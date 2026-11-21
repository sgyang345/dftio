from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
import re
from tqdm import tqdm
from collections import Counter
from dftio.constants import orbitalId, SIESTA2DFTIO,anglrMId
import ase
import dpdata
import os
import numpy as np
from collections import Counter
from dftio.io.parse import Parser, ParserRegister, find_target_line
from dftio.data import _keys
import sisl


@ParserRegister.register("siesta")
class SiestaParser(Parser):
    def __init__(
            self,
            root,
            prefix,
            **kwargs
            ):
        super(SiestaParser, self).__init__(root, prefix, **kwargs)
    
    @staticmethod
    def convert_kpoints_bohr_inv_to_twopi_over_a(kpoints_bohr_inv, lattice_const_a_angstrom):
        """
        Convert k-points from units of 1/Bohr to units of 2π/a, where 'a' is the lattice constant.

        Parameters
        ----------
        kpoints_bohr_inv : array-like or float
            K-points in units of inverse Bohr radius (1/Bohr).
        lattice_const_a_angstrom : float
            Lattice constant 'a' in Angstroms.

        Returns
        -------
        kpoints_twopi_over_a : array-like or float
            K-points in units of 2π/a.

        Notes
        -----
        The conversion uses the Bohr radius (a₀ = 0.529177 Å) and the provided lattice constant.
        """
        a0 = 0.529177  # Bohr radius in Å
        scale = lattice_const_a_angstrom / (a0 * 2*np.pi)
        kpoints_twopi_over_a = np.dot(kpoints_bohr_inv, scale)
        return kpoints_twopi_over_a

    @staticmethod
    def find_content(path, str_to_find, 
                     for_system_label=False,
                     for_Kpt_bands=False):
        if for_system_label and for_Kpt_bands:
            raise ValueError("for_system_label and for_Kpt_bands \
                             cannot both be True at the same time.")
        file_path = None
        system_label_content = None
        targeted_files = []
        for root, _, files in os.walk(path):
            for file in files:
                    if not for_Kpt_bands and not file.endswith('.fdf'):
                        # Skip files that are not FDF files except for k-point bands search
                        continue

                    if for_Kpt_bands:
                        # SIESTA outputs the kpoints for band in log when 
                        #           WriteKbands is set to true.
                        # However, the name of the log file is not fixed.
                        # Therefore, we target the log file by searching for 'WELCOME' 
                        # in all the files in the directory except for some specific file types.
                        skip_format = ['.TSHS', '.DM', 'xml','BONDS','ORB_INDX',\
                                       'psf','nc','ion','psdump','xml']
                        if any(file.endswith(ext) for ext in skip_format):
                            continue
                    
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            # Search for the string in the content with case insensitivity
                            # because SIESTA input files are case-insensitive.
                            match = re.search(r'\b'+str_to_find+r'\b\s*(\S+)', content, re.IGNORECASE)
                            if match:
                                targeted_files.append(file_path)
                                if for_system_label:
                                    system_label_content = match.group(1)
                    except Exception as e:
                        print(f"Skipping file {file_path} due to error: {e}")
                        continue
                                
        
        if for_system_label and system_label_content is None:
            print(f"Warning: Don't find {str_to_find} in {file_path}.\
                     use SIESTA default value: siesta")
            system_label_content = "siesta"

        if len(targeted_files) > 1:
            file_path = targeted_files[-1]
            print(f"Warning: Multiple files found containing '{str_to_find}': {targeted_files}.\n \
                            Using the last one: {file_path}.")
        elif len(targeted_files) == 1:
            file_path = targeted_files[0]
        elif len(targeted_files) == 0:
            raise FileNotFoundError(f"No file found containing '{str_to_find}' in {path}")
        
        return file_path, system_label_content   


    # essential
    def get_structure(self,idx):
        path = self.raw_datas[idx]
        lattice_vectors,_ = SiestaParser.find_content(path= path,str_to_find='LatticeVectors')
        struct,_ = SiestaParser.find_content(path= path,str_to_find='AtomicCoordinatesAndAtomicSpecies')
        chemspecis,_ = SiestaParser.find_content(path= path,str_to_find='ChemicalSpeciesLabel')

        with open(lattice_vectors, 'r') as file:
            lines = file.readlines()
        counter_start_end = []
        for i in range(len(lines)):
            if 'LatticeVectors' in lines[i].split():
                counter_start_end.append(i)
        assert len(counter_start_end) == 2, "LatticeVectors section not found or malformed."
        # Extract lattice vectors
        lattice_vec = np.array([lines[i].split()[0:3] for i in range(counter_start_end[0]+1, counter_start_end[1])], dtype=self.real_dtype)

        with open(struct, 'r') as file:
            lines = file.readlines()
        counter_start_end = []
        for i in range(len(lines)):
            if 'AtomicCoordinatesAndAtomicSpecies' in lines[i].split():
                counter_start_end.append(i)
        struct_xyz = np.array([lines[i].split()[0:3] for i in range(counter_start_end[0]+1, counter_start_end[1])], dtype=self.real_dtype)
        element_type = [lines[i].split()[3] for i in range(counter_start_end[0]+1, counter_start_end[1])]

        with open(chemspecis, 'r') as file:
            lines = file.readlines()
        counter_start_end = []
        for i in range(len(lines)):
            if 'ChemicalSpeciesLabel' in lines[i].split():
                counter_start_end.append(i)
        element_index = {}
        for i in range(counter_start_end[0]+1, counter_start_end[1]):
            element_index[int(lines[i].split()[0])] = lines[i].split()[1]
        element_index_all = [element_index[int(e)] for e in element_type]


        # struct = sisl.get_sile(struct).read_geometry()
        structure = {
            _keys.ATOMIC_NUMBERS_KEY: np.array(element_index_all, dtype=np.int32),
            _keys.PBC_KEY: np.array([True, True, True]) # abacus does not allow non-pbc structure
        }
        structure[_keys.POSITIONS_KEY] = struct_xyz.astype(self.real_dtype)[np.newaxis, :, :]
        structure[_keys.CELL_KEY] = lattice_vec.astype(self.real_dtype)[np.newaxis, :, :]

        return structure
    
    # essential
    def get_eigenvalue(self, idx, band_index_min=0):
        """
        Extracts the eigenvalues and k-points from SIESTA output files for a given calculation index.
        k-point coordinates are extracted from log file ( ensure "WriteKbands    .true." in SIESTA input file).
        Eigenvalues are extracted from the corresponding ".bands" file. 
        Now it supports only non-spin-polarized calculations.
        Args:
            idx (int): Index of the calculation in `self.raw_datas` to process.
            band_index_min (int, optional): Minimum band index to include in the output. Defaults to 0.
        Returns:
            dict: A dictionary containing:
                - _keys.ENERGY_EIGENVALUE_KEY (np.ndarray): Eigenvalues of shape (1, num_kpts, num_bands), with the requested precision.
                - _keys.KPOINT_KEY (np.ndarray): K-point coordinates of shape (num_kpts, 3), with the requested precision.
        """
        
        # Check if WriteKbands is set to true in the SIESTA input file
        RUN_file, _ = SiestaParser.find_content(path=self.raw_datas[idx],
                                       str_to_find='WriteKbands',
                                       for_Kpt_bands=False)
        with open(RUN_file, 'r') as file:
            lines = file.readlines()
        for line in lines:
            if 'WriteKbands' in line.split():
                if 'true' not in line.lower() or "t" not in line.lower():
                    raise ValueError("WriteKbands is not set to true in the SIESTA input file. \
                                      Cannot extract k-points and eigenvalues.")

        # Determine log_file, system_label, and eigs_file(.bands)
        log_file,_ = SiestaParser.find_content(path=self.raw_datas[idx], 
                                       str_to_find='WELCOME',
                                       for_Kpt_bands=True)
        assert os.path.exists(log_file), f"Log file {log_file} does not exist."
        _,system_label = SiestaParser.find_content(path=self.raw_datas[idx], 
                                           str_to_find='SystemLabel', 
                                           for_system_label=True)
        assert system_label is not None, "SystemLabel not defined."
        eigs_file = os.path.join(self.raw_datas[idx], system_label + ".bands")
        assert os.path.exists(eigs_file), f"Eigenvalue file {eigs_file} does not exist."

        # Extract k-points from the log file
        kpts = []
        with open(log_file, 'r') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            if 'siesta: Band k vectors' in line:
                for data_line in lines[i+2:]:
                    if not data_line.strip():  # skip empty lines
                        break
                    match = re.match(r'\s*\d+\s+([-\d\.Ee+]+)\s+([-\d\.Ee+]+)\s+([-\d\.Ee+]+)', data_line)
                    if match:
                        kvec = tuple(float(match.group(j)) for j in range(1, 4))
                        kpts.append(kvec)
                    else:
                        break 
                break  
        if len(kpts) == 0:
            raise ValueError("No k-points found in the log file.")
        kpts = np.array(kpts) # in units of Bohr^-1

        # unit from Bohr^-1 to 2π/a
        lattice_vec_path,_ = SiestaParser.find_content(path=self.raw_datas[idx],str_to_find='LatticeVectors')
        with open(lattice_vec_path, 'r') as file:
            lines = file.readlines()
        counter_start_end = []
        for i in range(len(lines)):
            if 'LatticeVectors' in lines[i].split():
                counter_start_end.append(i)
        lattice_vec = np.array([lines[i].split()[0:3] for i in range(counter_start_end[0]+1, counter_start_end[1])], dtype=self.real_dtype)
        kpts = SiestaParser.convert_kpoints_bohr_inv_to_twopi_over_a(kpts, lattice_vec) # in units of 2π/a
        

        # Extract eigenvalues from the eigs_file(.bands)
        eigs = []
        eigs_k = [] # eigenvalues for each k-point
        with open(eigs_file, 'r') as file:
            lines = file.readlines()
        ## Determine the Gamma-only case or band case
        isGamma :bool = None
        start_line : int = None
        if len(lines[2].split()) == 3: ### Gamma-only case
            isGamma = True
            start_line = 2
            assert lines[2].split()[2] == '1', \
                "The detected file is Gamma-only, but the number of k-points is not 1."
        elif len(lines[3].split()) == 3: ### Band case
            isGamma = False
            start_line = 3
            assert lines[3].split()[2] == str(len(kpts)), \
                "The detected file is a band case, but the number of k-points does not match the number of k-points found in the log file."
        else:
            raise ValueError("Unexpected format in .band file. Expected Gamma-only or band case.")
            
        for idx, line in enumerate(lines[start_line:]):
            # Etract the number of bands, spin_degree and k-points from the first line
            if idx == 0:
                Num_bands = int(line.split()[0])
                spin_degree = int(line.split()[1])
                if spin_degree != 1:
                    raise NotImplementedError("Only support non-spin-polarized calculations.")
                Num_kpts = int(line.split()[2])
                assert Num_kpts == len(kpts), f"Number of k-points in file ({Num_kpts}) does not match the number of k-points found ({len(kpts)})."
                continue
            # Extract eigenvalues for each k-point
            if line.strip():  # eliminate empty lines
                eigs_k.extend([float(val) for val in line.strip().split()])
                if (isGamma and len(eigs_k) == Num_bands + 3) \
                    or (not isGamma and len(eigs_k) == Num_bands + 1):
                    skip = 3 if isGamma else 1
                    eigs.append(eigs_k[skip:])
                    eigs_k = []
            
            if len(eigs) == Num_kpts: # if we have collected all k-points
                break

        eigs = np.array(eigs)[np.newaxis, :, band_index_min:]
        assert eigs.shape[1] == len(kpts), \
            f"Number of kpoints for eigenvalues ({eigs.shape[1]}) does not match the number of k-points in Kline ({len(kpts)})." 
    
        return {_keys.ENERGY_EIGENVALUE_KEY: eigs.astype(self.real_dtype), _keys.KPOINT_KEY: kpts.astype(self.real_dtype)}


    # essential
    def get_basis(self,idx):
        # {"Si": "2s2p1d"}
        path = self.raw_datas[idx]
        _,system_label = SiestaParser.find_content(path=path,str_to_find='SystemLabel',for_system_label=True)
        if system_label is None:
            system_label = "siesta"

        # tshs = self.raw_datas[idx]+ "/"+system_label+".TSHS"
        # hamil =  sisl.Hamiltonian.read(tshs)
        ORB_INDX = self.raw_datas[idx]+ "/"+system_label+".ORB_INDX"

        with open(ORB_INDX , 'r') as file:
            lines = file.readlines()

        basis = {}
        all_orb_num = lines[0].split()[0]
        all_elements = []
        all_ia = []
        all_l_info = []

        for i in range(3, 3+int(all_orb_num)):
            line = lines[i].split()
            all_elements.append(line[3])
            all_ia.append(int(line[1]))
            all_l_info.append(line[6])

        element_type = list(set(all_elements))
        all_ia = np.array(all_ia)
        all_l_info = np.array(all_l_info)


        for e in element_type:
            first_e_ia = all_elements.index(e)
            mask = all_ia == all_ia[first_e_ia]
            select_l_info = Counter(all_l_info[mask])
            counted_basis_list = []
            for l in select_l_info.keys():
                if l == '0':
                    counted_basis_list.append(str(int(select_l_info['0']/1))+'s')
                elif l == '1':
                    counted_basis_list.append(str(int(select_l_info['1']/3))+'p')
                elif l == '2':
                    counted_basis_list.append(str(int(select_l_info['2']/5))+'d')
                elif l == '3':
                    counted_basis_list.append(str(int(select_l_info['3']/7))+'f')
            basis[str(e)] = "".join(counted_basis_list)
        
        return basis


    # essential
    def get_blocks(self, idx, hamiltonian: bool = False, overlap: bool = False, density_matrix: bool = False):
        path = self.raw_datas[idx]
        _,system_label = SiestaParser.find_content(path=self.raw_datas[idx],
                                           str_to_find='SystemLabel', 
                                           for_system_label=True)
        if system_label is None:
            system_label = "siesta"
        hamiltonian_dict, overlap_dict, density_matrix_dict = None, None, None
        struct,_ = SiestaParser.find_content(path= path,
                                     str_to_find='AtomicCoordinatesAndAtomicSpecies')
        chemspecis,_ = SiestaParser.find_content(path= path,
                                         str_to_find='ChemicalSpeciesLabel')
        
        with open(struct, 'r') as file:
            lines = file.readlines()
        counter_start_end = []
        for i in range(len(lines)):
            line = lines[i].split()
            if 'AtomicCoordinatesAndAtomicSpecies' in line:
                counter_start_end.append(i)
        element_type = [lines[i].split()[3] for i in range(counter_start_end[0]+1, counter_start_end[1])]
        na = len(element_type)

        with open(chemspecis, 'r') as file:
            lines = file.readlines()
        counter_start_end = []
        for i in range(len(lines)):
            if 'ChemicalSpeciesLabel' in lines[i].split():
                counter_start_end.append(i)
        element_symbol = {}
        for i in range(counter_start_end[0]+1, counter_start_end[1]):
            element_symbol[int(lines[i].split()[0])] = lines[i].split()[2]
        element = [element_symbol[int(e)] for e in element_type]
        
        tshs = path+ "/"+system_label+".TSHS"
        if os.path.exists(tshs):
            hamil =  sisl.Hamiltonian.read(tshs)
        else:
            raise FileNotFoundError("Hamiltonian file not found.")        
        site_norbits = np.array([hamil.atoms[i].no for i in range(hamil.na)])
        site_norbits_cumsum = site_norbits.cumsum()

        basis = self.get_basis(idx)      
        spinful = False #TODO: add support for spinful


        central_cell = [int(np.floor(hamil.nsc[i]/2)) for i in range(3)]
        Rvec_list = []
        for rx in range(central_cell[0],hamil.nsc[0]):
            for ry in range(hamil.nsc[1]):
                for rz in range(hamil.nsc[2]):
                    Rvec_list.append([rx-central_cell[0],ry-central_cell[1],rz-central_cell[2]])
        Rvec = np.array(Rvec_list)


        hamiltonian_dict = {}
        overlap_dict = {}
        density_matrix_dict = {}


        l_dict = {}
        # count norbs
        count = {}
        for at in basis:
            count[at] = 0
            l_dict[at] = []
            for iorb in range(int(len(basis[at]) / 2)):
                n, o = int(basis[at][2*iorb]), basis[at][2*iorb+1]
                count[at] += n * (2*anglrMId[o]+1)
                l_dict[at] += [anglrMId[o]] * n

        cut_tol_ham = 1e-5
        cut_tol_ovp = 1e-5
        cut_tol_dm = 1e-5


        if hamiltonian:
           
            hamil_csr = hamil.tocsr()
            hamil_blocks = []
            hamil_mask = []
            for i in range(Rvec.shape[0]):
                off = hamil.geometry.sc_index(Rvec[i]) * hamil.geometry.no
                if np.abs(hamil_csr[:,off:off+hamil.geometry.no].toarray()).max() > cut_tol_ham:
                    hamil_mask.append(True)
                    hamil_blocks.append(hamil_csr[:,off:off+hamil.geometry.no].toarray())
                else:
                    hamil_mask.append(False)

            hamil_mask = np.array(hamil_mask)
            hamil_Rvec = Rvec[hamil_mask]
            hamil_blocks = np.stack(hamil_blocks).astype(self.real_dtype)

            for i in range(na):
                si = element[i]
                for j in range(na):
                    sj = element[j]
                    keys = map(lambda x: "_".join([str(i),str(j),str(x[0].astype(np.int32)),\
                                str(x[1].astype(np.int32)),str(x[2].astype(np.int32))]), hamil_Rvec)
                    i_norbs = site_norbits[i]
                    i_orbs_start =site_norbits_cumsum[i] - i_norbs
                    j_norbs = site_norbits[j]
                    j_orbs_start =site_norbits_cumsum[j] - j_norbs
                    block = self.transform(hamil_blocks[:,i_orbs_start:i_orbs_start+i_norbs,j_orbs_start:j_orbs_start+j_norbs],\
                                            l_dict[si], l_dict[sj])
                    # block = hamil_blocks[:,i_orbs_start:i_orbs_start+i_norbs,j_orbs_start:j_orbs_start+j_norbs]
                    block_mask = np.abs(block).max(axis=(1,2)) > cut_tol_ham

                    if np.any(block_mask):
                        keys = list(keys)
                        keys = [keys[k] for k,t in enumerate(block_mask) if t]
                        hamiltonian_dict.update(dict(zip(keys, block[block_mask])))
            
        if overlap:
            if os.path.exists(tshs):
                ovp =  sisl.Overlap.read(tshs)
            else:
                raise FileNotFoundError("Overlap file not found.")
            
            ovp_csr = ovp.tocsr()
            ovp_blocks = []
            ovp_mask = []
            for i in range(Rvec.shape[0]):
                off = ovp.geometry.sc_index(Rvec[i]) * ovp.geometry.no
                if np.abs(ovp_csr[:,off:off+ovp.geometry.no].toarray()).max() > cut_tol_ovp:
                    ovp_mask.append(True)
                    ovp_blocks.append(ovp_csr[:,off:off+ovp.geometry.no].toarray())
                else:
                    ovp_mask.append(False)
                
            ovp_blocks = np.stack(ovp_blocks).astype(self.real_dtype)
            ovp_mask = np.array(ovp_mask)
            ovp_Rvec = Rvec[ovp_mask]

            for i in range(na):
                si = element[i]
                for j in range(na):
                    sj = element[j]
                    keys = map(lambda x: "_".join([str(i),str(j),str(x[0].astype(np.int32)),\
                                str(x[1].astype(np.int32)),str(x[2].astype(np.int32))]), ovp_Rvec)
                    i_norbs = site_norbits[i]
                    i_orbs_start =site_norbits_cumsum[i] - i_norbs
                    j_norbs = site_norbits[j]
                    j_orbs_start =site_norbits_cumsum[j] - j_norbs
                    block = self.transform(ovp_blocks[:,i_orbs_start:i_orbs_start+i_norbs,j_orbs_start:j_orbs_start+j_norbs],\
                                             l_dict[si], l_dict[sj])
                    # block = ovp_blocks[:,i_orbs_start:i_orbs_start+i_norbs,j_orbs_start:j_orbs_start+j_norbs]
                    block_mask = np.abs(block).max(axis=(1,2)) > cut_tol_ovp

                    if np.any(block_mask):
                        keys = list(keys)
                        keys = [keys[k] for k,t in enumerate(block_mask) if t]
                        overlap_dict.update(dict(zip(keys, block[block_mask])))

        if density_matrix:
            _,system_label = SiestaParser.find_content(path=self.raw_datas[idx],
                                               str_to_find='SystemLabel', 
                                               for_system_label=True)
            if system_label is None:
                system_label = "siesta"
            DM_path = self.raw_datas[idx]+ "/"+system_label+".DM"
            if os.path.exists(DM_path):
                DM =  sisl.DensityMatrix.read(DM_path)
            else:
                raise FileNotFoundError("Density Matrix file not found.")
            
            DM_csr = DM.tocsr()
            DM_blocks = []
            DM_mask = []
            for i in range(Rvec.shape[0]):
                off = DM.geometry.sc_index(Rvec[i]) * DM.no
                if np.abs(DM_csr[:,off:off+DM.geometry.no].toarray()).max() > cut_tol_dm:
                    DM_mask.append(True)
                    DM_blocks.append(DM_csr[:,off:off+DM.geometry.no].toarray())
                else:
                    DM_mask.append(False)
                
            DM_blocks = np.stack(DM_blocks).astype(self.real_dtype)
            DM_mask = np.array(DM_mask)
            DM_Rvec = Rvec[DM_mask]

            for i in range(na):
                si = element[i]
                for j in range(na):
                    sj = element[j]
                    keys = map(lambda x: "_".join([str(i),str(j),str(x[0].astype(np.int32)),\
                                str(x[1].astype(np.int32)),str(x[2].astype(np.int32))]), DM_Rvec)
                    i_norbs = site_norbits[i]
                    i_orbs_start =site_norbits_cumsum[i] - i_norbs
                    j_norbs = site_norbits[j]
                    j_orbs_start =site_norbits_cumsum[j] - j_norbs
                    block = self.transform(DM_blocks[:,i_orbs_start:i_orbs_start+i_norbs,j_orbs_start:j_orbs_start+j_norbs],\
                                            l_dict[si], l_dict[sj])
                    
                    block_mask = np.abs(block).max(axis=(1,2)) > cut_tol_dm

                    if np.any(block_mask):
                        keys = list(keys)
                        keys = [keys[k] for k,t in enumerate(block_mask) if t]
                        density_matrix_dict.update(dict(zip(keys, block)))

            
        
        return [hamiltonian_dict], [overlap_dict], [density_matrix_dict]
    

    
    def transform(self, mat, l_lefts, l_rights):
        # ssppd   l_lefts=[0,0,1,1,2] l_rights=[0,0,1,1,2]

        if max(*l_lefts, *l_rights) > 5:
            raise NotImplementedError("Only support l = s, p, d, f, g, h.")
        block_lefts = block_diag(*[SIESTA2DFTIO[l_left] for l_left in l_lefts])
        block_rights = block_diag(*[SIESTA2DFTIO[l_right] for l_right in l_rights])

        return block_lefts @ mat @ block_rights.T
