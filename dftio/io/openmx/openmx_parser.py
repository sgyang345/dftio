import os
import numpy as np
import re
import glob
import warnings
from dftio.io.parse import Parser, ParserRegister, find_target_line
from dftio.data import _keys
import ase.data
from collections import Counter
from dftio.constants import orbitalId, OPENMX2DFTIO, Bohr2Ang, Harte2eV
import dpdata
from ase.io import read
from scipy.linalg import block_diag

@ParserRegister.register("openmx")
class OpenmxParser(Parser):
    def __init__(
            self,
            root,
            prefix,
            **kwargs
            ):
        super(OpenmxParser, self).__init__(root, prefix)
        self._cache_idx = -1
        self._cache_data = None
        mode = self.get_mode(idx=0)
        if mode in ['Nomd']:
            self.raw_sys = [
                dpdata.System(self._read_cif_without_spacegroup_warning(f), fmt="ase/structure")
                for idx in range(len(self.raw_datas))
                for f in glob.glob(os.path.join(self.raw_datas[idx], "*.cif"))
            ]
            # self.raw_sys = [dpdata.System(read(os.path.join(self.raw_datas[idx], "openmx.cif")), fmt="ase/structure") for idx in range(len(self.raw_datas))]
        elif mode in ['md']:
            self.raw_sys = [dpdata.LabeledSystem(os.path.join(self.raw_datas[idx], "openmx"), fmt='openmx/md') for idx in range(len(self.raw_datas))]

    def _read_cif_without_spacegroup_warning(self, path):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"crystal system '.*' is not interpreted for space group .*",
                category=UserWarning,
            )
            return read(path)

    def get_mode(self, idx):
        with open(glob.glob(os.path.join(self.raw_datas[idx], "*.out"))[0], 'r') as f:
            line = find_target_line(f, "MD.Type")
            # assert line is not None, 'Cannot find "MD.Type" in input file'
            # mode_val = line.split()[1]
            # if mode_val == "Nomd":
            #     pass
            # else:
            #     mode_val = 'md' # Fallback
            if line is not None:
                mode_val = line.split()[1]
                if mode_val == "Nomd":
                    pass
                else:
                    mode_val = 'md'
            else:
                mode_val = "Nomd"
        return mode_val


    def get_structure(self, idx):
        sys = self.raw_sys[idx]
        
        # Use corrected geometry from scfout input restoration if available
        # _read_scfout_file returns data in Bohr
        data = self._read_scfout_file(idx, read_matrices=False)
        tv_bohr = data["tv"]
        Gxyz_bohr = data["Gxyz"]
        
        # Convert to Angstrom for dftio output
        tv_ang = tv_bohr * Bohr2Ang
        Gxyz_ang = Gxyz_bohr * Bohr2Ang
        
        structure = {
            _keys.ATOMIC_NUMBERS_KEY: np.array([ase.data.atomic_numbers[i] for i in sys.data["atom_names"]], dtype=np.int32)[sys.data["atom_types"]],
            _keys.PBC_KEY: np.array([True, True, True])
        }
        
        # Ensure dimensions are (nframes, ...)
        # Nomd mode -> 1 frame
        structure[_keys.POSITIONS_KEY] = Gxyz_ang[np.newaxis, :, :].astype(np.float32)
        structure[_keys.CELL_KEY] = tv_ang[np.newaxis, :, :].astype(np.float32)

        return structure


    def get_eigenvalue(self, idx, band_index_min=0):
        path = self.raw_datas[idx]
        mode = self.get_mode(idx)

        if mode in ['Nomd']:
            assert os.path.exists(glob.glob(os.path.join(self.raw_datas[idx], "*.out"))[0]), "openmx.out not found"
            with open(glob.glob(os.path.join(self.raw_datas[idx], "*.out"))[0]) as f:
                solver = find_target_line(f, "scf.EigenvalueSolver")
                dispersion = find_target_line(f, "Band.dispersion")
                nkpath_line = find_target_line(f, "Band.Nkpath")
                kpath_block = find_target_line(f, "<Band.kpath")

            if (solver and "band" in solver.lower() and dispersion and "on" in dispersion.lower() and nkpath_line and kpath_block):
                nkpath = int(nkpath_line.split()[1])
                assert os.path.exists(glob.glob(os.path.join(self.raw_datas[idx], "*.Band"))[0]), "openmx.Band not found"
                return self._read_band_file(glob.glob(os.path.join(self.raw_datas[idx], "*.Band"))[0], nkpath, band_index_min)
            else:
                with open(glob.glob(os.path.join(self.raw_datas[idx], "*.out"))[0]) as f:
                    kpts = []
                    eigenvalues = []
                    # Determine spin polarization
                    line = find_target_line(f, "scf.SpinPolarization")
                    nspin = 1 if line.split()[1].lower() == "off" else 2
                    # Find Eigenvalues section
                    line = find_target_line(f, "Eigenvalues (Hartree) of SCF KS-eq.")
                    assert line, "Cannot find 'Eigenvalues (Hartree) of SCF KS-eq.' in openmx.out"
                    while True:
                        line = f.readline()
                        if "kloop" in line:
                            break
                    while "kloop" in line:
                        # New k-point
                        line = f.readline() # k1= ...
                        assert "k1" in line
                        parts = line.replace("=", " ").split()
                        kpts.append([float(parts[1]), float(parts[3]), float(parts[5])])
                        # Prepare storage for this k-point
                        eigs_up = []
                        eigs_down = []
                        f.readline() # Empty line
                        while True:
                            line = f.readline()
                            if not line or not line.strip():
                                break
                            parts = line.split()
                            vals = [float(x) for x in parts[1:]]
                            eigs_up.append(vals[0])
                            eigs_down.append(vals[1])
                        eigenvalues.append([eigs_up, eigs_down] if len(eigs_down) > 0 else [eigs_up])
                        line = f.readline()
                        while line and "kloop" not in line :
                            break
            eigs_np = np.moveaxis(np.array(eigenvalues), 1, 0)
            if nspin == 1 and eigs_np.shape[0] == 2:
                 eigs_np = eigs_np[0:1]
            kpts_np = np.array(kpts)
            eigs_np = eigs_np[:, :, band_index_min:] * Harte2eV
            return {_keys.ENERGY_EIGENVALUE_KEY: eigs_np.astype(np.float32), _keys.KPOINT_KEY: kpts_np.astype(np.float32)}

        elif mode == "md" or mode == "relax":
            raise NotImplementedError("output eigenvalues from MD trajectory is not supported by OpenMX.")
        else:
            raise NotImplementedError("mode {} is not supported.".format(mode))


    def _read_band_file(self, band_path, nkpath, band_index_min=0):
        """
        Read eigenvalues and k-points from openmx.Band file.
        """
        with open(band_path, 'r') as f:
            line = f.readline().split()
            nbands = int(line[0])
            spin_pol = int(line[1]) # 0: non-spin, 1: spin
            f.readline()
            line = f.readline().strip()
            assert int(line) == nkpath
            for _ in range(nkpath):
                f.readline()
            kpts = []
            eigenvalues = [] # [kpt][spin][band]

            if spin_pol == 0:
                while True:
                    line = f.readline()
                    if not line or not line.strip():
                        break
                    parts = line.split()
                    assert len(parts) == 4 and parts[0] == str(nbands)
                    kpts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    line = f.readline()
                    parts = line.split()
                    assert len(parts) == nbands
                    eigenvalues.append([[float(x) for x in parts]])
            else:
                while True:
                    line = f.readline()
                    if not line or not line.strip():
                        break
                    eigs_up = []
                    eigs_down = []
                    parts1 = line.split()
                    assert len(parts1) == 4 and parts1[0] == str(nbands)
                    kpts.append([float(parts1[1]), float(parts1[2]), float(parts1[3])])
                    line = f.readline()
                    parts = line.split()
                    assert len(parts) == nbands
                    eigs_up.extend([float(x) for x in parts])

                    line = f.readline()
                    parts2 = line.split()
                    assert parts2 == parts1
                    line = f.readline()
                    parts = line.split()
                    assert len(parts) == nbands
                    eigs_down.extend([float(x) for x in parts])
                    eigenvalues.append([eigs_up, eigs_down])

        eigs_np = np.array(eigenvalues)
        eigs_np = np.moveaxis(eigs_np, 1, 0)
        eigs_np = eigs_np[:, :, band_index_min:] * Harte2eV
        kpts_np = np.array(kpts)
        return {_keys.ENERGY_EIGENVALUE_KEY: eigs_np.astype(np.float32), _keys.KPOINT_KEY: kpts_np.astype(np.float32)}


    def get_basis(self, idx):
        data = self._read_scfout_file(idx, read_matrices=False)
        orbital_types = data["orbital_types"]
        element_list = data["element_list"]

        basis = {}
        for i, elem_num in enumerate(element_list):
            elem_sym = ase.data.chemical_symbols[elem_num]
            if elem_sym not in basis:
                orbs = orbital_types[i]
                counts = Counter(orbs)
                basis_str = ""
                for l in sorted(counts.keys()):
                    basis_str += f"{counts[l]}{orbitalId[l]}"
                basis[elem_sym] = basis_str
        return basis

    def get_blocks(self, idx, hamiltonian=True, overlap=False, density_matrix=False):
        mode = self.get_mode(idx)
        hamiltonians = {}
        overlaps = {}
        density_matrixs = {}

        if mode == 'Nomd':
            data = self._read_scfout_file(idx, read_matrices=True, one_frame_only=True)

            if hamiltonian:
                Hk = data["Hk"]
                iHk = data["iHk"]
                SpinP_switch = data["SpinP_switch"]

                if SpinP_switch == 3:
                    # Merge Hk and iHk
                    for i in range(len(Hk[3])):
                        for j in range(len(Hk[3][i])):
                            Hk[3][i][j] += iHk[2][i][j]
                            iHk[2][i][j] = -Hk[3][i][j]
                hamiltonians = self._build_general_matrix(data, 'hamiltonian')
                for i in hamiltonians.keys():
                    hamiltonians[i] *= Harte2eV
            if overlap:
                overlaps = self._build_general_matrix(data, 'overlap')
            if density_matrix:
                density_matrixs = self._build_general_matrix(data, 'density_matrix')
            return [hamiltonians], [overlaps], [density_matrixs]

        elif mode == 'md':
            sys = self.raw_sys[idx]
            nframes = sys.get_nframes()
        else:
             raise NotImplementedError("mode {} is not supported.".format(mode))

    def _build_general_matrix(self, data, matrix_type):
        matrices = {}
        SpinP_switch = data["SpinP_switch"]
        FNAN = data["FNAN"]
        natn = data["natn"]
        ncn = data["ncn"]
        atv_ijk = data["atv_ijk"]
        Total_NumOrbs = data["Total_NumOrbs"]
        nsites = data["atomnum"]
        orbital_types = data["orbital_types"]

        # Determine input data based on matrix_type
        if matrix_type == 'hamiltonian':
            Mat_real = data["Hk"]
            Mat_imag = data["iHk"]
        elif matrix_type == 'overlap':
            Mat_real = [data["OLP"]]
            Mat_imag = None
        elif matrix_type == 'density_matrix':
            Mat_real = data["DM"]
            Mat_imag = data.get("iDM") # iDM might be None if not read or not applicable
        else:
            raise ValueError(f"Unknown matrix_type: {matrix_type}")

        # For overlap Matrix
        if matrix_type == 'overlap':
             for site_i in range(nsites):
                for index_nn_i in range(FNAN[site_i]):
                    site_j = natn[site_i][index_nn_i]
                    R = atv_ijk[ncn[site_i][index_nn_i]]
                    key = f"{site_i}_{site_j}_{R[0]}_{R[1]}_{R[2]}"

                    # OLP is [atom][neighbor]
                    # Mat_real is [[atom][neighbor]] (wrapped)
                    block = Mat_real[0][site_i][index_nn_i]
                    block = self.transform(block, orbital_types[site_i], orbital_types[site_j])
                    matrices[key] = block.astype(np.float32)
             return matrices

        # For Hamiltonian and Density Matrix
        if SpinP_switch == 0:
            for site_i in range(nsites):
                for index_nn_i in range(FNAN[site_i]):
                    site_j = natn[site_i][index_nn_i]
                    R = atv_ijk[ncn[site_i][index_nn_i]]
                    key = f"{site_i}_{site_j}_{R[0]}_{R[1]}_{R[2]}"


                    block = Mat_real[0][site_i][index_nn_i]
                    block = self.transform(block, orbital_types[site_i], orbital_types[site_j])
                    matrices[key] = block.astype(np.float32)

        elif SpinP_switch == 3:
            for site_i in range(nsites):
                for index_nn_i in range(FNAN[site_i]):
                    site_j = natn[site_i][index_nn_i]
                    R = atv_ijk[ncn[site_i][index_nn_i]]
                    key = f"{site_i}_{site_j}_{R[0]}_{R[1]}_{R[2]}"
                    key_inv = f"{site_j}_{site_i}_{-R[0]}_{-R[1]}_{-R[2]}"

                    len_i_wo_spin = Total_NumOrbs[site_i]
                    len_j_wo_spin = Total_NumOrbs[site_j]

                    if key not in matrices:
                        matrices[key] = np.zeros((len_i_wo_spin * 2, len_j_wo_spin * 2), dtype=np.complex128)
                    if key_inv not in matrices:
                        matrices[key_inv] = np.zeros((len_j_wo_spin * 2, len_i_wo_spin * 2), dtype=np.complex128)

                    for spini in range(2):
                        for spinj in range(spini, 2):
                            if spini == 0:
                                if spinj == 0:
                                    real, imag = Mat_real[0][site_i][index_nn_i], Mat_imag[0][site_i][index_nn_i]
                                else:
                                    # Note: For H, we did preprocessing: Hk[3] += iHk[2] -> Hk[3] holds the full imaginary part.
                                    # For DM, we use DM[2] and DM[3] directly.
                                    real, imag = Mat_real[2][site_i][index_nn_i], Mat_real[3][site_i][index_nn_i]
                            else:
                                real, imag = Mat_real[1][site_i][index_nn_i], Mat_imag[1][site_i][index_nn_i]

                            block = real.T + 1j * imag.T

                            matrices[key][spini * len_i_wo_spin : (spini + 1) * len_i_wo_spin,
                                            spinj * len_j_wo_spin : (spinj + 1) * len_j_wo_spin] = block

                            if spini == 0 and spinj == 1:
                                matrices[key_inv][1 * len_j_wo_spin : (1 + 1) * len_j_wo_spin,
                                                    0 * len_i_wo_spin : (0 + 1) * len_i_wo_spin] = block.conj().T

                    # Apply transform to the full spinful matrix
                    matrices[key] = self.transform(matrices[key], orbital_types[site_i] * 2, orbital_types[site_j] * 2)
                    matrices[key_inv] = self.transform(matrices[key_inv], orbital_types[site_j] * 2, orbital_types[site_i] * 2)

        return matrices

    def transform(self, mat, l_lefts, l_rights):
        if max(*l_lefts, *l_rights) > 3:
            # OPENMX2DFTIO supports up to f (3)
             raise NotImplementedError("Only support l = s, p, d, f.")
            
        block_lefts = block_diag(*[OPENMX2DFTIO[l_left] for l_left in l_lefts])
        block_rights = block_diag(*[OPENMX2DFTIO[l_right] for l_right in l_rights])

        return block_lefts @ mat @ block_rights.T


    @staticmethod
    def _multiread(f, dtype, count):
        return np.fromfile(f, dtype=dtype, count=count)

    @staticmethod
    def _read_mixed_matrix(f, dtype, dims):
        ret = []
        for i in dims:
            t = np.fromfile(f, dtype=dtype, count=i)
            ret.append(t)
        return ret

    @staticmethod
    def _read_matrix_in_mixed_matrix(f, dtype, spins, atomnum, FNAN, natn, Total_NumOrbs):
        ret = []
        for spin in range(spins):
            t_spin = []
            for ai in range(atomnum):
                t_ai = []
                for aj_inner in range(FNAN[ai]):
                    rows = Total_NumOrbs[natn[ai][aj_inner]]
                    cols = Total_NumOrbs[ai]
                    t = np.fromfile(f, dtype=dtype, count=rows*cols).reshape((cols, rows))
                    t_ai.append(t)
                t_spin.append(t_ai)
            ret.append(t_spin)
        return ret

    @staticmethod
    def _read_3d_vecs(f, dtype, num):
        data = np.fromfile(f, dtype=dtype, count=4 * num)
        data = data.reshape((num, 4))
        return data[:, 1:4]


    def _read_scfout_file(self, idx, read_matrices=False, one_frame_only=False):
        if self._cache_idx == idx:
            return self._cache_data

        with open(glob.glob(os.path.join(self.raw_datas[idx], "*.scfout"))[0])  as f:
            header = self._multiread(f, np.int32, 7)
            atomnum, SpinP_switch, Catomnum, Latomnum, Ratomnum, TCpyCell, order_max = header
            SpinP_switch &= 0x03

            atv = self._read_3d_vecs(f, np.float64, TCpyCell + 1) # atv
            atv_ijk = self._read_3d_vecs(f, np.int32, TCpyCell + 1)

            Total_NumOrbs = self._multiread(f, np.int32, atomnum)
            FNAN = self._multiread(f, np.int32, atomnum)
            FNAN += 1

            natn = self._read_mixed_matrix(f, np.int32, FNAN)
            natn = [x - 1 for x in natn]
            ncn = self._read_mixed_matrix(f, np.int32, FNAN)

            # Initialize lists for frame-dependent data
            tv = self._read_3d_vecs(f, np.float64, 3)
            rtv = self._read_3d_vecs(f, np.float64, 3) # rtv
            Gxyz = self._read_3d_vecs(f, np.float64, atomnum) # Gxyz

            Hk, iHk, OLP, DM, iDM = None, None, None, None, None

            # Always read matrices to advance file pointer correctly
            Hk = self._read_matrix_in_mixed_matrix(f, np.float64, SpinP_switch + 1, atomnum, FNAN, natn, Total_NumOrbs)

            if SpinP_switch == 3:
                iHk = self._read_matrix_in_mixed_matrix(f, np.float64, 3, atomnum, FNAN, natn, Total_NumOrbs) if SpinP_switch == 3 else None


            OLP = self._read_matrix_in_mixed_matrix(f, np.float64, 1, atomnum, FNAN, natn, Total_NumOrbs)[0]

            for dir_ in range(3):
                for order in range(order_max):
                    self._read_matrix_in_mixed_matrix(f, np.float64, 1, atomnum, FNAN, natn, Total_NumOrbs) # OLP_r components

            self._read_matrix_in_mixed_matrix(f, np.float64, 3, atomnum, FNAN, natn, Total_NumOrbs) # OLP_p
            DM = self._read_matrix_in_mixed_matrix(f, np.float64, SpinP_switch + 1, atomnum, FNAN, natn, Total_NumOrbs)
            iDM = self._read_matrix_in_mixed_matrix(f, np.float64, 2, atomnum, FNAN, natn, Total_NumOrbs)



        with open(glob.glob(os.path.join(self.raw_datas[idx], "*.out"))[0]) as f:
            #basis
            line = find_target_line(f, "Species.Number")
            num_species= int(line.split()[-1])
            line = find_target_line(f, "Definition.of.Atomic.Species")
            assert line is not None, 'Cannot find "ATOM TYPE" in openmx.out file'
            orbital_types_element = {}
            for i in range(num_species):
                line = f.readline().strip()
                parts = line.split()
                atom_label = parts[0]
                assert atom_label in ase.data.atomic_numbers, "Atom label should be in periodic table"
                orbital_types_str = parts[1].split("-")[1]
                orbs = re.findall(r'[A-Za-z]', orbital_types_str)
                nums = [int(x) for x in re.findall(r'\d+', orbital_types_str)]
                orb2l = {"s": 0, "p": 1, "d": 2, "f": 3, "g":4, "h":5}
                types = []
                for orb, num in zip(orbs, nums):
                    for _ in range(num):
                        types.append(orb2l[orb])
                orbital_types_element[atom_label] = types


            find_target_line(f, "Fractional coordinates of the final structure")
            f.readline()
            f.readline()
            f.readline()

            orbital_types = []
            element_list = []
            # atom_frac_pos = np.zeros((atomnum, 3)) # Not needed

            for i in range(atomnum):
                line = f.readline()
                parts = line.split()
                element_str = parts[1]
                orbital_types.append(orbital_types_element[element_str])
                element_list.append(ase.data.atomic_numbers[element_str])
                # atom_frac_pos[i, :] = [float(parts[2]), float(parts[3]), float(parts[4])]

            # atom_pos = np.dot(atom_frac_pos, tv) # Not needed

        data = {
            "atomnum": atomnum,
            "SpinP_switch": SpinP_switch,
            "atv_ijk": atv_ijk,
            "Total_NumOrbs": Total_NumOrbs,
            "FNAN": FNAN,
            "natn": natn,
            "ncn": ncn,
            "tv": tv, # List of frames
            "rtv": rtv,
            "Gxyz": Gxyz,
            "Hk": Hk, # List of frames
            "iHk": iHk, # List of frames
            "OLP": OLP, # List of frames
            "DM": DM, # List of frames
            "iDM": iDM, # List of frames
            "orbital_types": orbital_types,
            "element_list": element_list,
        }
        self._cache_idx = idx
        self._cache_data = data
        return data
