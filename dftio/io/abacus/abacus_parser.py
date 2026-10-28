from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
import re
from tqdm import tqdm
from collections import Counter
from dftio.constants import orbitalId, ABACUS2DFTIO
import ase
from ase.io import read
import dpdata
import os
import numpy as np
from dftio.io.parse import Parser, ParserRegister, find_target_line
from dftio.data import _keys
from dftio.register import Register
import lmdb
import logging
import glob

log = logging.getLogger(__name__)
import pickle
import shutil
import subprocess

@ParserRegister.register("abacus")
class AbacusParser(Parser):
    def __init__(
            self,
            root,
            prefix,
            **kwargs
            ):
        super(AbacusParser, self).__init__(root, prefix)
        self.raw_sys = []
        for idx in range(len(self.raw_datas)):
            mode = self.get_mode(idx)
            if mode in ['nscf', "scf"]:
                system = dpdata.System(
                    read(os.path.join(self._get_output_dir(idx), "STRU.cif")),
                    fmt="ase/structure",
                )
            elif mode.lower() == "get_s":
                stru_path = os.path.join(self.raw_datas[idx], "STRU")
                if not os.path.isfile(stru_path):
                    raise FileNotFoundError(
                        f"No ABACUS STRU file found for get_S calculation: {stru_path}"
                    )
                system = dpdata.System(stru_path, fmt="abacus/stru")
            else:
                system = dpdata.LabeledSystem(
                    self.raw_datas[idx], fmt='abacus/' + mode
                )
            self.raw_sys.append(system)

    def _get_output_dir(self, idx):
        """
        Get the ABACUS output directory for a given index.

        First checks for 'OUT.ABACUS', and if not found, searches for directories
        matching the pattern 'OUT.*' (e.g., 'OUT.suffix' when suffix is specified).

        Parameters
        ----------
        idx : int
            Index of the structure/trajectory

        Returns
        -------
        str
            Path to the output directory

        Raises
        ------
        FileNotFoundError
            If no suitable output directory is found
        """
        base_path = self.raw_datas[idx]

        # First try the default OUT.ABACUS directory
        out_abacus_path = os.path.join(base_path, "OUT.ABACUS")
        if os.path.exists(out_abacus_path):
            return out_abacus_path

        # If OUT.ABACUS doesn't exist, search for OUT.* pattern
        search_pattern = os.path.join(base_path, "OUT.*")
        out_dirs = glob.glob(search_pattern)

        # Filter to directories only
        out_dirs = [d for d in out_dirs if os.path.isdir(d)]

        if len(out_dirs) == 0:
            raise FileNotFoundError(f"No ABACUS output directory found in {base_path}. "
                                  f"Looked for OUT.ABACUS or OUT.* pattern.")
        elif len(out_dirs) == 1:
            log.info(f"Using output directory: {out_dirs[0]} (OUT.ABACUS not found, using OUT.* pattern)")
            return out_dirs[0]
        else:
            # If multiple directories found, prefer the one with most common suffixes
            # or the one that was most recently modified
            log.warning(f"Multiple output directories found: {out_dirs}. Using the most recently modified one.")
            out_dirs.sort(key=os.path.getmtime, reverse=True)
            return out_dirs[0]

    def _get_input_file(self, idx, prefer_output_dir=False):
        """
        Resolve the ABACUS input file path.

        Supports both legacy `INPUT` and newer `INPUT.info` names while
        preserving the original lookup order for each caller.
        """
        candidate_dirs = []
        if prefer_output_dir:
            candidate_dirs.append(self._get_output_dir(idx))
            candidate_dirs.append(self.raw_datas[idx])
        else:
            candidate_dirs.append(self.raw_datas[idx])
            candidate_dirs.append(self._get_output_dir(idx))

        for candidate_dir in candidate_dirs:
            for filename in ("INPUT", "INPUT.info"):
                input_path = os.path.join(candidate_dir, filename)
                if os.path.exists(input_path):
                    return input_path

        raise FileNotFoundError(
            f"No ABACUS input file found for {self.raw_datas[idx]}. "
            f"Looked for INPUT and INPUT.info in {candidate_dirs}."
        )

    # essential
    def get_structure(self, idx):
        sys = self.raw_sys[idx]
        
        structure = {
            _keys.ATOMIC_NUMBERS_KEY: np.array([ase.atom.atomic_numbers[i] for i in sys.data["atom_names"]], dtype=np.int32)[sys.data["atom_types"]],
            _keys.PBC_KEY: np.array([True, True, True]) # abacus does not allow non-pbc structure
        }
        structure[_keys.POSITIONS_KEY] = sys.data["coords"].astype(np.float32)
        structure[_keys.CELL_KEY] = sys.data["cells"].astype(np.float32)

        return structure
    
    def get_mode(self, idx):
        with open(self._get_input_file(idx, prefer_output_dir=True), 'r') as f:
            line = find_target_line(f, "calculation")
            assert line is not None, 'Cannot find "MODE" in log file'
            mode = line.split()[1]
            f.close()

        return mode

    def _parse_atom_label_line(self, line, logfile):
        lower_line = line.lower()
        assert "atom label =" in lower_line, 'Cannot find "atom label =" in log file'
        atom_label = line.split()[-1]
        atom_label = ''.join(re.findall(r'[A-Za-z]', atom_label))
        assert atom_label in ase.data.atomic_numbers, f"Atom label {atom_label} should be in periodic table ({logfile})"
        return atom_label

    def _parse_coordinate_atom_label(self, token, logfile):
        if "tau" in token.lower():
            atom_label = ''.join(re.findall(r'[A-Za-z]', token[5:]))
        else:
            atom_label = ''.join(re.findall(r'[A-Za-z]', token))
        assert atom_label in ase.data.atomic_numbers, f"Atom label {atom_label} should be in periodic table ({logfile})"
        return atom_label

    def _resolve_matrix_path(self, output_dir, *candidates):
        for candidate in candidates:
            matrix_path = os.path.join(output_dir, candidate)
            if os.path.exists(matrix_path):
                return matrix_path
        raise FileNotFoundError(
            f"No ABACUS matrix file found in {output_dir}. Tried: {candidates}"
        )

    def _resolve_output_file(self, output_dir, *candidates):
        for candidate in candidates:
            file_path = os.path.join(output_dir, candidate)
            if os.path.exists(file_path):
                return file_path
        raise FileNotFoundError(
            f"No ABACUS output file found in {output_dir}. Tried: {candidates}"
        )

    @staticmethod
    def _is_matrix_block_header(line):
        parts = line.split()
        if len(parts) != 4:
            return False
        try:
            [int(part) for part in parts]
        except ValueError:
            return False
        return True

    @staticmethod
    def _read_numeric_block(file_obj, count):
        tokens = []
        while len(tokens) < count:
            line = file_obj.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens.extend(stripped.split())
        if len(tokens) != count:
            raise ValueError(f"Expected {count} numeric entries, got {len(tokens)}")
        return tokens
    
    # essential
    def get_eigenvalue(self, idx, band_index_min=0):
        mode = self.get_mode(idx)
        if mode in ["scf", "nscf"]:
            output_dir = self._get_output_dir(idx)
            eig_path = self._resolve_output_file(output_dir, "BANDS_1.dat", "band.txt")
            if os.path.basename(eig_path) == "BANDS_1.dat":
                eigs = np.loadtxt(eig_path)[np.newaxis, :, 2 + band_index_min:]
            else:
                eigs = np.loadtxt(eig_path)[np.newaxis, :, 1 + band_index_min:]

            kpoint_path = self._resolve_output_file(output_dir, "kpoints", "KPT.info")
            kpts = []
            with open(kpoint_path, "r") as f:
                line = find_target_line(f, "nkstot now")
                nkstot = line.strip().split()[-1]
                line = find_target_line(f, "KPOINTS ")
                if not line:
                    f.seek(0)
                    line = find_target_line(f, "KPT ")
                if not line: 
                    raise Exception("Cannot find KPT or KPOINTS in kpoints file")
                for _ in range(int(nkstot)):
                    line = f.readline()
                    kpt = []
                    line = line.strip().split()
                    kpt.extend([float(line[1]), float(line[2]), float(line[3])])
                    kpts.append(kpt)
                kpts = np.array(kpts)
        elif mode == "md" or mode == "relax":
            raise NotImplementedError("output eigenvalues from MD trajectory is not supported by ABACUS.")
        
        else:
            raise NotImplementedError("mode {} is not supported.".format(mode))
            
        return {_keys.ENERGY_EIGENVALUE_KEY: eigs.astype(np.float32), _keys.KPOINT_KEY: kpts.astype(np.float32)}
    
    # essential
    def get_basis(self, idx):
        # {"Si": "2s2p1d"}
        mode = self.get_mode(idx)
        logfile = "running_"+mode+".log"
        sys = self.raw_sys[idx]
        with open(os.path.join(self._get_output_dir(idx), logfile), 'r') as f:
            orbital_types_dict = {}
            for index_type in range(len(sys.data["atom_numbs"])):
                tmp = find_target_line(f, "READING ATOM TYPE")
                assert tmp is not None, 'Cannot find "ATOM TYPE" in log file'
                assert tmp.split()[-1] == str(index_type + 1)
                if tmp is None:
                    raise Exception(f"Cannot find ATOM {index_type} in {logfile}")

                line = f.readline()
                atom_label = self._parse_atom_label_line(line, logfile)

                current_orbital_types = []
                while True:
                    line = f.readline()
                    if "number of zeta" in line:
                        tmp = line.split()
                        L = int(tmp[0][2:-1])
                        num_L = int(tmp[-1])
                        current_orbital_types.extend([L] * num_L)
                    else:
                        break
                orbital_types_dict[atom_label] = current_orbital_types
        basis = {}
        for k,v in orbital_types_dict.items():
            counter = Counter(v)
            basis[k] = [str(counter[l])+orbitalId[l] for l in range(max(counter.keys())+1) if counter.get(l, 0) > 0]
            basis[k] = "".join(basis[k])
        
        return basis
    
    # essential
    def get_blocks(self, idx, hamiltonian=True, overlap=False, density_matrix=False):
        mode = self.get_mode(idx)
        is_get_s = mode.lower() == "get_s"
        if is_get_s and (hamiltonian or density_matrix):
            requested = []
            if hamiltonian:
                requested.append("Hamiltonian")
            if density_matrix:
                requested.append("density matrix")
            raise ValueError(
                "ABACUS get_S calculations only provide the overlap matrix; "
                f"cannot extract {' and '.join(requested)}."
            )

        logfile = "running_"+mode+".log"
        hamiltonian_dict, overlap_dict, density_matrix_dict = None, None, None
        sys = self.raw_sys[idx]
        nsites = sys.data["atom_types"].shape[0]
        output_dir = self._get_output_dir(idx)
        if os.path.exists(os.path.join(output_dir, "hscsr.tgz")):
            # os.system(f"tar -xzf {os.path.join(output_dir, 'hscsr.tgz')} -C {os.path.join(self.raw_datas[idx])}")
            subprocess.run(
                ["tar", "-xzf", os.path.join(output_dir, 'hscsr.tgz'), "-C", os.path.join(self.raw_datas[idx])],
                check=True
            )
        with open(os.path.join(output_dir, logfile), 'r') as f:
            site_norbits_dict = {}
            orbital_types_dict = {}
            for index_type in range(len(sys.data["atom_numbs"])):
                tmp = find_target_line(f, "READING ATOM TYPE")
                assert tmp is not None, 'Cannot find "ATOM TYPE" in log file'
                assert tmp.split()[-1] == str(index_type + 1)
                if tmp is None:
                    raise Exception(f"Cannot find ATOM {index_type} in {logfile}")

                line = f.readline()
                atom_label = self._parse_atom_label_line(line, logfile)
                atom_type = ase.data.atomic_numbers[atom_label]

                current_site_norbits = 0
                current_orbital_types = []
                while True:
                    line = f.readline()
                    if "number of zeta" in line:
                        tmp = line.split()
                        L = int(tmp[0][2:-1])
                        num_L = int(tmp[-1])
                        current_site_norbits += (2 * L + 1) * num_L
                        current_orbital_types.extend([L] * num_L)
                    else:
                        break
                site_norbits_dict[atom_type] = current_site_norbits
                orbital_types_dict[atom_type] = current_orbital_types


            line = find_target_line(f, " COORDINATES")
            assert "atom" in f.readline()
            site_norbits = np.zeros(nsites, dtype=int)
            element = np.zeros(nsites, dtype=int)
            for index_site in range(nsites):
                line = f.readline()
                tmp = line.split()
                atom_label = self._parse_coordinate_atom_label(tmp[0], logfile)
                element[index_site] = ase.data.atomic_numbers[atom_label]
                site_norbits[index_site] = site_norbits_dict[element[index_site]]

            if hamiltonian is False and overlap is True:
                spinful = False
            else:
                line = find_target_line(f, "nspin")
                if line is None:
                    line = find_target_line(f, "NSPIN")
                assert line is not None, 'Cannot find "NSPIN" in log file'
                lower_line = line.lower()
                if "nspin == 1" in lower_line or "nspin = 1" in lower_line:
                    spinful = False
                elif "nspin == 4" in lower_line or "nspin = 4" in lower_line:
                    spinful = True
                else:
                    raise ValueError(f'{line} is not supported')

        if mode in ["scf", "nscf"] or is_get_s:
            if hamiltonian:
                hamiltonian_dict, tmp = self.parse_matrix(
                    matrix_path=self._resolve_matrix_path(output_dir, "data-HR-sparse_SPIN0.csr", "hrs1_nao.csr"), 
                    nsites=nsites,
                    site_norbits=site_norbits,
                    orbital_types_dict=orbital_types_dict,
                    element=element,
                    factor=13.605698, # Ryd2eV
                    spinful=spinful
                    )
                assert tmp == int(np.sum(site_norbits)) * (1 + spinful)
                hamiltonian_dict = [hamiltonian_dict]
            
            if overlap:
                overlap_candidates = (
                    ("SR.csr", "data-SR-sparse_SPIN0.csr", "srs1_nao.csr")
                    if is_get_s
                    else ("data-SR-sparse_SPIN0.csr", "srs1_nao.csr")
                )
                overlap_dict, tmp = self.parse_matrix(
                    matrix_path=self._resolve_matrix_path(output_dir, *overlap_candidates),
                    nsites=nsites,
                    site_norbits=site_norbits,
                    orbital_types_dict=orbital_types_dict,
                    element=element,
                    factor=1,
                    spinful=spinful
                    )
                assert tmp == int(np.sum(site_norbits)) * (1 + spinful)

                if spinful:
                    overlap_dict_spinless = {}
                    for k, v in overlap_dict.items():
                        overlap_dict_spinless[k] = v[:v.shape[0] // 2, :v.shape[1] // 2].real
                    overlap_dict_spinless, overlap_dict = overlap_dict, overlap_dict_spinless

                overlap_dict = [overlap_dict]

            if density_matrix:
                density_matrix_dict, tmp = self.parse_matrix(
                    matrix_path=self._resolve_matrix_path(output_dir, "data-DMR-sparse_SPIN0.csr", "dmrs1_nao.csr"), 
                    nsites=nsites,
                    site_norbits=site_norbits,
                    orbital_types_dict=orbital_types_dict,
                    element=element,
                    factor=1,
                    spinful=spinful
                    )
                assert tmp == int(np.sum(site_norbits)) * (1 + spinful)

                density_matrix_dict = [density_matrix_dict]

        elif mode == "md":
            if hamiltonian:
                hamiltonian_dict = []
                for i in range(sys.get_nframes()):
                    if os.path.exists(os.path.join(output_dir, "matrix/")):
                        hamil, tmp = self.parse_matrix(
                            matrix_path=self._resolve_matrix_path(output_dir, "matrix/"+str(i)+"_data-HR-sparse_SPIN0.csr"), 
                            nsites=nsites,
                            site_norbits=site_norbits,
                            orbital_types_dict=orbital_types_dict,
                            element=element,
                            factor=13.605698, # Ryd2eV
                            spinful=spinful
                            )
                    else:
                        hamil, tmp = self.parse_matrix(
                            matrix_path=self._resolve_matrix_path(output_dir, "data-HR-sparse_SPIN0.csr", "hrs1_nao.csr"), 
                            nsites=nsites,
                            site_norbits=site_norbits,
                            orbital_types_dict=orbital_types_dict,
                            element=element,
                            factor=13.605698, # Ryd2eV
                            spinful=spinful,
                            step=i
                            )
                    assert tmp == int(np.sum(site_norbits)) * (1 + spinful)
                    hamiltonian_dict.append(hamil)

            if overlap:
                overlap_dict = []
                for i in range(sys.get_nframes()):
                    if os.path.exists(os.path.join(output_dir, "matrix/")):
                        ovp, tmp = self.parse_matrix(
                            matrix_path=self._resolve_matrix_path(output_dir, "matrix/"+str(i)+"_data-SR-sparse_SPIN0.csr"), 
                            nsites=nsites,
                            site_norbits=site_norbits,
                            orbital_types_dict=orbital_types_dict,
                            element=element,
                            factor=1,
                            spinful=spinful
                            )
                    else:
                        ovp, tmp = self.parse_matrix(
                            matrix_path=self._resolve_matrix_path(output_dir, "data-SR-sparse_SPIN0.csr", "srs1_nao.csr"), 
                            nsites=nsites,
                            site_norbits=site_norbits,
                            orbital_types_dict=orbital_types_dict,
                            element=element,
                            factor=1,
                            spinful=spinful,
                            step=i
                            )
                    assert tmp == int(np.sum(site_norbits)) * (1 + spinful)

                    if spinful:
                        ovp_spinless = {}
                        for k, v in ovp.items():
                            ovp_spinless[k] = v[:v.shape[0] // 2, :v.shape[1] // 2].real

                        ovp_spinless, ovp = ovp, ovp_spinless
                    overlap_dict.append(ovp)
            
            if density_matrix:
                density_matrix_dict = []
                for i in range(sys.get_nframes()):
                    if os.path.exists(os.path.join(output_dir, "matrix/")):
                        dm, tmp = self.parse_matrix(
                            matrix_path=self._resolve_matrix_path(output_dir, "matrix/"+str(i)+"_data-DMR-sparse_SPIN0.csr"), 
                            nsites=nsites,
                            site_norbits=site_norbits,
                            orbital_types_dict=orbital_types_dict,
                            element=element,
                            factor=1,
                            spinful=spinful
                            )
                    else:
                        dm, tmp = self.parse_matrix(
                            matrix_path=self._resolve_matrix_path(output_dir, "data-DMR-sparse_SPIN0.csr", "dmrs1_nao.csr"), 
                            nsites=nsites,
                            site_norbits=site_norbits,
                            orbital_types_dict=orbital_types_dict,
                            element=element,
                            factor=1,
                            spinful=spinful,
                            step=i
                            )

                    assert tmp == int(np.sum(site_norbits)) * (1 + spinful)
                    density_matrix_dict.append(dm)
        else:
            raise NotImplementedError("mode {} is not supported.".format(mode))
        
        return hamiltonian_dict, overlap_dict, density_matrix_dict

    def parse_matrix(self, matrix_path, nsites, site_norbits, orbital_types_dict, element, factor, spinful=False, step=0):
        site_norbits_cumsum = np.cumsum(site_norbits)
        norbits = int(np.sum(site_norbits))
        matrix_dict = dict()
        with open(matrix_path, 'r') as f:
            line = f.readline()
            first_matrix_line = None

            if "Matrix Dimension of" in line:
                assert step == 0
                f.readline() # read "Matrix number of ..."
                norbits = int(line.split()[-1])
            elif "STEP" in line:
                # ABACUS >= 3.0 sparse matrix format
                step_found = False
                while line and not step_found:
                    if "STEP" in line:
                        stp = int(line.split()[-1])
                        if stp == step:
                            step_found = True
                        else:
                            line = f.readline()
                    else:
                        line = f.readline()
                line = f.readline()
                assert "Matrix Dimension of" in line
                f.readline() # read "Matrix number of ..."
                norbits = int(line.split()[-1])
            else:
                # Newer density-matrix files contain a descriptive header before CSR data.
                while line:
                    if "# number of localized basis" in line:
                        norbits = int(line.split()[0])
                    if self._is_matrix_block_header(line):
                        first_matrix_line = line
                        break
                    line = f.readline()

            line = first_matrix_line if first_matrix_line else f.readline()
            while line:
                line1 = line.split()
                if len(line1) == 0 or len(line1) == 2:
                    break
                num_element = int(line1[3])
                if num_element != 0:
                    R_cur = np.array(line1[:3]).astype(int)
                    line2 = self._read_numeric_block(f, num_element)
                    line3 = self._read_numeric_block(f, num_element)
                    line4 = self._read_numeric_block(f, norbits + 1)
                    if not spinful:
                        hamiltonian_cur = csr_matrix((np.array(line2).astype(np.float32), np.array(line3).astype(int),
                                                        np.array(line4).astype(np.int32)), shape=(norbits, norbits), dtype=np.float32).toarray()
                    else:
                        line2 = np.char.replace(line2, '(', '')
                        line2 = np.char.replace(line2, ')', 'j')
                        line2 = np.char.replace(line2, ',', '+')
                        line2 = np.char.replace(line2, '+-', '-')
                        hamiltonian_cur = csr_matrix((np.array(line2).astype(np.complex64), np.array(line3).astype(int),
                                                    np.array(line4).astype(np.int32)), shape=(norbits, norbits), dtype=np.complex64).toarray()
                    for index_site_i in range(nsites):
                        for index_site_j in range(nsites):
                            key_str = f"{index_site_i}_{index_site_j}_{R_cur[0]}_{R_cur[1]}_{R_cur[2]}"
                            mat = hamiltonian_cur[(site_norbits_cumsum[index_site_i]
                                                    - site_norbits[index_site_i]) * (1 + spinful):
                                                    site_norbits_cumsum[index_site_i] * (1 + spinful),
                                    (site_norbits_cumsum[index_site_j] - site_norbits[index_site_j]) * (1 + spinful):
                                    site_norbits_cumsum[index_site_j] * (1 + spinful)]
                            if abs(mat).max() < 1e-10:
                                continue
                            if not spinful:
                                mat = self.transform(mat, orbital_types_dict[element[index_site_i]],
                                                            orbital_types_dict[element[index_site_j]])
                            else:
                                mat = mat.reshape((site_norbits[index_site_i], 2, site_norbits[index_site_j], 2))
                                mat = mat.transpose((1, 0, 3, 2)).reshape((2 * site_norbits[index_site_i],
                                                                        2 * site_norbits[index_site_j]))
                                mat = self.transform(mat, orbital_types_dict[element[index_site_i]] * 2,
                                                        orbital_types_dict[element[index_site_j]] * 2)
                            matrix_dict[key_str] = mat * factor
                line = f.readline()
        return matrix_dict, norbits
    
    def transform(self, mat, l_lefts, l_rights):

        if max(*l_lefts, *l_rights) > 5:
            raise NotImplementedError("Only support l = s, p, d, f, g, h.")
        block_lefts = block_diag(*[ABACUS2DFTIO[l_left] for l_left in l_lefts])
        block_rights = block_diag(*[ABACUS2DFTIO[l_right] for l_right in l_rights])

        return block_lefts @ mat @ block_rights.T

    @staticmethod
    def _extract_energy_from_log(loglines, mode, dump_freq=1):
        """
        Extract total energy from ABACUS log file.

        Note that this extractor is only validated for ABACUS versions 3.9.0.

        For SCF/NSCF, extracts the final total energy.
        For MD, extracts energies at each dump interval, filtering out unconverged frames.
        For RELAX, extracts energies for the final converged structure.

        Parameters
        ----------
        loglines : list of str
            Lines from the log file
        mode : str
            Calculation mode (scf, nscf, md, relax)
        dump_freq : int, optional
            Dump frequency for MD calculations (default: 1)

        Returns
        -------
        tuple or None
            Tuple of (energy_array, unconverged_indices) where:
            - energy_array: np.ndarray of energies in eV
              * SCF/NSCF: shape (1,)
              * MD: energies for converged frames at dump intervals
              * RELAX: energies for converged relaxation steps.
            - unconverged_indices: list of frame indices that did not converge
              (currently only used for MD).

            Returns None if extraction fails or all frames are unconverged. In RELAX
            mode a ValueError is raised instead when relaxation does not converge or
            no energies are found.
        """
        energy = []

        if mode in ["scf", "nscf"]:
            # For SCF/NSCF, search from the end for the final energy
            for line in reversed(loglines):
                if "final etot is" in line:
                    # LTS version format: "final etot is <value> eV"
                    Etot = float(line.split()[-2])
                    return (np.array([Etot], dtype=np.float64), [])
                elif "TOTAL ENERGY" in line:
                    # Develop version format
                    Etot = float(line.split()[-2])
                    return (np.array([Etot], dtype=np.float64), [])
                elif "convergence has NOT been achieved!" in line or \
                     "convergence has not been achieved" in line:
                    # SCF did not converge
                    return None
            return None

        elif mode == "md":
            # For MD, extract all energies at dump intervals
            nenergy = 0
            for line in loglines:
                if "final etot is" in line:
                    if nenergy % dump_freq == 0:
                        energy.append(float(line.split()[-2]))
                    nenergy += 1
                elif "!! convergence has not been achieved" in line:
                    if nenergy % dump_freq == 0:
                        energy.append(np.nan)
                    nenergy += 1

            if len(energy) == 0:
                return None

            # Filter out unconverged frames (NaN values) and track their indices
            energy = np.array(energy, dtype=np.float64)
            valid_mask = ~np.isnan(energy)
            if not valid_mask.any():
                return None

            # Get indices of unconverged frames
            unconverged_indices = np.where(~valid_mask)[0].tolist()

            # Filter to keep only converged frames
            energy = energy[valid_mask]
            return (energy, unconverged_indices)

        elif mode == "relax":
            relax_success = False
            # For RELAX, extract energy for the converged structures
            for line in loglines:
                if "Relaxation is converged!" in line:
                    relax_success = True
                if "!FINAL_ETOT_IS" in line:
                    energy.append(float(line.split()[-2]))

            if relax_success and len(energy) > 0:
                return (np.array(energy, dtype=np.float64), [])
            else:
                # raise error when relaxation did not converge or no energies found
                raise ValueError("Relaxation did not converge or no energies found.")

        else:
            return None

    def get_etot(self, idx):
        """
        Extract total energy (Etot) from ABACUS output.

        Parameters
        ----------
        idx : int
            Index of the structure/trajectory

        Returns
        -------
        dict
            Dictionary with:
            - _keys.TOTAL_ENERGY_KEY: energy array in eV
              * SCF/NSCF: shape (1,)
              * MD/RELAX: length equals the number of converged frames at dump intervals.
            - _keys.UNCONVERGED_FRAME_INDICES_KEY: list of unconverged frame indices
              (empty if all frames converged).
            Returns None if energy extraction fails or all frames are unconverged.
            In RELAX mode, a ValueError is raised if relaxation did not converge or
            no energies were found.
        """
        mode = self.get_mode(idx)
        logfile = "running_" + mode + ".log"
        logpath = os.path.join(self._get_output_dir(idx), logfile)

        # Check if log file exists
        if not os.path.exists(logpath):
            raise FileNotFoundError(f"Log file {logpath} does not exist.")

        # Read log file
        with open(logpath, 'r') as f:
            loglines = f.readlines()

        # Get dump frequency for MD mode
        dump_freq = 1
        if mode == "md":
            input_path = self._get_input_file(idx)
            with open(input_path, 'r') as f:
                for line in f:
                    if len(line) > 0 and "md_dumpfreq" in line and "md_dumpfreq" == line.split()[0]:
                        dump_freq = int(line.split()[1])
                        break

        # Extract energy
        result = self._extract_energy_from_log(loglines, mode, dump_freq)

        if result is None:
            return None

        energy, unconverged_indices = result

        # Log warning if there are unconverged frames
        if len(unconverged_indices) > 0:
            log.warning(f"Energy extraction: frames {unconverged_indices} did not converge")

        return {
            _keys.TOTAL_ENERGY_KEY: energy,
            _keys.UNCONVERGED_FRAME_INDICES_KEY: unconverged_indices
        }

    def get_abs_h0_folders(self, h0_root):
        # Build a map of all directory names to their full paths to avoid repeated os.walk calls
        folder_path_map = {}
        for sub_root, dirs, _ in os.walk(h0_root):
            for dirname in dirs:
                folder_path_map[dirname] = os.path.join(sub_root, dirname)

        abs_h0_folders = []
        valid_idx_list = []
        for valid_idx, a_H_folder in enumerate(self.raw_datas):
            a_leaf_folder_name = os.path.split(a_H_folder)[-1]

            # Use pre-built map for O(1) lookup instead of calling slow find_leaf_folder()
            a_leaf_folder_path = folder_path_map.get(a_leaf_folder_name)
            found_flag = a_leaf_folder_path is not None

            if found_flag:
                abs_h0_folders.append(a_leaf_folder_path)
                valid_idx_list.append(valid_idx)
            else:
                abs_h0_folders.append(None)
                valid_idx_list.append(None)
        return abs_h0_folders, valid_idx_list

    def add_h0_delta_h(self, h0_src_root, old_lmdb_path, new_lmdb_path, keep_old_lmdb: bool = True,
                       keep_delta_ham_only: bool = True):
        h0_root = os.path.abspath(h0_src_root)
        self.raw_datas, valid_idx_list = self.get_abs_h0_folders(h0_root=h0_root)

        os.makedirs(new_lmdb_path, exist_ok=True)
        counter = 0
        batch_size = 50

        # Process in batches
        for batch_start in tqdm(range(0, len(valid_idx_list), batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(valid_idx_list))
            batch_indices = valid_idx_list[batch_start:batch_end]

            # Open LMDB environments for each batch
            old_db_env = lmdb.open(old_lmdb_path, readonly=True, lock=False)
            new_db_env = lmdb.open(new_lmdb_path, map_size=1048576000000, lock=True)

            with old_db_env.begin() as old_txn, new_db_env.begin(write=True) as new_txn:
                for idx in batch_indices:
                    if idx == None:
                        continue
                    # Get data from old LMDB
                    data_dict = old_txn.get(idx.to_bytes(length=4, byteorder='big'))
                    data_dict = pickle.loads(data_dict)

                    # Get H0 block
                    h0_block, _0, _ = self.get_blocks(idx, hamiltonian=True, overlap=False, density_matrix=False)
                    h0_block = h0_block[0]
                    old_ham_block = data_dict['hamiltonian']

                    # Calculate delta blocks
                    delta_block = dict()
                    for a_block_name in old_ham_block.keys():
                        a_delta_block = old_ham_block[a_block_name] - h0_block[a_block_name]
                        delta_block[a_block_name] = a_delta_block

                    # Update data dictionary
                    data_dict['hamiltonian'] = delta_block
                    if not keep_delta_ham_only:
                        data_dict['hamiltonian_full'] = old_ham_block
                        data_dict['hamiltonian_0'] = h0_block

                    # Store in new LMDB
                    data_dict = pickle.dumps(data_dict)
                    new_txn.put(counter.to_bytes(length=4, byteorder='big'), data_dict)
                    counter = counter + 1

            # Close LMDB environments after each batch
            old_db_env.close()
            new_db_env.close()

        # Remove old LMDB if requested
        if not keep_old_lmdb:
            shutil.rmtree(old_lmdb_path)
