import os
import glob
import json

from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import h5py
import ase
import pickle
import lmdb
from dftio.data import _keys
from dftio.utils import j_must_have
from dftio.register import Register
from ase.io.trajectory import Trajectory
import logging
log = logging.getLogger(__name__)


def resolve_output_prefix(output_prefix, default):
    """Return a safe filename prefix without allowing path components."""
    if output_prefix is None:
        return default
    if not output_prefix or output_prefix in (".", ".."):
        raise ValueError("output_prefix must be a non-empty filename prefix")
    if "/" in output_prefix or "\\" in output_prefix:
        raise ValueError("output_prefix must not contain path separators")
    return output_prefix


def find_target_line(f, target):
    line = f.readline()
    while line:
        if target in line:
            return line
        line = f.readline()
    return None


class ParserRegister:
    _register = Register()

    def register(target):
        return ParserRegister._register.register(target)
    
    def __new__(cls, mode: str, **kwargs):
        if mode in ParserRegister._register.keys():
            return ParserRegister._register[mode](**kwargs)
        else:
            raise Exception(f"Descriptor mode: {mode} is not registered!")

class Parser(ABC):
    def __init__(
            self,
            root,
            prefix,
            **kwargs
            ):
        """All DFT parser need to inherit this class and implement the abstract methods.
        The subclass can call the check_structure, check_eigenvalue, check_blocks to check the parsed data structure.
        To see if the parsed data structure are in the right shape and dtype.

        Parameters
        ----------
        root : The root of the DFT calculation output files
        prefix : str
            The prefix of the DFT calculation folders or files
        """

        self.root = root
        self.prefix = prefix
        self.failed_inputs = []
        if '/' in prefix:
            log.warning(f"There are '/' in the prefix: {prefix}, which should not appear. You should consider use '-r' "
                        f"to change the root rather than include path in the prefix, which might cause error.")

        if isinstance(root, list) and all(isinstance(item, str) for item in root):
            self.raw_datas = root
        else:
            self.raw_datas = glob.glob(os.path.join(root, '*' + prefix + '*'))

        assert(len(self.raw_datas) != 0, 'There are no folders that meet the requirements in the directory!')

    def _load_valid_inputs(self, loader):
        """Load inputs eagerly while excluding individual failures."""
        valid_raw_datas = []
        loaded_inputs = []
        for idx, raw_data in enumerate(self.raw_datas):
            try:
                loaded_input = loader(idx)
            except Exception:
                self.failed_inputs.append(os.path.basename(os.path.normpath(raw_data)))
                continue
            valid_raw_datas.append(raw_data)
            loaded_inputs.append(loaded_input)
        self.raw_datas = valid_raw_datas
        return loaded_inputs
    
    def __len__(self):
        return len(self.raw_datas)
    
    def __getitem__(self, idx):
        return self.raw_datas[idx]
    
    @abstractmethod
    def get_structure(self, idx):
        pass # return a dict of structure information, keys includes: [_keys.ATOMIC_NUMBERS_KEY, _keys.POSITIONS_KEY, _keys.CELL_KEY, _keys.PBC_KEY], CELL could be None if no PBC is applied
    
    # def get_hash(self, idx):

    #     buffer = yaml.dump(self.get_structure(idx=idx)).encode("ascii")
    #     # And hash it:
    #     param_hash = hashlib.sha1(buffer).hexdigest()
    #     return param_hash
    
    def formula(self, idx):
        structure = self.get_structure(idx)
        atomic_number = j_must_have(structure, _keys.ATOMIC_NUMBERS_KEY)

        return ase.Atoms(numbers=atomic_number).get_chemical_formula()

    def structure_to_ase(self, structure):
        self.check_structure(idx=None, structure=structure)
        cell = structure[_keys.CELL_KEY]
        nframe = cell.shape[0] 
        ase_lists = []
        for i in range(nframe):
            ase_lists.append(ase.Atoms(
                numbers=structure[_keys.ATOMIC_NUMBERS_KEY],
                positions=structure[_keys.POSITIONS_KEY][i],
                pbc=structure[_keys.PBC_KEY],
                cell=cell[i]
            ))
        return ase_lists
    
    def ase_to_structure(self, sys):
        if isinstance(sys, list):
            pos = []
            cell = []
            pbc = sys[0].pbc
            atom_numbs = sys[0].get_atomic_numbers()
            for s in sys:
                assert isinstance(s, ase.Atoms), "The input system is not a list of ase.Atoms object!"
                pos.append(s.positions)
                cell.append(s.cell)
                assert np.all(s.pbc == pbc), "The input system is not a list of ase.Atoms object with same PBC!"
                assert np.all(s.get_atomic_numbers() == atom_numbs), "The input system is not a list of ase.Atoms object with same atomic numbers!"
            
            structure = {
                _keys.ATOMIC_NUMBERS_KEY: atom_numbs.astype(np.int32),
                _keys.PBC_KEY: pbc,
                _keys.POSITIONS_KEY: np.array(pos).astype(np.float32),
                _keys.CELL_KEY: np.array(cell).astype(np.float32)
            }

        elif isinstance(sys, ase.Atoms):
            structure = {
            _keys.ATOMIC_NUMBERS_KEY: sys.get_atomic_numbers().astype(np.int32),
            _keys.PBC_KEY: sys.pbc,
            _keys.POSITIONS_KEY: sys.positions[np.newaxis,:,:].astype(np.float32),
            _keys.CELL_KEY: sys.cell[np.newaxis,:,:].astype(np.float32)
            }
        else:
            raise ValueError("The input system is not a list or ase.Atoms object!")
        
        self.check_structure(idx=None, structure=structure)

        return structure
    
    @abstractmethod
    def get_eigenvalue(self, idx, band_index_min):
        pass # return a dict with energy eigenvalue and kpoint as key, keys includes: [_keys.ENERGY_EIGENVALUE_KEY, _keys.KPOINT_KEY]

    @abstractmethod
    def get_blocks(self, idx, hamiltonian: bool=False, overlap: bool=False, density_matrix: bool=False):
        pass # return a list of hamiltonian, overlap, density_matrix dict, with i_j_Rx_Ry_Rz as key, and the block as value

    # @abstractmethod
    # def get_field():
    #     pass
    
    def check_structure(self, idx, structure=None):
        if structure is None:
            structure = self.get_structure(idx)
        atomic_number = j_must_have(structure, _keys.ATOMIC_NUMBERS_KEY)
        pos = j_must_have(structure, _keys.POSITIONS_KEY)
        cell = j_must_have(structure, _keys.CELL_KEY)
        pbc = j_must_have(structure, _keys.PBC_KEY)

        # check the shape
        assert len(pos.shape) == 3 and pos.shape[2] == 3, "The shape of structure should be (n_frame, natom, 3)"
        n_frame, natom, _ = pos.shape
        assert len(cell.shape) == 3 and cell.shape[0] == n_frame and cell.shape[1] == 3 and cell.shape[2] == 3, "The shape of cell should be (n_frame, 3, 3)"
        assert len(atomic_number.shape) == 1 and atomic_number.shape[0] == natom, "The shape of atomic_number should be (natom,)"
        assert len(pbc.shape) == 1 and pbc.shape[0] == 3, "The shape of pbc should be (3)"
        
        # check dtype
        assert atomic_number.dtype == np.int32, "The dtype of atomic_number should be int"
        assert cell.dtype == np.float32, "The dtype of cell should be float"
        assert pos.dtype == np.float32, "The dtype of pos should be float"
        assert pbc.dtype == np.bool8, "The dtype of pbc should be bool"

        return True
    
    def check_eigenvalue(self, idx, eigstatus=None):
        if eigstatus is None:
            eigstatus = self.get_eigenvalue(idx)
        eigs = j_must_have(eigstatus, _keys.ENERGY_EIGENVALUE_KEY)
        kpts = j_must_have(eigstatus, _keys.KPOINT_KEY)

        # check the shape
        assert len(eigs.shape) == 3, "The shape of eigenvalue should be (n_frame, n_kpt, n_band)"
        nf, nk, nb = eigs.shape
        assert len(kpts.shape) == 2 and kpts.shape[0] == nk and kpts.shape[1] == 3, "The shape of kpoint should be (n_kpt, 3)"

        # check dtype
        assert eigs.dtype == np.float32, "The dtype of eigenvalue should be float"
        assert kpts.dtype == np.float32, "The dtype of kpoint should be float"

        return True
    
    def check_blocks(self, idx, hamiltonian: bool=False, overlap: bool=False, density_matrix: bool=False):
        ham, ovp, dm = self.get_blocks(idx, hamiltonian, overlap, density_matrix)
        structure = self.get_structure(idx)
        if hamiltonian:
            assert ham is not None, "Hamiltonian should not be None"
            assert len(ham) == structure[_keys.POSITIONS_KEY].shape[0], "The number of hamiltonian blocks should be equal to the number of frames"

            assert all([isinstance(h, dict) for h in ham]), "Hamiltonian should be a list of dict"
            assert ham[0].get("0_0_0_0_0") is not None, "Hamiltonian should at least have key 0_0_0_0_0"
            assert ham[0].get("0_0_0_0_0").dtype in [np.float32, np.complex64], "The dtype of hamiltonian block should be float real or complex"

        if overlap:
            assert ovp is not None, "Overlap should not be None"
            assert len(ovp) == structure[_keys.POSITIONS_KEY].shape[0], "The number of overlap blocks should be equal to the number of frames"
        if density_matrix:
            assert dm is not None, "Density matrix should not be None"
            assert len(dm) == structure[_keys.POSITIONS_KEY].shape[0], "The number of density matrix blocks should be equal to the number of frames"

        return True
    
    def write(self, idx, outroot, format, eigenvalue, hamiltonian, overlap, density_matrix, band_index_min, energy=False, output_prefix=None, **kwargs):
        if format == "hdf5":
            self.write_hdf5(idx=idx, outroot=outroot, eigenvalue=eigenvalue, hamiltonian=hamiltonian, overlap=overlap, density_matrix=density_matrix,band_index_min=band_index_min, energy=energy)
        elif format in ["dat", "ase"]:
            self.write_dat(idx=idx, outroot=outroot, fmt=format, eigenvalue=eigenvalue, hamiltonian=hamiltonian, overlap=overlap, density_matrix=density_matrix,band_index_min=band_index_min, energy=energy, output_prefix=output_prefix)
        elif format == "lmdb":
            self.write_lmdb(idx=idx, outroot=outroot, eigenvalue=eigenvalue, hamiltonian=hamiltonian, overlap=overlap, density_matrix=density_matrix,band_index_min=band_index_min, energy=energy, output_prefix=output_prefix)
        else:
            raise NotImplementedError(f"Format: {format} is not implemented!")
        
    def write_hdf5(self, idx, outroot, eigenvalue: bool=False, hamiltonian: bool=False, overlap: bool=False, density_matrix: bool=False, band_index_min=0):
        pass
    
    def write_struct(self, structure, out_dir, fmt='dat'):
        # write structure
        if fmt == 'dat':
            # The abacus must have PBC, so here we save cell by default
            np.savetxt(os.path.join(out_dir, "cell.dat"), structure[_keys.CELL_KEY].reshape(-1, 3))
            np.savetxt(os.path.join(out_dir, "positions.dat"), structure[_keys.POSITIONS_KEY].reshape(-1, 3))
            np.savetxt(os.path.join(out_dir, "atomic_numbers.dat"), structure[_keys.ATOMIC_NUMBERS_KEY], fmt='%d')
            np.savetxt(os.path.join(out_dir, "pbc.dat"), structure[_keys.PBC_KEY])
        
        elif fmt=='ase':
            ase_list = self.structure_to_ase(structure)
            trajfile = Trajectory(os.path.join(out_dir, "xdat.traj"), 'w')
            for istr in ase_list:
                trajfile.write(istr)
            trajfile.close()
        else:
            raise NotImplementedError(f"Format: {fmt} is not implemented!")

    def write_info(self, out_dir, structure, fmt='dat', eigstatus=None):
        """Write metadata describing the exported structure and eigenvalues.

        Eigenvalues have already been sliced by ``band_index_min`` in each
        parser before they reach this method.  Consequently, the exported
        array has its own zero-based band indices and ``band_max`` is the
        exclusive upper bound used by DeePTB.
        """
        positions = structure[_keys.POSITIONS_KEY]
        atomic_numbers = structure[_keys.ATOMIC_NUMBERS_KEY]
        pbc = structure[_keys.PBC_KEY]
        info = {
            "nframes": int(positions.shape[0]),
            "natoms": int(atomic_numbers.shape[0]),
            "pos_type": "ase" if fmt == "ase" else "cart",
            "pbc": np.asarray(pbc, dtype=bool).tolist(),
        }
        if eigstatus is not None:
            eigenvalues = eigstatus[_keys.ENERGY_EIGENVALUE_KEY]
            info["bandinfo"] = {
                "band_min": 0,
                "band_max": int(eigenvalues.shape[-1]),
                "emin": None,
                "emax": None,
            }

        with open(os.path.join(out_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=4)
            f.write("\n")

    def write_dat(self, idx, outroot, fmt='dat', eigenvalue=False, hamiltonian=False, overlap=False, density_matrix=False, band_index_min=0, energy=False, output_prefix=None):
        # write structure
        os.makedirs(outroot, exist_ok=True)

        structure = self.get_structure(idx)

        output_prefix = resolve_output_prefix(output_prefix, self.formula(idx=idx))
        out_dir = os.path.join(outroot, f"{output_prefix}.{idx}")
        os.makedirs(out_dir, exist_ok=True)
        # The abacus must have PBC, so here we save cell by default
        # np.savetxt(os.path.join(out_dir, "cell.dat"), structure[_keys.CELL_KEY].reshape(-1, 3))
        # np.savetxt(os.path.join(out_dir, "positions.dat"), structure[_keys.POSITIONS_KEY].reshape(-1, 3))
        # np.savetxt(os.path.join(out_dir, "atomic_numbers.dat"), structure[_keys.ATOMIC_NUMBERS_KEY], fmt='%d')
        # np.savetxt(os.path.join(out_dir, "pbc.dat"), structure[_keys.PBC_KEY])

        # write structure
        self.write_struct(structure, out_dir, fmt=fmt)

        # write eigenvalue
        if eigenvalue:
            eigstatus = self.get_eigenvalue(idx=idx, band_index_min=band_index_min)
            self.check_eigenvalue(idx=idx, eigstatus=eigstatus)
            np.save(os.path.join(out_dir, "kpoints.npy"), eigstatus[_keys.KPOINT_KEY])
            np.save(os.path.join(out_dir, "eigenvalues.npy"), eigstatus[_keys.ENERGY_EIGENVALUE_KEY])

        # write energy
        if energy:
            if hasattr(self, 'get_etot'):
                energy_data = self.get_etot(idx)
                if energy_data is not None:
                    np.savetxt(os.path.join(out_dir, "total_energy.dat"), energy_data[_keys.TOTAL_ENERGY_KEY])

                    # Write unconverged frame indices if present
                    if _keys.UNCONVERGED_FRAME_INDICES_KEY in energy_data:
                        unconverged_indices = energy_data[_keys.UNCONVERGED_FRAME_INDICES_KEY]
                        if len(unconverged_indices) > 0:
                            with open(os.path.join(out_dir, "unconverged_frames.dat"), 'w') as f:
                                f.write("# Frame indices that did not converge during MD/RELAX\n")
                                for idx_frame in unconverged_indices:
                                    f.write(f"{idx_frame}\n")
                else:
                    log.warning(f"Failed to extract energy for structure {idx}")
            else:
                log.warning(f"Parser does not implement get_etot method")

        # write blocks
        if any([hamiltonian is not None, overlap is not None, density_matrix is not None]) and any([hamiltonian, overlap, density_matrix]):
            with open(os.path.join(out_dir, "basis.dat"), 'w') as f:
               f.write(str(self.get_basis(idx)))

            ham, ovp, dm = self.get_blocks(idx, hamiltonian, overlap, density_matrix)
            if hamiltonian:
                with h5py.File(os.path.join(out_dir, "hamiltonians.h5"), 'w') as fid:
                    for i in range(len(ham)):
                        default_group = fid.create_group(str(i))
                        for key_str, value in ham[i].items():
                            default_group.create_dataset(key_str, data=value)
            del ham

            if overlap:
                with h5py.File(os.path.join(out_dir, "overlaps.h5"), 'w') as fid:
                    for i in range(len(ovp)):
                        default_group = fid.create_group(str(i))
                        for key_str, value in ovp[i].items():
                            default_group.create_dataset(key_str, data=value)
            del ovp

            if density_matrix:
                with h5py.File(os.path.join(out_dir, "density_matrices.h5"), 'w') as fid:
                    for i in range(len(dm)):
                        default_group = fid.create_group(str(i))
                        for key_str, value in dm[i].items():
                            default_group.create_dataset(key_str, data=value)

            del dm

        self.write_info(
            out_dir,
            structure,
            fmt=fmt,
            eigstatus=eigstatus if eigenvalue else None,
        )

        return True
    
    def write_lmdb(self, idx, outroot, eigenvalue: bool=False, hamiltonian: bool=False, overlap: bool=False, density_matrix: bool=False,band_index_min=0, energy: bool=False, output_prefix=None):
        os.makedirs(outroot, exist_ok=True)
        output_prefix = resolve_output_prefix(output_prefix, "data")
        out_dir = os.path.join(outroot, f"{output_prefix}.{os.getpid()}.lmdb")
        structure = self.get_structure(idx)
        if any([hamiltonian, overlap, density_matrix]):
            ham, ovp, dm = self.get_blocks(idx, hamiltonian, overlap, density_matrix)
        if eigenvalue:
            eigstatus = self.get_eigenvalue(idx=idx, band_index_min=band_index_min)
        if energy:
            if hasattr(self, 'get_etot'):
                energy_data = self.get_etot(idx)
            else:
                energy_data = None
                log.warning(f"Parser does not implement get_etot method")

        # Build frame index mapping for energy data
        # If there are unconverged frames, energy array will be shorter than n_frames
        energy_frame_mapping = None
        if energy and energy_data is not None:
            unconverged_indices = energy_data.get(_keys.UNCONVERGED_FRAME_INDICES_KEY, [])
            if len(unconverged_indices) > 0:
                # Build mapping: structure_frame_idx -> energy_array_idx
                energy_frame_mapping = {}
                energy_idx = 0
                n_frames_total = structure[_keys.POSITIONS_KEY].shape[0]
                for frame_idx in range(n_frames_total):
                    if frame_idx not in unconverged_indices:
                        energy_frame_mapping[frame_idx] = energy_idx
                        energy_idx += 1

        n_frames = structure[_keys.POSITIONS_KEY].shape[0]
        lmdb_env = lmdb.open(out_dir, map_size=1048576000000, lock=True)
        for nf in range(n_frames):
            data_dict = {}
            data_dict[_keys.ATOMIC_NUMBERS_KEY] = structure[_keys.ATOMIC_NUMBERS_KEY]
            data_dict[_keys.CELL_KEY] = structure[_keys.CELL_KEY][nf]
            data_dict[_keys.POSITIONS_KEY] = structure[_keys.POSITIONS_KEY][nf]
            data_dict[_keys.PBC_KEY] = structure[_keys.PBC_KEY]

            if eigenvalue:
                data_dict[_keys.ENERGY_EIGENVALUE_KEY] = eigstatus[_keys.ENERGY_EIGENVALUE_KEY][nf]
                data_dict[_keys.KPOINT_KEY] = eigstatus[_keys.KPOINT_KEY]

            if energy and energy_data is not None:
                # For single structure (SCF/NSCF), energy_data has shape [1,]
                # For trajectories (MD/RELAX), energy_data has shape [nframes,] or less if unconverged
                if energy_data[_keys.TOTAL_ENERGY_KEY].shape[0] == 1:
                    # Single structure case
                    data_dict[_keys.TOTAL_ENERGY_KEY] = energy_data[_keys.TOTAL_ENERGY_KEY][0]
                else:
                    # Trajectory case - use mapping if unconverged frames exist
                    if energy_frame_mapping is not None:
                        if nf in energy_frame_mapping:
                            energy_idx = energy_frame_mapping[nf]
                            data_dict[_keys.TOTAL_ENERGY_KEY] = energy_data[_keys.TOTAL_ENERGY_KEY][energy_idx]
                        # else: skip energy for unconverged frames (don't add to data_dict)
                    else:
                        # No unconverged frames, direct indexing
                        data_dict[_keys.TOTAL_ENERGY_KEY] = energy_data[_keys.TOTAL_ENERGY_KEY][nf]

            if hamiltonian:
                data_dict["hamiltonian"] = ham[nf]
            if overlap:
                data_dict["overlap"] = ovp[nf]
            if density_matrix:
                data_dict["density_matrix"] = dm[nf]

            data_dict["idx"] = idx
            data_dict["nf"] = nf

            data_dict = pickle.dumps(data_dict)

            # write
            with lmdb_env.begin(write=True) as txn:
                entries = lmdb_env.stat()["entries"]
                txn.put(entries.to_bytes(length=4, byteorder='big'), data_dict)

        lmdb_env.close()
