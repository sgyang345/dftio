import os
import h5py
import numpy as np
import pytest
import shutil
from dftio.io.abacus.abacus_parser import AbacusParser
from dftio.data import _keys

@pytest.fixture
def abacus_parser(tmp_path):
    """Fixture for AbacusParser that creates a temporary directory structure."""
    # The parser expects a directory structure like:
    # root/
    #   prefix_001/
    #     OUT.ABACUS/
    #       ...
    #   prefix_002/
    #     OUT.ABACUS/
    #       ...
    
    # Create a temporary directory structure that the parser expects
    calc_dir = tmp_path / "calculation"
    calc_dir.mkdir()
    out_abacus_dir = calc_dir / "OUT.ABACUS"
    out_abacus_dir.mkdir()

    # Copy the test data to the temporary directory
    test_data_src = "test/data/abacus_scf/OUT.ABACUS"
    for item in os.listdir(test_data_src):
        s = os.path.join(test_data_src, item)
        d = os.path.join(out_abacus_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks=False, ignore=None)
        else:
            shutil.copy2(s, d)

    # Create a dummy kpoints file
    with open(out_abacus_dir / "kpoints", "w") as f:
        f.write("nkstot now = 2\n")
        f.write("KPOINTS \n")
        f.write("0.0 0.0 0.0 1.0\n")
        f.write("0.5 0.5 0.5 1.0\n")

    return AbacusParser(
        root=str(tmp_path),
        prefix='calculation'
    )


@pytest.fixture
def abacus_gets_parser():
    """Parser backed by a real ABACUS get_S calculation."""
    return AbacusParser(
        root="test/data",
        prefix="abacus_gets",
    )

def test_abacus_parser_init(abacus_parser, tmp_path):
    """Test AbacusParser initialization."""
    assert abacus_parser.root == str(tmp_path)
    assert abacus_parser.prefix == 'calculation'
    assert len(abacus_parser.raw_datas) == 1
    assert abacus_parser.raw_datas[0] == str(tmp_path / "calculation")

def test_get_structure(abacus_parser):
    """Test parsing of structure files."""
    structure = abacus_parser.get_structure(0)
    assert structure is not None
    assert _keys.ATOMIC_NUMBERS_KEY in structure
    assert _keys.POSITIONS_KEY in structure
    assert _keys.CELL_KEY in structure
    assert _keys.PBC_KEY in structure
    assert len(structure[_keys.ATOMIC_NUMBERS_KEY]) == 1
    assert structure[_keys.ATOMIC_NUMBERS_KEY][0] == 13
    assert structure[_keys.POSITIONS_KEY].shape == (1, 1, 3)
    assert structure[_keys.CELL_KEY].shape == (1, 3, 3)

def test_get_eigenvalue(abacus_parser):
    """Test parsing of eigenvalues."""
    eigenvalues = abacus_parser.get_eigenvalue(0)
    assert eigenvalues is not None
    assert _keys.ENERGY_EIGENVALUE_KEY in eigenvalues
    assert _keys.KPOINT_KEY in eigenvalues
    assert eigenvalues[_keys.ENERGY_EIGENVALUE_KEY].shape == (1, 47, 16)
    assert eigenvalues[_keys.KPOINT_KEY].shape == (2, 3)

def test_get_basis(abacus_parser):
    """Test parsing of basis set."""
    basis = abacus_parser.get_basis(0)
    assert basis is not None
    assert isinstance(basis, dict)
    assert "Al" in basis

def test_get_blocks(abacus_parser):
    """Test parsing of blocks (Hamiltonian/Overlap)."""
    # This test requires the sparse matrix files to be present.
    # The fixture copies 'test/data/abacus/OUT.ABACUS' which contains 'data-HR-sparse_SPIN0.csr' etc.
    
    # We need to check if the parser can read them.
    # Note: get_blocks reads 'running_scf.log' (or similar) to get orbital info first.
    # The fixture copies 'running_scf.log' as well.
    
    # However, the fixture creates a dummy kpoints file but copies other files.
    # We need to make sure 'running_scf.log' is consistent with the sparse matrices if the parser checks dimensions.
    # The parser reads 'Matrix Dimension of ...' from the csr file.
    
    # Let's try to run it.
    ham, ovp, dm = abacus_parser.get_blocks(0, hamiltonian=True, overlap=True, density_matrix=False)
    
    assert ham is not None
    assert isinstance(ham, list)
    assert len(ham) > 0
    assert isinstance(ham[0], dict)
    
    assert ovp is not None
    assert isinstance(ovp, list)
    assert len(ovp) > 0
    assert isinstance(ovp[0], dict)


def test_abacus_gets_structure_and_basis(abacus_gets_parser):
    assert abacus_gets_parser.get_mode(0) == "get_S"

    structure = abacus_gets_parser.get_structure(0)
    assert structure[_keys.ATOMIC_NUMBERS_KEY].tolist() == [13]
    assert structure[_keys.POSITIONS_KEY].shape == (1, 1, 3)
    assert structure[_keys.CELL_KEY].shape == (1, 3, 3)
    assert abacus_gets_parser.get_basis(0) == {"Al": "4s4p1d"}


def test_abacus_gets_overlap_blocks(abacus_gets_parser):
    ham, ovp, dm = abacus_gets_parser.get_blocks(
        0,
        hamiltonian=False,
        overlap=True,
        density_matrix=False,
    )

    assert ham is None
    assert dm is None
    assert isinstance(ovp, list) and len(ovp) == 1

    blocks = ovp[0]
    assert len(blocks) == 13
    onsite = blocks["0_0_0_0_0"]
    assert onsite.shape == (21, 21)
    assert onsite.dtype == np.float32
    assert np.allclose(np.diag(onsite), 1.0)
    assert np.allclose(
        blocks["0_0_-1_0_0"],
        blocks["0_0_1_0_0"].T,
    )


@pytest.mark.parametrize(
    ("hamiltonian", "density_matrix", "requested"),
    [
        (True, False, "Hamiltonian"),
        (False, True, "density matrix"),
    ],
)
def test_abacus_gets_rejects_unavailable_matrices(
    abacus_gets_parser,
    hamiltonian,
    density_matrix,
    requested,
):
    with pytest.raises(ValueError, match=requested):
        abacus_gets_parser.get_blocks(
            0,
            hamiltonian=hamiltonian,
            overlap=True,
            density_matrix=density_matrix,
        )


def test_abacus_gets_writes_overlaps_h5(abacus_gets_parser, tmp_path):
    abacus_gets_parser.write(
        idx=0,
        outroot=tmp_path,
        format="dat",
        eigenvalue=False,
        hamiltonian=False,
        overlap=True,
        density_matrix=False,
        band_index_min=0,
    )

    output_dir = tmp_path / "Al.0"
    assert (output_dir / "basis.dat").exists()
    assert (output_dir / "info.json").exists()

    with h5py.File(output_dir / "overlaps.h5", "r") as overlap_file:
        assert list(overlap_file.keys()) == ["0"]
        assert len(overlap_file["0"]) == 13
        onsite = overlap_file["0"]["0_0_0_0_0"][:]
        assert onsite.shape == (21, 21)
        assert np.allclose(np.diag(onsite), 1.0)
