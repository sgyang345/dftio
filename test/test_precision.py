"""Optional precision must preserve source digits and legacy defaults."""

import json
import os
import pickle
import shutil
from pathlib import Path

import ase
import ase.io
import h5py
import lmdb
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from dftio.__main__ import parse_args, wapper
from dftio.constants import ABACUS2DFTIO, Harte2eV
from dftio.data import _keys
from dftio.io.abacus.abacus_parser import AbacusParser
from dftio.io.parse import ParserRegister
from dftio.io.rescu.rescu_parser import Hartree2eV


DATA = Path(__file__).parent / "data"
MODES = ("abacus", "vasp", "siesta", "rescu", "gaussian", "pyatb", "openmx")
VALUE = 1.23456789012345


def make_openmx(directory, spinful=False):
    """Small binary fixture: on-disk doubles must not depend on output precision."""
    directory.mkdir()
    ase.io.write(directory / "C.cif", ase.Atoms("C", cell=np.eye(3) * 4, pbc=True))
    (directory / "C.out").write_text(
        "MD.Type Nomd\nscf.EigenvalueSolver band\nBand.dispersion on\n"
        "Band.Nkpath 1\n<Band.kpath\n"
        "Species.Number 1\n<Definition.of.Atomic.Species\nC C6.0-s1 C_PBE\n"
        "Fractional coordinates of the final structure\n\n\n\n1 C 0 0 0\n"
    )
    (directory / "C.Band").write_text(
        f"1 0\nunused lattice\n1\nunused path\n1 {VALUE} 0 0\n{VALUE}\n"
    )
    with (directory / "C.scfout").open("wb") as stream:
        def ints(values):
            np.array(values, dtype=np.int32).tofile(stream)

        def doubles(values):
            np.array(values, dtype=np.float64).tofile(stream)

        ints([1, 3 if spinful else 0, 1, 0, 0, 0, 0])
        doubles([0, 0, 0, 0])  # atv
        ints([0, 0, 0, 0])  # atv_ijk
        ints([1, 0, 1, 0])  # Total_NumOrbs, FNAN, natn, ncn
        doubles(np.column_stack((np.zeros(3), np.eye(3) * VALUE)))  # tv
        doubles(np.column_stack((np.zeros(3), np.eye(3))))  # rtv
        doubles([0, VALUE, 0, 0])  # Gxyz
        doubles([VALUE] * (4 if spinful else 1))  # Hk
        if spinful:
            doubles([VALUE] * 3)  # iHk
        doubles([VALUE])  # overlap
        doubles([0] * 3)  # OLP_p
        doubles([VALUE] * (4 if spinful else 1))  # density matrix
        doubles([VALUE] * 2)  # iDM
    return directory


def make_rescu(directory):
    directory.mkdir()
    with h5py.File(directory / "scf.mat", "w") as stream:
        stream["info/calculationType"] = np.array([list(map(ord, "self-consistent"))])
        stream["atom/xyz"] = np.array([[VALUE], [0], [0]])
        stream["atom/element"] = [[1]]
        stream["element/species"] = np.array([[ord("C")]], dtype=np.uint16)
        stream["domain/boundary"] = [[1, 1, 1]]
        stream["domain/latvec"] = np.eye(3) * VALUE
        stream["LCAO/orbInfo/Aorb"] = [[1]]
        stream["LCAO/orbInfo/Lorb"] = [[0]]
    with h5py.File(directory / "blocks.h5", "w") as stream:
        stream["LCAO/Rvec"] = np.zeros((3, 1))
        stream["LCAO/hamiltonian1"] = [[VALUE]]
        stream["LCAO/overlap1"] = [[VALUE]]
    with h5py.File(directory / "band.mat", "w") as stream:
        stream["info/calculationType"] = np.array([list(map(ord, "band-structure"))])
        stream["band/kdirect"] = np.array([[VALUE], [0], [0]])
        stream["band/ksnrg"] = [[[VALUE]]]
    return directory


@pytest.fixture
def calculations(tmp_path):
    abacus = tmp_path / "abacus"
    shutil.copytree(DATA / "abacus_scf", abacus)
    output = abacus / "OUT.ABACUS"
    nk = np.loadtxt(output / "BANDS_1.dat").shape[0]
    (output / "kpoints").write_text(
        f"nkstot now = {nk}\nKPOINTS \n"
        + "".join(f"{i} {VALUE} 0 0\n" for i in range(nk))
    )
    shutil.copyfile(output / "data-HR-sparse_SPIN0.csr", output / "data-DMR-sparse_SPIN0.csr")

    vasp = tmp_path / "vasp"
    vasp.mkdir()
    ase.io.write(vasp / "POSCAR", ase.Atoms("C", positions=[[VALUE, 0, 0]],
                                          cell=np.eye(3) * 4, pbc=True), format="vasp")
    (vasp / "EIGENVAL").write_text(
        f"1 1 1 1\n0 0 0 0\n1\nCAR\nSystem\n1 1 1\n\n{VALUE} 0 0 1\n1 {VALUE}\n"
    )

    pyatb = tmp_path / "pyatb"
    band = pyatb / "pyatb" / "Out" / "Band_Structure"
    band.mkdir(parents=True)
    shutil.copyfile(DATA / "abacus_scf" / "STRU", pyatb / "pyatb" / "STRU")
    np.savetxt(band / "band.dat", [[VALUE, VALUE + 1], [VALUE + 2, VALUE + 3]])
    np.savetxt(band / "kpt.dat", [[VALUE, 0, 0], [0, VALUE, 0]])

    gaussian_log = DATA / "gaussian" / "example_folder" / "id_1" / "gau.log"
    valid_logs = tmp_path / "gaussian_logs.txt"
    valid_logs.write_text(str(gaussian_log) + "\n")
    return {
        "abacus": (abacus, {}),
        "vasp": (vasp, {}),
        "siesta": (DATA / "siesta" / "siesta_out_withband", {}),
        "rescu": (make_rescu(tmp_path / "rescu"), {}),
        "gaussian": (gaussian_log, {"valid_gau_info_path": str(valid_logs)}),
        "pyatb": (pyatb, {}),
        "openmx": (make_openmx(tmp_path / "openmx"), {}),
    }


def get_parser(calculations, mode, precision=None):
    path, options = calculations[mode]
    if precision is not None:
        options = dict(options, precision=precision)
    return ParserRegister(mode, root=[str(path)], prefix="", **options)


@pytest.mark.parametrize("mode", MODES)
def test_invalid_precision_rejected_before_reading(mode):
    with pytest.raises(ValueError, match="precision"):
        ParserRegister(mode, root=["unused"], prefix="", precision="invalid")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("precision", [None, "single", "double"])
def test_all_backend_precisions(calculations, mode, precision):
    parser = get_parser(calculations, mode, precision)
    real_dtype = np.float64 if precision == "double" else np.float32
    structure = parser.get_structure(0)
    assert structure[_keys.POSITIONS_KEY].dtype == real_dtype
    assert structure[_keys.CELL_KEY].dtype == real_dtype
    if mode != "gaussian":
        assert parser.check_structure(0, structure)
        eigstatus = parser.get_eigenvalue(0)
        # RESCU eigenvalues historically preserve the file dtype, even by default.
        eig_dtype = np.float64 if mode == "rescu" else real_dtype
        assert all(value.dtype == eig_dtype for value in eigstatus.values())
        if mode != "rescu":
            assert parser.check_eigenvalue(0, eigstatus)
    if mode in ("vasp", "pyatb"):
        with pytest.raises(NotImplementedError):
            parser.get_blocks(0, hamiltonian=True)
        return
    ham, ovp, dm = parser.get_blocks(0, hamiltonian=True, overlap=True,
                                    density_matrix=mode != "rescu")
    block_dtype = np.float64 if mode in ("gaussian", "rescu") else real_dtype
    for frames in (ham, ovp) if mode == "rescu" else (ham, ovp, dm):
        assert frames[0]
        assert all(block.dtype == block_dtype for frame in frames for block in frame.values())
    if precision == "double":
        assert parser.check_blocks(0, hamiltonian=True, overlap=True,
                                   density_matrix=mode != "rescu")


@pytest.mark.parametrize("mode", ["abacus", "vasp", "pyatb", "rescu", "openmx"])
def test_source_digits_survive_double_reading(calculations, mode):
    parser = get_parser(calculations, mode, "double")
    if mode == "rescu":
        ham, ovp, _ = parser.get_blocks(0, hamiltonian=True, overlap=True)
        assert ham[0]["0_0_0_0_0"][0, 0] == VALUE * Hartree2eV
        assert ovp[0]["0_0_0_0_0"][0, 0] == VALUE
    else:
        eigstatus = parser.get_eigenvalue(0)
        assert eigstatus[_keys.KPOINT_KEY][0, 0] == VALUE
        if mode != "abacus":
            factor = Harte2eV if mode == "openmx" else 1
            assert eigstatus[_keys.ENERGY_EIGENVALUE_KEY][0, 0, 0] == VALUE * factor


def test_siesta_source_matrix_digits_preserved(calculations):
    import sisl

    parser = get_parser(calculations, "siesta", "double")
    path, _ = calculations["siesta"]
    ham, ovp, _ = parser.get_blocks(0, hamiltonian=True, overlap=True)
    for kind, filename, frames in (
        (sisl.Hamiltonian, "Au_cell.TSHS", ham),
        (sisl.Overlap, "Au_cell.TSHS", ovp),
    ):
        source = kind.read(str(path / filename))
        offset = source.geometry.sc_index([0, 0, 0]) * source.geometry.no
        block = frames[0]["0_0_0_0_0"]
        # The first two orbitals are s orbitals, unchanged by the basis transform.
        np.testing.assert_array_equal(block[:2, :2], source.tocsr()[:2, offset:offset + 2].toarray())


@pytest.mark.parametrize("matrix_index", [0, 1, 2])
def test_double_checks_all_matrices_and_frames(calculations, monkeypatch, matrix_index):
    parser = get_parser(calculations, "abacus", "double")
    structure = parser.get_structure(0)
    structure[_keys.POSITIONS_KEY] = np.repeat(structure[_keys.POSITIONS_KEY], 2, axis=0)
    structure[_keys.CELL_KEY] = np.repeat(structure[_keys.CELL_KEY], 2, axis=0)
    frames = [[{"0_0_0_0_0": np.ones((1, 1))}, {"0_0_0_0_0": np.ones((1, 1))}]
              for _ in range(3)]
    frames[matrix_index][1]["0_0_0_0_0"] = np.ones((1, 1), dtype=np.float32)
    monkeypatch.setattr(parser, "get_structure", lambda idx: structure)
    monkeypatch.setattr(parser, "get_blocks", lambda *args: frames)
    with pytest.raises(AssertionError, match="double precision"):
        parser.check_blocks(0, hamiltonian=True, overlap=True, density_matrix=True)


@pytest.mark.parametrize("spinful", [False, True])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("step", [0, 1])
def test_abacus_parses_source_digits_and_spin_order(tmp_path, spinful, precision, step):
    parser = AbacusParser(root=[str(DATA / "abacus_scf")], prefix="", precision=precision)
    size = 6 if spinful else 3
    source = np.arange(size * size, dtype=np.float64).reshape(size, size) + VALUE
    if spinful:
        source = source + 1j * (source + VALUE)
    sparse = csr_matrix(source)
    if spinful:
        values = " ".join(f"({v.real:.17g},{v.imag:.17g})" for v in sparse.data)
    else:
        values = " ".join(f"{v:.17g}" for v in sparse.data)
    matrix_text = (
        f"Matrix Dimension of H(R): {size}\nMatrix number of H(R): 1\n"
        f"0 0 0 {sparse.nnz}\n{values}\n"
        + " ".join(map(str, sparse.indices)) + "\n"
        + " ".join(map(str, sparse.indptr)) + "\n"
    )
    path = tmp_path / "matrix.csr"
    path.write_text("STEP: 0\n" + matrix_text + "STEP: 1\n" + matrix_text)
    blocks, norbits = parser.parse_matrix(str(path), 1, np.array([3]), {6: [1]},
                                          np.array([6]), 13.605698, spinful, step)
    dtype = parser.complex_dtype if spinful else parser.real_dtype
    expected = source.astype(dtype)
    transform = ABACUS2DFTIO[1]
    if spinful:
        expected = expected.reshape(3, 2, 3, 2).transpose(1, 0, 3, 2).reshape(6, 6)
        transform = np.kron(np.eye(2, dtype=np.float32), transform)
    expected = (transform @ expected @ transform.T) * 13.605698
    assert norbits == size
    assert blocks["0_0_0_0_0"].dtype == dtype
    np.testing.assert_array_equal(blocks["0_0_0_0_0"], expected)
    if precision == "double":
        rounded = source.astype(np.complex64 if spinful else np.float32).astype(dtype)
        assert np.max(np.abs(source - rounded)) > 1e-8


@pytest.mark.parametrize("precision", ["single", "double"])
def test_openmx_soc_preserves_existing_double_default(tmp_path, precision):
    path = make_openmx(tmp_path / "soc", spinful=True)
    parser = ParserRegister("openmx", root=[str(path)], prefix="", precision=precision)
    ham, ovp, dm = parser.get_blocks(0, hamiltonian=True, overlap=True, density_matrix=True)
    assert ham[0]["0_0_0_0_0"].dtype == np.complex128
    assert dm[0]["0_0_0_0_0"].dtype == np.complex128
    assert ovp[0]["0_0_0_0_0"].dtype == parser.real_dtype
    assert ham[0]["0_0_0_0_0"][0, 0] == complex(VALUE, VALUE) * Harte2eV


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fmt", ["dat", "lmdb"])
def test_double_export_roundtrip(calculations, tmp_path, mode, fmt):
    parser = get_parser(calculations, mode, "double")
    matrices = mode not in ("vasp", "pyatb")
    outroot = tmp_path / "export"
    assert parser.write(0, outroot=str(outroot), format=fmt,
                        eigenvalue=mode not in ("gaussian", "rescu"),
                        hamiltonian=matrices, overlap=matrices,
                        density_matrix=matrices and mode != "rescu", band_index_min=0,
                        output_prefix="double") is not False
    if fmt == "dat":
        output = outroot / "double.0"
        assert json.loads((output / "info.json").read_text())["nframes"] == 1
        if matrices:
            for filename in ("hamiltonians.h5", "overlaps.h5"):
                with h5py.File(output / filename) as stream:
                    assert stream["0"]
                    assert all(value.dtype == np.float64 for value in stream["0"].values())
        if mode not in ("gaussian", "rescu"):
            assert np.load(output / "eigenvalues.npy").dtype == np.float64
            assert np.load(output / "kpoints.npy").dtype == np.float64
    else:
        with lmdb.open(str(outroot / f"double.{os.getpid()}.lmdb"), readonly=True, lock=False) as env:
            with env.begin() as txn:
                frame = pickle.loads(txn.get((0).to_bytes(4, "big")))
        assert frame[_keys.POSITIONS_KEY].dtype == np.float64
        assert frame[_keys.CELL_KEY].dtype == np.float64
        if matrices:
            assert all(block.dtype == np.float64 for block in frame["hamiltonian"].values())
            assert all(block.dtype == np.float64 for block in frame["overlap"].values())


def test_cli_default_and_double_option():
    assert parse_args(["parse"]).precision == "single"
    assert parse_args(["parse", "--precision", "double"]).precision == "double"
    with pytest.raises(SystemExit):
        parse_args(["parse", "--precision", "invalid"])


def test_double_precision_reaches_multiprocessing(calculations, tmp_path):
    from multiprocessing.pool import Pool

    path, _ = calculations["abacus"]
    args = dict(mode="abacus", root=[str(path)], prefix="", precision="double",
                outroot=str(tmp_path / "parallel"), format="dat", eigenvalue=True,
                hamiltonian=True, overlap=True, density_matrix=True, band_index_min=0)
    with Pool(2) as pool:
        results = list(pool.imap(wapper(args), [0]))
    assert results == [(True, "abacus")]
    with h5py.File(tmp_path / "parallel" / "Al.0" / "hamiltonians.h5") as stream:
        assert all(value.dtype == np.float64 for value in stream["0"].values())
