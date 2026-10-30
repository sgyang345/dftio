import json
import os

import lmdb
import numpy as np
import pytest

from dftio.data import _keys
from dftio.io.parse import Parser


class _InfoParser(Parser):
    def get_structure(self, idx):
        return {
            _keys.ATOMIC_NUMBERS_KEY: np.array([42, 34], dtype=np.int32),
            _keys.POSITIONS_KEY: np.zeros((2, 2, 3), dtype=np.float32),
            _keys.CELL_KEY: np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
            _keys.PBC_KEY: np.array([True, True, False]),
        }

    def get_eigenvalue(self, idx, band_index_min):
        return {
            _keys.ENERGY_EIGENVALUE_KEY: np.zeros((2, 4, 6), dtype=np.float32),
            _keys.KPOINT_KEY: np.zeros((4, 3), dtype=np.float32),
        }

    def get_blocks(self, idx, hamiltonian=False, overlap=False, density_matrix=False):
        return None, None, None


@pytest.fixture
def parser():
    return _InfoParser(root=["unused"], prefix="")


@pytest.mark.parametrize(("fmt", "pos_type"), [("dat", "cart"), ("ase", "ase")])
def test_structure_output_writes_info_json(parser, tmp_path, fmt, pos_type):
    parser.write(
        idx=0,
        outroot=tmp_path,
        format=fmt,
        eigenvalue=True,
        hamiltonian=False,
        overlap=False,
        density_matrix=False,
        band_index_min=3,
    )

    info_path = tmp_path / "MoSe.0" / "info.json"
    info = json.loads(info_path.read_text())
    assert info == {
        "nframes": 2,
        "natoms": 2,
        "pos_type": pos_type,
        "pbc": [True, True, False],
        "bandinfo": {
            "band_min": 0,
            "band_max": 6,
            "emin": None,
            "emax": None,
        },
    }


def test_lmdb_output_writes_info_json(parser, tmp_path):
    parser.write(
        idx=0,
        outroot=tmp_path,
        format="lmdb",
        eigenvalue=True,
        hamiltonian=False,
        overlap=False,
        density_matrix=False,
        band_index_min=2,
    )

    lmdb_dir = tmp_path / f"data.{os.getpid()}.lmdb"
    info = json.loads((lmdb_dir / "info.json").read_text())
    assert info == {
        "nframes": 2,
        "natoms": 2,
        "pos_type": "cart",
        "pbc": [True, True, False],
        "bandinfo": {
            "band_min": 0,
            "band_max": 6,
            "emin": None,
            "emax": None,
        },
    }

    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False)
    with env.begin() as txn:
        assert txn.stat()["entries"] == 2
    env.close()


@pytest.mark.parametrize("fmt", ["dat", "ase"])
def test_structure_output_accepts_custom_prefix(parser, tmp_path, fmt):
    parser.write(
        idx=3,
        outroot=tmp_path,
        format=fmt,
        eigenvalue=False,
        hamiltonian=False,
        overlap=False,
        density_matrix=False,
        band_index_min=0,
        output_prefix="training_set",
    )

    assert (tmp_path / "training_set.3" / "info.json").is_file()


def test_lmdb_output_accepts_custom_prefix(parser, tmp_path):
    parser.write(
        idx=0,
        outroot=tmp_path,
        format="lmdb",
        eigenvalue=False,
        hamiltonian=False,
        overlap=False,
        density_matrix=False,
        band_index_min=0,
        output_prefix="training_set",
    )

    assert (tmp_path / f"training_set.{os.getpid()}.lmdb" / "info.json").is_file()


@pytest.mark.parametrize("output_prefix", ["", ".", "..", "nested/name", "nested\\name"])
def test_output_prefix_rejects_path_components(parser, tmp_path, output_prefix):
    with pytest.raises(ValueError, match="output_prefix"):
        parser.write(
            idx=0,
            outroot=tmp_path,
            format="dat",
            eigenvalue=False,
            hamiltonian=False,
            overlap=False,
            density_matrix=False,
            band_index_min=0,
            output_prefix=output_prefix,
        )


def test_info_json_omits_bandinfo_without_eigenvalues(parser, tmp_path):
    parser.write(
        idx=0,
        outroot=tmp_path,
        format="dat",
        eigenvalue=False,
        hamiltonian=False,
        overlap=False,
        density_matrix=False,
        band_index_min=0,
    )

    info = json.loads((tmp_path / "MoSe.0" / "info.json").read_text())
    assert "bandinfo" not in info
