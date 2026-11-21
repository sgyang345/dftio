# dftio

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://deepmodeling.github.io/dftio/)
[![Tests](https://github.com/deepmodeling/dftio/workflows/Tests/badge.svg)](https://github.com/deepmodeling/dftio/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/)


dftio is to assist machine learning communities in transcribing and manipulating DFT output into a format that is easy to read or used by machine learning models. 

dftio uses multiprocessing to paralleling the processing, and provide a standard dataset class that reads the processed dataset directly.

## Installation

### Using the install script (Recommended)

The easiest way to install dftio is using the provided installation script:

```bash
# CPU version (default)
./install.sh

# Or for GPU with CUDA 12.1
./install.sh cu121
```

### Manual installation with UV

If you prefer manual installation:

```bash
# Install UV if you haven't already
pip install uv

# Install dftio (CPU version)
uv sync

# Or for GPU version
uv sync --find-links https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

### Using pip (from PyPI - coming soon)

```bash
pip install dftio
```

**Note**: dftio depends on `torch-scatter` which requires special handling. The install script automatically manages this for you.

## Supports

Current:

| Package  | Structure | Eigenvalues | Hamiltonian | Density matrix | Overlap matrix | Total energy |
|:--------:|:---------:|:-----------:|    :----:   |     :----:     |     :----:     |   :----:     |
|  ABACUS  |     √     |      √      | √           | √              | √              | √            |
|  RESCU   |     √     |             | √           |                | √              |              |
|  SIESTA  |     √     |             | √           | √              | √              |              |
| Gaussian |     √     |             | √           | √              | √              |              |
|   VASP   |     √     |      √      |             |                |                | √            |
|  PYATB   |     √     |      √      |             |                |                |              |
|  OpenMX  |     √     |      √      | √           | √              | √              |              |

Ongoing:

- Charge Density
- Atomic Orbitals
- Wave Function
- Wave Function Coefficients
- Total Energy


## How to use
To parse the DFT output files into readable data format, user can follows:

```bash
usage: dftio parse [-h] [-ll {DEBUG,3,INFO,2,WARNING,1,ERROR,0}] [-lp LOG_PATH] [-m MODE] [-n NUM_WORKERS] [-r ROOT] [-p PREFIX] [-o OUTROOT] [-f FORMAT] [-ham] [-ovp] [-dm] [-eig] [-min BAND_INDEX_MIN]

options:
  -h, --help            show this help message and exit
  -ll {DEBUG,3,INFO,2,WARNING,1,ERROR,0}, --log-level {DEBUG,3,INFO,2,WARNING,1,ERROR,0}
                        set verbosity level by string or number, 0=ERROR, 1=WARNING, 2=INFO and 3=DEBUG (default: INFO)
  -lp LOG_PATH, --log-path LOG_PATH
                        set log file to log messages to disk, if not specified, the logs will only be output to console (default: None)
  -m MODE, --mode MODE  The name of the DFT software, currently support abacus/rescu/siesta/gaussian/pyatb/openmx (default: abacus)
  -n NUM_WORKERS, --num_workers NUM_WORKERS
                        The number of workers used to parse the dataset. (For n>1, we use the multiprocessing to accelerate io.) (default: 1)
  -r ROOT, --root ROOT  The root directory of the DFT files. (default: ./)
  -p PREFIX, --prefix PREFIX
                        The prefix of the DFT files under root. (default: frame)
  -o OUTROOT, --outroot OUTROOT
                        The output root directory. (default: ./)
  -f FORMAT, --format FORMAT
                        The output file format, should be dat, ase or lmdb. (default: dat)
  -ham, --hamiltonian   Whether to parse the Hamiltonian matrix. (default: False)
  -ovp, --overlap       Whether to parse the Overlap matrix (default: False)
  -dm, --density_matrix
                        Whether to parse the Density matrix (default: False)
  -eig, --eigenvalue    Whether to parse the kpoints and eigenvalues (default: False)
  -min BAND_INDEX_MIN, --band_index_min BAND_INDEX_MIN
                        The initial band index for eigenvalues to save.(0-band_index_min) bands will be ignored! (default: 0)
  -energy, --energy     Whether to parse the total energy from DFT output (default: False)
```

### Optional double-precision reading

All supported backends (ABACUS, VASP, SIESTA, RESCU, Gaussian, PYATB, and OpenMX)
accept `--precision double` to retain `float64`/`complex128` when reading their
supported data. The default `--precision single` preserves the original
conversions, including arrays that already used double precision. For example:

```bash
dftio parse --mode abacus --root /path/to/calculations --prefix calculation \
  --hamiltonian --overlap --precision double --outroot ./parsed_double
```

The Python API accepts the same option:

```python
from dftio.io.parse import ParserRegister

parser = ParserRegister("abacus", root=["/path/to/calculation"], prefix="",
                        precision="double")
hamiltonians, overlaps, _ = parser.get_blocks(0, hamiltonian=True, overlap=True)
```

This applies to each backend's supported data, including Hamiltonian, overlap,
and density matrices and existing SOC/MD parsing paths. Units, orbital/spin
ordering, and export layouts are unchanged;
dat exports retain matrix dtypes in HDF5, and LMDB retains array dtypes. Total
energies already use `float64` and are unchanged. This option does not add
previously unsupported data types or calculation modes (for example, VASP and
PYATB still do not supply matrix blocks). Previously rounded single-precision
exports must be regenerated from the original DFT output;
casting them to double precision cannot restore lost digits. Downstream
machine-learning data loaders may still convert exported arrays to their own
configured precision.

### OpenMX

For a static OpenMX calculation (`MD.Type Nomd`), dftio can read the structure, eigenvalues, Hamiltonian, overlap matrix, and density matrix. The calculation directory should contain an OpenMX `.out` file, the corresponding `.scfout` file, and a `.cif` structure file. dftio uses the CIF to initialize the structure and restores the final geometry from `.scfout` when parsing. Eigenvalues are read from the SCF output, or from the `.Band` file when band-dispersion output is enabled.

For example, to extract the Hamiltonian and overlap matrices:

```bash
dftio parse \
  --mode openmx \
  --root /path/to/openmx/calculations \
  --prefix calculation \
  --hamiltonian \
  --overlap \
  --format dat \
  --outroot ./parsed_openmx
```

Add `--eigenvalue` to extract eigenvalues and k-points, or `--density_matrix` to extract the density matrix. The parser currently does not extract OpenMX total energies; matrix parsing for MD trajectories is not supported.

## Call for Contributors
dftio is an open-source tool that calls for enthusiastic developers to contribute their talent. One can contribute through raising function requirement issues, or contact the current developer directly.

### Current Contributors (in alphabetical order)
Qiangqiang Gu, Jijie Zou, Mingkang Liu, Zixi Gan, Zhanghao Zhouyin
