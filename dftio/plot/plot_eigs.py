import numpy as np
import matplotlib.pyplot as plt
import os
import logging

from ase.io.trajectory import Trajectory
from ase.io import read, write

log=logging.getLogger(__name__)

class BandPlot(object):
    def __init__(self, root, **kwargs):
        self.path = root
        
    def load_dat(self, fmt=None):
        # load structure
        assert os.path.exists(self.path), f"Path {self.path} does not exist!"
        # 判断文件格式       
        fmt_1 = False
        fmt_2 = False
        if os.path.exists(os.path.join(self.path, "cell.dat")) \
                and os.path.exists(os.path.join(self.path, "positions.dat")) \
                and os.path.exists(os.path.join(self.path, "atomic_numbers.dat")) \
                and os.path.exists(os.path.join(self.path, "pbc.dat")):
            log.info(".dat file exists, Loading data from dat format!")
            fmt_1 = True

        if os.path.exists(os.path.join(self.path, "xdat.traj")):
            fmt_2 = True

        if fmt_1 and fmt_2:
            if fmt is None:
                raise ValueError("Both .dat and .traj files exist, please specify the format!")
            else:
                log.warning(f"Both .dat and .traj files exist, using user defined fmt {fmt}!")

        
        if fmt_1 and not fmt_2:
            self.cell = np.loadtxt(os.path.join(self.path, "cell.dat")).reshape(-1, 3)
            self.positions = np.loadtxt(os.path.join(self.path, "positions.dat")).reshape(-1, 3)
            self.atomic_numbers = np.loadtxt(os.path.join(self.path, "atomic_numbers.dat"), dtype=np.int32)
            self.pbc = np.loadtxt(os.path.join(self.path, "pbc.dat"))
            
            if not(fmt is None or fmt == 'dat'):
                log.warning(f"detect the .dat format, and user defiend {fmt}, using dat format instead!")
        
        if fmt_2 and not fmt_1:
            trajfile = Trajectory(os.path.join(self.path, "xdat.traj"), 'r')
            self.cell = trajfile[0].cell
            self.positions = trajfile[0].positions
            self.atomic_numbers = trajfile[0].numbers
            self.pbc = trajfile[0].pbc
            
            if not(fmt is None or fmt == 'ase'):
                log.warning(f"detect the .traj format, and user defiend {fmt}, using ase format instead!")

        if not fmt_1 and not fmt_2:
            raise ValueError("No .dat or .traj files exist, please check the path!")

        self.kpoints = np.load(os.path.join(self.path, "kpoints.npy"))
        self.eigs = np.load(os.path.join(self.path, "eigenvalues.npy"))

        if len(self.eigs.shape)==3 :
            assert self.eigs.shape[0]== 1, "only 1 band structure is supported!"
            self.eigs = self.eigs[0]
        
        assert self.eigs.shape[0] == self.kpoints.shape[0], "eigenvalues and kpoints shape mismatch!"

    def plot(self, bmin:int=0, bmax:int=None):
        # plot band structure
        fig, ax = plt.subplots()
        
        nbands = self.eigs.shape[1]
        if bmax is None:
            bmax = nbands
        else:
            bmax = min(bmax, nbands)
        assert bmax>bmin, "max should be larger than min!"

        kpoint_indices = np.arange(self.kpoints.shape[0], dtype=np.int64)
        plotted_eigenvalues = self.eigs[:, bmin:bmax]
        data = np.column_stack((kpoint_indices, plotted_eigenvalues))
        band_columns = " ".join(
            f"band_{band_index}" for band_index in range(bmin, bmax)
        )
        np.savetxt(
            os.path.join(self.path, "band_structure.dat"),
            data,
            fmt="%.10g",
            header=f"kpoint_index {band_columns}",
        )

        if self.kpoints.shape[0] > 1:
            ax.plot(self.eigs[:,bmin:bmax], 'b-', lw=1)
            ax.set_xlabel("k-point")
            ax.set_title("Band Structure")
        else:
            for e in self.eigs[0, bmin:bmax]:
                ax.hlines(e, xmin=-0.5, xmax=0.5, color='b', linewidth=1)
                ax.set_xticks([])
            ax.set_title("Energy levels")
        ax.text(0.5, 0.5, f"band windown: {bmin} - {bmax}", ha='center', va='center', transform=ax.transAxes)
        ax.set_ylabel("Energy (eV)")
        plt.savefig(os.path.join(self.path, "band_structure.png"), dpi=300)
        plt.tight_layout()
        # if has gui? then plot show
        if plt.get_backend() != 'Agg':
            plt.show()
        else:
            plt.close()
