import numpy as np

from dftio.plot.plot_eigs import BandPlot


def test_band_plot_writes_selected_eigenvalues(tmp_path):
    np.savetxt(tmp_path / "cell.dat", np.eye(3))
    np.savetxt(tmp_path / "positions.dat", [[0.0, 0.0, 0.0]])
    np.savetxt(tmp_path / "atomic_numbers.dat", [14], fmt="%d")
    np.savetxt(tmp_path / "pbc.dat", [True, True, True], fmt="%d")

    kpoints = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    eigenvalues = np.array(
        [[[ -2.0, 0.5, 3.0], [-1.5, 1.0, 3.5], [-1.0, 1.5, 4.0]]]
    )
    np.save(tmp_path / "kpoints.npy", kpoints)
    np.save(tmp_path / "eigenvalues.npy", eigenvalues)

    band_plot = BandPlot(root=str(tmp_path))
    band_plot.load_dat()
    band_plot.plot(bmin=1, bmax=3)

    exported = np.loadtxt(tmp_path / "band_structure.dat")
    expected = np.column_stack((np.arange(3), eigenvalues[0, :, 1:3]))
    assert exported.shape == (3, 3)
    assert np.array_equal(exported[:, 0], [0, 1, 2])
    assert np.allclose(exported, expected)
    assert (tmp_path / "band_structure.png").exists()


def test_single_kpoint_band_export(tmp_path):
    np.savetxt(tmp_path / "cell.dat", np.eye(3))
    np.savetxt(tmp_path / "positions.dat", [[0.0, 0.0, 0.0]])
    np.savetxt(tmp_path / "atomic_numbers.dat", [14], fmt="%d")
    np.savetxt(tmp_path / "pbc.dat", [True, True, True], fmt="%d")
    np.save(tmp_path / "kpoints.npy", [[0.0, 0.0, 0.0]])
    np.save(tmp_path / "eigenvalues.npy", [[[-1.0, 2.0]]])

    band_plot = BandPlot(root=str(tmp_path))
    band_plot.load_dat()
    band_plot.plot()

    exported = np.loadtxt(tmp_path / "band_structure.dat")
    assert np.allclose(exported, [[0, -1, 2]])
