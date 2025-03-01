from discretize.utils import mkvc
import numpy as np

from pathlib import Path

outdir = Path("outputs")
fout = Path("topo.txt")

[x_topo, y_topo] = np.meshgrid(np.linspace(-200, 200, 41), np.linspace(-200,
 200, 41))
z_topo = -15 * np.exp(-(x_topo**2 + y_topo**2) / 80**2)
x_topo, y_topo, z_topo = mkvc(x_topo), mkvc(y_topo), mkvc(z_topo)
xyz_topo = np.c_[x_topo, y_topo, z_topo]

if not outdir.is_dir():
    outdir.mkdir()

np.savetxt(outdir / fout, np.c_[xyz_topo], fmt="%.4e")
