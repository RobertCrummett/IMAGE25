from pathlib import Path

import numpy as np

outdir = Path("outputs")
ftopo  = outdir / Path("topo.txt")

def create_topo():
    x_linspace = np.linspace(-200, 200, 41)
    y_linspace = np.linspace(-200, 200, 41)

    x_topo, y_topo = np.meshgrid(x_linspace, y_linspace)
    z_topo = -15 * np.exp(-(x_topo**2 + y_topo**2) / 80**2)

    x_flat = x_topo.flatten(order = "f")
    y_flat = y_topo.flatten(order = "f")
    z_flat = z_topo.flatten(order = "f")

    return np.column_stack((x_flat, y_flat, z_flat))

def write_topo(xyz_topo, path = ftopo):
    try:
        path.parent.mkdir()
    except FileExistsError as _:
        pass
    finally:
        np.savetxt(ftopo, xyz_topo, fmt="%.4e", delimiter=" ")

def load_topo(path = ftopo):
    if path.exists():
        return np.loadtxt(path)
    else:
        return create_topo()

if __name__ == "__main__":
    xyz_topo = create_topo()
    write_topo(xyz_topo)
