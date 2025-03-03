# https://docs.simpeg.xyz/latest/content/tutorials/04-magnetics/plot_2a_magnetics_induced.html#sphx-glr-content-tutorials-04-magnetics-plot-2a-magnetics-induced-py
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from simpeg import maps

from collections import ChainMap

from topography import create_topo
from forward import create_simulation_params, create_model, add_noise_to_model

susceptibility = dict(background = 1e-4, sphere = 1e-2)
xyz_topo = create_topo()
simulation_params = create_simulation_params(xyz_topo)

params = ChainMap(simulation_params, susceptibility)

mesh = params["mesh"]
ind_active = params["active_cells"]
    
true_model = create_model(xyz_topo, params)
model = add_noise_to_model(true_model, sigma = 0.1)

fig = plt.figure(figsize=(9, 4))

plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
ax1 = fig.add_axes([0.1, 0.12, 0.73, 0.78])
mesh.plot_slice(
    plotting_map * model,
    normal="Y",
    ax=ax1,
    ind=int(mesh.shape_cells[1] / 2),
    grid=True,
    clim=(np.min(model), np.max(model)),
)
ax1.set_title("Model slice at y = 0 m")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")

ax2 = fig.add_axes([0.85, 0.12, 0.05, 0.78])
norm = mpl.colors.Normalize(vmin=np.min(model), vmax=np.max(model))
cbar = mpl.colorbar.ColorbarBase(ax2, norm=norm, orientation="vertical")
cbar.set_label("Magnetic Susceptibility (SI)", rotation=270, labelpad=15, size=12)

plt.show()
