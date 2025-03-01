import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from discretize import TensorMesh
from discretize.utils import active_from_xyz
from simpeg.potential_fields import magnetics
from simpeg.utils import plot2Ddata, model_builder
from simpeg import (
    maps,
    data,
    inverse_problem,
    data_misfit,
    regularization,
    optimization,
    directives,
    inversion,
    utils,
)

from pathlib import Path

inclination = 0
declination = 0
strength = 50000
noise_percent = 5

outdir = Path("outputs")
ftopo = Path("topo.txt")
fanom = Path(f"anom_i{inclination}_d{declination}.txt")

topo_xyz = np.loadtxt(outdir / ftopo)
dobs = np.loadtxt(outdir / fanom)

receiver_locations = dobs[:, 0:3]
dobs = dobs[:, -1]

fig = plt.figure(figsize=(6, 5))
v_max = np.max(np.abs(dobs))
ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])
plot2Ddata(
    receiver_locations,
    dobs,
    ax=ax1,
    ncontour=30,
    clim=(-v_max, v_max),
    contourOpts={"cmap": "bwr"},
)
ax1.set_title("TMI Anomaly")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
ax2 = fig.add_axes([0.85, 0.05, 0.05, 0.9])
norm = mpl.colors.Normalize(vmin=-np.max(np.abs(dobs)), vmax=np.max(np.abs(dobs)))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.bwr
)
cbar.set_label("$nT$", rotation=270, labelpad=15, size=12)
plt.show()


maximum_anomaly = np.max(np.abs(dobs))
std = noise_percent / 100 * maximum_anomaly * np.ones(len(dobs))
components = ["tmi"]
receiver_list = magnetics.receivers.Point(receiver_locations, components=components)
receiver_list = [receiver_list]
source_field = magnetics.sources.UniformBackgroundField(
    receiver_list=receiver_list,
    amplitude=strength,
    inclination=inclination,
    declination=declination,
)
survey = magnetics.survey.Survey(source_field)

data_object = data.Data(survey, dobs=dobs, standard_deviation=std)

dh = 5.0
hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
hy = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
hz = [(dh, 5, -1.3), (dh, 15)]
mesh = TensorMesh([hx, hy, hz], "CCN")

background_susceptibility = 1e-4
assert background_susceptibility > 0, "Must be greater than zero to converge!"

active_cells = active_from_xyz(mesh, topo_xyz)

nC = int(active_cells.sum())
model_map = maps.IdentityMap(nP=nC)

starting_model = background_susceptibility * np.ones(nC)

simulation = magnetics.simulation.Simulation3DIntegral(
    survey=survey,
    mesh=mesh,
    model_type="scalar",
    chiMap=model_map,
    active_cells=active_cells,
    engine="choclo",
)
dmis = data_misfit.L2DataMisfit(data=data_object, simulation=simulation)

reg = regularization.Sparse(
    mesh,
    active_cells=active_cells,
    mapping=model_map,
    reference_model=starting_model,
    gradient_type="total",
)

reg.norms = [0, 0, 0, 0]
opt = optimization.ProjectedGNCG(
    maxIter=20, lower=0.0, upper=1.0, maxIterLS=20, maxIterCG=10, tolCG=1e-3
)

inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=5)
save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
update_IRLS = directives.UpdateIRLS(
    f_min_change=1e-4,
    max_irls_iterations=30,
    cooling_factor=1.5,
    misfit_tolerance=1e-2,
)
update_jacobi = directives.UpdatePreconditioner()
target_misfit = directives.TargetMisfit(chifact=1)
sensitivity_weights = directives.UpdateSensitivityWeights(every_iteration=False)
directives_list = [
    sensitivity_weights,
    starting_beta,
    save_iteration,
    update_IRLS,
    update_jacobi,
]


inv = inversion.BaseInversion(inv_prob, directives_list)
recovered_model = inv.run(starting_model)

background_susceptibility = 0.0001
sphere_susceptibility = 0.01

true_model = background_susceptibility * np.ones(nC)
ind_sphere = model_builder.get_indices_sphere(
    np.r_[0.0, 0.0, -45.0], 15.0, mesh.cell_centers
)
ind_sphere = ind_sphere[active_cells]
true_model[ind_sphere] = sphere_susceptibility


fig = plt.figure(figsize=(9, 4))
plotting_map = maps.InjectActiveCells(mesh, active_cells, np.nan)
ax1 = fig.add_axes([0.08, 0.1, 0.75, 0.8])
mesh.plot_slice(
    plotting_map * true_model,
    normal="Y",
    ax=ax1,
    ind=int(mesh.shape_cells[1] / 2),
    grid=True,
    clim=(np.min(true_model), np.max(true_model)),
    pcolor_opts={"cmap": "viridis"},
)
ax1.set_title("Model slice at y = 0 m")
ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])
norm = mpl.colors.Normalize(vmin=np.min(true_model), vmax=np.max(true_model))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis, format="%.1e"
)
cbar.set_label("SI", rotation=270, labelpad=15, size=12)
plt.show()


fig = plt.figure(figsize=(9, 4))
plotting_map = maps.InjectActiveCells(mesh, active_cells, np.nan)
ax1 = fig.add_axes([0.08, 0.1, 0.75, 0.8])
mesh.plot_slice(
    plotting_map * recovered_model,
    normal="Y",
    ax=ax1,
    ind=int(mesh.shape_cells[1] / 2),
    grid=True,
    clim=(np.min(recovered_model), np.max(recovered_model)),
    pcolor_opts={"cmap": "viridis"},
)
ax1.set_title("Model slice at y = 0 m")
ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])
norm = mpl.colors.Normalize(vmin=np.min(recovered_model), vmax=np.max(recovered_model))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis, format="%.1e"
)
cbar.set_label("SI", rotation=270, labelpad=15, size=12)
plt.show()

dpred = inv_prob.dpred

data_array = np.c_[dobs, dpred, (dobs - dpred) / std]

fig = plt.figure(figsize=(17, 4))
plot_title = ["Observed", "Predicted", "Normalized Misfit"]
plot_units = ["nT", "nT", ""]
ax1 = 3 * [None]
ax2 = 3 * [None]
norm = 3 * [None]
cbar = 3 * [None]
cplot = 3 * [None]
v_lim = [np.max(np.abs(dobs)), np.max(np.abs(dobs)), np.max(np.abs(data_array[:, 2]))]
for ii in range(0, 3):
    ax1[ii] = fig.add_axes([0.33 * ii + 0.03, 0.11, 0.25, 0.84])
    cplot[ii] = plot2Ddata(
        receiver_list[0].locations,
        data_array[:, ii],
        ax=ax1[ii],
        ncontour=30,
        clim=(-v_lim[ii], v_lim[ii]),
        contourOpts={"cmap": "bwr"},
    )
    ax1[ii].set_title(plot_title[ii])
    ax1[ii].set_xlabel("x (m)")
    ax1[ii].set_ylabel("y (m)")

    ax2[ii] = fig.add_axes([0.33 * ii + 0.27, 0.11, 0.01, 0.84])
    norm[ii] = mpl.colors.Normalize(vmin=-v_lim[ii], vmax=v_lim[ii])
    cbar[ii] = mpl.colorbar.ColorbarBase(
        ax2[ii], norm=norm[ii], orientation="vertical", cmap=mpl.cm.bwr
    )
    cbar[ii].set_label(plot_units[ii], rotation=270, labelpad=15, size=12)
plt.show()
