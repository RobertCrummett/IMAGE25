import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib as mpl
import matplotlib.pyplot as plt

from discretize import TensorMesh
from discretize.utils import mkvc, active_from_xyz
from simpeg.utils import plot2Ddata, model_builder
from simpeg import maps
from simpeg.potential_fields import magnetics

from pathlib import Path

np.random.seed(12345)

inclination = 0
declination = 0
strength = 50000

background_susceptibility = 0.0001
sphere_susceptibility = 0.01
noise_sigma = 0.05;

outdir = Path("outputs")
ftopo = Path("topo.txt")
fanom = Path(f"anom_i{inclination}_d{declination}.txt")

xyz_topo = np.loadtxt(outdir / ftopo)

x_topo = xyz_topo[:,0]
y_topo = xyz_topo[:,1]
z_topo = xyz_topo[:,2]

x = np.linspace(-80.0, 80.0, 17)
y = np.linspace(-80.0, 80.0, 17)
x, y = np.meshgrid(x, y)
x, y = mkvc(x.T), mkvc(y.T)
fun_interp = LinearNDInterpolator(np.c_[x_topo, y_topo], z_topo)
z = fun_interp(np.c_[x, y]) + 10
receiver_locations = np.c_[x, y, z]

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

dh = 5.0
hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
hy = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
hz = [(dh, 5, -1.3), (dh, 15)]
mesh = TensorMesh([hx, hy, hz], "CCN")

ind_active = active_from_xyz(mesh, xyz_topo)

nC = int(ind_active.sum())
model_map = maps.IdentityMap(nP=nC)

model = background_susceptibility * np.ones(nC) * \
        np.random.lognormal(mean=0, sigma=noise_sigma, size=nC)
ind_sphere = model_builder.get_indices_sphere(
    np.r_[0.0, 0.0, -45.0], 15.0, mesh.cell_centers
)
ind_sphere = ind_sphere[ind_active]
model[ind_sphere] = sphere_susceptibility 

simulation = magnetics.simulation.Simulation3DIntegral(
    survey=survey,
    mesh=mesh,
    model_type="scalar",
    chiMap=model_map,
    active_cells=ind_active,
    store_sensitivities="forward_only",
    engine="choclo",
)

dpred = simulation.dpred(model)

if not outdir.is_dir():
    outdir.mkdir()

np.savetxt(outdir / fanom, np.c_[receiver_locations, dpred], fmt="%.4e")
