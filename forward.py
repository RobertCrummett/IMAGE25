import numpy as np
from scipy.interpolate import LinearNDInterpolator

from discretize import TensorMesh
from discretize.utils import active_from_xyz
from simpeg import maps
from simpeg.utils import model_builder
from simpeg.potential_fields import magnetics

from collections import ChainMap
from pathlib import Path

from topography import load_topo

FORWARD_SEED = 12345
np.random.seed(FORWARD_SEED)

outdir = Path("outputs")

def create_tmi_receivers(xyz_topo):
    x_linspace = np.linspace(-80.0, 80.0, 17)
    y_linspace = np.linspace(-80.0, 80.0, 17)
    z_offset = 10

    x_recv, y_recv = np.meshgrid(x_linspace, y_linspace)
    xy_flat = np.column_stack((x_recv.flatten(), y_recv.flatten()))

    xy_topo, z_topo = xyz_topo[...,:-1], xyz_topo[...,-1]

    interp = LinearNDInterpolator(xy_topo, z_topo)
    z_flat = interp(xy_flat) + z_offset

    receiver_locations = np.column_stack((xy_flat, z_flat))

    return magnetics.receivers.Point(receiver_locations, components="tmi")

def create_survey(receivers, background_field):
    source_field = magnetics.sources.UniformBackgroundField(
        receiver_list=[receivers],
        amplitude=background_field["strength"],
        inclination=background_field["inclination"],
        declination=background_field["declination"],
    )
    return magnetics.survey.Survey(source_field)

def create_mesh():
    dh = 5.0
    hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
    hy = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
    hz = [(dh, 5, -1.3), (dh, 15)]
    return TensorMesh([hx, hy, hz], "CCN")

def create_simulation_params(xyz_topo):
    mesh = create_mesh()  
    ind_active = active_from_xyz(mesh, xyz_topo)
    
    nC = int(ind_active.sum())
    model_map = maps.IdentityMap(nP=nC)

    return dict(
            mesh=mesh,
            model_type="scalar",
            chiMap=model_map,
            active_cells=ind_active,
            store_sensitivities="forward_only",
            engine="choclo",
        )

def create_model(xyz_topo, susceptibility, simulation_params):
    active_cells = simulation_params["active_cells"]
    nC = int(active_cells.sum())
    model = susceptibility["background"] * np.ones(nC)

    mesh = simulation_params["mesh"]
    ind_sphere = model_builder.get_indices_sphere(
            np.r_[0.0, 0.0, -45.0], 15.0, mesh.cell_centers
        )

    ind_sphere = ind_sphere[active_cells]
    model[ind_sphere] = susceptibility["sphere"]

    return model

# this is multiplicative noise, so `add` is deceptive
# `add` here means insert, or put into, not addition the math op
def add_noise_to_model(model, sigma):
    np.random.seed(FORWARD_SEED)
    noise = np.random.lognormal(mean=0, sigma=sigma, size=model.shape)
    return model * noise

def survey_the_model(model, survey, simulation_params):
    simulation = magnetics.simulation.Simulation3DIntegral(
        survey=survey,
        **simulation_params.maps[0],
    )
    return simulation.dpred(model)

def write_survey_results(receivers, survey_results, background_field):
    create_anom_path = lambda i, d: outdir / Path(f"anom_i{i}_d{d}.txt")

    inclination = background_field["inclination"]
    declination = background_field["declination"]

    fsurvey = create_anom_path(inclination, declination)

    survey_with_xyz = np.column_stack(
            (receivers.locations, survey_results)
        )

    try:
        fsurvey.parent.mkdir()
    except FileExistsError as _:
        pass
    finally:
        np.savetxt(fsurvey, survey_with_xyz, fmt="%.4e", delimiter=" ")

if __name__ == "__main__":
    background_field = dict(
            strength = 50000, inclination = 0, declination = 0
        )
    susceptibility = dict(
            background = 1e-4, sphere = 1e-2
        )
    
    xyz_topo = load_topo()
    receivers = create_tmi_receivers(xyz_topo)

    survey = create_survey(receivers, background_field)
    simulation_params = create_simulation_params(xyz_topo)

    params = ChainMap(simulation_params, background_field, susceptibility)

    model = create_model(xyz_topo, susceptibility, params)
    noisy_model = add_noise_to_model(model, sigma = 0.1)

    results = survey_the_model(noisy_model, survey, params)

    write_survey_results(receivers, results, params)
