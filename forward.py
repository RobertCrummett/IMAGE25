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

def create_model(xyz_topo, params):
    active_cells = params["active_cells"]
    nC = int(active_cells.sum())
    model = params["background"] * np.ones(nC)

    mesh = params["mesh"]
    ind_sphere = model_builder.get_indices_sphere(
            np.r_[0.0, 0.0, -45.0], 15.0, mesh.cell_centers
        )

    ind_sphere = ind_sphere[active_cells]
    model[ind_sphere] = params["sphere"]

    return model

"""
this is multiplicative noise, so `add` is deceptive
`add` here means insert, or put into, not addition the math op
"""
def add_noise_to_model(model, sigma):
    noise = np.random.lognormal(mean=0, sigma=sigma, size=model.shape)
    return model * noise

def survey_the_model(model, survey, params):
    simulation = magnetics.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=params["mesh"],
        model_type=params["model_type"],
        chiMap=params["chiMap"],
        active_cells=params["active_cells"],
        store_sensitivities=params["store_sensitivities"],
        engine=params["engine"],
    )
    return simulation.dpred(model)

def write_survey_results(receivers, survey_results, params):
    inclination = params["inclination"]
    declination = params["declination"]
    fsurvey = outdir / Path(f"anom_i{inclination}_d{declination}.txt")

    survey_with_xyz = np.column_stack((receivers.locations, survey_results))

    try:
        fsurvey.parent.mkdir()
    except FileExistsError as _:
        pass
    finally:
        np.savetxt(fsurvey, survey_with_xyz, fmt="%.4e", delimiter=" ")

def load_survey_results(path):
    survey_results_xyz = np.loadtxt(path)

    receiver_locations = survey_results_xyz[...,:-1]
    survey_results = survey_results_xyz[...,-1]

    return receiver_locations, survey_results

if __name__ == "__main__":
    np.random.seed(FORWARD_SEED)

    xyz_topo = load_topo()
    receivers = create_tmi_receivers(xyz_topo)

    background_field = dict(strength = 50000, inclination = 0, declination = 0)
    susceptibility = dict(background = 1e-4, sphere = 1e-2)
    simulation_params = create_simulation_params(xyz_topo)

    params = ChainMap(simulation_params, background_field, susceptibility)

    for inclination in range(0, 100, 10):
        params["inclination"] = inclination

        survey = create_survey(receivers, params)

        model = create_model(xyz_topo, params)
        noisy_model = add_noise_to_model(model, sigma = 0.1)

        results = survey_the_model(noisy_model, survey, params)

        write_survey_results(receivers, results, params)

        print(f"Wrote forward model for {inclination} degree inclination")

