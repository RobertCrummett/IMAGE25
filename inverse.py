import numpy as np
from scipy.interpolate import LinearNDInterpolator

from discretize import TensorMesh
from discretize.utils import active_from_xyz
from simpeg import maps, data, data_misfit, regularization, optimization, inversion, inverse_problem, directives
from simpeg.utils import model_builder
from simpeg.potential_fields import magnetics

from collections import ChainMap
from pathlib import Path

from topography import load_topo

INVERSE_SEED = 12345

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

def create_survey_params(**kwargs):
    xyz_topo = load_topo()

    # Defaults
    background_field = dict(strength = 50000, inclination = 0, declination = 0)
    susceptibility = dict(background = 1e-4, sphere = 1e-2)
    simulation_params = create_simulation_params(xyz_topo)
    params = ChainMap(simulation_params, background_field, susceptibility)

    for key, value in kwargs.items():
        params[key] = value
    return params


def create_survey_results(**kwargs):
    xyz_topo = load_topo()
    receivers = create_tmi_receivers(xyz_topo)

    params = create_survey_params(**kwargs)
    survey = create_survey(receivers, params)
 
    model = create_model(xyz_topo, params)
    noisy_model = add_noise_to_model(model, sigma = 0.1)
 
    results = survey_the_model(noisy_model, survey, params)
    
    return receivers, results, params

def invert_survey_results(receivers, results, params):
    survey = create_survey(receivers, params)
    maximum_anomaly = np.max(np.abs(results))
    std = 0.02 * maximum_anomaly * np.ones(len(results))
    data_object = data.Data(survey, dobs=results, standard_deviation=std)

    simulation = magnetics.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=params["mesh"],
        model_type=params["model_type"],
        chiMap=params["chiMap"],
        active_cells=params["active_cells"],
        store_sensitivities=params["store_sensitivities"],
        engine=params["engine"],
    )

    dmis = data_misfit.L2DataMisfit(data=data_object, simulation=simulation)

    nC = int(params["active_cells"].sum())
    model_map = maps.IdentityMap(nP=nC)
    starting_model = params["background"] * np.ones(nC)

    reg = regularization.Sparse(
        params["mesh"],
        active_cells=params["active_cells"],
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
    return inv.run(starting_model)

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

def load_survey_results(declination, inclination):
    params = {}
    params["inclination"] = inclination
    params["declination"] = declination

    path = Path(f"outputs/anom_i{inclination}_d{declination}.txt")

    if path.exists():
        survey_results_xyz = np.loadtxt(path)
        receiver_locations = survey_results_xyz[...,:-1]
        survey_results = survey_results_xyz[...,-1]
        return receiver_locations, survey_results, params
    else:
        receivers, survey_results, params = create_survey_results(**params)
        return receivers.locations, survey_results, params

if __name__ == "__main__":
    np.random.seed(INVERSE_SEED)

    inclination = 0
    results = create_survey_results(inclination = inclination)

    inversion_result = invert_survey_results(*results)

    receiver_locations, survey_results, params = results

    dobs = survey_results
    dpred = inversion_result.dpred
    
    data_array = np.c_[dobs, dpred]
    
    fig = plt.figure(figsize=(17, 4))
    plot_title = ["Observed", "Predicted", "Normalized Misfit"]
    plot_units = ["nT", "nT", ""]
    
    ax1 = 3 * [None]
    ax2 = 3 * [None]
    norm = 3 * [None]
    cbar = 3 * [None]
    cplot = 3 * [None]
    v_lim = [np.max(np.abs(dobs)), np.max(np.abs(dobs))]
    
    for ii in range(2):
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
