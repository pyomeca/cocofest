import numpy as np
import matplotlib.pyplot as plt
from bioptim import (
    SolutionMerge,
    OdeSolver,
    OptimalControlProgram,
    ObjectiveFcn,
    Node,
    ControlType,
    ObjectiveList,
    BoundsList,
    InitialGuessList,
)
import pytest
from cocofest import (
    DingModelPulseWidthFrequency,
    IvpFes,
    ModelMaker,
    OcpFesId,
    FES_plot,
    DingModelFrequency,
    DingModelPulseIntensityFrequency,
)
from cocofest.identification.identification_method import DataExtraction

stim_time = np.linspace(0, 1, 34).tolist()
tested_index = np.linspace(0, 726, 34, dtype=int)  # Test values at 34 different times out of 727


def check_values(result, tested_values):
    for i in range(len(tested_values)):
        np.testing.assert_almost_equal(result[i], tested_values[i])


ding2003_model = ModelMaker.create_model("ding2003", stim_time=stim_time, sum_stim_truncation=10)
ding2003_with_fatigue_model = ModelMaker.create_model(
    "ding2003_with_fatigue", stim_time=stim_time, sum_stim_truncation=10
)


def simulate_data_ding2003(model, final_time: int, n_integration_steps):
    """
    Simulate the data using the pulse intensity method.
    Returns a dictionary with time, force, stim_time.
    """
    stim_time = model.stim_time

    fes_parameters = {"model": model}
    ivp_parameters = {
        "final_time": final_time,
        "use_sx": True,
        "ode_solver": OdeSolver.RK4(n_integration_steps=n_integration_steps),
    }
    ivp = IvpFes(fes_parameters, ivp_parameters)

    result, time = ivp.integrate()
    data = {
        "time": time,
        "force": result["F"][0],
        "stim_time": stim_time,
    }
    return data


def prepare_ocp_ding2003(
    model,
    final_time,
    simulated_data,  # Dictionary from simulate_data
    key_parameter_to_identify,  # List of parameters to identify
):
    n_shooting = model.get_n_shooting(final_time)
    force_at_node = DataExtraction.force_at_node_in_ocp(
        simulated_data["time"], simulated_data["force"], n_shooting, final_time
    )

    numerical_data_time_series, stim_idx_at_node_list = model.get_numerical_data_time_series(n_shooting, final_time)
    dynamics = OcpFesId.declare_dynamics(model=model, numerical_data_timeseries=numerical_data_time_series)

    x_bounds, x_init = OcpFesId.set_x_bounds(
        model=model,
        force_tracking=force_at_node,
    )
    u_bounds, u_init = BoundsList(), InitialGuessList()

    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_STATE,
        key="F",
        weight=1,
        target=np.array(force_at_node)[np.newaxis, :],
        node=Node.ALL,
        quadratic=True,
    )
    additional_key_settings = OcpFesId.set_default_values(model)
    parameters, parameters_bounds, parameters_init = OcpFesId.set_parameters(
        parameter_to_identify=key_parameter_to_identify,
        parameter_setting=additional_key_settings,
        use_sx=True,
    )
    OcpFesId.update_model_param(model, parameters)

    return OptimalControlProgram(
        bio_model=[model],
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=final_time,
        x_init=x_init,
        x_bounds=x_bounds,
        u_init=u_init,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        parameters=parameters,
        parameter_bounds=parameters_bounds,
        parameter_init=parameters_init,
        control_type=ControlType.CONSTANT,
        use_sx=True,
        ode_solver=OdeSolver.RK4(n_integration_steps=10),
        n_threads=20,
    )


@pytest.mark.parametrize("model", [ding2003_model, ding2003_with_fatigue_model])
def test_ding2003_id(model):
    # Parameters for simulation and identification
    n_stim = 33
    final_time = 2
    integration_steps = 10
    stim_time = list(np.linspace(0, 1, n_stim + 1)[:-1])
    model = ModelMaker.create_model("ding2003", stim_time=stim_time, sum_stim_truncation=10)

    sim_data = simulate_data_ding2003(model, final_time, n_integration_steps=integration_steps)

    ocp = prepare_ocp_ding2003(
        model,
        final_time,
        simulated_data=sim_data,
        key_parameter_to_identify=[
            "a_rest",
            "km_rest",
            "tau1_rest",
            "tau2",
        ],
    )
    sol = ocp.solve()

    result_force = [sol.stepwise_states(to_merge=SolutionMerge.NODES)["F"][0].tolist()[i] for i in tested_index]
    result_param = sol.parameters

    if model._with_fatigue:
        pass
    else:
        tested_values_force = [
            0.0,
            107.87317924748636,
            178.7303591800682,
            218.77104258782373,
            241.05914664185326,
            253.4393116503343,
            260.3142001309098,
            264.13181451221783,
            266.2517237996478,
            267.42890249475937,
            268.08258596111745,
            268.44557424419503,
            268.6471404462057,
            268.75906949649874,
            268.8212233306958,
            268.8557371507923,
            268.8749025617931,
            257.24663940949284,
            159.2458935385931,
            57.94842473471988,
            18.142634108710155,
            5.5493362036915785,
            1.6907791804003263,
            0.5147840777942536,
            0.15671358723425183,
            0.047706532911022004,
            0.014522694341757077,
            0.004420956155577021,
            0.001345814368518724,
            0.0004096888113336389,
            0.0001247162501628999,
            3.796575013442903e-05,
            1.155742079494496e-05,
            3.5182756815234537e-06,
        ]
        tested_values_param = {
            "a_rest": [3009.00005647],
            "km_rest": [0.10300001],
            "tau1_rest": [0.050957],
            "tau2": [0.06],
        }

    check_values(result=result_force, tested_values=tested_values_force)
    for key in tested_values_param.keys():
        np.testing.assert_almost_equal(result_param[key], tested_values_param[key])


ding2007_model = ModelMaker.create_model("ding2007", stim_time=stim_time, sum_stim_truncation=10)
ding2007_with_fatigue_model = ModelMaker.create_model(
    "ding2007_with_fatigue", stim_time=stim_time, sum_stim_truncation=10
)


def simulate_data_ding2007(model, final_time: int, pulse_width_values: list, n_integration_steps):
    """
    Returns a dictionary with time, force, stim_time, and pulse_width.
    """
    stim_time = model.stim_time

    fes_parameters = {"model": model, "pulse_width": pulse_width_values}
    ivp_parameters = {
        "final_time": final_time,
        "use_sx": True,
        "ode_solver": OdeSolver.RK4(n_integration_steps=n_integration_steps),
    }
    ivp = IvpFes(fes_parameters, ivp_parameters)

    result, time = ivp.integrate()
    data = {
        "time": time,
        "force": result["F"][0],
        "stim_time": stim_time,
        "pulse_width": pulse_width_values,
    }
    return data


def prepare_ocp_ding2007(
    model,
    final_time,
    pulse_width_values,
    simulated_data,
    key_parameter_to_identify,
):
    n_shooting = model.get_n_shooting(final_time)
    force_at_node = DataExtraction.force_at_node_in_ocp(
        simulated_data["time"], simulated_data["force"], n_shooting, final_time
    )

    numerical_data_time_series, stim_idx_at_node_list = model.get_numerical_data_time_series(n_shooting, final_time)
    dynamics = OcpFesId.declare_dynamics(model=model, numerical_data_timeseries=numerical_data_time_series)

    x_bounds, x_init = OcpFesId.set_x_bounds(
        model=model,
        force_tracking=force_at_node,
    )
    u_bounds, u_init = OcpFesId.set_u_bounds(
        model=model,
        control_value=pulse_width_values,
        stim_idx_at_node_list=stim_idx_at_node_list,
        n_shooting=n_shooting,
    )

    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_STATE,
        key="F",
        weight=1,
        target=np.array(force_at_node)[np.newaxis, :],
        node=Node.ALL,
        quadratic=True,
    )
    additional_key_settings = OcpFesId.set_default_values(model)
    parameters, parameters_bounds, parameters_init = OcpFesId.set_parameters(
        parameter_to_identify=key_parameter_to_identify,
        parameter_setting=additional_key_settings,
        use_sx=True,
    )
    OcpFesId.update_model_param(model, parameters)

    return OptimalControlProgram(
        bio_model=[model],
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=final_time,
        x_init=x_init,
        x_bounds=x_bounds,
        u_init=u_init,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        parameters=parameters,
        parameter_bounds=parameters_bounds,
        parameter_init=parameters_init,
        control_type=ControlType.CONSTANT,
        use_sx=True,
        ode_solver=OdeSolver.RK4(n_integration_steps=10),
        n_threads=20,
    )


@pytest.mark.parametrize("model", [ding2007_model, ding2007_with_fatigue_model])
def test_ding2007_id(model):
    # Parameters for simulation and identification
    n_stim = 33
    final_time = 2
    integration_steps = 10
    stim_time = list(np.linspace(0, 1, n_stim + 1)[:-1])
    model = ModelMaker.create_model("ding2007", stim_time=stim_time, sum_stim_truncation=10)
    np.random.seed(7)
    pulse_width_values = list(np.random.uniform(0.0002, 0.0006, n_stim))

    sim_data = simulate_data_ding2007(
        model, final_time, pulse_width_values=pulse_width_values, n_integration_steps=integration_steps
    )

    ocp = prepare_ocp_ding2007(
        model,
        final_time,
        pulse_width_values,
        simulated_data=sim_data,
        key_parameter_to_identify=[
            "km_rest",
            "tau1_rest",
            "tau2",
            "pd0",
            "pdt",
            "a_scale",
        ],
    )
    sol = ocp.solve()

    result_force = [sol.stepwise_states(to_merge=SolutionMerge.NODES)["F"][0].tolist()[i] for i in tested_index]
    result_param = sol.parameters
    if model._with_fatigue:
        pass
    else:
        tested_values_force = [
            0.0,
            90.25994758759484,
            140.9028421065209,
            163.40179387925275,
            132.36402502358163,
            142.71532162865827,
            168.18346654092912,
            129.89896905380124,
            154.62535727666264,
            146.33750754522316,
            128.35683032619974,
            164.20865767072732,
            154.88240594142906,
            142.49415905470514,
            163.43282741422757,
            164.46647214487567,
            151.36703018215917,
            117.68447116780446,
            46.598428770371946,
            17.167134312209058,
            6.315060889833131,
            2.3229881601844444,
            0.8545083328222599,
            0.3143298360241792,
            0.1156258423866465,
            0.04253282347203855,
            0.015645646640603417,
            0.005755231814402106,
            0.0021170549225844396,
            0.0007787560414201352,
            0.00028646444906965836,
            0.00010537559417341437,
            3.8762282312734746e-05,
            1.425865772694475e-05,
        ]
        tested_values_param = {
            "km_rest": [0.13700005],
            "tau1_rest": [0.060601],
            "tau2": [0.001],
            "pd0": [0.0001314],
            "pdt": [0.00019414],
            "a_scale": [4920.00051479],
        }

    check_values(result=result_force, tested_values=tested_values_force)
    for key in tested_values_param.keys():
        np.testing.assert_almost_equal(result_param[key], tested_values_param[key])


hmed2018_model = ModelMaker.create_model("hmed2018", stim_time=stim_time, sum_stim_truncation=10)
hmed2018_with_fatigue_model = ModelMaker.create_model(
    "hmed2018_with_fatigue", stim_time=stim_time, sum_stim_truncation=10
)


def simulate_data_hmed2018(model, final_time: int, pulse_intensity_values: list, n_integration_steps):
    """
    Simulate the data using the pulse intensity method.
    Returns a dictionary with time, force, stim_time, and pulse_intensity.
    """
    stim_time = model.stim_time

    fes_parameters = {"model": model, "pulse_intensity": pulse_intensity_values}
    ivp_parameters = {
        "final_time": final_time,
        "use_sx": True,
        "ode_solver": OdeSolver.RK4(n_integration_steps=n_integration_steps),
    }
    ivp = IvpFes(fes_parameters, ivp_parameters)

    result, time = ivp.integrate()
    data = {
        "time": time,
        "force": result["F"][0],
        "stim_time": stim_time,
        "pulse_intensity": pulse_intensity_values,
    }
    return data


def prepare_ocp_hmed2018(
    model,
    final_time,
    pulse_intensity_values,
    simulated_data,
    key_parameter_to_identify,
):
    n_shooting = model.get_n_shooting(final_time)
    force_at_node = DataExtraction.force_at_node_in_ocp(
        simulated_data["time"], simulated_data["force"], n_shooting, final_time
    )

    numerical_data_time_series, stim_idx_at_node_list = model.get_numerical_data_time_series(n_shooting, final_time)
    dynamics = OcpFesId.declare_dynamics(model=model, numerical_data_timeseries=numerical_data_time_series)

    x_bounds, x_init = OcpFesId.set_x_bounds(
        model=model,
        force_tracking=force_at_node,
    )
    u_bounds, u_init = OcpFesId.set_u_bounds(
        model=model,
        control_value=pulse_intensity_values,
        stim_idx_at_node_list=stim_idx_at_node_list,
        n_shooting=n_shooting,
    )

    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_STATE,
        key="F",
        weight=1,
        target=np.array(force_at_node)[np.newaxis, :],
        node=Node.ALL,
        quadratic=True,
    )
    additional_key_settings = OcpFesId.set_default_values(model)
    parameters, parameters_bounds, parameters_init = OcpFesId.set_parameters(
        parameter_to_identify=key_parameter_to_identify,
        parameter_setting=additional_key_settings,
        use_sx=True,
    )
    OcpFesId.update_model_param(model, parameters)

    return OptimalControlProgram(
        bio_model=[model],
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=final_time,
        x_init=x_init,
        x_bounds=x_bounds,
        u_init=u_init,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        parameters=parameters,
        parameter_bounds=parameters_bounds,
        parameter_init=parameters_init,
        control_type=ControlType.CONSTANT,
        use_sx=True,
        ode_solver=OdeSolver.RK4(n_integration_steps=10),
        n_threads=20,
    )


@pytest.mark.parametrize("model", [hmed2018_model, hmed2018_with_fatigue_model])
def test_hmed2018_id(model):
    # Parameters for simulation and identification
    n_stim = 33
    final_time = 2
    integration_steps = 10
    stim_time = list(np.linspace(0, 1, n_stim + 1)[:-1])
    model = ModelMaker.create_model("hmed2018", stim_time=stim_time, sum_stim_truncation=10)
    np.random.seed(7)
    pulse_intensity_values = list(np.random.randint(40, 130, n_stim))

    sim_data = simulate_data_hmed2018(
        model, final_time, pulse_intensity_values=pulse_intensity_values, n_integration_steps=integration_steps
    )

    ocp = prepare_ocp_hmed2018(
        model,
        final_time,
        pulse_intensity_values,
        simulated_data=sim_data,
        key_parameter_to_identify=[
            "a_rest",
            "km_rest",
            "tau1_rest",
            "tau2",
            "ar",
            "bs",
            "Is",
            "cr",
        ],
    )
    sol = ocp.solve()

    result_force = [sol.stepwise_states(to_merge=SolutionMerge.NODES)["F"][0].tolist()[i] for i in tested_index]
    result_param = sol.parameters
    if model._with_fatigue:
        pass
    else:
        tested_values_force = [
            0.0,
            104.04035782777072,
            170.50460362940066,
            209.32823006643278,
            224.87815826270773,
            230.85336734772173,
            244.90580998514395,
            239.72260536328864,
            247.446605123523,
            241.8202060647484,
            235.38838075758986,
            238.79684986564615,
            234.80303660023972,
            244.78038612594935,
            253.62816351295785,
            252.5331694714469,
            256.09052428207275,
            248.03028764437445,
            152.33433411401035,
            55.15270603406743,
            17.256399862334085,
            5.277869614643382,
            1.6080496723596438,
            0.4895944387805321,
            0.14904501724892028,
            0.04537203054510817,
            0.013812017755770527,
            0.004204609942214798,
            0.0012799536416889337,
            0.000389639302963552,
            0.00011861272226183607,
            3.610769695395068e-05,
            1.0991787004840968e-05,
            3.3460838477592833e-06,
        ]
        tested_values_param = {
            "a_rest": [3009.00944301],
            "km_rest": [0.10313384],
            "tau1_rest": [0.05095696],
            "tau2": [0.05999946],
            "ar": [0.58673292],
            "bs": [0.02600106],
            "Is": [63.10015773],
            "cr": [0.83302601],
        }

    check_values(result=result_force, tested_values=tested_values_force)
    for key in tested_values_param.keys():
        np.testing.assert_almost_equal(result_param[key], tested_values_param[key])
