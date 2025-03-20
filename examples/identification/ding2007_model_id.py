"""
This example demonstrates the way of identifying the Ding 2007 model parameter using simulated data.
First we integrate the model with a given parameter set.
Finally, we use the data to identify the model parameters.
"""

import matplotlib.pyplot as plt
import numpy as np

from bioptim import SolutionMerge, OdeSolver, OptimalControlProgram, ObjectiveFcn, Node, ControlType, ObjectiveList

from cocofest import (
    DingModelPulseWidthFrequency,
    IvpFes,
    ModelMaker,
    OcpFesId,
    FES_plot,
)
from cocofest.identification.identification_method import DataExtraction


def simulate_data(model, final_time: int, pulse_width_values: list, n_integration_steps):
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


def prepare_ocp(
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


def main(plot=True):
    # Parameters for simulation and identification
    n_stim = 33
    final_time = 2
    integration_steps = 10
    stim_time = list(np.linspace(0, 1, n_stim + 1)[:-1])
    model = ModelMaker.create_model("ding2007", stim_time=stim_time, sum_stim_truncation=10)
    pulse_width_values = list(np.random.uniform(0.0002, 0.0006, n_stim))

    sim_data = simulate_data(
        model, final_time, pulse_width_values=pulse_width_values, n_integration_steps=integration_steps
    )

    ocp = prepare_ocp(
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

    if plot:
        default_model = DingModelPulseWidthFrequency()

        FES_plot(data=sol).plot(
            title="Identification of Ding 2007 parameters",
            tracked_data=sim_data,
            default_model=default_model,
            show_bounds=False,
            show_stim=False,
        )


if __name__ == "__main__":
    main()
