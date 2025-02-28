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


def extract_identified_parameters(identified, keys):
    """
    For each parameter in keys, use the identified value if available.
    Returns a dictionary mapping parameter names to their values.
    """
    return {key: identified.parameters[key][0] for key in keys}


def annotate_parameters(ax, identified_params, default_model, start_x=0.7, y_start=0.4, y_step=0.05):
    """
    Annotate the plot with parameter names, the identified values, and default values.
    The names are annotated in black, identified values in red, and default values in blue.
    """
    for i, key in enumerate(identified_params.keys()):
        y = y_start - i * y_step
        ax.annotate(f"{key} :", xy=(start_x, y), xycoords="axes fraction", color="black")
        ax.annotate(
            f"{round(identified_params[key], 5)}", xy=(start_x + 0.08, y), xycoords="axes fraction", color="red"
        )
        ax.annotate(f"{getattr(default_model, key)}", xy=(start_x + 0.15, y), xycoords="axes fraction", color="blue")


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


def main():
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

    param_keys = [
        "km_rest",
        "tau1_rest",
        "tau2",
        "pd0",
        "pdt",
        "a_scale",
    ]
    identified_params = extract_identified_parameters(sol, param_keys)

    print("Identified parameters:")
    for key, value in identified_params.items():
        print(f"  {key}: {value}")

    sim_data_time = sim_data["time"]
    sim_data_force = sim_data["force"]
    sol_time = sol.stepwise_time(to_merge=SolutionMerge.NODES).T[0]
    sol_force = sol.stepwise_states(to_merge=SolutionMerge.NODES)["F"][0]

    # Plot the simulation and identification results
    fig, ax = plt.subplots()
    ax.set_title("Force state result")
    ax.plot(sim_data_time, sim_data_force, color="blue", label="simulated")
    ax.plot(sol_time, sol_force, color="green", label="identified")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("force (N)")

    default_model = DingModelPulseWidthFrequency()
    annotate_parameters(ax, identified_params, default_model)

    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
