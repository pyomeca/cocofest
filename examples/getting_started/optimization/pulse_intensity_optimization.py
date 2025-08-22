"""
This example will do a 10 stimulation example with Ding's 2003 frequency model associated to Hmed's 2018 work.
This ocp was build to match a force value of 200N at the end of the last node.
"""

import numpy as np
from bioptim import OdeSolver, ObjectiveList, ObjectiveFcn, OptimalControlProgram, ControlType, Node
from cocofest import DingModelPulseIntensityFrequencyWithFatigue, OcpFes

# --- Build ocp --- #
# This ocp was build to match a force value of 135 N at the end of the last node.
# The stimulation won't be optimized and is already set to one pulse every 0.1 seconds (n_stim/final_time).
# Plus the pulsation intensity will be optimized between 0 and 130 mA and are not the same across the problem.


def prepare_ocp(model, final_time, pi_max):
    # --- Set dynamics --- #
    n_shooting = model.get_n_shooting(final_time=final_time)
    numerical_data_time_series, stim_idx_at_node_list = model.get_numerical_data_time_series(n_shooting, final_time)
    dynamics = OcpFes.declare_dynamics(
        model, numerical_data_time_series, ode_solver=OdeSolver.RK4(n_integration_steps=10)
    )

    # --- Set initial guesses and bounds for states and controls --- #
    x_bounds, x_init = OcpFes.set_x_bounds(model)
    u_bounds, u_init = OcpFes.set_u_bounds(model, max_bound=pi_max)

    # --- Set objective functions --- #
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="F", weight=1, quadratic=True)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="F", node=Node.END, target=135, weight=1e5, quadratic=True
    )

    # --- Set parameters (required for intensity models) --- #
    use_sx = True
    parameters, parameters_bounds, parameters_init = OcpFes.set_parameters(
        model=model,
        max_pulse_intensity=pi_max,
        use_sx=use_sx,
    )

    # --- Set constraints (required for intensity models) --- #
    constraints = OcpFes.set_constraints(
        model,
        n_shooting,
        stim_idx_at_node_list,
    )

    return OptimalControlProgram(
        bio_model=[model],
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=final_time,
        objective_functions=objective_functions,
        x_init=x_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        u_init=u_init,
        parameters=parameters,
        parameter_bounds=parameters_bounds,
        parameter_init=parameters_init,
        constraints=constraints,
        control_type=ControlType.CONSTANT,
        use_sx=use_sx,
        n_threads=20,
    )


def main(plot=True):
    final_time = 1
    model = DingModelPulseIntensityFrequencyWithFatigue(
        stim_time=list(np.linspace(0, final_time, 11)[:-1]), sum_stim_truncation=10
    )
    pi_max = 130
    ocp = prepare_ocp(model=model, final_time=final_time, pi_max=pi_max)
    # --- Solve the program --- #
    sol = ocp.solve()
    # --- Show results --- #
    if plot:
        sol.graphs()


if __name__ == "__main__":
    main()
