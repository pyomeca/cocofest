"""
This example will do a 10 stimulation example with Ding's 2003 frequency model associated to Hmed's 2018 intensity work
This ocp was build to match a force curve across all optimization.
"""

import matplotlib.pyplot as plt
import numpy as np

from bioptim import SolutionMerge, ObjectiveList, ObjectiveFcn, OptimalControlProgram, ControlType, OdeSolver, Node
from cocofest import (
    ModelMaker,
    FourierSeries,
    OcpFes,
)


def prepare_ocp(model, final_time, pi_max, force_tracking):
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
    force_to_track = OcpFes.check_and_adjust_dimensions_for_objective_fun(
        force_to_track=force_tracking, n_shooting=n_shooting, final_time=final_time
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_STATE,
        key="F",
        weight=100,
        target=force_to_track,
        node=Node.ALL,
        quadratic=True,
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
    model = ModelMaker.create_model("hmed2018", stim_time=list(np.linspace(0, 1, 34)[:-1]))

    # --- Building force to track ---#
    time = np.linspace(0, 1, 1001)
    force = abs(np.sin(time * 5) + np.random.normal(scale=0.1, size=len(time))) * 100
    force_tracking = [time, force]

    ocp = prepare_ocp(model=model, final_time=final_time, pi_max=130, force_tracking=force_tracking)
    sol = ocp.solve()

    # --- Show results from solution --- #
    if plot:
        sol_merged = sol.stepwise_states(to_merge=[SolutionMerge.NODES])
        sol_time = sol.stepwise_time(to_merge=[SolutionMerge.NODES]).T[0]

        fourier_fun = FourierSeries()
        fourier_coef = fourier_fun.compute_real_fourier_coeffs(time, force, 50)
        y_approx = FourierSeries().fit_func_by_fourier_series_with_real_coeffs(time, fourier_coef)
        plt.title("Comparison between given and simulated force after parameter optimization")
        plt.plot(time, force, color="red", label="force from file")
        plt.plot(time, y_approx, color="orange", label="force after fourier transform")
        plt.plot(
            sol_time,
            sol_merged["F"].squeeze(),
            color="blue",
            label="force from optimized stimulation",
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Force (N)")
        plt.legend()
        plt.show()

        # --- Show the optimization results --- #
        sol.graphs()


if __name__ == "__main__":
    main()
