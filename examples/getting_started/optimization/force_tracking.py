"""
This example will do a 10 stimulation example with Ding's 2007 model.
This ocp was build to match a force curve across all optimization.
"""

import matplotlib.pyplot as plt
import numpy as np

from bioptim import SolutionMerge, ObjectiveList, ObjectiveFcn, OptimalControlProgram, ControlType, OdeSolver, Node
from cocofest import ModelMaker, OcpFes


def prepare_ocp(model, final_time, pw_max, force_tracking):
    # --- Set dynamics --- #
    n_shooting = model.get_n_shooting(final_time=final_time)
    numerical_data_time_series, stim_idx_at_node_list = model.get_numerical_data_time_series(n_shooting, final_time)
    dynamics = OcpFes.declare_dynamics(
        model, numerical_data_time_series, ode_solver=OdeSolver.RK4(n_integration_steps=10)
    )

    # --- Set initial guesses and bounds for states and controls --- #
    x_bounds, x_init = OcpFes.set_x_bounds(model)
    u_bounds, u_init = OcpFes.set_u_bounds(model, max_bound=pw_max)

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
        constraints=constraints,
        control_type=ControlType.CONSTANT,
        n_threads=20,
    )


def main(plot=True):
    final_time = 1
    stim = 33
    model = ModelMaker.create_model("ding2007", stim_time=list(np.linspace(0, 1, stim, endpoint=False)))

    # --- Building force to track ---#
    time = np.linspace(0, 1, 1001)
    force = abs(np.sin(time * 5) + np.random.normal(scale=0.02, size=len(time))) * 100
    force_tracking = [time, force]

    ocp = prepare_ocp(model=model, final_time=final_time, pw_max=0.0006, force_tracking=force_tracking)
    sol = ocp.solve()

    # --- Show results from solution --- #
    if plot:
        sol_state = sol.stepwise_states(to_merge=[SolutionMerge.NODES])
        sol_control = sol.stepwise_controls(to_merge=[SolutionMerge.NODES])
        sol_time = sol.stepwise_time(to_merge=[SolutionMerge.NODES]).T[0]

        fig = plt.figure()
        gs = fig.add_gridspec(2, height_ratios=[3, 1], hspace=0)
        axs = gs.subplots(sharex=True, sharey=False)

        axs[0].plot(sol_time, sol_state["F"].squeeze(), color="red", label="Optimized force profile", lw=2)
        axs[0].plot(time, force, color="grey", label="Force to track", lw=0.5)
        axs[0].set_ylabel("Force (N)", fontsize=14, fontweight="bold", color="black")

        bar_width = final_time / stim
        for i in range(sol_control["last_pulse_width"].shape[1]):
            axs[1].bar(final_time * (i/stim), sol_control["last_pulse_width"][0][i] * 10e5, width=bar_width, align='edge', color='blue', alpha=0.3, edgecolor='black')

        axs[1].set_ylabel("Pulse width (µs)", fontsize=14, fontweight="bold", color="black")
        axs[1].set_xlabel("Time (s)", fontsize=14, fontweight="bold", color="black")
        axs[1].set_ylim(model.pd0 * 10e5 - 20, 300)

        for ax in axs:
            ax.label_outer()

        plt.legend()
        plt.show()

        # --- Show the optimization results --- #
        sol.graphs()


if __name__ == "__main__":
    main()
