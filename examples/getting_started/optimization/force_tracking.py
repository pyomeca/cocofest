"""
This example will do a 10 stimulation example with Ding's 2007 model.
This ocp was build to match a force curve across all optimization.
"""

import matplotlib.pyplot as plt
import numpy as np

from bioptim import SolutionMerge, ObjectiveList, ObjectiveFcn, OptimalControlProgram, ControlType, OdeSolver, Node
from cocofest import ModelMaker, OcpFes, FesModel


def prepare_ocp(model: FesModel, final_time: float, pw_max: float, force_tracking: list) -> OptimalControlProgram:
    """
    Prepare the Optimal Control Program by setting dynamics, bounds and cost functions.

    Parameters
    ----------
    model : DingModelPulseWidthFrequency
        The chosen FES model to use as muscle dynamics.
    final_time : float
        The ending time for the simulation.
    pw_max : float
        The maximum pulse width, used for stimulation bounds.
    force_tracking : list
        The force to track.

    Returns
    -------
    ocp : OptimalControlProgram
        The Optimal Control Program to solve.
    """
    # --- Set dynamics --- #
    n_shooting = model.get_n_shooting(final_time=final_time)  # Create the number of shooting points for the OCP
    time_series, stim_idx_at_node_list = model.get_numerical_data_time_series(
        n_shooting, final_time
    )  # Retrieve time and indexes at which occurs the stimulation for the FES dynamic
    dynamics = OcpFes.declare_dynamics(
        model,
        time_series,
        ode_solver=OdeSolver.RK4(n_integration_steps=10),
        # ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3, method="radau"),  # Possibility to use a different solver
    )

    # --- Set initial guesses and bounds for states and controls --- #
    x_bounds = OcpFes.set_x_bounds(model)
    x_init = OcpFes.set_x_init(model)
    u_bounds = OcpFes.set_u_bounds(model, max_bound=pw_max)
    u_init = OcpFes.set_u_init(model)

    # --- Set objective functions --- #
    objective_functions = ObjectiveList()
    force_to_track = force_tracking[np.newaxis, :]  # Reshape list to track to match Bioptim's target size
    objective_functions.add(
        ObjectiveFcn.Mayer.TRACK_STATE,
        key="F",
        target=force_to_track,
        node=Node.ALL,
        quadratic=True,
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
        control_type=ControlType.CONSTANT,
        n_threads=20,
    )


def main(plot=True):
    final_time = 1
    stim = 33
    model = ModelMaker.create_model("ding2007", stim_time=list(np.linspace(0, 1, stim, endpoint=False)))

    # --- Building force to track ---#
    time = np.linspace(0, 1, 34)
    force = 10 + (150 - 10) * np.abs(np.sin(time * 5))  # Example of force to track between 10 and 150 N
    force[0] = 0.0  # Ensuring the force starts at 0 N

    ocp = prepare_ocp(model=model, final_time=final_time, pw_max=0.0006, force_tracking=force)
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
        axs[0].scatter(time, force, color="grey", label="Force to track", lw=0.5)
        axs[0].set_ylabel("Force (N)", fontsize=14, fontweight="bold", color="black")

        bar_width = final_time / stim
        for i in range(sol_control["last_pulse_width"].shape[1]):
            axs[1].bar(
                final_time * (i / stim),
                sol_control["last_pulse_width"][0][i] * 10e5,
                width=bar_width,
                align="edge",
                color="blue",
                alpha=0.3,
                edgecolor="black",
            )

        axs[1].set_ylabel("Pulse width (µs)", fontsize=14, fontweight="bold", color="black")
        axs[1].set_xlabel("Time (s)", fontsize=14, fontweight="bold", color="black")
        axs[1].set_ylim(model.pd0 * 10e5 - 20, 400)

        for ax in axs:
            ax.label_outer()

        plt.show()

        # --- Show the optimization results --- #
        sol.graphs()


if __name__ == "__main__":
    main()
