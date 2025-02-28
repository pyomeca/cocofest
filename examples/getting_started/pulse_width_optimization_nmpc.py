"""
This example showcases a moving time horizon simulation problem of cyclic muscle force tracking at last node.
The FES model used here is Ding's 2007 pulse width and frequency model with fatigue.
Only the pulse width is optimized to minimize muscle force production, frequency is fixed at 33 Hz.
The nmpc cyclic problem stops once the last cycle is reached.
"""

import numpy as np

from bioptim import (
    CostType,
    ConstraintFcn,
    MultiCyclicCycleSolutions,
    MultiCyclicNonlinearModelPredictiveControl,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    SolutionMerge,
    Solution,
    Solver,
)

from cocofest import (
    DingModelPulseWidthFrequencyWithFatigue,
    OcpFes,
    FesModel,
    FES_plot,
    FesNmpc,
)


def prepare_nmpc(
    model: FesModel,
    cycle_duration: int | float,
    n_cycles_to_advance: int,
    n_cycles_simultaneous: int,
    max_pulse_width: float,
    use_sx: bool = False,
    minimize_force: bool = True,
    minimize_fatigue: bool = False,
):
    total_cycle_len = model.get_n_shooting(cycle_duration * n_cycles_simultaneous)
    total_cycle_duration = cycle_duration * n_cycles_simultaneous
    cycle_len = int(total_cycle_len / n_cycles_simultaneous)

    numerical_data_time_series, stim_idx_at_node_list = model.get_numerical_data_time_series(
        total_cycle_len, total_cycle_duration
    )

    dynamics = OcpFes.declare_dynamics(model, numerical_data_time_series)

    x_bounds, x_init = OcpFes.set_x_bounds(model)
    u_bounds, u_init = OcpFes.set_u_bounds(model, max_pulse_width)

    constraints = OcpFes.set_constraints(
        model,
        total_cycle_len,
        stim_idx_at_node_list,
    )

    stim_interval = int(1 / (model.stim_time[1] - model.stim_time[0]))
    constraints_node_list = [stim_interval * (1 + 2 * i) for i in range(total_cycle_len // (2 * stim_interval))]
    for i in constraints_node_list:
        constraints.add(ConstraintFcn.TRACK_STATE, key="F", node=i, min_bound=200, max_bound=210)

    # Define objective functions
    objective_functions = ObjectiveList()

    if minimize_force:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="F", weight=1, quadratic=True)

    if minimize_fatigue:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="A", weight=-1, quadratic=True)

    return FesNmpc(
        bio_model=model,
        dynamics=dynamics,
        cycle_len=cycle_len,
        cycle_duration=cycle_duration,
        n_cycles_simultaneous=n_cycles_simultaneous,
        n_cycles_to_advance=n_cycles_to_advance,
        common_objective_functions=objective_functions,
        constraints=constraints,
        x_bounds=x_bounds,
        x_init=x_init,
        u_bounds=u_bounds,
        u_init=u_init,
        ode_solver=OdeSolver.RK4(n_integration_steps=10),
        n_threads=20,
        use_sx=use_sx,
    )


def main():
    """
    Main function to configure and solve the optimal control problem in NMPC.
    """
    # --- Set nmpc parameters --- #
    cycle_duration = 2  # Duration of a cycle in seconds
    n_cycles_simultaneous = 3  # Number of cycles to solve simultaneously
    n_cycles_to_advance = 1  # Number of cycles to advance at each iteration
    n_cycles = 5  # Number of total cycles to perform

    # --- Set stimulation time apparition --- #
    stimulation_frequency = 33  # Stimulation frequency in Hz
    final_time = cycle_duration * n_cycles_simultaneous
    stim_time = [
        val
        for start in range(0, final_time, 2)
        for val in np.linspace(start, start + 1, stimulation_frequency + 1)[:-1]
    ]

    # --- Build FES model --- #
    fes_model = DingModelPulseWidthFrequencyWithFatigue(stim_time=stim_time, sum_stim_truncation=10)

    nmpc = prepare_nmpc(
        model=fes_model,
        cycle_duration=cycle_duration,
        n_cycles_to_advance=n_cycles_to_advance,
        n_cycles_simultaneous=n_cycles_simultaneous,
        max_pulse_width=0.0006,
        minimize_force=True,
        minimize_fatigue=False,
        use_sx=False,
    )

    def update_functions(_nmpc: MultiCyclicNonlinearModelPredictiveControl, cycle_idx: int, _sol: Solution):
        return cycle_idx < n_cycles  # True if there are still some cycle to perform

    # Add the penalty cost function plot
    nmpc.add_plot_penalty(CostType.ALL)
    # Solve the optimal control problem
    sol = nmpc.solve_fes_nmpc(
        update_functions,
        solver=Solver.IPOPT(show_online_optim=False, _max_iter=1000, show_options=dict(show_bounds=True)),
        total_cycles=n_cycles,
        cycle_solutions=MultiCyclicCycleSolutions.ALL_CYCLES,
        get_all_iterations=True,
        cyclic_options={"states": {}},
        max_consecutive_failing=3,
    )

    result = sol[0].stepwise_states(to_merge=[SolutionMerge.NODES])
    time = sol[0].stepwise_time(to_merge=[SolutionMerge.NODES]).T[0]
    result["time"] = time
    result["pulse_width"] = sol[0].stepwise_controls(to_merge=[SolutionMerge.NODES])["last_pulse_width"]

    # Plotting the force state result
    FES_plot(data=result).plot(title="NMPC FES model optimization")


if __name__ == "__main__":
    main()
