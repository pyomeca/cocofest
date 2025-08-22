"""
Warning : The model output are not those intended, use this model cautiously.

This example will do a 10 stimulation example with Marion's 2013 frequency model (possibility to add pulse width).
The stimulation will be optimized the pulsation width between 0 and 0.0006 seconds to match a force value of 90N at the
end of the last node while minimizing the muscle force state.
"""

import numpy as np
from bioptim import (
    Solver,
    ObjectiveList,
    ObjectiveFcn,
    OptimalControlProgram,
    ControlType,
    Node,
    BoundsList,
    InitialGuessList,
    InterpolationType,
)
from cocofest import OcpFes, ModelMaker
from cocofest.models.marion2013.marion2013_modified import Marion2013ModelPulseWidthFrequency


def prepare_ocp(model, final_time, pw_max=0.0006):
    # --- Set dynamics --- #
    n_shooting = model.get_n_shooting(final_time=final_time)
    numerical_data_time_series, stim_idx_at_node_list = model.get_numerical_data_time_series(n_shooting, final_time)
    dynamics = OcpFes.declare_dynamics(model, numerical_data_time_series)

    # --- Set initial guesses and bounds for states and controls --- #
    x_bounds, x_init = OcpFes.set_x_bounds(model)
    x_bounds.add(
        key="theta",
        min_bound=np.array([[90, 0, 0]]),
        max_bound=np.array([[90, 90, 90]]),
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )
    x_bounds.add(
        key="dtheta_dt",
        min_bound=np.array([[0, -100, -100]]),
        max_bound=np.array([[0, 100, 100]]),
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )
    if isinstance(model, Marion2013ModelPulseWidthFrequency):
        u_bounds, u_init = OcpFes.set_u_bounds(model, max_bound=pw_max)
    else:
        u_bounds, u_init = BoundsList(), InitialGuessList()

    # --- Set objective functions --- #
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="F", weight=1, quadratic=True)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="F", node=Node.END, target=90, weight=1e5, quadratic=True
    )

    return OptimalControlProgram(
        bio_model=[model],
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=final_time,
        objective_functions=objective_functions,
        x_init=x_init,
        x_bounds=x_bounds,
        u_init=u_init,
        u_bounds=u_bounds,
        control_type=ControlType.CONSTANT,
        use_sx=True,
        n_threads=20,
    )


def main(with_pulse_width=True, with_fatigue=False, plot=True):
    final_time = 0.2
    chosen_model = "marion2013_modified" if with_pulse_width else "marion2013"
    chosen_model = chosen_model + "_with_fatigue" if with_fatigue else chosen_model
    model = ModelMaker.create_model(
        chosen_model,
        sum_stim_truncation=10,
        stim_time=list(np.linspace(0, final_time, 11)[:-1]),
        previous_stim={"time": [-0.15, -0.10, -0.05]},
    )
    ocp = prepare_ocp(model=model, final_time=final_time, pw_max=0.0006)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT())

    # --- Show results --- #
    if plot:
        sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
