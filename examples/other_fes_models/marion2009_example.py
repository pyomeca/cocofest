"""
This example will do a 10 stimulation example with Marion's 2009 frequency model (possibility to add pulse width).
The stimulation will be optimized the pulsation width between 0 and 0.0006 seconds to match a force value of 200N at the
end of the last node while minimizing the muscle force state.
"""

import numpy as np
from bioptim import (
    Solver,
    ObjectiveList,
    ObjectiveFcn,
    OdeSolver,
    OptimalControlProgram,
    ControlType,
    Node,
    BoundsList,
    InitialGuessList,
)
from cocofest import OcpFes, ModelMaker, FES_plot
from cocofest.models.marion2009.marion2009_modified import Marion2009ModelPulseWidthFrequency


def prepare_ocp(model, final_time, pw_max=0.0006):
    # --- Set dynamics --- #
    n_shooting = model.get_n_shooting(final_time=final_time)
    numerical_data_time_series, stim_idx_at_node_list = model.get_numerical_data_time_series(n_shooting, final_time)
    dynamics_options = OcpFes.declare_dynamics_options(
        numerical_time_series=numerical_data_time_series, ode_solver=OdeSolver.RK4(n_integration_steps=10)
    )

    # --- Set initial guesses and bounds for states and controls --- #
    x_bounds = OcpFes.set_x_bounds(model)
    x_init = OcpFes.set_x_init(model)

    if isinstance(model, Marion2009ModelPulseWidthFrequency):
        u_bounds = OcpFes.set_u_bounds(model, max_bound=pw_max)
        u_init = OcpFes.set_u_init(model)
    else:
        u_bounds, u_init = BoundsList(), InitialGuessList()

    # --- Set objective functions --- #
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="F", weight=1, quadratic=True)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="F", node=Node.END, target=200, weight=1e5, quadratic=True
    )

    return OptimalControlProgram(
        bio_model=[model],
        dynamics=dynamics_options,
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
    chosen_model = "marion2009_modified" if with_pulse_width else "marion2009"
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
        FES_plot(data=sol).plot(title="Optimize pulse width", show_bounds=False, show_stim=False)


if __name__ == "__main__":
    main()
