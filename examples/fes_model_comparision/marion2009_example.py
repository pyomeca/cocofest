"""
This example will do a 10 stimulation example with Ding's 2007 pulse width and frequency model.
The stimulation will be optimized the pulsation width between 0 and 0.0006 seconds to match a force value of 200N at the
end of the last node while minimizing the muscle force state.
"""

import numpy as np
from bioptim import Solver, ObjectiveList, ObjectiveFcn, OptimalControlProgram, ControlType, Node, BiorbdModel
from cocofest import OcpFes, ModelMaker, FES_plot


def prepare_ocp(model, final_time, pw_max):
    # --- Set dynamics --- #
    n_shooting = model.get_n_shooting(final_time=final_time)
    numerical_data_time_series, stim_idx_at_node_list = model.get_numerical_data_time_series(n_shooting, final_time)
    dynamics = OcpFes.declare_dynamics(model, numerical_data_time_series)

    # --- Set initial guesses and bounds for states and controls --- #
    x_bounds, x_init = OcpFes.set_x_bounds(model)
    u_bounds, u_init = OcpFes.set_u_bounds(model, max_bound=pw_max)

    # --- Set objective functions --- #
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="F", weight=1, quadratic=True)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="F", node=Node.END, target=200, weight=1e5, quadratic=True
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
        use_sx=True,
        n_threads=20,
    )


def main(with_fatigue=True, plot=True):
    final_time = 0.2
    model = ModelMaker.create_model(
        "marion2009_with_fatigue" if with_fatigue else "marion2009",
        sum_stim_truncation=10,
        stim_time=list(np.linspace(0, final_time, 11)[:-1]),
        previous_stim={"time": [-0.15, -0.10, -0.05]},
    )
    biorbd_path = "../model_msk/arm26_biceps_1dof.bioMod"
    model.bio_model = BiorbdModel(biorbd_path, parameters=None, external_force_set=None)
    ocp = prepare_ocp(model=model, final_time=final_time, pw_max=0.0006)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT())

    # --- Show results --- #
    if plot:
        FES_plot(data=sol).plot(title="Optimize pulse width", show_bounds=False, show_stim=False)


if __name__ == "__main__":
    main()
