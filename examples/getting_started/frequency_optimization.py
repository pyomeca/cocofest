"""
This example will do a 10 stimulation example with Ding's 2003 frequency model.
This ocp was build to match a force value of 270N at the end of the last node.
No optimization will be done on the stimulation, the frequency is fixed to 1Hz.
"""

from bioptim import ObjectiveList, ObjectiveFcn, OptimalControlProgram, ControlType, OdeSolver, Node
from cocofest import OcpFes, ModelMaker


def prepare_ocp(model, final_time):
    # --- Set dynamics --- #
    n_shooting = model.get_n_shooting(final_time=final_time)
    numerical_data_time_series, stim_idx_at_node_list = model.get_numerical_data_time_series(n_shooting, final_time)
    dynamics = OcpFes.declare_dynamics(model, numerical_data_time_series)

    # --- Set initial guesses and bounds for states --- #
    x_bounds, x_init = OcpFes.set_x_bounds(model)

    # --- Set objective functions --- #
    objective_functions = ObjectiveList()
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
        control_type=ControlType.CONSTANT,
        use_sx=True,
        ode_solver=OdeSolver.RK4(n_integration_steps=10),
        n_threads=20,
    )


def main():
    final_time = 1
    model = ModelMaker.create_model("ding2003", stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ocp = prepare_ocp(model, final_time)
    sol = ocp.solve()

    # --- Show results from solution --- #
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
