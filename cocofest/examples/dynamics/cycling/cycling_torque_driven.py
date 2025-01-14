"""
This example will do an optimal control program of a 100 steps tracking a hand cycling motion with a torque driven
control.
"""

import numpy as np

from bioptim import (
    Axis,
    BiorbdModel,
    BoundsList,
    CostType,
    ConstraintList,
    ConstraintFcn,
    DynamicsFcn,
    DynamicsList,
    InitialGuessList,
    InterpolationType,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    PhaseDynamics,
)

from cocofest import (
    get_circle_coord,
    inverse_kinematics_cycling,
    inverse_dynamics_cycling,
)


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    final_time: int,
    objective: dict,
    initial_guess_warm_start: bool = False,
) -> OptimalControlProgram:

    # Adding the model
    bio_model = BiorbdModel(
        biorbd_model_path,
    )

    # Adding an objective function to track a marker in a circular trajectory
    x_center = objective["cycling"]["x_center"]
    y_center = objective["cycling"]["y_center"]
    radius = objective["cycling"]["radius"]
    # circle_coord_list = np.array(
    #     [get_circle_coord(theta, x_center, y_center, radius)[:-1] for theta in np.linspace(0, -2 * np.pi, n_shooting)]
    # ).T

    # Objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=0, node=Node.ALL_SHOOTING)

    # objective_functions.add(
    #     ObjectiveFcn.Mayer.TRACK_MARKERS,
    #     weight=100,
    #     axes=[Axis.X, Axis.Y],
    #     marker_index=0,
    #     target=circle_coord_list,
    #     node=Node.ALL_SHOOTING,
    #     phase=0,
    #     quadratic=True,
    # )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        DynamicsFcn.TORQUE_DRIVEN,
        expand_dynamics=True,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
    )

    # Path constraint
    x_bounds = BoundsList()
    q_x_bounds = bio_model.bounds_from_ranges("q")
    qdot_x_bounds = bio_model.bounds_from_ranges("qdot")
    x_bounds.add(key="q", bounds=q_x_bounds, phase=0)
    x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=0)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(key="tau", min_bound=np.array([-50, -50, 0]), max_bound=np.array([50, 50, 0]), phase=0)

    # Initial q guess
    x_init = InitialGuessList()
    u_init = InitialGuessList()

    # The initial guess is the result of the inverse kinematics and dynamics
    biorbd_model_path = "../../msk_models/simplified_UL_Seth_pedal_aligned_test_one_marker.bioMod"
    q_guess, qdot_guess, qddotguess = inverse_kinematics_cycling(
        biorbd_model_path, n_shooting, x_center, y_center, radius, ik_method="trf"
    )
    x_init.add("q", q_guess, interpolation=InterpolationType.EACH_FRAME)
    x_init.add("qdot", qdot_guess, interpolation=InterpolationType.EACH_FRAME)
    u_guess = inverse_dynamics_cycling(biorbd_model_path, q_guess, qdot_guess, qddotguess)
    u_init.add("tau", u_guess, interpolation=InterpolationType.EACH_FRAME)

    # Constraints
    constraints = ConstraintList()
    cardinal_node_list = [i * int(n_shooting / 4) for i in range(4 + 1)]
    for i in cardinal_node_list:
        min_bound = x_init["q"].init[2][i]-x_init["q"].init[2][i]*0.01
        max_bound = x_init["q"].init[2][i]+x_init["q"].init[2][i]*0.01
        if min_bound > max_bound:
            max_bound, min_bound = min_bound, max_bound
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE,
                        key="q",
                        index=2,
                        target=x_init["q"].init[2][i],
                        # min_bound=[min_bound],
                        # max_bound=[max_bound],
                        node=i,
                        weight=100000,
                        )

    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS,
                    node=Node.START,
                    first_marker="wheel_center",
                    second_marker="global_wheel_center",
                    axes=[Axis.X, Axis.Y],)

    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        node=Node.ALL_SHOOTING,
        marker_index=bio_model.marker_index("wheel_center"),
        axes=[Axis.X, Axis.Y],
    )

    return OptimalControlProgram(
        [bio_model],
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=OdeSolver.RK4(),
        n_threads=8,
    )


def main():
    # --- Prepare the ocp --- #
    ocp = prepare_ocp(
        biorbd_model_path="../../msk_models/simplified_UL_Seth_pedal_aligned.bioMod",
        n_shooting=100,
        final_time=1,
        objective={"cycling": {"x_center": 0.35, "y_center": 0, "radius": 0.1}},
        initial_guess_warm_start=True,
    )
    ocp.add_plot_penalty(CostType.ALL)
    sol = ocp.solve()
    sol.animate(viewer="pyorerun")
    sol.graphs()


if __name__ == "__main__":
    main()
