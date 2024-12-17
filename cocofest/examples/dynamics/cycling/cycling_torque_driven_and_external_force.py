"""
This example will do an optimal control program of a 100 steps tracking a hand cycling motion with a torque driven and
a torque resistance at the handle.
"""

import numpy as np
from scipy.interpolate import interp1d

from bioptim import (
    Axis,
    BiorbdModel,
    BoundsList,
    ConstraintList,
    ConstraintFcn,
    CostType,
    DynamicsFcn,
    DynamicsList,
    ExternalForceSetTimeSeries,
    InitialGuessList,
    InterpolationType,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    PhaseDynamics,
    Solver,
    PhaseTransitionList,
    PhaseTransitionFcn,
    ControlType,
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
    turn_number: int,
    objective: dict,
    initial_guess_warm_start: bool = False,
) -> OptimalControlProgram:

    # External forces
    external_force_set = ExternalForceSetTimeSeries(nb_frames=n_shooting)
    external_force_array = np.array([0, 0, -1])
    reshape_values_array = np.tile(external_force_array[:, np.newaxis], (1, n_shooting))
    external_force_set.add_torque(segment="wheel", values=reshape_values_array)

    # Phase transitions not available with numerical time series
    # phase_transitions = PhaseTransitionList()  # TODO : transition phase cyclic
    # phase_transitions.add(PhaseTransitionFcn.CYCLIC)

    # Dynamics
    numerical_time_series = {"external_forces": external_force_set.to_numerical_time_series()}
    bio_model = BiorbdModel(biorbd_model_path, external_force_set=external_force_set)

    # Adding an objective function to track a marker in a circular trajectory
    x_center = objective["cycling"]["x_center"]
    y_center = objective["cycling"]["y_center"]
    radius = objective["cycling"]["radius"]

    f = interp1d(np.linspace(0, -360*turn_number, 360*turn_number+1), np.linspace(0, -360*turn_number, 360*turn_number+1), kind="linear")
    x_new = f(np.linspace(0, -360*turn_number, n_shooting+1))
    x_new_rad = np.deg2rad(x_new)

    circle_coord_list = np.array(
        [
            get_circle_coord(theta, x_center, y_center, radius)[:-1]
            for theta in x_new_rad
        ]
    ).T

    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Mayer.TRACK_MARKERS,
        weight=100000,
        axes=[Axis.X, Axis.Y],
        marker_index=0,
        target=circle_coord_list,
        node=Node.ALL,
        phase=0,
        quadratic=True,
    )

    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, quadratic=True)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        DynamicsFcn.TORQUE_DRIVEN,
        expand_dynamics=True,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        numerical_data_timeseries=numerical_time_series,
        with_contact=True,
    )

    # Path constraint
    x_bounds = BoundsList()
    q_x_bounds = bio_model.bounds_from_ranges("q")
    qdot_x_bounds = bio_model.bounds_from_ranges("qdot")
    x_bounds.add(key="q", bounds=q_x_bounds, phase=0)
    x_bounds["q"].min[-1, :] = x_bounds["q"].min[-1, :] * turn_number  # Allow the wheel to spin as much as needed
    x_bounds["q"].max[-1, :] = x_bounds["q"].max[-1, :] * turn_number

    # Modifying pedal speed bounds
    qdot_x_bounds.max[2] = [0, 0, 0]
    x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=0)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(key="tau", min_bound=np.array([-50, -50, -0]), max_bound=np.array([50, 50, 0]), phase=0)

    # Initial q guess
    x_init = InitialGuessList()
    u_init = InitialGuessList()
    # # If warm start, the initial guess is the result of the inverse kinematics and dynamics
    if initial_guess_warm_start:
        biorbd_model_path = "../../msk_models/simplified_UL_Seth_pedal_aligned_test_one_marker.bioMod"
        q_guess, qdot_guess, qddotguess = inverse_kinematics_cycling(
            biorbd_model_path, n_shooting, x_center, y_center, radius, ik_method="lm", cycling_number=turn_number
        )
        x_init.add("q", q_guess, interpolation=InterpolationType.EACH_FRAME)
        x_init.add("qdot", qdot_guess, interpolation=InterpolationType.EACH_FRAME)
        # u_guess = inverse_dynamics_cycling(biorbd_model_path, q_guess, qdot_guess, qddotguess)
        # u_init.add("tau", u_guess, interpolation=InterpolationType.EACH_FRAME)

    constraints = ConstraintList()
    # constraints.add(
    #     ConstraintFcn.TRACK_MARKERS_VELOCITY,
    #     node=Node.START,
    #     marker_index=bio_model.marker_index("wheel_center"),
    #     axes=[Axis.X, Axis.Y],
    # )
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        first_marker="wheel_center",
        second_marker="global_wheel_center",
        node=Node.ALL,
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
        ode_solver=OdeSolver.RK4(),
        n_threads=8,
        constraints=constraints,
    )


def main():
    # --- Prepare the ocp --- #
    ocp = prepare_ocp(
        biorbd_model_path="../../msk_models/simplified_UL_Seth_pedal_aligned_test.bioMod",
        n_shooting=100,
        final_time=5,
        turn_number=5,
        objective={"cycling": {"x_center": 0.35, "y_center": 0, "radius": 0.1}},
        initial_guess_warm_start=True,
    )
    ocp.add_plot_penalty(CostType.ALL)
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=10000)) #, show_options=dict(show_bounds=True)))#, show_options=dict(show_bounds=True)))
    sol.animate(viewer="pyorerun", show_tracked_markers=True)
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
