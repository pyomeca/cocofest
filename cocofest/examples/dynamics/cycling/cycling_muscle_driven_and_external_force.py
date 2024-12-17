"""
This example will do an optimal control program of a 100 steps tracking a hand cycling motion with a torque driven and
a torque resistance at the handle.
"""

import numpy as np

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
    circle_coord_list = np.array(
        [get_circle_coord(theta, x_center, y_center, radius)[:-1] for theta in np.linspace(0, -2 * np.pi, n_shooting)]
    ).T
    objective_functions = ObjectiveList()

    objective_functions.add(
        ObjectiveFcn.Mayer.TRACK_MARKERS,
        weight=100000,
        axes=[Axis.X, Axis.Y],
        marker_index=0,
        target=circle_coord_list,
        node=Node.ALL_SHOOTING,
        phase=0,
        quadratic=True,
    )

    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=100, quadratic=True)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        DynamicsFcn.MUSCLE_DRIVEN,
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

    # Modifying pedal speed bounds
    qdot_x_bounds.max[2] = [0, 0, 0]
    x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=0)

    # Define control path constraint
    muscle_min, muscle_max, muscle_init = 0.0, 1.0, 0.5
    u_bounds = BoundsList()
    u_bounds["muscles"] = [muscle_min] * bio_model.nb_muscles, [muscle_max] * bio_model.nb_muscles
    u_init = InitialGuessList()
    u_init["muscles"] = [muscle_init] * bio_model.nb_muscles

    # Initial q guess
    x_init = InitialGuessList()
    # # If warm start, the initial guess is the result of the inverse kinematics and dynamics
    if initial_guess_warm_start:
        biorbd_model_path = "../../msk_models/simplified_UL_Seth_pedal_aligned_test_one_marker.bioMod"
        q_guess, qdot_guess, qddotguess = inverse_kinematics_cycling(
            biorbd_model_path, n_shooting, x_center, y_center, radius, ik_method="trf"
        )
        x_init.add("q", q_guess, interpolation=InterpolationType.EACH_FRAME)
        x_init.add("qdot", qdot_guess, interpolation=InterpolationType.EACH_FRAME)

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        node=Node.START,
        marker_index=bio_model.marker_index("wheel_center"),
        axes=[Axis.X, Axis.Y],
    )
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        first_marker="wheel_center",
        second_marker="global_wheel_center",
        node=Node.START,
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
        # phase_transitions=phase_transitions,
    )


def main():
    # --- Prepare the ocp --- #
    ocp = prepare_ocp(
        biorbd_model_path="../../msk_models/simplified_UL_Seth_pedal_aligned_test.bioMod",
        n_shooting=100,
        final_time=1,
        objective={"cycling": {"x_center": 0.35, "y_center": 0, "radius": 0.1}},
        initial_guess_warm_start=True,
    )
    ocp.add_plot_penalty(CostType.ALL)
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=10000))#, show_options=dict(show_bounds=True)))
    sol.animate(viewer="pyorerun")
    # sol.animate()
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
