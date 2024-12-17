"""
This example will do an optimal control program of a 100 steps tracking a hand cycling motion with a torque driven and
a torque resistance at the handle.
"""

import platform
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
    MultiCyclicNonlinearModelPredictiveControl,
    Dynamics,
    Objective,
    Solution,
    SolutionMerge,
    MultiCyclicCycleSolutions,
    ControlType,
)

from cocofest import (
    get_circle_coord,
    inverse_kinematics_cycling,
    inverse_dynamics_cycling,
)


class MyCyclicNMPC(MultiCyclicNonlinearModelPredictiveControl):
    def advance_window_bounds_states(self, sol, n_cycles_simultaneous=None):
        # Reimplementation of the advance_window method so the rotation of the wheel restart at -pi
        super(MyCyclicNMPC, self).advance_window_bounds_states(sol)  # Allow the wheel to spin as much as needed
        self.nlp[0].x_bounds["q"].min[-1, :] = self.nlp[0].model.bounds_from_ranges("q").min[-1] * n_cycles_simultaneous
        self.nlp[0].x_bounds["q"].max[-1, :] = self.nlp[0].model.bounds_from_ranges("q").max[-1] * n_cycles_simultaneous
        return True

    def advance_window_initial_guess_states(self, sol, n_cycles_simultaneous=None):
        # Reimplementation of the advance_window method so the rotation of the wheel restart at -pi
        super(MyCyclicNMPC, self).advance_window_initial_guess_states(sol)
        q = sol.decision_states(to_merge=SolutionMerge.NODES)["q"]
        self.nlp[0].x_init["q"].init[-1, :] = q[-1, :]  # Keep the previously found value for the wheel
        return True


def prepare_nmpc(
    biorbd_model_path: str,
    cycle_len: int,
    cycle_duration: int | float,
    n_cycles_to_advance: int,
    n_cycles_simultaneous: int,
    total_n_cycles: int,
    objective: dict,
    initial_guess_warm_start: bool = False,
    dynamics_torque_driven: bool = True,
    with_residual_torque: bool = False,

):
    if with_residual_torque and dynamics_torque_driven:
        raise ValueError("Residual torque is only available for muscle driven dynamics")

    total_n_shooting = cycle_len * n_cycles_simultaneous

    # External forces
    total_external_forces_frame = total_n_cycles * cycle_len if total_n_cycles >= n_cycles_simultaneous else total_n_shooting
    external_force_set = ExternalForceSetTimeSeries(nb_frames=total_external_forces_frame)
    external_force_array = np.array([0, 0, -1])
    reshape_values_array = np.tile(external_force_array[:, np.newaxis], (1, total_external_forces_frame))
    external_force_set.add_torque(segment="wheel", values=reshape_values_array)

    # Dynamics
    numerical_time_series = {"external_forces": external_force_set.to_numerical_time_series()}
    bio_model = BiorbdModel(biorbd_model_path, external_force_set=external_force_set)
    bio_model.current_n_cycles = 1

    # Adding an objective function to track a marker in a circular trajectory
    x_center = objective["cycling"]["x_center"]
    y_center = objective["cycling"]["y_center"]
    radius = objective["cycling"]["radius"]

    from scipy.interpolate import interp1d
    f = interp1d(np.linspace(0, -360 * n_cycles_simultaneous, 360 * n_cycles_simultaneous + 1),
                 np.linspace(0, -360 * n_cycles_simultaneous, 360 * n_cycles_simultaneous + 1), kind="linear")
    x_new = f(np.linspace(0, -360 * n_cycles_simultaneous, total_n_shooting + 1))
    x_new_rad = np.deg2rad(x_new)

    circle_coord_list = np.array(
        [
            get_circle_coord(theta, x_center, y_center, radius)[:-1]
            for theta in x_new_rad
        ]
    ).T

    objective_functions = ObjectiveList()
    for i in [5, 10, 15, 20]:
        objective_functions.add(
            ObjectiveFcn.Mayer.TRACK_MARKERS,
            weight=100000,
            axes=[Axis.X, Axis.Y],
            marker_index=0,
            target=circle_coord_list[:, i],
            node=i,
            phase=0,
            quadratic=True,
        )

    control_key = "tau" if dynamics_torque_driven else "muscles"
    weight = 100 if dynamics_torque_driven else 1
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key=control_key, weight=weight, quadratic=True)
    if with_residual_torque:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, quadratic=True)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTACT_FORCES, weight=0.0001, quadratic=True)

    # Dynamics
    chosen_dynamics = DynamicsFcn.TORQUE_DRIVEN if dynamics_torque_driven else DynamicsFcn.MUSCLE_DRIVEN
    dynamics = DynamicsList()
    dynamics.add(
        chosen_dynamics,
        expand_dynamics=True,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        numerical_data_timeseries=numerical_time_series,
        with_contact=True,
        # with_residual_torque=with_residual_torque,
    )

    # Path constraint
    x_bounds = BoundsList()
    q_x_bounds = bio_model.bounds_from_ranges("q")
    qdot_x_bounds = bio_model.bounds_from_ranges("qdot")
    x_bounds.add(key="q", bounds=q_x_bounds, phase=0)
    x_bounds["q"].min[-1, :] = x_bounds["q"].min[-1, :] * n_cycles_simultaneous  # Allow the wheel to spin as much as needed
    x_bounds["q"].max[-1, :] = x_bounds["q"].max[-1, :] * n_cycles_simultaneous

    # Modifying pedal speed bounds
    qdot_x_bounds.max[2] = [0, 0, 0]
    x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=0, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,)

    # Define control path constraint
    u_bounds = BoundsList()
    if dynamics_torque_driven:
        u_bounds.add(key="tau", min_bound=np.array([-50, -50, -0]), max_bound=np.array([50, 50, 0]), phase=0, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,)
    else:
        if with_residual_torque:
            u_bounds.add(key="tau", min_bound=np.array([-50, -50, -0]), max_bound=np.array([50, 50, 0]), phase=0, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,)
        muscle_min, muscle_max, muscle_init = 0.0, 1.0, 0.5
        u_bounds["muscles"] = [muscle_min] * bio_model.nb_muscles, [muscle_max] * bio_model.nb_muscles
        u_init = InitialGuessList()
        u_init["muscles"] = [muscle_init] * bio_model.nb_muscles

    # Initial q guess
    x_init = InitialGuessList()
    u_init = InitialGuessList()
    # # If warm start, the initial guess is the result of the inverse kinematics and dynamics
    if initial_guess_warm_start:
        biorbd_model_path = "../../msk_models/simplified_UL_Seth_pedal_aligned_test_one_marker.bioMod"
        q_guess, qdot_guess, qddotguess = inverse_kinematics_cycling(
            biorbd_model_path, total_n_shooting, x_center, y_center, radius, ik_method="trf", cycling_number=n_cycles_simultaneous
        )
        x_init.add("q", q_guess, interpolation=InterpolationType.EACH_FRAME)
        x_init.add("qdot", qdot_guess, interpolation=InterpolationType.EACH_FRAME)
        # if not dynamics_torque_driven and with_residual_torque:
        # u_guess = inverse_dynamics_cycling(biorbd_model_path, q_guess, qdot_guess, qddotguess)
        # u_init.add("tau", u_guess, interpolation=InterpolationType.EACH_FRAME)

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

    return MyCyclicNMPC(
            [bio_model],
            dynamics,
            cycle_len=cycle_len,
            cycle_duration=cycle_duration,
            n_cycles_simultaneous=n_cycles_simultaneous,
            n_cycles_to_advance=n_cycles_to_advance,
            common_objective_functions=objective_functions,
            constraints=constraints,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            x_init=x_init,
            u_init=u_init,
            ode_solver=OdeSolver.RK4(),
            n_threads=8,
        )


def main():
    cycle_duration = 1
    cycle_len = 20
    n_cycles_to_advance = 1
    n_cycles_simultaneous = 2
    n_cycles = 2

    nmpc = prepare_nmpc(
        biorbd_model_path="../../msk_models/simplified_UL_Seth_pedal_aligned_test.bioMod",
        cycle_len=cycle_len,
        cycle_duration=cycle_duration,
        n_cycles_to_advance=n_cycles_to_advance,
        n_cycles_simultaneous=n_cycles_simultaneous,
        total_n_cycles=n_cycles,
        objective={"cycling": {"x_center": 0.35, "y_center": 0, "radius": 0.1}},
        initial_guess_warm_start=True,
        dynamics_torque_driven=True,
        with_residual_torque=False,
    )

    def update_functions(_nmpc: MultiCyclicNonlinearModelPredictiveControl, cycle_idx: int, _sol: Solution):
        return cycle_idx < n_cycles  # True if there are still some cycle to perform

    # Solve the program
    sol = nmpc.solve(
        update_functions,
        solver=Solver.IPOPT(show_online_optim=False, _max_iter=1000, show_options=dict(show_bounds=True)),
        n_cycles_simultaneous=n_cycles_simultaneous,
        cyclic_options={"states": {}},
        cycle_solutions=MultiCyclicCycleSolutions.ALL_CYCLES,
        get_all_iterations=True,
    )

    # sol.print_cost()
    # sol.graphs(show_bounds=True)
    sol[1][0].graphs(show_bounds=True)
    sol[1][1].graphs(show_bounds=True)

    sol[0].graphs(show_bounds=True)
    sol[0].animate(n_frames=100)
    # sol.animate(n_frames=200, show_tracked_markers=True)


if __name__ == "__main__":
    main()
