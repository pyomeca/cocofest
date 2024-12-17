"""
This example will do an optimal control program of a 100 steps tracking a hand cycling motion with a torque driven and
a torque resistance at the handle.
"""

import numpy as np

from bioptim import (
    Axis,
    ConstraintFcn,
    DynamicsList,
    ExternalForceSetTimeSeries,
    InterpolationType,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    PhaseDynamics,
    Solver,
    MultiCyclicNonlinearModelPredictiveControl,
    Solution,
    SolutionMerge,
    ControlType,
)

from cocofest import get_circle_coord, OcpFesMsk, FesMskModel, DingModelPulseWidthFrequencyWithFatigue, OcpFes


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
        model: FesMskModel,
        stim_time: list,
        pulse_width: dict,
        cycle_duration: int | float,
        n_cycles_to_advance: int,
        n_cycles_simultaneous: int,
        total_n_cycles: int,
        objective: dict,
):
    # cycle_len = OcpFes.prepare_n_shooting(stim_time, cycle_duration)
    cycle_len = 20
    total_n_shooting = cycle_len * n_cycles_simultaneous

    # --- EXTERNAL FORCES --- #
    total_external_forces_frame = total_n_cycles * cycle_len if total_n_cycles >= n_cycles_simultaneous else total_n_shooting
    external_force_set = ExternalForceSetTimeSeries(nb_frames=total_external_forces_frame)
    external_force_array = np.array([0, 0, -1])
    reshape_values_array = np.tile(external_force_array[:, np.newaxis], (1, total_external_forces_frame))
    external_force_set.add_torque(segment="wheel", values=reshape_values_array)

    # --- OBJECTIVE FUNCTION --- #
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

    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, quadratic=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTACT_FORCES, weight=0.0001, quadratic=True)

    # --- DYNAMICS --- #
    numerical_time_series = {"external_forces": external_force_set.to_numerical_time_series()}
    dynamics = DynamicsList()
    dynamics.add(
        model.declare_model_variables,
        dynamic_function=model.muscle_dynamic,
        expand_dynamics=True,
        expand_continuity=False,
        phase=0,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        numerical_data_timeseries=numerical_time_series,
        with_contact=True,
    )

    # --- BOUNDS AND INITIAL GUESS --- #
    # Path constraint: x_bounds, x_init
    x_bounds, x_init = OcpFesMsk._set_bounds_fes(model)
    x_bounds, x_init = OcpFesMsk._set_bounds_msk(x_bounds, x_init, model, msk_info={"bound_type": None})

    q_guess, qdot_guess = OcpFesMsk._prepare_initial_guess_cycling(model.biorbd_path,
                                                                   cycle_len,
                                                                   x_center,
                                                                   y_center,
                                                                   radius,
                                                                   n_cycles_simultaneous)

    x_initial_guess = {"q_guess": q_guess, "qdot_guess": qdot_guess}
    x_bounds, x_init = OcpFesMsk._set_bounds_msk_for_cycling(x_bounds, x_init, model, x_initial_guess,
                                                             n_cycles_simultaneous)

    # Define control path constraint: u_bounds, u_init
    u_bounds, u_init = OcpFesMsk._set_u_bounds_fes(model)
    u_bounds, u_init = OcpFesMsk._set_u_bounds_msk(u_bounds, u_init, model, with_residual_torque=True)
    u_bounds.add(key="tau", min_bound=np.array([-50, -50, -0]), max_bound=np.array([50, 50, 0]), phase=0,
                 interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

    # --- CONSTRAINTS --- #
    constraints = OcpFesMsk._build_constraints(
        model,
        cycle_len,
        cycle_duration,
        stim_time,
        ControlType.CONSTANT,
        custom_constraint=None,
        external_forces=True,
        simultaneous_cycle=n_cycles_simultaneous,
    )

    # constraints.add(
    #     ConstraintFcn.TRACK_MARKERS_VELOCITY,
    #     node=Node.START,
    #     marker_index=model.marker_index("wheel_center"),
    #     axes=[Axis.X, Axis.Y],
    # )

    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        first_marker="wheel_center",
        second_marker="global_wheel_center",
        node=Node.ALL,
        axes=[Axis.X, Axis.Y],
    )

    # --- PARAMETERS --- #
    (parameters,
     parameters_bounds,
     parameters_init,
     parameter_objectives,
     ) = OcpFesMsk._build_parameters(
        model=model,
        stim_time=stim_time,
        pulse_event=None,
        pulse_width=pulse_width,
        pulse_intensity=None,
        use_sx=True,
    )

    # rebuilding model for the OCP
    model = FesMskModel(
        name=model.name,
        biorbd_path=model.biorbd_path,
        muscles_model=model.muscles_dynamics_model,
        activate_force_length_relationship=model.activate_force_length_relationship,
        activate_force_velocity_relationship=model.activate_force_velocity_relationship,
        activate_residual_torque=model.activate_residual_torque,
        parameters=parameters,
        external_force_set=external_force_set,
        for_cycling=True,
    )

    return MyCyclicNMPC(
        [model],
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
        parameters=parameters,
        parameter_bounds=parameters_bounds,
        parameter_init=parameters_init,
        parameter_objectives=parameter_objectives,
        ode_solver=OdeSolver.RK4(),
        control_type=ControlType.CONSTANT,
        n_threads=8,
        use_sx=True,
    )


def main():
    cycle_duration = 1
    n_cycles_to_advance = 1
    n_cycles_simultaneous = 2
    n_cycles = 2

    model = FesMskModel(
        name=None,
        biorbd_path="../../msk_models/simplified_UL_Seth_pedal_aligned_one_muscle.bioMod",
        muscles_model=[
            DingModelPulseWidthFrequencyWithFatigue(muscle_name="DeltoideusClavicle_A", is_approximated=False),
            # DingModelPulseWidthFrequencyWithFatigue(muscle_name="DeltoideusScapula_P", is_approximated=False),
            # DingModelPulseWidthFrequencyWithFatigue(muscle_name="TRIlong", is_approximated=False),
            # DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIC_long", is_approximated=False),
            # DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIC_brevis", is_approximated=False),
        ],
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_residual_torque=True,
        external_force_set=None,  # External forces will be added
    )

    minimum_pulse_width = DingModelPulseWidthFrequencyWithFatigue().pd0

    nmpc = prepare_nmpc(
        model=model,
        stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        # stim_time=[0],
        pulse_width={
            "min": minimum_pulse_width,
            "max": 0.0006,
            "bimapping": False,
            "same_for_all_muscles": False,
            "fixed": False,
        },
        cycle_duration=cycle_duration,
        n_cycles_to_advance=n_cycles_to_advance,
        n_cycles_simultaneous=n_cycles_simultaneous,
        total_n_cycles=n_cycles,
        objective={"cycling": {"x_center": 0.35, "y_center": 0, "radius": 0.1},
                   "minimize_residual_torque": True},
    )

    def update_functions(_nmpc: MultiCyclicNonlinearModelPredictiveControl, cycle_idx: int, _sol: Solution):
        return cycle_idx < n_cycles  # True if there are still some cycle to perform

    # Solve the program
    sol = nmpc.solve(
        update_functions,
        solver=Solver.IPOPT(show_online_optim=False, _max_iter=1000, show_options=dict(show_bounds=True)),
        n_cycles_simultaneous=n_cycles_simultaneous,
        # get_all_iterations=True,
        cyclic_options={"states": {}},
    )

    sol.print_cost()
    sol.graphs(show_bounds=True)
    sol.animate(n_frames=200, show_tracked_markers=True)


if __name__ == "__main__":
    main()
