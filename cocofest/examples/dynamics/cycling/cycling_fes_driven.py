"""
This example will do an optimal control program of a 10 stimulation example with Ding's 2007 pulse width model.
Those ocp were build to produce a cycling motion.
The stimulation frequency will be set to 10Hz and pulse width will be optimized to satisfy the motion meanwhile
reducing residual torque.
"""

# import numpy as np
#
# from bioptim import CostType, Solver
#
# import biorbd
#
# from cocofest import (
#     DingModelPulseWidthFrequencyWithFatigue,
#     OcpFesMsk,
#     PlotCyclingResult,
#     SolutionToPickle,
#     FesMskModel,
#     PickleAnimate,
# )
#
#
# def main():
#     minimum_pulse_width = DingModelPulseWidthFrequencyWithFatigue().pd0
#
#     model = FesMskModel(
#         name=None,
#         biorbd_path="../../msk_models/simplified_UL_Seth.bioMod",
#         muscles_model=[
#             DingModelPulseWidthFrequencyWithFatigue(muscle_name="DeltoideusClavicle_A"),
#             DingModelPulseWidthFrequencyWithFatigue(muscle_name="DeltoideusScapula_P"),
#             DingModelPulseWidthFrequencyWithFatigue(muscle_name="TRIlong"),
#             DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIC_long"),
#             DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIC_brevis"),
#         ],
#         activate_force_length_relationship=True,
#         activate_force_velocity_relationship=True,
#         activate_residual_torque=True,
#     )
#
#     ocp = OcpFesMsk.prepare_ocp(
#         model=model,
#         stim_time=list(np.round(np.linspace(0, 1, 11), 3))[:-1],
#         final_time=1,
#         pulse_width={
#             "min": minimum_pulse_width,
#             "max": 0.0006,
#             "bimapping": False,
#         },
#         msk_info={"with_residual_torque": True},
#         objective={
#             "cycling": {"x_center": 0.35, "y_center": 0, "radius": 0.1, "target": "marker"},
#             "minimize_residual_torque": True,
#         },
#         initial_guess_warm_start=False,
#         n_threads=5,
#     )
#     ocp.add_plot_penalty(CostType.ALL)
#     sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=10000))
#     SolutionToPickle(sol, "cycling_fes_driven_min_residual_torque.pkl", "").pickle()
#
#     biorbd_model = biorbd.Model("../../msk_models/simplified_UL_Seth_full_mesh.bioMod")
#     PickleAnimate("cycling_fes_driven_min_residual_torque.pkl").animate(model=biorbd_model)
#
#     sol.graphs(show_bounds=False)
#     PlotCyclingResult(sol).plot(starting_location="E")
#
#
# if __name__ == "__main__":
#     main()


"""
This example will do an optimal control program of a 100 steps tracking a hand cycling motion with a torque driven and
a torque resistance at the handle.
"""


import numpy as np

from bioptim import (
    Axis,
    BiorbdModel,
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
    MultiCyclicCycleSolutions,
    Solution,
    SolutionMerge,
    ControlType,
    OptimalControlProgram,
)

from cocofest import OcpFesMsk, FesMskModel, DingModelPulseWidthFrequency, OcpFes, CustomObjective


def prepare_ocp(
        model: FesMskModel,
        pulse_width: dict,
        cycle_duration: int | float,
        n_cycles_simultaneous: int,
        objective: dict,
):
    cycle_len = OcpFes.prepare_n_shooting(model.muscles_dynamics_model[0].stim_time, cycle_duration)
    total_n_shooting = cycle_len * n_cycles_simultaneous

    # --- EXTERNAL FORCES --- #
    total_external_forces_frame = total_n_shooting
    external_force_set = ExternalForceSetTimeSeries(nb_frames=total_external_forces_frame)
    external_force_array = np.array([0, 0, -1])
    reshape_values_array = np.tile(external_force_array[:, np.newaxis], (1, total_external_forces_frame))
    external_force_set.add_torque(segment="wheel", values=reshape_values_array)

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

    # --- OBJECTIVE FUNCTION --- #
    # Adding an objective function to track a marker in a circular trajectory
    x_center = objective["cycling"]["x_center"]
    y_center = objective["cycling"]["y_center"]
    radius = objective["cycling"]["radius"]

    objective_functions = ObjectiveList()
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1000, quadratic=True)
    # objective_functions.add(CustomObjective.minimize_overall_muscle_force_production, custom_type=ObjectiveFcn.Lagrange, weight=10, quadratic=True)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTACT_FORCES, weight=0.0001, quadratic=True)

    # --- BOUNDS AND INITIAL GUESS --- #
    # Path constraint: x_bounds, x_init
    x_bounds, x_init = OcpFesMsk._set_bounds_fes(model)
    q_guess, qdot_guess = OcpFesMsk._prepare_initial_guess_cycling(model.biorbd_path,
                                                                   cycle_len,
                                                                   x_center,
                                                                   y_center,
                                                                   radius,
                                                                   n_cycles_simultaneous)

    # import matplotlib.pyplot as plt
    # plt.plot(q_guess[0])
    # plt.plot(q_guess[1])
    # plt.show()
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=0, target=q_guess[0][:-1], weight=1000,
                            quadratic=True)
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=1, target=q_guess[1][:-1], weight=1000,
                            quadratic=True)


    x_init.add("q", q_guess, interpolation=InterpolationType.EACH_FRAME)
    x_init.add("qdot", qdot_guess, interpolation=InterpolationType.EACH_FRAME)

    q_x_bounds = model.bio_model.bounds_from_ranges("q")
    q_x_bounds.min[0] = [-1]
    q_x_bounds.max[0] = [2]
    q_x_bounds.min[1] = [1]
    q_x_bounds.min[2] = [q_x_bounds.min[2][0] * n_cycles_simultaneous]
    q_x_bounds.max[2] = [5]

    # x_min_bound = []
    # x_max_bound = []
    # for i in range(q_x_bounds.min.shape[0]):
    #     x_min_bound.append([q_x_bounds.min[i][0]] * (cycle_len * n_cycles_simultaneous + 1))
    #     x_max_bound.append([q_x_bounds.max[i][0]] * (cycle_len * n_cycles_simultaneous + 1))
    #
    # # # Resize bounds to reduce search space
    # x_min_bound[0] = [-1] * (cycle_len * n_cycles_simultaneous + 1)
    # x_max_bound[0] = [2] * (cycle_len * n_cycles_simultaneous + 1)
    # x_min_bound[1] = [1] * (cycle_len * n_cycles_simultaneous + 1)
    # x_max_bound[2] = [5] * (cycle_len * n_cycles_simultaneous + 1)
    # x_min_bound[2] = [x_min_bound[2][0] * n_cycles_simultaneous] * (cycle_len * n_cycles_simultaneous + 1)  # Allow the wheel to spin as much as needed

    # cardinal_node_list = [i * int(cycle_len / 4) for i in range(4 * n_cycles_simultaneous + 1)]
    # for i in cardinal_node_list:
    #     x_min_bound[0][i] = x_init["q"].init[0][i] #- x_init["q"].init[0][i] * 0.01
    #     x_max_bound[0][i] = x_init["q"].init[0][i] #+ #x_init["q"].init[0][i] * 0.01
    #     x_min_bound[1][i] = x_init["q"].init[1][i] #- x_init["q"].init[1][i] * 0.01
    #     x_max_bound[1][i] = x_init["q"].init[1][i] #+ x_init["q"].init[1][i] * 0.01

    # x_bounds.add(key="q", min_bound=x_min_bound, max_bound=x_max_bound, phase=0,
    #              interpolation=InterpolationType.EACH_FRAME)

    x_bounds.add(key="q", bounds=q_x_bounds, phase=0)

    # Modifying pedal speed bounds
    qdot_x_bounds = model.bio_model.bounds_from_ranges("qdot")
    qdot_x_bounds.max[2] = [0, 0, 0]
    x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=0)

    # Define control path constraint: u_bounds, u_init
    u_bounds, u_init = OcpFesMsk._set_u_bounds_fes(model)
    u_bounds, u_init = OcpFesMsk._set_u_bounds_msk(u_bounds, u_init, model, with_residual_torque=True)
    u_bounds.add(key="tau", min_bound=np.array([-500, -500, -0]), max_bound=np.array([500, 500, 0]), phase=0,
                 interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

    # --- CONSTRAINTS --- #
    constraints = OcpFesMsk._build_constraints(
        model,
        cycle_len,
        cycle_duration,
        ControlType.CONSTANT,
        custom_constraint=None,
        external_forces=True,
        simultaneous_cycle=n_cycles_simultaneous,
    )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        node=Node.ALL,
        marker_index=model.marker_index("wheel_center"),
        axes=[Axis.X, Axis.Y],
    )

    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        first_marker="wheel_center",
        second_marker="global_wheel_center",
        node=Node.START,
        axes=[Axis.X, Axis.Y],
    )

    # cardinal_node_list = [i * int(cycle_len / 2) for i in range(2 * n_cycles_simultaneous + 1)]
    # for i in cardinal_node_list:
    #     objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", index=0, node=i, weight=10000000, target=q_guess[0][i], quadratic=True)
    #     objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", index=1, node=i, weight=10000000, target=q_guess[1][i], quadratic=True)
        # constraints.add(ConstraintFcn.TRACK_STATE, key="q", index=0, node=i, target=q_guess[0][i])
        # constraints.add(ConstraintFcn.TRACK_STATE, key="q", index=1, node=i, target=q_guess[1][i])
        # constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", index=0, node=i, target=qdot_guess[0][i])
        # constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", index=1, node=i, target=qdot_guess[1][i])

    # --- PARAMETERS --- #
    (parameters,
     parameters_bounds,
     parameters_init,
     parameter_objectives,
     ) = OcpFesMsk._build_parameters(
        model=model,
        pulse_width=pulse_width,
        pulse_intensity=None,
        use_sx=True,
    )

    # rebuilding model for the OCP
    model = FesMskModel(
        name=model.name,
        biorbd_path=model.biorbd_path,
        muscles_model=model.muscles_dynamics_model,
        stim_time=model.muscles_dynamics_model[0].stim_time,
        previous_stim=model.muscles_dynamics_model[0].previous_stim,
        activate_force_length_relationship=model.activate_force_length_relationship,
        activate_force_velocity_relationship=model.activate_force_velocity_relationship,
        activate_residual_torque=model.activate_residual_torque,
        parameters=parameters,
        external_force_set=external_force_set,
        for_cycling=True,
    )

    return OptimalControlProgram(
        [model],
        dynamics,
        n_shooting=total_n_shooting,
        phase_time=3,
        objective_functions=objective_functions,
        constraints=constraints,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        parameters=parameters,
        parameter_bounds=parameters_bounds,
        parameter_init=parameters_init,
        parameter_objectives=parameter_objectives,
        ode_solver=OdeSolver.RK1(n_integration_steps=3),
        control_type=ControlType.CONSTANT,
        n_threads=8,
        use_sx=True,
    )


def main():
    cycle_duration = 1
    n_cycles_simultaneous = 3

    model = FesMskModel(
        name=None,
        biorbd_path="../../msk_models/simplified_UL_Seth_pedal_aligned.bioMod",
        muscles_model=[
            DingModelPulseWidthFrequency(muscle_name="DeltoideusClavicle_A", is_approximated=False),
            DingModelPulseWidthFrequency(muscle_name="DeltoideusScapula_P", is_approximated=False),
            DingModelPulseWidthFrequency(muscle_name="TRIlong", is_approximated=False),
            DingModelPulseWidthFrequency(muscle_name="BIC_long", is_approximated=False),
            DingModelPulseWidthFrequency(muscle_name="BIC_brevis", is_approximated=False),
        ],
        stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                   1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                   2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_residual_torque=True,
        external_force_set=None,  # External forces will be added
    )

    minimum_pulse_width = DingModelPulseWidthFrequency().pd0

    ocp = prepare_ocp(
        model=model,
        pulse_width={
            "min": minimum_pulse_width,
            "max": 0.0006,
            "bimapping": False,
            "same_for_all_muscles": False,
            "fixed": False,
        },
        cycle_duration=cycle_duration,
        n_cycles_simultaneous=n_cycles_simultaneous,
        objective={"cycling": {"x_center": 0.35, "y_center": 0, "radius": 0.1},
                   "minimize_residual_torque": True},
    )

    # Solve the program
    sol = ocp.solve(
        solver=Solver.IPOPT(show_online_optim=False, _max_iter=1000, show_options=dict(show_bounds=True)),
    )

    sol.graphs(show_bounds=True)
    sol.animate(n_frames=200, show_tracked_markers=True)
    print(sol.constraints)
    print(sol.parameters)
    print(sol.detailed_cost)


if __name__ == "__main__":
    main()

