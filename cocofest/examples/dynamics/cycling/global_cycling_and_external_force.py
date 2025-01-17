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
    FesMskModel,
    CustomObjective,
    DingModelPulseWidthFrequency,
    OcpFesMsk,
    OcpFes,
)


def prepare_ocp(
    model: BiorbdModel | FesMskModel,
    n_shooting: int,
    final_time: int,
    turn_number: int,
    pedal_config: dict,
    pulse_width: dict,
    dynamics_type: str = "torque_driven",
    use_sx: bool = True,
) -> OptimalControlProgram:

    # Dynamics
    numerical_time_series, external_force_set = set_external_forces(n_shooting, torque=-1)
    dynamics = set_dynamics(model=model, numerical_time_series=numerical_time_series, dynamics_type_str=dynamics_type)

    # Define objective functions
    objective_functions = set_objective_functions(model, dynamics_type)

    # Initial q guess
    x_init = set_x_init(n_shooting, pedal_config, turn_number)

    # Path constraint
    x_bounds = set_bounds(model=model,
                          x_init=x_init,
                          n_shooting=n_shooting,
                          turn_number=turn_number,
                          interpolation_type=InterpolationType.EACH_FRAME,
                          cardinal=1)
    # x_bounds = set_bounds(bio_model=bio_model, x_init=x_init, n_shooting=n_shooting, interpolation_type=InterpolationType.CONSTANT)

    # Control path constraint
    u_bounds, u_init = set_u_bounds_and_init(model, dynamics_type_str=dynamics_type)

    # Constraints
    constraints = set_constraints(model, n_shooting, turn_number)

    # Parameters
    parameters = None
    parameters_bounds = None
    parameters_init = None
    parameter_objectives = None
    if isinstance(model, FesMskModel) and isinstance(pulse_width, dict):
        (parameters,
         parameters_bounds,
         parameters_init,
         parameter_objectives,
         ) = OcpFesMsk._build_parameters(
            model=model,
            pulse_width=pulse_width,
            pulse_intensity=None,
            use_sx=use_sx,
        )

    # Update model
    model = updating_model(model=model, external_force_set=external_force_set, parameters=parameters)

    return OptimalControlProgram(
        [model],
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        ode_solver=OdeSolver.RK1(n_integration_steps=20),
        n_threads=20,
        constraints=constraints,
        parameters=parameters,
        parameter_bounds=parameters_bounds,
        parameter_init=parameters_init,
        parameter_objectives=parameter_objectives,
        use_sx=use_sx,
    )


def set_external_forces(n_shooting, torque):
    external_force_set = ExternalForceSetTimeSeries(nb_frames=n_shooting)
    external_force_array = np.array([0, 0, torque])
    reshape_values_array = np.tile(external_force_array[:, np.newaxis], (1, n_shooting))
    external_force_set.add_torque(segment="wheel", values=reshape_values_array)
    numerical_time_series = {"external_forces": external_force_set.to_numerical_time_series()}
    return numerical_time_series, external_force_set


def updating_model(model, external_force_set, parameters=None):
    if isinstance(model, FesMskModel):
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
    else:
        model = BiorbdModel(model.path, external_force_set=external_force_set)

    return model


def set_dynamics(model, numerical_time_series, dynamics_type_str="torque_driven"):
    dynamics_type = (DynamicsFcn.TORQUE_DRIVEN if dynamics_type_str == "torque_driven"
                         else DynamicsFcn.MUSCLE_DRIVEN if dynamics_type_str == "muscle_driven"
                         else model.declare_model_variables if dynamics_type_str == "fes_driven"
                         else None)
    if dynamics_type is None:
        raise ValueError("Dynamics type not recognized")

    dynamics = DynamicsList()
    dynamics.add(
        dynamics_type=dynamics_type,
        dynamic_function=None if dynamics_type in (DynamicsFcn.TORQUE_DRIVEN, DynamicsFcn.MUSCLE_DRIVEN) else model.muscle_dynamic,
        expand_dynamics=True,
        expand_continuity=False,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        numerical_data_timeseries=numerical_time_series,
        with_contact=True,
        phase=0,
    )
    return dynamics


def set_objective_functions(model, dynamics_type):
    objective_functions = ObjectiveList()
    if isinstance(model, FesMskModel):
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100000, quadratic=True)
        # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MARKERS_ACCELERATION, marker_index=model.marker_index("hand"), weight=100, quadratic=True)
        # objective_functions.add(CustomObjective.minimize_overall_muscle_force_production, custom_type=ObjectiveFcn.Lagrange, weight=1, quadratic=True)
    else:
        control_key = "tau" if dynamics_type == "torque_driven" else "muscles"
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key=control_key, weight=1000, quadratic=True)
    # if isinstance(model, BiorbdModel):
    #     objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MARKERS_ACCELERATION,
    #                             marker_index=model.marker_index("hand"), weight=100, quadratic=True)
    # else:  # TO DO: check
    #     objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MARKERS_ACCELERATION,
    #                             marker_index=model.marker_index("hand"), weight=100, quadratic=True)

    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTACT_FORCES, weight=1, quadratic=True)

    return objective_functions


def set_x_init(n_shooting, pedal_config, turn_number):
    x_init = InitialGuessList()

    biorbd_model_path = "../../msk_models/simplified_UL_Seth_pedal_aligned_test_one_marker.bioMod"
    # biorbd_model_path = "../../msk_models/arm26_cycling_pedal_aligned_contact_one_marker.bioMod"
    q_guess, qdot_guess, qddotguess = inverse_kinematics_cycling(
        biorbd_model_path,
        n_shooting,
        x_center=pedal_config["x_center"],
        y_center=pedal_config["y_center"],
        radius=pedal_config["radius"],
        ik_method="lm",
        cycling_number=turn_number
    )
    x_init.add("q", q_guess, interpolation=InterpolationType.EACH_FRAME)
    # x_init.add("qdot", qdot_guess, interpolation=InterpolationType.EACH_FRAME)
    # u_guess = inverse_dynamics_cycling(biorbd_model_path, q_guess, qdot_guess, qddotguess)
    # u_init.add("tau", u_guess, interpolation=InterpolationType.EACH_FRAME)

    return x_init


def set_u_bounds_and_init(bio_model, dynamics_type_str):
    u_bounds = BoundsList()
    u_init = InitialGuessList()
    if dynamics_type_str == "torque_driven":
        u_bounds.add(key="tau", min_bound=np.array([-50, -50, -0]), max_bound=np.array([50, 50, 0]), phase=0)
    elif dynamics_type_str == "muscle_driven":
        muscle_min, muscle_max, muscle_init = 0.0, 1.0, 0.5
        u_bounds.add(key="muscles",
                     min_bound=np.array([muscle_min] * bio_model.nb_muscles),
                     max_bound=np.array([muscle_max] * bio_model.nb_muscles),
                     phase=0)
        u_init.add(key="muscles",
                   initial_guess=np.array([muscle_init] * bio_model.nb_muscles),
                   phase=0)
    if dynamics_type_str == "fes_driven":
        u_bounds.add(key="tau", min_bound=np.array([-50, -50, -0]), max_bound=np.array([50, 50, 0]), phase=0)

    return u_bounds, u_init


def set_bounds(model, x_init, n_shooting, turn_number, interpolation_type=InterpolationType.CONSTANT, cardinal=4):
    x_bounds = BoundsList()
    if isinstance(model, FesMskModel):
        x_bounds, _ = OcpFesMsk._set_bounds_fes(model)

    q_x_bounds = model.bounds_from_ranges("q")
    qdot_x_bounds = model.bounds_from_ranges("qdot")

    if interpolation_type == InterpolationType.EACH_FRAME:
        x_min_bound = []
        x_max_bound = []
        for i in range(q_x_bounds.min.shape[0]):
            x_min_bound.append([q_x_bounds.min[i][0]] * (n_shooting + 1))
            x_max_bound.append([q_x_bounds.max[i][0]] * (n_shooting + 1))

        cardinal_node_list = [i * int(n_shooting / ((n_shooting/(n_shooting/turn_number)) * cardinal)) for i in range(int((n_shooting/(n_shooting/turn_number)) * cardinal + 1))]
        slack = 10*(np.pi/180)
        for i in cardinal_node_list:
            x_min_bound[2][i] = x_init["q"].init[2][i] - slack
            x_max_bound[2][i] = x_init["q"].init[2][i] + slack

        x_bounds.add(key="q", min_bound=x_min_bound, max_bound=x_max_bound, phase=0,
                     interpolation=InterpolationType.EACH_FRAME)

    else:
        x_bounds.add(key="q", bounds=q_x_bounds, phase=0)

    # Modifying pedal speed bounds
    # qdot_x_bounds.max[2] = [0, 0, 0]
    # qdot_x_bounds.min[2] = [-60, -60, -60]
    x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=0)
    return x_bounds


def set_constraints(bio_model, n_shooting, turn_number):
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        node=Node.START,
        marker_index=bio_model.marker_index("wheel_center"),
        axes=[Axis.X, Axis.Y],
    )

    superimpose_marker_list = [i * int(n_shooting / ((n_shooting / (n_shooting / turn_number)) * 1)) for i in
                          range(int((n_shooting / (n_shooting / turn_number)) * 1 + 1))]
    for i in superimpose_marker_list:
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            first_marker="wheel_center",
            second_marker="global_wheel_center",
            node=i,
            axes=[Axis.X, Axis.Y],
        )

    # cardinal_node_list = [i * int(300 / ((300 / 100) * 2)) for i in
    #                       range(int((300 / 100) * 2 + 1))]
    # for i in cardinal_node_list:
    #     constraints.add(ConstraintFcn.TRACK_STATE, key="q", index=0, node=i, target=x_init["q"].init[2][i])

    return constraints


def main():
    # --- Prepare the ocp --- #
    dynamics_type = "fes_driven"
    model_path = "../../msk_models/simplified_UL_Seth_pedal_aligned.bioMod"
    pulse_width = None
    n_shooting = 300
    final_time = 1
    if dynamics_type == "torque_driven" or dynamics_type == "muscle_driven":
        model = BiorbdModel(model_path)
    elif dynamics_type == "fes_driven":
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
            # stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
            #            1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
            #            2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
            # stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            stim_time=list(np.linspace(0, 1, 34)[:-1]),
            activate_force_length_relationship=True,
            activate_force_velocity_relationship=True,
            activate_residual_torque=True,
            external_force_set=None,  # External forces will be added
        )
        pulse_width = {"min": DingModelPulseWidthFrequency().pd0, "max": 0.0006, "bimapping": False, "same_for_all_muscles": False,
                       "fixed": False}
        # n_shooting = OcpFes.prepare_n_shooting(model.muscles_dynamics_model[0].stim_time, final_time)
        n_shooting = 33
    else:
        raise ValueError("Dynamics type not recognized")

    ocp = prepare_ocp(
        model=model,
        n_shooting=n_shooting,
        final_time=final_time,
        turn_number=1,
        pedal_config={"x_center": 0.35, "y_center": 0, "radius": 0.1},
        pulse_width=pulse_width,
        dynamics_type=dynamics_type,
    )
    ocp.add_plot_penalty(CostType.ALL)
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=10000)) #, show_options=dict(show_bounds=True)))#, show_options=dict(show_bounds=True)))
    sol.graphs(show_bounds=True)
    sol.animate(viewer="pyorerun")

    # 914 iter before recuperation


if __name__ == "__main__":
    main()
