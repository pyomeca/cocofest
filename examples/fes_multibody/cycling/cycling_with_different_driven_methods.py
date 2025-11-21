"""
This example will do an optimal control program of a 100 steps hand cycling motion with either a torque driven /
muscle driven / FES driven dynamics and includes a resistive torque at the handle.
"""

from sys import platform
import numpy as np

from bioptim import (
    Axis,
    BiorbdModel,
    BoundsList,
    ConstraintList,
    ConstraintFcn,
    ControlType,
    CostType,
    ExternalForceSetTimeSeries,
    InitialGuessList,
    InterpolationType,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    ParameterList,
    PhaseDynamics,
    Solver,
    VariableScalingList,
    DynamicsOptionsList,
    DynamicsOptions,
)

from cocofest import (
    CustomObjective,
    DingModelPulseWidthFrequencyWithFatigue,
    FesMskModel,
    inverse_kinematics_cycling,
    OcpFesMsk,
    DingModelPulseWidthFrequency,
)


def set_external_forces(n_shooting: int, torque: int | float) -> tuple[dict, ExternalForceSetTimeSeries]:
    """
    Create an external force time series applying a constant torque.

    Parameters
    ----------
        n_shooting: int
            Number of shooting nodes.
        torque: int | float
            Torque value to be applied.

    Returns
    -------
        A tuple with a numerical time series dictionary and the ExternalForceSetTimeSeries object.
    """
    external_force_set = ExternalForceSetTimeSeries(nb_frames=n_shooting)
    external_force_array = np.array([0, 0, torque])
    reshape_values_array = np.tile(external_force_array[:, np.newaxis], (1, n_shooting))
    external_force_set.add_torque(segment="wheel", values=reshape_values_array, force_name="resistive_torque")
    numerical_time_series = {"external_forces": external_force_set.to_numerical_time_series()}
    return numerical_time_series, external_force_set


def update_model(
    model: BiorbdModel | FesMskModel, external_force_set: ExternalForceSetTimeSeries, parameters: ParameterList = None
) -> BiorbdModel | FesMskModel:
    """
    Update the model with external forces and parameters if necessary.

    Parameters
    ----------
    model: BiorbdModel | FesMskModel
        The initial model.
    external_force_set: ExternalForceSetTimeSeries
        The external forces to be applied.
    parameters: ParameterList
        Optional parameters for the FES model.

    Returns
    -------
    Updated model instance.
    """
    if isinstance(model, FesMskModel):
        model = FesMskModel(
            name=model.name,
            biorbd_path=model.biorbd_path,
            muscles_model=model.muscles_dynamics_model,
            stim_time=model.muscles_dynamics_model[0].stim_time,
            previous_stim=model.muscles_dynamics_model[0].previous_stim,
            activate_force_length_relationship=model.activate_force_length_relationship,
            activate_force_velocity_relationship=model.activate_force_velocity_relationship,
            activate_passive_force_relationship=model.activate_force_velocity_relationship,
            activate_residual_torque=model.activate_residual_torque,
            parameters=parameters,
            external_force_set=external_force_set,
            with_contact=model.with_contact
        )
    else:
        model = BiorbdModel(model.path, external_force_set=external_force_set)

    return model


# def set_dynamics(
#     model: BiorbdModel | FesMskModel,
#     numerical_time_series: dict,
#     dynamics_type_str: str = "torque_driven",
#     ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=10),
# ) -> DynamicsList:
#     """
#     Set the dynamics of the optimal control program based on the chosen dynamics type.
#
#     Parameters
#     ----------
#     model: BiorbdModel | FesMskModel
#         The biomechanical model.
#     numerical_time_series: dict
#         External numerical data (e.g., external forces).
#     dynamics_type_str: str
#         Type of dynamics ("torque_driven", "muscle_driven", or "fes_driven").
#
#     Returns
#     -------
#         A DynamicsList configured for the problem.
#     """
#     dynamics_type = (
#         DynamicsFcn.TORQUE_DRIVEN
#         if dynamics_type_str == "torque_driven"
#         else (
#             DynamicsFcn.MUSCLE_DRIVEN
#             if dynamics_type_str == "muscle_driven"
#             else model.declare_model_variables if dynamics_type_str == "fes_driven" else None
#         )
#     )
#     if dynamics_type is None:
#         raise ValueError("Dynamics type not recognized")
#
#     dynamics = DynamicsList()
#     dynamics.add(
#         dynamics_type=dynamics_type,
#         dynamic_function=(
#             None if dynamics_type in (DynamicsFcn.TORQUE_DRIVEN, DynamicsFcn.MUSCLE_DRIVEN) else model.muscle_dynamic
#         ),
#         expand_dynamics=True,
#         expand_continuity=False,
#         phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
#         numerical_data_timeseries=numerical_time_series,
#         phase=0,
#         ode_solver=ode_solver,
#         contact_type=[ContactType.RIGID_EXPLICIT],
#     )
#     return dynamics


def set_objective_functions(model: BiorbdModel | FesMskModel, dynamics_type: str, init_x) -> ObjectiveList:
    """
    Configure the objective functions for the optimal control problem.

    Parameters
    ----------
    model: BiorbdModel | FesMskModel
        The biomechanical model.
    dynamics_type: str
        The type of dynamics used.

    Returns
    -------
    An ObjectiveList with the appropriate objectives.
    """
    objective_functions = ObjectiveList()
    if isinstance(model, FesMskModel):
        objective_functions.add(
            CustomObjective.minimize_overall_muscle_force_production,
            custom_type=ObjectiveFcn.Lagrange,
            node=Node.ALL,
            weight=10000,
            quadratic=True,
        )

    else:
        control_key = "tau" if dynamics_type == "torque_driven" else "muscles"
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key=control_key, weight=1000, quadratic=True)

    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        key="q",
        index=2,
        node=Node.END,
        weight=10,
        target=init_x,
        quadratic=True,
    )

    return objective_functions


def set_x_init(
    n_shooting: int, pedal_config: dict, turn_number: int, ode_solver: OdeSolver, model_path: str
) -> InitialGuessList:
    """
    Set the initial guess for the state variables based on inverse kinematics.

    Parameters
    ----------
    n_shooting: int
        Number of shooting nodes.
    pedal_config: dict
        Dictionary with keys "x_center", "y_center", and "radius".
    turn_number: int
        Number of complete turns.
    ode_solver: OdeSolver
        The ODE solver used in the optimal control problem.
    model_path
        Path to the biomechanical model used for inverse kinematics.

    Returns
    -------
    An InitialGuessList for the state variables.
    """
    x_init = InitialGuessList()
    # Path to the biomechanical model used for inverse kinematics

    n_shooting = (
        n_shooting * (ode_solver.polynomial_degree + 1) if isinstance(ode_solver, OdeSolver.COLLOCATION) else n_shooting
    )

    q_guess, qdot_guess, qddotguess = inverse_kinematics_cycling(
        model_path,
        n_shooting,
        x_center=pedal_config["x_center"],
        y_center=pedal_config["y_center"],
        radius=pedal_config["radius"],
        ik_method="trf",
        cycling_number=turn_number,
    )

    # --- Set q and qdot initial guesses values obtained by inverse kinematics --- #
    if isinstance(ode_solver, OdeSolver.COLLOCATION):
        x_init.add("q", q_guess, interpolation=InterpolationType.ALL_POINTS)
        x_init.add("qdot", qdot_guess, interpolation=InterpolationType.ALL_POINTS)
    elif isinstance(ode_solver, OdeSolver.RK1 | OdeSolver.RK2 | OdeSolver.RK4):
        x_init.add("q", q_guess, interpolation=InterpolationType.EACH_FRAME)
        x_init.add("qdot", qdot_guess, interpolation=InterpolationType.EACH_FRAME)
    else:
        raise RuntimeError("ode_solver must be COLLOCATION or RK4")

    return x_init


def set_u_bounds_and_init(
    model: BiorbdModel | FesMskModel, dynamics_type_str: str
) -> tuple[InitialGuessList, BoundsList, VariableScalingList]:
    """
    Define the control bounds and initial guess for the optimal control problem.

    Parameters
    ----------
    model: BiorbdModel | FesMskModel
        The biomechanical model.
    dynamics_type_str: str
        Type of dynamics ("torque_driven" or "muscle_driven").

    Returns
    -------
    A tuple containing the initial guess list for controls and the bounds list.
    """
    u_bounds = BoundsList()
    u_init = InitialGuessList()
    u_scaling = VariableScalingList()
    if dynamics_type_str == "torque_driven":
        u_bounds.add(key="tau", min_bound=np.array([-50, -50, -0]), max_bound=np.array([50, 50, 0]), phase=0)
    elif dynamics_type_str == "muscle_driven":
        muscle_min, muscle_max, muscle_init = 0.0, 1.0, 0.5
        u_bounds.add(
            key="muscles",
            min_bound=np.array([muscle_min] * model.nb_muscles),
            max_bound=np.array([muscle_max] * model.nb_muscles),
            phase=0,
        )
        u_init.add(key="muscles", initial_guess=np.array([muscle_init] * model.nb_muscles), phase=0)
    elif dynamics_type_str == "fes_driven":
        if isinstance(model.muscles_dynamics_model[0], DingModelPulseWidthFrequency):
            for model in model.muscles_dynamics_model:
                key = "last_pulse_width_" + str(model.muscle_name)
                u_init.add(key=key, initial_guess=[model.pd0], phase=0)
                u_bounds.add(key=key, min_bound=[model.pd0], max_bound=[0.0006], phase=0)
                u_scaling.add(key=key, scaling=[1 / 400])

    return u_init, u_bounds, u_scaling


def set_state_bounds(
    model: BiorbdModel | FesMskModel,
    x_init: InitialGuessList,
    n_shooting: int,
    ode_solver: OdeSolver,
) -> tuple[BoundsList, InitialGuessList]:
    """
    Set the bounds for the state variables.

    Parameters
    ----------
    model: BiorbdModel | FesMskModel
        The biomechanical model.
    x_init: InitialGuessList
        Initial guess for states.

    Returns
    -------
    A BoundsList object with the defined state bounds.
    """
    # --- Set interpolation type according to ode_solver type --- #
    interpolation_type = InterpolationType.EACH_FRAME
    if isinstance(ode_solver, OdeSolver.COLLOCATION):
        n_shooting = n_shooting * (ode_solver.polynomial_degree + 1)
        interpolation_type = InterpolationType.ALL_POINTS

    # --- Initialize default FES bounds and intial guess --- #
    if hasattr(model, "muscles_dynamics_model"):
        x_bounds, x_init_fes = OcpFesMsk.set_x_bounds_fes(model)

        # --- Setting FES initial guesses --- #
        for key in x_init_fes.keys():
            initial_guess = np.array([[x_init_fes[key].init[0][0]] * (n_shooting + 1)])
            x_init.add(key=key, initial_guess=initial_guess, phase=0, interpolation=interpolation_type)
    else:
        x_bounds = BoundsList()

    # --- Setting q bounds --- #
    q_x_bounds = model.bounds_from_ranges("q")

    # --- First: enter general bound values in radiant --- #
    arm_q = [0, 1.5]  # Arm min_max q bound in radiant
    forarm_q = [0.5, 2.5]  # Forarm min_max q bound in radiant
    slack = 0.05  # Wheel rotation slack
    wheel_q = [x_init["q"].init[2][-1] - slack, x_init["q"].init[2][0] + slack]  # Wheel min_max q bound in radiant

    # --- Second: set general bound values in radiant, CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT mandatory for qdot --- #
    q_x_bounds.min[0] = [arm_q[0], arm_q[0], arm_q[0]]
    q_x_bounds.max[0] = [arm_q[1], arm_q[1], arm_q[1]]
    q_x_bounds.min[1] = [forarm_q[0], forarm_q[0], forarm_q[0]]
    q_x_bounds.max[1] = [forarm_q[1], forarm_q[1], forarm_q[1]]
    q_x_bounds.min[2] = [x_init["q"].init[2][0], wheel_q[0] - 2, x_init["q"].init[2][-1] - slack]
    q_x_bounds.max[2] = [x_init["q"].init[2][0], wheel_q[1] + 2, x_init["q"].init[2][-1] + slack]

    x_bounds.add(
        key="q", bounds=q_x_bounds, phase=0, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT
    )

    # --- Setting qdot bounds --- #
    qdot_x_bounds = model.bounds_from_ranges("qdot")

    # --- First: enter general bound values in radiant --- #
    arm_qdot = [-10, 10]  # Arm min_max qdot bound in radiant
    forarm_qdot = [-14, 10]  # Forarm min_max qdot bound in radiant
    wheel_qdot = [-2 * np.pi - 3, -2 * np.pi + 3]  # Wheel min_max qdot bound in radiant

    # --- Second: set general bound values in radiant, CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT mandatory for qdot --- #
    qdot_x_bounds.min[0] = [arm_qdot[0], arm_qdot[0], arm_qdot[0]]
    qdot_x_bounds.max[0] = [arm_qdot[1], arm_qdot[1], arm_qdot[1]]
    qdot_x_bounds.min[1] = [forarm_qdot[0], forarm_qdot[0], forarm_qdot[0]]
    qdot_x_bounds.max[1] = [forarm_qdot[1], forarm_qdot[1], forarm_qdot[1]]
    qdot_x_bounds.min[2] = [wheel_qdot[0], wheel_qdot[0], wheel_qdot[0]]
    qdot_x_bounds.max[2] = [wheel_qdot[1], wheel_qdot[1], wheel_qdot[1]]

    x_bounds.add(
        key="qdot",
        bounds=qdot_x_bounds,
        phase=0,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    return x_bounds, x_init


def set_constraints(model: BiorbdModel | FesMskModel) -> ConstraintList:
    """
    Set constraints for the optimal control problem.

    Parameters
    ----------
    model: BiorbdModel | FesMskModel
        The biomechanical model.

    Returns
    -------
        A ConstraintList with the defined constraints.
    """
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        node=Node.START,
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
    return constraints


def prepare_ocp(
    model: BiorbdModel | FesMskModel,
    n_shooting: int,
    final_time: int,
    turn_number: int,
    pedal_config: dict,
    dynamics_type: str = "torque_driven",
    use_sx: bool = True,
    ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=10),
    torque: int | float = -1,
    initial_guess_model_path: str = None,
) -> OptimalControlProgram:
    """
    Prepare the optimal control program (OCP) with the provided configuration.

    Parameters
    ----------
    model: BiorbdModel | FesMskModel
        The biomechanical model.
    n_shooting: int
        Number of shooting nodes.
    final_time: int
        Total time of the motion.
    turn_number: int
        Number of complete turns.
    pedal_config: dict
        Dictionary with pedal configuration (e.g., center and radius).
    dynamics_type: str
        Type of dynamics ("torque_driven", "muscle_driven", or "fes_driven").
    use_sx: bool
        Whether to use CasADi SX for symbolic computations.

    Returns
    -------
        An OptimalControlProgram instance configured for the problem.
    """
    # Set external forces (e.g., resistive torque at the handle)
    numerical_time_series, external_force_set = set_external_forces(n_shooting, torque=torque)

    # Set stimulation time in numerical_data_time_series
    if isinstance(model, FesMskModel):
        numerical_data_time_series, stim_idx_at_node_list = model.muscles_dynamics_model[
            0
        ].get_numerical_data_time_series(n_shooting, final_time)
        numerical_time_series.update(numerical_data_time_series)

    # Set dynamics based on the chosen dynamics type
    # dynamics = set_dynamics(
    #     model,
    #     numerical_time_series,
    #     dynamics_type_str=dynamics_type,
    #     ode_solver=ode_solver,
    # )
    dynamics_options = DynamicsOptionsList()
    dynamics_options.add(
        DynamicsOptions(
            expand_dynamics=True,
            expand_continuity=False,
            phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
            ode_solver=ode_solver,
            numerical_data_timeseries=numerical_time_series,
        )
    )

    # Set initial guess for state variables
    x_init = set_x_init(
        n_shooting, pedal_config, turn_number, ode_solver=ode_solver, model_path=initial_guess_model_path
    )

    # Define state bounds
    x_bounds, x_init = set_state_bounds(
        model=model,
        x_init=x_init,
        n_shooting=n_shooting,
        ode_solver=ode_solver,
    )

    # Define control bounds and initial guess
    u_init, u_bounds, u_scaling = set_u_bounds_and_init(model, dynamics_type_str=dynamics_type)

    # Set constraints
    constraints = set_constraints(model)

    # Configure objective functions
    objective_functions = set_objective_functions(model, dynamics_type, np.array([x_init["q"].init[2][-1]]))

    # Update the model with external forces and parameters
    model = update_model(model, external_force_set, parameters=ParameterList(use_sx=use_sx))

    return OptimalControlProgram(
        [model],
        n_shooting,
        final_time,
        dynamics_options,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        u_scaling=u_scaling,
        objective_functions=objective_functions,
        n_threads=32,
        constraints=constraints,
        use_sx=False,
        control_type=ControlType.CONSTANT,
    )


def main(
    plot=True,
    model_path: str = "../../msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod",
    initial_guess_model_path: str = "../../msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling_for_IK.bioMod",
):
    """
    Main function to configure and solve the optimal control problem.
    """
    # --- Configuration --- #
    dynamics_type = "fes_driven"  # Available options: "torque_driven", "muscle_driven", "fes_driven"
    # --- Supplementary available configurations --- #
    # dynamics_type = "torque_driven"
    # dynamics_type = "muscle_driven"
    # model_path = "../../msk_models/Seth/Modified_UL_Seth_2D_Cycling.bioMod"
    # model_path = "../../msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod"
    # IK_biorbd_model_path = "../../msk_models/Seth/Modified_UL_Seth_2D_Cycling_for_IK.bioMod"
    # IK_biorbd_model_path = "../../msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling_for_IK.bioMod"

    final_time = 2
    turn_number = 2
    pedal_config = {"x_center": 0.35, "y_center": 0.0, "radius": 0.1}

    # --- Load the appropriate model --- #
    if dynamics_type in ["torque_driven", "muscle_driven"]:
        model = BiorbdModel(model_path)
        n_shooting = 100 * final_time
    elif dynamics_type == "fes_driven":
        # Set FES model (set to Ding et al. 2007 + fatigue, for now)
        dummy_biomodel = BiorbdModel(model_path)
        muscle_name_list = dummy_biomodel.muscle_names
        muscles_model = [
            DingModelPulseWidthFrequencyWithFatigue(muscle_name=muscle, sum_stim_truncation=6)
            for muscle in muscle_name_list
        ]

        parameter_dict = {
            "Biceps": {"Fmax": 149, "a_scale": 3314.7, "alpha_a": -5.6 * 10e-2, "tau_fat": 179.6},
            "Triceps": {"Fmax": 617, "a_scale": 7036.3, "alpha_a": -2.4 * 10e-2, "tau_fat": 76.2},
            "Delt_ant": {"Fmax": 48, "a_scale": 1148.6, "alpha_a": -1.4 * 10e-1, "tau_fat": 445.5},
            "Delt_post": {"Fmax": 51, "a_scale": 1234.5, "alpha_a": -1.1 * 10e-1, "tau_fat": 342.7},
        }

        for model in muscles_model:
            muscle_name = model.muscle_name
            model.a_scale = parameter_dict[muscle_name]["a_scale"]
            model.a_rest = parameter_dict[muscle_name]["a_scale"]
            model.fmax = parameter_dict[muscle_name]["Fmax"]
            model.alpha_a = parameter_dict[muscle_name]["alpha_a"]
            model.tau_fat = parameter_dict[muscle_name]["tau_fat"]

        stim_time = list(np.linspace(0, final_time, 33 * final_time + 1)[:-1])
        model = FesMskModel(
            name=None,
            biorbd_path=model_path,
            muscles_model=muscles_model,
            stim_time=stim_time,
            activate_force_length_relationship=True,
            activate_force_velocity_relationship=True,
            activate_passive_force_relationship=True,
            activate_residual_torque=False,
            external_force_set=None,  # External forces will be added later
            with_contact=True
        )
        # Adjust n_shooting based on the stimulation time
        n_shooting = model.muscles_dynamics_model[0].get_n_shooting(final_time)
    else:
        raise ValueError(f"Dynamics type '{dynamics_type}' not recognized")

    ocp = prepare_ocp(
        model=model,
        n_shooting=n_shooting,
        final_time=final_time,
        turn_number=turn_number,
        pedal_config=pedal_config,
        dynamics_type=dynamics_type,
        use_sx=False,
        ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3, method="radau"),
        # ode_solver=OdeSolver.RK4(n_integration_steps=5),
        torque=-0.3,
        initial_guess_model_path=initial_guess_model_path,
    )

    # Add the penalty cost function plot
    ocp.add_plot_penalty(CostType.ALL)

    # Solve the optimal control problem
    linear_solver = "ma57" if platform == "linux" else "mumps"
    sol = ocp.solve(
        Solver.IPOPT(
            show_online_optim=False, _max_iter=10000, show_options=dict(show_bounds=True), _linear_solver=linear_solver
        )
    )
    sol.print_cost()
    if plot:
        sol.animate(viewer="pyorerun")
        sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
