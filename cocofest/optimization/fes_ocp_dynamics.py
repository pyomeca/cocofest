import numpy as np

from bioptim import (
    Axis,
    BoundsList,
    ConstraintList,
    ControlType,
    DynamicsList,
    InitialGuessList,
    InterpolationType,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OdeSolverBase,
    OptimalControlProgram,
    ParameterList,
    ParameterObjectiveList,
    PhaseDynamics,
    VariableScaling,
)

from ..custom_constraints import CustomConstraint
from ..custom_objectives import CustomObjective
from ..dynamics.inverse_kinematics_and_dynamics import get_circle_coord
from ..dynamics.warm_start import get_initial_guess
from ..models.ding2003 import DingModelFrequency
from ..models.ding2007 import DingModelPulseDurationFrequency
from ..models.dynamical_model import FesMskModel
from ..models.fes_model import FesModel
from ..models.hmed2018 import DingModelIntensityFrequency
from ..optimization.fes_ocp import OcpFes


class OcpFesMsk:
    @staticmethod
    def prepare_ocp(
        biorbd_model_path: str,
        bound_type: str = None,
        bound_data: list = None,
        fes_muscle_models: list[FesModel] = None,
        n_stim: int = None,
        n_shooting: int = None,
        final_time: int | float = None,
        stim_time: list = None,
        pulse_event: dict = None,
        pulse_duration: dict = None,
        pulse_intensity: dict = None,
        objective: dict = None,
        custom_constraint: ConstraintList = None,
        with_residual_torque: bool = False,
        activate_force_length_relationship: bool = False,
        activate_force_velocity_relationship: bool = False,
        minimize_muscle_fatigue: bool = False,
        minimize_muscle_force: bool = False,
        use_sx: bool = True,
        warm_start: bool = False,
        ode_solver: OdeSolverBase = OdeSolver.RK4(n_integration_steps=1),
        control_type: ControlType = ControlType.CONSTANT,
        n_threads: int = 1,
    ):
        """
        Prepares the Optimal Control Program (OCP) with a musculoskeletal model for a movement to be solved.

        Parameters
        ----------
        biorbd_model_path : str
            The path to the bioMod file.
        bound_type : str
            The type of bound to use (start, end, start_end).
        bound_data : list
            The data to use for the bound.
        fes_muscle_models : list[FesModel]
            The FES model type used for the OCP.
        n_stim : int
            Number of stimulations that will occur during the OCP, also referred to as phases.
        n_shooting : int
            Number of shooting points for each individual phase.
        final_time : int | float
            The final time of the OCP.
        pulse_event : dict
            Dictionary containing parameters related to the appearance of the pulse.
            It should contain the following keys: "min", "max", "bimapping", "frequency", "round_down", "pulse_mode".
        pulse_duration : dict
            Dictionary containing parameters related to the duration of the pulse.
            It should contain the following keys: "fixed", "min", "max", "bimapping", "similar_for_all_muscles".
            Optional if not using the Ding2007 models
        pulse_intensity : dict
            Dictionary containing parameters related to the intensity of the pulse.
            It should contain the following keys: "fixed", "min", "max", "bimapping", "similar_for_all_muscles".
            Optional if not using the Hmed2018 models
        objective : dict
            Dictionary containing parameters related to the objective of the optimization.
        custom_constraint : ConstraintList,
            Custom constraints for the OCP.
        with_residual_torque : bool
            If residual torque is used.
        activate_force_length_relationship : bool
            If the force length relationship is used.
        activate_force_velocity_relationship : bool
            If the force velocity relationship is used.
        minimize_muscle_fatigue : bool
            Minimize the muscle fatigue.
        minimize_muscle_force : bool
            Minimize the muscle force.
        use_sx : bool
            The nature of the CasADi variables. MX are used if False.
        warm_start : bool
            If a warm start is run to get the problem initial guesses.
        ode_solver : OdeSolverBase
            The ODE solver to use.
        control_type : ControlType
            The type of control to use.
        n_threads : int
            The number of threads to use while solving (multi-threading if > 1).

        Returns
        -------
        OptimalControlProgram
            The prepared Optimal Control Program.
        """

        (pulse_event, pulse_duration, pulse_intensity, objective) = OcpFes._fill_dict(
            pulse_event, pulse_duration, pulse_intensity, objective
        )

        time_min = pulse_event["min"]
        time_max = pulse_event["max"]
        time_bimapping = pulse_event["bimapping"]
        frequency = pulse_event["frequency"]
        round_down = pulse_event["round_down"]
        pulse_mode = pulse_event["pulse_mode"]

        fixed_pulse_duration = pulse_duration["fixed"]
        pulse_duration_min = pulse_duration["min"]
        pulse_duration_max = pulse_duration["max"]
        pulse_duration_bimapping = pulse_duration["bimapping"]
        key_in_dict = "similar_for_all_muscles" in pulse_duration
        pulse_duration_similar_for_all_muscles = pulse_duration["similar_for_all_muscles"] if key_in_dict else False

        fixed_pulse_intensity = pulse_intensity["fixed"]
        pulse_intensity_min = pulse_intensity["min"]
        pulse_intensity_max = pulse_intensity["max"]
        pulse_intensity_bimapping = pulse_intensity["bimapping"]
        key_in_dict = "similar_for_all_muscles" in pulse_intensity
        pulse_intensity_similar_for_all_muscles = pulse_intensity["similar_for_all_muscles"] if key_in_dict else False

        force_tracking = objective["force_tracking"]
        end_node_tracking = objective["end_node_tracking"]
        cycling_objective = objective["cycling"]
        custom_objective = objective["custom"]
        key_in_dict = "q_tracking" in objective
        q_tracking = objective["q_tracking"] if key_in_dict else None

        OcpFes._sanity_check(
            model=fes_muscle_models[0],
            n_stim=n_stim,
            n_shooting=n_shooting,
            final_time=final_time,
            pulse_mode=pulse_mode,
            frequency=frequency,
            time_min=time_min,
            time_max=time_max,
            time_bimapping=time_bimapping,
            fixed_pulse_duration=fixed_pulse_duration,
            pulse_duration_min=pulse_duration_min,
            pulse_duration_max=pulse_duration_max,
            pulse_duration_bimapping=pulse_duration_bimapping,
            fixed_pulse_intensity=fixed_pulse_intensity,
            pulse_intensity_min=pulse_intensity_min,
            pulse_intensity_max=pulse_intensity_max,
            pulse_intensity_bimapping=pulse_intensity_bimapping,
            custom_objective=custom_objective,
            use_sx=use_sx,
            ode_solver=ode_solver,
            n_threads=n_threads,
        )

        OcpFesMsk._sanity_check_fes_models_inputs(
            biorbd_model_path=biorbd_model_path,
            bound_type=bound_type,
            bound_data=bound_data,
            fes_muscle_models=fes_muscle_models,
            force_tracking=force_tracking,
            end_node_tracking=end_node_tracking,
            cycling_objective=cycling_objective,
            q_tracking=q_tracking,
            with_residual_torque=with_residual_torque,
            activate_force_length_relationship=activate_force_length_relationship,
            activate_force_velocity_relationship=activate_force_velocity_relationship,
            minimize_muscle_fatigue=minimize_muscle_fatigue,
            minimize_muscle_force=minimize_muscle_force,
        )

        OcpFes._sanity_check_frequency(n_stim=n_stim, final_time=final_time, frequency=frequency, round_down=round_down)

        OcpFesMsk._sanity_check_muscle_model(biorbd_model_path=biorbd_model_path, fes_muscle_models=fes_muscle_models)

        force_fourier_coef = [] if force_tracking else None
        if force_tracking:
            for i in range(len(force_tracking[1])):
                force_fourier_coef.append(OcpFes._build_fourier_coefficient([force_tracking[0], force_tracking[1][i]]))

        q_fourier_coef = [] if q_tracking else None
        if q_tracking:
            for i in range(len(q_tracking[1])):
                q_fourier_coef.append(OcpFes._build_fourier_coefficient([q_tracking[0], q_tracking[1][i]]))

        (
            parameters,
            parameters_bounds,
            parameters_init,
            parameter_objectives,
            constraints,
        ) = OcpFesMsk._build_parameters(
            model=fes_muscle_models,
            n_stim=n_stim,
            stim_time=stim_time,
            time_min=time_min,
            time_max=time_max,
            time_bimapping=time_bimapping,
            fixed_pulse_duration=fixed_pulse_duration,
            pulse_duration_min=pulse_duration_min,
            pulse_duration_max=pulse_duration_max,
            pulse_duration_bimapping=pulse_duration_bimapping,
            pulse_duration_similar_for_all_muscles=pulse_duration_similar_for_all_muscles,
            fixed_pulse_intensity=fixed_pulse_intensity,
            pulse_intensity_min=pulse_intensity_min,
            pulse_intensity_max=pulse_intensity_max,
            pulse_intensity_bimapping=pulse_intensity_bimapping,
            pulse_intensity_similar_for_all_muscles=pulse_intensity_similar_for_all_muscles,
            use_sx=use_sx,
        )

        constraints = OcpFesMsk._set_constraints(constraints, custom_constraint)

        if len(constraints) == 0 and len(parameters) == 0:
            raise ValueError(
                "This is not an optimal control problem,"
                " add parameter to optimize or use the IvpFes method to build your problem"
            )

        bio_models = [
            FesMskModel(
                name=None,
                biorbd_path=biorbd_model_path,
                muscles_model=fes_muscle_models,
                activate_force_length_relationship=activate_force_length_relationship,
                activate_force_velocity_relationship=activate_force_velocity_relationship,
            )
        ]

        dynamics = OcpFesMsk._declare_dynamics(bio_models)
        initial_state = (
            get_initial_guess(biorbd_model_path, final_time, n_stim, n_shooting, objective) if warm_start else None
        )

        x_bounds, x_init = OcpFesMsk._set_bounds(
            bio_models,
            fes_muscle_models,
            bound_type,
            bound_data,
            n_stim,
            initial_state,
        )
        u_bounds, u_init = OcpFesMsk._set_controls(bio_models, n_stim, with_residual_torque)
        muscle_force_key = ["F_" + fes_muscle_models[i].muscle_name for i in range(len(fes_muscle_models))]
        objective_functions = OcpFesMsk._set_objective(
            n_stim,
            n_shooting,
            force_fourier_coef,
            end_node_tracking,
            cycling_objective,
            custom_objective,
            q_fourier_coef,
            minimize_muscle_fatigue,
            minimize_muscle_force,
            muscle_force_key,
            time_min,
            time_max,
        )

        return OptimalControlProgram(
            bio_model=bio_models,
            dynamics=dynamics,
            n_shooting=n_shooting,
            phase_time=final_time,
            objective_functions=objective_functions,
            x_init=x_init,
            x_bounds=x_bounds,
            u_init=u_init,
            u_bounds=u_bounds,
            constraints=constraints,
            parameters=parameters,
            parameter_bounds=parameters_bounds,
            parameter_init=parameters_init,
            parameter_objectives=parameter_objectives,
            control_type=control_type,
            use_sx=use_sx,
            ode_solver=ode_solver,
            n_threads=n_threads,
        )

    @staticmethod
    def _declare_dynamics(bio_models):
        dynamics = DynamicsList()
        dynamics.add(
            bio_models[0].declare_model_variables,
            dynamic_function=bio_models[0].muscle_dynamic,
            expand_dynamics=True,
            expand_continuity=False,
            phase=0,
            phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        )
        return dynamics

    @staticmethod
    def _build_parameters(
        model,
        n_stim,
        stim_time,
        time_min,
        time_max,
        time_bimapping,
        fixed_pulse_duration,
        pulse_duration_min,
        pulse_duration_max,
        pulse_duration_bimapping,
        pulse_duration_similar_for_all_muscles,
        fixed_pulse_intensity,
        pulse_intensity_min,
        pulse_intensity_max,
        pulse_intensity_bimapping,
        pulse_intensity_similar_for_all_muscles,
        use_sx,
    ):
        parameters = ParameterList(use_sx=use_sx)
        parameters_bounds = BoundsList()
        parameters_init = InitialGuessList()
        parameter_objectives = ParameterObjectiveList()
        constraints = ConstraintList()

        # if time_min:
        parameters.add(
            name="pulse_apparition_time",
            function=DingModelFrequency.set_pulse_apparition_time,
            size=n_stim,
            scaling=VariableScaling("pulse_apparition_time", [1] * n_stim),
        )

        parameters_bounds.add(
            "pulse_apparition_time",
            min_bound=np.array(stim_time),
            max_bound=np.array(stim_time),
            interpolation=InterpolationType.CONSTANT,
        )

        parameters_init["pulse_apparition_time"] = np.array(stim_time)

        for i in range(len(model)):
            if isinstance(model[i], DingModelPulseDurationFrequency):
                parameter_name = (
                    "pulse_duration"
                    if pulse_duration_similar_for_all_muscles
                    else "pulse_duration" + "_" + model[i].muscle_name
                )
                if fixed_pulse_duration:  # TODO : ADD SEVERAL INDIVIDUAL FIXED PULSE DURATION FOR EACH MUSCLE
                    if (
                        pulse_duration_similar_for_all_muscles and i == 0
                    ) or not pulse_duration_similar_for_all_muscles:
                        parameters.add(
                            name=parameter_name,
                            function=DingModelPulseDurationFrequency.set_impulse_duration,
                            size=n_stim,
                            scaling=VariableScaling(parameter_name, [1] * n_stim),
                        )
                        if isinstance(fixed_pulse_duration, list):
                            parameters_bounds.add(
                                parameter_name,
                                min_bound=np.array(fixed_pulse_duration),
                                max_bound=np.array(fixed_pulse_duration),
                                interpolation=InterpolationType.CONSTANT,
                            )
                            parameters_init.add(key=parameter_name, initial_guess=np.array(fixed_pulse_duration))
                        else:
                            parameters_bounds.add(
                                parameter_name,
                                min_bound=np.array([fixed_pulse_duration] * n_stim),
                                max_bound=np.array([fixed_pulse_duration] * n_stim),
                                interpolation=InterpolationType.CONSTANT,
                            )
                            parameters_init[parameter_name] = np.array([fixed_pulse_duration] * n_stim)

                elif (
                    pulse_duration_min and pulse_duration_max
                ):  # TODO : ADD SEVERAL MIN MAX PULSE DURATION FOR EACH MUSCLE
                    if (
                        pulse_duration_similar_for_all_muscles and i == 0
                    ) or not pulse_duration_similar_for_all_muscles:
                        parameters_bounds.add(
                            parameter_name,
                            min_bound=[pulse_duration_min],
                            max_bound=[pulse_duration_max],
                            interpolation=InterpolationType.CONSTANT,
                        )
                        pulse_duration_avg = (pulse_duration_max + pulse_duration_min) / 2
                        parameters_init[parameter_name] = np.array([pulse_duration_avg] * n_stim)
                        parameters.add(
                            name=parameter_name,
                            function=DingModelPulseDurationFrequency.set_impulse_duration,
                            size=n_stim,
                            scaling=VariableScaling(parameter_name, [1] * n_stim),
                        )

                if pulse_duration_bimapping:
                    pass
                    # parameter_bimapping.add(name="pulse_duration", to_second=[0 for _ in range(n_stim)], to_first=[0])
                    # TODO : Fix Bimapping in Bioptim

            if isinstance(model[i], DingModelIntensityFrequency):
                parameter_name = (
                    "pulse_intensity"
                    if pulse_intensity_similar_for_all_muscles
                    else "pulse_intensity" + "_" + model[i].muscle_name
                )
                if fixed_pulse_intensity:  # TODO : ADD SEVERAL INDIVIDUAL FIXED PULSE INTENSITY FOR EACH MUSCLE
                    if (
                        pulse_intensity_similar_for_all_muscles and i == 0
                    ) or not pulse_intensity_similar_for_all_muscles:
                        parameters.add(
                            name=parameter_name,
                            function=DingModelIntensityFrequency.set_impulse_intensity,
                            size=n_stim,
                            scaling=VariableScaling(parameter_name, [1] * n_stim),
                        )
                        if isinstance(fixed_pulse_intensity, list):
                            parameters_bounds.add(
                                parameter_name,
                                min_bound=np.array(fixed_pulse_intensity),
                                max_bound=np.array(fixed_pulse_intensity),
                                interpolation=InterpolationType.CONSTANT,
                            )
                            parameters_init.add(key=parameter_name, initial_guess=np.array(fixed_pulse_intensity))
                        else:
                            parameters_bounds.add(
                                parameter_name,
                                min_bound=np.array([fixed_pulse_intensity] * n_stim),
                                max_bound=np.array([fixed_pulse_intensity] * n_stim),
                                interpolation=InterpolationType.CONSTANT,
                            )
                            parameters_init[parameter_name] = np.array([fixed_pulse_intensity] * n_stim)

                elif (
                    pulse_intensity_min and pulse_intensity_max
                ):  # TODO : ADD SEVERAL MIN MAX PULSE INTENSITY FOR EACH MUSCLE
                    if (
                        pulse_intensity_similar_for_all_muscles and i == 0
                    ) or not pulse_intensity_similar_for_all_muscles:
                        parameters_bounds.add(
                            parameter_name,
                            min_bound=[pulse_intensity_min],
                            max_bound=[pulse_intensity_max],
                            interpolation=InterpolationType.CONSTANT,
                        )
                        intensity_avg = (pulse_intensity_min + pulse_intensity_max) / 2
                        parameters_init[parameter_name] = np.array([intensity_avg] * n_stim)
                        parameters.add(
                            name=parameter_name,
                            function=DingModelIntensityFrequency.set_impulse_intensity,
                            size=n_stim,
                            scaling=VariableScaling(parameter_name, [1] * n_stim),
                        )

                if pulse_intensity_bimapping:
                    pass
                    # parameter_bimapping.add(name="pulse_intensity",
                    #                         to_second=[0 for _ in range(n_stim)],
                    #                         to_first=[0])
                    # TODO : Fix Bimapping in Bioptim

        return parameters, parameters_bounds, parameters_init, parameter_objectives, constraints

    @staticmethod
    def _set_constraints(constraints, custom_constraint):
        if custom_constraint:
            for i in range(len(custom_constraint)):
                if custom_constraint[i]:
                    for j in range(len(custom_constraint[i])):
                        constraints.add(custom_constraint[i][j])
        return constraints

    @staticmethod
    def _set_bounds(bio_models, fes_muscle_models, bound_type, bound_data, n_stim, initial_state):
        # ---- STATE BOUNDS REPRESENTATION ---- #
        #
        #                    |‾‾‾‾‾‾‾‾‾‾x_max_middle‾‾‾‾‾‾‾‾‾‾‾‾x_max_end‾
        #                    |          max_bounds              max_bounds
        #    x_max_start     |
        #   _starting_bounds_|
        #   ‾starting_bounds‾|
        #    x_min_start     |
        #                    |          min_bounds              min_bounds
        #                     ‾‾‾‾‾‾‾‾‾‾x_min_middle‾‾‾‾‾‾‾‾‾‾‾‾x_min_end‾

        # Sets the bound for all the phases
        x_bounds = BoundsList()
        x_init = InitialGuessList()
        for model in fes_muscle_models:
            muscle_name = model.muscle_name
            variable_bound_list = [model.name_dof[i] + "_" + muscle_name for i in range(len(model.name_dof))]

            starting_bounds, min_bounds, max_bounds = (
                model.standard_rest_values(),
                model.standard_rest_values(),
                model.standard_rest_values(),
            )

            for i in range(len(variable_bound_list)):
                if variable_bound_list[i] == "Cn_" + muscle_name:
                    max_bounds[i] = 10
                elif variable_bound_list[i] == "F_" + muscle_name:
                    max_bounds[i] = 1000
                elif variable_bound_list[i] == "Tau1_" + muscle_name or variable_bound_list[i] == "Km_" + muscle_name:
                    max_bounds[i] = 1
                elif variable_bound_list[i] == "A_" + muscle_name:
                    min_bounds[i] = 0

            starting_bounds_min = np.concatenate((starting_bounds, min_bounds, min_bounds), axis=1)
            starting_bounds_max = np.concatenate((starting_bounds, max_bounds, max_bounds), axis=1)

            for j in range(len(variable_bound_list)):
                x_bounds.add(
                    variable_bound_list[j],
                    min_bound=np.array([starting_bounds_min[j]]),
                    max_bound=np.array([starting_bounds_max[j]]),
                    phase=0,
                    interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
                )

            for j in range(len(variable_bound_list)):
                x_init.add(variable_bound_list[j], model.standard_rest_values()[j], phase=0)

        if bound_type == "start_end":
            start_bounds = []
            end_bounds = []
            for i in range(bio_models[0].nb_q):
                start_bounds.append(3.14 / (180 / bound_data[0][i]) if bound_data[0][i] != 0 else 0)
                end_bounds.append(3.14 / (180 / bound_data[1][i]) if bound_data[1][i] != 0 else 0)

        elif bound_type == "start":
            start_bounds = []
            for i in range(bio_models[0].nb_q):
                start_bounds.append(3.14 / (180 / bound_data[i]) if bound_data[i] != 0 else 0)

        elif bound_type == "end":
            end_bounds = []
            for i in range(bio_models[0].nb_q):
                end_bounds.append(3.14 / (180 / bound_data[i]) if bound_data[i] != 0 else 0)

        q_x_bounds = bio_models[0].bounds_from_ranges("q")
        qdot_x_bounds = bio_models[0].bounds_from_ranges("qdot")

        if bound_type == "start_end":
            for j in range(bio_models[0].nb_q):
                q_x_bounds[j, [0]] = start_bounds[j]
                q_x_bounds[j, [-1]] = end_bounds[j]
        elif bound_type == "start":
            for j in range(bio_models[0].nb_q):
                q_x_bounds[j, [0]] = start_bounds[j]
        elif bound_type == "end":
            for j in range(bio_models[0].nb_q):
                q_x_bounds[j, [-1]] = end_bounds[j]
        qdot_x_bounds[:, [0]] = 0  # Start without any velocity

        x_bounds.add(key="q", bounds=q_x_bounds, phase=0)
        x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=0)

        # Sets the initial state of q, qdot and muscle forces for all the phases if a warm start is used
        if initial_state:
            muscle_names = bio_models[0].muscle_names
            x_init.add(
                key="q", initial_guess=initial_state["q"][0], interpolation=InterpolationType.EACH_FRAME, phase=0
            )
            x_init.add(
                key="qdot",
                initial_guess=initial_state["qdot"][0],
                interpolation=InterpolationType.EACH_FRAME,
                phase=0,
            )
            for j in range(len(muscle_names)):
                x_init.add(
                    key="F_" + muscle_names[j],
                    initial_guess=initial_state[muscle_names[j]][0],
                    interpolation=InterpolationType.EACH_FRAME,
                    phase=0,
                )
        else:
            for i in range(n_stim):
                x_init.add(key="q", initial_guess=[0] * bio_models[0].nb_q, phase=0)

        return x_bounds, x_init

    @staticmethod
    def _set_controls(bio_models, n_stim, with_residual_torque):
        # Controls bounds
        nb_tau = bio_models[0].nb_tau
        if with_residual_torque:  # TODO : ADD SEVERAL INDIVIDUAL FIXED RESIDUAL TORQUE FOR EACH JOINT
            tau_min, tau_max, tau_init = [-50] * nb_tau, [50] * nb_tau, [0] * nb_tau
        else:
            tau_min, tau_max, tau_init = [0] * nb_tau, [0] * nb_tau, [0] * nb_tau

        u_bounds = BoundsList()
        u_bounds.add(key="tau", min_bound=tau_min, max_bound=tau_max, phase=0)

        # Controls initial guess
        u_init = InitialGuessList()
        u_init.add(key="tau", initial_guess=tau_init, phase=0)

        return u_bounds, u_init

    @staticmethod
    def _set_objective(
        n_stim,
        n_shooting,
        force_fourier_coef,
        end_node_tracking,
        cycling_objective,
        custom_objective,
        q_fourier_coef,
        minimize_muscle_fatigue,
        minimize_muscle_force,
        muscle_force_key,
        time_min,
        time_max,
    ):
        # Creates the objective for our problem
        objective_functions = ObjectiveList()
        if custom_objective:
            for i in range(len(custom_objective)):
                if custom_objective[i]:
                    for j in range(len(custom_objective[i])):
                        objective_functions.add(custom_objective[i][j])

        if force_fourier_coef is not None:
            for j in range(len(muscle_force_key)):
                for phase in range(n_stim):
                    for i in range(n_shooting[phase]):
                        objective_functions.add(
                            CustomObjective.track_state_from_time,
                            custom_type=ObjectiveFcn.Mayer,
                            node=i,
                            fourier_coeff=force_fourier_coef[j],
                            key=muscle_force_key[j],
                            quadratic=True,
                            weight=1,
                            phase=phase,
                        )

        if end_node_tracking is not None:
            for j in range(len(muscle_force_key)):
                objective_functions.add(
                    ObjectiveFcn.Mayer.MINIMIZE_STATE,
                    node=Node.END,
                    key=muscle_force_key[j],
                    quadratic=True,
                    weight=1,
                    target=end_node_tracking[j],
                    phase=n_stim - 1,
                )

        if cycling_objective:
            x_center = cycling_objective["x_center"]
            y_center = cycling_objective["y_center"]
            radius = cycling_objective["radius"]
            circle_coord_list = np.array(
                [
                    get_circle_coord(theta, x_center, y_center, radius)[:-1]
                    for theta in np.linspace(0, -2 * np.pi, n_shooting[0] * n_stim + 1)
                ]
            )
            for phase in range(n_stim):
                objective_functions.add(
                    ObjectiveFcn.Mayer.TRACK_MARKERS,
                    weight=100000,
                    axes=[Axis.X, Axis.Y],
                    marker_index=0,
                    target=circle_coord_list[n_shooting[0] * phase : n_shooting[0] * (phase + 1) + 1].T,
                    node=Node.ALL,
                    phase=phase,
                    quadratic=True,
                )

        if q_fourier_coef:
            for j in range(len(q_fourier_coef)):
                for phase in range(n_stim):
                    for i in range(n_shooting[phase]):
                        objective_functions.add(
                            CustomObjective.track_state_from_time,
                            custom_type=ObjectiveFcn.Mayer,
                            node=i,
                            fourier_coeff=q_fourier_coef[j],
                            key="q",
                            quadratic=True,
                            weight=1,
                            phase=phase,
                            index=j,
                        )

        if minimize_muscle_fatigue:
            objective_functions.add(
                CustomObjective.minimize_overall_muscle_fatigue,
                custom_type=ObjectiveFcn.Mayer,
                node=Node.END,
                quadratic=True,
                weight=-1,
                phase=n_stim - 1,
            )

        if minimize_muscle_force:
            for i in range(n_stim):
                objective_functions.add(
                    CustomObjective.minimize_overall_muscle_force_production,
                    custom_type=ObjectiveFcn.Lagrange,
                    quadratic=True,
                    weight=1,
                    phase=i,
                )

        if time_min and time_max:
            for i in range(n_stim):
                objective_functions.add(
                    ObjectiveFcn.Mayer.MINIMIZE_TIME,
                    weight=0.001 / n_shooting[i],
                    min_bound=time_min,
                    max_bound=time_max,
                    quadratic=True,
                    phase=i,
                )

        return objective_functions

    @staticmethod
    def _sanity_check_muscle_model(biorbd_model_path, fes_muscle_models):
        tested_bio_model = FesMskModel(name=None, biorbd_path=biorbd_model_path, muscles_model=fes_muscle_models)
        fes_muscle_models_name_list = [fes_muscle_models[x].muscle_name for x in range(len(fes_muscle_models))]
        for biorbd_muscle in tested_bio_model.muscle_names:
            if biorbd_muscle not in fes_muscle_models_name_list:
                raise ValueError(
                    f"The muscle {biorbd_muscle} is not in the fes muscle model "
                    f"please add it into the fes_muscle_models list by providing the muscle_name ="
                    f" {biorbd_muscle}"
                )

    @staticmethod
    def _sanity_check_fes_models_inputs(
        biorbd_model_path,
        bound_type,
        bound_data,
        fes_muscle_models,
        force_tracking,
        end_node_tracking,
        cycling_objective,
        q_tracking,
        with_residual_torque,
        activate_force_length_relationship,
        activate_force_velocity_relationship,
        minimize_muscle_fatigue,
        minimize_muscle_force,
    ):
        if not isinstance(biorbd_model_path, str):
            raise TypeError("biorbd_model_path should be a string")

        if bound_type:
            tested_bio_model = FesMskModel(name=None, biorbd_path=biorbd_model_path, muscles_model=fes_muscle_models)
            if not isinstance(bound_type, str) or bound_type not in ["start", "end", "start_end"]:
                raise ValueError("bound_type should be a string and should be equal to start, end or start_end")
            if not isinstance(bound_data, list):
                raise TypeError("bound_data should be a list")
            if bound_type == "start_end":
                if len(bound_data) != 2 or not isinstance(bound_data[0], list) or not isinstance(bound_data[1], list):
                    raise TypeError("bound_data should be a list of two list")
                if len(bound_data[0]) != tested_bio_model.nb_q or len(bound_data[1]) != tested_bio_model.nb_q:
                    raise ValueError(f"bound_data should be a list of {tested_bio_model.nb_q} elements")
                for i in range(len(bound_data[0])):
                    if not isinstance(bound_data[0][i], int | float) or not isinstance(bound_data[1][i], int | float):
                        raise TypeError(
                            f"bound data index {i}: {bound_data[0][i]} and {bound_data[1][i]} should be an int or float"
                        )
            if bound_type == "start" or bound_type == "end":
                if len(bound_data) != tested_bio_model.nb_q:
                    raise ValueError(f"bound_data should be a list of {tested_bio_model.nb_q} element")
                for i in range(len(bound_data)):
                    if not isinstance(bound_data[i], int | float):
                        raise TypeError(f"bound data index {i}: {bound_data[i]} should be an int or float")

        for i in range(len(fes_muscle_models)):
            if not isinstance(fes_muscle_models[i], FesModel):
                raise TypeError("model must be a FesModel type")

        if force_tracking:
            if isinstance(force_tracking, list):
                if len(force_tracking) != 2:
                    raise ValueError("force_tracking must of size 2")
                if not isinstance(force_tracking[0], np.ndarray):
                    raise TypeError(f"force_tracking index 0: {force_tracking[0]} must be np.ndarray type")
                if not isinstance(force_tracking[1], list):
                    raise TypeError(f"force_tracking index 1: {force_tracking[1]} must be list type")
                if len(force_tracking[1]) != len(fes_muscle_models):
                    raise ValueError(
                        "force_tracking index 1 list must have the same size as the number of muscles in fes_muscle_models"
                    )
                for i in range(len(force_tracking[1])):
                    if len(force_tracking[0]) != len(force_tracking[1][i]):
                        raise ValueError("force_tracking time and force argument must be the same length")
            else:
                raise TypeError(f"force_tracking: {force_tracking} must be list type")

        if end_node_tracking:
            if not isinstance(end_node_tracking, list):
                raise TypeError(f"force_tracking: {end_node_tracking} must be list type")
            if len(end_node_tracking) != len(fes_muscle_models):
                raise ValueError(
                    "end_node_tracking list must have the same size as the number of muscles in fes_muscle_models"
                )
            for i in range(len(end_node_tracking)):
                if not isinstance(end_node_tracking[i], int | float):
                    raise TypeError(f"end_node_tracking index {i}: {end_node_tracking[i]} must be int or float type")

        if cycling_objective:
            if not isinstance(cycling_objective, dict):
                raise TypeError(f"cycling_objective: {cycling_objective} must be dictionary type")

            if len(cycling_objective) != 4:
                raise ValueError(
                    "cycling_objective dictionary must have the same size as the number of muscles in fes_muscle_models"
                )

            cycling_objective_keys = ["x_center", "y_center", "radius", "target"]
            if not all([cycling_objective_keys[i] in cycling_objective for i in range(len(cycling_objective_keys))]):
                raise ValueError(
                    f"cycling_objective dictionary must contain the following keys: {cycling_objective_keys}"
                )

            if not all([isinstance(cycling_objective[key], int | float) for key in cycling_objective_keys[:3]]):
                raise TypeError(f"cycling_objective x_center, y_center and radius inputs must be int or float")

            if isinstance(cycling_objective[cycling_objective_keys[-1]], str):
                if (
                    cycling_objective[cycling_objective_keys[-1]] != "marker"
                    and cycling_objective[cycling_objective_keys[-1]] != "q"
                ):
                    raise ValueError(
                        f"{cycling_objective[cycling_objective_keys[-1]]} not implemented chose between 'marker' and 'q' as 'target'"
                    )
            else:
                raise TypeError(f"cycling_objective target must be string type")

        if q_tracking:
            if not isinstance(q_tracking, list) and len(q_tracking) != 2:
                raise TypeError("q_tracking should be a list of size 2")
            tested_bio_model = FesMskModel(name=None, biorbd_path=biorbd_model_path, muscles_model=fes_muscle_models)
            if not isinstance(q_tracking[0], list | np.ndarray):
                raise ValueError("q_tracking[0] should be a list or array type")
            if len(q_tracking[1]) != tested_bio_model.nb_q:
                raise ValueError("q_tracking[1] should have the same size as the number of generalized coordinates")
            for i in range(tested_bio_model.nb_q):
                if len(q_tracking[0]) != len(q_tracking[1][i]):
                    raise ValueError("q_tracking[0] and q_tracking[1] should have the same size")

        list_to_check = [
            with_residual_torque,
            activate_force_length_relationship,
            activate_force_velocity_relationship,
            minimize_muscle_fatigue,
            minimize_muscle_force,
        ]

        list_to_check_name = [
            "with_residual_torque",
            "activate_force_length_relationship",
            "activate_force_velocity_relationship",
            "minimize_muscle_fatigue",
            "minimize_muscle_force",
        ]

        for i in range(len(list_to_check)):
            if list_to_check[i]:
                if not isinstance(list_to_check[i], bool):
                    raise TypeError(f"{list_to_check_name[i]} should be a boolean")
