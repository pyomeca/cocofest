import numpy as np

from bioptim import (
    BoundsList,
    ConstraintList,
    ControlType,
    DynamicsList,
    ExternalForceSetTimeSeries,
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

from ..custom_objectives import CustomObjective
from ..models.ding2007 import DingModelPulseWidthFrequency
from ..models.dynamical_model import FesMskModel
from ..models.hmed2018 import DingModelPulseIntensityFrequency
from ..optimization.fes_ocp import OcpFes
from ..fourier_approx import FourierSeries
from ..custom_constraints import CustomConstraint


class OcpFesMsk:
    @staticmethod
    def _prepare_optimization_problem(input_dict: dict) -> dict:

        (pulse_width, pulse_intensity, objective) = OcpFes._fill_dict(
            input_dict["pulse_width"],
            input_dict["pulse_intensity"],
            input_dict["objective"],
        )

        (
            pulse_width,
            pulse_intensity,
            objective,
            msk_info,
        ) = OcpFesMsk._fill_msk_dict(pulse_width, pulse_intensity, objective, input_dict["msk_info"])

        OcpFes._sanity_check(
            model=input_dict["model"],
            n_shooting=input_dict["n_shooting"],
            final_time=input_dict["final_time"],
            objective=objective,
            use_sx=input_dict["use_sx"],
            ode_solver=input_dict["ode_solver"],
            n_threads=input_dict["n_threads"],
        )

        OcpFesMsk._sanity_check_msk_inputs(
            model=input_dict["model"],
            msk_info=msk_info,
            objective=objective,
        )

        (
            parameters,
            parameters_bounds,
            parameters_init,
            parameter_objectives,
        ) = OcpFesMsk._build_parameters(
            model=input_dict["model"],
            pulse_intensity=pulse_intensity,
            use_sx=input_dict["use_sx"],
        )

        if input_dict["external_forces"]:
            input_dict["n_total_cycles"] = input_dict["n_total_cycles"] if "n_total_cycles" in input_dict.keys() else 1
            numerical_time_series, with_contact, external_force_set = OcpFesMsk._prepare_numerical_time_series(
                input_dict
            )
        else:
            numerical_time_series, with_contact, external_force_set = None, False, None

        numerical_data_time_series, stim_idx_at_node_list = (
            input_dict["model"]
            .muscles_dynamics_model[0]
            .get_numerical_data_time_series(input_dict["n_shooting"], input_dict["final_time"])
        )

        if numerical_time_series:
            numerical_time_series.update(numerical_data_time_series)
        else:
            numerical_time_series = numerical_data_time_series

        dynamics = OcpFesMsk._declare_dynamics(
            input_dict["model"], numerical_time_series=numerical_time_series, with_contact=with_contact
        )

        x_bounds, x_init = OcpFesMsk._set_bounds_fes(input_dict["model"])
        x_bounds, x_init = OcpFesMsk._set_bounds_msk(x_bounds, x_init, input_dict["model"], msk_info)

        u_bounds, u_init = OcpFesMsk._set_u_bounds_fes(input_dict["model"])
        u_bounds, u_init = OcpFesMsk._set_u_bounds_msk(
            u_bounds, u_init, input_dict["model"], msk_info["with_residual_torque"]
        )

        constraints = OcpFesMsk._build_constraints(input_dict["model"], input_dict["n_shooting"], stim_idx_at_node_list)

        muscle_force_key = [
            "F_" + input_dict["model"].muscles_dynamics_model[i].muscle_name
            for i in range(len(input_dict["model"].muscles_dynamics_model))
        ]

        objective_functions = OcpFesMsk._set_objective(
            input_dict["n_shooting"],
            muscle_force_key,
            objective,
            input_dict["n_cycles_simultaneous"] if "n_cycles_simultaneous" in input_dict.keys() else 1,
        )

        # rebuilding model for the OCP
        model = FesMskModel(
            name=input_dict["model"].name,
            biorbd_path=input_dict["model"].biorbd_path,
            muscles_model=input_dict["model"].muscles_dynamics_model,
            stim_time=input_dict["model"].muscles_dynamics_model[0].stim_time,
            previous_stim=input_dict["model"].muscles_dynamics_model[0].previous_stim,
            activate_force_length_relationship=input_dict["model"].activate_force_length_relationship,
            activate_force_velocity_relationship=input_dict["model"].activate_force_velocity_relationship,
            activate_residual_torque=input_dict["model"].activate_residual_torque,
            parameters=parameters,
            external_force_set=external_force_set,
        )

        optimization_dict = {
            "model": model,
            "dynamics": dynamics,
            "n_shooting": input_dict["n_shooting"],
            "final_time": input_dict["final_time"],
            "objective_functions": objective_functions,
            "x_init": x_init,
            "x_bounds": x_bounds,
            "u_init": u_init,
            "u_bounds": u_bounds,
            "constraints": constraints,
            "parameters": parameters,
            "parameters_bounds": parameters_bounds,
            "parameters_init": parameters_init,
            "parameter_objectives": parameter_objectives,
            "use_sx": input_dict["use_sx"],
            "ode_solver": input_dict["ode_solver"],
            "n_threads": input_dict["n_threads"],
            "control_type": input_dict["control_type"],
        }

        return optimization_dict

    @staticmethod
    def prepare_ocp(
        model: FesMskModel = None,
        final_time: int | float = None,
        pulse_width: dict = None,
        pulse_intensity: dict = None,
        objective: dict = None,
        msk_info: dict = None,
        use_sx: bool = True,
        initial_guess_warm_start: bool = False,
        ode_solver: OdeSolverBase = OdeSolver.RK4(n_integration_steps=1),
        control_type: ControlType = ControlType.CONSTANT,
        n_threads: int = 1,
        external_forces: dict = None,
    ):
        """
        Prepares the Optimal Control Program (OCP) with a musculoskeletal model for a movement to be solved.

        Parameters
        ----------
        model : FesModel
            The FES model to use.
        final_time : int | float
            The final time of the OCP.
            It should contain the following keys: "min", "max", "bimapping", "frequency", "round_down", "pulse_mode".
        pulse_width : dict
            Dictionary containing parameters related to the duration of the pulse.
            It should contain the following keys: "fixed", "min", "max", "bimapping", "similar_for_all_muscles".
            Optional if not using the Ding2007 models
        pulse_intensity : dict
            Dictionary containing parameters related to the intensity of the pulse.
            It should contain the following keys: "fixed", "min", "max", "bimapping", "similar_for_all_muscles".
            Optional if not using the Hmed2018 models
        objective : dict
            Dictionary containing parameters related to the objective of the optimization.
        msk_info : dict
            Dictionary containing parameters related to the musculoskeletal model.
        use_sx : bool
            The nature of the CasADi variables. MX are used if False.
        initial_guess_warm_start : bool
            If a warm start is run to get the problem initial guesses.
        ode_solver : OdeSolverBase
            The ODE solver to use.
        control_type : ControlType
            The type of control to use.
        n_threads : int
            The number of threads to use while solving (multi-threading if > 1).
        external_forces : dict
            Dictionary containing the parameters related to the external forces.

        Returns
        -------
        OptimalControlProgram
            The prepared Optimal Control Program.
        """

        input_dict = {
            "model": model,
            "n_shooting": OcpFes.prepare_n_shooting(model.muscles_dynamics_model[0].stim_time, final_time),
            "final_time": final_time,
            "pulse_width": pulse_width,
            "pulse_intensity": pulse_intensity,
            "objective": objective,
            "msk_info": msk_info,
            "initial_guess_warm_start": initial_guess_warm_start,
            "use_sx": use_sx,
            "ode_solver": ode_solver,
            "n_threads": n_threads,
            "control_type": control_type,
            "external_forces": external_forces,
        }

        optimization_dict = OcpFesMsk._prepare_optimization_problem(input_dict)

        return OptimalControlProgram(
            bio_model=[optimization_dict["model"]],
            dynamics=optimization_dict["dynamics"],
            n_shooting=optimization_dict["n_shooting"],
            phase_time=optimization_dict["final_time"],
            objective_functions=optimization_dict["objective_functions"],
            x_init=optimization_dict["x_init"],
            x_bounds=optimization_dict["x_bounds"],
            u_init=optimization_dict["u_init"],
            u_bounds=optimization_dict["u_bounds"],
            constraints=optimization_dict["constraints"],
            parameters=optimization_dict["parameters"],
            parameter_bounds=optimization_dict["parameters_bounds"],
            parameter_init=optimization_dict["parameters_init"],
            parameter_objectives=optimization_dict["parameter_objectives"],
            control_type=control_type,
            use_sx=optimization_dict["use_sx"],
            ode_solver=optimization_dict["ode_solver"],
            n_threads=optimization_dict["n_threads"],
        )

    @staticmethod
    def _fill_msk_dict(pulse_width, pulse_intensity, objective, msk_info):

        pulse_width = pulse_width if pulse_width else {}
        default_pulse_width = {
            "fixed": None,
            "min": None,
            "max": None,
            "bimapping": False,
            "same_for_all_muscles": False,
        }

        pulse_intensity = pulse_intensity if pulse_intensity else {}
        default_pulse_intensity = {
            "fixed": None,
            "min": None,
            "max": None,
            "bimapping": False,
            "same_for_all_muscles": False,
        }

        objective = objective if objective else {}
        default_objective = {
            "force_tracking": None,
            "end_node_tracking": None,
            "custom": None,
            "q_tracking": None,
            "minimize_muscle_fatigue": False,
            "minimize_muscle_force": False,
            "minimize_residual_torque": False,
        }

        msk_info = msk_info if msk_info else {}
        default_msk_info = {
            "bound_type": None,
            "bound_data": None,
            "with_residual_torque": False,
            "custom_constraint": None,
        }

        pulse_width = {**default_pulse_width, **pulse_width}
        pulse_intensity = {**default_pulse_intensity, **pulse_intensity}
        objective = {**default_objective, **objective}
        msk_info = {**default_msk_info, **msk_info}

        return pulse_width, pulse_intensity, objective, msk_info

    @staticmethod
    def _prepare_numerical_time_series(input_dict):
        n_cycles_simultaneous = (
            input_dict["n_cycles_simultaneous"] if "n_cycles_simultaneous" in input_dict.keys() else 1
        )
        total_n_shooting = input_dict["n_shooting"] * n_cycles_simultaneous
        total_external_forces_frame = (
            input_dict["n_total_cycles"] * input_dict["n_shooting"]
            if input_dict["n_total_cycles"] >= n_cycles_simultaneous
            else total_n_shooting
        )
        external_force_set = ExternalForceSetTimeSeries(nb_frames=total_external_forces_frame)

        external_force_array = np.array(input_dict["external_forces"]["torque"])
        reshape_values_array = np.tile(external_force_array[:, np.newaxis], (1, total_external_forces_frame))
        external_force_set.add_torque(
            segment=input_dict["external_forces"]["Segment_application"], values=reshape_values_array
        )

        numerical_time_series = {"external_forces": external_force_set.to_numerical_time_series()}
        with_contact = (
            input_dict["external_forces"]["with_contact"]
            if "with_contact" in input_dict["external_forces"].keys()
            else False
        )

        return numerical_time_series, with_contact, external_force_set

    @staticmethod
    def _declare_dynamics(bio_models, numerical_time_series, with_contact):
        dynamics = DynamicsList()
        dynamics.add(
            bio_models.declare_model_variables,
            dynamic_function=bio_models.muscle_dynamic,
            expand_dynamics=True,
            expand_continuity=False,
            phase=0,
            phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
            numerical_data_timeseries=numerical_time_series,
            with_contact=with_contact,
        )
        return dynamics

    @staticmethod
    def _build_parameters(
        model: FesMskModel,
        pulse_intensity: dict,
        use_sx: bool = True,
    ):
        parameters = ParameterList(use_sx=use_sx)
        parameters_bounds = BoundsList()
        parameters_init = InitialGuessList()
        parameter_objectives = ParameterObjectiveList()

        n_stim = len(model.muscles_dynamics_model[0].stim_time)

        for i in range(len(model.muscles_dynamics_model)):
            if isinstance(model.muscles_dynamics_model[i], DingModelPulseIntensityFrequency):
                if pulse_intensity["bimapping"]:
                    n_stim = 1
                parameter_name = (
                    "pulse_intensity"
                    if pulse_intensity["same_for_all_muscles"]
                    else "pulse_intensity" + "_" + model.muscles_dynamics_model[i].muscle_name
                )
                if pulse_intensity["fixed"]:  # TODO : ADD SEVERAL INDIVIDUAL FIXED PULSE INTENSITY FOR EACH MUSCLE
                    if (pulse_intensity["same_for_all_muscles"] and i == 0) or not pulse_intensity[
                        "same_for_all_muscles"
                    ]:
                        parameters.add(
                            name=parameter_name,
                            function=DingModelPulseIntensityFrequency.set_impulse_intensity,
                            size=n_stim,
                            scaling=VariableScaling(parameter_name, [1] * n_stim),
                        )
                        if isinstance(pulse_intensity["fixed"], list):
                            parameters_bounds.add(
                                parameter_name,
                                min_bound=np.array(pulse_intensity["fixed"]),
                                max_bound=np.array(pulse_intensity["fixed"]),
                                interpolation=InterpolationType.CONSTANT,
                            )
                            parameters_init.add(
                                key=parameter_name,
                                initial_guess=np.array(pulse_intensity["fixed"]),
                            )
                        else:
                            parameters_bounds.add(
                                parameter_name,
                                min_bound=np.array([pulse_intensity["fixed"]] * n_stim),
                                max_bound=np.array([pulse_intensity["fixed"]] * n_stim),
                                interpolation=InterpolationType.CONSTANT,
                            )
                            parameters_init[parameter_name] = np.array([pulse_intensity["fixed"]] * n_stim)

                elif (
                    pulse_intensity["min"] and pulse_intensity["max"]
                ):  # TODO : ADD SEVERAL MIN MAX PULSE INTENSITY FOR EACH MUSCLE
                    if (pulse_intensity["same_for_all_muscles"] and i == 0) or not pulse_intensity[
                        "same_for_all_muscles"
                    ]:
                        parameters_bounds.add(
                            parameter_name,
                            min_bound=[pulse_intensity["min"]],
                            max_bound=[pulse_intensity["max"]],
                            interpolation=InterpolationType.CONSTANT,
                        )
                        intensity_avg = (pulse_intensity["min"] + pulse_intensity["max"]) / 2
                        parameters_init[parameter_name] = np.array([intensity_avg] * n_stim)
                        parameters.add(
                            name=parameter_name,
                            function=DingModelPulseIntensityFrequency.set_impulse_intensity,
                            size=n_stim,
                            scaling=VariableScaling(parameter_name, [1] * n_stim),
                        )

        return (
            parameters,
            parameters_bounds,
            parameters_init,
            parameter_objectives,
        )

    @staticmethod
    def _build_constraints(models, n_shooting, stim_idx_at_node_list, custom_constraint=None):
        constraints = ConstraintList()
        if isinstance(models.muscles_dynamics_model[0], DingModelPulseIntensityFrequency):
            for model in models.muscles_dynamics_model:
                for i in range(n_shooting):

                    # last_stim_idx = stim_idx_at_node_list[i][-1] - model._sum_stim_truncation
                    # bimapped_parameters_idx = bimapped_parameters_idx_list[i] if bimapped_for_nmpc else \
                    # bimapped_parameters_idx_list[0]

                    last_stim_idx = stim_idx_at_node_list[i][-1]

                    constraints.add(
                        CustomConstraint.pulse_intensity_sliding_window_constraint,
                        # last_stim_index=last_stim_idx if not bimapped_parameters else bimapped_parameters_idx,
                        last_stim_idx=last_stim_idx,
                        muscle_name=model.muscle_name,
                        node=i,
                    )

        if custom_constraint:
            for i in range(len(custom_constraint)):
                if custom_constraint[i]:
                    for j in range(len(custom_constraint[i])):
                        constraints.add(custom_constraint[i][j])

        return constraints

    @staticmethod
    def _set_bounds_fes(bio_models):
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
        for model in bio_models.muscles_dynamics_model:
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

        return x_bounds, x_init

    @staticmethod
    def _set_bounds_msk(x_bounds, x_init, bio_models, msk_info):
        if msk_info["bound_type"] == "start_end":
            start_bounds = []
            end_bounds = []
            for i in range(bio_models.nb_q):
                start_bounds.append(
                    3.14 / (180 / msk_info["bound_data"][0][i]) if msk_info["bound_data"][0][i] != 0 else 0
                )
                end_bounds.append(
                    3.14 / (180 / msk_info["bound_data"][1][i]) if msk_info["bound_data"][1][i] != 0 else 0
                )

        elif msk_info["bound_type"] == "start":
            start_bounds = []
            for i in range(bio_models.nb_q):
                start_bounds.append(3.14 / (180 / msk_info["bound_data"][i]) if msk_info["bound_data"][i] != 0 else 0)

        elif msk_info["bound_type"] == "end":
            end_bounds = []
            for i in range(bio_models.nb_q):
                end_bounds.append(3.14 / (180 / msk_info["bound_data"][i]) if msk_info["bound_data"][i] != 0 else 0)

        q_x_bounds = bio_models.bounds_from_ranges("q")
        qdot_x_bounds = bio_models.bounds_from_ranges("qdot")

        if msk_info["bound_type"] == "start_end":
            for j in range(bio_models.nb_q):
                q_x_bounds[j, [0]] = start_bounds[j]
                q_x_bounds[j, [-1]] = end_bounds[j]
        elif msk_info["bound_type"] == "start":
            for j in range(bio_models.nb_q):
                q_x_bounds[j, [0]] = start_bounds[j]
        elif msk_info["bound_type"] == "end":
            for j in range(bio_models.nb_q):
                q_x_bounds[j, [-1]] = end_bounds[j]
        qdot_x_bounds[:, [0]] = 0  # Start without any velocity

        x_bounds.add(key="q", bounds=q_x_bounds, phase=0)
        x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=0)

        return x_bounds, x_init

    @staticmethod
    def _set_u_bounds_fes(bio_models):
        u_bounds = BoundsList()  # Controls bounds
        u_init = InitialGuessList()  # Controls initial guess
        models = bio_models.muscles_dynamics_model
        if isinstance(models[0], DingModelPulseWidthFrequency):
            for model in models:
                key = "last_pulse_width_" + str(model.muscle_name)
                u_init.add(key=key, initial_guess=[0], phase=0)
                u_bounds.add(key=key, min_bound=[model.pd0], max_bound=[0.0006], phase=0)

        return u_bounds, u_init

    @staticmethod
    def _set_u_bounds_msk(u_bounds, u_init, bio_models, with_residual_torque):
        if with_residual_torque:  # TODO : ADD SEVERAL INDIVIDUAL FIXED RESIDUAL TORQUE FOR EACH JOINT
            nb_tau = bio_models.nb_tau
            tau_min, tau_max, tau_init = [-200] * nb_tau, [200] * nb_tau, [0] * nb_tau
            u_bounds.add(
                key="tau", min_bound=tau_min, max_bound=tau_max, phase=0, interpolation=InterpolationType.CONSTANT
            )
            u_init.add(key="tau", initial_guess=tau_init, phase=0)

        models = bio_models.muscles_dynamics_model
        if isinstance(models[0], DingModelPulseWidthFrequency):
            for model in models:
                key = "last_pulse_width_" + str(model.muscle_name)
                u_init.add(key=key, initial_guess=[0], phase=0)
                u_bounds.add(key=key, min_bound=[model.pd0], max_bound=[0.0006], phase=0)

        if isinstance(models[0], DingModelPulseIntensityFrequency):
            for model in models:
                key = "pulse_intensity_" + str(model.muscle_name)
                u_init.add(key=key, initial_guess=[0] * model._sum_stim_truncation, phase=0)
                u_bounds.add(
                    key=key,
                    min_bound=[model.min_pulse_intensity()] * model._sum_stim_truncation,
                    max_bound=[130] * model._sum_stim_truncation,
                    phase=0,
                )

        return u_bounds, u_init

    @staticmethod
    def _set_objective(
        n_shooting,
        muscle_force_key,
        objective,
        n_simultaneous_cycle: int = 1,
    ):
        # Creates the objective for our problem
        objective_functions = ObjectiveList()
        if objective["custom"]:
            for i in range(len(objective["custom"])):
                if objective["custom"][i]:
                    for j in range(len(objective["custom"][i])):
                        objective_functions.add(objective["custom"][i][j])

        if objective["force_tracking"]:
            force_fourier_coef = []
            for i in range(len(objective["force_tracking"][1])):
                force_fourier_coef.append(
                    OcpFes._build_fourier_coefficient(
                        [
                            objective["force_tracking"][0],
                            objective["force_tracking"][1][i],
                        ]
                    )
                )

            force_to_track = []
            for i in range(len(force_fourier_coef)):
                force_to_track.append(
                    FourierSeries().fit_func_by_fourier_series_with_real_coeffs(
                        np.linspace(0, 1, n_shooting + 1),
                        force_fourier_coef[i],
                    )[np.newaxis, :]
                )

            for j in range(len(muscle_force_key)):
                objective_functions.add(
                    ObjectiveFcn.Lagrange.TRACK_STATE,
                    key=muscle_force_key[j],
                    weight=100,
                    target=force_to_track[j],
                    node=Node.ALL,
                    quadratic=True,
                )

        if objective["end_node_tracking"] is not None:
            for j in range(len(muscle_force_key)):
                objective_functions.add(
                    ObjectiveFcn.Mayer.MINIMIZE_STATE,
                    node=Node.END,
                    key=muscle_force_key[j],
                    quadratic=True,
                    weight=1,
                    target=objective["end_node_tracking"][j],
                    phase=0,
                )

        if objective["q_tracking"]:
            q_fourier_coef = []
            for i in range(len(objective["q_tracking"][1])):
                q_fourier_coef.append(
                    OcpFes._build_fourier_coefficient([objective["q_tracking"][0], objective["q_tracking"][1][i]])
                )

            q_to_track = []
            for i in range(len(q_fourier_coef)):
                q_to_track.append(
                    FourierSeries().fit_func_by_fourier_series_with_real_coeffs(
                        np.linspace(0, 1, n_shooting + 1),
                        q_fourier_coef[i],
                    )[np.newaxis, :]
                )

            for j in range(len(q_to_track)):
                objective_functions.add(
                    ObjectiveFcn.Lagrange.TRACK_STATE,
                    key="q",
                    weight=100,
                    target=q_to_track[j],
                    node=Node.ALL,
                    quadratic=True,
                )

        if objective["minimize_muscle_fatigue"]:
            objective_functions.add(
                CustomObjective.minimize_overall_muscle_fatigue,
                custom_type=ObjectiveFcn.Mayer,
                node=Node.END,
                quadratic=True,
                weight=1,
                phase=0,
            )

        if objective["minimize_muscle_force"]:
            objective_functions.add(
                CustomObjective.minimize_overall_muscle_force_production,
                custom_type=ObjectiveFcn.Lagrange,
                node=Node.ALL,
                quadratic=True,
                weight=1,
                phase=0,
            )

        if objective["minimize_residual_torque"]:
            objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
                key="tau",
                weight=10000,
                quadratic=True,
                phase=0,
            )

        return objective_functions

    @staticmethod
    def _sanity_check_msk_inputs(
        model,
        msk_info,
        objective,
    ):
        if msk_info["bound_type"]:
            if not isinstance(msk_info["bound_type"], str) or msk_info["bound_type"] not in [
                "start",
                "end",
                "start_end",
            ]:
                raise ValueError("bound_type should be a string and should be equal to start, end or start_end")
            if not isinstance(msk_info["bound_data"], list):
                raise TypeError("bound_data should be a list")
            if msk_info["bound_type"] == "start_end":
                if (
                    len(msk_info["bound_data"]) != 2
                    or not isinstance(msk_info["bound_data"][0], list)
                    or not isinstance(msk_info["bound_data"][1], list)
                ):
                    raise TypeError("bound_data should be a list of two list")
                if len(msk_info["bound_data"][0]) != model.nb_q or len(msk_info["bound_data"][1]) != model.nb_q:
                    raise ValueError(f"bound_data should be a list of {model.nb_q} elements")
                for i in range(len(msk_info["bound_data"][0])):
                    if not isinstance(msk_info["bound_data"][0][i], int | float) or not isinstance(
                        msk_info["bound_data"][1][i], int | float
                    ):
                        raise TypeError(
                            f"bound data index {i}: {msk_info['bound_data'][0][i]} and {msk_info['bound_data'][1][i]} should be an int or float"
                        )
            if msk_info["bound_type"] == "start" or msk_info["bound_type"] == "end":
                if len(msk_info["bound_data"]) != model.nb_q:
                    raise ValueError(f"bound_data should be a list of {model.nb_q} element")
                for i in range(len(msk_info["bound_data"])):
                    if not isinstance(msk_info["bound_data"][i], int | float):
                        raise TypeError(f"bound data index {i}: {msk_info['bound_data'][i]} should be an int or float")

        if objective["force_tracking"]:
            if isinstance(objective["force_tracking"], list):
                if len(objective["force_tracking"]) != 2:
                    raise ValueError("force_tracking must of size 2")
                if not isinstance(objective["force_tracking"][0], np.ndarray):
                    raise TypeError(f"force_tracking index 0: {objective['force_tracking'][0]} must be np.ndarray type")
                if not isinstance(objective["force_tracking"][1], list):
                    raise TypeError(f"force_tracking index 1: {objective['force_tracking'][1]} must be list type")
                if len(objective["force_tracking"][1]) != len(model.muscles_dynamics_model):
                    raise ValueError(
                        "force_tracking index 1 list must have the same size as the number of muscles in model.muscles_dynamics_model"
                    )
                for i in range(len(objective["force_tracking"][1])):
                    if len(objective["force_tracking"][0]) != len(objective["force_tracking"][1][i]):
                        raise ValueError("force_tracking time and force argument must be the same length")
            else:
                raise TypeError(f"force_tracking: {objective['force_tracking']} must be list type")

        if objective["end_node_tracking"]:
            if not isinstance(objective["end_node_tracking"], list):
                raise TypeError(f"force_tracking: {objective['end_node_tracking']} must be list type")
            if len(objective["end_node_tracking"]) != len(model.muscles_dynamics_model):
                raise ValueError(
                    "end_node_tracking list must have the same size as the number of muscles in fes_muscle_models"
                )
            for i in range(len(objective["end_node_tracking"])):
                if not isinstance(objective["end_node_tracking"][i], int | float):
                    raise TypeError(
                        f"end_node_tracking index {i}: {objective['end_node_tracking'][i]} must be int or float type"
                    )

        if objective["q_tracking"]:
            if not isinstance(objective["q_tracking"], list) and len(objective["q_tracking"]) != 2:
                raise TypeError("q_tracking should be a list of size 2")
            if not isinstance(objective["q_tracking"][0], list | np.ndarray):
                raise ValueError("q_tracking[0] should be a list or array type")
            if len(objective["q_tracking"][1]) != model.nb_q:
                raise ValueError("q_tracking[1] should have the same size as the number of generalized coordinates")
            for i in range(model.nb_q):
                if len(objective["q_tracking"][0]) != len(objective["q_tracking"][1][i]):
                    raise ValueError("q_tracking[0] and q_tracking[1] should have the same size")

        list_to_check = [
            msk_info["with_residual_torque"],
            objective["minimize_muscle_fatigue"],
            objective["minimize_muscle_force"],
        ]

        list_to_check_name = [
            "with_residual_torque",
            "minimize_muscle_fatigue",
            "minimize_muscle_force",
        ]

        for i in range(len(list_to_check)):
            if list_to_check[i]:
                if not isinstance(list_to_check[i], bool):
                    raise TypeError(f"{list_to_check_name[i]} should be a boolean")
