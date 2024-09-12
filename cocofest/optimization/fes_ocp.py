import numpy as np

from bioptim import (
    BoundsList,
    ConstraintList,
    ControlType,
    DynamicsList,
    InitialGuessList,
    InterpolationType,
    Node,
    Objective,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    ParameterList,
    ParameterObjectiveList,
    PhaseDynamics,
    VariableScaling,
)

from ..custom_objectives import CustomObjective
from ..custom_constraints import CustomConstraint
from ..fourier_approx import FourierSeries

from ..models.fes_model import FesModel
from ..models.ding2007 import DingModelPulseDurationFrequency
from ..models.ding2007_with_fatigue import DingModelPulseDurationFrequencyWithFatigue
from ..models.ding2003 import DingModelFrequency
from ..models.hmed2018 import DingModelIntensityFrequency
from ..models.hmed2018_with_fatigue import DingModelIntensityFrequencyWithFatigue


class OcpFes:
    """
    The main class to define an ocp. This class prepares the full program and gives all
    the needed parameters to solve a functional electrical stimulation ocp

    Methods
    -------
    from_frequency_and_final_time(self, frequency: int | float, final_time: float, round_down: bool)
        Calculates the number of stim (phases) for the ocp from frequency and final time
    from_frequency_and_n_stim(self, frequency: int | float, n_stim: int)
        Calculates the final ocp time from frequency and stimulation number
    """

    @staticmethod
    def prepare_ocp(
        model: FesModel = None,
        n_stim: int = None,
        n_shooting: int = None,
        final_time: int | float = None,
        pulse_event: dict = None,
        pulse_duration: dict = None,
        pulse_intensity: dict = None,
        objective: dict = None,
        use_sx: bool = True,
        ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
        n_threads: int = 1,
        stim_time: list = None,
    ):
        """
        Prepares the Optimal Control Program (OCP) to be solved.

        Parameters
        ----------
        model : FesModel
            The model type used for the OCP.
        n_stim : int
            Number of stimulations that will occur during the OCP, also referred to as phases.
        n_shooting : int
            Number of shooting points for each individual phase.
        final_time : int | float
            The final time of the OCP.
        pulse_event : dict
            Dictionary containing parameters related to the appearance of the pulse.
        pulse_duration : dict
            Dictionary containing parameters related to the duration of the pulse.
            Optional if not using DingModelPulseDurationFrequency or DingModelPulseDurationFrequencyWithFatigue.
        pulse_intensity : dict
            Dictionary containing parameters related to the intensity of the pulse.
            Optional if not using DingModelIntensityFrequency or DingModelIntensityFrequencyWithFatigue.
        objective : dict
            Dictionary containing parameters related to the optimization objective.
        use_sx : bool
            The nature of the CasADi variables. MX are used if False.
        ode_solver : OdeSolver
            The ODE solver to use.
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

        fixed_pulse_intensity = pulse_intensity["fixed"]
        pulse_intensity_min = pulse_intensity["min"]
        pulse_intensity_max = pulse_intensity["max"]
        pulse_intensity_bimapping = pulse_intensity["bimapping"]

        force_tracking = objective["force_tracking"]
        end_node_tracking = objective["end_node_tracking"]
        custom_objective = objective["custom"]

        OcpFes._sanity_check(
            model=model,
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
            force_tracking=force_tracking,
            end_node_tracking=end_node_tracking,
            custom_objective=custom_objective,
            use_sx=use_sx,
            ode_solver=ode_solver,
            n_threads=n_threads,
        )

        OcpFes._sanity_check_frequency(n_stim=n_stim, final_time=final_time, frequency=frequency, round_down=round_down)

        n_stim, final_time = OcpFes._build_phase_parameter(
            n_stim=n_stim, final_time=final_time, frequency=frequency, pulse_mode=pulse_mode, round_down=round_down
        )

        force_fourier_coefficient = (
            None if force_tracking is None else OcpFes._build_fourier_coefficient(force_tracking)
        )

        parameters, parameters_bounds, parameters_init, parameter_objectives, constraints = OcpFes._build_parameters(
            model=model,
            n_stim=n_stim,
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
            use_sx=use_sx,
            stim_time=stim_time,
        )

        if len(constraints) == 0 and len(parameters) == 0:
            raise ValueError(
                "This is not an optimal control problem,"
                " add parameter to optimize or use the IvpFes method to build your problem"
            )

        dynamics = OcpFes._declare_dynamics(model)
        x_bounds, x_init = OcpFes._set_bounds(model)
        objective_functions = OcpFes._set_objective(
            n_stim, n_shooting, force_fourier_coefficient, end_node_tracking, custom_objective, time_min, time_max
        )

        return OptimalControlProgram(
            bio_model=[model],
            dynamics=dynamics,
            n_shooting=n_shooting,
            phase_time=[final_time],
            objective_functions=objective_functions,
            x_init=x_init,
            x_bounds=x_bounds,
            constraints=constraints,
            parameters=parameters,
            parameter_bounds=parameters_bounds,
            parameter_init=parameters_init,
            parameter_objectives=parameter_objectives,
            control_type=ControlType.CONSTANT,
            use_sx=use_sx,
            ode_solver=ode_solver,
            n_threads=n_threads,
        )

    @staticmethod
    def _fill_dict(pulse_event, pulse_duration, pulse_intensity, objective):
        """
        This method fills the provided dictionaries with default values if they are not set.

        Parameters
        ----------
        pulse_event : dict
            Dictionary containing parameters related to the appearance of the pulse.
            Expected keys are 'min', 'max', 'bimapping', 'frequency', 'round_down', and 'pulse_mode'.

        pulse_duration : dict
            Dictionary containing parameters related to the duration of the pulse.
            Expected keys are 'fixed', 'min', 'max', and 'bimapping'.

        pulse_intensity : dict
            Dictionary containing parameters related to the intensity of the pulse.
            Expected keys are 'fixed', 'min', 'max', and 'bimapping'.

        objective : dict
            Dictionary containing parameters related to the objective of the optimization.
            Expected keys are 'force_tracking', 'end_node_tracking', and 'custom'.

        Returns
        -------
        Returns four dictionaries: pulse_event, pulse_duration, pulse_intensity, and objective.
        Each dictionary is filled with default values for any keys that were not initially set.
        """

        default_pulse_event = {
            "min": None,
            "max": None,
            "bimapping": False,
            "frequency": None,
            "round_down": False,
            "pulse_mode": "single",
        }

        default_pulse_duration = {
            "fixed": None,
            "min": None,
            "max": None,
            "bimapping": False,
        }

        default_pulse_intensity = {
            "fixed": None,
            "min": None,
            "max": None,
            "bimapping": False,
        }

        default_objective = {
            "force_tracking": None,
            "end_node_tracking": None,
            "cycling": None,
            "custom": None,
        }
        dict_list = [pulse_event, pulse_duration, pulse_intensity, objective]
        default_dict_list = [
            default_pulse_event,
            default_pulse_duration,
            default_pulse_intensity,
            default_objective,
        ]

        for i in range(len(dict_list)):
            if dict_list[i] is None:
                dict_list[i] = {}

        for i in range(len(dict_list)):
            for key in default_dict_list[i]:
                if key not in dict_list[i]:
                    dict_list[i][key] = default_dict_list[i][key]

        return dict_list[0], dict_list[1], dict_list[2], dict_list[3]

    @staticmethod
    def _sanity_check(
        model=None,
        n_stim=None,
        n_shooting=None,
        final_time=None,
        pulse_mode=None,
        frequency=None,
        time_min=None,
        time_max=None,
        time_bimapping=None,
        fixed_pulse_duration=None,
        pulse_duration_min=None,
        pulse_duration_max=None,
        pulse_duration_bimapping=None,
        fixed_pulse_intensity=None,
        pulse_intensity_min=None,
        pulse_intensity_max=None,
        pulse_intensity_bimapping=None,
        force_tracking=None,
        end_node_tracking=None,
        custom_objective=None,
        use_sx=None,
        ode_solver=None,
        n_threads=None,
    ):
        if not isinstance(model, FesModel):
            raise TypeError(
                f"The current model type used is {type(model)}, it must be a FesModel type."
                f"Current available models are: DingModelFrequency, DingModelFrequencyWithFatigue,"
                f"DingModelPulseDurationFrequency, DingModelPulseDurationFrequencyWithFatigue,"
                f"DingModelIntensityFrequency, DingModelIntensityFrequencyWithFatigue"
            )

        if n_stim:
            if isinstance(n_stim, int):
                if n_stim <= 0:
                    raise ValueError("n_stim must be positive")
            else:
                raise TypeError("n_stim must be int type")

        if n_shooting:
            if isinstance(n_shooting, int):
                if n_shooting <= 0:
                    raise ValueError("n_shooting must be positive")
            else:
                raise TypeError("n_shooting must be int type")

        if final_time:
            if isinstance(final_time, int | float):
                if final_time <= 0:
                    raise ValueError("final_time must be positive")
            else:
                raise TypeError("final_time must be int or float type")

        if pulse_mode:
            if pulse_mode != "single":
                raise NotImplementedError(f"Pulse mode '{pulse_mode}' is not yet implemented")

        if frequency:
            if isinstance(frequency, int | float):
                if frequency <= 0:
                    raise ValueError("frequency must be positive")
            else:
                raise TypeError("frequency must be int or float type")

        if [time_min, time_max].count(None) == 1:
            raise ValueError("time_min and time_max must be both entered or none of them in order to work")

        if time_bimapping:
            if not isinstance(time_bimapping, bool):
                raise TypeError("time_bimapping must be bool type")

        if isinstance(model, DingModelPulseDurationFrequency | DingModelPulseDurationFrequencyWithFatigue):
            if fixed_pulse_duration is None and [pulse_duration_min, pulse_duration_max].count(None) != 0:
                raise ValueError("pulse duration or pulse duration min max bounds need to be set for this model")
            if all([fixed_pulse_duration, pulse_duration_min, pulse_duration_max]):
                raise ValueError("Either pulse duration or pulse duration min max bounds need to be set for this model")

            minimum_pulse_duration = (
                0 if model.pd0 is None else model.pd0
            )  # Set it to 0 if used for the identification process

            if fixed_pulse_duration is not None:
                if isinstance(fixed_pulse_duration, int | float):
                    if fixed_pulse_duration < minimum_pulse_duration:
                        raise ValueError(
                            f"The pulse duration set ({fixed_pulse_duration})"
                            f" is lower than minimum duration required."
                            f" Set a value above {minimum_pulse_duration} seconds "
                        )
                elif isinstance(fixed_pulse_duration, list):
                    if not all(isinstance(x, int | float) for x in fixed_pulse_duration):
                        raise TypeError("pulse_duration must be int or float type")
                    if not all(x >= minimum_pulse_duration for x in fixed_pulse_duration):
                        raise ValueError(
                            f"The pulse duration set ({fixed_pulse_duration})"
                            f" is lower than minimum duration required."
                            f" Set a value above {minimum_pulse_duration} seconds "
                        )
                else:
                    raise TypeError("Wrong pulse_duration type, only int or float accepted")

            elif pulse_duration_min is not None and pulse_duration_max is not None:
                if not isinstance(pulse_duration_min, int | float) or not isinstance(pulse_duration_max, int | float):
                    raise TypeError("pulse_duration_min and pulse_duration_max must be int or float type")
                if pulse_duration_max < pulse_duration_min:
                    raise ValueError("The set minimum pulse duration is higher than maximum pulse duration.")
                if pulse_duration_min < minimum_pulse_duration:
                    raise ValueError(
                        f"The pulse duration set ({pulse_duration_min})"
                        f" is lower than minimum duration required."
                        f" Set a value above {minimum_pulse_duration} seconds "
                    )

            if not isinstance(pulse_duration_bimapping, None | bool):
                raise NotImplementedError("If added, pulse duration parameter mapping must be a bool type")

        if isinstance(model, DingModelIntensityFrequency | DingModelIntensityFrequencyWithFatigue):
            if fixed_pulse_intensity is None and [pulse_intensity_min, pulse_intensity_max].count(None) != 0:
                raise ValueError("Pulse intensity or pulse intensity min max bounds need to be set for this model")
            if all([fixed_pulse_intensity, pulse_intensity_min, pulse_intensity_max]):
                raise ValueError(
                    "Either pulse intensity or pulse intensity min max bounds need to be set for this model"
                )

            check_for_none_type = [model.cr, model.bs, model.Is]
            minimum_pulse_intensity = (
                0 if None in check_for_none_type else model.min_pulse_intensity()
            )  # Set it to 0 if used for the identification process

            if fixed_pulse_intensity is not None:
                if isinstance(fixed_pulse_intensity, int | float):
                    if fixed_pulse_intensity < minimum_pulse_intensity:
                        raise ValueError(
                            f"The pulse intensity set ({fixed_pulse_intensity})"
                            f" is lower than minimum intensity required."
                            f" Set a value above {minimum_pulse_intensity} mA "
                        )
                elif isinstance(fixed_pulse_intensity, list):
                    if not all(isinstance(x, int | float) for x in fixed_pulse_intensity):
                        raise TypeError("pulse_intensity must be int or float type")
                    if not all(x >= minimum_pulse_intensity for x in fixed_pulse_intensity):
                        raise ValueError(
                            f"The pulse intensity set ({fixed_pulse_intensity})"
                            f" is lower than minimum intensity required."
                            f" Set a value above {minimum_pulse_intensity} seconds "
                        )
                else:
                    raise TypeError("pulse_intensity must be int or float type")

            elif pulse_intensity_min is not None and pulse_intensity_max is not None:
                if not isinstance(pulse_intensity_min, int | float) or not isinstance(pulse_intensity_max, int | float):
                    raise TypeError("pulse_intensity_min and pulse_intensity_max must be int or float type")
                if pulse_intensity_max < pulse_intensity_min:
                    raise ValueError("The set minimum pulse intensity is higher than maximum pulse intensity.")
                if pulse_intensity_min < minimum_pulse_intensity:
                    raise ValueError(
                        f"The pulse intensity set ({pulse_intensity_min})"
                        f" is lower than minimum intensity required."
                        f" Set a value above {minimum_pulse_intensity} mA "
                    )

            if not isinstance(pulse_intensity_bimapping, None | bool):
                raise NotImplementedError("If added, pulse intensity parameter mapping must be a bool type")

        if force_tracking is not None:
            if isinstance(force_tracking, list):
                if isinstance(force_tracking[0], np.ndarray) and isinstance(force_tracking[1], np.ndarray):
                    if len(force_tracking[0]) != len(force_tracking[1]) or len(force_tracking) != 2:
                        raise ValueError(
                            "force_tracking time and force argument must be same length and force_tracking "
                            "list size 2"
                        )
                else:
                    raise TypeError("force_tracking argument must be np.ndarray type")
            else:
                raise TypeError("force_tracking must be list type")

        if end_node_tracking:
            if not isinstance(end_node_tracking, int | float):
                raise TypeError("end_node_tracking must be int or float type")

        if custom_objective:
            if not isinstance(custom_objective, ObjectiveList):
                raise TypeError("custom_objective must be a ObjectiveList type")
            if not all(isinstance(x, Objective) for x in custom_objective[0]):
                raise TypeError("All elements in ObjectiveList must be an Objective type")

        if not isinstance(ode_solver, (OdeSolver.RK1, OdeSolver.RK2, OdeSolver.RK4, OdeSolver.COLLOCATION)):
            raise TypeError("ode_solver must be a OdeSolver type")

        if not isinstance(use_sx, bool):
            raise TypeError("use_sx must be a bool type")

        if not isinstance(n_threads, int):
            raise TypeError("n_thread must be a int type")

    @staticmethod
    def _sanity_check_frequency(n_stim, final_time, frequency, round_down):
        if [n_stim, final_time, frequency].count(None) == 2:
            raise ValueError("At least two variable must be set from n_stim, final_time or frequency")

        if n_stim and final_time and frequency:
            if n_stim != final_time * frequency:
                raise ValueError(
                    "Can not satisfy n_stim equal to final_time * frequency with the given parameters."
                    "Consider setting only two of the three parameters"
                )

        if round_down:
            if not isinstance(round_down, bool):
                raise TypeError("round_down must be bool type")

    @staticmethod
    def _build_fourier_coefficient(force_tracking):
        return FourierSeries().compute_real_fourier_coeffs(force_tracking[0], force_tracking[1], 50)

    @staticmethod
    def _build_phase_time(final_time, n_stim, pulse_mode, time_min, time_max):
        final_time_phase = None
        if time_min is None and time_max is None:
            if pulse_mode == "single":
                step = final_time / n_stim
                final_time_phase = (step,)
                for i in range(n_stim - 1):
                    final_time_phase = final_time_phase + (step,)
        else:
            final_time_phase = [final_time / n_stim] * n_stim

        return final_time_phase

    @staticmethod
    def _build_parameters(
        model,
        n_stim,
        time_min,
        time_max,
        time_bimapping,
        fixed_pulse_duration,
        pulse_duration_min,
        pulse_duration_max,
        pulse_duration_bimapping,
        fixed_pulse_intensity,
        pulse_intensity_min,
        pulse_intensity_max,
        pulse_intensity_bimapping,
        use_sx,
        stim_time,
    ):
        parameters = ParameterList(use_sx=use_sx)
        parameters_bounds = BoundsList()
        parameters_init = InitialGuessList()
        parameter_objectives = ParameterObjectiveList()
        constraints = ConstraintList()

        parameters.add(
            name="pulse_apparition_time",
            function=DingModelFrequency.set_pulse_apparition_time,
            size=n_stim,
            scaling=VariableScaling("pulse_apparition_time", [1] * n_stim),
        )

        if time_min and time_max:
            time_min_list = np.array([0] + list(np.cumsum([time_min]*(n_stim-1))))
            time_max_list = np.array([0] + list(np.cumsum([time_max]*(n_stim-1))))
            parameters_init["pulse_apparition_time"] = np.array([(time_max_list[i] + time_min_list[i]) / 2 for i in range(n_stim)])
        else:
            time_min_list = stim_time
            time_max_list = stim_time
            parameters_init["pulse_apparition_time"] = np.array(stim_time)

        parameters_bounds.add(
            "pulse_apparition_time",
            min_bound=time_min_list,
            max_bound=time_max_list,
            interpolation=InterpolationType.CONSTANT,
        )

        if time_bimapping and time_min and time_max:
            for i in range(n_stim):
                constraints.add(CustomConstraint.equal_to_first_pulse_interval_time, node=Node.START, target=0, phase=i)
                #TODO : Modify this for single phase

        if isinstance(model, DingModelPulseDurationFrequency):
            if fixed_pulse_duration:
                parameters.add(
                    name="pulse_duration",
                    function=DingModelPulseDurationFrequency.set_impulse_duration,
                    size=n_stim,
                    scaling=VariableScaling("pulse_duration", [1] * n_stim),
                )
                if isinstance(fixed_pulse_duration, list):
                    parameters_bounds.add(
                        "pulse_duration",
                        min_bound=np.array(fixed_pulse_duration),
                        max_bound=np.array(fixed_pulse_duration),
                        interpolation=InterpolationType.CONSTANT,
                    )
                    parameters_init.add(key="pulse_duration", initial_guess=np.array(fixed_pulse_duration))
                else:
                    parameters_bounds.add(
                        "pulse_duration",
                        min_bound=np.array([fixed_pulse_duration] * n_stim),
                        max_bound=np.array([fixed_pulse_duration] * n_stim),
                        interpolation=InterpolationType.CONSTANT,
                    )
                    parameters_init["pulse_duration"] = np.array([fixed_pulse_duration] * n_stim)

            elif pulse_duration_min is not None and pulse_duration_max is not None:
                parameters_bounds.add(
                    "pulse_duration",
                    min_bound=pulse_duration_min,
                    max_bound=pulse_duration_max,
                    interpolation=InterpolationType.CONSTANT,
                )
                parameters_init["pulse_duration"] = np.array([(pulse_duration_min + pulse_duration_max) / 2 for i in range(n_stim)])
                parameters.add(
                    name="pulse_duration",
                    function=DingModelPulseDurationFrequency.set_impulse_duration,
                    size=n_stim,
                    scaling=VariableScaling("pulse_duration", [1] * n_stim),
                )

            if pulse_duration_bimapping is True:
                for i in range(1, n_stim):
                    constraints.add(CustomConstraint.equal_to_first_pulse_duration, node=Node.START, target=0, phase=i)

        if isinstance(model, DingModelIntensityFrequency):
            if fixed_pulse_intensity:
                parameters.add(
                    name="pulse_intensity",
                    function=DingModelIntensityFrequency.set_impulse_intensity,
                    size=n_stim,
                    scaling=VariableScaling("pulse_intensity", [1] * n_stim),
                )
                if isinstance(fixed_pulse_intensity, list):
                    parameters_bounds.add(
                        "pulse_intensity",
                        min_bound=np.array(fixed_pulse_intensity),
                        max_bound=np.array(fixed_pulse_intensity),
                        interpolation=InterpolationType.CONSTANT,
                    )
                    parameters_init.add(key="pulse_intensity", initial_guess=np.array(fixed_pulse_intensity))
                else:
                    parameters_bounds.add(
                        "pulse_intensity",
                        min_bound=np.array([fixed_pulse_intensity] * n_stim),
                        max_bound=np.array([fixed_pulse_intensity] * n_stim),
                        interpolation=InterpolationType.CONSTANT,
                    )
                    parameters_init["pulse_intensity"] = np.array([fixed_pulse_intensity] * n_stim)

            elif pulse_intensity_min is not None and pulse_intensity_max is not None:
                parameters_bounds.add(
                    "pulse_intensity",
                    min_bound=[pulse_intensity_min],
                    max_bound=[pulse_intensity_max],
                    interpolation=InterpolationType.CONSTANT,
                )
                intensity_avg = (pulse_intensity_min + pulse_intensity_max) / 2
                parameters_init["pulse_intensity"] = np.array([intensity_avg] * n_stim)
                parameters.add(
                    name="pulse_intensity",
                    function=DingModelIntensityFrequency.set_impulse_intensity,
                    size=n_stim,
                    scaling=VariableScaling("pulse_intensity", [1] * n_stim),
                )

            if pulse_intensity_bimapping is True:
                for i in range(1, n_stim):
                    constraints.add(CustomConstraint.equal_to_first_pulse_intensity, node=Node.START, target=0, phase=i)

        return parameters, parameters_bounds, parameters_init, parameter_objectives, constraints

    @staticmethod
    def _declare_dynamics(model):
        dynamics = DynamicsList()
        dynamics.add(
            model.declare_ding_variables,
            dynamic_function=model.dynamics,
            expand_dynamics=True,
            phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        )

        return dynamics

    @staticmethod
    def _set_bounds(model):
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
        variable_bound_list = model.name_dof
        starting_bounds, min_bounds, max_bounds = (
            model.standard_rest_values(),
            model.standard_rest_values(),
            model.standard_rest_values(),
        )

        for i in range(len(variable_bound_list)):
            if variable_bound_list[i] == "Cn" or variable_bound_list[i] == "F":
                max_bounds[i] = 1000
            elif variable_bound_list[i] == "Tau1" or variable_bound_list[i] == "Km":
                max_bounds[i] = 1
            elif variable_bound_list[i] == "A":
                min_bounds[i] = 0

        starting_bounds_min = np.concatenate((starting_bounds, min_bounds, min_bounds), axis=1)
        starting_bounds_max = np.concatenate((starting_bounds, max_bounds, max_bounds), axis=1)

        for j in range(len(variable_bound_list)):
            x_bounds.add(
                variable_bound_list[j],
                min_bound=np.array([starting_bounds_min[j]]),
                max_bound=np.array([starting_bounds_max[j]]),
                interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
            )

        x_init = InitialGuessList()
        for j in range(len(variable_bound_list)):
            x_init.add(variable_bound_list[j], model.standard_rest_values()[j])

        return x_bounds, x_init

    @staticmethod
    def _set_objective(
        n_stim, n_shooting, force_fourier_coefficient, end_node_tracking, custom_objective, time_min, time_max
    ):
        # Creates the objective for our problem
        objective_functions = ObjectiveList()
        if custom_objective:
            for i in range(len(custom_objective)):
                objective_functions.add(custom_objective[0][i])

        if force_fourier_coefficient is not None:

            force_to_track = FourierSeries().fit_func_by_fourier_series_with_real_coeffs(
            np.linspace(0, 1, n_shooting + 1),
            force_fourier_coefficient,
        )[np.newaxis, :]

            objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="F", weight=100, target=force_to_track, node=Node.ALL, quadratic=True)

        if end_node_tracking:
            if isinstance(end_node_tracking, int | float):
                objective_functions.add(
                    ObjectiveFcn.Mayer.MINIMIZE_STATE,
                    node=Node.END,
                    key="F",
                    quadratic=True,
                    weight=1,
                    target=end_node_tracking,
                )

        if time_min and time_max:
            objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.001, quadratic=True)

        return objective_functions

    @staticmethod
    def _build_phase_parameter(n_stim, final_time, frequency=None, pulse_mode="single", round_down=False):
        pulse_mode_multiplier = 1 if pulse_mode == "single" else 2 if pulse_mode == "doublet" else 3
        if n_stim and frequency:
            final_time = n_stim / frequency / pulse_mode_multiplier

        if final_time and frequency:
            n_stim = final_time * frequency * pulse_mode_multiplier
            if round_down or n_stim.is_integer():
                n_stim = int(n_stim)
            else:
                raise ValueError(
                    "The number of stimulation needs to be integer within the final time t, set round down"
                    "to True or set final_time * frequency to make the result a integer."
                )

        return n_stim, final_time
