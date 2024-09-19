import numpy as np

from bioptim import (
    BoundsList,
    ConstraintList,
    ControlType,
    InitialGuessList,
    InterpolationType,
    Objective,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    ParameterList,
    PhaseTransitionFcn,
    PhaseTransitionList,
    VariableScaling,
    Node,
)

from ..custom_objectives import CustomObjective
from ..models.fes_model import FesModel
from ..models.ding2007 import DingModelPulseDurationFrequency
from ..models.ding2007_with_fatigue import DingModelPulseDurationFrequencyWithFatigue
from ..models.ding2003 import DingModelFrequency
from ..models.ding2003_with_fatigue import DingModelFrequencyWithFatigue
from ..models.hmed2018 import DingModelIntensityFrequency
from ..models.hmed2018_with_fatigue import DingModelIntensityFrequencyWithFatigue
from ..optimization.fes_ocp import OcpFes


class OcpFesId(OcpFes):
    def __init__(self):
        super(OcpFesId, self).__init__()

    @staticmethod
    def prepare_ocp(
        model: FesModel = None,
        n_shooting: int = None,
        final_time: float | int = None,
        stim_time: list = None,
        pulse_duration: int | float | list = None,
        pulse_intensity: int | float | list = None,
        objective: dict = None,
        key_parameter_to_identify: list = None,
        additional_key_settings: dict = None,
        custom_objective: list[Objective] = None,
        discontinuity_in_ocp: list = None,
        use_sx: bool = True,
        ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
        n_threads: int = 1,
        **kwargs,
    ):
        """
        The main class to define an ocp. This class prepares the full program and gives all
        the needed parameters to solve a functional electrical stimulation ocp

        Attributes
        ----------
        model:  FesModel
            The model used to solve the ocp
        final_time: float, int
            The final time of each phase, it corresponds to the stimulation apparition time
        n_shooting: list[int],
            The number of shooting points for each phase
        pulse_duration: int | float | list[int] | list[float],
            The duration of the stimulation
        pulse_intensity: int | float | list[int] | list[float],
            The intensity of the stimulation
        objective: dict,
            The objective to minimize
        discontinuity_in_ocp: list[int],
            The phases where the continuity is not respected
        ode_solver: OdeSolver
            The ode solver to use
        use_sx: bool
            The nature of the casadi variables. MX are used if False.
        n_thread: int
            The number of thread to use while solving (multi-threading if > 1)
        """
        (
            pulse_event,
            pulse_duration,
            pulse_intensity,
            temp_objective,
        ) = OcpFes._fill_dict({}, pulse_duration, pulse_intensity, {})

        OcpFesId._sanity_check(
            model=model,
            n_shooting=n_shooting,
            final_time=final_time,
            pulse_event=pulse_event,
            pulse_duration=pulse_duration,
            pulse_intensity=pulse_intensity,
            objective=temp_objective,
            use_sx=use_sx,
            ode_solver=ode_solver,
            n_threads=n_threads,
        )

        OcpFesId._sanity_check_id(
            model=model,
            n_shooting=n_shooting,
            final_time=final_time,
            objective=objective,
            pulse_duration=pulse_duration,
            pulse_intensity=pulse_intensity,
        )

        n_stim = len(stim_time)

        constraints = ConstraintList()
        parameters, parameters_bounds, parameters_init = OcpFesId._set_parameters(
            n_stim=n_stim,
            stim_apparition_time=stim_time,
            parameter_to_identify=key_parameter_to_identify,
            parameter_setting=additional_key_settings,
            pulse_duration=pulse_duration["fixed"],
            pulse_intensity=pulse_intensity["fixed"],
            use_sx=use_sx,
        )
        dynamics = OcpFesId._declare_dynamics(model=model)
        x_bounds, x_init = OcpFesId._set_bounds(
            model=model,
            n_stim=n_stim,
            n_shooting=n_shooting,
            force_tracking=objective["force_tracking"],
            discontinuity_in_ocp=discontinuity_in_ocp,
        )
        objective_functions = OcpFesId._set_objective(
            model=model,
            n_stim=n_stim,
            n_shooting=n_shooting,
            force_tracking=objective["force_tracking"],
            custom_objective=custom_objective,
        )
        # phase_transitions = OcpFesId._set_phase_transition(discontinuity_in_ocp)

        return OptimalControlProgram(
            bio_model=[model],
            dynamics=dynamics,
            n_shooting=n_shooting,
            phase_time=final_time,
            x_init=x_init,
            x_bounds=x_bounds,
            objective_functions=objective_functions,
            constraints=constraints,
            ode_solver=ode_solver,
            control_type=ControlType.CONSTANT,
            use_sx=use_sx,
            parameters=parameters,
            parameter_bounds=parameters_bounds,
            parameter_init=parameters_init,
            # phase_transitions=phase_transitions,
            n_threads=n_threads,
        )

    @staticmethod
    def _sanity_check_id(
        model=None,
        n_shooting=None,
        final_time=None,
        objective=None,
        pulse_duration=None,
        pulse_intensity=None,
    ):
        if not isinstance(n_shooting, int):
            raise TypeError(
                f"n_shooting must be list type,"
                f" currently n_shooting is {type(n_shooting)}) type."
            )

        if not isinstance(final_time, int | float):
            raise TypeError(f"final_time must be int or float type.")

        if not isinstance(objective["force_tracking"], list):
            raise TypeError(
                f"force_tracking must be list type,"
                f" currently force_tracking is {type(objective['force_tracking'])}) type."
            )
        else:
            if not all(
                isinstance(val, int | float) for val in objective["force_tracking"]
            ):
                raise TypeError(f"force_tracking must be list of int or float type.")

        if isinstance(model, DingModelPulseDurationFrequency):
            if not isinstance(pulse_duration, list):
                raise TypeError(
                    f"pulse_duration must be list type,"
                    f" currently pulse_duration is {type(pulse_duration)}) type."
                )

        if isinstance(model, DingModelIntensityFrequency):
            if isinstance(pulse_intensity, dict):
                if not isinstance(pulse_intensity["fixed"], int | float | list):
                    raise ValueError(
                        f"fixed pulse_intensity must be a int, float or list type."
                    )

            else:
                raise TypeError(
                    f"pulse_intensity must be dict type,"
                    f" currently pulse_intensity is {type(pulse_intensity)}) type."
                )

    @staticmethod
    def _set_bounds(
        model=None,
        n_stim=None,
        n_shooting=None,
        force_tracking=None,
        discontinuity_in_ocp=None,
    ):
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
            if variable_bound_list[i] == "Cn":
                max_bounds[i] = 10
            elif variable_bound_list[i] == "F":
                max_bounds[i] = 500
            elif variable_bound_list[i] == "Tau1" or variable_bound_list[i] == "Km":
                max_bounds[i] = 1
            elif variable_bound_list[i] == "A":
                min_bounds[i] = 0

        starting_bounds_min = np.concatenate(
            (starting_bounds, min_bounds, min_bounds), axis=1
        )
        starting_bounds_max = np.concatenate(
            (starting_bounds, max_bounds, max_bounds), axis=1
        )

        for j in range(len(variable_bound_list)):
            x_bounds.add(
                variable_bound_list[j],
                min_bound=np.array([starting_bounds_min[j]]),
                max_bound=np.array([starting_bounds_max[j]]),
                phase=0,
                interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
            )

        x_init = InitialGuessList()

        x_init.add(
            "F",
            np.array([force_tracking]),
            phase=0,
            interpolation=InterpolationType.EACH_FRAME,
        )
        x_init.add("Cn", [0], phase=0, interpolation=InterpolationType.CONSTANT)
        if model._with_fatigue:
            for j in range(len(variable_bound_list)):
                if variable_bound_list[j] == "F" or variable_bound_list[j] == "Cn":
                    pass
                else:
                    x_init.add(variable_bound_list[j], model.standard_rest_values()[j])

        return x_bounds, x_init

    @staticmethod
    def _set_objective(
        model, n_stim, n_shooting, force_tracking, custom_objective, **kwargs
    ):
        # Creates the objective for our problem (in this case, match a force curve)
        objective_functions = ObjectiveList()

        if force_tracking:
            objective_functions.add(
                ObjectiveFcn.Lagrange.TRACK_STATE,
                key="F",
                weight=1,
                target=np.array(force_tracking)[np.newaxis, :],
                node=Node.ALL,
                quadratic=True,
            )

        if custom_objective:
            for i in range(len(custom_objective)):
                objective_functions.add(custom_objective[i])

        return objective_functions

    @staticmethod
    def _set_parameters(
        n_stim,
        stim_apparition_time,
        parameter_to_identify,
        parameter_setting,
        use_sx,
        pulse_duration=None,
        pulse_intensity=None,
    ):
        parameters = ParameterList(use_sx=use_sx)
        parameters_bounds = BoundsList()
        parameters_init = InitialGuessList()

        parameters.add(
            name="pulse_apparition_time",
            function=DingModelFrequency.set_pulse_apparition_time,
            size=n_stim,
            scaling=VariableScaling("pulse_apparition_time", [1] * n_stim),
        )

        parameters_init["pulse_apparition_time"] = np.array(stim_apparition_time)

        parameters_bounds.add(
            "pulse_apparition_time",
            min_bound=stim_apparition_time,
            max_bound=stim_apparition_time,
            interpolation=InterpolationType.CONSTANT,
        )

        for i in range(len(parameter_to_identify)):
            parameters.add(
                name=parameter_to_identify[i],
                function=parameter_setting[parameter_to_identify[i]]["function"],
                size=1,
                scaling=VariableScaling(
                    parameter_to_identify[i],
                    [parameter_setting[parameter_to_identify[i]]["scaling"]],
                ),
            )
            parameters_bounds.add(
                parameter_to_identify[i],
                min_bound=np.array(
                    [parameter_setting[parameter_to_identify[i]]["min_bound"]]
                ),
                max_bound=np.array(
                    [parameter_setting[parameter_to_identify[i]]["max_bound"]]
                ),
                interpolation=InterpolationType.CONSTANT,
            )
            parameters_init.add(
                key=parameter_to_identify[i],
                initial_guess=np.array(
                    [parameter_setting[parameter_to_identify[i]]["initial_guess"]]
                ),
            )

        if pulse_duration:
            parameters.add(
                name="pulse_duration",
                function=DingModelPulseDurationFrequency.set_impulse_duration,
                size=n_stim,
                scaling=VariableScaling("pulse_duration", [1] * n_stim),
            )
            if isinstance(pulse_duration, list):
                parameters_bounds.add(
                    "pulse_duration",
                    min_bound=np.array(pulse_duration),
                    max_bound=np.array(pulse_duration),
                    interpolation=InterpolationType.CONSTANT,
                )
                parameters_init.add(
                    key="pulse_duration", initial_guess=np.array(pulse_duration)
                )
            else:
                parameters_bounds.add(
                    "pulse_duration",
                    min_bound=np.array([pulse_duration] * n_stim),
                    max_bound=np.array([pulse_duration] * n_stim),
                    interpolation=InterpolationType.CONSTANT,
                )
                parameters_init.add(
                    key="pulse_duration",
                    initial_guess=np.array([pulse_duration] * n_stim),
                )

        if pulse_intensity:
            parameters.add(
                name="pulse_intensity",
                function=DingModelIntensityFrequency.set_impulse_intensity,
                size=n_stim,
                scaling=VariableScaling("pulse_intensity", [1] * n_stim),
            )
            if isinstance(pulse_intensity, list):
                parameters_bounds.add(
                    "pulse_intensity",
                    min_bound=np.array(pulse_intensity),
                    max_bound=np.array(pulse_intensity),
                    interpolation=InterpolationType.CONSTANT,
                )
                parameters_init.add(
                    key="pulse_intensity", initial_guess=np.array(pulse_intensity)
                )
            else:
                parameters_bounds.add(
                    "pulse_intensity",
                    min_bound=np.array([pulse_intensity] * n_stim),
                    max_bound=np.array([pulse_intensity] * n_stim),
                    interpolation=InterpolationType.CONSTANT,
                )
                parameters_init.add(
                    key="pulse_intensity",
                    initial_guess=np.array([pulse_intensity] * n_stim),
                )

        return parameters, parameters_bounds, parameters_init

    @staticmethod
    def _set_phase_transition(discontinuity_in_ocp):
        phase_transitions = PhaseTransitionList()
        if discontinuity_in_ocp:
            for i in range(len(discontinuity_in_ocp)):
                phase_transitions.add(
                    PhaseTransitionFcn.DISCONTINUOUS,
                    phase_pre_idx=discontinuity_in_ocp[i] - 1,
                )
        return phase_transitions
