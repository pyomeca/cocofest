import numpy as np

from bioptim import (
    BoundsList,
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

from ..models.fes_model import FesModel

from ..models.ding2007 import DingModelPulseWidthFrequency
from ..models.hmed2018 import DingModelPulseIntensityFrequency
from ..optimization.fes_ocp import OcpFes


class OcpFesId(OcpFes):
    def __init__(self):
        super(OcpFesId, self).__init__()

    @staticmethod
    def prepare_ocp(
        model: FesModel = None,
        final_time: float | int = None,
        pulse_width: dict = None,
        pulse_intensity: dict = None,
        objective: dict = None,
        key_parameter_to_identify: list = None,
        additional_key_settings: dict = None,
        custom_objective: list[Objective] = None,
        discontinuity_in_ocp: list = None,
        use_sx: bool = True,
        ode_solver: OdeSolver.RK1 | OdeSolver.RK2 | OdeSolver.RK4 = OdeSolver.RK4(n_integration_steps=10),
        n_threads: int = 1,
        control_type: ControlType = ControlType.CONSTANT,
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
        pulse_width: dict,
            The duration of the stimulation
        pulse_intensity: dict,
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
            pulse_width,
            pulse_intensity,
            temp_objective,
        ) = OcpFes._fill_dict(pulse_width, pulse_intensity, {})

        n_shooting = OcpFes.prepare_n_shooting(model.stim_time, final_time)
        OcpFesId._sanity_check(
            model=model,
            n_shooting=n_shooting,
            final_time=final_time,
            objective=temp_objective,
            use_sx=use_sx,
            ode_solver=ode_solver,
            n_threads=n_threads,
        )

        OcpFesId._sanity_check_id(
            model=model,
            final_time=final_time,
            objective=objective,
            pulse_width=pulse_width,
            pulse_intensity=pulse_intensity,
        )

        parameters, parameters_bounds, parameters_init = OcpFesId._set_parameters(
            parameter_to_identify=key_parameter_to_identify,
            parameter_setting=additional_key_settings,
            use_sx=use_sx,
        )

        OcpFesId.update_model_param(model, parameters)

        numerical_data_time_series, stim_idx_at_node_list = model.get_numerical_data_time_series(n_shooting, final_time)

        dynamics = OcpFesId.declare_dynamics(model=model, numerical_data_timeseries=numerical_data_time_series)
        x_bounds, x_init = OcpFesId.set_x_bounds(
            model=model,
            force_tracking=objective["force_tracking"],
            discontinuity_in_ocp=discontinuity_in_ocp,
        )
        objective_functions = OcpFesId._set_objective(model=model, objective=objective)

        control_value = (
            pulse_width["fixed"]
            if isinstance(model, DingModelPulseWidthFrequency)
            else pulse_intensity["fixed"] if isinstance(model, DingModelPulseIntensityFrequency) else None
        )
        u_bounds, u_init = OcpFesId.set_u_bounds(
            model=model, control_value=control_value, stim_idx_at_node_list=stim_idx_at_node_list, n_shooting=n_shooting
        )

        return OptimalControlProgram(
            bio_model=[model],
            dynamics=dynamics,
            n_shooting=n_shooting,
            phase_time=final_time,
            x_init=x_init,
            x_bounds=x_bounds,
            u_init=u_init,
            u_bounds=u_bounds,
            objective_functions=objective_functions,
            ode_solver=ode_solver,
            control_type=control_type,
            use_sx=use_sx,
            parameters=parameters,
            parameter_bounds=parameters_bounds,
            parameter_init=parameters_init,
            n_threads=n_threads,
        )

    @staticmethod
    def _sanity_check_id(
        model=None,
        final_time=None,
        objective=None,
        pulse_width=None,
        pulse_intensity=None,
    ):
        if not isinstance(final_time, int | float):
            raise TypeError(f"final_time must be int or float type.")

        if not isinstance(objective["force_tracking"], list):
            raise TypeError(
                f"force_tracking must be list type,"
                f" currently force_tracking is {type(objective['force_tracking'])}) type."
            )
        else:
            if not all(isinstance(val, int | float) for val in objective["force_tracking"]):
                raise TypeError(f"force_tracking must be list of int or float type.")

        if isinstance(model, DingModelPulseWidthFrequency):
            if not isinstance(pulse_width, dict):
                raise TypeError(
                    f"pulse_width must be dict type," f" currently pulse_width is {type(pulse_width)}) type."
                )

        if isinstance(model, DingModelPulseIntensityFrequency):
            if isinstance(pulse_intensity, dict):
                if not isinstance(pulse_intensity["fixed"], int | float | list):
                    raise ValueError(f"fixed pulse_intensity must be a int, float or list type.")

            else:
                raise TypeError(
                    f"pulse_intensity must be dict type,"
                    f" currently pulse_intensity is {type(pulse_intensity)}) type."
                )

    @staticmethod
    def set_x_bounds(
        model: FesModel = None,
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
    def _set_objective(model, objective):
        # Creates the objective for our problem (in this case, match a force curve)
        objective_functions = ObjectiveList()

        if objective["force_tracking"]:
            objective_functions.add(
                ObjectiveFcn.Lagrange.TRACK_STATE,
                key="F",
                weight=1,
                target=np.array(objective["force_tracking"])[np.newaxis, :],
                node=Node.ALL,
                quadratic=True,
            )

        if "custom" in objective and objective["custom"] is not None:
            for i in range(len(objective["custom"])):
                objective_functions.add(objective["custom"][i])

        return objective_functions

    @staticmethod
    def _set_parameters(
        parameter_to_identify,
        parameter_setting,
        use_sx,
    ):
        parameters = ParameterList(use_sx=use_sx)
        parameters_bounds = BoundsList()
        parameters_init = InitialGuessList()

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
                min_bound=np.array([parameter_setting[parameter_to_identify[i]]["min_bound"]]),
                max_bound=np.array([parameter_setting[parameter_to_identify[i]]["max_bound"]]),
                interpolation=InterpolationType.CONSTANT,
            )
            parameters_init.add(
                key=parameter_to_identify[i],
                initial_guess=np.array([parameter_setting[parameter_to_identify[i]]["initial_guess"]]),
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

    @staticmethod
    def set_u_bounds(model, control_value: list, stim_idx_at_node_list: list, n_shooting: int):
        # Controls bounds
        u_bounds = BoundsList()
        # Controls initial guess
        u_init = InitialGuessList()
        if isinstance(model, DingModelPulseWidthFrequency):
            if len(control_value) != 1:
                last_stim_idx = [stim_idx_at_node_list[i][-1] for i in range(len(stim_idx_at_node_list) - 1)]
                control_bounds = [control_value[last_stim_idx[i]] for i in range(len(last_stim_idx))]
            else:
                control_bounds = [control_value] * n_shooting
            u_init.add(key="last_pulse_width", initial_guess=[0], phase=0)
            u_bounds.add(
                "last_pulse_width",
                min_bound=np.array([control_bounds]),
                max_bound=np.array([control_bounds]),
                interpolation=InterpolationType.EACH_FRAME,
            )

        if isinstance(model, DingModelPulseIntensityFrequency):
            control_list = [
                [
                    (
                        control_value[stim_idx_at_node_list[j - i][j - i]]
                        if i < j + 1
                        else control_value[stim_idx_at_node_list[i][0]]
                    )
                    for i in range(n_shooting)
                ]
                for j in range(model.sum_stim_truncation)
            ]

            u_init.add(key="pulse_intensity", initial_guess=np.array(control_list)[:, 0], phase=0)
            u_bounds.add(
                "pulse_intensity",
                min_bound=np.array(control_list),
                max_bound=np.array(control_list),
                interpolation=InterpolationType.EACH_FRAME,
            )

        return u_bounds, u_init
