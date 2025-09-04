import numpy as np

from bioptim import (
    BoundsList,
    ConstraintList,
    DynamicsList,
    InitialGuessList,
    InterpolationType,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    ParameterList,
    PhaseDynamics,
    VariableScaling,
    OdeSolver,
)

from ..fourier_approx import FourierSeries
from cocofest.models.ding2007.ding2007 import DingModelPulseWidthFrequency
from cocofest.models.hmed2018.hmed2018 import DingModelPulseIntensityFrequency
from ..custom_constraints import CustomConstraint


class OcpFes:
    """
    The main class to define an ocp. This class prepares the full program and gives all
    the needed parameters to solve a functional electrical stimulation ocp.
    """

    @staticmethod
    def set_parameters(
        model,
        max_pulse_intensity,
        use_sx,
    ):
        parameters = ParameterList(use_sx=use_sx)
        parameters_bounds = BoundsList()
        parameters_init = InitialGuessList()
        n_stim = len(model.stim_time)

        if isinstance(model, DingModelPulseIntensityFrequency):
            parameters.add(
                name="pulse_intensity",
                function=DingModelPulseIntensityFrequency.set_impulse_intensity,
                size=n_stim,
                scaling=VariableScaling("pulse_intensity", [1] * n_stim),
            )
            parameters_bounds.add(
                "pulse_intensity",
                min_bound=[model.min_pulse_intensity()],
                max_bound=[max_pulse_intensity],
                interpolation=InterpolationType.CONSTANT,
            )
            intensity_avg = (model.min_pulse_intensity() + max_pulse_intensity) / 2
            parameters_init["pulse_intensity"] = np.array([intensity_avg] * n_stim)

        return parameters, parameters_bounds, parameters_init

    @staticmethod
    def set_constraints(model, n_shooting, stim_idx_at_node_list):
        constraints = ConstraintList()
        if isinstance(model, DingModelPulseIntensityFrequency):
            for i in range(n_shooting):
                last_stim_idx = stim_idx_at_node_list[i][-1]
                constraints.add(
                    CustomConstraint.pulse_intensity_sliding_window_constraint,
                    last_stim_idx=last_stim_idx,
                    muscle_name=model.muscle_name,
                    node=i,
                )

        return constraints

    @staticmethod
    def declare_dynamics(model, numerical_data_timeseries=None, ode_solver=OdeSolver.RK4(n_integration_steps=10)):
        dynamics = DynamicsList()
        dynamics.add(
            model.declare_ding_variables,
            dynamic_function=model.dynamics,
            expand_dynamics=True,
            phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
            numerical_data_timeseries=numerical_data_timeseries,
            ode_solver=ode_solver,
        )
        return dynamics

    @staticmethod
    def set_x_bounds(model):
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
                max_bounds[i] = 2
            if variable_bound_list[i] == "F":
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

        return x_bounds

    @staticmethod
    def set_x_init(model):
        variable_bound_list = model.name_dof
        x_init = InitialGuessList()
        for j in range(len(variable_bound_list)):
            x_init.add(variable_bound_list[j], model.standard_rest_values()[j])

        return x_init

    @staticmethod
    def set_u_bounds(model, max_bound: int | float):
        u_bounds = BoundsList()  # Controls bounds

        if isinstance(model, DingModelPulseWidthFrequency):
            min_pulse_width = model.pd0 if isinstance(model.pd0, int | float) else 0
            u_bounds.add(
                "last_pulse_width",
                min_bound=np.array([[min_pulse_width] * 3]),
                max_bound=np.array([[max_bound] * 3]),
                interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
            )

        if isinstance(model, DingModelPulseIntensityFrequency):
            min_pulse_intensity = (
                model.min_pulse_intensity() if isinstance(model.min_pulse_intensity(), int | float) else 0
            )
            u_bounds.add(
                "pulse_intensity",
                min_bound=[min_pulse_intensity] * model.sum_stim_truncation,
                max_bound=[max_bound] * model.sum_stim_truncation,
                interpolation=InterpolationType.CONSTANT,
            )

        return u_bounds

    @staticmethod
    def set_u_init(model):
        u_init = InitialGuessList()  # Controls initial guess

        if isinstance(model, DingModelPulseWidthFrequency):
            u_init.add(key="last_pulse_width", initial_guess=[0], phase=0)

        if isinstance(model, DingModelPulseIntensityFrequency):
            u_init.add(key="pulse_intensity", initial_guess=[0] * model.sum_stim_truncation, phase=0)

        return u_init

    # TODO: Remove this method
    @staticmethod
    def _set_objective(n_shooting, objective):
        # Creates the objective for our problem
        objective_functions = ObjectiveList()
        if objective["custom"]:
            for i in range(len(objective["custom"])):
                objective_functions.add(objective["custom"][0][i])

        if objective["force_tracking"]:
            force_fourier_coefficient = (
                None
                if objective["force_tracking"] is None
                else OcpFes._build_fourier_coefficient(objective["force_tracking"])
            )
            force_to_track = FourierSeries().fit_func_by_fourier_series_with_real_coeffs(
                np.linspace(0, 1, n_shooting + 1),
                force_fourier_coefficient,
            )[np.newaxis, :]

            objective_functions.add(
                ObjectiveFcn.Lagrange.TRACK_STATE,
                key="F",
                weight=100,
                target=force_to_track,
                node=Node.ALL,
                quadratic=True,
            )

        if objective["end_node_tracking"]:
            objective_functions.add(
                ObjectiveFcn.Mayer.MINIMIZE_STATE,
                node=Node.END,
                key="F",
                quadratic=True,
                weight=1,
                target=objective["end_node_tracking"],
            )

        return objective_functions

    @staticmethod
    def check_and_adjust_dimensions_for_objective_fun(force_to_track, n_shooting, final_time):
        if len(force_to_track[0]) != len(force_to_track[1]):
            raise ValueError("force_tracking time and force argument must be same length")
        if len(force_to_track) != 2:
            raise ValueError("force_tracking list size 2")

        force_fourier_coefficient = FourierSeries().compute_real_fourier_coeffs(
            force_to_track[0], force_to_track[1], 50
        )
        force_to_track = FourierSeries().fit_func_by_fourier_series_with_real_coeffs(
            np.linspace(0, final_time, n_shooting + 1),
            force_fourier_coefficient,
        )[np.newaxis, :]

        return force_to_track

    @staticmethod
    def update_model_param(model, parameters):
        for param_key in parameters:
            if parameters[param_key].function:
                param_scaling = parameters[param_key].scaling.scaling
                param_reduced = parameters[param_key].cx
                parameters[param_key].function(model, param_reduced * param_scaling, **parameters[param_key].kwargs)
