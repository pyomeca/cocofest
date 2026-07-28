"""
Builds the optimal control problem used to optimize a single-muscle FES model.
"""

import numpy as np

from bioptim import (
    BoundsList,
    ConstraintList,
    DynamicsOptionsList,
    DynamicsOptions,
    InitialGuessList,
    InterpolationType,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    ParameterList,
    PhaseDynamics,
    VariableScaling,
)

from ..misc.fourier_approx import FourierSeries
from cocofest.models.ding2007.ding2007 import DingModelPulseWidthFrequency
from cocofest.models.hmed2018.hmed2018 import DingModelPulseIntensityFrequency
from .penalties.custom_constraints import CustomConstraint


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
        """
        Declare the pulse intensity as an optimization parameter, if the model uses one.

        Parameters
        ----------
        model: FesModel
            The FES model used in the ocp
        max_pulse_intensity: int | float
            The maximum pulse intensity allowed
        use_sx: bool
            The nature of the casadi variables. MX are used if False.

        Returns
        -------
        tuple
            All the parameters to optimize of the program, their bounds and their initial guess
        """
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
        """
        Declare the pulse intensity sliding window constraint, if the model uses pulse intensity.

        Parameters
        ----------
        model: FesModel
            The FES model used in the ocp
        n_shooting: int
            The number of shooting points of the phase
        stim_idx_at_node_list: list
            The list of stimulation indices considered at each node

        Returns
        -------
        ConstraintList
            All the constraints of the program
        """
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
    def declare_dynamics_options(numerical_time_series, ode_solver):
        """
        Build the ocp's dynamics options, sharing dynamics between nodes and expanding them for speed.

        Parameters
        ----------
        numerical_time_series: dict
            The numerical timeseries at each node. ex: the experimental external forces data should go here.
        ode_solver: OdeSolver
            The integrator to use to integrate this dynamics.

        Returns
        -------
        DynamicsOptionsList
            The dynamics of the phase
        """
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
        return dynamics_options

    @staticmethod
    def set_x_bounds(model):
        """
        Build the state bounds from the model's rest values, widening them to the model's physiological range.

        Parameters
        ----------
        model: FesModel
            The FES model used in the ocp

        Returns
        -------
        BoundsList
            The bounds for the states
        """
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
        variable_bound_list = model.name_dofs
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
        """
        Build the state initial guess from the model's rest values.

        Parameters
        ----------
        model: FesModel
            The FES model used in the ocp

        Returns
        -------
        InitialGuessList
            The initial guesses for the states
        """
        variable_bound_list = model.name_dofs
        x_init = InitialGuessList()
        for j in range(len(variable_bound_list)):
            x_init.add(variable_bound_list[j], model.standard_rest_values()[j])

        return x_init

    @staticmethod
    def set_u_bounds(model, max_bound: int | float):
        """
        Build the control bounds (pulse width or pulse intensity, depending on the model).

        Parameters
        ----------
        model: FesModel
            The FES model used in the ocp
        max_bound: int | float
            The maximum control value allowed (pulse width or pulse intensity)

        Returns
        -------
        BoundsList
            The bounds for the controls
        """
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
        """
        Build the control initial guess (pulse width or pulse intensity, depending on the model).

        Parameters
        ----------
        model: FesModel
            The FES model used in the ocp

        Returns
        -------
        InitialGuessList
            The initial guesses for the controls
        """
        u_init = InitialGuessList()  # Controls initial guess

        if isinstance(model, DingModelPulseWidthFrequency):
            u_init.add(key="last_pulse_width", initial_guess=[0], phase=0)

        if isinstance(model, DingModelPulseIntensityFrequency):
            u_init.add(key="pulse_intensity", initial_guess=[0] * model.sum_stim_truncation, phase=0)

        return u_init

    # TODO: Remove this method
    @staticmethod
    def _set_objective(n_shooting, objective):
        """
        Build the objective functions of the program from a custom, force-tracking and/or end-node-tracking entry.

        Parameters
        ----------
        n_shooting: int
            The number of shooting points of the phase
        objective: dict
            The ocp objective, including custom, force_tracking and end_node_tracking entries

        Returns
        -------
        ObjectiveList
            All the objective function of the program
        """
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
        """
        Fit a tracked force curve with a Fourier series, then resample it at every shooting node.

        Parameters
        ----------
        force_to_track: list
            A [time, force] pair of equal-length lists to track
        n_shooting: int
            The number of shooting points
        final_time: float
            The ocp final time

        Returns
        -------
        The tracked force resampled at each shooting node
        """
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
        """
        Apply every identified/optimized bioptim parameter back onto the model via its setter function.

        Parameters
        ----------
        model: FesModel
            The FES model to update
        parameters: ParameterList
            All the parameters to optimize of the program
        """
        for param_key in parameters:
            if parameters[param_key].function:
                param_scaling = parameters[param_key].scaling.scaling
                param_reduced = parameters[param_key].cx
                parameters[param_key].function(model, param_reduced * param_scaling, **parameters[param_key].kwargs)
