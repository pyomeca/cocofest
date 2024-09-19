import time as time_package
import numpy as np

from bioptim import Solver, Objective, OdeSolver
from ..models.ding2007 import DingModelPulseDurationFrequency
from ..identification.ding2003_force_parameter_identification import (
    DingModelFrequencyForceParameterIdentification,
)
from ..optimization.fes_identification_ocp import OcpFesId
from .identification_method import (
    full_data_extraction,
    average_data_extraction,
    sparse_data_extraction,
    node_shooting_list_creation,
    force_at_node_in_ocp,
)


class DingModelPulseDurationFrequencyForceParameterIdentification(
    DingModelFrequencyForceParameterIdentification
):
    """
    This class extends the DingModelFrequencyForceParameterIdentification class and is used to define an optimal control problem (OCP).
    It prepares the full program and provides all the necessary parameters to solve a functional electrical stimulation OCP.
    """

    def __init__(
        self,
        model: DingModelPulseDurationFrequency,
        data_path: str | list[str] = None,
        identification_method: str = "full",
        double_step_identification: bool = False,
        key_parameter_to_identify: list = None,
        additional_key_settings: dict = None,
        n_shooting: int = 5,
        custom_objective: list[Objective] = None,
        use_sx: bool = True,
        ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
        n_threads: int = 1,
        **kwargs,
    ):
        """
        Initializes the DingModelPulseDurationFrequencyForceParameterIdentification class.

        Parameters
        ----------
        model: DingModelPulseDurationFrequency
            The model to use for the OCP.
        data_path: str | list[str]
            The path to the force model data.
        identification_method: str
            The method to use for the force model identification. Options are "full" for objective function on all data,
            "average" for objective function on average data, and "sparse" for objective function at the beginning and end of the data.
        double_step_identification: bool
            If True, the identification process will be performed in two steps.
        key_parameter_to_identify: list
            List of keys of the parameters to identify.
        additional_key_settings: dict
            Additional settings for the keys.
        n_shooting: int
            The number of shooting points for the OCP.
        custom_objective: list[Objective]
            List of custom objectives.
        use_sx: bool
            The nature of the CasADi variables. MX are used if False.
        ode_solver: OdeSolver
            The ODE solver to use.
        n_threads: int
            The number of threads to use while solving (multi-threading if > 1).
        **kwargs: dict
            Additional keyword arguments.
        """
        super(
            DingModelPulseDurationFrequencyForceParameterIdentification, self
        ).__init__(
            model=model,
            data_path=data_path,
            identification_method=identification_method,
            double_step_identification=double_step_identification,
            key_parameter_to_identify=key_parameter_to_identify,
            additional_key_settings=additional_key_settings,
            n_shooting=n_shooting,
            custom_objective=custom_objective,
            use_sx=use_sx,
            ode_solver=ode_solver,
            n_threads=n_threads,
            **kwargs,
        )

    def _set_default_values(self, model):
        """
        Sets the default values for the identified parameters (initial guesses, bounds, scaling and function).
        If the user does not provide additional_key_settings for a specific parameter, the default value will be used.

        Parameters
        ----------
        model: FesModel
            The model to use for the OCP.

        Returns
        -------
        dict
            A dictionary of default values for the identified parameters.
        """
        return {
            "tau1_rest": {
                "initial_guess": 0.5,
                "min_bound": 0.0001,
                "max_bound": 1,
                "function": model.set_tau1_rest,
                "scaling": 1,  # 10000
            },
            "tau2": {
                "initial_guess": 0.5,
                "min_bound": 0.0001,
                "max_bound": 1,
                "function": model.set_tau2,
                "scaling": 1,  # 10000
            },
            "km_rest": {
                "initial_guess": 0.5,
                "min_bound": 0.001,
                "max_bound": 1,
                "function": model.set_km_rest,
                "scaling": 1,  # 10000
            },
            "a_scale": {
                "initial_guess": 5000,
                "min_bound": 1,
                "max_bound": 10000,
                "function": model.set_a_scale,
                "scaling": 1,
            },
            "pd0": {
                "initial_guess": 1e-4,
                "min_bound": 1e-4,
                "max_bound": 6e-4,
                "function": model.set_pd0,
                "scaling": 1,  # 1000
            },
            "pdt": {
                "initial_guess": 1e-4,
                "min_bound": 1e-4,
                "max_bound": 6e-4,
                "function": model.set_pdt,
                "scaling": 1,  # 1000
            },
        }

    def _set_default_parameters_list(self):
        """
        Sets the default parameters list for the model.
        """
        self.numeric_parameters = [
            self.model.tau1_rest,
            self.model.tau2,
            self.model.km_rest,
            self.model.a_scale,
            self.model.pd0,
            self.model.pdt,
        ]
        self.key_parameters = ["tau1_rest", "tau2", "km_rest", "a_scale", "pd0", "pdt"]

    @staticmethod
    def _set_model_parameters(model, model_parameters_value):
        """
        Sets the model parameters.

        Parameters
        ----------
        model: FesModel
            The model to use for the OCP.
        model_parameters_value: list
            List of values for the model parameters.

        Returns
        -------
        FesModel
            The model with updated parameters.
        """
        model.a_scale = model_parameters_value[0]
        model.km_rest = model_parameters_value[1]
        model.tau1_rest = model_parameters_value[2]
        model.tau2 = model_parameters_value[3]
        model.pd0 = model_parameters_value[4]
        model.pdt = model_parameters_value[5]
        return model

    @staticmethod
    def pulse_duration_extraction(data_path: str) -> list[float]:
        """
        Extracts the pulse duration from the data.

        Parameters
        ----------
        data_path: str
            The path to the data.

        Returns
        -------
        list[float]
            A list of pulse durations.
        """
        import pickle

        pulse_duration = []
        for i in range(len(data_path)):
            with open(data_path[i], "rb") as f:
                data = pickle.load(f)
            pulse_duration.append(data["pulse_duration"])
        pulse_duration = [item for sublist in pulse_duration for item in sublist]
        return pulse_duration

    def _force_model_identification_for_initial_guess(self):
        """
        Performs the force model identification for the initial guess.

        Returns
        -------
        dict
            A dictionary of initial guesses for the parameters.
        """
        self.input_sanity(
            self.model,
            self.data_path,
            self.force_model_identification_method,
            self.double_step_identification,
            self.key_parameter_to_identify,
            self.additional_key_settings,
            self.n_shooting,
        )
        self.check_experiment_force_format(self.data_path)
        # --- Data extraction --- #
        # --- Force model --- #
        stimulated_n_shooting = self.n_shooting
        force_curve_number = None

        time, stim, force, discontinuity = average_data_extraction(self.data_path)
        pulse_duration = self.pulse_duration_extraction(self.data_path)
        n_shooting, final_time_phase = node_shooting_list_creation(
            stim, stimulated_n_shooting
        )
        force_at_node = force_at_node_in_ocp(
            time, force, n_shooting, final_time_phase, force_curve_number
        )

        # --- Building force ocp --- #
        self.force_ocp = OcpFesId.prepare_ocp(
            model=self.model,
            n_shooting=n_shooting,
            final_time_phase=final_time_phase,
            force_tracking=force_at_node,
            custom_objective=self.custom_objective,
            discontinuity_in_ocp=discontinuity,
            pulse_duration=pulse_duration,
            km_rest=self.km_rest,
            tau1_rest=self.tau1_rest,
            tau2=self.tau2,
            use_sx=self.use_sx,
            ode_solver=self.ode_solver,
            n_threads=self.n_threads,
        )

        self.force_identification_result = self.force_ocp.solve(
            Solver.IPOPT()
        )  # _hessian_approximation="limited-memory"

        initial_guess = {}
        for key in self.key_parameter_to_identify:
            initial_guess[key] = self.force_identification_result.parameters[key][0][0]

        return initial_guess

    def force_model_identification(self) -> dict[str, np.ndarray]:
        """
        Performs the force model identification.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary of identified parameters.
        """

        if not self.double_step_identification:
            self.input_sanity(
                self.model,
                self.data_path,
                self.force_model_identification_method,
                self.double_step_identification,
                self.key_parameter_to_identify,
                self.additional_key_settings,
                self.n_shooting,
            )
            self.check_experiment_force_format(self.data_path)

        # --- Data extraction --- #
        # --- Force model --- #
        stimulated_n_shooting = self.n_shooting
        force_curve_number = None
        stim = None
        time = None
        force = None

        if self.force_model_identification_method == "full":
            time, stim, force, discontinuity = full_data_extraction(self.data_path)
            pulse_duration = self.pulse_duration_extraction(self.data_path)

        elif self.force_model_identification_method == "average":
            time, stim, force, discontinuity = average_data_extraction(self.data_path)
            pulse_duration = np.mean(
                np.array(self.pulse_duration_extraction(self.data_path))
            )

        elif self.force_model_identification_method == "sparse":
            force_curve_number = (
                self.kwargs["force_curve_number"]
                if "force_curve_number" in self.kwargs
                else 5
            )
            time, stim, force, discontinuity = sparse_data_extraction(
                self.data_path, force_curve_number
            )
            pulse_duration = self.pulse_duration_extraction(
                self.data_path
            )  # TODO : adapt this for sparse data

        n_shooting, final_time_phase = node_shooting_list_creation(
            stim, stimulated_n_shooting
        )
        force_at_node = force_at_node_in_ocp(
            time, force, n_shooting, final_time_phase, force_curve_number
        )

        if self.double_step_identification:
            initial_guess = self._force_model_identification_for_initial_guess()

            for key in self.key_parameter_to_identify:
                self.additional_key_settings[key]["initial_guess"] = initial_guess[key]

        # --- Building force ocp --- #
        start_time = time_package.time()
        self.force_ocp = OcpFesId.prepare_ocp(
            model=self.model,
            n_shooting=n_shooting,
            final_time_phase=final_time_phase,
            force_tracking=force_at_node,
            key_parameter_to_identify=self.key_parameter_to_identify,
            additional_key_settings=self.additional_key_settings,
            custom_objective=self.custom_objective,
            discontinuity_in_ocp=discontinuity,
            pulse_duration=pulse_duration,
            use_sx=self.use_sx,
            ode_solver=self.ode_solver,
            n_threads=self.n_threads,
        )

        print(f"OCP creation time : {time_package.time() - start_time} seconds")

        self.force_identification_result = self.force_ocp.solve(Solver.IPOPT())

        identified_parameters = {}
        for key in self.key_parameter_to_identify:
            identified_parameters[key] = self.force_identification_result.parameters[
                key
            ][0]

        return identified_parameters
