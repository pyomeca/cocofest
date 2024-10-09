from typing import Callable

from casadi import MX, vertcat, tanh
import numpy as np

from bioptim import (
    ConfigureProblem,
    DynamicsEvaluation,
    NonLinearProgram,
    OptimalControlProgram,
)
from .ding2003 import DingModelFrequency
from .state_configue import StateConfigure


class DingModelIntensityFrequency(DingModelFrequency):
    """
    This is a custom models that inherits from bioptim. CustomModel.
    As CustomModel is an abstract class, some methods are mandatory and must be implemented.
    Such as serialize, name_dof, nb_state.

    This is the Hmed 2018 model using the stimulation frequency and pulse intensity in input.

    Hmed, A. B., Bakir, T., Garnier, Y. M., Sakly, A., Lepers, R., & Binczak, S. (2018).
    An approach to a muscle force model with force-pulse amplitude relationship of human quadriceps muscles.
    Computers in Biology and Medicine, 101, 218-228.
    """

    def __init__(
        self,
        model_name: str = "hmed2018",
        muscle_name: str = None,
        sum_stim_truncation: int = None,
    ):
        super(DingModelIntensityFrequency, self).__init__(
            model_name=model_name,
            muscle_name=muscle_name,
            sum_stim_truncation=sum_stim_truncation,
        )
        self._with_fatigue = False
        self.stim_pulse_intensity_prev = []
        # ---- Custom values for the example ---- #
        # ---- Force models ---- #
        self.ar = 0.586  # (-) Translation of axis coordinates.
        self.bs = 0.026  # (-) Fiber muscle recruitment constant identification.
        self.Is = 63.1  # (mA) Muscle saturation intensity.
        self.cr = 0.833  # (-) Translation of axis coordinates.
        self.impulse_intensity = None

    @property
    def identifiable_parameters(self):
        return {
            "a_rest": self.a_rest,
            "tau1_rest": self.tau1_rest,
            "km_rest": self.km_rest,
            "tau2": self.tau2,
            "ar": self.ar,
            "bs": self.bs,
            "Is": self.Is,
            "cr": self.cr,
        }

    def set_ar(self, model, ar: MX | float):
        # models is required for bioptim compatibility
        self.ar = ar

    def set_bs(self, model, bs: MX | float):
        self.bs = bs

    def set_Is(self, model, Is: MX | float):
        self.Is = Is

    def set_cr(self, model, cr: MX | float):
        self.cr = cr

    # ---- Absolutely needed methods ---- #
    def serialize(self) -> tuple[Callable, dict]:
        # This is where you can serialize your models
        # This is useful if you want to save your models and load it later
        return (
            DingModelIntensityFrequency,
            {
                "tauc": self.tauc,
                "a_rest": self.a_rest,
                "tau1_rest": self.tau1_rest,
                "km_rest": self.km_rest,
                "tau2": self.tau2,
                "ar": self.ar,
                "bs": self.bs,
                "Is": self.Is,
                "cr": self.cr,
            },
        )

    def system_dynamics(
        self,
        cn: MX,
        f: MX,
        cn_sum: MX = None,
        force_length_relationship: MX | float = 1,
        force_velocity_relationship: MX | float = 1,
    ) -> MX:
        """
        The system dynamics is the function that describes the models.

        Parameters
        ----------
        cn: MX
            The value of the ca_troponin_complex (unitless)
        f: MX
            The value of the force (N)
        force_length_relationship: MX | float
            The force length relationship value (unitless)
        force_velocity_relationship: MX | float
            The force velocity relationship value (unitless)

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        cn_dot = self.cn_dot_fun(cn=cn, cn_sum=cn_sum)  # Equation n°1
        f_dot = self.f_dot_fun(
            cn,
            f,
            self.a_rest,
            self.tau1_rest,
            self.km_rest,
            force_length_relationship=force_length_relationship,
            force_velocity_relationship=force_velocity_relationship,
        )  # Equation n°2
        return vertcat(cn_dot, f_dot)

    def lambda_i_calculation(self, intensity_stim: MX):
        """
        Parameters
        ----------
        intensity_stim: MX
            The pulsation intensity of the current stimulation (mA)
        Returns
        -------
        The lambda factor, part of the n°1 equation
        """
        lambda_i = self.ar * (tanh(self.bs * (intensity_stim - self.Is)) + self.cr)  # equation include intensity
        return lambda_i

    @staticmethod
    def lambda_i_calculation_identification(
        intensity_stim: MX, ar: MX | float, bs: MX | float, Is: MX | float, cr: MX | float
    ):
        """
        Parameters
        ----------
        intensity_stim: MX
            The pulsation intensity of the current stimulation (mA)
        ar: MX | float
            Translation of axis coordinates (-)
        bs: MX | float
            Fiber muscle recruitment constant identification.
        Is: MX | float
            Muscle saturation intensity (mA)
        cr: MX | float
            Translation of axis coordinates (-)
        Returns
        -------
        The lambda factor, part of the n°1 equation
        """
        lambda_i = ar * (tanh(bs * (intensity_stim - Is)) + cr)  # equation include intensity
        return lambda_i

    def set_impulse_intensity(self, value: MX):
        """
        Sets the impulse intensity for each pulse (phases) according to the ocp parameter "impulse_intensity"

        Parameters
        ----------
        value: MX
            The pulsation intensity list (s)
        """
        self.impulse_intensity = []
        for i in range(value.shape[0]):
            self.impulse_intensity.append(value[i])

    @staticmethod
    def dynamics(
        time: MX,
        states: MX,
        controls: MX,
        parameters: MX,
        algebraic_states: MX,
        numerical_timeseries: MX,
        nlp: NonLinearProgram,
        fes_model: NonLinearProgram = None,
        force_length_relationship: MX | float = 1,
        force_velocity_relationship: MX | float = 1,
    ) -> DynamicsEvaluation:
        """
        Functional electrical stimulation dynamic

        Parameters
        ----------
        time: MX
            The system's current node time
        states: MX
            The state of the system CN, F, A, Tau1, Km
        controls: MX
            The controls of the system, none
        parameters: MX
            The parameters acting on the system, final time of each phase
        algebraic_states: MX
            The stochastic variables of the system, none
        numerical_timeseries: MX
            The numerical timeseries of the system
        nlp: NonLinearProgram
            A reference to the phase
        fes_model: DingModelIntensityFrequency
            The current phase fes model
        force_length_relationship: MX | float
            The force length relationship value (unitless)
        force_velocity_relationship: MX | float
            The force velocity relationship value (unitless)
        Returns
        -------
        The derivative of the states in the tuple[MX] format
        """

        dxdt_fun = fes_model.system_dynamics if fes_model else nlp.model.system_dynamics

        return DynamicsEvaluation(
            dxdt=dxdt_fun(
                cn=states[0],
                f=states[1],
                cn_sum=controls[0],
                force_length_relationship=force_length_relationship,
                force_velocity_relationship=force_velocity_relationship,
            ),
            defects=None,
        )

    def declare_ding_variables(
        self,
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        numerical_data_timeseries: dict[str, np.ndarray] = None,
    ):
        """
        Tell the program which variables are states and controls.
        The user is expected to use the ConfigureProblem.configure_xxx functions.
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node. Experimental external forces should be included here.
        """
        StateConfigure().configure_all_fes_model_states(ocp, nlp, fes_model=self)
        StateConfigure().configure_cn_sum(ocp, nlp)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, dyn_func=self.dynamics)

    def min_pulse_intensity(self):
        """
        Returns
        -------
        The minimum pulse intensity threshold of the model
        For lambda_i = ar * (tanh(bs * (intensity_stim - Is)) + cr) > 0
        """
        return (np.arctanh(-self.cr) / self.bs) + self.Is
