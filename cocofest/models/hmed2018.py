from typing import Callable

from casadi import MX, vertcat, tanh, if_else
import numpy as np

from bioptim import (
    ConfigureProblem,
    DynamicsEvaluation,
    NonLinearProgram,
    OptimalControlProgram,
    ParameterList,
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

    def __init__(self, model_name: str = "hmed2018", muscle_name: str = None, sum_stim_truncation: int = None):
        super(DingModelIntensityFrequency, self).__init__(
            model_name=model_name, muscle_name=muscle_name, sum_stim_truncation=sum_stim_truncation
        )
        self._with_fatigue = False
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
        t: MX = None,
        t_stim_prev: list[MX] | list[float] = None,
        intensity_stim: list[MX] | list[float] = None,
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
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_prev: list[MX]
            The time list of the previous stimulations (ms)
        intensity_stim: list[MX]
            The pulsation intensity of the current stimulation (mA)
        force_length_relationship: MX | float
            The force length relationship value (unitless)
        force_velocity_relationship: MX | float
            The force velocity relationship value (unitless)

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        r0 = self.km_rest + self.r0_km_relationship  # Simplification
        t_stim_prev = self.slice_stim(t, t_stim_prev)
        cn_dot = self.cn_dot_fun(cn, r0, t, t_stim_prev=t_stim_prev, intensity_stim=intensity_stim)  # Equation n°1
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

    def cn_dot_fun(
        self, cn: MX, r0: MX | float, t: MX, t_stim_prev: list[MX], intensity_stim: list[MX] = None
    ) -> MX | float:
        """
        Parameters
        ----------
        cn: MX
            The previous step value of ca_troponin_complex (unitless)
        r0: MX
            Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli (unitless)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_prev: list[MX]
            The time list of the previous stimulations (ms)
        intensity_stim: list[MX]
            The pulsation intensity of the current stimulation (mA)
        Returns
        -------
        The value of the derivative ca_troponin_complex (unitless)
        """
        sum_multiplier = self.cn_sum_fun(r0, t, t_stim_prev=t_stim_prev, intensity_stim=intensity_stim)

        return (1 / self.tauc) * sum_multiplier - (cn / self.tauc)  # Eq(1)

    def cn_sum_fun(
        self, r0: MX | float, t: MX, t_stim_prev: list[MX] = None, intensity_stim: list[MX] = None
    ) -> MX | float:
        """
        Parameters
        ----------
        r0: MX | float
            Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli (unitless)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_prev: list[MX]
            The time list of the previous stimulations (ms)
        intensity_stim: list[MX]
            The pulsation intensity of the current stimulation (mA)

        Returns
        -------
        A part of the n°1 equation
        """
        sum_multiplier = 0
        # enough_stim_to_truncate = self._sum_stim_truncation and len(t_stim_prev) > self._sum_stim_truncation
        # if enough_stim_to_truncate:
        #     t_stim_prev = t_stim_prev[-self._sum_stim_truncation :]
        # TODO : Truncation not working for single phase

        for i in range(len(t_stim_prev)):
            previous_phase_time = t_stim_prev[i] - t_stim_prev[i - 1]
            ri = 1 if i == 0 else self.ri_fun(r0, previous_phase_time)  # Part of Eq n°1
            exp_time = self.exp_time_fun(t, t_stim_prev[i])  # Part of Eq n°1
            lambda_i = self.lambda_i_calculation(intensity_stim[i])
            sum_multiplier += lambda_i * ri * exp_time  # Part of Eq n°1
        return sum_multiplier

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
    def get_intensity_parameters(nlp, parameters: ParameterList, muscle_name: str = None) -> MX:
        """
        Get the nlp list of intensity parameters

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        parameters: ParameterList
            The nlp list parameter
        muscle_name: str
            The muscle name

        Returns
        -------
        The list of intensity parameters
        """
        intensity_parameters = vertcat()
        for j in range(parameters.shape[0]):
            if muscle_name:
                if "pulse_intensity_" + muscle_name in nlp.parameters.scaled.cx[j].str():
                    intensity_parameters = vertcat(intensity_parameters, parameters[j])
            elif "pulse_intensity" in nlp.parameters.scaled.cx[j].str():
                intensity_parameters = vertcat(intensity_parameters, parameters[j])

        return intensity_parameters

    @staticmethod
    def dynamics(
        time: MX,
        states: MX,
        controls: MX,
        parameters: MX,
        algebraic_states: MX,
        numerical_timeseries: MX,
        nlp: NonLinearProgram,
        stim_prev: list[float] = None,
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
        stim_prev: list[float]
            The previous stimulation values
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
        intensity_stim_prev = (
            []
        )  # Every stimulation intensity before the current phase, i.e.: the intensity of each phase
        intensity_parameters = (
            nlp.model.get_intensity_parameters(nlp, parameters)
            if fes_model is None
            else fes_model.get_intensity_parameters(nlp, parameters, muscle_name=fes_model.muscle_name)
        )

        dxdt_fun = fes_model.system_dynamics if fes_model else nlp.model.system_dynamics
        stim_apparition = (
            (
                fes_model.get_stim_prev(nlp=nlp, parameters=parameters, idx=nlp.phase_idx)
                if fes_model
                else nlp.model.get_stim_prev(nlp=nlp, parameters=parameters, idx=nlp.phase_idx)
            )
            if stim_prev is None
            else stim_prev
        )  # Get the previous stimulation apparition time from the parameters
        # if not provided from stim_prev, this way of getting the list is not optimal, but it is the only way to get it.
        # Otherwise, it will create issues with free variables or wrong mx or sx type while calculating the dynamics

        return DynamicsEvaluation(
            dxdt=dxdt_fun(
                cn=states[0],
                f=states[1],
                t=time,
                t_stim_prev=stim_apparition,
                intensity_stim=intensity_parameters,
                force_length_relationship=force_length_relationship,
                force_velocity_relationship=force_velocity_relationship,
            ),
            defects=None,
        )

    def declare_ding_variables(
        self, ocp: OptimalControlProgram, nlp: NonLinearProgram, numerical_data_timeseries: dict[str, np.ndarray] = None
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
        stim_prev = (
            self._build_t_stim_prev(ocp, nlp.phase_idx)
            if "pulse_apparition_time" not in nlp.parameters.keys()
            else None
        )
        ConfigureProblem.configure_dynamics_function(ocp, nlp, dyn_func=self.dynamics, stim_prev=stim_prev)

    def min_pulse_intensity(self):
        """
        Returns
        -------
        The minimum pulse intensity threshold of the model
        For lambda_i = ar * (tanh(bs * (intensity_stim - Is)) + cr) > 0
        """
        return (np.arctanh(-self.cr) / self.bs) + self.Is
