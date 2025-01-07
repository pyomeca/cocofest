from typing import Callable

import numpy as np
from casadi import MX, vertcat, exp, if_else

from bioptim import (
    ConfigureProblem,
    DynamicsEvaluation,
    NonLinearProgram,
    OptimalControlProgram,
    ParameterList,
)
from .ding2003 import DingModelFrequency
from .state_configure import StateConfigure


class DingModelPulseWidthFrequency(DingModelFrequency):
    """
    This is a custom models that inherits from bioptim. CustomModel.
    As CustomModel is an abstract class, some methods are mandatory and must be implemented.
    Such as serialize, name_dof, nb_state.

    This is the Ding 2007 model using the stimulation frequency and pulse width in input.

    Ding, J., Chou, L. W., Kesar, T. M., Lee, S. C., Johnston, T. E., Wexler, A. S., & Binder‐Macleod, S. A. (2007).
    Mathematical model that predicts the force–intensity and force–frequency relationships after spinal cord injuries.
    Muscle & Nerve: Official Journal of the American Association of Electrodiagnostic Medicine, 36(2), 214-222.
    """

    def __init__(
        self,
        model_name: str = "ding_2007",
        muscle_name: str = None,
        stim_time: list[float] = None,
        previous_stim: dict = None,
        sum_stim_truncation: int = None,
        is_approximated: bool = False,
        tauc: float = None,
        a_rest: float = None,
        tau1_rest: float = None,
        km_rest: float = None,
        tau2: float = None,
        pd0: float = None,
        pdt: float = None,
        a_scale: float = None,
        alpha_a: float = None,
        alpha_tau1: float = None,
        alpha_km: float = None,
        tau_fat: float = None,
    ):
        super(DingModelPulseWidthFrequency, self).__init__(
            model_name=model_name,
            muscle_name=muscle_name,
            stim_time=stim_time,
            previous_stim=previous_stim,
            sum_stim_truncation=sum_stim_truncation,
            is_approximated=is_approximated,
        )
        self._with_fatigue = False
        self.pulse_width = None

        # --- Default values --- #
        A_SCALE_DEFAULT = 4920  # Value from Ding's 2007 article (N/s)
        PD0_DEFAULT = 0.000131405  # Value from Ding's 2007 article (s)
        PDT_DEFAULT = 0.000194138  # Value from Ding's 2007 article (s)
        TAU1_REST_DEFAULT = 0.060601  # Value from Ding's 2003 article (s)
        TAU2_DEFAULT = 0.001  # Value from Ding's 2007 article (s)
        KM_REST_DEFAULT = 0.137  # Value from Ding's 2007 article (unitless)
        TAUC_DEFAULT = 0.011  # Value from Ding's 2007 article (s)

        # ---- Custom values for the example ---- #
        # ---- Force models ---- #
        self.a_scale = A_SCALE_DEFAULT
        self.pd0 = PD0_DEFAULT
        self.pdt = PDT_DEFAULT
        self.tau1_rest = TAU1_REST_DEFAULT
        self.tau2 = TAU2_DEFAULT
        self.km_rest = KM_REST_DEFAULT
        self.tauc = TAUC_DEFAULT

    @property
    def identifiable_parameters(self):
        return {
            "a_scale": self.a_scale,
            "tau1_rest": self.tau1_rest,
            "km_rest": self.km_rest,
            "tau2": self.tau2,
            "pd0": self.pd0,
            "pdt": self.pdt,
        }

    def set_a_scale(self, model, a_scale: MX | float):
        # models is required for bioptim compatibility
        self.a_scale = a_scale

    def set_pd0(self, model, pd0: MX | float):
        self.pd0 = pd0

    def set_pdt(self, model, pdt: MX | float):
        self.pdt = pdt

    def serialize(self) -> tuple[Callable, dict]:
        # This is where you can serialize your models
        # This is useful if you want to save your models and load it later
        return (
            DingModelPulseWidthFrequency,
            {
                "tauc": self.tauc,
                "a_rest": self.a_rest,
                "tau1_rest": self.tau1_rest,
                "km_rest": self.km_rest,
                "tau2": self.tau2,
                "a_scale": self.a_scale,
                "pd0": self.pd0,
                "pdt": self.pdt,
            },
        )

    def system_dynamics(
        self,
        cn: MX,
        f: MX,
        t: MX = None,
        pulse_width: MX = None,
        cn_sum: MX = None,
        a_scale: MX = None,
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
            The current time at which the dynamics is evaluated (s)
        pulse_width: MX
            The pulsation duration of the current stimulation (s)
        cn_sum: MX | float
            The sum of the ca_troponin_complex (unitless)
        a_scale: MX | float
            The scaling factor of the current stimulation (unitless)
        force_length_relationship: MX | float
            The force length relationship value (unitless)
        force_velocity_relationship: MX | float
            The force velocity relationship value (unitless)

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        t_stim_prev = self.all_stim
        if self.all_stim != self.stim_time and not self.is_approximated:
            pulse_width = self.previous_stim["pulse_width"] + pulse_width
        cn_dot = self.calculate_cn_dot(cn, cn_sum, t, t_stim_prev)
        a_scale = (
            a_scale
            if self.is_approximated
            else self.a_calculation(
                a_scale=self.a_scale,
                pulse_width=pulse_width,
                t=t,
                t_stim_prev=t_stim_prev,
            )
        )

        f_dot = self.f_dot_fun(
            cn,
            f,
            a_scale,
            self.tau1_rest,
            self.km_rest,
            force_length_relationship=force_length_relationship,
            force_velocity_relationship=force_velocity_relationship,
        )  # Equation n°2 from Ding's 2003 article
        return vertcat(cn_dot, f_dot)

    def a_calculation(
        self,
        a_scale: float | MX,
        pulse_width: MX,
        t=None,
        t_stim_prev: list[float] | list[MX] = None,
    ) -> MX:
        """
        Parameters
        ----------
        a_scale: float | MX
            The scaling factor of the current stimulation (unitless)
        pulse_width: MX
            The pulsation duration of the current stimulation (s)
        t: MX
            The current time at which the dynamics is evaluated (s)
        t_stim_prev: list[float] | list[MX]
            The time list of the previous stimulations (s)
        Returns
        -------
        The value of scaling factor (unitless)
        """
        if self.is_approximated:
            return a_scale * (1 - exp(-(pulse_width - self.pd0) / self.pdt))
        else:
            pulse_width_list = pulse_width
            for i in range(len(t_stim_prev)):
                if i == 0:
                    pulse_width = pulse_width_list[0]
                else:
                    coefficient = if_else(t_stim_prev[i] <= t, 1, 0)
                    temp_pulse_width = pulse_width_list[i] * coefficient
                    pulse_width = if_else(temp_pulse_width != 0, temp_pulse_width, pulse_width)
            return a_scale * (1 - exp(-(pulse_width - self.pd0) / self.pdt))

    def a_calculation_identification(
        self,
        a_scale: float | MX,
        pulse_width: MX,
        pd0: float | MX,
        pdt: float | MX,
    ) -> MX:
        """
        Parameters
        ----------
        a_scale: float | MX
            The scaling factor of the current stimulation (unitless)
        pulse_width: MX
            The pulsation duration of the current stimulation (s)
        pd0: float | MX
            The pd0 value (s)
        pdt: float | MX
            The pdt value (s)

        Returns
        -------
        The value of scaling factor (unitless)
        """
        return a_scale * (1 - exp(-(pulse_width - pd0) / pdt))

    def set_impulse_width(self, value: list[MX]):
        """
        Sets the pulse width for each pulse (phases) according to the ocp parameter "pulse_width"

        Parameters
        ----------
        value: list[MX]
            The pulsation duration list (s)
        """
        self.pulse_width = value

    @staticmethod
    def get_pulse_width_parameters(nlp, parameters: ParameterList, muscle_name: str = None) -> list[MX]:
        """
        Get the nlp list of pulse_width parameters

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
        The list of list of pulse_width parameters
        """

        pulse_width_parameters = []
        for j in range(parameters.shape[0]):
            if muscle_name:
                if "pulse_width_" + muscle_name in nlp.parameters.scaled.cx[j].str():
                    pulse_width_parameters.append(parameters[j])
            elif "pulse_width" in nlp.parameters.scaled.cx[j].str():
                pulse_width_parameters.append(parameters[j])
        return pulse_width_parameters

    @staticmethod
    def dynamics(
        time: MX,
        states: MX,
        controls: MX,
        parameters: MX,
        algebraic_states: MX,
        numerical_timeseries: MX,
        nlp: NonLinearProgram,
        fes_model=None,
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
        fes_model: DingModelPulseWidthFrequency
            The current phase fes model
        force_length_relationship: MX | float
            The force length relationship value (unitless)
        force_velocity_relationship: MX | float
            The force velocity relationship value (unitless)
        Returns
        -------
        The derivative of the states in the tuple[MX] format
        """
        model = fes_model if fes_model else nlp.model
        dxdt_fun = model.system_dynamics

        if model.is_approximated:
            pulse_width = None
            cn_sum = controls[0]
            a_scale = controls[1]
        else:
            pulse_width = model.get_pulse_width_parameters(nlp, parameters)

            if len(pulse_width) == 1 and len(nlp.model.stim_time) != 1:
                pulse_width = pulse_width * len(nlp.model.stim_time)
            cn_sum = None
            a_scale = None

        return DynamicsEvaluation(
            dxdt=dxdt_fun(
                cn=states[0],
                f=states[1],
                t=time,
                pulse_width=pulse_width,
                cn_sum=cn_sum,
                a_scale=a_scale,
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
        if self.is_approximated:
            StateConfigure().configure_cn_sum(ocp, nlp)
            StateConfigure().configure_a_calculation(ocp, nlp)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, dyn_func=self.dynamics)
