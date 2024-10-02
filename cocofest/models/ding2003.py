from typing import Callable

import numpy as np
from casadi import MX, exp, vertcat

from bioptim import (
    ConfigureProblem,
    DynamicsEvaluation,
    NonLinearProgram,
    OptimalControlProgram,
)

from .state_configue import StateConfigure
from .fes_model import FesModel


class DingModelFrequency(FesModel):
    """
    This is a custom model of the Bioptim package. As CustomModel, some methods are mandatory and must be implemented.
    to make it work with bioptim.

    This is the Ding 2003 model using the stimulation frequency as a control input.

    Notes
    -----

    Ding, J., Wexler, A. S., & Binder-Macleod, S. A. (2003).
    Mathematical models for fatigue minimization during functional electrical stimulation.
    Journal of Electromyography and Kinesiology, 13(6), 575-588.
    """

    def __init__(
        self,
        model_name: str = "ding2003",
        muscle_name: str = None,
        sum_stim_truncation: int = None,
    ):
        super().__init__()
        self._model_name = model_name
        self._muscle_name = muscle_name
        self._sum_stim_truncation = sum_stim_truncation
        self._with_fatigue = False
        self._is_integrate = False
        self.pulse_apparition_time = None
        self.stim_prev = []
        # ---- Custom values for the example ---- #
        self.tauc = 0.020  # Value from Ding's experimentation [1] (s)
        self.r0_km_relationship = 1.04  # (unitless)
        # ---- Different values for each person ---- #
        # ---- Force models ---- #
        self.a_rest = 3009  # Value from Ding's experimentation [1] (N.s-1)
        self.tau1_rest = 0.050957  # Value from Ding's experimentation [1] (s)
        self.tau2 = 0.060  # Close value from Ding's experimentation [2] (s)
        self.km_rest = 0.103  # Value from Ding's experimentation [1] (unitless)

    def set_a_rest(self, model, a_rest: MX | float):
        # models is required for bioptim compatibility
        self.a_rest = a_rest

    def set_km_rest(self, model, km_rest: MX | float):
        self.km_rest = km_rest

    def set_tau1_rest(self, model, tau1_rest: MX | float):
        self.tau1_rest = tau1_rest

    def set_tau2(self, model, tau2: MX | float):
        self.tau2 = tau2

    def standard_rest_values(self) -> np.array:
        """
        Returns
        -------
        The rested values of the states Cn, F
        """
        return np.array([[0], [0]])

    # ---- Absolutely needed methods ---- #
    def serialize(self) -> tuple[Callable, dict]:
        # This is where you can serialize your models
        # This is useful if you want to save your models and load it later
        return (
            DingModelFrequency,
            {
                "tauc": self.tauc,
                "a_rest": self.a_rest,
                "tau1_rest": self.tau1_rest,
                "km_rest": self.km_rest,
                "tau2": self.tau2,
            },
        )

    # ---- Needed for the example ---- #
    @property
    def name_dof(self, with_muscle_name: bool = False) -> list[str]:
        muscle_name = (
            "_" + self.muscle_name if self.muscle_name and with_muscle_name else ""
        )
        return ["Cn" + muscle_name, "F" + muscle_name]

    @property
    def nb_state(self) -> int:
        return 2

    @property
    def model_name(self) -> None | str:
        return self._model_name

    @property
    def muscle_name(self) -> None | str:
        return self._muscle_name

    @property
    def with_fatigue(self):
        return self._with_fatigue

    @property
    def identifiable_parameters(self):
        return {
            "a_rest": self.a_rest,
            "tau1_rest": self.tau1_rest,
            "km_rest": self.km_rest,
            "tau2": self.tau2,
        }

    # ---- Model's dynamics ---- #
    def system_dynamics(
        self,
        cn: MX,
        f: MX,
        cn_sum: MX | float = None,
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
        cn_dot = self.cn_dot_fun(cn, cn_sum=cn_sum)  # Equation n°1
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

    def exp_time_fun(self, t: MX, t_stim_i: MX) -> MX | float:
        """
        Parameters
        ----------
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_i: MX
            Time when the stimulation i occurred (ms)

        Returns
        -------
        A part of the n°1 equation
        """
        return exp(-(t - t_stim_i) / self.tauc)  # Part of Eq n°1

    def ri_fun(self, r0: MX | float, time_between_stim: MX) -> MX | float:
        """
        Parameters
        ----------
        r0: MX | float
            Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli (unitless)
        time_between_stim: MX
            Time between the last stimulation i and the current stimulation i (ms)

        Returns
        -------
        A part of the n°1 equation
        """
        return 1 + (r0 - 1) * exp(-time_between_stim / self.tauc)  # Part of Eq n°1

    def cn_sum_fun(self, r0: MX | float, t: MX, t_stim_prev: list[MX], lambda_i: list[MX]) -> MX | float:
        """
        Parameters
        ----------
        r0: MX | float
            Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli (unitless)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_prev: list[MX]
            The time list of the previous stimulations (ms)

        Returns
        -------
        A part of the n°1 equation
        """
        sum_multiplier = 0

        for i in range(len(t_stim_prev)):
            previous_phase_time = t_stim_prev[i] - t_stim_prev[i - 1]
            ri = 1 if i == 0 else self.ri_fun(r0, previous_phase_time)  # Part of Eq n°1
            exp_time = self.exp_time_fun(t, t_stim_prev[i])  # Part of Eq n°1
            sum_multiplier += ri * exp_time * lambda_i[i]
        return sum_multiplier

    def cn_dot_fun(self, cn: MX, cn_sum: MX) -> MX | float:
        """
        Parameters
        ----------
        cn: MX
            The previous step value of ca_troponin_complex (unitless)

        Returns
        -------
        The value of the derivative ca_troponin_complex (unitless)
        """

        return (1 / self.tauc) * cn_sum - (cn / self.tauc)  # Equation n°1

    def f_dot_fun(
        self,
        cn: MX,
        f: MX,
        a: MX | float,
        tau1: MX | float,
        km: MX | float,
        force_length_relationship: MX | float = 1,
        force_velocity_relationship: MX | float = 1,
    ) -> MX | float:
        """
        Parameters
        ----------
        cn: MX
            The previous step value of ca_troponin_complex (unitless)
        f: MX
            The previous step value of force (N)
        a: MX | float
            The previous step value of scaling factor (unitless)
        tau1: MX | float
            The previous step value of time_state_force_no_cross_bridge (ms)
        km: MX | float
            The previous step value of cross_bridges (unitless)
        force_length_relationship: MX | float
            The force length relationship value (unitless)
        force_velocity_relationship: MX | float
            The force velocity relationship value (unitless)

        Returns
        -------
        The value of the derivative force (N)
        """
        return (
            (a * (cn / (km + cn)) - (f / (tau1 + self.tau2 * (cn / (km + cn)))))
            * force_length_relationship
            * force_velocity_relationship
        )  # Equation n°2

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
        fes_model: DingModelFrequency
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

    def set_pulse_apparition_time(self, value: list[MX]):
        """
        Sets the pulse apparition time for each pulse (phases) according to the ocp parameter "pulse_apparition_time"

        Parameters
        ----------
        value: list[MX]
            The pulse apparition time list (s)
        """
        self.pulse_apparition_time = value
