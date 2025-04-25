from typing import Callable

from casadi import MX, vertcat
import numpy as np

from bioptim import (
    ConfigureProblem,
    DynamicsEvaluation,
    NonLinearProgram,
    OptimalControlProgram,
)
from .ding2007 import DingModelPulseWidthFrequency
from .state_configure import StateConfigure


class DingModelPulseWidthFrequencyWithFatigue(DingModelPulseWidthFrequency):
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
        model_name: str = "ding_2007_with_fatigue",
        muscle_name: str = None,
        stim_time: list[float] = None,
        previous_stim: dict = None,
        sum_stim_truncation: int = 20,
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
        super(DingModelPulseWidthFrequencyWithFatigue, self).__init__(
            model_name=model_name,
            muscle_name=muscle_name,
            stim_time=stim_time,
            previous_stim=previous_stim,
            sum_stim_truncation=sum_stim_truncation,
        )
        self._with_fatigue = True
        self.stim_time = stim_time
        self.fmax = 315

        # --- Default values --- #
        ALPHA_A_DEFAULT = -4.0 * 10e-2  # Value from Ding's experimentation [1] (s^-2)
        TAU_FAT_DEFAULT = 127  # Value from Ding's experimentation [1] (s)
        ALPHA_TAU1_DEFAULT = 2.1 * 10e-6  # Value from Ding's experimentation [1] (N^-1)
        ALPHA_KM_DEFAULT = 1.9 * 10e-6  # Value from Ding's experimentation [1] (s^-1.N^-1)

        # ---- Fatigue models ---- #
        self.alpha_a = ALPHA_A_DEFAULT
        self.alpha_tau1 = ALPHA_TAU1_DEFAULT
        self.tau_fat = TAU_FAT_DEFAULT
        self.alpha_km = ALPHA_KM_DEFAULT

    # ---- Absolutely needed methods ---- #
    @property
    def name_dof(self, with_muscle_name: bool = False) -> list[str]:
        muscle_name = "_" + self.muscle_name if self.muscle_name and with_muscle_name else ""
        return [
            "Cn" + muscle_name,
            "F" + muscle_name,
            "A" + muscle_name,
            "Tau1" + muscle_name,
            "Km" + muscle_name,
        ]

    @property
    def nb_state(self) -> int:
        return 5

    @property
    def identifiable_parameters(self):
        return {
            "a_scale": self.a_scale,
            "tau1_rest": self.tau1_rest,
            "km_rest": self.km_rest,
            "tau2": self.tau2,
            "pd0": self.pd0,
            "pdt": self.pdt,
            "alpha_a": self.alpha_a,
            "alpha_tau1": self.alpha_tau1,
            "alpha_km": self.alpha_km,
            "tau_fat": self.tau_fat,
        }

    def standard_rest_values(self) -> np.array:
        """
        Returns
        -------
        The rested values of Cn, F, A, Tau1, Km
        """
        return np.array([[0], [0], [self.a_scale], [self.tau1_rest], [self.km_rest]])

    def serialize(self) -> tuple[Callable, dict]:
        # This is where you can serialize your models
        # This is useful if you want to save your models and load it later
        return (
            DingModelPulseWidthFrequencyWithFatigue,
            {
                "tauc": self.tauc,
                "a_rest": self.a_rest,
                "tau1_rest": self.tau1_rest,
                "km_rest": self.km_rest,
                "tau2": self.tau2,
                "alpha_a": self.alpha_a,
                "alpha_tau1": self.alpha_tau1,
                "alpha_km": self.alpha_km,
                "tau_fat": self.tau_fat,
                "a_scale": self.a_scale,
                "pd0": self.pd0,
                "pdt": self.pdt,
                "stim_time": self.stim_time,
                "previous_stim": self.previous_stim,
            },
        )

    def system_dynamics(
        self,
        cn: MX,
        f: MX,
        a: MX = None,
        tau1: MX = None,
        km: MX = None,
        t: MX = None,
        t_stim_prev: MX = None,
        pulse_width: MX = None,
        force_length_relationship: MX | float = 1,
        force_velocity_relationship: MX | float = 1,
        passive_force_relationship: MX | float = 0,
    ) -> MX:
        """
        The system dynamics is the function that describes the models.

        Parameters
        ----------
        cn: MX
            The value of the ca_troponin_complex (unitless)
        f: MX
            The value of the force (N)
        a: MX
            The value of the scaling factor (unitless)
        tau1: MX
            The value of the time_state_force_no_cross_bridge (ms)
        km: MX
            The value of the cross_bridges (unitless)
        t: MX
            The current time at which the dynamics is evaluated (s)
        t_stim_prev: MX
            The time of the previous stimulation (s)
        pulse_width: MX
            The time of the impulse (s)
        force_length_relationship: MX | float
            The force length relationship value (unitless)
        force_velocity_relationship: MX | float
            The force velocity relationship value (unitless)
        passive_force_relationship: MX | float
            The passive force relationship value (unitless)

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        cn_dot = self.calculate_cn_dot(cn, t, t_stim_prev)
        a_scale = self.a_calculation(a_scale=a, pulse_width=pulse_width)

        f_dot = self.f_dot_fun(
            cn,
            f,
            a_scale,
            tau1,
            km,
            force_length_relationship=force_length_relationship,
            force_velocity_relationship=force_velocity_relationship,
            passive_force_relationship=passive_force_relationship,
        )  # Equation n°2 from Ding's 2003 article

        a_dot = self.a_dot_fun(a, f)
        tau1_dot = self.tau1_dot_fun(tau1, f)  # Equation n°9 from Ding's 2003 article
        km_dot = self.km_dot_fun(km, f)  # Equation n°11 from Ding's 2003 article
        return vertcat(cn_dot, f_dot, a_dot, tau1_dot, km_dot)

    def a_dot_fun(self, a: MX, f: MX) -> MX | float:
        """
        Parameters
        ----------
        a: MX
            The previous step value of scaling factor (unitless)
        f: MX
            The previous step value of force (N)

        Returns
        -------
        The value of the derivative scaling factor (unitless)
        """
        return -(a - self.a_scale) / self.tau_fat + self.alpha_a * f  # Equation n°5

    def tau1_dot_fun(self, tau1: MX, f: MX) -> MX | float:
        """
        Parameters
        ----------
        tau1: MX
            The previous step value of time_state_force_no_cross_bridge (ms)
        f: MX
            The previous step value of force (N)

        Returns
        -------
        The value of the derivative time_state_force_no_cross_bridge (ms)
        """
        return -(tau1 - self.tau1_rest) / self.tau_fat + self.alpha_tau1 * f  # Equation n°9

    def km_dot_fun(self, km: MX, f: MX) -> MX | float:
        """
        Parameters
        ----------
        km: MX
            The previous step value of cross_bridges (unitless)
        f: MX
            The previous step value of force (N)

        Returns
        -------
        The value of the derivative cross_bridges (unitless)
        """
        return -(km - self.km_rest) / self.tau_fat + self.alpha_km * f  # Equation n°11

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
        passive_force_relationship: MX | float = 0,
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
        fes_model: DingModelPulseWidthFrequencyWithFatigue
            The current phase fes model
        force_length_relationship: MX | float
            The force length relationship value (unitless)
        force_velocity_relationship: MX | float
            The force velocity relationship value (unitless)
        passive_force_relationship: MX | float
            The passive force relationship value (unitless)
        Returns
        -------
        The derivative of the states in the tuple[MX] format
        """
        model = fes_model if fes_model else nlp.model
        dxdt_fun = model.system_dynamics

        return DynamicsEvaluation(
            dxdt=dxdt_fun(
                cn=states[0],
                f=states[1],
                a=states[2],
                tau1=states[3],
                km=states[4],
                t=time,
                t_stim_prev=numerical_timeseries,
                pulse_width=controls[0],
                force_length_relationship=force_length_relationship,
                force_velocity_relationship=force_velocity_relationship,
                passive_force_relationship=passive_force_relationship,
            ),
            defects=None,
        )

    def declare_ding_variables(
        self,
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        numerical_data_timeseries: dict[str, np.ndarray] = None,
        contact_type: list = (),
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
        contact_type: list
            A list of contact types. This is used to define the contact forces in the dynamics. Not used in this model.
        """
        StateConfigure().configure_all_fes_model_states(ocp, nlp, fes_model=self)
        StateConfigure().configure_last_pulse_width(ocp, nlp)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, dyn_func=self.dynamics)
