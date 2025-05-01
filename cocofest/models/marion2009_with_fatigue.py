from typing import Callable

import numpy as np
from casadi import MX, vertcat

from bioptim import (
    DynamicsEvaluation,
    NonLinearProgram,
    DynamicsFunctions,
)
from .marion2009 import Marion2009ModelFrequency


class Marion2009ModelFrequencyWithFatigue(Marion2009ModelFrequency):
    """
    This model extends the Marion 2009 model to include fatigue states.
    
    It combines the angle-dependent force-fatigue relationship from Marion 2009 with
    explicit fatigue states tracking (A, Tau1, Km) as done in Ding's models.
    
    Marion, M. S., Wexler, A. S., Hull, M. L., & Binder-Macleod, S. A. (2009).
    Predicting the effect of muscle length on fatigue during electrical stimulation.
    Muscle & Nerve: Official Journal of the American Association of Electrodiagnostic Medicine, 40(4), 573-581.
    """

    def __init__(
        self,
        model_name: str = "marion_2009_with_fatigue",
        muscle_name: str = None,
        stim_time: list[float] = None,
        previous_stim: dict = None,
        sum_stim_truncation: int = 20,
        tauc: float = None,
        a_rest: float = None,
        tau1_rest: float = None,
        km_rest: float = None,
        tau2: float = None,
        alpha_a: float = None,
        alpha_tau1: float = None,
        alpha_km: float = None,
        tau_fat: float = None,
        theta_star: float = 90.0,
        a_theta: float = None,
        b_theta: float = None,
    ):
        super().__init__(
            model_name=model_name,
            muscle_name=muscle_name,
            stim_time=stim_time,
            previous_stim=previous_stim,
            sum_stim_truncation=sum_stim_truncation,
            tauc=tauc,
            a_rest=a_rest,
            tau1_rest=tau1_rest,
            km_rest=km_rest,
            tau2=tau2,
            theta_star=theta_star,
            a_theta=a_theta,
            b_theta=b_theta,
        )
        self._with_fatigue = True
        self.stim_time = stim_time
        self.fmax = 315  # Maximum force value from Marion 2009 paper

        # --- Default values for fatigue parameters --- #
        ALPHA_A_DEFAULT = -2.006 * 10e-2  # Value from Marion's 2009 article in figure n°3 (s^-2)
        TAU_FAT_DEFAULT = 97.48  # Value from Marion's 2009 article in figure n°3 (s)
        # /!\ Minus sign removed from alpha_tau1 and alpha_km to enable correct calculation, difference with article
        ALPHA_TAU1_DEFAULT = 1.563 * 10e-5  # Value from Marion's 2009 article in figure n°3 (N^-1)
        ALPHA_KM_DEFAULT = 6.269 * 10e-6  # Value from Marion's 2009 article in figure n°3 (s^-1.N^-1)

        # ---- Fatigue model parameters ---- #
        self.alpha_a = alpha_a if alpha_a is not None else ALPHA_A_DEFAULT
        self.alpha_tau1 = alpha_tau1 if alpha_tau1 is not None else ALPHA_TAU1_DEFAULT
        self.tau_fat = tau_fat if tau_fat is not None else TAU_FAT_DEFAULT
        self.alpha_km = alpha_km if alpha_km is not None else ALPHA_KM_DEFAULT

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
        params = super().identifiable_parameters
        params.update({
            "alpha_a": self.alpha_a,
            "alpha_tau1": self.alpha_tau1,
            "alpha_km": self.alpha_km,
            "tau_fat": self.tau_fat,
        })
        return params

    def standard_rest_values(self) -> np.array:
        """
        Returns
        -------
        The rested values of Cn, F, A, Tau1, Km
        """
        return np.array([[0], [0], [self.a_rest], [self.tau1_rest], [self.km_rest]])

    def serialize(self) -> tuple[Callable, dict]:
        base_params = super().serialize()[1]
        base_params.update({
            "alpha_a": self.alpha_a,
            "alpha_tau1": self.alpha_tau1,
            "alpha_km": self.alpha_km,
            "tau_fat": self.tau_fat,
        })
        return (Marion2009ModelFrequencyWithFatigue, base_params)

    def system_dynamics(
        self,
        cn: MX,
        f: MX,
        a: MX = None,
        tau1: MX = None,
        km: MX = None,
        t: MX = None,
        t_stim_prev: list[float] | list[MX] = None,
        theta: MX = None,
    ) -> MX:
        """
        The system dynamics incorporating both angle dependency and fatigue states.

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
        t_stim_prev: list[float] | list[MX]
            The time list of the previous stimulations (s)
        theta: MX
            The current knee angle in degrees

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        cn_dot = self.calculate_cn_dot(cn, t, t_stim_prev)
        
        # Apply angle scaling
        angle_factor = self.angle_scaling_factor(theta)
        a = a * angle_factor
        
        f_dot = self.f_dot_fun(
            cn,
            f,
            a,
            tau1,
            km,
        )

        # Add fatigue state derivatives
        a_dot = self.a_dot_fun(a, f)
        tau1_dot = self.tau1_dot_fun(tau1, f)
        km_dot = self.km_dot_fun(km, f)
        
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
        return -(a - self.a_rest) / self.tau_fat + self.alpha_a * f

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
        return -(tau1 - self.tau1_rest) / self.tau_fat + self.alpha_tau1 * f

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
        return -(km - self.km_rest) / self.tau_fat + self.alpha_km * f

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
    ) -> DynamicsEvaluation:
        """
        Functional electrical stimulation dynamic including angle dependency and fatigue

        Parameters
        ----------
        time: MX
            The system's current node time
        states: MX
            The state of the system CN, F, A, Tau1, Km
        controls: MX
            The controls of the system: pulse_width, theta
        parameters: MX
            The parameters acting on the system, final time of each phase
        algebraic_states: MX
            The stochastic variables of the system, none
        numerical_timeseries: MX
            The numerical timeseries of the system
        nlp: NonLinearProgram
            A reference to the phase
        fes_model: Marion2009ModelFrequencyWithFatigue
            The current phase fes model
            
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
                theta=controls[0] if controls.shape[0] > 0 else 90,
            ),
            defects=None,
        )
