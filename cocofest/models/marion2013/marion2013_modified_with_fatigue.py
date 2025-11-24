from typing import Callable
from casadi import MX, vertcat
import numpy as np

from .marion2013_modified import Marion2013ModelPulseWidthFrequency


class Marion2013ModelPulseWidthFrequencyWithFatigue(Marion2013ModelPulseWidthFrequency):
    """
    Extension of Marion2013 model that includes fatigue effects

    Warning: This model was not validated from Marion's experiment as the pulse with is added.
    This model should be used with caution.

    Marion, M. S., Wexler, A. S., & Hull, M. L. (2013).
    Predicting non-isometric fatigue induced by electrical stimulation pulse trains as a function of pulse duration.
    Journal of neuroengineering and rehabilitation, 10, 1-16.
    """

    def __init__(
        self,
        model_name: str = "marion_2013_modified_with_fatigue",
        muscle_name: str = None,
        stim_time: list[float] = None,
        previous_stim: dict = None,
        sum_stim_truncation: int = 10,
    ):
        super().__init__(
            model_name=model_name,
            muscle_name=muscle_name,
            stim_time=stim_time,
            previous_stim=previous_stim,
            sum_stim_truncation=sum_stim_truncation,
        )
        self._with_fatigue = True

        # Default fatigue parameter values from Marion 2013 paper
        ALPHA_A_DEFAULT = -4.03e-2  # Force scaling factor for A90 (s^-2)
        TAU_FAT_DEFAULT = 99.4  # Time constant for fatigue (s)
        BETA_TAU1_DEFAULT = 8.54e-07  # Angular velocity x force scaling factor in fatigue model for force-motion model parameter τ1 (s.deg^-1.N^-1)
        # /!\ Minus sign removed from alpha_tau1 and alpha_km to enable correct calculation, difference with article
        ALPHA_TAU1_DEFAULT = 2.93e-6  # Force scaling factor for tau1 (N^-1)
        ALPHA_KM_DEFAULT = 1.36e-6  # Force scaling factor for Km (s^-1.N^-1)

        # Following values are not used in the original paper but are included for completeness:
        # The fatigue model was simplified by eliminating the parameters βA and βKm from the fatigue model and generating
        # a new equation for βτ1 (Equation 9) as a function of existing force-motion-fatigue model parameters (Marion et al., 2013).
        BETA_KM_DEFAULT = 0  # Angular velocity x force scaling factor in fatigue model for force-motion model parameter Km (deg^-1.N^-1)
        BETA_A_DEFAULT = 0  # Angular velocity x force scaling factor in fatigue model for force-motion model parameter A90 (s^-1.deg^-1)

        # Fatigue model parameters
        self.alpha_a = ALPHA_A_DEFAULT
        self.alpha_km = ALPHA_KM_DEFAULT
        self.alpha_tau1 = ALPHA_TAU1_DEFAULT
        self.tau_fat = TAU_FAT_DEFAULT
        self.beta_tau1 = BETA_TAU1_DEFAULT
        self.beta_km = BETA_KM_DEFAULT
        self.beta_a = BETA_A_DEFAULT

    @property
    def name_dofs(self, with_muscle_name: bool = False) -> list[str]:
        muscle_name = "_" + self.muscle_name if self.muscle_name is not None else ""
        return [
            "Cn" + muscle_name,
            "F" + muscle_name,
            "theta" + muscle_name,
            "dtheta_dt" + muscle_name,
            "A" + muscle_name,
            "Tau1" + muscle_name,
            "Km" + muscle_name,
        ]

    @property
    def nb_state(self) -> int:
        return 7

    @property
    def identifiable_parameters(self):
        params = super().identifiable_parameters
        params.update(
            {
                "alpha_a": self.alpha_a,
                "alpha_km": self.alpha_km,
                "alpha_tau1": self.alpha_tau1,
                "tau_fat": self.tau_fat,
            }
        )
        return params

    def standard_rest_values(self) -> np.array:
        """
        Returns
        -------
        The rested values of all states including fatigue parameters
        """
        base_values = super().standard_rest_values()
        fatigue_values = np.array([[self.a_rest], [self.tau1_rest], [self.km_rest]])
        return np.vstack((base_values, fatigue_values))

    def serialize(self) -> tuple[Callable, dict]:
        base_dict = super().serialize()[1]
        base_dict.update(
            {
                "alpha_a": self.alpha_a,
                "alpha_km": self.alpha_km,
                "alpha_tau1": self.alpha_tau1,
                "tau_fat": self.tau_fat,
            }
        )
        return (Marion2013ModelPulseWidthFrequencyWithFatigue, base_dict)

    def a_dot_fun(self, a: MX, f: MX, velocity: MX) -> MX | float:
        """
        Parameters
        ----------
        a: MX
            The previous step value of A scaling factor at 90° (unitless)
        f: MX
            The previous step value of force (N)
        velocity: MX
            The previous step value of angular velocity (deg/s)

        Returns
        -------
        The value of the derivative a scaling factor at 90° (unitless)
        """
        return -(a - self.a_rest) / self.tau_fat + (self.alpha_a + self.beta_a * velocity) * f

    def tau1_dot_fun(self, tau1: MX, f: MX, velocity: MX) -> MX | float:
        """
        Parameters
        ----------
        tau1: MX
            The previous step value of time_state_force_no_cross_bridge (ms)
        f: MX
            The previous step value of force (N)
        velocity: MX
            The previous step value of angular velocity (deg/s)

        Returns
        -------
        The value of the derivative time_state_force_no_cross_bridge (ms)
        """
        return -(tau1 - self.tau1_rest) / self.tau_fat + (self.alpha_tau1 + self.beta_tau1 * velocity) * f

    def km_dot_fun(self, km: MX, f: MX, velocity: MX) -> MX | float:
        """
        Parameters
        ----------
        km: MX
            The previous step value of cross_bridges (unitless)
        f: MX
            The previous step value of force (N)
        velocity: MX
            The previous step value of angular velocity (deg/s)

        Returns
        -------
        The value of the derivative cross_bridges (unitless)
        """
        return -(km - self.km_rest) / self.tau_fat + (self.alpha_km + self.beta_km * velocity) * f

    def system_dynamics(
        self,
        time: MX,
        states: MX,
        controls: MX,
        numerical_timeseries: MX,
    ) -> MX:
        """
        The system dynamics including fatigue effects.

        Parameters
        ----------
        time: MX
            The system's current node time
        states: MX
            The state of the system CN, F, theta, dtheta, A, Tau1, Km
        controls: MX
            The controls of the system, pulse_width, Fload
        numerical_timeseries: MX
            The numerical timeseries of the system

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        t = time
        cn = states[0]
        f = states[1]
        theta = states[2]
        dtheta_dt = states[3]
        a = states[4]
        tau1 = states[5]
        km = states[6]
        pulse_width = controls[0]
        Fload = controls[1] if controls.shape[0] > 1 else 0.0
        t_stim_prev = numerical_timeseries

        # Get CN dynamics from Ding model
        cn_dot = self.calculate_cn_dot(cn, t, t_stim_prev)

        # Calculate base a_scale with pulse width dependency
        base_a_scale = self.a_calculation(a_scale=self.a_scale, pulse_width=pulse_width)

        # Calculate Marion-specific force scaling terms with fatigue-affected A90
        A = self.calculate_A(base_a_scale, theta)  # Using current A90 value
        G = self.calculate_G(theta, dtheta_dt)

        # Use Ding's force equation with G+A as the scaling term
        f_dot = self.f_dot_fun(
            cn,
            f,
            G + A,
            tau1,
            km,
        )

        # Motion dynamics
        lambda_angle = 90.0 - theta
        d2theta_dt2 = self.calculate_acceleration(
            theta,
            lambda_angle,
            f,
            Fload,
        )

        # Fatigue dynamics
        a_dot = self.a_dot_fun(a, f, dtheta_dt)
        km_dot = self.km_dot_fun(km, f, dtheta_dt)
        tau1_dot = self.tau1_dot_fun(tau1, f, dtheta_dt)

        return vertcat(cn_dot, f_dot, dtheta_dt, d2theta_dt2, a_dot, tau1_dot, km_dot)
