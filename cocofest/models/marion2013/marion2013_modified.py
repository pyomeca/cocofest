from typing import Callable
from casadi import MX, vertcat, exp, cos, pi
import numpy as np

from bioptim import (
    DynamicsEvaluation,
    NonLinearProgram,
)
from cocofest.models.marion2009.marion2009_modified import Marion2009ModelPulseWidthFrequency


class Marion2013ModelPulseWidthFrequency(Marion2009ModelPulseWidthFrequency):
    """
    Implementation of the Marion 2013 force-motion model for electrical stimulation

    Warning: This model was not validated from Marion's experiment as the pulse with is added.
    This model should be used with caution.

    Marion, M. S., Wexler, A. S., & Hull, M. L. (2013).
    Predicting non-isometric fatigue induced by electrical stimulation pulse trains as a function of pulse duration.
    Journal of neuroengineering and rehabilitation, 10, 1-16.
    """

    def __init__(
        self,
        model_name: str = "marion_2013",
        muscle_name: str = None,
        stim_time: list[float] = None,
        previous_stim: dict = None,
        sum_stim_truncation: int = 20,
    ):
        # Initialize Ding 2007 parent class
        super().__init__(
            model_name=model_name,
            muscle_name=muscle_name,
            stim_time=stim_time,
            previous_stim=previous_stim,
            sum_stim_truncation=sum_stim_truncation,
        )

        # Default values from Marion 2013 paper
        A_REST_DEFAULT = 2100  # N/s (using same default as Ding's a_scale)
        TAU1_REST_DEFAULT = 0.0361  # Value from Marion's 2013 article in figure n°3 (s)
        TAU2_DEFAULT = 0.0521  # Value from Marion's 2013 article in figure n°3 (s)
        KM_REST_DEFAULT = 0.352  # Value from Marion's 2013 article in figure n°3 (unitless)
        TAUC_DEFAULT = 0.020  # Value from Marion's 2013 article in figure n°3 (s)
        R0_KM_RELATIONSHIP_DEFAULT = 2  # Value from Marion's 2013 article in figure n°3 (unitless)
        A_THETA_DEFAULT = -0.000449  # Value from Marion's 2013 article in figure n°3 (deg^-2)
        B_THETA_DEFAULT = 0.0344  # Value from Marion's 2013 article in figure n°3 (deg^-1)
        V1_DEFAULT = 0.371  # Value from Marion's 2013 article in figure n°3 (N.deg^-2)
        V2_DEFAULT = 0.0229  # Value from Marion's 2013 article in figure n°3 (deg^-1)
        L_I_DEFAULT = 9.85  # Value from Marion's 2013 article in figure n°3 (kg^-1.m^-1)
        FM_DEFAULT = 247.5  # Value from Marion's 2013 article in figure n°3 (N)

        # Model parameters with default values
        self.a_rest = A_REST_DEFAULT
        self.a_scale = A_REST_DEFAULT
        self.km_rest = KM_REST_DEFAULT
        self.tau1_rest = TAU1_REST_DEFAULT
        self.tau2 = TAU2_DEFAULT
        self.tauc = TAUC_DEFAULT
        self.r0_km_relationship = R0_KM_RELATIONSHIP_DEFAULT
        self.a_theta = A_THETA_DEFAULT
        self.b_theta = B_THETA_DEFAULT
        self.V1 = V1_DEFAULT
        self.V2 = V2_DEFAULT
        self.L_I = L_I_DEFAULT
        self.FM = FM_DEFAULT

    @property
    def name_dofs(self, with_muscle_name: bool = False) -> list[str]:
        muscle_name = "_" + self.muscle_name if self.muscle_name is not None else ""
        return [
            "Cn" + muscle_name,
            "F" + muscle_name,
            "theta" + muscle_name,
            "dtheta_dt" + muscle_name,
        ]

    @property
    def nb_state(self) -> int:
        return 4

    @property
    def identifiable_parameters(self):
        return {
            "a": self.a_rest,
            "a_theta": self.a_theta,
            "b_theta": self.b_theta,
            "V1": self.V1,
            "V2": self.V2,
            "L_I": self.L_I,
            "FM": self.FM,
        }

    def standard_rest_values(self) -> np.array:
        """
        Returns
        -------
        The rested values of CN, F, theta, dtheta_dt
        """
        return np.array([[0], [0], [90], [0]])

    def serialize(self) -> tuple[Callable, dict]:
        return (
            Marion2013ModelPulseWidthFrequency,
            {
                "a": self.a_rest,
                "a_theta": self.a_theta,
                "b_theta": self.b_theta,
                "V1": self.V1,
                "V2": self.V2,
                "L_I": self.L_I,
                "FM": self.FM,
                "stim_time": self.stim_time,
                "previous_stim": self.previous_stim,
            },
        )

    def calculate_A(self, a: MX, theta: MX) -> MX:
        """Calculate angle-dependent scaling term A"""
        return a * (self.a_theta * (90 - theta) ** 2 + self.b_theta * (90 - theta) + 1)

    def calculate_G(self, theta: MX, dtheta_dt: MX) -> MX:
        """Calculate velocity-dependent scaling term G"""
        return self.V1 * theta * exp(-self.V2 * theta) * dtheta_dt

    def calculate_acceleration(self, theta: MX, lambda_angle: MX, f: MX, Fload: MX) -> MX:
        return self.L_I * ((Fload + self.FM) * cos(pi / 180 * (theta + lambda_angle)) - f)

    def system_dynamics(
        self,
        time: MX,
        states: MX,
        controls: MX,
        numerical_timeseries: MX,
    ) -> MX:
        """
        The system dynamics that describes the model.

        Parameters
        ----------
        time: MX
            The system's current node time
        states: MX
            The state of the system CN, F, theta, dtheta
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
        pulse_width = controls[0]
        Fload = controls[1] if controls.shape[0] > 1 else 0.0
        t_stim_prev = numerical_timeseries

        # Get CN dynamics from Ding model
        cn_dot = self.calculate_cn_dot(cn, t, t_stim_prev)

        # Calculate base a_scale with pulse width dependency
        base_a_scale = self.a_calculation(a_scale=self.a_scale, pulse_width=pulse_width)

        # Calculate Marion-specific force scaling terms
        A = self.calculate_A(base_a_scale, theta)
        G = self.calculate_G(theta, dtheta_dt)

        # Use Ding's force equation with G+A as the scaling term instead of a_scale
        f_dot = self.f_dot_fun(
            cn,
            f,
            G + A,  # Replace a_scale with G+A from Marion 2013
            self.tau1_rest,
            self.km_rest,
        )

        # Motion dynamics specific to Marion 2013
        lambda_angle = 90.0 - theta  # Resting angle adjustment
        d2theta_dt2 = self.calculate_acceleration(
            theta,
            lambda_angle,
            f,
            Fload,
        )

        return vertcat(cn_dot, f_dot, dtheta_dt, d2theta_dt2)
