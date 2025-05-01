from typing import Callable

import numpy as np
from casadi import MX, vertcat


from bioptim import (
    ConfigureProblem,
    DynamicsEvaluation,
    NonLinearProgram,
    OptimalControlProgram,
)
from .ding2007 import DingModelFrequency
from .state_configure import StateConfigure


class Marion2009ModelFrequency(DingModelFrequency):
    """
    This is a custom model that inherits from DingModelFrequency.

    This implements the Marion 2009 model which adds angle dependency to the force-fatigue relationship.
    
    Marion, M. S., Wexler, A. S., Hull, M. L., & Binder-Macleod, S. A. (2009).
    Predicting the effect of muscle length on fatigue during electrical stimulation.
    Muscle & Nerve: Official Journal of the American Association of Electrodiagnostic Medicine, 40(4), 573-581.
    """

    def __init__(
        self,
        model_name: str = "marion_2009",
        muscle_name: str = None,
        stim_time: list[float] = None,
        previous_stim: dict = None,
        sum_stim_truncation: int = 20,
    ):

        super().__init__(
            model_name=model_name,
            muscle_name=muscle_name,
            stim_time=stim_time,
            previous_stim=previous_stim,
            sum_stim_truncation=sum_stim_truncation,
        )
        
        # --- Default values --- #
        A_THETA_DEFAULT = 1473  # Value from Marion's 2009 article in figure n°3 (N/s)
        TAU1_REST_DEFAULT = 0.04298  # Value from Marion's 2009 article in figure n°3 (s)
        TAU2_DEFAULT = 0.10536  # Value from Marion's 2009 article in figure n°3 (s)
        KM_REST_DEFAULT = 0.128  # Value from Marion's 2009 article in figure n°3 (unitless)
        TAUC_DEFAULT = 0.020  # Value from Marion's 2009 article in figure n°3 (s)
        R0_KM_RELATIONSHIP_DEFAULT = 1.168  # Value from Marion's 2009 article in figure n°3 (unitless)
        A_COEF_DEFAULT = -0.000449  # Value from Marion's 2013 article in figure n°3 (deg^-2), couldn't be found in 2009
        B_COEF_DEFAULT = 0.0344  # Value from Marion's 2013 article in figure n°3 (deg^-1), couldn't be found in 2009

        # --- Model parameters with default values --- #
        self.tauc = TAUC_DEFAULT
        self.a_rest = A_THETA_DEFAULT
        self.tau1_rest = TAU1_REST_DEFAULT
        self.km_rest = KM_REST_DEFAULT
        self.tau2 = TAU2_DEFAULT
        self.r0_km_relationship = R0_KM_RELATIONSHIP_DEFAULT

        # Angle-specific parameters
        self.theta_star = 90  # Reference angle of identified parameters
        self.a_theta = A_COEF_DEFAULT
        self.b_theta = B_COEF_DEFAULT
        self.activate_residual_torque = False

    @property
    def identifiable_parameters(self):
        params = super().identifiable_parameters
        params.update({
            "theta_star": self.theta_star,
            "a_theta": self.a_theta,
            "b_theta": self.b_theta,
        })
        return params

    def serialize(self) -> tuple[Callable, dict]:
        base_params = super().serialize()[1]
        base_params.update({
            "theta_star": self.theta_star,
            "a_theta": self.a_theta,
            "b_theta": self.b_theta,
        })
        return (Marion2009ModelFrequency, base_params)

    def angle_scaling_factor(self, theta: MX) -> MX:
        """
        Calculate the angle-dependent scaling factor A(θ) according to equation 2a from Marion 2009.
        
        Parameters
        ----------
        theta: MX
            Current knee angle in degrees
            
        Returns
        -------
        The angle scaling factor (unitless)
        """
        delta_theta = self.theta_star - theta
        return 1 + self.a_theta * delta_theta**2 + self.b_theta * delta_theta

    def system_dynamics(
        self,
        cn: MX,
        f: MX,
        t: MX = None,
        t_stim_prev: list[float] | list[MX] = None,
        theta: MX = None,
    ) -> MX:
        """
        The system dynamics incorporating angle dependency.

        Parameters
        ----------
        cn: MX
            The value of the ca_troponin_complex (unitless)
        f: MX
            The value of the force (N)
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
        a = self.a_rest * angle_factor
        
        f_dot = self.f_dot_fun(
            cn,
            f,
            a,
            self.tau1_rest,
            self.km_rest,
        )
        
        return vertcat(cn_dot, f_dot)

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
        Functional electrical stimulation dynamic including angle dependency

        Parameters
        ----------
        time: MX
            The system's current node time
        states: MX
            The state of the system CN, F
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
        fes_model: Marion2009ModelFrequency
            The current phase fes model
        force_length_relationship: MX | float
            The force length relationship value (unitless), not considered for this model
        force_velocity_relationship: MX | float
            The force velocity relationship value (unitless), not considered for this model
        passive_force_relationship: MX | float
            The passive force coefficient of the muscle (unitless), not considered for this model
            
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
                t=time,
                t_stim_prev=numerical_timeseries,
                theta=controls[0] if controls.shape[0] > 0 else 90,
            ),
            defects=None,
        )
