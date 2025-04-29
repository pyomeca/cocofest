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

class Marion2013ModelPulseWidthFrequency(DingModelPulseWidthFrequency):
    """
    Implementation of the Marion 2013 force-motion model for electrical stimulation

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
        A90: float = None,
        a_coef: float = None,
        b_coef: float = None,
        V1: float = None,
        V2: float = None,
        L_I: float = None,
        FM: float = None,
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
        A90_DEFAULT = 4920  # N/s (using same default as Ding's a_scale)
        A_COEF_DEFAULT = -0.0002  # deg^-2 (estimated from paper's figures)
        B_COEF_DEFAULT = 0.05  # deg^-1 (estimated from paper's figures)
        V1_DEFAULT = 10  # N.deg^-2 (estimated from paper's figures)
        V2_DEFAULT = 0.1  # deg^-1 (estimated from paper's figures)
        L_I_DEFAULT = 0.2  # Lumped parameter (estimated from paper's figures)
        FM_DEFAULT = 5  # N (estimated from paper's figures)
        
        # Model parameters with default values
        self.A90 = A90 if A90 is not None else A90_DEFAULT
        self.a_coef = a_coef if a_coef is not None else A_COEF_DEFAULT
        self.b_coef = b_coef if b_coef is not None else B_COEF_DEFAULT
        self.V1 = V1 if V1 is not None else V1_DEFAULT
        self.V2 = V2 if V2 is not None else V2_DEFAULT
        self.L_I = L_I if L_I is not None else L_I_DEFAULT
        self.FM = FM if FM is not None else FM_DEFAULT

    @property
    def name_dof(self, with_muscle_name: bool = False) -> list[str]:
        muscle_name = "_" + self.muscle_name if self.muscle_name and with_muscle_name else ""
        return [
            "CN" + muscle_name,
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
            "A90": self.A90,
            "a_coef": self.a_coef,
            "b_coef": self.b_coef,
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
                "A90": self.A90,
                "a_coef": self.a_coef,
                "b_coef": self.b_coef,
                "V1": self.V1,
                "V2": self.V2,
                "L_I": self.L_I,
                "FM": self.FM,
                "stim_time": self.stim_time,
                "previous_stim": self.previous_stim,
            },
        )
    
    def calculate_A(self, theta: MX) -> MX:
        """Calculate angle-dependent scaling term A"""
        return self.A90 * (self.a_coef * (90 - theta)**2 + self.b_coef * (90 - theta) + 1)
    
    def calculate_G(self, theta: MX, dtheta_dt: MX) -> MX:
        """Calculate velocity-dependent scaling term G"""
        return self.V1 * theta * MX.exp(-self.V2 * theta) * dtheta_dt

    def system_dynamics(
        self,
        cn: MX,
        f: MX,
        theta: MX,
        dtheta_dt: MX,
        t: MX = None,
        t_stim_prev: MX = None,
        Fload: MX = 0.0,
        force_length_relationship: MX | float = 0,
        force_velocity_relationship: MX | float = 0,
        passive_force_relationship: MX | float = 0,
    ) -> MX:
        """
        The system dynamics that describes the model.

        Parameters
        ----------
        cn: MX
            The normalized Ca2+-troponin complex concentration
        f: MX
            The instantaneous force
        theta: MX
            The flexion angle
        dtheta_dt: MX
            The angular velocity
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_prev: MX
            The time of the previous stimulation (ms)
        Fload: MX
            External load force (N)
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
        # Get CN dynamics from Ding model
        cn_dot = self.calculate_cn_dot(cn, t, t_stim_prev)
            
        # Calculate Marion-specific force scaling terms
        A = self.calculate_A(theta)
        G = self.calculate_G(theta, dtheta_dt)
        
        # Use Ding's force equation with G+A as the scaling term instead of a_scale
        f_dot = self.f_dot_fun(
            cn,
            f,
            G + A,  # Replace a_scale with G+A from Marion 2013
            self.tau1_rest,
            self.km_rest,
            force_length_relationship=force_length_relationship,
            force_velocity_relationship=force_velocity_relationship,
            passive_force_relationship=passive_force_relationship,
        )
        
        # Motion dynamics specific to Marion 2013
        lambda_angle = 90.0 - theta  # Resting angle adjustment
        d2theta_dt2 = (self.L_I * ((Fload + self.FM) * MX.cos(MX.pi/180 * (theta + lambda_angle)) - f))
        
        return vertcat(cn_dot, f_dot, dtheta_dt, d2theta_dt2)

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
            The state of the system CN, F, theta, dtheta_dt
        controls: MX
            The controls of the system
        parameters: MX
            The parameters acting on the system
        algebraic_states: MX
            The stochastic variables of the system
        numerical_timeseries: MX
            The numerical timeseries of the system
        nlp: NonLinearProgram
            A reference to the phase
        fes_model: Marion2013ModelPulseWidthFrequency
            The current phase fes model

        Returns
        -------
        The derivative of the states in the tuple[MX] format
        """
        model = fes_model if fes_model else nlp.model
        # TODO: Check if it is linked to a model with q or qdot else raise an error
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        dxdt_fun = model.system_dynamics

        return DynamicsEvaluation(
            dxdt=dxdt_fun(
                cn=states[0],
                f=states[1],
                theta=q,
                dtheta_dt=qdot,
                t=time,
                t_stim_prev=numerical_timeseries,
                Fload=controls[0] if controls.shape[0] > 0 else 0.0,
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
        contact_type: tuple = (),
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
        contact_type: tuple
            The type of contact to use for the model
        """
        StateConfigure().configure_all_fes_model_states(ocp, nlp, fes_model=self)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, dyn_func=self.dynamics) 