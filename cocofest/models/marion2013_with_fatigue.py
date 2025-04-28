from typing import Callable
from casadi import MX, vertcat
import numpy as np

from bioptim import (
    ConfigureProblem,
    DynamicsEvaluation,
    NonLinearProgram,
    OptimalControlProgram,
)
from .marion2013 import Marion2013
from .state_configure import StateConfigure


class Marion2013WithFatigue(Marion2013):
    """
    Extension of Marion2013 model that includes fatigue effects
    Based on: Marion et al. (2013) - Predicting non-isometric fatigue induced by electrical 
    stimulation pulse trains as a function of pulse duration
    """
    
    def __init__(
        self,
        model_name: str = "marion_2013_with_fatigue",
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
        alpha_a: float = None,
        alpha_tau1: float = None,
        alpha_km: float = None,
        tau_fat: float = None,
    ):
        super().__init__(
            model_name=model_name,
            muscle_name=muscle_name,
            stim_time=stim_time,
            previous_stim=previous_stim,
            sum_stim_truncation=sum_stim_truncation,
            A90=A90,
            a_coef=a_coef,
            b_coef=b_coef,
            V1=V1,
            V2=V2,
            L_I=L_I,
            FM=FM,
        )
        self._with_fatigue = True
        
        # Default fatigue parameter values from Marion 2013 paper
        ALPHA_A_DEFAULT = -4.0e-2  # Force scaling factor for A90 (s^-2)
        ALPHA_KM_DEFAULT = 1.9e-6  # Force scaling factor for Km (s^-1.N^-1)
        ALPHA_TAU1_DEFAULT = 2.1e-6  # Force scaling factor for tau1 (N^-1)
        TAU_FAT_DEFAULT = 127  # Time constant for fatigue (s)
        
        # Fatigue model parameters
        self.alpha_a = alpha_a if alpha_a is not None else ALPHA_A_DEFAULT
        self.alpha_km = alpha_km if alpha_km is not None else ALPHA_KM_DEFAULT
        self.alpha_tau1 = alpha_tau1 if alpha_tau1 is not None else ALPHA_TAU1_DEFAULT
        self.tau_fat = tau_fat if tau_fat is not None else TAU_FAT_DEFAULT
        
        # Initial non-fatigue values
        self.A90_0 = self.A90
        self.km_0 = self.km_rest
        self.tau1_0 = self.tau1_rest

    @property
    def name_dof(self, with_muscle_name: bool = False) -> list[str]:
        muscle_name = "_" + self.muscle_name if self.muscle_name and with_muscle_name else ""
        return [
            "CN" + muscle_name,
            "F" + muscle_name,
            "theta" + muscle_name,
            "dtheta_dt" + muscle_name,
            "A90" + muscle_name,
            "Km" + muscle_name,
            "tau1" + muscle_name,
        ]

    @property
    def nb_state(self) -> int:
        return 7

    @property
    def identifiable_parameters(self):
        params = super().identifiable_parameters
        params.update({
            "alpha_a": self.alpha_a,
            "alpha_km": self.alpha_km,
            "alpha_tau1": self.alpha_tau1,
            "tau_fat": self.tau_fat,
        })
        return params

    def standard_rest_values(self) -> np.array:
        """
        Returns
        -------
        The rested values of all states including fatigue parameters
        """
        base_values = super().standard_rest_values()
        fatigue_values = np.array([[self.A90_0], [self.km_0], [self.tau1_0]])
        return np.vstack((base_values, fatigue_values))

    def serialize(self) -> tuple[Callable, dict]:
        base_dict = super().serialize()[1]
        base_dict.update({
            "alpha_a": self.alpha_a,
            "alpha_km": self.alpha_km,
            "alpha_tau1": self.alpha_tau1,
            "tau_fat": self.tau_fat,
        })
        return (Marion2013WithFatigue, base_dict)

    def a_dot_fun(self, a90: MX, f: MX) -> MX | float:
        """
        Parameters
        ----------
        a90: MX
            The previous step value of A90 scaling factor (unitless)
        f: MX
            The previous step value of force (N)

        Returns
        -------
        The value of the derivative A90 scaling factor (unitless)
        """
        return -(a90 - self.A90_0) / self.tau_fat + self.alpha_a * f

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
        return -(tau1 - self.tau1_0) / self.tau_fat + self.alpha_tau1 * f

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
        return -(km - self.km_0) / self.tau_fat + self.alpha_km * f

    def system_dynamics(
        self,
        cn: MX,
        f: MX,
        theta: MX,
        dtheta_dt: MX,
        A90: MX,
        Km: MX,
        tau1: MX,
        t: MX = None,
        t_stim_prev: MX = None,
        Fload: MX = 0.0,
        force_length_relationship: MX | float = 1,
        force_velocity_relationship: MX | float = 1,
        passive_force_relationship: MX | float = 0,
    ) -> MX:
        """
        The system dynamics including fatigue effects.

        Parameters
        ----------
        cn: MX
            The normalized Ca2+-troponin complex concentration
        f: MX
            The instantaneous force near ankle
        theta: MX
            The knee flexion angle
        dtheta_dt: MX
            The angular velocity
        A90: MX
            The current A90 value affected by fatigue
        Km: MX
            The current Km value affected by fatigue
        tau1: MX
            The current tau1 value affected by fatigue
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
            
        # Calculate Marion-specific force scaling terms with fatigue-affected A90
        A = self.calculate_A(theta)  # Using current A90 value
        G = self.calculate_G(theta, dtheta_dt)
        
        # Use Ding's force equation with G+A as the scaling term
        f_dot = self.f_dot_fun(
            cn,
            f,
            G + A,
            tau1,  # Use fatigue-affected tau1
            Km,    # Use fatigue-affected Km
            force_length_relationship=force_length_relationship,
            force_velocity_relationship=force_velocity_relationship,
            passive_force_relationship=passive_force_relationship,
        )
        
        # Motion dynamics
        lambda_angle = 90.0 - theta
        d2theta_dt2 = (self.L_I * ((Fload + self.FM) * MX.cos(MX.pi/180 * (theta + lambda_angle)) - f))
        
        # Fatigue dynamics
        dA90 = self.a_dot_fun(A90, f)
        dKm = self.km_dot_fun(Km, f)
        dtau1 = self.tau1_dot_fun(tau1, f)
        
        return vertcat(cn_dot, f_dot, dtheta_dt, d2theta_dt2, dA90, dKm, dtau1)

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
        Functional electrical stimulation dynamic with fatigue

        Parameters
        ----------
        time: MX
            The system's current node time
        states: MX
            The state of the system including fatigue states
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
        fes_model: Marion2013WithFatigue
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
                A90=states[4],
                Km=states[5],
                tau1=states[6],
                t=time,
                t_stim_prev=numerical_timeseries,
                Fload=controls[0] if controls.shape[0] > 0 else 0.0,
                force_length_relationship=force_length_relationship,
                force_velocity_relationship=force_velocity_relationship,
                passive_force_relationship=passive_force_relationship,
            ),
            defects=None,
        )

    def declare_model_variables(
        self,
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        numerical_data_timeseries: dict[str, np.ndarray] = None,
    ):
        """
        Tell the program which variables are states and controls.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node
        """
        StateConfigure().configure_all_fes_model_states(ocp, nlp, fes_model=self)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, dyn_func=self.dynamics) 