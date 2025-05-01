from typing import Callable

import numpy as np
from casadi import MX, SX, vertcat, Function

from biorbd_casadi import GeneralizedCoordinates

from bioptim import (
    ConfigureProblem,
    DynamicsEvaluation,
    NonLinearProgram,
    OptimalControlProgram,
    DynamicsFunctions,
)
from .ding2007 import DingModelPulseWidthFrequency
from .state_configure import StateConfigure


class Marion2009ModelPulseWidthFrequency(DingModelPulseWidthFrequency):
    """
    This is a custom model that inherits from DingModelPulseWidthFrequency.
    
    This implements the Marion 2009 model which adds angle dependency to the force-fatigue relationship.

    Warning: This model was not validated from Marion's experiment as the pulse with is added.
    This model should be used with caution.
    
    Marion, M. S., Wexler, A. S., Hull, M. L., & Binder-Macleod, S. A. (2009).
    Predicting the effect of muscle length on fatigue during electrical stimulation.
    Muscle & Nerve: Official Journal of the American Association of Electrodiagnostic Medicine, 40(4), 573-581.
    """

    def __init__(
        self,
        model_name: str = "marion_2009_modified",
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
        theta_star: float = 90.0,  # Reference angle in degrees (90° in the paper)
        a_theta: float = None,  # Parabolic coefficient a
        b_theta: float = None,  # Parabolic coefficient b
        bio_model=None,  # The biorbd model to use for q and qdot
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
            pd0=pd0,
            pdt=pdt,
            a_scale=a_scale,
        )

        # --- Default values --- #
        A_THETA_DEFAULT = 1473  # Value from Marion's 2009 article in figure n°3 (N/s)
        TAU1_REST_DEFAULT = 0.04298  # Value from Marion's 2009 article in figure n°3 (s)
        TAU2_DEFAULT = 0.10536  # Value from Marion's 2009 article in figure n°3 (s)
        KM_REST_DEFAULT = 0.128  # Value from Marion's 2009 article in figure n°3 (unitless)
        TAUC_DEFAULT = 0.020  # Value from Marion's 2009 article in figure n°3 (s)
        R0_KM_RELATIONSHIP_DEFAULT = 1.168  # Value from Marion's 2009 article in figure n°3 (unitless)
        PD0_DEFAULT = 0.000131405  # Value from Ding's 2007 article (s)
        PDT_DEFAULT = 0.000194138  # Value from Ding's 2007 article (s)

        # --- Model parameters with default values --- #
        self.tauc = tauc if tauc is not None else TAUC_DEFAULT
        self.a_rest = a_rest if a_rest is not None else A_THETA_DEFAULT
        self.tau1_rest = tau1_rest if tau1_rest is not None else TAU1_REST_DEFAULT
        self.km_rest = km_rest if km_rest is not None else KM_REST_DEFAULT
        self.tau2 = tau2 if tau2 is not None else TAU2_DEFAULT
        self.r0_km_relationship = R0_KM_RELATIONSHIP_DEFAULT
        self.pd0 = PD0_DEFAULT if pd0 is not None else PD0_DEFAULT
        self.pdt = PDT_DEFAULT if pdt is not None else PDT_DEFAULT
        
        # Angle-specific parameters
        self.theta_star = theta_star  # Reference angle
        self.a_theta = a_theta if a_theta is not None else -0.0001  # Default from paper
        self.b_theta = b_theta if b_theta is not None else 0.01  # Default from paper
        self.activate_residual_torque = False
        self.bio_model = bio_model

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
        return (Marion2009ModelPulseWidthFrequency, base_params)

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
        pulse_width: MX = None,
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
        pulse_width: MX
            The pulsation duration of the current stimulation (s)
        theta: MX
            The current knee angle in degrees

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        cn_dot = self.calculate_cn_dot(cn, t, t_stim_prev)
        
        # Calculate base a_scale with pulse width dependency
        base_a_scale = self.a_calculation(a_scale=self.a_scale, pulse_width=pulse_width)
        
        # Apply angle scaling
        angle_factor = self.angle_scaling_factor(theta)
        a_scale = base_a_scale * angle_factor
        
        f_dot = self.f_dot_fun(
            cn,
            f,
            a_scale,
            self.tau1_rest,
            self.km_rest,
        )
        
        return vertcat(cn_dot, f_dot)


    def muscle_dynamic(
        self,
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        algebraic_states: MX | SX,
        numerical_data_timeseries: MX | SX,
        nlp: NonLinearProgram,
    ) -> DynamicsEvaluation:
        """
        The custom dynamics function that provides the derivative of the states: dxdt = f(t, x, u, p, s)

        Parameters
        ----------
        time: MX | SX
            The time of the system
        states: MX | SX
            The state of the system
        controls: MX | SX
            The controls of the system
        parameters: MX | SX
            The parameters acting on the system
        algebraic_states: MX | SX
            The stochastic variables of the system
        numerical_data_timeseries: MX | SX
            A list of values to pass to the dynamics at each node. Experimental external forces should be included here.
        nlp: NonLinearProgram
            A reference to the phase
        Returns
        -------
        The derivative of the states in the tuple[MX | SX] format
        """

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls) if "tau" in nlp.controls.keys() else 0

        muscles_tau, dxdt_muscle_list = self.muscles_joint_torque(
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_data_timeseries,
            nlp,
            q,
            qdot,
        )

        # You can directly call biorbd function (as for ddq) or call bioptim accessor (as for dq)
        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        total_torque = muscles_tau + tau if self.activate_residual_torque else muscles_tau
        external_forces = nlp.get_external_forces(states, controls, algebraic_states, numerical_data_timeseries)
        with_contact = (
            True if nlp.model.bio_model.contact_names != () else False
        )  # TODO: Add a better way of with_contact=True
        ddq = nlp.model.bio_model.forward_dynamics(with_contact=with_contact)(
            q, qdot, total_torque, external_forces, parameters
        )  # q, qdot, tau, external_forces, parameters
        dxdt = vertcat(dxdt_muscle_list, dq, ddq)

        return DynamicsEvaluation(dxdt=dxdt, defects=None)

    @staticmethod
    def muscles_joint_torque(
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        algebraic_states: MX | SX,
        numerical_data_timeseries: MX | SX,
        nlp: NonLinearProgram,
        q: MX | SX = None,
        qdot: MX | SX = None,
    ):

        Q = nlp.model.bio_model.q
        Qdot = nlp.model.bio_model.qdot

        updatedModel = nlp.model.bio_model.model.UpdateKinematicsCustom(Q, Qdot)
        nlp.model.bio_model.model.updateMuscles(updatedModel, Q, Qdot)
        updated_muscle_length_jacobian = nlp.model.bio_model.model.musclesLengthJacobian(updatedModel, Q, False).to_mx()
        updated_muscle_length_jacobian = Function("musclesLengthJacobian", [Q, Qdot], [updated_muscle_length_jacobian])(
            q, qdot
        )

        external_force_in_numerical_data_timeseries = (
            True if "external_force" in str(numerical_data_timeseries) else False
        )
        fes_numerical_data_timeseries = (
            numerical_data_timeseries[3 : numerical_data_timeseries.shape[0]]
            if external_force_in_numerical_data_timeseries
            else numerical_data_timeseries
        )

        muscle_dxdt = nlp.model.dynamics(
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            fes_numerical_data_timeseries,
            nlp,
        ).dxdt

        muscle_forces = DynamicsFunctions.get(nlp.states["F"], states)

        muscle_moment_arm_matrix = updated_muscle_length_jacobian
        muscle_joint_torques = -muscle_moment_arm_matrix.T @ muscle_forces

        return muscle_joint_torques, muscle_dxdt

    @staticmethod
    def forces_from_fes_driven(
        time: MX.sym,
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        algebraic_states: MX.sym,
        numerical_timeseries: MX.sym,
        nlp,
        with_passive_torque: bool = False,
        with_ligament: bool = False,
        state_name_list=None,
    ) -> MX:
        """
        Contact forces of a forward dynamics driven by muscles activations and joint torques with contact constraints.

        Parameters
        ----------
        time: MX.sym
            The time of the system
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        algebraic_states: MX.sym
            The algebraic states of the system
        numerical_timeseries: MX.sym
            The numerical timeseries of the system
        nlp: NonLinearProgram
            The definition of the system
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        state_name_list: list[str]
            The states names list
        Returns
        ----------
        MX.sym
            The contact forces that ensure no acceleration at these contact points
        """

        q = nlp.get_var_from_states_or_controls("q", states, controls)
        qdot = nlp.get_var_from_states_or_controls("qdot", states, controls)
        residual_tau = nlp.get_var_from_states_or_controls("tau", states, controls) if "tau" in nlp.controls else None

        muscles_joint_torque, _ = Marion2009ModelPulseWidthFrequency.muscles_joint_torque(
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            nlp,
            nlp.model.muscles_dynamics_model,
            state_name_list,
            q,
            qdot,
        )

        tau = muscles_joint_torque + residual_tau if residual_tau is not None else muscles_joint_torque
        tau = tau + nlp.model.passive_joint_torque()(q, qdot, nlp.parameters.cx) if with_passive_torque else tau
        tau = tau + nlp.model.ligament_joint_torque()(q, qdot, nlp.parameters.cx) if with_ligament else tau

        external_forces = nlp.get_external_forces(states, controls, algebraic_states, numerical_timeseries[0:3])

        return nlp.model.contact_forces()(q, qdot, tau, external_forces, nlp.parameters.cx)

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
        fes_model: Marion2009ModelPulseWidthFrequency
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
        q = DynamicsFunctions.get(nlp.states["q"], states)
        dxdt_fun = model.system_dynamics

        return DynamicsEvaluation(
            dxdt=dxdt_fun(
                cn=states[0],
                f=states[1],
                t=time,
                t_stim_prev=numerical_timeseries,
                pulse_width=controls[0],
                theta=q,
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
        StateConfigure().configure_q_for_marion_model(ocp, nlp, as_states=True, as_controls=False)
        StateConfigure().configure_qdot_for_marion_model(ocp, nlp, as_states=True, as_controls=False)
        StateConfigure().configure_last_pulse_width(ocp, nlp)
        if self.activate_residual_torque:
            ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, dyn_func=self.muscle_dynamic)

    def reshape_qdot(self, k_stab=1) -> Function:
        biorbd_return = self.bio_model.model.computeQdot(
            GeneralizedCoordinates(self.bio_model.q),
            GeneralizedCoordinates(self.bio_model.qdot),  # mistake in biorbd
            k_stab,
        ).to_mx()
        casadi_fun = Function(
            "reshape_qdot",
            [self.bio_model.q, self.bio_model.qdot, self.bio_model.parameters],
            [biorbd_return],
            ["q", "qdot", "parameters"],
            ["Reshaped qdot"],
        )
        return casadi_fun
