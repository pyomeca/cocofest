from typing import Callable
import numpy as np

from casadi import vertcat, MX, SX
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsFunctions,
    DynamicsEvaluation,
)

from ..models.fes_model import FesModel
from ..models.ding2003 import DingModelFrequency
from .state_configue import StateConfigure
from .hill_coefficients import muscle_force_length_coefficient, muscle_force_velocity_coefficient


class FesMskModel(BiorbdModel):
    def __init__(
        self,
        name: str = None,
        biorbd_path: str = None,
        muscles_model: list[FesModel] = None,
        activate_force_length_relationship: bool = False,
        activate_force_velocity_relationship: bool = False,
    ):
        """
        The custom model that will be used in the optimal control program for the FES-MSK models

        Parameters
        ----------
        name: str
            The model name
        biorbd_path: str
            The path to the biorbd model
        muscles_model: DingModelFrequency
            The muscle model that will be used in the model
        activate_force_length_relationship: bool
            If the force-length relationship should be activated
        activate_force_velocity_relationship: bool
            If the force-velocity relationship should be activated
        """

        super().__init__(biorbd_path)
        self._name = name
        self.bio_model = BiorbdModel(biorbd_path)

        self._model_sanity(muscles_model, activate_force_length_relationship, activate_force_velocity_relationship)
        self.muscles_dynamics_model = muscles_model
        self.bio_stim_model = [self.bio_model] + self.muscles_dynamics_model

        self.activate_force_length_relationship = activate_force_length_relationship
        self.activate_force_velocity_relationship = activate_force_velocity_relationship

    # ---- Absolutely needed methods ---- #
    def serialize(self) -> tuple[Callable, dict]:
        return (
            FesMskModel,
            {
                "name": self._name,
                "biorbd_path": self.bio_model.path,
                "muscles_model": self.muscles_dynamics_model,
                "activate_force_length_relationship": self.activate_force_length_relationship,
                "activate_force_velocity_relationship": self.activate_force_velocity_relationship,
            },
        )

    # ---- Needed for the example ---- #
    @property
    def name_dof(self) -> tuple[str]:
        return self.bio_model.name_dof

    def muscle_name_dof(self, index: int = 0) -> list[str]:
        return self.muscles_dynamics_model[index].name_dof(with_muscle_name=True)

    @property
    def nb_state(self) -> int:
        nb_state = 0
        for muscle_model in self.muscles_dynamics_model:
            nb_state += muscle_model.nb_state
        nb_state += self.bio_model.nb_q
        return nb_state

    @property
    def name(self) -> None | str:
        return self._name

    def muscle_dynamic(
        self,
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        algebraic_states: MX | SX,
        numerical_data_timeseries: MX | SX,
        nlp: NonLinearProgram,
        muscle_models: list[FesModel],
        state_name_list=None,
        stim_prev: list[float] = None,
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
        muscle_models: list[FesModel]
            The list of the muscle models
        state_name_list: list[str]
            The states names list
        stim_prev: list[float]
            The previous stimulation values
        Returns
        -------
        The derivative of the states in the tuple[MX | SX] format
        """

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

        muscles_tau, dxdt_muscle_list = self.muscles_joint_torque(
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_data_timeseries,
            nlp,
            muscle_models,
            state_name_list,
            stim_prev,
            q,
            qdot,
        )

        # You can directly call biorbd function (as for ddq) or call bioptim accessor (as for dq)
        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = nlp.model.forward_dynamics(q, qdot, muscles_tau + tau)

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
        muscle_models: list[FesModel],
        state_name_list=None,
        stim_prev: list[float] = None,
        q: MX | SX = None,
        qdot: MX | SX = None,
    ):

        dxdt_muscle_list = vertcat()
        muscle_forces = vertcat()
        muscle_idx_list = []

        updatedModel = nlp.model.bio_model.model.UpdateKinematicsCustom(q, qdot)
        nlp.model.bio_model.model.updateMuscles(updatedModel, q, qdot)
        updated_muscle_length_jacobian = nlp.model.bio_model.model.musclesLengthJacobian(updatedModel, q, False).to_mx()

        bio_muscle_names_at_index = []
        for i in range(len(nlp.model.bio_model.model.muscles())):
            bio_muscle_names_at_index.append(nlp.model.bio_model.model.muscle(i).name().to_string())

        for muscle_model in muscle_models:
            muscle_states_idxs = [
                i for i in range(len(state_name_list)) if muscle_model.muscle_name in state_name_list[i]
            ]
            muscle_states = vertcat()
            for i in range(len(muscle_states_idxs)):
                muscle_states = vertcat(muscle_states, states[muscle_states_idxs[i]])

            muscle_idx = bio_muscle_names_at_index.index(muscle_model.muscle_name)

            muscle_force_length_coeff = (
                muscle_force_length_coefficient(
                    model=updatedModel, muscle=nlp.model.bio_model.model.muscle(muscle_idx), q=q
                )
                if nlp.model.activate_force_velocity_relationship
                else 1
            )

            muscle_force_velocity_coeff = (
                muscle_force_velocity_coefficient(
                    model=updatedModel, muscle=nlp.model.bio_model.model.muscle(muscle_idx), q=q, qdot=qdot
                )
                if nlp.model.activate_force_velocity_relationship
                else 1
            )

            muscle_dxdt = muscle_model.dynamics(
                time,
                muscle_states,
                controls,
                parameters,
                algebraic_states,
                numerical_data_timeseries,
                nlp,
                stim_prev=stim_prev,
                fes_model=muscle_model,
                force_length_relationship=muscle_force_length_coeff,
                force_velocity_relationship=muscle_force_velocity_coeff,
            ).dxdt

            dxdt_muscle_list = vertcat(dxdt_muscle_list, muscle_dxdt)
            muscle_idx_list.append(muscle_idx)

            muscle_forces = vertcat(
                muscle_forces, DynamicsFunctions.get(nlp.states["F_" + muscle_model.muscle_name], states)
            )

        muscle_moment_arm_matrix = updated_muscle_length_jacobian[
            muscle_idx_list, :
        ]  # reorganize the muscle moment arm matrix according to the muscle index list
        muscle_joint_torques = -muscle_moment_arm_matrix.T @ muscle_forces

        return muscle_joint_torques, dxdt_muscle_list

    def declare_model_variables(
        self, ocp: OptimalControlProgram, nlp: NonLinearProgram, numerical_data_timeseries: dict[str, np.ndarray] = None
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
        state_name_list = StateConfigure().configure_all_muscle_states(self.muscles_dynamics_model, ocp, nlp)
        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        state_name_list.append("q")
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        state_name_list.append("qdot")
        ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
        stim_prev = (
            DingModelFrequency._build_t_stim_prev(ocp, nlp.phase_idx)
            if "pulse_apparition_time" not in nlp.parameters.keys()
            else None
        )
        ConfigureProblem.configure_dynamics_function(
            ocp,
            nlp,
            dyn_func=self.muscle_dynamic,
            muscle_models=self.muscles_dynamics_model,
            state_name_list=state_name_list,
            stim_prev=stim_prev,
        )

    @staticmethod
    def _model_sanity(muscles_model, activate_force_length_relationship, activate_force_velocity_relationship):
        if not isinstance(muscles_model, list):
            for muscle_model in muscles_model:
                if not isinstance(muscle_model, FesModel):
                    raise TypeError(
                        f"The current model type used is {type(muscles_model)}, it must be a FesModel type."
                        f"Current available models are: DingModelFrequency, DingModelFrequencyWithFatigue,"
                        f"DingModelPulseDurationFrequency, DingModelPulseDurationFrequencyWithFatigue,"
                        f"DingModelIntensityFrequency, DingModelIntensityFrequencyWithFatigue"
                    )

            raise TypeError("The given muscles_model must be a list of FesModel")

        if not isinstance(activate_force_length_relationship, bool):
            raise TypeError("The activate_force_length_relationship must be a boolean")

        if not isinstance(activate_force_velocity_relationship, bool):
            raise TypeError("The activate_force_velocity_relationship must be a boolean")
