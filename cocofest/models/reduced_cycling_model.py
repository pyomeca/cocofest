"""Bioptim model combining reduced cycling mechanics with exact Ding states."""

from __future__ import annotations

from collections.abc import Sequence

from casadi import MX, SX, vertcat
from bioptim import (
    ConfigureVariables,
    DynamicsEvaluation,
    DynamicsFunctions,
    NonLinearProgram,
    OdeSolver,
    StateDynamics,
)

from cocofest.dynamics.reduced_cycling import ReducedCyclingDynamics
from cocofest.models.ding2007.ding2007 import DingModelPulseWidthFrequency
from cocofest.models.fes_model import FesModel
from cocofest.models.state_configure import StateConfigure


class ReducedFesCyclingModel(StateDynamics):
    """Twenty Ding states plus physical crank angle and angular velocity.

    The four five-state Ding models and their pulse-width controls are retained
    exactly.  The three-coordinate constrained multibody subsystem is replaced
    by the tangent-projected two-state system ``theta_dot = omega`` and
    ``omega_dot = f(theta, omega, muscle forces)``.
    """

    def __init__(
        self,
        *,
        reduced_dynamics: ReducedCyclingDynamics,
        muscles_model: Sequence[FesModel],
        external_crank_torque: float = 0.0,
        activate_force_length_relationship: bool = True,
        activate_force_velocity_relationship: bool = True,
        activate_passive_force_relationship: bool = True,
        name: str = "reduced_fes_cycling",
    ):
        super().__init__()
        self.reduced_dynamics = reduced_dynamics
        self.muscles_dynamics_model = list(muscles_model)
        self.external_crank_torque = float(external_crank_torque)
        self.activate_force_length_relationship = bool(
            activate_force_length_relationship
        )
        self.activate_force_velocity_relationship = bool(
            activate_force_velocity_relationship
        )
        self.activate_passive_force_relationship = bool(
            activate_passive_force_relationship
        )
        self._name = str(name)

        model_names = tuple(
            str(model.muscle_name) for model in self.muscles_dynamics_model
        )
        if model_names != self.reduced_dynamics.muscle_names:
            raise ValueError(
                "Reduced profile muscles and Ding models must have identical "
                f"ordering; received {model_names} and "
                f"{self.reduced_dynamics.muscle_names}."
            )
        if len(model_names) != 4:
            raise ValueError(
                "The cycling reduction currently expects four muscles "
                f"(20 Ding states), received {len(model_names)}."
            )
        invalid_state_models = [
            f"{model.muscle_name}:{model.nb_state}"
            for model in self.muscles_dynamics_model
            if int(model.nb_state) != 5
        ]
        if invalid_state_models:
            raise ValueError(
                "Reduced cycling requires exactly five Ding states per muscle "
                "(Cn, F, A, Tau1, Km); incompatible models: "
                + ", ".join(invalid_state_models)
                + "."
            )
        if (
            self.activate_force_length_relationship
            or self.activate_force_velocity_relationship
            or self.activate_passive_force_relationship
        ) and self.reduced_dynamics.muscle_geometry is None:
            raise ValueError(
                "The reduced profile must include muscle geometry when Hill "
                "relationships are enabled."
            )

    @property
    def name(self) -> str:
        return self._name

    @property
    def name_dofs(self) -> list[str]:
        return ["theta"]

    @property
    def nb_state(self) -> int:
        return 22

    @property
    def contact_types(self) -> tuple:
        return ()

    @property
    def state_configuration_functions(self):
        state_dictionary = StateConfigure().state_dictionary
        functions = []
        for muscle_model in self.muscles_dynamics_model:
            for state_key in muscle_model.name_dof:
                if state_key not in state_dictionary:
                    continue
                functions.append(
                    lambda ocp,
                    nlp,
                    state_key=state_key,
                    muscle_model=muscle_model: state_dictionary[state_key](
                        ocp=ocp,
                        nlp=nlp,
                        as_states=True,
                        as_controls=False,
                        muscle_name=muscle_model.muscle_name,
                    )
                )
        functions.extend(
            (
                lambda ocp, nlp: ConfigureVariables.configure_new_variable(
                    "theta", ["physical_crank_angle"], ocp, nlp, as_states=True
                ),
                lambda ocp, nlp: ConfigureVariables.configure_new_variable(
                    "omega", ["physical_crank_angular_velocity"], ocp, nlp, as_states=True
                ),
            )
        )
        return functions

    @property
    def control_configuration_functions(self):
        functions = []
        for muscle_model in self.muscles_dynamics_model:
            if isinstance(muscle_model, DingModelPulseWidthFrequency):
                functions.append(
                    lambda ocp,
                    nlp,
                    muscle_model=muscle_model: StateConfigure().configure_last_pulse_width(
                        ocp, nlp, muscle_model.muscle_name
                    )
                )
        return functions

    @property
    def algebraic_configuration_functions(self):
        return []

    @property
    def extra_configuration_functions(self):
        return []

    @property
    def extra_dynamics(self):
        return None

    def serialize(self):
        return (
            ReducedFesCyclingModel,
            {
                "reduced_dynamics": self.reduced_dynamics,
                "muscles_model": self.muscles_dynamics_model,
                "external_crank_torque": self.external_crank_torque,
                "activate_force_length_relationship": self.activate_force_length_relationship,
                "activate_force_velocity_relationship": self.activate_force_velocity_relationship,
                "activate_passive_force_relationship": self.activate_passive_force_relationship,
                "name": self._name,
            },
        )

    def dynamics(
        self,
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        algebraic_states: MX | SX,
        numerical_data_timeseries: MX | SX,
        nlp: NonLinearProgram,
    ) -> DynamicsEvaluation:
        theta = DynamicsFunctions.get(nlp.states["theta"], states)
        omega = DynamicsFunctions.get(nlp.states["omega"], states)
        if (
            self.activate_force_length_relationship
            or self.activate_force_velocity_relationship
            or self.activate_passive_force_relationship
        ):
            force_length, force_velocity, passive_force = (
                self.reduced_dynamics.casadi_muscle_relationships(theta, omega)
            )
        else:
            force_length = [1.0] * len(self.muscles_dynamics_model)
            force_velocity = [1.0] * len(self.muscles_dynamics_model)
            passive_force = [0.0] * len(self.muscles_dynamics_model)

        muscle_derivatives = []
        muscle_forces = []
        for muscle_index, muscle_model in enumerate(
            self.muscles_dynamics_model
        ):
            muscle_states = vertcat(
                *[
                    DynamicsFunctions.get(
                        nlp.states[f"{state_key}_{muscle_model.muscle_name}"],
                        states,
                    )
                    for state_key in muscle_model.name_dof
                ]
            )
            control_key = f"last_pulse_width_{muscle_model.muscle_name}"
            pulse_width = DynamicsFunctions.get(nlp.controls[control_key], controls)
            muscle_derivatives.append(
                muscle_model.dynamics(
                    time,
                    muscle_states,
                    pulse_width,
                    parameters,
                    algebraic_states,
                    numerical_data_timeseries,
                    nlp,
                    fes_model=muscle_model,
                    force_length_relationship=(
                        force_length[muscle_index]
                        if self.activate_force_length_relationship
                        else 1.0
                    ),
                    force_velocity_relationship=(
                        force_velocity[muscle_index]
                        if self.activate_force_velocity_relationship
                        else 1.0
                    ),
                    passive_force_relationship=(
                        passive_force[muscle_index]
                        if self.activate_passive_force_relationship
                        else 0.0
                    ),
                ).dxdt
            )
            muscle_forces.append(
                DynamicsFunctions.get(
                    nlp.states[f"F_{muscle_model.muscle_name}"], states
                )
            )

        omega_dot = self.reduced_dynamics.casadi_acceleration(
            theta,
            omega,
            vertcat(*muscle_forces),
            self.external_crank_torque,
        )
        dxdt = vertcat(*muscle_derivatives, omega, omega_dot)
        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):
            defects = (
                nlp.states_dot.scaled.cx * nlp.dt - dxdt * nlp.dt
            )
        return DynamicsEvaluation(dxdt=dxdt, defects=defects)
