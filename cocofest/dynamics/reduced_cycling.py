"""One-degree-of-freedom mechanical reduction for constrained cycling.

The cycling model has three generalized coordinates and two holonomic contact
constraints at the crank centre.  Its admissible mechanical configurations
therefore form a one-dimensional manifold.  This module parameterizes that
manifold by the physical crank angle while retaining crank angular velocity as
an independent state.

The reduction is deliberately independent from Bioptim.  It can be generated
offline from a numerical biorbd model, validated against constrained forward
dynamics, and embedded in CasADi/ACADOS through smooth Fourier expressions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np


TWO_PI = 2.0 * np.pi
REDUCED_PROFILE_SCHEMA_VERSION = 2


def _as_numpy(value) -> np.ndarray:
    """Convert numerical biorbd or CasADi-backed values to NumPy."""

    if hasattr(value, "to_array"):
        value = value.to_array()
    elif hasattr(value, "to_mx"):
        from casadi import evalf

        value = evalf(value.to_mx())
    return np.asarray(value, dtype=float)


def _load_numerical_biorbd():
    try:
        import biorbd
    except ImportError:  # pragma: no cover - exercised on source-built Linux CI
        import biorbd_casadi as biorbd
    return biorbd


def _model_marker_names(model) -> list[str]:
    return [
        model.markerNames()[index].to_string()
        for index in range(model.nbMarkers())
    ]


def _model_muscle_names(model) -> list[str]:
    return [
        model.muscle(index).name().to_string()
        for index in range(model.nbMuscles())
    ]


def _model_dof_names(model) -> list[str]:
    return [
        model.nameDof()[index].to_string()
        for index in range(model.nbQ())
    ]


@dataclass(frozen=True)
class PeriodicFourierSeries:
    """Smooth periodic vector-valued Fourier approximation."""

    offset: np.ndarray
    cosine: np.ndarray
    sine: np.ndarray

    def __post_init__(self):
        offset = np.asarray(self.offset, dtype=float).reshape(-1)
        cosine = np.asarray(self.cosine, dtype=float)
        sine = np.asarray(self.sine, dtype=float)
        if cosine.ndim == 1:
            cosine = cosine.reshape((offset.size, -1))
        if sine.ndim == 1:
            sine = sine.reshape((offset.size, -1))
        if cosine.shape != sine.shape:
            raise ValueError("Fourier cosine and sine arrays must have the same shape.")
        if cosine.shape[0] != offset.size:
            raise ValueError("Fourier outputs must match the offset size.")
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "cosine", cosine)
        object.__setattr__(self, "sine", sine)

    @property
    def order(self) -> int:
        return int(self.cosine.shape[1])

    @property
    def output_size(self) -> int:
        return int(self.offset.size)

    @classmethod
    def fit(
        cls,
        phase: np.ndarray,
        values: np.ndarray,
        *,
        order: int,
    ) -> "PeriodicFourierSeries":
        """Fit samples defined over one or more complete turns."""

        phase = np.asarray(phase, dtype=float).reshape(-1)
        values = np.asarray(values, dtype=float)
        if order < 1:
            raise ValueError("Fourier order must be strictly positive.")
        if phase.size < 2 * order + 1:
            raise ValueError(
                "At least 2*order+1 samples are required for the Fourier fit."
            )
        if values.ndim == 1:
            values = values.reshape((1, -1))
        if values.ndim != 2 or values.shape[1] != phase.size:
            raise ValueError("Fourier values must have shape (outputs, samples).")
        if not np.all(np.isfinite(phase)) or not np.all(np.isfinite(values)):
            raise ValueError("Fourier samples must be finite.")

        harmonics = np.arange(1, order + 1, dtype=float)
        design = np.column_stack(
            (
                np.ones(phase.size),
                *[
                    basis(harmonic * phase)
                    for harmonic in harmonics
                    for basis in (np.cos, np.sin)
                ],
            )
        )
        coefficients, *_ = np.linalg.lstsq(design, values.T, rcond=None)
        return cls(
            offset=coefficients[0, :],
            cosine=coefficients[1::2, :].T,
            sine=coefficients[2::2, :].T,
        )

    def evaluate(self, phase, *, derivative: int = 0) -> np.ndarray:
        """Evaluate the series or its first two phase derivatives."""

        if derivative not in (0, 1, 2):
            raise ValueError("Only derivatives 0, 1 and 2 are supported.")
        phase_array = np.asarray(phase, dtype=float)
        scalar = phase_array.ndim == 0
        flat_phase = phase_array.reshape(-1)
        harmonics = np.arange(1, self.order + 1, dtype=float)
        arguments = harmonics[:, None] * flat_phase[None, :]

        if derivative == 0:
            values = (
                self.offset[:, None]
                + self.cosine @ np.cos(arguments)
                + self.sine @ np.sin(arguments)
            )
        elif derivative == 1:
            values = (
                -(self.cosine * harmonics[None, :]) @ np.sin(arguments)
                + (self.sine * harmonics[None, :]) @ np.cos(arguments)
            )
        else:
            squared = harmonics**2
            values = (
                -(self.cosine * squared[None, :]) @ np.cos(arguments)
                - (self.sine * squared[None, :]) @ np.sin(arguments)
            )

        if scalar:
            return values[:, 0]
        return values.reshape((self.output_size, *phase_array.shape))

    def casadi(self, phase, *, derivative: int = 0):
        """Return a smooth CasADi expression for the series."""

        if derivative not in (0, 1, 2):
            raise ValueError("Only derivatives 0, 1 and 2 are supported.")
        from casadi import cos, sin, vertcat

        outputs = []
        for output_index in range(self.output_size):
            value = float(self.offset[output_index]) if derivative == 0 else 0.0
            for harmonic in range(1, self.order + 1):
                cosine = float(self.cosine[output_index, harmonic - 1])
                sine = float(self.sine[output_index, harmonic - 1])
                argument = harmonic * phase
                if derivative == 0:
                    value += cosine * cos(argument) + sine * sin(argument)
                elif derivative == 1:
                    value += harmonic * (
                        -cosine * sin(argument) + sine * cos(argument)
                    )
                else:
                    value -= harmonic**2 * (
                        cosine * cos(argument) + sine * sin(argument)
                    )
            outputs.append(value)
        return vertcat(*outputs)


@dataclass(frozen=True)
class ReducedCyclingKinematics:
    """Full generalized kinematics parameterized by physical crank angle."""

    theta_origin: float
    direction: int
    winding_numbers: np.ndarray
    periodic_residual: PeriodicFourierSeries

    def __post_init__(self):
        if self.direction not in (-1, 1):
            raise ValueError("Cycling direction must be -1 or 1.")
        winding_numbers = np.asarray(self.winding_numbers, dtype=float).reshape(-1)
        if winding_numbers.size != self.periodic_residual.output_size:
            raise ValueError("One winding number is required per generalized coordinate.")
        object.__setattr__(self, "winding_numbers", winding_numbers)

    @property
    def nb_q(self) -> int:
        return int(self.winding_numbers.size)

    @classmethod
    def fit(
        cls,
        theta: np.ndarray,
        q: np.ndarray,
        *,
        order: int = 12,
        turn_tolerance: float = 1e-5,
    ) -> "ReducedCyclingKinematics":
        """Fit one contact-consistent turn sampled by physical crank angle."""

        theta = np.asarray(theta, dtype=float).reshape(-1)
        q = np.asarray(q, dtype=float)
        if q.ndim != 2 or q.shape[1] != theta.size:
            raise ValueError("q must have shape (nb_q, number_of_theta_samples).")
        delta_theta = float(theta[-1] - theta[0])
        if np.isclose(delta_theta, 0.0):
            raise ValueError("Crank-angle samples must span one complete turn.")
        direction = 1 if delta_theta > 0.0 else -1
        progress = direction * (theta - theta[0])
        if np.any(np.diff(progress) <= 0.0):
            raise ValueError("Crank-angle samples must be strictly monotonic.")
        if not np.isclose(progress[-1], TWO_PI, atol=turn_tolerance, rtol=0.0):
            raise ValueError(
                f"Crank-angle samples span {progress[-1]:.12g} rad instead of 2*pi."
            )

        winding_numbers = np.rint((q[:, -1] - q[:, 0]) / TWO_PI)
        periodic_values = q - winding_numbers[:, None] * progress[None, :]
        periodic_residual = PeriodicFourierSeries.fit(
            progress, periodic_values, order=order
        )
        return cls(
            theta_origin=float(theta[0]),
            direction=direction,
            winding_numbers=winding_numbers,
            periodic_residual=periodic_residual,
        )

    def progress(self, theta):
        return self.direction * (theta - self.theta_origin)

    def q(self, theta) -> np.ndarray:
        progress = self.progress(np.asarray(theta, dtype=float))
        periodic = self.periodic_residual.evaluate(progress)
        if np.asarray(progress).ndim == 0:
            return self.winding_numbers * float(progress) + periodic
        winding = self.winding_numbers.reshape(
            (-1,) + (1,) * np.asarray(progress).ndim
        )
        return winding * progress + periodic

    def tangent(self, theta) -> np.ndarray:
        """Return dq/dtheta."""

        progress = self.progress(np.asarray(theta, dtype=float))
        periodic_derivative = self.periodic_residual.evaluate(progress, derivative=1)
        if np.asarray(progress).ndim == 0:
            return self.direction * (
                self.winding_numbers + periodic_derivative
            )
        winding = self.winding_numbers.reshape(
            (-1,) + (1,) * np.asarray(progress).ndim
        )
        return self.direction * (winding + periodic_derivative)

    def curvature(self, theta) -> np.ndarray:
        """Return d2q/dtheta2."""

        progress = self.progress(np.asarray(theta, dtype=float))
        return self.periodic_residual.evaluate(progress, derivative=2)

    def generalized_kinematics(
        self,
        theta: float,
        omega: float,
        theta_acceleration: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = self.q(theta)
        tangent = self.tangent(theta)
        curvature = self.curvature(theta)
        return (
            q,
            tangent * float(omega),
            tangent * float(theta_acceleration)
            + curvature * float(omega) ** 2,
        )

    def project_generalized_trajectory(
        self,
        q: np.ndarray,
        qdot: np.ndarray | None = None,
        *,
        maximum_iterations: int = 20,
        tolerance: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray | None, dict]:
        """Project full generalized coordinates onto ``theta`` and ``omega``."""

        q = np.asarray(q, dtype=float)
        if q.ndim != 2 or q.shape[0] != self.nb_q:
            raise ValueError(
                f"q must have shape ({self.nb_q}, number_of_nodes)."
            )
        if qdot is not None:
            qdot = np.asarray(qdot, dtype=float)
            if qdot.shape != q.shape:
                raise ValueError("qdot must have the same shape as q.")

        winding_index = int(np.argmax(np.abs(self.winding_numbers)))
        winding = float(self.winding_numbers[winding_index])
        if np.isclose(winding, 0.0):
            raise RuntimeError(
                "At least one generalized coordinate must wind over a crank turn."
            )

        theta = np.empty(q.shape[1])
        omega = None if qdot is None else np.empty(q.shape[1])
        residuals = np.empty(q.shape[1])
        iterations = np.empty(q.shape[1], dtype=int)
        periodic_origin = self.periodic_residual.evaluate(0.0)
        previous_progress = None
        previous_q = None
        for node in range(q.shape[1]):
            if previous_progress is None:
                progress = (
                    q[winding_index, node] - periodic_origin[winding_index]
                ) / winding
            else:
                progress = previous_progress + (
                    q[winding_index, node] - previous_q[winding_index]
                ) / winding

            theta_value = self.theta_origin + self.direction * progress
            iteration = 0
            for iteration in range(1, maximum_iterations + 1):
                difference = self.q(theta_value) - q[:, node]
                tangent = self.tangent(theta_value)
                curvature = self.curvature(theta_value)
                gradient = float(tangent @ difference)
                hessian = float(
                    tangent @ tangent + curvature @ difference
                )
                if np.isclose(hessian, 0.0):
                    break
                step = gradient / hessian
                theta_value -= step
                if abs(step) <= tolerance:
                    break

            reconstructed = self.q(theta_value)
            tangent = self.tangent(theta_value)
            theta[node] = theta_value
            residuals[node] = float(
                np.linalg.norm(reconstructed - q[:, node], ord=np.inf)
            )
            iterations[node] = iteration
            if omega is not None:
                omega[node] = float(
                    tangent @ qdot[:, node] / (tangent @ tangent)
                )
            previous_progress = float(self.progress(theta_value))
            previous_q = q[:, node]

        return theta[np.newaxis, :], (
            None if omega is None else omega[np.newaxis, :]
        ), {
            "maximum_configuration_projection_error_rad": float(
                np.max(residuals)
            ),
            "maximum_projection_iterations": int(np.max(iterations)),
        }

    def casadi_kinematics(self, theta, omega=None, theta_acceleration=None):
        from casadi import DM

        progress = self.direction * (theta - self.theta_origin)
        winding = DM(self.winding_numbers)
        q = winding * progress + self.periodic_residual.casadi(progress)
        tangent = self.direction * (
            winding + self.periodic_residual.casadi(progress, derivative=1)
        )
        curvature = self.periodic_residual.casadi(progress, derivative=2)
        if omega is None:
            return q, tangent, curvature
        qdot = tangent * omega
        if theta_acceleration is None:
            return q, qdot
        qddot = tangent * theta_acceleration + curvature * omega**2
        return q, qdot, qddot


def solve_cycling_contact_kinematics(
    model_path: str | Path,
    *,
    sample_count: int = 181,
    theta_origin: float = 0.0,
    direction: int = -1,
    initial_q: Sequence[float] | None = None,
    hand_marker: str = "hand",
    wheel_center_marker: str = "wheel_center",
    global_center_marker: str = "global_wheel_center",
    residual_tolerance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Solve the exact contact manifold over one physical crank turn.

    Two residuals fix the wheel centre and one residual fixes the direction of
    the vector from the wheel centre to the hand/crank attachment.  The
    previous solution initializes the next angle, preserving the elbow branch.
    """

    if sample_count < 5:
        raise ValueError("At least five kinematic samples are required.")
    if direction not in (-1, 1):
        raise ValueError("Cycling direction must be -1 or 1.")

    from scipy.optimize import least_squares

    biorbd = _load_numerical_biorbd()
    model = biorbd.Model(str(Path(model_path)))
    marker_names = _model_marker_names(model)
    try:
        hand_index = marker_names.index(hand_marker)
        wheel_index = marker_names.index(wheel_center_marker)
        center_index = marker_names.index(global_center_marker)
    except ValueError as error:
        raise ValueError(
            "The reduced cycling model requires hand, wheel-centre and "
            "global-centre markers."
        ) from error

    def residual(q_values, theta_value):
        markers = model.markers(q_values)
        wheel = _as_numpy(markers[wheel_index]).reshape(-1)
        center = _as_numpy(markers[center_index]).reshape(-1)
        hand = _as_numpy(markers[hand_index]).reshape(-1)
        crank_vector = hand[:2] - center[:2]
        phase_residual = (
            crank_vector[0] * np.sin(theta_value)
            - crank_vector[1] * np.cos(theta_value)
        )
        return np.array(
            [
                wheel[0] - center[0],
                wheel[1] - center[1],
                phase_residual,
            ]
        )

    def residual_jacobian(q_values, theta_value):
        jacobians = model.markersJacobian(q_values)
        wheel = _as_numpy(jacobians[wheel_index])
        center = _as_numpy(jacobians[center_index])
        hand = _as_numpy(jacobians[hand_index])
        return np.vstack(
            (
                (wheel - center)[0, :],
                (wheel - center)[1, :],
                np.sin(theta_value) * hand[0, :]
                - np.cos(theta_value) * hand[1, :],
            )
        )

    theta = theta_origin + direction * np.linspace(0.0, TWO_PI, sample_count)
    current_q = (
        np.zeros(model.nbQ())
        if initial_q is None
        else np.asarray(initial_q, dtype=float).reshape(-1)
    )
    if current_q.size != model.nbQ():
        raise ValueError(f"initial_q must contain {model.nbQ()} values.")

    q_samples = np.empty((model.nbQ(), sample_count))
    residuals = np.empty((3, sample_count))
    crank_radii = np.empty(sample_count)
    forward_phase_projection = np.empty(sample_count)
    evaluations = 0
    for index, theta_value in enumerate(theta):
        result = least_squares(
            residual,
            current_q,
            jac=residual_jacobian,
            args=(theta_value,),
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=100,
        )
        current_q = result.x
        evaluations += int(result.nfev)
        current_residual = residual(current_q, theta_value)
        if np.max(np.abs(current_residual)) > residual_tolerance:
            raise RuntimeError(
                "Contact-manifold solve did not reach the requested tolerance "
                f"at sample {index}: {np.max(np.abs(current_residual)):.3e}."
            )
        q_samples[:, index] = current_q
        residuals[:, index] = current_residual

        markers = model.markers(current_q)
        center = _as_numpy(markers[center_index]).reshape(-1)
        hand = _as_numpy(markers[hand_index]).reshape(-1)
        crank_vector = hand[:2] - center[:2]
        crank_radii[index] = np.linalg.norm(crank_vector)
        forward_phase_projection[index] = np.dot(
            crank_vector,
            np.array([np.cos(theta_value), np.sin(theta_value)]),
        )

    if np.any(forward_phase_projection <= 0.0):
        raise RuntimeError("The contact solve switched to the opposite crank branch.")

    audit = {
        "sample_count": int(sample_count),
        "function_evaluations": int(evaluations),
        "maximum_contact_phase_residual": float(np.max(np.abs(residuals))),
        "minimum_crank_radius": float(np.min(crank_radii)),
        "maximum_crank_radius": float(np.max(crank_radii)),
        "maximum_cycle_closure_error": float(
            np.max(
                np.abs(
                    (q_samples[:, -1] - q_samples[:, 0])
                    - np.rint(
                        (q_samples[:, -1] - q_samples[:, 0]) / TWO_PI
                    )
                    * TWO_PI
                )
            )
        ),
    }
    return theta, q_samples, audit


@dataclass(frozen=True)
class ReducedCyclingDynamics:
    """Fourier profile of the tangent-projected mechanical dynamics."""

    kinematics: ReducedCyclingKinematics
    coefficients: PeriodicFourierSeries
    muscle_names: tuple[str, ...]
    crank_torque_dof_index: int
    muscle_geometry: PeriodicFourierSeries | None = None

    @property
    def _inertia_index(self) -> int:
        return 0

    @property
    def _gravity_index(self) -> int:
        return 1

    @property
    def _velocity_index(self) -> int:
        return 2

    @property
    def _muscle_slice(self) -> slice:
        return slice(3, 3 + len(self.muscle_names))

    @property
    def _external_index(self) -> int:
        return 3 + len(self.muscle_names)

    @property
    def _normalized_length_slice(self) -> slice:
        return slice(0, len(self.muscle_names))

    @property
    def _velocity_per_omega_slice(self) -> slice:
        muscle_count = len(self.muscle_names)
        return slice(muscle_count, 2 * muscle_count)

    @classmethod
    def build(
        cls,
        model_path: str | Path,
        kinematics: ReducedCyclingKinematics,
        *,
        sample_count: int = 181,
        order: int = 12,
        muscle_names: Sequence[str] | None = None,
        crank_torque_dof: str = "wheel_rotation_RotZ",
    ) -> "ReducedCyclingDynamics":
        """Sample and fit the projected mass, bias and muscle effectiveness."""

        if sample_count < 2 * order + 1:
            raise ValueError("sample_count must be at least 2*order+1.")
        biorbd = _load_numerical_biorbd()
        model = biorbd.Model(str(Path(model_path)))
        available_muscles = _model_muscle_names(model)
        selected_muscles = (
            available_muscles
            if muscle_names is None
            else [str(name) for name in muscle_names]
        )
        missing = sorted(set(selected_muscles) - set(available_muscles))
        if missing:
            raise ValueError(f"Unknown muscles: {', '.join(missing)}.")
        muscle_indices = [available_muscles.index(name) for name in selected_muscles]

        dof_names = _model_dof_names(model)
        if crank_torque_dof not in dof_names:
            raise ValueError(
                f"Unknown crank torque DoF '{crank_torque_dof}'. "
                f"Available DoFs: {', '.join(dof_names)}."
            )
        crank_torque_dof_index = dof_names.index(crank_torque_dof)

        progress = np.linspace(0.0, TWO_PI, sample_count)
        theta = kinematics.theta_origin + kinematics.direction * progress
        coefficient_samples = np.empty(
            (4 + len(selected_muscles), sample_count)
        )
        geometry_samples = np.empty((2 * len(selected_muscles), sample_count))
        for index, theta_value in enumerate(theta):
            q = kinematics.q(theta_value)
            tangent = kinematics.tangent(theta_value)
            curvature = kinematics.curvature(theta_value)
            mass_matrix = _as_numpy(model.massMatrix(q))
            zero_velocity = np.zeros(model.nbQ())
            nonlinear_zero = _as_numpy(
                model.NonLinearEffect(q, zero_velocity)
            ).reshape(-1)
            nonlinear_unit = _as_numpy(
                model.NonLinearEffect(q, tangent)
            ).reshape(-1)
            muscle_jacobian = _as_numpy(model.musclesLengthJacobian(q))[
                muscle_indices, :
            ]
            normalized_lengths = np.array(
                [
                    float(
                        _as_numpy(
                            model.muscle(muscle_index).length(
                                model.UpdateKinematicsCustom(q), q, True
                            )
                        ).reshape(-1)[0]
                    )
                    / float(
                        _as_numpy(
                            model.muscle(muscle_index)
                            .characteristics()
                            .optimalLength()
                        ).reshape(-1)[0]
                    )
                    for muscle_index in muscle_indices
                ]
            )
            velocity_per_omega = muscle_jacobian @ tangent

            effective_inertia = float(tangent @ mass_matrix @ tangent)
            if effective_inertia <= 0.0:
                raise RuntimeError("Reduced effective inertia must remain positive.")
            projected_gravity = float(tangent @ nonlinear_zero)
            projected_velocity = float(
                tangent
                @ (
                    nonlinear_unit
                    - nonlinear_zero
                    + mass_matrix @ curvature
                )
            )
            muscle_effectiveness = -(muscle_jacobian @ tangent)
            external_effectiveness = float(tangent[crank_torque_dof_index])
            coefficient_samples[:, index] = np.concatenate(
                (
                    np.array(
                        [
                            effective_inertia,
                            projected_gravity,
                            projected_velocity,
                        ]
                    ),
                    muscle_effectiveness,
                    np.array([external_effectiveness]),
                )
            )
            geometry_samples[:, index] = np.concatenate(
                (normalized_lengths, velocity_per_omega)
            )

        return cls(
            kinematics=kinematics,
            coefficients=PeriodicFourierSeries.fit(
                progress, coefficient_samples, order=order
            ),
            muscle_names=tuple(selected_muscles),
            crank_torque_dof_index=crank_torque_dof_index,
            muscle_geometry=PeriodicFourierSeries.fit(
                progress, geometry_samples, order=order
            ),
        )

    def coefficient_values(self, theta: float) -> dict:
        values = self.coefficients.evaluate(self.kinematics.progress(theta))
        return {
            "effective_inertia": float(values[self._inertia_index]),
            "projected_gravity": float(values[self._gravity_index]),
            "projected_velocity_quadratic": float(
                values[self._velocity_index]
            ),
            "muscle_effectiveness": values[self._muscle_slice].copy(),
            "external_torque_effectiveness": float(
                values[self._external_index]
            ),
        }

    def acceleration(
        self,
        theta: float,
        omega: float,
        muscle_forces: Sequence[float],
        *,
        external_crank_torque: float = 0.0,
    ) -> float:
        forces = np.asarray(muscle_forces, dtype=float).reshape(-1)
        if forces.size != len(self.muscle_names):
            raise ValueError(
                f"Expected {len(self.muscle_names)} muscle forces, "
                f"received {forces.size}."
            )
        values = self.coefficients.evaluate(self.kinematics.progress(theta))
        numerator = (
            values[self._muscle_slice] @ forces
            + values[self._external_index] * float(external_crank_torque)
            - values[self._gravity_index]
            - values[self._velocity_index] * float(omega) ** 2
        )
        return float(numerator / values[self._inertia_index])

    def casadi_acceleration(
        self,
        theta,
        omega,
        muscle_forces,
        external_crank_torque=0.0,
    ):
        from casadi import dot

        values = self.coefficients.casadi(self.kinematics.progress(theta))
        numerator = (
            dot(values[self._muscle_slice], muscle_forces)
            + values[self._external_index] * external_crank_torque
            - values[self._gravity_index]
            - values[self._velocity_index] * omega**2
        )
        return numerator / values[self._inertia_index]

    def casadi_muscle_relationships(self, theta, omega):
        """Return the De Groote active, velocity and passive coefficients.

        The algebraic laws are unchanged from the full model.  Only normalized
        muscle length and velocity-per-crank-speed are represented by smooth
        periodic Fourier profiles.
        """

        if self.muscle_geometry is None:
            raise RuntimeError(
                "This reduced profile predates muscle-geometry fitting. "
                "Rebuild it before enabling the full Ding cycling OCP."
            )
        from casadi import exp, log, sqrt

        geometry = self.muscle_geometry.casadi(self.kinematics.progress(theta))
        normalized_length = geometry[self._normalized_length_slice]
        normalized_velocity = (
            geometry[self._velocity_per_omega_slice] * omega / 10.0
        )

        active_force_length = (
            0.815
            * exp(
                -0.5 * (normalized_length - 1.055) ** 2
                / (0.162 + 0.063 * normalized_length) ** 2
            )
            + 0.433
            * exp(
                -0.5 * (normalized_length - 0.717) ** 2
                / (-0.030 + 0.200 * normalized_length) ** 2
            )
            + 0.100
            * exp(
                -0.5 * (normalized_length - 1.000) ** 2
                / (0.354 + 0.0 * normalized_length) ** 2
            )
        )
        velocity_argument = -8.149 * normalized_velocity - 0.374
        force_velocity = (
            -0.318
            * log(
                velocity_argument
                + sqrt(velocity_argument * velocity_argument + 1)
            )
            + 0.886
        )
        passive_force = (
            exp(4.0 * (normalized_length - 1.0) / 0.6) - 1.0
        ) / (np.exp(4.0) - 1.0)
        return active_force_length, force_velocity, passive_force

    def muscle_relationships(self, theta: float, omega: float):
        """Numerical counterpart of :meth:`casadi_muscle_relationships`."""

        if self.muscle_geometry is None:
            raise RuntimeError(
                "This reduced profile does not contain muscle geometry."
            )
        geometry = self.muscle_geometry.evaluate(
            self.kinematics.progress(theta)
        )
        normalized_length = geometry[self._normalized_length_slice]
        normalized_velocity = (
            geometry[self._velocity_per_omega_slice] * float(omega) / 10.0
        )
        active_force_length = (
            0.815
            * np.exp(
                -0.5 * (normalized_length - 1.055) ** 2
                / (0.162 + 0.063 * normalized_length) ** 2
            )
            + 0.433
            * np.exp(
                -0.5 * (normalized_length - 0.717) ** 2
                / (-0.030 + 0.200 * normalized_length) ** 2
            )
            + 0.100
            * np.exp(
                -0.5 * (normalized_length - 1.000) ** 2 / 0.354**2
            )
        )
        velocity_argument = -8.149 * normalized_velocity - 0.374
        force_velocity = (
            -0.318 * np.arcsinh(velocity_argument) + 0.886
        )
        passive_force = (
            np.exp(4.0 * (normalized_length - 1.0) / 0.6) - 1.0
        ) / (np.exp(4.0) - 1.0)
        return active_force_length, force_velocity, passive_force

    def save(self, path: str | Path) -> Path:
        """Save the generated profile without pickle or backend objects."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            schema_version=np.array([REDUCED_PROFILE_SCHEMA_VERSION], dtype=int),
            theta_origin=np.array([self.kinematics.theta_origin]),
            direction=np.array([self.kinematics.direction], dtype=int),
            winding_numbers=self.kinematics.winding_numbers,
            kinematic_offset=self.kinematics.periodic_residual.offset,
            kinematic_cosine=self.kinematics.periodic_residual.cosine,
            kinematic_sine=self.kinematics.periodic_residual.sine,
            dynamics_offset=self.coefficients.offset,
            dynamics_cosine=self.coefficients.cosine,
            dynamics_sine=self.coefficients.sine,
            muscle_names=np.asarray(self.muscle_names, dtype=str),
            crank_torque_dof_index=np.array(
                [self.crank_torque_dof_index], dtype=int
            ),
            **(
                {}
                if self.muscle_geometry is None
                else {
                    "muscle_geometry_offset": self.muscle_geometry.offset,
                    "muscle_geometry_cosine": self.muscle_geometry.cosine,
                    "muscle_geometry_sine": self.muscle_geometry.sine,
                }
            ),
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "ReducedCyclingDynamics":
        """Load a profile generated by :meth:`save`."""

        with np.load(Path(path), allow_pickle=False) as data:
            schema_version = (
                int(data["schema_version"][0])
                if "schema_version" in data.files
                else 1
            )
            if schema_version != REDUCED_PROFILE_SCHEMA_VERSION:
                raise ValueError(
                    "Reduced cycling profile schema "
                    f"{schema_version} is stale; expected "
                    f"{REDUCED_PROFILE_SCHEMA_VERSION}. Rebuild the profile."
                )
            kinematics = ReducedCyclingKinematics(
                theta_origin=float(data["theta_origin"][0]),
                direction=int(data["direction"][0]),
                winding_numbers=np.asarray(
                    data["winding_numbers"], dtype=float
                ),
                periodic_residual=PeriodicFourierSeries(
                    offset=data["kinematic_offset"],
                    cosine=data["kinematic_cosine"],
                    sine=data["kinematic_sine"],
                ),
            )
            return cls(
                kinematics=kinematics,
                coefficients=PeriodicFourierSeries(
                    offset=data["dynamics_offset"],
                    cosine=data["dynamics_cosine"],
                    sine=data["dynamics_sine"],
                ),
                muscle_names=tuple(str(name) for name in data["muscle_names"]),
                crank_torque_dof_index=int(
                    data["crank_torque_dof_index"][0]
                ),
                muscle_geometry=(
                    PeriodicFourierSeries(
                        offset=data["muscle_geometry_offset"],
                        cosine=data["muscle_geometry_cosine"],
                        sine=data["muscle_geometry_sine"],
                    )
                    if "muscle_geometry_offset" in data
                    else None
                ),
            )


def build_reduced_cycling_dynamics(
    model_path: str | Path,
    *,
    sample_count: int = 181,
    kinematic_order: int = 12,
    dynamics_order: int = 12,
    theta_origin: float = 0.0,
    direction: int = -1,
    initial_q: Sequence[float] | None = None,
) -> tuple[ReducedCyclingDynamics, dict]:
    """Build the complete contact-consistent reduced mechanical model."""

    start = perf_counter()
    theta, q_samples, contact_audit = solve_cycling_contact_kinematics(
        model_path,
        sample_count=sample_count,
        theta_origin=theta_origin,
        direction=direction,
        initial_q=initial_q,
    )
    kinematics = ReducedCyclingKinematics.fit(
        theta, q_samples, order=kinematic_order
    )
    reconstructed = kinematics.q(theta)
    kinematic_fit_error = float(np.max(np.abs(reconstructed - q_samples)))
    reduced = ReducedCyclingDynamics.build(
        model_path,
        kinematics,
        sample_count=sample_count,
        order=dynamics_order,
    )
    return reduced, {
        **contact_audit,
        "kinematic_fourier_order": int(kinematic_order),
        "dynamics_fourier_order": int(dynamics_order),
        "maximum_kinematic_fit_error_rad": kinematic_fit_error,
        "build_wall_time_s": float(perf_counter() - start),
    }


def validate_reduced_cycling_dynamics(
    model_path: str | Path,
    reduced: ReducedCyclingDynamics,
    *,
    sample_count: int = 50,
    omega_range: tuple[float, float] = (-9.0, -3.0),
    force_range: tuple[float, float] = (0.0, 600.0),
    external_crank_torque: float = -0.2,
    seed: int = 42,
) -> dict:
    """Compare reduced accelerations with biorbd constrained dynamics."""

    if sample_count < 1:
        raise ValueError("Validation sample_count must be strictly positive.")
    biorbd = _load_numerical_biorbd()
    model = biorbd.Model(str(Path(model_path)))
    available_muscles = _model_muscle_names(model)
    muscle_indices = [
        available_muscles.index(name) for name in reduced.muscle_names
    ]
    rng = np.random.default_rng(seed)
    theta_values = (
        reduced.kinematics.theta_origin
        + reduced.kinematics.direction
        * rng.uniform(0.0, TWO_PI, sample_count)
    )
    omega_values = rng.uniform(*omega_range, sample_count)
    force_values = rng.uniform(
        force_range[0],
        force_range[1],
        (sample_count, len(reduced.muscle_names)),
    )

    acceleration_errors = np.empty(sample_count)
    generalized_acceleration_errors = np.empty(sample_count)
    contact_position_errors = np.empty(sample_count)
    force_length_errors = np.empty((sample_count, len(muscle_indices)))
    force_velocity_errors = np.empty((sample_count, len(muscle_indices)))
    passive_force_errors = np.empty((sample_count, len(muscle_indices)))
    full_start = perf_counter()
    full_accelerations = np.empty(sample_count)
    for index, (theta, omega, forces) in enumerate(
        zip(theta_values, omega_values, force_values, strict=True)
    ):
        q = reduced.kinematics.q(theta)
        tangent = reduced.kinematics.tangent(theta)
        curvature = reduced.kinematics.curvature(theta)
        qdot = tangent * omega
        muscle_jacobian = _as_numpy(model.musclesLengthJacobian(q))[
            muscle_indices, :
        ]
        updated_model = model.UpdateKinematicsCustom(q)
        normalized_length = np.array(
            [
                float(
                    _as_numpy(
                        model.muscle(muscle_index).length(
                            updated_model, q, True
                        )
                    ).reshape(-1)[0]
                )
                / float(
                    _as_numpy(
                        model.muscle(muscle_index)
                        .characteristics()
                        .optimalLength()
                    ).reshape(-1)[0]
                )
                for muscle_index in muscle_indices
            ]
        )
        normalized_velocity = muscle_jacobian @ tangent * omega / 10.0
        exact_force_length = (
            0.815
            * np.exp(
                -0.5 * (normalized_length - 1.055) ** 2
                / (0.162 + 0.063 * normalized_length) ** 2
            )
            + 0.433
            * np.exp(
                -0.5 * (normalized_length - 0.717) ** 2
                / (-0.030 + 0.200 * normalized_length) ** 2
            )
            + 0.100
            * np.exp(-0.5 * (normalized_length - 1.0) ** 2 / 0.354**2)
        )
        exact_force_velocity = (
            -0.318
            * np.arcsinh(-8.149 * normalized_velocity - 0.374)
            + 0.886
        )
        exact_passive_force = (
            np.exp(4.0 * (normalized_length - 1.0) / 0.6) - 1.0
        ) / (np.exp(4.0) - 1.0)
        (
            reduced_force_length,
            reduced_force_velocity,
            reduced_passive_force,
        ) = reduced.muscle_relationships(theta, omega)
        force_length_errors[index] = np.abs(
            reduced_force_length - exact_force_length
        )
        force_velocity_errors[index] = np.abs(
            reduced_force_velocity - exact_force_velocity
        )
        passive_force_errors[index] = np.abs(
            reduced_passive_force - exact_passive_force
        )
        generalized_torque = -muscle_jacobian.T @ forces
        generalized_torque[reduced.crank_torque_dof_index] += (
            external_crank_torque
        )
        qddot_full = _as_numpy(
            model.ForwardDynamicsConstraintsDirect(
                q,
                qdot,
                generalized_torque,
                model.getConstraints(),
            )
        ).reshape(-1)
        full_acceleration = float(
            tangent @ (qddot_full - curvature * omega**2)
            / (tangent @ tangent)
        )
        full_accelerations[index] = full_acceleration

        markers = model.markers(q)
        marker_names = _model_marker_names(model)
        wheel = _as_numpy(
            markers[marker_names.index("wheel_center")]
        ).reshape(-1)
        center = _as_numpy(
            markers[marker_names.index("global_wheel_center")]
        ).reshape(-1)
        contact_position_errors[index] = np.max(
            np.abs(wheel[:2] - center[:2])
        )

        reduced_acceleration = reduced.acceleration(
            theta,
            omega,
            forces,
            external_crank_torque=external_crank_torque,
        )
        qddot_reduced = (
            tangent * reduced_acceleration + curvature * omega**2
        )
        acceleration_errors[index] = abs(
            reduced_acceleration - full_acceleration
        )
        generalized_acceleration_errors[index] = np.max(
            np.abs(qddot_reduced - qddot_full)
        )
    full_time = perf_counter() - full_start

    reduced_start = perf_counter()
    for theta, omega, forces in zip(
        theta_values, omega_values, force_values, strict=True
    ):
        reduced.acceleration(
            theta,
            omega,
            forces,
            external_crank_torque=external_crank_torque,
        )
    reduced_time = perf_counter() - reduced_start

    acceleration_scale = np.maximum(np.abs(full_accelerations), 1.0)
    relative_errors = acceleration_errors / acceleration_scale
    return {
        "sample_count": int(sample_count),
        "maximum_contact_position_error_m": float(
            np.max(contact_position_errors)
        ),
        "median_crank_acceleration_absolute_error_rad_s2": float(
            np.median(acceleration_errors)
        ),
        "maximum_crank_acceleration_absolute_error_rad_s2": float(
            np.max(acceleration_errors)
        ),
        "p95_crank_acceleration_relative_error": float(
            np.percentile(relative_errors, 95)
        ),
        "maximum_generalized_acceleration_error_rad_s2": float(
            np.max(generalized_acceleration_errors)
        ),
        "maximum_active_force_length_coefficient_error": float(
            np.max(force_length_errors)
        ),
        "maximum_force_velocity_coefficient_error": float(
            np.max(force_velocity_errors)
        ),
        "maximum_passive_force_coefficient_error": float(
            np.max(passive_force_errors)
        ),
        "full_constrained_evaluation_wall_time_s": float(full_time),
        "reduced_evaluation_wall_time_s": float(reduced_time),
        "numerical_evaluation_speedup": (
            float(full_time / reduced_time) if reduced_time > 0.0 else None
        ),
    }


def benchmark_reduced_casadi_mechanical_kernel(
    model_path: str | Path,
    reduced: ReducedCyclingDynamics,
    *,
    repeats: int = 1000,
    theta: float = -1.0,
    omega: float = -TWO_PI,
    muscle_forces: Sequence[float] | None = None,
    external_crank_torque: float = -0.2,
) -> dict:
    """Profile comparable full and reduced CasADi derivative kernels.

    Both functions map ``(theta, omega, muscle forces, external torque)`` to
    physical crank acceleration and evaluate its value, Jacobian and Hessian.
    The full expression reconstructs q(theta) and calls biorbd constrained
    forward dynamics; the reduced expression only evaluates Fourier profiles.
    """

    if repeats < 1:
        raise ValueError("CasADi benchmark repeats must be strictly positive.")
    import casadi as casadi

    try:
        import biorbd_casadi as symbolic_biorbd
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The full CasADi kernel benchmark requires biorbd_casadi."
        ) from error

    forces = (
        np.full(len(reduced.muscle_names), 100.0)
        if muscle_forces is None
        else np.asarray(muscle_forces, dtype=float).reshape(-1)
    )
    if forces.size != len(reduced.muscle_names):
        raise ValueError(
            f"Expected {len(reduced.muscle_names)} benchmark muscle forces."
        )

    theta_symbol = casadi.MX.sym("theta")
    omega_symbol = casadi.MX.sym("omega")
    forces_symbol = casadi.MX.sym("muscle_forces", len(reduced.muscle_names))
    external_symbol = casadi.MX.sym("external_crank_torque")
    inputs = casadi.vertcat(
        theta_symbol, omega_symbol, forces_symbol, external_symbol
    )

    q, tangent, curvature = reduced.kinematics.casadi_kinematics(theta_symbol)
    qdot = tangent * omega_symbol
    model = symbolic_biorbd.Model(str(Path(model_path)))
    available_muscles = _model_muscle_names(model)
    muscle_indices = [
        available_muscles.index(name) for name in reduced.muscle_names
    ]
    muscle_jacobian = model.musclesLengthJacobian(q).to_mx()[
        muscle_indices, :
    ]
    generalized_torque = -muscle_jacobian.T @ forces_symbol
    generalized_torque[reduced.crank_torque_dof_index] += external_symbol
    full_qddot = model.ForwardDynamicsConstraintsDirect(
        q,
        qdot,
        generalized_torque,
        model.getConstraints(),
    ).to_mx()
    full_acceleration = casadi.dot(
        tangent,
        full_qddot - curvature * omega_symbol**2,
    ) / casadi.dot(tangent, tangent)
    reduced_acceleration = reduced.casadi_acceleration(
        theta_symbol,
        omega_symbol,
        forces_symbol,
        external_symbol,
    )

    def derivative_function(name: str, expression):
        jacobian = casadi.jacobian(expression, inputs)
        hessian = casadi.hessian(expression, inputs)[0]
        outputs = casadi.vertcat(
            expression,
            casadi.reshape(jacobian, -1, 1),
            casadi.reshape(hessian, -1, 1),
        )
        return (
            casadi.Function(
                name,
                [inputs],
                [outputs],
                {"cse": True},
            ),
            outputs,
        )

    full_function, _ = derivative_function(
        "full_cycling_mechanical_kernel", full_acceleration
    )
    reduced_function, reduced_outputs = derivative_function(
        "reduced_cycling_mechanical_kernel", reduced_acceleration
    )
    argument = np.concatenate(
        (
            np.array([theta, omega]),
            forces,
            np.array([external_crank_torque]),
        )
    )
    full_result = np.asarray(full_function(argument), dtype=float).reshape(-1)
    reduced_result = np.asarray(
        reduced_function(argument), dtype=float
    ).reshape(-1)

    def evaluation_time(function) -> float:
        start = perf_counter()
        for _ in range(repeats):
            function(argument)
        return float(perf_counter() - start)

    full_time = evaluation_time(full_function)
    reduced_time = evaluation_time(reduced_function)
    report = {
        "repeats": int(repeats),
        "full_instruction_count": int(full_function.n_instructions()),
        "reduced_instruction_count": int(reduced_function.n_instructions()),
        "instruction_count_ratio": float(
            full_function.n_instructions()
            / reduced_function.n_instructions()
        ),
        "full_value_jacobian_hessian_wall_time_s": full_time,
        "reduced_value_jacobian_hessian_wall_time_s": reduced_time,
        "interpreted_derivative_kernel_speedup": float(
            full_time / reduced_time
        ),
        "crank_acceleration_difference_rad_s2": float(
            abs(full_result[0] - reduced_result[0])
        ),
    }

    try:
        reduced_jit = casadi.Function(
            "reduced_cycling_mechanical_kernel_jit",
            [inputs],
            [reduced_outputs],
            {
                "jit": True,
                "compiler": "shell",
                "jit_options": {"flags": "-O3"},
            },
        )
        reduced_jit(argument)
        reduced_jit_time = evaluation_time(reduced_jit)
        report.update(
            {
                "reduced_jit_supported": True,
                "reduced_jit_wall_time_s": reduced_jit_time,
                "full_interpreted_to_reduced_jit_speedup": float(
                    full_time / reduced_jit_time
                ),
            }
        )
    except RuntimeError as error:  # pragma: no cover - compiler dependent
        report.update(
            {
                "reduced_jit_supported": False,
                "reduced_jit_error": str(error),
            }
        )
    return report
