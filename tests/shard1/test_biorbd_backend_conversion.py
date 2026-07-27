import numpy as np
import pytest
from casadi import DM, MX
from pathlib import Path

import cocofest.dynamics.inverse_kinematics_and_dynamics as cycling_dynamics
from cocofest.dynamics.inverse_kinematics_and_dynamics import (
    _biorbd_marker_jacobian,
    _biorbd_marker_residual,
    _biorbd_vector_to_numpy,
)


class _EigenVector:
    def to_array(self):
        return np.array([[1.0], [2.0], [3.0]])


class _CasadiVector:
    def to_mx(self):
        return MX([1.0, 2.0, 3.0])


class _CasadiJacobian:
    def __init__(self, values):
        self.values = values

    def to_mx(self):
        return MX(DM(self.values))


def test_biorbd_eigen_vector_is_converted_to_flat_numpy_array():
    np.testing.assert_array_equal(
        _biorbd_vector_to_numpy(_EigenVector()),
        np.array([1.0, 2.0, 3.0]),
    )


def test_biorbd_casadi_vector_is_converted_to_flat_numpy_array():
    np.testing.assert_array_equal(
        _biorbd_vector_to_numpy(_CasadiVector()),
        np.array([1.0, 2.0, 3.0]),
    )


def test_biorbd_casadi_marker_callbacks_are_numpy_compatible():
    model_markers = np.array([_CasadiVector(), _CasadiVector()], dtype=object)
    real_markers = np.array(
        [
            [0.5, 1.0],
            [1.5, 2.0],
            [2.5, 3.0],
        ]
    )
    np.testing.assert_array_equal(
        _biorbd_marker_residual(model_markers, real_markers),
        np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0]),
    )

    jacobian = _biorbd_marker_jacobian(
        np.array(
            [
                _CasadiJacobian([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                _CasadiJacobian([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
            ],
            dtype=object,
        )
    )
    np.testing.assert_array_equal(
        jacobian,
        np.arange(1.0, 13.0).reshape(6, 2),
    )


def test_real_biorbd_casadi_inverse_kinematics_and_jacobian(monkeypatch):
    biorbd_casadi = pytest.importorskip("biorbd_casadi")
    model_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "msk_models"
        / "Wu"
        / "Modified_Wu_Shoulder_Model_Cycling_for_IK.bioMod"
    )
    model = biorbd_casadi.Model(str(model_path))
    q = np.zeros(model.nbQ())
    markers = np.array(model.markers(q), dtype=object)
    reference = np.column_stack(
        [_biorbd_vector_to_numpy(marker) for marker in markers]
    )
    analytical = _biorbd_marker_jacobian(
        np.array(model.markersJacobian(q), dtype=object)
    )
    finite_difference = np.empty_like(analytical)
    step = 1e-7
    for index in range(model.nbQ()):
        perturbed = q.copy()
        perturbed[index] += step
        residual = _biorbd_marker_residual(
            np.array(model.markers(perturbed), dtype=object),
            reference,
        )
        finite_difference[:, index] = residual / step
    np.testing.assert_allclose(
        analytical,
        finite_difference,
        rtol=1e-5,
        atol=1e-7,
    )

    monkeypatch.setattr(cycling_dynamics, "biorbd", biorbd_casadi)
    q_solution, qdot, qddot = cycling_dynamics.inverse_kinematics_cycling(
        str(model_path),
        n_shooting=3,
        x_center=0.35,
        y_center=0.0,
        radius=0.1,
    )
    assert q_solution.shape == qdot.shape == qddot.shape == (model.nbQ(), 4)
    assert np.all(np.isfinite(q_solution))
