import numpy as np
from casadi import DM, MX

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
