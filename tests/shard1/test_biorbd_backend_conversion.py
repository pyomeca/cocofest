import numpy as np
from casadi import MX

from cocofest.dynamics.inverse_kinematics_and_dynamics import _biorbd_vector_to_numpy


class _EigenVector:
    def to_array(self):
        return np.array([[1.0], [2.0], [3.0]])


class _CasadiVector:
    def to_mx(self):
        return MX([1.0, 2.0, 3.0])


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
