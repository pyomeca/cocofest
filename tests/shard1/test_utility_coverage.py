import os
import pickle
from types import SimpleNamespace

import matplotlib
import numpy as np
from casadi import Function, MX

matplotlib.use("Agg", force=True)

from cocofest.custom_constraints import CustomConstraint
from cocofest.fourier_approx import FourierSeries
from cocofest.result.pickle import SolutionToPickle


class _FakeBounds:
    def __init__(self, lower, upper):
        self.min = [[lower]]
        self.max = [[upper]]


class _FakeSolution:
    def __init__(self):
        self.ocp = SimpleNamespace(
            parameter_bounds={"pulse_width": _FakeBounds(0.1, 0.6)},
            nlp=[SimpleNamespace(model=SimpleNamespace(bio_model=SimpleNamespace(path="model.bioMod")))],
        )
        self.real_time_to_optimize = 1.25

    def decision_time(self, to_merge=None):
        return np.array([[0.0], [0.1], [0.1], [0.2]])

    def decision_states(self, to_merge=None):
        return {
            "single_state": np.array([[10.0, 11.0, 12.0, 13.0]]),
            "multi_state": np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        }

    def decision_controls(self, to_merge=None):
        return {"control": np.array([[0.0, 1.0, 2.0]])}

    def decision_parameters(self):
        return {"pulse_width": np.array([0.3])}


def test_fourier_series_reconstructs_first_harmonic(monkeypatch):
    series = FourierSeries()
    x = np.linspace(0.0, 1.0, 201)
    y = 2.0 + 3.0 * np.cos(2.0 * np.pi * x) + 4.0 * np.sin(2.0 * np.pi * x)

    coeffs = series.compute_real_fourier_coeffs(x, y, 1)
    np.testing.assert_allclose(coeffs[:, 0], [4.0, 3.0], atol=1e-3)
    np.testing.assert_allclose(coeffs[:, 1], [0.0, 4.0], atol=1e-3)

    approximation = series.fit_func_by_fourier_series_with_real_coeffs(x, coeffs)
    np.testing.assert_allclose(approximation, y, atol=1e-3)

    monkeypatch.setattr("matplotlib.pyplot.scatter", lambda *args, **kwargs: None)
    monkeypatch.setattr("matplotlib.pyplot.plot", lambda *args, **kwargs: None)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kwargs: None)
    np.testing.assert_allclose(series.fourier_approx(x, y, 1), y, atol=1e-3)


def test_fourier_series_supports_casadi_mode():
    series = FourierSeries()
    x = MX.sym("x")
    coeffs = np.array([[4.0, 0.0], [3.0, 4.0]])

    expression = series.fit_func_by_fourier_series_with_real_coeffs(x, coeffs, mode="casadi")
    value = Function("fourier", [x], [expression])(0.25)

    np.testing.assert_allclose(float(value), 6.0, atol=1e-12)


def test_solution_pickle_removes_duplicate_time_nodes(tmp_path):
    exporter = SolutionToPickle(_FakeSolution(), "solution.pkl", str(tmp_path) + os.sep)

    exporter.pickle()

    with open(tmp_path / "solution.pkl", "rb") as file:
        data = pickle.load(file)

    np.testing.assert_allclose(data["time"], [0.0, 0.1, 0.2])
    np.testing.assert_allclose(data["states"]["single_state"], [10.0, 11.0, 13.0])
    np.testing.assert_allclose(data["states"]["multi_state"], [[1.0, 2.0, 4.0], [5.0, 6.0, 8.0]])
    assert data["parameters_bounds"] == {"pulse_width": (0.1, 0.6)}
    assert data["bio_model_path"] == "model.bioMod"


def test_pulse_intensity_sliding_window_constraint_pads_missing_history():
    pulse_intensity = MX.sym("pulse_intensity")
    control = MX.sym("control", 3)
    controller = SimpleNamespace(
        parameters={"pulse_intensity": SimpleNamespace(cx=pulse_intensity)},
        controls={"pulse_intensity": SimpleNamespace(cx=control)},
        model=SimpleNamespace(min_pulse_intensity=lambda: 2.0),
    )

    constraint = CustomConstraint.pulse_intensity_sliding_window_constraint(controller, last_stim_idx=0)
    value = Function("constraint", [control, pulse_intensity], [constraint])([5.0, 6.0, 7.0], 1.0)

    np.testing.assert_allclose(np.array(value).reshape(-1), [3.0, 4.0, 6.0])


def test_pulse_intensity_sliding_window_constraint_trims_long_history():
    pulse_intensity = MX.sym("pulse_intensity", 4)
    control = MX.sym("control", 2)
    controller = SimpleNamespace(
        parameters={"pulse_intensity_Biceps": SimpleNamespace(cx=pulse_intensity)},
        controls={"pulse_intensity_Biceps": SimpleNamespace(cx=control)},
        model=SimpleNamespace(min_pulse_intensity=lambda: 2.0),
    )

    constraint = CustomConstraint.pulse_intensity_sliding_window_constraint(
        controller, last_stim_idx=3, muscle_name="Biceps"
    )
    value = Function("constraint", [control, pulse_intensity], [constraint])([10.0, 20.0], [1.0, 2.0, 3.0, 4.0])

    np.testing.assert_allclose(np.array(value).reshape(-1), [7.0, 16.0])
