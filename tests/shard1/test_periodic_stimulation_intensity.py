from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.models.ding2007.ding2007_with_fatigue_periodic import (
    DingModelPulseWidthFrequencyWithFatiguePeriodic,
)
from examples.fes_multibody.cycling.cycling_pulse_width_mhe_acados_periodic import (
    build_warmup_stimulation_intensity_timeseries,
    cyclic_moving_average,
    set_nmpc_stimulation_intensity,
    stimulation_intensity_control_consistency,
    stimulation_intensity_from_trajectories,
)


def _intensity_args(alpha=1.0, smoothing_window=1):
    return SimpleNamespace(
        stimulation_intensity_ipopt_source="pulse_width_and_fatigue",
        stimulation_intensity_reference_pulse_width=0.000365702,
        stimulation_intensity_fatigue_gain=8.0,
        stimulation_intensity_homotopy_alpha=alpha,
        stimulation_intensity_smoothing_window=smoothing_window,
        stimulation_intensity_clip_low=0.0,
        stimulation_intensity_clip_high=2.5,
        stimulation_intensity_node_convention="current",
    )


def _muscle_model():
    return DingModelPulseWidthFrequencyWithFatiguePeriodic(
        muscle_name="Biceps",
        stim_time=[0.0, 1 / 30],
        stimulation_intensity_index=0,
    )


def test_pulse_width_homotopy_preserves_fatigue_baseline_at_zero():
    model = _muscle_model()
    pulse_width = np.array([0.00015, 0.0003, 0.0005, 0.0006])
    fatigue = np.array([model.a_scale, 0.9 * model.a_scale] * 2)

    intensity = stimulation_intensity_from_trajectories(
        _intensity_args(alpha=0.0), model, pulse_width, fatigue
    )

    expected_fatigue = 1.0 + 8.0 * np.maximum(
        0.0, (model.a_scale - fatigue) / model.a_scale
    )
    np.testing.assert_allclose(intensity, expected_fatigue)


def test_cyclic_smoothing_reduces_fast_pulse_width_contrast():
    model = _muscle_model()
    pulse_width = np.array([0.00014, 0.0006] * 3)
    fatigue = np.full(pulse_width.shape, model.a_scale)

    raw = stimulation_intensity_from_trajectories(
        _intensity_args(alpha=1.0, smoothing_window=1),
        model,
        pulse_width,
        fatigue,
    )
    smoothed = stimulation_intensity_from_trajectories(
        _intensity_args(alpha=1.0, smoothing_window=3),
        model,
        pulse_width,
        fatigue,
    )

    assert np.ptp(smoothed) < np.ptp(raw)
    np.testing.assert_allclose(cyclic_moving_average(np.ones(6), window=3), np.ones(6))


def test_cyclic_smoothing_rejects_even_windows():
    with pytest.raises(ValueError, match="must be odd"):
        cyclic_moving_average(np.ones(6), window=2)


def test_warmup_intensity_is_derived_from_transferred_controls_and_states():
    model = _muscle_model()
    pulse_width = np.array([[0.0002, 0.0003, 0.0004, 0.0005]])
    fatigue = np.array(
        [
            [
                model.a_scale,
                0.98 * model.a_scale,
                0.95 * model.a_scale,
                0.9 * model.a_scale,
                0.85 * model.a_scale,
            ]
        ]
    )
    nlp = SimpleNamespace(
        ns=4,
        model=SimpleNamespace(muscles_dynamics_model=[model]),
        u_init={"last_pulse_width_Biceps": SimpleNamespace(init=pulse_width)},
        x_init={"A_Biceps": SimpleNamespace(init=fatigue)},
        numerical_data_timeseries={},
    )
    nmpc = SimpleNamespace(nlp=[nlp])
    args = _intensity_args(alpha=0.5, smoothing_window=1)

    intensity = build_warmup_stimulation_intensity_timeseries(args, nmpc)
    set_nmpc_stimulation_intensity(nmpc, intensity)
    metrics = stimulation_intensity_control_consistency(args, nmpc)

    assert intensity.shape == (1, 1, 5)
    assert metrics[0]["muscle"] == "Biceps"
    assert metrics[0]["rmse"] == 0.0
    assert metrics[0]["max_abs_error"] == 0.0
