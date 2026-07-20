import numpy as np
from types import SimpleNamespace

from cocofest.models.ding2007.ding2007_with_fatigue_periodic import (
    DingModelPulseWidthFrequencyWithFatiguePeriodic,
)
from examples.fes_multibody.cycling.cycling_pulse_width_mhe_acados_periodic import (
    pulse_width_initial_guess_summary,
)


def _muscle_model():
    return DingModelPulseWidthFrequencyWithFatiguePeriodic(
        muscle_name="Biceps",
        stim_time=[0.0, 1 / 30],
    )


def _dynamics(model, pulse_width, numerical_timeseries):
    return np.asarray(
        model.system_dynamics(
            cn=0.5,
            cn_sum=1.0,
            f=0.0,
            a=model.a_scale,
            tau1=model.tau1_rest,
            km=model.km_rest,
            pulse_width=pulse_width,
            numerical_timeseries=numerical_timeseries,
        ),
        dtype=float,
    ).squeeze()


def test_periodic_calcium_forcing_uses_fixed_unit_intensity():
    model = _muscle_model()

    baseline = _dynamics(model, pulse_width=0.0002, numerical_timeseries=None)
    arbitrary_numerical_data = _dynamics(
        model,
        pulse_width=0.0002,
        numerical_timeseries=np.array([100.0]),
    )

    np.testing.assert_allclose(arbitrary_numerical_data, baseline)
    expected_gain = model.periodic_cn_sum_gain()
    np.testing.assert_allclose(baseline[1], -1.0 / model.tauc + expected_gain)


def test_pulse_width_changes_force_recruitment_not_calcium_forcing():
    model = _muscle_model()

    short_pulse = _dynamics(model, pulse_width=0.00015, numerical_timeseries=None)
    long_pulse = _dynamics(model, pulse_width=0.0006, numerical_timeseries=None)

    np.testing.assert_allclose(short_pulse[:2], long_pulse[:2])
    assert long_pulse[2] > short_pulse[2]


def test_pulse_width_summary_preserves_ipopt_control_variation():
    pulse_widths = np.array([[0.00015, 0.0003, 0.0006]])
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                u_init={
                    "last_pulse_width_Biceps": SimpleNamespace(init=pulse_widths),
                }
            )
        ]
    )

    summary = pulse_width_initial_guess_summary(nmpc)

    assert summary == [
        {
            "key": "last_pulse_width_Biceps",
            "minimum": 0.00015,
            "mean": 0.00035,
            "maximum": 0.0006,
            "span": 0.00045,
        }
    ]
