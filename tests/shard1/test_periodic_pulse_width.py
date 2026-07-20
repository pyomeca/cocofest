import numpy as np
from types import SimpleNamespace
from bioptim import Solver

from cocofest.models.ding2007.ding2007_with_fatigue_periodic import (
    DingModelPulseWidthFrequencyWithFatiguePeriodic,
)
from examples.fes_multibody.cycling.cycling_pulse_width_mhe_acados_periodic import (
    pulse_width_initial_guess_summary,
    set_acados_unsafe_option,
    tile_one_cycle_solution_to_periodic_nmpc,
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


def test_one_cycle_solution_is_tiled_with_wheel_and_fatigue_drift():
    source = SimpleNamespace(
        decision_states=lambda to_merge=None: {
            "q": np.array(
                [
                    [1.0, 1.1, 1.1, 1.0],
                    [2.0, 2.1, 2.1, 2.0],
                    [0.0, -2.0, -4.0, -2 * np.pi],
                ]
            ),
            "qdot": np.array(
                [
                    [0.0, 0.1, -0.1, 0.0],
                    [0.0, 0.2, -0.2, 0.0],
                    [-2 * np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi],
                ]
            ),
            "F_Biceps": np.array([[0.0, 2.0, 1.0, 0.0]]),
            "A_Biceps": np.array([[10.0, 9.0, 8.0, 7.0]]),
        },
        decision_controls=lambda to_merge=None: {
            "last_pulse_width_Biceps": np.array([[0.0002, 0.0004, 0.0003]])
        },
    )

    def guess(shape):
        return SimpleNamespace(init=np.zeros(shape))

    def bounds(rows):
        return SimpleNamespace(
            min=np.full((rows, 3), -100.0),
            max=np.full((rows, 3), 100.0),
        )

    nlp = SimpleNamespace(
        x_init={
            "q": guess((3, 7)),
            "qdot": guess((3, 7)),
            "F_Biceps": guess((1, 7)),
            "A_Biceps": guess((1, 7)),
        },
        u_init={"last_pulse_width_Biceps": guess((1, 6))},
        x_bounds={"q": bounds(3), "qdot": bounds(3)},
    )
    nmpc = SimpleNamespace(
        nlp=[nlp],
        _correct_init_guess_to_fit_bounds=lambda corrected_input: None,
        _sync_acados_state_bounds=lambda: None,
    )

    summary = tile_one_cycle_solution_to_periodic_nmpc(nmpc, source)

    np.testing.assert_allclose(
        nlp.u_init["last_pulse_width_Biceps"].init,
        [[0.0002, 0.0004, 0.0003, 0.0002, 0.0004, 0.0003]],
    )
    np.testing.assert_allclose(
        nlp.x_init["q"].init[2],
        [0.0, -2.0, -4.0, -2 * np.pi, -2.0 - 2 * np.pi, -4.0 - 2 * np.pi, -4 * np.pi],
    )
    np.testing.assert_allclose(
        nlp.x_init["F_Biceps"].init,
        [[0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0]],
    )
    np.testing.assert_allclose(
        nlp.x_init["A_Biceps"].init,
        [[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0]],
    )
    assert summary["repeat_count"] == 2
    assert summary["max_transfer_seam_error"] == 0.0


def test_unsafe_acados_option_is_initialized_on_each_solver_instance():
    first_solver = Solver.ACADOS()
    second_solver = Solver.ACADOS()

    set_acados_unsafe_option(first_solver, 0.5, "test_repeated_option")
    set_acados_unsafe_option(second_solver, 0.25, "test_repeated_option")

    assert first_solver._test_repeated_option == 0.5
    assert second_solver._test_repeated_option == 0.25
