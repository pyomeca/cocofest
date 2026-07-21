import numpy as np
from types import SimpleNamespace
from bioptim import Solver

import examples.fes_multibody.cycling.cycling_pulse_width_mhe_acados_periodic as periodic_example
from cocofest.models.ding2007.ding2007_with_fatigue_periodic import (
    DingModelPulseWidthFrequencyWithFatiguePeriodic,
)
from examples.fes_multibody.cycling.cycling_pulse_width_mhe import MyCyclicNMPC
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


def test_acados_diagnostics_snapshot_is_independent_from_shared_solver(monkeypatch):
    live_diagnostics = {"status": 0, "residuals": np.array([1.0, 2.0])}
    monkeypatch.setattr(
        periodic_example,
        "collect_acados_diagnostics",
        lambda solution: live_diagnostics,
    )

    snapshot = periodic_example.snapshot_acados_diagnostics(SimpleNamespace())
    live_diagnostics["status"] = 2
    live_diagnostics["residuals"][0] = 99.0

    assert snapshot["status"] == 0
    np.testing.assert_allclose(snapshot["residuals"], [1.0, 2.0])


def test_acados_diagnostics_must_meet_strict_cache_tolerances():
    strict = {"residuals": np.array([0.1, 0.009, 0.0, 0.008])}
    relaxed_only = {"residuals": np.array([3.0, 0.009, 0.0, 0.008])}

    assert periodic_example.acados_diagnostics_meet_tolerances(
        strict, convergence_tolerance=1e-2, stationarity_tolerance=0.15
    )
    assert not periodic_example.acados_diagnostics_meet_tolerances(
        relaxed_only, convergence_tolerance=1e-2, stationarity_tolerance=0.15
    )


def test_codegen_signature_ignores_run_only_options():
    parser = periodic_example.build_argument_parser()
    reference = parser.parse_args([])
    longer_diagnostic_run = parser.parse_args(
        ["--n-windows", "20", "--acados-diagnostics", "--codegen-tag", "diagnostic"]
    )

    assert periodic_example._codegen_signature(
        reference
    ) == periodic_example._codegen_signature(longer_diagnostic_run)
    assert periodic_example._horizon_seed_cache_signature(
        reference
    ) == periodic_example._horizon_seed_cache_signature(longer_diagnostic_run)

    longer_diagnostic_run.stimulations_per_cycle += 1
    assert periodic_example._codegen_signature(
        reference
    ) != periodic_example._codegen_signature(longer_diagnostic_run)
    assert periodic_example._horizon_seed_cache_signature(
        reference
    ) != periodic_example._horizon_seed_cache_signature(longer_diagnostic_run)


def test_acados_qp_warm_start_cli_options_are_explicit():
    parser = periodic_example.build_argument_parser()
    default_args = parser.parse_args([])
    warm_args = parser.parse_args(
        [
            "--acados-qp-warm-start-level",
            "1",
            "--acados-warm-start-first-qp",
            "--acados-warm-start-first-qp-from-nlp",
        ]
    )

    assert default_args.acados_qp_warm_start_level == 0
    assert default_args.acados_warm_start_first_qp is False
    assert default_args.acados_warm_start_first_qp_from_nlp is False
    assert warm_args.acados_qp_warm_start_level == 1
    assert warm_args.acados_warm_start_first_qp is True
    assert warm_args.acados_warm_start_first_qp_from_nlp is True
    assert periodic_example._codegen_signature(
        default_args
    ) != periodic_example._codegen_signature(warm_args)


def test_signed_wheel_transfer_preserves_seam_and_terminal_turn():
    source = np.array([0.0, -1.0, -2.0, -6.0, -7.0, -8.0, -12.5])
    initial_guess = np.zeros((1, source.shape[0]))
    qdot_source = np.full_like(source, -6.5)
    qdot_initial_guess = np.zeros((1, source.shape[0]))
    nmpc = SimpleNamespace(
        nodes_per_cycle=3,
        cycle_duration=1.0,
        use_signed_wheel_shift=True,
        nlp=[
            SimpleNamespace(
                x_init={
                    "q": SimpleNamespace(init=initial_guess),
                    "qdot": SimpleNamespace(init=qdot_initial_guess),
                }
            )
        ],
        _wheel_cycle_shift=lambda states: -2 * np.pi,
    )
    nmpc.set_init_cyclical = lambda data, key, index: MyCyclicNMPC.set_init_cyclical(
        nmpc, data, key, index
    )
    states = {"q": source[None, :], "qdot": qdot_source[None, :]}

    MyCyclicNMPC.set_init_cyclical_wheel(nmpc, states, "q", 0)
    MyCyclicNMPC.set_init_cyclical_wheel_velocity(nmpc, states, "qdot", 0)

    transferred = initial_guess[0]
    source_final_increment = source[-1] - source[-2]
    transferred_seam_increment = transferred[3] - transferred[2]
    np.testing.assert_allclose(transferred_seam_increment, source_final_increment)
    np.testing.assert_allclose(transferred[-1], source[-1] - 2 * np.pi)
    expected_velocity_correction = -2 * np.pi - (source[-1] - source[-4])
    np.testing.assert_allclose(
        qdot_initial_guess[0, 3:],
        qdot_source[-4:] + expected_velocity_correction,
    )


def test_window_cache_callback_runs_before_window_is_advanced():
    events = []
    solution = SimpleNamespace(
        decision_states=lambda to_merge=None: events.append("decision_states") or {}
    )
    nmpc = SimpleNamespace(
        before_window_advance=lambda current_nmpc, current_solution: events.append(
            "before_window_advance"
        ),
        debugg_bounds=False,
        transfer_debug=False,
        update_stim=lambda: None,
        _sync_acados_state_bounds=lambda: None,
    )

    MyCyclicNMPC.advance_window_bounds_states(nmpc, solution)

    assert events == ["before_window_advance", "decision_states"]


def test_horizon_seed_recenters_kinematic_boundary_bounds():
    source = SimpleNamespace(
        decision_states=lambda to_merge=None: {
            "q": np.array([[-5.0, -6.0, -7.0]]),
            "qdot": np.array([[-1.0, -2.0, -3.0]]),
        },
        decision_controls=lambda to_merge=None: {
            "last_pulse_width_Biceps": np.array([[0.0002, 0.0003]])
        },
    )

    def guess(shape):
        return SimpleNamespace(init=np.zeros(shape))

    def bounds(half_width):
        return SimpleNamespace(
            min=np.array([[-half_width, -10.0, -half_width]]),
            max=np.array([[half_width, 10.0, half_width]]),
        )

    nlp = SimpleNamespace(
        x_init={"q": guess((1, 3)), "qdot": guess((1, 3))},
        u_init={"last_pulse_width_Biceps": guess((1, 2))},
        x_bounds={"q": bounds(0.1), "qdot": bounds(0.2)},
    )
    nmpc = SimpleNamespace(
        nlp=[nlp],
        _correct_init_guess_to_fit_bounds=lambda corrected_input: None,
        _sync_acados_state_bounds=lambda: None,
    )

    periodic_example.apply_solution_directly_to_periodic_nmpc_initial_guess(
        nmpc, source, recenter_kinematic_bounds=True
    )

    np.testing.assert_allclose(nlp.x_bounds["q"].min[:, [0, 2]], [[-5.1, -7.1]])
    np.testing.assert_allclose(nlp.x_bounds["q"].max[:, [0, 2]], [[-4.9, -6.9]])
    np.testing.assert_allclose(nlp.x_bounds["qdot"].min[:, [0, 2]], [[-1.2, -3.2]])
    np.testing.assert_allclose(nlp.x_bounds["qdot"].max[:, [0, 2]], [[-0.8, -2.8]])
