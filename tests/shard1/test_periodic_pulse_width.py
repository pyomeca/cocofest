import numpy as np
from casadi import Function, SX
from types import SimpleNamespace
from bioptim import Solver

import examples.fes_multibody.cycling.cycling_pulse_width_mhe_acados_periodic as periodic_example
from cocofest.models.ding2007.ding2007_with_fatigue_periodic import (
    DingModelPulseWidthFrequencyWithFatiguePeriodic,
)
from cocofest.models.ding2007.ding2007_with_fatigue_periodic_node import (
    DingModelPulseWidthFrequencyWithFatiguePeriodicNode,
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


def test_periodic_node_amplitude_matches_truncated_historical_sum():
    model = DingModelPulseWidthFrequencyWithFatiguePeriodicNode(
        muscle_name="Biceps",
        stim_time=[0.0, 1 / 30],
        sum_stim_truncation=6,
    )
    decay = np.exp(-(1 / 30) / model.tauc)
    ri = 1.0 + (model.get_r0(model.km_rest) - 1.0) * decay
    expected = decay**5 + ri * sum(decay**age for age in range(5))

    np.testing.assert_allclose(model.post_stimulation_amplitude(), expected)


def test_periodic_node_data_reconstructs_exact_within_interval_decay():
    model = DingModelPulseWidthFrequencyWithFatiguePeriodicNode(
        muscle_name="Biceps",
        stim_time=[0.0, 1 / 30],
        sum_stim_truncation=6,
    )
    data, _ = model.get_numerical_data_time_series(3, 0.1)
    stage_data = data["periodic_calcium"][:, 0, 1]
    local_time = 0.012
    absolute_time = stage_data[1] + local_time

    observed = float(model.calcium_history(np.array([absolute_time]), stage_data))
    expected = stage_data[0] * np.exp(-local_time / model.tauc)

    np.testing.assert_allclose(observed, expected)


def test_periodic_node_pulse_width_does_not_change_calcium_derivative():
    model = DingModelPulseWidthFrequencyWithFatiguePeriodicNode(
        muscle_name="Biceps",
        stim_time=[0.0, 1 / 30],
    )
    state = model.standard_rest_values().reshape(-1)
    state[0] = 0.5
    data = np.array([model.post_stimulation_amplitude(), 0.0])

    short = np.asarray(
        model.system_dynamics(
            states=state,
            controls=np.array([0.00015]),
            time=np.array([0.005]),
            numerical_timeseries=data,
        ),
        dtype=float,
    ).reshape(-1)
    long = np.asarray(
        model.system_dynamics(
            states=state,
            controls=np.array([0.0006]),
            time=np.array([0.005]),
            numerical_timeseries=data,
        ),
        dtype=float,
    ).reshape(-1)

    np.testing.assert_allclose(short[0], long[0])
    assert long[1] > short[1]


def test_time_dependent_rk4_map_retains_local_time_inside_interval():
    state = SX.sym("state", 1)
    control = SX.sym("control", 0)
    parameters = SX.sym("parameters", 0)
    local_time = SX.sym("local_time", 1)
    discrete_map = periodic_example.build_time_dependent_rk4_map(
        rhs=local_time,
        state=state,
        control=control,
        stage_parameters=parameters,
        local_time=local_time,
        interval_duration=2.0,
        n_substeps=2,
    )

    observed = float(Function("discrete_map", [state], [discrete_map])(3.0))

    np.testing.assert_allclose(observed, 5.0)


def test_periodic_node_irk_converts_acados_local_stage_time_to_absolute_time():
    observed = periodic_example._periodic_node_dynamics_time(
        "IRK", acados_time=0.012, interval_start=0.5, interval_duration=1 / 30
    )

    np.testing.assert_allclose(observed, 0.512)


def test_periodic_node_erk_uses_absolute_interval_midpoint():
    observed = periodic_example._periodic_node_dynamics_time(
        "ERK", acados_time=None, interval_start=0.5, interval_duration=1 / 30
    )

    np.testing.assert_allclose(observed, 0.5 + 1 / 60)


def test_acados_rhs_is_converted_to_scaled_state_derivative():
    rhs = SX.sym("rhs", 3)
    scaled_rhs = periodic_example._scaled_acados_dynamics_rhs(
        rhs, state_scaling=np.array([2.0, 10.0]), n_parameters=1
    )
    function = Function("scaled_rhs", [rhs], [scaled_rhs])

    observed = np.asarray(function(np.array([0.0, 4.0, 30.0]))).reshape(-1)

    np.testing.assert_allclose(observed, np.array([0.0, 2.0, 3.0]))


def test_high_accuracy_integrator_diagnostic_handles_time_dependent_dynamics():
    class Variables(dict):
        def __init__(self, values, shape):
            super().__init__(values)
            self.shape = shape

    state_variables = Variables({"x": SimpleNamespace(index=[0])}, shape=1)
    control_variables = Variables({"u": SimpleNamespace(index=[0])}, shape=1)
    nlp = SimpleNamespace(
        states=state_variables,
        controls=control_variables,
        x_init={"x": SimpleNamespace(init=np.array([[0.0, 0.125, 0.5]]))},
        u_init={"u": SimpleNamespace(init=np.zeros((1, 2)))},
        numerical_data_timeseries=None,
        dynamics_func=lambda time, state, control, parameters, algebraic, data: np.array(
            [time[0]]
        ),
    )
    nmpc = SimpleNamespace(nlp=[nlp], cycle_duration=1.0, cycle_len=2)

    rows = periodic_example.high_accuracy_integrator_map_diagnostics(
        nmpc, nodes=(0, 1), rk4_substeps=2
    )

    assert [row["node"] for row in rows] == [0, 1]
    assert max(row["trajectory_vs_reference"] for row in rows) < 1e-12
    assert max(row["rk4_vs_reference"] for row in rows) < 1e-12


def test_solution_trace_comparisons_reports_scaled_differences():
    reference = periodic_example._WarmupSolutionAdapter(
        states={"x": np.array([[0.0, 1.0, 2.0]])},
        controls={"u": np.array([[1.0, 2.0]])},
    )
    candidate = periodic_example._WarmupSolutionAdapter(
        states={"x": np.array([[0.0, 1.5, 2.0]])},
        controls={"u": np.array([[1.0, 3.0]])},
    )

    state_row = periodic_example.solution_trace_comparisons(
        reference, candidate, controls=False
    )[0]
    control_row = periodic_example.solution_trace_comparisons(
        reference, candidate, controls=True
    )[0]

    np.testing.assert_allclose(state_row["rmse"], 0.5 / np.sqrt(3))
    np.testing.assert_allclose(state_row["normalized_rmse"], 0.25 / np.sqrt(3))
    np.testing.assert_allclose(control_row["max_abs_error"], 1.0)


def test_pulse_width_trust_region_keeps_nodewise_centers():
    bounds = SimpleNamespace(
        min=np.array([[0.1, 0.1, 0.1]]), max=np.array([[0.6, 0.6, 0.6]])
    )
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                u_init={
                    "last_pulse_width_Biceps": SimpleNamespace(
                        init=np.array([[0.2, 0.4, 0.5]])
                    )
                },
                u_bounds={"last_pulse_width_Biceps": bounds},
            )
        ]
    )

    periodic_example.apply_pulse_width_control_trust_region(nmpc, radius=0.01)
    lower, upper = nmpc._cocofest_nodewise_control_bounds["last_pulse_width_Biceps"]

    np.testing.assert_allclose(lower, np.array([[0.19, 0.39, 0.49]]))
    np.testing.assert_allclose(upper, np.array([[0.21, 0.41, 0.51]]))


def test_continuation_source_inherits_requested_acados_tolerances(
    monkeypatch, tmp_path
):
    args = SimpleNamespace(
        cycles_per_window=2,
        n_windows=1,
        single_shot=False,
        acados_horizon_continuation=True,
        max_acados_iterations=100,
        acados_continuation_source_max_iterations=50,
        acados_tolerance=1e-4,
        acados_stationarity_tolerance=0.1,
        acados_diagnostics=True,
        codegen_tag="test",
    )
    observed = {}

    monkeypatch.setattr(
        periodic_example, "_continuation_cache_path", lambda _: tmp_path / "missing.npz"
    )

    def fake_solve_case(source_args, echo):
        observed["feasibility"] = source_args.acados_tolerance
        observed["stationarity"] = source_args.acados_stationarity_tolerance
        return {"status": 1, "solution": None}

    monkeypatch.setattr(periodic_example, "solve_case", fake_solve_case)

    with np.testing.assert_raises(RuntimeError):
        periodic_example.get_one_cycle_acados_continuation_source(args, echo=False)

    assert observed == {"feasibility": 1e-4, "stationarity": 0.1}


def test_proximal_phase_one_update_balances_reference_and_dynamics():
    observed = periodic_example._proximal_phase_one_update(
        reference=np.array([0.0, 10.0]),
        predicted=np.array([2.0, 20.0]),
        lower=np.array([-1.0, -1.0]),
        upper=np.array([1.0, 100.0]),
        proximity_weight=1.0,
        defect_weight=3.0,
    )

    np.testing.assert_allclose(observed, [1.0, 17.5])


def test_proximal_phase_one_rejects_collocation_layout():
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={"q": SimpleNamespace(init=np.zeros((3, 5)))},
                u_init={"u": SimpleNamespace(init=np.zeros((1, 1)))},
            )
        ]
    )

    with np.testing.assert_raises_regex(ValueError, "one state node"):
        periodic_example.project_full_dynamics_initial_guess(nmpc)


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


def test_tiled_fes_states_are_rolled_out_causally():
    class Variables(dict):
        def __init__(self, values, shape):
            super().__init__(values)
            self.shape = shape

    states = Variables(
        {
            "q": SimpleNamespace(index=[0]),
            "F_Biceps": SimpleNamespace(index=[1]),
        },
        shape=2,
    )
    controls = Variables(
        {"last_pulse_width_Biceps": SimpleNamespace(index=[0])}, shape=1
    )
    nlp = SimpleNamespace(
        model=SimpleNamespace(),
        states=states,
        controls=controls,
        x_init={
            "q": SimpleNamespace(init=np.zeros((1, 5))),
            "F_Biceps": SimpleNamespace(init=np.array([[0.0, 0.5, 1.0, 9.0, 9.0]])),
        },
        u_init={"last_pulse_width_Biceps": SimpleNamespace(init=np.ones((1, 4)))},
        numerical_data_timeseries=None,
        dynamics_func=lambda time, state, control, parameters, algebraic, data: np.array(
            [0.0, control[0]]
        ),
    )
    nmpc = SimpleNamespace(nlp=[nlp], cycle_duration=1.0, cycle_len=2)

    summary = periodic_example._rollout_tiled_fes_states(
        nmpc, start_node=2, n_substeps=2
    )

    assert summary["applied"] is True
    assert summary["start_node"] == 2
    np.testing.assert_allclose(nlp.x_init["F_Biceps"].init, [[0.0, 0.5, 1.0, 1.5, 2.0]])


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


def test_full_dynamics_transfer_rollout_reintegrates_appended_cycle():
    class Variables(dict):
        def __init__(self, *args, shape, **kwargs):
            super().__init__(*args, **kwargs)
            self.shape = shape

    state_variables = Variables(
        {
            "q": SimpleNamespace(index=[0]),
            "qdot": SimpleNamespace(index=[1]),
        },
        shape=2,
    )
    control_variables = Variables({"acceleration": SimpleNamespace(index=[0])}, shape=1)
    x_init = {
        "q": SimpleNamespace(init=np.array([[0.0, 0.5, 9.0, 9.0, 9.0]])),
        "qdot": SimpleNamespace(init=np.array([[1.0, 1.0, 9.0, 9.0, 9.0]])),
    }
    loose_bounds = SimpleNamespace(
        min=np.full((1, 3), -100.0), max=np.full((1, 3), 100.0)
    )
    nlp = SimpleNamespace(
        x_init=x_init,
        u_init={"acceleration": SimpleNamespace(init=np.zeros((1, 4)))},
        x_bounds={"q": loose_bounds, "qdot": loose_bounds},
        states=state_variables,
        controls=control_variables,
        numerical_data_timeseries=None,
        dynamics_func=lambda time, state, control, numerical, algebraic, parameters: np.array(
            [state[1], control[0]]
        ),
    )
    nmpc = SimpleNamespace(
        nlp=[nlp], nodes_per_cycle=2, cycle_duration=1.0, cycle_len=2
    )

    summary = periodic_example.rollout_transferred_cycle_full_dynamics(
        nmpc, n_substeps=2
    )

    assert summary["applied"] is True
    assert summary["start_node"] == 1
    assert summary["max_bound_violation"] == 0.0
    np.testing.assert_allclose(x_init["q"].init, [[0.0, 0.5, 1.0, 1.5, 2.0]])
    np.testing.assert_allclose(x_init["qdot"].init, [[1.0, 1.0, 1.0, 1.0, 1.0]])

    x_init["q"].init[:, 2:] = 9.0
    x_init["qdot"].init[:, 2:] = 9.0
    nlp.x_bounds["q"] = SimpleNamespace(
        min=np.full((1, 3), -100.0), max=np.full((1, 3), 1.0)
    )
    rejected = periodic_example.rollout_transferred_cycle_full_dynamics(
        nmpc, n_substeps=2, max_allowed_bound_violation=0.1
    )

    assert rejected["applied"] is False
    assert rejected["max_bound_violation"] == 1.0
    np.testing.assert_allclose(x_init["q"].init[:, 2:], 9.0)


def test_full_dynamics_rhs_passes_numerical_timeseries_as_data():
    recorded = {}

    def dynamics(
        time, states, controls, parameters, algebraic_states, numerical_timeseries
    ):
        recorded["parameters"] = np.asarray(parameters)
        recorded["algebraic_states"] = np.asarray(algebraic_states)
        recorded["numerical_timeseries"] = np.asarray(numerical_timeseries)
        return states

    nlp = SimpleNamespace(dynamics_func=dynamics)
    numerical_timeseries = np.array([1.0, 2.0, 3.0])

    result = periodic_example._full_dynamics_rhs(
        nlp,
        time=0.0,
        dt=0.1,
        state=np.array([4.0, 5.0]),
        control=np.array([6.0]),
        numerical_timeseries=numerical_timeseries,
    )

    np.testing.assert_allclose(result, [4.0, 5.0])
    assert recorded["parameters"].size == 0
    assert recorded["algebraic_states"].size == 0
    np.testing.assert_allclose(recorded["numerical_timeseries"], numerical_timeseries)
