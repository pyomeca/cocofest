import numpy as np
from casadi import Function, SX
from types import SimpleNamespace
from bioptim import Node, Solver

import examples.fes_multibody.cycling.cycling_pulse_width_mhe_acados_periodic as periodic_example
import examples.fes_multibody.cycling.cycling_fes_solver_comparison as comparison_example
from cocofest.optimization.receding_horizon_initial_guess import (
    audit_initial_guess,
    copy_container_values,
)
from cocofest.optimization.fes_nmpc_multibody import FesNmpcMsk
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

    nmpc.nlp[0].u_init["last_pulse_width_Biceps"].init[:, :] = [0.3, 0.5, 0.55]
    periodic_example.apply_pulse_width_control_trust_region(nmpc, radius=0.01)
    lower, upper = nmpc._cocofest_nodewise_control_bounds["last_pulse_width_Biceps"]

    np.testing.assert_allclose(lower, np.array([[0.29, 0.49, 0.54]]))
    np.testing.assert_allclose(upper, np.array([[0.31, 0.51, 0.56]]))


def test_control_homotopy_radii_are_parsed_as_an_increasing_sequence():
    parser = periodic_example.build_argument_parser()
    args = parser.parse_args(
        [
            "--acados-control-homotopy-radii",
            "1e-8,1e-7,1e-6",
            "--acados-control-homotopy-keep-final-radius",
            "--acados-control-homotopy-each-window",
            "--acados-control-homotopy-window-growth",
            "1.25",
            "--acados-control-homotopy-max-restarts",
            "3",
            "--acados-control-homotopy-stage-iterations",
            "40",
        ]
    )

    assert args.acados_control_homotopy_radii == (1e-8, 1e-7, 1e-6)
    assert args.acados_control_homotopy_keep_final_radius is True
    assert args.acados_control_homotopy_each_window is True
    assert args.acados_control_homotopy_window_growth == 1.25
    assert args.acados_control_homotopy_max_restarts == 3
    assert args.acados_control_homotopy_stage_iterations == 40


def test_proximal_control_weights_are_parsed_as_a_decreasing_sequence():
    args = periodic_example.build_argument_parser().parse_args(
        [
            "--acados-proximal-control-weights",
            "1e6,1e5,1e4",
            "--acados-proximal-control-each-window",
            "--acados-terminal-wheel-q-homotopy-slacks",
            "0.2,0.1,0.02",
            "--acados-terminal-wheel-q-homotopy-each-window",
            "--acados-proximal-control-stage-iterations",
            "30",
        ]
    )

    assert args.acados_proximal_control_weights == (1e6, 1e5, 1e4)
    assert args.acados_proximal_control_each_window is True
    assert args.acados_proximal_control_stage_iterations == 30


def test_terminal_wheel_bound_slacks_are_parsed_as_a_decreasing_sequence():
    args = periodic_example.build_argument_parser().parse_args(
        [
            "--acados-terminal-wheel-q-homotopy-slacks",
            "0.2,0.1,0.05,0.02",
            "--acados-terminal-wheel-q-homotopy-each-window",
        ]
    )

    assert args.acados_terminal_wheel_q_homotopy_slacks == (0.2, 0.1, 0.05, 0.02)
    assert args.acados_terminal_wheel_q_homotopy_each_window is True


def test_transfer_bound_homotopy_fractions_are_parsed():
    args = periodic_example.build_argument_parser().parse_args(
        [
            "--acados-transfer-irk-rollout",
            "--acados-transfer-bound-homotopy",
            "--acados-transfer-bound-homotopy-fractions",
            "0,0.5,1",
            "--acados-transfer-bound-homotopy-padding",
            "0.1",
            "--acados-transfer-bound-homotopy-iterations",
            "12",
        ]
    )

    assert args.acados_transfer_bound_homotopy is True
    assert args.acados_transfer_bound_homotopy_fractions == (0.0, 0.5, 1.0)
    assert args.acados_transfer_bound_homotopy_padding == 0.1
    assert args.acados_transfer_bound_homotopy_iterations == 12


def test_transfer_sqp_restart_options_are_parsed():
    parser = periodic_example.build_argument_parser()
    args = parser.parse_args(
        [
            "--acados-transfer-sqp-restarts",
            "4",
            "--acados-transfer-sqp-restart-iterations",
            "2",
            "--acados-transfer-sqp-restart-feasibility-tolerance",
            "0.02",
        ]
    )

    assert args.acados_transfer_sqp_restarts == 4
    assert args.acados_transfer_sqp_restart_iterations == 2
    assert args.acados_transfer_sqp_restart_feasibility_tolerance == 0.02
    assert periodic_example._codegen_signature(
        parser.parse_args([])
    ) != periodic_example._codegen_signature(args)


def test_acados_cyclical_transfer_extrapolates_by_default():
    args = periodic_example.build_argument_parser().parse_args([])

    assert args.acados_cyclical_transfer_mode == "extrapolate"
    assert args.acados_control_homotopy_stage_iterations == 54


def test_periodic_ipopt_window_cache_paths_are_window_specific(tmp_path, monkeypatch):
    args = periodic_example.build_argument_parser().parse_args([])
    model_path = tmp_path / "cycling.bioMod"
    model_path.write_text("version 4\n")
    monkeypatch.setattr(periodic_example, "_cache_root", lambda: tmp_path)

    first = periodic_example._periodic_ipopt_window_refinement_cache_path(
        args, model_path, 1
    )
    second = periodic_example._periodic_ipopt_window_refinement_cache_path(
        args, model_path, 2
    )

    assert first.parent == tmp_path
    assert first.name.endswith("_window_0001.npz")
    assert second.name.endswith("_window_0002.npz")
    assert first != second


def test_strict_fes_continuity_uses_a_distinct_periodic_ipopt_cache(
    tmp_path, monkeypatch
):
    parser = periodic_example.build_argument_parser()
    relaxed_args = parser.parse_args([])
    strict_args = parser.parse_args(["--acados-bind-first-node-fes-states"])
    model_path = tmp_path / "cycling.bioMod"
    model_path.write_text("version 4\n")
    monkeypatch.setattr(periodic_example, "_cache_root", lambda: tmp_path)

    relaxed = periodic_example._periodic_ipopt_refinement_cache_path(
        relaxed_args, model_path
    )
    strict = periodic_example._periodic_ipopt_refinement_cache_path(
        strict_args, model_path
    )

    assert relaxed != strict


def test_control_homotopy_stops_on_failure_and_restores_bounds(monkeypatch):
    class FakeSolver:
        def set_maximum_iterations(self, value):
            self.max_iterations = value

        def set_convergence_tolerance(self, value):
            self.tolerance = value

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
    solutions = iter(
        [
            SimpleNamespace(
                status=2,
                residuals=np.array([4e-4, 1e-6, 0.0, 1e-6]),
                solver_time_to_optimize=1.0,
                real_time_to_optimize=1.1,
            ),
            SimpleNamespace(
                status=0,
                residuals=np.zeros(4),
                solver_time_to_optimize=2.0,
                real_time_to_optimize=2.1,
            ),
            SimpleNamespace(
                status=2,
                residuals=np.array([2.0, 0.1, 0.0, 0.0]),
                solver_time_to_optimize=3.0,
                real_time_to_optimize=3.1,
            ),
        ]
    )
    stage_bounds = []

    def solve_stage():
        nodewise = getattr(nmpc, "_cocofest_nodewise_control_bounds", {})
        stage_bounds.append(
            {
                key: (lower.copy(), upper.copy())
                for key, (lower, upper) in nodewise.items()
            }
        )
        return next(solutions)

    applied_statuses = []
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda solution: {"residuals": solution.residuals},
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda _nmpc, solution: applied_statuses.append(solution.status),
    )

    summaries = periodic_example.run_acados_control_homotopy(
        nmpc,
        FakeSolver(),
        radii=(0.01, 0.02),
        convergence_tolerance=5e-4,
        fixed_control_tolerance=1e-8,
        echo=False,
        solve_stage=solve_stage,
        max_restarts=0,
        stage_iterations=25,
    )

    assert [summary["accepted"] for summary in summaries] == [True, True, False]
    assert applied_statuses == [2, 0]
    assert stage_bounds[0] == {}
    np.testing.assert_allclose(
        stage_bounds[1]["last_pulse_width_Biceps"][0], [[0.19, 0.39, 0.49]]
    )
    np.testing.assert_allclose(bounds.min, [[0.1, 0.1, 0.1]])
    np.testing.assert_allclose(bounds.max, [[0.6, 0.6, 0.6]])
    assert nmpc._cocofest_fix_controls_to_warmup is False
    assert nmpc._cocofest_nodewise_control_bounds == {}
    assert summaries[0]["stage"] == 0


def test_control_homotopy_restarts_a_nearly_feasible_stage(monkeypatch):
    class FakeSolver:
        def set_convergence_tolerance(self, value):
            self.tolerance = value

        def set_maximum_iterations(self, value):
            self.max_iterations = value

    bounds = SimpleNamespace(min=np.array([[0.1, 0.1]]), max=np.array([[0.6, 0.6]]))
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                u_init={
                    "last_pulse_width_Biceps": SimpleNamespace(
                        init=np.array([[0.2, 0.4]])
                    )
                },
                u_bounds={"last_pulse_width_Biceps": bounds},
            )
        ]
    )
    solutions = iter(
        [
            SimpleNamespace(
                status=2,
                residuals=np.array([0.2, 1e-6, 0.0, 1e-5]),
                solver_time_to_optimize=1.0,
                real_time_to_optimize=1.1,
            ),
            SimpleNamespace(
                status=0,
                residuals=np.array([4e-4, 1e-7, 0.0, 1e-6]),
                solver_time_to_optimize=2.0,
                real_time_to_optimize=2.1,
            ),
            SimpleNamespace(
                status=0,
                residuals=np.zeros(4),
                solver_time_to_optimize=3.0,
                real_time_to_optimize=3.1,
            ),
        ]
    )
    applied_statuses = []
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda solution: {"residuals": solution.residuals},
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda _nmpc, solution: applied_statuses.append(solution.status),
    )

    summaries = periodic_example.run_acados_control_homotopy(
        nmpc,
        FakeSolver(),
        radii=(0.01,),
        convergence_tolerance=5e-4,
        fixed_control_tolerance=1e-8,
        max_restarts=1,
        stage_iterations=25,
        echo=False,
        solve_stage=lambda: next(solutions),
    )

    assert [(item["stage"], item["attempt"]) for item in summaries] == [
        (0, 0),
        (0, 1),
        (1, 0),
    ]
    assert [item["accepted"] for item in summaries] == [False, True, True]
    assert summaries[0]["restartable"] is True
    assert applied_statuses == [2, 0, 0]


def test_proximal_control_continuation_reduces_weight_without_changing_bounds(
    monkeypatch,
):
    class FakeSolver:
        def set_convergence_tolerance(self, value):
            self.tolerance = value

        def set_maximum_iterations(self, value):
            self.max_iterations = value

    solutions = iter(
        [
            SimpleNamespace(
                status=0,
                residuals=np.zeros(4),
                solver_time_to_optimize=1.0,
                real_time_to_optimize=1.1,
            )
            for _ in range(3)
        ]
    )
    bounds = SimpleNamespace(
        min=np.array([[0.1, 0.1]]),
        max=np.array([[0.6, 0.6]]),
    )
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                u_bounds={"last_pulse_width_Biceps": bounds},
            )
        ]
    )
    applied_weights = []
    applied_statuses = []
    monkeypatch.setattr(
        periodic_example,
        "set_acados_runtime_control_regularization_weight",
        lambda _nmpc, weight: (
            applied_weights.append(weight)
            or {"applied": True, "reason": None, "weight": weight}
        ),
    )
    monkeypatch.setattr(
        periodic_example,
        "set_acados_runtime_max_iterations",
        lambda _nmpc, _iterations: True,
    )
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda solution: {"residuals": solution.residuals},
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda _nmpc, solution: applied_statuses.append(solution.status),
    )

    summaries = periodic_example.run_acados_proximal_control_continuation(
        nmpc,
        FakeSolver(),
        weights=(1e6, 1e5, 1e4),
        convergence_tolerance=5e-4,
        max_restarts=0,
        stage_iterations=20,
        echo=False,
        solve_stage=lambda: next(solutions),
    )

    assert [summary["accepted"] for summary in summaries] == [True, True, True]
    assert applied_weights == [1e6, 1e5, 1e4, 1e4]
    assert applied_statuses == [0, 0, 0]
    assert nmpc._cocofest_dual_warm_start_mode == "preserve"
    np.testing.assert_allclose(bounds.min, [[0.1, 0.1]])
    np.testing.assert_allclose(bounds.max, [[0.6, 0.6]])


def test_proximal_control_continuation_restarts_from_best_failed_qp_iterate(
    monkeypatch,
):
    class FakeSolver:
        def set_convergence_tolerance(self, value):
            self.tolerance = value

    solutions = iter(
        [
            SimpleNamespace(
                status=4,
                solver_time_to_optimize=1.0,
                real_time_to_optimize=1.1,
            ),
            SimpleNamespace(
                status=0,
                solver_time_to_optimize=2.0,
                real_time_to_optimize=2.1,
            ),
        ]
    )
    nmpc = SimpleNamespace(nlp=[SimpleNamespace(u_bounds={})])
    restored_iterates = []
    reset_calls = []
    monkeypatch.setattr(
        periodic_example,
        "set_acados_runtime_control_regularization_weight",
        lambda _nmpc, weight: {"applied": True, "weight": weight},
    )
    monkeypatch.setattr(
        periodic_example,
        "set_acados_runtime_max_iterations",
        lambda _nmpc, _iterations: True,
    )
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda solution: (
            {
                "residuals": np.array([1e8, 10.0, 0.0, 1e6]),
                "res_stat_all": np.array([3.0, 0.2, 1e8]),
                "res_eq_all": np.array([1.0, 2e-4, 10.0]),
                "res_ineq_all": np.zeros(3),
                "res_comp_all": np.array([0.1, 1e-5, 1e6]),
            }
            if solution.status == 4
            else {"residuals": np.zeros(4)}
        ),
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_acados_capsule_primal_to_initial_guess",
        lambda _nmpc, iterate_index=None: (
            restored_iterates.append(iterate_index)
            or {"applied": True, "iterate_index": iterate_index}
        ),
    )
    monkeypatch.setattr(
        periodic_example,
        "reset_acados_solver_memory",
        lambda _nmpc: reset_calls.append(True) or True,
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda _nmpc, _solution: None,
    )

    summaries = periodic_example.run_acados_proximal_control_continuation(
        nmpc,
        FakeSolver(),
        weights=(1e6,),
        convergence_tolerance=5e-4,
        max_restarts=1,
        stage_iterations=20,
        echo=False,
        solve_stage=lambda: next(solutions),
    )

    assert [summary["status"] for summary in summaries] == [4, 0]
    assert summaries[0]["restartable"] is True
    assert restored_iterates == [1]
    assert reset_calls == [True]


def test_ding_force_compensation_increases_width_when_capacity_drops():
    class FakeDingModel:
        muscle_name = "Biceps"

        def system_dynamics(
            self,
            cn,
            cn_sum,
            f,
            a,
            tau1,
            km,
            pulse_width,
        ):
            return np.array([0.0, 0.0, a * pulse_width - f, 0.0, 0.0, 0.0])

    state_values = {
        "Cn_Biceps": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Cn_sum_Biceps": [0.0, 0.0, 0.0, 0.0, 0.0],
        "F_Biceps": [2.0, 2.0, 2.0, 0.0, 0.0],
        "A_Biceps": [10.0, 10.0, 5.0, 5.0, 5.0],
        "Tau1_Biceps": [1.0, 1.0, 1.0, 1.0, 1.0],
        "Km_Biceps": [1.0, 1.0, 1.0, 1.0, 1.0],
    }
    control_key = "last_pulse_width_Biceps"
    controls = np.array([[0.2, 0.2, 0.2, 0.2]])
    nlp = SimpleNamespace(
        model=SimpleNamespace(muscles_dynamics_model=[FakeDingModel()]),
        x_init={
            key: SimpleNamespace(init=np.asarray([values], dtype=float))
            for key, values in state_values.items()
        },
        u_init={control_key: SimpleNamespace(init=controls)},
        u_bounds={
            control_key: SimpleNamespace(
                min=np.array([[0.1, 0.1, 0.1]]),
                max=np.array([[0.6, 0.6, 0.6]]),
            )
        },
    )
    nmpc = SimpleNamespace(nlp=[nlp], cycle_len=2, cycle_duration=2.0)

    summary = periodic_example.compensate_appended_pulse_widths_from_ding_force(
        nmpc,
        n_substeps=5,
        bisection_iterations=25,
    )

    assert summary["applied"] is True
    muscle_summary = summary["muscles"]["Biceps"]
    assert muscle_summary["gain_mean"] > 1.5
    assert (
        muscle_summary["compensated_force_rmse"] < muscle_summary["baseline_force_rmse"]
    )
    np.testing.assert_allclose(controls[0, 2:], 0.4, atol=1e-6)


def test_ding_force_compensation_uses_previous_solution_for_one_cycle_window():
    class FakeDingModel:
        muscle_name = "Biceps"

        def system_dynamics(self, cn, cn_sum, f, a, tau1, km, pulse_width):
            return np.array([0.0, 0.0, a * pulse_width - f, 0.0, 0.0, 0.0])

    state_values = {
        "Cn_Biceps": [0.0, 0.0, 0.0],
        "Cn_sum_Biceps": [0.0, 0.0, 0.0],
        "F_Biceps": [2.0, 0.0, 0.0],
        "A_Biceps": [5.0, 5.0, 5.0],
        "Tau1_Biceps": [1.0, 1.0, 1.0],
        "Km_Biceps": [1.0, 1.0, 1.0],
    }
    control_key = "last_pulse_width_Biceps"
    controls = np.array([[0.2, 0.2]])
    nlp = SimpleNamespace(
        model=SimpleNamespace(muscles_dynamics_model=[FakeDingModel()]),
        x_init={
            key: SimpleNamespace(init=np.asarray([values], dtype=float))
            for key, values in state_values.items()
        },
        u_init={control_key: SimpleNamespace(init=controls)},
        u_bounds={
            control_key: SimpleNamespace(
                min=np.array([[0.1, 0.1]]), max=np.array([[0.6, 0.6]])
            )
        },
    )
    previous_solution = SimpleNamespace(
        decision_states=lambda to_merge=None: {"F_Biceps": np.array([[2.0, 2.0, 2.0]])},
        decision_controls=lambda to_merge=None: {control_key: np.array([[0.2, 0.2]])},
    )
    nmpc = SimpleNamespace(nlp=[nlp], cycle_len=2, cycle_duration=2.0)

    summary = periodic_example.compensate_appended_pulse_widths_from_ding_force(
        nmpc,
        n_substeps=5,
        bisection_iterations=25,
        previous_solution=previous_solution,
    )

    assert summary["applied"] is True
    assert summary["previous_solution_used"] is True
    assert summary["start_node"] == 0
    np.testing.assert_allclose(controls, 0.4, atol=1e-6)


def test_ding_force_compensation_supports_periodic_node_calcium_forcing():
    class FakePeriodicNodeDingModel:
        muscle_name = "Biceps"

        @staticmethod
        def post_stimulation_amplitude():
            return 1.0

        def system_dynamics(self, states, controls, time, numerical_timeseries):
            cn, force, capacity, tau1, km = states
            return np.array([0.0, capacity * controls[0] - force, 0.0, 0.0, 0.0])

    state_values = {
        "Cn_Biceps": [0.0, 0.0, 0.0],
        "F_Biceps": [2.0, 0.0, 0.0],
        "A_Biceps": [5.0, 5.0, 5.0],
        "Tau1_Biceps": [1.0, 1.0, 1.0],
        "Km_Biceps": [1.0, 1.0, 1.0],
    }
    control_key = "last_pulse_width_Biceps"
    controls = np.array([[0.2, 0.2]])
    nlp = SimpleNamespace(
        model=SimpleNamespace(muscles_dynamics_model=[FakePeriodicNodeDingModel()]),
        x_init={
            key: SimpleNamespace(init=np.asarray([values], dtype=float))
            for key, values in state_values.items()
        },
        u_init={control_key: SimpleNamespace(init=controls)},
        u_bounds={
            control_key: SimpleNamespace(
                min=np.array([[0.1, 0.1]]), max=np.array([[0.6, 0.6]])
            )
        },
    )
    previous_solution = SimpleNamespace(
        decision_states=lambda to_merge=None: {"F_Biceps": np.array([[2.0, 2.0, 2.0]])},
        decision_controls=lambda to_merge=None: {control_key: np.array([[0.2, 0.2]])},
    )
    nmpc = SimpleNamespace(nlp=[nlp], cycle_len=2, cycle_duration=2.0)

    summary = periodic_example.compensate_appended_pulse_widths_from_ding_force(
        nmpc,
        n_substeps=5,
        bisection_iterations=25,
        previous_solution=previous_solution,
    )

    assert summary["applied"] is True
    np.testing.assert_allclose(controls, 0.4, atol=1e-6)


def test_terminal_wheel_bound_continuation_tightens_accepted_stages(monkeypatch):
    class FakeSolver:
        def set_convergence_tolerance(self, value):
            self.tolerance = value

    solutions = iter(
        [
            SimpleNamespace(
                status=0,
                residuals=np.array([1e-4, 1e-6, 0.0, 1e-5]),
                solver_time_to_optimize=0.1,
                real_time_to_optimize=0.2,
            )
            for _ in range(3)
        ]
    )
    applied_slacks = []
    applied_solutions = []
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_bounds={
                    "q": SimpleNamespace(
                        min=np.array(
                            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.2]]
                        ),
                        max=np.array(
                            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -0.8]]
                        ),
                    )
                }
            )
        ]
    )
    monkeypatch.setattr(
        periodic_example,
        "set_terminal_wheel_q_bound_slack",
        lambda _nmpc, slack: applied_slacks.append(slack),
    )
    monkeypatch.setattr(
        periodic_example,
        "set_acados_runtime_max_iterations",
        lambda _nmpc, _iterations: True,
    )
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda solution: {"residuals": solution.residuals},
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda _nmpc, solution: applied_solutions.append(solution.status),
    )

    summaries = periodic_example.run_acados_terminal_wheel_bound_continuation(
        nmpc,
        FakeSolver(),
        slacks=(0.2, 0.1, 0.02),
        convergence_tolerance=1e-3,
        stage_iterations=20,
        echo=False,
        solve_stage=lambda: next(solutions),
    )

    assert [item["accepted"] for item in summaries] == [True, True, True]
    assert applied_slacks == [0.2, 0.1, 0.02, 0.02]
    assert applied_solutions == [0, 0, 0]
    assert nmpc._cocofest_dual_warm_start_mode == "preserve"
    assert nmpc._cocofest_terminal_wheel_q_center == -1.0


def test_terminal_wheel_bound_is_recentered_before_a_new_continuation():
    sync_calls = []
    bounds = SimpleNamespace(
        min=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -7.0]]),
        max=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -6.8]]),
    )
    nmpc = SimpleNamespace(
        nlp=[SimpleNamespace(x_bounds={"q": bounds})],
        _sync_acados_state_bounds=lambda: sync_calls.append(True),
    )

    summary = periodic_example.recenter_terminal_wheel_q_bound_slack(nmpc, 0.2)

    assert summary == {
        "center": -6.9,
        "slack": 0.2,
        "lower": -7.1000000000000005,
        "upper": -6.7,
    }
    assert nmpc._cocofest_terminal_wheel_q_center == -6.9
    np.testing.assert_allclose(bounds.min[2, 2], -7.1)
    np.testing.assert_allclose(bounds.max[2, 2], -6.7)
    assert sync_calls == [True]


def test_acados_residual_history_selects_one_feasible_iterate():
    diagnostics = {
        "res_stat_all": np.array([5.0, 0.2, 1e-4]),
        "res_eq_all": np.array([1.0, 2e-4, 1e-2]),
        "res_ineq_all": np.array([0.0, 0.0, 0.0]),
        "res_comp_all": np.array([0.5, 1e-5, 1e-6]),
    }

    summary = periodic_example._acados_residual_history_summary(diagnostics)

    assert summary["best_index"] == 1
    np.testing.assert_allclose(summary["best"], [0.2, 2e-4, 0.0, 1e-5])
    np.testing.assert_allclose(summary["componentwise_best"], [1e-4, 2e-4, 0.0, 1e-6])


def test_control_homotopy_does_not_restart_a_linesearch_failure(monkeypatch):
    reset_calls = []

    class FakeAcadosSolver:
        def reset(self, reset_qp_solver_mem):
            reset_calls.append(reset_qp_solver_mem)

    class FakeSolver:
        nlp_solver_max_iter = 50

        def set_convergence_tolerance(self, value):
            self.tolerance = value

    nmpc = SimpleNamespace(
        ocp_solver=SimpleNamespace(ocp_solver=FakeAcadosSolver()),
        nlp=[SimpleNamespace(u_init={}, u_bounds={})],
    )
    solution = SimpleNamespace(
        status=3,
        residuals=np.array([1.0, 1e-6, 0.0, 1e-6]),
        solver_time_to_optimize=1.0,
        real_time_to_optimize=1.1,
    )
    applied_statuses = []
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda result: {"residuals": result.residuals},
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda _nmpc, result: applied_statuses.append(result.status),
    )

    summaries = periodic_example.run_acados_control_homotopy(
        nmpc,
        FakeSolver(),
        radii=(),
        convergence_tolerance=1e-3,
        fixed_control_tolerance=1e-8,
        max_restarts=3,
        stage_iterations=50,
        echo=False,
        solve_stage=lambda: solution,
    )

    assert summaries[0]["restartable"] is False
    assert summaries[0]["solver_reset"] is True
    assert applied_statuses == []
    assert reset_calls == [1]


def test_control_homotopy_reuses_compiled_acados_options(monkeypatch):
    option_change_flags = []
    runtime_options = []

    class FakeAcadosSolver:
        def options_set(self, key, value):
            runtime_options.append((key, value))

    class FakeSolver:
        nlp_solver_max_iter = 100

        def set_convergence_tolerance(self, value):
            self.tolerance = value

        def set_maximum_iterations(self, value):
            self.nlp_solver_max_iter = value

        def set_only_first_options_has_changed(self, value):
            option_change_flags.append(value)

    nmpc = SimpleNamespace(
        ocp_solver=SimpleNamespace(ocp_solver=FakeAcadosSolver()),
        nlp=[SimpleNamespace(u_init={}, u_bounds={})],
    )
    solution = SimpleNamespace(
        status=0,
        residuals=np.zeros(4),
        solver_time_to_optimize=1.0,
        real_time_to_optimize=1.1,
    )
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda _solution: {"residuals": np.zeros(4)},
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda *_args: None,
    )

    periodic_example.run_acados_control_homotopy(
        nmpc,
        FakeSolver(),
        radii=(),
        convergence_tolerance=1e-3,
        fixed_control_tolerance=1e-8,
        stage_iterations=50,
        echo=False,
        solve_stage=lambda: solution,
    )

    assert option_change_flags == [False]
    assert runtime_options == [("nlp_solver_max_iter", 50)]


def _benchmark_result(statuses, solver_success=False, success=False):
    cycle_count = 3
    shooting_per_cycle = 2
    return {
        "args": SimpleNamespace(stimulations_per_cycle=shooting_per_cycle),
        "window_statuses": statuses,
        "solver_success": solver_success,
        "success": success,
        "covered_cycles": cycle_count if solver_success else 0,
        "wheel_angle_trace": np.arange(cycle_count * shooting_per_cycle + 1),
        "state_traces": {
            "A_Biceps": np.linspace(100.0, 80.0, 7)[np.newaxis, :],
            "Tau1_Biceps": np.linspace(0.05, 0.06, 7)[np.newaxis, :],
        },
        "control_traces": {
            "last_pulse_width_Biceps": np.array(
                [[0.0002, 0.0006, 0.0003, 0.0004, 0.0005, 0.0006]]
            )
        },
        "control_bounds": {
            "last_pulse_width_Biceps": {"lower": 0.0001, "upper": 0.0006}
        },
    }


def test_benchmark_compares_only_the_successful_prefix():
    result = _benchmark_result([0, 0, 4])

    assert comparison_example._successful_prefix_length([0, 0, 4, 0]) == 2
    assert comparison_example._validated_cycle_count(result) == 2

    limited = comparison_example._truncate_result_to_cycles(result, 2)
    assert limited["wheel_angle_trace"].shape == (5,)
    assert limited["state_traces"]["A_Biceps"].shape == (1, 5)
    assert limited["control_traces"]["last_pulse_width_Biceps"].shape == (1, 4)


def test_benchmark_extracts_collocation_shooting_nodes_without_interpolation():
    result = _benchmark_result([0, 0], solver_success=True, success=True)
    result["args"] = SimpleNamespace(
        stimulations_per_cycle=2,
        ode_solver="collocation",
        collocation_degree=3,
    )
    result["exported_cycles"] = 2
    collocation_values = np.arange(17, dtype=float)
    result["wheel_angle_trace"] = collocation_values
    result["state_traces"] = {"q": collocation_values[np.newaxis, :]}
    result["control_traces"] = {"u": np.arange(4, dtype=float)[np.newaxis, :]}

    limited = comparison_example._truncate_result_to_cycles(result, 2)

    np.testing.assert_array_equal(limited["wheel_angle_trace"], [0, 4, 8, 12, 16])
    np.testing.assert_array_equal(limited["state_traces"]["q"], [[0, 4, 8, 12, 16]])
    np.testing.assert_array_equal(limited["control_traces"]["u"], [[0, 1, 2, 3]])


def test_endurance_metrics_report_fatigue_and_control_saturation():
    result = _benchmark_result([0, 0, 4])

    fatigue = comparison_example._fatigue_metrics(result, cycle_count=2)
    saturation = comparison_example._control_saturation_metrics(result, cycle_count=2)

    a_row = next(row for row in fatigue if row["key"] == "A_Biceps")
    np.testing.assert_allclose(a_row["relative_final"], (100 - 4 * 20 / 6) / 100)
    assert saturation[0]["upper_fraction"] == 0.25


def test_shared_capacity_limit_requires_two_independent_signals():
    ipopt = _benchmark_result([0, 0, 1])
    acados = _benchmark_result([0, 4])

    classification = comparison_example._shared_stop_classification(ipopt, acados)

    assert classification["label"] == "shared_capacity_limit_candidate"
    assert set(classification["evidence"]) == {
        "both_solvers_stop_at_similar_cycle",
        "pulse_width_upper_bound_active",
    }


def test_completed_endurance_horizon_uses_all_covered_cycles():
    result = _benchmark_result([0, 0], solver_success=True, success=True)

    assert comparison_example._validated_cycle_count(result) == 3
    assert comparison_example._stop_classification(result)["label"] == (
        "completed_requested_horizon"
    )


def test_receding_horizon_window_count_includes_single_cycle_horizons():
    assert periodic_example.receding_horizon_window_count(3, 1) == 3
    assert periodic_example.receding_horizon_window_count(30, 2) == 29

    with np.testing.assert_raises_regex(ValueError, "at least"):
        periodic_example.receding_horizon_window_count(1, 2)


def test_endurance_cli_stops_on_failure_and_keeps_robust_irk_defaults():
    args = comparison_example.build_cli().parse_args([])

    assert args.max_consecutive_failing == 1
    assert args.acados_integrator_type == "IRK"
    assert args.acados_sim_stages == 4
    assert args.acados_sim_steps == 5
    assert args.acados_dual_warm_start_mode == "reset"
    assert args.acados_transfer_pulse_width_trust_radius is None
    assert args.acados_proximal_control_weights is None
    assert args.periodic_ipopt_refinement_ode_solver == "target"


class _DualWarmStartSolver:
    def __init__(self, horizon, terminal_lam_size=1):
        self.values = {
            (stage, "lam"): np.array([stage + 1.0]) for stage in range(horizon + 1)
        }
        self.values[(horizon, "lam")] = np.full(terminal_lam_size, horizon + 1.0)
        self.values.update(
            {(stage, "pi"): np.array([10.0 + stage]) for stage in range(horizon)}
        )

    def get(self, stage, field):
        return self.values[(stage, field)].copy()

    def set(self, stage, field, values):
        self.values[(stage, field)] = np.asarray(values, dtype=float).copy()


def test_acados_dual_warm_start_can_reset_all_multipliers():
    solver = _DualWarmStartSolver(horizon=3)

    summary = periodic_example.apply_acados_dual_warm_start(
        solver, horizon=3, mode="reset", shift_stages=1
    )

    assert summary == {"mode": "reset", "shift_stages": 0, "zeroed_tail_stages": 4}
    assert all(not np.any(values) for values in solver.values.values())


def test_acados_dual_warm_start_can_shift_one_cycle_and_zero_tail():
    solver = _DualWarmStartSolver(horizon=3)

    summary = periodic_example.apply_acados_dual_warm_start(
        solver, horizon=3, mode="shift", shift_stages=1
    )

    assert summary == {"mode": "shift", "shift_stages": 1, "zeroed_tail_stages": 1}
    np.testing.assert_array_equal(
        [solver.values[(stage, "lam")][0] for stage in range(4)], [2, 3, 4, 0]
    )
    np.testing.assert_array_equal(
        [solver.values[(stage, "pi")][0] for stage in range(3)], [11, 12, 0]
    )


def test_acados_dual_shift_zeros_structurally_incompatible_terminal_multipliers():
    solver = _DualWarmStartSolver(horizon=3, terminal_lam_size=2)

    periodic_example.apply_acados_dual_warm_start(
        solver, horizon=3, mode="shift", shift_stages=1
    )

    np.testing.assert_array_equal(solver.values[(2, "lam")], [0.0])


def _ipopt_dual_warm_start_fixture():
    interface = SimpleNamespace(
        lam_g=None,
        lam_x=None,
        limits={"lbg": np.zeros(3), "x0": np.zeros(4)},
    )
    nmpc = SimpleNamespace(ocp_solver=interface, _is_warm_starting=False)
    solution = SimpleNamespace(
        lam_g=np.array([1.0, 2.0, 3.0]),
        lam_x=np.array([4.0, 5.0, 6.0, 7.0]),
    )
    return nmpc, solution


def test_ipopt_dual_warm_start_can_transfer_constraint_multipliers_only():
    nmpc, solution = _ipopt_dual_warm_start_fixture()

    summary = periodic_example.apply_ipopt_dual_warm_start(
        nmpc, solution, mode="constraints"
    )

    assert summary == {
        "mode": "constraints",
        "applied": True,
        "lam_g_size": 3,
        "lam_x_size": 0,
        "reason": None,
    }
    np.testing.assert_array_equal(nmpc.ocp_solver.lam_g, solution.lam_g)
    assert nmpc.ocp_solver.lam_x is None
    # Avoid Bioptim's aggressive set_warm_start_options(1e-10): the configured
    # IPOPT solver already accepts lam_g0 while retaining its robust mu_init.
    assert nmpc._is_warm_starting is False


def test_ipopt_dual_warm_start_can_include_bound_multipliers():
    nmpc, solution = _ipopt_dual_warm_start_fixture()

    summary = periodic_example.apply_ipopt_dual_warm_start(nmpc, solution, mode="all")

    assert summary["applied"] is True
    assert summary["lam_x_size"] == 4
    np.testing.assert_array_equal(nmpc.ocp_solver.lam_x, solution.lam_x)


def test_ipopt_dual_warm_start_can_transfer_bound_multipliers_only():
    nmpc, solution = _ipopt_dual_warm_start_fixture()

    summary = periodic_example.apply_ipopt_dual_warm_start(
        nmpc, solution, mode="bounds"
    )

    assert summary["applied"] is True
    assert summary["lam_g_size"] == 0
    assert summary["lam_x_size"] == 4
    assert nmpc.ocp_solver.lam_g is None
    np.testing.assert_array_equal(nmpc.ocp_solver.lam_x, solution.lam_x)


def test_ipopt_dual_warm_start_rejects_nonfinite_or_wrong_sized_duals():
    nmpc, solution = _ipopt_dual_warm_start_fixture()
    solution.lam_g = np.array([1.0, np.nan, 3.0])

    nonfinite = periodic_example.apply_ipopt_dual_warm_start(
        nmpc, solution, mode="constraints"
    )
    assert nonfinite["applied"] is False
    assert nonfinite["reason"] == "invalid_constraint_multipliers"

    solution.lam_g = np.ones(2)
    wrong_size = periodic_example.apply_ipopt_dual_warm_start(
        nmpc, solution, mode="constraints"
    )
    assert wrong_size["applied"] is False
    assert wrong_size["reason"] == "invalid_constraint_multipliers"


def test_ipopt_dual_warm_start_cli_defaults_to_bound_multipliers():
    periodic_args = periodic_example.build_argument_parser().parse_args([])
    comparison_args = comparison_example.build_cli().parse_args([])

    assert periodic_args.ipopt_dual_warm_start_mode == "bounds"
    assert comparison_args.ipopt_dual_warm_start_mode == "bounds"


def test_regularized_mhe_cli_exposes_previous_window_targets_and_terminal_slack():
    args = periodic_example.build_argument_parser().parse_args(
        [
            "--control-regularization-target-source",
            "previous",
            "--terminal-qdot-regularization-weight",
            "0.1",
            "--terminal-qdot-regularization-target-source",
            "first_node",
        ]
    )

    assert args.control_regularization_target_source == "previous"
    assert args.terminal_qdot_regularization_weight == 0.1
    assert args.terminal_qdot_regularization_target_source == "first_node"
    assert args.acados_terminal_wheel_q_slack == 0.2
    assert args.wheel_qdot_bound_margin == 3.0
    assert args.acados_globalization == "FUNNEL_L1PEN_LINESEARCH"
    assert args.periodic_ipopt_refinement_each_window is False


def test_previous_control_and_terminal_velocity_targets_are_recentered():
    control_penalty = SimpleNamespace(
        extra_parameters={"key": "last_pulse_width_Biceps"},
        node_idx=[0, 1, 2],
        node=[Node.ALL],
        rows=np.array([7]),
        target=None,
    )
    terminal_penalty = SimpleNamespace(
        extra_parameters={"key": "qdot"},
        node_idx=[2],
        node=[Node.END],
        rows=np.array([10, 11, 12]),
        target=np.zeros((3, 1)),
    )
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                J=[control_penalty, terminal_penalty],
                u_init={
                    "last_pulse_width_Biceps": SimpleNamespace(
                        init=np.array([[0.0002, 0.0004]])
                    )
                },
            )
        ]
    )

    keys = periodic_example.apply_initial_guess_control_regularization_targets(nmpc)
    updated_terminal = periodic_example.apply_terminal_qdot_regularization_target(
        nmpc, [-1.0, -2.0, -6.5]
    )

    assert keys == ["last_pulse_width_Biceps"]
    np.testing.assert_allclose(control_penalty.target, [[0.0002, 0.0004, 0.0004]])
    assert updated_terminal is True
    np.testing.assert_allclose(terminal_penalty.target[:, 0], [-1.0, -2.0, -6.5])


def test_updated_targets_are_copied_to_acados_cached_yrefs():
    control_penalty = SimpleNamespace(
        extra_parameters={"key": "last_pulse_width_Biceps"},
        node_idx=[0, 1],
        node=[Node.ALL],
        rows=np.array([7]),
        target=np.array([[0.0002, 0.0004]]),
    )
    terminal_penalty = SimpleNamespace(
        extra_parameters={"key": "qdot"},
        node_idx=[2],
        node=[Node.END],
        rows=np.array([10, 11, 12]),
        target=np.array([[-1.0], [-2.0], [-6.5]]),
    )
    interface = SimpleNamespace(
        y_ref=[[np.zeros((1, 1)), np.zeros((1, 1))]],
        y_ref_end=[np.zeros((3, 1))],
    )
    nlp = SimpleNamespace(
        J=[control_penalty, terminal_penalty],
        controls={"last_pulse_width_Biceps": SimpleNamespace(index=[7])},
        states={"qdot": SimpleNamespace(index=[10, 11, 12])},
    )
    nmpc = SimpleNamespace(nlp=[nlp], ocp_solver=interface)

    periodic_example.refresh_acados_cached_objective_targets(nmpc)

    np.testing.assert_allclose(
        [item[0, 0] for item in interface.y_ref[0]], [0.0002, 0.0004]
    )
    np.testing.assert_allclose(interface.y_ref_end[0][:, 0], [-1.0, -2.0, -6.5])


def test_all_muscle_targets_are_copied_to_acados_cached_yrefs():
    keys = [
        "last_pulse_width_Biceps",
        "last_pulse_width_Delt_ant",
        "last_pulse_width_Delt_post",
        "last_pulse_width_Triceps",
    ]
    penalties = [
        SimpleNamespace(
            extra_parameters={"key": key},
            node_idx=[0, 1],
            node=[Node.ALL],
            rows=np.array([index]),
            target=np.array([[index + 0.1, index + 0.2]]),
        )
        for index, key in enumerate(keys)
    ]
    interface = SimpleNamespace(
        y_ref=[[np.zeros((1, 1)), np.zeros((1, 1))] for _ in penalties],
        y_ref_end=[],
    )
    nlp = SimpleNamespace(
        J=penalties,
        controls={
            key: SimpleNamespace(index=[index]) for index, key in enumerate(keys)
        },
        states={},
    )

    periodic_example.refresh_acados_cached_objective_targets(
        SimpleNamespace(nlp=[nlp], ocp_solver=interface)
    )

    for index, references in enumerate(interface.y_ref):
        np.testing.assert_allclose(
            [reference[0, 0] for reference in references],
            [index + 0.1, index + 0.2],
        )


def test_runtime_proximal_weight_updates_only_pulse_width_blocks():
    class LagrangeFunction:
        pass

    class MayerFunction:
        pass

    class FakeGeneratedSolver:
        def __init__(self):
            self.calls = []

        def cost_set(self, stage, field, value, api=None):
            self.calls.append((stage, field, np.array(value, copy=True)))

    def penalty(key, node, size, penalty_type):
        return SimpleNamespace(
            extra_parameters={"key": key},
            node=[node],
            function=[SimpleNamespace(numel_out=lambda: size)],
            type=SimpleNamespace(get_type=lambda: penalty_type),
        )

    penalties = [
        penalty(None, Node.ALL, 2, LagrangeFunction),
        penalty("last_pulse_width_Biceps", Node.ALL, 1, LagrangeFunction),
        penalty("last_pulse_width_Triceps", Node.ALL, 1, LagrangeFunction),
        penalty("qdot", Node.END, 3, MayerFunction),
    ]
    generated_solver = FakeGeneratedSolver()
    interface = SimpleNamespace(
        ocp_solver=generated_solver,
        acados_ocp=SimpleNamespace(solver_options=SimpleNamespace(N_horizon=3)),
        W=np.diag([7.0, 8.0, 9.0, 10.0]),
        W_0=np.diag([7.0, 8.0, 9.0, 10.0]),
        W_e=np.diag([7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]),
    )
    nmpc = SimpleNamespace(
        nlp=[SimpleNamespace(J=penalties)],
        ocp_solver=interface,
    )

    summary = periodic_example.set_acados_runtime_control_regularization_weight(
        nmpc, 100.0
    )

    assert summary["applied"] is True
    assert [call[0] for call in generated_solver.calls] == [0, 1, 2, 3]
    for _, field, matrix in generated_solver.calls:
        assert field == "W"
        np.testing.assert_allclose(np.diag(matrix)[:2], [7.0, 8.0])
        np.testing.assert_allclose(np.diag(matrix)[2:4], [100.0, 100.0])
    np.testing.assert_allclose(
        np.diag(generated_solver.calls[-1][2])[4:], [11.0, 12.0, 13.0]
    )


def test_terminal_wheel_slack_is_independent_from_first_node_slack():
    q_bounds = SimpleNamespace(
        min=np.zeros((3, 3)),
        max=np.zeros((3, 3)),
    )
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={
                    "q": SimpleNamespace(
                        init=np.array([[0, 0, 0], [0, 0, 0], [-5, -6, -7]])
                    )
                },
                x_bounds={"q": q_bounds},
            )
        ],
        _sync_acados_state_bounds=lambda: None,
    )

    periodic_example.set_terminal_wheel_q_bound_slack(nmpc, 0.2)

    assert q_bounds.min[2, 0] == 0.0
    assert q_bounds.max[2, 0] == 0.0
    np.testing.assert_allclose(q_bounds.min[2, 2], -7.2)
    np.testing.assert_allclose(q_bounds.max[2, 2], -6.8)


class _BoundComplementaritySolver:
    def get(self, stage, field):
        values = {
            "x": np.array([3.0]),
            "u": np.array([2.0]),
            "lam": np.array([0.1, 0.2, 0.3, 0.4]),
        }
        return values[field]

    def constraints_get(self, stage, field):
        values = {
            "lbu": np.array([0.0]),
            "ubu": np.array([4.0]),
            "lbx": np.array([1.0]),
            "ubx": np.array([5.0]),
        }
        return values[field]


def test_acados_bound_complementarity_identifies_largest_product():
    rows = periodic_example._acados_bound_complementarity_rows(
        _BoundComplementaritySolver(),
        n_stages=1,
        state_labels=["force"],
        control_labels=["pulse_width"],
    )

    assert rows[0] == {
        "stage": 0,
        "variable": "force",
        "side": "upper",
        "value": 3.0,
        "bound": 5.0,
        "distance": 2.0,
        "multiplier": 0.4,
        "product": 0.8,
    }


def test_control_bounds_summary_preserves_physical_units():
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                controls={"last_pulse_width_Biceps": SimpleNamespace()},
                u_bounds={
                    "last_pulse_width_Biceps": SimpleNamespace(
                        min=np.array([[0.000131405]]),
                        max=np.array([[0.0006]]),
                    )
                },
            )
        ]
    )

    assert periodic_example._control_bounds_summary(nmpc) == {
        "last_pulse_width_Biceps": {"lower": 0.000131405, "upper": 0.0006}
    }

    nmpc._cocofest_original_control_bounds = {
        "last_pulse_width_Biceps": (
            np.array([[0.0001]]),
            np.array([[0.0007]]),
        )
    }
    assert periodic_example._control_bounds_summary(nmpc) == {
        "last_pulse_width_Biceps": {"lower": 0.0001, "upper": 0.0007}
    }


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
    source = np.array(
        [
            0.0,
            -1.0,
            -2.0,
            -2 * np.pi,
            -2 * np.pi - 1.0,
            -2 * np.pi - 2.0,
            -4 * np.pi,
        ]
    )
    initial_guess = np.zeros((1, source.shape[0]))
    qdot_source = np.full_like(source, -6.5)
    qdot_initial_guess = np.zeros((1, source.shape[0]))
    nmpc = SimpleNamespace(
        nodes_per_cycle=3,
        cycle_duration=1.0,
        use_signed_wheel_shift=True,
        transfer_initial_guess_mode="anchored",
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
    np.testing.assert_allclose(transferred[:4], source[3:])
    np.testing.assert_allclose(transferred[4:], source[4:] - 2 * np.pi)
    transferred_seam_increment = transferred[4] - transferred[3]
    np.testing.assert_allclose(transferred_seam_increment, source[4] - source[3])
    np.testing.assert_allclose(transferred[-1], source[-1] - 2 * np.pi)
    np.testing.assert_allclose(qdot_initial_guess[0], qdot_source)


def test_one_cycle_transfer_preserves_profiles_instead_of_broadcasting_terminal_node():
    nodes_per_cycle = 3
    q_source = np.array([0.0, -1.0, -2.0, -2 * np.pi])
    cyclical_source = np.array([1.0, 2.0, 3.0, 1.5])
    fatigue_source = np.array([10.0, 10.5, 11.0, 12.0])
    nmpc = SimpleNamespace(
        nodes_per_cycle=nodes_per_cycle,
        use_signed_wheel_shift=True,
        transfer_initial_guess_mode="anchored",
        nlp=[
            SimpleNamespace(
                x_init={
                    "q": SimpleNamespace(init=np.zeros((1, nodes_per_cycle + 1))),
                    "F": SimpleNamespace(init=np.zeros((1, nodes_per_cycle + 1))),
                    "A": SimpleNamespace(init=np.zeros((1, nodes_per_cycle + 1))),
                }
            )
        ],
        _wheel_cycle_shift=lambda _states: -2 * np.pi,
    )
    states = {
        "q": q_source[None, :],
        "F": cyclical_source[None, :],
        "A": fatigue_source[None, :],
    }

    MyCyclicNMPC.set_init_cyclical_wheel(nmpc, states, "q", 0)
    MyCyclicNMPC.set_init_cyclical(nmpc, states, "F", 0)
    MyCyclicNMPC.set_init_continuous(nmpc, states, "A", 0)

    np.testing.assert_allclose(nmpc.nlp[0].x_init["q"].init[0], q_source - 2 * np.pi)
    np.testing.assert_allclose(nmpc.nlp[0].x_init["F"].init[0], cyclical_source)
    np.testing.assert_allclose(nmpc.nlp[0].x_init["A"].init[0], fatigue_source + 2.0)


def test_cyclical_transfer_keeps_complete_state_cycle_and_repeats_controls():
    nmpc = SimpleNamespace(
        nodes_per_cycle=3,
        transfer_initial_guess_mode="anchored",
        nlp=[
            SimpleNamespace(
                x_init={"state": SimpleNamespace(init=np.zeros((1, 7)))},
                u_init={"control": SimpleNamespace(init=np.zeros((1, 6)))},
            )
        ],
    )
    states = {"state": np.arange(7, dtype=float)[None, :]}
    controls = {"control": np.arange(6, dtype=float)[None, :]}

    MyCyclicNMPC.set_init_cyclical(nmpc, states, "state", 0)
    MyCyclicNMPC.set_init_cyclical(nmpc, controls, "control", 0, state=False)

    np.testing.assert_allclose(
        nmpc.nlp[0].x_init["state"].init[0], [3, 4, 5, 6, 7, 8, 9]
    )
    np.testing.assert_allclose(
        nmpc.nlp[0].u_init["control"].init[0], [3, 4, 5, 3, 4, 5]
    )


def test_cyclical_transfer_can_repeat_states_without_extrapolating_cycle_delta():
    nmpc = SimpleNamespace(
        nodes_per_cycle=3,
        transfer_initial_guess_mode="anchored",
        repeat_cyclical_state_initial_guess=True,
        nlp=[SimpleNamespace(x_init={"state": SimpleNamespace(init=np.zeros((1, 7)))})],
    )
    states = {"state": np.arange(7, dtype=float)[None, :]}

    MyCyclicNMPC.set_init_cyclical(nmpc, states, "state", 0)

    np.testing.assert_allclose(
        nmpc.nlp[0].x_init["state"].init[0], [3, 4, 5, 6, 4, 5, 6]
    )


def test_continuous_transfer_extrapolates_cycle_delta_without_duplicate_seam():
    nmpc = SimpleNamespace(
        nodes_per_cycle=3,
        transfer_initial_guess_mode="anchored",
        nlp=[
            SimpleNamespace(x_init={"fatigue": SimpleNamespace(init=np.zeros((1, 7)))})
        ],
    )
    states = {"fatigue": np.array([[0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 14.0]])}

    MyCyclicNMPC.set_init_continuous(nmpc, states, "fatigue", 0)

    np.testing.assert_allclose(
        nmpc.nlp[0].x_init["fatigue"].init[0], [10, 11, 12, 14, 15, 16, 18]
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
        "q": SimpleNamespace(init=np.array([[0.0, 0.5, 1.0, 9.0, 9.0]])),
        "qdot": SimpleNamespace(init=np.array([[1.0, 1.0, 1.0, 9.0, 9.0]])),
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
    assert summary["start_node"] == 2
    assert summary["max_bound_violation"] == 0.0
    np.testing.assert_allclose(x_init["q"].init, [[0.0, 0.5, 1.0, 1.5, 2.0]])
    np.testing.assert_allclose(x_init["qdot"].init, [[1.0, 1.0, 1.0, 1.0, 1.0]])

    x_init["q"].init[:, 3:] = 9.0
    x_init["qdot"].init[:, 3:] = 9.0
    nlp.x_bounds["q"] = SimpleNamespace(
        min=np.full((1, 3), -100.0), max=np.full((1, 3), 1.0)
    )
    rejected = periodic_example.rollout_transferred_cycle_full_dynamics(
        nmpc, n_substeps=2, max_allowed_bound_violation=0.1
    )

    assert rejected["applied"] is False
    assert rejected["max_bound_violation"] == 1.0
    np.testing.assert_allclose(x_init["q"].init[:, 3:], 9.0)


def test_appended_pulse_width_scaling_preserves_retained_cycle_and_clips():
    values = np.array([[0.1, 0.2, 0.2, 0.3]])
    nmpc = SimpleNamespace(
        nodes_per_cycle=2,
        nlp=[
            SimpleNamespace(
                u_init={"last_pulse_width_Biceps": SimpleNamespace(init=values)},
                u_bounds={
                    "last_pulse_width_Biceps": SimpleNamespace(
                        min=np.zeros((1, 3)), max=np.full((1, 3), 0.6)
                    )
                },
            )
        ],
    )

    summary = periodic_example.scale_appended_pulse_width_controls(nmpc, 2.5)

    np.testing.assert_allclose(values, [[0.1, 0.2, 0.5, 0.6]])
    assert summary["start_node"] == 2
    assert summary["controls"]["last_pulse_width_Biceps"]["clipped_count"] == 1


def test_acados_irk_transfer_rollout_uses_scaled_variables_and_stage_data():
    class Variables(dict):
        def __init__(self, *args, shape, **kwargs):
            super().__init__(*args, **kwargs)
            self.shape = shape

    class FakeSimulator:
        def __init__(self):
            self.acados_sim = SimpleNamespace(dims=SimpleNamespace(nx=2, nu=1))
            self.calls = []
            self.settings = []

        def set(self, field, value):
            assert field in ("T", "t0")
            self.settings.append((field, value.copy()))

        def simulate(self, x, u, p):
            self.calls.append((x.copy(), u.copy(), p.copy()))
            return x + np.array([u[0], p[0]])

        def get(self, field):
            assert field == "time_tot"
            return 0.001

    states = Variables(
        {"q": SimpleNamespace(index=[0]), "qdot": SimpleNamespace(index=[1])},
        shape=2,
    )
    controls = Variables({"acceleration": SimpleNamespace(index=[0])}, shape=1)
    x_init = {
        "q": SimpleNamespace(init=np.array([[0.0, 1.0, 2.0, 9.0, 9.0]])),
        "qdot": SimpleNamespace(init=np.array([[2.0, 2.0, 2.0, 9.0, 9.0]])),
    }
    scaling = lambda value: SimpleNamespace(  # noqa: E731 - compact test fixture.
        scaling=np.array([[value]])
    )
    loose_bounds = SimpleNamespace(
        min=np.full((1, 3), -100.0), max=np.full((1, 3), 100.0)
    )
    numerical_data = {"periodic_calcium": np.arange(1.0, 6.0).reshape((1, 1, 5))}
    nlp = SimpleNamespace(
        x_init=x_init,
        u_init={"acceleration": SimpleNamespace(init=np.full((1, 4), 6.0))},
        x_bounds={"q": loose_bounds, "qdot": loose_bounds},
        states=states,
        controls=controls,
        x_scaling={"q": scaling(2.0), "qdot": scaling(4.0)},
        u_scaling={"acceleration": scaling(3.0)},
        numerical_data_timeseries=numerical_data,
    )
    simulator = FakeSimulator()
    nmpc = SimpleNamespace(
        nlp=[nlp],
        nodes_per_cycle=2,
        cycle_duration=1.0,
        cycle_len=2,
        _cocofest_acados_sim_solver=simulator,
    )

    summary = periodic_example.rollout_transferred_cycle_acados_irk(nmpc)

    assert summary["applied"] is True
    assert summary["simulator_built"] is False
    np.testing.assert_allclose(summary["simulation_time_s"], 0.002)
    np.testing.assert_allclose(x_init["q"].init, [[0.0, 1.0, 2.0, 6.0, 10.0]])
    np.testing.assert_allclose(x_init["qdot"].init, [[2.0, 2.0, 2.0, 14.0, 30.0]])
    np.testing.assert_allclose(simulator.calls[0][0], [1.0, 0.5])
    np.testing.assert_allclose(simulator.calls[0][1], [2.0])
    np.testing.assert_allclose(simulator.calls[0][2], [3.0])
    assert [field for field, _ in simulator.settings] == ["T", "t0", "T", "t0"]
    np.testing.assert_allclose(simulator.settings[0][1], [0.5])
    np.testing.assert_allclose(simulator.settings[1][1], [1.0])


def test_acados_irk_transfer_rejects_dimension_mismatch_without_mutating_guess():
    class Variables(dict):
        def __init__(self, *args, shape, **kwargs):
            super().__init__(*args, **kwargs)
            self.shape = shape

    simulator = SimpleNamespace(
        acados_sim=SimpleNamespace(dims=SimpleNamespace(nx=3, nu=1))
    )
    states = Variables({"q": SimpleNamespace(index=[0])}, shape=1)
    controls = Variables({"u": SimpleNamespace(index=[0])}, shape=1)
    guess = np.zeros((1, 3))
    nlp = SimpleNamespace(
        x_init={"q": SimpleNamespace(init=guess)},
        u_init={"u": SimpleNamespace(init=np.zeros((1, 2)))},
        x_bounds={"q": SimpleNamespace(min=np.full((1, 3), -1.0), max=np.ones((1, 3)))},
        states=states,
        controls=controls,
        x_scaling={"q": SimpleNamespace(scaling=np.ones((1, 1)))},
        u_scaling={"u": SimpleNamespace(scaling=np.ones((1, 1)))},
    )
    nmpc = SimpleNamespace(
        nlp=[nlp], nodes_per_cycle=1, _cocofest_acados_sim_solver=simulator
    )

    with np.testing.assert_raises_regex(ValueError, "dimensions do not match"):
        periodic_example.rollout_transferred_cycle_acados_irk(nmpc)

    np.testing.assert_allclose(guess, 0.0)


def test_transfer_bound_homotopy_never_relaxes_first_node():
    bounds = SimpleNamespace(
        min=np.array([[-0.1, -1.0, -2.1]]),
        max=np.array([[0.1, 1.0, -1.9]]),
    )
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={"qdot": SimpleNamespace(init=np.array([[0.0, -4.0, -5.0]]))},
                x_bounds={"qdot": bounds},
            )
        ],
        _sync_acados_state_bounds=lambda: None,
    )
    original = periodic_example._copy_state_bounds(nmpc)
    relaxed, expansion = periodic_example.build_relaxed_transfer_state_bounds(
        nmpc, padding=0.1
    )

    np.testing.assert_allclose(relaxed["qdot"][0][:, 0], [-0.1])
    np.testing.assert_allclose(relaxed["qdot"][1][:, 0], [0.1])
    assert relaxed["qdot"][0][0, 1] < -4.0
    assert relaxed["qdot"][0][0, 2] < -5.0
    assert expansion["qdot"] > 0.0

    periodic_example.apply_transfer_state_bound_fraction(
        nmpc, original, relaxed, fraction=0.5
    )
    np.testing.assert_allclose(bounds.min[:, 0], [-0.1])
    np.testing.assert_allclose(bounds.max[:, 0], [0.1])
    assert relaxed["qdot"][0][0, 1] < bounds.min[0, 1] < -1.0


def test_transfer_bound_homotopy_restores_physical_bounds(monkeypatch):
    class FakeSolver:
        def __init__(self):
            self.nlp_solver_max_iter = 100

        def set_convergence_tolerance(self, value):
            self.tolerance = value

        def set_maximum_iterations(self, value):
            self.nlp_solver_max_iter = value

    bounds = SimpleNamespace(
        min=np.array([[-0.1, -1.0, -2.1]]),
        max=np.array([[0.1, 1.0, -1.9]]),
    )
    original_min = bounds.min.copy()
    original_max = bounds.max.copy()
    nlp = SimpleNamespace(
        x_init={"qdot": SimpleNamespace(init=np.array([[0.0, -4.0, -5.0]]))},
        u_init={"u": SimpleNamespace(init=np.zeros((1, 2)))},
        x_bounds={"qdot": bounds},
    )
    nmpc = SimpleNamespace(
        nlp=[nlp],
        ocp_solver=None,
        _cocofest_fix_controls_to_warmup=True,
        _correct_init_guess_to_fit_bounds=lambda corrected_input: None,
        _sync_acados_state_bounds=lambda: None,
    )
    solutions = [
        SimpleNamespace(
            status=0, solver_time_to_optimize=0.1, real_time_to_optimize=0.2
        ),
        SimpleNamespace(
            status=0, solver_time_to_optimize=0.1, real_time_to_optimize=0.2
        ),
    ]
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda solution: {"residuals": np.zeros(4)},
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda periodic_nmpc, solution: None,
    )

    summary = periodic_example.run_acados_transfer_bound_homotopy(
        nmpc,
        FakeSolver(),
        fractions=(0.0, 1.0),
        padding=0.1,
        convergence_tolerance=1e-4,
        stage_iterations=10,
        echo=False,
        solve_stage=lambda: solutions.pop(0),
    )

    assert summary["completed"] is True
    assert [stage["accepted"] for stage in summary["stages"]] == [True, True]
    np.testing.assert_allclose(bounds.min, original_min)
    np.testing.assert_allclose(bounds.max, original_max)
    assert nmpc._cocofest_fix_controls_to_warmup is True


def test_transfer_sqp_restarts_from_nearly_feasible_iterate(monkeypatch):
    runtime_options = []
    reset_calls = []

    class FakeAcadosSolver:
        def options_set(self, key, value):
            runtime_options.append((key, value))

        def reset(self, reset_qp_solver_mem):
            reset_calls.append(reset_qp_solver_mem)

    class FakeSolver:
        nlp_solver_max_iter = 100

        def set_maximum_iterations(self, value):
            self.nlp_solver_max_iter = value

        def set_only_first_options_has_changed(self, value):
            self.options_changed = value

    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={"q": SimpleNamespace(init=np.zeros((1, 2)))},
                u_init={"u": SimpleNamespace(init=np.zeros((1, 1)))},
            )
        ],
        ocp_solver=SimpleNamespace(ocp_solver=FakeAcadosSolver()),
    )
    solutions = iter(
        [
            SimpleNamespace(
                status=4,
                residuals=np.array([0.7, 2.0, 0.0, 0.0]),
                solver_time_to_optimize=0.1,
                real_time_to_optimize=0.2,
            ),
            SimpleNamespace(
                status=0,
                residuals=np.array([1e-6, 1e-8, 0.0, 1e-7]),
                solver_time_to_optimize=0.3,
                real_time_to_optimize=0.4,
            ),
        ]
    )
    applied_statuses = []
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda solution: (
            {
                "residuals": solution.residuals,
                "res_stat_all": np.array([0.7, 0.6]),
                "res_eq_all": np.array([2.0, 3e-4]),
                "res_ineq_all": np.zeros(2),
                "res_comp_all": np.array([0.0, 1.4e-3]),
            }
            if solution.status == 4
            else {"residuals": solution.residuals}
        ),
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda _nmpc, solution: applied_statuses.append(solution.status),
    )

    summary = periodic_example.run_acados_transfer_sqp_restarts(
        nmpc,
        FakeSolver(),
        max_restarts=3,
        stage_iterations=1,
        feasibility_tolerance=1e-2,
        echo=False,
        solve_stage=lambda: next(solutions),
    )

    assert summary["completed"] is True
    assert [item["status"] for item in summary["attempts"]] == [4, 0]
    np.testing.assert_allclose(
        summary["attempts"][0]["reported_residuals"], [0.7, 2.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        summary["attempts"][0]["residuals"], [0.6, 3e-4, 0.0, 1.4e-3]
    )
    assert applied_statuses == [4, 0]
    assert reset_calls == [1]
    assert runtime_options == [
        ("nlp_solver_max_iter", 1),
        ("nlp_solver_max_iter", 1),
        ("nlp_solver_max_iter", 100),
    ]


def test_failed_acados_capsule_primal_is_unscaled_into_initial_guess():
    class Variables(dict):
        def __init__(self, values, shape):
            super().__init__(values)
            self.shape = shape

    class FakeCapsule:
        def get(self, stage, field):
            values = {
                (0, "x"): np.array([1.0, 2.0]),
                (1, "x"): np.array([3.0, 4.0]),
                (0, "u"): np.array([5.0]),
            }
            return values[(stage, field)]

    states = Variables(
        {"q": SimpleNamespace(index=[0]), "qdot": SimpleNamespace(index=[1])},
        shape=2,
    )
    controls = Variables({"u": SimpleNamespace(index=[0])}, shape=1)
    nlp = SimpleNamespace(
        states=states,
        controls=controls,
        parameters=SimpleNamespace(shape=0),
        x_init={
            "q": SimpleNamespace(init=np.zeros((1, 2))),
            "qdot": SimpleNamespace(init=np.zeros((1, 2))),
        },
        u_init={"u": SimpleNamespace(init=np.zeros((1, 1)))},
        x_scaling={
            "q": SimpleNamespace(scaling=np.array([[2.0]])),
            "qdot": SimpleNamespace(scaling=np.array([[4.0]])),
        },
        u_scaling={"u": SimpleNamespace(scaling=np.array([[3.0]]))},
    )
    nmpc = SimpleNamespace(
        nlp=[nlp], ocp_solver=SimpleNamespace(ocp_solver=FakeCapsule())
    )

    summary = periodic_example.apply_acados_capsule_primal_to_initial_guess(nmpc)

    assert summary["applied"] is True
    np.testing.assert_allclose(nlp.x_init["q"].init, [[2.0, 6.0]])
    np.testing.assert_allclose(nlp.x_init["qdot"].init, [[8.0, 16.0]])
    np.testing.assert_allclose(nlp.u_init["u"].init, [[15.0]])


def test_transferred_guess_is_projected_after_bounds_move():
    state_guess = SimpleNamespace(init=np.array([[2.0, -3.0]]))
    control_guess = SimpleNamespace(init=np.array([[4.0]]))
    calls = []

    def correct(corrected_input):
        calls.append(corrected_input)
        if corrected_input == "states":
            state_guess.init[:] = np.clip(state_guess.init, -1.0, 1.0)
        else:
            control_guess.init[:] = np.clip(control_guess.init, 0.0, 2.0)

    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={"state": state_guess}, u_init={"control": control_guess}
            )
        ],
        _correct_init_guess_to_fit_bounds=correct,
        _sync_acados_state_bounds=lambda: calls.append("sync"),
    )

    summary = periodic_example.project_transferred_initial_guess_to_bounds(nmpc)

    assert calls == ["states", "controls", "sync"]
    assert summary["state_max_change"] == 2.0
    assert summary["control_max_change"] == 2.0
    np.testing.assert_allclose(state_guess.init, [[1.0, -1.0]])
    np.testing.assert_allclose(control_guess.init, [[2.0]])


def test_initial_guess_audit_is_solver_independent_and_deterministic():
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={"q": SimpleNamespace(init=np.array([[1.0, 2.0]]))},
                u_init={"tau": SimpleNamespace(init=np.array([[3.0]]))},
            )
        ]
    )

    first = audit_initial_guess(nmpc)
    second = audit_initial_guess(nmpc)

    assert first["signature"] == second["signature"]
    assert first["finite"] is True
    assert first["state_shapes"] == {"q": (1, 2)}
    assert first["control_shapes"] == {"tau": (1, 1)}
    nmpc.nlp[0].x_init["q"].init[0, 1] += 1.0
    assert audit_initial_guess(nmpc)["signature"] != first["signature"]


def test_generic_initial_guess_copy_reports_incompatible_grids():
    source = {"q": SimpleNamespace(init=np.ones((1, 3)))}
    target = {"q": SimpleNamespace(init=np.zeros((1, 2)))}

    with np.testing.assert_raises_regex(ValueError, "shape"):
        copy_container_values(source, target, "init")


def test_shared_transfer_rollout_cli_is_available_to_ipopt():
    args = periodic_example.build_argument_parser().parse_args(
        [
            "--solver",
            "ipopt",
            "--transfer-full-dynamics-rollout",
            "--transfer-phase-one",
            "--transfer-rollout-substeps",
            "7",
        ]
    )
    comparison_args = comparison_example.build_cli().parse_args(
        [
            "--shared-transfer-full-dynamics-rollout",
            "--shared-transfer-phase-one",
            "--shared-initial-phase-one",
            "--shared-transfer-rollout-substeps",
            "7",
            "--shared-transfer-ding-force-compensation",
            "--shared-transfer-ding-force-compensation-substeps",
            "6",
            "--acados-transfer-ding-force-compensation",
            "--acados-proximal-control-weights",
            "1e6,1e5",
            "--acados-proximal-control-each-window",
            "--acados-terminal-wheel-q-homotopy-slacks",
            "0.2,0.1,0.02",
            "--acados-terminal-wheel-q-homotopy-each-window",
        ]
    )

    assert args.acados_transfer_full_dynamics_rollout is True
    assert args.acados_transfer_phase_one is True
    assert args.acados_transfer_rollout_substeps == 7
    assert comparison_args.shared_transfer_full_dynamics_rollout is True
    assert comparison_args.shared_transfer_phase_one is True
    assert comparison_args.shared_initial_phase_one is True
    assert comparison_args.shared_transfer_rollout_substeps == 7
    assert comparison_args.shared_transfer_ding_force_compensation is True
    assert comparison_args.shared_transfer_ding_force_compensation_substeps == 6
    assert comparison_args.acados_transfer_ding_force_compensation is True
    assert comparison_args.acados_proximal_control_weights == (1e6, 1e5)
    assert comparison_args.acados_proximal_control_each_window is True
    assert comparison_args.acados_terminal_wheel_q_homotopy_slacks == (
        0.2,
        0.1,
        0.02,
    )
    assert comparison_args.acados_terminal_wheel_q_homotopy_each_window is True
    assert (
        comparison_example.IPOPT_PROFILE_DEFAULTS["acados_like"]["model_formulation"]
        == "periodic_node"
    )
    assert (
        comparison_example.IPOPT_PROFILE_DEFAULTS["acados_like"][
            "disable_periodic_fes_warmup_projection"
        ]
        is False
    )


def test_periodic_collocation_ipopt_profile_is_available():
    args = comparison_example.build_cli().parse_args(
        ["--ipopt-profile", "periodic-collocation"]
    )
    defaults = comparison_example.IPOPT_PROFILE_DEFAULTS["periodic_collocation"]

    assert comparison_example._normalize_ipopt_profile(args.ipopt_profile) == (
        "periodic_collocation"
    )
    assert defaults["model_formulation"] == "periodic_node"
    assert defaults["torque_application"] == "constant"
    assert defaults["ode_solver"] == "collocation"
    assert defaults["use_sx"] is False


def test_fes_nmpc_reports_incomplete_export_as_solver_failure():
    nmpc = object.__new__(FesNmpcMsk)
    nmpc.n_cycles_simultaneous = 1

    def fail_while_assembling_window(*_args, **_kwargs):
        raise IndexError("index 31 is out of bounds for axis 1 with size 31")

    nmpc.solve = fail_while_assembling_window
    with np.testing.assert_raises_regex(RuntimeError, "exported window"):
        nmpc.solve_fes_nmpc(
            update_functions=None,
            solver=SimpleNamespace(),
            total_cycles=1,
            external_force=None,
            cycle_solutions=SimpleNamespace(),
        )


def test_shared_initial_guess_comparison_detects_exact_and_biased_seeds():
    shared = {
        "initial_guess_state_traces": {"q": np.array([[1.0, 2.0]])},
        "initial_guess_control_traces": {"u": np.array([[3.0]])},
        "initial_guess_audits": [{"signature": "same"}],
    }
    exact = comparison_example._shared_initial_guess_comparison(shared, shared)
    changed = {
        **shared,
        "initial_guess_control_traces": {"u": np.array([[4.0]])},
        "initial_guess_audits": [{"signature": "different"}],
    }
    biased = comparison_example._shared_initial_guess_comparison(shared, changed)

    assert exact["comparable"] is True
    assert exact["exact"] is True
    assert exact["max_abs_error"] == 0.0
    assert biased["comparable"] is True
    assert biased["exact"] is False
    assert biased["max_abs_error"] == 1.0


def test_failed_first_window_keeps_initial_guess_for_backend_comparison():
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                controls={"u": object()},
                u_bounds={
                    "u": SimpleNamespace(min=np.array([[0.0]]), max=np.array([[1.0]]))
                },
            )
        ]
    )
    states = {"q": np.array([[0.0, 0.0], [0.0, 0.0], [0.0, -1.0]])}
    controls = {"u": np.array([[0.5]])}

    summary = periodic_example.build_failed_solve_summary(
        nmpc,
        SimpleNamespace(n_windows=2),
        RuntimeError("no iterate"),
        states,
        controls,
    )

    assert summary["success"] is False
    assert summary["attempted_windows"] == 0
    assert summary["diagnostics"]["issues"] == ["no_solver_solution"]
    assert summary["initial_guess_state_traces"] is states
    assert summary["initial_guess_control_traces"] is controls
    assert summary["final_wheel_angle"] == -1.0


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
