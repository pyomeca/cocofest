"""
Compare IPOPT and ACADOS on the cycling pulse-width FES MHE using solver-specific but validated configurations.

IPOPT uses the historically robust collocation-based transcription.
ACADOS uses the solver-compatible periodic Ding surrogate with the lightweight RK4 setup.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cycling_pulse_width_mhe_acados_periodic import (
    ACADOS_STATUS_NAMES,
    build_argument_parser,
    solve_case,
)

EXAMPLE_DIR = Path(__file__).resolve().parent


def _namespace_from_cli(**overrides) -> argparse.Namespace:
    parser = build_argument_parser()
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _format_metric(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _format_control_metric(value) -> str:
    if value is None:
        return "None"
    return f"{float(value):.6g}"


def _status_label(status) -> str:
    if status is None:
        return "None"
    return ACADOS_STATUS_NAMES.get(status, str(status))


def _solver_status_label(solver_name: str, status) -> str:
    if solver_name == "ACADOS":
        return _status_label(status)
    return "None" if status is None else str(status)


def _format_array(values) -> str:
    if values is None:
        return "None"
    if isinstance(values, dict):
        return f"error={values.get('error')}"
    return np.array2string(
        np.asarray(values, dtype=float),
        precision=3,
        suppress_small=False,
    )


def _effective_status(result: dict):
    if result.get("status") is not None:
        return result["status"]
    for status in result.get("window_statuses") or []:
        if status != 0:
            return status
    return result.get("status")


def _resample_trace(trace: np.ndarray, target_len: int) -> np.ndarray:
    trace = np.asarray(trace, dtype=float).squeeze()
    if trace.size == target_len:
        return trace
    current_grid = np.linspace(0.0, 1.0, trace.size)
    target_grid = np.linspace(0.0, 1.0, target_len)
    return np.interp(target_grid, current_grid, trace)


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2 * np.pi) - np.pi


def _trace_comparison(ipopt_result: dict, acados_result: dict) -> dict:
    ipopt_trace = np.asarray(ipopt_result["wheel_angle_trace"], dtype=float).squeeze()
    acados_trace = np.asarray(acados_result["wheel_angle_trace"], dtype=float).squeeze()
    common_len = max(ipopt_trace.size, acados_trace.size)
    ipopt_common = _resample_trace(np.unwrap(ipopt_trace), common_len)
    acados_common = _resample_trace(np.unwrap(acados_trace), common_len)
    unwrapped_diff = acados_common - ipopt_common
    unwrapped_turn_offset = int(np.rint(np.median(unwrapped_diff / (2 * np.pi))))
    turn_aligned_unwrapped_diff = unwrapped_diff - unwrapped_turn_offset * 2 * np.pi
    phase_diff = _wrap_to_pi(unwrapped_diff)
    raw_final_error = float(acados_trace[-1] - ipopt_trace[-1])
    return {
        "common_len": common_len,
        "unwrapped_wheel_rmse": float(np.sqrt(np.mean(unwrapped_diff**2))),
        "unwrapped_wheel_max_abs_error": float(np.max(np.abs(unwrapped_diff))),
        "unwrapped_wheel_final_error": float(acados_common[-1] - ipopt_common[-1]),
        "turn_aligned_unwrapped_rmse": float(
            np.sqrt(np.mean(turn_aligned_unwrapped_diff**2))
        ),
        "turn_aligned_unwrapped_max_abs_error": float(
            np.max(np.abs(turn_aligned_unwrapped_diff))
        ),
        "turn_aligned_unwrapped_final_error": float(turn_aligned_unwrapped_diff[-1]),
        "unwrapped_turn_offset": unwrapped_turn_offset,
        "phase_rmse": float(np.sqrt(np.mean(phase_diff**2))),
        "phase_max_abs_error": float(np.max(np.abs(phase_diff))),
        "phase_final_error": float(phase_diff[-1]),
        "raw_final_error": raw_final_error,
        "raw_final_turn_offset": int(np.rint(raw_final_error / (2 * np.pi))),
    }


def _control_comparisons(ipopt_result: dict, acados_result: dict) -> list[dict]:
    ipopt_controls = ipopt_result.get("control_traces", {})
    acados_controls = acados_result.get("control_traces", {})
    common_keys = sorted(set(ipopt_controls).intersection(acados_controls))
    comparisons = []

    for key in common_keys:
        ipopt_values = np.asarray(ipopt_controls[key], dtype=float).reshape(-1)
        acados_values = np.asarray(acados_controls[key], dtype=float).reshape(-1)
        if ipopt_values.size == 0 or acados_values.size == 0:
            continue
        common_len = max(ipopt_values.size, acados_values.size)
        ipopt_common = _resample_trace(ipopt_values, common_len)
        acados_common = _resample_trace(acados_values, common_len)
        diff = acados_common - ipopt_common
        comparisons.append(
            {
                "key": key,
                "common_len": common_len,
                "rmse": float(np.sqrt(np.mean(diff**2))),
                "mae": float(np.mean(np.abs(diff))),
                "max_abs_error": float(np.max(np.abs(diff))),
                "final_error": float(diff[-1]),
                "ipopt_mean": float(np.mean(ipopt_common)),
                "acados_mean": float(np.mean(acados_common)),
                "ipopt_sum": float(np.sum(ipopt_common)),
                "acados_sum": float(np.sum(acados_common)),
                "ipopt_min": float(np.min(ipopt_common)),
                "ipopt_max": float(np.max(ipopt_common)),
                "acados_min": float(np.min(acados_common)),
                "acados_max": float(np.max(acados_common)),
            }
        )

    return comparisons


def _state_comparisons(ipopt_result: dict, acados_result: dict) -> list[dict]:
    ipopt_states = ipopt_result.get("state_traces", {})
    acados_states = acados_result.get("state_traces", {})
    return _state_trace_comparisons(ipopt_states, acados_states, "ipopt", "acados")


def _state_trace_comparisons(
    reference_states: dict,
    compared_states: dict,
    reference_prefix: str,
    compared_prefix: str,
) -> list[dict]:
    common_keys = sorted(set(reference_states).intersection(compared_states))
    comparisons = []

    for key in common_keys:
        reference_values = np.asarray(reference_states[key], dtype=float)
        compared_values = np.asarray(compared_states[key], dtype=float)
        if reference_values.ndim == 1:
            reference_values = reference_values[np.newaxis, :]
        if compared_values.ndim == 1:
            compared_values = compared_values[np.newaxis, :]
        if reference_values.size == 0 or compared_values.size == 0:
            continue

        for row in range(min(reference_values.shape[0], compared_values.shape[0])):
            common_len = max(reference_values.shape[1], compared_values.shape[1])
            reference_common = _resample_trace(reference_values[row, :], common_len)
            compared_common = _resample_trace(compared_values[row, :], common_len)
            diff = compared_common - reference_common
            comparisons.append(
                {
                    "key": (key if reference_values.shape[0] == 1 else f"{key}[{row}]"),
                    "common_len": common_len,
                    "rmse": float(np.sqrt(np.mean(diff**2))),
                    "mae": float(np.mean(np.abs(diff))),
                    "max_abs_error": float(np.max(np.abs(diff))),
                    "final_error": float(diff[-1]),
                    f"{reference_prefix}_mean": float(np.mean(reference_common)),
                    f"{compared_prefix}_mean": float(np.mean(compared_common)),
                    f"{reference_prefix}_range": (
                        float(np.min(reference_common)),
                        float(np.max(reference_common)),
                    ),
                    f"{compared_prefix}_range": (
                        float(np.min(compared_common)),
                        float(np.max(compared_common)),
                    ),
                }
            )

    return sorted(comparisons, key=lambda item: item["rmse"], reverse=True)


def _initial_guess_state_comparisons(acados_result: dict) -> list[dict]:
    initial_guess_states = acados_result.get("initial_guess_state_traces", {})
    solution_states = acados_result.get("state_traces", {})
    return _state_trace_comparisons(
        initial_guess_states, solution_states, "initial_guess", "solution"
    )


def _solver_config(
    solver_name: str,
    objective: str,
    objective_shape: str,
    cycles_per_window: int,
    stimulations_per_cycle: int,
    n_windows: int,
    resistive_torque: float,
    codegen_tag: str | None,
    ipopt_max_iter: int,
    ipopt_linear_solver: str,
    acados_max_iter: int,
    control_regularization_weight: float,
    control_regularization_target: float | None,
    control_regularization_target_source: str,
    wheel_qdot_regularization_weight: float,
    wheel_qdot_regularization_target: float,
    state_scaling: str,
    pulse_width_scaling: float,
    acados_tolerance: float | None,
    acados_qp_iter_max: int,
    acados_levenberg_marquardt: float,
    acados_regularize_method: str,
    acados_hessian_approx: str,
    acados_nlp_solver_type: str,
    acados_search_direction_mode: str,
    acados_globalization: str,
    acados_fixed_step_length: float,
    acados_nlp_qp_tol_strategy: str,
    acados_qpscaling_scale_objective: str,
    acados_qpscaling_scale_constraints: str,
    acados_ext_qp_res: bool,
    acados_project_qdot_from_q: bool,
    disable_periodic_fes_warmup_projection: bool,
    periodic_fes_warmup_projection_weight: float,
    periodic_fes_warmup_projection_mode: str,
    periodic_fes_warmup_projection_strategy: str,
    periodic_fes_warmup_projection_substeps: int,
    periodic_fes_warmup_projection_defect_weight: float,
    acados_diagnostics: bool,
    periodic_ipopt_refinement: bool,
    periodic_ipopt_refinement_iterations: int,
    periodic_ipopt_refinement_use_sx: bool,
    warmup_state_comparison_limit: int,
) -> argparse.Namespace:
    if solver_name == "ipopt":
        return _namespace_from_cli(
            solver="ipopt",
            single_shot=False,
            model_formulation="standard",
            torque_application="external_forces",
            cycles_per_window=cycles_per_window,
            stimulations_per_cycle=stimulations_per_cycle,
            objective=objective,
            objective_shape=objective_shape,
            constant_crank_torque=resistive_torque,
            max_ipopt_iterations=ipopt_max_iter,
            ipopt_linear_solver=ipopt_linear_solver,
            n_windows=n_windows,
            ode_solver="collocation",
            collocation_degree=3,
            collocation_method="radau",
            rk_steps=1,
            use_sx=False,
            enforce_start_constraints=True,
            disable_standard_ipopt_warmup=False,
            max_consecutive_failing=1,
            codegen_tag=codegen_tag,
            control_regularization_weight=control_regularization_weight,
            control_regularization_target=control_regularization_target,
            control_regularization_target_source=control_regularization_target_source,
            wheel_qdot_regularization_weight=wheel_qdot_regularization_weight,
            wheel_qdot_regularization_target=wheel_qdot_regularization_target,
            state_scaling=state_scaling,
            pulse_width_scaling=pulse_width_scaling,
            acados_tolerance=acados_tolerance,
            acados_qp_iter_max=acados_qp_iter_max,
            acados_levenberg_marquardt=acados_levenberg_marquardt,
            acados_regularize_method=acados_regularize_method,
            acados_hessian_approx=acados_hessian_approx,
            acados_nlp_solver_type=acados_nlp_solver_type,
            acados_search_direction_mode=acados_search_direction_mode,
            acados_globalization=acados_globalization,
            acados_fixed_step_length=acados_fixed_step_length,
            acados_nlp_qp_tol_strategy=acados_nlp_qp_tol_strategy,
            acados_qpscaling_scale_objective=acados_qpscaling_scale_objective,
            acados_qpscaling_scale_constraints=acados_qpscaling_scale_constraints,
            acados_ext_qp_res=False,
            acados_project_qdot_from_q=False,
            disable_periodic_fes_warmup_projection=True,
            periodic_fes_warmup_projection_weight=periodic_fes_warmup_projection_weight,
            periodic_fes_warmup_projection_mode=periodic_fes_warmup_projection_mode,
            periodic_fes_warmup_projection_strategy=(
                periodic_fes_warmup_projection_strategy
            ),
            periodic_fes_warmup_projection_substeps=(
                periodic_fes_warmup_projection_substeps
            ),
            periodic_fes_warmup_projection_defect_weight=(
                periodic_fes_warmup_projection_defect_weight
            ),
            acados_diagnostics=False,
            periodic_ipopt_refinement=False,
            periodic_ipopt_refinement_iterations=periodic_ipopt_refinement_iterations,
            periodic_ipopt_refinement_use_sx=False,
            warmup_state_comparison_limit=warmup_state_comparison_limit,
        )

    if solver_name == "acados":
        return _namespace_from_cli(
            solver="acados",
            single_shot=False,
            model_formulation="periodic",
            torque_application="constant",
            cycles_per_window=cycles_per_window,
            stimulations_per_cycle=stimulations_per_cycle,
            objective=objective,
            objective_shape=objective_shape,
            constant_crank_torque=resistive_torque,
            max_acados_iterations=acados_max_iter,
            ipopt_linear_solver=ipopt_linear_solver,
            n_windows=n_windows,
            ode_solver="rk4",
            rk_steps=5,
            collocation_degree=3,
            collocation_method="radau",
            use_sx=True,
            enforce_start_constraints=False,
            disable_standard_ipopt_warmup=False,
            max_consecutive_failing=max(n_windows, 10),
            codegen_tag=codegen_tag,
            control_regularization_weight=control_regularization_weight,
            control_regularization_target=control_regularization_target,
            control_regularization_target_source=control_regularization_target_source,
            wheel_qdot_regularization_weight=wheel_qdot_regularization_weight,
            wheel_qdot_regularization_target=wheel_qdot_regularization_target,
            state_scaling=state_scaling,
            pulse_width_scaling=pulse_width_scaling,
            acados_tolerance=acados_tolerance,
            acados_qp_iter_max=acados_qp_iter_max,
            acados_levenberg_marquardt=acados_levenberg_marquardt,
            acados_regularize_method=acados_regularize_method,
            acados_hessian_approx=acados_hessian_approx,
            acados_nlp_solver_type=acados_nlp_solver_type,
            acados_search_direction_mode=acados_search_direction_mode,
            acados_globalization=acados_globalization,
            acados_fixed_step_length=acados_fixed_step_length,
            acados_nlp_qp_tol_strategy=acados_nlp_qp_tol_strategy,
            acados_qpscaling_scale_objective=acados_qpscaling_scale_objective,
            acados_qpscaling_scale_constraints=acados_qpscaling_scale_constraints,
            acados_ext_qp_res=acados_ext_qp_res,
            acados_project_qdot_from_q=acados_project_qdot_from_q,
            disable_periodic_fes_warmup_projection=(
                disable_periodic_fes_warmup_projection
            ),
            periodic_fes_warmup_projection_weight=periodic_fes_warmup_projection_weight,
            periodic_fes_warmup_projection_mode=periodic_fes_warmup_projection_mode,
            periodic_fes_warmup_projection_strategy=(
                periodic_fes_warmup_projection_strategy
            ),
            periodic_fes_warmup_projection_substeps=(
                periodic_fes_warmup_projection_substeps
            ),
            periodic_fes_warmup_projection_defect_weight=(
                periodic_fes_warmup_projection_defect_weight
            ),
            acados_diagnostics=acados_diagnostics,
            periodic_ipopt_refinement=periodic_ipopt_refinement,
            periodic_ipopt_refinement_iterations=periodic_ipopt_refinement_iterations,
            periodic_ipopt_refinement_use_sx=periodic_ipopt_refinement_use_sx,
            warmup_state_comparison_limit=warmup_state_comparison_limit,
        )

    raise ValueError(f"Unsupported solver_name '{solver_name}'")


def print_comparison(
    ipopt_result: dict,
    acados_result: dict,
    print_traces: bool = False,
    state_comparison_limit: int = 12,
) -> None:
    print(
        "solver | success | solver_success | physical_success | status | status_label | objective | solver_time_s | wall_time_s | final_wheel_angle | "
        "requested_cycles | attempted_windows | successful_windows | exported_cycles | covered_cycles"
    )
    for label, result in (("IPOPT", ipopt_result), ("ACADOS", acados_result)):
        status = _effective_status(result)
        print(
            f"{label} | "
            f"{_format_metric(result.get('success'))} | "
            f"{_format_metric(result.get('solver_success'))} | "
            f"{_format_metric(result.get('physical_success'))} | "
            f"{_format_metric(status)} | "
            f"{_solver_status_label(label, status)} | "
            f"{_format_metric(result['objective'])} | "
            f"{_format_metric(result['solver_time_s'])} | "
            f"{_format_metric(result['wall_time_s'])} | "
            f"{_format_metric(result['final_wheel_angle'])} | "
            f"{_format_metric(result.get('requested_windows'))} | "
            f"{_format_metric(result.get('attempted_windows'))} | "
            f"{_format_metric(result.get('successful_windows'))} | "
            f"{_format_metric(result.get('exported_cycles'))} | "
            f"{_format_metric(result.get('covered_cycles'))}"
        )
        if print_traces:
            print(
                f"{label} wheel angle trace: "
                f"{np.array2string(result['wheel_angle_trace'], precision=4, suppress_small=False)}"
            )
        diagnostics = result.get("diagnostics", {})
        print(
            f"{label} diagnostics: physical={diagnostics.get('is_physical')} "
            f"issues={diagnostics.get('issues')} "
            f"max_abs_angle={_format_metric(diagnostics.get('max_abs_angle'))} "
            f"max_step={_format_metric(diagnostics.get('max_step'))} "
            f"window_statuses={result.get('window_statuses')}"
        )
        if label == "ACADOS" and result.get("window_statuses"):
            labels = [_status_label(status) for status in result["window_statuses"]]
            print(f"{label} window_status_labels: {labels}")

        acados_diagnostics = result.get("acados_diagnostics") or []
        if acados_diagnostics:
            for idx, item in enumerate(acados_diagnostics):
                print(
                    f"{label} acados window[{idx}] | "
                    f"status={item.get('status')} ({item.get('status_label')}) | "
                    f"residuals={_format_array(item.get('residuals'))} | "
                    f"sqp_iter={_format_array(item.get('sqp_iter'))} | "
                    f"qp_iter={_format_array(item.get('qp_iter'))} | "
                    f"qp_stat={_format_array(item.get('qp_stat'))}"
                )

    if not (ipopt_result.get("success") and acados_result.get("success")):
        print(
            "wheel trace comparison warning: at least one solver did not cover all requested cycles successfully."
        )
    trace_metrics = _trace_comparison(ipopt_result, acados_result)
    print(
        "wheel trace comparison | "
        f"common_len={trace_metrics['common_len']} | "
        f"unwrapped_rmse={trace_metrics['unwrapped_wheel_rmse']:.6f} | "
        f"unwrapped_max_abs_error={trace_metrics['unwrapped_wheel_max_abs_error']:.6f} | "
        f"unwrapped_final_error={trace_metrics['unwrapped_wheel_final_error']:.6f} | "
        f"turn_aligned_unwrapped_rmse={trace_metrics['turn_aligned_unwrapped_rmse']:.6f} | "
        f"turn_aligned_unwrapped_max_abs_error={trace_metrics['turn_aligned_unwrapped_max_abs_error']:.6f} | "
        f"turn_aligned_unwrapped_final_error={trace_metrics['turn_aligned_unwrapped_final_error']:.6f} | "
        f"unwrapped_turn_offset={trace_metrics['unwrapped_turn_offset']} | "
        f"phase_rmse={trace_metrics['phase_rmse']:.6f} | "
        f"phase_max_abs_error={trace_metrics['phase_max_abs_error']:.6f} | "
        f"phase_final_error={trace_metrics['phase_final_error']:.6f}"
    )
    print(
        "raw wheel final-angle representation | "
        f"raw_final_error={trace_metrics['raw_final_error']:.6f} | "
        f"raw_final_turn_offset={trace_metrics['raw_final_turn_offset']}"
    )

    control_metrics = _control_comparisons(ipopt_result, acados_result)
    if control_metrics:
        print(
            "control comparison | key | common_len | rmse | mae | max_abs_error | final_error | "
            "ipopt_mean | acados_mean | ipopt_sum | acados_sum | ipopt_range | acados_range"
        )
        for metric in control_metrics:
            print(
                "control comparison | "
                f"{metric['key']} | "
                f"{metric['common_len']} | "
                f"{_format_control_metric(metric['rmse'])} | "
                f"{_format_control_metric(metric['mae'])} | "
                f"{_format_control_metric(metric['max_abs_error'])} | "
                f"{_format_control_metric(metric['final_error'])} | "
                f"{_format_control_metric(metric['ipopt_mean'])} | "
                f"{_format_control_metric(metric['acados_mean'])} | "
                f"{_format_control_metric(metric['ipopt_sum'])} | "
                f"{_format_control_metric(metric['acados_sum'])} | "
                f"[{_format_control_metric(metric['ipopt_min'])}, {_format_control_metric(metric['ipopt_max'])}] | "
                f"[{_format_control_metric(metric['acados_min'])}, {_format_control_metric(metric['acados_max'])}]"
            )
    else:
        print("control comparison warning: no common control keys were found.")

    state_metrics = _state_comparisons(ipopt_result, acados_result)
    if state_metrics and state_comparison_limit:
        print(
            "state comparison | key | common_len | rmse | mae | max_abs_error | final_error | "
            "ipopt_mean | acados_mean | ipopt_range | acados_range"
        )
        for metric in state_metrics[:state_comparison_limit]:
            ipopt_min, ipopt_max = metric["ipopt_range"]
            acados_min, acados_max = metric["acados_range"]
            print(
                "state comparison | "
                f"{metric['key']} | "
                f"{metric['common_len']} | "
                f"{_format_control_metric(metric['rmse'])} | "
                f"{_format_control_metric(metric['mae'])} | "
                f"{_format_control_metric(metric['max_abs_error'])} | "
                f"{_format_control_metric(metric['final_error'])} | "
                f"{_format_control_metric(metric['ipopt_mean'])} | "
                f"{_format_control_metric(metric['acados_mean'])} | "
                f"[{_format_control_metric(ipopt_min)}, {_format_control_metric(ipopt_max)}] | "
                f"[{_format_control_metric(acados_min)}, {_format_control_metric(acados_max)}]"
            )
    elif not state_metrics:
        print("state comparison warning: no common state keys were found.")

    initial_guess_state_metrics = _initial_guess_state_comparisons(acados_result)
    if initial_guess_state_metrics and state_comparison_limit:
        print(
            "acados initial guess vs solution | key | common_len | rmse | mae | max_abs_error | "
            "final_error | initial_guess_mean | solution_mean | initial_guess_range | solution_range"
        )
        for metric in initial_guess_state_metrics[:state_comparison_limit]:
            initial_min, initial_max = metric["initial_guess_range"]
            solution_min, solution_max = metric["solution_range"]
            print(
                "acados initial guess vs solution | "
                f"{metric['key']} | "
                f"{metric['common_len']} | "
                f"{_format_control_metric(metric['rmse'])} | "
                f"{_format_control_metric(metric['mae'])} | "
                f"{_format_control_metric(metric['max_abs_error'])} | "
                f"{_format_control_metric(metric['final_error'])} | "
                f"{_format_control_metric(metric['initial_guess_mean'])} | "
                f"{_format_control_metric(metric['solution_mean'])} | "
                f"[{_format_control_metric(initial_min)}, {_format_control_metric(initial_max)}] | "
                f"[{_format_control_metric(solution_min)}, {_format_control_metric(solution_max)}]"
            )
    elif acados_result.get("initial_guess_state_traces") is not None:
        print(
            "acados initial guess vs solution warning: no common state keys were found."
        )

    ipopt_solver_time = ipopt_result["solver_time_s"]
    acados_solver_time = acados_result["solver_time_s"]
    ipopt_wall_time = ipopt_result["wall_time_s"]
    acados_wall_time = acados_result["wall_time_s"]
    if not (ipopt_result.get("success") and acados_result.get("success")):
        print(
            "timing ratio warning: at least one solver failed; ratios compare elapsed diagnostic attempts, not successful solves."
        )
    if ipopt_solver_time and acados_solver_time:
        print(
            f"solver-time ratio IPOPT/ACADOS: {ipopt_solver_time / acados_solver_time:.3f}x "
            f"(ACADOS/IPOPT: {acados_solver_time / ipopt_solver_time:.3f}x)"
        )
    if ipopt_wall_time and acados_wall_time:
        print(
            f"wall-time ratio IPOPT/ACADOS: {ipopt_wall_time / acados_wall_time:.3f}x "
            f"(ACADOS/IPOPT: {acados_wall_time / ipopt_wall_time:.3f}x)"
        )


def main(
    objective: str = "force",
    objective_shape: str = "quadratic",
    cycles_per_window: int = 2,
    stimulations_per_cycle: int = 30,
    n_windows: int = 2,
    resistive_torque: float = -0.2,
    acados_dir: str | None = None,
    codegen_tag: str | None = None,
    ipopt_max_iter: int = 2000,
    ipopt_linear_solver: str = "ma57",
    acados_max_iter: int = 100,
    control_regularization_weight: float = 0.0,
    acados_control_regularization_weight: float | None = None,
    control_regularization_target: float | None = None,
    control_regularization_target_source: str = "constant",
    acados_control_regularization_target_source: str | None = None,
    wheel_qdot_regularization_weight: float = 0.0,
    acados_wheel_qdot_regularization_weight: float | None = None,
    wheel_qdot_regularization_target: float = -float(2 * np.pi),
    state_scaling: str = "none",
    acados_state_scaling: str | None = None,
    pulse_width_scaling: float = 1 / 400,
    acados_pulse_width_scaling: float | None = None,
    acados_tolerance: float | None = None,
    acados_qp_iter_max: int = 50,
    acados_levenberg_marquardt: float = 0.0,
    acados_regularize_method: str = "GERSHGORIN_LEVENBERG_MARQUARDT",
    acados_hessian_approx: str = "GAUSS_NEWTON",
    acados_nlp_solver_type: str = "SQP",
    acados_search_direction_mode: str = "NOMINAL_QP",
    acados_globalization: str = "MERIT_BACKTRACKING",
    acados_fixed_step_length: float = 1.0,
    acados_nlp_qp_tol_strategy: str = "ADAPTIVE_QPSCALING",
    acados_qpscaling_scale_objective: str = "OBJECTIVE_GERSHGORIN",
    acados_qpscaling_scale_constraints: str = "INF_NORM",
    acados_ext_qp_res: bool = False,
    acados_project_qdot_from_q: bool = False,
    disable_periodic_fes_warmup_projection: bool = False,
    periodic_fes_warmup_projection_weight: float = 1.0,
    periodic_fes_warmup_projection_mode: str = "all",
    periodic_fes_warmup_projection_strategy: str = "sequential",
    periodic_fes_warmup_projection_substeps: int = 10,
    periodic_fes_warmup_projection_defect_weight: float = 100.0,
    acados_diagnostics: bool = False,
    periodic_ipopt_refinement: bool = False,
    periodic_ipopt_refinement_iterations: int = 300,
    periodic_ipopt_refinement_use_sx: bool = False,
    warmup_state_comparison_limit: int = 12,
    state_comparison_limit: int = 12,
    print_traces: bool = False,
):
    os.chdir(EXAMPLE_DIR)
    if acados_dir:
        os.environ["ACADOS_SOURCE_DIR"] = str(Path(acados_dir).resolve())

    ipopt_args = _solver_config(
        "ipopt",
        objective=objective,
        objective_shape=objective_shape,
        cycles_per_window=cycles_per_window,
        stimulations_per_cycle=stimulations_per_cycle,
        n_windows=n_windows,
        resistive_torque=resistive_torque,
        codegen_tag=codegen_tag,
        ipopt_max_iter=ipopt_max_iter,
        ipopt_linear_solver=ipopt_linear_solver,
        acados_max_iter=acados_max_iter,
        control_regularization_weight=control_regularization_weight,
        control_regularization_target=control_regularization_target,
        control_regularization_target_source=control_regularization_target_source,
        wheel_qdot_regularization_weight=wheel_qdot_regularization_weight,
        wheel_qdot_regularization_target=wheel_qdot_regularization_target,
        state_scaling=state_scaling,
        pulse_width_scaling=pulse_width_scaling,
        acados_tolerance=acados_tolerance,
        acados_qp_iter_max=acados_qp_iter_max,
        acados_levenberg_marquardt=acados_levenberg_marquardt,
        acados_regularize_method=acados_regularize_method,
        acados_hessian_approx=acados_hessian_approx,
        acados_nlp_solver_type=acados_nlp_solver_type,
        acados_search_direction_mode=acados_search_direction_mode,
        acados_globalization=acados_globalization,
        acados_fixed_step_length=acados_fixed_step_length,
        acados_nlp_qp_tol_strategy=acados_nlp_qp_tol_strategy,
        acados_qpscaling_scale_objective=acados_qpscaling_scale_objective,
        acados_qpscaling_scale_constraints=acados_qpscaling_scale_constraints,
        acados_ext_qp_res=False,
        acados_project_qdot_from_q=False,
        disable_periodic_fes_warmup_projection=True,
        periodic_fes_warmup_projection_weight=periodic_fes_warmup_projection_weight,
        periodic_fes_warmup_projection_mode=periodic_fes_warmup_projection_mode,
        periodic_fes_warmup_projection_strategy=periodic_fes_warmup_projection_strategy,
        periodic_fes_warmup_projection_substeps=periodic_fes_warmup_projection_substeps,
        periodic_fes_warmup_projection_defect_weight=(
            periodic_fes_warmup_projection_defect_weight
        ),
        acados_diagnostics=acados_diagnostics,
        periodic_ipopt_refinement=False,
        periodic_ipopt_refinement_iterations=periodic_ipopt_refinement_iterations,
        periodic_ipopt_refinement_use_sx=False,
        warmup_state_comparison_limit=warmup_state_comparison_limit,
    )
    acados_args = _solver_config(
        "acados",
        objective=objective,
        objective_shape=objective_shape,
        cycles_per_window=cycles_per_window,
        stimulations_per_cycle=stimulations_per_cycle,
        n_windows=n_windows,
        resistive_torque=resistive_torque,
        codegen_tag=codegen_tag,
        ipopt_max_iter=ipopt_max_iter,
        ipopt_linear_solver=ipopt_linear_solver,
        acados_max_iter=acados_max_iter,
        control_regularization_weight=(
            acados_control_regularization_weight
            if acados_control_regularization_weight is not None
            else control_regularization_weight
        ),
        control_regularization_target=control_regularization_target,
        control_regularization_target_source=(
            acados_control_regularization_target_source
            if acados_control_regularization_target_source is not None
            else control_regularization_target_source
        ),
        wheel_qdot_regularization_weight=(
            acados_wheel_qdot_regularization_weight
            if acados_wheel_qdot_regularization_weight is not None
            else wheel_qdot_regularization_weight
        ),
        wheel_qdot_regularization_target=wheel_qdot_regularization_target,
        state_scaling=(
            acados_state_scaling if acados_state_scaling is not None else state_scaling
        ),
        pulse_width_scaling=(
            acados_pulse_width_scaling
            if acados_pulse_width_scaling is not None
            else pulse_width_scaling
        ),
        acados_tolerance=acados_tolerance,
        acados_qp_iter_max=acados_qp_iter_max,
        acados_levenberg_marquardt=acados_levenberg_marquardt,
        acados_regularize_method=acados_regularize_method,
        acados_hessian_approx=acados_hessian_approx,
        acados_nlp_solver_type=acados_nlp_solver_type,
        acados_search_direction_mode=acados_search_direction_mode,
        acados_globalization=acados_globalization,
        acados_fixed_step_length=acados_fixed_step_length,
        acados_nlp_qp_tol_strategy=acados_nlp_qp_tol_strategy,
        acados_qpscaling_scale_objective=acados_qpscaling_scale_objective,
        acados_qpscaling_scale_constraints=acados_qpscaling_scale_constraints,
        acados_ext_qp_res=acados_ext_qp_res,
        acados_project_qdot_from_q=acados_project_qdot_from_q,
        disable_periodic_fes_warmup_projection=disable_periodic_fes_warmup_projection,
        periodic_fes_warmup_projection_weight=periodic_fes_warmup_projection_weight,
        periodic_fes_warmup_projection_mode=periodic_fes_warmup_projection_mode,
        periodic_fes_warmup_projection_strategy=periodic_fes_warmup_projection_strategy,
        periodic_fes_warmup_projection_substeps=periodic_fes_warmup_projection_substeps,
        periodic_fes_warmup_projection_defect_weight=(
            periodic_fes_warmup_projection_defect_weight
        ),
        acados_diagnostics=acados_diagnostics,
        periodic_ipopt_refinement=periodic_ipopt_refinement,
        periodic_ipopt_refinement_iterations=periodic_ipopt_refinement_iterations,
        periodic_ipopt_refinement_use_sx=periodic_ipopt_refinement_use_sx,
        warmup_state_comparison_limit=warmup_state_comparison_limit,
    )

    print("Running IPOPT reference configuration...")
    ipopt_result = solve_case(ipopt_args, echo=True)
    print()
    print("Running ACADOS-compatible configuration...")
    acados_result = solve_case(acados_args, echo=True)
    print()
    print_comparison(
        ipopt_result,
        acados_result,
        print_traces=print_traces,
        state_comparison_limit=state_comparison_limit,
    )
    return {"ipopt": ipopt_result, "acados": acados_result}


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", default="force")
    parser.add_argument(
        "--objective-shape", default="quadratic", choices=("quadratic", "linear")
    )
    parser.add_argument("--cycles-per-window", type=int, default=2)
    parser.add_argument("--stimulations-per-cycle", type=int, default=30)
    parser.add_argument("--n-windows", type=int, default=2)
    parser.add_argument("--resistive-torque", type=float, default=-0.2)
    parser.add_argument("--acados-dir", default=os.environ.get("ACADOS_SOURCE_DIR"))
    parser.add_argument("--codegen-tag", default="fes_compare")
    parser.add_argument("--ipopt-max-iter", type=int, default=2000)
    parser.add_argument("--ipopt-linear-solver", default="ma57")
    parser.add_argument("--acados-max-iter", type=int, default=100)
    parser.add_argument("--control-regularization-weight", type=float, default=0.0)
    parser.add_argument(
        "--acados-control-regularization-weight", type=float, default=None
    )
    parser.add_argument("--control-regularization-target", type=float, default=None)
    parser.add_argument(
        "--control-regularization-target-source",
        choices=("constant", "warmup"),
        default="constant",
        help="Use a constant pulse-width target or the IPOPT warmup control trajectory.",
    )
    parser.add_argument(
        "--acados-control-regularization-target-source",
        choices=("constant", "warmup"),
        default=None,
        help="Override --control-regularization-target-source for ACADOS only.",
    )
    parser.add_argument("--wheel-qdot-regularization-weight", type=float, default=0.0)
    parser.add_argument(
        "--acados-wheel-qdot-regularization-weight", type=float, default=None
    )
    parser.add_argument(
        "--wheel-qdot-regularization-target", type=float, default=-float(2 * np.pi)
    )
    parser.add_argument(
        "--state-scaling", choices=("none", "fes", "full"), default="none"
    )
    parser.add_argument(
        "--acados-state-scaling", choices=("none", "fes", "full"), default=None
    )
    parser.add_argument("--pulse-width-scaling", type=float, default=1 / 400)
    parser.add_argument("--acados-pulse-width-scaling", type=float, default=None)
    parser.add_argument("--acados-tolerance", type=float, default=None)
    parser.add_argument("--acados-qp-iter-max", type=int, default=50)
    parser.add_argument("--acados-levenberg-marquardt", type=float, default=0.0)
    parser.add_argument(
        "--acados-regularize-method",
        choices=(
            "NO_REGULARIZE",
            "MIRROR",
            "PROJECT",
            "PROJECT_REDUC_HESS",
            "CONVEXIFY",
            "GERSHGORIN_LEVENBERG_MARQUARDT",
        ),
        default="GERSHGORIN_LEVENBERG_MARQUARDT",
    )
    parser.add_argument(
        "--acados-hessian-approx",
        choices=("GAUSS_NEWTON", "EXACT"),
        default="GAUSS_NEWTON",
    )
    parser.add_argument(
        "--acados-nlp-solver-type",
        choices=("SQP", "SQP_WITH_FEASIBLE_QP"),
        default="SQP",
    )
    parser.add_argument(
        "--acados-search-direction-mode",
        choices=("NOMINAL_QP", "BYRD_OMOJOKUN", "FEASIBILITY_QP"),
        default="NOMINAL_QP",
    )
    parser.add_argument(
        "--acados-globalization",
        choices=("FIXED_STEP", "MERIT_BACKTRACKING", "FUNNEL_L1PEN_LINESEARCH"),
        default="MERIT_BACKTRACKING",
    )
    parser.add_argument("--acados-fixed-step-length", type=float, default=1.0)
    parser.add_argument(
        "--acados-nlp-qp-tol-strategy",
        choices=("FIXED_QP_TOL", "ADAPTIVE_CURRENT_RES_JOINT", "ADAPTIVE_QPSCALING"),
        default="ADAPTIVE_QPSCALING",
    )
    parser.add_argument(
        "--acados-qpscaling-scale-objective",
        choices=("NO_OBJECTIVE_SCALING", "OBJECTIVE_GERSHGORIN"),
        default="OBJECTIVE_GERSHGORIN",
    )
    parser.add_argument(
        "--acados-qpscaling-scale-constraints",
        choices=("NO_CONSTRAINT_SCALING", "INF_NORM"),
        default="INF_NORM",
    )
    parser.add_argument("--acados-ext-qp-res", action="store_true")
    parser.add_argument("--acados-project-qdot-from-q", action="store_true")
    parser.add_argument("--acados-diagnostics", action="store_true")
    parser.add_argument("--disable-periodic-fes-warmup-projection", action="store_true")
    parser.add_argument(
        "--periodic-fes-warmup-projection-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--periodic-fes-warmup-projection-mode",
        choices=("calcium", "all", "all_except_force"),
        default="all",
    )
    parser.add_argument(
        "--periodic-fes-warmup-projection-strategy",
        choices=("rollout", "sequential", "least_squares"),
        default="sequential",
    )
    parser.add_argument(
        "--periodic-fes-warmup-projection-substeps", type=int, default=10
    )
    parser.add_argument(
        "--periodic-fes-warmup-projection-defect-weight", type=float, default=100.0
    )
    parser.add_argument(
        "--periodic-ipopt-refinement",
        action="store_true",
        help="Run the optional periodic IPOPT refinement before the ACADOS comparison solve.",
    )
    parser.add_argument(
        "--periodic-ipopt-refinement-iterations",
        type=int,
        default=300,
        help="Maximum IPOPT iterations for the periodic ACADOS warmstart refinement.",
    )
    parser.add_argument(
        "--periodic-ipopt-refinement-use-sx",
        action="store_true",
        help=(
            "Build the auxiliary periodic IPOPT refinement with SX graphs. "
            "By default it uses MX to reduce memory pressure."
        ),
    )
    parser.add_argument("--state-comparison-limit", type=int, default=12)
    parser.add_argument("--warmup-state-comparison-limit", type=int, default=12)
    parser.add_argument("--print-traces", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_cli().parse_args()
    main(
        objective=args.objective,
        objective_shape=args.objective_shape,
        cycles_per_window=args.cycles_per_window,
        stimulations_per_cycle=args.stimulations_per_cycle,
        n_windows=args.n_windows,
        resistive_torque=args.resistive_torque,
        acados_dir=args.acados_dir,
        codegen_tag=args.codegen_tag,
        ipopt_max_iter=args.ipopt_max_iter,
        ipopt_linear_solver=args.ipopt_linear_solver,
        acados_max_iter=args.acados_max_iter,
        control_regularization_weight=args.control_regularization_weight,
        acados_control_regularization_weight=args.acados_control_regularization_weight,
        control_regularization_target=args.control_regularization_target,
        control_regularization_target_source=args.control_regularization_target_source,
        acados_control_regularization_target_source=args.acados_control_regularization_target_source,
        wheel_qdot_regularization_weight=args.wheel_qdot_regularization_weight,
        acados_wheel_qdot_regularization_weight=args.acados_wheel_qdot_regularization_weight,
        wheel_qdot_regularization_target=args.wheel_qdot_regularization_target,
        state_scaling=args.state_scaling,
        acados_state_scaling=args.acados_state_scaling,
        pulse_width_scaling=args.pulse_width_scaling,
        acados_pulse_width_scaling=args.acados_pulse_width_scaling,
        acados_tolerance=args.acados_tolerance,
        acados_qp_iter_max=args.acados_qp_iter_max,
        acados_levenberg_marquardt=args.acados_levenberg_marquardt,
        acados_regularize_method=args.acados_regularize_method,
        acados_hessian_approx=args.acados_hessian_approx,
        acados_nlp_solver_type=args.acados_nlp_solver_type,
        acados_search_direction_mode=args.acados_search_direction_mode,
        acados_globalization=args.acados_globalization,
        acados_fixed_step_length=args.acados_fixed_step_length,
        acados_nlp_qp_tol_strategy=args.acados_nlp_qp_tol_strategy,
        acados_qpscaling_scale_objective=args.acados_qpscaling_scale_objective,
        acados_qpscaling_scale_constraints=args.acados_qpscaling_scale_constraints,
        acados_ext_qp_res=args.acados_ext_qp_res,
        acados_project_qdot_from_q=args.acados_project_qdot_from_q,
        disable_periodic_fes_warmup_projection=(
            args.disable_periodic_fes_warmup_projection
        ),
        periodic_fes_warmup_projection_weight=(
            args.periodic_fes_warmup_projection_weight
        ),
        periodic_fes_warmup_projection_mode=args.periodic_fes_warmup_projection_mode,
        periodic_fes_warmup_projection_strategy=(
            args.periodic_fes_warmup_projection_strategy
        ),
        periodic_fes_warmup_projection_substeps=(
            args.periodic_fes_warmup_projection_substeps
        ),
        periodic_fes_warmup_projection_defect_weight=(
            args.periodic_fes_warmup_projection_defect_weight
        ),
        acados_diagnostics=args.acados_diagnostics,
        periodic_ipopt_refinement=args.periodic_ipopt_refinement,
        periodic_ipopt_refinement_iterations=args.periodic_ipopt_refinement_iterations,
        periodic_ipopt_refinement_use_sx=args.periodic_ipopt_refinement_use_sx,
        warmup_state_comparison_limit=args.warmup_state_comparison_limit,
        state_comparison_limit=args.state_comparison_limit,
        print_traces=args.print_traces,
    )
