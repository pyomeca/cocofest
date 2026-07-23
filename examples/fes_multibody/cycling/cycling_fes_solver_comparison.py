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
from time import perf_counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .cycling_pulse_width_mhe_acados_periodic import (
        ACADOS_STATUS_NAMES,
        build_argument_parser,
        parse_proximal_control_weights,
        parse_terminal_wheel_q_slacks,
        solve_case,
    )
except ImportError:
    from cycling_pulse_width_mhe_acados_periodic import (
        ACADOS_STATUS_NAMES,
        build_argument_parser,
        parse_proximal_control_weights,
        parse_terminal_wheel_q_slacks,
        solve_case,
    )

EXAMPLE_DIR = Path(__file__).resolve().parent

IPOPT_PROFILE_DEFAULTS = {
    "historical": {
        "model_formulation": "standard",
        "torque_application": "external_forces",
        "ode_solver": "collocation",
        "rk_steps": 1,
        "collocation_degree": 3,
        "collocation_method": "radau",
        "use_sx": False,
        "enforce_start_constraints": True,
        "disable_standard_ipopt_warmup": False,
        "disable_periodic_fes_warmup_projection": True,
        "fatigue_warmstart_mode": "continuous",
    },
    "acados_like": {
        "model_formulation": "periodic_node",
        "torque_application": "constant",
        "ode_solver": "rk4",
        "rk_steps": 5,
        "collocation_degree": 3,
        "collocation_method": "radau",
        "use_sx": True,
        "enforce_start_constraints": False,
        "disable_standard_ipopt_warmup": False,
        "disable_periodic_fes_warmup_projection": False,
        "fatigue_warmstart_mode": None,
    },
    "periodic_collocation": {
        "model_formulation": "periodic_node",
        "torque_application": "constant",
        "ode_solver": "collocation",
        "rk_steps": 1,
        "collocation_degree": 3,
        "collocation_method": "radau",
        "use_sx": False,
        "enforce_start_constraints": False,
        "disable_standard_ipopt_warmup": False,
        "disable_periodic_fes_warmup_projection": False,
        "fatigue_warmstart_mode": None,
    },
}


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


def _successful_prefix_length(statuses: list | None) -> int:
    length = 0
    for status in statuses or []:
        if status != 0:
            break
        length += 1
    return length


def _validated_cycle_count(result: dict) -> int:
    prefix = _successful_prefix_length(result.get("window_statuses"))
    if result.get("solver_success"):
        return int(result.get("covered_cycles") or prefix)
    return prefix


def _configured_state_node_stride(result: dict) -> int:
    args = result["args"]
    ode_solver = str(getattr(args, "ode_solver", "")).lower()
    if ode_solver in {"collocation", "irk"}:
        return int(getattr(args, "collocation_degree", 3)) + 1
    return 1


def _exported_cycle_count(result: dict) -> int:
    if result.get("exported_cycles") is not None:
        return int(result["exported_cycles"])

    shooting_per_cycle = int(result["args"].stimulations_per_cycle)
    state_stride = _configured_state_node_stride(result)
    state_node_count = np.asarray(result["wheel_angle_trace"]).size
    interval_count, remainder = divmod(state_node_count - 1, state_stride)
    cycle_count, cycle_remainder = divmod(interval_count, shooting_per_cycle)
    if remainder or cycle_remainder:
        raise ValueError(
            "Cannot infer the number of exported cycles from the state trace."
        )
    return cycle_count


def _shooting_node_state_trace(
    values: np.ndarray, result: dict, cycle_count: int
) -> np.ndarray:
    values = np.asarray(values)
    shooting_per_cycle = int(result["args"].stimulations_per_cycle)
    exported_cycles = _exported_cycle_count(result)
    exported_intervals = exported_cycles * shooting_per_cycle
    if exported_intervals <= 0:
        return values[..., :0]

    available_intervals, remainder = divmod(values.shape[-1] - 1, exported_intervals)
    if remainder:
        raise ValueError(
            "State trace cannot be mapped exactly to the shooting-node grid: "
            f"{values.shape[-1]} values for {exported_intervals} intervals."
        )
    stride = available_intervals
    expected_stride = _configured_state_node_stride(result)
    if stride != expected_stride:
        raise ValueError(
            "Unexpected number of state points per shooting interval: "
            f"observed {stride}, expected {expected_stride}."
        )

    requested_intervals = cycle_count * shooting_per_cycle
    return values[..., : requested_intervals * stride + 1 : stride]


def _shooting_node_control_trace(
    values: np.ndarray, result: dict, cycle_count: int
) -> np.ndarray:
    values = np.asarray(values)
    shooting_per_cycle = int(result["args"].stimulations_per_cycle)
    exported_intervals = _exported_cycle_count(result) * shooting_per_cycle
    if exported_intervals <= 0:
        return values[..., :0]

    stride, remainder = divmod(values.shape[-1], exported_intervals)
    if remainder or stride <= 0:
        raise ValueError(
            "Control trace cannot be mapped exactly to the shooting-node grid: "
            f"{values.shape[-1]} values for {exported_intervals} intervals."
        )
    requested_intervals = cycle_count * shooting_per_cycle
    return values[..., : requested_intervals * stride : stride]


def _truncate_result_to_cycles(result: dict, cycle_count: int) -> dict:
    return {
        **result,
        "wheel_angle_trace": _shooting_node_state_trace(
            result["wheel_angle_trace"], result, cycle_count
        ),
        "state_traces": {
            key: _shooting_node_state_trace(values, result, cycle_count)
            for key, values in result.get("state_traces", {}).items()
        },
        "control_traces": {
            key: _shooting_node_control_trace(values, result, cycle_count)
            for key, values in result.get("control_traces", {}).items()
        },
    }


def _window_performance(result: dict) -> dict:
    windows = result.get("window_solutions") or []
    rows = []
    for index, solution in enumerate(windows):
        rows.append(
            {
                "window": index,
                "status": solution.status,
                "success": solution.status == 0,
                "solver_time_s": solution.solver_time_to_optimize,
                "wall_time_s": solution.real_time_to_optimize,
            }
        )

    prefix = _successful_prefix_length(result.get("window_statuses"))
    successful_rows = rows[:prefix]

    def total(key: str) -> float:
        return float(sum(float(row[key] or 0.0) for row in successful_rows))

    validated_cycles = _validated_cycle_count(result)
    successful_solver_time = total("solver_time_s")
    successful_wall_time = total("wall_time_s")
    return {
        "rows": rows,
        "successful_prefix_windows": prefix,
        "validated_cycles": validated_cycles,
        "successful_solver_time_s": successful_solver_time,
        "successful_wall_time_s": successful_wall_time,
        "solver_time_per_cycle_s": (
            successful_solver_time / validated_cycles if validated_cycles else None
        ),
        "wall_time_per_cycle_s": (
            successful_wall_time / validated_cycles if validated_cycles else None
        ),
    }


def _fatigue_metrics(result: dict, cycle_count: int) -> list[dict]:
    if cycle_count <= 0:
        return []
    limited = _truncate_result_to_cycles(result, cycle_count)
    rows = []
    for key, values in sorted(limited.get("state_traces", {}).items()):
        if not key.startswith(("A_", "Tau1_", "Km_")):
            continue
        trace = np.asarray(values, dtype=float).reshape(-1)
        if trace.size == 0 or not np.all(np.isfinite(trace)):
            continue
        initial = float(trace[0])
        final = float(trace[-1])
        rows.append(
            {
                "key": key,
                "initial": initial,
                "final": final,
                "relative_final": final / initial if initial else None,
                "minimum": float(np.min(trace)),
                "maximum": float(np.max(trace)),
            }
        )
    return rows


def _control_saturation_metrics(result: dict, cycle_count: int) -> list[dict]:
    if cycle_count <= 0:
        return []
    limited = _truncate_result_to_cycles(result, cycle_count)
    bounds = result.get("control_bounds", {})
    rows = []
    for key, values in sorted(limited.get("control_traces", {}).items()):
        if key not in bounds:
            continue
        trace = np.asarray(values, dtype=float).reshape(-1)
        lower = float(bounds[key]["lower"])
        upper = float(bounds[key]["upper"])
        span = upper - lower
        tolerance = max(1e-12, span * 1e-3)
        rows.append(
            {
                "key": key,
                "lower": lower,
                "upper": upper,
                "lower_fraction": float(np.mean(trace <= lower + tolerance)),
                "upper_fraction": float(np.mean(trace >= upper - tolerance)),
                "maximum": float(np.max(trace)),
                "mean": float(np.mean(trace)),
            }
        )
    return rows


def _stop_classification(result: dict) -> dict:
    validated_cycles = _validated_cycle_count(result)
    statuses = result.get("window_statuses") or []
    if result.get("success"):
        return {
            "label": "completed_requested_horizon",
            "validated_cycles": validated_cycles,
            "first_failure_status": None,
        }
    first_failure = next((status for status in statuses if status != 0), None)
    return {
        "label": (
            "numerical_failure_before_valid_cycle"
            if validated_cycles == 0
            else "solver_failure_after_valid_cycles"
        ),
        "validated_cycles": validated_cycles,
        "first_failure_status": first_failure,
    }


def _shared_stop_classification(ipopt_result: dict, acados_result: dict) -> dict:
    ipopt = _stop_classification(ipopt_result)
    acados = _stop_classification(acados_result)
    if ipopt_result.get("success") and acados_result.get("success"):
        return {"label": "horizon_completed", "evidence": []}

    evidence = []
    if not ipopt_result.get("success") and not acados_result.get("success"):
        cycle_gap = abs(ipopt["validated_cycles"] - acados["validated_cycles"])
        if cycle_gap <= 1:
            evidence.append("both_solvers_stop_at_similar_cycle")

    saturation = []
    for result, stop in ((ipopt_result, ipopt), (acados_result, acados)):
        saturation.extend(_control_saturation_metrics(result, stop["validated_cycles"]))
    if saturation and max(row["upper_fraction"] for row in saturation) >= 0.1:
        evidence.append("pulse_width_upper_bound_active")

    if set(evidence) == {
        "both_solvers_stop_at_similar_cycle",
        "pulse_width_upper_bound_active",
    }:
        label = "shared_capacity_limit_candidate"
    else:
        label = "numerical_or_unconfirmed_limit"
    return {"label": label, "evidence": evidence}


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2 * np.pi) - np.pi


def _trace_comparison(ipopt_result: dict, acados_result: dict) -> dict:
    ipopt_trace = np.asarray(ipopt_result["wheel_angle_trace"], dtype=float).squeeze()
    acados_trace = np.asarray(acados_result["wheel_angle_trace"], dtype=float).squeeze()
    if ipopt_trace.size != acados_trace.size:
        raise ValueError("Shooting-node wheel traces must have the same length.")
    common_len = ipopt_trace.size
    ipopt_common = np.unwrap(ipopt_trace)
    acados_common = np.unwrap(acados_trace)
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
        if ipopt_values.size != acados_values.size:
            raise ValueError(
                f"Shooting-node control traces for '{key}' must have the same length."
            )
        common_len = ipopt_values.size
        ipopt_common = ipopt_values
        acados_common = acados_values
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
            common_len = min(reference_values.shape[1], compared_values.shape[1])
            reference_common = reference_values[row, :common_len]
            compared_common = compared_values[row, :common_len]
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


def _shared_initial_guess_comparison(ipopt_result: dict, acados_result: dict) -> dict:
    """Compare the physical primal supplied to each backend before its first solve."""

    categories = (
        ("states", "initial_guess_state_traces"),
        ("controls", "initial_guess_control_traces"),
    )
    max_error = 0.0
    mismatches = []
    common_values = 0
    for category, result_key in categories:
        ipopt_values = ipopt_result.get(result_key, {})
        acados_values = acados_result.get(result_key, {})
        if set(ipopt_values) != set(acados_values):
            mismatches.append(f"{category}_keys")
            continue
        for key in sorted(ipopt_values):
            ipopt_array = np.asarray(ipopt_values[key], dtype=float)
            acados_array = np.asarray(acados_values[key], dtype=float)
            if ipopt_array.shape != acados_array.shape:
                mismatches.append(f"{category}:{key}:shape")
                continue
            common_values += ipopt_array.size
            if ipopt_array.size:
                max_error = max(
                    max_error, float(np.max(np.abs(ipopt_array - acados_array)))
                )

    ipopt_audits = ipopt_result.get("initial_guess_audits") or []
    acados_audits = acados_result.get("initial_guess_audits") or []
    ipopt_signature = ipopt_audits[0].get("signature") if ipopt_audits else None
    acados_signature = acados_audits[0].get("signature") if acados_audits else None
    comparable = not mismatches and common_values > 0
    return {
        "comparable": comparable,
        "exact": comparable and max_error == 0.0,
        "max_abs_error": max_error if comparable else None,
        "common_values": common_values,
        "mismatches": mismatches,
        "ipopt_signature": ipopt_signature,
        "acados_signature": acados_signature,
    }


def _normalize_ipopt_profile(profile: str) -> str:
    return profile.replace("-", "_")


def _pick(value, fallback):
    return fallback if value is None else value


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
    ipopt_dual_warm_start_mode: str,
    acados_max_iter: int,
    control_regularization_weight: float,
    control_regularization_target: float | None,
    control_regularization_target_source: str,
    wheel_qdot_regularization_weight: float,
    wheel_qdot_regularization_target: float,
    wheel_qdot_bound_margin: float,
    terminal_qdot_regularization_weight: float,
    terminal_qdot_regularization_target_source: str,
    acados_terminal_wheel_q_slack: float,
    state_scaling: str,
    pulse_width_scaling: float,
    acados_pulse_width_trust_radius: float | None,
    acados_transfer_pulse_width_trust_radius: float | None,
    acados_fes_state_trust_radius: float | None,
    acados_fatigue_warmstart_mode: str,
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
    periodic_fes_warmup_projection_proximity_weight: float,
    periodic_fes_warmup_projection_defect_weight: float,
    periodic_fes_warmup_projection_trust_radius: float | None,
    periodic_fes_warmup_projection_max_iterations: int,
    periodic_fes_warmup_force_projection_weight: float,
    periodic_fes_warmup_force_qdot_defect_limit: float,
    periodic_fes_warmup_force_adaptive_steps: int,
    acados_diagnostics: bool,
    periodic_ipopt_refinement: bool,
    periodic_ipopt_refinement_iterations: int,
    periodic_ipopt_refinement_use_sx: bool,
    warmup_state_comparison_limit: int,
    ipopt_profile: str = "historical",
    ipopt_model_formulation: str | None = None,
    ipopt_torque_application: str | None = None,
    ipopt_ode_solver: str | None = None,
    ipopt_rk_steps: int | None = None,
    ipopt_collocation_degree: int | None = None,
    ipopt_collocation_method: str | None = None,
    ipopt_use_sx: bool | None = None,
    ipopt_enforce_start_constraints: bool | None = None,
    ipopt_disable_standard_warmup: bool | None = None,
    ipopt_disable_periodic_fes_warmup_projection: bool | None = None,
    ipopt_fatigue_warmstart_mode: str | None = None,
    ipopt_disable_historical_initial_guess: bool = False,
) -> argparse.Namespace:
    if solver_name == "ipopt":
        normalized_profile = _normalize_ipopt_profile(ipopt_profile)
        if normalized_profile not in IPOPT_PROFILE_DEFAULTS:
            raise ValueError(
                "--ipopt-profile must be 'historical', 'periodic_collocation', "
                "or 'acados_like'."
            )

        defaults = IPOPT_PROFILE_DEFAULTS[normalized_profile]
        disable_projection_default = defaults["disable_periodic_fes_warmup_projection"]
        if disable_projection_default is None:
            disable_projection_default = disable_periodic_fes_warmup_projection

        fatigue_warmstart_default = defaults["fatigue_warmstart_mode"]
        if fatigue_warmstart_default is None:
            fatigue_warmstart_default = acados_fatigue_warmstart_mode

        return _namespace_from_cli(
            solver="ipopt",
            single_shot=False,
            model_formulation=_pick(
                ipopt_model_formulation, defaults["model_formulation"]
            ),
            torque_application=_pick(
                ipopt_torque_application, defaults["torque_application"]
            ),
            cycles_per_window=cycles_per_window,
            stimulations_per_cycle=stimulations_per_cycle,
            objective=objective,
            objective_shape=objective_shape,
            constant_crank_torque=resistive_torque,
            max_ipopt_iterations=ipopt_max_iter,
            ipopt_linear_solver=ipopt_linear_solver,
            ipopt_dual_warm_start_mode=ipopt_dual_warm_start_mode,
            n_windows=n_windows,
            ode_solver=_pick(ipopt_ode_solver, defaults["ode_solver"]),
            collocation_degree=_pick(
                ipopt_collocation_degree, defaults["collocation_degree"]
            ),
            collocation_method=_pick(
                ipopt_collocation_method, defaults["collocation_method"]
            ),
            rk_steps=_pick(ipopt_rk_steps, defaults["rk_steps"]),
            use_sx=_pick(ipopt_use_sx, defaults["use_sx"]),
            enforce_start_constraints=_pick(
                ipopt_enforce_start_constraints,
                defaults["enforce_start_constraints"],
            ),
            disable_standard_ipopt_warmup=_pick(
                ipopt_disable_standard_warmup,
                defaults["disable_standard_ipopt_warmup"],
            ),
            max_consecutive_failing=1,
            codegen_tag=codegen_tag,
            control_regularization_weight=control_regularization_weight,
            control_regularization_target=control_regularization_target,
            control_regularization_target_source=control_regularization_target_source,
            wheel_qdot_regularization_weight=wheel_qdot_regularization_weight,
            wheel_qdot_regularization_target=wheel_qdot_regularization_target,
            wheel_qdot_bound_margin=wheel_qdot_bound_margin,
            terminal_qdot_regularization_weight=(terminal_qdot_regularization_weight),
            terminal_qdot_regularization_target_source=(
                terminal_qdot_regularization_target_source
            ),
            acados_terminal_wheel_q_slack=acados_terminal_wheel_q_slack,
            state_scaling=state_scaling,
            pulse_width_scaling=pulse_width_scaling,
            acados_pulse_width_trust_radius=None,
            acados_transfer_pulse_width_trust_radius=None,
            acados_fes_state_trust_radius=None,
            acados_fatigue_warmstart_mode=_pick(
                ipopt_fatigue_warmstart_mode, fatigue_warmstart_default
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
            acados_ext_qp_res=False,
            acados_project_qdot_from_q=False,
            disable_periodic_fes_warmup_projection=_pick(
                ipopt_disable_periodic_fes_warmup_projection,
                disable_projection_default,
            ),
            periodic_fes_warmup_projection_weight=periodic_fes_warmup_projection_weight,
            periodic_fes_warmup_projection_mode=periodic_fes_warmup_projection_mode,
            periodic_fes_warmup_projection_strategy=(
                periodic_fes_warmup_projection_strategy
            ),
            periodic_fes_warmup_projection_substeps=(
                periodic_fes_warmup_projection_substeps
            ),
            periodic_fes_warmup_projection_proximity_weight=(
                periodic_fes_warmup_projection_proximity_weight
            ),
            periodic_fes_warmup_projection_defect_weight=(
                periodic_fes_warmup_projection_defect_weight
            ),
            periodic_fes_warmup_projection_trust_radius=(
                periodic_fes_warmup_projection_trust_radius
            ),
            periodic_fes_warmup_projection_max_iterations=(
                periodic_fes_warmup_projection_max_iterations
            ),
            periodic_fes_warmup_force_projection_weight=(
                periodic_fes_warmup_force_projection_weight
            ),
            periodic_fes_warmup_force_qdot_defect_limit=(
                periodic_fes_warmup_force_qdot_defect_limit
            ),
            periodic_fes_warmup_force_adaptive_steps=(
                periodic_fes_warmup_force_adaptive_steps
            ),
            acados_diagnostics=False,
            periodic_ipopt_refinement=False,
            periodic_ipopt_refinement_iterations=periodic_ipopt_refinement_iterations,
            periodic_ipopt_refinement_use_sx=False,
            warmup_state_comparison_limit=warmup_state_comparison_limit,
            disable_historical_ipopt_initial_guess=(
                ipopt_disable_historical_initial_guess
            ),
        )

    if solver_name == "acados":
        return _namespace_from_cli(
            solver="acados",
            single_shot=False,
            model_formulation="periodic_node",
            torque_application="constant",
            cycles_per_window=cycles_per_window,
            stimulations_per_cycle=stimulations_per_cycle,
            objective=objective,
            objective_shape=objective_shape,
            constant_crank_torque=resistive_torque,
            max_acados_iterations=acados_max_iter,
            ipopt_linear_solver=ipopt_linear_solver,
            ipopt_dual_warm_start_mode=ipopt_dual_warm_start_mode,
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
            wheel_qdot_bound_margin=wheel_qdot_bound_margin,
            terminal_qdot_regularization_weight=(terminal_qdot_regularization_weight),
            terminal_qdot_regularization_target_source=(
                terminal_qdot_regularization_target_source
            ),
            acados_terminal_wheel_q_slack=acados_terminal_wheel_q_slack,
            state_scaling=state_scaling,
            pulse_width_scaling=pulse_width_scaling,
            acados_pulse_width_trust_radius=acados_pulse_width_trust_radius,
            acados_transfer_pulse_width_trust_radius=(
                acados_transfer_pulse_width_trust_radius
            ),
            acados_fes_state_trust_radius=acados_fes_state_trust_radius,
            acados_fatigue_warmstart_mode=acados_fatigue_warmstart_mode,
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
            periodic_fes_warmup_projection_proximity_weight=(
                periodic_fes_warmup_projection_proximity_weight
            ),
            periodic_fes_warmup_projection_defect_weight=(
                periodic_fes_warmup_projection_defect_weight
            ),
            periodic_fes_warmup_projection_trust_radius=(
                periodic_fes_warmup_projection_trust_radius
            ),
            periodic_fes_warmup_projection_max_iterations=(
                periodic_fes_warmup_projection_max_iterations
            ),
            periodic_fes_warmup_force_projection_weight=(
                periodic_fes_warmup_force_projection_weight
            ),
            periodic_fes_warmup_force_qdot_defect_limit=(
                periodic_fes_warmup_force_qdot_defect_limit
            ),
            periodic_fes_warmup_force_adaptive_steps=(
                periodic_fes_warmup_force_adaptive_steps
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
    performance = {
        "IPOPT": _window_performance(ipopt_result),
        "ACADOS": _window_performance(acados_result),
    }
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
        if result.get("error"):
            print(f"{label} solve_error: {result['error']}")
        if result.get("window_iterations"):
            iterations = [
                value for value in result["window_iterations"] if value is not None
            ]
            print(
                f"{label} window_iterations={result['window_iterations']} "
                f"total_iterations={sum(iterations)}"
            )
        solver_per_cycle = performance[label]["solver_time_per_cycle_s"]
        wall_per_cycle = performance[label]["wall_time_per_cycle_s"]
        print(
            f"{label} benchmark timing: "
            f"validated_cycles={performance[label]['validated_cycles']} "
            f"successful_prefix_windows={performance[label]['successful_prefix_windows']} "
            f"solver_time_per_cycle_s={_format_metric(solver_per_cycle)} "
            f"wall_time_per_cycle_s={_format_metric(wall_per_cycle)} "
            "initial_guess_preparation_time_s="
            f"{_format_metric(result.get('initial_guess_preparation_time_s'))} "
            "standard_warmup_cache_hit="
            f"{_format_metric(result.get('standard_warmup_cache_hit'))} "
            f"end_to_end_wall_time_s={_format_metric(result.get('end_to_end_wall_time_s'))}"
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

    initial_guess_comparison = _shared_initial_guess_comparison(
        ipopt_result, acados_result
    )
    print(
        "initial guess fairness | "
        f"comparable={initial_guess_comparison['comparable']} | "
        f"exact={initial_guess_comparison['exact']} | "
        f"max_abs_error={_format_metric(initial_guess_comparison['max_abs_error'])} | "
        f"common_values={initial_guess_comparison['common_values']} | "
        f"ipopt_signature={initial_guess_comparison['ipopt_signature']} | "
        f"acados_signature={initial_guess_comparison['acados_signature']} | "
        f"mismatches={initial_guess_comparison['mismatches']}"
    )

    common_validated_cycles = min(
        performance["IPOPT"]["validated_cycles"],
        performance["ACADOS"]["validated_cycles"],
    )
    if not (ipopt_result.get("success") and acados_result.get("success")):
        print(
            "wheel trace comparison warning: at least one solver did not cover all requested cycles successfully."
        )
    if common_validated_cycles == 0:
        print("solution comparison unavailable: no commonly validated cycle.")
        shared_stop = _shared_stop_classification(ipopt_result, acados_result)
        print(
            "benchmark stop classification | "
            f"label={shared_stop['label']} | evidence={shared_stop['evidence']}"
        )
        return

    comparison_ipopt = _truncate_result_to_cycles(ipopt_result, common_validated_cycles)
    comparison_acados = _truncate_result_to_cycles(
        acados_result, common_validated_cycles
    )
    print(f"solution comparison validated cycles: {common_validated_cycles}")
    print("solution comparison grid: shooting_nodes_only")
    trace_metrics = _trace_comparison(comparison_ipopt, comparison_acados)
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

    control_metrics = _control_comparisons(comparison_ipopt, comparison_acados)
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

    state_metrics = _state_comparisons(comparison_ipopt, comparison_acados)
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

    for label, result in (("IPOPT", ipopt_result), ("ACADOS", acados_result)):
        validated_cycles = performance[label]["validated_cycles"]
        fatigue = _fatigue_metrics(result, validated_cycles)
        saturation = _control_saturation_metrics(result, validated_cycles)
        a_ratios = [
            row["relative_final"]
            for row in fatigue
            if row["key"].startswith("A_") and row["relative_final"] is not None
        ]
        print(
            f"{label} endurance: "
            f"stop={_stop_classification(result)['label']} "
            f"min_A_capacity_ratio={_format_metric(min(a_ratios) if a_ratios else None)} "
            "max_pulse_width_upper_fraction="
            f"{_format_metric(max((row['upper_fraction'] for row in saturation), default=None))}"
        )

    shared_stop = _shared_stop_classification(ipopt_result, acados_result)
    print(
        "benchmark stop classification | "
        f"label={shared_stop['label']} | evidence={shared_stop['evidence']}"
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
    ipopt_dual_warm_start_mode: str = "bounds",
    acados_max_iter: int = 100,
    control_regularization_weight: float = 0.0,
    acados_control_regularization_weight: float | None = None,
    control_regularization_target: float | None = None,
    control_regularization_target_source: str = "constant",
    acados_control_regularization_target_source: str | None = None,
    wheel_qdot_regularization_weight: float = 0.0,
    acados_wheel_qdot_regularization_weight: float | None = None,
    wheel_qdot_regularization_target: float = -float(2 * np.pi),
    wheel_qdot_bound_margin: float = 3.0,
    terminal_qdot_regularization_weight: float = 0.0,
    terminal_qdot_regularization_target_source: str = "previous",
    acados_terminal_wheel_q_slack: float = 0.2,
    acados_terminal_wheel_q_homotopy_slacks: tuple[float, ...] | None = None,
    acados_terminal_wheel_q_homotopy_each_window: bool = False,
    state_scaling: str = "none",
    acados_state_scaling: str | None = None,
    pulse_width_scaling: float = 1 / 400,
    acados_pulse_width_scaling: float | None = None,
    acados_pulse_width_trust_radius: float | None = None,
    acados_transfer_pulse_width_trust_radius: float | None = None,
    acados_proximal_control_weights: tuple[float, ...] | None = None,
    acados_proximal_control_each_window: bool = False,
    acados_proximal_control_tolerance: float = 5e-4,
    acados_proximal_control_stage_iterations: int = 50,
    acados_proximal_control_max_restarts: int = 1,
    acados_transfer_sqp_restarts: int = 0,
    acados_transfer_sqp_restart_iterations: int = 1,
    acados_transfer_sqp_restart_feasibility_tolerance: float = 1e-2,
    acados_fes_state_trust_radius: float | None = None,
    acados_fatigue_warmstart_mode: str = "continuous",
    acados_tolerance: float | None = None,
    acados_stationarity_tolerance: float | None = None,
    acados_qp_iter_max: int = 50,
    acados_dual_warm_start_mode: str = "reset",
    acados_levenberg_marquardt: float = 0.0,
    acados_regularize_method: str = "GERSHGORIN_LEVENBERG_MARQUARDT",
    acados_hessian_approx: str = "GAUSS_NEWTON",
    acados_nlp_solver_type: str = "SQP",
    acados_search_direction_mode: str = "NOMINAL_QP",
    acados_globalization: str = "FUNNEL_L1PEN_LINESEARCH",
    acados_fixed_step_length: float = 1.0,
    acados_nlp_qp_tol_strategy: str = "ADAPTIVE_QPSCALING",
    acados_qpscaling_scale_objective: str = "OBJECTIVE_GERSHGORIN",
    acados_qpscaling_scale_constraints: str = "INF_NORM",
    acados_ext_qp_res: bool = False,
    acados_project_qdot_from_q: bool = False,
    shared_transfer_full_dynamics_rollout: bool = False,
    shared_transfer_phase_one: bool = False,
    shared_initial_phase_one: bool = False,
    shared_transfer_rollout_substeps: int = 5,
    shared_transfer_rollout_max_bound_violation: float = 1.0,
    shared_transfer_ding_force_compensation: bool = False,
    shared_transfer_ding_force_compensation_substeps: int = 5,
    shared_transfer_ding_force_compensation_iterations: int = 20,
    acados_transfer_ding_force_compensation: bool = False,
    acados_integrator_type: str = "IRK",
    acados_collocation_type: str = "GAUSS_LEGENDRE",
    acados_sim_stages: int = 4,
    acados_sim_steps: int = 5,
    disable_periodic_fes_warmup_projection: bool = False,
    periodic_fes_warmup_projection_weight: float = 1.0,
    periodic_fes_warmup_projection_mode: str = "all",
    periodic_fes_warmup_projection_strategy: str = "sequential",
    periodic_fes_warmup_projection_substeps: int = 10,
    periodic_fes_warmup_projection_proximity_weight: float = 1.0,
    periodic_fes_warmup_projection_defect_weight: float = 100.0,
    periodic_fes_warmup_projection_trust_radius: float | None = None,
    periodic_fes_warmup_projection_max_iterations: int = 200,
    periodic_fes_warmup_force_projection_weight: float = 0.25,
    periodic_fes_warmup_force_qdot_defect_limit: float = 3.0,
    periodic_fes_warmup_force_adaptive_steps: int = 10,
    acados_diagnostics: bool = False,
    periodic_ipopt_refinement: bool = False,
    periodic_ipopt_refinement_iterations: int = 300,
    periodic_ipopt_refinement_use_sx: bool = False,
    periodic_ipopt_refinement_ode_solver: str = "target",
    warmup_state_comparison_limit: int = 12,
    state_comparison_limit: int = 12,
    print_traces: bool = False,
    ipopt_profile: str = "historical",
    ipopt_model_formulation: str | None = None,
    ipopt_torque_application: str | None = None,
    ipopt_ode_solver: str | None = None,
    ipopt_rk_steps: int | None = None,
    ipopt_collocation_degree: int | None = None,
    ipopt_collocation_method: str | None = None,
    ipopt_use_sx: bool | None = None,
    ipopt_enforce_start_constraints: bool | None = None,
    ipopt_disable_standard_warmup: bool | None = None,
    ipopt_disable_periodic_fes_warmup_projection: bool | None = None,
    ipopt_fatigue_warmstart_mode: str | None = None,
    ipopt_disable_historical_initial_guess: bool = False,
    max_consecutive_failing: int = 1,
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
        ipopt_dual_warm_start_mode=ipopt_dual_warm_start_mode,
        acados_max_iter=acados_max_iter,
        control_regularization_weight=control_regularization_weight,
        control_regularization_target=control_regularization_target,
        control_regularization_target_source=control_regularization_target_source,
        wheel_qdot_regularization_weight=wheel_qdot_regularization_weight,
        wheel_qdot_regularization_target=wheel_qdot_regularization_target,
        wheel_qdot_bound_margin=wheel_qdot_bound_margin,
        terminal_qdot_regularization_weight=terminal_qdot_regularization_weight,
        terminal_qdot_regularization_target_source=(
            terminal_qdot_regularization_target_source
        ),
        acados_terminal_wheel_q_slack=acados_terminal_wheel_q_slack,
        state_scaling=state_scaling,
        pulse_width_scaling=pulse_width_scaling,
        acados_pulse_width_trust_radius=None,
        acados_transfer_pulse_width_trust_radius=None,
        acados_fes_state_trust_radius=None,
        acados_fatigue_warmstart_mode="continuous",
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
        disable_periodic_fes_warmup_projection=(disable_periodic_fes_warmup_projection),
        periodic_fes_warmup_projection_weight=periodic_fes_warmup_projection_weight,
        periodic_fes_warmup_projection_mode=periodic_fes_warmup_projection_mode,
        periodic_fes_warmup_projection_strategy=periodic_fes_warmup_projection_strategy,
        periodic_fes_warmup_projection_substeps=periodic_fes_warmup_projection_substeps,
        periodic_fes_warmup_projection_proximity_weight=(
            periodic_fes_warmup_projection_proximity_weight
        ),
        periodic_fes_warmup_projection_defect_weight=(
            periodic_fes_warmup_projection_defect_weight
        ),
        periodic_fes_warmup_projection_trust_radius=(
            periodic_fes_warmup_projection_trust_radius
        ),
        periodic_fes_warmup_projection_max_iterations=(
            periodic_fes_warmup_projection_max_iterations
        ),
        periodic_fes_warmup_force_projection_weight=(
            periodic_fes_warmup_force_projection_weight
        ),
        periodic_fes_warmup_force_qdot_defect_limit=(
            periodic_fes_warmup_force_qdot_defect_limit
        ),
        periodic_fes_warmup_force_adaptive_steps=(
            periodic_fes_warmup_force_adaptive_steps
        ),
        acados_diagnostics=acados_diagnostics,
        periodic_ipopt_refinement=False,
        periodic_ipopt_refinement_iterations=periodic_ipopt_refinement_iterations,
        periodic_ipopt_refinement_use_sx=False,
        warmup_state_comparison_limit=warmup_state_comparison_limit,
        ipopt_profile=ipopt_profile,
        ipopt_model_formulation=ipopt_model_formulation,
        ipopt_torque_application=ipopt_torque_application,
        ipopt_ode_solver=ipopt_ode_solver,
        ipopt_rk_steps=ipopt_rk_steps,
        ipopt_collocation_degree=ipopt_collocation_degree,
        ipopt_collocation_method=ipopt_collocation_method,
        ipopt_use_sx=ipopt_use_sx,
        ipopt_enforce_start_constraints=ipopt_enforce_start_constraints,
        ipopt_disable_standard_warmup=ipopt_disable_standard_warmup,
        ipopt_disable_periodic_fes_warmup_projection=(
            ipopt_disable_periodic_fes_warmup_projection
        ),
        ipopt_fatigue_warmstart_mode=ipopt_fatigue_warmstart_mode,
        ipopt_disable_historical_initial_guess=(ipopt_disable_historical_initial_guess),
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
        ipopt_dual_warm_start_mode=ipopt_dual_warm_start_mode,
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
        wheel_qdot_bound_margin=wheel_qdot_bound_margin,
        terminal_qdot_regularization_weight=terminal_qdot_regularization_weight,
        terminal_qdot_regularization_target_source=(
            terminal_qdot_regularization_target_source
        ),
        acados_terminal_wheel_q_slack=acados_terminal_wheel_q_slack,
        state_scaling=(
            acados_state_scaling if acados_state_scaling is not None else state_scaling
        ),
        pulse_width_scaling=(
            acados_pulse_width_scaling
            if acados_pulse_width_scaling is not None
            else pulse_width_scaling
        ),
        acados_pulse_width_trust_radius=acados_pulse_width_trust_radius,
        acados_transfer_pulse_width_trust_radius=(
            acados_transfer_pulse_width_trust_radius
        ),
        acados_fes_state_trust_radius=acados_fes_state_trust_radius,
        acados_fatigue_warmstart_mode=acados_fatigue_warmstart_mode,
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
        periodic_fes_warmup_projection_proximity_weight=(
            periodic_fes_warmup_projection_proximity_weight
        ),
        periodic_fes_warmup_projection_defect_weight=(
            periodic_fes_warmup_projection_defect_weight
        ),
        periodic_fes_warmup_projection_trust_radius=(
            periodic_fes_warmup_projection_trust_radius
        ),
        periodic_fes_warmup_projection_max_iterations=(
            periodic_fes_warmup_projection_max_iterations
        ),
        periodic_fes_warmup_force_projection_weight=(
            periodic_fes_warmup_force_projection_weight
        ),
        periodic_fes_warmup_force_qdot_defect_limit=(
            periodic_fes_warmup_force_qdot_defect_limit
        ),
        periodic_fes_warmup_force_adaptive_steps=(
            periodic_fes_warmup_force_adaptive_steps
        ),
        acados_diagnostics=acados_diagnostics,
        periodic_ipopt_refinement=periodic_ipopt_refinement,
        periodic_ipopt_refinement_iterations=periodic_ipopt_refinement_iterations,
        periodic_ipopt_refinement_use_sx=periodic_ipopt_refinement_use_sx,
        warmup_state_comparison_limit=warmup_state_comparison_limit,
    )
    ipopt_args.max_consecutive_failing = max_consecutive_failing
    acados_args.max_consecutive_failing = max_consecutive_failing
    acados_args.acados_integrator_type = acados_integrator_type
    acados_args.acados_collocation_type = acados_collocation_type
    acados_args.acados_sim_stages = acados_sim_stages
    acados_args.acados_sim_steps = acados_sim_steps
    acados_args.periodic_ipopt_refinement_ode_solver = (
        periodic_ipopt_refinement_ode_solver
    )
    acados_args.acados_stationarity_tolerance = acados_stationarity_tolerance
    acados_args.acados_dual_warm_start_mode = acados_dual_warm_start_mode
    acados_args.acados_proximal_control_weights = acados_proximal_control_weights
    acados_args.acados_proximal_control_each_window = (
        acados_proximal_control_each_window
    )
    acados_args.acados_proximal_control_tolerance = acados_proximal_control_tolerance
    acados_args.acados_proximal_control_stage_iterations = (
        acados_proximal_control_stage_iterations
    )
    acados_args.acados_proximal_control_max_restarts = (
        acados_proximal_control_max_restarts
    )
    acados_args.acados_transfer_sqp_restarts = acados_transfer_sqp_restarts
    acados_args.acados_transfer_sqp_restart_iterations = (
        acados_transfer_sqp_restart_iterations
    )
    acados_args.acados_transfer_sqp_restart_feasibility_tolerance = (
        acados_transfer_sqp_restart_feasibility_tolerance
    )
    acados_args.acados_terminal_wheel_q_homotopy_slacks = (
        acados_terminal_wheel_q_homotopy_slacks
    )
    acados_args.acados_terminal_wheel_q_homotopy_each_window = (
        acados_terminal_wheel_q_homotopy_each_window
    )
    for solver_args in (ipopt_args, acados_args):
        solver_args.acados_transfer_full_dynamics_rollout = (
            shared_transfer_full_dynamics_rollout
        )
        solver_args.acados_transfer_phase_one = shared_transfer_phase_one
        solver_args.full_dynamics_phase_one = shared_initial_phase_one
        solver_args.acados_transfer_rollout_substeps = shared_transfer_rollout_substeps
        solver_args.acados_transfer_rollout_max_bound_violation = (
            shared_transfer_rollout_max_bound_violation
        )
        solver_args.transfer_ding_force_compensation = (
            shared_transfer_ding_force_compensation
        )
        solver_args.transfer_ding_force_compensation_substeps = (
            shared_transfer_ding_force_compensation_substeps
        )
        solver_args.transfer_ding_force_compensation_iterations = (
            shared_transfer_ding_force_compensation_iterations
        )
    acados_args.transfer_ding_force_compensation = bool(
        shared_transfer_ding_force_compensation
        or acados_transfer_ding_force_compensation
    )

    normalized_ipopt_profile = _normalize_ipopt_profile(ipopt_profile)
    ipopt_label = {
        "historical": "historical reference",
        "periodic_collocation": "periodic-collocation bridge",
        "acados_like": "ACADOS-like diagnostic",
    }[normalized_ipopt_profile]
    print(f"Running IPOPT configuration ({ipopt_label})...")
    start = perf_counter()
    ipopt_result = solve_case(ipopt_args, echo=True)
    ipopt_result["end_to_end_wall_time_s"] = perf_counter() - start
    print()
    print("Running ACADOS-compatible configuration...")
    start = perf_counter()
    acados_result = solve_case(acados_args, echo=True)
    acados_result["end_to_end_wall_time_s"] = perf_counter() - start
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
    parser.add_argument(
        "--max-consecutive-failing",
        type=int,
        default=1,
        help=(
            "Stop the endurance benchmark after this many consecutive failed "
            "windows. The default avoids benchmarking trajectories returned by "
            "failed solves."
        ),
    )
    parser.add_argument("--resistive-torque", type=float, default=-0.2)
    parser.add_argument("--acados-dir", default=os.environ.get("ACADOS_SOURCE_DIR"))
    parser.add_argument("--codegen-tag", default="fes_compare")
    parser.add_argument("--ipopt-max-iter", type=int, default=2000)
    parser.add_argument("--ipopt-linear-solver", default="ma57")
    parser.add_argument(
        "--ipopt-dual-warm-start-mode",
        choices=("off", "constraints", "bounds", "all"),
        default="bounds",
        help=(
            "Reuse no IPOPT duals, constraint multipliers, bound multipliers, "
            "or both between receding-horizon windows."
        ),
    )
    parser.add_argument(
        "--shared-transfer-full-dynamics-rollout",
        action="store_true",
        help=(
            "Apply the same complete-dynamics RK4 rollout to the appended cycle "
            "for IPOPT and ACADOS."
        ),
    )
    parser.add_argument(
        "--shared-transfer-phase-one",
        action="store_true",
        help="Apply the same bounded phase-I projection between windows for both solvers.",
    )
    parser.add_argument(
        "--shared-initial-phase-one",
        action="store_true",
        help="Apply the same complete-dynamics phase-I projection before the first solve.",
    )
    parser.add_argument("--shared-transfer-rollout-substeps", type=int, default=5)
    parser.add_argument(
        "--shared-transfer-rollout-max-bound-violation", type=float, default=1.0
    )
    parser.add_argument(
        "--shared-transfer-ding-force-compensation",
        action="store_true",
        help=(
            "Apply the same per-muscle Ding pulse-width compensation to IPOPT "
            "and ACADOS transfers; use with --ipopt-profile acados_like."
        ),
    )
    parser.add_argument(
        "--shared-transfer-ding-force-compensation-substeps", type=int, default=5
    )
    parser.add_argument(
        "--shared-transfer-ding-force-compensation-iterations", type=int, default=20
    )
    parser.add_argument(
        "--acados-transfer-ding-force-compensation",
        action="store_true",
        help=(
            "Apply Ding pulse-width compensation only to the periodic ACADOS "
            "transfer, leaving the historical IPOPT reference unchanged."
        ),
    )
    parser.add_argument(
        "--ipopt-profile",
        choices=(
            "historical",
            "periodic_collocation",
            "periodic-collocation",
            "acados_like",
            "acados-like",
        ),
        default="historical",
        help=(
            "Base IPOPT configuration. 'historical' keeps the robust reference "
            "problem; 'periodic_collocation' isolates the periodic dynamics and "
            "constant torque with robust collocation; 'acados_like' additionally "
            "switches IPOPT to the explicit RK setup used to diagnose ACADOS."
        ),
    )
    parser.add_argument(
        "--ipopt-model-formulation",
        choices=("standard", "periodic", "periodic_node"),
        default=None,
        help="Override the IPOPT model formulation selected by --ipopt-profile.",
    )
    parser.add_argument(
        "--ipopt-torque-application",
        choices=("constant", "external_forces"),
        default=None,
        help="Override how the IPOPT-side crank torque is applied.",
    )
    parser.add_argument(
        "--ipopt-ode-solver",
        choices=("rk4", "rk8", "irk", "collocation"),
        default=None,
        help="Override the IPOPT transcription/integration scheme.",
    )
    parser.add_argument(
        "--ipopt-rk-steps",
        type=int,
        default=None,
        help="Override IPOPT RK integration steps per shooting interval.",
    )
    parser.add_argument(
        "--ipopt-collocation-degree",
        type=int,
        default=None,
        help="Override IPOPT collocation/IRK polynomial degree.",
    )
    parser.add_argument(
        "--ipopt-collocation-method",
        default=None,
        help="Override IPOPT collocation/IRK method.",
    )
    ipopt_sx_group = parser.add_mutually_exclusive_group()
    ipopt_sx_group.add_argument(
        "--ipopt-use-sx",
        dest="ipopt_use_sx",
        action="store_true",
        default=None,
        help="Build the IPOPT-side diagnostic problem with CasADi SX graphs.",
    )
    ipopt_sx_group.add_argument(
        "--ipopt-no-use-sx",
        dest="ipopt_use_sx",
        action="store_false",
        help="Build the IPOPT-side diagnostic problem with CasADi MX graphs.",
    )
    ipopt_start_group = parser.add_mutually_exclusive_group()
    ipopt_start_group.add_argument(
        "--ipopt-enforce-start-constraints",
        dest="ipopt_enforce_start_constraints",
        action="store_true",
        default=None,
        help="Enable historical start constraints in the IPOPT-side problem.",
    )
    ipopt_start_group.add_argument(
        "--ipopt-disable-start-constraints",
        dest="ipopt_enforce_start_constraints",
        action="store_false",
        help="Disable historical start constraints in the IPOPT-side problem.",
    )
    ipopt_warmup_group = parser.add_mutually_exclusive_group()
    ipopt_warmup_group.add_argument(
        "--ipopt-enable-standard-warmup",
        dest="ipopt_disable_standard_warmup",
        action="store_false",
        default=None,
        help="Enable the standard IPOPT warmup before a periodic IPOPT diagnostic solve.",
    )
    ipopt_warmup_group.add_argument(
        "--ipopt-disable-standard-warmup",
        dest="ipopt_disable_standard_warmup",
        action="store_true",
        help="Disable the standard IPOPT warmup before a periodic IPOPT diagnostic solve.",
    )
    ipopt_projection_group = parser.add_mutually_exclusive_group()
    ipopt_projection_group.add_argument(
        "--ipopt-enable-periodic-fes-warmup-projection",
        dest="ipopt_disable_periodic_fes_warmup_projection",
        action="store_false",
        default=None,
        help="Apply the periodic FES warmup projection to the IPOPT-side diagnostic problem.",
    )
    ipopt_projection_group.add_argument(
        "--ipopt-disable-periodic-fes-warmup-projection",
        dest="ipopt_disable_periodic_fes_warmup_projection",
        action="store_true",
        help="Skip the periodic FES warmup projection for the IPOPT-side diagnostic problem.",
    )
    parser.add_argument(
        "--ipopt-fatigue-warmstart-mode",
        choices=("continuous", "cyclical"),
        default=None,
        help="Override fatigue-state warmstart shifting for periodic IPOPT diagnostics.",
    )
    parser.add_argument(
        "--ipopt-disable-historical-initial-guess",
        action="store_true",
        help="Do not load the historical initial guess file for the direct IPOPT-side solve.",
    )
    parser.add_argument("--acados-max-iter", type=int, default=100)
    parser.add_argument(
        "--acados-integrator-type", choices=("ERK", "IRK", "DISCRETE"), default="IRK"
    )
    parser.add_argument(
        "--acados-collocation-type",
        choices=("GAUSS_LEGENDRE", "GAUSS_RADAU_IIA", "EXPLICIT_RUNGE_KUTTA"),
        default="GAUSS_LEGENDRE",
    )
    parser.add_argument("--acados-sim-stages", type=int, default=4)
    parser.add_argument(
        "--acados-sim-steps",
        type=int,
        default=5,
        help=(
            "Integration steps per shooting interval. Five is the robust IRK "
            "default; one can be used with an experimental IPOPT-IRK bridge."
        ),
    )
    parser.add_argument("--control-regularization-weight", type=float, default=0.0)
    parser.add_argument(
        "--acados-control-regularization-weight", type=float, default=None
    )
    parser.add_argument("--control-regularization-target", type=float, default=None)
    parser.add_argument(
        "--control-regularization-target-source",
        choices=("constant", "warmup", "previous"),
        default="constant",
        help="Use a constant pulse-width target or the IPOPT warmup control trajectory.",
    )
    parser.add_argument(
        "--acados-control-regularization-target-source",
        choices=("constant", "warmup", "previous"),
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
    parser.add_argument("--wheel-qdot-bound-margin", type=float, default=3.0)
    parser.add_argument(
        "--terminal-qdot-regularization-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--terminal-qdot-regularization-target-source",
        choices=("initial", "previous"),
        default="previous",
    )
    parser.add_argument("--acados-terminal-wheel-q-slack", type=float, default=0.2)
    parser.add_argument(
        "--state-scaling", choices=("none", "fes", "full"), default="none"
    )
    parser.add_argument(
        "--acados-state-scaling", choices=("none", "fes", "full"), default=None
    )
    parser.add_argument("--pulse-width-scaling", type=float, default=1 / 400)
    parser.add_argument("--acados-pulse-width-scaling", type=float, default=None)
    parser.add_argument("--acados-pulse-width-trust-radius", type=float, default=None)
    parser.add_argument(
        "--acados-transfer-pulse-width-trust-radius", type=float, default=None
    )
    parser.add_argument(
        "--acados-proximal-control-weights",
        type=parse_proximal_control_weights,
        default=None,
        help="Strictly decreasing control-proximity weights used only by ACADOS.",
    )
    parser.add_argument("--acados-proximal-control-each-window", action="store_true")
    parser.add_argument("--acados-proximal-control-tolerance", type=float, default=5e-4)
    parser.add_argument(
        "--acados-proximal-control-stage-iterations", type=int, default=50
    )
    parser.add_argument("--acados-proximal-control-max-restarts", type=int, default=1)
    parser.add_argument(
        "--acados-transfer-sqp-restarts",
        type=int,
        default=0,
        help="Short SQP transfer-repair attempts before each ACADOS MHE window.",
    )
    parser.add_argument("--acados-transfer-sqp-restart-iterations", type=int, default=1)
    parser.add_argument(
        "--acados-transfer-sqp-restart-feasibility-tolerance",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--acados-terminal-wheel-q-homotopy-slacks",
        type=parse_terminal_wheel_q_slacks,
        default=None,
    )
    parser.add_argument(
        "--acados-terminal-wheel-q-homotopy-each-window",
        action="store_true",
    )
    parser.add_argument("--acados-fes-state-trust-radius", type=float, default=None)
    parser.add_argument(
        "--acados-fatigue-warmstart-mode",
        choices=("continuous", "cyclical"),
        default="continuous",
    )
    parser.add_argument("--acados-tolerance", type=float, default=None)
    parser.add_argument(
        "--acados-stationarity-tolerance",
        type=float,
        default=None,
        help="Stationarity tolerance applied independently from ACADOS feasibility.",
    )
    parser.add_argument("--acados-qp-iter-max", type=int, default=50)
    parser.add_argument(
        "--acados-dual-warm-start-mode",
        choices=("preserve", "reset", "shift"),
        default="reset",
        help="Preserve, reset, or cycle-shift ACADOS dual variables between MHE windows.",
    )
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
        default="FUNNEL_L1PEN_LINESEARCH",
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
        choices=(
            "calcium",
            "all",
            "all_except_force",
            "all_force_blend",
            "all_force_adaptive_blend",
        ),
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
        "--periodic-fes-warmup-projection-proximity-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--periodic-fes-warmup-projection-defect-weight", type=float, default=100.0
    )
    parser.add_argument(
        "--periodic-fes-warmup-projection-trust-radius", type=float, default=None
    )
    parser.add_argument(
        "--periodic-fes-warmup-projection-max-iterations", type=int, default=200
    )
    parser.add_argument(
        "--periodic-fes-warmup-force-projection-weight", type=float, default=0.25
    )
    parser.add_argument(
        "--periodic-fes-warmup-force-qdot-defect-limit", type=float, default=3.0
    )
    parser.add_argument(
        "--periodic-fes-warmup-force-adaptive-steps", type=int, default=10
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
    parser.add_argument(
        "--periodic-ipopt-refinement-ode-solver",
        choices=("target", "collocation", "rk4", "irk"),
        default="target",
        help=(
            "Integrator for the IPOPT bridge used to initialize ACADOS; "
            "collocation is the robust periodic bridge."
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
        ipopt_dual_warm_start_mode=args.ipopt_dual_warm_start_mode,
        acados_max_iter=args.acados_max_iter,
        control_regularization_weight=args.control_regularization_weight,
        acados_control_regularization_weight=args.acados_control_regularization_weight,
        control_regularization_target=args.control_regularization_target,
        control_regularization_target_source=args.control_regularization_target_source,
        acados_control_regularization_target_source=args.acados_control_regularization_target_source,
        wheel_qdot_regularization_weight=args.wheel_qdot_regularization_weight,
        acados_wheel_qdot_regularization_weight=args.acados_wheel_qdot_regularization_weight,
        wheel_qdot_regularization_target=args.wheel_qdot_regularization_target,
        wheel_qdot_bound_margin=args.wheel_qdot_bound_margin,
        terminal_qdot_regularization_weight=(args.terminal_qdot_regularization_weight),
        terminal_qdot_regularization_target_source=(
            args.terminal_qdot_regularization_target_source
        ),
        acados_terminal_wheel_q_slack=args.acados_terminal_wheel_q_slack,
        acados_terminal_wheel_q_homotopy_slacks=(
            args.acados_terminal_wheel_q_homotopy_slacks
        ),
        acados_terminal_wheel_q_homotopy_each_window=(
            args.acados_terminal_wheel_q_homotopy_each_window
        ),
        state_scaling=args.state_scaling,
        acados_state_scaling=args.acados_state_scaling,
        pulse_width_scaling=args.pulse_width_scaling,
        acados_pulse_width_scaling=args.acados_pulse_width_scaling,
        acados_pulse_width_trust_radius=args.acados_pulse_width_trust_radius,
        acados_transfer_pulse_width_trust_radius=(
            args.acados_transfer_pulse_width_trust_radius
        ),
        acados_proximal_control_weights=args.acados_proximal_control_weights,
        acados_proximal_control_each_window=(args.acados_proximal_control_each_window),
        acados_proximal_control_tolerance=(args.acados_proximal_control_tolerance),
        acados_proximal_control_stage_iterations=(
            args.acados_proximal_control_stage_iterations
        ),
        acados_proximal_control_max_restarts=(
            args.acados_proximal_control_max_restarts
        ),
        acados_transfer_sqp_restarts=args.acados_transfer_sqp_restarts,
        acados_transfer_sqp_restart_iterations=(
            args.acados_transfer_sqp_restart_iterations
        ),
        acados_transfer_sqp_restart_feasibility_tolerance=(
            args.acados_transfer_sqp_restart_feasibility_tolerance
        ),
        acados_fes_state_trust_radius=args.acados_fes_state_trust_radius,
        acados_fatigue_warmstart_mode=args.acados_fatigue_warmstart_mode,
        acados_tolerance=args.acados_tolerance,
        acados_stationarity_tolerance=args.acados_stationarity_tolerance,
        acados_qp_iter_max=args.acados_qp_iter_max,
        acados_dual_warm_start_mode=args.acados_dual_warm_start_mode,
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
        shared_transfer_full_dynamics_rollout=(
            args.shared_transfer_full_dynamics_rollout
        ),
        shared_transfer_phase_one=args.shared_transfer_phase_one,
        shared_initial_phase_one=args.shared_initial_phase_one,
        shared_transfer_rollout_substeps=args.shared_transfer_rollout_substeps,
        shared_transfer_rollout_max_bound_violation=(
            args.shared_transfer_rollout_max_bound_violation
        ),
        shared_transfer_ding_force_compensation=(
            args.shared_transfer_ding_force_compensation
        ),
        shared_transfer_ding_force_compensation_substeps=(
            args.shared_transfer_ding_force_compensation_substeps
        ),
        shared_transfer_ding_force_compensation_iterations=(
            args.shared_transfer_ding_force_compensation_iterations
        ),
        acados_transfer_ding_force_compensation=(
            args.acados_transfer_ding_force_compensation
        ),
        acados_integrator_type=args.acados_integrator_type,
        acados_collocation_type=args.acados_collocation_type,
        acados_sim_stages=args.acados_sim_stages,
        acados_sim_steps=args.acados_sim_steps,
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
        periodic_fes_warmup_projection_proximity_weight=(
            args.periodic_fes_warmup_projection_proximity_weight
        ),
        periodic_fes_warmup_projection_defect_weight=(
            args.periodic_fes_warmup_projection_defect_weight
        ),
        periodic_fes_warmup_projection_trust_radius=(
            args.periodic_fes_warmup_projection_trust_radius
        ),
        periodic_fes_warmup_projection_max_iterations=(
            args.periodic_fes_warmup_projection_max_iterations
        ),
        periodic_fes_warmup_force_projection_weight=(
            args.periodic_fes_warmup_force_projection_weight
        ),
        periodic_fes_warmup_force_qdot_defect_limit=(
            args.periodic_fes_warmup_force_qdot_defect_limit
        ),
        periodic_fes_warmup_force_adaptive_steps=(
            args.periodic_fes_warmup_force_adaptive_steps
        ),
        acados_diagnostics=args.acados_diagnostics,
        periodic_ipopt_refinement=args.periodic_ipopt_refinement,
        periodic_ipopt_refinement_iterations=args.periodic_ipopt_refinement_iterations,
        periodic_ipopt_refinement_use_sx=args.periodic_ipopt_refinement_use_sx,
        periodic_ipopt_refinement_ode_solver=(
            args.periodic_ipopt_refinement_ode_solver
        ),
        warmup_state_comparison_limit=args.warmup_state_comparison_limit,
        state_comparison_limit=args.state_comparison_limit,
        print_traces=args.print_traces,
        ipopt_profile=args.ipopt_profile,
        ipopt_model_formulation=args.ipopt_model_formulation,
        ipopt_torque_application=args.ipopt_torque_application,
        ipopt_ode_solver=args.ipopt_ode_solver,
        ipopt_rk_steps=args.ipopt_rk_steps,
        ipopt_collocation_degree=args.ipopt_collocation_degree,
        ipopt_collocation_method=args.ipopt_collocation_method,
        ipopt_use_sx=args.ipopt_use_sx,
        ipopt_enforce_start_constraints=args.ipopt_enforce_start_constraints,
        ipopt_disable_standard_warmup=args.ipopt_disable_standard_warmup,
        ipopt_disable_periodic_fes_warmup_projection=(
            args.ipopt_disable_periodic_fes_warmup_projection
        ),
        ipopt_fatigue_warmstart_mode=args.ipopt_fatigue_warmstart_mode,
        ipopt_disable_historical_initial_guess=(
            args.ipopt_disable_historical_initial_guess
        ),
        max_consecutive_failing=args.max_consecutive_failing,
    )
