"""Benchmark IPOPT, ACADOS, MadNLP, and Alpaqa on the cycling FES MHE.

IPOPT, MadNLP, and Alpaqa share the historically robust collocation
transcription and warm start. ACADOS retains its validated solver-specific
periodic formulation. The default objective is fatigue only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
import traceback

import numpy as np
from bioptim import SolutionMerge

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .cycling_pulse_width_mhe_acados_periodic import (
        ACADOS_STATUS_NAMES,
        DEFAULT_CRANK_TORQUE_NM,
        NLP_SOLVER_NAMES,
        build_argument_parser,
        parse_control_homotopy_radii,
        parse_crank_assistance,
        parse_proximal_control_weights,
        parse_terminal_wheel_q_slacks,
        parse_transfer_bound_homotopy_fractions,
        solve_case,
    )
except ImportError:
    from cycling_pulse_width_mhe_acados_periodic import (
        ACADOS_STATUS_NAMES,
        DEFAULT_CRANK_TORQUE_NM,
        NLP_SOLVER_NAMES,
        build_argument_parser,
        parse_control_homotopy_radii,
        parse_crank_assistance,
        parse_proximal_control_weights,
        parse_terminal_wheel_q_slacks,
        parse_transfer_bound_homotopy_fractions,
        solve_case,
    )

EXAMPLE_DIR = Path(__file__).resolve().parent
BENCHMARK_SOLVERS = ("ipopt", "acados", "fatrop", "madnlp")
BENCHMARK_STIMULATION_PATTERN_CYCLES = (1, 2, 3, 4, 5, 10, 30, 100)
BENCHMARK_CONFIGURATION_FIELDS = (
    "solver",
    "benchmark_profile",
    "transcription_profile",
    "ipopt_profile",
    "profile_version",
    "profile_hash",
    "profile_integrity",
    "scientific_status",
    "single_shot",
    "objective",
    "objective_shape",
    "model_formulation",
    "mechanical_formulation",
    "mechanical_equivalence_audit",
    "full_contact_constraints_terminal",
    "full_contact_position_terminal",
    "full_contact_position_tolerance",
    "full_contact_constraints_all_nodes",
    "full_contact_position_all_nodes",
    "transfer_contact_manifold_projection",
    "transfer_contact_manifold_projection_mode",
    "torque_application",
    "ode_solver",
    "collocation_degree",
    "collocation_method",
    "calcium_forcing_formulation",
    "calcium_initialization_regime",
    "calcium_tau_s",
    "calcium_stimulation_interval_s",
    "calcium_post_stimulation_amplitude",
    "calcium_analytical_periodic_value",
    "ding_sum_stim_truncation",
    "ding_states_per_muscle",
    "muscle_names",
    "control_decisions_per_cycle",
    "activate_force_length_relationship",
    "activate_force_velocity_relationship",
    "activate_passive_force_relationship",
    "use_sx",
    "enforce_start_constraints",
    "nlp_ordering_strategy",
    "state_scaling",
    "pulse_width_scaling",
    "pulse_width_active_set",
    "pulse_width_active_threshold",
    "pulse_width_active_margin",
    "wheel_qdot_regularization_target",
    "wheel_qdot_bound_margin",
    "acados_wheel_qdot_fast_bound_margin",
    "acados_wheel_qdot_slow_bound_margin",
    "acados_wheel_q_slack",
    "acados_wheel_qdot_slack",
    "acados_terminal_wheel_q_slack",
    "acados_terminal_wheel_q_target_slack",
    "acados_terminal_wheel_q_homotopy_slacks",
    "acados_terminal_wheel_q_homotopy_each_window",
    "rho_state_continuity_mode",
    "max_acados_iterations",
    "acados_tolerance",
    "acados_stationarity_tolerance",
    "acados_qp_solver",
    "acados_integrator_type",
    "acados_collocation_type",
    "acados_sim_stages",
    "acados_sim_steps",
    "acados_newton_iter",
    "acados_qp_cond_n",
    "acados_qp_warm_start_level",
    "acados_warm_start_first_qp",
    "acados_warm_start_first_qp_from_nlp",
    "acados_hessian_approx",
    "acados_nlp_solver_type",
    "acados_search_direction_mode",
    "acados_dual_warm_start_mode",
    "acados_disable_direction_mode_switch_to_nominal",
    "acados_use_constraint_hessian_in_feas_qp",
    "acados_store_iterates",
    "acados_maxiter_retries",
    "acados_maxiter_retry_iterations",
    "acados_maxiter_retry_feasibility_tolerance",
    "acados_reset_solver_before_solve",
    "acados_check_reuse_possible",
    "acados_code_reuse_tolerance",
    "acados_with_anderson_acceleration",
    "acados_anderson_activation_threshold",
    "acados_byrd_omojokon_slack_relaxation_factor",
    "acados_assisted_hot_start",
    "acados_control_homotopy_radii",
    "acados_control_homotopy_tolerance",
    "acados_control_homotopy_stage_iterations",
    "acados_control_homotopy_max_restarts",
    "acados_control_homotopy_keep_final_radius",
    "acados_control_homotopy_window_growth",
    "acados_control_homotopy_window_max_radius",
    "acados_transfer_irk_rollout",
    "acados_transfer_rollout_max_bound_violation",
    "acados_transfer_phase_one",
    "acados_transfer_bound_homotopy",
    "acados_transfer_bound_homotopy_fractions",
    "acados_transfer_bound_homotopy_padding",
    "acados_transfer_bound_homotopy_iterations",
    "acados_transfer_bound_homotopy_tolerance",
    "acados_transfer_bound_homotopy_solver_tolerance",
    "acados_transfer_bound_homotopy_min_fraction_step",
    "acados_transfer_bound_homotopy_max_refinements",
    "acados_transfer_mechanical_restoration",
    "acados_transfer_sqp_restarts",
    "acados_transfer_active_set_guard_radius",
    "acados_transfer_active_set_guard_margin",
    "acados_transfer_active_set_threshold",
    "acados_cyclical_transfer_mode",
    "terminal_wheel_q_reference_mode",
    "cycles_per_window",
    "stimulations_per_cycle",
    "n_windows",
    "n_threads",
    "constant_crank_torque",
    "crank_torque_role",
    "crank_assistance_nm",
    "expected_external_crank_power_w",
    "max_consecutive_failing",
    "nlp_tolerance",
    "primal_feasibility_threshold",
    "ipopt_linear_solver",
    "warmup_ipopt_linear_solver",
    "standard_warmup_seed",
    "standard_warmup_seed_continuation",
    "legacy_standard_warmup_seed_signed_torque",
    "common_initial_solution",
    "common_initial_solution_output",
    "allow_partial_receding_horizon_solution_output",
    "ipopt_hsl_library",
    "ipopt_c_compile",
    "ipopt_print_level",
    "ipopt_print_timing_statistics",
    "ipopt_linear_system_scaling",
    "ipopt_linear_scaling_on_demand",
    "ipopt_ma57_automatic_scaling",
    "ipopt_ma57_pivot_order",
    "ipopt_ma57_pivtol",
    "ipopt_ma57_pivtolmax",
    "ipopt_ma57_pre_alloc",
    "ipopt_ma57_block_size",
    "ipopt_ma57_node_amalgamation",
    "ipopt_ma57_small_pivot_flag",
    "ipopt_dual_warm_start_mode",
    "max_ipopt_iterations",
    "standard_warmup_max_iterations",
    "disable_historical_ipopt_initial_guess",
    "fatrop_dual_warm_start_mode",
    "fatrop_c_compile",
    "fatrop_structure_detection",
    "fatrop_bound_tightening_factor",
    "fatrop_print_level",
    "max_fatrop_iterations",
    "madnlp_dual_warm_start_mode",
    "madnlp_c_compile",
    "madnlp_linear_solver",
    "max_madnlp_iterations",
    "alpaqa_dual_warm_start_mode",
    "max_alpaqa_iterations",
    "alpaqa_alm_max_iterations",
    "alpaqa_lbfgs_memory",
    "alpaqa_max_wall_time",
    "alpaqa_initial_penalty",
    "alpaqa_initial_tolerance",
    "alpaqa_penalty_update_factor",
    "alpaqa_maximum_penalty",
    "alpaqa_panoc_max_wall_time",
    "alpaqa_max_no_progress",
    "nlp_periodic_ipopt_hot_start",
    "terminal_wheel_regularization_weight",
)

IPOPT_PROFILE_DEFAULTS = {
    "historical": {
        "model_formulation": "standard",
        "torque_application": "external_forces",
        "ode_solver": "collocation",
        "rk_steps": 1,
        "collocation_degree": 3,
        "collocation_method": "radau",
        "use_sx": True,
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
        "use_sx": True,
        "enforce_start_constraints": False,
        "disable_standard_ipopt_warmup": False,
        "disable_periodic_fes_warmup_projection": False,
        "fatigue_warmstart_mode": None,
    },
    "scientific_radau5": {
        "model_formulation": "periodic_node",
        "torque_application": "constant",
        "ode_solver": "collocation",
        "rk_steps": 1,
        "collocation_degree": 5,
        "collocation_method": "radau",
        "use_sx": True,
        "enforce_start_constraints": True,
        "disable_standard_ipopt_warmup": False,
        "disable_periodic_fes_warmup_projection": False,
        "fatigue_warmstart_mode": None,
    },
    "scientific_radau4": {
        "model_formulation": "periodic_node",
        "torque_application": "constant",
        "ode_solver": "collocation",
        "rk_steps": 1,
        "collocation_degree": 4,
        "collocation_method": "radau",
        "use_sx": True,
        "enforce_start_constraints": True,
        "disable_standard_ipopt_warmup": False,
        "disable_periodic_fes_warmup_projection": False,
        "fatigue_warmstart_mode": None,
    },
    "scientific_radau6": {
        "model_formulation": "periodic_node",
        "torque_application": "constant",
        "ode_solver": "collocation",
        "rk_steps": 1,
        "collocation_degree": 6,
        "collocation_method": "radau",
        "use_sx": True,
        "enforce_start_constraints": True,
        "disable_standard_ipopt_warmup": False,
        "disable_periodic_fes_warmup_projection": False,
        "fatigue_warmstart_mode": None,
    },
}
IPOPT_PROFILE_VERSION = 1


def _profile_hash(profile: str) -> str:
    contract = {
        "name": profile,
        "version": IPOPT_PROFILE_VERSION,
        "defaults": IPOPT_PROFILE_DEFAULTS[profile],
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _namespace_from_cli(**overrides) -> argparse.Namespace:
    parser = build_argument_parser()
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def parse_solver_names(raw_solvers: str) -> tuple[str, ...]:
    values = tuple(
        dict.fromkeys(
            item.strip().lower() for item in raw_solvers.split(",") if item.strip()
        )
    )
    invalid = set(values) - set(BENCHMARK_SOLVERS)
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported solver(s): {', '.join(sorted(invalid))}."
        )
    if not values:
        raise argparse.ArgumentTypeError("Select at least one solver.")
    return values


def _format_metric(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _finite_float(value) -> float | None:
    """Return a scalar finite float suitable for console and JSON summaries."""

    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size != 1 or not np.isfinite(array[0]):
        return None
    return float(array[0])


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


def _validated_window_prefix(result: dict) -> int:
    prefix = _successful_prefix_length(result.get("window_statuses"))
    feasibility = result.get("window_feasibility") or []
    for index in range(min(prefix, len(feasibility))):
        if not feasibility[index].get("passes_tolerance", False):
            return index
    return prefix


def _validated_cycle_count(result: dict) -> int:
    prefix = _validated_window_prefix(result)
    if result.get("solver_success"):
        return int(result.get("covered_cycles") or prefix)
    return prefix


def _physically_validated_cycle_count(result: dict) -> int:
    """Return the longest NLP-valid prefix that retains the physical certificate."""

    nlp_validated_cycles = _validated_cycle_count(result)
    if nlp_validated_cycles <= 0:
        return 0

    mechanical_audit = result.get("mechanical_equivalence_audit") or {}
    if (
        mechanical_audit.get("available") is True
        and mechanical_audit.get("passes_tolerance") is False
    ):
        return 0

    diagnostics = result.get("physical_crank_diagnostics")
    if diagnostics is None:
        return 0 if result.get("physical_success") is False else nlp_validated_cycles

    absolute_tolerance = diagnostics.get("absolute_cycle_tolerance")
    progress_tolerance = diagnostics.get("cycle_progress_tolerance")
    if absolute_tolerance is None or progress_tolerance is None:
        return 0 if result.get("physical_success") is False else nlp_validated_cycles

    physical_prefix = 0
    for cycle_count in range(1, nlp_validated_cycles + 1):
        metrics = _cycle_boundary_wheel_angle_metrics(result, cycle_count)
        absolute_error = metrics["maximum_absolute_error_rad"]
        progress_error = metrics["maximum_cycle_progress_error_rad"]
        if (
            absolute_error is None
            or progress_error is None
            or absolute_error > float(absolute_tolerance)
            or progress_error > float(progress_tolerance)
        ):
            break
        physical_prefix = cycle_count
    return physical_prefix


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


def _shooting_node_physical_trace(
    values: np.ndarray, result: dict, cycle_count: int
) -> np.ndarray:
    """Truncate an already projected crank trace without assuming a full tail.

    Mechanical certification is deliberately restricted to the contiguous
    accepted ACADOS prefix.  Consequently, these projected traces can contain
    fewer cycles than the raw state export whenever a later RHO fails.
    Collocation traces still contain their internal points; ACADOS shooting
    traces do not.  The audit records this stride explicitly.
    """

    values = np.asarray(values)
    shooting_per_cycle = int(result["args"].stimulations_per_cycle)
    mechanical_audit = result.get("mechanical_equivalence_audit") or {}
    stride = int(
        mechanical_audit.get(
            "cadence_audit_node_stride", _configured_state_node_stride(result)
        )
    )
    if stride < 1:
        raise ValueError("Physical crank trace stride must be strictly positive.")
    available_cycles, remainder = divmod(
        values.shape[-1] - 1, shooting_per_cycle * stride
    )
    if remainder:
        raise ValueError(
            "Physical crank trace cannot be mapped exactly to complete cycles: "
            f"{values.shape[-1]} values for {shooting_per_cycle} intervals/cycle "
            f"and stride {stride}."
        )
    requested_cycles = min(int(cycle_count), available_cycles)
    return values[..., : requested_cycles * shooting_per_cycle * stride + 1 : stride]


def _truncate_result_to_cycles(result: dict, cycle_count: int) -> dict:
    truncated = {
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
    for key in ("physical_crank_angle_trace", "physical_crank_velocity_trace"):
        if result.get(key) is not None:
            truncated[key] = _shooting_node_physical_trace(
                result[key], result, cycle_count
            )
    return truncated


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

    prefix = _validated_window_prefix(result)
    successful_rows = rows[:prefix]

    def total(key: str) -> float:
        return float(sum(float(row[key] or 0.0) for row in successful_rows))

    nlp_validated_cycles = _validated_cycle_count(result)
    validated_cycles = _physically_validated_cycle_count(result)
    successful_solver_time = total("solver_time_s")
    successful_wall_time = total("wall_time_s")
    hot_rows = successful_rows[1:]

    def hot_stat(key: str, percentile: float) -> float | None:
        values = np.asarray(
            [
                float(row[key])
                for row in hot_rows
                if row[key] is not None and np.isfinite(float(row[key]))
            ],
            dtype=float,
        )
        return float(np.percentile(values, percentile)) if values.size else None

    return {
        "rows": rows,
        "successful_prefix_windows": prefix,
        "nlp_validated_cycles": nlp_validated_cycles,
        "physically_validated_cycles": validated_cycles,
        "validated_cycles": validated_cycles,
        "successful_solver_time_s": successful_solver_time,
        "successful_wall_time_s": successful_wall_time,
        "solver_time_per_cycle_s": (
            successful_solver_time / validated_cycles if validated_cycles else None
        ),
        "wall_time_per_cycle_s": (
            successful_wall_time / validated_cycles if validated_cycles else None
        ),
        "hot_window_count": len(hot_rows),
        "hot_solver_time_median_s": hot_stat("solver_time_s", 50),
        "hot_solver_time_p90_s": hot_stat("solver_time_s", 90),
        "hot_wall_time_median_s": hot_stat("wall_time_s", 50),
        "hot_wall_time_p90_s": hot_stat("wall_time_s", 90),
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
        capacity_scale = result.get("fatigue_capacity_scales", {}).get(key)
        normalization_reference = (
            float(capacity_scale)
            if key.startswith("A_") and capacity_scale is not None
            else initial
        )
        if normalization_reference:
            normalized_fatigue = 1.0 - trace / normalization_reference
            mean_normalized_fatigue = float(np.mean(normalized_fatigue))
            if trace.size > 1:
                dx = float(cycle_count) / (trace.size - 1)
                fatigue_auc_cycles = float(
                    np.sum(
                        0.5 * (normalized_fatigue[:-1] + normalized_fatigue[1:]) * dx
                    )
                )
            else:
                fatigue_auc_cycles = 0.0
        else:
            mean_normalized_fatigue = None
            fatigue_auc_cycles = None
        rows.append(
            {
                "key": key,
                "initial": initial,
                "final": final,
                "relative_final": (
                    final / normalization_reference if normalization_reference else None
                ),
                "minimum": float(np.min(trace)),
                "maximum": float(np.max(trace)),
                "normalization_reference": normalization_reference,
                "normalization_source": (
                    "a_scale"
                    if key.startswith("A_") and capacity_scale is not None
                    else "initial"
                ),
                "mean_normalized_fatigue": mean_normalized_fatigue,
                "fatigue_auc_cycles": fatigue_auc_cycles,
            }
        )
    return rows


def _minimum_a_capacity_ratio(fatigue: list[dict]) -> float | None:
    ratios = [
        row["relative_final"]
        for row in fatigue
        if row["key"].startswith("A_") and row["relative_final"] is not None
    ]
    return min(ratios) if ratios else None


def _executed_fatigue_objective_by_muscle(
    result: dict, cycle_count: int, *, weight: float = 10_000.0
) -> list[dict]:
    """Re-evaluate each muscle's fatigue cost on unique exported cycles.

    Receding-horizon window objectives overlap whenever a horizon contains
    more than one cycle but advances by one cycle. This common trapezoidal
    quadrature instead scores only the states that were actually exported.
    """

    if cycle_count <= 0:
        return []
    limited = _truncate_result_to_cycles(result, cycle_count)
    rows = []
    for key, values in sorted(limited.get("state_traces", {}).items()):
        if not key.startswith("A_"):
            continue
        capacity_scale = result.get("fatigue_capacity_scales", {}).get(key)
        trace = np.asarray(values, dtype=float).reshape(-1)
        if (
            capacity_scale is None
            or not np.isfinite(float(capacity_scale))
            or float(capacity_scale) == 0.0
            or trace.size < 2
            or not np.all(np.isfinite(trace))
        ):
            continue
        normalized_fatigue = 1.0 - trace / float(capacity_scale)
        squared_fatigue = np.square(normalized_fatigue)
        dt = float(cycle_count) / (squared_fatigue.size - 1)
        squared_integral = np.sum(
            0.5 * (squared_fatigue[:-1] + squared_fatigue[1:]) * dt
        )
        fatigue_integral = np.sum(
            0.5 * (normalized_fatigue[:-1] + normalized_fatigue[1:]) * dt
        )
        rows.append(
            {
                "muscle": key.removeprefix("A_"),
                "state_key": key,
                "executed_fatigue_objective": float(weight * squared_integral),
                "cumulative_normalized_fatigue_cycles": float(fatigue_integral),
                "final_capacity_ratio": float(trace[-1] / float(capacity_scale)),
            }
        )
    return rows


def _executed_fatigue_objective(
    result: dict, cycle_count: int, *, weight: float = 10_000.0
) -> float | None:
    """Return the sum of the independently re-evaluated muscle costs."""

    rows = _executed_fatigue_objective_by_muscle(result, cycle_count, weight=weight)
    if not rows:
        return None
    return float(sum(row["executed_fatigue_objective"] for row in rows))


def _external_crank_power_metrics(result: dict, cycle_count: int) -> dict:
    """Report whether the configured crank torque resists or drives the motion.

    Generalized mechanical power is ``tau * qdot``.  In the historical cycling
    data the crank angle decreases, so a negative torque has positive power and
    drives the crank; a positive torque is the genuinely resistive sign.
    """

    metrics = {
        "torque_nm": None,
        "mean_power_w": None,
        "minimum_power_w": None,
        "maximum_power_w": None,
        "role": "unavailable",
    }
    if cycle_count <= 0:
        return metrics

    args = result.get("args")
    torque = getattr(args, "constant_crank_torque", None)
    if torque is None or not np.isfinite(float(torque)):
        return metrics
    metrics["torque_nm"] = float(torque)

    limited = _truncate_result_to_cycles(result, cycle_count)
    state_traces = limited.get("state_traces", {})
    qdot = state_traces.get("qdot")
    velocity_index = 2
    if qdot is None:
        qdot = state_traces.get("omega")
        velocity_index = 0
    if qdot is None:
        return metrics
    qdot = np.asarray(qdot, dtype=float)
    if qdot.ndim != 2 or qdot.shape[0] <= velocity_index or qdot.shape[1] == 0:
        return metrics

    power = float(torque) * qdot[velocity_index, :]
    if not np.all(np.isfinite(power)):
        return metrics
    if power.size == 1:
        mean_power = float(power[0])
    else:
        mean_power = float(np.sum(0.5 * (power[:-1] + power[1:])) / (power.size - 1))

    power_tolerance = 1e-12
    role = (
        "driving"
        if mean_power > power_tolerance
        else "resistive"
        if mean_power < -power_tolerance
        else "neutral"
    )
    metrics.update(
        mean_power_w=mean_power,
        minimum_power_w=float(np.min(power)),
        maximum_power_w=float(np.max(power)),
        role=role,
    )
    return metrics


def _cycle_boundary_wheel_angle_metrics(result: dict, cycle_count: int) -> dict:
    """Measure completed-turn accuracy on the unique executed trajectory."""

    metrics = {
        "signed_cycle_shift_rad": None,
        "absolute_reference_rad": None,
        "maximum_absolute_error_rad": None,
        "final_error_rad": None,
        "errors_rad": [],
        "maximum_cycle_progress_error_rad": None,
        "cycle_progress_errors_rad": [],
    }
    if cycle_count <= 0:
        return metrics

    limited = _truncate_result_to_cycles(result, cycle_count)
    wheel_angle = np.asarray(
        limited.get("physical_crank_angle_trace")
        if limited.get("physical_crank_angle_trace") is not None
        else limited.get("wheel_angle_trace", []),
        dtype=float,
    ).reshape(-1)
    shooting_per_cycle = int(result["args"].stimulations_per_cycle)
    expected_size = cycle_count * shooting_per_cycle + 1
    if (
        wheel_angle.size != expected_size
        or not np.all(np.isfinite(wheel_angle))
        or np.isclose(wheel_angle[-1], wheel_angle[0])
    ):
        return metrics

    cycle_shift = float(np.sign(wheel_angle[-1] - wheel_angle[0]) * 2.0 * np.pi)
    boundaries = wheel_angle[::shooting_per_cycle]
    reference = result.get("physical_crank_absolute_reference")
    if reference is None:
        reference = result.get("absolute_wheel_q_reference")
    if reference is None:
        reference = wheel_angle[0]
    reference = float(reference)
    expected = reference + cycle_shift * np.arange(cycle_count + 1)
    errors = boundaries - expected
    progress_errors = np.diff(boundaries) - cycle_shift
    metrics.update(
        signed_cycle_shift_rad=cycle_shift,
        absolute_reference_rad=reference,
        maximum_absolute_error_rad=float(np.max(np.abs(errors))),
        final_error_rad=float(errors[-1]),
        errors_rad=errors.tolist(),
        maximum_cycle_progress_error_rad=float(np.max(np.abs(progress_errors))),
        cycle_progress_errors_rad=progress_errors.tolist(),
    )
    return metrics


def _format_a_capacity_by_muscle(fatigue: list[dict]) -> str:
    rows = [
        f"{row['key'].removeprefix('A_')}={row['relative_final']:.6f}"
        for row in fatigue
        if row["key"].startswith("A_") and row["relative_final"] is not None
    ]
    return ",".join(rows) if rows else "None"


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
            metric_key = key if reference_values.shape[0] == 1 else f"{key}[{row}]"
            if metric_key == "q[2]":
                reference_common = np.unwrap(reference_common)
                compared_common = np.unwrap(compared_common)
                turn_offset = int(
                    np.rint(
                        np.median((compared_common - reference_common) / (2 * np.pi))
                    )
                )
                compared_common = compared_common - turn_offset * 2 * np.pi
            diff = compared_common - reference_common
            comparisons.append(
                {
                    "key": metric_key,
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


def _print_solution_metrics(
    ipopt_result: dict,
    acados_result: dict,
    *,
    line_prefix: str = "",
    state_comparison_limit: int = 12,
) -> None:
    trace_metrics = _trace_comparison(ipopt_result, acados_result)
    print(
        f"{line_prefix}wheel trace comparison | "
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
        f"{line_prefix}raw wheel final-angle representation | "
        f"raw_final_error={trace_metrics['raw_final_error']:.6f} | "
        f"raw_final_turn_offset={trace_metrics['raw_final_turn_offset']}"
    )

    control_metrics = _control_comparisons(ipopt_result, acados_result)
    if control_metrics:
        print(
            f"{line_prefix}control comparison | key | common_len | rmse | mae | max_abs_error | final_error | "
            "ipopt_mean | acados_mean | ipopt_sum | acados_sum | ipopt_range | acados_range"
        )
        for metric in control_metrics:
            print(
                f"{line_prefix}control comparison | "
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
        print(
            f"{line_prefix}control comparison warning: no common control keys were found."
        )

    state_metrics = _state_comparisons(ipopt_result, acados_result)
    if state_metrics and state_comparison_limit:
        print(
            f"{line_prefix}state comparison | key | common_len | rmse | mae | max_abs_error | final_error | "
            "ipopt_mean | acados_mean | ipopt_range | acados_range"
        )
        for metric in state_metrics[:state_comparison_limit]:
            ipopt_min, ipopt_max = metric["ipopt_range"]
            acados_min, acados_max = metric["acados_range"]
            print(
                f"{line_prefix}state comparison | "
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
        print(
            f"{line_prefix}state comparison warning: no common state keys were found."
        )


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
                    max_error,
                    float(np.max(np.abs(ipopt_array - acados_array))),
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
    single_shot: bool,
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
    acados_assisted_hot_start: bool,
    acados_control_homotopy_radii: tuple[float, ...] | None,
    acados_control_homotopy_tolerance: float,
    acados_control_homotopy_stage_iterations: int,
    acados_control_homotopy_max_restarts: int,
    control_regularization_weight: float,
    control_regularization_target: float | None,
    control_regularization_target_source: str,
    wheel_qdot_regularization_weight: float,
    wheel_qdot_regularization_target: float,
    wheel_qdot_bound_margin: float,
    acados_wheel_qdot_fast_bound_margin: float | None,
    acados_wheel_qdot_slow_bound_margin: float | None,
    terminal_qdot_regularization_weight: float,
    terminal_qdot_regularization_target_source: str,
    first_node_wheel_q_slack: float,
    acados_terminal_wheel_q_slack: float,
    state_scaling: str,
    pulse_width_scaling: float,
    pulse_width_active_set: str,
    pulse_width_active_threshold: float,
    pulse_width_active_margin: int,
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
                "'scientific_radau4', 'scientific_radau5', "
                "'scientific_radau6', or 'acados_like'."
            )

        defaults = IPOPT_PROFILE_DEFAULTS[normalized_profile]
        disable_projection_default = defaults["disable_periodic_fes_warmup_projection"]
        if disable_projection_default is None:
            disable_projection_default = disable_periodic_fes_warmup_projection

        fatigue_warmstart_default = defaults["fatigue_warmstart_mode"]
        if fatigue_warmstart_default is None:
            fatigue_warmstart_default = acados_fatigue_warmstart_mode

        solver_args = _namespace_from_cli(
            solver="ipopt",
            benchmark_profile=normalized_profile.replace("_", "-"),
            ipopt_profile=normalized_profile,
            transcription_profile=normalized_profile.replace("_", "-"),
            profile_version=IPOPT_PROFILE_VERSION,
            profile_hash=_profile_hash(normalized_profile),
            scientific_status=(
                "candidate"
                if normalized_profile == "scientific_radau5"
                else "diagnostic"
            ),
            single_shot=single_shot,
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
            acados_wheel_qdot_fast_bound_margin=(
                acados_wheel_qdot_fast_bound_margin
            ),
            acados_wheel_qdot_slow_bound_margin=(
                acados_wheel_qdot_slow_bound_margin
            ),
            terminal_qdot_regularization_weight=(terminal_qdot_regularization_weight),
            terminal_qdot_regularization_target_source=(
                terminal_qdot_regularization_target_source
            ),
            acados_wheel_q_slack=first_node_wheel_q_slack,
            acados_terminal_wheel_q_slack=acados_terminal_wheel_q_slack,
            state_scaling=state_scaling,
            pulse_width_scaling=pulse_width_scaling,
            pulse_width_active_set=pulse_width_active_set,
            pulse_width_active_threshold=pulse_width_active_threshold,
            pulse_width_active_margin=pulse_width_active_margin,
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
            periodic_ipopt_refinement_use_sx=True,
            warmup_state_comparison_limit=warmup_state_comparison_limit,
            disable_historical_ipopt_initial_guess=(
                ipopt_disable_historical_initial_guess
            ),
        )
        contract_fields = (
            ("model_formulation", "model_formulation"),
            ("torque_application", "torque_application"),
            ("ode_solver", "ode_solver"),
            ("rk_steps", "rk_steps"),
            ("collocation_degree", "collocation_degree"),
            ("collocation_method", "collocation_method"),
            ("use_sx", "use_sx"),
            ("enforce_start_constraints", "enforce_start_constraints"),
            ("disable_standard_ipopt_warmup", "disable_standard_ipopt_warmup"),
            (
                "disable_periodic_fes_warmup_projection",
                "disable_periodic_fes_warmup_projection",
            ),
            ("acados_fatigue_warmstart_mode", "fatigue_warmstart_mode"),
        )
        mismatches = {
            attribute: (getattr(solver_args, attribute), defaults[default_key])
            for attribute, default_key in contract_fields
            if defaults[default_key] is not None
            and getattr(solver_args, attribute) != defaults[default_key]
        }
        solver_args.profile_integrity = not mismatches
        if normalized_profile.startswith("scientific_radau") and mismatches:
            details = ", ".join(
                f"{key}={observed!r} (expected {expected!r})"
                for key, (observed, expected) in mismatches.items()
            )
            raise ValueError(
                f"The {normalized_profile.replace('_', '-')} profile is a fixed scientific "
                f"contract and cannot be overridden: {details}."
            )
        return solver_args

    if solver_name == "acados":
        return _namespace_from_cli(
            solver="acados",
            single_shot=single_shot,
            model_formulation="periodic_node",
            torque_application="constant",
            cycles_per_window=cycles_per_window,
            stimulations_per_cycle=stimulations_per_cycle,
            objective=objective,
            objective_shape=objective_shape,
            constant_crank_torque=resistive_torque,
            max_acados_iterations=acados_max_iter,
            acados_assisted_hot_start=acados_assisted_hot_start,
            acados_control_homotopy_radii=acados_control_homotopy_radii,
            acados_control_homotopy_tolerance=acados_control_homotopy_tolerance,
            acados_control_homotopy_stage_iterations=(
                acados_control_homotopy_stage_iterations
            ),
            acados_control_homotopy_max_restarts=(acados_control_homotopy_max_restarts),
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
            acados_wheel_qdot_fast_bound_margin=(
                acados_wheel_qdot_fast_bound_margin
            ),
            acados_wheel_qdot_slow_bound_margin=(
                acados_wheel_qdot_slow_bound_margin
            ),
            terminal_qdot_regularization_weight=(terminal_qdot_regularization_weight),
            terminal_qdot_regularization_target_source=(
                terminal_qdot_regularization_target_source
            ),
            acados_wheel_q_slack=first_node_wheel_q_slack,
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


def _nlp_solver_config(
    solver_name: str,
    reference_args: argparse.Namespace,
    *,
    tolerance: float,
    max_iterations: int,
    dual_warm_start_mode: str,
    alpaqa_lbfgs_memory: int = 20,
    alpaqa_max_wall_time: float | None = None,
    alpaqa_initial_penalty: float | None = None,
    alpaqa_alm_max_iterations: int | None = None,
    madnlp_linear_solver: str | None = None,
    alpaqa_initial_tolerance: float | None = None,
    alpaqa_penalty_update_factor: float | None = None,
    alpaqa_maximum_penalty: float | None = None,
    alpaqa_panoc_max_wall_time: float | None = None,
    alpaqa_max_no_progress: int | None = None,
    fatrop_c_compile: bool = False,
    fatrop_structure_detection: str = "auto",
    fatrop_bound_tightening_factor: float = 1e-8,
    fatrop_print_level: int = 0,
    periodic_ipopt_hot_start: bool = False,
) -> argparse.Namespace:
    """Clone the IPOPT transcription so only the nonlinear solver changes."""

    if solver_name not in {"fatrop", "madnlp", "alpaqa"}:
        raise ValueError(
            "The optional NLP solver must be 'fatrop', 'madnlp' or 'alpaqa'."
        )
    args = argparse.Namespace(**vars(reference_args))
    args.solver = solver_name
    args.nlp_tolerance = tolerance
    setattr(args, f"max_{solver_name}_iterations", max_iterations)
    setattr(args, f"{solver_name}_dual_warm_start_mode", dual_warm_start_mode)
    args.nlp_periodic_ipopt_hot_start = periodic_ipopt_hot_start
    if solver_name == "fatrop":
        args.fatrop_c_compile = fatrop_c_compile
        args.fatrop_structure_detection = fatrop_structure_detection
        args.fatrop_bound_tightening_factor = fatrop_bound_tightening_factor
        args.fatrop_print_level = fatrop_print_level
    if solver_name == "madnlp":
        args.madnlp_linear_solver = madnlp_linear_solver
    if solver_name == "alpaqa":
        args.alpaqa_lbfgs_memory = alpaqa_lbfgs_memory
        args.alpaqa_alm_max_iterations = alpaqa_alm_max_iterations
        args.alpaqa_max_wall_time = alpaqa_max_wall_time
        args.alpaqa_initial_penalty = alpaqa_initial_penalty
        args.alpaqa_initial_tolerance = alpaqa_initial_tolerance
        args.alpaqa_penalty_update_factor = alpaqa_penalty_update_factor
        args.alpaqa_maximum_penalty = alpaqa_maximum_penalty
        args.alpaqa_panoc_max_wall_time = alpaqa_panoc_max_wall_time
        args.alpaqa_max_no_progress = alpaqa_max_no_progress
    return args


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
    _print_solution_metrics(
        comparison_ipopt,
        comparison_acados,
        state_comparison_limit=state_comparison_limit,
    )

    common_exported_cycles = min(
        _exported_cycle_count(ipopt_result),
        _exported_cycle_count(acados_result),
    )
    if common_exported_cycles > common_validated_cycles:
        print(
            "exported diagnostic warning: includes windows without a successful solver status; "
            "use these metrics to inspect trajectories, not as a convergence certificate."
        )
        print(f"exported diagnostic cycles: {common_exported_cycles}")
        exported_ipopt = _truncate_result_to_cycles(
            ipopt_result, common_exported_cycles
        )
        exported_acados = _truncate_result_to_cycles(
            acados_result, common_exported_cycles
        )
        _print_solution_metrics(
            exported_ipopt,
            exported_acados,
            line_prefix="exported diagnostic ",
            state_comparison_limit=state_comparison_limit,
        )

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
        validated_fatigue = _fatigue_metrics(result, validated_cycles)
        saturation = _control_saturation_metrics(result, validated_cycles)
        print(
            f"{label} validated endurance: "
            f"stop={_stop_classification(result)['label']} "
            f"cycles={validated_cycles} "
            "min_A_capacity_ratio="
            f"{_format_metric(_minimum_a_capacity_ratio(validated_fatigue))} "
            f"A_capacity_by_muscle={_format_a_capacity_by_muscle(validated_fatigue)} "
            "max_pulse_width_upper_fraction="
            f"{_format_metric(max((row['upper_fraction'] for row in saturation), default=None))}"
        )
        exported_cycles = _exported_cycle_count(result)
        if exported_cycles > validated_cycles:
            exported_fatigue = _fatigue_metrics(result, exported_cycles)
            print(
                f"{label} exported diagnostic endurance: "
                f"cycles={exported_cycles} "
                "min_A_capacity_ratio="
                f"{_format_metric(_minimum_a_capacity_ratio(exported_fatigue))} "
                f"A_capacity_by_muscle={_format_a_capacity_by_muscle(exported_fatigue)}"
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


def _failed_solver_result(
    args: argparse.Namespace, error: Exception, wall_time_s: float
) -> dict:
    """Return a benchmark-shaped failure so one missing plugin does not abort the matrix."""

    return {
        "args": args,
        "success": False,
        "solver_success": False,
        "physical_success": False,
        "status": None,
        "objective": None,
        "solver_time_s": 0.0,
        "wall_time_s": wall_time_s,
        "end_to_end_wall_time_s": wall_time_s,
        "final_wheel_angle": None,
        "requested_windows": args.n_windows,
        "attempted_windows": 0,
        "successful_windows": 0,
        "exported_cycles": 0,
        "covered_cycles": 0,
        "wheel_angle_trace": np.array([], dtype=float),
        "state_traces": {},
        "state_boundary_jumps": {
            "available": False,
            "boundary_count": 0,
            "by_state": {},
        },
        "control_traces": {},
        "control_bounds": {},
        "window_statuses": [],
        "window_solutions": [],
        "diagnostics": {
            "is_physical": False,
            "issues": ["solver_unavailable_or_construction_failed"],
        },
        "error": f"{type(error).__name__}: {error}",
    }


def _run_benchmark_case(
    solver_name: str,
    args: argparse.Namespace,
    *,
    echo: bool = True,
    print_traceback: bool = False,
) -> dict:
    label = solver_name.upper()
    print(f"Running {label} configuration...")
    start = perf_counter()
    try:
        uses_c_codegen = (
            (solver_name == "ipopt" and getattr(args, "ipopt_c_compile", False))
            or (solver_name == "fatrop" and getattr(args, "fatrop_c_compile", False))
            or (solver_name == "madnlp" and getattr(args, "madnlp_c_compile", False))
        )
        if uses_c_codegen:
            previous_cwd = Path.cwd()
            # CasADi emits the fixed filenames nlp.c/nlp.so in the current
            # directory. Keep generated files isolated without changing the
            # meaning of user-provided relative inputs or outputs.
            for path_attribute in (
                "standard_warmup_seed",
                "common_initial_solution",
                "common_initial_solution_output",
                "receding_horizon_solution_output",
                "reduced_cycling_profile",
            ):
                path_value = getattr(args, path_attribute, None)
                if path_value is None:
                    continue
                if not isinstance(path_value, (str, os.PathLike)):
                    raise TypeError(
                        f"{path_attribute} must be a filesystem path, got "
                        f"{type(path_value).__name__}."
                    )
                path_value = Path(path_value).expanduser()
                if not path_value.is_absolute():
                    setattr(args, path_attribute, previous_cwd / path_value)
            with TemporaryDirectory(
                prefix=f"cocofest_{solver_name}_codegen_"
            ) as codegen_dir:
                try:
                    os.chdir(codegen_dir)
                    result = solve_case(args, echo=echo)
                finally:
                    os.chdir(previous_cwd)
        else:
            result = solve_case(args, echo=echo)
    except (AttributeError, ImportError, RuntimeError) as error:
        if print_traceback:
            traceback.print_exc()
        result = _failed_solver_result(args, error, perf_counter() - start)
        print(f"{label} unavailable or failed during setup: {result['error']}")
        return result
    result["end_to_end_wall_time_s"] = perf_counter() - start
    return result


def _benchmark_window_rows(result: dict) -> list[dict]:
    """Return per-window evidence used to validate timing and warm starts."""

    solutions = result.get("window_solutions") or []
    statuses = result.get("window_statuses") or []
    iterations = result.get("window_iterations") or []
    objectives = result.get("window_objectives") or []
    feasibility = result.get("window_feasibility") or []
    stats_by_window = {
        int(stats["window"]): stats
        for stats in (result.get("nlp_solver_stats") or [])
        if stats.get("window") is not None
    }
    count = max(
        len(solutions),
        len(statuses),
        len(iterations),
        len(objectives),
        len(feasibility),
    )
    validated_prefix = _validated_window_prefix(result)
    rows = []
    for index in range(count):
        solution = solutions[index] if index < len(solutions) else None
        status = (
            int(statuses[index])
            if index < len(statuses) and statuses[index] is not None
            else None
        )
        window_feasibility = feasibility[index] if index < len(feasibility) else None
        solver_stats = stats_by_window.get(index) or {}
        madnlp_stats = solver_stats.get("madnlp") or {}
        native_status = next(
            (
                value
                for value in (
                    solver_stats.get("unified_return_status"),
                    solver_stats.get("return_status"),
                    madnlp_stats.get("status"),
                )
                if value is not None
            ),
            None,
        )
        if native_status is None and index == count - 1:
            native_status = result.get("native_solver_status")
        rows.append(
            {
                "window": index,
                "rho": index + 1,
                "status": status,
                "native_status": (
                    str(native_status) if native_status is not None else None
                ),
                "solver_converged": status == 0,
                "primal_feasible": (
                    bool(window_feasibility.get("passes_tolerance", False))
                    if window_feasibility is not None
                    else None
                ),
                "validated": index < validated_prefix,
                "iterations": (iterations[index] if index < len(iterations) else None),
                "objective": (
                    _finite_float(objectives[index])
                    if index < len(objectives)
                    else None
                ),
                "solver_time_s": _finite_float(
                    getattr(solution, "solver_time_to_optimize", None)
                ),
                "wall_time_s": _finite_float(
                    getattr(solution, "real_time_to_optimize", None)
                ),
                "feasibility": window_feasibility,
            }
        )
    return rows


def _stimulation_pattern_snapshot(result: dict, cycle: int) -> dict:
    """Extract the executed pulse-width pattern of one validated one-cycle RHO."""

    validated_cycles = _physically_validated_cycle_count(result)
    cycles_per_window = int(getattr(result["args"], "cycles_per_window", 1))
    snapshot = {
        "cycle": int(cycle),
        "rho": int(cycle) if cycles_per_window == 1 else None,
        "rho_equals_cycle": cycles_per_window == 1,
        "available": False,
        "reason": None,
        "stimulations_per_cycle": int(result["args"].stimulations_per_cycle),
        "phase_fraction": [],
        "crank_angle_rad": [],
        "crank_phase_rad": [],
        "crank_velocity_rad_s": [],
        "muscles": {},
    }
    if cycle < 1:
        snapshot["reason"] = "cycle_must_be_positive"
        return snapshot
    if cycle > validated_cycles:
        snapshot[
            "reason"
        ] = f"only_{validated_cycles}_cycles_belong_to_the_converged_prefix"
        return snapshot

    shooting_per_cycle = snapshot["stimulations_per_cycle"]
    limited = _truncate_result_to_cycles(result, cycle)
    controls = limited.get("control_traces", {})
    start = (cycle - 1) * shooting_per_cycle
    stop = cycle * shooting_per_cycle
    wheel_angle = np.asarray(
        limited.get("physical_crank_angle_trace")
        if limited.get("physical_crank_angle_trace") is not None
        else limited.get("wheel_angle_trace", []),
        dtype=float,
    ).reshape(-1)
    if wheel_angle.size < stop + 1 or not np.all(
        np.isfinite(wheel_angle[start : stop + 1])
    ):
        snapshot["reason"] = "invalid_crank_angle_trace"
        return snapshot
    cycle_wheel_angle = wheel_angle[start:stop]
    signed_cycle_progress = wheel_angle[stop] - wheel_angle[start]
    crank_direction = np.sign(signed_cycle_progress)
    if crank_direction == 0:
        snapshot["reason"] = "zero_crank_progress"
        return snapshot
    snapshot["crank_angle_rad"] = cycle_wheel_angle.tolist()
    snapshot["crank_phase_rad"] = np.mod(
        crank_direction * (cycle_wheel_angle - wheel_angle[start]),
        2.0 * np.pi,
    ).tolist()
    state_traces = limited.get("state_traces", {})
    qdot = limited.get("physical_crank_velocity_trace")
    velocity_index = 0
    if qdot is None:
        qdot = state_traces.get("qdot")
        velocity_index = 2
    if qdot is None:
        qdot = state_traces.get("omega")
        velocity_index = 0
    if qdot is not None:
        qdot = np.asarray(qdot, dtype=float)
        if qdot.ndim == 1 and qdot.size >= stop:
            crank_velocity = qdot[start:stop]
            if np.all(np.isfinite(crank_velocity)):
                snapshot["crank_velocity_rad_s"] = crank_velocity.tolist()
        elif (
            qdot.ndim == 2 and qdot.shape[0] > velocity_index and qdot.shape[1] >= stop
        ):
            crank_velocity = qdot[velocity_index, start:stop]
            if np.all(np.isfinite(crank_velocity)):
                snapshot["crank_velocity_rad_s"] = crank_velocity.tolist()

    bounds = result.get("control_bounds", {})
    for key, values in sorted(controls.items()):
        if not key.startswith("last_pulse_width_"):
            continue
        trace = np.asarray(values, dtype=float).reshape(-1)
        cycle_values = trace[start:stop]
        if cycle_values.size != shooting_per_cycle or not np.all(
            np.isfinite(cycle_values)
        ):
            snapshot["reason"] = f"invalid_control_trace_for_{key}"
            snapshot["muscles"] = {}
            return snapshot
        control_bounds = bounds.get(key) or {}
        lower = _finite_float(control_bounds.get("lower"))
        upper = _finite_float(control_bounds.get("upper"))
        if lower is not None and upper is not None and upper > lower:
            normalized = (cycle_values - lower) / (upper - lower)
            bound_tolerance = max(1e-12, (upper - lower) * 1e-3)
            lower_fraction = float(np.mean(cycle_values <= lower + bound_tolerance))
            upper_fraction = float(np.mean(cycle_values >= upper - bound_tolerance))
        else:
            normalized = np.full(cycle_values.shape, np.nan)
            lower_fraction = None
            upper_fraction = None
        maximum_index = int(np.argmax(cycle_values))
        snapshot["muscles"][key.removeprefix("last_pulse_width_")] = {
            "control_key": key,
            "pulse_width_s": cycle_values.tolist(),
            "pulse_width_us": (1e6 * cycle_values).tolist(),
            "normalized_to_bounds": [
                float(value) if np.isfinite(value) else None for value in normalized
            ],
            "minimum_s": float(np.min(cycle_values)),
            "mean_s": float(np.mean(cycle_values)),
            "maximum_s": float(np.max(cycle_values)),
            "lower_bound_s": lower,
            "upper_bound_s": upper,
            "lower_bound_fraction": lower_fraction,
            "upper_bound_fraction": upper_fraction,
            "maximum_phase_rad": snapshot["crank_phase_rad"][maximum_index],
        }

    if not snapshot["muscles"]:
        snapshot["reason"] = "no_pulse_width_controls"
        return snapshot
    snapshot["available"] = True
    snapshot["phase_fraction"] = (
        np.arange(shooting_per_cycle, dtype=float) / shooting_per_cycle
    ).tolist()
    return snapshot


def stimulation_pattern_snapshots(
    result: dict,
    cycles: tuple[int, ...] = BENCHMARK_STIMULATION_PATTERN_CYCLES,
) -> dict[str, dict]:
    """Return JSON-safe stimulation evidence at selected one-cycle RHO indices."""

    return {
        f"cycle_{cycle}": _stimulation_pattern_snapshot(result, cycle)
        for cycle in cycles
    }


def isolated_window_checkpoint_snapshots(
    result: dict,
    cycles: tuple[int, ...] = BENCHMARK_STIMULATION_PATTERN_CYCLES,
) -> dict[str, dict]:
    """Preserve selected RHO decisions even after the strict prefix stops.

    These checkpoints are deliberately separate from ``stimulation_patterns``:
    they are numerical diagnostics, not an executed endurance trajectory.
    Keeping their terminal A states and PW vectors makes the 100th isolated
    full solve auditable without accidentally certifying all preceding seams.
    """

    cycles_per_window = int(getattr(result["args"], "cycles_per_window", 1))
    windows = result.get("window_solutions") or []
    objectives = result.get("window_objectives") or []
    feasibility = result.get("window_feasibility") or []
    strict_prefix = _physically_validated_cycle_count(result)
    capacity_scales = result.get("fatigue_capacity_scales") or {}
    checkpoint_cycles = list(cycles)
    if cycles_per_window == 1:
        for window_index, solution in enumerate(windows):
            feasibility_passes = (
                window_index < len(feasibility)
                and bool(feasibility[window_index].get("passes_tolerance", False))
            )
            if getattr(solution, "status", None) == 0 and feasibility_passes:
                continue
            failed_cycle = window_index + 1
            checkpoint_cycles.extend(
                cycle
                for cycle in (failed_cycle - 1, failed_cycle, failed_cycle + 1)
                if 1 <= cycle <= len(windows)
            )
            break
    checkpoint_cycles = tuple(dict.fromkeys(checkpoint_cycles))
    control_bounds = result.get("control_bounds") or {}
    snapshots = {}
    for cycle in checkpoint_cycles:
        snapshot = {
            "cycle": int(cycle),
            "rho": int(cycle) if cycles_per_window == 1 else None,
            "available": False,
            "diagnostic_only": True,
            "belongs_to_strict_prefix": bool(cycle <= strict_prefix),
            "reason": None,
            "status": None,
            "objective": None,
            "primal_feasible": None,
            "capacity_states": {},
            "pulse_width_us": {},
            "pulse_width_active_set": {},
        }
        snapshots[f"cycle_{cycle}"] = snapshot
        if cycles_per_window != 1:
            snapshot["reason"] = "requires_one_cycle_per_window"
            continue
        if cycle < 1:
            snapshot["reason"] = "cycle_must_be_positive"
            continue
        if cycle > len(windows):
            snapshot["reason"] = f"only_{len(windows)}_windows_were_attempted"
            continue

        window_index = cycle - 1
        solution = windows[window_index]
        snapshot["status"] = getattr(solution, "status", None)
        if window_index < len(objectives):
            snapshot["objective"] = _finite_float(objectives[window_index])
        if window_index < len(feasibility):
            snapshot["primal_feasible"] = bool(
                feasibility[window_index].get("passes_tolerance", False)
            )
        try:
            states = solution.decision_states(to_merge=SolutionMerge.NODES)
            controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
        except (AttributeError, KeyError, RuntimeError, ValueError) as error:
            snapshot["reason"] = f"decision_extraction_failed:{type(error).__name__}"
            continue

        for key, values in sorted(states.items()):
            if not key.startswith("A_"):
                continue
            trace = np.asarray(values, dtype=float).reshape(-1)
            if trace.size == 0 or not np.all(np.isfinite(trace)):
                continue
            scale = _finite_float(capacity_scales.get(key))
            snapshot["capacity_states"][key] = {
                "initial": float(trace[0]),
                "terminal": float(trace[-1]),
                "scale": scale,
                "initial_ratio": (
                    None if scale in (None, 0.0) else float(trace[0] / scale)
                ),
                "terminal_ratio": (
                    None if scale in (None, 0.0) else float(trace[-1] / scale)
                ),
            }
        for key, values in sorted(controls.items()):
            if not key.startswith("last_pulse_width_"):
                continue
            trace = np.asarray(values, dtype=float).reshape(-1)
            if trace.size == 0 or not np.all(np.isfinite(trace)):
                continue
            muscle = key.removeprefix("last_pulse_width_")
            snapshot["pulse_width_us"][muscle] = (1e6 * trace).tolist()
            bounds = control_bounds.get(key) or {}
            lower = _finite_float(bounds.get("lower"))
            upper = _finite_float(bounds.get("upper"))
            if lower is None or upper is None or upper <= lower:
                continue
            tolerance = max(1e-12, (upper - lower) * 1e-3)
            snapshot["pulse_width_active_set"][muscle] = {
                "lower_bound_us": 1e6 * lower,
                "upper_bound_us": 1e6 * upper,
                "classification_tolerance_us": 1e6 * tolerance,
                "lower_active_indices": np.flatnonzero(
                    trace <= lower + tolerance
                ).astype(int).tolist(),
                "upper_active_indices": np.flatnonzero(
                    trace >= upper - tolerance
                ).astype(int).tolist(),
            }
        snapshot["available"] = bool(
            snapshot["capacity_states"] or snapshot["pulse_width_us"]
        )
        if not snapshot["available"]:
            snapshot["reason"] = "no_capacity_or_pulse_width_decisions"
    return snapshots


def pulse_width_cycle_variation(result: dict, cycle_count: int) -> dict:
    """Measure aligned pulse-width changes between consecutive executed cycles.

    These percentiles are observations, not proposed hard bounds. An ACADOS
    trust region must retain a safety margin and an out-of-distribution recovery
    path.
    """

    summary = {
        "available": False,
        "reason": None,
        "cycle_count": int(cycle_count),
        "transition_count": max(0, int(cycle_count) - 1),
        "stimulations_per_cycle": int(result["args"].stimulations_per_cycle),
        "muscles": [],
        "pooled_absolute_change_us": {},
    }
    if cycle_count < 2:
        summary["reason"] = "at_least_two_validated_cycles_are_required"
        return summary

    shooting_per_cycle = summary["stimulations_per_cycle"]
    limited = _truncate_result_to_cycles(result, cycle_count)
    pooled_changes = []
    for key, values in sorted(limited.get("control_traces", {}).items()):
        if not key.startswith("last_pulse_width_"):
            continue
        trace = np.asarray(values, dtype=float).reshape(-1)
        expected = cycle_count * shooting_per_cycle
        if trace.size < expected or not np.all(np.isfinite(trace[:expected])):
            summary["reason"] = f"invalid_control_trace_for_{key}"
            summary["muscles"] = []
            return summary
        cycles = trace[:expected].reshape(cycle_count, shooting_per_cycle)
        changes_us = 1e6 * np.diff(cycles, axis=0)
        absolute_us = np.abs(changes_us)
        pooled_changes.append(absolute_us.reshape(-1))
        transitions = []
        for index, transition in enumerate(changes_us):
            transitions.append(
                {
                    "from_cycle": index + 1,
                    "to_cycle": index + 2,
                    "mean_absolute_change_us": float(np.mean(np.abs(transition))),
                    "root_mean_square_change_us": float(
                        np.sqrt(np.mean(np.square(transition)))
                    ),
                    "maximum_absolute_change_us": float(np.max(np.abs(transition))),
                }
            )
        summary["muscles"].append(
            {
                "muscle": key.removeprefix("last_pulse_width_"),
                "control_key": key,
                "sample_count": int(absolute_us.size),
                "mean_absolute_change_us": float(np.mean(absolute_us)),
                "median_absolute_change_us": float(np.median(absolute_us)),
                "p90_absolute_change_us": float(np.percentile(absolute_us, 90)),
                "p95_absolute_change_us": float(np.percentile(absolute_us, 95)),
                "p99_absolute_change_us": float(np.percentile(absolute_us, 99)),
                "maximum_absolute_change_us": float(np.max(absolute_us)),
                "transitions": transitions,
            }
        )

    if not pooled_changes:
        summary["reason"] = "no_pulse_width_controls"
        return summary
    pooled = np.concatenate(pooled_changes)
    summary["pooled_absolute_change_us"] = {
        "sample_count": int(pooled.size),
        "mean": float(np.mean(pooled)),
        "median": float(np.median(pooled)),
        "p90": float(np.percentile(pooled, 90)),
        "p95": float(np.percentile(pooled, 95)),
        "p99": float(np.percentile(pooled, 99)),
        "maximum": float(np.max(pooled)),
    }
    summary["available"] = True
    return summary


def acados_transfer_restoration_timing(result: dict) -> dict:
    """Attribute inter-window feasibility-restoration solves to the next RHO."""

    by_rho: dict[int, float] = {}
    stages = []
    for homotopy in result.get("transfer_bound_homotopy_summaries") or []:
        source_rho = homotopy.get("window")
        try:
            target_rho = int(source_rho) + 1
        except (TypeError, ValueError):
            target_rho = None
        for stage in homotopy.get("stages") or []:
            wall_time_s = _finite_float(stage.get("wall_time_s"))
            if wall_time_s is None:
                continue
            stage_row = {
                "kind": "transfer_bound_homotopy",
                "source_rho": source_rho,
                "target_rho": target_rho,
                "fraction": _finite_float(stage.get("fraction")),
                "attempt": stage.get("attempt"),
                "accepted": bool(stage.get("accepted")),
                "wall_time_s": wall_time_s,
                "solver_time_s": _finite_float(stage.get("solver_time_s")),
            }
            stages.append(stage_row)
            if target_rho is not None:
                by_rho[target_rho] = by_rho.get(target_rho, 0.0) + wall_time_s
    terminal_stages = []
    for stage in result.get("terminal_wheel_bound_summaries") or []:
        terminal_stages.append((stage, None, 1))
    for stage in result.get("inter_window_terminal_wheel_bound_summaries") or []:
        source_rho = stage.get("window")
        try:
            target_rho = int(source_rho) + 1
        except (TypeError, ValueError):
            target_rho = None
        terminal_stages.append((stage, source_rho, target_rho))
    for stage, source_rho, target_rho in terminal_stages:
        wall_time_s = _finite_float(stage.get("wall_time_s"))
        if wall_time_s is None:
            continue
        stage_row = {
            "kind": "terminal_wheel_bound_homotopy",
            "source_rho": source_rho,
            "target_rho": target_rho,
            "slack": _finite_float(stage.get("slack")),
            "attempt": stage.get("attempt"),
            "accepted": bool(stage.get("accepted")),
            "wall_time_s": wall_time_s,
            "solver_time_s": _finite_float(stage.get("solver_time_s")),
        }
        stages.append(stage_row)
        if target_rho is not None:
            by_rho[target_rho] = by_rho.get(target_rho, 0.0) + wall_time_s
    return {
        "available": bool(stages),
        "total_wall_time_s": sum(row["wall_time_s"] for row in stages),
        "by_target_rho_wall_time_s": by_rho,
        "stages": stages,
    }


def _first_failed_rho(
    window_rows: list[dict], physical_success: bool | None
) -> int | None:
    """Keep a later strict-prefix failure when the aggregate physical audit also fails."""

    first_failed_rho = next(
        (window["rho"] for window in window_rows if not window["validated"]),
        None,
    )
    if first_failed_rho is None and physical_success is False and window_rows:
        return 1
    return first_failed_rho


def _prefix_fatigue_checkpoints(result: dict, validated_cycles: int) -> dict:
    """Report like-for-like cumulative fatigue at the standard RHO checkpoints."""

    checkpoints = {}
    for cycle in BENCHMARK_STIMULATION_PATTERN_CYCLES:
        if cycle > validated_cycles:
            continue
        fatigue_rows = _fatigue_metrics(result, cycle)
        a_rows = [row for row in fatigue_rows if row["key"].startswith("A_")]
        muscle_rows = _executed_fatigue_objective_by_muscle(result, cycle)
        checkpoints[f"cycle_{cycle}"] = {
            "cycle": cycle,
            "executed_fatigue_objective": _executed_fatigue_objective(result, cycle),
            "fatigue_auc_cycles": float(
                sum(
                    row["fatigue_auc_cycles"]
                    for row in a_rows
                    if row["fatigue_auc_cycles"] is not None
                )
            ),
            "min_A_capacity_ratio": min(
                (
                    row["relative_final"]
                    for row in a_rows
                    if row["relative_final"] is not None
                ),
                default=None,
            ),
            "muscle_fatigue": muscle_rows,
        }
    return checkpoints


def solver_overview_rows(results: dict[str, dict]) -> list[dict]:
    """Build JSON-safe fatigue and timing outcomes for every selected backend."""

    rows = []
    for solver_name, result in results.items():
        performance = _window_performance(result)
        window_rows = _benchmark_window_rows(result)
        dual_warm_start_summaries = (
            result.get("acados_dual_warm_start_summaries")
            or result.get("nlp_dual_warm_start_summaries")
            or []
        )
        dual_warm_start_by_window = {
            item["window"]: item
            for item in dual_warm_start_summaries
            if item.get("window") is not None
        }
        restoration_timing = acados_transfer_restoration_timing(result)
        restoration_by_rho = restoration_timing["by_target_rho_wall_time_s"]
        for window in window_rows:
            applied_dual = dual_warm_start_by_window.get(window["window"])
            window["dual_warm_start_mode"] = (
                None if applied_dual is None else applied_dual.get("mode")
            )
            restoration_wall_time = restoration_by_rho.get(window["rho"], 0.0)
            window["feasibility_restoration_wall_time_s"] = restoration_wall_time
            window["effective_wall_time_s"] = (
                None
                if window["wall_time_s"] is None
                else window["wall_time_s"] + restoration_wall_time
            )
        consecutive_failures = 0
        maximum_consecutive_failures = 0
        for window in window_rows:
            if window["solver_converged"] and window["primal_feasible"] is True:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                maximum_consecutive_failures = max(
                    maximum_consecutive_failures, consecutive_failures
                )
        first_failed_rho = _first_failed_rho(
            window_rows, result.get("physical_success")
        )
        fatigue = _fatigue_metrics(result, performance["validated_cycles"])
        muscle_fatigue = _executed_fatigue_objective_by_muscle(
            result, performance["validated_cycles"]
        )
        saturation = _control_saturation_metrics(
            result, performance["validated_cycles"]
        )
        a_rows = [row for row in fatigue if row["key"].startswith("A_")]
        mean_fatigue = max(
            (
                row["mean_normalized_fatigue"]
                for row in a_rows
                if row["mean_normalized_fatigue"] is not None
            ),
            default=None,
        )
        fatigue_auc = sum(
            row["fatigue_auc_cycles"]
            for row in a_rows
            if row["fatigue_auc_cycles"] is not None
        )
        external_crank_power = _external_crank_power_metrics(
            result, performance["validated_cycles"]
        )
        cycle_boundary_wheel_angle = _cycle_boundary_wheel_angle_metrics(
            result, performance["validated_cycles"]
        )
        attempted_rho_wall_time = sum(
            window["wall_time_s"]
            for window in window_rows
            if window["wall_time_s"] is not None
        )
        attempted_effective_wall_time = sum(
            window["effective_wall_time_s"]
            for window in window_rows
            if window["effective_wall_time_s"] is not None
        )
        strict_hot_effective_wall_times = [
            window["effective_wall_time_s"]
            for window in window_rows[1 : performance["successful_prefix_windows"]]
            if window["validated"] and window["effective_wall_time_s"] is not None
        ]
        phase_one_summaries = result.get("transfer_phase_one_summaries") or []
        phase_one_wall_time_by_window = {}
        for phase_one_summary in phase_one_summaries:
            phase_one_window = phase_one_summary.get("window")
            phase_one_wall_time = _finite_float(phase_one_summary.get("wall_time_s"))
            if phase_one_window is None or phase_one_wall_time is None:
                continue
            phase_one_window = int(phase_one_window)
            phase_one_wall_time_by_window[phase_one_window] = (
                phase_one_wall_time_by_window.get(phase_one_window, 0.0)
                + phase_one_wall_time
            )
        attempted_effective_plus_phase_one_wall_times = [
            window["effective_wall_time_s"]
            + phase_one_wall_time_by_window.get(int(window["window"]), 0.0)
            for window in window_rows
            if window["effective_wall_time_s"] is not None
        ]
        strict_hot_effective_plus_phase_one_wall_times = [
            window["effective_wall_time_s"]
            + phase_one_wall_time_by_window.get(int(window["window"]), 0.0)
            for window in window_rows[1 : performance["successful_prefix_windows"]]
            if window["validated"] and window["effective_wall_time_s"] is not None
        ]
        finite_phase_one_wall_times = list(phase_one_wall_time_by_window.values())
        transfer_phase_one_timing = {
            "count": len(phase_one_summaries),
            "accepted_count": sum(
                bool(phase_one_summary.get("accepted"))
                for phase_one_summary in phase_one_summaries
            ),
            "total_wall_time_s": (
                float(sum(finite_phase_one_wall_times))
                if finite_phase_one_wall_times
                else 0.0
            ),
            "median_wall_time_s": (
                float(np.median(finite_phase_one_wall_times))
                if finite_phase_one_wall_times
                else None
            ),
            "p90_wall_time_s": (
                float(np.percentile(finite_phase_one_wall_times, 90))
                if finite_phase_one_wall_times
                else None
            ),
            "max_wall_time_s": (
                float(max(finite_phase_one_wall_times))
                if finite_phase_one_wall_times
                else None
            ),
        }
        end_to_end_wall_time = _finite_float(result.get("end_to_end_wall_time_s"))
        preparation_time = _finite_float(result.get("initial_guess_preparation_time_s"))
        unattributed_wall_time = (
            end_to_end_wall_time
            - preparation_time
            - attempted_effective_wall_time
            - sum(finite_phase_one_wall_times)
            if end_to_end_wall_time is not None and preparation_time is not None
            else None
        )
        status = _effective_status(result)
        if status is None and result.get("window_statuses"):
            status = result["window_statuses"][-1]
        validated_prefix_objectives = [
            window["objective"]
            for window in window_rows[: performance["successful_prefix_windows"]]
            if window["validated"] and window["objective"] is not None
        ]
        rows.append(
            {
                "solver": solver_name,
                "mode": result.get("mode"),
                "success": bool(result.get("success")),
                "solver_success": bool(result.get("solver_success")),
                "physical_success": bool(result.get("physical_success")),
                "status": None if status is None else int(status),
                "requested_cycles": result.get("requested_cycles")
                or result.get("requested_windows"),
                "exported_cycles": result.get("exported_cycles"),
                "covered_cycles": result.get("covered_cycles"),
                "validated_windows": performance["successful_prefix_windows"],
                "nlp_validated_cycles": performance["nlp_validated_cycles"],
                "physically_validated_cycles": performance[
                    "physically_validated_cycles"
                ],
                "validated_cycles": performance["validated_cycles"],
                "attempted_windows": result.get("attempted_windows"),
                "successful_windows": result.get("successful_windows"),
                "status_zero_windows": sum(
                    window["solver_converged"] for window in window_rows
                ),
                "primal_feasible_windows": sum(
                    window["primal_feasible"] is True for window in window_rows
                ),
                "first_failed_rho": first_failed_rho,
                "maximum_consecutive_failures": maximum_consecutive_failures,
                "objective": _finite_float(result.get("objective")),
                "window_objective_sum": _finite_float(result.get("objective")),
                "validated_prefix_window_objective_sum": (
                    float(sum(validated_prefix_objectives))
                    if validated_prefix_objectives
                    else None
                ),
                "executed_fatigue_objective": _executed_fatigue_objective(
                    result, performance["validated_cycles"]
                ),
                "solver_time_s": _finite_float(result.get("solver_time_s")),
                "wall_time_s": _finite_float(result.get("wall_time_s")),
                "end_to_end_wall_time_s": _finite_float(
                    result.get("end_to_end_wall_time_s")
                ),
                "initial_guess_preparation_time_s": preparation_time,
                "reduced_profile_build_time_s": _finite_float(
                    result.get("reduced_profile_build_time_s")
                ),
                "attempted_rho_wall_time_sum_s": attempted_rho_wall_time,
                "attempted_rho_effective_wall_time_sum_s": (
                    attempted_effective_wall_time
                ),
                "attempted_rho_effective_plus_phase_one_wall_time_sum_s": (
                    float(sum(attempted_effective_plus_phase_one_wall_times))
                ),
                "feasibility_restoration": restoration_timing,
                "unattributed_wall_time_s": unattributed_wall_time,
                "validated_solver_time_s": performance["successful_solver_time_s"],
                "validated_wall_time_s": performance["successful_wall_time_s"],
                "solver_time_per_cycle_s": performance["solver_time_per_cycle_s"],
                "wall_time_per_cycle_s": performance["wall_time_per_cycle_s"],
                "hot_window_count": performance["hot_window_count"],
                "hot_solver_time_median_s": performance["hot_solver_time_median_s"],
                "hot_solver_time_p90_s": performance["hot_solver_time_p90_s"],
                "hot_wall_time_median_s": performance["hot_wall_time_median_s"],
                "hot_wall_time_p90_s": performance["hot_wall_time_p90_s"],
                "hot_effective_wall_time_median_s": (
                    None
                    if not strict_hot_effective_wall_times
                    else float(np.median(strict_hot_effective_wall_times))
                ),
                "hot_effective_wall_time_p90_s": (
                    None
                    if not strict_hot_effective_wall_times
                    else float(np.percentile(strict_hot_effective_wall_times, 90))
                ),
                "hot_effective_plus_phase_one_wall_time_median_s": (
                    None
                    if not strict_hot_effective_plus_phase_one_wall_times
                    else float(
                        np.median(strict_hot_effective_plus_phase_one_wall_times)
                    )
                ),
                "hot_effective_plus_phase_one_wall_time_p90_s": (
                    None
                    if not strict_hot_effective_plus_phase_one_wall_times
                    else float(
                        np.percentile(
                            strict_hot_effective_plus_phase_one_wall_times, 90
                        )
                    )
                ),
                "hot_effective_plus_phase_one_wall_time_max_s": (
                    None
                    if not strict_hot_effective_plus_phase_one_wall_times
                    else float(max(strict_hot_effective_plus_phase_one_wall_times))
                ),
                "min_A_capacity_ratio": _minimum_a_capacity_ratio(fatigue),
                "max_mean_normalized_fatigue": mean_fatigue,
                "fatigue_auc_cycles": fatigue_auc if a_rows else None,
                "muscle_fatigue": muscle_fatigue,
                "fatigue_by_state": fatigue,
                "prefix_fatigue_checkpoints": _prefix_fatigue_checkpoints(
                    result, performance["validated_cycles"]
                ),
                "external_crank_power": external_crank_power,
                "cycle_boundary_wheel_angle": cycle_boundary_wheel_angle,
                "mechanical_equivalence_audit": result.get(
                    "mechanical_equivalence_audit"
                ),
                "nlp_crank_diagnostics": result.get("nlp_crank_diagnostics"),
                "physical_crank_diagnostics": result.get("physical_crank_diagnostics"),
                "state_boundary_jumps": result.get("state_boundary_jumps")
                or {
                    "available": False,
                    "boundary_count": 0,
                    "by_state": {},
                },
                "control_saturation": saturation,
                "pulse_width_active_set_summary": result.get(
                    "pulse_width_active_set_summary"
                )
                or [],
                "stop": _stop_classification(result),
                "native_solver_status": result.get("native_solver_status"),
                "windows": window_rows,
                "stimulation_patterns": stimulation_pattern_snapshots(result),
                "isolated_window_checkpoints": (
                    isolated_window_checkpoint_snapshots(result)
                ),
                "high_accuracy_trace_rollout": result.get(
                    "high_accuracy_trace_rollout"
                ),
                "pulse_width_cycle_variation": pulse_width_cycle_variation(
                    result, performance["validated_cycles"]
                ),
                "nlp_solver_stats": result.get("nlp_solver_stats") or [],
                "compiled_nlp_reuse": result.get("compiled_nlp_reuse"),
                "acados_maxiter_retry_summaries": (
                    result.get("acados_maxiter_retry_summaries") or []
                ),
                "transfer_phase_one_summaries": (
                    phase_one_summaries
                ),
                "transfer_phase_one_timing": transfer_phase_one_timing,
                "terminal_wheel_bound_summaries": (
                    result.get("terminal_wheel_bound_summaries") or []
                ),
                "inter_window_terminal_wheel_bound_summaries": (
                    result.get("inter_window_terminal_wheel_bound_summaries") or []
                ),
                "warm_start": {
                    "initial_guess_audits": result.get("initial_guess_audits") or [],
                    "dual_summaries": dual_warm_start_summaries,
                    "historical_cache_hit": result.get("standard_warmup_cache_hit"),
                },
                "fatigue_capacity_scales": result.get("fatigue_capacity_scales", {}),
                "error": result.get("error"),
            }
        )
    return rows


def print_solver_overview(results: dict[str, dict]) -> None:
    """Print comparable fatigue and timing outcomes for every selected backend."""

    print(
        "solver overview | solver | success | validated_cycles | "
        "window_objective_sum | executed_fatigue_objective | "
        "solver_time_per_cycle_s | hot_solver_time_median_s | "
        "hot_solver_time_p90_s | min_A_capacity_ratio | "
        "max_mean_normalized_fatigue | fatigue_auc_cycles | error"
    )
    for row in solver_overview_rows(results):
        print(
            "solver overview | "
            f"{row['solver'].upper()} | "
            f"{row['success']} | "
            f"{row['validated_cycles']} | "
            f"{_format_metric(row['window_objective_sum'])} | "
            f"{_format_metric(row['executed_fatigue_objective'])} | "
            f"{_format_metric(row['solver_time_per_cycle_s'])} | "
            f"{_format_metric(row['hot_solver_time_median_s'])} | "
            f"{_format_metric(row['hot_solver_time_p90_s'])} | "
            f"{_format_metric(row['min_A_capacity_ratio'])} | "
            f"{_format_metric(row['max_mean_normalized_fatigue'])} | "
            f"{_format_metric(row['fatigue_auc_cycles'])} | "
            f"{row['error']}"
        )


def write_benchmark_summary(output_path: str | Path, results: dict[str, dict]) -> Path:
    """Persist a compact, reproducible summary without serializing Solution objects."""

    output_path = Path(output_path).expanduser().resolve()
    configurations = {}
    for solver_name, result in results.items():
        args = result.get("args")
        configurations[solver_name] = {
            field: getattr(args, field, None)
            for field in BENCHMARK_CONFIGURATION_FIELDS
        }

    payload = {
        "schema_version": 3,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "JULIA_NUM_THREADS",
                )
            },
            "provenance": {
                name: os.environ.get(name)
                for name in (
                    "COCOFEST_BENCHMARK_COMMIT",
                    "BIOPTIM_BENCHMARK_COMMIT",
                    "BIOPTIM_INTEGRATION_BRANCH",
                    "LIBMAD_BENCHMARK_COMMIT",
                    "GITHUB_RUN_ID",
                    "GITHUB_RUN_ATTEMPT",
                    "RUNNER_NAME",
                )
            },
        },
        "configurations": configurations,
        "results": solver_overview_rows(results),
    }
    try:
        import bioptim
        import casadi

        payload["runtime"]["bioptim"] = getattr(bioptim, "__version__", None)
        payload["runtime"]["casadi"] = getattr(casadi, "__version__", None)
    except ImportError:
        pass
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            _json_compatible(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def _json_compatible(value):
    """Recursively convert NumPy/CasADi-adjacent values to strict JSON data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return str(value)
    if array.ndim == 0:
        return _json_compatible(array.item())
    return _json_compatible(array.tolist())


def _set_common_primal_feasibility_threshold(
    solver_args: tuple[argparse.Namespace, ...],
    threshold: float | None,
) -> None:
    """Forward the public physical threshold to every compared backend."""

    for args in solver_args:
        args.primal_feasibility_threshold = threshold


def main(
    objective: str = "fatigue",
    objective_shape: str = "quadratic",
    solvers: tuple[str, ...] = BENCHMARK_SOLVERS,
    single_shot: bool = False,
    cycles_per_window: int = 1,
    stimulations_per_cycle: int = 30,
    n_windows: int = 2,
    mechanical_formulation: str = "full",
    reduced_cycling_profile: str | Path | None = None,
    experimental_reduced_acados: bool = False,
    full_contact_constraints_terminal: bool = False,
    full_contact_position_terminal: bool = False,
    full_contact_position_tolerance: float = 0.0,
    full_contact_constraints_all_nodes: bool = False,
    full_contact_position_all_nodes: bool = False,
    n_threads: int | None = None,
    compact_rho_output: bool = False,
    resistive_torque: float = DEFAULT_CRANK_TORQUE_NM,
    acados_dir: str | None = None,
    codegen_tag: str | None = None,
    ipopt_max_iter: int = 2000,
    standard_warmup_max_iter: int | None = None,
    ipopt_linear_solver: str = "ma57",
    warmup_ipopt_linear_solver: str | None = None,
    standard_warmup_seed: str | Path | None = None,
    standard_warmup_seed_continuation: bool = False,
    legacy_standard_warmup_seed_signed_torque: float | None = None,
    common_initial_solution: str | Path | None = None,
    adopt_common_initial_solution_warmup_cycles: bool = False,
    common_initial_solution_output: str | Path | None = None,
    receding_horizon_solution_output: str | Path | None = None,
    allow_partial_receding_horizon_solution_output: bool = False,
    ipopt_hsl_library: str | None = None,
    ipopt_c_compile: bool = False,
    ipopt_print_level: int = 0,
    ipopt_print_timing_statistics: bool = False,
    ipopt_linear_system_scaling: str | None = None,
    ipopt_linear_scaling_on_demand: str | None = None,
    ipopt_ma57_automatic_scaling: bool | None = None,
    ipopt_ma57_pivot_order: int | None = None,
    ipopt_ma57_pivtol: float | None = None,
    ipopt_ma57_pivtolmax: float | None = None,
    ipopt_ma57_pre_alloc: float | None = None,
    ipopt_ma57_block_size: int | None = None,
    ipopt_ma57_node_amalgamation: int | None = None,
    ipopt_ma57_small_pivot_flag: int | None = None,
    ipopt_dual_warm_start_mode: str = "bounds",
    nlp_tolerance: float = 1e-6,
    primal_feasibility_threshold: float | None = None,
    fatrop_max_iter: int = 1000,
    fatrop_dual_warm_start_mode: str = "off",
    fatrop_c_compile: bool = False,
    fatrop_structure_detection: str = "auto",
    fatrop_bound_tightening_factor: float = 1e-8,
    fatrop_print_level: int = 0,
    fatrop_state_scaling: str = "none",
    madnlp_max_iter: int = 2000,
    madnlp_dual_warm_start_mode: str = "off",
    madnlp_c_compile: bool = False,
    madnlp_linear_solver: str | None = None,
    alpaqa_max_iter: int = 2000,
    alpaqa_alm_max_iter: int | None = None,
    alpaqa_dual_warm_start_mode: str = "constraints",
    alpaqa_lbfgs_memory: int = 20,
    alpaqa_max_wall_time: float | None = None,
    alpaqa_initial_penalty: float | None = None,
    alpaqa_initial_tolerance: float | None = None,
    alpaqa_penalty_update_factor: float | None = None,
    alpaqa_maximum_penalty: float | None = None,
    alpaqa_panoc_max_wall_time: float | None = None,
    alpaqa_max_no_progress: int | None = None,
    optional_nlp_periodic_ipopt_hot_start: bool = True,
    acados_max_iter: int = 100,
    acados_assisted_hot_start: bool = True,
    acados_control_homotopy_radii: tuple[float, ...] | None = None,
    acados_control_homotopy_tolerance: float = 5e-4,
    acados_control_homotopy_stage_iterations: int = 50,
    acados_control_homotopy_max_restarts: int = 1,
    acados_control_homotopy_keep_final_radius: bool | None = None,
    acados_control_homotopy_window_growth: float = 1.0,
    acados_control_homotopy_window_max_radius: float | None = None,
    control_regularization_weight: float = 0.0,
    acados_control_regularization_weight: float | None = None,
    control_regularization_target: float | None = None,
    control_regularization_target_source: str = "constant",
    acados_control_regularization_target_source: str | None = None,
    wheel_qdot_regularization_weight: float = 0.0,
    acados_wheel_qdot_regularization_weight: float | None = None,
    wheel_qdot_regularization_target: float = -float(2 * np.pi),
    wheel_qdot_bound_margin: float = 3.0,
    acados_wheel_qdot_fast_bound_margin: float | None = None,
    acados_wheel_qdot_slow_bound_margin: float | None = None,
    terminal_qdot_regularization_weight: float = 0.0,
    terminal_qdot_regularization_target_source: str = "previous",
    first_node_wheel_q_slack: float = 0.0,
    acados_terminal_wheel_q_slack: float = 0.002,
    acados_terminal_wheel_q_homotopy_slacks: tuple[float, ...] | None = None,
    acados_terminal_wheel_q_homotopy_each_window: bool = False,
    state_scaling: str = "full",
    acados_state_scaling: str | None = None,
    pulse_width_scaling: float = 1 / 400,
    pulse_width_active_set: str = "none",
    pulse_width_active_threshold: float = 0.01,
    pulse_width_active_margin: int = 3,
    acados_pulse_width_scaling: float | None = None,
    acados_pulse_width_trust_radius: float | None = None,
    acados_transfer_pulse_width_trust_radius: float | None = None,
    acados_proximal_control_weights: tuple[float, ...] | None = None,
    acados_proximal_control_each_window: bool = False,
    acados_proximal_control_tolerance: float = 5e-4,
    acados_proximal_control_stage_iterations: int = 50,
    acados_proximal_control_max_restarts: int = 1,
    acados_proximal_control_restart_feasibility_factor: float = 1.0,
    acados_proximal_control_try_next_weight_on_failure: bool = False,
    continue_after_acados_transfer_failure: bool = False,
    acados_transfer_mechanical_restoration: bool = False,
    acados_transfer_mechanical_control_radius: float = 5e-5,
    acados_transfer_mechanical_regularization: float = 1e-2,
    acados_transfer_mechanical_substeps: int = 5,
    acados_transfer_sqp_restarts: int = 0,
    acados_transfer_sqp_restart_iterations: int = 1,
    acados_transfer_sqp_restart_feasibility_tolerance: float = 1e-2,
    acados_transfer_active_set_guard_radius: float | None = None,
    acados_transfer_active_set_guard_margin: int = 1,
    acados_transfer_active_set_threshold: float = 1e-6,
    acados_fes_state_trust_radius: float | None = None,
    acados_fatigue_warmstart_mode: str = "continuous",
    acados_tolerance: float | None = None,
    acados_stationarity_tolerance: float | None = None,
    acados_qp_iter_max: int = 50,
    acados_qp_cond_n: int | None = None,
    acados_qp_warm_start_level: int = 0,
    acados_warm_start_first_qp: bool = False,
    acados_warm_start_first_qp_from_nlp: bool = False,
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
    acados_store_iterates: bool = False,
    acados_maxiter_retries: int = 0,
    acados_maxiter_retry_iterations: int = 20,
    acados_maxiter_retry_feasibility_tolerance: float = 2.5e-3,
    acados_reset_solver_before_solve: bool = False,
    acados_check_reuse_possible: bool = False,
    acados_code_reuse_tolerance: float = 1e-12,
    acados_with_anderson_acceleration: bool = False,
    acados_anderson_activation_threshold: float = 0.1,
    acados_byrd_omojokon_slack_relaxation_factor: float = 1.00001,
    acados_project_qdot_from_q: bool = False,
    shared_transfer_full_dynamics_rollout: bool = False,
    shared_transfer_contact_projection: bool = False,
    shared_transfer_contact_projection_mode: str = "position",
    shared_transfer_phase_one: bool = False,
    acados_transfer_phase_one: bool = False,
    acados_transfer_phase_one_mode: str = "all",
    acados_transfer_phase_one_lookback_nodes: int | None = None,
    acados_cyclical_transfer_mode: str = "extrapolate",
    acados_transfer_phase_one_proximity_weight: float = 1.0,
    acados_transfer_phase_one_defect_weight: float = 10.0,
    acados_transfer_phase_one_substeps: int = 5,
    acados_transfer_phase_one_max_state_change: float | None = None,
    acados_transfer_phase_one_max_q_change: float | None = None,
    acados_transfer_phase_one_max_qdot_change: float | None = None,
    acados_transfer_phase_one_max_fes_change: float | None = None,
    acados_transfer_bound_homotopy: bool = False,
    acados_transfer_bound_homotopy_fractions: tuple[float, ...] = (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ),
    acados_transfer_bound_homotopy_padding: float = 0.05,
    acados_transfer_bound_homotopy_iterations: int = 30,
    acados_transfer_bound_homotopy_tolerance: float = 1e-4,
    acados_transfer_bound_homotopy_solver_tolerance: float | None = None,
    acados_transfer_bound_homotopy_min_fraction_step: float = 0.0,
    acados_transfer_bound_homotopy_max_refinements: int = 0,
    shared_initial_phase_one: bool = False,
    shared_transfer_rollout_substeps: int = 5,
    shared_transfer_rollout_max_bound_violation: float = 1.0,
    acados_transfer_select_projected_candidate: bool = False,
    acados_transfer_selector_max_q_bound_violation_rad: float = 1.0,
    acados_transfer_selector_max_qdot_bound_violation_rad_s: float = 12.0,
    acados_transfer_selector_max_other_scaled_bound_violation: float = 1.0,
    acados_transfer_selector_max_scaled_q_defect: float = 0.1,
    acados_transfer_selector_max_scaled_qdot_defect: float = 0.1,
    acados_transfer_selector_improvement_ratio: float = 0.95,
    shared_transfer_ding_force_compensation: bool = False,
    shared_transfer_ding_force_compensation_substeps: int = 5,
    shared_transfer_ding_force_compensation_iterations: int = 20,
    acados_transfer_ding_force_compensation: bool = False,
    acados_integrator_type: str = "IRK",
    acados_collocation_type: str = "GAUSS_LEGENDRE",
    acados_sim_stages: int = 4,
    acados_sim_steps: int = 5,
    acados_newton_iter: int = 5,
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
    initial_guess_diagnostics: bool = False,
    exact_initial_nlp_audit: bool = False,
    periodic_ipopt_refinement: bool = True,
    periodic_ipopt_refinement_iterations: int = 300,
    periodic_ipopt_refinement_use_sx: bool = True,
    periodic_ipopt_refinement_ode_solver: str = "target",
    warmup_state_comparison_limit: int = 12,
    state_comparison_limit: int = 12,
    print_traces: bool = False,
    validate_integrator_maps: bool = False,
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
    output_json: str | Path | None = None,
):
    invocation_cwd = Path.cwd()

    def resolve_invocation_path(value):
        if value is None:
            return None
        path = Path(value).expanduser()
        return path if path.is_absolute() else invocation_cwd / path

    # ``main`` historically changes into the example directory so model paths
    # remain stable. Resolve every user-facing path first so both interpreted
    # and C-compiled runs retain CLI-relative path semantics.
    standard_warmup_seed = resolve_invocation_path(standard_warmup_seed)
    common_initial_solution = resolve_invocation_path(common_initial_solution)
    common_initial_solution_output = resolve_invocation_path(
        common_initial_solution_output
    )
    receding_horizon_solution_output = resolve_invocation_path(
        receding_horizon_solution_output
    )
    reduced_cycling_profile = resolve_invocation_path(reduced_cycling_profile)
    ipopt_hsl_library = resolve_invocation_path(ipopt_hsl_library)
    output_json = resolve_invocation_path(output_json)
    os.chdir(EXAMPLE_DIR)
    if n_threads is None:
        n_threads = os.cpu_count() or 1
    if n_threads < 1:
        raise ValueError("n_threads must be at least 1.")
    if single_shot and n_windows != cycles_per_window:
        raise ValueError(
            "--single-shot requires --n-windows to equal --cycles-per-window "
            "so the reported cycle count matches the optimized horizon."
        )
    if isinstance(solvers, str):
        solvers = parse_solver_names(solvers)
    solvers = tuple(dict.fromkeys(name.lower() for name in solvers))
    invalid_solvers = set(solvers) - set(BENCHMARK_SOLVERS)
    if invalid_solvers or not solvers:
        raise ValueError(
            "solvers must contain at least one of: " f"{', '.join(BENCHMARK_SOLVERS)}."
        )
    objective_names = {
        item.strip().lower() for item in objective.split(",") if item.strip()
    }
    if objective_names != {"fatigue"}:
        print(
            "benchmark objective warning: the requested objective is not fatigue-only; "
            "solver relevance conclusions should use --objective fatigue."
        )
    if acados_dir:
        os.environ["ACADOS_SOURCE_DIR"] = str(Path(acados_dir).resolve())

    ipopt_args = _solver_config(
        "ipopt",
        single_shot=single_shot,
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
        acados_assisted_hot_start=acados_assisted_hot_start,
        acados_control_homotopy_radii=acados_control_homotopy_radii,
        acados_control_homotopy_tolerance=acados_control_homotopy_tolerance,
        acados_control_homotopy_stage_iterations=(
            acados_control_homotopy_stage_iterations
        ),
        acados_control_homotopy_max_restarts=(acados_control_homotopy_max_restarts),
        control_regularization_weight=control_regularization_weight,
        control_regularization_target=control_regularization_target,
        control_regularization_target_source=control_regularization_target_source,
        wheel_qdot_regularization_weight=wheel_qdot_regularization_weight,
        wheel_qdot_regularization_target=wheel_qdot_regularization_target,
        wheel_qdot_bound_margin=wheel_qdot_bound_margin,
        acados_wheel_qdot_fast_bound_margin=None,
        acados_wheel_qdot_slow_bound_margin=None,
        terminal_qdot_regularization_weight=terminal_qdot_regularization_weight,
        terminal_qdot_regularization_target_source=(
            terminal_qdot_regularization_target_source
        ),
        first_node_wheel_q_slack=first_node_wheel_q_slack,
        acados_terminal_wheel_q_slack=acados_terminal_wheel_q_slack,
        state_scaling=state_scaling,
        pulse_width_scaling=pulse_width_scaling,
        pulse_width_active_set=pulse_width_active_set,
        pulse_width_active_threshold=pulse_width_active_threshold,
        pulse_width_active_margin=pulse_width_active_margin,
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
        periodic_ipopt_refinement_use_sx=True,
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
        single_shot=single_shot,
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
        acados_assisted_hot_start=acados_assisted_hot_start,
        acados_control_homotopy_radii=acados_control_homotopy_radii,
        acados_control_homotopy_tolerance=acados_control_homotopy_tolerance,
        acados_control_homotopy_stage_iterations=(
            acados_control_homotopy_stage_iterations
        ),
        acados_control_homotopy_max_restarts=(acados_control_homotopy_max_restarts),
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
        acados_wheel_qdot_fast_bound_margin=(
            acados_wheel_qdot_fast_bound_margin
        ),
        acados_wheel_qdot_slow_bound_margin=(
            acados_wheel_qdot_slow_bound_margin
        ),
        terminal_qdot_regularization_weight=terminal_qdot_regularization_weight,
        terminal_qdot_regularization_target_source=(
            terminal_qdot_regularization_target_source
        ),
        first_node_wheel_q_slack=first_node_wheel_q_slack,
        acados_terminal_wheel_q_slack=acados_terminal_wheel_q_slack,
        state_scaling=(
            acados_state_scaling if acados_state_scaling is not None else state_scaling
        ),
        pulse_width_scaling=(
            acados_pulse_width_scaling
            if acados_pulse_width_scaling is not None
            else pulse_width_scaling
        ),
        pulse_width_active_set=pulse_width_active_set,
        pulse_width_active_threshold=pulse_width_active_threshold,
        pulse_width_active_margin=pulse_width_active_margin,
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
    _set_common_primal_feasibility_threshold(
        (ipopt_args, acados_args),
        primal_feasibility_threshold,
    )
    ipopt_args.max_consecutive_failing = max_consecutive_failing
    acados_args.max_consecutive_failing = max_consecutive_failing
    ipopt_args.warmup_ipopt_linear_solver = warmup_ipopt_linear_solver
    ipopt_args.standard_warmup_seed = standard_warmup_seed
    acados_args.standard_warmup_seed = standard_warmup_seed
    ipopt_args.standard_warmup_seed_continuation = standard_warmup_seed_continuation
    acados_args.standard_warmup_seed_continuation = standard_warmup_seed_continuation
    ipopt_args.standard_warmup_max_iterations = standard_warmup_max_iter
    acados_args.standard_warmup_max_iterations = standard_warmup_max_iter
    ipopt_args.legacy_standard_warmup_seed_signed_torque = (
        legacy_standard_warmup_seed_signed_torque
    )
    acados_args.legacy_standard_warmup_seed_signed_torque = (
        legacy_standard_warmup_seed_signed_torque
    )
    ipopt_args.common_initial_solution = common_initial_solution
    acados_args.common_initial_solution = common_initial_solution
    ipopt_args.adopt_common_initial_solution_warmup_cycles = (
        adopt_common_initial_solution_warmup_cycles
    )
    acados_args.adopt_common_initial_solution_warmup_cycles = (
        adopt_common_initial_solution_warmup_cycles
    )
    ipopt_args.common_initial_solution_output = common_initial_solution_output
    acados_args.common_initial_solution_output = common_initial_solution_output
    ipopt_args.receding_horizon_solution_output = receding_horizon_solution_output
    acados_args.receding_horizon_solution_output = receding_horizon_solution_output
    ipopt_args.allow_partial_receding_horizon_solution_output = (
        allow_partial_receding_horizon_solution_output
    )
    acados_args.allow_partial_receding_horizon_solution_output = (
        allow_partial_receding_horizon_solution_output
    )
    acados_args.experimental_reduced_acados = experimental_reduced_acados
    for name, value in (
        ("ipopt_print_level", ipopt_print_level),
        ("ipopt_print_timing_statistics", ipopt_print_timing_statistics),
        ("ipopt_linear_system_scaling", ipopt_linear_system_scaling),
        ("ipopt_linear_scaling_on_demand", ipopt_linear_scaling_on_demand),
        ("ipopt_ma57_automatic_scaling", ipopt_ma57_automatic_scaling),
        ("ipopt_ma57_pivot_order", ipopt_ma57_pivot_order),
        ("ipopt_ma57_pivtol", ipopt_ma57_pivtol),
        ("ipopt_ma57_pivtolmax", ipopt_ma57_pivtolmax),
        ("ipopt_ma57_pre_alloc", ipopt_ma57_pre_alloc),
        ("ipopt_ma57_block_size", ipopt_ma57_block_size),
        ("ipopt_ma57_node_amalgamation", ipopt_ma57_node_amalgamation),
        ("ipopt_ma57_small_pivot_flag", ipopt_ma57_small_pivot_flag),
    ):
        setattr(ipopt_args, name, value)
    ipopt_args.compact_rho_output = compact_rho_output
    acados_args.compact_rho_output = compact_rho_output
    ipopt_args.validate_integrator_maps = validate_integrator_maps
    acados_args.validate_integrator_maps = validate_integrator_maps
    acados_args.acados_integrator_type = acados_integrator_type
    acados_args.acados_collocation_type = acados_collocation_type
    acados_args.acados_sim_stages = acados_sim_stages
    acados_args.acados_sim_steps = acados_sim_steps
    acados_args.ipopt_profile = None
    acados_args.benchmark_profile = "acados-runtime"
    acados_args.transcription_profile = (
        f"acados-{acados_integrator_type.lower()}-"
        f"{acados_sim_stages}stage-{acados_sim_steps}step"
    )
    acados_args.profile_version = 1
    acados_args.profile_hash = hashlib.sha256(
        acados_args.transcription_profile.encode()
    ).hexdigest()
    acados_args.profile_integrity = True
    acados_args.scientific_status = "candidate"
    acados_args.acados_newton_iter = acados_newton_iter
    acados_args.periodic_ipopt_refinement_ode_solver = (
        periodic_ipopt_refinement_ode_solver
    )
    acados_args.acados_stationarity_tolerance = acados_stationarity_tolerance
    acados_args.acados_control_homotopy_keep_final_radius = (
        acados_control_homotopy_keep_final_radius
    )
    acados_args.acados_control_homotopy_window_growth = (
        acados_control_homotopy_window_growth
    )
    acados_args.acados_control_homotopy_window_max_radius = (
        acados_control_homotopy_window_max_radius
    )
    acados_args.acados_dual_warm_start_mode = acados_dual_warm_start_mode
    acados_args.acados_qp_cond_n = acados_qp_cond_n
    acados_args.acados_qp_warm_start_level = acados_qp_warm_start_level
    acados_args.acados_warm_start_first_qp = acados_warm_start_first_qp
    acados_args.acados_warm_start_first_qp_from_nlp = (
        acados_warm_start_first_qp_from_nlp
    )
    acados_args.acados_store_iterates = acados_store_iterates
    acados_args.acados_maxiter_retries = acados_maxiter_retries
    acados_args.acados_maxiter_retry_iterations = acados_maxiter_retry_iterations
    acados_args.acados_maxiter_retry_feasibility_tolerance = (
        acados_maxiter_retry_feasibility_tolerance
    )
    acados_args.acados_reset_solver_before_solve = acados_reset_solver_before_solve
    acados_args.acados_check_reuse_possible = acados_check_reuse_possible
    acados_args.acados_code_reuse_tolerance = acados_code_reuse_tolerance
    acados_args.acados_with_anderson_acceleration = acados_with_anderson_acceleration
    acados_args.acados_anderson_activation_threshold = (
        acados_anderson_activation_threshold
    )
    acados_args.acados_byrd_omojokon_slack_relaxation_factor = (
        acados_byrd_omojokon_slack_relaxation_factor
    )
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
    acados_args.acados_proximal_control_restart_feasibility_factor = (
        acados_proximal_control_restart_feasibility_factor
    )
    acados_args.acados_proximal_control_try_next_weight_on_failure = (
        acados_proximal_control_try_next_weight_on_failure
    )
    acados_args.continue_after_acados_transfer_failure = (
        continue_after_acados_transfer_failure
    )
    acados_args.acados_transfer_mechanical_restoration = (
        acados_transfer_mechanical_restoration
    )
    acados_args.acados_transfer_mechanical_control_radius = (
        acados_transfer_mechanical_control_radius
    )
    acados_args.acados_transfer_mechanical_regularization = (
        acados_transfer_mechanical_regularization
    )
    acados_args.acados_transfer_mechanical_substeps = (
        acados_transfer_mechanical_substeps
    )
    acados_args.acados_transfer_sqp_restarts = acados_transfer_sqp_restarts
    acados_args.acados_transfer_sqp_restart_iterations = (
        acados_transfer_sqp_restart_iterations
    )
    acados_args.acados_transfer_sqp_restart_feasibility_tolerance = (
        acados_transfer_sqp_restart_feasibility_tolerance
    )
    acados_args.acados_transfer_active_set_guard_radius = (
        acados_transfer_active_set_guard_radius
    )
    acados_args.acados_transfer_active_set_guard_margin = (
        acados_transfer_active_set_guard_margin
    )
    acados_args.acados_transfer_active_set_threshold = (
        acados_transfer_active_set_threshold
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
        solver_args.transfer_contact_manifold_projection = (
            shared_transfer_contact_projection
        )
        solver_args.transfer_contact_manifold_projection_mode = (
            shared_transfer_contact_projection_mode
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
    acados_args.acados_transfer_select_projected_candidate = (
        acados_transfer_select_projected_candidate
    )
    acados_args.acados_transfer_selector_max_q_bound_violation_rad = (
        acados_transfer_selector_max_q_bound_violation_rad
    )
    acados_args.acados_transfer_selector_max_qdot_bound_violation_rad_s = (
        acados_transfer_selector_max_qdot_bound_violation_rad_s
    )
    acados_args.acados_transfer_selector_max_other_scaled_bound_violation = (
        acados_transfer_selector_max_other_scaled_bound_violation
    )
    acados_args.acados_transfer_selector_max_scaled_q_defect = (
        acados_transfer_selector_max_scaled_q_defect
    )
    acados_args.acados_transfer_selector_max_scaled_qdot_defect = (
        acados_transfer_selector_max_scaled_qdot_defect
    )
    acados_args.acados_transfer_selector_improvement_ratio = (
        acados_transfer_selector_improvement_ratio
    )
    acados_args.acados_transfer_phase_one = bool(
        shared_transfer_phase_one or acados_transfer_phase_one
    )
    acados_args.acados_transfer_phase_one_mode = acados_transfer_phase_one_mode
    acados_args.acados_transfer_phase_one_lookback_nodes = (
        acados_transfer_phase_one_lookback_nodes
    )
    acados_args.acados_cyclical_transfer_mode = acados_cyclical_transfer_mode
    acados_args.full_dynamics_phase_one_proximity_weight = (
        acados_transfer_phase_one_proximity_weight
    )
    acados_args.full_dynamics_phase_one_defect_weight = (
        acados_transfer_phase_one_defect_weight
    )
    acados_args.full_dynamics_phase_one_substeps = acados_transfer_phase_one_substeps
    acados_args.full_dynamics_phase_one_max_state_change = (
        acados_transfer_phase_one_max_state_change
    )
    acados_args.full_dynamics_phase_one_max_q_change = (
        acados_transfer_phase_one_max_q_change
    )
    acados_args.full_dynamics_phase_one_max_qdot_change = (
        acados_transfer_phase_one_max_qdot_change
    )
    acados_args.full_dynamics_phase_one_max_fes_change = (
        acados_transfer_phase_one_max_fes_change
    )
    acados_args.acados_transfer_bound_homotopy = acados_transfer_bound_homotopy
    # The bound homotopy may become primal-feasible before its last SQP
    # iterate. Retain those iterates so the continuation can restart from the
    # certified best primal instead of a later, degraded MAXITER iterate.
    acados_args.acados_store_iterates = bool(
        acados_store_iterates or acados_transfer_bound_homotopy
    )
    acados_args.acados_transfer_bound_homotopy_fractions = (
        acados_transfer_bound_homotopy_fractions
    )
    acados_args.acados_transfer_bound_homotopy_padding = (
        acados_transfer_bound_homotopy_padding
    )
    acados_args.acados_transfer_bound_homotopy_iterations = (
        acados_transfer_bound_homotopy_iterations
    )
    acados_args.acados_transfer_bound_homotopy_tolerance = (
        acados_transfer_bound_homotopy_tolerance
    )
    acados_args.acados_transfer_bound_homotopy_solver_tolerance = (
        acados_transfer_bound_homotopy_solver_tolerance
    )
    acados_args.acados_transfer_bound_homotopy_min_fraction_step = (
        acados_transfer_bound_homotopy_min_fraction_step
    )
    acados_args.acados_transfer_bound_homotopy_max_refinements = (
        acados_transfer_bound_homotopy_max_refinements
    )

    fatrop_args = _nlp_solver_config(
        "fatrop",
        ipopt_args,
        tolerance=nlp_tolerance,
        max_iterations=fatrop_max_iter,
        dual_warm_start_mode=fatrop_dual_warm_start_mode,
        fatrop_c_compile=fatrop_c_compile,
        fatrop_structure_detection=fatrop_structure_detection,
        fatrop_bound_tightening_factor=fatrop_bound_tightening_factor,
        fatrop_print_level=fatrop_print_level,
        periodic_ipopt_hot_start=optional_nlp_periodic_ipopt_hot_start,
    )
    fatrop_args.state_scaling = fatrop_state_scaling
    madnlp_args = _nlp_solver_config(
        "madnlp",
        ipopt_args,
        tolerance=nlp_tolerance,
        max_iterations=madnlp_max_iter,
        dual_warm_start_mode=madnlp_dual_warm_start_mode,
        madnlp_linear_solver=madnlp_linear_solver,
        periodic_ipopt_hot_start=optional_nlp_periodic_ipopt_hot_start,
    )
    ipopt_args.ipopt_c_compile = ipopt_c_compile
    ipopt_args.ipopt_hsl_library = ipopt_hsl_library
    fatrop_args.ipopt_c_compile = False
    fatrop_args.ipopt_hsl_library = None
    madnlp_args.ipopt_c_compile = False
    madnlp_args.ipopt_hsl_library = None
    madnlp_args.madnlp_c_compile = madnlp_c_compile
    alpaqa_args = _nlp_solver_config(
        "alpaqa",
        ipopt_args,
        tolerance=nlp_tolerance,
        max_iterations=alpaqa_max_iter,
        dual_warm_start_mode=alpaqa_dual_warm_start_mode,
        alpaqa_alm_max_iterations=alpaqa_alm_max_iter,
        alpaqa_lbfgs_memory=alpaqa_lbfgs_memory,
        alpaqa_max_wall_time=alpaqa_max_wall_time,
        alpaqa_initial_penalty=alpaqa_initial_penalty,
        alpaqa_initial_tolerance=alpaqa_initial_tolerance,
        alpaqa_penalty_update_factor=alpaqa_penalty_update_factor,
        alpaqa_maximum_penalty=alpaqa_maximum_penalty,
        alpaqa_panoc_max_wall_time=alpaqa_panoc_max_wall_time,
        alpaqa_max_no_progress=alpaqa_max_no_progress,
        periodic_ipopt_hot_start=optional_nlp_periodic_ipopt_hot_start,
    )
    for solver_args_for_threads in (
        ipopt_args,
        acados_args,
        fatrop_args,
        madnlp_args,
        alpaqa_args,
    ):
        solver_args_for_threads.n_threads = n_threads

    ipopt_args.nlp_tolerance = nlp_tolerance
    ipopt_args.ipopt_dual_warm_start_mode = ipopt_dual_warm_start_mode

    normalized_ipopt_profile = _normalize_ipopt_profile(ipopt_profile)
    ipopt_label = {
        "historical": "historical reference",
        "periodic_collocation": "periodic-collocation bridge",
        "scientific_radau4": "scientific Radau-4 diagnostic",
        "scientific_radau5": "scientific Radau-5",
        "scientific_radau6": "scientific Radau-6 diagnostic",
        "acados_like": "ACADOS-like diagnostic",
    }[normalized_ipopt_profile]
    solver_args = {
        "ipopt": ipopt_args,
        "acados": acados_args,
        "fatrop": fatrop_args,
        "madnlp": madnlp_args,
        "alpaqa": alpaqa_args,
    }
    # A benchmark comparison must use a common physical crank coordinate.
    # Full q/qdot traces are therefore projected onto the same contact
    # manifold as the reduced theta/omega OCP before phase and cadence metrics
    # are computed.
    for solver_configuration in solver_args.values():
        solver_configuration.initial_guess_diagnostics = bool(
            initial_guess_diagnostics
        )
        solver_configuration.exact_initial_nlp_audit = bool(
            exact_initial_nlp_audit
            and solver_configuration.solver in NLP_SOLVER_NAMES
        )
        solver_configuration.mechanical_equivalence_audit = True
        solver_configuration.full_contact_constraints_terminal = bool(
            full_contact_constraints_terminal
        )
        solver_configuration.full_contact_position_terminal = bool(
            full_contact_position_terminal
        )
        solver_configuration.full_contact_position_tolerance = float(
            full_contact_position_tolerance
        )
        solver_configuration.full_contact_constraints_all_nodes = bool(
            full_contact_constraints_all_nodes
        )
        solver_configuration.full_contact_position_all_nodes = bool(
            full_contact_position_all_nodes
        )
        solver_configuration.reduced_cycling_profile = (
            None if reduced_cycling_profile is None else Path(reduced_cycling_profile)
        )
    if mechanical_formulation not in ("full", "reduced"):
        raise ValueError("mechanical_formulation must be 'full' or 'reduced'.")
    if mechanical_formulation == "reduced":
        supported = {"ipopt", "fatrop", "madnlp"}
        if experimental_reduced_acados:
            supported.add("acados")
        unsupported = set(solvers) - supported
        if unsupported:
            raise ValueError(
                "Reduced mechanics are currently certified only for IPOPT, "
                f"Fatrop and MadNLP; remove {', '.join(sorted(unsupported))}."
            )
        for solver_name in supported:
            solver_args[solver_name].mechanical_formulation = "reduced"
            solver_args[solver_name].reduced_cycling_profile = (
                None
                if reduced_cycling_profile is None
                else Path(reduced_cycling_profile)
            )
            solver_args[solver_name].model_formulation = "periodic_node"
            solver_args[solver_name].torque_application = "constant"
            if solver_name != "acados":
                solver_args[solver_name].disable_periodic_fes_warmup_projection = True
    else:
        for solver_configuration in solver_args.values():
            solver_configuration.mechanical_formulation = "full"
        # The constrained full dynamics keep the wheel centre fixed at the
        # acceleration level, but still require a position/velocity anchor at
        # the first node. ACADOS v0.5.5 support for Node.START nonlinear
        # constraints is provided by the pinned Bioptim integration branch.
        # Repeating the same holonomic equalities at every node is redundant
        # and was observed to make the QP substantially less robust.
        solver_args["acados"].enforce_start_constraints = True
    print(f"NLP reference profile: {ipopt_label}")
    results = {}
    for solver_name in solvers:
        results[solver_name] = _run_benchmark_case(
            solver_name,
            solver_args[solver_name],
            echo=True,
            print_traceback=print_traces,
        )
        print()

    print_solver_overview(results)
    if output_json is not None:
        written_path = write_benchmark_summary(output_json, results)
        print(f"benchmark JSON: {written_path}")
    if "ipopt" in results and "acados" in results:
        print()
        print_comparison(
            results["ipopt"],
            results["acados"],
            print_traces=print_traces,
            state_comparison_limit=state_comparison_limit,
        )
    return results


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solvers",
        type=parse_solver_names,
        default=BENCHMARK_SOLVERS,
        help="Comma-separated solver matrix: ipopt,acados,fatrop,madnlp.",
    )
    parser.add_argument(
        "--objective",
        default="fatigue",
        help="Benchmark objective; use 'fatigue' for the endurance comparison.",
    )
    parser.add_argument(
        "--objective-shape",
        default="quadratic",
        choices=("quadratic", "linear"),
    )
    parser.add_argument("--cycles-per-window", type=int, default=1)
    parser.add_argument("--stimulations-per-cycle", type=int, default=30)
    parser.add_argument(
        "--single-shot",
        action="store_true",
        help=(
            "Solve one OCP spanning --cycles-per-window cycles instead of "
            "entering the receding-horizon loop."
        ),
    )
    parser.add_argument(
        "--n-windows",
        type=int,
        default=2,
        help=(
            "Number of cycles to validate; overlapping solve count is "
            "n_windows - cycles_per_window + 1."
        ),
    )
    parser.add_argument(
        "--mechanical-formulation",
        choices=("full", "reduced"),
        default="full",
        help=(
            "Benchmark the full constrained mechanics or the experimental "
            "theta/omega reduction. Reduced mode accepts IPOPT and MadNLP."
        ),
    )
    parser.add_argument(
        "--reduced-cycling-profile",
        type=Path,
        default=None,
        help="Optional cached .npz profile used in reduced mode.",
    )
    parser.add_argument(
        "--experimental-reduced-acados",
        action="store_true",
        help=(
            "Enable the uncertified reduced ACADOS SQP path for one-cycle "
            "rollout/projection diagnostics."
        ),
    )
    parser.add_argument(
        "--full-contact-constraints-terminal",
        action="store_true",
        help=(
            "For full mechanics, close wheel-centre position and velocity at "
            "the terminal node to make the RHO seam mechanically feasible."
        ),
    )
    parser.add_argument(
        "--full-contact-position-terminal",
        action="store_true",
        help=(
            "For full mechanics, close only wheel-centre position at the "
            "terminal node to target the observed RHO seam."
        ),
    )
    parser.add_argument(
        "--full-contact-position-tolerance",
        type=float,
        default=0.0,
        metavar="M",
        help=(
            "Symmetric tolerance on the full wheel-centre position "
            "constraints; zero preserves the historical equality."
        ),
    )
    parser.add_argument(
        "--full-contact-constraints-all-nodes",
        action="store_true",
        help=(
            "For full mechanics, constrain wheel-centre position and velocity "
            "at every shooting node to prevent contact-manifold drift."
        ),
    )
    parser.add_argument(
        "--full-contact-position-all-nodes",
        action="store_true",
        help=(
            "For full mechanics, constrain wheel-centre position at every "
            "node without adding a redundant velocity path constraint."
        ),
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=os.cpu_count() or 1,
        help=(
            "Number of Bioptim/CasADi workers. Defaults to all logical CPUs; "
            "avoid adding nested BLAS threads unless measured."
        ),
    )
    parser.add_argument("--compact-rho-output", action="store_true")
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
    torque_group = parser.add_mutually_exclusive_group()
    torque_group.add_argument(
        "--crank-assistance",
        dest="resistive_torque",
        type=parse_crank_assistance,
        default=DEFAULT_CRANK_TORQUE_NM,
        metavar="N_M",
        help=(
            "Non-negative crank assistance magnitude. The default 0.2 N.m is "
            "converted to the signed cycling torque -0.2 N.m."
        ),
    )
    torque_group.add_argument(
        "--signed-crank-torque",
        "--resistive-torque",
        dest="resistive_torque",
        type=float,
        metavar="N_M",
        help=(
            "Expert signed-torque override. Negative values assist the cycling "
            "direction; positive values are resistive. --resistive-torque is "
            "retained as a legacy alias."
        ),
    )
    parser.add_argument("--acados-dir", default=os.environ.get("ACADOS_SOURCE_DIR"))
    parser.add_argument("--codegen-tag", default="fes_compare")
    parser.add_argument("--ipopt-max-iter", type=int, default=2000)
    parser.add_argument(
        "--standard-warmup-max-iter",
        type=int,
        default=None,
        help=(
            "Maximum IPOPT iterations used only to build a standard warmup; "
            "--ipopt-max-iter remains the per-RHO budget."
        ),
    )
    parser.add_argument("--ipopt-linear-solver", default="ma57")
    parser.add_argument(
        "--warmup-ipopt-linear-solver",
        default=None,
        help="Fixed linear solver used to build a shared warmup across targets.",
    )
    parser.add_argument(
        "--standard-warmup-seed",
        type=Path,
        default=None,
        help=(
            "Explicit warmup .npz used as the primal seed. Use this for a "
            "neighbouring 0.02 N.m torque-continuation step."
        ),
    )
    parser.add_argument(
        "--standard-warmup-seed-continuation",
        action="store_true",
        help=(
            "Allow the documented seed torque to differ from the target torque. "
            "The target NLP must still converge and pass the physical checks."
        ),
    )
    parser.add_argument(
        "--legacy-standard-warmup-seed-signed-torque",
        type=float,
        default=None,
        metavar="N_M",
        help=(
            "Explicit signed-torque assertion required to reuse a legacy "
            "--standard-warmup-seed without metadata."
        ),
    )
    parser.add_argument(
        "--common-initial-solution",
        type=Path,
        default=None,
        help=(
            "Converged periodic target solution applied identically to every "
            "selected backend."
        ),
    )
    parser.add_argument(
        "--adopt-common-initial-solution-warmup-cycles",
        action="store_true",
        help=(
            "Preserve the common seed's warmup-cycle chronology when the consumer "
            "intentionally disables its own redundant standard warmup."
        ),
    )
    parser.add_argument(
        "--common-initial-solution-output",
        type=Path,
        default=None,
        help=(
            "Write the first converged target window as a solver-neutral "
            "periodic initial solution."
        ),
    )
    parser.add_argument(
        "--ipopt-hsl-library",
        default=None,
        help="Absolute CoinHSL library passed to IPOPT's hsllib option.",
    )
    parser.add_argument("--ipopt-print-level", type=int, default=0)
    parser.add_argument("--ipopt-print-timing-statistics", action="store_true")
    parser.add_argument(
        "--ipopt-linear-system-scaling",
        choices=("none", "mc19", "slack-based"),
        default=None,
    )
    parser.add_argument(
        "--ipopt-linear-scaling-on-demand",
        choices=("yes", "no"),
        default=None,
    )
    ma57_scaling = parser.add_mutually_exclusive_group()
    ma57_scaling.add_argument(
        "--ipopt-ma57-automatic-scaling",
        dest="ipopt_ma57_automatic_scaling",
        action="store_true",
    )
    ma57_scaling.add_argument(
        "--ipopt-no-ma57-automatic-scaling",
        dest="ipopt_ma57_automatic_scaling",
        action="store_false",
    )
    parser.set_defaults(ipopt_ma57_automatic_scaling=None)
    parser.add_argument("--ipopt-ma57-pivot-order", type=int, default=None)
    parser.add_argument("--ipopt-ma57-pivtol", type=float, default=None)
    parser.add_argument("--ipopt-ma57-pivtolmax", type=float, default=None)
    parser.add_argument("--ipopt-ma57-pre-alloc", type=float, default=None)
    parser.add_argument("--ipopt-ma57-block-size", type=int, default=None)
    parser.add_argument("--ipopt-ma57-node-amalgamation", type=int, default=None)
    parser.add_argument(
        "--ipopt-ma57-small-pivot-flag",
        type=int,
        choices=(0, 1),
        default=None,
    )
    parser.add_argument(
        "--ipopt-c-compile",
        action="store_true",
        help=(
            "Generate and compile the CasADi NLP before IPOPT solves. Compare "
            "hot window times separately from the one-off compilation cost."
        ),
    )
    parser.add_argument(
        "--ipopt-dual-warm-start-mode",
        choices=("off", "constraints", "bounds", "all"),
        default="bounds",
        help=(
            "Reuse no IPOPT duals, constraint multipliers, bound multipliers, "
            "or both between receding-horizon windows."
        ),
    )
    parser.add_argument("--nlp-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--primal-feasibility-threshold",
        type=float,
        default=None,
        help=(
            "Common absolute primal-feasibility threshold used to validate "
            "windows independently of solver-specific convergence tolerances."
        ),
    )
    parser.add_argument(
        "--fatrop-max-iter",
        type=int,
        default=1000,
        help="Maximum Fatrop iterations per RHO (CasADi 3.7.2 limits this to 1000).",
    )
    parser.add_argument(
        "--fatrop-structure-detection",
        choices=("auto", "manual"),
        default="auto",
        help=(
            "CasADi Fatrop OCP-structure detection. Automatic detection is the "
            "portable benchmark default."
        ),
    )
    parser.add_argument(
        "--fatrop-c-compile",
        action="store_true",
        help="Experimentally generate and compile the CasADi NLP used by Fatrop.",
    )
    parser.add_argument(
        "--fatrop-bound-tightening-factor",
        type=float,
        default=1e-8,
        help=(
            "Compensatory tightening of Fatrop interval bounds. The default "
            "offsets its native 1e-8 relative relaxation."
        ),
    )
    parser.add_argument("--fatrop-print-level", type=int, default=0)
    parser.add_argument(
        "--fatrop-state-scaling",
        choices=("none", "full"),
        default="none",
        help=(
            "Fatrop-specific state scaling. Automatic OCP structure detection "
            "currently requires 'none'; 'full' is retained for diagnostics."
        ),
    )
    parser.add_argument(
        "--fatrop-dual-warm-start-mode",
        choices=("off", "constraints", "bounds", "all"),
        default="off",
        help=(
            "Reuse no Fatrop multipliers by default; the shifted time-major "
            "multiplier blocks have not yet been independently validated."
        ),
    )
    parser.add_argument("--madnlp-max-iter", type=int, default=2000)
    parser.add_argument(
        "--madnlp-linear-solver",
        default=None,
        choices=(
            "mumps",
            "umfpack",
            "lapack_cpu",
            "pardiso_mkl",
            "cudss",
            "lapack_gpu",
            "cucholesky",
        ),
        help=(
            "MadNLP C-runtime linear solver. The default is mumps; pardiso_mkl "
            "requires the x86-64 libMad MKL runtime, and GPU choices require a "
            "compatible runtime and runner."
        ),
    )
    parser.add_argument(
        "--madnlp-c-compile",
        action="store_true",
        help="Experimentally generate and compile the CasADi NLP used by MadNLP.",
    )
    parser.add_argument(
        "--madnlp-dual-warm-start-mode",
        choices=("off", "constraints", "bounds", "all"),
        default="off",
        help=(
            "Reuse no MadNLP multipliers by default because multiplier blocks "
            "are not yet shifted with the receding horizon."
        ),
    )
    parser.add_argument("--alpaqa-max-iter", type=int, default=2000)
    parser.add_argument("--alpaqa-alm-max-iter", type=int, default=None)
    parser.add_argument(
        "--alpaqa-dual-warm-start-mode",
        choices=("off", "constraints"),
        default="constraints",
    )
    parser.add_argument("--alpaqa-lbfgs-memory", type=int, default=20)
    parser.add_argument("--alpaqa-max-wall-time", type=float, default=None)
    parser.add_argument("--alpaqa-initial-penalty", type=float, default=None)
    parser.add_argument("--alpaqa-initial-tolerance", type=float, default=None)
    parser.add_argument("--alpaqa-penalty-update-factor", type=float, default=None)
    parser.add_argument("--alpaqa-maximum-penalty", type=float, default=None)
    parser.add_argument("--alpaqa-panoc-max-wall-time", type=float, default=None)
    parser.add_argument("--alpaqa-max-no-progress", type=int, default=None)
    optional_seed_group = parser.add_mutually_exclusive_group()
    optional_seed_group.add_argument(
        "--optional-nlp-periodic-ipopt-hot-start",
        dest="optional_nlp_periodic_ipopt_hot_start",
        action="store_true",
        help=(
            "Seed the first Fatrop/MadNLP/Alpaqa window with a feasibility-certified "
            "periodic IPOPT solution (default)."
        ),
    )
    optional_seed_group.add_argument(
        "--no-optional-nlp-periodic-ipopt-hot-start",
        dest="optional_nlp_periodic_ipopt_hot_start",
        action="store_false",
        help="Start Fatrop/MadNLP/Alpaqa directly from the projected standard warmup.",
    )
    parser.set_defaults(optional_nlp_periodic_ipopt_hot_start=True)
    parser.add_argument(
        "--shared-transfer-full-dynamics-rollout",
        action="store_true",
        help=(
            "Apply the same complete-dynamics RK4 rollout to the appended cycle "
            "for IPOPT and ACADOS."
        ),
    )
    parser.add_argument(
        "--shared-transfer-contact-projection",
        action="store_true",
        help=(
            "Project free full q/qdot components onto the contact manifold "
            "between RHO solves for every selected solver."
        ),
    )
    parser.add_argument(
        "--shared-transfer-contact-projection-mode",
        choices=("position", "position_velocity"),
        default="position",
    )
    parser.add_argument(
        "--shared-transfer-phase-one",
        action="store_true",
        help="Apply the same bounded phase-I projection between windows for both solvers.",
    )
    parser.add_argument(
        "--acados-transfer-phase-one",
        action="store_true",
        help=(
            "Apply the bounded phase-I transfer projection only to ACADOS, "
            "leaving the historical IPOPT reference unchanged."
        ),
    )
    parser.add_argument(
        "--acados-transfer-phase-one-mode",
        choices=("all", "mechanical"),
        default="all",
    )
    parser.add_argument(
        "--acados-transfer-phase-one-lookback-nodes",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--acados-cyclical-transfer-mode",
        choices=("extrapolate", "repeat"),
        default="extrapolate",
        help="Construct the appended ACADOS cycle by extrapolation or repetition.",
    )
    parser.add_argument(
        "--acados-transfer-phase-one-proximity-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--acados-transfer-phase-one-defect-weight",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--acados-transfer-phase-one-substeps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--acados-transfer-phase-one-max-state-change",
        type=float,
        default=None,
    )
    for block in ("q", "qdot", "fes"):
        parser.add_argument(
            f"--acados-transfer-phase-one-max-{block}-change",
            type=float,
            default=None,
        )
    parser.add_argument(
        "--acados-transfer-bound-homotopy",
        action="store_true",
        help=(
            "Restore the transferred ACADOS trajectory through relaxed state "
            "bounds before the next fatigue-optimal RHO solve."
        ),
    )
    parser.add_argument(
        "--acados-transfer-bound-homotopy-fractions",
        type=parse_transfer_bound_homotopy_fractions,
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument(
        "--acados-transfer-bound-homotopy-padding",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--acados-transfer-bound-homotopy-iterations",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--acados-transfer-bound-homotopy-tolerance",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--acados-transfer-bound-homotopy-solver-tolerance",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--acados-transfer-bound-homotopy-min-fraction-step",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--acados-transfer-bound-homotopy-max-refinements",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--shared-initial-phase-one",
        action="store_true",
        help="Apply the same complete-dynamics phase-I projection before the first solve.",
    )
    parser.add_argument("--shared-transfer-rollout-substeps", type=int, default=5)
    parser.add_argument(
        "--shared-transfer-rollout-max-bound-violation",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--acados-transfer-select-projected-candidate",
        action="store_true",
    )
    parser.add_argument(
        "--acados-transfer-selector-max-q-bound-violation-rad",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--acados-transfer-selector-max-qdot-bound-violation-rad-s",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--acados-transfer-selector-max-other-scaled-bound-violation",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--acados-transfer-selector-max-scaled-q-defect",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--acados-transfer-selector-max-scaled-qdot-defect",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--acados-transfer-selector-improvement-ratio",
        type=float,
        default=0.95,
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
        "--shared-transfer-ding-force-compensation-substeps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--shared-transfer-ding-force-compensation-iterations",
        type=int,
        default=20,
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
        "--benchmark-profile",
        dest="ipopt_profile",
        choices=(
            "historical",
            "periodic_collocation",
            "periodic-collocation",
            "scientific_radau4",
            "scientific-radau4",
            "scientific_radau5",
            "scientific-radau5",
            "scientific_radau6",
            "scientific-radau6",
            "acados_like",
            "acados-like",
        ),
        default="historical",
        help=(
            "Base IPOPT configuration. 'historical' keeps the robust reference "
            "problem; 'periodic_collocation' isolates the periodic dynamics and "
            "constant torque with robust collocation; the scientific Radau-4/5/6 "
            "profiles fix the corrected SX contracts (Radau-5 is the candidate; "
            "4 and 6 are diagnostics); 'acados_like' additionally "
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
    assisted_hot_start_group = parser.add_mutually_exclusive_group()
    assisted_hot_start_group.add_argument(
        "--acados-assisted-hot-start",
        dest="acados_assisted_hot_start",
        action="store_true",
        default=True,
    )
    assisted_hot_start_group.add_argument(
        "--disable-acados-assisted-hot-start",
        dest="acados_assisted_hot_start",
        action="store_false",
    )
    parser.add_argument(
        "--acados-control-homotopy-radii",
        type=parse_control_homotopy_radii,
        default=None,
    )
    parser.add_argument(
        "--acados-control-homotopy-tolerance",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--acados-control-homotopy-stage-iterations",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--acados-control-homotopy-max-restarts",
        type=int,
        default=1,
    )
    retained_radius_group = parser.add_mutually_exclusive_group()
    retained_radius_group.add_argument(
        "--acados-control-homotopy-keep-final-radius",
        dest="acados_control_homotopy_keep_final_radius",
        action="store_true",
        default=None,
    )
    retained_radius_group.add_argument(
        "--acados-control-homotopy-release-final-radius",
        dest="acados_control_homotopy_keep_final_radius",
        action="store_false",
    )
    parser.add_argument(
        "--acados-control-homotopy-window-growth",
        type=float,
        default=1.0,
        help="Factor applied to the retained control radius after each RHO window.",
    )
    parser.add_argument(
        "--acados-control-homotopy-window-max-radius",
        type=float,
        default=None,
        help="Maximum retained control radius, in seconds, after inter-window growth.",
    )
    parser.add_argument(
        "--acados-integrator-type",
        choices=("ERK", "IRK", "DISCRETE"),
        default="IRK",
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
    parser.add_argument("--acados-newton-iter", type=int, default=5)
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
        "--wheel-qdot-regularization-target",
        type=float,
        default=-float(2 * np.pi),
    )
    parser.add_argument("--wheel-qdot-bound-margin", type=float, default=3.0)
    parser.add_argument(
        "--acados-wheel-qdot-fast-bound-margin",
        type=float,
        default=None,
        help=(
            "Tighten only the ACADOS lower/fast wheel-speed state bound; "
            "the physical audit keeps --wheel-qdot-bound-margin."
        ),
    )
    parser.add_argument(
        "--acados-wheel-qdot-slow-bound-margin",
        type=float,
        default=None,
        help=(
            "Optional ACADOS upper/slow wheel-speed state margin; defaults "
            "to the physical audit margin."
        ),
    )
    parser.add_argument(
        "--terminal-qdot-regularization-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--terminal-qdot-regularization-target-source",
        choices=("initial", "previous"),
        default="previous",
    )
    parser.add_argument(
        "--first-node-wheel-q-slack",
        type=float,
        default=0.0,
        help=(
            "Crank-angle continuity slack between consecutive RHO windows. "
            "Zero preserves the executed cycle boundary exactly."
        ),
    )
    parser.add_argument(
        "--terminal-wheel-q-slack",
        "--acados-terminal-wheel-q-slack",
        dest="acados_terminal_wheel_q_slack",
        type=float,
        default=0.002,
        help=(
            "Backend-independent terminal crank-angle slack in rad around the "
            "absolute initial-angle plus signed-cycle-count reference."
        ),
    )
    parser.add_argument(
        "--state-scaling", choices=("none", "fes", "full"), default="full"
    )
    parser.add_argument(
        "--acados-state-scaling", choices=("none", "fes", "full"), default=None
    )
    parser.add_argument("--pulse-width-scaling", type=float, default=1 / 400)
    parser.add_argument(
        "--pulse-width-active-set",
        choices=("none", "historical", "warmup"),
        default="none",
    )
    parser.add_argument("--pulse-width-active-threshold", type=float, default=0.01)
    parser.add_argument("--pulse-width-active-margin", type=int, default=3)
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
        "--acados-proximal-control-restart-feasibility-factor",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--acados-proximal-control-try-next-weight-on-failure",
        action="store_true",
    )
    parser.add_argument(
        "--continue-after-acados-transfer-failure",
        action="store_true",
    )
    parser.add_argument(
        "--acados-transfer-mechanical-restoration",
        action="store_true",
    )
    parser.add_argument(
        "--acados-transfer-mechanical-control-radius",
        type=float,
        default=5e-5,
    )
    parser.add_argument(
        "--acados-transfer-mechanical-regularization",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--acados-transfer-mechanical-substeps",
        type=int,
        default=5,
    )
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
        "--acados-transfer-active-set-guard-radius",
        type=float,
        default=None,
        help=(
            "Locally enlarge the ACADOS PW trust region at phase-aligned "
            "recruitment transitions."
        ),
    )
    parser.add_argument(
        "--acados-transfer-active-set-guard-margin",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--acados-transfer-active-set-threshold",
        type=float,
        default=1e-6,
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
    parser.add_argument("--acados-qp-cond-n", type=int, default=None)
    parser.add_argument(
        "--acados-qp-warm-start-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=0,
    )
    parser.add_argument("--acados-warm-start-first-qp", action="store_true")
    parser.add_argument("--acados-warm-start-first-qp-from-nlp", action="store_true")
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
        choices=("SQP", "SQP_RTI", "SQP_WITH_FEASIBLE_QP"),
        default="SQP",
    )
    parser.add_argument(
        "--acados-search-direction-mode",
        choices=("NOMINAL_QP", "BYRD_OMOJOKUN", "FEASIBILITY_QP"),
        default="NOMINAL_QP",
    )
    parser.add_argument(
        "--acados-globalization",
        choices=(
            "FIXED_STEP",
            "MERIT_BACKTRACKING",
            "FUNNEL_L1PEN_LINESEARCH",
        ),
        default="FUNNEL_L1PEN_LINESEARCH",
    )
    parser.add_argument("--acados-fixed-step-length", type=float, default=1.0)
    parser.add_argument(
        "--acados-nlp-qp-tol-strategy",
        choices=(
            "FIXED_QP_TOL",
            "ADAPTIVE_CURRENT_RES_JOINT",
            "ADAPTIVE_QPSCALING",
        ),
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
    parser.add_argument("--acados-store-iterates", action="store_true")
    parser.add_argument("--acados-maxiter-retries", type=int, default=0)
    parser.add_argument(
        "--acados-maxiter-retry-iterations", type=int, default=20
    )
    parser.add_argument(
        "--acados-maxiter-retry-feasibility-tolerance",
        type=float,
        default=2.5e-3,
    )
    parser.add_argument("--acados-reset-solver-before-solve", action="store_true")
    parser.add_argument("--acados-check-reuse-possible", action="store_true")
    parser.add_argument("--acados-code-reuse-tolerance", type=float, default=1e-12)
    parser.add_argument("--acados-with-anderson-acceleration", action="store_true")
    parser.add_argument(
        "--acados-anderson-activation-threshold", type=float, default=0.1
    )
    parser.add_argument(
        "--acados-byrd-omojokon-slack-relaxation-factor",
        type=float,
        default=1.00001,
    )
    parser.add_argument("--acados-project-qdot-from-q", action="store_true")
    parser.add_argument("--acados-diagnostics", action="store_true")
    parser.add_argument(
        "--initial-guess-diagnostics",
        action="store_true",
        help="Store solver-neutral primal-seed defect diagnostics for every backend.",
    )
    parser.add_argument(
        "--exact-initial-nlp-audit",
        action="store_true",
        help=(
            "Evaluate canonical g(x0) immediately before each CasADi NLP solve; "
            "disabled for ACADOS and excluded from solver timing."
        ),
    )
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
        "--periodic-fes-warmup-projection-defect-weight",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--periodic-fes-warmup-projection-trust-radius",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--periodic-fes-warmup-projection-max-iterations",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--periodic-fes-warmup-force-projection-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--periodic-fes-warmup-force-qdot-defect-limit",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--periodic-fes-warmup-force-adaptive-steps", type=int, default=10
    )
    periodic_refinement_group = parser.add_mutually_exclusive_group()
    periodic_refinement_group.add_argument(
        "--periodic-ipopt-refinement",
        dest="periodic_ipopt_refinement",
        action="store_true",
        default=True,
        help="Run the periodic IPOPT refinement before ACADOS (enabled by default).",
    )
    periodic_refinement_group.add_argument(
        "--disable-periodic-ipopt-refinement",
        dest="periodic_ipopt_refinement",
        action="store_false",
        help="Skip the periodic IPOPT refinement before ACADOS.",
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
        default=True,
        help=(
            "Build the auxiliary periodic IPOPT refinement with SX graphs "
            "(the benchmark default and supported production mode)."
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
    parser.add_argument(
        "--validate-integrator-maps",
        action="store_true",
        help=(
            "Reintegrate the exported RHO trace with the common high-accuracy "
            "DOP853 audit used by the scientific collocation gate."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for a compact JSON summary of every selected solver.",
    )
    parser.add_argument(
        "--receding-horizon-solution-output",
        type=Path,
        default=None,
        help=(
            "Save the complete validated RHO trajectory as a multi-cycle .npz "
            "initial solution."
        ),
    )
    parser.add_argument(
        "--allow-partial-receding-horizon-solution-output",
        action="store_true",
        help=(
            "Export the longest physically valid RHO prefix when a later "
            "window fails."
        ),
    )
    return parser


if __name__ == "__main__":
    args = build_cli().parse_args()
    main(
        solvers=args.solvers,
        single_shot=args.single_shot,
        output_json=args.output_json,
        objective=args.objective,
        objective_shape=args.objective_shape,
        cycles_per_window=args.cycles_per_window,
        stimulations_per_cycle=args.stimulations_per_cycle,
        n_windows=args.n_windows,
        mechanical_formulation=args.mechanical_formulation,
        reduced_cycling_profile=args.reduced_cycling_profile,
        experimental_reduced_acados=args.experimental_reduced_acados,
        full_contact_constraints_terminal=(args.full_contact_constraints_terminal),
        full_contact_position_terminal=args.full_contact_position_terminal,
        full_contact_position_tolerance=(args.full_contact_position_tolerance),
        full_contact_constraints_all_nodes=(args.full_contact_constraints_all_nodes),
        full_contact_position_all_nodes=(args.full_contact_position_all_nodes),
        n_threads=args.n_threads,
        compact_rho_output=args.compact_rho_output,
        validate_integrator_maps=args.validate_integrator_maps,
        resistive_torque=args.resistive_torque,
        acados_dir=args.acados_dir,
        codegen_tag=args.codegen_tag,
        ipopt_max_iter=args.ipopt_max_iter,
        standard_warmup_max_iter=args.standard_warmup_max_iter,
        ipopt_linear_solver=args.ipopt_linear_solver,
        warmup_ipopt_linear_solver=args.warmup_ipopt_linear_solver,
        standard_warmup_seed=args.standard_warmup_seed,
        standard_warmup_seed_continuation=(args.standard_warmup_seed_continuation),
        legacy_standard_warmup_seed_signed_torque=(
            args.legacy_standard_warmup_seed_signed_torque
        ),
        common_initial_solution=args.common_initial_solution,
        adopt_common_initial_solution_warmup_cycles=(
            args.adopt_common_initial_solution_warmup_cycles
        ),
        common_initial_solution_output=args.common_initial_solution_output,
        receding_horizon_solution_output=(args.receding_horizon_solution_output),
        allow_partial_receding_horizon_solution_output=(
            args.allow_partial_receding_horizon_solution_output
        ),
        ipopt_hsl_library=args.ipopt_hsl_library,
        ipopt_c_compile=args.ipopt_c_compile,
        ipopt_print_level=args.ipopt_print_level,
        ipopt_print_timing_statistics=args.ipopt_print_timing_statistics,
        ipopt_linear_system_scaling=args.ipopt_linear_system_scaling,
        ipopt_linear_scaling_on_demand=args.ipopt_linear_scaling_on_demand,
        ipopt_ma57_automatic_scaling=args.ipopt_ma57_automatic_scaling,
        ipopt_ma57_pivot_order=args.ipopt_ma57_pivot_order,
        ipopt_ma57_pivtol=args.ipopt_ma57_pivtol,
        ipopt_ma57_pivtolmax=args.ipopt_ma57_pivtolmax,
        ipopt_ma57_pre_alloc=args.ipopt_ma57_pre_alloc,
        ipopt_ma57_block_size=args.ipopt_ma57_block_size,
        ipopt_ma57_node_amalgamation=args.ipopt_ma57_node_amalgamation,
        ipopt_ma57_small_pivot_flag=args.ipopt_ma57_small_pivot_flag,
        ipopt_dual_warm_start_mode=args.ipopt_dual_warm_start_mode,
        nlp_tolerance=args.nlp_tolerance,
        primal_feasibility_threshold=args.primal_feasibility_threshold,
        fatrop_max_iter=args.fatrop_max_iter,
        fatrop_dual_warm_start_mode=args.fatrop_dual_warm_start_mode,
        fatrop_c_compile=args.fatrop_c_compile,
        fatrop_structure_detection=args.fatrop_structure_detection,
        fatrop_bound_tightening_factor=args.fatrop_bound_tightening_factor,
        fatrop_print_level=args.fatrop_print_level,
        fatrop_state_scaling=args.fatrop_state_scaling,
        madnlp_max_iter=args.madnlp_max_iter,
        madnlp_dual_warm_start_mode=args.madnlp_dual_warm_start_mode,
        madnlp_c_compile=args.madnlp_c_compile,
        madnlp_linear_solver=args.madnlp_linear_solver,
        alpaqa_max_iter=args.alpaqa_max_iter,
        alpaqa_alm_max_iter=args.alpaqa_alm_max_iter,
        alpaqa_dual_warm_start_mode=args.alpaqa_dual_warm_start_mode,
        alpaqa_lbfgs_memory=args.alpaqa_lbfgs_memory,
        alpaqa_max_wall_time=args.alpaqa_max_wall_time,
        alpaqa_initial_penalty=args.alpaqa_initial_penalty,
        alpaqa_initial_tolerance=args.alpaqa_initial_tolerance,
        alpaqa_penalty_update_factor=args.alpaqa_penalty_update_factor,
        alpaqa_maximum_penalty=args.alpaqa_maximum_penalty,
        alpaqa_panoc_max_wall_time=args.alpaqa_panoc_max_wall_time,
        alpaqa_max_no_progress=args.alpaqa_max_no_progress,
        optional_nlp_periodic_ipopt_hot_start=(
            args.optional_nlp_periodic_ipopt_hot_start
        ),
        acados_max_iter=args.acados_max_iter,
        acados_assisted_hot_start=args.acados_assisted_hot_start,
        acados_control_homotopy_radii=args.acados_control_homotopy_radii,
        acados_control_homotopy_tolerance=(args.acados_control_homotopy_tolerance),
        acados_control_homotopy_stage_iterations=(
            args.acados_control_homotopy_stage_iterations
        ),
        acados_control_homotopy_max_restarts=(
            args.acados_control_homotopy_max_restarts
        ),
        acados_control_homotopy_keep_final_radius=(
            args.acados_control_homotopy_keep_final_radius
        ),
        acados_control_homotopy_window_growth=(
            args.acados_control_homotopy_window_growth
        ),
        acados_control_homotopy_window_max_radius=(
            args.acados_control_homotopy_window_max_radius
        ),
        control_regularization_weight=args.control_regularization_weight,
        acados_control_regularization_weight=args.acados_control_regularization_weight,
        control_regularization_target=args.control_regularization_target,
        control_regularization_target_source=args.control_regularization_target_source,
        acados_control_regularization_target_source=args.acados_control_regularization_target_source,
        wheel_qdot_regularization_weight=args.wheel_qdot_regularization_weight,
        acados_wheel_qdot_regularization_weight=args.acados_wheel_qdot_regularization_weight,
        wheel_qdot_regularization_target=args.wheel_qdot_regularization_target,
        wheel_qdot_bound_margin=args.wheel_qdot_bound_margin,
        acados_wheel_qdot_fast_bound_margin=(
            args.acados_wheel_qdot_fast_bound_margin
        ),
        acados_wheel_qdot_slow_bound_margin=(
            args.acados_wheel_qdot_slow_bound_margin
        ),
        terminal_qdot_regularization_weight=(args.terminal_qdot_regularization_weight),
        terminal_qdot_regularization_target_source=(
            args.terminal_qdot_regularization_target_source
        ),
        first_node_wheel_q_slack=args.first_node_wheel_q_slack,
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
        pulse_width_active_set=args.pulse_width_active_set,
        pulse_width_active_threshold=args.pulse_width_active_threshold,
        pulse_width_active_margin=args.pulse_width_active_margin,
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
        acados_proximal_control_restart_feasibility_factor=(
            args.acados_proximal_control_restart_feasibility_factor
        ),
        acados_proximal_control_try_next_weight_on_failure=(
            args.acados_proximal_control_try_next_weight_on_failure
        ),
        continue_after_acados_transfer_failure=(
            args.continue_after_acados_transfer_failure
        ),
        acados_transfer_mechanical_restoration=(
            args.acados_transfer_mechanical_restoration
        ),
        acados_transfer_mechanical_control_radius=(
            args.acados_transfer_mechanical_control_radius
        ),
        acados_transfer_mechanical_regularization=(
            args.acados_transfer_mechanical_regularization
        ),
        acados_transfer_mechanical_substeps=(args.acados_transfer_mechanical_substeps),
        acados_transfer_sqp_restarts=args.acados_transfer_sqp_restarts,
        acados_transfer_sqp_restart_iterations=(
            args.acados_transfer_sqp_restart_iterations
        ),
        acados_transfer_sqp_restart_feasibility_tolerance=(
            args.acados_transfer_sqp_restart_feasibility_tolerance
        ),
        acados_transfer_active_set_guard_radius=(
            args.acados_transfer_active_set_guard_radius
        ),
        acados_transfer_active_set_guard_margin=(
            args.acados_transfer_active_set_guard_margin
        ),
        acados_transfer_active_set_threshold=(
            args.acados_transfer_active_set_threshold
        ),
        acados_fes_state_trust_radius=args.acados_fes_state_trust_radius,
        acados_fatigue_warmstart_mode=args.acados_fatigue_warmstart_mode,
        acados_tolerance=args.acados_tolerance,
        acados_stationarity_tolerance=args.acados_stationarity_tolerance,
        acados_qp_iter_max=args.acados_qp_iter_max,
        acados_qp_cond_n=args.acados_qp_cond_n,
        acados_qp_warm_start_level=args.acados_qp_warm_start_level,
        acados_warm_start_first_qp=args.acados_warm_start_first_qp,
        acados_warm_start_first_qp_from_nlp=(args.acados_warm_start_first_qp_from_nlp),
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
        acados_store_iterates=args.acados_store_iterates,
        acados_maxiter_retries=args.acados_maxiter_retries,
        acados_maxiter_retry_iterations=(args.acados_maxiter_retry_iterations),
        acados_maxiter_retry_feasibility_tolerance=(
            args.acados_maxiter_retry_feasibility_tolerance
        ),
        acados_reset_solver_before_solve=args.acados_reset_solver_before_solve,
        acados_check_reuse_possible=args.acados_check_reuse_possible,
        acados_code_reuse_tolerance=args.acados_code_reuse_tolerance,
        acados_with_anderson_acceleration=(args.acados_with_anderson_acceleration),
        acados_anderson_activation_threshold=(
            args.acados_anderson_activation_threshold
        ),
        acados_byrd_omojokon_slack_relaxation_factor=(
            args.acados_byrd_omojokon_slack_relaxation_factor
        ),
        acados_project_qdot_from_q=args.acados_project_qdot_from_q,
        shared_transfer_full_dynamics_rollout=(
            args.shared_transfer_full_dynamics_rollout
        ),
        shared_transfer_contact_projection=(args.shared_transfer_contact_projection),
        shared_transfer_contact_projection_mode=(
            args.shared_transfer_contact_projection_mode
        ),
        shared_transfer_phase_one=args.shared_transfer_phase_one,
        acados_transfer_phase_one=args.acados_transfer_phase_one,
        acados_transfer_phase_one_mode=args.acados_transfer_phase_one_mode,
        acados_transfer_phase_one_lookback_nodes=(
            args.acados_transfer_phase_one_lookback_nodes
        ),
        acados_cyclical_transfer_mode=args.acados_cyclical_transfer_mode,
        acados_transfer_phase_one_proximity_weight=(
            args.acados_transfer_phase_one_proximity_weight
        ),
        acados_transfer_phase_one_defect_weight=(
            args.acados_transfer_phase_one_defect_weight
        ),
        acados_transfer_phase_one_substeps=(args.acados_transfer_phase_one_substeps),
        acados_transfer_phase_one_max_state_change=(
            args.acados_transfer_phase_one_max_state_change
        ),
        acados_transfer_phase_one_max_q_change=(
            args.acados_transfer_phase_one_max_q_change
        ),
        acados_transfer_phase_one_max_qdot_change=(
            args.acados_transfer_phase_one_max_qdot_change
        ),
        acados_transfer_phase_one_max_fes_change=(
            args.acados_transfer_phase_one_max_fes_change
        ),
        acados_transfer_bound_homotopy=(args.acados_transfer_bound_homotopy),
        acados_transfer_bound_homotopy_fractions=(
            args.acados_transfer_bound_homotopy_fractions
        ),
        acados_transfer_bound_homotopy_padding=(
            args.acados_transfer_bound_homotopy_padding
        ),
        acados_transfer_bound_homotopy_iterations=(
            args.acados_transfer_bound_homotopy_iterations
        ),
        acados_transfer_bound_homotopy_tolerance=(
            args.acados_transfer_bound_homotopy_tolerance
        ),
        acados_transfer_bound_homotopy_solver_tolerance=(
            args.acados_transfer_bound_homotopy_solver_tolerance
        ),
        acados_transfer_bound_homotopy_min_fraction_step=(
            args.acados_transfer_bound_homotopy_min_fraction_step
        ),
        acados_transfer_bound_homotopy_max_refinements=(
            args.acados_transfer_bound_homotopy_max_refinements
        ),
        shared_initial_phase_one=args.shared_initial_phase_one,
        shared_transfer_rollout_substeps=args.shared_transfer_rollout_substeps,
        shared_transfer_rollout_max_bound_violation=(
            args.shared_transfer_rollout_max_bound_violation
        ),
        acados_transfer_select_projected_candidate=(
            args.acados_transfer_select_projected_candidate
        ),
        acados_transfer_selector_max_q_bound_violation_rad=(
            args.acados_transfer_selector_max_q_bound_violation_rad
        ),
        acados_transfer_selector_max_qdot_bound_violation_rad_s=(
            args.acados_transfer_selector_max_qdot_bound_violation_rad_s
        ),
        acados_transfer_selector_max_other_scaled_bound_violation=(
            args.acados_transfer_selector_max_other_scaled_bound_violation
        ),
        acados_transfer_selector_max_scaled_q_defect=(
            args.acados_transfer_selector_max_scaled_q_defect
        ),
        acados_transfer_selector_max_scaled_qdot_defect=(
            args.acados_transfer_selector_max_scaled_qdot_defect
        ),
        acados_transfer_selector_improvement_ratio=(
            args.acados_transfer_selector_improvement_ratio
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
        acados_newton_iter=args.acados_newton_iter,
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
        initial_guess_diagnostics=args.initial_guess_diagnostics,
        exact_initial_nlp_audit=args.exact_initial_nlp_audit,
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
