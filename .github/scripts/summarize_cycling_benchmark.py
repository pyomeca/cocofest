#!/usr/bin/env python3
"""Aggregate per-solver cycling benchmark JSON files into Kevin-ready artifacts."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path

COMPARABILITY_FIELDS = (
    "objective",
    "objective_shape",
    "model_formulation",
    "mechanical_formulation",
    "torque_application",
    "ode_solver",
    "collocation_method",
    "calcium_forcing_formulation",
    "ding_sum_stim_truncation",
    "activate_force_length_relationship",
    "activate_force_velocity_relationship",
    "activate_passive_force_relationship",
    "control_decisions_per_cycle",
    "wheel_qdot_regularization_target",
    "wheel_qdot_bound_margin",
    "collocation_degree",
    "acados_terminal_wheel_q_slack",
    "cycles_per_window",
    "stimulations_per_cycle",
    "n_windows",
    "n_threads",
    "constant_crank_torque",
    "crank_torque_role",
    "primal_feasibility_threshold",
    "use_sx",
)
DEFAULT_EXPECTED_SOLVERS = ("ipopt", "madnlp")
DEFAULT_EXPECTED_CASES = (
    "ipopt/full",
    "madnlp-mumps/full",
    "acados/full",
    "ipopt/reduced",
    "madnlp-mumps/reduced",
    "acados/reduced",
)


def _finite(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _fmt(value, digits: int = 3) -> str:
    value = _finite(value)
    return "—" if value is None else f"{value:.{digits}f}"


def _fmt_scientific(value, digits: int = 1) -> str:
    value = _finite(value)
    return "—" if value is None else f"{value:.{digits}e}"


def _mechanical_formulation(entry: dict) -> str:
    return str(
        entry.get("configuration", {}).get("mechanical_formulation") or "full"
    ).lower()


def _entry_case(entry: dict) -> str:
    return f"{_solver_variant(entry)}/{_mechanical_formulation(entry)}"


def _entry_base_case(entry: dict) -> str:
    """Return the physical solver case independently of evaluator codegen."""

    return _entry_case(entry).replace("-compiled/", "/")


def _solver_comparison_family(entry: dict) -> str:
    """Pair full/reduced physics even when only one evaluator is C-compiled."""

    return _solver_variant(entry).replace("-compiled", "")


def _solver_variant(entry: dict) -> str:
    solver = entry["result"].get("solver", "unknown").lower()
    configuration = entry.get("configuration", {})
    if solver == "fatrop":
        transcription = str(configuration.get("ode_solver") or "unknown").lower()
        compilation = (
            "-compiled" if configuration.get("fatrop_c_compile") is True else ""
        )
        return f"fatrop-{transcription}{compilation}"
    if solver == "ipopt":
        variant = (
            "ipopt-compiled"
            if configuration.get("ipopt_c_compile") is True
            else "ipopt"
        )
        degree = configuration.get("collocation_degree")
        if degree not in (None, 3):
            variant = variant.replace("ipopt", f"ipopt-radau{degree}", 1)
        return variant
    if solver != "madnlp":
        return solver
    linear_solver = str(
        entry.get("configuration", {}).get("madnlp_linear_solver") or ""
    ).lower()
    if linear_solver in {"pardiso_mkl", "pardisomklsolver"}:
        variant = "madnlp-pardiso"
    elif linear_solver in {"mumps", "mumpssolver"}:
        variant = "madnlp-mumps"
    else:
        variant = solver
    degree = configuration.get("collocation_degree")
    if degree not in (None, 3):
        variant += f"-radau{degree}"
    if configuration.get("madnlp_c_compile") is True:
        variant += "-compiled"
    return variant


def _entry_label(entry: dict) -> str:
    return _entry_case(entry).upper()


def _requested_rho_count(entry: dict) -> int | None:
    requested_cycles = entry.get("configuration", {}).get("n_windows")
    cycles_per_window = entry.get("configuration", {}).get("cycles_per_window") or 1
    try:
        requested_cycles = int(requested_cycles)
        cycles_per_window = int(cycles_per_window)
    except (TypeError, ValueError):
        return None
    return max(0, requested_cycles - cycles_per_window + 1)


def _ratio(numerator, denominator) -> float | None:
    numerator = _finite(numerator)
    denominator = _finite(denominator)
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def fatrop_internal_timing_rows(entry: dict) -> list[dict]:
    """Return one normalized Fatrop oracle-timing row per solved RHO."""

    if str(entry["result"].get("solver", "")).lower() != "fatrop":
        return []
    rows = []
    for stats in entry["result"].get("nlp_solver_stats") or []:
        native = stats.get("fatrop") or {}
        window = stats.get("window")
        iterations = native.get("iterations_count", stats.get("iter_count"))
        total = native.get("time_total", stats.get("t_wall_total"))
        hessian = native.get("eval_hess_time", stats.get("t_wall_nlp_hess_l"))
        jacobian = native.get("eval_jac_time", stats.get("t_wall_nlp_jac_g"))
        constraints = native.get("eval_cv_time", stats.get("t_wall_nlp_g"))
        structure = native.get("compute_sd_time")
        hessian_count = native.get("eval_hess_count", stats.get("n_call_nlp_hess_l"))
        jacobian_count = native.get("eval_jac_count", stats.get("n_call_nlp_jac_g"))
        rows.append(
            {
                "solver": "fatrop",
                "mechanical_formulation": _mechanical_formulation(entry),
                "case": _entry_case(entry),
                "window": window,
                "rho": int(window) + 1 if window is not None else None,
                "iterations": iterations,
                "total_wall_time_s": _finite(total),
                "hessian_wall_time_s": _finite(hessian),
                "jacobian_wall_time_s": _finite(jacobian),
                "constraint_wall_time_s": _finite(constraints),
                "structure_detection_wall_time_s": _finite(structure),
                "hessian_evaluations": hessian_count,
                "jacobian_evaluations": jacobian_count,
                "total_wall_time_per_iteration_s": _ratio(total, iterations),
                "hessian_wall_time_per_iteration_s": _ratio(hessian, iterations),
                "jacobian_wall_time_per_iteration_s": _ratio(jacobian, iterations),
                "hessian_wall_time_per_evaluation_s": _ratio(hessian, hessian_count),
                "jacobian_wall_time_per_evaluation_s": _ratio(jacobian, jacobian_count),
                "derivative_wall_time_fraction": _ratio(
                    sum(
                        value
                        for value in (_finite(hessian), _finite(jacobian))
                        if value is not None
                    ),
                    total,
                ),
            }
        )
    return rows


def fatrop_internal_timing_summary(entry: dict) -> dict | None:
    """Aggregate Fatrop derivative and structure-detection costs over a run."""

    rows = fatrop_internal_timing_rows(entry)
    if not rows:
        return None

    def total(field: str) -> float:
        return sum(
            value for row in rows if (value := _finite(row.get(field))) is not None
        )

    iterations = total("iterations")
    total_wall = total("total_wall_time_s")
    hessian = total("hessian_wall_time_s")
    jacobian = total("jacobian_wall_time_s")
    return {
        "case": _entry_case(entry),
        "rho_count": len(rows),
        "mean_iterations": iterations / len(rows) if rows else None,
        "total_wall_time_s": total_wall,
        "hessian_wall_time_s": hessian,
        "jacobian_wall_time_s": jacobian,
        "structure_detection_wall_time_s": total("structure_detection_wall_time_s"),
        "total_wall_time_per_iteration_s": _ratio(total_wall, iterations),
        "hessian_wall_time_per_iteration_s": _ratio(hessian, iterations),
        "jacobian_wall_time_per_iteration_s": _ratio(jacobian, iterations),
        "derivative_wall_time_fraction": _ratio(hessian + jacobian, total_wall),
    }


def load_benchmark_files(paths: list[Path]) -> list[dict]:
    entries = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        configurations = payload.get("configurations") or {}
        for result in payload.get("results") or []:
            solver = result["solver"]
            entries.append(
                {
                    "source": str(path),
                    "runtime": payload.get("runtime") or {},
                    "configuration": configurations.get(solver) or {},
                    "result": result,
                }
            )
    mechanics_order = {"full": 0, "reduced": 1}
    entries.sort(
        key=lambda entry: (
            mechanics_order.get(_mechanical_formulation(entry), 99),
            _solver_variant(entry),
        )
    )
    return entries


def configuration_mismatches(entries: list[dict]) -> list[dict]:
    if not entries:
        return []
    mismatches = []
    formulations = dict.fromkeys(_mechanical_formulation(entry) for entry in entries)
    for formulation in formulations:
        group = [
            entry for entry in entries if _mechanical_formulation(entry) == formulation
        ]
        reference = next(
            (entry for entry in group if entry["result"].get("solver") == "ipopt"),
            group[0],
        )
        for entry in group:
            for field in COMPARABILITY_FIELDS:
                expected = reference["configuration"].get(field)
                observed = entry["configuration"].get(field)
                if observed != expected:
                    mismatches.append(
                        {
                            "case": _entry_case(entry),
                            "reference_case": _entry_case(reference),
                            "field": field,
                            "expected": expected,
                            "observed": observed,
                        }
                    )
    return mismatches


def _pattern_comparison(reference: dict, compared: dict) -> dict | None:
    reference_values = reference.get("pulse_width_s") or []
    compared_values = compared.get("pulse_width_s") or []
    if len(reference_values) != len(compared_values) or not reference_values:
        return None
    differences = [
        float(compared_value) - float(reference_value)
        for reference_value, compared_value in zip(reference_values, compared_values)
    ]
    mae = sum(abs(value) for value in differences) / len(differences)
    rmse = math.sqrt(sum(value * value for value in differences) / len(differences))
    reference_mean = sum(reference_values) / len(reference_values)
    compared_mean = sum(compared_values) / len(compared_values)
    covariance = sum(
        (reference_value - reference_mean) * (compared_value - compared_mean)
        for reference_value, compared_value in zip(reference_values, compared_values)
    )
    reference_energy = sum((value - reference_mean) ** 2 for value in reference_values)
    compared_energy = sum((value - compared_mean) ** 2 for value in compared_values)
    denominator = math.sqrt(reference_energy * compared_energy)
    return {
        "sample_count": len(differences),
        "mean_absolute_error_s": mae,
        "root_mean_square_error_s": rmse,
        "maximum_absolute_error_s": max(abs(value) for value in differences),
        "mean_absolute_error_us": 1e6 * mae,
        "root_mean_square_error_us": 1e6 * rmse,
        "maximum_absolute_error_us": 1e6 * max(abs(value) for value in differences),
        "correlation": covariance / denominator if denominator else None,
    }


def _periodic_linear_interpolation(
    source_phase: list[float],
    source_values: list[float],
    target_phase: list[float],
) -> list[float] | None:
    if (
        len(source_phase) != len(source_values)
        or len(source_phase) < 2
        or not target_phase
    ):
        return None
    period = 2.0 * math.pi
    pairs = sorted(
        (float(phase) % period, float(value))
        for phase, value in zip(source_phase, source_values)
    )
    phases: list[float] = []
    values: list[float] = []
    for phase, value in pairs:
        if phases and math.isclose(phase, phases[-1], abs_tol=1e-12):
            values[-1] = 0.5 * (values[-1] + value)
        else:
            phases.append(phase)
            values.append(value)
    if len(phases) < 2:
        return None

    extended_phase = [phases[-1] - period, *phases, phases[0] + period]
    extended_values = [values[-1], *values, values[0]]
    interpolated = []
    for target in target_phase:
        target = float(target) % period
        upper = bisect.bisect_right(extended_phase, target)
        lower = max(0, upper - 1)
        upper = min(upper, len(extended_phase) - 1)
        phase_span = extended_phase[upper] - extended_phase[lower]
        if phase_span <= 0.0:
            interpolated.append(extended_values[lower])
            continue
        fraction = (target - extended_phase[lower]) / phase_span
        interpolated.append(
            extended_values[lower]
            + fraction * (extended_values[upper] - extended_values[lower])
        )
    return interpolated


def _phase_aligned_pattern_comparison(
    reference: dict,
    compared: dict,
    reference_phase: list[float],
    compared_phase: list[float],
) -> dict | None:
    reference_values = reference.get("pulse_width_s") or []
    compared_values = compared.get("pulse_width_s") or []
    if len(reference_phase) != len(reference_values):
        return None
    aligned_values = _periodic_linear_interpolation(
        compared_phase,
        compared_values,
        reference_phase,
    )
    if aligned_values is None:
        return None
    return _pattern_comparison(
        {"pulse_width_s": reference_values},
        {"pulse_width_s": aligned_values},
    )


def _paired_stimulation_comparisons(
    reference_entry: dict,
    compared_entry: dict,
    *,
    comparison_kind: str,
) -> list[dict]:
    rows = []
    reference_patterns = reference_entry["result"].get("stimulation_patterns") or {}
    compared_patterns = compared_entry["result"].get("stimulation_patterns") or {}
    for checkpoint, reference_snapshot in reference_patterns.items():
        compared_snapshot = compared_patterns.get(checkpoint) or {}
        base = {
            "comparison_kind": comparison_kind,
            "reference_case": _entry_case(reference_entry),
            "case": _entry_case(compared_entry),
            "solver": compared_entry["result"]["solver"],
            "mechanical_formulation": _mechanical_formulation(compared_entry),
            "checkpoint": checkpoint,
        }
        if not (
            reference_snapshot.get("available") and compared_snapshot.get("available")
        ):
            rows.append(
                {
                    **base,
                    "available": False,
                    "reason": (
                        reference_snapshot.get("reason")
                        or compared_snapshot.get("reason")
                        or "missing_snapshot"
                    ),
                }
            )
            continue
        common_muscles = sorted(
            set(reference_snapshot.get("muscles") or {})
            & set(compared_snapshot.get("muscles") or {})
        )
        reference_phase = reference_snapshot.get("crank_phase_rad") or []
        compared_phase = compared_snapshot.get("crank_phase_rad") or []
        if len(reference_phase) == len(compared_phase) and reference_phase:
            crank_phase_rmse = math.sqrt(
                sum(
                    math.atan2(
                        math.sin(float(compared) - float(reference)),
                        math.cos(float(compared) - float(reference)),
                    )
                    ** 2
                    for reference, compared in zip(reference_phase, compared_phase)
                )
                / len(reference_phase)
            )
        else:
            crank_phase_rmse = None
        for muscle in common_muscles:
            metrics = _pattern_comparison(
                reference_snapshot["muscles"][muscle],
                compared_snapshot["muscles"][muscle],
            )
            phase_aligned_metrics = _phase_aligned_pattern_comparison(
                reference_snapshot["muscles"][muscle],
                compared_snapshot["muscles"][muscle],
                reference_phase,
                compared_phase,
            )
            rows.append(
                {
                    **base,
                    "cycle": reference_snapshot.get("cycle"),
                    "muscle": muscle,
                    "available": metrics is not None,
                    "crank_phase_root_mean_square_error_rad": crank_phase_rmse,
                    **(metrics or {"reason": "incompatible_pattern_dimensions"}),
                    **(
                        {
                            f"phase_aligned_{key}": value
                            for key, value in phase_aligned_metrics.items()
                        }
                        if phase_aligned_metrics
                        else {}
                    ),
                }
            )
    return rows


def stimulation_comparisons(entries: list[dict]) -> list[dict]:
    rows = []
    for formulation in dict.fromkeys(
        _mechanical_formulation(entry) for entry in entries
    ):
        group = [
            entry for entry in entries if _mechanical_formulation(entry) == formulation
        ]
        reference_entry = next(
            (entry for entry in group if entry["result"].get("solver") == "ipopt"),
            None,
        )
        if reference_entry is None:
            continue
        for entry in group:
            if entry is reference_entry:
                continue
            rows.extend(
                _paired_stimulation_comparisons(
                    reference_entry,
                    entry,
                    comparison_kind="solver_within_formulation",
                )
            )
    return rows


def mechanical_stimulation_comparisons(entries: list[dict]) -> list[dict]:
    """Compare full and reduced mechanics for each solver at cycles 10 and 30."""

    rows = []
    for solver_family in dict.fromkeys(
        _solver_comparison_family(entry) for entry in entries
    ):
        full_entry = next(
            (
                entry
                for entry in entries
                if _solver_comparison_family(entry) == solver_family
                and _mechanical_formulation(entry) == "full"
            ),
            None,
        )
        reduced_entry = next(
            (
                entry
                for entry in entries
                if _solver_comparison_family(entry) == solver_family
                and _mechanical_formulation(entry) == "reduced"
            ),
            None,
        )
        if full_entry is None or reduced_entry is None:
            continue
        rows.extend(
            _paired_stimulation_comparisons(
                full_entry,
                reduced_entry,
                comparison_kind="reduced_against_full",
            )
        )
    return rows


def write_rho_csv(path: Path, entries: list[dict]) -> None:
    fieldnames = (
        "solver",
        "mechanical_formulation",
        "case",
        "rho",
        "status",
        "native_status",
        "solver_converged",
        "primal_feasible",
        "validated",
        "iterations",
        "objective",
        "solver_time_s",
        "wall_time_s",
        "effective_primal_infeasibility",
        "inf_pr_available",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            solver = entry["result"]["solver"]
            for window in entry["result"].get("windows") or []:
                feasibility = window.get("feasibility") or {}
                writer.writerow(
                    {
                        "solver": solver,
                        "mechanical_formulation": _mechanical_formulation(entry),
                        "case": _entry_case(entry),
                        **{
                            field: window.get(field)
                            for field in fieldnames
                            if field
                            not in {
                                "solver",
                                "mechanical_formulation",
                                "case",
                                "effective_primal_infeasibility",
                                "inf_pr_available",
                            }
                        },
                        "effective_primal_infeasibility": feasibility.get(
                            "effective_primal_infeasibility"
                        ),
                        "inf_pr_available": feasibility.get("inf_pr_available"),
                    }
                )


def write_fatrop_internal_timing_csv(path: Path, entries: list[dict]) -> None:
    fieldnames = (
        "solver",
        "mechanical_formulation",
        "case",
        "window",
        "rho",
        "iterations",
        "total_wall_time_s",
        "hessian_wall_time_s",
        "jacobian_wall_time_s",
        "constraint_wall_time_s",
        "structure_detection_wall_time_s",
        "hessian_evaluations",
        "jacobian_evaluations",
        "total_wall_time_per_iteration_s",
        "hessian_wall_time_per_iteration_s",
        "jacobian_wall_time_per_iteration_s",
        "hessian_wall_time_per_evaluation_s",
        "jacobian_wall_time_per_evaluation_s",
        "derivative_wall_time_fraction",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerows(fatrop_internal_timing_rows(entry))


def write_stimulation_csv(path: Path, entries: list[dict]) -> None:
    fieldnames = (
        "solver",
        "mechanical_formulation",
        "case",
        "cycle",
        "rho",
        "muscle",
        "stimulation_index",
        "phase_fraction",
        "crank_angle_rad",
        "crank_phase_rad",
        "crank_velocity_rad_s",
        "pulse_width_s",
        "pulse_width_us",
        "normalized_to_bounds",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            solver = entry["result"]["solver"]
            patterns = entry["result"].get("stimulation_patterns") or {}
            for snapshot in patterns.values():
                if not snapshot.get("available"):
                    continue
                for muscle, pattern in (snapshot.get("muscles") or {}).items():
                    values = pattern.get("pulse_width_s") or []
                    for index, pulse_width in enumerate(values):
                        writer.writerow(
                            {
                                "solver": solver,
                                "mechanical_formulation": _mechanical_formulation(
                                    entry
                                ),
                                "case": _entry_case(entry),
                                "cycle": snapshot.get("cycle"),
                                "rho": snapshot.get("rho"),
                                "muscle": muscle,
                                "stimulation_index": index + 1,
                                "phase_fraction": snapshot["phase_fraction"][index],
                                "crank_angle_rad": snapshot["crank_angle_rad"][index],
                                "crank_phase_rad": snapshot["crank_phase_rad"][index],
                                "crank_velocity_rad_s": (
                                    snapshot["crank_velocity_rad_s"][index]
                                    if snapshot.get("crank_velocity_rad_s")
                                    else None
                                ),
                                "pulse_width_s": pulse_width,
                                "pulse_width_us": pattern["pulse_width_us"][index],
                                "normalized_to_bounds": pattern["normalized_to_bounds"][
                                    index
                                ],
                            }
                        )


def write_muscle_fatigue_csv(path: Path, entries: list[dict]) -> None:
    fieldnames = (
        "solver",
        "mechanical_formulation",
        "case",
        "muscle",
        "executed_fatigue_objective",
        "cumulative_normalized_fatigue_cycles",
        "final_capacity_ratio",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            for row in entry["result"].get("muscle_fatigue") or []:
                writer.writerow(
                    {
                        "solver": entry["result"]["solver"],
                        "mechanical_formulation": _mechanical_formulation(entry),
                        "case": _entry_case(entry),
                        **{field: row.get(field) for field in fieldnames[3:]},
                    }
                )


def write_pulse_width_variation_csv(path: Path, entries: list[dict]) -> None:
    fieldnames = (
        "solver",
        "mechanical_formulation",
        "case",
        "muscle",
        "from_cycle",
        "to_cycle",
        "mean_absolute_change_us",
        "root_mean_square_change_us",
        "maximum_absolute_change_us",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            variation = entry["result"].get("pulse_width_cycle_variation") or {}
            for muscle in variation.get("muscles") or []:
                for transition in muscle.get("transitions") or []:
                    writer.writerow(
                        {
                            "solver": entry["result"]["solver"],
                            "mechanical_formulation": _mechanical_formulation(entry),
                            "case": _entry_case(entry),
                            "muscle": muscle.get("muscle"),
                            **{
                                field: transition.get(field) for field in fieldnames[4:]
                            },
                        }
                    )


def render_markdown(
    entries: list[dict],
    mismatches: list[dict],
    missing_cases: tuple[str, ...] = (),
    *,
    missing_solvers: tuple[str, ...] | None = None,
) -> str:
    if missing_solvers and not missing_cases:
        missing_cases = missing_solvers
    bioptim_commits = {
        entry["runtime"].get("provenance", {}).get("BIOPTIM_BENCHMARK_COMMIT")
        for entry in entries
        if entry["runtime"].get("provenance", {}).get("BIOPTIM_BENCHMARK_COMMIT")
    }
    requested_horizons = {
        _requested_rho_count(entry)
        for entry in entries
        if _requested_rho_count(entry) is not None
    }
    horizon_label = (
        f"{next(iter(requested_horizons))} RHO"
        if len(requested_horizons) == 1
        else "horizons mixtes"
    )
    if missing_cases:
        comparability = (
            "Comparabilité du problème et des critères physiques : "
            "**INCOMPLÈTE** "
            f"(cas manquants : {', '.join(name.upper() for name in missing_cases)})"
        )
    elif mismatches:
        comparability = (
            "Comparabilité du problème et des critères physiques : "
            f"**ÉCHEC ({len(mismatches)} écarts)**"
        )
    else:
        comparability = "Comparabilité du problème et des critères physiques : **OK**"
    lines = [
        f"# Benchmark cyclage FES — {horizon_label}",
        "",
        comparability,
        (
            "Intégration Bioptim : **commit commun**"
            if len(bioptim_commits) <= 1
            else "**Attention : branches d’intégration Bioptim différentes selon le backend.**"
        ),
        (
            "Les tolérances internes sont propres à chaque backend et ne font "
            "volontairement pas partie de ce verdict; elles sont affichées "
            "séparément. Le même seuil de faisabilité physique est exigé."
        ),
        "",
        "| Solveur/formulation | Graphe | Tol. interne | Seuil physique | Convergence | RHO résolus | Préfixe strict | 1er échec | Mur-à-mur (s) | Préparation (s) | Profil réduit (s) | Solve/RHO médian (s) | Effectif/RHO médian (s) | Effectif/RHO P90 (s) | Arrêt |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for entry in entries:
        row = entry["result"]
        requested = _requested_rho_count(entry)
        validated_windows = row.get("physically_validated_cycles")
        if validated_windows is None:
            validated_windows = row.get("validated_cycles")
        if validated_windows is None:
            validated_windows = row.get(
                "validated_windows",
                sum(
                    bool(window.get("validated")) for window in row.get("windows") or []
                ),
            )
        lines.append(
            "| {case} | {graph} | {solver_tolerance} | {physical_threshold} | {success} | {successful}/{attempted} | {validated}/{requested} | {first_failed_rho} | {e2e} | {prep} | {profile} | {median} | {effective_median} | {effective_p90} | {stop} |".format(
                case=_entry_label(entry),
                graph="SX"
                if entry["configuration"].get("use_sx") is True
                else "NON-SX",
                solver_tolerance=_fmt_scientific(
                    entry["configuration"].get("nlp_tolerance")
                ),
                physical_threshold=_fmt_scientific(
                    entry["configuration"].get("primal_feasibility_threshold")
                ),
                success="oui" if row.get("success") else "non",
                successful=row.get("successful_windows", 0),
                attempted=row.get("attempted_windows", 0),
                validated=validated_windows,
                requested=requested if requested is not None else "—",
                first_failed_rho=row.get("first_failed_rho") or "—",
                e2e=_fmt(row.get("end_to_end_wall_time_s")),
                prep=_fmt(row.get("initial_guess_preparation_time_s")),
                profile=_fmt(row.get("reduced_profile_build_time_s")),
                median=_fmt(row.get("hot_wall_time_median_s")),
                effective_median=_fmt(
                    row.get(
                        "hot_effective_wall_time_median_s",
                        row.get("hot_wall_time_median_s"),
                    )
                ),
                effective_p90=_fmt(
                    row.get(
                        "hot_effective_wall_time_p90_s",
                        row.get("hot_wall_time_p90_s"),
                    )
                ),
                stop=(row.get("stop") or {}).get("label", "—"),
            )
        )

    fatrop_timings = [
        summary
        for entry in entries
        if (summary := fatrop_internal_timing_summary(entry)) is not None
    ]
    if fatrop_timings:
        lines.extend(
            [
                "",
                "## Décomposition interne Fatrop",
                "",
                "Les temps proviennent des oracles natifs Fatrop/CasADi. Le coût moyen par itération permet de distinguer un grand nombre d’itérations d’un graphe de dérivées intrinsèquement coûteux.",
                "",
                "| Formulation | RHO | Itérations moyennes | Hessienne (s) | Jacobienne (s) | Détection structure (s) | Temps/itération (s) | Hessienne/itération (s) | Jacobienne/itération (s) | Fraction dérivées |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for timing in fatrop_timings:
            lines.append(
                f"| {timing['case'].upper()} | {timing['rho_count']} | "
                f"{_fmt(timing.get('mean_iterations'), 2)} | "
                f"{_fmt(timing.get('hessian_wall_time_s'))} | "
                f"{_fmt(timing.get('jacobian_wall_time_s'))} | "
                f"{_fmt(timing.get('structure_detection_wall_time_s'))} | "
                f"{_fmt(timing.get('total_wall_time_per_iteration_s'), 6)} | "
                f"{_fmt(timing.get('hessian_wall_time_per_iteration_s'), 6)} | "
                f"{_fmt(timing.get('jacobian_wall_time_per_iteration_s'), 6)} | "
                f"{_fmt(timing.get('derivative_wall_time_fraction'), 4)} |"
            )

    lines.extend(
        [
            "",
            "`RHO résolus` compte chaque fenêtre dont le solveur converge et dont la faisabilité indépendante est certifiée. "
            "Le `préfixe strict` s’arrête au premier échec, même si les fenêtres suivantes récupèrent.",
            "",
            "## Coût et fatigue cumulée",
            "",
            "Le coût est réévalué sur les cycles réellement exécutés, sans recompter les horizons qui se chevauchent. "
            "La fatigue cumulée est l’intégrale en cycles de `1 - A/A_scale`.",
            "",
            "| Solveur/formulation | Coût fatigue exécuté | Fatigue cumulée, 4 muscles (cycles) | Minimum A/A_scale |",
            "|---|---:|---:|---:|",
        ]
    )
    for entry in entries:
        row = entry["result"]
        lines.append(
            f"| {_entry_label(entry)} | "
            f"{_fmt(row.get('executed_fatigue_objective'))} | "
            f"{_fmt(row.get('fatigue_auc_cycles'))} | "
            f"{_fmt(row.get('min_A_capacity_ratio'), 6)} |"
        )

    lines.extend(
        [
            "",
            "### Détail des quatre muscles",
            "",
            "| Solveur/formulation | Muscle | Coût fatigue exécuté | Fatigue cumulée (cycles) | A final/A_scale |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for entry in entries:
        for row in entry["result"].get("muscle_fatigue") or []:
            lines.append(
                f"| {_entry_label(entry)} | {row.get('muscle')} | "
                f"{_fmt(row.get('executed_fatigue_objective'))} | "
                f"{_fmt(row.get('cumulative_normalized_fatigue_cycles'))} | "
                f"{_fmt(row.get('final_capacity_ratio'), 6)} |"
            )

    if mismatches:
        lines.extend(
            [
                "",
                "## Écarts de configuration",
                "",
                "| Cas | Référence | Champ | Référence | Observé |",
                "|---|---|---|---|---|",
            ]
        )
        for mismatch in mismatches:
            lines.append(
                f"| {mismatch['case']} | {mismatch['reference_case']} | "
                f"{mismatch['field']} | "
                f"`{mismatch['expected']}` | `{mismatch['observed']}` |"
            )

    lines.extend(
        [
            "",
            "## Temps de chaque RHO",
            "",
            "Le temps effectif inclut la restauration de faisabilité effectuée après le RHO précédent pour préparer celui-ci.",
            "",
            "| Solveur/formulation | RHO | Statut | Statut natif | Faisable | Validé | Itérations | Solve principal (s) | Restauration (s) | Effectif (s) | Solveur (s) |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for entry in entries:
        solver = _entry_label(entry)
        for row in entry["result"].get("windows") or []:
            lines.append(
                f"| {solver} | {row.get('rho')} | {row.get('status')} | "
                f"{row.get('native_status')} | {row.get('primal_feasible')} | "
                f"{row.get('validated')} | "
                f"{row.get('iterations')} | {_fmt(row.get('wall_time_s'))} | "
                f"{_fmt(row.get('feasibility_restoration_wall_time_s'))} | "
                f"{_fmt(row.get('effective_wall_time_s', row.get('wall_time_s')))} | "
                f"{_fmt(row.get('solver_time_s'))} |"
            )

    restoration_entries = [
        entry
        for entry in entries
        if (entry["result"].get("feasibility_restoration") or {}).get("available")
    ]
    if restoration_entries:
        lines.extend(
            [
                "",
                "### Restauration de faisabilité ACADOS",
                "",
                "| Solveur/formulation | Temps total (s) | Étapes auxiliaires |",
                "|---|---:|---:|",
            ]
        )
        for entry in restoration_entries:
            restoration = entry["result"]["feasibility_restoration"]
            lines.append(
                f"| {_entry_label(entry)} | "
                f"{_fmt(restoration.get('total_wall_time_s'))} | "
                f"{len(restoration.get('stages') or [])} |"
            )

    lines.extend(
        [
            "",
            "## Patrons de stimulation",
            "",
            "Les points de contrôle sont des cycles/RHO du même run, pas des OCP contenant simultanément ce nombre de cycles.",
            "",
            "| Solveur/formulation | Cycle | Muscle | Min (µs) | Moyenne (µs) | Max (µs) | Borne basse | Borne haute |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for entry in entries:
        solver = _entry_label(entry)
        for snapshot in (entry["result"].get("stimulation_patterns") or {}).values():
            if not snapshot.get("available"):
                lines.append(
                    f"| {solver} | {snapshot.get('cycle')} | indisponible: "
                    f"{snapshot.get('reason')} | — | — | — | — | — |"
                )
                continue
            for muscle, pattern in snapshot["muscles"].items():
                lines.append(
                    f"| {solver} | {snapshot['cycle']} | {muscle} | "
                    f"{_fmt(1e6 * pattern['minimum_s'], 1)} | "
                    f"{_fmt(1e6 * pattern['mean_s'], 1)} | "
                    f"{_fmt(1e6 * pattern['maximum_s'], 1)} | "
                    f"{_fmt(pattern.get('lower_bound_fraction'))} | "
                    f"{_fmt(pattern.get('upper_bound_fraction'))} |"
                )

    lines.extend(
        [
            "",
            "### Variations de PW entre deux cycles",
            "",
            "Les percentiles décrivent les changements observés à phase de stimulation identique. "
            "Ils ne constituent pas encore des bornes dures ACADOS; une marge et un mécanisme de récupération restent nécessaires.",
            "",
            "| Solveur/formulation | Muscle | ΔPW médian (µs) | P95 (µs) | P99 (µs) | Max (µs) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for entry in entries:
        variation = entry["result"].get("pulse_width_cycle_variation") or {}
        if not variation.get("available"):
            lines.append(
                f"| {_entry_label(entry)} | indisponible: "
                f"{variation.get('reason', 'non_mesuré')} | — | — | — | — |"
            )
            continue
        for row in variation.get("muscles") or []:
            lines.append(
                f"| {_entry_label(entry)} | {row.get('muscle')} | "
                f"{_fmt(row.get('median_absolute_change_us'))} | "
                f"{_fmt(row.get('p95_absolute_change_us'))} | "
                f"{_fmt(row.get('p99_absolute_change_us'))} | "
                f"{_fmt(row.get('maximum_absolute_change_us'))} |"
            )

    lines.extend(
        [
            "",
            "### Écarts des patrons par rapport à IPOPT de la même formulation",
            "",
            "La comparaison brute est faite par indice de stimulation. La comparaison réalignée interpole le patron du solveur selon l’angle réel du pédalier sur la grille angulaire IPOPT.",
            "",
            "| Solveur/formulation | Cycle | Muscle | RMSE brute (µs) | RMSE réalignée (µs) | Max abs. brut (µs) | Corr. brute | Corr. réalignée | RMSE phase (rad) |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for comparison in stimulation_comparisons(entries):
        if not comparison.get("available"):
            lines.append(
                f"| {comparison['case'].upper()} | "
                f"{comparison.get('cycle', '—')} | indisponible: "
                f"{comparison.get('reason')} | — | — | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {comparison['case'].upper()} | {comparison['cycle']} | "
            f"{comparison['muscle']} | "
            f"{_fmt(comparison['root_mean_square_error_us'])} | "
            f"{_fmt(comparison.get('phase_aligned_root_mean_square_error_us'))} | "
            f"{_fmt(comparison['maximum_absolute_error_us'])} | "
            f"{_fmt(comparison.get('correlation'))} | "
            f"{_fmt(comparison.get('phase_aligned_correlation'))} | "
            f"{_fmt(comparison.get('crank_phase_root_mean_square_error_rad'))} |"
        )
    lines.extend(
        [
            "",
            "### Effet de la réduction mécanique sur les patrons",
            "",
            "La formulation complète est la référence; les écarts sont calculés séparément pour chaque solveur et chaque transcription.",
            "",
            "| Solveur/formulation réduite | Cycle | Muscle | RMSE brute (µs) | RMSE réalignée (µs) | Max abs. brut (µs) | Corr. brute | Corr. réalignée | RMSE phase (rad) |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for comparison in mechanical_stimulation_comparisons(entries):
        if not comparison.get("available"):
            lines.append(
                f"| {comparison['case'].upper()} | "
                f"{comparison.get('cycle', '—')} | indisponible: "
                f"{comparison.get('reason')} | — | — | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {comparison['case'].upper()} | {comparison['cycle']} | "
            f"{comparison['muscle']} | "
            f"{_fmt(comparison['root_mean_square_error_us'])} | "
            f"{_fmt(comparison.get('phase_aligned_root_mean_square_error_us'))} | "
            f"{_fmt(comparison['maximum_absolute_error_us'])} | "
            f"{_fmt(comparison.get('correlation'))} | "
            f"{_fmt(comparison.get('phase_aligned_correlation'))} | "
            f"{_fmt(comparison.get('crank_phase_root_mean_square_error_rad'))} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_files", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-solvers",
        default=",".join(DEFAULT_EXPECTED_SOLVERS),
        help=(
            "Legacy comma-separated solver set required for a complete report. "
            "Ignored when --expected-cases is provided."
        ),
    )
    parser.add_argument(
        "--expected-cases",
        default=None,
        help=(
            "Comma-separated solver/formulation cases required for a complete "
            "report, for example ipopt/full,ipopt/reduced."
        ),
    )
    args = parser.parse_args()

    entries = load_benchmark_files(args.json_files)
    if not entries:
        raise SystemExit("No solver result was found in the supplied JSON files.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mismatches = configuration_mismatches(entries)
    comparisons = stimulation_comparisons(entries)
    mechanical_comparisons = mechanical_stimulation_comparisons(entries)
    expected_solvers = tuple(
        dict.fromkeys(
            solver.strip().lower()
            for solver in args.expected_solvers.split(",")
            if solver.strip()
        )
    )
    present_solvers = {entry["result"].get("solver", "").lower() for entry in entries}
    missing_solvers = tuple(
        solver for solver in expected_solvers if solver not in present_solvers
    )
    expected_cases = (
        tuple(
            dict.fromkeys(
                case.strip().lower()
                for case in args.expected_cases.split(",")
                if case.strip()
            )
        )
        if args.expected_cases is not None
        else ()
    )
    present_cases = {_entry_base_case(entry) for entry in entries}
    missing_cases = (
        tuple(case for case in expected_cases if case not in present_cases)
        if expected_cases
        else missing_solvers
    )
    bioptim_commits = sorted(
        {
            entry["runtime"].get("provenance", {}).get("BIOPTIM_BENCHMARK_COMMIT")
            for entry in entries
            if entry["runtime"].get("provenance", {}).get("BIOPTIM_BENCHMARK_COMMIT")
        }
    )
    combined = {
        "schema_version": 1,
        "complete_solver_matrix": not missing_cases,
        "expected_solvers": expected_solvers,
        "present_solvers": sorted(present_solvers),
        "missing_solvers": missing_solvers,
        "expected_cases": expected_cases,
        "present_cases": sorted(present_cases),
        "missing_cases": missing_cases,
        "comparable_configuration": not mismatches and not missing_cases,
        "comparable_problem_and_physical_criteria": (
            not mismatches and not missing_cases
        ),
        "configuration_mismatches": mismatches,
        "same_bioptim_commit": len(bioptim_commits) <= 1,
        "bioptim_commits": bioptim_commits,
        "entries": entries,
        "stimulation_comparisons_against_ipopt": comparisons,
        "stimulation_comparisons_reduced_against_full": mechanical_comparisons,
    }
    (args.output_dir / "benchmark-comparison.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_rho_csv(args.output_dir / "rho-timings.csv", entries)
    write_fatrop_internal_timing_csv(
        args.output_dir / "fatrop-internal-timings.csv", entries
    )
    write_stimulation_csv(args.output_dir / "stimulation-patterns.csv", entries)
    write_muscle_fatigue_csv(args.output_dir / "muscle-fatigue.csv", entries)
    write_pulse_width_variation_csv(
        args.output_dir / "pulse-width-cycle-variation.csv", entries
    )
    (args.output_dir / "benchmark-comparison.md").write_text(
        render_markdown(entries, mismatches, missing_cases),
        encoding="utf-8",
    )
    if missing_cases:
        raise SystemExit(
            "Incomplete solver/formulation matrix; missing: " + ", ".join(missing_cases)
        )


if __name__ == "__main__":
    main()
