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
    "torque_application",
    "ode_solver",
    "collocation_degree",
    "collocation_method",
    "use_sx",
    "state_scaling",
    "pulse_width_scaling",
    "pulse_width_active_set",
    "acados_terminal_wheel_q_slack",
    "cycles_per_window",
    "stimulations_per_cycle",
    "n_windows",
    "n_threads",
    "constant_crank_torque",
    "crank_torque_role",
    "primal_feasibility_threshold",
)
DEFAULT_EXPECTED_SOLVERS = ("ipopt", "madnlp", "alpaqa")


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
    entries.sort(key=lambda entry: entry["result"]["solver"])
    return entries


def configuration_mismatches(entries: list[dict]) -> list[dict]:
    if not entries:
        return []
    reference = next(
        (
            entry
            for entry in entries
            if entry["result"].get("solver") == "ipopt"
        ),
        entries[0],
    )
    mismatches = []
    for entry in entries:
        for field in COMPARABILITY_FIELDS:
            expected = reference["configuration"].get(field)
            observed = entry["configuration"].get(field)
            if observed != expected:
                mismatches.append(
                    {
                        "solver": entry["result"]["solver"],
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
        for reference_value, compared_value in zip(
            reference_values, compared_values
        )
    ]
    mae = sum(abs(value) for value in differences) / len(differences)
    rmse = math.sqrt(sum(value * value for value in differences) / len(differences))
    reference_mean = sum(reference_values) / len(reference_values)
    compared_mean = sum(compared_values) / len(compared_values)
    covariance = sum(
        (reference_value - reference_mean) * (compared_value - compared_mean)
        for reference_value, compared_value in zip(
            reference_values, compared_values
        )
    )
    reference_energy = sum(
        (value - reference_mean) ** 2 for value in reference_values
    )
    compared_energy = sum(
        (value - compared_mean) ** 2 for value in compared_values
    )
    denominator = math.sqrt(reference_energy * compared_energy)
    return {
        "sample_count": len(differences),
        "mean_absolute_error_s": mae,
        "root_mean_square_error_s": rmse,
        "maximum_absolute_error_s": max(abs(value) for value in differences),
        "mean_absolute_error_us": 1e6 * mae,
        "root_mean_square_error_us": 1e6 * rmse,
        "maximum_absolute_error_us": 1e6
        * max(abs(value) for value in differences),
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


def stimulation_comparisons(entries: list[dict]) -> list[dict]:
    reference_entry = next(
        (
            entry
            for entry in entries
            if entry["result"].get("solver") == "ipopt"
        ),
        None,
    )
    if reference_entry is None:
        return []
    reference_patterns = reference_entry["result"].get("stimulation_patterns") or {}
    rows = []
    for entry in entries:
        solver = entry["result"]["solver"]
        if solver == "ipopt":
            continue
        compared_patterns = entry["result"].get("stimulation_patterns") or {}
        for checkpoint, reference_snapshot in reference_patterns.items():
            compared_snapshot = compared_patterns.get(checkpoint) or {}
            if not (
                reference_snapshot.get("available")
                and compared_snapshot.get("available")
            ):
                rows.append(
                    {
                        "reference_solver": "ipopt",
                        "solver": solver,
                        "checkpoint": checkpoint,
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
                        for reference, compared in zip(
                            reference_phase, compared_phase
                        )
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
                        "reference_solver": "ipopt",
                        "solver": solver,
                        "checkpoint": checkpoint,
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


def write_rho_csv(path: Path, entries: list[dict]) -> None:
    fieldnames = (
        "solver",
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
                        **{
                            field: window.get(field)
                            for field in fieldnames
                            if field
                            not in {
                                "solver",
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


def write_stimulation_csv(path: Path, entries: list[dict]) -> None:
    fieldnames = (
        "solver",
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
                                "normalized_to_bounds": pattern[
                                    "normalized_to_bounds"
                                ][index],
                            }
                        )


def render_markdown(
    entries: list[dict],
    mismatches: list[dict],
    missing_solvers: tuple[str, ...] = (),
) -> str:
    bioptim_commits = {
        entry["runtime"].get("provenance", {}).get("BIOPTIM_BENCHMARK_COMMIT")
        for entry in entries
        if entry["runtime"].get("provenance", {}).get("BIOPTIM_BENCHMARK_COMMIT")
    }
    requested_horizons = {
        entry["configuration"].get("n_windows")
        for entry in entries
        if entry["configuration"].get("n_windows") is not None
    }
    horizon_label = (
        f"{next(iter(requested_horizons))} RHO"
        if len(requested_horizons) == 1
        else "horizons mixtes"
    )
    if missing_solvers:
        comparability = (
            "Comparabilité des configurations : **INCOMPLÈTE** "
            f"(solveurs manquants : {', '.join(name.upper() for name in missing_solvers)})"
        )
    elif mismatches:
        comparability = (
            f"Comparabilité des configurations : **ÉCHEC ({len(mismatches)} écarts)**"
        )
    else:
        comparability = "Comparabilité des configurations : **OK**"
    lines = [
        f"# Benchmark cyclage FES — {horizon_label}",
        "",
        comparability,
        (
            "Intégration Bioptim : **commit commun**"
            if len(bioptim_commits) <= 1
            else "**Attention : branches d’intégration Bioptim différentes selon le backend.**"
        ),
        "",
        "| Solveur | Tol. interne | Seuil physique | Convergence | Cycles validés | Mur-à-mur (s) | Préparation (s) | Mur/RHO médian (s) | Mur/RHO P90 (s) | Arrêt |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for entry in entries:
        row = entry["result"]
        requested = entry["configuration"].get("n_windows")
        lines.append(
            "| {solver} | {solver_tolerance} | {physical_threshold} | {success} | {validated}/{requested} | {e2e} | {prep} | {median} | {p90} | {stop} |".format(
                solver=row["solver"].upper(),
                solver_tolerance=_fmt_scientific(
                    entry["configuration"].get("nlp_tolerance")
                ),
                physical_threshold=_fmt_scientific(
                    entry["configuration"].get("primal_feasibility_threshold")
                ),
                success="oui" if row.get("success") else "non",
                validated=row.get("validated_cycles", 0),
                requested=requested if requested is not None else "—",
                e2e=_fmt(row.get("end_to_end_wall_time_s")),
                prep=_fmt(row.get("initial_guess_preparation_time_s")),
                median=_fmt(row.get("hot_wall_time_median_s")),
                p90=_fmt(row.get("hot_wall_time_p90_s")),
                stop=(row.get("stop") or {}).get("label", "—"),
            )
        )

    if mismatches:
        lines.extend(
            [
                "",
                "## Écarts de configuration",
                "",
                "| Solveur | Champ | Référence | Observé |",
                "|---|---|---|---|",
            ]
        )
        for mismatch in mismatches:
            lines.append(
                f"| {mismatch['solver']} | {mismatch['field']} | "
                f"`{mismatch['expected']}` | `{mismatch['observed']}` |"
            )

    lines.extend(
        [
            "",
            "## Temps de chaque RHO",
            "",
            "| Solveur | RHO | Statut | Statut natif | Faisable | Validé | Itérations | Mur (s) | Solveur (s) |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for entry in entries:
        solver = entry["result"]["solver"].upper()
        for row in entry["result"].get("windows") or []:
            lines.append(
                f"| {solver} | {row.get('rho')} | {row.get('status')} | "
                f"{row.get('native_status')} | {row.get('primal_feasible')} | "
                f"{row.get('validated')} | "
                f"{row.get('iterations')} | {_fmt(row.get('wall_time_s'))} | "
                f"{_fmt(row.get('solver_time_s'))} |"
            )

    lines.extend(
        [
            "",
            "## Patrons de stimulation",
            "",
            "Les points de contrôle sont des cycles/RHO du même run, pas des OCP contenant simultanément ce nombre de cycles.",
            "",
            "| Solveur | Cycle | Muscle | Min (µs) | Moyenne (µs) | Max (µs) | Borne basse | Borne haute |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for entry in entries:
        solver = entry["result"]["solver"].upper()
        for snapshot in (
            entry["result"].get("stimulation_patterns") or {}
        ).values():
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
            "### Écarts des patrons par rapport à IPOPT",
            "",
            "La comparaison brute est faite par indice de stimulation. La comparaison réalignée interpole le patron du solveur selon l’angle réel du pédalier sur la grille angulaire IPOPT.",
            "",
            "| Solveur | Cycle | Muscle | RMSE brute (µs) | RMSE réalignée (µs) | Max abs. brut (µs) | Corr. brute | Corr. réalignée | RMSE phase (rad) |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for comparison in stimulation_comparisons(entries):
        if not comparison.get("available"):
            lines.append(
                f"| {comparison['solver'].upper()} | "
                f"{comparison.get('cycle', '—')} | indisponible: "
                f"{comparison.get('reason')} | — | — | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {comparison['solver'].upper()} | {comparison['cycle']} | "
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
        help="Comma-separated solver set required for a complete report.",
    )
    args = parser.parse_args()

    entries = load_benchmark_files(args.json_files)
    if not entries:
        raise SystemExit("No solver result was found in the supplied JSON files.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mismatches = configuration_mismatches(entries)
    comparisons = stimulation_comparisons(entries)
    expected_solvers = tuple(
        dict.fromkeys(
            solver.strip().lower()
            for solver in args.expected_solvers.split(",")
            if solver.strip()
        )
    )
    present_solvers = {
        entry["result"].get("solver", "").lower() for entry in entries
    }
    missing_solvers = tuple(
        solver for solver in expected_solvers if solver not in present_solvers
    )
    bioptim_commits = sorted(
        {
            entry["runtime"].get("provenance", {}).get("BIOPTIM_BENCHMARK_COMMIT")
            for entry in entries
            if entry["runtime"]
            .get("provenance", {})
            .get("BIOPTIM_BENCHMARK_COMMIT")
        }
    )
    combined = {
        "schema_version": 1,
        "complete_solver_matrix": not missing_solvers,
        "expected_solvers": expected_solvers,
        "present_solvers": sorted(present_solvers),
        "missing_solvers": missing_solvers,
        "comparable_configuration": not mismatches and not missing_solvers,
        "configuration_mismatches": mismatches,
        "same_bioptim_commit": len(bioptim_commits) <= 1,
        "bioptim_commits": bioptim_commits,
        "entries": entries,
        "stimulation_comparisons_against_ipopt": comparisons,
    }
    (args.output_dir / "benchmark-comparison.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_rho_csv(args.output_dir / "rho-timings.csv", entries)
    write_stimulation_csv(args.output_dir / "stimulation-patterns.csv", entries)
    (args.output_dir / "benchmark-comparison.md").write_text(
        render_markdown(entries, mismatches, missing_solvers),
        encoding="utf-8",
    )
    if missing_solvers:
        raise SystemExit(
            "Incomplete solver matrix; missing: " + ", ".join(missing_solvers)
        )


if __name__ == "__main__":
    main()
