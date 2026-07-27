#!/usr/bin/env python3
"""Summarize the focused Alpaqa and MadNLP option screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

DEFAULT_EXPECTED_VARIANTS = (
    "alpaqa-panoc60-default",
    "alpaqa-panoc20-default",
    "alpaqa-panoc5-default",
    "alpaqa-panoc20-auto-penalty",
    "alpaqa-panoc20-penalty0p01",
    "alpaqa-panoc20-penalty0p01-factor2",
    "madnlp-tol1e-6-default",
    "madnlp-tol1e-7-default",
    "madnlp-tol1e-8-default",
    "madnlp-tol1e-7-mumps",
    "madnlp-tol1e-7-umfpack",
)


def _finite(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _median(values) -> float | None:
    finite = [item for value in values if (item := _finite(value)) is not None]
    return statistics.median(finite) if finite else None


def _sum_evaluations(stats: dict) -> int:
    return sum(
        int(value)
        for key, value in stats.items()
        if key.startswith("n_call_") and _finite(value) is not None
    )


def load_rows(paths: list[Path], common_feasibility_threshold: float) -> list[dict]:
    rows = []
    variants = set()
    for path in paths:
        variant = path.stem
        if variant in variants:
            raise ValueError(f"Duplicate screen variant: {variant}")
        variants.add(variant)

        payload = json.loads(path.read_text(encoding="utf-8"))
        results = payload.get("results") or []
        if len(results) != 1:
            raise ValueError(f"{path} must contain exactly one solver result")
        result = results[0]
        solver = str(result.get("solver", "")).lower()
        configuration = (payload.get("configurations") or {}).get(solver) or {}
        windows = result.get("windows") or []
        infeasibilities = [
            _finite(
                (window.get("feasibility") or {}).get("effective_primal_infeasibility")
            )
            for window in windows
        ]
        common_feasible = [
            value is not None and value <= common_feasibility_threshold
            for value in infeasibilities
        ]
        common_validated = [
            feasible and bool(window.get("solver_converged"))
            for window, feasible in zip(windows, common_feasible)
        ]
        common_prefix = 0
        for validated in common_validated:
            if not validated:
                break
            common_prefix += 1
        stats = result.get("nlp_solver_stats") or []
        evaluation_counts = [_sum_evaluations(item) for item in stats]
        rows.append(
            {
                "variant": variant,
                "solver": solver,
                "configured_tolerance": _finite(configuration.get("nlp_tolerance")),
                "linear_solver": configuration.get("madnlp_linear_solver"),
                "panoc_max_wall_time_s": _finite(
                    configuration.get("alpaqa_panoc_max_wall_time")
                ),
                "initial_penalty": _finite(configuration.get("alpaqa_initial_penalty")),
                "penalty_update_factor": _finite(
                    configuration.get("alpaqa_penalty_update_factor")
                ),
                "success": bool(result.get("success")),
                "validated_cycles_internal": int(result.get("validated_cycles") or 0),
                "attempted_windows": int(result.get("attempted_windows") or 0),
                "common_feasible_windows": sum(common_feasible),
                "common_validated_prefix": common_prefix,
                "first_failed_rho_internal": result.get("first_failed_rho"),
                "first_effective_infeasibility": (
                    infeasibilities[0] if infeasibilities else None
                ),
                "maximum_effective_infeasibility": max(
                    (value for value in infeasibilities if value is not None),
                    default=None,
                ),
                "first_objective": (
                    _finite(windows[0].get("objective")) if windows else None
                ),
                "first_native_status": (
                    windows[0].get("native_status") if windows else None
                ),
                "last_native_status": (
                    windows[-1].get("native_status") if windows else None
                ),
                "first_iterations": (windows[0].get("iterations") if windows else None),
                "total_oracle_evaluations": sum(evaluation_counts),
                "first_oracle_evaluations": (
                    evaluation_counts[0] if evaluation_counts else None
                ),
                "rho_wall_median_s": _median(
                    window.get("wall_time_s") for window in windows
                ),
                "rho_solver_median_s": _median(
                    window.get("solver_time_s") for window in windows
                ),
                "end_to_end_wall_time_s": _finite(result.get("end_to_end_wall_time_s")),
                "initial_guess_preparation_time_s": _finite(
                    result.get("initial_guess_preparation_time_s")
                ),
                "stop": (result.get("stop") or {}).get("label"),
                "error": result.get("error"),
                "source": str(path),
            }
        )
    return sorted(rows, key=lambda row: (row["solver"], row["variant"]))


def ranked_rows(rows: list[dict]) -> list[dict]:
    """Rank progress without confusing a solver tolerance with physical validity."""

    def key(row):
        infeasibility = row["maximum_effective_infeasibility"]
        wall = row["rho_wall_median_s"]
        if row["solver"] == "alpaqa":
            return (
                -row["common_validated_prefix"],
                math.inf if infeasibility is None else infeasibility,
                math.inf if row["first_objective"] is None else row["first_objective"],
                math.inf if wall is None else wall,
            )
        return (
            -row["common_validated_prefix"],
            math.inf if infeasibility is None else infeasibility,
            math.inf if wall is None else wall,
        )

    ranked = []
    for solver in ("alpaqa", "madnlp"):
        for rank, row in enumerate(
            sorted((row for row in rows if row["solver"] == solver), key=key),
            start=1,
        ):
            ranked.append({**row, "rank_within_solver": rank})
    return ranked


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = tuple(rows[0]) if rows else ("variant",)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value, digits=3) -> str:
    value = _finite(value)
    return "—" if value is None else f"{value:.{digits}g}"


def render_markdown(
    rows: list[dict],
    common_feasibility_threshold: float,
    missing_variants: tuple[str, ...],
) -> str:
    lines = [
        "# Écran d’options Alpaqa/MadNLP",
        "",
        (
            "Le classement applique le même seuil physique indépendant "
            f"`{common_feasibility_threshold:.1e}` à toutes les variantes. "
            "La tolérance interne du solveur reste affichée séparément."
        ),
        "",
    ]
    if missing_variants:
        lines.extend(
            [
                f"**Matrice incomplète — variantes absentes : "
                f"{', '.join(missing_variants)}.**",
                "",
            ]
        )
    lines.extend(
        [
            "| Rang | Variante | Tol. | Préfixe valide commun | Infais. max | "
            "Obj. RHO 1 | Mur RHO médian (s) | Préparation (s) | Évals oracle | "
            "Statut natif |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for solver in ("alpaqa", "madnlp"):
        solver_rows = [row for row in rows if row["solver"] == solver]
        if not solver_rows:
            continue
        lines.append(f"|  | **{solver.upper()}** |  |  |  |  |  |  |  |  |")
        for row in solver_rows:
            lines.append(
                f"| {row['rank_within_solver']} | `{row['variant']}` | "
                f"{_fmt(row['configured_tolerance'])} | "
                f"{row['common_validated_prefix']}/{row['attempted_windows']} | "
                f"{_fmt(row['maximum_effective_infeasibility'])} | "
                f"{_fmt(row['first_objective'])} | "
                f"{_fmt(row['rho_wall_median_s'])} | "
                f"{_fmt(row['initial_guess_preparation_time_s'])} | "
                f"{row['total_oracle_evaluations']} | "
                f"{row['last_native_status'] or '—'} |"
            )
        lines.append("")
    lines.extend(
        [
            "Le total d’évaluations est la somme des compteurs CasADi "
            "`n_call_*`; il sert à comparer l’activité de l’oracle, pas à "
            "reconstituer exactement les itérations PANOC/ALM.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_files", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--common-feasibility-threshold", type=float, default=1e-5)
    parser.add_argument(
        "--expected-variants",
        default=",".join(DEFAULT_EXPECTED_VARIANTS),
    )
    args = parser.parse_args()
    if args.common_feasibility_threshold <= 0:
        raise SystemExit("--common-feasibility-threshold must be positive")

    rows = ranked_rows(load_rows(args.json_files, args.common_feasibility_threshold))
    expected = tuple(
        item.strip() for item in args.expected_variants.split(",") if item.strip()
    )
    present = {row["variant"] for row in rows}
    missing = tuple(variant for variant in expected if variant not in present)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "solver-option-screen.csv", rows)
    (args.output_dir / "solver-option-screen.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "common_feasibility_threshold": args.common_feasibility_threshold,
                "complete": not missing,
                "missing_variants": missing,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "solver-option-screen.md").write_text(
        render_markdown(rows, args.common_feasibility_threshold, missing),
        encoding="utf-8",
    )
    if missing:
        raise SystemExit("Incomplete option screen: " + ", ".join(missing))


if __name__ == "__main__":
    main()
