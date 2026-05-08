from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import cycling_weight_exploration as cwe
from analyze_failure_feasibility import cycle_end_indices, feasibility_metrics, load_run

OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_outputs"


def choose_snapshot_cycle_indices(run: dict, num_snapshots: int) -> list[tuple[int, int]]:
    cycle_indices = cycle_end_indices(
        total_points=run["time"].shape[0],
        n_shooting_per_cycle=int(run["n_shooting_per_cycle"].item()),
        polynomial_order=int(run["polynomial_order"].item()),
    )
    if len(cycle_indices) == 0:
        return []
    anchors = np.unique(np.round(np.linspace(1, len(cycle_indices), num_snapshots)).astype(int))
    return [(cycle, int(cycle_indices[cycle - 1])) for cycle in anchors]


def fatigue_ratios(run: dict, idx: int) -> np.ndarray:
    values = []
    for muscle_name in cwe.MUSCLE_LIST:
        a_value = float(run[f"A_{muscle_name}"][0, idx])
        a_rest = float(cwe.PARAMETERS[muscle_name]["a_scale"])
        values.append(a_value / a_rest)
    return np.array(values, dtype=float)


def intrinsic_vulnerability() -> dict[str, float]:
    theta, torque_profiles = cwe.compute_torque_profiles()
    theta_deg = np.degrees(theta)
    positive_torque = np.maximum(torque_profiles, 0.0)
    support_mask = positive_torque > 1e-9
    vulnerability = {}
    for i, muscle_name in enumerate(cwe.MUSCLE_LIST):
        duty_cycle = float(np.mean(support_mask[i]))
        fatigue_summary, _, _ = cwe.analyze_fatigue_for_muscle(
            duty_cycle=duty_cycle, parameters=cwe.PARAMETERS[muscle_name]
        )
        cycles_fail = fatigue_summary.cycles_to_failure_high_demand
        vulnerability[muscle_name] = 1e-4 if cycles_fail is None else 1.0 / float(cycles_fail)
    return vulnerability


def normalize_positive(signal: dict[str, float]) -> dict[str, float]:
    positive_values = [v for v in signal.values() if v > 0]
    scale = min(positive_values) if positive_values else 1.0
    return {k: float(v / scale) for k, v in signal.items()}


def derive_weights(run: dict, num_snapshots: int):
    theta, torque_profiles = cwe.compute_torque_profiles()
    snapshot_indices = choose_snapshot_cycle_indices(run, num_snapshots=num_snapshots)
    if not snapshot_indices:
        raise RuntimeError("No cycle-end snapshots could be extracted from this run.")

    snapshots = []
    for cycle, idx in snapshot_indices:
        ratios = fatigue_ratios(run, idx)
        metrics = feasibility_metrics(theta, torque_profiles, ratios)
        snapshots.append((cycle, ratios, metrics))

    max_deficit = max(snapshot[2]["deficit_area"] for snapshot in snapshots)
    criticality = {m: 0.0 for m in cwe.MUSCLE_LIST}
    trajectory_vulnerability = {m: 0.0 for m in cwe.MUSCLE_LIST}
    snapshot_rows = []

    for cycle, ratios, metrics in snapshots:
        severity = metrics["deficit_area"] / max(max_deficit, 1e-8)
        weight = 0.25 + 0.75 * severity
        row = {
            "cycle": int(cycle),
            "severity_weight": float(weight),
            "risk_fraction": metrics["risk_fraction"],
            "deficit_area": metrics["deficit_area"],
            "ratios": {m: float(ratios[i]) for i, m in enumerate(cwe.MUSCLE_LIST)},
            "restore_gain": {},
        }
        for i, muscle_name in enumerate(cwe.MUSCLE_LIST):
            restore_gain = metrics["per_muscle"][muscle_name]["deficit_area_reduction_if_restored"]
            support_risk = metrics["per_muscle"][muscle_name]["support_in_risk_area_nm_rad"]
            criticality[muscle_name] += weight * (restore_gain + 0.25 * support_risk)
            trajectory_vulnerability[muscle_name] += weight * (1.0 - ratios[i])
            row["restore_gain"][muscle_name] = float(restore_gain)
        snapshot_rows.append(row)

    intrinsic = intrinsic_vulnerability()
    criticality_norm = normalize_positive(criticality)
    intrinsic_norm = normalize_positive(intrinsic)
    trajectory_norm = normalize_positive(trajectory_vulnerability)

    vulnerability = {m: float(np.sqrt(intrinsic_norm[m] * trajectory_norm[m])) for m in cwe.MUSCLE_LIST}
    raw_weights = {m: float(criticality_norm[m] * vulnerability[m]) for m in cwe.MUSCLE_LIST}
    fixed_weights = normalize_positive(raw_weights)

    return {
        "snapshots": snapshot_rows,
        "criticality_signal": criticality,
        "criticality_normalized": criticality_norm,
        "intrinsic_vulnerability": intrinsic,
        "intrinsic_vulnerability_normalized": intrinsic_norm,
        "trajectory_vulnerability": trajectory_vulnerability,
        "trajectory_vulnerability_normalized": trajectory_norm,
        "combined_vulnerability": vulnerability,
        "candidate_fixed_weights_raw": raw_weights,
        "candidate_fixed_weights_normalized": fixed_weights,
    }


def plot_weight_derivation(summary: dict, stem: str):
    fig = Figure(figsize=(12, 9))
    FigureCanvas(fig)
    axes = fig.subplots(3, 1)

    cycles = [row["cycle"] for row in summary["snapshots"]]
    deficit = [row["deficit_area"] for row in summary["snapshots"]]
    risk_fraction = [row["risk_fraction"] for row in summary["snapshots"]]

    ax = axes[0]
    ax.plot(cycles, deficit, marker="o", lw=2.0, label="Deficit area")
    ax.plot(cycles, risk_fraction, marker="s", lw=2.0, label="Risk fraction")
    ax.set_xlabel("Cycle")
    ax.set_title("Snapshot severity along the MHE trajectory")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    for muscle_name in cwe.MUSCLE_LIST:
        values = [row["restore_gain"][muscle_name] for row in summary["snapshots"]]
        ax.plot(cycles, values, marker="o", lw=2.0, label=muscle_name)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Deficit reduction if restored")
    ax.set_title("Restore-gain profiles across snapshots")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)

    ax = axes[2]
    x = np.arange(len(cwe.MUSCLE_LIST))
    criticality = [summary["criticality_normalized"][m] for m in cwe.MUSCLE_LIST]
    vulnerability = [summary["combined_vulnerability"][m] for m in cwe.MUSCLE_LIST]
    weights = [summary["candidate_fixed_weights_normalized"][m] for m in cwe.MUSCLE_LIST]
    ax.bar(x - 0.25, criticality, width=0.25, label="Criticality")
    ax.bar(x, vulnerability, width=0.25, label="Vulnerability")
    ax.bar(x + 0.25, weights, width=0.25, label="Candidate fixed weight")
    ax.set_xticks(x)
    ax.set_xticklabels(cwe.MUSCLE_LIST)
    ax.set_yscale("log")
    ax.set_title("Snapshot-derived signals and final fixed weights")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}_snapshot_weight_derivation.png", dpi=180)


def main():
    parser = argparse.ArgumentParser(
        description="Derive constant muscle weights from fatigue snapshots of a saved MHE run."
    )
    parser.add_argument("npz", type=Path, help="Path to the saved MHE .npz file.")
    parser.add_argument("--num-snapshots", type=int, default=8, help="Number of cycle-end snapshots to use.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    run = load_run(args.npz)
    summary = derive_weights(run, num_snapshots=args.num_snapshots)
    stem = args.npz.stem
    plot_weight_derivation(summary, stem)
    out_json = OUTPUT_DIR / f"{stem}_snapshot_weight_derivation.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(out_json)
    print(OUTPUT_DIR / f"{stem}_snapshot_weight_derivation.png")


if __name__ == "__main__":
    main()
