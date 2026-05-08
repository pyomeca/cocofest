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

OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_outputs"


def cycle_end_indices(total_points: int, n_shooting_per_cycle: int, polynomial_order: int) -> np.ndarray:
    points_per_cycle = n_shooting_per_cycle * (polynomial_order + 1)
    last_indices = np.arange(points_per_cycle, total_points, points_per_cycle, dtype=int)
    return last_indices


def load_run(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    return {k: np.asarray(data[k]) for k in data.files}


def fatigue_ratios_at_index(run: dict, idx: int) -> np.ndarray:
    ratios = []
    for muscle_name in cwe.MUSCLE_LIST:
        a_value = float(run[f"A_{muscle_name}"][0, idx])
        a_rest = float(cwe.PARAMETERS[muscle_name]["a_scale"])
        ratios.append(a_value / a_rest)
    return np.array(ratios, dtype=float)


def feasibility_metrics(theta: np.ndarray, torque_profiles: np.ndarray, ratios: np.ndarray) -> dict:
    scaled_profiles = torque_profiles * ratios[:, None]
    positive_profiles = np.maximum(scaled_profiles, 0.0)
    combined_positive = positive_profiles.sum(axis=0)
    deficit = np.maximum(cwe.TASK_TORQUE_THRESHOLD - combined_positive, 0.0)
    theta_deg = np.degrees(theta)

    per_muscle = {}
    for i, muscle_name in enumerate(cwe.MUSCLE_LIST):
        without_i = combined_positive - positive_profiles[i]
        deficit_without_i = np.maximum(cwe.TASK_TORQUE_THRESHOLD - without_i, 0.0)
        restore_ratios = ratios.copy()
        restore_ratios[i] = 1.0
        restored_positive = np.maximum(torque_profiles * restore_ratios[:, None], 0.0).sum(axis=0)
        deficit_if_restored = np.maximum(cwe.TASK_TORQUE_THRESHOLD - restored_positive, 0.0)
        support_risk_area = float(
            np.trapezoid(positive_profiles[i] * (combined_positive < cwe.TASK_TORQUE_THRESHOLD), theta)
        )

        per_muscle[muscle_name] = {
            "max_positive_torque_nm": float(positive_profiles[i].max()),
            "positive_area_nm_rad": float(np.trapezoid(positive_profiles[i], theta)),
            "support_intervals_deg": cwe.wrap_intervals(positive_profiles[i] > 1e-9, theta_deg),
            "extra_deficit_area_if_removed": float(np.trapezoid(deficit_without_i - deficit, theta)),
            "deficit_area_reduction_if_restored": float(np.trapezoid(deficit - deficit_if_restored, theta)),
            "support_in_risk_area_nm_rad": support_risk_area,
        }

    risk_mask = combined_positive < cwe.TASK_TORQUE_THRESHOLD
    share_in_risk = {}
    if np.any(risk_mask):
        risk_positive = positive_profiles[:, risk_mask]
        totals = risk_positive.sum(axis=0)
        shares = np.divide(risk_positive, totals, out=np.zeros_like(risk_positive), where=totals > 1e-12)
        for i, muscle_name in enumerate(cwe.MUSCLE_LIST):
            share_in_risk[muscle_name] = float(shares[i].mean())
    else:
        for muscle_name in cwe.MUSCLE_LIST:
            share_in_risk[muscle_name] = 0.0

    return {
        "ratios": {name: float(r) for name, r in zip(cwe.MUSCLE_LIST, ratios)},
        "combined_positive_torque": combined_positive,
        "risk_intervals_deg": cwe.wrap_intervals(risk_mask, theta_deg),
        "risk_fraction": float(np.mean(risk_mask)),
        "deficit_area": float(np.trapezoid(deficit, theta)),
        "per_muscle": per_muscle,
        "share_in_risk_zone": share_in_risk,
    }


def choose_snapshot_indices(run: dict, requested_cycles: list[int] | None) -> list[tuple[str, int]]:
    idx = cycle_end_indices(
        total_points=run["time"].shape[0],
        n_shooting_per_cycle=int(run["n_shooting_per_cycle"].item()),
        polynomial_order=int(run["polynomial_order"].item()),
    )
    if requested_cycles:
        selected = []
        for cycle in requested_cycles:
            cycle = max(1, min(cycle, len(idx)))
            selected.append((f"cycle_{cycle}", int(idx[cycle - 1])))
        return selected

    if len(idx) <= 8:
        return [(f"cycle_{i+1}", int(v)) for i, v in enumerate(idx)]

    anchors = np.unique(np.round(np.linspace(1, len(idx), 6)).astype(int))
    return [(f"cycle_{cycle}", int(idx[cycle - 1])) for cycle in anchors]


def plot_failure_diagnostics(theta_deg: np.ndarray, snapshots: list[tuple[str, dict]], stem: str):
    fig = Figure(figsize=(12, 9))
    FigureCanvas(fig)
    axes = fig.subplots(3, 1, sharex=False)

    ax = axes[0]
    for label, snapshot in snapshots:
        ax.plot(theta_deg, snapshot["combined_positive_torque"], lw=2.0, label=label)
    ax.axhline(cwe.TASK_TORQUE_THRESHOLD, color="black", ls="--", lw=1.5, label="0.20 Nm threshold")
    ax.set_ylabel("Combined positive torque (Nm)")
    ax.set_title("Feasible crank-torque support at selected snapshots")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)

    ax = axes[1]
    labels = [label for label, _ in snapshots]
    risk_fraction = [snapshot["risk_fraction"] for _, snapshot in snapshots]
    deficit_area = [snapshot["deficit_area"] for _, snapshot in snapshots]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, risk_fraction, width=0.35, label="Risk fraction")
    ax.bar(x + 0.18, deficit_area, width=0.35, label="Deficit area (rad.Nm)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_title("Task-feasibility degradation across snapshots")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[2]
    final_snapshot = snapshots[-1][1]
    restore_gain = [final_snapshot["per_muscle"][m]["deficit_area_reduction_if_restored"] for m in cwe.MUSCLE_LIST]
    risk_share = [final_snapshot["share_in_risk_zone"][m] for m in cwe.MUSCLE_LIST]
    x = np.arange(len(cwe.MUSCLE_LIST))
    ax.bar(x - 0.18, restore_gain, width=0.35, label="Deficit reduction if restored")
    ax.bar(x + 0.18, risk_share, width=0.35, label="Average share in risk zone")
    ax.set_xticks(x)
    ax.set_xticklabels(cwe.MUSCLE_LIST)
    ax.set_title(f"Final snapshot bottleneck analysis: {snapshots[-1][0]}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}_failure_diagnostics.png", dpi=180)


def main():
    parser = argparse.ArgumentParser(description="Analyze failure feasibility from a saved cycling MHE .npz file.")
    parser.add_argument("npz", type=Path, help="Path to the saved MHE .npz file.")
    parser.add_argument(
        "--snapshot-cycles",
        type=int,
        nargs="*",
        default=None,
        help="Cycle numbers to analyze explicitly. Default: a small evenly spaced subset plus the last cycle.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    run = load_run(args.npz)
    theta, torque_profiles = cwe.compute_torque_profiles()
    theta_deg = np.degrees(theta)

    snapshots = []
    serializable = {}
    for label, idx in choose_snapshot_indices(run, args.snapshot_cycles):
        metrics = feasibility_metrics(theta, torque_profiles, fatigue_ratios_at_index(run, idx))
        snapshots.append((label, metrics))
        serializable[label] = {
            "ratios": metrics["ratios"],
            "risk_intervals_deg": metrics["risk_intervals_deg"],
            "risk_fraction": metrics["risk_fraction"],
            "deficit_area": metrics["deficit_area"],
            "share_in_risk_zone": metrics["share_in_risk_zone"],
            "per_muscle": metrics["per_muscle"],
        }

    stem = args.npz.stem
    plot_failure_diagnostics(theta_deg, snapshots, stem)

    out_json = OUTPUT_DIR / f"{stem}_failure_analysis.json"
    out_json.write_text(json.dumps(serializable, indent=2))
    print(out_json)
    print(OUTPUT_DIR / f"{stem}_failure_diagnostics.png")


if __name__ == "__main__":
    main()
