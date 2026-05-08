from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import cycling_weight_exploration as cwe
from derive_offline_task_weights import simulate_fatigue_ratios

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis_outputs"
SUMMARY_PATH = OUTPUT_DIR / "offline_1500_cycles_weight_derivation.json"


def load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text())


def plot_formula_breakdown(summary: dict):
    muscles = cwe.MUSCLE_LIST
    criticality = [summary["criticality_normalized"][m] for m in muscles]
    vulnerability = [summary["combined_vulnerability"][m] for m in muscles]
    vulnerability_sq = [v**2 for v in vulnerability]
    raw = [summary["candidate_fixed_weights_normalized"][m] for m in muscles]

    fig = Figure(figsize=(12, 8.5))
    FigureCanvas(fig)
    axes = fig.subplots(2, 2)
    colors = ["#8c5e34", "#6b4f4f", "#1f78b4", "#1b9e77"]

    ax = axes[0, 0]
    ax.bar(muscles, criticality, color=colors)
    ax.set_yscale("log")
    ax.set_title("Normalized criticality")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    ax.bar(muscles, vulnerability, color=colors)
    ax.set_yscale("log")
    ax.set_title("Combined vulnerability")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    ax.bar(muscles, vulnerability_sq, color=colors)
    ax.set_yscale("log")
    ax.set_title("Combined vulnerability squared")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    ax.bar(muscles, raw, color=colors)
    ax.set_yscale("log")
    ax.set_title(r"Raw offline weights: $w_i \propto C_i \times V_i^2$")
    ax.grid(axis="y", alpha=0.25)

    for i, muscle in enumerate(muscles):
        ax.text(
            i,
            raw[i] * 1.15,
            f"{criticality[i]:.1f} x {vulnerability[i]:.1f}^2",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "offline_1500_cycles_formula_breakdown.png", dpi=180)


def plot_synthetic_fatigue(summary: dict):
    muscles = cwe.MUSCLE_LIST
    target_cycles = int(summary["target_cycles"])
    rho = float(summary["rho_for_offline_fatigue"])
    cycles = np.arange(1, target_cycles + 1)

    theta, torque_profiles = cwe.compute_torque_profiles()
    support_mask = np.maximum(torque_profiles, 0.0) > 1e-9

    fig = Figure(figsize=(12, 7.5))
    FigureCanvas(fig)
    axes = fig.subplots(2, 2)
    axes = axes.flatten()
    colors = {
        "Delt_ant": "#8c5e34",
        "Delt_post": "#6b4f4f",
        "Biceps": "#1f78b4",
        "Triceps": "#1b9e77",
    }

    for ax, muscle in zip(axes, muscles):
        duty_cycle = float(np.mean(support_mask[muscles.index(muscle)]))
        ratios = simulate_fatigue_ratios(
            duty_cycle=duty_cycle,
            parameters=cwe.PARAMETERS[muscle],
            total_cycles=target_cycles,
            rho=rho,
        )
        ax.plot(cycles, ratios, color=colors[muscle], lw=2.2)
        ax.set_title(muscle)
        ax.set_xlabel("Synthetic cycle")
        ax.set_ylabel("A / A_rest")
        ax.grid(alpha=0.25)

    fig.suptitle("Synthetic fatigue trajectories used for offline vulnerability", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "offline_1500_cycles_synthetic_fatigue.png", dpi=180, bbox_inches="tight")


def plot_snapshot_criticality(summary: dict):
    muscles = cwe.MUSCLE_LIST
    snapshots = summary["snapshots"]
    cycles = [row["cycle"] for row in snapshots]

    fig = Figure(figsize=(12, 8))
    FigureCanvas(fig)
    axes = fig.subplots(2, 1)
    colors = {
        "Delt_ant": "#8c5e34",
        "Delt_post": "#6b4f4f",
        "Biceps": "#1f78b4",
        "Triceps": "#1b9e77",
    }

    ax = axes[0]
    ax.plot(
        cycles, [row["deficit_area"] for row in snapshots], color="#d00000", marker="o", lw=2.2, label="Deficit area"
    )
    ax.plot(
        cycles, [row["risk_fraction"] for row in snapshots], color="#222222", marker="s", lw=2.0, label="Risk fraction"
    )
    ax.set_title("Synthetic snapshot severity")
    ax.set_xlabel("Synthetic cycle")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    for muscle in muscles:
        ax.plot(
            cycles,
            [row["restore_gain"][muscle] for row in snapshots],
            marker="o",
            lw=2.0,
            label=muscle,
            color=colors[muscle],
        )
    ax.set_title("Restore-gain contribution at each synthetic snapshot")
    ax.set_xlabel("Synthetic cycle")
    ax.set_ylabel("Deficit reduction if muscle is restored")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "offline_1500_cycles_snapshot_criticality.png", dpi=180)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary = load_summary()
    plot_formula_breakdown(summary)
    plot_synthetic_fatigue(summary)
    plot_snapshot_criticality(summary)

    print(OUTPUT_DIR / "offline_1500_cycles_formula_breakdown.png")
    print(OUTPUT_DIR / "offline_1500_cycles_synthetic_fatigue.png")
    print(OUTPUT_DIR / "offline_1500_cycles_snapshot_criticality.png")


if __name__ == "__main__":
    main()
