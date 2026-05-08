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


def article_figure(summary: dict):
    muscles = cwe.MUSCLE_LIST
    colors = {
        "Delt_ant": "#8c5e34",
        "Delt_post": "#6b4f4f",
        "Biceps": "#1f78b4",
        "Triceps": "#1b9e77",
    }

    fig = Figure(figsize=(13, 8.5))
    FigureCanvas(fig)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.2, 1.0])

    ax_fatigue = fig.add_subplot(gs[0, 0])
    ax_restore = fig.add_subplot(gs[0, 1])
    ax_weights = fig.add_subplot(gs[1, :])

    target_cycles = int(summary["target_cycles"])
    rho = float(summary["rho_for_offline_fatigue"])
    cycles = np.arange(1, target_cycles + 1)

    theta, torque_profiles = cwe.compute_torque_profiles()
    support_mask = np.maximum(torque_profiles, 0.0) > 1e-9

    for i, muscle in enumerate(muscles):
        duty_cycle = float(np.mean(support_mask[i]))
        ratios = simulate_fatigue_ratios(
            duty_cycle=duty_cycle,
            parameters=cwe.PARAMETERS[muscle],
            total_cycles=target_cycles,
            rho=rho,
        )
        ax_fatigue.plot(cycles, ratios, color=colors[muscle], lw=2.4, label=muscle)

    ax_fatigue.set_xlabel("Synthetic cycle")
    ax_fatigue.set_ylabel(r"$A / A_{rest}$")
    ax_fatigue.set_title("A. Synthetic Fatigue Trajectories")
    ax_fatigue.grid(alpha=0.25)
    ax_fatigue.legend(ncol=2, fontsize=9)

    snapshot_cycles = [row["cycle"] for row in summary["snapshots"]]
    for muscle in muscles:
        restore = [row["restore_gain"][muscle] for row in summary["snapshots"]]
        ax_restore.plot(
            snapshot_cycles,
            restore,
            marker="o",
            ms=4.5,
            lw=2.1,
            color=colors[muscle],
            label=muscle,
        )

    ax_restore.set_xlabel("Synthetic snapshot cycle")
    ax_restore.set_ylabel("Deficit reduction if restored")
    ax_restore.set_title("B. Snapshot-Based Mechanical Criticality")
    ax_restore.grid(alpha=0.25)

    x = np.arange(len(muscles))
    criticality = [summary["criticality_normalized"][m] for m in muscles]
    vulnerability = [summary["combined_vulnerability"][m] for m in muscles]
    raw_weights = [summary["candidate_fixed_weights_normalized"][m] for m in muscles]

    ax_weights.bar(x - 0.24, criticality, width=0.22, color="#bdb2a7", label="Criticality")
    ax_weights.bar(x, vulnerability, width=0.22, color="#a8dadc", label="Vulnerability")
    ax_weights.bar(x + 0.24, raw_weights, width=0.22, color=[colors[m] for m in muscles], label="Raw weight")
    ax_weights.set_xticks(x)
    ax_weights.set_xticklabels(muscles)
    ax_weights.set_yscale("log")
    ax_weights.set_title(r"C. Raw Offline Weights: $w_i \propto C_i \times V_i^2$")
    ax_weights.grid(axis="y", alpha=0.25)
    ax_weights.legend(ncol=3, fontsize=9)

    for i, muscle in enumerate(muscles):
        ax_weights.text(
            x[i] + 0.24,
            raw_weights[i] * 1.12,
            f"{int(round(raw_weights[i]))}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=90,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "offline_1500_cycles_article_figure.png", dpi=220)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary = load_summary()
    article_figure(summary)
    print(OUTPUT_DIR / "offline_1500_cycles_article_figure.png")


if __name__ == "__main__":
    main()
