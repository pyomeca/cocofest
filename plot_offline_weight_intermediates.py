from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Patch

import cycling_weight_exploration as cwe
import derive_offline_task_weights as dotw

OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_outputs"


def build_intermediate_summary(
    target_cycles: int,
    num_snapshots: int,
    rho: float,
    muscle_name: str,
    pre_risk_width_deg: float,
    include_gravity: bool,
) -> dict:
    if include_gravity:
        theta, torque_profiles, gravity_torque = cwe.compute_torque_profiles(return_gravity_profile=True)
    else:
        theta, torque_profiles = cwe.compute_torque_profiles()
        gravity_torque = np.zeros_like(theta)
    theta_deg = np.degrees(theta)
    positive_torque = np.maximum(torque_profiles, 0.0)
    combined_positive = gravity_torque + positive_torque.sum(axis=0)
    risk_mask_fresh = combined_positive < cwe.TASK_TORQUE_THRESHOLD
    pre_risk_mask_fresh = dotw.build_pre_risk_mask(
        theta_deg=theta_deg,
        risk_mask=risk_mask_fresh,
        pre_risk_width_deg=pre_risk_width_deg,
    )

    support_mask = positive_torque > 1e-9
    muscle_index = cwe.MUSCLE_LIST.index(muscle_name)
    duty_cycle = float(np.mean(support_mask[muscle_index]))
    fatigue_ratios = {
        name: dotw.simulate_fatigue_ratios(
            duty_cycle=float(np.mean(support_mask[i])),
            parameters=cwe.PARAMETERS[name],
            total_cycles=target_cycles,
            rho=rho,
        )
        for i, name in enumerate(cwe.MUSCLE_LIST)
    }
    intrinsic = {
        name: dotw.intrinsic_vulnerability(
            duty_cycle=float(np.mean(support_mask[i])),
            parameters=cwe.PARAMETERS[name],
            rho=rho,
            total_cycles=target_cycles,
        )
        for i, name in enumerate(cwe.MUSCLE_LIST)
    }

    snapshot_cycles = dotw.choose_snapshot_cycles(target_cycles, num_snapshots=num_snapshots)
    snapshot_rows = []
    deficit_areas = []
    criticality_terms_global_max = 0.0
    for cycle in snapshot_cycles:
        ratios = np.array([fatigue_ratios[m][cycle - 1] for m in cwe.MUSCLE_LIST], dtype=float)
        metrics = dotw.feasibility_metrics(
            theta,
            torque_profiles,
            ratios,
            gravity_torque=gravity_torque,
            pre_risk_width_deg=pre_risk_width_deg,
        )
        deficit_areas.append(metrics["deficit_area"])
        pm = metrics["per_muscle"][muscle_name]
        for name in cwe.MUSCLE_LIST:
            per_muscle_metrics = metrics["per_muscle"][name]
            criticality_terms_global_max = max(
                criticality_terms_global_max,
                per_muscle_metrics["restore_gain"],
                per_muscle_metrics["support_in_pre_risk_area"],
                per_muscle_metrics["support_in_risk_area"],
                per_muscle_metrics["unique_support_area"],
            )
        snapshot_rows.append(
            {
                "cycle": int(cycle),
                "fatigue_ratio": float(ratios[muscle_index]),
                "risk_fraction": float(metrics["risk_fraction"]),
                "deficit_area": float(metrics["deficit_area"]),
                "restore_gain": float(pm["restore_gain"]),
                "support_in_pre_risk_area": float(pm["support_in_pre_risk_area"]),
                "support_in_risk_area": float(pm["support_in_risk_area"]),
                "unique_support_area": float(pm["unique_support_area"]),
                "low_redundancy_support_area": float(pm["low_redundancy_support_area"]),
            }
        )

    max_deficit = max(deficit_areas) if deficit_areas else 1.0
    for row in snapshot_rows:
        severity = row["deficit_area"] / max(max_deficit, 1e-8)
        row["severity_weight"] = float(0.25 + 0.75 * severity)

    full_summary = dotw.derive_weights(
        total_cycles=target_cycles,
        num_snapshots=num_snapshots,
        rho=rho,
        pre_risk_width_deg=pre_risk_width_deg,
        include_gravity=include_gravity,
    )

    return {
        "muscle": muscle_name,
        "target_cycles": target_cycles,
        "rho": rho,
        "pre_risk_width_deg": pre_risk_width_deg,
        "include_gravity": include_gravity,
        "theta_deg": theta_deg.tolist(),
        "torque_profile": torque_profiles[muscle_index].tolist(),
        "gravity_torque": gravity_torque.tolist(),
        "combined_positive_torque": combined_positive.tolist(),
        "risk_intervals_deg": cwe.wrap_intervals(risk_mask_fresh, theta_deg),
        "pre_risk_intervals_deg": cwe.wrap_intervals(pre_risk_mask_fresh, theta_deg),
        "fatigue_curve_cycles": list(range(1, target_cycles + 1)),
        "fatigue_curve_ratio": fatigue_ratios[muscle_name].tolist(),
        "snapshot_rows": snapshot_rows,
        "intrinsic_vulnerability": intrinsic[muscle_name],
        "all_muscles": {
            "criticality_normalized": full_summary["criticality_normalized"],
            "trajectory_vulnerability_normalized": full_summary["trajectory_vulnerability_normalized"],
            "intrinsic_vulnerability_normalized": full_summary["intrinsic_vulnerability_normalized"],
            "combined_vulnerability": full_summary["combined_vulnerability"],
            "candidate_fixed_weights_normalized": full_summary["candidate_fixed_weights_normalized"],
        },
        "plot_ranges": {
            "torque_ymin": float(
                min(np.min(torque_profiles), np.min(combined_positive), cwe.TASK_TORQUE_THRESHOLD * -0.1)
            ),
            "torque_ymax": float(
                max(np.max(torque_profiles), np.max(combined_positive), cwe.TASK_TORQUE_THRESHOLD) * 1.05
            ),
            "criticality_ymax": float(max(criticality_terms_global_max * 1.05, 1e-6)),
        },
    }


def _shade(ax, intervals, color, alpha):
    for start, end in intervals:
        if start <= end:
            ax.axvspan(start, end, color=color, alpha=alpha, zorder=0)
        else:
            ax.axvspan(0.0, end, color=color, alpha=alpha, zorder=0)
            ax.axvspan(start, 360.0, color=color, alpha=alpha, zorder=0)


def plot_intermediates(summary: dict, stem: str) -> Path:
    muscle_name = summary["muscle"]
    theta_deg = np.array(summary["theta_deg"], dtype=float)
    torque_profile = np.array(summary["torque_profile"], dtype=float)
    gravity_torque = np.array(summary["gravity_torque"], dtype=float)
    combined_positive = np.array(summary["combined_positive_torque"], dtype=float)
    fatigue_cycles = np.array(summary["fatigue_curve_cycles"], dtype=float)
    fatigue_ratio = np.array(summary["fatigue_curve_ratio"], dtype=float)
    snapshots = summary["snapshot_rows"]

    fig = Figure(figsize=(13, 10))
    FigureCanvas(fig)
    axs = fig.subplots(2, 2)

    ax = axs[0, 0]
    _shade(ax, summary["pre_risk_intervals_deg"], "#ffb703", 0.16)
    _shade(ax, summary["risk_intervals_deg"], "#d00000", 0.12)
    positive_profile = np.where(torque_profile > 0, torque_profile, np.nan)
    negative_profile = np.where(torque_profile < 0, torque_profile, np.nan)
    ax.plot(theta_deg, positive_profile, lw=2.6, color="#1d3557", label=f"{muscle_name} positive")
    ax.plot(theta_deg, negative_profile, lw=2.6, ls=":", color="#1d3557", label=f"{muscle_name} negative")
    # Keep the gravity trace silent unless it reaches a visible magnitude
    # relative to the task. In the current model it is quasi null, so plotting
    # it usually adds clutter without changing interpretation.
    if np.max(np.abs(gravity_torque)) > dotw.GRAVITY_NEGLIGIBLE_TORQUE_NM:
        ax.plot(theta_deg, gravity_torque, lw=1.8, color="#7c3aed", alpha=0.9, label="Gravity quasi-static")
    ax.plot(theta_deg, combined_positive, lw=2.0, color="#d00000", alpha=0.8, label="Total positive support")
    ax.axhline(cwe.TASK_TORQUE_THRESHOLD, color="black", ls="--", lw=1.2, label="0.20 Nm threshold")
    ax.axhline(0.0, color="#666666", lw=1.0)
    ax.set_xlim(0, 360)
    ax.set_ylim(summary["plot_ranges"]["torque_ymin"], summary["plot_ranges"]["torque_ymax"])
    ax.set_xlabel("Pedal angle (deg)")
    ax.set_ylabel("Tangential torque (Nm)")
    ax.set_title(f"A. {muscle_name}: torque and critical angular zones")
    ax.grid(alpha=0.25)
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    legend_handles.extend(
        [
            Patch(facecolor="#ffb703", edgecolor="none", alpha=0.16),
            Patch(facecolor="#d00000", edgecolor="none", alpha=0.12),
        ]
    )
    legend_labels.extend(["Pre-risk zone", "Risk zone"])
    ax.legend(legend_handles, legend_labels, fontsize=8, ncol=3)

    ax = axs[0, 1]
    ax.plot(fatigue_cycles, fatigue_ratio, lw=2.6, color="#457b9d", label=r"$A/A_{rest}$ synthetic trajectory")
    snap_cycles = np.array([row["cycle"] for row in snapshots], dtype=float)
    snap_ratios = np.array([row["fatigue_ratio"] for row in snapshots], dtype=float)
    sev = np.array([row["severity_weight"] for row in snapshots], dtype=float)
    sc = ax.scatter(snap_cycles, snap_ratios, c=sev, cmap="magma", s=70, zorder=5, label="Snapshots")
    ax.set_xlabel("Synthetic cycle index")
    ax.set_ylabel(r"Fatigue ratio $A/A_{rest}$")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"B. {muscle_name}: synthetic fatigue trajectory")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.03, label="Severity weight")

    ax = axs[1, 0]
    ax.plot(snap_cycles, [row["restore_gain"] for row in snapshots], marker="o", lw=2.2, label="Restore gain")
    ax.plot(
        snap_cycles,
        [row["support_in_pre_risk_area"] for row in snapshots],
        marker="s",
        lw=2.0,
        label="Pre-risk support",
    )
    ax.plot(snap_cycles, [row["support_in_risk_area"] for row in snapshots], marker="^", lw=2.0, label="Risk support")
    ax.plot(snap_cycles, [row["unique_support_area"] for row in snapshots], marker="d", lw=2.0, label="Unique support")
    ax.set_xlabel("Synthetic cycle index")
    ax.set_ylabel("Area / gain")
    ax.set_ylim(0.0, summary["plot_ranges"]["criticality_ymax"])
    ax.set_title(f"C. {muscle_name}: snapshot-level criticality terms")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    ax = axs[1, 1]
    labels = [
        "Criticality\n(normalized)",
        "Trajectory vuln.\n(normalized)",
        "Intrinsic vuln.\n(normalized)",
        "Combined vuln.",
        "Final raw weight\n(normalized)",
    ]
    values = [
        summary["all_muscles"]["criticality_normalized"][muscle_name],
        summary["all_muscles"]["trajectory_vulnerability_normalized"][muscle_name],
        summary["all_muscles"]["intrinsic_vulnerability_normalized"][muscle_name],
        summary["all_muscles"]["combined_vulnerability"][muscle_name],
        summary["all_muscles"]["candidate_fixed_weights_normalized"][muscle_name],
    ]
    colors = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#e76f51"]
    ax.bar(np.arange(len(labels)), values, color=colors)
    ax.axvline(0.5, color="#555555", lw=1.2, ls=":")
    ax.axvline(3.5, color="#555555", lw=1.2, ls=":")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Value (log scale)")
    ax.set_title(f"D. {muscle_name}: from intermediate signals to final weight")
    ax.grid(axis="y", alpha=0.25)

    gravity_text = ""
    fig.suptitle(
        f"Offline weight derivation intermediates for {muscle_name} ({int(round(summary['pre_risk_width_deg']))} deg pre-risk{gravity_text})",
        y=0.99,
    )
    fig.tight_layout()
    out_path = OUTPUT_DIR / f"{stem}_{muscle_name}_intermediates.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Illustrate offline weighting intermediates for a selected muscle.")
    parser.add_argument("--muscle", choices=cwe.MUSCLE_LIST, default="Delt_ant")
    parser.add_argument("--target-cycles", type=int, default=1500)
    parser.add_argument("--num-snapshots", type=int, default=8)
    parser.add_argument("--rho", type=float, default=cwe.HIGH_DEMAND_FRACTION)
    parser.add_argument("--pre-risk-width-deg", type=float, default=120.0)
    parser.add_argument(
        "--include-gravity",
        action="store_true",
        help="Optional sensitivity check; quasi-static gravity is hidden from the plot when negligible.",
    )
    parser.add_argument(
        "--all-muscles", action="store_true", help="Generate one figure per muscle with shared y-scales."
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    gravity_suffix = "_with_gravity" if args.include_gravity else ""
    width_suffix = f"_{int(round(args.pre_risk_width_deg))}deg_prerisk"
    stem = f"offline_{args.target_cycles}_cycles{width_suffix}{gravity_suffix}"
    muscles = cwe.MUSCLE_LIST if args.all_muscles else [args.muscle]
    for muscle_name in muscles:
        summary = build_intermediate_summary(
            target_cycles=args.target_cycles,
            num_snapshots=args.num_snapshots,
            rho=args.rho,
            muscle_name=muscle_name,
            pre_risk_width_deg=args.pre_risk_width_deg,
            include_gravity=args.include_gravity,
        )
        out_json = OUTPUT_DIR / f"{stem}_{muscle_name}_intermediates.json"
        out_json.write_text(json.dumps(summary, indent=2))
        out_png = plot_intermediates(summary, stem)
        print(out_json)
        print(out_png)


if __name__ == "__main__":
    main()
