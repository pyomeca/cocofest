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
GRAVITY_NEGLIGIBLE_TORQUE_NM = 1e-3


def simulate_fatigue_ratios(
    duty_cycle: float,
    parameters: dict,
    total_cycles: int,
    rho: float,
) -> np.ndarray:
    a_rest = float(parameters["a_scale"])
    alpha_a = float(parameters["alpha_a"])
    tau_fat = float(parameters["tau_fat"])
    f_max = float(parameters["Fmax"])

    active_duration = duty_cycle
    rest_duration = 1.0 - duty_cycle
    force_demand = rho * f_max

    a_value = a_rest
    ratios = np.zeros(total_cycles, dtype=float)
    for cycle_idx in range(total_cycles):
        a_value = cwe.fatigue_discrete_map(
            a_value=a_value,
            a_rest=a_rest,
            alpha_a=alpha_a,
            force_value=force_demand,
            tau_fat=tau_fat,
            active_duration=active_duration,
            rest_duration=rest_duration,
        )
        ratios[cycle_idx] = a_value / a_rest
    return ratios


def intrinsic_vulnerability(duty_cycle: float, parameters: dict, rho: float, total_cycles: int) -> dict[str, float]:
    fatigue_summary, _, _ = cwe.analyze_fatigue_for_muscle(
        duty_cycle=duty_cycle, parameters=parameters, rho=rho, max_cycles=total_cycles
    )
    cycles_fail = fatigue_summary.cycles_to_failure_high_demand
    inv_cycles = 1e-4 if cycles_fail is None else 1.0 / float(cycles_fail)
    low_sustainable_fraction = max(1.0 - float(fatigue_summary.rho_crit), 1e-6)
    return {
        "inverse_cycles_to_failure": inv_cycles,
        "one_minus_rho_crit": low_sustainable_fraction,
    }


def normalize_positive(signal: dict[str, float]) -> dict[str, float]:
    positive_values = [float(v) for v in signal.values() if v > 0]
    scale = min(positive_values) if positive_values else 1.0
    return {k: float(v / scale) for k, v in signal.items()}


def choose_snapshot_cycles(total_cycles: int, num_snapshots: int) -> np.ndarray:
    anchors = np.unique(np.round(np.linspace(1, total_cycles, num_snapshots)).astype(int))
    return anchors


def build_pre_risk_mask(theta_deg: np.ndarray, risk_mask: np.ndarray, pre_risk_width_deg: float = 120.0) -> np.ndarray:
    theta_deg = np.asarray(theta_deg, dtype=float)
    risk_mask = np.asarray(risk_mask, dtype=bool)
    onset_mask = risk_mask & ~np.roll(risk_mask, 1)
    pre_risk_mask = np.zeros_like(risk_mask, dtype=bool)
    onset_angles = theta_deg[onset_mask]
    for onset_angle in onset_angles:
        angular_distance_to_onset = np.mod(onset_angle - theta_deg, 360.0)
        pre_risk_mask |= (angular_distance_to_onset > 0.0) & (angular_distance_to_onset <= pre_risk_width_deg)
    pre_risk_mask &= ~risk_mask
    return pre_risk_mask


def feasibility_metrics(
    theta: np.ndarray,
    torque_profiles: np.ndarray,
    ratios: np.ndarray,
    gravity_torque: np.ndarray | None = None,
    pre_risk_width_deg: float = 120.0,
) -> dict:
    theta_deg = np.degrees(theta)
    scaled_profiles = torque_profiles * ratios[:, None]
    positive_profiles = np.maximum(scaled_profiles, 0.0)
    # Keep the gravity contribution optional and diagnostic-only. In the current
    # Wu cycling model, the quasi-static hand force required to balance the arm
    # with zero shoulder/elbow torques projects to an almost null pedal torque,
    # so it has a negligible impact on the resulting weights.
    gravity_torque = np.zeros_like(theta) if gravity_torque is None else np.asarray(gravity_torque, dtype=float)
    combined_positive = gravity_torque + positive_profiles.sum(axis=0)
    deficit = np.maximum(cwe.TASK_TORQUE_THRESHOLD - combined_positive, 0.0)
    redundancy_count = (positive_profiles > 1e-9).sum(axis=0)
    risk_mask = combined_positive < cwe.TASK_TORQUE_THRESHOLD
    pre_risk_mask = build_pre_risk_mask(theta_deg=theta_deg, risk_mask=risk_mask, pre_risk_width_deg=pre_risk_width_deg)

    per_muscle = {}
    for i, muscle_name in enumerate(cwe.MUSCLE_LIST):
        without_i = combined_positive - positive_profiles[i]
        deficit_without_i = np.maximum(cwe.TASK_TORQUE_THRESHOLD - without_i, 0.0)

        restored_ratios = ratios.copy()
        restored_ratios[i] = 1.0
        restored_combined = np.maximum(torque_profiles * restored_ratios[:, None], 0.0).sum(axis=0)
        deficit_if_restored = np.maximum(cwe.TASK_TORQUE_THRESHOLD - restored_combined, 0.0)

        per_muscle[muscle_name] = {
            "restore_gain": float(np.trapezoid(deficit - deficit_if_restored, theta)),
            "extra_deficit_if_removed": float(np.trapezoid(deficit_without_i - deficit, theta)),
            "support_in_risk_area": float(np.trapezoid(positive_profiles[i] * risk_mask, theta)),
            "support_in_pre_risk_area": float(np.trapezoid(positive_profiles[i] * pre_risk_mask, theta)),
            "unique_support_area": float(
                np.trapezoid(np.where(redundancy_count == 1, positive_profiles[i], 0.0), theta)
            ),
            "low_redundancy_support_area": float(
                np.trapezoid(np.where(redundancy_count <= 2, positive_profiles[i], 0.0), theta)
            ),
        }

    return {
        "risk_fraction": float(np.mean(combined_positive < cwe.TASK_TORQUE_THRESHOLD)),
        "deficit_area": float(np.trapezoid(deficit, theta)),
        "gravity_support_mean": float(np.mean(gravity_torque)),
        "gravity_support_min": float(np.min(gravity_torque)),
        "gravity_support_max": float(np.max(gravity_torque)),
        "per_muscle": per_muscle,
    }


def derive_weights(total_cycles: int, num_snapshots: int, rho: float, pre_risk_width_deg: float, include_gravity: bool):
    if include_gravity:
        # This path is intentionally kept as a quiet sensitivity check. For the
        # present model, the gravity-induced pedal torque stays orders of
        # magnitude below the 0.20 Nm task threshold and does not materially
        # modify the offline ranking.
        theta, torque_profiles, gravity_torque = cwe.compute_torque_profiles(return_gravity_profile=True)
    else:
        theta, torque_profiles = cwe.compute_torque_profiles()
        gravity_torque = np.zeros_like(theta)
    positive_torque = np.maximum(torque_profiles, 0.0)
    support_mask = positive_torque > 1e-9

    fatigue_ratio_trajectories = {}
    vulnerability_intrinsic = {}
    for i, muscle_name in enumerate(cwe.MUSCLE_LIST):
        duty_cycle = float(np.mean(support_mask[i]))
        fatigue_ratio_trajectories[muscle_name] = simulate_fatigue_ratios(
            duty_cycle=duty_cycle,
            parameters=cwe.PARAMETERS[muscle_name],
            total_cycles=total_cycles,
            rho=rho,
        )
        vulnerability_intrinsic[muscle_name] = intrinsic_vulnerability(
            duty_cycle=duty_cycle,
            parameters=cwe.PARAMETERS[muscle_name],
            rho=rho,
            total_cycles=total_cycles,
        )

    snapshots = []
    for cycle in choose_snapshot_cycles(total_cycles, num_snapshots=num_snapshots):
        ratios = np.array([fatigue_ratio_trajectories[m][cycle - 1] for m in cwe.MUSCLE_LIST], dtype=float)
        metrics = feasibility_metrics(
            theta,
            torque_profiles,
            ratios,
            gravity_torque=gravity_torque,
            pre_risk_width_deg=pre_risk_width_deg,
        )
        snapshots.append((cycle, ratios, metrics))

    max_deficit = max(snapshot[2]["deficit_area"] for snapshot in snapshots)
    criticality = {m: 0.0 for m in cwe.MUSCLE_LIST}
    trajectory_vulnerability = {m: 0.0 for m in cwe.MUSCLE_LIST}
    snapshot_rows = []

    for cycle, ratios, metrics in snapshots:
        severity = metrics["deficit_area"] / max(max_deficit, 1e-8)
        snapshot_weight = 0.25 + 0.75 * severity
        row = {
            "cycle": int(cycle),
            "severity_weight": float(snapshot_weight),
            "risk_fraction": float(metrics["risk_fraction"]),
            "deficit_area": float(metrics["deficit_area"]),
            "gravity_support_mean": float(metrics["gravity_support_mean"]),
            "ratios": {m: float(ratios[i]) for i, m in enumerate(cwe.MUSCLE_LIST)},
            "restore_gain": {},
        }
        for i, muscle_name in enumerate(cwe.MUSCLE_LIST):
            restore_gain = metrics["per_muscle"][muscle_name]["restore_gain"]
            support_risk = metrics["per_muscle"][muscle_name]["support_in_risk_area"]
            support_pre_risk = metrics["per_muscle"][muscle_name]["support_in_pre_risk_area"]
            unique_support = metrics["per_muscle"][muscle_name]["unique_support_area"]
            low_redundancy_support = metrics["per_muscle"][muscle_name]["low_redundancy_support_area"]
            criticality[muscle_name] += snapshot_weight * (
                restore_gain + 0.75 * unique_support + 1.00 * support_pre_risk + 0.10 * support_risk
            )
            trajectory_vulnerability[muscle_name] += snapshot_weight * (1.0 - ratios[i])
            row["restore_gain"][muscle_name] = float(restore_gain)
        snapshot_rows.append(row)

    intrinsic_combined = {
        m: np.sqrt(
            vulnerability_intrinsic[m]["inverse_cycles_to_failure"] * vulnerability_intrinsic[m]["one_minus_rho_crit"]
        )
        for m in cwe.MUSCLE_LIST
    }

    criticality_norm = normalize_positive(criticality)
    trajectory_norm = normalize_positive(trajectory_vulnerability)
    intrinsic_norm = normalize_positive(intrinsic_combined)
    combined_vulnerability = {m: float(np.sqrt(trajectory_norm[m] * intrinsic_norm[m])) for m in cwe.MUSCLE_LIST}
    vulnerability_exponent = 2.0
    raw_weights = {
        m: float(criticality_norm[m] * (combined_vulnerability[m] ** vulnerability_exponent)) for m in cwe.MUSCLE_LIST
    }
    normalized_weights = normalize_positive(raw_weights)
    sqrt_compressed = {m: float(np.sqrt(v)) for m, v in normalized_weights.items()}
    fourth_root_compressed = {m: float(v**0.25) for m, v in normalized_weights.items()}

    return {
        "rho_for_offline_fatigue": float(rho),
        "target_cycles": int(total_cycles),
        "pre_risk_width_deg": float(pre_risk_width_deg),
        "include_gravity": bool(include_gravity),
        "gravity_support_summary": {
            "mean": float(np.mean(gravity_torque)),
            "min": float(np.min(gravity_torque)),
            "max": float(np.max(gravity_torque)),
        },
        "snapshot_cycles": [row["cycle"] for row in snapshot_rows],
        "snapshots": snapshot_rows,
        "criticality_signal": criticality,
        "criticality_normalized": criticality_norm,
        "trajectory_vulnerability": trajectory_vulnerability,
        "trajectory_vulnerability_normalized": trajectory_norm,
        "intrinsic_vulnerability_components": vulnerability_intrinsic,
        "intrinsic_vulnerability_combined": intrinsic_combined,
        "intrinsic_vulnerability_normalized": intrinsic_norm,
        "combined_vulnerability": combined_vulnerability,
        "vulnerability_exponent": vulnerability_exponent,
        "candidate_fixed_weights_raw": raw_weights,
        "candidate_fixed_weights_normalized": normalized_weights,
        "candidate_fixed_weights_sqrt_compressed": sqrt_compressed,
        "candidate_fixed_weights_fourth_root_compressed": fourth_root_compressed,
    }


def plot_offline_weight_derivation(summary: dict, stem: str):
    fig = Figure(figsize=(12, 9))
    FigureCanvas(fig)
    axes = fig.subplots(3, 1)

    cycles = [row["cycle"] for row in summary["snapshots"]]
    deficit = [row["deficit_area"] for row in summary["snapshots"]]
    risk_fraction = [row["risk_fraction"] for row in summary["snapshots"]]

    ax = axes[0]
    ax.plot(cycles, deficit, marker="o", lw=2.0, label="Deficit area")
    ax.plot(cycles, risk_fraction, marker="s", lw=2.0, label="Risk fraction")
    ax.set_xlabel("Synthetic cycle index")
    ax.set_title("Offline synthetic fatigue trajectory severity")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    for muscle_name in cwe.MUSCLE_LIST:
        values = [row["restore_gain"][muscle_name] for row in summary["snapshots"]]
        ax.plot(cycles, values, marker="o", lw=2.0, label=muscle_name)
    ax.set_xlabel("Synthetic cycle index")
    ax.set_ylabel("Deficit reduction if restored")
    ax.set_title("Offline restore-gain trajectories")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)

    ax = axes[2]
    x = np.arange(len(cwe.MUSCLE_LIST))
    sqrt_weights = [summary["candidate_fixed_weights_sqrt_compressed"][m] for m in cwe.MUSCLE_LIST]
    fourth_root = [summary["candidate_fixed_weights_fourth_root_compressed"][m] for m in cwe.MUSCLE_LIST]
    raw = [summary["candidate_fixed_weights_normalized"][m] for m in cwe.MUSCLE_LIST]
    ax.bar(x - 0.25, raw, width=0.25, label="Raw normalized")
    ax.bar(x, sqrt_weights, width=0.25, label="Sqrt-compressed")
    ax.bar(x + 0.25, fourth_root, width=0.25, label="Fourth-root compressed")
    ax.set_xticks(x)
    ax.set_xticklabels(cwe.MUSCLE_LIST)
    ax.set_yscale("log")
    ax.set_title("Offline task/physiology-derived fixed weights")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}_offline_weight_derivation.png", dpi=180)


def main():
    parser = argparse.ArgumentParser(
        description="Derive constant muscle weights from offline task and physiology only."
    )
    parser.add_argument("--target-cycles", type=int, default=1500, help="Synthetic endurance horizon.")
    parser.add_argument("--num-snapshots", type=int, default=8, help="Number of synthetic snapshots.")
    parser.add_argument("--rho", type=float, default=cwe.HIGH_DEMAND_FRACTION, help="Standardized effort fraction.")
    parser.add_argument("--pre-risk-width-deg", type=float, default=120.0, help="Angular width before risk onset.")
    parser.add_argument(
        "--include-gravity",
        action="store_true",
        help="Optional sensitivity check; quasi-static gravity impact is negligible in the current model.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    summary = derive_weights(
        total_cycles=args.target_cycles,
        num_snapshots=args.num_snapshots,
        rho=args.rho,
        pre_risk_width_deg=args.pre_risk_width_deg,
        include_gravity=args.include_gravity,
    )
    gravity_suffix = "_with_gravity" if args.include_gravity else ""
    width_suffix = f"_{int(round(args.pre_risk_width_deg))}deg_prerisk"
    stem = f"offline_{args.target_cycles}_cycles{width_suffix}{gravity_suffix}"
    plot_offline_weight_derivation(summary, stem)
    out_json = OUTPUT_DIR / f"{stem}_weight_derivation.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(out_json)
    print(OUTPUT_DIR / f"{stem}_offline_weight_derivation.png")


if __name__ == "__main__":
    main()
