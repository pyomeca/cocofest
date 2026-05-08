from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import biorbd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

from cocofest.dynamics.inverse_kinematics_and_dynamics import inverse_kinematics_cycling

ROOT = Path(__file__).resolve().parent
COCOFEST_ROOT = ROOT if (ROOT / "examples").exists() else ROOT / "cocofest"
OUTPUT_DIR = ROOT / "analysis_outputs"

MODEL_PATH = COCOFEST_ROOT / "examples" / "msk_models" / "Wu" / "Modified_Wu_Shoulder_Model_Cycling.bioMod"
IK_MODEL_PATH = COCOFEST_ROOT / "examples" / "msk_models" / "Wu" / "Modified_Wu_Shoulder_Model_Cycling_for_IK.bioMod"

MUSCLE_LIST = ["Delt_ant", "Delt_post", "Biceps", "Triceps"]
PARAMETERS = {
    "Biceps": {"Fmax": 149.0, "a_scale": 3314.7, "alpha_a": -5.6e-2, "tau_fat": 179.6, "pcsa": 7.33},
    "Triceps": {"Fmax": 262.0, "a_scale": 4915.5, "alpha_a": -3.4e-2, "tau_fat": 109.1, "pcsa": 10.87},
    "Delt_ant": {"Fmax": 48.0, "a_scale": 1148.6, "alpha_a": -1.4e-1, "tau_fat": 445.5, "pcsa": 2.54},
    "Delt_post": {"Fmax": 51.0, "a_scale": 1234.5, "alpha_a": -1.1e-1, "tau_fat": 342.7, "pcsa": 2.73},
}

TASK_TORQUE_THRESHOLD = 0.20
HIGH_DEMAND_FRACTION = 0.80


@dataclass
class FatigueSummary:
    duty_cycle: float
    rho_crit: float
    cycles_to_failure_high_demand: float | None
    asymptotic_capacity_ratio_at_high_demand: float
    fit_r2: dict[str, float]


def to_numpy(x):
    if hasattr(x, "to_array"):
        return np.array(x.to_array(), dtype=float).squeeze()
    return np.array(x, dtype=float).squeeze()


def name_to_str(x):
    if hasattr(x, "to_string"):
        return x.to_string()
    if hasattr(x, "toString"):
        return x.toString()
    return str(x)


def marker_dict(model, q):
    markers = model.markers(q)
    names = [name_to_str(m) for m in model.markerNames()]
    return {names[i]: to_numpy(markers[i])[:3] for i in range(len(markers))}


def hand_position(model, q):
    return marker_dict(model, q)["hand"][:2]


def wheel_center_position(model, q):
    return marker_dict(model, q)["global_wheel_center"][:2]


def hand_jacobian_fd(model, q, eps=1e-7):
    nq = len(q)
    jac = np.zeros((2, nq))
    for j in range(nq):
        dq = np.zeros(nq)
        dq[j] = eps
        jac[:, j] = (hand_position(model, q + dq) - hand_position(model, q - dq)) / (2 * eps)
    return jac


def equivalent_hand_force_xy_from_joint_torque(j_hand_xy, tau_q):
    f_xy, *_ = np.linalg.lstsq(j_hand_xy.T, tau_q, rcond=None)
    return f_xy


def gravity_balancing_hand_force_xy(j_hand_xy, tau_q_gravity, arm_dof_indices=(0, 1)):
    arm_dof_indices = tuple(arm_dof_indices)
    arm_jacobian_xy = j_hand_xy[:, arm_dof_indices]
    arm_gravity_torque = np.asarray(tau_q_gravity, dtype=float)[list(arm_dof_indices)]
    f_xy, *_ = np.linalg.lstsq(arm_jacobian_xy.T, -arm_gravity_torque, rcond=None)
    return f_xy


def useful_tangent_and_radius(hand_xy, center_xy):
    r = hand_xy - center_xy
    radius = np.linalg.norm(r)
    if radius < 1e-10:
        raise RuntimeError("Hand too close to wheel center")
    tangent = np.array([-r[1], r[0]]) / radius
    return tangent, radius


def make_states_one_muscle_active(model, muscle_index, activation=1.0):
    states = model.stateSet()
    for s in states:
        if hasattr(s, "setExcitation"):
            s.setExcitation(0.0)
        if hasattr(s, "setActivation"):
            s.setActivation(0.0)

    target = states[muscle_index]
    if hasattr(target, "setExcitation"):
        target.setExcitation(activation)
    if hasattr(target, "setActivation"):
        target.setActivation(activation)
    return states


def wrap_intervals(mask: np.ndarray, theta_deg: np.ndarray) -> list[tuple[float, float]]:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return []

    regions = []
    in_region = False
    start = None
    for i, value in enumerate(mask):
        if value and not in_region:
            start = i
            in_region = True
        elif not value and in_region:
            regions.append((start, i - 1))
            in_region = False
    if in_region:
        regions.append((start, len(mask) - 1))

    if len(regions) > 1 and mask[0] and mask[-1]:
        first = regions[0]
        last = regions[-1]
        merged = (last[0], first[1])
        regions = [merged] + regions[1:-1]

    return [(float(theta_deg[s]), float(theta_deg[e])) for s, e in regions]


def draw_interval_segments(ax, intervals, y, color="black", linestyle=":", linewidth=2.0, alpha=0.9, zorder=5):
    for start, end in intervals:
        if start <= end:
            ax.plot([start, end], [y, y], color=color, ls=linestyle, lw=linewidth, alpha=alpha, zorder=zorder)
        else:
            ax.plot([0.0, end], [y, y], color=color, ls=linestyle, lw=linewidth, alpha=alpha, zorder=zorder)
            ax.plot([start, 360.0], [y, y], color=color, ls=linestyle, lw=linewidth, alpha=alpha, zorder=zorder)


def shade_interval_regions(ax, intervals, color="#d00000", alpha=0.10, zorder=0):
    for start, end in intervals:
        if start <= end:
            ax.axvspan(start, end, color=color, alpha=alpha, zorder=zorder)
        else:
            ax.axvspan(0.0, end, color=color, alpha=alpha, zorder=zorder)
            ax.axvspan(start, 360.0, color=color, alpha=alpha, zorder=zorder)


def angular_edges(theta_deg: np.ndarray) -> np.ndarray:
    theta_deg = np.asarray(theta_deg, dtype=float)
    if theta_deg.ndim != 1 or theta_deg.size < 2:
        raise ValueError("theta_deg must be a 1D array with at least two samples")
    edges = np.empty(theta_deg.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (theta_deg[:-1] + theta_deg[1:])
    edges[0] = 0.0
    edges[-1] = 360.0
    return edges


def get_q_trajectory():
    q_guess, qdot_guess, _ = inverse_kinematics_cycling(
        str(IK_MODEL_PATH),
        120,
        x_center=0.35,
        y_center=0.0,
        radius=0.1,
        ik_method="trf",
        cycling_number=1,
    )
    return q_guess, qdot_guess


def compute_torque_profiles(return_gravity_profile: bool = False):
    model = biorbd.Model(str(MODEL_PATH))
    for i, muscle_name in enumerate(MUSCLE_LIST):
        model.muscle(i).setForceIsoMax(PARAMETERS[muscle_name]["Fmax"])

    q_ref, qdot_ref = get_q_trajectory()
    theta = np.mod(np.unwrap(q_ref[2, :]), 2 * np.pi)
    sort_idx = np.argsort(theta)
    theta = theta[sort_idx]
    q_ref = q_ref[:, sort_idx]
    qdot_ref = qdot_ref[:, sort_idx]

    torque_profiles = np.zeros((len(MUSCLE_LIST), q_ref.shape[1]))
    gravity_torque_profile = np.zeros(q_ref.shape[1])

    for k in range(q_ref.shape[1]):
        qk = q_ref[:, k]
        qdotk = qdot_ref[:, k]
        updated_model = model.UpdateKinematicsCustom(qk, qdotk)
        model.updateMuscles(updated_model, qk, qdotk)

        hand_xy = hand_position(model, qk)
        center_xy = wheel_center_position(model, qk)
        e_t, radius = useful_tangent_and_radius(hand_xy, center_xy)
        j_hand_xy = hand_jacobian_fd(model, qk)
        zero = np.zeros_like(qk)
        tau_q_gravity = to_numpy(model.InverseDynamics(qk, zero, zero))
        f_xy_gravity = gravity_balancing_hand_force_xy(j_hand_xy, tau_q_gravity, arm_dof_indices=(0, 1))
        gravity_torque_profile[k] = radius * float(np.dot(f_xy_gravity, e_t))

        for i, _ in enumerate(MUSCLE_LIST):
            states = make_states_one_muscle_active(model, i, activation=1.0)
            tau_q = to_numpy(model.muscularJointTorque(states, qk, qdotk))
            f_xy = equivalent_hand_force_xy_from_joint_torque(j_hand_xy, tau_q)
            torque_profiles[i, k] = radius * float(np.dot(f_xy, e_t))

    if return_gravity_profile:
        return theta, torque_profiles, gravity_torque_profile
    return theta, torque_profiles


def fatigue_discrete_map(a_value, a_rest, alpha_a, force_value, tau_fat, active_duration, rest_duration):
    exp_active = math.exp(-active_duration / tau_fat)
    exp_rest = math.exp(-rest_duration / tau_fat)
    beta = alpha_a * force_value * tau_fat

    a_after_active = (a_rest + beta) * (1 - exp_active) + exp_active * a_value
    a_after_rest = a_rest * (1 - exp_rest) + exp_rest * a_after_active
    return a_after_rest


def fit_fatigue_models(cycles: np.ndarray, deficits: np.ndarray) -> dict[str, float]:
    def linear_model(x, a, b):
        return a * x + b

    def quadratic_model(x, a, b, c):
        return a * x**2 + b * x + c

    def exponential_model(x, l_inf, rate, offset):
        return l_inf * (1 - np.exp(-rate * x)) + offset

    candidate_models = {
        "linear": (linear_model, [deficits[-1] / max(len(cycles), 1), 0.0]),
        "quadratic": (quadratic_model, [0.0, deficits[-1] / max(len(cycles), 1), 0.0]),
        "exponential": (exponential_model, [max(deficits[-1], 1e-6), 0.01, 0.0]),
    }

    fit_r2 = {}
    sst = float(np.sum((deficits - deficits.mean()) ** 2))
    for name, (fun, guess) in candidate_models.items():
        popt, _ = curve_fit(fun, cycles, deficits, p0=guess, maxfev=20000)
        predicted = fun(cycles, *popt)
        ssr = float(np.sum((deficits - predicted) ** 2))
        fit_r2[name] = 1.0 - ssr / sst if sst > 1e-12 else 1.0
    return fit_r2


def analyze_fatigue_for_muscle(duty_cycle: float, parameters: dict, rho=HIGH_DEMAND_FRACTION, max_cycles=1500):
    a_rest = parameters["a_scale"]
    alpha_a = parameters["alpha_a"]
    tau_fat = parameters["tau_fat"]
    f_max = parameters["Fmax"]

    active_duration = duty_cycle
    rest_duration = 1.0 - duty_cycle
    exp_active = math.exp(-active_duration / tau_fat)
    exp_rest = math.exp(-rest_duration / tau_fat)
    lam = exp_active * exp_rest
    slope = exp_rest * (1 - exp_active) * alpha_a * f_max * tau_fat / ((1 - lam) * a_rest)
    rho_crit = 1.0 / (1.0 - slope)

    a_inf = a_rest * (1.0 + slope * rho)
    capacity_ratio_inf = a_inf / a_rest
    force_demand = rho * f_max

    a_value = a_rest
    deficits = []
    cycles = []
    cycles_to_failure = None

    for cycle_idx in range(1, max_cycles + 1):
        a_value = fatigue_discrete_map(
            a_value=a_value,
            a_rest=a_rest,
            alpha_a=alpha_a,
            force_value=force_demand,
            tau_fat=tau_fat,
            active_duration=active_duration,
            rest_duration=rest_duration,
        )
        deficits.append(1.0 - a_value / a_rest)
        cycles.append(cycle_idx)
        if cycles_to_failure is None and a_value < rho * a_rest:
            cycles_to_failure = cycle_idx
            break

    cycles = np.array(cycles, dtype=float)
    deficits = np.array(deficits, dtype=float)
    fit_r2 = fit_fatigue_models(cycles, deficits)

    return (
        FatigueSummary(
            duty_cycle=duty_cycle,
            rho_crit=rho_crit,
            cycles_to_failure_high_demand=cycles_to_failure,
            asymptotic_capacity_ratio_at_high_demand=capacity_ratio_inf,
            fit_r2=fit_r2,
        ),
        cycles,
        deficits,
    )


def build_summary(theta, torque_profiles):
    theta_deg = np.degrees(theta)
    positive_torque = np.maximum(torque_profiles, 0.0)
    support_mask = positive_torque > 1e-9
    redundancy_count = support_mask.sum(axis=0)
    combined_positive_torque = positive_torque.sum(axis=0)

    base_deficit = np.maximum(TASK_TORQUE_THRESHOLD - combined_positive_torque, 0.0)
    base_deficit_area = float(np.trapezoid(base_deficit, theta))

    muscle_summaries = {}
    fatigue_curves = {}
    for i, muscle_name in enumerate(MUSCLE_LIST):
        positive_profile = positive_torque[i]
        duty_cycle = float(np.mean(support_mask[i]))
        unique_contribution = np.where(redundancy_count == 1, positive_profile, 0.0)
        contribution_when_shared = positive_profile / np.maximum(redundancy_count, 1)

        without_this_muscle = combined_positive_torque - positive_profile
        deficit_without_this_muscle = np.maximum(TASK_TORQUE_THRESHOLD - without_this_muscle, 0.0)
        extra_deficit_area = float(np.trapezoid(deficit_without_this_muscle - base_deficit, theta))

        fatigue_summary, cycles, deficits = analyze_fatigue_for_muscle(
            duty_cycle=duty_cycle,
            parameters=PARAMETERS[muscle_name],
        )
        fatigue_curves[muscle_name] = {"cycles": cycles, "deficits": deficits}

        muscle_summaries[muscle_name] = {
            "max_positive_torque_nm": float(np.max(positive_profile)),
            "max_negative_torque_nm": float(np.min(torque_profiles[i])),
            "support_fraction": duty_cycle,
            "support_intervals_deg": wrap_intervals(support_mask[i], theta_deg),
            "positive_area": float(np.trapezoid(positive_profile, theta)),
            "shared_contribution_area": float(np.trapezoid(contribution_when_shared, theta)),
            "unique_contribution_area": float(np.trapezoid(unique_contribution, theta)),
            "extra_deficit_area_if_removed": extra_deficit_area,
            "fatigue": {
                "rho_crit": fatigue_summary.rho_crit,
                "cycles_to_failure_high_demand": fatigue_summary.cycles_to_failure_high_demand,
                "asymptotic_capacity_ratio_at_high_demand": fatigue_summary.asymptotic_capacity_ratio_at_high_demand,
                "fit_r2": fatigue_summary.fit_r2,
            },
        }

    vulnerability_signal = {}
    for muscle_name in MUSCLE_LIST:
        cycles_to_failure = muscle_summaries[muscle_name]["fatigue"]["cycles_to_failure_high_demand"]
        if cycles_to_failure is None:
            vulnerability_signal[muscle_name] = 1e-4
        else:
            vulnerability_signal[muscle_name] = 1.0 / cycles_to_failure

    criticality_signal = {
        muscle_name: (
            muscle_summaries[muscle_name]["unique_contribution_area"]
            + muscle_summaries[muscle_name]["extra_deficit_area_if_removed"]
            + 0.25 * muscle_summaries[muscle_name]["shared_contribution_area"]
        )
        for muscle_name in MUSCLE_LIST
    }

    raw_weight_signal = {
        muscle_name: vulnerability_signal[muscle_name] * criticality_signal[muscle_name] for muscle_name in MUSCLE_LIST
    }
    min_non_zero = min(v for v in raw_weight_signal.values() if v > 0)
    candidate_fixed_weights = {
        muscle_name: float(max(raw_weight_signal[muscle_name] / min_non_zero, 1e-3)) for muscle_name in MUSCLE_LIST
    }

    summary = {
        "task_threshold_nm": TASK_TORQUE_THRESHOLD,
        "high_demand_fraction_for_fatigue": HIGH_DEMAND_FRACTION,
        "redundancy_regions": {
            "no_support_intervals_deg": wrap_intervals(redundancy_count == 0, theta_deg),
            "single_support_intervals_deg": wrap_intervals(redundancy_count == 1, theta_deg),
            "double_support_intervals_deg": wrap_intervals(redundancy_count == 2, theta_deg),
            "triple_or_more_support_intervals_deg": wrap_intervals(redundancy_count >= 3, theta_deg),
        },
        "threshold_regions": {
            "below_0p20_nm_intervals_deg": wrap_intervals(combined_positive_torque < TASK_TORQUE_THRESHOLD, theta_deg),
        },
        "base_deficit_area_at_threshold": base_deficit_area,
        "muscles": muscle_summaries,
        "candidate_fixed_weights_raw": raw_weight_signal,
        "candidate_fixed_weights_normalized": candidate_fixed_weights,
    }
    return summary, fatigue_curves, combined_positive_torque, redundancy_count


def plot_cycle_diagnostics(theta_deg, torque_profiles, redundancy_count, combined_positive_torque):
    fig = Figure(figsize=(12, 8.8))
    FigureCanvas(fig)
    axs = fig.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [2.5, 1.8, 1.2]})
    risk_intervals = wrap_intervals(combined_positive_torque < TASK_TORQUE_THRESHOLD, theta_deg)
    support_mask = torque_profiles > 1e-9
    theta_edges_deg = angular_edges(theta_deg)
    muscle_edges = np.arange(len(MUSCLE_LIST) + 1, dtype=float) - 0.5

    ax = axs[0]
    colors = {
        "Delt_ant": "#9c6644",
        "Delt_post": "#6c584c",
        "Biceps": "#1982c4",
        "Triceps": "#2a9d8f",
    }
    shade_interval_regions(ax, risk_intervals)
    for i, muscle_name in enumerate(MUSCLE_LIST):
        positive_profile = np.where(torque_profiles[i] > 0, torque_profiles[i], np.nan)
        negative_profile = np.where(torque_profiles[i] < 0, torque_profiles[i], np.nan)
        ax.plot(theta_deg, positive_profile, lw=2.2, label=muscle_name, color=colors[muscle_name])
        ax.plot(theta_deg, negative_profile, lw=2.2, ls=":", color=colors[muscle_name], alpha=0.95)
    ax.plot(theta_deg, combined_positive_torque, lw=3, color="#d00000", label="Sum of positive torques")
    ax.axhline(TASK_TORQUE_THRESHOLD, color="black", lw=1.5, ls="--", label="0.20 Nm threshold")
    ax.axhline(0.0, color="#555555", lw=1.0, ls="-", alpha=0.7)
    ax.set_xlim(0, 360)
    ax.set_ylabel("Tangential crank torque capacity (Nm)")
    ax.set_title("Tangential crank torque; dotted segments indicate negative torque")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=9)

    ax = axs[1]
    shade_interval_regions(ax, risk_intervals)
    ax.pcolormesh(
        theta_edges_deg,
        muscle_edges,
        support_mask,
        cmap="Greens",
        shading="flat",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_ylim(len(MUSCLE_LIST) - 0.5, -0.5)
    ax.set_yticks(range(len(MUSCLE_LIST)))
    ax.set_yticklabels(MUSCLE_LIST)
    ax.set_ylabel("Muscle")
    ax.set_title("Positive-support map: green = positive crank torque")

    ax = axs[2]
    shade_interval_regions(ax, risk_intervals)
    ax.fill_between(theta_deg, 0, redundancy_count, step="mid", color="#457b9d", alpha=0.9)
    ax.set_ylim(0, 4.5)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_ylabel("Count")
    ax.set_xlabel("Pedal angle (deg)")
    ax.set_title("Redundancy count")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cycle_diagnostics.png", dpi=180)
    fig.savefig(OUTPUT_DIR / "torque_capacity.png", dpi=180)
    fig.savefig(OUTPUT_DIR / "redundancy_map.png", dpi=180)


def plot_fatigue_curves(fatigue_curves):
    fig = Figure(figsize=(11, 7.5))
    FigureCanvas(fig)
    axs = fig.subplots(2, 2, sharex=False, sharey=False)
    axes = axs.flatten()

    for ax, muscle_name in zip(axes, MUSCLE_LIST):
        cycles = fatigue_curves[muscle_name]["cycles"]
        deficits = fatigue_curves[muscle_name]["deficits"]
        ax.plot(cycles, deficits, color="#1d3557", lw=2.2, label="Fatigue deficit")
        ax.set_title(muscle_name)
        ax.set_xlabel("Cycle index")
        ax.set_ylabel("Normalized fatigue deficit")
        ax.grid(alpha=0.25)

    fig.suptitle("Repeated same-effort cycle simulation at 80% of each muscle's own max profile", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fatigue_growth.png", dpi=180, bbox_inches="tight")


def plot_candidate_weights(summary):
    weights = summary["candidate_fixed_weights_normalized"]
    rho_crit = [summary["muscles"][m]["fatigue"]["rho_crit"] for m in MUSCLE_LIST]
    inverse_cycles_fail = []
    for muscle_name in MUSCLE_LIST:
        cycles_fail = summary["muscles"][muscle_name]["fatigue"]["cycles_to_failure_high_demand"]
        inverse_cycles_fail.append(0.0 if cycles_fail is None else 1.0 / cycles_fail)

    fig = Figure(figsize=(11, 4.8))
    FigureCanvas(fig)
    axs = fig.subplots(1, 2)
    axs[0].bar(MUSCLE_LIST, [weights[m] for m in MUSCLE_LIST], color=["#9c6644", "#6c584c", "#1982c4", "#2a9d8f"])
    axs[0].set_yscale("log")
    axs[0].set_ylabel("Normalized fixed weight (log scale)")
    axs[0].set_title("Candidate fixed weights from contribution x vulnerability")
    axs[0].grid(axis="y", alpha=0.25)

    x = np.arange(len(MUSCLE_LIST))
    axs[1].bar(x - 0.18, [1.0 - val for val in rho_crit], width=0.35, label="1 - sustainable fraction", color="#457b9d")
    axs[1].bar(x + 0.18, inverse_cycles_fail, width=0.35, label="1 / cycles to failure @ 80%", color="#e76f51")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(MUSCLE_LIST)
    axs[1].set_title("Fatigue vulnerability indicators")
    axs[1].set_yscale("log")
    axs[1].grid(axis="y", alpha=0.25)
    axs[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "candidate_weights.png", dpi=180)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    theta, torque_profiles = compute_torque_profiles()
    theta_deg = np.degrees(theta)
    summary, fatigue_curves, combined_positive_torque, redundancy_count = build_summary(theta, torque_profiles)

    plot_cycle_diagnostics(theta_deg, torque_profiles, redundancy_count, combined_positive_torque)
    plot_fatigue_curves(fatigue_curves)
    plot_candidate_weights(summary)

    summary_path = OUTPUT_DIR / "cycling_weight_exploration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Summary written to {summary_path}")
    print("Candidate fixed weights:")
    for muscle_name in MUSCLE_LIST:
        print(f"  {muscle_name}: {summary['candidate_fixed_weights_normalized'][muscle_name]:.3f}")


if __name__ == "__main__":
    main()
