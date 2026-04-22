import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import biorbd
from cocofest.dynamics.inverse_kinematics_and_dynamics import inverse_kinematics_cycling

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis_outputs"

MODEL_PATH = ROOT / "../../msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod"
IK_MODEL_PATH = ROOT / "../../msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling_for_IK.bioMod"

MUSCLE_LIST = ["Delt_ant", "Delt_post", "Biceps", "Triceps"]
PARAMETERS = {
    "Biceps": {"Fmax": 149.0, "a_scale": 3314.7, "alpha_a": -5.6e-2, "tau_fat": 179.6, "pcsa": 7.33},
    "Triceps": {"Fmax": 262.0, "a_scale": 4915.5, "alpha_a": -3.4e-2, "tau_fat": 109.1, "pcsa": 10.87},
    "Delt_ant": {"Fmax": 48.0, "a_scale": 1148.6, "alpha_a": -1.4e-1, "tau_fat": 445.5, "pcsa": 2.54},
    "Delt_post": {"Fmax": 51.0, "a_scale": 1234.5, "alpha_a": -1.1e-1, "tau_fat": 342.7, "pcsa": 2.73},
}

TASK_TORQUE_THRESHOLD = 0.20
HIGH_DEMAND_FRACTION = 0.80
DEFAULT_GRAVITY_TORQUE_NM = 0.0


def to_numpy(value):
    if hasattr(value, "to_array"):
        return np.array(value.to_array(), dtype=float).squeeze()
    return np.array(value, dtype=float).squeeze()


def name_to_str(value):
    if hasattr(value, "to_string"):
        return value.to_string()
    if hasattr(value, "toString"):
        return value.toString()
    return str(value)


def marker_dict(model, q):
    markers = model.markers(q)
    names = [name_to_str(marker_name) for marker_name in model.markerNames()]
    return {names[i]: to_numpy(markers[i])[:3] for i in range(len(markers))}


def hand_position(model, q):
    return marker_dict(model, q)["hand"][:2]


def wheel_center_position(model, q):
    return marker_dict(model, q)["global_wheel_center"][:2]


def hand_jacobian_fd(model, q, eps=1e-7):
    nq = len(q)
    jacobian = np.zeros((2, nq))
    for joint_index in range(nq):
        dq = np.zeros(nq)
        dq[joint_index] = eps
        jacobian[:, joint_index] = (hand_position(model, q + dq) - hand_position(model, q - dq)) / (2 * eps)
    return jacobian


def equivalent_hand_force_xy_from_joint_torque(hand_jacobian_xy, joint_torque):
    force_xy, *_ = np.linalg.lstsq(hand_jacobian_xy.T, joint_torque, rcond=None)
    return force_xy


def useful_tangent_and_radius(hand_xy, center_xy):
    radius_vector = hand_xy - center_xy
    radius = np.linalg.norm(radius_vector)
    if radius < 1e-10:
        raise RuntimeError("Hand too close to wheel center")
    tangent = np.array([-radius_vector[1], radius_vector[0]]) / radius
    return tangent, radius


def make_states_one_muscle_active(model, muscle_index, activation=1.0):
    states = model.stateSet()
    for state in states:
        if hasattr(state, "setExcitation"):
            state.setExcitation(0.0)
        if hasattr(state, "setActivation"):
            state.setActivation(0.0)

    target_state = states[muscle_index]
    if hasattr(target_state, "setExcitation"):
        target_state.setExcitation(activation)
    if hasattr(target_state, "setActivation"):
        target_state.setActivation(activation)
    return states


def wrap_intervals(mask: np.ndarray, theta_deg: np.ndarray) -> list[tuple[float, float]]:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return []

    regions = []
    in_region = False
    start = None

    for index, value in enumerate(mask):
        if value and not in_region:
            start = index
            in_region = True
        elif not value and in_region:
            regions.append((start, index - 1))
            in_region = False

    if in_region:
        regions.append((start, len(mask) - 1))

    if len(regions) > 1 and mask[0] and mask[-1]:
        first = regions[0]
        last = regions[-1]
        regions = [(last[0], first[1])] + regions[1:-1]

    return [(float(theta_deg[start]), float(theta_deg[end])) for start, end in regions]


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


def ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


def save_figure(fig: Figure, *paths: Path, dpi: int = 180, bbox_inches=None):
    FigureCanvas(fig)
    for path in paths:
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)


def get_q_trajectory():
    q_guess, qdot_guess, _ = inverse_kinematics_cycling(
        str(IK_MODEL_PATH),
        120,
        x_center=0.35,  # Position of the crank axis
        y_center=0.0,
        radius=0.1,
        ik_method="trf",
        cycling_number=1,
    )
    return q_guess, qdot_guess


def gravity_torque_profile(theta: np.ndarray) -> np.ndarray:
    return np.full_like(theta, DEFAULT_GRAVITY_TORQUE_NM, dtype=float)


def compute_torque_profiles(return_gravity_profile: bool = False):
    model = biorbd.Model(str(MODEL_PATH))
    for muscle_index, muscle_name in enumerate(MUSCLE_LIST):
        model.muscle(muscle_index).setForceIsoMax(PARAMETERS[muscle_name]["Fmax"])

    q_ref, qdot_ref = get_q_trajectory()
    theta = np.mod(np.unwrap(q_ref[2, :]), 2 * np.pi)
    sort_index = np.argsort(theta)
    theta = theta[sort_index]
    q_ref = q_ref[:, sort_index]
    qdot_ref = qdot_ref[:, sort_index]

    torque_profiles = np.zeros((len(MUSCLE_LIST), q_ref.shape[1]))

    for sample_index in range(q_ref.shape[1]):
        qk = q_ref[:, sample_index]
        qdotk = qdot_ref[:, sample_index]
        updated_model = model.UpdateKinematicsCustom(qk, qdotk)
        model.updateMuscles(updated_model, qk, qdotk)

        hand_xy = hand_position(model, qk)
        center_xy = wheel_center_position(model, qk)
        tangent, radius = useful_tangent_and_radius(hand_xy, center_xy)
        hand_jacobian_xy = hand_jacobian_fd(model, qk)

        for muscle_index, _ in enumerate(MUSCLE_LIST):
            states = make_states_one_muscle_active(model, muscle_index, activation=1.0)
            joint_torque = to_numpy(model.muscularJointTorque(states, qk, qdotk))
            force_xy = equivalent_hand_force_xy_from_joint_torque(hand_jacobian_xy, joint_torque)
            torque_profiles[muscle_index, sample_index] = radius * float(np.dot(force_xy, tangent))

    if return_gravity_profile:
        return theta, torque_profiles, gravity_torque_profile(theta)
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

    def exponential_model(x, limit_value, rate, offset):
        return limit_value * (1 - np.exp(-rate * x)) + offset

    candidate_models = {
        "linear": (linear_model, [deficits[-1] / max(len(cycles), 1), 0.0]),
        "quadratic": (quadratic_model, [0.0, deficits[-1] / max(len(cycles), 1), 0.0]),
        "exponential": (exponential_model, [max(deficits[-1], 1e-6), 0.01, 0.0]),
    }

    fit_r2 = {}
    sst = float(np.sum((deficits - deficits.mean()) ** 2))
    for model_name, (function, guess) in candidate_models.items():
        try:
            params, _ = curve_fit(function, cycles, deficits, p0=guess, maxfev=20000)
            predicted = function(cycles, *params)
            ssr = float(np.sum((deficits - predicted) ** 2))
            fit_r2[model_name] = 1.0 - ssr / sst if sst > 1e-12 else 1.0
        except Exception:
            fit_r2[model_name] = float("nan")
    return fit_r2


def analyze_fatigue_for_muscle(duty_cycle: float, parameters: dict, rho=HIGH_DEMAND_FRACTION, max_cycles=1500):
    a_rest = float(parameters["a_scale"])
    alpha_a = float(parameters["alpha_a"])
    tau_fat = float(parameters["tau_fat"])
    f_max = float(parameters["Fmax"])

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

    cycles_array = np.array(cycles, dtype=float)
    deficits_array = np.array(deficits, dtype=float)
    fit_r2 = fit_fatigue_models(cycles_array, deficits_array)

    summary = {
        "duty_cycle": duty_cycle,
        "rho_crit": rho_crit,
        "cycles_to_failure_high_demand": cycles_to_failure,
        "asymptotic_capacity_ratio_at_high_demand": capacity_ratio_inf,
        "fit_r2": fit_r2,
    }

    return summary, cycles_array, deficits_array


def build_cycle_analyse_summary(theta, torque_profiles):
    theta_deg = np.degrees(theta)
    positive_torque = np.maximum(torque_profiles, 0.0)
    support_mask = positive_torque > 1e-9
    redundancy_count = support_mask.sum(axis=0)
    combined_positive_torque = positive_torque.sum(axis=0)

    base_deficit = np.maximum(TASK_TORQUE_THRESHOLD - combined_positive_torque, 0.0)
    base_deficit_area = float(np.trapezoid(base_deficit, theta))

    muscle_summaries = {}
    fatigue_curves = {}

    for muscle_index, muscle_name in enumerate(MUSCLE_LIST):
        positive_profile = positive_torque[muscle_index]
        duty_cycle = float(np.mean(support_mask[muscle_index]))
        unique_contribution = np.where(redundancy_count == 1, positive_profile, 0.0)
        shared_contribution = positive_profile / np.maximum(redundancy_count, 1)

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
            "max_negative_torque_nm": float(np.min(torque_profiles[muscle_index])),
            "support_fraction": duty_cycle,
            "support_intervals_deg": wrap_intervals(support_mask[muscle_index], theta_deg),
            "positive_area": float(np.trapezoid(positive_profile, theta)),
            "shared_contribution_area": float(np.trapezoid(shared_contribution, theta)),
            "unique_contribution_area": float(np.trapezoid(unique_contribution, theta)),
            "extra_deficit_area_if_removed": extra_deficit_area,
            "fatigue": {
                "rho_crit": fatigue_summary["rho_crit"],
                "cycles_to_failure_high_demand": fatigue_summary["cycles_to_failure_high_demand"],
                "asymptotic_capacity_ratio_at_high_demand": fatigue_summary["asymptotic_capacity_ratio_at_high_demand"],
                "fit_r2": fatigue_summary["fit_r2"],
            },
        }

    vulnerability_signal = {}
    for muscle_name in MUSCLE_LIST:
        cycles_to_failure = muscle_summaries[muscle_name]["fatigue"]["cycles_to_failure_high_demand"]
        vulnerability_signal[muscle_name] = 1e-4 if cycles_to_failure is None else 1.0 / cycles_to_failure

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
    candidate_fixed_weights = normalize_positive(raw_weight_signal)

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


def plot_cycle_analysis(theta_deg, torque_profiles, redundancy_count, combined_positive_torque, save):
    fig, axes = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(12, 8.8),
        gridspec_kw={"height_ratios": [2.5, 1.8, 1.2]},
    )

    risk_intervals = wrap_intervals(combined_positive_torque < TASK_TORQUE_THRESHOLD, theta_deg)
    support_mask = torque_profiles > 1e-9
    theta_edges_deg = angular_edges(theta_deg)
    muscle_edges = np.arange(len(MUSCLE_LIST) + 1, dtype=float) - 0.5

    colors = {
        "Delt_ant": "#9c6644",
        "Delt_post": "#6c584c",
        "Biceps": "#1982c4",
        "Triceps": "#2a9d8f",
    }

    ax = axes[0]
    shade_interval_regions(ax, risk_intervals)
    for muscle_index, muscle_name in enumerate(MUSCLE_LIST):
        positive_profile = np.where(torque_profiles[muscle_index] > 0, torque_profiles[muscle_index], np.nan)
        negative_profile = np.where(torque_profiles[muscle_index] < 0, torque_profiles[muscle_index], np.nan)
        ax.plot(theta_deg, positive_profile, lw=2.2, label=muscle_name, color=colors[muscle_name])
        ax.plot(theta_deg, negative_profile, lw=2.2, ls=":", color=colors[muscle_name], alpha=0.95)
    ax.plot(theta_deg, combined_positive_torque, lw=3, color="#d00000", label="Sum of positive torques")
    ax.axhline(TASK_TORQUE_THRESHOLD, color="black", lw=1.5, ls="--", label="0.20 Nm threshold")
    ax.axhline(0.0, color="#555555", lw=1.0, alpha=0.7)
    ax.set_xlim(0, 360)
    ax.set_ylabel("Tangential crank torque capacity (Nm)")
    ax.set_title("Tangential crank torque; dotted segments indicate negative torque")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=9)

    ax = axes[1]
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

    ax = axes[2]
    shade_interval_regions(ax, risk_intervals)
    ax.fill_between(theta_deg, 0, redundancy_count, step="mid", color="#457b9d", alpha=0.9)
    ax.set_ylim(0, 4.5)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_ylabel("Count")
    ax.set_xlabel("Pedal angle (deg)")
    ax.set_title("Redundancy count")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    plt.show()

    if save:
        save_figure(
            fig,
            OUTPUT_DIR / "cycle_analysis.png",
        )


def plot_fatigue_curves(fatigue_curves, save):
    fig, axes = plt.subplots(
        2,
        2,
        sharex=False,
        sharey=False,
        figsize=(11, 7.5),
    )
    axes = axes.flatten()

    for ax, muscle_name in zip(axes, MUSCLE_LIST):
        cycles = fatigue_curves[muscle_name]["cycles"]
        deficits = fatigue_curves[muscle_name]["deficits"]
        ax.plot(cycles, deficits, color="#1d3557", lw=2.2)
        ax.set_title(muscle_name)
        ax.set_xlabel("Cycle index")
        ax.set_ylabel("Normalized fatigue deficit")
        ax.grid(alpha=0.25)

    fig.suptitle("Repeated same-effort cycle simulation at 80% of each muscle's own max profile", y=1.02)
    fig.tight_layout()
    plt.show()

    if save:
        save_figure(fig, OUTPUT_DIR / "fatigue_growth.png", bbox_inches="tight")


def plot_candidate_weights(summary, save):
    weights = summary["candidate_fixed_weights_normalized"]
    rho_crit = [summary["muscles"][muscle_name]["fatigue"]["rho_crit"] for muscle_name in MUSCLE_LIST]
    inverse_cycles_to_failure = []
    for muscle_name in MUSCLE_LIST:
        cycles_to_failure = summary["muscles"][muscle_name]["fatigue"]["cycles_to_failure_high_demand"]
        inverse_cycles_to_failure.append(0.0 if cycles_to_failure is None else 1.0 / cycles_to_failure)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.8),
    )

    axes[0].bar(
        MUSCLE_LIST,
        [weights[muscle_name] for muscle_name in MUSCLE_LIST],
        color=["#9c6644", "#6c584c", "#1982c4", "#2a9d8f"],
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Normalized fixed weight (log scale)")
    axes[0].set_title("Candidate fixed weights from contribution x vulnerability")
    axes[0].grid(axis="y", alpha=0.25)

    x_positions = np.arange(len(MUSCLE_LIST))
    axes[1].bar(
        x_positions - 0.18,
        [1.0 - value for value in rho_crit],
        width=0.35,
        label="1 - sustainable fraction",
        color="#457b9d",
    )
    axes[1].bar(
        x_positions + 0.18, inverse_cycles_to_failure, width=0.35, label="1 / cycles to failure @ 80%", color="#e76f51"
    )
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(MUSCLE_LIST)
    axes[1].set_title("Fatigue vulnerability indicators")
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    plt.show()

    if save:
        save_figure(fig, OUTPUT_DIR / "candidate_weights.png")


def simulate_fatigue_ratios(duty_cycle: float, parameters: dict, total_cycles: int, rho: float) -> np.ndarray:
    a_rest = float(parameters["a_scale"])
    alpha_a = float(parameters["alpha_a"])
    tau_fat = float(parameters["tau_fat"])
    f_max = float(parameters["Fmax"])

    active_duration = duty_cycle
    rest_duration = 1.0 - duty_cycle
    force_demand = rho * f_max

    a_value = a_rest
    ratios = np.zeros(total_cycles, dtype=float)
    for cycle_index in range(total_cycles):
        a_value = fatigue_discrete_map(
            a_value=a_value,
            a_rest=a_rest,
            alpha_a=alpha_a,
            force_value=force_demand,
            tau_fat=tau_fat,
            active_duration=active_duration,
            rest_duration=rest_duration,
        )
        ratios[cycle_index] = a_value / a_rest
    return ratios


def intrinsic_vulnerability(duty_cycle: float, parameters: dict, rho: float, total_cycles: int) -> dict[str, float]:
    fatigue_summary, _, _ = analyze_fatigue_for_muscle(
        duty_cycle=duty_cycle,
        parameters=parameters,
        rho=rho,
        max_cycles=total_cycles,
    )
    cycles_to_failure = fatigue_summary["cycles_to_failure_high_demand"]
    inverse_cycles = 1e-4 if cycles_to_failure is None else 1.0 / float(cycles_to_failure)
    low_sustainable_fraction = max(1.0 - float(fatigue_summary["rho_crit"]), 1e-6)
    return {
        "inverse_cycles_to_failure": inverse_cycles,
        "one_minus_rho_crit": low_sustainable_fraction,
    }


def normalize_positive(signal: dict[str, float]) -> dict[str, float]:
    positive_values = [float(value) for value in signal.values() if value > 0]
    scale = min(positive_values) if positive_values else 1.0
    return {key: float(value / scale) for key, value in signal.items()}


def choose_snapshot_cycles(total_cycles: int, num_snapshots: int) -> np.ndarray:
    return np.unique(np.round(np.linspace(1, total_cycles, num_snapshots)).astype(int))


def build_pre_risk_mask(theta_deg: np.ndarray, risk_mask: np.ndarray, pre_risk_width_deg: float = 120.0) -> np.ndarray:
    theta_deg = np.asarray(theta_deg, dtype=float)
    risk_mask = np.asarray(risk_mask, dtype=bool)
    onset_mask = risk_mask & ~np.roll(risk_mask, 1)
    pre_risk_mask = np.zeros_like(risk_mask, dtype=bool)

    for onset_angle in theta_deg[onset_mask]:
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
    # Where muscle torque contribution is inferior at cycling resistive torque
    theta_deg = np.degrees(theta)
    scaled_profiles = torque_profiles * ratios[:, None]
    positive_profiles = np.maximum(scaled_profiles, 0.0)
    gravity_torque = np.zeros_like(theta) if gravity_torque is None else np.asarray(gravity_torque, dtype=float)
    combined_positive = gravity_torque + positive_profiles.sum(axis=0)
    deficit = np.maximum(TASK_TORQUE_THRESHOLD - combined_positive, 0.0)
    redundancy_count = (positive_profiles > 1e-9).sum(axis=0)
    risk_mask = combined_positive < TASK_TORQUE_THRESHOLD
    pre_risk_mask = build_pre_risk_mask(theta_deg=theta_deg, risk_mask=risk_mask, pre_risk_width_deg=pre_risk_width_deg)

    per_muscle = {}
    for muscle_index, muscle_name in enumerate(MUSCLE_LIST):
        without_this_muscle = combined_positive - positive_profiles[muscle_index]
        deficit_without_this_muscle = np.maximum(TASK_TORQUE_THRESHOLD - without_this_muscle, 0.0)

        restored_ratios = ratios.copy()
        restored_ratios[muscle_index] = 1.0
        restored_combined = gravity_torque + np.maximum(torque_profiles * restored_ratios[:, None], 0.0).sum(axis=0)
        deficit_if_restored = np.maximum(TASK_TORQUE_THRESHOLD - restored_combined, 0.0)

        per_muscle[muscle_name] = {
            "restore_gain": float(np.trapezoid(deficit - deficit_if_restored, theta)),
            "extra_deficit_if_removed": float(np.trapezoid(deficit_without_this_muscle - deficit, theta)),
            "support_in_risk_area": float(np.trapezoid(positive_profiles[muscle_index] * risk_mask, theta)),
            "support_in_pre_risk_area": float(np.trapezoid(positive_profiles[muscle_index] * pre_risk_mask, theta)),
            "unique_support_area": float(
                np.trapezoid(np.where(redundancy_count == 1, positive_profiles[muscle_index], 0.0), theta)
            ),
            "low_redundancy_support_area": float(
                np.trapezoid(np.where(redundancy_count <= 2, positive_profiles[muscle_index], 0.0), theta)
            ),
        }

    return {
        "risk_fraction": float(np.mean(risk_mask)),
        "deficit_area": float(np.trapezoid(deficit, theta)),
        "gravity_support_mean": float(np.mean(gravity_torque)),
        "gravity_support_min": float(np.min(gravity_torque)),
        "gravity_support_max": float(np.max(gravity_torque)),
        "per_muscle": per_muscle,
    }


def derive_offline_weights(config: dict) -> dict:
    if config["include_gravity"]:
        theta, torque_profiles, gravity_torque = compute_torque_profiles(return_gravity_profile=True)
    else:
        theta, torque_profiles = compute_torque_profiles()
        gravity_torque = np.zeros_like(theta)

    positive_torque = np.maximum(torque_profiles, 0.0)
    support_mask = positive_torque > 1e-9

    # Physiological
    fatigue_ratio_trajectories = {}
    vulnerability_intrinsic = {}
    for muscle_index, muscle_name in enumerate(MUSCLE_LIST):
        duty_cycle = float(np.mean(support_mask[muscle_index]))
        fatigue_ratio_trajectories[muscle_name] = simulate_fatigue_ratios(
            duty_cycle=duty_cycle,
            parameters=PARAMETERS[muscle_name],
            total_cycles=config["target_cycles"],
            rho=config["rho"],
        )
        vulnerability_intrinsic[muscle_name] = intrinsic_vulnerability(
            duty_cycle=duty_cycle,
            parameters=PARAMETERS[muscle_name],
            rho=config["rho"],
            total_cycles=config["target_cycles"],
        )

    # Task contribution
    snapshots = []
    for cycle in choose_snapshot_cycles(config["target_cycles"], num_snapshots=config["num_snapshots"]):
        ratios = np.array(
            [fatigue_ratio_trajectories[muscle_name][cycle - 1] for muscle_name in MUSCLE_LIST], dtype=float
        )
        metrics = feasibility_metrics(
            theta,
            torque_profiles,
            ratios,
            gravity_torque=gravity_torque,
            pre_risk_width_deg=config["pre_risk_width_deg"],
        )
        snapshots.append((cycle, ratios, metrics))

    max_deficit = max(snapshot[2]["deficit_area"] for snapshot in snapshots)
    criticality = {muscle_name: 0.0 for muscle_name in MUSCLE_LIST}
    trajectory_vulnerability = {muscle_name: 0.0 for muscle_name in MUSCLE_LIST}
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
            "ratios": {muscle_name: float(ratios[i]) for i, muscle_name in enumerate(MUSCLE_LIST)},
            "restore_gain": {},
        }

        for muscle_index, muscle_name in enumerate(MUSCLE_LIST):
            restore_gain = metrics["per_muscle"][muscle_name]["restore_gain"]
            support_in_risk = metrics["per_muscle"][muscle_name]["support_in_risk_area"]
            support_in_pre_risk = metrics["per_muscle"][muscle_name]["support_in_pre_risk_area"]
            unique_support = metrics["per_muscle"][muscle_name]["unique_support_area"]

            criticality[muscle_name] += snapshot_weight * (
                restore_gain + 0.75 * unique_support + 1.00 * support_in_pre_risk + 0.10 * support_in_risk
            )
            trajectory_vulnerability[muscle_name] += snapshot_weight * (1.0 - ratios[muscle_index])
            row["restore_gain"][muscle_name] = float(restore_gain)

        snapshot_rows.append(row)

    intrinsic_combined = {
        muscle_name: np.sqrt(
            vulnerability_intrinsic[muscle_name]["inverse_cycles_to_failure"]
            * vulnerability_intrinsic[muscle_name]["one_minus_rho_crit"]
        )
        for muscle_name in MUSCLE_LIST
    }

    criticality_norm = normalize_positive(criticality)
    trajectory_norm = normalize_positive(trajectory_vulnerability)
    intrinsic_norm = normalize_positive(intrinsic_combined)
    combined_vulnerability = {
        muscle_name: float(np.sqrt(trajectory_norm[muscle_name] * intrinsic_norm[muscle_name]))
        for muscle_name in MUSCLE_LIST
    }

    vulnerability_exponent = 2.0
    raw_weights = {
        muscle_name: float(
            criticality_norm[muscle_name] * (combined_vulnerability[muscle_name] ** vulnerability_exponent)
        )
        for muscle_name in MUSCLE_LIST
    }
    normalized_weights = normalize_positive(raw_weights)

    return {
        "rho_for_offline_fatigue": float(config["rho"]),
        "target_cycles": int(config["target_cycles"]),
        "pre_risk_width_deg": float(config["pre_risk_width_deg"]),
        "include_gravity": bool(config["include_gravity"]),
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
        "candidate_fixed_weights_sqrt_compressed": {m: float(np.sqrt(v)) for m, v in normalized_weights.items()},
        "candidate_fixed_weights_fourth_root_compressed": {m: float(v**0.25) for m, v in normalized_weights.items()},
    }


def plot_offline_weight_derivation(summary: dict, stem: str, show_plot: bool, save: bool):
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 9),
    )

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
    for muscle_name in MUSCLE_LIST:
        values = [row["restore_gain"][muscle_name] for row in summary["snapshots"]]
        ax.plot(cycles, values, marker="o", lw=2.0, label=muscle_name)
    ax.set_xlabel("Synthetic cycle index")
    ax.set_ylabel("Deficit reduction if restored")
    ax.set_title("Offline restore-gain trajectories")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)

    ax = axes[2]
    x_positions = np.arange(len(MUSCLE_LIST))
    raw = [summary["candidate_fixed_weights_normalized"][muscle_name] for muscle_name in MUSCLE_LIST]
    sqrt_weights = [summary["candidate_fixed_weights_sqrt_compressed"][muscle_name] for muscle_name in MUSCLE_LIST]
    fourth_root = [
        summary["candidate_fixed_weights_fourth_root_compressed"][muscle_name] for muscle_name in MUSCLE_LIST
    ]
    ax.bar(x_positions - 0.25, raw, width=0.25, label="Raw normalized")
    ax.bar(x_positions, sqrt_weights, width=0.25, label="Sqrt-compressed")
    ax.bar(x_positions + 0.25, fourth_root, width=0.25, label="Fourth-root compressed")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(MUSCLE_LIST)
    ax.set_yscale("log")
    ax.set_title("Offline task/physiology-derived fixed weights")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    if show_plot:
        plt.show()
    if save:
        save_figure(fig, OUTPUT_DIR / f"{stem}_offline_weight_derivation.png")


def run_cycle_analyse(show_plot=False, save=False):
    ensure_output_dir()
    theta, torque_profiles = compute_torque_profiles()
    theta_deg = np.degrees(theta)
    summary, fatigue_curves, combined_positive_torque, redundancy_count = build_cycle_analyse_summary(
        theta, torque_profiles
    )

    if show_plot:
        plot_cycle_analysis(theta_deg, torque_profiles, redundancy_count, combined_positive_torque, save)
        plot_fatigue_curves(fatigue_curves, save)
        plot_candidate_weights(summary, save)

    if save:
        summary_path = OUTPUT_DIR / "cycling_weight_exploration_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        return summary_path, summary
    else:
        return None, summary


def run_weight_derivation(config: dict, show_plot=False, save=False):
    ensure_output_dir()
    summary = derive_offline_weights(config)

    gravity_suffix = "_with_gravity" if config["include_gravity"] else ""
    width_suffix = f"_{int(round(config["pre_risk_width_deg"]))}deg_prerisk"
    stem = f"offline_{config["target_cycles"]}_cycles{width_suffix}{gravity_suffix}"

    if show_plot or save:
        plot_offline_weight_derivation(summary, stem, show_plot, save)

    if save:
        output_json = OUTPUT_DIR / f"{stem}_weight_derivation.json"
        output_json.write_text(json.dumps(summary, indent=2))
        return output_json, summary

    else:
        return None, summary


def main():
    config = {
        "target_cycles": 1500,
        "num_snapshots": 8,
        "rho": 0.80,
        "pre_risk_width_deg": 90.0,
        "include_gravity": False,
        "mode": "both",
    }

    show_plot = True
    save = True

    if config["mode"] in {"analyse", "both"}:
        summary_path, summary = run_cycle_analyse(show_plot, save)
        if save:
            print(f"Analysis of cycling summary saved in {summary_path}")
        print("Candidate fixed weights based on task:")
        for muscle_name in MUSCLE_LIST:
            print(f"  {muscle_name}: {summary['candidate_fixed_weights_normalized'][muscle_name]:.3f}")

    if config["mode"] in {"derive", "both"}:
        output_json, summary = run_weight_derivation(config, show_plot, save)
        if save:
            print(f"Weight derivation written to {output_json}")
        print("Derive candidate fixed weights:")
        for muscle_name in MUSCLE_LIST:
            print(f"  {muscle_name}: {summary['candidate_fixed_weights_normalized'][muscle_name]:.3f}")


if __name__ == "__main__":
    main()
