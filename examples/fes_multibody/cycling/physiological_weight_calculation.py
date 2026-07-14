from __future__ import annotations

import time
import math
from pathlib import Path
import numpy as np

import biorbd
from cocofest.dynamics.inverse_kinematics_and_dynamics import inverse_kinematics_cycling

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "../../msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod"
IK_MODEL_PATH = ROOT / "../../msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling_for_IK.bioMod"

MUSCLE_LIST = ["Delt_ant", "Delt_post", "Biceps", "Triceps"]

PARAMETERS = {
    "Biceps": {
        "Fmax": 149.0,
        "a_scale": 3314.7,
        "alpha_a": -5.6e-2,
        "tau_fat": 179.6,
    },
    "Triceps": {
        "Fmax": 262.0,
        "a_scale": 4915.5,
        "alpha_a": -3.4e-2,
        "tau_fat": 109.1,
    },
    "Delt_ant": {
        "Fmax": 48.0,
        "a_scale": 1148.6,
        "alpha_a": -1.4e-1,
        "tau_fat": 445.5,
    },
    "Delt_post": {
        "Fmax": 51.0,
        "a_scale": 1234.5,
        "alpha_a": -1.1e-1,
        "tau_fat": 342.7,
    },
}

TASK_TORQUE_THRESHOLD = 0.20


# ============================================================
# Generic helpers
# ============================================================
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


def normalize_positive(signal, new_min=0.0, new_max=1.0):
    min_val = min(signal.values())
    max_val = max(signal.values())

    if max_val == min_val:
        return {key: float(new_min) for key in signal}

    return {
        key: float(new_min + (value - min_val) * (new_max - new_min) / (max_val - min_val))
        for key, value in signal.items()
    }


# ============================================================
# Geometry and torque helpers
# ============================================================


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


# ============================================================
# Equation n°1: torque profile
# ============================================================


def compute_torque_profiles():
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
            states = make_states_one_muscle_active(
                model,
                muscle_index,
                activation=1.0,
            )

            joint_torque = to_numpy(model.muscularJointTorque(states, qk, qdotk))

            force_xy = equivalent_hand_force_xy_from_joint_torque(
                hand_jacobian_xy,
                joint_torque,
            )

            torque_profiles[muscle_index, sample_index] = radius * float(np.dot(force_xy, tangent))

    return theta, torque_profiles


# ============================================================
# Fatigue model
# ============================================================


def fatigue_discrete_map(
    a_value,
    a_rest,
    alpha_a,
    force_value,
    tau_fat,
    active_duration,
    rest_duration,
):
    exp_active = math.exp(-active_duration / tau_fat)
    exp_rest = math.exp(-rest_duration / tau_fat)
    beta = alpha_a * force_value * tau_fat

    a_after_active = (a_rest + beta) * (1 - exp_active) + exp_active * a_value
    a_after_rest = a_rest * (1 - exp_rest) + exp_rest * a_after_active

    return a_after_rest


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


# ============================================================
# Mechanical contribution helpers
# ============================================================


def build_pre_risk_mask(
    theta_deg: np.ndarray,
    risk_mask: np.ndarray,
    pre_risk_width_deg: float = 120.0,
) -> np.ndarray:
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
    pre_risk_width_deg: float = 120.0,
) -> dict:
    theta_deg = np.degrees(theta)

    scaled_profiles = torque_profiles * ratios[:, None]
    positive_profiles = np.maximum(scaled_profiles, 0.0)

    redundancy_count = (positive_profiles > 1e-9).sum(axis=0)
    risk_mask = positive_profiles.sum(axis=0) < TASK_TORQUE_THRESHOLD

    pre_risk_mask = build_pre_risk_mask(
        theta_deg=theta_deg,
        risk_mask=risk_mask,
        pre_risk_width_deg=pre_risk_width_deg,
    )

    metric = {}

    for muscle_index, muscle_name in enumerate(MUSCLE_LIST):
        metric[muscle_name] = {
            "support_in_pre_risk_area": float(
                np.trapezoid(
                    positive_profiles[muscle_index] * pre_risk_mask,
                    theta,
                )
            ),
            "unique_support_area": float(
                np.trapezoid(
                    np.where(
                        redundancy_count == 1,
                        positive_profiles[muscle_index],
                        0.0,
                    ),
                    theta,
                )
            ),
        }

    return metric


# ============================================================
# Simplified weight calculation
# ============================================================


def weight_calculation(config: dict) -> dict:
    theta, torque_profiles = compute_torque_profiles()

    positive_torque = np.maximum(torque_profiles, 0.0)
    support_mask = positive_torque > 1e-9

    # Muscle fatigability
    fatigue_dynamics = {}
    fatigability = {}

    for muscle_index, muscle_name in enumerate(MUSCLE_LIST):
        duty_cycle = float(np.mean(support_mask[muscle_index]))

        fatigue_dynamics[muscle_name] = simulate_fatigue_ratios(
            duty_cycle=duty_cycle,
            parameters=PARAMETERS[muscle_name],
            total_cycles=config["target_cycles"],
            rho=config["rho"],
        )

        fatigability[muscle_name] = (fatigue_dynamics[muscle_name][0] - fatigue_dynamics[muscle_name][-1]) / config[
            "target_cycles"
        ]

    # Mechanical contribution
    support_in_pre_risk = {}
    unique_support = {}
    support_in_pre_risk_list = {name: [] for name in MUSCLE_LIST}
    unique_support_list = {name: [] for name in MUSCLE_LIST}
    mechanical_contribution = {name: 0.0 for name in MUSCLE_LIST}

    for cycle_index in range(config["target_cycles"]):
        ratios = np.array(
            [fatigue_dynamics[muscle_name][cycle_index] for muscle_name in MUSCLE_LIST],
            dtype=float,
        )

        metrics = feasibility_metrics(
            theta,
            torque_profiles,
            ratios,
            pre_risk_width_deg=config["pre_risk_width_deg"],
        )

        for muscle_name in MUSCLE_LIST:
            support_in_pre_risk_list[muscle_name].append(metrics[muscle_name]["support_in_pre_risk_area"])

            unique_support_list[muscle_name].append(metrics[muscle_name]["unique_support_area"])

    for muscle_name in MUSCLE_LIST:
        support_in_pre_risk[muscle_name] = float(np.mean(support_in_pre_risk_list[muscle_name]))

        unique_support[muscle_name] = float(np.mean(unique_support_list[muscle_name]))

        mechanical_contribution[muscle_name] = support_in_pre_risk[muscle_name] + unique_support[muscle_name]

    # Final weights
    raw_weights = {
        muscle_name: float(mechanical_contribution[muscle_name] * fatigability[muscle_name] ** 2)
        for muscle_name in MUSCLE_LIST
    }

    normalized_weights = normalize_positive(
        raw_weights,
    )

    return {
        "rho": float(config["rho"]),
        "target_cycles": int(config["target_cycles"]),
        "pre_risk_width_deg": float(config["pre_risk_width_deg"]),
        "theta": theta,
        "torque_profiles": torque_profiles,
        "positive_torque": positive_torque,
        "support_mask": support_mask,
        "fatigue_dynamics": fatigue_dynamics,
        "fatigability": fatigability,
        "support_in_pre_risk": support_in_pre_risk,
        "unique_support": unique_support,
        "mechanical_contribution": mechanical_contribution,
        "raw_weights": raw_weights,
        "normalized_weights": normalized_weights,
    }


# ============================================================
# Runner
# ============================================================


def main():
    config = {
        "target_cycles": 1500,
        "rho": 0.80,
        "pre_risk_width_deg": 90.0,
    }

    start = time.time()

    summary = weight_calculation(config)

    end = time.time()

    print(f"\nWeights were calculated in: {end - start:.2f} s")
    print("\nDerived candidate fixed weights:")

    for muscle_name in MUSCLE_LIST:
        print(f"  {muscle_name}: " f"{summary['normalized_weights'][muscle_name]:.3f}")


if __name__ == "__main__":
    main()
