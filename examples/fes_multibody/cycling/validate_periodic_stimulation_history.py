"""
Validate periodic Ding stimulation-history surrogates against the base Ding model.

The base Ding 2007 fatigue model evaluates the calcium driving term from the
truncated list of previous stimulation times. The periodic variant replaces this
history by the additional state Cn_sum. This script compares a mean-intensity
periodic approximation and a pulse-aware periodic state update with the same
stimulation times, pulse-width command history, and optional IPOPT-derived
effective stimulation intensity history.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

STATE_LABELS = ("Cn", "Cn_sum", "F", "A", "Tau1", "Km")
COMPARABLE_STATE_LABELS = ("Cn", "F", "A", "Tau1", "Km")


@dataclass(frozen=True)
class Ding2007Parameters:
    tauc: float = 0.011
    r0_km_relationship: float = 1.04
    a_scale: float = 4920.0
    tau1_rest: float = 0.060601
    tau2: float = 0.001
    km_rest: float = 0.137
    pd0: float = 0.000131405
    pdt: float = 0.000194138
    alpha_a: float = -4.0 * 10e-2
    tau_fat: float = 127.0
    alpha_tau1: float = 2.1 * 10e-6
    alpha_km: float = 1.9 * 10e-6


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-duration", type=float, default=1.0)
    parser.add_argument("--n-cycles", type=int, default=2)
    parser.add_argument("--prehistory-cycles", type=int, default=5)
    parser.add_argument("--stimulations-per-cycle", type=int, default=30)
    parser.add_argument("--sum-stim-truncation", type=int, default=20)
    parser.add_argument("--substeps-per-stimulation", type=int, default=20)
    parser.add_argument("--pulse-width", type=float, default=0.000365702)
    parser.add_argument(
        "--pulse-width-profile",
        choices=("constant", "sinusoidal", "staircase"),
        default="constant",
    )
    parser.add_argument(
        "--activation-profile",
        choices=(
            "constant",
            "sinusoidal",
            "staircase",
            "alternating",
            "random",
            "ipopt",
        ),
        default="constant",
        help=(
            "Effective stimulation intensity profile used as lambda_i in the base "
            "calcium history. Use 'ipopt' to derive lambda_i from a saved IPOPT "
            "solution."
        ),
    )
    parser.add_argument("--activation-low", type=float, default=0.2)
    parser.add_argument("--activation-high", type=float, default=1.8)
    parser.add_argument("--activation-clip-low", type=float, default=None)
    parser.add_argument("--activation-clip-high", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--ipopt-data-file",
        type=Path,
        default=None,
        help=(
            "Saved IPOPT .npz solution used when --activation-profile ipopt. "
            "Defaults to result/initial_guess/2_initial_guess_collocation_3_radau.npz "
            "when it exists."
        ),
    )
    parser.add_argument("--ipopt-muscle", default="Triceps")
    parser.add_argument("--ipopt-control-key", default=None)
    parser.add_argument(
        "--ipopt-activation-source",
        choices=("pulse_width", "fatigue", "pulse_width_and_fatigue"),
        default="pulse_width_and_fatigue",
    )
    parser.add_argument("--fatigue-activation-gain", type=float, default=8.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "result"
        / "periodic_stimulation_history_validation",
    )
    parser.add_argument("--show", action="store_true")
    return parser


def pulse_width_at_time(
    time: float,
    cycle_duration: float,
    base_pulse_width: float,
    profile: str,
) -> float:
    phase = (time % cycle_duration) / cycle_duration
    if profile == "sinusoidal":
        return float(base_pulse_width * (1.0 + 0.25 * np.sin(2.0 * np.pi * phase)))
    if profile == "staircase":
        return float(base_pulse_width * (0.85 if phase < 0.5 else 1.15))
    return float(base_pulse_width)


def finite_stimulation_history(
    stim_times: np.ndarray,
    time: float,
    truncation: int,
) -> np.ndarray:
    history = stim_times[stim_times <= time + 1e-12]
    if history.size == 0:
        return np.array([-1e7], dtype=float)
    return history[-truncation:]


def finite_stimulation_history_indices(
    stim_times: np.ndarray,
    time: float,
    truncation: int,
) -> np.ndarray:
    indices = np.where(stim_times <= time + 1e-12)[0]
    if indices.size == 0:
        return np.array([], dtype=int)
    return indices[-truncation:]


def build_activation_values(
    stim_times: np.ndarray,
    cycle_duration: float,
    args: argparse.Namespace,
    parameters: Ding2007Parameters,
) -> np.ndarray:
    profile = args.activation_profile
    low = args.activation_low
    high = args.activation_high
    if low < 0.0 or high < 0.0:
        raise ValueError("--activation-low and --activation-high must be non-negative.")
    if high < low:
        raise ValueError("--activation-high must be >= --activation-low.")
    if (
        args.activation_clip_low is not None
        and args.activation_clip_high is not None
        and args.activation_clip_high < args.activation_clip_low
    ):
        raise ValueError("--activation-clip-high must be >= --activation-clip-low.")

    if profile == "constant":
        return np.ones(stim_times.shape)

    if profile == "ipopt":
        values = load_ipopt_activation_values(
            stim_times, cycle_duration, args, parameters
        )
        return clip_activation_values(values, args)

    phase = (stim_times % cycle_duration) / cycle_duration
    mid = 0.5 * (low + high)
    amplitude = 0.5 * (high - low)
    if profile == "sinusoidal":
        return mid + amplitude * np.sin(2.0 * np.pi * phase)
    if profile == "staircase":
        return np.where(phase < 0.5, low, high)
    if profile == "alternating":
        return np.where(np.arange(stim_times.size) % 2 == 0, low, high)

    rng = np.random.default_rng(args.random_seed)
    return rng.uniform(low, high, size=stim_times.shape)


def clip_activation_values(values: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    clipped = np.asarray(values, dtype=float)
    if args.activation_clip_low is not None:
        clipped = np.maximum(clipped, args.activation_clip_low)
    if args.activation_clip_high is not None:
        clipped = np.minimum(clipped, args.activation_clip_high)
    return clipped


def load_ipopt_activation_values(
    stim_times: np.ndarray,
    cycle_duration: float,
    args: argparse.Namespace,
    parameters: Ding2007Parameters,
) -> np.ndarray:
    data_file = args.ipopt_data_file or (
        Path(__file__).resolve().parent
        / "result"
        / "initial_guess"
        / "2_initial_guess_collocation_3_radau.npz"
    )
    if not data_file.exists():
        raise FileNotFoundError(
            f"Could not find IPOPT data file '{data_file}'. Pass --ipopt-data-file."
        )

    with np.load(data_file, allow_pickle=True) as data:
        pulse_width_values, source_times = load_ipopt_pulse_width_series(
            data,
            cycle_duration,
            args,
        )
        activation = np.ones(pulse_width_values.shape)
        if args.ipopt_activation_source in ("pulse_width", "pulse_width_and_fatigue"):
            activation *= pulse_width_to_activation(
                parameters,
                pulse_width_values,
                args.pulse_width,
            )
        if args.ipopt_activation_source in ("fatigue", "pulse_width_and_fatigue"):
            activation *= fatigue_compensation_from_ipopt(
                data,
                source_times,
                args,
            )

    return sample_periodic_series_at_stimulations(
        stim_times,
        source_times,
        activation,
    )


def load_ipopt_pulse_width_series(
    data: np.lib.npyio.NpzFile,
    cycle_duration: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    key = args.ipopt_control_key or f"last_pulse_width_{args.ipopt_muscle}"
    if key not in data.files:
        available_controls = sorted(
            k for k in data.files if k.startswith("last_pulse_width_")
        )
        if not available_controls:
            raise KeyError(
                "No last_pulse_width_* control found in the IPOPT data file."
            )
        key = available_controls[0]

    pulse_width_values = np.asarray(data[key], dtype=float).squeeze()
    if pulse_width_values.ndim != 1:
        raise ValueError(f"IPOPT control '{key}' must be a one-dimensional series.")

    source_times = build_ipopt_control_times(
        data, pulse_width_values.size, cycle_duration, args
    )
    return pulse_width_values, source_times


def build_ipopt_control_times(
    data: np.lib.npyio.NpzFile,
    value_count: int,
    cycle_duration: float,
    args: argparse.Namespace,
) -> np.ndarray:
    if "stim_time" in data.files:
        stim_time = np.asarray(data["stim_time"], dtype=float).squeeze()
        if stim_time.ndim == 1 and stim_time.size == value_count:
            return stim_time

    if "n_shooting_per_cycle" in data.files:
        n_per_cycle = int(np.asarray(data["n_shooting_per_cycle"]).item())
    else:
        n_per_cycle = args.stimulations_per_cycle
    stim_interval = cycle_duration / n_per_cycle
    return np.arange(value_count, dtype=float) * stim_interval


def pulse_width_to_activation(
    parameters: Ding2007Parameters,
    pulse_width_values: np.ndarray,
    reference_pulse_width: float,
) -> np.ndarray:
    denominator = pulse_width_recruitment(parameters, reference_pulse_width)
    if denominator <= 0.0:
        raise ValueError(
            "--pulse-width must be above pd0 to define an activation reference."
        )
    return pulse_width_recruitment(parameters, pulse_width_values) / denominator


def pulse_width_recruitment(
    parameters: Ding2007Parameters,
    pulse_width_values: float | np.ndarray,
) -> np.ndarray:
    pulse_width_array = np.asarray(pulse_width_values, dtype=float)
    effective_pulse_width = np.maximum(0.0, pulse_width_array - parameters.pd0)
    return 1.0 - np.exp(-effective_pulse_width / parameters.pdt)


def fatigue_compensation_from_ipopt(
    data: np.lib.npyio.NpzFile,
    source_times: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    key = f"A_{args.ipopt_muscle}"
    if key not in data.files or "time" not in data.files:
        return np.ones(source_times.shape)

    time = np.asarray(data["time"], dtype=float).squeeze()
    a_values = np.asarray(data[key], dtype=float).squeeze()
    if time.ndim != 1 or a_values.ndim != 1 or time.size != a_values.size:
        return np.ones(source_times.shape)

    sampled_a = np.interp(source_times, time, a_values)
    rest_a = max(float(sampled_a[0]), 1e-12)
    fatigue_index = np.maximum(0.0, (rest_a - sampled_a) / rest_a)
    return 1.0 + args.fatigue_activation_gain * fatigue_index


def sample_periodic_series_at_stimulations(
    stim_times: np.ndarray,
    source_times: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    if values.size == 0:
        raise ValueError("Cannot sample an empty IPOPT activation series.")
    if values.size == 1:
        return np.full(stim_times.shape, float(values[0]))

    source_times = np.asarray(source_times, dtype=float)
    values = np.asarray(values, dtype=float)
    dt = float(np.median(np.diff(source_times)))
    period = float(source_times[-1] - source_times[0] + dt)
    wrapped_times = np.mod(stim_times - source_times[0], period)
    indices = np.floor(wrapped_times / dt + 0.5).astype(int) % values.size
    return values[indices]


def time_is_stimulation_node(stim_times: np.ndarray, time: float) -> bool:
    return bool(np.any(np.isclose(stim_times, time, rtol=0.0, atol=1e-12)))


def pre_stimulation_endpoint_time(
    rhs_time: float,
    step_end_time: float,
    endpoint_is_stimulation: bool,
) -> float:
    if endpoint_is_stimulation and np.isclose(
        rhs_time, step_end_time, rtol=0.0, atol=1e-12
    ):
        return float(rhs_time - 1e-10)
    return rhs_time


def activation_at_time(
    stim_times: np.ndarray,
    activation_values: np.ndarray,
    time: float,
) -> float:
    previous_indices = np.where(stim_times <= time + 1e-12)[0]
    if previous_indices.size == 0:
        return float(activation_values[0])
    return float(activation_values[previous_indices[-1]])


def base_cn_sum(
    parameters: Ding2007Parameters,
    stim_times: np.ndarray,
    activation_values: np.ndarray,
    time: float,
    truncation: int,
) -> float:
    history_indices = finite_stimulation_history_indices(stim_times, time, truncation)
    if history_indices.size == 0:
        return 0.0

    history = stim_times[history_indices]
    history_activation = activation_values[history_indices]
    r0 = parameters.km_rest + parameters.r0_km_relationship
    value = 0.0
    for idx, (stim_time, activation) in enumerate(
        zip(history, history_activation, strict=True)
    ):
        if idx == 0:
            ri = 1.0
        else:
            ri = 1.0 + (r0 - 1.0) * np.exp(
                -(stim_time - history[idx - 1]) / parameters.tauc
            )
        value += ri * np.exp(-(time - stim_time) / parameters.tauc) * activation
    return float(value)


def cn_dot(parameters: Ding2007Parameters, cn: float, cn_sum: float) -> float:
    return (cn_sum - cn) / parameters.tauc


def a_calculation(
    parameters: Ding2007Parameters,
    a_scale: float,
    pulse_width: float,
) -> float:
    return float(
        a_scale * (1.0 - np.exp(-(pulse_width - parameters.pd0) / parameters.pdt))
    )


def f_dot(
    parameters: Ding2007Parameters,
    cn: float,
    force: float,
    a: float,
    tau1: float,
    km: float,
) -> float:
    calcium_ratio = cn / (km + cn) if km + cn != 0.0 else 0.0
    return float(a * calcium_ratio - force / (tau1 + parameters.tau2 * calcium_ratio))


def fatigue_rhs(
    parameters: Ding2007Parameters,
    force: float,
    a: float,
    tau1: float,
    km: float,
) -> tuple[float, float, float]:
    a_dot = -(a - parameters.a_scale) / parameters.tau_fat + parameters.alpha_a * force
    tau1_dot = (
        -(tau1 - parameters.tau1_rest) / parameters.tau_fat
        + parameters.alpha_tau1 * force
    )
    km_dot = (
        -(km - parameters.km_rest) / parameters.tau_fat + parameters.alpha_km * force
    )
    return float(a_dot), float(tau1_dot), float(km_dot)


def periodic_cn_sum_gain(
    parameters: Ding2007Parameters,
    stim_interval: float,
    activation: float = 1.0,
) -> float:
    decay = np.exp(-stim_interval / parameters.tauc)
    ri = 1.0 + ((parameters.km_rest + parameters.r0_km_relationship) - 1.0) * decay
    return float(activation * ri / (parameters.tauc * (1.0 - decay)))


def base_rhs(
    parameters: Ding2007Parameters,
    state: np.ndarray,
    time: float,
    pulse_width: float,
    stim_times: np.ndarray,
    activation_values: np.ndarray,
    truncation: int,
) -> np.ndarray:
    cn, force, a, tau1, km = state
    history_sum = base_cn_sum(
        parameters, stim_times, activation_values, time, truncation
    )
    effective_a = a_calculation(parameters, a, pulse_width)
    force_dot = f_dot(parameters, cn, force, effective_a, tau1, km)
    a_dot, tau1_dot, km_dot = fatigue_rhs(parameters, force, a, tau1, km)
    return np.array(
        [
            cn_dot(parameters, cn, history_sum),
            force_dot,
            a_dot,
            tau1_dot,
            km_dot,
        ],
        dtype=float,
    )


def periodic_rhs(
    parameters: Ding2007Parameters,
    state: np.ndarray,
    pulse_width: float,
    stim_interval: float,
    activation: float = 1.0,
) -> np.ndarray:
    cn, cn_sum, force, a, tau1, km = state
    effective_a = a_calculation(parameters, a, pulse_width)
    force_dot = f_dot(parameters, cn, force, effective_a, tau1, km)
    a_dot, tau1_dot, km_dot = fatigue_rhs(parameters, force, a, tau1, km)
    return np.array(
        [
            cn_dot(parameters, cn, cn_sum),
            -cn_sum / parameters.tauc
            + periodic_cn_sum_gain(parameters, stim_interval, activation),
            force_dot,
            a_dot,
            tau1_dot,
            km_dot,
        ],
        dtype=float,
    )


def pulse_periodic_rhs(
    parameters: Ding2007Parameters,
    state: np.ndarray,
    pulse_width: float,
) -> np.ndarray:
    cn, cn_sum, force, a, tau1, km = state
    effective_a = a_calculation(parameters, a, pulse_width)
    force_dot = f_dot(parameters, cn, force, effective_a, tau1, km)
    a_dot, tau1_dot, km_dot = fatigue_rhs(parameters, force, a, tau1, km)
    return np.array(
        [
            cn_dot(parameters, cn, cn_sum),
            -cn_sum / parameters.tauc,
            force_dot,
            a_dot,
            tau1_dot,
            km_dot,
        ],
        dtype=float,
    )


def stimulation_increment(
    parameters: Ding2007Parameters,
    stim_times: np.ndarray,
    stim_index: int,
) -> float:
    if stim_index == 0:
        return 1.0
    r0 = parameters.km_rest + parameters.r0_km_relationship
    stim_interval = stim_times[stim_index] - stim_times[stim_index - 1]
    return float(1.0 + (r0 - 1.0) * np.exp(-stim_interval / parameters.tauc))


def apply_stimulation_jumps(
    state: np.ndarray,
    stim_index: int,
    stim_times: np.ndarray,
    activation_values: np.ndarray,
    time: float,
    parameters: Ding2007Parameters,
) -> int:
    while stim_index < stim_times.size and stim_times[stim_index] <= time + 1e-12:
        state[1] += (
            stimulation_increment(parameters, stim_times, stim_index)
            * activation_values[stim_index]
        )
        stim_index += 1
    return stim_index


def rk4_step(rhs, state: np.ndarray, time: float, dt: float) -> np.ndarray:
    k1 = rhs(state, time)
    k2 = rhs(state + 0.5 * dt * k1, time + 0.5 * dt)
    k3 = rhs(state + 0.5 * dt * k2, time + 0.5 * dt)
    k4 = rhs(state + dt * k3, time + dt)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def simulate(
    scenario: str,
    args: argparse.Namespace,
) -> dict[str, np.ndarray | dict[str, float]]:
    parameters = Ding2007Parameters()
    cycle_duration = args.cycle_duration
    stim_interval = cycle_duration / args.stimulations_per_cycle
    final_time = args.n_cycles * cycle_duration
    prehistory_time = (
        args.prehistory_cycles * cycle_duration if scenario == "steady_history" else 0.0
    )
    start_time = -prehistory_time
    dt = stim_interval / args.substeps_per_stimulation

    stim_times = np.arange(
        start_time,
        final_time + 0.5 * stim_interval,
        stim_interval,
        dtype=float,
    )
    activation_values = build_activation_values(
        stim_times,
        cycle_duration,
        args,
        parameters,
    )
    periodic_activation = float(np.mean(activation_values))
    base_state = np.array(
        [
            0.0,
            0.0,
            parameters.a_scale,
            parameters.tau1_rest,
            parameters.km_rest,
        ],
        dtype=float,
    )
    periodic_state = np.array(
        [
            0.0,
            0.0,
            0.0,
            parameters.a_scale,
            parameters.tau1_rest,
            parameters.km_rest,
        ],
        dtype=float,
    )
    pulse_periodic_state = periodic_state.copy()
    next_pulse_stim_index = 0

    times = np.arange(start_time, final_time + 0.5 * dt, dt, dtype=float)
    base_states = np.empty((times.size, base_state.size))
    periodic_states = np.empty((times.size, periodic_state.size))
    pulse_periodic_states = np.empty((times.size, pulse_periodic_state.size))
    cn_sum_history = np.empty(times.size)
    pulse_width = np.empty(times.size)
    activation = np.empty(times.size)

    for idx, time in enumerate(times):
        next_pulse_stim_index = apply_stimulation_jumps(
            pulse_periodic_state,
            next_pulse_stim_index,
            stim_times,
            activation_values,
            time,
            parameters,
        )
        base_states[idx, :] = base_state
        periodic_states[idx, :] = periodic_state
        pulse_periodic_states[idx, :] = pulse_periodic_state
        cn_sum_history[idx] = base_cn_sum(
            parameters, stim_times, activation_values, time, args.sum_stim_truncation
        )
        pulse_width[idx] = pulse_width_at_time(
            time,
            cycle_duration,
            args.pulse_width,
            args.pulse_width_profile,
        )
        activation[idx] = activation_at_time(stim_times, activation_values, time)
        if idx == times.size - 1:
            break

        step_end_time = times[idx + 1]
        endpoint_is_stimulation = time_is_stimulation_node(stim_times, step_end_time)

        def base_step_rhs(state, rhs_time):
            input_time = pre_stimulation_endpoint_time(
                rhs_time,
                step_end_time,
                endpoint_is_stimulation,
            )
            return base_rhs(
                parameters,
                state,
                input_time,
                pulse_width_at_time(
                    input_time,
                    cycle_duration,
                    args.pulse_width,
                    args.pulse_width_profile,
                ),
                stim_times,
                activation_values,
                args.sum_stim_truncation,
            )

        def periodic_step_rhs(state, rhs_time):
            input_time = pre_stimulation_endpoint_time(
                rhs_time,
                step_end_time,
                endpoint_is_stimulation,
            )
            return periodic_rhs(
                parameters,
                state,
                pulse_width_at_time(
                    input_time,
                    cycle_duration,
                    args.pulse_width,
                    args.pulse_width_profile,
                ),
                stim_interval,
                periodic_activation,
            )

        def pulse_periodic_step_rhs(state, rhs_time):
            input_time = pre_stimulation_endpoint_time(
                rhs_time,
                step_end_time,
                endpoint_is_stimulation,
            )
            return pulse_periodic_rhs(
                parameters,
                state,
                pulse_width_at_time(
                    input_time,
                    cycle_duration,
                    args.pulse_width,
                    args.pulse_width_profile,
                ),
            )

        base_state = rk4_step(base_step_rhs, base_state, time, dt)
        periodic_state = rk4_step(periodic_step_rhs, periodic_state, time, dt)
        pulse_periodic_state = rk4_step(
            pulse_periodic_step_rhs, pulse_periodic_state, time, dt
        )

    validation_mask = times >= -1e-12
    validation_times = times[validation_mask]
    validation_base_states = base_states[validation_mask, :]
    validation_periodic_states = periodic_states[validation_mask, :]
    validation_pulse_periodic_states = pulse_periodic_states[validation_mask, :]
    validation_cn_sum_history = cn_sum_history[validation_mask]
    validation_pulse_width = pulse_width[validation_mask]
    validation_activation = activation[validation_mask]

    node_times = np.arange(0.0, final_time + 0.5 * stim_interval, stim_interval)
    node_indices = np.array(
        [
            int(np.argmin(np.abs(validation_times - node_time)))
            for node_time in node_times
        ]
    )
    node_times = validation_times[node_indices]
    node_base = validation_base_states[node_indices, :]
    node_periodic = validation_periodic_states[node_indices, :]
    node_pulse_periodic = validation_pulse_periodic_states[node_indices, :]
    node_cn_sum_history = validation_cn_sum_history[node_indices]
    node_pulse_width = validation_pulse_width[node_indices]
    node_activation = validation_activation[node_indices]

    comparable_periodic = node_periodic[:, [0, 2, 3, 4, 5]]
    node_error = comparable_periodic - node_base
    node_cn_sum_error = node_periodic[:, 1] - node_cn_sum_history
    node_rhs_error = rhs_equivalence_error(
        parameters,
        node_times,
        node_base,
        node_cn_sum_history,
        node_pulse_width,
        activation_values,
        stim_times,
        args.sum_stim_truncation,
        stim_interval,
    )
    dense_periodic = validation_periodic_states[:, [0, 2, 3, 4, 5]]
    dense_error = dense_periodic - validation_base_states
    dense_cn_sum_error = validation_periodic_states[:, 1] - validation_cn_sum_history
    pulse_comparable_periodic = node_pulse_periodic[:, [0, 2, 3, 4, 5]]
    pulse_node_error = pulse_comparable_periodic - node_base
    pulse_node_cn_sum_error = node_pulse_periodic[:, 1] - node_cn_sum_history
    pulse_dense_periodic = validation_pulse_periodic_states[:, [0, 2, 3, 4, 5]]
    pulse_dense_error = pulse_dense_periodic - validation_base_states
    pulse_dense_cn_sum_error = (
        validation_pulse_periodic_states[:, 1] - validation_cn_sum_history
    )

    metrics = {
        "cn_sum_node_rmse": rmse(node_cn_sum_error),
        "cn_sum_node_max_abs": max_abs(node_cn_sum_error),
        "cn_sum_dense_rmse": rmse(dense_cn_sum_error),
        "cn_sum_dense_max_abs": max_abs(dense_cn_sum_error),
        "pulse_cn_sum_node_rmse": rmse(pulse_node_cn_sum_error),
        "pulse_cn_sum_node_max_abs": max_abs(pulse_node_cn_sum_error),
        "pulse_cn_sum_dense_rmse": rmse(pulse_dense_cn_sum_error),
        "pulse_cn_sum_dense_max_abs": max_abs(pulse_dense_cn_sum_error),
    }
    for state_idx, label in enumerate(COMPARABLE_STATE_LABELS):
        metrics[f"{label}_node_rmse"] = rmse(node_error[:, state_idx])
        metrics[f"{label}_node_max_abs"] = max_abs(node_error[:, state_idx])
        metrics[f"{label}_dense_rmse"] = rmse(dense_error[:, state_idx])
        metrics[f"{label}_dense_max_abs"] = max_abs(dense_error[:, state_idx])
        metrics[f"pulse_{label}_node_rmse"] = rmse(pulse_node_error[:, state_idx])
        metrics[f"pulse_{label}_node_max_abs"] = max_abs(pulse_node_error[:, state_idx])
        metrics[f"pulse_{label}_dense_rmse"] = rmse(pulse_dense_error[:, state_idx])
        metrics[f"pulse_{label}_dense_max_abs"] = max_abs(
            pulse_dense_error[:, state_idx]
        )
        metrics[f"{label}_rhs_node_rmse"] = rmse(node_rhs_error[:, state_idx])
        metrics[f"{label}_rhs_node_max_abs"] = max_abs(node_rhs_error[:, state_idx])

    return {
        "scenario": scenario,
        "times": validation_times,
        "base_states": validation_base_states,
        "periodic_states": validation_periodic_states,
        "pulse_periodic_states": validation_pulse_periodic_states,
        "cn_sum_history": validation_cn_sum_history,
        "pulse_width": validation_pulse_width,
        "activation": validation_activation,
        "node_times": node_times,
        "node_base_states": node_base,
        "node_periodic_states": node_periodic,
        "node_pulse_periodic_states": node_pulse_periodic,
        "node_cn_sum_history": node_cn_sum_history,
        "node_pulse_width": node_pulse_width,
        "node_activation": node_activation,
        "periodic_activation": periodic_activation,
        "node_rhs_error": node_rhs_error,
        "metrics": metrics,
    }


def rhs_equivalence_error(
    parameters: Ding2007Parameters,
    node_times: np.ndarray,
    node_base_states: np.ndarray,
    node_cn_sum_history: np.ndarray,
    node_pulse_width: np.ndarray,
    activation_values: np.ndarray,
    stim_times: np.ndarray,
    truncation: int,
    stim_interval: float,
) -> np.ndarray:
    errors = np.empty((node_times.size, len(COMPARABLE_STATE_LABELS)))
    for idx, (time, state, cn_sum, pulse_width) in enumerate(
        zip(
            node_times,
            node_base_states,
            node_cn_sum_history,
            node_pulse_width,
            strict=True,
        )
    ):
        base_derivative = base_rhs(
            parameters,
            state,
            time,
            pulse_width,
            stim_times,
            activation_values,
            truncation,
        )
        periodic_state = np.array(
            [state[0], cn_sum, state[1], state[2], state[3], state[4]],
            dtype=float,
        )
        periodic_derivative = periodic_rhs(
            parameters,
            periodic_state,
            pulse_width,
            stim_interval,
            activation_at_time(stim_times, activation_values, time),
        )
        errors[idx, :] = periodic_derivative[[0, 2, 3, 4, 5]] - base_derivative
    return errors


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def max_abs(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.max(np.abs(values)))


def plot_cn_sum(results: dict[str, dict], output_dir: Path) -> Path:
    fig, axes = plt.subplots(
        len(results),
        1,
        figsize=(11, 6.5),
        sharex=True,
        constrained_layout=True,
    )
    if len(results) == 1:
        axes = [axes]

    for axis, (scenario, result) in zip(axes, results.items(), strict=True):
        times = result["times"]
        axis.plot(
            times,
            result["cn_sum_history"],
            label="base history Cn_sum(t)",
            color="tab:blue",
            linewidth=1.8,
        )
        axis.plot(
            times,
            result["periodic_states"][:, 1],
            label="periodic mean-intensity Cn_sum(t)",
            color="tab:orange",
            linewidth=1.8,
        )
        axis.plot(
            times,
            result["pulse_periodic_states"][:, 1],
            label="periodic pulse-aware Cn_sum(t)",
            color="tab:red",
            linewidth=1.4,
            linestyle="--",
        )
        axis.scatter(
            result["node_times"],
            result["node_cn_sum_history"],
            label="base at stim nodes",
            color="tab:blue",
            s=12,
            zorder=3,
        )
        axis.scatter(
            result["node_times"],
            result["node_periodic_states"][:, 1],
            label="periodic mean-intensity at stim nodes",
            color="tab:orange",
            s=12,
            zorder=3,
        )
        axis.scatter(
            result["node_times"],
            result["node_pulse_periodic_states"][:, 1],
            label="pulse-aware at stim nodes",
            color="tab:red",
            s=9,
            zorder=3,
        )
        axis.set_ylabel("Cn_sum")
        axis.set_title(scenario.replace("_", " "))
        axis.grid(True, alpha=0.25)

    axes[-1].set_xlabel("time (s)")
    axes[0].legend(loc="upper right", ncol=2)
    path = output_dir / "01_cn_sum_history_vs_periodic.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_states(results: dict[str, dict], output_dir: Path) -> Path:
    result = results["steady_history"]
    times = result["times"]
    base = result["base_states"]
    periodic = result["periodic_states"]
    pulse_periodic = result["pulse_periodic_states"]
    periodic_comparable = {
        "Cn": periodic[:, 0],
        "F": periodic[:, 2],
        "A": periodic[:, 3],
        "Tau1": periodic[:, 4],
        "Km": periodic[:, 5],
    }
    pulse_periodic_comparable = {
        "Cn": pulse_periodic[:, 0],
        "F": pulse_periodic[:, 2],
        "A": pulse_periodic[:, 3],
        "Tau1": pulse_periodic[:, 4],
        "Km": pulse_periodic[:, 5],
    }

    fig, axes = plt.subplots(
        3, 2, figsize=(12, 8), sharex=True, constrained_layout=True
    )
    axes = axes.ravel()
    for idx, label in enumerate(COMPARABLE_STATE_LABELS):
        axis = axes[idx]
        axis.plot(times, base[:, idx], label="base", color="tab:blue", linewidth=1.8)
        axis.plot(
            times,
            periodic_comparable[label],
            label="periodic mean",
            color="tab:orange",
            linewidth=1.8,
            linestyle="--",
        )
        axis.plot(
            times,
            pulse_periodic_comparable[label],
            label="periodic pulse-aware",
            color="tab:red",
            linewidth=1.4,
            linestyle=":",
        )
        axis.scatter(
            result["node_times"],
            result["node_base_states"][:, idx],
            color="tab:blue",
            s=8,
            alpha=0.7,
        )
        axis.scatter(
            result["node_times"],
            result["node_periodic_states"][:, [0, 2, 3, 4, 5][idx]],
            color="tab:orange",
            s=8,
            alpha=0.7,
        )
        axis.scatter(
            result["node_times"],
            result["node_pulse_periodic_states"][:, [0, 2, 3, 4, 5][idx]],
            color="tab:red",
            s=6,
            alpha=0.7,
        )
        axis.set_title(label)
        axis.grid(True, alpha=0.25)
    axes[-1].plot(
        times,
        result["pulse_width"] * 1e6,
        color="tab:green",
        label="pulse width",
    )
    activation_axis = axes[-1].twinx()
    activation_axis.step(
        times,
        result["activation"],
        color="tab:purple",
        where="post",
        alpha=0.75,
        label="lambda",
    )
    activation_axis.axhline(
        result["periodic_activation"],
        color="tab:purple",
        linestyle=":",
        alpha=0.8,
        label="mean lambda",
    )
    axes[-1].set_title("pulse width and effective stimulation intensity")
    axes[-1].set_ylabel("pulse width (us)")
    activation_axis.set_ylabel("lambda")
    axes[-1].grid(True, alpha=0.25)
    handles, labels = axes[-1].get_legend_handles_labels()
    activation_handles, activation_labels = activation_axis.get_legend_handles_labels()
    axes[-1].legend(
        handles + activation_handles,
        labels + activation_labels,
        loc="best",
    )
    for axis in axes[-2:]:
        axis.set_xlabel("time (s)")
    axes[0].legend(loc="best")
    path = output_dir / "02_steady_history_state_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_error_summary(results: dict[str, dict], output_dir: Path) -> Path:
    labels = ("Cn_sum", *COMPARABLE_STATE_LABELS)
    x = np.arange(len(labels))
    bar_specs = [
        ("steady_history", "mean", "", -0.3),
        ("steady_history", "pulse-aware", "pulse_", -0.1),
        ("from_rest", "mean", "", 0.1),
        ("from_rest", "pulse-aware", "pulse_", 0.3),
    ]
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, metric_suffix, title in (
        (axes[0], "node_rmse", "RMSE at stimulation nodes"),
        (axes[1], "dense_rmse", "RMSE on dense integration grid"),
    ):
        for scenario, model_name, prefix, offset in bar_specs:
            result = results[scenario]
            metrics = result["metrics"]
            values = [
                metrics[f"{prefix}cn_sum_{metric_suffix}"],
                *[
                    metrics[f"{prefix}{label}_{metric_suffix}"]
                    for label in COMPARABLE_STATE_LABELS
                ],
            ]
            label = f"{scenario.replace('_', ' ')} / {model_name}"
            axis.bar(x + offset, values, width=width, label=label)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=30, ha="right")
        axis.set_yscale("symlog", linthresh=1e-6)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
    axes[0].legend(loc="best")
    path = output_dir / "03_periodic_history_error_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_rhs_equivalence(results: dict[str, dict], output_dir: Path) -> Path:
    labels = COMPARABLE_STATE_LABELS
    x = np.arange(len(labels))
    width = 0.35

    fig, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for offset, (scenario, result) in zip(
        (-0.5 * width, 0.5 * width),
        results.items(),
        strict=True,
    ):
        metrics = result["metrics"]
        values = [metrics[f"{label}_rhs_node_rmse"] for label in labels]
        axis.bar(x + offset, values, width=width, label=scenario.replace("_", " "))

    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    max_value = max(
        metrics[f"{label}_rhs_node_rmse"]
        for result in results.values()
        for metrics in (result["metrics"],)
        for label in labels
    )
    axis.set_ylim(0.0, max(1e-12, 1.1 * max_value))
    axis.set_title("RHS mismatch at stimulation nodes, same state and same Cn_sum")
    axis.set_ylabel("RMSE")
    if max_value == 0.0:
        axis.text(
            0.5,
            0.55,
            "All comparable Ding RHS terms match exactly at stimulation nodes\n"
            "when the periodic state Cn_sum is set to the base history value.",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(loc="best")
    path = output_dir / "04_rhs_equivalence_at_nodes.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def print_metrics(results: dict[str, dict]) -> None:
    steady_activation = results["steady_history"]["node_activation"]
    print(
        "activation | "
        f"node_min {np.min(steady_activation):.6g} | "
        f"node_mean {np.mean(steady_activation):.6g} | "
        f"node_max {np.max(steady_activation):.6g}"
    )
    print(
        "scenario | periodic_model | quantity | node_rmse | node_max_abs | "
        "dense_rmse | dense_max_abs"
    )
    for scenario, result in results.items():
        metrics = result["metrics"]
        for model_name, prefix in (("mean", ""), ("pulse-aware", "pulse_")):
            rows = ("cn_sum", *COMPARABLE_STATE_LABELS)
            for row in rows:
                print(
                    f"{scenario} | {model_name} | {row} | "
                    f"{metrics[f'{prefix}{row}_node_rmse']:.6g} | "
                    f"{metrics[f'{prefix}{row}_node_max_abs']:.6g} | "
                    f"{metrics[f'{prefix}{row}_dense_rmse']:.6g} | "
                    f"{metrics[f'{prefix}{row}_dense_max_abs']:.6g}"
                )
        print("scenario | rhs_quantity | rhs_node_rmse | rhs_node_max_abs")
        for row in COMPARABLE_STATE_LABELS:
            print(
                f"{scenario} | {row} | "
                f"{metrics[f'{row}_rhs_node_rmse']:.6g} | "
                f"{metrics[f'{row}_rhs_node_max_abs']:.6g}"
            )


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.n_cycles < 1:
        raise ValueError("--n-cycles must be >= 1")
    if args.prehistory_cycles < 0:
        raise ValueError("--prehistory-cycles must be >= 0")
    if args.stimulations_per_cycle < 1:
        raise ValueError("--stimulations-per-cycle must be >= 1")
    if args.substeps_per_stimulation < 1:
        raise ValueError("--substeps-per-stimulation must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "steady_history": simulate("steady_history", args),
        "from_rest": simulate("from_rest", args),
    }

    figure_paths = [
        plot_cn_sum(results, args.output_dir),
        plot_states(results, args.output_dir),
        plot_error_summary(results, args.output_dir),
        plot_rhs_equivalence(results, args.output_dir),
    ]
    print_metrics(results)
    print("figures:")
    for path in figure_paths:
        print(path.resolve())

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
