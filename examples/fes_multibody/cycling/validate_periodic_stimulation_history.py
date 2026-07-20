"""
Validate the periodic Ding stimulation-history surrogate against the base Ding model.

The base Ding 2007 fatigue model evaluates the calcium driving term from the
truncated list of previous stimulation times. The periodic variant replaces this
history by the additional state Cn_sum. This script compares both formulations
with the same stimulation times and pulse-width command history.
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


def base_cn_sum(
    parameters: Ding2007Parameters,
    stim_times: np.ndarray,
    time: float,
    truncation: int,
) -> float:
    history = finite_stimulation_history(stim_times, time, truncation)
    r0 = parameters.km_rest + parameters.r0_km_relationship
    value = 0.0
    for idx, stim_time in enumerate(history):
        if idx == 0:
            ri = 1.0
        else:
            ri = 1.0 + (r0 - 1.0) * np.exp(
                -(stim_time - history[idx - 1]) / parameters.tauc
            )
        value += ri * np.exp(-(time - stim_time) / parameters.tauc)
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
) -> float:
    decay = np.exp(-stim_interval / parameters.tauc)
    ri = 1.0 + ((parameters.km_rest + parameters.r0_km_relationship) - 1.0) * decay
    return float(ri / (parameters.tauc * (1.0 - decay)))


def base_rhs(
    parameters: Ding2007Parameters,
    state: np.ndarray,
    time: float,
    pulse_width: float,
    stim_times: np.ndarray,
    truncation: int,
) -> np.ndarray:
    cn, force, a, tau1, km = state
    history_sum = base_cn_sum(parameters, stim_times, time, truncation)
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
) -> np.ndarray:
    cn, cn_sum, force, a, tau1, km = state
    effective_a = a_calculation(parameters, a, pulse_width)
    force_dot = f_dot(parameters, cn, force, effective_a, tau1, km)
    a_dot, tau1_dot, km_dot = fatigue_rhs(parameters, force, a, tau1, km)
    return np.array(
        [
            cn_dot(parameters, cn, cn_sum),
            -cn_sum / parameters.tauc + periodic_cn_sum_gain(parameters, stim_interval),
            force_dot,
            a_dot,
            tau1_dot,
            km_dot,
        ],
        dtype=float,
    )


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

    times = np.arange(start_time, final_time + 0.5 * dt, dt, dtype=float)
    base_states = np.empty((times.size, base_state.size))
    periodic_states = np.empty((times.size, periodic_state.size))
    cn_sum_history = np.empty(times.size)
    pulse_width = np.empty(times.size)

    for idx, time in enumerate(times):
        base_states[idx, :] = base_state
        periodic_states[idx, :] = periodic_state
        cn_sum_history[idx] = base_cn_sum(
            parameters, stim_times, time, args.sum_stim_truncation
        )
        pulse_width[idx] = pulse_width_at_time(
            time,
            cycle_duration,
            args.pulse_width,
            args.pulse_width_profile,
        )
        if idx == times.size - 1:
            break

        def base_step_rhs(state, rhs_time):
            return base_rhs(
                parameters,
                state,
                rhs_time,
                pulse_width_at_time(
                    rhs_time,
                    cycle_duration,
                    args.pulse_width,
                    args.pulse_width_profile,
                ),
                stim_times,
                args.sum_stim_truncation,
            )

        def periodic_step_rhs(state, rhs_time):
            return periodic_rhs(
                parameters,
                state,
                pulse_width_at_time(
                    rhs_time,
                    cycle_duration,
                    args.pulse_width,
                    args.pulse_width_profile,
                ),
                stim_interval,
            )

        base_state = rk4_step(base_step_rhs, base_state, time, dt)
        periodic_state = rk4_step(periodic_step_rhs, periodic_state, time, dt)

    validation_mask = times >= -1e-12
    validation_times = times[validation_mask]
    validation_base_states = base_states[validation_mask, :]
    validation_periodic_states = periodic_states[validation_mask, :]
    validation_cn_sum_history = cn_sum_history[validation_mask]
    validation_pulse_width = pulse_width[validation_mask]

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
    node_cn_sum_history = validation_cn_sum_history[node_indices]
    node_pulse_width = validation_pulse_width[node_indices]

    comparable_periodic = node_periodic[:, [0, 2, 3, 4, 5]]
    node_error = comparable_periodic - node_base
    node_cn_sum_error = node_periodic[:, 1] - node_cn_sum_history
    node_rhs_error = rhs_equivalence_error(
        parameters,
        node_times,
        node_base,
        node_cn_sum_history,
        node_pulse_width,
        stim_times,
        args.sum_stim_truncation,
        stim_interval,
    )
    dense_periodic = validation_periodic_states[:, [0, 2, 3, 4, 5]]
    dense_error = dense_periodic - validation_base_states
    dense_cn_sum_error = validation_periodic_states[:, 1] - validation_cn_sum_history

    metrics = {
        "cn_sum_node_rmse": rmse(node_cn_sum_error),
        "cn_sum_node_max_abs": max_abs(node_cn_sum_error),
        "cn_sum_dense_rmse": rmse(dense_cn_sum_error),
        "cn_sum_dense_max_abs": max_abs(dense_cn_sum_error),
    }
    for state_idx, label in enumerate(COMPARABLE_STATE_LABELS):
        metrics[f"{label}_node_rmse"] = rmse(node_error[:, state_idx])
        metrics[f"{label}_node_max_abs"] = max_abs(node_error[:, state_idx])
        metrics[f"{label}_dense_rmse"] = rmse(dense_error[:, state_idx])
        metrics[f"{label}_dense_max_abs"] = max_abs(dense_error[:, state_idx])
        metrics[f"{label}_rhs_node_rmse"] = rmse(node_rhs_error[:, state_idx])
        metrics[f"{label}_rhs_node_max_abs"] = max_abs(node_rhs_error[:, state_idx])

    return {
        "scenario": scenario,
        "times": validation_times,
        "base_states": validation_base_states,
        "periodic_states": validation_periodic_states,
        "cn_sum_history": validation_cn_sum_history,
        "pulse_width": validation_pulse_width,
        "node_times": node_times,
        "node_base_states": node_base,
        "node_periodic_states": node_periodic,
        "node_cn_sum_history": node_cn_sum_history,
        "node_pulse_width": node_pulse_width,
        "node_rhs_error": node_rhs_error,
        "metrics": metrics,
    }


def rhs_equivalence_error(
    parameters: Ding2007Parameters,
    node_times: np.ndarray,
    node_base_states: np.ndarray,
    node_cn_sum_history: np.ndarray,
    node_pulse_width: np.ndarray,
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
            label="periodic state Cn_sum(t)",
            color="tab:orange",
            linewidth=1.8,
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
            label="periodic at stim nodes",
            color="tab:orange",
            s=12,
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
    periodic_comparable = {
        "Cn": periodic[:, 0],
        "F": periodic[:, 2],
        "A": periodic[:, 3],
        "Tau1": periodic[:, 4],
        "Km": periodic[:, 5],
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
            label="periodic",
            color="tab:orange",
            linewidth=1.8,
            linestyle="--",
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
        axis.set_title(label)
        axis.grid(True, alpha=0.25)
    axes[-1].plot(times, result["pulse_width"] * 1e6, color="tab:green")
    axes[-1].set_title("pulse width command")
    axes[-1].set_ylabel("us")
    axes[-1].grid(True, alpha=0.25)
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
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, metric_suffix, title in (
        (axes[0], "node_rmse", "RMSE at stimulation nodes"),
        (axes[1], "dense_rmse", "RMSE on dense integration grid"),
    ):
        for offset, (scenario, result) in zip(
            (-0.5 * width, 0.5 * width),
            results.items(),
            strict=True,
        ):
            metrics = result["metrics"]
            values = [
                metrics[f"cn_sum_{metric_suffix}"],
                *[
                    metrics[f"{label}_{metric_suffix}"]
                    for label in COMPARABLE_STATE_LABELS
                ],
            ]
            axis.bar(x + offset, values, width=width, label=scenario.replace("_", " "))
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
    print("scenario | quantity | node_rmse | node_max_abs | dense_rmse | dense_max_abs")
    for scenario, result in results.items():
        metrics = result["metrics"]
        rows = ("cn_sum", *COMPARABLE_STATE_LABELS)
        for row in rows:
            print(
                f"{scenario} | {row} | "
                f"{metrics[f'{row}_node_rmse']:.6g} | "
                f"{metrics[f'{row}_node_max_abs']:.6g} | "
                f"{metrics[f'{row}_dense_rmse']:.6g} | "
                f"{metrics[f'{row}_dense_max_abs']:.6g}"
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
