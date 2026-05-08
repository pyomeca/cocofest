from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_outputs"


def load_bo_npz(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    arr = np.load(path, allow_pickle=False)
    muscle_names = [str(x) for x in arr["muscle_names"]]
    weights = np.asarray(arr["weights"], dtype=float)
    metric = np.asarray(arr["metric"], dtype=float)
    return muscle_names, weights, metric


def analyze_near_optimal(
    muscle_names: list[str],
    weights: np.ndarray,
    metric: np.ndarray,
    metric_min: float,
    metric_max: float,
    fraction: float,
) -> dict:
    threshold = metric_min + fraction * (metric_max - metric_min)
    selected = metric >= threshold
    near_weights = weights[selected]
    near_metric = metric[selected]
    triceps_idx = muscle_names.index("Triceps")
    ratio = near_weights / near_weights[:, [triceps_idx]]

    ordering_counts = Counter()
    for row in ratio:
        order = tuple(muscle_names[i] for i in np.argsort(-row))
        ordering_counts[order] += 1

    summary = {
        "threshold": float(threshold),
        "n_selected": int(selected.sum()),
        "n_total": int(weights.shape[0]),
        "selected_metric_min": float(near_metric.min()),
        "selected_metric_mean": float(near_metric.mean()),
        "selected_metric_max": float(near_metric.max()),
        "raw_summary": {},
        "ratio_to_triceps_summary": {},
        "top_orderings": [
            {"order": list(order), "count": int(count)} for order, count in ordering_counts.most_common(10)
        ],
    }

    for j, muscle_name in enumerate(muscle_names):
        raw = near_weights[:, j]
        normalized = ratio[:, j]
        summary["raw_summary"][muscle_name] = {
            "mean": float(raw.mean()),
            "median": float(np.median(raw)),
            "std": float(raw.std()),
            "min": float(raw.min()),
            "max": float(raw.max()),
        }
        summary["ratio_to_triceps_summary"][muscle_name] = {
            "mean": float(normalized.mean()),
            "median": float(np.median(normalized)),
            "std": float(normalized.std()),
            "min": float(normalized.min()),
            "max": float(normalized.max()),
        }

    return summary, selected, ratio


def plot_near_optimal(
    muscle_names: list[str],
    weights: np.ndarray,
    metric: np.ndarray,
    selected: np.ndarray,
    ratio: np.ndarray,
    threshold: float,
    stem: str,
):
    near_metric = metric[selected]
    near_weights = weights[selected]
    sum_normalized = near_weights / near_weights.sum(axis=1, keepdims=True)

    fig = Figure(figsize=(12, 8))
    FigureCanvas(fig)
    axes = fig.subplots(2, 1)

    ax = axes[0]
    idx = np.arange(metric.size)
    ax.plot(idx, metric, marker="o", lw=1.2, ms=3, color="#8d99ae")
    ax.scatter(idx[selected], metric[selected], color="#d00000", s=28, label="Near-optimal")
    ax.axhline(threshold, color="black", ls="--", lw=1.5, label=f"95% threshold = {threshold:.2f}")
    ax.set_xlabel("BO evaluation")
    ax.set_ylabel("Turns before failing")
    ax.set_title("Near-optimal Bayesian optimization solutions")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    positions = np.arange(len(muscle_names))
    for row in ratio:
        ax.plot(positions, row, color="#457b9d", alpha=0.18)
    ax.plot(positions, np.median(ratio, axis=0), color="#d00000", lw=2.5, marker="o", label="Median")
    ax.set_xticks(positions)
    ax.set_xticklabels(muscle_names)
    ax.set_yscale("log")
    ax.set_ylabel("Weight / Triceps")
    ax.set_title("Near-optimal weights normalized by Triceps")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}_near_optimal_triceps_normalized.png", dpi=180)

    fig = Figure(figsize=(12, 8))
    FigureCanvas(fig)
    axes = fig.subplots(2, 1)

    ax = axes[0]
    idx = np.arange(metric.size)
    ax.plot(idx, metric, marker="o", lw=1.2, ms=3, color="#8d99ae")
    ax.scatter(idx[selected], metric[selected], color="#d00000", s=28, label="Near-optimal")
    ax.axhline(threshold, color="black", ls="--", lw=1.5, label=f"95% threshold = {threshold:.2f}")
    ax.set_xlabel("BO evaluation")
    ax.set_ylabel("Turns before failing")
    ax.set_title("Near-optimal Bayesian optimization solutions")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    positions = np.arange(len(muscle_names))
    for row in sum_normalized:
        ax.plot(positions, row, color="#2a9d8f", alpha=0.22)
    ax.plot(positions, np.median(sum_normalized, axis=0), color="#d00000", lw=2.5, marker="o", label="Median")
    ax.set_xticks(positions)
    ax.set_xticklabels(muscle_names)
    ax.set_ylabel("Weight / Sum(weights)")
    ax.set_title("Near-optimal weights normalized by their sum")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}_near_optimal_sum_normalized.png", dpi=180)


def main():
    parser = argparse.ArgumentParser(description="Plot near-optimal BO solutions normalized by Triceps.")
    parser.add_argument("npz", type=Path, help="Path to bo_iter_arrays.npz")
    parser.add_argument("--metric-min", type=float, required=True)
    parser.add_argument("--metric-max", type=float, required=True)
    parser.add_argument("--fraction", type=float, default=0.95, help="Relative fraction of the [min, max] range.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    muscle_names, weights, metric = load_bo_npz(args.npz)
    summary, selected, ratio = analyze_near_optimal(
        muscle_names=muscle_names,
        weights=weights,
        metric=metric,
        metric_min=args.metric_min,
        metric_max=args.metric_max,
        fraction=args.fraction,
    )

    stem = args.npz.stem
    plot_near_optimal(muscle_names, weights, metric, selected, ratio, summary["threshold"], stem)

    out_json = OUTPUT_DIR / f"{stem}_near_optimal_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(out_json)
    print(OUTPUT_DIR / f"{stem}_near_optimal_triceps_normalized.png")
    print(OUTPUT_DIR / f"{stem}_near_optimal_sum_normalized.png")


if __name__ == "__main__":
    main()
