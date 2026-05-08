from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_outputs"
BO_DIR = Path(__file__).resolve().parent / "examples" / "fes_multibody" / "cycling" / "result" / "bo"


def load_bo_table() -> tuple[list[str], np.ndarray, np.ndarray]:
    npz_path = BO_DIR / "bo_iter_arrays.npz"
    pkl_path = BO_DIR / "bo_iter_log.pkl"

    if npz_path.exists():
        arr = np.load(npz_path, allow_pickle=False)
        muscle_names = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in arr["muscle_names"]]
        return muscle_names, np.asarray(arr["weights"], dtype=float), np.asarray(arr["metric"], dtype=float)

    if pkl_path.exists():
        with open(pkl_path, "rb") as file:
            log = pickle.load(file)
        ordered_idx = sorted(log.keys())
        first_row = log[ordered_idx[0]]
        muscle_names = [
            k
            for k in first_row.keys()
            if k != "metric"
            and not k.endswith("time_per_ocp")
            and k
            not in {
                "solving_time_per_ocp",
                "total_solving_time",
                "iter_per_ocp",
                "average_solving_time_per_iter_list",
                "average_solving_time_per_iter",
            }
        ]
        weights = np.array([[float(log[i][name]) for name in muscle_names] for i in ordered_idx], dtype=float)
        metric = np.array([float(log[i]["metric"]) for i in ordered_idx], dtype=float)
        return muscle_names, weights, metric

    raise FileNotFoundError(f"No BO results found in {BO_DIR}")


def summarize_quasi_optimal(muscle_names: list[str], weights: np.ndarray, metric: np.ndarray, tol: float) -> dict:
    valid = np.isfinite(metric)
    metric = metric[valid]
    weights = weights[valid]
    if metric.size == 0:
        raise RuntimeError("No finite BO metric available.")

    best_metric = float(metric.max())
    cutoff = (1.0 - tol) * best_metric
    quasi = metric >= cutoff
    quasi_weights = weights[quasi]
    quasi_metric = metric[quasi]

    summary = {
        "n_evals": int(metric.size),
        "best_metric": best_metric,
        "quasi_optimal_cutoff": float(cutoff),
        "n_quasi_optimal": int(quasi.sum()),
        "muscles": {},
    }
    for j, muscle_name in enumerate(muscle_names):
        vals = quasi_weights[:, j]
        summary["muscles"][muscle_name] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "median": float(np.median(vals)),
        }
    summary["quasi_optimal_metric_range"] = {
        "min": float(quasi_metric.min()),
        "max": float(quasi_metric.max()),
    }
    return summary


def plot_bo_summary(muscle_names: list[str], weights: np.ndarray, metric: np.ndarray, tol: float, stem: str):
    valid = np.isfinite(metric)
    metric = metric[valid]
    weights = weights[valid]
    best_metric = metric.max()
    cutoff = (1.0 - tol) * best_metric
    quasi = metric >= cutoff

    fig = Figure(figsize=(12, 8))
    FigureCanvas(fig)
    axes = fig.subplots(2, 1)

    ax = axes[0]
    ax.plot(np.arange(metric.size), metric, marker="o", lw=1.5, ms=3)
    ax.axhline(best_metric, color="black", ls="--", lw=1.5, label="Best")
    ax.axhline(cutoff, color="#d00000", ls=":", lw=1.5, label=f"Quasi-optimal cutoff ({int((1-tol)*100)}%)")
    ax.set_xlabel("BO evaluation")
    ax.set_ylabel("Turns before failing")
    ax.set_title("Bayesian optimization history")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    x = np.arange(len(muscle_names))
    means = [weights[quasi, j].mean() for j in range(len(muscle_names))]
    stds = [weights[quasi, j].std() for j in range(len(muscle_names))]
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(muscle_names)
    ax.set_yscale("log")
    ax.set_title("Quasi-optimal weight distribution")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}_bo_quasi_optimal.png", dpi=180)


def main():
    parser = argparse.ArgumentParser(description="Analyze quasi-optimal Bayesian optimization results.")
    parser.add_argument("--tol", type=float, default=0.05, help="Relative tolerance around the best metric.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    try:
        muscle_names, weights, metric = load_bo_table()
    except FileNotFoundError as exc:
        print(exc)
        return
    summary = summarize_quasi_optimal(muscle_names, weights, metric, tol=args.tol)
    plot_bo_summary(muscle_names, weights, metric, tol=args.tol, stem="cycling")

    out_json = OUTPUT_DIR / "cycling_bo_quasi_optimal_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(out_json)
    print(OUTPUT_DIR / "cycling_bo_quasi_optimal.png")


if __name__ == "__main__":
    main()
