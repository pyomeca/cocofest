"""
Compare a compact cycling moving-horizon estimator solved with IPOPT and ACADOS.

This script uses the same torque-driven cycling MHE setup for both solvers so their solve time, status, objective, and
estimated wheel angle can be compared directly. It intentionally avoids the pulse-width FES MHE because bioptim 3.4
cannot export its numerical time series to ACADOS code generation.
"""

from __future__ import annotations

import os
import sys
from time import perf_counter

import numpy as np
from bioptim import MovingHorizonEstimator, SolutionMerge
from bioptim.misc.enums import SolverType

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.fes_multibody.cycling.cycling_mhe_acados import (
    EXAMPLE_DIR,
    REPO_ROOT,
    _resolve_example_path,
    _validate_window_len,
    build_reference_series,
    build_update_function,
    configure_acados_solver,
    configure_ipopt_solver,
    prepare_mhe,
    solve_windows,
)


def summarize_solution(label: str, sol) -> dict:
    q_wheel = sol.decision_states(to_merge=SolutionMerge.NODES)["q"][2, :]
    objective = float(np.nansum(sol.cost)) if getattr(sol, "cost", None) is not None else float("nan")
    return {
        "solver": label,
        "status": sol.status,
        "objective": objective,
        "wall_time_s": float(sol.real_time_to_optimize) if sol.real_time_to_optimize is not None else float("nan"),
        "final_wheel_angle": float(q_wheel[-1]),
        "wheel_angle_trace": q_wheel,
    }


def solve_windows_with_timing(mhe: MovingHorizonEstimator, update_function, solver):
    wall_clock_start = perf_counter()
    final_solution, window_solutions = solve_windows(mhe, update_function, solver)
    wall_time = perf_counter() - wall_clock_start
    solver_time = float(sum((window_sol.real_time_to_optimize or 0.0) for window_sol in window_solutions))
    if solver.type == SolverType.ACADOS and len(window_solutions) > 1:
        solver_time = float(sum((window_sol.real_time_to_optimize or 0.0) for window_sol in window_solutions[1:]))
    return {
        "final_solution": final_solution,
        "window_solutions": window_solutions,
        "solver_time_s": solver_time,
        "wall_time_s": wall_time,
    }


def run_comparison(
    model_path: str,
    n_windows: int = 2,
    window_len: int = 5,
    total_angle: float = -1.0,
    perturbation_scale: float = 0.03,
    acados_dir: str | None = None,
    acados_codegen_dir: str = "result/acados/c_generated_code_compare",
    ipopt_max_iter: int = 500,
    acados_max_iter: int = 100,
):
    _validate_window_len(window_len)
    full_target = build_reference_series(
        window_len=window_len,
        n_windows=n_windows,
        total_angle=total_angle,
        perturbation_scale=perturbation_scale,
    )

    ipopt_mhe = prepare_mhe(model_path, window_len=window_len, total_angle=total_angle, use_sx=False, n_threads=1)
    ipopt_solver = configure_ipopt_solver(max_iter=ipopt_max_iter, linear_solver="ma57", tolerance=1e-4)
    ipopt_run = solve_windows_with_timing(ipopt_mhe, build_update_function(full_target, window_len), solver=ipopt_solver)

    acados_mhe = prepare_mhe(model_path, window_len=window_len, total_angle=total_angle, use_sx=True, n_threads=1)
    acados_solver = configure_acados_solver(
        acados_dir=acados_dir or os.environ.get("ACADOS_SOURCE_DIR"),
        codegen_dir=acados_codegen_dir,
        model_name="cycling_mhe_compare",
        max_iter=acados_max_iter,
    )
    acados_run = solve_windows_with_timing(
        acados_mhe, build_update_function(full_target, window_len), solver=acados_solver
    )

    summaries = [
        summarize_solution("IPOPT", ipopt_run["final_solution"]),
        summarize_solution("ACADOS", acados_run["final_solution"]),
    ]
    summaries[0]["solver_time_s"] = ipopt_run["solver_time_s"]
    summaries[0]["wall_time_s"] = ipopt_run["wall_time_s"]
    summaries[1]["solver_time_s"] = acados_run["solver_time_s"]
    summaries[1]["wall_time_s"] = acados_run["wall_time_s"]

    print("solver | status | objective | solver_time_s | wall_time_s | final_wheel_angle")
    for summary in summaries:
        print(
            f"{summary['solver']} | {summary['status']} | {summary['objective']:.6f} | "
            f"{summary['solver_time_s']:.6f} | {summary['wall_time_s']:.6f} | {summary['final_wheel_angle']:.6f}"
        )
        print(
            f"{summary['solver']} wheel angle trace: "
            f"{np.array2string(summary['wheel_angle_trace'], precision=4, suppress_small=False)}"
        )
    print(
        "reference wheel angle trace: "
        f"{np.array2string(full_target[-(window_len + 1) :], precision=4, suppress_small=False)}"
    )
    print("window | reference_end | ipopt_end | acados_end")
    for window_idx, (ipopt_window, acados_window) in enumerate(
        zip(ipopt_run["window_solutions"], acados_run["window_solutions"], strict=False)
    ):
        reference_end = full_target[window_idx + window_len]
        ipopt_end = ipopt_window.decision_states(to_merge=SolutionMerge.NODES)["q"][2, -1]
        acados_end = acados_window.decision_states(to_merge=SolutionMerge.NODES)["q"][2, -1]
        print(f"{window_idx:>6} | {reference_end:>13.6f} | {ipopt_end:>9.6f} | {acados_end:>10.6f}")

    ipopt_solver_time = summaries[0]["solver_time_s"]
    acados_solver_time = summaries[1]["solver_time_s"]
    ipopt_wall_time = summaries[0]["wall_time_s"]
    acados_wall_time = summaries[1]["wall_time_s"]

    if np.isfinite(ipopt_solver_time) and np.isfinite(acados_solver_time) and acados_solver_time > 0:
        print(
            f"solver-time ratio IPOPT/ACADOS: {ipopt_solver_time / acados_solver_time:.3f}x "
            f"(ACADOS/IPOPT: {acados_solver_time / ipopt_solver_time:.3f}x)"
        )
    if np.isfinite(ipopt_wall_time) and np.isfinite(acados_wall_time) and acados_wall_time > 0:
        print(
            f"wall-time ratio IPOPT/ACADOS: {ipopt_wall_time / acados_wall_time:.3f}x "
            f"(ACADOS/IPOPT: {acados_wall_time / ipopt_wall_time:.3f}x)"
        )

    return {"ipopt": ipopt_run, "acados": acados_run, "summaries": summaries}


def main(
    model_path: str = "../../msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod",
    n_windows: int = 2,
    window_len: int = 5,
    total_angle: float = -1.0,
    perturbation_scale: float = 0.03,
    acados_dir: str | None = None,
    acados_codegen_dir: str = "result/acados/c_generated_code_compare",
    ipopt_max_iter: int = 500,
    acados_max_iter: int = 100,
):
    os.chdir(EXAMPLE_DIR)
    run_comparison(
        model_path=_resolve_example_path(model_path, EXAMPLE_DIR),
        n_windows=n_windows,
        window_len=window_len,
        total_angle=total_angle,
        perturbation_scale=perturbation_scale,
        acados_dir=acados_dir or os.environ.get("ACADOS_SOURCE_DIR"),
        acados_codegen_dir=_resolve_example_path(acados_codegen_dir, REPO_ROOT),
        ipopt_max_iter=ipopt_max_iter,
        acados_max_iter=acados_max_iter,
    )


if __name__ == "__main__":
    main()
