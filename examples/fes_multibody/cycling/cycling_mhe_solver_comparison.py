"""
Compare a compact cycling moving-horizon estimator solved with IPOPT and ACADOS.

This script uses the same torque-driven cycling MHE setup for both solvers so their solve time, status, objective, and
estimated wheel angle can be compared directly. It intentionally avoids the pulse-width FES MHE because bioptim 3.4
cannot export its numerical time series to ACADOS code generation.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from bioptim import MovingHorizonEstimator, SolutionMerge
from bioptim.misc.enums import SolverType
from bioptim.optimization.receding_horizon_optimization import RecedingHorizonOptimization

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.fes_multibody.cycling.cycling_mhe_acados import (
    EXAMPLE_DIR,
    REPO_ROOT,
    _resolve_example_path,
    _validate_window_len,
    configure_acados_solver,
    configure_ipopt_solver,
    prepare_mhe,
)


def build_update_function(full_target: np.ndarray, window_len: int):
    def update_functions(mhe: MovingHorizonEstimator, window_idx: int, _sol):
        target = full_target[window_idx : window_idx + window_len + 1]
        mhe.update_objectives_target(target=target[np.newaxis, :], list_index=0)
        return window_idx < len(full_target) - window_len - 1

    return update_functions


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


def solve_windows(mhe: MovingHorizonEstimator, update_function, solver):
    sol = None
    total_time = 0.0
    wall_clock_start = perf_counter()
    compiled = False
    window_solutions = []

    mhe.total_optimization_run = 0
    while update_function(mhe, mhe.total_optimization_run, sol):
        sol = super(RecedingHorizonOptimization, mhe).solve(solver=solver, warm_start=None)
        if not compiled and solver.type == SolverType.ACADOS:
            wall_clock_start = perf_counter()
            compiled = True
        if mhe.total_optimization_run == 0 and solver.type == SolverType.IPOPT:
            solver.online_optim = None

        total_time += sol.real_time_to_optimize or 0.0
        window_solutions.append(sol)
        mhe.advance_window(sol)
        mhe.total_optimization_run += 1

    return {
        "final_solution": sol,
        "window_solutions": window_solutions,
        "solver_time_s": total_time,
        "wall_time_s": perf_counter() - wall_clock_start,
    }


def run_comparison(
    model_path: str,
    n_windows: int = 2,
    window_len: int = 5,
    total_angle: float = -1.0,
    acados_dir: str | None = None,
    acados_codegen_dir: str = "result/acados/c_generated_code_compare",
    ipopt_max_iter: int = 500,
    acados_max_iter: int = 100,
):
    _validate_window_len(window_len)
    full_target = np.linspace(0, total_angle, window_len + n_windows + 1)

    ipopt_mhe = prepare_mhe(model_path, window_len=window_len, total_angle=total_angle, use_sx=False, n_threads=1)
    ipopt_solver = configure_ipopt_solver(max_iter=ipopt_max_iter, linear_solver="ma57", tolerance=1e-4)
    ipopt_run = solve_windows(ipopt_mhe, build_update_function(full_target, window_len), solver=ipopt_solver)

    acados_mhe = prepare_mhe(model_path, window_len=window_len, total_angle=total_angle, use_sx=True, n_threads=1)
    acados_solver = configure_acados_solver(
        acados_dir=acados_dir or os.environ.get("ACADOS_SOURCE_DIR"),
        codegen_dir=acados_codegen_dir,
        model_name="cycling_mhe_compare",
        max_iter=acados_max_iter,
    )
    acados_run = solve_windows(acados_mhe, build_update_function(full_target, window_len), solver=acados_solver)

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
        acados_dir=acados_dir or os.environ.get("ACADOS_SOURCE_DIR"),
        acados_codegen_dir=_resolve_example_path(acados_codegen_dir, REPO_ROOT),
        ipopt_max_iter=ipopt_max_iter,
        acados_max_iter=acados_max_iter,
    )


if __name__ == "__main__":
    main()
