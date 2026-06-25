"""
Run one compact solver-compatible cycling moving-horizon problem with ACADOS.

This wrapper stays intentionally small; the reusable setup and comparison logic live in
`cycling_mhe_common.py` so we can benchmark multiple problem definitions consistently.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from cycling_mhe_common import (
    DEFAULT_MODEL_PATH,
    EXAMPLE_DIR,
    PROBLEM_LIBRARY,
    REPO_ROOT,
    build_reference_series,
    build_reference_velocity,
    build_update_function,
    build_objectives,
    configure_acados_solver,
    prepare_mhe,
    print_single_solver_result,
    resolve_example_path,
    resolve_problem,
    solve_windows_with_timing,
)


def main(
    problem: str = "tracking_control",
    model_path: str = DEFAULT_MODEL_PATH,
    n_windows: int = 2,
    window_len: int = 5,
    total_angle: float = -1.0,
    acados_dir: str | None = None,
    codegen_dir: str = "result/acados/c_generated_code",
    acados_max_iter: int = 100,
):
    os.chdir(EXAMPLE_DIR)
    problem_spec = resolve_problem(problem)
    model_path = resolve_example_path(model_path, EXAMPLE_DIR)
    codegen_dir = resolve_example_path(codegen_dir, REPO_ROOT)

    full_target = build_reference_series(
        window_len=window_len,
        n_windows=n_windows,
        total_angle=total_angle,
        perturbation_scale=problem_spec.perturbation_scale,
    )
    velocity_target = build_reference_velocity(full_target, window_duration=1.0, window_len=window_len)
    objective_indices = build_objectives(
        problem_spec,
        np.linspace(0, total_angle, window_len + 1),
        build_reference_velocity(np.linspace(0, total_angle, window_len + 1), 1.0, window_len),
    )[1]
    mhe = prepare_mhe(model_path, problem=problem_spec, window_len=window_len, total_angle=total_angle, use_sx=True)
    solver = configure_acados_solver(
        acados_dir=acados_dir or os.environ.get("ACADOS_SOURCE_DIR"),
        codegen_dir=codegen_dir,
        model_name=f"cycling_{problem_spec.name}_acados",
        max_iter=acados_max_iter,
    )
    acados_run = solve_windows_with_timing(
        mhe,
        build_update_function(
            full_target,
            window_len,
            wheel_target_objective_index=objective_indices["wheel_q"],
            velocity_target=velocity_target if "wheel_qdot" in objective_indices else None,
            wheel_velocity_objective_index=objective_indices.get("wheel_qdot"),
        ),
        solver=solver,
    )
    print_single_solver_result(problem_spec, "ACADOS", acados_run, full_target)
    return acados_run


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="tracking_control", choices=sorted(PROBLEM_LIBRARY))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--n-windows", type=int, default=2)
    parser.add_argument("--window-len", type=int, default=5)
    parser.add_argument("--total-angle", type=float, default=-1.0)
    parser.add_argument("--acados-dir", default=None)
    parser.add_argument("--codegen-dir", default="result/acados/c_generated_code")
    parser.add_argument("--acados-max-iter", type=int, default=100)
    parser.add_argument(
        "--list-problems",
        action="store_true",
        help="Print the available predefined problems and exit.",
    )
    return parser


if __name__ == "__main__":
    args = build_cli().parse_args()
    if args.list_problems:
        for name in sorted(PROBLEM_LIBRARY):
            print(f"{name}: {PROBLEM_LIBRARY[name].description}")
    else:
        main(
            problem=args.problem,
            model_path=args.model_path,
            n_windows=args.n_windows,
            window_len=args.window_len,
            total_angle=args.total_angle,
            acados_dir=args.acados_dir,
            codegen_dir=args.codegen_dir,
            acados_max_iter=args.acados_max_iter,
        )
