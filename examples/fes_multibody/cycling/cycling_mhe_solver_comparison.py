"""
Compare IPOPT and ACADOS on one or several compact solver-compatible cycling MHE problems.

The pulse-width FES MHE remains IPOPT-only in bioptim 3.4 because its time-varying numerical series are not exported
to ACADOS code generation. This script focuses on a shared torque-driven benchmark family that is easy to maintain and
extend.
"""

from __future__ import annotations

import argparse
import os

from cycling_mhe_common import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROBLEM_SUITE,
    EXAMPLE_DIR,
    PROBLEM_LIBRARY,
    REPO_ROOT,
    compare_problem,
    print_problem_comparison,
    resolve_example_path,
    run_problem_suite,
)


def main(
    problem: str | None = None,
    model_path: str = DEFAULT_MODEL_PATH,
    n_windows: int = 2,
    window_len: int = 5,
    total_angle: float = -1.0,
    acados_dir: str | None = None,
    acados_codegen_dir: str = "result/acados/c_generated_code_compare",
    acados_codegen_root: str = "result/acados/problem_suite",
    ipopt_max_iter: int = 500,
    acados_max_iter: int = 100,
):
    os.chdir(EXAMPLE_DIR)
    model_path = resolve_example_path(model_path, EXAMPLE_DIR)

    if problem:
        result = compare_problem(
            problem=problem,
            model_path=model_path,
            n_windows=n_windows,
            window_len=window_len,
            total_angle=total_angle,
            acados_dir=acados_dir or os.environ.get("ACADOS_SOURCE_DIR"),
            acados_codegen_dir=resolve_example_path(acados_codegen_dir, REPO_ROOT),
            ipopt_max_iter=ipopt_max_iter,
            acados_max_iter=acados_max_iter,
        )
        print_problem_comparison(result)
        return result

    return run_problem_suite(
        problems=list(DEFAULT_PROBLEM_SUITE),
        model_path=model_path,
        n_windows=n_windows,
        window_len=window_len,
        total_angle=total_angle,
        acados_dir=acados_dir or os.environ.get("ACADOS_SOURCE_DIR"),
        acados_codegen_root=acados_codegen_root,
        ipopt_max_iter=ipopt_max_iter,
        acados_max_iter=acados_max_iter,
    )


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default=None, choices=sorted(PROBLEM_LIBRARY))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--n-windows", type=int, default=2)
    parser.add_argument("--window-len", type=int, default=5)
    parser.add_argument("--total-angle", type=float, default=-1.0)
    parser.add_argument("--acados-dir", default=None)
    parser.add_argument("--acados-codegen-dir", default="result/acados/c_generated_code_compare")
    parser.add_argument("--acados-codegen-root", default="result/acados/problem_suite")
    parser.add_argument("--ipopt-max-iter", type=int, default=500)
    parser.add_argument("--acados-max-iter", type=int, default=100)
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run the default stable comparison suite instead of a single problem.",
    )
    parser.add_argument(
        "--list-problems",
        action="store_true",
        help="Print the available predefined problems and exit.",
    )
    return parser


if __name__ == "__main__":
    args = build_cli().parse_args()
    if args.list_problems:
        print("Default suite:")
        for name in DEFAULT_PROBLEM_SUITE:
            print(f"  {name}: {PROBLEM_LIBRARY[name].description}")
        print("Additional problems:")
        for name in sorted(set(PROBLEM_LIBRARY) - set(DEFAULT_PROBLEM_SUITE)):
            print(f"  {name}: {PROBLEM_LIBRARY[name].description}")
    else:
        main(
            problem=None if args.suite else args.problem,
            model_path=args.model_path,
            n_windows=args.n_windows,
            window_len=args.window_len,
            total_angle=args.total_angle,
            acados_dir=args.acados_dir,
            acados_codegen_dir=args.acados_codegen_dir,
            acados_codegen_root=args.acados_codegen_root,
            ipopt_max_iter=args.ipopt_max_iter,
            acados_max_iter=args.acados_max_iter,
        )
