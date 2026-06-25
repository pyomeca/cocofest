"""
Compare IPOPT and ACADOS on one or several compact solver-compatible cycling MHE problems.

The pulse-width FES MHE remains IPOPT-only in bioptim 3.4 because its time-varying numerical series are not exported
to ACADOS code generation. This script focuses on a shared torque-driven benchmark family that is easy to maintain and
extend.
"""

from __future__ import annotations

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


if __name__ == "__main__":
    main()
