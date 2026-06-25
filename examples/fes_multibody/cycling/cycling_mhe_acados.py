"""
Run one compact solver-compatible cycling moving-horizon problem with ACADOS.

This wrapper stays intentionally small; the reusable setup and comparison logic live in
`cycling_mhe_common.py` so we can benchmark multiple problem definitions consistently.
"""

from __future__ import annotations

import os

from cycling_mhe_common import (
    DEFAULT_MODEL_PATH,
    EXAMPLE_DIR,
    REPO_ROOT,
    build_reference_series,
    build_update_function,
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
    mhe = prepare_mhe(model_path, problem=problem_spec, window_len=window_len, total_angle=total_angle, use_sx=True)
    solver = configure_acados_solver(
        acados_dir=acados_dir or os.environ.get("ACADOS_SOURCE_DIR"),
        codegen_dir=codegen_dir,
        model_name=f"cycling_{problem_spec.name}_acados",
        max_iter=acados_max_iter,
    )
    acados_run = solve_windows_with_timing(mhe, build_update_function(full_target, window_len), solver=solver)
    print_single_solver_result(problem_spec, "ACADOS", acados_run, full_target)
    return acados_run


if __name__ == "__main__":
    main()
