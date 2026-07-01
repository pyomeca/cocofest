"""
Compare IPOPT and ACADOS on the cycling pulse-width FES MHE using solver-specific but validated configurations.

IPOPT uses the historically robust collocation-based transcription.
ACADOS uses the solver-compatible periodic Ding surrogate with the lightweight RK4 setup.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cycling_pulse_width_mhe_acados_periodic import build_argument_parser, solve_case


EXAMPLE_DIR = Path(__file__).resolve().parent


def _namespace_from_cli(**overrides) -> argparse.Namespace:
    parser = build_argument_parser()
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _format_metric(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _solver_config(
    solver_name: str,
    objective: str,
    objective_shape: str,
    cycles_per_window: int,
    stimulations_per_cycle: int,
    n_windows: int,
    resistive_torque: float,
    codegen_tag: str | None,
    ipopt_max_iter: int,
    acados_max_iter: int,
) -> argparse.Namespace:
    if solver_name == "ipopt":
        return _namespace_from_cli(
            solver="ipopt",
            single_shot=False,
            model_formulation="standard",
            torque_application="external_forces",
            cycles_per_window=cycles_per_window,
            stimulations_per_cycle=stimulations_per_cycle,
            objective=objective,
            objective_shape=objective_shape,
            constant_crank_torque=resistive_torque,
            max_ipopt_iterations=ipopt_max_iter,
            n_windows=n_windows,
            ode_solver="collocation",
            collocation_degree=3,
            collocation_method="radau",
            rk_steps=1,
            use_sx=False,
            enforce_start_constraints=True,
            disable_standard_ipopt_warmup=False,
            max_consecutive_failing=1,
            codegen_tag=codegen_tag,
        )

    if solver_name == "acados":
        return _namespace_from_cli(
            solver="acados",
            single_shot=False,
            model_formulation="periodic",
            torque_application="constant",
            cycles_per_window=cycles_per_window,
            stimulations_per_cycle=stimulations_per_cycle,
            objective=objective,
            objective_shape=objective_shape,
            constant_crank_torque=resistive_torque,
            max_acados_iterations=acados_max_iter,
            n_windows=n_windows,
            ode_solver="rk4",
            rk_steps=1,
            collocation_degree=3,
            collocation_method="radau",
            use_sx=True,
            enforce_start_constraints=False,
            disable_standard_ipopt_warmup=True,
            max_consecutive_failing=10,
            codegen_tag=codegen_tag,
        )

    raise ValueError(f"Unsupported solver_name '{solver_name}'")


def print_comparison(ipopt_result: dict, acados_result: dict) -> None:
    print(
        "solver | success | status | objective | solver_time_s | wall_time_s | final_wheel_angle | "
        "requested_windows | achieved_windows"
    )
    for label, result in (("IPOPT", ipopt_result), ("ACADOS", acados_result)):
        print(
            f"{label} | "
            f"{_format_metric(result.get('success'))} | "
            f"{_format_metric(result['status'])} | "
            f"{_format_metric(result['objective'])} | "
            f"{_format_metric(result['solver_time_s'])} | "
            f"{_format_metric(result['wall_time_s'])} | "
            f"{_format_metric(result['final_wheel_angle'])} | "
            f"{_format_metric(result.get('requested_windows'))} | "
            f"{_format_metric(result.get('achieved_windows'))}"
        )
        print(
            f"{label} wheel angle trace: "
            f"{np.array2string(result['wheel_angle_trace'], precision=4, suppress_small=False)}"
        )
        diagnostics = result.get("diagnostics", {})
        print(
            f"{label} diagnostics: physical={diagnostics.get('is_physical')} "
            f"issues={diagnostics.get('issues')} "
            f"max_abs_angle={_format_metric(diagnostics.get('max_abs_angle'))} "
            f"max_step={_format_metric(diagnostics.get('max_step'))}"
        )

    ipopt_solver_time = ipopt_result["solver_time_s"]
    acados_solver_time = acados_result["solver_time_s"]
    ipopt_wall_time = ipopt_result["wall_time_s"]
    acados_wall_time = acados_result["wall_time_s"]
    if ipopt_solver_time and acados_solver_time:
        print(
            f"solver-time ratio IPOPT/ACADOS: {ipopt_solver_time / acados_solver_time:.3f}x "
            f"(ACADOS/IPOPT: {acados_solver_time / ipopt_solver_time:.3f}x)"
        )
    if ipopt_wall_time and acados_wall_time:
        print(
            f"wall-time ratio IPOPT/ACADOS: {ipopt_wall_time / acados_wall_time:.3f}x "
            f"(ACADOS/IPOPT: {acados_wall_time / ipopt_wall_time:.3f}x)"
        )


def main(
    objective: str = "force",
    objective_shape: str = "quadratic",
    cycles_per_window: int = 2,
    stimulations_per_cycle: int = 30,
    n_windows: int = 2,
    resistive_torque: float = -0.2,
    acados_dir: str | None = None,
    codegen_tag: str | None = None,
    ipopt_max_iter: int = 500,
    acados_max_iter: int = 50,
):
    os.chdir(EXAMPLE_DIR)
    if acados_dir:
        os.environ["ACADOS_SOURCE_DIR"] = str(Path(acados_dir).resolve())

    ipopt_args = _solver_config(
        "ipopt",
        objective=objective,
        objective_shape=objective_shape,
        cycles_per_window=cycles_per_window,
        stimulations_per_cycle=stimulations_per_cycle,
        n_windows=n_windows,
        resistive_torque=resistive_torque,
        codegen_tag=codegen_tag,
        ipopt_max_iter=ipopt_max_iter,
        acados_max_iter=acados_max_iter,
    )
    acados_args = _solver_config(
        "acados",
        objective=objective,
        objective_shape=objective_shape,
        cycles_per_window=cycles_per_window,
        stimulations_per_cycle=stimulations_per_cycle,
        n_windows=n_windows,
        resistive_torque=resistive_torque,
        codegen_tag=codegen_tag,
        ipopt_max_iter=ipopt_max_iter,
        acados_max_iter=acados_max_iter,
    )

    print("Running IPOPT reference configuration...")
    ipopt_result = solve_case(ipopt_args, echo=True)
    print()
    print("Running ACADOS-compatible configuration...")
    acados_result = solve_case(acados_args, echo=True)
    print()
    print_comparison(ipopt_result, acados_result)
    return {"ipopt": ipopt_result, "acados": acados_result}


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", default="force")
    parser.add_argument("--objective-shape", default="quadratic", choices=("quadratic", "linear"))
    parser.add_argument("--cycles-per-window", type=int, default=2)
    parser.add_argument("--stimulations-per-cycle", type=int, default=30)
    parser.add_argument("--n-windows", type=int, default=2)
    parser.add_argument("--resistive-torque", type=float, default=-0.2)
    parser.add_argument("--acados-dir", default=os.environ.get("ACADOS_SOURCE_DIR"))
    parser.add_argument("--codegen-tag", default="fes_compare")
    parser.add_argument("--ipopt-max-iter", type=int, default=500)
    parser.add_argument("--acados-max-iter", type=int, default=50)
    return parser


if __name__ == "__main__":
    args = build_cli().parse_args()
    main(
        objective=args.objective,
        objective_shape=args.objective_shape,
        cycles_per_window=args.cycles_per_window,
        stimulations_per_cycle=args.stimulations_per_cycle,
        n_windows=args.n_windows,
        resistive_torque=args.resistive_torque,
        acados_dir=args.acados_dir,
        codegen_tag=args.codegen_tag,
        ipopt_max_iter=args.ipopt_max_iter,
        acados_max_iter=args.acados_max_iter,
    )
