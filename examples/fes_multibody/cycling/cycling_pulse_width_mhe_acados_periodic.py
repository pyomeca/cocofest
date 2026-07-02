"""
CLI-friendly ACADOS example for the periodic Ding pulse-width cycling MHE with a constant crank torque.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
from sys import platform as sys_platform

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bioptim import MultiCyclicCycleSolutions, OdeSolver, SolutionMerge, Solver
from bioptim.optimization.receding_horizon_optimization import (
    RecedingHorizonOptimization,
)

from cycling_pulse_width_mhe import prepare_nmpc, set_fes_model

OBJECTIVE_TO_WEIGHT_INDEX = {"force": 0, "fatigue": 1, "control": 2}


def parse_objectives(raw_objective: str) -> set[str]:
    values = {item.strip().lower() for item in raw_objective.split(",") if item.strip()}
    allowed = set(OBJECTIVE_TO_WEIGHT_INDEX) | {"none"}
    invalid = values - allowed
    if invalid:
        raise ValueError(f"Unsupported objectives: {', '.join(sorted(invalid))}")
    if "none" in values and len(values) > 1:
        raise ValueError("'none' cannot be combined with other objectives.")
    return values or {"force"}


def build_cost_fun_weight(objectives: set[str]) -> list[int]:
    weights = [0, 0, 0]
    for objective in objectives:
        if objective in OBJECTIVE_TO_WEIGHT_INDEX:
            weights[OBJECTIVE_TO_WEIGHT_INDEX[objective]] = 1
    return weights


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-windows",
        type=int,
        default=2,
        help="Number of successive MHE windows to solve.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=("acados", "ipopt"),
        default="acados",
        help="Solver backend used for the MHE windows.",
    )
    parser.add_argument(
        "--single-shot",
        action="store_true",
        help="Solve only one window directly, without the receding-horizon loop.",
    )
    parser.add_argument(
        "--model-formulation",
        type=str,
        choices=("periodic", "standard"),
        default="periodic",
        help="Use the ACADOS-friendly periodic Ding formulation or the historical standard one.",
    )
    parser.add_argument(
        "--torque-application",
        type=str,
        choices=("constant", "external_forces"),
        default="constant",
        help="Apply the resistive crank torque directly on wheel_rotation_RotZ or through external_forces.",
    )
    parser.add_argument(
        "--cycles-per-window",
        type=int,
        default=1,
        help="Number of pedaling cycles simultaneously optimized in each MHE window.",
    )
    parser.add_argument(
        "--ode-solver",
        type=str,
        choices=("rk4", "collocation"),
        default="rk4",
        help="Integration scheme used to transcribe the window dynamics.",
    )
    parser.add_argument(
        "--rk-steps",
        type=int,
        default=5,
        help="Number of RK4 integration steps per shooting interval when --ode-solver=rk4.",
    )
    parser.add_argument(
        "--collocation-degree",
        type=int,
        default=3,
        help="Polynomial degree when --ode-solver=collocation.",
    )
    parser.add_argument(
        "--collocation-method",
        type=str,
        default="radau",
        help="Collocation method name when --ode-solver=collocation.",
    )
    parser.add_argument(
        "--stimulations-per-cycle",
        type=int,
        default=30,
        help="Number of stimulation events per pedaling cycle.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="force",
        help="Objective(s) to minimize: force, fatigue, control, none, or comma-separated combinations.",
    )
    parser.add_argument(
        "--objective-shape",
        type=str,
        choices=("quadratic", "linear"),
        default="quadratic",
        help="Shape of the objective terms passed to bioptim.",
    )
    parser.add_argument(
        "--control-regularization-weight",
        type=float,
        default=0.0,
        help="Optional quadratic MINIMIZE_CONTROL weight on each pulse-width control.",
    )
    parser.add_argument(
        "--control-regularization-target",
        type=float,
        default=None,
        help="Optional pulse-width target, in seconds, for the control regularization.",
    )
    parser.add_argument(
        "--control-regularization-target-source",
        choices=("constant", "warmup"),
        default="constant",
        help="Use a constant pulse-width target or the standard IPOPT warmup controls as the target.",
    )
    parser.add_argument(
        "--wheel-qdot-regularization-weight",
        type=float,
        default=0.0,
        help="Optional quadratic MINIMIZE_STATE weight on the crank/wheel qdot.",
    )
    parser.add_argument(
        "--wheel-qdot-regularization-target",
        type=float,
        default=-float(2 * np.pi),
        help="Target wheel angular velocity, in rad/s, for qdot regularization.",
    )
    parser.add_argument(
        "--constant-crank-torque",
        type=float,
        default=-0.2,
        help="Resistive torque magnitude in N.m, used either as a constant crank torque or as external_forces.",
    )
    parser.add_argument(
        "--max-acados-iterations",
        type=int,
        default=100,
        help="Maximum number of ACADOS SQP iterations per window.",
    )
    parser.add_argument(
        "--max-ipopt-iterations",
        type=int,
        default=2000,
        help="Maximum number of IPOPT iterations per window.",
    )
    parser.add_argument(
        "--ipopt-linear-solver",
        type=str,
        default="ma57",
        help="Linear solver used by IPOPT for direct IPOPT runs and the standard warmup.",
    )
    parser.add_argument(
        "--max-consecutive-failing",
        type=int,
        default=10,
        help="Maximum number of consecutive failing MHE windows tolerated before stopping.",
    )
    parser.add_argument(
        "--codegen-tag",
        type=str,
        default=None,
        help="Optional suffix added to the generated ACADOS code folder and model name.",
    )
    parser.add_argument(
        "--disable-standard-ipopt-warmup",
        action="store_true",
        help="Skip the one-window IPOPT warmup with the standard Ding formulation before periodic MHE.",
    )
    parser.add_argument(
        "--disable-historical-ipopt-initial-guess",
        action="store_true",
        help="Do not reuse the historical IPOPT initial guess file from result/initial_guess when it exists.",
    )
    parser.add_argument(
        "--use-sx",
        dest="use_sx",
        action="store_true",
        help="Build the problem with CasADi SX graphs.",
    )
    parser.add_argument(
        "--no-use-sx",
        dest="use_sx",
        action="store_false",
        help="Build the problem with CasADi MX graphs.",
    )
    parser.add_argument(
        "--enforce-start-constraints",
        dest="enforce_start_constraints",
        action="store_true",
        help="Enable the historical start-of-window posture constraints.",
    )
    parser.add_argument(
        "--disable-start-constraints",
        dest="enforce_start_constraints",
        action="store_false",
        help="Disable the start-of-window posture constraints.",
    )
    parser.set_defaults(use_sx=True, enforce_start_constraints=False)
    return parser


def build_ode_solver(args: argparse.Namespace):
    if args.ode_solver == "collocation":
        return OdeSolver.COLLOCATION(
            polynomial_degree=args.collocation_degree, method=args.collocation_method
        )
    return OdeSolver.RK4(n_integration_steps=args.rk_steps)


def _ode_solver_suffix(ode_solver) -> str:
    if isinstance(ode_solver, OdeSolver.COLLOCATION):
        return f"collocation_{ode_solver.polynomial_degree}_{ode_solver.method}"
    if isinstance(ode_solver, OdeSolver.RK4):
        return f"rk4_{ode_solver.n_integration_steps}"
    raise RuntimeError("ode_solver must be COLLOCATION or RK4")


def _historical_initial_guess_path(cycles_per_window: int, ode_solver) -> Path | None:
    filename = f"{cycles_per_window}_initial_guess_{_ode_solver_suffix(ode_solver)}.pkl"
    candidates = (
        Path.cwd() / "result" / "initial_guess" / filename,
        Path(__file__).resolve().parent / "result" / "initial_guess" / filename,
    )
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def ensure_acados_environment(acados_source_dir: str | None = None) -> Path:
    acados_dir = Path(
        acados_source_dir
        or os.environ.get(
            "ACADOS_SOURCE_DIR", str(Path.home() / "Documents/bioptim/external/acados")
        )
    ).resolve()
    os.environ["ACADOS_SOURCE_DIR"] = str(acados_dir)

    acados_lib_dir = acados_dir / "lib"
    for env_name in (
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "LD_LIBRARY_PATH",
    ):
        current = os.environ.get(env_name, "")
        paths = [p for p in current.split(":") if p]
        if str(acados_lib_dir) not in paths:
            os.environ[env_name] = (
                ":".join([str(acados_lib_dir), *paths])
                if paths
                else str(acados_lib_dir)
            )

    if sys_platform == "darwin":
        _preload_acados_libraries(acados_lib_dir)

    return acados_dir


def _shared_lib_ext() -> str:
    if sys_platform == "darwin":
        return ".dylib"
    if sys_platform.startswith("win"):
        return ".dll"
    return ".so"


def _short_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def _source_stamp(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def _cache_root() -> Path:
    path = Path(__file__).resolve().parent / "result" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _warmup_cache_signature(
    args: argparse.Namespace,
    model_path: Path,
    simulation_conditions: dict,
    cycling_info: dict,
) -> str:
    payload = {
        "kind": "warmup",
        "model_path": str(model_path.resolve()),
        "cycles_per_window": args.cycles_per_window,
        "stimulations_per_cycle": args.stimulations_per_cycle,
        "objective": args.objective,
        "objective_shape": args.objective_shape,
        "constant_crank_torque": args.constant_crank_torque,
        "torque_application": args.torque_application,
        "ipopt_linear_solver": args.ipopt_linear_solver,
        "simulation_conditions": simulation_conditions,
        "cycling_info_keys": sorted(cycling_info.keys()),
        "sources": [
            _source_stamp(Path(__file__).resolve()),
            _source_stamp(
                (
                    Path(__file__).resolve().parent / "cycling_pulse_width_mhe.py"
                ).resolve()
            ),
        ],
    }
    return _short_hash(payload)


def _warmup_cache_path(
    args: argparse.Namespace,
    model_path: Path,
    simulation_conditions: dict,
    cycling_info: dict,
) -> Path:
    return (
        _cache_root()
        / f"warmup_{_warmup_cache_signature(args, model_path, simulation_conditions, cycling_info)}.npz"
    )


def _save_warmup_cache(cache_path: Path, solution) -> None:
    states = solution.decision_states(to_merge=SolutionMerge.NODES)
    controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
    payload = {}
    for key, values in states.items():
        payload[f"states__{key}"] = np.asarray(values)
    for key, values in controls.items():
        payload[f"controls__{key}"] = np.asarray(values)
    np.savez(cache_path, **payload)


def _load_warmup_cache(cache_path: Path) -> "_WarmupSolutionAdapter":
    data = np.load(cache_path, allow_pickle=False)
    states = {
        key.split("__", 1)[1]: data[key]
        for key in data.files
        if key.startswith("states__")
    }
    controls = {
        key.split("__", 1)[1]: data[key]
        for key in data.files
        if key.startswith("controls__")
    }
    return _WarmupSolutionAdapter(states, controls)


def _codegen_signature(args: argparse.Namespace) -> str:
    payload = {
        "solver": args.solver,
        "model_formulation": args.model_formulation,
        "torque_application": args.torque_application,
        "cycles_per_window": args.cycles_per_window,
        "stimulations_per_cycle": args.stimulations_per_cycle,
        "ode_solver": args.ode_solver,
        "rk_steps": args.rk_steps,
        "collocation_degree": args.collocation_degree,
        "collocation_method": args.collocation_method,
        "objective": args.objective,
        "objective_shape": args.objective_shape,
        "constant_crank_torque": args.constant_crank_torque,
        "use_sx": args.use_sx,
        "enforce_start_constraints": args.enforce_start_constraints,
        "control_regularization_weight": args.control_regularization_weight,
        "control_regularization_target": args.control_regularization_target,
        "control_regularization_target_source": args.control_regularization_target_source,
        "wheel_qdot_regularization_weight": args.wheel_qdot_regularization_weight,
        "wheel_qdot_regularization_target": args.wheel_qdot_regularization_target,
        "max_acados_iterations": args.max_acados_iterations,
        "sources": [
            _source_stamp(Path(__file__).resolve()),
            _source_stamp(
                (
                    Path(__file__).resolve().parent / "cycling_pulse_width_mhe.py"
                ).resolve()
            ),
        ],
    }
    return _short_hash(payload)


def _preload_acados_libraries(acados_lib_dir: Path) -> None:
    mode = ctypes.RTLD_GLOBAL
    if hasattr(os, "RTLD_NOW"):
        mode |= os.RTLD_NOW

    library_order = (
        "libqpOASES_e.dylib",
        "libhpipm.dylib",
        "libblasfeo.dylib",
        "libacados.dylib",
    )
    for library_name in library_order:
        library_path = acados_lib_dir / library_name
        if library_path.exists():
            ctypes.CDLL(str(library_path), mode=mode)


def configure_acados_solver(
    model_name: str,
    generated_code_path: str,
    max_iterations: int,
    sim_method_num_steps: int,
    print_level: int = 0,
) -> Solver.ACADOS:
    solver = Solver.ACADOS()
    solver.set_acados_dir(str(ensure_acados_environment()))
    solver.set_c_generated_code_path(generated_code_path)
    solver.set_acados_model_name(model_name)
    shared_lib_path = (
        Path(generated_code_path)
        / f"libacados_ocp_solver_{model_name}{_shared_lib_ext()}"
    )
    solver.set_c_compile(not shared_lib_path.exists())
    solver.set_qp_solver("PARTIAL_CONDENSING_HPIPM")
    solver.set_integrator_type("IRK")
    solver.set_nlp_solver_type("SQP")
    solver.set_hessian_approx("GAUSS_NEWTON")
    solver.set_sim_method_num_stages(4)
    solver.set_sim_method_num_steps(sim_method_num_steps)
    solver.set_sim_method_newton_iter(5)
    solver.set_maximum_iterations(max_iterations)
    solver.set_print_level(print_level)
    # Favor numerical robustness over raw speed for this periodic MHE.
    solver.set_option_unsafe("MERIT_BACKTRACKING", "globalization")
    solver.set_option_unsafe(1, "globalization_line_search_use_sufficient_descent")
    solver.set_option_unsafe(0, "globalization_use_SOC")
    solver.set_option_unsafe("ROBUST", "hpipm_mode")
    solver.set_option_unsafe("GERSHGORIN_LEVENBERG_MARQUARDT", "regularize_method")
    solver.set_option_unsafe("OBJECTIVE_GERSHGORIN", "qpscaling_scale_objective")
    solver.set_option_unsafe("INF_NORM", "qpscaling_scale_constraints")
    solver.set_option_unsafe("ADAPTIVE_QPSCALING", "nlp_qp_tol_strategy")
    solver.set_option_unsafe(0, "qp_solver_warm_start")
    solver.set_option_unsafe(0, "qp_solver_ric_alg")
    solver.set_option_unsafe(0, "qp_solver_cond_ric_alg")
    solver.set_option_unsafe(False, "nlp_solver_warm_start_first_qp")
    solver.set_option_unsafe(False, "nlp_solver_warm_start_first_qp_from_nlp")
    return solver


def configure_ipopt_solver(
    max_iterations: int, linear_solver: str = "ma57"
) -> Solver.IPOPT:
    solver = Solver.IPOPT(
        show_online_optim=False,
        _max_iter=max_iterations,
        show_options=dict(show_bounds=True),
    )
    solver.set_warm_start_init_point("yes")
    solver.set_mu_init(1e-2)
    solver.set_tol(1e-6)
    solver.set_dual_inf_tol(1e-6)
    solver.set_constr_viol_tol(1e-6)
    solver.set_linear_solver(linear_solver)
    return solver


def _split_receding_solution(sol) -> tuple:
    merged_solution = sol[0]
    source_window_solutions = []
    exported_cycle_solutions = []

    if len(sol) == 3:
        source_window_solutions = sol[1]
        exported_cycle_solutions = sol[2]
    elif len(sol) == 2:
        exported_cycle_solutions = sol[1]

    return merged_solution, source_window_solutions, exported_cycle_solutions


def _wheel_trace_from_exported_cycles(
    merged_solution, exported_cycle_solutions: list
) -> np.ndarray:
    if not exported_cycle_solutions:
        return merged_solution.decision_states(to_merge=SolutionMerge.NODES)["q"][2, :]

    cycle_traces = [
        cycle_solution.decision_states(to_merge=SolutionMerge.NODES)["q"][2, :]
        for cycle_solution in exported_cycle_solutions
    ]
    return np.concatenate(
        [trace[:-1] for trace in cycle_traces[:-1]] + [cycle_traces[-1]]
    )


def _control_traces_from_exported_cycles(
    merged_solution, exported_cycle_solutions: list
) -> dict[str, np.ndarray]:
    if not exported_cycle_solutions:
        controls = merged_solution.decision_controls(to_merge=SolutionMerge.NODES)
        return {key: np.asarray(values) for key, values in controls.items()}

    control_traces = {}
    reference_controls = exported_cycle_solutions[0].decision_controls(
        to_merge=SolutionMerge.NODES
    )
    for key in reference_controls.keys():
        cycle_values = []
        for cycle_solution in exported_cycle_solutions:
            controls = cycle_solution.decision_controls(to_merge=SolutionMerge.NODES)
            values = np.asarray(controls[key])
            if values.ndim == 1:
                values = values[np.newaxis, :]
            cycle_values.append(values)
        control_traces[key] = np.concatenate(cycle_values, axis=1)

    return control_traces


def _status_is_success(status) -> bool:
    return status == 0


def _window_accounting(
    source_window_solutions: list,
    exported_cycle_solutions: list,
    cycles_per_window: int,
) -> dict:
    attempted_windows = len(source_window_solutions)
    window_statuses = [
        window_solution.status for window_solution in source_window_solutions
    ]
    successful_windows = sum(_status_is_success(status) for status in window_statuses)
    failed_windows = attempted_windows - successful_windows
    exported_cycles = len(exported_cycle_solutions)
    covered_cycles = successful_windows
    if attempted_windows and successful_windows == attempted_windows:
        covered_cycles += cycles_per_window - 1

    return {
        "attempted_windows": attempted_windows,
        "successful_windows": successful_windows,
        "failed_windows": failed_windows,
        "exported_cycles": exported_cycles,
        "covered_cycles": covered_cycles,
        "window_statuses": window_statuses,
    }


def summarize_windows(sol, requested_windows: int, cycles_per_window: int) -> None:
    def _fmt(value) -> str:
        return "None" if value is None else f"{value:.6f}"

    merged_solution, source_window_solutions, exported_cycle_solutions = (
        _split_receding_solution(sol)
    )
    accounting = _window_accounting(
        source_window_solutions, exported_cycle_solutions, cycles_per_window
    )
    wheel_trace = _wheel_trace_from_exported_cycles(
        merged_solution, exported_cycle_solutions
    )
    diagnostics = diagnose_wheel_trace(wheel_trace, requested_windows=requested_windows)
    solver_success = (
        accounting["covered_cycles"] >= requested_windows
        and accounting["failed_windows"] == 0
    )
    physical_success = (
        diagnostics["is_physical"]
        and accounting["exported_cycles"] >= requested_windows
    )
    success = solver_success and physical_success

    print(f"merged_status: {merged_solution.status}")
    print(f"merged_cost: {merged_solution.cost}")
    print(f"merged_solver_time_s: {_fmt(merged_solution.solver_time_to_optimize)}")
    print(f"merged_wall_time_s: {_fmt(merged_solution.real_time_to_optimize)}")
    print(f"requested_windows: {requested_windows}")
    print(f"attempted_windows: {accounting['attempted_windows']}")
    print(f"successful_windows: {accounting['successful_windows']}")
    print(f"failed_windows: {accounting['failed_windows']}")
    print(f"exported_cycles: {accounting['exported_cycles']}")
    print(f"covered_cycles: {accounting['covered_cycles']}")
    print(f"solver_success: {solver_success}")
    print(f"physical_success: {physical_success}")
    print(f"success: {success}")
    print(f"final_wheel_angle: {diagnostics['final_angle']:.6f}")
    print(f"max_wheel_step: {diagnostics['max_step']:.6f}")
    if diagnostics["issues"]:
        print(f"diagnostic_issues: {', '.join(diagnostics['issues'])}")

    if source_window_solutions:
        print(f"source_window_count: {len(source_window_solutions)}")
        for idx, window_solution in enumerate(source_window_solutions):
            print(
                f"window[{idx}] status={window_solution.status} "
                f"solver_time_s={_fmt(window_solution.solver_time_to_optimize)} "
                f"wall_time_s={_fmt(window_solution.real_time_to_optimize)}"
            )


def build_window_summary(sol, requested_windows: int, cycles_per_window: int) -> dict:
    merged_solution, source_window_solutions, exported_cycle_solutions = (
        _split_receding_solution(sol)
    )
    accounting = _window_accounting(
        source_window_solutions, exported_cycle_solutions, cycles_per_window
    )
    wheel_trace = _wheel_trace_from_exported_cycles(
        merged_solution, exported_cycle_solutions
    )
    control_traces = _control_traces_from_exported_cycles(
        merged_solution, exported_cycle_solutions
    )
    objective = (
        float(np.nansum(merged_solution.cost))
        if getattr(merged_solution, "cost", None) is not None
        else float("nan")
    )
    diagnostics = diagnose_wheel_trace(wheel_trace, requested_windows=requested_windows)
    solver_success = (
        accounting["covered_cycles"] >= requested_windows
        and accounting["failed_windows"] == 0
    )
    physical_success = (
        diagnostics["is_physical"]
        and accounting["exported_cycles"] >= requested_windows
    )
    success = solver_success and physical_success
    return {
        "mode": "rho",
        "status": merged_solution.status,
        "objective": objective,
        "solver_time_s": merged_solution.solver_time_to_optimize,
        "wall_time_s": merged_solution.real_time_to_optimize,
        "requested_windows": requested_windows,
        "achieved_windows": accounting["attempted_windows"],
        "attempted_windows": accounting["attempted_windows"],
        "successful_windows": accounting["successful_windows"],
        "failed_windows": accounting["failed_windows"],
        "exported_cycles": accounting["exported_cycles"],
        "covered_cycles": accounting["covered_cycles"],
        "window_statuses": accounting["window_statuses"],
        "solver_success": solver_success,
        "physical_success": physical_success,
        "window_count": accounting["attempted_windows"],
        "final_wheel_angle": float(wheel_trace[-1]),
        "wheel_angle_trace": wheel_trace,
        "control_traces": control_traces,
        "solution": merged_solution,
        "window_solutions": source_window_solutions,
        "cycle_solutions": exported_cycle_solutions,
        "diagnostics": diagnostics,
        "success": success,
    }


def summarize_single_shot(sol) -> None:
    def _fmt(value) -> str:
        return "None" if value is None else f"{value:.6f}"

    print(f"single_shot_status: {sol.status}")
    print(f"single_shot_cost: {sol.cost}")
    print(f"single_shot_solver_time_s: {_fmt(sol.solver_time_to_optimize)}")
    print(f"single_shot_wall_time_s: {_fmt(sol.real_time_to_optimize)}")


def build_single_shot_summary(sol) -> dict:
    wheel_trace = sol.decision_states(to_merge=SolutionMerge.NODES)["q"][2, :]
    control_traces = {
        key: np.asarray(values)
        for key, values in sol.decision_controls(to_merge=SolutionMerge.NODES).items()
    }
    objective = (
        float(np.nansum(sol.cost))
        if getattr(sol, "cost", None) is not None
        else float("nan")
    )
    diagnostics = diagnose_wheel_trace(wheel_trace, requested_windows=1)
    return {
        "mode": "single_shot",
        "status": sol.status,
        "objective": objective,
        "solver_time_s": sol.solver_time_to_optimize,
        "wall_time_s": sol.real_time_to_optimize,
        "final_wheel_angle": float(wheel_trace[-1]),
        "wheel_angle_trace": wheel_trace,
        "control_traces": control_traces,
        "solution": sol,
        "window_solutions": [],
        "diagnostics": diagnostics,
        "success": diagnostics["is_physical"],
    }


def diagnose_wheel_trace(wheel_trace: np.ndarray, requested_windows: int) -> dict:
    trace = np.asarray(wheel_trace, dtype=float).squeeze()
    finite = bool(np.all(np.isfinite(trace)))
    final_angle = float(trace[-1]) if trace.size else float("nan")
    max_abs_angle = float(np.max(np.abs(trace))) if trace.size else float("nan")
    max_step = float(np.max(np.abs(np.diff(trace)))) if trace.size > 1 else 0.0
    expected_scale = max(2 * np.pi * max(requested_windows, 1), 1.0)
    angle_limit = 10.0 * expected_scale
    jump_limit = 2.0 * np.pi
    issues = []
    if not finite:
        issues.append("non_finite_wheel_trace")
    if finite and max_abs_angle > angle_limit:
        issues.append("wheel_angle_out_of_bounds")
    if finite and max_step > jump_limit:
        issues.append("wheel_angle_jump_out_of_bounds")

    return {
        "is_physical": not issues,
        "issues": issues,
        "final_angle": final_angle,
        "max_abs_angle": max_abs_angle,
        "max_step": max_step,
        "angle_limit": angle_limit,
        "jump_limit": jump_limit,
    }


def _shift_cyclical_trajectory(values: np.ndarray, nodes_per_cycle: int) -> np.ndarray:
    n_plus_one_cycles = values[nodes_per_cycle:-1]
    last_cycle = values[-nodes_per_cycle - 1 :]
    return (
        last_cycle
        if n_plus_one_cycles.size == 0
        else np.concatenate((n_plus_one_cycles, last_cycle))
    )


def estimate_periodic_cn_sum_from_cn(
    cn_values: np.ndarray, tauc: float, dt: float
) -> np.ndarray:
    cn_dot = np.gradient(cn_values, dt)
    return cn_values + tauc * cn_dot


class _WarmupSolutionAdapter:
    def __init__(self, states: dict[str, np.ndarray], controls: dict[str, np.ndarray]):
        self._states = states
        self._controls = controls

    def decision_states(self, to_merge=None):
        return self._states

    def decision_controls(self, to_merge=None):
        return self._controls


def _resample_warmup_data(
    values: np.ndarray, target_len: int, has_terminal_node: bool
) -> np.ndarray:
    current_len = values.shape[1]
    if current_len == target_len:
        return values

    if has_terminal_node:
        if (
            current_len > 1
            and target_len > 1
            and (current_len - 1) % (target_len - 1) == 0
        ):
            stride = (current_len - 1) // (target_len - 1)
            return values[:, ::stride]
    else:
        if current_len % target_len == 0:
            stride = current_len // target_len
            return values[:, ::stride][:, :target_len]

    raise ValueError(
        f"Cannot resample warmup data of length {current_len} to target length {target_len} "
        f"(has_terminal_node={has_terminal_node})."
    )


def _adapt_warmup_solution_to_periodic_nodes(
    periodic_nmpc, warmup_solution
) -> _WarmupSolutionAdapter:
    warmup_states = warmup_solution.decision_states(to_merge=SolutionMerge.NODES)
    warmup_controls = warmup_solution.decision_controls(to_merge=SolutionMerge.NODES)

    first_state_key = next(iter(periodic_nmpc.nlp[0].x_init.keys()))
    first_control_key = next(iter(periodic_nmpc.nlp[0].u_init.keys()))
    target_state_len = periodic_nmpc.nlp[0].x_init[first_state_key].init.shape[1]
    target_control_len = periodic_nmpc.nlp[0].u_init[first_control_key].init.shape[1]

    adapted_states = {
        key: _resample_warmup_data(values, target_state_len, has_terminal_node=True)
        for key, values in warmup_states.items()
    }
    adapted_controls = {
        key: _resample_warmup_data(values, target_control_len, has_terminal_node=False)
        for key, values in warmup_controls.items()
    }
    return _WarmupSolutionAdapter(adapted_states, adapted_controls)


def apply_standard_warmup_to_periodic_nmpc(periodic_nmpc, warmup_solution):
    adapted_solution = _adapt_warmup_solution_to_periodic_nodes(
        periodic_nmpc, warmup_solution
    )
    periodic_nmpc.advance_window_bounds_states(adapted_solution)
    periodic_nmpc.advance_window_initial_guess_states(adapted_solution)
    periodic_nmpc.advance_window_initial_guess_controls(adapted_solution)

    warmup_states = adapted_solution.decision_states(to_merge=SolutionMerge.NODES)
    dt = periodic_nmpc.cycle_duration / periodic_nmpc.cycle_len
    muscle_models = {
        model.muscle_name: model
        for model in periodic_nmpc.nlp[0].model.muscles_dynamics_model
    }
    for key in periodic_nmpc.nlp[0].states.keys():
        if not key.startswith("Cn_sum_"):
            continue

        source_key = key.replace("Cn_sum_", "Cn_")
        if source_key not in warmup_states:
            continue

        muscle_name = key.replace("Cn_sum_", "")
        tauc = muscle_models[muscle_name].tauc
        source_values = warmup_states[source_key][0]
        cn_sum_values = estimate_periodic_cn_sum_from_cn(
            source_values, tauc=tauc, dt=dt
        )
        shifted_values = _shift_cyclical_trajectory(
            cn_sum_values, periodic_nmpc.nodes_per_cycle
        )
        periodic_nmpc.nlp[0].x_init[key].init[0, :] = shifted_values
        if getattr(periodic_nmpc, "bound_first_node_all_states", True):
            center = cn_sum_values[periodic_nmpc.nodes_per_cycle]
            slack = (
                periodic_nmpc._state_slack_for(key, 0)
                if hasattr(periodic_nmpc, "_state_slack_for")
                else 0.0
            )
            periodic_nmpc.nlp[0].x_bounds[key].min[0, 0] = center - slack
            periodic_nmpc.nlp[0].x_bounds[key].max[0, 0] = center + slack

    return adapted_solution


def apply_warmup_control_regularization_targets(
    periodic_nmpc, adapted_warmup_solution
) -> list[str]:
    warmup_controls = adapted_warmup_solution.decision_controls(
        to_merge=SolutionMerge.NODES
    )
    updated_keys = []
    for penalty in periodic_nmpc.nlp[0].J:
        if not penalty:
            continue

        key = getattr(penalty, "extra_parameters", {}).get("key")
        if key not in warmup_controls:
            continue

        target = np.asarray(warmup_controls[key], dtype=float)
        if target.ndim == 1:
            target = target[np.newaxis, :]

        target_len = len(penalty.node_idx)
        if target.shape[1] == target_len - 1:
            target = np.concatenate((target, target[:, -1:]), axis=1)
        elif target.shape[1] != target_len:
            target = _resample_warmup_data(target, target_len, has_terminal_node=False)

        penalty.target = target
        updated_keys.append(key)

    return updated_keys


def run_standard_ipopt_warmup(
    args: argparse.Namespace,
    mhe_info: dict,
    cycling_info: dict,
    simulation_conditions: dict,
    model_path: Path,
):
    cache_path = _warmup_cache_path(
        args, model_path, simulation_conditions, cycling_info
    )
    if cache_path.exists():
        print(f"warmup_cache: hit ({cache_path.name})")
        return _load_warmup_cache(cache_path)

    warmup_mhe_info = dict(mhe_info)
    warmup_mhe_info["ode_solver"] = OdeSolver.COLLOCATION(
        polynomial_degree=3, method="radau"
    )
    warmup_mhe_info["use_sx"] = False

    warmup_cycling_info = dict(cycling_info)
    warmup_cycling_info["enforce_start_constraints"] = True

    stim_time = list(
        np.linspace(
            0,
            warmup_mhe_info["cycle_duration"] * args.cycles_per_window,
            args.stimulations_per_cycle * args.cycles_per_window,
            endpoint=False,
        )
    )
    warmup_model = set_fes_model(
        str(model_path), stim_time, periodic_cn_sum_approximation=False
    )
    warmup_nmpc = prepare_nmpc(
        warmup_model, warmup_mhe_info, warmup_cycling_info, dict(simulation_conditions)
    )
    warmup_nmpc.n_cycles_simultaneous = args.cycles_per_window

    warmup_solver = configure_ipopt_solver(
        max_iterations=args.max_ipopt_iterations,
        linear_solver=args.ipopt_linear_solver,
    )
    warmup_sol = super(RecedingHorizonOptimization, warmup_nmpc).solve(
        solver=warmup_solver,
        warm_start=None,
    )
    _save_warmup_cache(cache_path, warmup_sol)
    print(f"warmup_cache: saved ({cache_path.name})")
    return warmup_sol


def build_codegen_names(args: argparse.Namespace) -> tuple[str, str]:
    objective_slug = args.objective.replace(",", "_")
    signature = _codegen_signature(args)
    suffix = args.codegen_tag or (
        f"{args.solver}_{args.model_formulation}_{objective_slug}_{args.objective_shape}_{args.n_windows}mhe_{args.cycles_per_window}cyc"
    )
    return (
        f"cycling_fes_periodic_{suffix}_{signature}",
        f"result/acados/c_generated_code_{suffix}_{signature}",
    )


def solve_case(args: argparse.Namespace, echo: bool = True) -> dict:
    objectives = parse_objectives(args.objective)

    if args.n_windows < 1:
        raise ValueError("--n-windows must be >= 1")
    if args.cycles_per_window < 1:
        raise ValueError("--cycles-per-window must be >= 1")
    if args.stimulations_per_cycle < 1:
        raise ValueError("--stimulations-per-cycle must be >= 1")

    example_dir = Path(__file__).resolve().parent
    model_path = (
        example_dir / "../../msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod"
    )
    cycle_duration = 1.0
    total_window_duration = cycle_duration * args.cycles_per_window
    total_stimulations = args.stimulations_per_cycle * args.cycles_per_window
    stim_time = list(
        np.linspace(0, total_window_duration, total_stimulations, endpoint=False)
    )
    periodic_cn_sum_approximation = args.model_formulation == "periodic"
    use_external_forces = args.torque_application == "external_forces"
    ode_solver = build_ode_solver(args)
    historical_init_guess_path = None
    if args.solver == "ipopt" and not args.disable_historical_ipopt_initial_guess:
        historical_init_guess_path = _historical_initial_guess_path(
            args.cycles_per_window, ode_solver
        )
    model = set_fes_model(
        str(model_path),
        stim_time,
        periodic_cn_sum_approximation=periodic_cn_sum_approximation,
    )

    mhe_info = {
        "cycle_duration": cycle_duration,
        "n_cycles_to_advance": 1,
        "n_cycles": args.n_windows,
        "ode_solver": ode_solver,
        "use_sx": args.use_sx,
        "cycle_len": args.stimulations_per_cycle,
        "n_cycles_simultaneous": args.cycles_per_window,
    }
    cycling_info = {
        "turn_number": args.cycles_per_window,
        "pedal_config": {"x_center": 0.35, "y_center": 0.0, "radius": 0.1},
        "enforce_start_constraints": args.enforce_start_constraints,
        "periodic_cn_sum_approximation": periodic_cn_sum_approximation,
    }
    if use_external_forces:
        cycling_info["resistive_torque"] = {
            "Segment_application": "wheel",
            "torque": np.array([0.0, 0.0, args.constant_crank_torque]),
        }
    else:
        cycling_info["constant_crank_torque"] = args.constant_crank_torque
    simulation_conditions = {
        "n_cycles_simultaneous": args.cycles_per_window,
        "stimulation": total_stimulations,
        "minimize_force": "force" in objectives,
        "minimize_fatigue": "fatigue" in objectives,
        "minimize_control": "control" in objectives,
        "cost_fun_weight": build_cost_fun_weight(objectives),
        "objective_shape": args.objective_shape,
        "control_regularization_weight": args.control_regularization_weight,
        "control_regularization_target": args.control_regularization_target,
        "wheel_qdot_regularization_weight": args.wheel_qdot_regularization_weight,
        "wheel_qdot_regularization_target": args.wheel_qdot_regularization_target,
        "init_guess_file_path": (
            str(historical_init_guess_path)
            if historical_init_guess_path is not None
            else None
        ),
    }
    nmpc_simulation_conditions = dict(simulation_conditions)
    if (
        args.solver == "acados"
        and args.control_regularization_target_source == "warmup"
    ):
        nmpc_simulation_conditions["control_regularization_target"] = None

    nmpc = prepare_nmpc(model, mhe_info, cycling_info, nmpc_simulation_conditions)
    nmpc.n_cycles_simultaneous = args.cycles_per_window
    if args.solver == "acados":
        nmpc.first_node_state_slack = {
            "q": [0.0, 0.0, 0.02],
            "qdot": [0.0, 0.0, 0.5],
            "Cn_": 5e-3,
            "Cn_sum_": 5e-3,
            "F_": 1e-2,
            "A_": 5e-3,
            "Tau1_": 5e-3,
            "Km_": 5e-3,
        }
        nmpc.bound_first_node_all_states = False
        nmpc.bound_first_node_wheel_qdot = False
        nmpc.advance_wheel_q_bounds = True
        nmpc.use_signed_wheel_shift = True
        nmpc.transfer_debug = echo

    if echo:
        print(f"model_formulation: {args.model_formulation}")
        print(f"torque_application: {args.torque_application}")
        print(f"resistive_torque_nm: {args.constant_crank_torque}")
        print(f"single_shot: {args.single_shot}")
        print(f"ode_solver: {args.ode_solver}")
        if args.ode_solver == "rk4":
            print(f"rk_steps: {args.rk_steps}")
        else:
            print(f"collocation_degree: {args.collocation_degree}")
            print(f"collocation_method: {args.collocation_method}")
        print(f"use_sx: {args.use_sx}")
        print(f"enforce_start_constraints: {args.enforce_start_constraints}")
        print(f"control_regularization_weight: {args.control_regularization_weight}")
        print(f"control_regularization_target: {args.control_regularization_target}")
        print(
            "control_regularization_target_source: "
            f"{args.control_regularization_target_source}"
        )
        print(
            f"wheel_qdot_regularization_weight: {args.wheel_qdot_regularization_weight}"
        )
        print(
            f"wheel_qdot_regularization_target: {args.wheel_qdot_regularization_target}"
        )
        if args.solver == "ipopt" or (
            periodic_cn_sum_approximation and not args.disable_standard_ipopt_warmup
        ):
            print(f"ipopt_linear_solver: {args.ipopt_linear_solver}")
        if args.solver == "ipopt":
            print(
                "historical_initial_guess: "
                f"{historical_init_guess_path if historical_init_guess_path else 'None'}"
            )

    if periodic_cn_sum_approximation and not args.disable_standard_ipopt_warmup:
        if echo:
            print("running_standard_ipopt_warmup: True")
        warmup_simulation_conditions = dict(simulation_conditions)
        if (
            args.solver == "acados"
            and args.control_regularization_target_source == "warmup"
        ):
            warmup_simulation_conditions["control_regularization_weight"] = 0.0
            warmup_simulation_conditions["control_regularization_target"] = None
        warmup_solution = run_standard_ipopt_warmup(
            args, mhe_info, cycling_info, warmup_simulation_conditions, model_path
        )
        adapted_warmup_solution = apply_standard_warmup_to_periodic_nmpc(
            nmpc, warmup_solution
        )
        if (
            args.solver == "acados"
            and args.control_regularization_target_source == "warmup"
            and args.control_regularization_weight
        ):
            target_keys = apply_warmup_control_regularization_targets(
                nmpc, adapted_warmup_solution
            )
            if echo:
                print(
                    "warmup_control_regularization_targets: "
                    f"{', '.join(target_keys) if target_keys else 'None'}"
                )

    def update_functions(_nmpc, cycle_idx, _sol):
        print(f"window {cycle_idx}")
        if echo and _sol is not None:
            states = _sol.decision_states(to_merge=SolutionMerge.NODES)
            print(
                f"window {cycle_idx} terminal wheel q={states['q'][2, -1]:.6f} "
                f"qdot={states['qdot'][2, -1]:.6f}"
            )
        return cycle_idx + 1 < args.n_windows

    if args.solver == "acados":
        model_name, generated_code_path = build_codegen_names(args)
        solver = configure_acados_solver(
            model_name=model_name,
            generated_code_path=generated_code_path,
            max_iterations=args.max_acados_iterations,
            sim_method_num_steps=max(3, args.rk_steps),
        )
    else:
        solver = configure_ipopt_solver(
            max_iterations=args.max_ipopt_iterations,
            linear_solver=args.ipopt_linear_solver,
        )

    if args.single_shot:
        sol = super(RecedingHorizonOptimization, nmpc).solve(
            solver=solver,
            warm_start=None,
        )
        if echo:
            summarize_single_shot(sol)
        summary = build_single_shot_summary(sol)
        summary["args"] = args
        return summary

    sol = nmpc.solve_fes_nmpc(
        update_functions,
        solver=solver,
        total_cycles=args.n_windows,
        external_force=cycling_info.get("resistive_torque"),
        cycle_solutions=MultiCyclicCycleSolutions.ALL_CYCLES,
        get_all_iterations=True,
        cyclic_options={"states": {}},
        max_consecutive_failing=args.max_consecutive_failing,
    )
    if echo:
        summarize_windows(
            sol,
            requested_windows=args.n_windows,
            cycles_per_window=args.cycles_per_window,
        )
    summary = build_window_summary(
        sol, requested_windows=args.n_windows, cycles_per_window=args.cycles_per_window
    )
    summary["args"] = args
    return summary


def main(cli_args: list[str] | None = None):
    parser = build_argument_parser()
    args = parser.parse_args(cli_args)
    ensure_acados_environment()
    solve_case(args, echo=True)


if __name__ == "__main__":
    main()
