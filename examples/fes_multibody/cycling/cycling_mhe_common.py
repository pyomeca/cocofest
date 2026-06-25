from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from bioptim import (
    BoundsList,
    DynamicsOptions,
    InitialGuessList,
    InterpolationType,
    MovingHorizonEstimator,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    PhaseDynamics,
    SolutionMerge,
    Solver,
    TorqueBiorbdModel,
)
from bioptim.misc.enums import SolverType
from bioptim.optimization.receding_horizon_optimization import RecedingHorizonOptimization

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[3]
DEFAULT_MODEL_PATH = "../../msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod"


@dataclass(frozen=True)
class CyclingMheProblem:
    name: str
    description: str
    track_wheel_weight: float = 10.0
    shoulder_posture_weight: float = 0.0
    elbow_posture_weight: float = 0.0
    wheel_velocity_tracking_weight: float = 0.0
    control_weight: float = 1e-4
    qdot_weight: float = 0.0
    terminal_tracking_weight: float = 1e-2
    perturbation_scale: float = 0.03
    shoulder_target: float = 0.75
    elbow_target: float = 1.5


PROBLEM_LIBRARY: dict[str, CyclingMheProblem] = {
    "tracking": CyclingMheProblem(
        name="tracking",
        description="Wheel-angle tracking only.",
        track_wheel_weight=10.0,
        control_weight=0.0,
        qdot_weight=0.0,
        terminal_tracking_weight=1e-2,
    ),
    "tracking_control": CyclingMheProblem(
        name="tracking_control",
        description="Wheel-angle tracking with light torque regularization.",
        track_wheel_weight=10.0,
        control_weight=1e-4,
        qdot_weight=0.0,
        terminal_tracking_weight=1e-2,
    ),
    "posture_tracking_control": CyclingMheProblem(
        name="posture_tracking_control",
        description="Wheel-angle tracking with shoulder/elbow posture regulation and light torque regularization.",
        track_wheel_weight=10.0,
        shoulder_posture_weight=2e-1,
        elbow_posture_weight=2e-1,
        control_weight=1e-4,
        qdot_weight=0.0,
        terminal_tracking_weight=1e-2,
    ),
    "tracking_strong_control": CyclingMheProblem(
        name="tracking_strong_control",
        description="Wheel-angle tracking with stronger torque regularization.",
        track_wheel_weight=10.0,
        control_weight=5e-4,
        qdot_weight=0.0,
        terminal_tracking_weight=1e-2,
    ),
    "tracking_velocity_control": CyclingMheProblem(
        name="tracking_velocity_control",
        description="Wheel-angle tracking with wheel-speed and torque regularization (experimental).",
        track_wheel_weight=10.0,
        shoulder_posture_weight=2e-1,
        elbow_posture_weight=2e-1,
        wheel_velocity_tracking_weight=5e-4,
        control_weight=1e-4,
        qdot_weight=0.0,
        terminal_tracking_weight=1e-2,
    ),
}

DEFAULT_PROBLEM_SUITE = ["tracking", "tracking_control", "posture_tracking_control"]


def resolve_problem(problem: str | CyclingMheProblem | None) -> CyclingMheProblem:
    if isinstance(problem, CyclingMheProblem):
        return problem
    if problem is None:
        return PROBLEM_LIBRARY["tracking_control"]
    if problem not in PROBLEM_LIBRARY:
        available = ", ".join(sorted(PROBLEM_LIBRARY))
        raise ValueError(f"Unknown problem '{problem}'. Available problems: {available}")
    return PROBLEM_LIBRARY[problem]


def resolve_example_path(path: str | os.PathLike, root: Path) -> str:
    path = Path(path)
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())


def validate_window_len(window_len: int) -> None:
    if window_len < 2:
        raise ValueError(
            "window_len must be >= 2 for the ACADOS cycling MHE. "
            "Smaller horizons can generate invalid ACADOS code because the intermediate cost uses N - 1 stages."
        )


def build_reference_series(
    window_len: int,
    n_windows: int,
    total_angle: float,
    perturbation_scale: float = 0.03,
) -> np.ndarray:
    n_nodes = window_len + n_windows + 1
    base_target = np.linspace(0, total_angle, n_nodes)
    if perturbation_scale == 0:
        return base_target

    amplitude = perturbation_scale * max(abs(total_angle), 1.0)
    phase = np.linspace(0, 2 * np.pi, n_nodes)
    perturbation = amplitude * np.sin(phase)
    perturbation[0] = 0
    perturbation[-1] = 0
    return base_target + perturbation


def build_reference_velocity(reference: np.ndarray, window_duration: float, window_len: int) -> np.ndarray:
    dt = window_duration / window_len
    return np.gradient(reference, dt)


def build_update_function(
    full_target: np.ndarray,
    window_len: int,
    wheel_target_objective_index: int = 0,
    velocity_target: np.ndarray | None = None,
    wheel_velocity_objective_index: int | None = None,
):
    def update_functions(mhe: MovingHorizonEstimator, window_idx: int, _sol):
        target = full_target[window_idx : window_idx + window_len + 1]
        mhe.update_objectives_target(target=target[np.newaxis, :], list_index=wheel_target_objective_index)
        if velocity_target is not None and wheel_velocity_objective_index is not None:
            qdot_target = velocity_target[window_idx : window_idx + window_len]
            mhe.update_objectives_target(target=qdot_target[np.newaxis, :], list_index=wheel_velocity_objective_index)
        return window_idx < len(full_target) - window_len - 1

    return update_functions


def solve_windows(mhe: MovingHorizonEstimator, update_function, solver):
    sol = None
    window_solutions = []
    mhe.total_optimization_run = 0

    while update_function(mhe, mhe.total_optimization_run, sol):
        sol = super(RecedingHorizonOptimization, mhe).solve(solver=solver, warm_start=None)
        if mhe.total_optimization_run == 0 and solver.type == SolverType.IPOPT:
            solver.online_optim = None
        window_solutions.append(sol)
        mhe.advance_window(sol)
        mhe.total_optimization_run += 1

    return sol, window_solutions


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


def summarize_solution(label: str, problem_name: str, sol) -> dict:
    q_wheel = sol.decision_states(to_merge=SolutionMerge.NODES)["q"][2, :]
    objective = float(np.nansum(sol.cost)) if getattr(sol, "cost", None) is not None else float("nan")
    return {
        "problem": problem_name,
        "solver": label,
        "status": sol.status,
        "objective": objective,
        "wall_time_s": float(sol.real_time_to_optimize) if sol.real_time_to_optimize is not None else float("nan"),
        "final_wheel_angle": float(q_wheel[-1]),
        "wheel_angle_trace": q_wheel,
    }


def configure_acados_solver(
    acados_dir: str | None = None,
    codegen_dir: str = "result/acados/c_generated_code",
    model_name: str = "cycling_mhe_acados",
    max_iter: int = 100,
):
    solver = Solver.ACADOS()
    if acados_dir:
        solver.set_acados_dir(acados_dir)
    solver.set_c_generated_code_path(codegen_dir)
    solver.set_acados_model_name(model_name)
    solver.set_maximum_iterations(max_iter)
    solver.set_print_level(0)
    return solver


def configure_ipopt_solver(
    max_iter: int = 500,
    linear_solver: str | None = None,
    tolerance: float = 1e-4,
):
    solver = Solver.IPOPT(show_online_optim=False, _max_iter=max_iter, show_options=dict(show_bounds=True))
    solver.set_convergence_tolerance(tolerance)
    solver.set_constraint_tolerance(tolerance)
    solver.set_dual_inf_tol(tolerance)
    solver.set_hessian_approximation("limited-memory")
    if linear_solver:
        solver.set_linear_solver(linear_solver)
    return solver


def build_objectives(
    problem: CyclingMheProblem, q_target: np.ndarray, qdot_target: np.ndarray | None = None
) -> tuple[ObjectiveList, dict[str, int]]:
    objectives = ObjectiveList()
    target_objective_indices = {}
    objective_count = 0
    if problem.track_wheel_weight:
        target_objective_indices["wheel_q"] = objective_count
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            key="q",
            index=2,
            node=Node.ALL,
            target=q_target[np.newaxis, :],
            weight=problem.track_wheel_weight,
            quadratic=True,
            multi_thread=False,
        )
        objective_count += 1
    if problem.shoulder_posture_weight:
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            key="q",
            index=0,
            node=Node.ALL_SHOOTING,
            target=np.full((1, q_target.shape[0] - 1), problem.shoulder_target),
            weight=problem.shoulder_posture_weight,
            quadratic=True,
            multi_thread=False,
        )
        objective_count += 1
    if problem.elbow_posture_weight:
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            key="q",
            index=1,
            node=Node.ALL_SHOOTING,
            target=np.full((1, q_target.shape[0] - 1), problem.elbow_target),
            weight=problem.elbow_posture_weight,
            quadratic=True,
            multi_thread=False,
        )
        objective_count += 1
    if problem.wheel_velocity_tracking_weight:
        if qdot_target is None:
            raise ValueError("qdot_target is required when wheel_velocity_tracking_weight is non-zero.")
        target_objective_indices["wheel_qdot"] = objective_count
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            key="qdot",
            index=2,
            node=Node.ALL_SHOOTING,
            target=qdot_target[:-1][np.newaxis, :],
            weight=problem.wheel_velocity_tracking_weight,
            quadratic=True,
            multi_thread=False,
        )
        objective_count += 1
    if problem.qdot_weight:
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            key="qdot",
            index=2,
            weight=problem.qdot_weight,
            quadratic=True,
            multi_thread=False,
        )
        objective_count += 1
    if problem.control_weight:
        objectives.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="tau",
            weight=problem.control_weight,
            quadratic=True,
            multi_thread=False,
        )
        objective_count += 1
    if problem.terminal_tracking_weight:
        objectives.add(
            ObjectiveFcn.Mayer.MINIMIZE_STATE,
            key="q",
            index=2,
            node=Node.END,
            target=np.array([[q_target[-1]]]),
            weight=problem.terminal_tracking_weight,
            quadratic=True,
            multi_thread=False,
        )
    return objectives, target_objective_indices


def prepare_mhe(
    model_path: str,
    problem: str | CyclingMheProblem | None = None,
    window_len: int = 5,
    window_duration: float = 1.0,
    total_angle: float = -1.0,
    use_sx: bool = True,
    n_threads: int = 1,
):
    validate_window_len(window_len)
    problem = resolve_problem(problem)
    model = TorqueBiorbdModel(model_path)
    n_nodes = window_len + 1

    x_bounds = BoundsList()
    q_bounds = model.bounds_from_ranges("q")
    q_bounds.min[:] = [-50]
    q_bounds.max[:] = [50]
    x_bounds.add("q", bounds=q_bounds, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

    qdot_bounds = model.bounds_from_ranges("qdot")
    qdot_bounds.min[:] = [-50]
    qdot_bounds.max[:] = [50]
    x_bounds.add("qdot", bounds=qdot_bounds, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

    u_bounds = BoundsList()
    u_bounds.add(
        "tau",
        min_bound=np.full((model.nb_tau, 3), -1000),
        max_bound=np.full((model.nb_tau, 3), 1000),
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    q_target = np.linspace(0, total_angle, n_nodes)
    qdot_target = build_reference_velocity(q_target, window_duration=window_duration, window_len=window_len)
    x_init = InitialGuessList()
    x_init.add(
        "q",
        np.vstack([np.full(n_nodes, 0.75), np.full(n_nodes, 1.5), q_target]),
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_init.add("qdot", np.zeros((model.nb_qdot, n_nodes)), interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialGuessList()
    u_init.add("tau", [0] * model.nb_tau)

    return MovingHorizonEstimator(
        model,
        window_len,
        window_duration,
        dynamics=DynamicsOptions(
            ode_solver=OdeSolver.RK4(n_integration_steps=1),
            expand_dynamics=True,
            phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        ),
        common_objective_functions=build_objectives(problem, q_target, qdot_target)[0],
        x_bounds=x_bounds,
        x_init=x_init,
        u_bounds=u_bounds,
        u_init=u_init,
        use_sx=use_sx,
        n_threads=n_threads,
    )


def compare_problem(
    problem: str | CyclingMheProblem,
    model_path: str,
    n_windows: int = 2,
    window_len: int = 5,
    total_angle: float = -1.0,
    acados_dir: str | None = None,
    acados_codegen_dir: str = "result/acados/c_generated_code_compare",
    ipopt_max_iter: int = 500,
    acados_max_iter: int = 100,
):
    problem = resolve_problem(problem)
    full_target = build_reference_series(
        window_len=window_len,
        n_windows=n_windows,
        total_angle=total_angle,
        perturbation_scale=problem.perturbation_scale,
    )
    velocity_target = build_reference_velocity(full_target, window_duration=1.0, window_len=window_len)
    objective_indices = build_objectives(
        problem, np.linspace(0, total_angle, window_len + 1), build_reference_velocity(np.linspace(0, total_angle, window_len + 1), 1.0, window_len)
    )[1]

    ipopt_mhe = prepare_mhe(model_path, problem=problem, window_len=window_len, total_angle=total_angle, use_sx=False)
    ipopt_solver = configure_ipopt_solver(max_iter=ipopt_max_iter, linear_solver="ma57", tolerance=1e-4)
    ipopt_run = solve_windows_with_timing(
        ipopt_mhe,
        build_update_function(
            full_target,
            window_len,
            wheel_target_objective_index=objective_indices["wheel_q"],
            velocity_target=velocity_target if "wheel_qdot" in objective_indices else None,
            wheel_velocity_objective_index=objective_indices.get("wheel_qdot"),
        ),
        solver=ipopt_solver,
    )

    acados_mhe = prepare_mhe(model_path, problem=problem, window_len=window_len, total_angle=total_angle, use_sx=True)
    acados_solver = configure_acados_solver(
        acados_dir=acados_dir or os.environ.get("ACADOS_SOURCE_DIR"),
        codegen_dir=acados_codegen_dir,
        model_name=f"cycling_{problem.name}_compare",
        max_iter=acados_max_iter,
    )
    acados_run = solve_windows_with_timing(
        acados_mhe,
        build_update_function(
            full_target,
            window_len,
            wheel_target_objective_index=objective_indices["wheel_q"],
            velocity_target=velocity_target if "wheel_qdot" in objective_indices else None,
            wheel_velocity_objective_index=objective_indices.get("wheel_qdot"),
        ),
        solver=acados_solver,
    )

    summaries = [
        summarize_solution("IPOPT", problem.name, ipopt_run["final_solution"]),
        summarize_solution("ACADOS", problem.name, acados_run["final_solution"]),
    ]
    summaries[0]["solver_time_s"] = ipopt_run["solver_time_s"]
    summaries[0]["wall_time_s"] = ipopt_run["wall_time_s"]
    summaries[1]["solver_time_s"] = acados_run["solver_time_s"]
    summaries[1]["wall_time_s"] = acados_run["wall_time_s"]

    return {
        "problem": problem,
        "reference": full_target,
        "ipopt": ipopt_run,
        "acados": acados_run,
        "summaries": summaries,
    }


def print_problem_comparison(result: dict) -> None:
    problem: CyclingMheProblem = result["problem"]
    summaries = result["summaries"]
    window_len = len(result["reference"]) - len(result["ipopt"]["window_solutions"]) - 1

    print(f"problem: {problem.name}")
    print(problem.description)
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

    reference_tail = result["reference"][-(window_len + 1) :]
    print(f"reference wheel angle trace: {np.array2string(reference_tail, precision=4, suppress_small=False)}")
    print("window | reference_end | ipopt_end | acados_end")
    for window_idx, (ipopt_window, acados_window) in enumerate(
        zip(result["ipopt"]["window_solutions"], result["acados"]["window_solutions"], strict=False)
    ):
        reference_end = result["reference"][window_idx + window_len]
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


def print_single_solver_result(problem: CyclingMheProblem, solver_name: str, result: dict, reference: np.ndarray) -> None:
    summary = summarize_solution(solver_name, problem.name, result["final_solution"])
    summary["solver_time_s"] = result["solver_time_s"]
    summary["wall_time_s"] = result["wall_time_s"]
    print(f"problem: {problem.name}")
    print(problem.description)
    print("solver | status | objective | solver_time_s | wall_time_s | final_wheel_angle")
    print(
        f"{summary['solver']} | {summary['status']} | {summary['objective']:.6f} | "
        f"{summary['solver_time_s']:.6f} | {summary['wall_time_s']:.6f} | {summary['final_wheel_angle']:.6f}"
    )
    print(
        f"{summary['solver']} wheel angle trace: "
        f"{np.array2string(summary['wheel_angle_trace'], precision=4, suppress_small=False)}"
    )
    reference_tail = reference[-(len(summary["wheel_angle_trace"])) :]
    print(f"reference wheel angle trace: {np.array2string(reference_tail, precision=4, suppress_small=False)}")
    print("window | reference_end | estimated_end | delta")
    for window_idx, window_sol in enumerate(result["window_solutions"]):
        reference_end = reference[window_idx + len(summary["wheel_angle_trace"]) - 1]
        estimated_end = window_sol.decision_states(to_merge=SolutionMerge.NODES)["q"][2, -1]
        print(
            f"{window_idx:>6} | {reference_end:>13.6f} | {estimated_end:>13.6f} | {estimated_end - reference_end:>+.6f}"
        )


def run_problem_suite(
    problems: list[str] | None = None,
    model_path: str = DEFAULT_MODEL_PATH,
    n_windows: int = 2,
    window_len: int = 5,
    total_angle: float = -1.0,
    acados_dir: str | None = None,
    acados_codegen_root: str = "result/acados/problem_suite",
    ipopt_max_iter: int = 500,
    acados_max_iter: int = 100,
) -> list[dict]:
    selected = problems or list(DEFAULT_PROBLEM_SUITE)
    results = []
    for problem_name in selected:
        result = compare_problem(
            problem=problem_name,
            model_path=model_path,
            n_windows=n_windows,
            window_len=window_len,
            total_angle=total_angle,
            acados_dir=acados_dir,
            acados_codegen_dir=resolve_example_path(f"{acados_codegen_root}/{problem_name}", REPO_ROOT),
            ipopt_max_iter=ipopt_max_iter,
            acados_max_iter=acados_max_iter,
        )
        print_problem_comparison(result)
        print("")
        results.append(result)
    return results
