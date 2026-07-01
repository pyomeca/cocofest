"""
CLI-friendly ACADOS example for the periodic Ding pulse-width cycling MHE with a constant crank torque.
"""

import argparse
import os
from pathlib import Path
import sys
from sys import platform as sys_platform

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bioptim import MultiCyclicCycleSolutions, OdeSolver, SolutionMerge, Solver
from bioptim.optimization.receding_horizon_optimization import RecedingHorizonOptimization

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
    parser.add_argument("--n-windows", type=int, default=2, help="Number of successive MHE windows to solve.")
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
        default=1,
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
        default=4,
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
        "--constant-crank-torque",
        type=float,
        default=-0.2,
        help="Resistive torque magnitude in N.m, used either as a constant crank torque or as external_forces.",
    )
    parser.add_argument(
        "--max-acados-iterations",
        type=int,
        default=10,
        help="Maximum number of ACADOS SQP iterations per window.",
    )
    parser.add_argument(
        "--max-ipopt-iterations",
        type=int,
        default=2000,
        help="Maximum number of IPOPT iterations per window.",
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
        return OdeSolver.COLLOCATION(polynomial_degree=args.collocation_degree, method=args.collocation_method)
    return OdeSolver.RK4(n_integration_steps=args.rk_steps)


def configure_acados_solver(
    model_name: str, generated_code_path: str, max_iterations: int, print_level: int = 0
) -> Solver.ACADOS:
    solver = Solver.ACADOS()
    solver.set_acados_dir(os.environ.get("ACADOS_SOURCE_DIR", str(Path.home() / "Documents/bioptim/external/acados")))
    solver.set_c_generated_code_path(generated_code_path)
    solver.set_acados_model_name(model_name)
    solver.set_qp_solver("FULL_CONDENSING_HPIPM")
    solver.set_integrator_type("IRK")
    solver.set_nlp_solver_type("SQP")
    solver.set_hessian_approx("GAUSS_NEWTON")
    solver.set_sim_method_num_stages(4)
    solver.set_sim_method_num_steps(3)
    solver.set_sim_method_newton_iter(5)
    solver.set_maximum_iterations(max_iterations)
    solver.set_print_level(print_level)
    return solver


def configure_ipopt_solver(max_iterations: int) -> Solver.IPOPT:
    solver = Solver.IPOPT(show_online_optim=False, _max_iter=max_iterations, show_options=dict(show_bounds=True))
    solver.set_warm_start_init_point("yes")
    solver.set_mu_init(1e-2)
    solver.set_tol(1e-6)
    solver.set_dual_inf_tol(1e-6)
    solver.set_constr_viol_tol(1e-6)
    linear_solver = "ma57" if sys_platform == "linux" else "mumps"
    solver.set_linear_solver(linear_solver)
    return solver


def summarize_windows(sol, requested_windows: int) -> None:
    def _fmt(value) -> str:
        return "None" if value is None else f"{value:.6f}"

    merged_solution = sol[0]
    window_solutions = sol[1] if len(sol) > 1 else []
    achieved_windows = 1 + len(window_solutions)

    print(f"merged_status: {merged_solution.status}")
    print(f"merged_cost: {merged_solution.cost}")
    print(f"merged_solver_time_s: {_fmt(merged_solution.solver_time_to_optimize)}")
    print(f"merged_wall_time_s: {_fmt(merged_solution.real_time_to_optimize)}")
    print(f"requested_windows: {requested_windows}")
    print(f"achieved_windows: {achieved_windows}")

    if window_solutions:
        print(f"additional_window_count: {len(window_solutions)}")
        for idx, window_solution in enumerate(window_solutions):
            print(
                f"window[{idx}] status={window_solution.status} "
                f"solver_time_s={_fmt(window_solution.solver_time_to_optimize)} "
                f"wall_time_s={_fmt(window_solution.real_time_to_optimize)}"
            )


def build_window_summary(sol, requested_windows: int) -> dict:
    merged_solution = sol[0]
    window_solutions = sol[1] if len(sol) > 1 else []
    achieved_windows = 1 + len(window_solutions)
    wheel_trace = merged_solution.decision_states(to_merge=SolutionMerge.NODES)["q"][2, :]
    objective = float(np.nansum(merged_solution.cost)) if getattr(merged_solution, "cost", None) is not None else float("nan")
    return {
        "mode": "rho",
        "status": merged_solution.status,
        "objective": objective,
        "solver_time_s": merged_solution.solver_time_to_optimize,
        "wall_time_s": merged_solution.real_time_to_optimize,
        "requested_windows": requested_windows,
        "achieved_windows": achieved_windows,
        "window_count": len(window_solutions),
        "final_wheel_angle": float(wheel_trace[-1]),
        "wheel_angle_trace": wheel_trace,
        "solution": merged_solution,
        "window_solutions": window_solutions,
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
    objective = float(np.nansum(sol.cost)) if getattr(sol, "cost", None) is not None else float("nan")
    return {
        "mode": "single_shot",
        "status": sol.status,
        "objective": objective,
        "solver_time_s": sol.solver_time_to_optimize,
        "wall_time_s": sol.real_time_to_optimize,
        "final_wheel_angle": float(wheel_trace[-1]),
        "wheel_angle_trace": wheel_trace,
        "solution": sol,
        "window_solutions": [],
    }


def _shift_cyclical_trajectory(values: np.ndarray, nodes_per_cycle: int) -> np.ndarray:
    n_plus_one_cycles = values[nodes_per_cycle:-1]
    last_cycle = values[-nodes_per_cycle - 1 :]
    return last_cycle if n_plus_one_cycles.size == 0 else np.concatenate((n_plus_one_cycles, last_cycle))


def estimate_periodic_cn_sum_from_cn(cn_values: np.ndarray, tauc: float, dt: float) -> np.ndarray:
    cn_dot = np.gradient(cn_values, dt)
    return cn_values + tauc * cn_dot


def apply_standard_warmup_to_periodic_nmpc(periodic_nmpc, warmup_solution) -> None:
    periodic_nmpc.advance_window_bounds_states(warmup_solution)
    periodic_nmpc.advance_window_initial_guess_states(warmup_solution)
    periodic_nmpc.advance_window_initial_guess_controls(warmup_solution)

    warmup_states = warmup_solution.decision_states(to_merge=SolutionMerge.NODES)
    dt = periodic_nmpc.cycle_duration / periodic_nmpc.cycle_len
    muscle_models = {model.muscle_name: model for model in periodic_nmpc.nlp[0].model.muscles_dynamics_model}
    for key in periodic_nmpc.nlp[0].states.keys():
        if not key.startswith("Cn_sum_"):
            continue

        source_key = key.replace("Cn_sum_", "Cn_")
        if source_key not in warmup_states:
            continue

        muscle_name = key.replace("Cn_sum_", "")
        tauc = muscle_models[muscle_name].tauc
        source_values = warmup_states[source_key][0]
        cn_sum_values = estimate_periodic_cn_sum_from_cn(source_values, tauc=tauc, dt=dt)
        shifted_values = _shift_cyclical_trajectory(cn_sum_values, periodic_nmpc.nodes_per_cycle)
        periodic_nmpc.nlp[0].x_init[key].init[0, :] = shifted_values
        periodic_nmpc.nlp[0].x_bounds[key].min[0, 0] = cn_sum_values[periodic_nmpc.nodes_per_cycle]
        periodic_nmpc.nlp[0].x_bounds[key].max[0, 0] = cn_sum_values[periodic_nmpc.nodes_per_cycle]


def run_standard_ipopt_warmup(
    args: argparse.Namespace,
    mhe_info: dict,
    cycling_info: dict,
    simulation_conditions: dict,
    model_path: Path,
):
    stim_time = list(
        np.linspace(
            0,
            mhe_info["cycle_duration"] * args.cycles_per_window,
            args.stimulations_per_cycle * args.cycles_per_window,
            endpoint=False,
        )
    )
    warmup_model = set_fes_model(str(model_path), stim_time, periodic_cn_sum_approximation=False)
    warmup_nmpc = prepare_nmpc(warmup_model, dict(mhe_info), dict(cycling_info), dict(simulation_conditions))
    warmup_nmpc.n_cycles_simultaneous = args.cycles_per_window

    warmup_solver = configure_ipopt_solver(max_iterations=args.max_ipopt_iterations)
    warmup_sol = super(RecedingHorizonOptimization, warmup_nmpc).solve(
        solver=warmup_solver,
        warm_start=None,
    )
    return warmup_sol


def build_codegen_names(args: argparse.Namespace) -> tuple[str, str]:
    objective_slug = args.objective.replace(",", "_")
    suffix = args.codegen_tag or (
        f"{args.solver}_{args.model_formulation}_{objective_slug}_{args.objective_shape}_{args.n_windows}mhe_{args.cycles_per_window}cyc"
    )
    return (
        f"cycling_fes_periodic_{suffix}",
        f"result/acados/c_generated_code_{suffix}",
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
    model_path = example_dir / "../../msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod"
    cycle_duration = 1.0
    total_window_duration = cycle_duration * args.cycles_per_window
    total_stimulations = args.stimulations_per_cycle * args.cycles_per_window
    stim_time = list(np.linspace(0, total_window_duration, total_stimulations, endpoint=False))
    periodic_cn_sum_approximation = args.model_formulation == "periodic"
    use_external_forces = args.torque_application == "external_forces"
    ode_solver = build_ode_solver(args)
    model = set_fes_model(str(model_path), stim_time, periodic_cn_sum_approximation=periodic_cn_sum_approximation)

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
        "init_guess_file_path": None,
    }

    nmpc = prepare_nmpc(model, mhe_info, cycling_info, simulation_conditions)
    nmpc.n_cycles_simultaneous = args.cycles_per_window

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

    if periodic_cn_sum_approximation and not args.disable_standard_ipopt_warmup:
        if echo:
            print("running_standard_ipopt_warmup: True")
        warmup_solution = run_standard_ipopt_warmup(args, mhe_info, cycling_info, simulation_conditions, model_path)
        apply_standard_warmup_to_periodic_nmpc(nmpc, warmup_solution)

    def update_functions(_nmpc, cycle_idx, _sol):
        print(f"window {cycle_idx}")
        return cycle_idx + 1 < args.n_windows

    if args.solver == "acados":
        model_name, generated_code_path = build_codegen_names(args)
        solver = configure_acados_solver(
            model_name=model_name,
            generated_code_path=generated_code_path,
            max_iterations=args.max_acados_iterations,
        )
    else:
        solver = configure_ipopt_solver(max_iterations=args.max_ipopt_iterations)

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
        cycle_solutions=MultiCyclicCycleSolutions.FIRST_CYCLES,
        get_all_iterations=False,
        cyclic_options={"states": {}},
        max_consecutive_failing=args.max_consecutive_failing,
    )
    if echo:
        summarize_windows(sol, requested_windows=args.n_windows)
    summary = build_window_summary(sol, requested_windows=args.n_windows)
    summary["args"] = args
    return summary


def main(cli_args: list[str] | None = None):
    parser = build_argument_parser()
    args = parser.parse_args(cli_args)
    solve_case(args, echo=True)


if __name__ == "__main__":
    main()
