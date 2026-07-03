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
ACADOS_STATUS_NAMES = {
    0: "ACADOS_SUCCESS",
    1: "ACADOS_NAN_DETECTED",
    2: "ACADOS_MAXITER",
    3: "ACADOS_MINSTEP",
    4: "ACADOS_QP_FAILURE",
    5: "ACADOS_READY",
    6: "ACADOS_UNBOUNDED",
    7: "ACADOS_TIMEOUT",
    8: "ACADOS_QPSCALING_BOUNDS_NOT_SATISFIED",
}


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
        choices=("rk4", "rk8", "irk", "collocation"),
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
        "--state-scaling",
        type=str,
        choices=("none", "fes", "full"),
        default="none",
        help="Scale optimization states: none, FES-only, or FES plus q/qdot.",
    )
    parser.add_argument(
        "--pulse-width-scaling",
        type=float,
        default=1 / 400,
        help="Scaling divisor, in seconds, for pulse-width controls.",
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
        "--acados-tolerance",
        type=float,
        default=None,
        help="Optional ACADOS NLP convergence tolerance applied to stationarity, constraints and complementarity.",
    )
    parser.add_argument(
        "--acados-collocation-type",
        type=str,
        choices=("GAUSS_LEGENDRE", "GAUSS_RADAU_IIA", "EXPLICIT_RUNGE_KUTTA"),
        default="GAUSS_LEGENDRE",
        help="Collocation tableau used by the ACADOS integrator.",
    )
    parser.add_argument(
        "--acados-sim-stages",
        type=int,
        default=4,
        help="Number of ACADOS integration stages per integration step.",
    )
    parser.add_argument(
        "--acados-sim-steps",
        type=int,
        default=None,
        help="Number of ACADOS integration substeps per shooting interval. Defaults to max(3, --rk-steps).",
    )
    parser.add_argument(
        "--acados-newton-iter",
        type=int,
        default=5,
        help="Number of Newton iterations used by the ACADOS implicit integrator.",
    )
    parser.add_argument(
        "--acados-newton-tol",
        type=float,
        default=None,
        help="Optional Newton tolerance for the ACADOS implicit integrator.",
    )
    parser.add_argument(
        "--acados-jac-reuse",
        type=int,
        choices=(0, 1),
        default=0,
        help="Reuse the ACADOS implicit integrator Jacobian across Newton iterations.",
    )
    parser.add_argument(
        "--acados-hessian-approx",
        choices=("GAUSS_NEWTON", "EXACT"),
        default="GAUSS_NEWTON",
        help="Hessian approximation used by ACADOS.",
    )
    parser.add_argument(
        "--acados-nlp-solver-type",
        choices=("SQP", "SQP_WITH_FEASIBLE_QP"),
        default="SQP",
        help="ACADOS NLP solver type.",
    )
    parser.add_argument(
        "--acados-search-direction-mode",
        choices=("NOMINAL_QP", "BYRD_OMOJOKUN", "FEASIBILITY_QP"),
        default="NOMINAL_QP",
        help="Search direction mode used by ACADOS, mainly for SQP_WITH_FEASIBLE_QP.",
    )
    parser.add_argument(
        "--acados-use-constraint-hessian-in-feas-qp",
        action="store_true",
        help="Use constraint Hessians in the feasibility QP of SQP_WITH_FEASIBLE_QP.",
    )
    parser.add_argument(
        "--acados-disable-direction-mode-switch-to-nominal",
        action="store_true",
        help="Keep ACADOS in the selected non-nominal search direction mode.",
    )
    parser.add_argument(
        "--acados-regularize-method",
        choices=(
            "NO_REGULARIZE",
            "MIRROR",
            "PROJECT",
            "PROJECT_REDUC_HESS",
            "CONVEXIFY",
            "GERSHGORIN_LEVENBERG_MARQUARDT",
        ),
        default="GERSHGORIN_LEVENBERG_MARQUARDT",
        help="ACADOS Hessian regularization method.",
    )
    parser.add_argument(
        "--acados-levenberg-marquardt",
        type=float,
        default=0.0,
        help="Additional Levenberg-Marquardt diagonal regularization for ACADOS.",
    )
    parser.add_argument(
        "--acados-globalization",
        choices=("FIXED_STEP", "MERIT_BACKTRACKING", "FUNNEL_L1PEN_LINESEARCH"),
        default="MERIT_BACKTRACKING",
        help="ACADOS globalization strategy.",
    )
    parser.add_argument(
        "--acados-fixed-step-length",
        type=float,
        default=1.0,
        help="Step length used when --acados-globalization=FIXED_STEP.",
    )
    parser.add_argument(
        "--acados-nlp-qp-tol-strategy",
        choices=("FIXED_QP_TOL", "ADAPTIVE_CURRENT_RES_JOINT", "ADAPTIVE_QPSCALING"),
        default="ADAPTIVE_QPSCALING",
        help="Strategy used by ACADOS to set QP tolerances inside SQP.",
    )
    parser.add_argument(
        "--acados-qp-iter-max",
        type=int,
        default=50,
        help="Maximum number of HPIPM QP iterations.",
    )
    parser.add_argument(
        "--acados-ext-qp-res",
        action="store_true",
        help="Ask ACADOS to log extended QP residuals in the statistics table.",
    )
    parser.add_argument(
        "--acados-diagnostics",
        action="store_true",
        help="Print ACADOS residuals, SQP/QP stats and finite-value checks after the solve.",
    )
    parser.add_argument(
        "--acados-print-level",
        type=int,
        default=0,
        help="Verbosity passed to ACADOS.",
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
        "--disable-periodic-fes-warmup-projection",
        action="store_true",
        help="Do not project periodic Ding FES initial guesses with a local rollout before ACADOS.",
    )
    parser.add_argument(
        "--periodic-fes-warmup-projection-weight",
        type=float,
        default=1.0,
        help="Blend weight between the original warmup FES states and the periodic Ding rollout.",
    )
    parser.add_argument(
        "--periodic-fes-warmup-projection-mode",
        choices=("calcium", "all"),
        default="all",
        help="Project only the periodic calcium states or all Ding fatigue states.",
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
    if args.ode_solver == "irk":
        return OdeSolver.IRK(
            polynomial_degree=args.collocation_degree, method=args.collocation_method
        )
    if args.ode_solver == "rk8":
        return OdeSolver.RK8(n_integration_steps=args.rk_steps)
    return OdeSolver.RK4(n_integration_steps=args.rk_steps)


def _ode_solver_suffix(ode_solver) -> str:
    if isinstance(ode_solver, OdeSolver.IRK):
        return f"irk_{ode_solver.polynomial_degree}_{ode_solver.method}"
    if isinstance(ode_solver, OdeSolver.COLLOCATION):
        return f"collocation_{ode_solver.polynomial_degree}_{ode_solver.method}"
    if isinstance(ode_solver, OdeSolver.RK8):
        return f"rk8_{ode_solver.n_integration_steps}"
    if isinstance(ode_solver, OdeSolver.RK4):
        return f"rk4_{ode_solver.n_integration_steps}"
    raise RuntimeError("ode_solver must be COLLOCATION, IRK, RK8, or RK4")


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
        "state_scaling": args.state_scaling,
        "pulse_width_scaling": args.pulse_width_scaling,
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
        "state_scaling": args.state_scaling,
        "pulse_width_scaling": args.pulse_width_scaling,
        "max_acados_iterations": args.max_acados_iterations,
        "acados_tolerance": args.acados_tolerance,
        "acados_collocation_type": args.acados_collocation_type,
        "acados_sim_stages": args.acados_sim_stages,
        "acados_sim_steps": args.acados_sim_steps,
        "acados_newton_iter": args.acados_newton_iter,
        "acados_newton_tol": args.acados_newton_tol,
        "acados_jac_reuse": args.acados_jac_reuse,
        "acados_hessian_approx": args.acados_hessian_approx,
        "acados_nlp_solver_type": args.acados_nlp_solver_type,
        "acados_search_direction_mode": args.acados_search_direction_mode,
        "acados_use_constraint_hessian_in_feas_qp": (
            args.acados_use_constraint_hessian_in_feas_qp
        ),
        "acados_disable_direction_mode_switch_to_nominal": (
            args.acados_disable_direction_mode_switch_to_nominal
        ),
        "acados_regularize_method": args.acados_regularize_method,
        "acados_levenberg_marquardt": args.acados_levenberg_marquardt,
        "acados_globalization": args.acados_globalization,
        "acados_fixed_step_length": args.acados_fixed_step_length,
        "acados_nlp_qp_tol_strategy": args.acados_nlp_qp_tol_strategy,
        "acados_qp_iter_max": args.acados_qp_iter_max,
        "acados_ext_qp_res": args.acados_ext_qp_res,
        "acados_print_level": args.acados_print_level,
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
    convergence_tolerance: float | None,
    collocation_type: str,
    sim_method_num_stages: int,
    sim_method_num_steps: int,
    sim_method_newton_iter: int,
    sim_method_newton_tol: float | None,
    sim_method_jac_reuse: int,
    hessian_approx: str,
    nlp_solver_type: str,
    search_direction_mode: str,
    use_constraint_hessian_in_feas_qp: bool,
    allow_direction_mode_switch_to_nominal: bool,
    regularize_method: str,
    levenberg_marquardt: float,
    globalization: str,
    fixed_step_length: float,
    nlp_qp_tol_strategy: str,
    qp_iter_max: int,
    ext_qp_res: bool,
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
    solver.set_nlp_solver_type(nlp_solver_type)
    solver.set_hessian_approx(hessian_approx)
    solver.set_sim_method_num_stages(sim_method_num_stages)
    solver.set_sim_method_num_steps(sim_method_num_steps)
    solver.set_sim_method_newton_iter(sim_method_newton_iter)
    solver.set_maximum_iterations(max_iterations)
    if convergence_tolerance is not None:
        solver.set_convergence_tolerance(convergence_tolerance)
    solver.set_print_level(print_level)
    solver.set_option_unsafe(collocation_type, "collocation_type")
    solver.set_option_unsafe(sim_method_jac_reuse, "sim_method_jac_reuse")
    solver.set_option_unsafe(search_direction_mode, "search_direction_mode")
    solver.set_option_unsafe(
        use_constraint_hessian_in_feas_qp,
        "use_constraint_hessian_in_feas_qp",
    )
    solver.set_option_unsafe(
        allow_direction_mode_switch_to_nominal,
        "allow_direction_mode_switch_to_nominal",
    )
    if sim_method_newton_tol is not None:
        solver.set_option_unsafe(float(sim_method_newton_tol), "sim_method_newton_tol")
    # Favor numerical robustness over raw speed for this periodic MHE.
    solver.set_option_unsafe(globalization, "globalization")
    solver.set_option_unsafe(fixed_step_length, "globalization_fixed_step_length")
    solver.set_option_unsafe(1, "globalization_line_search_use_sufficient_descent")
    solver.set_option_unsafe(0, "globalization_use_SOC")
    solver.set_option_unsafe("ROBUST", "hpipm_mode")
    solver.set_option_unsafe(regularize_method, "regularize_method")
    solver.set_option_unsafe(levenberg_marquardt, "levenberg_marquardt")
    solver.set_option_unsafe("OBJECTIVE_GERSHGORIN", "qpscaling_scale_objective")
    solver.set_option_unsafe("INF_NORM", "qpscaling_scale_constraints")
    solver.set_option_unsafe(nlp_qp_tol_strategy, "nlp_qp_tol_strategy")
    solver.set_option_unsafe(qp_iter_max, "qp_solver_iter_max")
    solver.set_option_unsafe(1 if ext_qp_res else 0, "nlp_solver_ext_qp_res")
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


def _status_label(status) -> str:
    if status is None:
        return "None"
    return ACADOS_STATUS_NAMES.get(status, str(status))


def _array_finite_summary(values) -> dict:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    finite_values = array[finite]
    return {
        "shape": array.shape,
        "finite": bool(np.all(finite)),
        "nonfinite_count": int(array.size - np.count_nonzero(finite)),
        "min": float(np.min(finite_values)) if finite_values.size else float("nan"),
        "max": float(np.max(finite_values)) if finite_values.size else float("nan"),
    }


def _dict_finite_summary(values_by_key: dict) -> dict:
    summary = {}
    for key, values in values_by_key.items():
        item = _array_finite_summary(values)
        if not item["finite"]:
            summary[key] = item
    return summary


def _get_acados_template_solver(solution):
    ocp = getattr(solution, "ocp", None)
    interface = getattr(ocp, "ocp_solver", None)
    return getattr(interface, "ocp_solver", None)


def _safe_acados_stat(acados_solver, field: str):
    try:
        return acados_solver.get_stats(field)
    except Exception as exc:  # noqa: BLE001 - diagnostics should not mask a solve.
        return {"error": str(exc)}


def _safe_acados_stage_field(acados_solver, stage: int, field: str):
    try:
        return acados_solver.get(stage, field)
    except Exception as exc:  # noqa: BLE001 - diagnostics should not mask a solve.
        return {"error": str(exc)}


def collect_acados_diagnostics(solution) -> dict:
    diagnostics = {
        "status": solution.status,
        "status_label": _status_label(solution.status),
        "state_nonfinite": {},
        "control_nonfinite": {},
        "solver_available": False,
    }

    try:
        states = solution.decision_states(to_merge=SolutionMerge.NODES)
        diagnostics["state_nonfinite"] = _dict_finite_summary(states)
    except Exception as exc:  # noqa: BLE001 - diagnostics should not mask a solve.
        diagnostics["state_error"] = str(exc)

    try:
        controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
        diagnostics["control_nonfinite"] = _dict_finite_summary(controls)
    except Exception as exc:  # noqa: BLE001 - diagnostics should not mask a solve.
        diagnostics["control_error"] = str(exc)

    acados_solver = _get_acados_template_solver(solution)
    if acados_solver is None:
        return diagnostics

    diagnostics["solver_available"] = True
    for field in (
        "sqp_iter",
        "nlp_iter",
        "qp_iter",
        "qp_stat",
        "alpha",
        "residuals",
        "qpscaling_status",
        "time_tot",
        "time_qp",
        "time_qp_solver_call",
        "time_sim",
        "time_glob",
        "time_reg",
        "time_qpscaling",
    ):
        diagnostics[field] = _safe_acados_stat(acados_solver, field)

    stats = _safe_acados_stat(acados_solver, "statistics")
    diagnostics["statistics"] = stats
    if not isinstance(stats, dict):
        stats_array = np.asarray(stats, dtype=float)
        diagnostics["statistics_finite"] = _array_finite_summary(stats_array)
        if stats_array.size and stats_array.ndim == 2:
            diagnostics["statistics_last_column"] = stats_array[:, -1]

    stage_nonfinite = []
    stage = 0
    while True:
        stage_items = {}
        any_available = False
        for field in ("x", "u", "pi", "lam"):
            values = _safe_acados_stage_field(acados_solver, stage, field)
            if isinstance(values, dict):
                continue
            any_available = True
            item = _array_finite_summary(values)
            if not item["finite"]:
                stage_items[field] = item
        if stage_items:
            stage_nonfinite.append({"stage": stage, "fields": stage_items})
        if not any_available:
            break
        stage += 1
        if stage > 10000:
            diagnostics["stage_scan_error"] = "Stopped after 10000 stages."
            break

    diagnostics["stage_nonfinite"] = stage_nonfinite
    diagnostics["n_stages_scanned"] = stage
    return diagnostics


def _format_array(values) -> str:
    if values is None:
        return "None"
    if isinstance(values, dict):
        return f"error={values.get('error')}"
    array = np.asarray(values, dtype=float).squeeze()
    return np.array2string(array, precision=3, suppress_small=False)


def print_acados_diagnostics(label: str, diagnostics: dict) -> None:
    print(
        f"{label} acados_status={diagnostics.get('status')} "
        f"({diagnostics.get('status_label')})"
    )
    print(f"{label} solver_available={diagnostics.get('solver_available')}")
    if diagnostics.get("state_nonfinite"):
        print(f"{label} state_nonfinite={diagnostics['state_nonfinite']}")
    if diagnostics.get("control_nonfinite"):
        print(f"{label} control_nonfinite={diagnostics['control_nonfinite']}")
    if diagnostics.get("stage_nonfinite"):
        print(f"{label} stage_nonfinite={diagnostics['stage_nonfinite'][:5]}")
    print(f"{label} residuals={_format_array(diagnostics.get('residuals'))}")
    print(f"{label} sqp_iter={_format_array(diagnostics.get('sqp_iter'))}")
    print(f"{label} qp_iter={_format_array(diagnostics.get('qp_iter'))}")
    print(f"{label} qp_stat={_format_array(diagnostics.get('qp_stat'))}")
    print(f"{label} alpha={_format_array(diagnostics.get('alpha'))}")
    print(
        f"{label} qpscaling_status="
        f"{_format_array(diagnostics.get('qpscaling_status'))}"
    )
    if "statistics_finite" in diagnostics:
        print(f"{label} statistics_finite={diagnostics['statistics_finite']}")
    if "statistics_last_column" in diagnostics:
        print(
            f"{label} statistics_last_column="
            f"{_format_array(diagnostics['statistics_last_column'])}"
        )


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


def _to_numpy_vector(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape((-1,))


def _ding_state_keys(muscle_name: str) -> tuple[str, str, str, str, str, str]:
    return (
        f"Cn_{muscle_name}",
        f"Cn_sum_{muscle_name}",
        f"F_{muscle_name}",
        f"A_{muscle_name}",
        f"Tau1_{muscle_name}",
        f"Km_{muscle_name}",
    )


def _state_trajectory_bounds(
    periodic_nmpc, key: str, n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    bounds = periodic_nmpc.nlp[0].x_bounds[key]
    lower = np.empty(n_nodes)
    upper = np.empty(n_nodes)
    for node in range(n_nodes):
        column = 0 if node == 0 else 2 if node == n_nodes - 1 else 1
        lower[node] = bounds.min[0, column]
        upper[node] = bounds.max[0, column]
    return lower, upper


def _clip_state_trajectory_to_bounds(
    periodic_nmpc, key: str, values: np.ndarray
) -> np.ndarray:
    lower, upper = _state_trajectory_bounds(periodic_nmpc, key, values.size)
    return np.minimum(np.maximum(values, lower), upper)


def _periodic_ding_rhs(
    muscle_model, state: np.ndarray, pulse_width: float
) -> np.ndarray:
    return _to_numpy_vector(
        muscle_model.system_dynamics(
            cn=state[0],
            cn_sum=state[1],
            f=state[2],
            a=state[3],
            tau1=state[4],
            km=state[5],
            pulse_width=pulse_width,
        )
    )


def _periodic_calcium_rhs(muscle_model, state: np.ndarray) -> np.ndarray:
    cn = state[0]
    cn_sum = state[1]
    return np.array(
        [
            float(muscle_model.cn_dot_fun(cn, cn_sum)),
            float(muscle_model.cn_sum_dot_fun(cn_sum)),
        ]
    )


def _rk4_periodic_ding_step(
    muscle_model, state: np.ndarray, pulse_width: float, dt: float
) -> np.ndarray:
    k1 = _periodic_ding_rhs(muscle_model, state, pulse_width)
    k2 = _periodic_ding_rhs(muscle_model, state + 0.5 * dt * k1, pulse_width)
    k3 = _periodic_ding_rhs(muscle_model, state + 0.5 * dt * k2, pulse_width)
    k4 = _periodic_ding_rhs(muscle_model, state + dt * k3, pulse_width)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def _rk4_periodic_calcium_step(
    muscle_model, state: np.ndarray, dt: float
) -> np.ndarray:
    k1 = _periodic_calcium_rhs(muscle_model, state)
    k2 = _periodic_calcium_rhs(muscle_model, state + 0.5 * dt * k1)
    k3 = _periodic_calcium_rhs(muscle_model, state + 0.5 * dt * k2)
    k4 = _periodic_calcium_rhs(muscle_model, state + dt * k3)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def _periodic_fes_rollout_defects(periodic_nmpc) -> dict[str, float]:
    dt = periodic_nmpc.cycle_duration / periodic_nmpc.cycle_len
    defects = {}
    for muscle_model in periodic_nmpc.nlp[0].model.muscles_dynamics_model:
        muscle_name = muscle_model.muscle_name
        state_keys = _ding_state_keys(muscle_name)
        control_key = f"last_pulse_width_{muscle_name}"
        if any(key not in periodic_nmpc.nlp[0].x_init.keys() for key in state_keys):
            continue
        if control_key not in periodic_nmpc.nlp[0].u_init.keys():
            continue

        states = np.vstack(
            [periodic_nmpc.nlp[0].x_init[key].init[0, :] for key in state_keys]
        )
        controls = periodic_nmpc.nlp[0].u_init[control_key].init[0, :]
        max_defect = 0.0
        for node, pulse_width in enumerate(controls):
            expected = _rk4_periodic_ding_step(
                muscle_model, states[:, node], pulse_width, dt
            )
            max_defect = max(
                max_defect, float(np.max(np.abs(states[:, node + 1] - expected)))
            )
        defects[muscle_name] = max_defect
    return defects


def _projection_state_keys(muscle_name: str, projection_mode: str) -> tuple[str, ...]:
    state_keys = _ding_state_keys(muscle_name)
    if projection_mode == "calcium":
        return state_keys[:2]
    if projection_mode == "all":
        return state_keys
    raise ValueError(
        "--periodic-fes-warmup-projection-mode must be 'calcium' or 'all'."
    )


def _project_periodic_states(
    muscle_model,
    original_states: np.ndarray,
    controls: np.ndarray,
    dt: float,
    projection_mode: str,
) -> np.ndarray:
    projected_states = np.empty_like(original_states)
    projected_states[:, 0] = original_states[:, 0]
    for node in range(controls.size):
        if projection_mode == "calcium":
            projected_states[:, node + 1] = _rk4_periodic_calcium_step(
                muscle_model, projected_states[:, node], dt
            )
        else:
            projected_states[:, node + 1] = _rk4_periodic_ding_step(
                muscle_model, projected_states[:, node], controls[node], dt
            )
    return projected_states


def project_periodic_fes_initial_guess(
    periodic_nmpc,
    projection_weight: float = 1.0,
    projection_mode: str = "all",
) -> dict[str, float]:
    if projection_weight < 0.0 or projection_weight > 1.0:
        raise ValueError(
            "--periodic-fes-warmup-projection-weight must be between 0 and 1."
        )

    dt = periodic_nmpc.cycle_duration / periodic_nmpc.cycle_len
    defects_before = _periodic_fes_rollout_defects(periodic_nmpc)
    projected_muscles = 0
    clipped_values = 0
    max_bound_violation = 0.0

    for muscle_model in periodic_nmpc.nlp[0].model.muscles_dynamics_model:
        muscle_name = muscle_model.muscle_name
        state_keys = _projection_state_keys(muscle_name, projection_mode)
        control_key = f"last_pulse_width_{muscle_name}"
        if any(key not in periodic_nmpc.nlp[0].x_init.keys() for key in state_keys):
            continue
        if control_key not in periodic_nmpc.nlp[0].u_init.keys():
            continue

        original_states = np.vstack(
            [periodic_nmpc.nlp[0].x_init[key].init[0, :] for key in state_keys]
        )
        controls = periodic_nmpc.nlp[0].u_init[control_key].init[0, :]
        projected_states = _project_periodic_states(
            muscle_model, original_states, controls, dt, projection_mode
        )

        blended_states = (
            projection_weight * projected_states
            + (1.0 - projection_weight) * original_states
        )
        for state_idx, key in enumerate(state_keys):
            values = blended_states[state_idx, :]
            lower, upper = _state_trajectory_bounds(periodic_nmpc, key, values.size)
            lower_violation = np.maximum(lower - values, 0.0)
            upper_violation = np.maximum(values - upper, 0.0)
            violations = lower_violation + upper_violation
            clipped_values += int(np.count_nonzero(violations))
            max_bound_violation = max(max_bound_violation, float(np.max(violations)))
            periodic_nmpc.nlp[0].x_init[key].init[0, :] = np.minimum(
                np.maximum(values, lower), upper
            )
        projected_muscles += 1

    defects_after = _periodic_fes_rollout_defects(periodic_nmpc)
    periodic_nmpc._correct_init_guess_to_fit_bounds(corrected_input="states")
    periodic_nmpc._sync_acados_state_bounds()

    max_before = max(defects_before.values(), default=0.0)
    max_after = max(defects_after.values(), default=0.0)
    return {
        "projected_muscles": projected_muscles,
        "projection_weight": projection_weight,
        "projection_mode": projection_mode,
        "max_defect_before": max_before,
        "max_defect_after": max_after,
        "clipped_values": clipped_values,
        "max_bound_violation": max_bound_violation,
    }


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


def apply_standard_warmup_to_periodic_nmpc(
    periodic_nmpc,
    warmup_solution,
    project_fes_warmup: bool = True,
    projection_weight: float = 1.0,
    projection_mode: str = "all",
    echo: bool = False,
):
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

    if project_fes_warmup:
        projection_summary = project_periodic_fes_initial_guess(
            periodic_nmpc,
            projection_weight=projection_weight,
            projection_mode=projection_mode,
        )
        if echo:
            print(
                "periodic_fes_warmup_projection: "
                f"projected_muscles={projection_summary['projected_muscles']} "
                f"mode={projection_summary['projection_mode']} "
                f"weight={projection_summary['projection_weight']:.3g} "
                f"max_defect_before={projection_summary['max_defect_before']:.6g} "
                f"max_defect_after={projection_summary['max_defect_after']:.6g} "
                f"clipped_values={projection_summary['clipped_values']} "
                f"max_bound_violation={projection_summary['max_bound_violation']:.6g}"
            )

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
        "state_scaling": args.state_scaling,
        "pulse_width_scaling": args.pulse_width_scaling,
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
        if args.ode_solver in ("rk4", "rk8"):
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
        print(f"state_scaling: {args.state_scaling}")
        print(f"pulse_width_scaling: {args.pulse_width_scaling}")
        if args.solver == "acados":
            print(
                "periodic_fes_warmup_projection: "
                f"{not args.disable_periodic_fes_warmup_projection}"
            )
            print(
                "periodic_fes_warmup_projection_weight: "
                f"{args.periodic_fes_warmup_projection_weight}"
            )
            print(
                "periodic_fes_warmup_projection_mode: "
                f"{args.periodic_fes_warmup_projection_mode}"
            )
            print(f"acados_collocation_type: {args.acados_collocation_type}")
            print(f"acados_sim_stages: {args.acados_sim_stages}")
            print(
                "acados_sim_steps: "
                f"{args.acados_sim_steps if args.acados_sim_steps is not None else max(3, args.rk_steps)}"
            )
            print(f"acados_newton_iter: {args.acados_newton_iter}")
            print(f"acados_newton_tol: {args.acados_newton_tol}")
            print(f"acados_jac_reuse: {args.acados_jac_reuse}")
            print(f"acados_tolerance: {args.acados_tolerance}")
            print(f"acados_hessian_approx: {args.acados_hessian_approx}")
            print(f"acados_nlp_solver_type: {args.acados_nlp_solver_type}")
            print(
                "acados_search_direction_mode: " f"{args.acados_search_direction_mode}"
            )
            print(
                "acados_use_constraint_hessian_in_feas_qp: "
                f"{args.acados_use_constraint_hessian_in_feas_qp}"
            )
            print(
                "acados_allow_direction_mode_switch_to_nominal: "
                f"{not args.acados_disable_direction_mode_switch_to_nominal}"
            )
            print(f"acados_regularize_method: {args.acados_regularize_method}")
            print(f"acados_levenberg_marquardt: {args.acados_levenberg_marquardt}")
            print(f"acados_globalization: {args.acados_globalization}")
            print(f"acados_fixed_step_length: {args.acados_fixed_step_length}")
            print(f"acados_nlp_qp_tol_strategy: {args.acados_nlp_qp_tol_strategy}")
            print(f"acados_qp_iter_max: {args.acados_qp_iter_max}")
            print(f"acados_ext_qp_res: {args.acados_ext_qp_res}")
            print(f"acados_diagnostics: {args.acados_diagnostics}")
            print(f"acados_print_level: {args.acados_print_level}")
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
            nmpc,
            warmup_solution,
            project_fes_warmup=not args.disable_periodic_fes_warmup_projection,
            projection_weight=args.periodic_fes_warmup_projection_weight,
            projection_mode=args.periodic_fes_warmup_projection_mode,
            echo=echo,
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
            convergence_tolerance=args.acados_tolerance,
            collocation_type=args.acados_collocation_type,
            sim_method_num_stages=args.acados_sim_stages,
            sim_method_num_steps=(
                args.acados_sim_steps
                if args.acados_sim_steps is not None
                else max(3, args.rk_steps)
            ),
            sim_method_newton_iter=args.acados_newton_iter,
            sim_method_newton_tol=args.acados_newton_tol,
            sim_method_jac_reuse=args.acados_jac_reuse,
            hessian_approx=args.acados_hessian_approx,
            nlp_solver_type=args.acados_nlp_solver_type,
            search_direction_mode=args.acados_search_direction_mode,
            use_constraint_hessian_in_feas_qp=(
                args.acados_use_constraint_hessian_in_feas_qp
            ),
            allow_direction_mode_switch_to_nominal=(
                not args.acados_disable_direction_mode_switch_to_nominal
            ),
            regularize_method=args.acados_regularize_method,
            levenberg_marquardt=args.acados_levenberg_marquardt,
            globalization=args.acados_globalization,
            fixed_step_length=args.acados_fixed_step_length,
            nlp_qp_tol_strategy=args.acados_nlp_qp_tol_strategy,
            qp_iter_max=args.acados_qp_iter_max,
            ext_qp_res=args.acados_ext_qp_res,
            print_level=args.acados_print_level,
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
            if args.solver == "acados" and args.acados_diagnostics:
                print_acados_diagnostics("single_shot", collect_acados_diagnostics(sol))
        summary = build_single_shot_summary(sol)
        if args.solver == "acados" and args.acados_diagnostics:
            summary["acados_diagnostics"] = collect_acados_diagnostics(sol)
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
        if args.solver == "acados" and args.acados_diagnostics:
            _, source_window_solutions, _ = _split_receding_solution(sol)
            if source_window_solutions:
                for idx, window_solution in enumerate(source_window_solutions):
                    print_acados_diagnostics(
                        f"window[{idx}]",
                        collect_acados_diagnostics(window_solution),
                    )
            else:
                print_acados_diagnostics("merged", collect_acados_diagnostics(sol[0]))
    summary = build_window_summary(
        sol, requested_windows=args.n_windows, cycles_per_window=args.cycles_per_window
    )
    if args.solver == "acados" and args.acados_diagnostics:
        _, source_window_solutions, _ = _split_receding_solution(sol)
        summary["acados_diagnostics"] = [
            collect_acados_diagnostics(window_solution)
            for window_solution in source_window_solutions
        ]
    summary["args"] = args
    return summary


def main(cli_args: list[str] | None = None):
    parser = build_argument_parser()
    args = parser.parse_args(cli_args)
    ensure_acados_environment()
    solve_case(args, echo=True)


if __name__ == "__main__":
    main()
