"""
Torque-driven cycling moving horizon example solved with ACADOS.

The FES pulse-width MHE uses time-varying stimulation and external-force numerical series. In bioptim 3.4 these series
are not exported as ACADOS model parameters, so this compact example keeps the same cycling model family but removes
the FES numerical time series to exercise the ACADOS moving-horizon pipeline end to end.
"""

import os
from pathlib import Path

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


def _resolve_example_path(path: str | os.PathLike, root: Path) -> str:
    path = Path(path)
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())


def _validate_window_len(window_len: int) -> None:
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


def build_update_function(full_target: np.ndarray, window_len: int):
    def update_functions(mhe: MovingHorizonEstimator, window_idx: int, _sol):
        target = full_target[window_idx : window_idx + window_len + 1]
        mhe.update_objectives_target(target=target[np.newaxis, :], list_index=0)
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


def prepare_mhe(
    model_path: str,
    window_len: int = 5,
    window_duration: float = 1.0,
    total_angle: float = -1.0,
    use_sx: bool = True,
    n_threads: int = 1,
):
    _validate_window_len(window_len)
    model = TorqueBiorbdModel(model_path)
    n_nodes = window_len + 1

    x_bounds = BoundsList()
    q_bounds = model.bounds_from_ranges("q")
    q_bounds.min[:] = [-50]
    q_bounds.max[:] = [50]
    x_bounds.add(
        "q",
        bounds=q_bounds,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    qdot_bounds = model.bounds_from_ranges("qdot")
    qdot_bounds.min[:] = [-50]
    qdot_bounds.max[:] = [50]
    x_bounds.add(
        "qdot",
        bounds=qdot_bounds,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    u_bounds = BoundsList()
    u_bounds.add(
        "tau",
        min_bound=np.full((model.nb_tau, 3), -1000),
        max_bound=np.full((model.nb_tau, 3), 1000),
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    q_target = np.linspace(0, total_angle, n_nodes)
    x_init = InitialGuessList()
    x_init.add(
        "q",
        np.vstack([np.full(n_nodes, 0.75), np.full(n_nodes, 1.5), q_target]),
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_init.add("qdot", np.zeros((model.nb_qdot, n_nodes)), interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialGuessList()
    u_init.add("tau", [0] * model.nb_tau)

    objectives = ObjectiveList()
    objectives.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        key="q",
        index=2,
        node=Node.ALL,
        target=q_target[np.newaxis, :],
        weight=10,
        multi_thread=False,
    )
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1e-4, multi_thread=False)

    return MovingHorizonEstimator(
        model,
        window_len,
        window_duration,
        dynamics=DynamicsOptions(
            ode_solver=OdeSolver.RK4(n_integration_steps=1),
            expand_dynamics=True,
            phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        ),
        common_objective_functions=objectives,
        x_bounds=x_bounds,
        x_init=x_init,
        u_bounds=u_bounds,
        u_init=u_init,
        use_sx=use_sx,
        n_threads=n_threads,
    )


def main(
    n_windows: int = 2,
    window_len: int = 5,
    total_angle: float = -1.0,
    perturbation_scale: float = 0.03,
    acados_dir: str | None = None,
    codegen_dir: str = "result/acados/c_generated_code",
):
    os.chdir(EXAMPLE_DIR)
    model_path = _resolve_example_path("../../msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod", EXAMPLE_DIR)
    codegen_dir = _resolve_example_path(codegen_dir, REPO_ROOT)
    mhe = prepare_mhe(model_path, window_len=window_len, total_angle=total_angle)
    full_target = build_reference_series(
        window_len=window_len,
        n_windows=n_windows,
        total_angle=total_angle,
        perturbation_scale=perturbation_scale,
    )

    solver = configure_acados_solver(
        acados_dir=acados_dir or os.environ.get("ACADOS_SOURCE_DIR"), codegen_dir=codegen_dir
    )
    sol, window_solutions = solve_windows(mhe, build_update_function(full_target, window_len), solver=solver)
    q_wheel = sol.decision_states(to_merge=SolutionMerge.NODES)["q"][2, :]
    print("window | target_end | estimated_end | delta")
    for window_idx, window_sol in enumerate(window_solutions):
        estimated_end = window_sol.decision_states(to_merge=SolutionMerge.NODES)["q"][2, -1]
        target_end = full_target[window_idx + window_len]
        print(
            f"{window_idx:>6} | {target_end:>10.6f} | {estimated_end:>13.6f} | {estimated_end - target_end:>+.6f}"
        )
    print("ACADOS cycling MHE status:", sol.status)
    print("Estimated wheel angle:", np.array2string(q_wheel, precision=4))
    print("Reference wheel angle:", np.array2string(full_target[-(window_len + 1) :], precision=4))
    return sol, window_solutions


if __name__ == "__main__":
    main()
