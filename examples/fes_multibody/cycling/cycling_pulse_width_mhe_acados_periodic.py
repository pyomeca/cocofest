"""
Minimal ACADOS example for the periodic Ding pulse-width cycling MHE with a constant crank torque.

This variant keeps the ACADOS-friendly periodic approximation of the Ding 2007 dynamics and
solves multiple MHE windows so the example behaves like a true receding-horizon problem.
"""

import os
from pathlib import Path

import numpy as np

from bioptim import MultiCyclicCycleSolutions, OdeSolver, Solver

from cycling_pulse_width_mhe import prepare_nmpc, set_fes_model


def configure_acados_solver(model_name: str, generated_code_path: str) -> Solver.ACADOS:
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
    solver.set_maximum_iterations(10)
    solver.set_print_level(0)
    return solver


def summarize_windows(sol) -> None:
    def _fmt(value) -> str:
        return "None" if value is None else f"{value:.6f}"

    merged_solution = sol[0]
    window_solutions = sol[1] if len(sol) > 1 else []

    print(f"merged_status: {merged_solution.status}")
    print(f"merged_cost: {merged_solution.cost}")
    print(f"merged_solver_time_s: {_fmt(merged_solution.solver_time_to_optimize)}")
    print(f"merged_wall_time_s: {_fmt(merged_solution.real_time_to_optimize)}")

    if window_solutions:
        print(f"window_count: {len(window_solutions)}")
        for idx, window_solution in enumerate(window_solutions):
            print(
                f"window[{idx}] status={window_solution.status} "
                f"solver_time_s={_fmt(window_solution.solver_time_to_optimize)} "
                f"wall_time_s={_fmt(window_solution.real_time_to_optimize)}"
            )


def main():
    example_dir = Path(__file__).resolve().parent
    model_path = example_dir / "../../msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod"
    stim_time = list(np.linspace(0, 1.0, 4, endpoint=False))
    model = set_fes_model(str(model_path), stim_time, periodic_cn_sum_approximation=True)
    n_windows = 2

    mhe_info = {
        "cycle_duration": 1,
        "n_cycles_to_advance": 1,
        "n_cycles": n_windows,
        "ode_solver": OdeSolver.RK4(n_integration_steps=1),
        "use_sx": True,
        "cycle_len": len(stim_time),
        "n_cycles_simultaneous": 1,
    }
    cycling_info = {
        "turn_number": 1,
        "pedal_config": {"x_center": 0.35, "y_center": 0.0, "radius": 0.1},
        "constant_crank_torque": -0.2,
        "enforce_start_constraints": False,
        "periodic_cn_sum_approximation": True,
    }
    simulation_conditions = {
        "n_cycles_simultaneous": 1,
        "stimulation": len(stim_time),
        "minimize_force": True,
        "minimize_fatigue": False,
        "minimize_control": False,
        "cost_fun_weight": [1, 0, 0],
        "init_guess_file_path": None,
    }

    nmpc = prepare_nmpc(model, mhe_info, cycling_info, simulation_conditions)
    nmpc.n_cycles_simultaneous = 1

    def update_functions(_nmpc, cycle_idx, _sol):
        print(f"window {cycle_idx}")
        return cycle_idx < n_windows

    solver = configure_acados_solver(
        model_name="cycling_fes_periodic",
        generated_code_path="result/acados/c_generated_code_fes_periodic",
    )

    sol = nmpc.solve_fes_nmpc(
        update_functions,
        solver=solver,
        total_cycles=n_windows,
        external_force=None,
        cycle_solutions=MultiCyclicCycleSolutions.FIRST_CYCLES,
        get_all_iterations=False,
        cyclic_options={"states": {}},
        max_consecutive_failing=1,
    )
    summarize_windows(sol)


if __name__ == "__main__":
    main()
