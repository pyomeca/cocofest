"""
Minimal ACADOS example for the periodic Ding pulse-width cycling MHE with a constant crank torque.
"""

import os
from pathlib import Path

import numpy as np

from bioptim import MultiCyclicCycleSolutions, OdeSolver, Solver

from cycling_pulse_width_mhe import prepare_nmpc, set_fes_model


def main():
    example_dir = Path(__file__).resolve().parent
    model_path = example_dir / "../../msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod"
    stim_time = list(np.linspace(0, 1.0, 4, endpoint=False))
    model = set_fes_model(str(model_path), stim_time, periodic_cn_sum_approximation=True)

    mhe_info = {
        "cycle_duration": 1,
        "n_cycles_to_advance": 1,
        "n_cycles": 1,
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
        return cycle_idx < 1

    solver = Solver.ACADOS()
    solver.set_acados_dir(os.environ.get("ACADOS_SOURCE_DIR", str(Path.home() / "Documents/bioptim/external/acados")))
    solver.set_c_generated_code_path("result/acados/c_generated_code_fes_periodic")
    solver.set_acados_model_name("cycling_fes_periodic")
    solver.set_maximum_iterations(5)
    solver.set_print_level(0)

    sol = nmpc.solve_fes_nmpc(
        update_functions,
        solver=solver,
        total_cycles=1,
        external_force=None,
        cycle_solutions=MultiCyclicCycleSolutions.FIRST_CYCLES,
        get_all_iterations=False,
        cyclic_options={"states": {}},
        max_consecutive_failing=1,
    )
    print(f"solutions: {len(sol)}")
    print(f"status: {sol[0].status}")
    print(f"cost: {sol[0].cost}")


if __name__ == "__main__":
    main()
