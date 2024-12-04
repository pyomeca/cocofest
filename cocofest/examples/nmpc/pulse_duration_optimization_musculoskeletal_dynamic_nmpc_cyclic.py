"""
This example will do a nmpc of 10 stimulation example with Ding's 2007 frequency model.
This ocp was build to produce a elbow motion from 5 to 120 degrees.
The pulse width between minimal sensitivity threshold and 600us to satisfy the flexion and minimizing required elbow
torque control.
"""

import biorbd
from bioptim import Solver, MultiCyclicNonlinearModelPredictiveControl, Solution
from cocofest import (
    DingModelPulseWidthFrequencyWithFatigue,
    NmpcFesMsk,
    FesMskModel,
)

model = FesMskModel(
    name=None,
    biorbd_path="../msk_models/arm26_biceps_1dof.bioMod",
    muscles_model=[DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIClong")],
    activate_force_length_relationship=True,
    activate_force_velocity_relationship=True,
    activate_residual_torque=True,
    external_force_set=None,
)

minimum_pulse_width = DingModelPulseWidthFrequencyWithFatigue().pd0

nmpc_fes_msk = NmpcFesMsk
nmpc = nmpc_fes_msk.prepare_nmpc(
    model=model,
    stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    cycle_duration=1,
    n_cycles_to_advance=1,
    n_cycles_simultaneous=3,
    pulse_width={
        "min": minimum_pulse_width,
        "max": 0.0006,
        "bimapping": False,
    },
    objective={"minimize_residual_torque": True},
    msk_info={
        "bound_type": "start_end",
        "bound_data": [[5], [120]],
        "with_residual_torque": True,
    },
)

n_cycles = 5


def update_functions(_nmpc: MultiCyclicNonlinearModelPredictiveControl, cycle_idx: int, _sol: Solution):
    return cycle_idx < n_cycles  # True if there are still some cycle to perform


sol = nmpc.solve(
    update_functions,
    solver=Solver.IPOPT(),
    cyclic_options={"states": {}},
    # get_all_iterations=True,
)

biorbd_model = biorbd.Model("../msk_models/arm26_biceps_1dof.bioMod")
sol.print_cost()
sol.graphs(show_bounds=True)
sol.animate(n_frames=200, show_tracked_markers=True)
