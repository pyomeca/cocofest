"""
This example will do a nmpc of 10 stimulation example with Ding's 2007 frequency model.
This ocp was build to produce a elbow motion from 5 to 120 degrees.
The pulse duration between minimal sensitivity threshold and 600us to satisfy the flexion and minimizing required elbow
torque control.
"""

import os
import biorbd
from bioptim import Solver
from cocofest import (
    DingModelPulseDurationFrequencyWithFatigue,
    NmpcFesMsk,
    FesMskModel,
    SolutionToPickle,
    PickleAnimate,
)

model = FesMskModel(
    name=None,
    biorbd_path="../msk_models/arm26_biceps_1dof.bioMod",
    muscles_model=[DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong")],
    activate_force_length_relationship=True,
    activate_force_velocity_relationship=True,
)

minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0

nmpc_fes_msk = NmpcFesMsk()
nmpc = nmpc_fes_msk.prepare_nmpc(
    model=model,
    stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    cycle_len=100,
    cycle_duration=1,
    pulse_duration={
        "min": minimum_pulse_duration,
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
nmpc_fes_msk.n_cycles = 2
sol = nmpc.solve(
    nmpc_fes_msk.update_functions,
    solver=Solver.IPOPT(),
    cyclic_options={"states": {}},
    get_all_iterations=True,
)

biorbd_model = biorbd.Model("../msk_models/arm26_biceps_1dof.bioMod")
temp_pickle_file_path = "pw_optim_dynamic_nmpc_full.pkl"
SolutionToPickle(sol[0], temp_pickle_file_path, "").pickle()

PickleAnimate(temp_pickle_file_path).animate(model=biorbd_model)

os.remove(temp_pickle_file_path)
