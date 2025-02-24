"""
This example will do a nmpc of 10 stimulation example with Ding's 2007 frequency model.
This ocp was build to produce a elbow motion from 5 to 120 degrees.
The pulse width between minimal sensitivity threshold and 600us to satisfy the flexion and minimizing required elbow
torque control.
"""

import numpy as np
import biorbd
from bioptim import Solver, ControlType
from cocofest import (
    DingModelPulseWidthFrequencyWithFatigue,
    NmpcFesMsk,
    FesMskModel,
    PickleAnimate,
    SolutionToPickle,
)


minimum_pulse_width = DingModelPulseWidthFrequencyWithFatigue().pd0
DeltoideusClavicle_A_model = DingModelPulseWidthFrequencyWithFatigue(muscle_name="DeltoideusClavicle_A")
DeltoideusScapula_P_model = DingModelPulseWidthFrequencyWithFatigue(muscle_name="DeltoideusScapula_P")
TRIlong_model = DingModelPulseWidthFrequencyWithFatigue(muscle_name="TRIlong")
BIC_long_model = DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIC_long")
BIC_brevis_model = DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIC_brevis")

DeltoideusClavicle_A_model.alpha_a = -4.0 * 10e-1
DeltoideusScapula_P_model.alpha_a = -4.0 * 10e-1
TRIlong_model.alpha_a = -4.0 * 10e-1
BIC_long_model.alpha_a = -4.0 * 10e-1
BIC_brevis_model.alpha_a = -4.0 * 10e-1

model = FesMskModel(
    name=None,
    biorbd_path="../model_msk/simplified_UL_Seth.bioMod",
    muscles_model=[
        DeltoideusClavicle_A_model,
        DeltoideusScapula_P_model,
        TRIlong_model,
        BIC_long_model,
        BIC_brevis_model,
    ],
    activate_force_length_relationship=True,
    activate_force_velocity_relationship=True,
    activate_residual_torque=True,
)

nmpc = NmpcFesMsk.prepare_nmpc(
    model=model,
    stim_time=list(np.round(np.linspace(0, 1, 11), 2))[:-1],
    cycle_len=100,
    cycle_duration=1,
    n_cycles_simultaneous=1,
    n_cycles_to_advance=1,
    pulse_width={
        "min": minimum_pulse_width,
        "max": 0.0006,
        "bimapping": False,
    },
    msk_info={"with_residual_torque": True},
    objective={
        "cycling": {"x_center": 0.35, "y_center": 0, "radius": 0.1, "target": "marker"},
        "minimize_muscle_fatigue": True,
        "minimize_residual_torque": True,
    },
    initial_guess_warm_start=True,
    n_threads=8,
    control_type=ControlType.CONSTANT,  # ControlType.LINEAR_CONTINUOUS don't work for nmpc in bioptim
)

n_cycles_total = 8


def update_functions(_nmpc, cycle_idx, _sol):
    return cycle_idx < n_cycles_total  # True if there are still some cycle to perform


if __name__ == "__main__":
    sol = nmpc.solve(
        update_functions,
        solver=Solver.IPOPT(_max_iter=10000),
        cyclic_options={"states": {}},
        get_all_iterations=True,
        # n_cycles_simultaneous=1,
    )

    SolutionToPickle(sol[0], "results/cycling_fes_driven_nmpc_full_fatigue.pkl", "").pickle()
    [
        SolutionToPickle(sol[1][i], "results/cycling_fes_driven_nmpc_" + str(i) + "_fatigue.pkl", "").pickle()
        for i in range(len(sol[1]))
    ]

    biorbd_model = biorbd.Model("../model_msk/simplified_UL_Seth_full_mesh.bioMod")
    PickleAnimate("results/cycling_fes_driven_nmpc_full_fatigue.pkl").animate(model=biorbd_model)
