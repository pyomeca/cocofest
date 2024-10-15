"""
This example will do a 10 stimulation example with Ding's 2007 frequency model.
This ocp was build to produce a elbow motion from 5 to 120 degrees.
The stimulation frequency will be optimized between 10 and 100 Hz and pulse duration between minimal sensitivity
threshold and 600us to satisfy the flexion and minimizing required elbow torque control.
External forces will be applied to the system to simulate a real-world scenario.
"""
import numpy as np
from bioptim import Solver
from cocofest import DingModelPulseDurationFrequencyWithFatigue, OcpFesMsk, FesMskModel, SolutionToPickle, PickleAnimate
import biorbd

model = FesMskModel(
    name=None,
    # biorbd_path="../msk_models/arm26_cycling_pedal_aligned.bioMod",
    biorbd_path="../msk_models/simplified_UL_Seth_pedal_aligned.bioMod",
    muscles_model=[
        DingModelPulseDurationFrequencyWithFatigue(muscle_name="DeltoideusClavicle_A"),
        DingModelPulseDurationFrequencyWithFatigue(muscle_name="DeltoideusScapula_P"),
        DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
        DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIC_long"),
        DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIC_brevis"),
    ],
    activate_force_length_relationship=True,
    activate_force_velocity_relationship=True,
    activate_residual_torque=True,
    # segments_to_apply_external_forces=["r_ulna_radius_hand"],
)

minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0
ocp = OcpFesMsk.prepare_ocp(
    model=model,
    stim_time=list(np.round(np.linspace(0, 1, 11), 3))[:-1],
    n_shooting=100,
    final_time=1,
    pulse_duration={
        "min": minimum_pulse_duration,
        "max": 0.0006,
        "bimapping": False,
    },
    msk_info={"with_residual_torque": True},
    objective={
        "cycling": {"x_center": 0.3, "y_center": 0, "radius": 0.1, "target": "marker"},
        "minimize_residual_torque": True,
        "minimize_muscle_force": True,
    },
    warm_start=False,
    n_threads=5,
    # external_forces={"Global": True, "Segment_application": "r_ulna_radius_hand", "torque": np.array([0, 0, -5]), "force": np.array([0, 0, 0]), "point_of_application": np.array([0, 0, 0])},
)

# sol = ocp.solve(solver=Solver.IPOPT(_max_iter=1))
sol = ocp.solve()
# SolutionToPickle(sol, "oui.pkl", "").pickle()
# biorbd_model = biorbd.Model("../msk_models/arm26_cycling_pedal.bioMod")
# PickleAnimate("oui.pkl").animate(model=biorbd_model)


sol.animate(show_tracked_markers=True)
# sol.animate(viewer="pyorerun")
sol.graphs(show_bounds=False)
