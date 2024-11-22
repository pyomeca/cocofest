"""
This example will do a 10 stimulation example with Ding's 2007 frequency model.
This ocp was build to produce a elbow motion from 5 to 120 degrees.
The stimulation frequency will be optimized between 10 and 100 Hz and pulse duration between minimal sensitivity
threshold and 600us to satisfy the flexion and minimizing required elbow torque control.
External forces will be applied to the system to simulate a real-world scenario.
"""
import numpy as np
from bioptim import Solver
from cocofest import DingModelPulseWidthFrequencyWithFatigue, OcpFesMsk, FesMskModel, SolutionToPickle, PickleAnimate
import biorbd

model = FesMskModel(
    name=None,
    # biorbd_path="../msk_models/arm26_cycling_pedal_aligned.bioMod",
    biorbd_path="../msk_models/simplified_UL_Seth_pedal_aligned.bioMod",
    muscles_model=[
        DingModelPulseWidthFrequencyWithFatigue(muscle_name="DeltoideusClavicle_A", is_approximated=True),
        DingModelPulseWidthFrequencyWithFatigue(muscle_name="DeltoideusScapula_P", is_approximated=True),
        DingModelPulseWidthFrequencyWithFatigue(muscle_name="TRIlong", is_approximated=True),
        DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIC_long", is_approximated=True),
        DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIC_brevis", is_approximated=True),
    ],
    activate_force_length_relationship=True,
    activate_force_velocity_relationship=True,
    activate_residual_torque=True,
    external_force_set=None,  # External forces will be added
)

minimum_pulse_width = DingModelPulseWidthFrequencyWithFatigue().pd0
ocp = OcpFesMsk.prepare_ocp(
    model=model,
    stim_time=list(np.round(np.linspace(0, 1, 11), 3))[:-1],
    final_time=1,
    pulse_width={
        "min": minimum_pulse_width,
        "max": 0.0006,
        "bimapping": False,
    },
    msk_info={"with_residual_torque": True},
    objective={
        "cycling": {"x_center": 0.3, "y_center": 0, "radius": 0.1, "target": "marker"},
        "minimize_residual_torque": True,
        "minimize_muscle_force": True,
    },
    initial_guess_warm_start=False,
    external_forces={"Segment_application": "r_ulna_radius_hand", "torque": np.array([0, 0, -1]), "with_contact": True},
)

# sol = ocp.solve(solver=Solver.IPOPT(_max_iter=1))
sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=10000))
SolutionToPickle(sol, "hand_cycling_external_forces.pkl", "").pickle()
biorbd_model = biorbd.Model("../msk_models/simplified_UL_Seth_pedal_aligned.bioMod")
PickleAnimate("hand_cycling_external_forces.pkl").animate(model=biorbd_model)


sol.animate(show_tracked_markers=True)
# sol.animate(viewer="pyorerun")
sol.graphs(show_bounds=False)
