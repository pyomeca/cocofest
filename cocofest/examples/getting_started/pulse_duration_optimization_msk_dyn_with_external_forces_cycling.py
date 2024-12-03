"""
This example will do a 10 stimulation example with Ding's 2007 frequency model.
This ocp was build to produce a hand cycling motion.
The pulse duration will be optimized between minimal sensitivity threshold and 600us to satisfy the motion while
minimizing residual joints torques and produced muscular forces.
External forces will be applied to the system to simulate a real-world scenario with contacts at the pedal center.
"""
import numpy as np
from bioptim import Solver, ControlType
from cocofest import DingModelPulseWidthFrequencyWithFatigue, OcpFesMsk, FesMskModel, SolutionToPickle, PickleAnimate
import biorbd

model = FesMskModel(
    name=None,
    biorbd_path="../msk_models/simplified_UL_Seth_pedal_aligned_test.bioMod",
    muscles_model=[
        DingModelPulseWidthFrequencyWithFatigue(muscle_name="DeltoideusClavicle_A", is_approximated=False),
        DingModelPulseWidthFrequencyWithFatigue(muscle_name="DeltoideusScapula_P", is_approximated=False),
        DingModelPulseWidthFrequencyWithFatigue(muscle_name="TRIlong", is_approximated=False),
        DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIC_long", is_approximated=False),
        DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIC_brevis", is_approximated=False),
    ],
    activate_force_length_relationship=True,
    activate_force_velocity_relationship=True,
    activate_residual_torque=True,
    external_force_set=None,  # External forces will be added
)

minimum_pulse_width = DingModelPulseWidthFrequencyWithFatigue().pd0
ocp = OcpFesMsk.prepare_ocp_for_cycling(
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
        "cycling": {"x_center": 0.35, "y_center": 0, "radius": 0.1, "target": "marker"},
        "minimize_residual_torque": True,
        "minimize_muscle_force": True,
    },
    initial_guess_warm_start=False,
    external_forces={"Segment_application": "wheel", "torque": np.array([0, 0, 0]), "with_contact": True},
    control_type=ControlType.CONSTANT,
)

sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=10000))
SolutionToPickle(sol, "hand_cycling_external_forces1.pkl", "").pickle()
biorbd_model = biorbd.Model("../msk_models/simplified_UL_Seth_pedal_aligned_test.bioMod")
PickleAnimate("hand_cycling_external_forces1.pkl").animate(model=biorbd_model)

sol.animate(show_tracked_markers=True)
sol.graphs(show_bounds=True)
