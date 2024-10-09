"""
This example will do a 10 stimulation example with Ding's 2007 frequency model.
This ocp was build to produce a elbow motion from 5 to 120 degrees.
The stimulation frequency will be optimized between 10 and 100 Hz and pulse duration between minimal sensitivity
threshold and 600us to satisfy the flexion and minimizing required elbow torque control.
External forces will be applied to the system to simulate a real-world scenario.
"""
import numpy as np

from cocofest import DingModelPulseDurationFrequencyWithFatigue, OcpFesMsk, FesMskModel

model = FesMskModel(
    name=None,
    biorbd_path="../msk_models/arm26_biceps_1dof.bioMod",
    muscles_model=[DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong")],
    activate_force_length_relationship=True,
    activate_force_velocity_relationship=True,
    activate_residual_torque=True,
    segments_to_apply_external_forces=["r_ulna_radius_hand"],
)

minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0
ocp = OcpFesMsk.prepare_ocp(
    model=model,
    stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    n_shooting=100,
    final_time=1,
    pulse_duration={
        "min": minimum_pulse_duration,
        "max": 0.0006,
        "bimapping": False,
    },
    objective={"minimize_residual_torque": True},
    msk_info={
        "bound_type": "start_end",
        "bound_data": [[5], [120]],
        "with_residual_torque": True},
    external_forces={"Global": True, "Segment_application": "r_ulna_radius_hand", "torque": np.array([0, 0, 0]), "force": np.array([0, 10, 0]), "point_of_application": np.array([0, 0, 0])},
)

sol = ocp.solve()
sol.animate()
sol.graphs(show_bounds=False)
