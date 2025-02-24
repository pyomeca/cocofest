"""
This example will do a 10 stimulation example with Ding's 2003 frequency model associated to Hmed's 2018 work.
This ocp was build to produce a elbow motion from 5 to 120 degrees.
The stimulation frequency will be optimized between 10 and 100 Hz and pulse intensity between minimal sensitivity
threshold and 130mA to satisfy the flexion and minimizing required elbow torque control.
"""

import numpy as np

from cocofest import DingModelPulseIntensityFrequencyWithFatigue, OcpFesMsk, FesMskModel


minimum_pulse_intensity = DingModelPulseIntensityFrequencyWithFatigue.min_pulse_intensity(
    DingModelPulseIntensityFrequencyWithFatigue()
)

model = FesMskModel(
    biorbd_path="../msk_models/arm26_biceps_1dof.bioMod",
    muscles_model=[DingModelPulseIntensityFrequencyWithFatigue(muscle_name="BIClong")],
    stim_time=np.linspace(0, 1, 34)[:-1].tolist(),
    activate_force_length_relationship=True,
    activate_force_velocity_relationship=True,
    activate_passive_force_relationship=True,
    activate_residual_torque=True,
)

ocp = OcpFesMsk.prepare_ocp(
    model=model,
    final_time=1,
    pulse_intensity={"min": minimum_pulse_intensity, "max": 130, "bimapping": False},
    objective={"minimize_residual_torque": True},
    msk_info={
        "with_residual_torque": True,
        "bound_type": "start_end",
        "bound_data": [[5], [120]],
    },
)

sol = ocp.solve()
sol.animate(viewer="pyorerun", n_frames=1000)
sol.graphs(show_bounds=False)
