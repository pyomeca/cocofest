"""
This example will do a 10 stimulation example with Ding's 2003 frequency model associated to Hmed's 2018 work.
This ocp was build to produce a elbow motion from 5 to 120 degrees.
The stimulation frequency will be optimized between 10 and 100 Hz and pulse intensity between minimal sensitivity
threshold and 130mA to satisfy the flexion and minimizing required elbow torque control.
"""

from cocofest import DingModelPulseIntensityFrequencyWithFatigue, OcpFesMsk, FesMskModel


minimum_pulse_intensity = DingModelPulseIntensityFrequencyWithFatigue.min_pulse_intensity(
    DingModelPulseIntensityFrequencyWithFatigue()
)

model = FesMskModel(
    biorbd_path="../msk_models/arm26_biceps_1dof.bioMod",
    muscles_model=[DingModelPulseIntensityFrequencyWithFatigue(muscle_name="BIClong")],
    activate_force_length_relationship=True,
    activate_force_velocity_relationship=True,
    activate_residual_torque=True,
)

ocp = OcpFesMsk.prepare_ocp(
    model=model,
    stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
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
sol.animate()
sol.graphs(show_bounds=False)
