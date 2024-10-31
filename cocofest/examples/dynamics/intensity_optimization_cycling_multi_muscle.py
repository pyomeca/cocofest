"""
This example will do a 10 stimulation example with Ding's 2003 frequency model associated to Hmed's 2018 intensity work.
This ocp was build to maintain an elbow angle of 90 degrees.
The stimulation frequency will be optimized between 1 and 10 Hz as well as the pulse intensity between minimal
sensitivity threshold and 130mA to satisfy the maintained elbow. No residual torque is allowed.
"""

import numpy as np

from cocofest import DingModelIntensityFrequency, OcpFesMsk, FesMskModel


track_q = [
    np.array([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]),
    [
        np.array([1.1339, 0.9943, 0.7676, 0.5757, 0.4536, 0.6280, 1.0292, 1.0990, 1.1339]),
        np.array([0.6629, 0.7676, 1.0641, 1.3781, 1.4653, 1.3781, 0.9594, 0.8373, 0.6629]),
    ],
]

minimum_pulse_intensity = DingModelIntensityFrequency.min_pulse_intensity(DingModelIntensityFrequency())

model = FesMskModel(
    name=None,
    biorbd_path="../msk_models/arm26.bioMod",
    muscles_model=[
        DingModelIntensityFrequency(muscle_name="BIClong"),
        DingModelIntensityFrequency(muscle_name="BICshort"),
        DingModelIntensityFrequency(muscle_name="TRIlong"),
        DingModelIntensityFrequency(muscle_name="TRIlat"),
        DingModelIntensityFrequency(muscle_name="TRImed"),
        DingModelIntensityFrequency(muscle_name="BRA"),
    ],
    activate_force_length_relationship=True,
    activate_force_velocity_relationship=True,
    activate_residual_torque=True,
)

ocp = OcpFesMsk.prepare_ocp(
    model=model,
    stim_time=list(np.round(np.linspace(0, 1, 31), 3))[:-1],
    final_time=1,
    pulse_event={"min": 0.05, "max": 1, "bimapping": True},
    pulse_intensity={
        "min": minimum_pulse_intensity,
        "max": 130,
        "bimapping": False,
    },
    msk_info={
        "with_residual_torque": True,
        "bound_type": "start_end",
        "bound_data": [[65, 38], [65, 38]],
    },
    objective={"minimize_residual_torque": True, "q_tracking": track_q},
    use_sx=False,
    n_threads=5,
)

sol = ocp.solve()
sol.animate()
sol.graphs(show_bounds=False)
