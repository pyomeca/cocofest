"""
This example will do a 10 stimulation example with Ding's 2003 frequency model.
This ocp was build to produce an elbow motion from 5 to 120 degrees starting and ending with the arm at the vertical.
The stimulation frequency will be optimized between 10 and 100 Hz to satisfy the flexion and minimizing required
elbow torque control.
"""

import numpy as np

from cocofest import DingModelFrequencyWithFatigue, OcpFesMsk, FesMskModel

model = FesMskModel(
    name=None,
    biorbd_path="../msk_models/arm26_biceps.bioMod",
    muscles_model=[DingModelFrequencyWithFatigue(muscle_name="BIClong")],
    activate_force_length_relationship=True,
    activate_force_velocity_relationship=True,
    activate_residual_torque=True,
)

ocp = OcpFesMsk.prepare_ocp(
    model=model,
    stim_time=np.linspace(0, 1, 11)[:-1],
    final_time=1,
    pulse_event={"min": 0.01, "max": 0.1, "bimapping": True},
    objective={"minimize_residual_torque": True},
    msk_info={
        "with_residual_torque": True,
        "bound_type": "start_end",
        "bound_data": [[0, 5], [0, 120]],
    },
    n_threads=5,
    use_sx=False,
)

sol = ocp.solve()
sol.animate()
sol.graphs(show_bounds=False)
