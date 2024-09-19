"""
This example will do a 10 stimulation example with Ding's 2003 frequency model.
This ocp was build to produce a elbow motion from 5 to 120 degrees.
The stimulation frequency will be optimized between 10 and 100 Hz to satisfy the flexion and minimizing required
elbow torque control.
"""

import numpy as np

from cocofest import DingModelFrequencyWithFatigue, OcpFesMsk, FesMskModel


model = FesMskModel(
    name=None,
    biorbd_path="../msk_models/arm26_biceps_1dof.bioMod",
    muscles_model=[DingModelFrequencyWithFatigue(muscle_name="BIClong")],
    activate_force_length_relationship=True,
    activate_force_velocity_relationship=True,
)

ocp = OcpFesMsk.prepare_ocp(
    model=model,
    stim_time=np.linspace(0, 1, 11)[:-1],
    n_shooting=100,
    final_time=1,
    pulse_event={"min": 0.01, "max": 0.1, "bimapping": True},
    objective={"minimize_residual_torque": True},
    msk_info={
        "with_residual_torque": True,
        "bound_type": "start_end",
        "bound_data": [[5], [120]],
    },
    use_sx=True,
    n_threads=5,
)

sol = ocp.solve()
sol.animate()
sol.graphs(show_bounds=False)
