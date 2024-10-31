"""
This example will do a 10 stimulation example with Ding's 2007 frequency model.
This ocp was build to produce a elbow motion from 5 to 120 degrees.
The stimulation frequency will be optimized between 10 and 100 Hz and pulse duration between minimal sensitivity
threshold and 600us to satisfy the flexion and minimizing required elbow torque control.
"""

from bioptim import Solver
from cocofest import DingModelPulseDurationFrequencyWithFatigue, OcpFesMsk, FesMskModel

model = FesMskModel(
    name=None,
    biorbd_path="../msk_models/arm26_biceps_1dof.bioMod",
    muscles_model=[DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong")],
    activate_force_length_relationship=True,
    activate_force_velocity_relationship=True,
    activate_residual_torque=True,
)

minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0
ocp = OcpFesMsk.prepare_ocp(
    model=model,
    stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
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
        "with_residual_torque": True,
    },
)

if __name__ == '__main__':
    sol = ocp.solve(Solver.IPOPT(show_online_optim=True))
    sol.animate()
    sol.graphs(show_bounds=False)

