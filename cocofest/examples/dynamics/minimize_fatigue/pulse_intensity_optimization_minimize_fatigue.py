"""
This example will do a 10 stimulation example with Ding's 2003 frequency model associated to Hmed's 2018 intensity work.
Those ocp were build to move the elbow from 0 to 90 degrees angle.
The stimulation frequency will be set to 10Hz and pulse intensity will be optimized to satisfy the motion and to minimize the overall muscle fatigue.
Intensity can be optimized from sensitivity threshold to 130mA. No residual torque is allowed.
"""

import numpy as np

from bioptim import Node, ObjectiveFcn, ObjectiveList, Solver

from cocofest import DingModelPulseIntensityFrequencyWithFatigue, OcpFesMsk, FesMskModel


objective_functions = ObjectiveList()
objective_functions.add(
    ObjectiveFcn.Mayer.MINIMIZE_STATE,
    key="qdot",
    index=[0, 1],
    node=Node.END,
    target=np.array([[0, 0]]).T,
    weight=100,
    quadratic=True,
    phase=0,
)

minimum_pulse_intensity = DingModelPulseIntensityFrequencyWithFatigue.min_pulse_intensity(
    DingModelPulseIntensityFrequencyWithFatigue()
)
model = FesMskModel(
    name=None,
    biorbd_path="../../msk_models/arm26_biceps_triceps.bioMod",
    muscles_model=[
        DingModelPulseIntensityFrequencyWithFatigue(muscle_name="BIClong"),
        DingModelPulseIntensityFrequencyWithFatigue(muscle_name="TRIlong"),
    ],
    activate_force_length_relationship=True,
    activate_force_velocity_relationship=True,
    activate_residual_torque=False,
)

ocp = OcpFesMsk.prepare_ocp(
    model=model,
    stim_time=list(np.linspace(0, 1, 11))[:-1],
    final_time=1,
    pulse_intensity={
        "min": minimum_pulse_intensity,
        "max": 130,
        "bimapping": False,
    },
    objective={"custom": objective_functions, "minimize_fatigue": True},
    msk_info={
        "with_residual_torque": False,
        "bound_type": "start_end",
        "bound_data": [[0, 5], [0, 90]],
    },
    n_threads=5,
)

sol = ocp.solve(Solver.IPOPT(_max_iter=1000))
sol.animate()
sol.graphs(show_bounds=False)
