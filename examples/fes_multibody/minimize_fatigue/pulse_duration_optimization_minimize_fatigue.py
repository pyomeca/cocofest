"""
This example will do a 10 stimulation example with Ding's 2007 pulse width model.
Those ocp were build to move the elbow from 0 to 90 degrees angle.
The stimulation frequency will be set to 10Hz and pulse width will be optimized to satisfy the motion and to minimize the overall muscle fatigue.
Intensity can be optimized from sensitivity threshold to 600us. No residual torque is allowed.
"""

import numpy as np

from bioptim import Node, ObjectiveFcn, ObjectiveList, Solver

from cocofest import DingModelPulseWidthFrequencyWithFatigue, OcpFesMsk, FesMskModel

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

minimum_pulse_width = DingModelPulseWidthFrequencyWithFatigue().pd0
model = FesMskModel(
    name=None,
    biorbd_path="../../model_msk/arm26_biceps_triceps.bioMod",
    muscles_model=[
        DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIClong"),
        DingModelPulseWidthFrequencyWithFatigue(muscle_name="TRIlong"),
    ],
    activate_force_length_relationship=True,
    activate_force_velocity_relationship=True,
    activate_residual_torque=False,
)

ocp = OcpFesMsk.prepare_ocp(
    model=model,
    stim_time=list(np.linspace(0, 1, 11))[:-1],
    final_time=1,
    pulse_width={
        "min": minimum_pulse_width,
        "max": 0.0006,
        "bimapping": False,
    },
    objective={"custom": objective_functions, "minimize_fatigue": True},
    msk_info={
        "bound_type": "start_end",
        "bound_data": [[0, 5], [0, 90]],
        "with_residual_torque": False,
    },
    n_threads=5,
)

sol = ocp.solve(Solver.IPOPT(_max_iter=3000))
sol.animate()
sol.graphs(show_bounds=False)
