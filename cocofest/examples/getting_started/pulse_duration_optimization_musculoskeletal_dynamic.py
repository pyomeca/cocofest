"""
This example will do a 10 stimulation example with Ding's 2007 frequency model.
This ocp was build to produce a elbow motion from 5 to 120 degrees.
The stimulation frequency will be optimized between 10 and 100 Hz and pulse duration between minimal sensitivity
threshold and 600us to satisfy the flexion and minimizing required elbow torque control.
"""

from bioptim import (
    ObjectiveFcn,
    ObjectiveList,
    Solver,
)

from cocofest import DingModelPulseDurationFrequencyWithFatigue, OcpFesMsk


objective_functions = ObjectiveList()
n_stim = 10

objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, quadratic=True, phase=0)

minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0
ocp = OcpFesMsk.prepare_ocp(
    biorbd_model_path="../msk_models/arm26_biceps_1dof.bioMod",
    bound_type="start_end",
    bound_data=[[5], [120]],
    fes_muscle_models=[DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong")],
    n_stim=n_stim,
    n_shooting=100,
    final_time=1,
    stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    pulse_duration={
        "min": minimum_pulse_duration,
        "max": 0.0006,
        "bimapping": False,
    },
    objective={"custom": objective_functions},
    with_residual_torque=True,
    activate_force_length_relationship=True,
    activate_force_velocity_relationship=True,
)

sol = ocp.solve(Solver.IPOPT(_max_iter=2000))
sol.animate()
sol.graphs(show_bounds=False)
