"""
This example is used to compare the effect of the muscle force-length and force-velocity relationships
on the joint angle.
"""

import numpy as np

import matplotlib.pyplot as plt

from bioptim import SolutionMerge

from cocofest import DingModelPulseDurationFrequencyWithFatigue, OcpFesMsk, FesMskModel

minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0

sol_list = []
sol_time = []
activate_force_length_relationship = [False, True]

for i in range(2):

    model = FesMskModel(
        name=None,
        biorbd_path="../msk_models/arm26_biceps_1dof.bioMod",
        muscles_model=[DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong")],
        activate_force_length_relationship=activate_force_length_relationship[i],
        activate_force_velocity_relationship=activate_force_length_relationship[i],
    )

    ocp = OcpFesMsk.prepare_ocp(
        model=model,
        stim_time=np.linspace(0, 1, 11)[:-1],
        n_shooting=100,
        final_time=1,
        pulse_duration={"fixed": 0.00025},
        msk_info={
            "bound_type": "start",
            "bound_data": [0],
            "with_residual_torque": False,
        },
        use_sx=False,
    )
    sol = ocp.solve()
    sol_list.append(sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES]))
    time = np.concatenate(
        sol.stepwise_time(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES], duplicated_times=False),
        axis=0,
    )
    index = 0
    for j in range(len(sol.ocp.nlp) - 1):
        index = index + 1 + sol.ocp.nlp[j].ns
        time = np.insert(time, index, time[index - 1])

    sol_time.append(time)

plt.plot(sol_time[0], np.degrees(sol_list[0]["q"][0]), label="without relationships")
plt.plot(sol_time[1], np.degrees(sol_list[1]["q"][0]), label="with relationships")

plt.xlabel("Time (s)")
plt.ylabel("Angle (°)")
plt.legend()
plt.show()

joint_overestimation = np.degrees(sol_list[0]["q"][0][-1]) - np.degrees(sol_list[1]["q"][0][-1])
print(f"Joint overestimation: {joint_overestimation} degrees")
