"""
This example will do a 10 stimulation example with Ding's 2003 frequency model associated to Hmed's 2018 intensity work.
This ocp was build to maintain an elbow angle of 90 degrees.
The stimulation frequency will be optimized between 1 and 10 Hz as well as the pulse intensity between minimal
sensitivity threshold and 130mA to satisfy the maintained elbow. No residual torque is allowed.
"""

import numpy as np

from bioptim import Node, ObjectiveFcn, ObjectiveList

from cocofest import DingModelIntensityFrequencyWithFatigue, OcpFesMsk, FesMskModel

n_shooting = 100
objective_functions = ObjectiveList()

objective_functions.add(
    ObjectiveFcn.Mayer.MINIMIZE_STATE,
    key="q",
    index=[0, 1],
    node=Node.ALL,
    target=np.array([[0, 1.57]] * (n_shooting + 1)).T,
    weight=10,
    quadratic=True,
    phase=0,
)
objective_functions.add(
    ObjectiveFcn.Mayer.MINIMIZE_STATE,
    key="qdot",
    index=[0, 1],
    node=Node.ALL,
    target=np.array([[0, 0]] * (n_shooting + 1)).T,
    weight=10,
    quadratic=True,
    phase=0,
)

minimum_pulse_intensity = DingModelIntensityFrequencyWithFatigue.min_pulse_intensity(
    DingModelIntensityFrequencyWithFatigue()
)
model = FesMskModel(
    name=None,
    biorbd_path="../msk_models/arm26.bioMod",
    muscles_model=[
        DingModelIntensityFrequencyWithFatigue(muscle_name="BIClong"),
        DingModelIntensityFrequencyWithFatigue(muscle_name="BICshort"),
        DingModelIntensityFrequencyWithFatigue(muscle_name="TRIlong"),
        DingModelIntensityFrequencyWithFatigue(muscle_name="TRIlat"),
        DingModelIntensityFrequencyWithFatigue(muscle_name="TRImed"),
        DingModelIntensityFrequencyWithFatigue(muscle_name="BRA"),
    ],
    activate_force_length_relationship=True,
    activate_force_velocity_relationship=True,
    activate_residual_torque=False,
)

ocp = OcpFesMsk.prepare_ocp(
    model=model,
    stim_time=np.linspace(0, 1, 11)[:-1],
    final_time=1,
    pulse_event={"min": 0.1, "max": 1, "bimapping": True},
    pulse_intensity={
        "min": minimum_pulse_intensity,
        "max": 130,
        "bimapping": False,
    },
    objective={"custom": objective_functions},
    msk_info={
        "with_residual_torque": False,
        "bound_type": "start",
        "bound_data": [0, 90],
    },
    use_sx=False,
    n_threads=5,
)

sol = ocp.solve()
sol.animate()
sol.graphs(show_bounds=False)
