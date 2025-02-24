"""
This example will do a 10 stimulation example with Ding's 2007 pulse width model.
Those ocp were build to move the elbow from 0 to 90 degrees angle.
The stimulation frequency will be set to 10Hz and pulse width will be optimized to satisfy the motion and to minimize the overall muscle fatigue.
Intensity can be optimized from sensitivity threshold to 600us. No residual torque is allowed.
"""

import numpy as np

from bioptim import Node, ObjectiveFcn, ObjectiveList

from cocofest import (
    OcpFesMsk,
    FesMskModel,
    DingModelPulseWidthFrequencyWithFatigue,
    DingModelPulseIntensityFrequencyWithFatigue,
    DingModelFrequencyWithFatigue,
)


def prepare_ocp(fes_model):
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

    if fes_model == DingModelPulseWidthFrequencyWithFatigue:
        pulse_width = {
            "min": DingModelPulseWidthFrequencyWithFatigue().pd0,
            "max": 0.0006,
            "bimapping": False,
        }
    else:
        pulse_width = None

    if fes_model == DingModelPulseIntensityFrequencyWithFatigue:
        pulse_intensity = {
            "min": DingModelPulseIntensityFrequencyWithFatigue.min_pulse_intensity(
                DingModelPulseIntensityFrequencyWithFatigue()
            ),
            "max": 130,
            "bimapping": False,
        }

    else:
        pulse_intensity = None

    bicep = fes_model(muscle_name="BIClong", stim_time=list(np.linspace(0, 1, 11))[:-1])
    tricep = fes_model(muscle_name="TRIlong", stim_time=list(np.linspace(0, 1, 11))[:-1])

    model = FesMskModel(
        name=None,
        biorbd_path="../model_msk/arm26_biceps_triceps.bioMod",
        muscles_model=[bicep, tricep],
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_residual_torque=False,
        stim_time=list(np.linspace(0, 1, 11))[:-1],
    )

    return OcpFesMsk.prepare_ocp(
        model=model,
        final_time=1,
        pulse_width=pulse_width,
        pulse_intensity=pulse_intensity,
        objective={"custom": objective_functions, "minimize_fatigue": True},
        msk_info={
            "bound_type": "start_end",
            "bound_data": [[0, 5], [0, 90]],
            "with_residual_torque": False,
        },
        n_threads=5,
    )


def main():
    # fes_model = DingModelPulseWidthFrequencyWithFatigue
    # fes_model = DingModelPulseIntensityFrequencyWithFatigue
    fes_model = DingModelFrequencyWithFatigue

    ocp = prepare_ocp(fes_model)
    sol = ocp.solve()

    # --- Show results from solution --- #
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
