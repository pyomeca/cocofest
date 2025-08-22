import os

import numpy as np
from bioptim import (
    Solver,
    SolutionMerge,
)

from cocofest import (
    DingModelPulseWidthFrequencyWithFatigue,
    DingModelPulseIntensityFrequencyWithFatigue,
    FesMskModel,
)

from examples.msk_models import init as model_path
from examples.fes_multibody.elbow_flexion import elbow_flexion_task as ocp_module

biomodel_folder = os.path.dirname(model_path.__file__)
biorbd_model_path = biomodel_folder + "/arm26_biceps_triceps.bioMod"


def test_pulse_width_multi_muscle_fes_dynamics():
    model = FesMskModel(
        biorbd_path=biorbd_model_path,
        muscles_model=[
            DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIClong", sum_stim_truncation=10),
            DingModelPulseWidthFrequencyWithFatigue(muscle_name="TRIlong", sum_stim_truncation=10),
        ],
        stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=True,
        activate_residual_torque=False,
    )

    ocp = ocp_module.prepare_ocp(
        model=model,
        final_time=1,
        max_bound=0.0006,
        msk_info={
            "with_residual_torque": False,
            "bound_type": "start_end",
            "bound_data": [[0, 5], [0, 120]],
        },
        minimize_force=True,
        minimize_fatigue=False,
    )

    sol = ocp.solve(Solver.IPOPT(_max_iter=1000))

    np.testing.assert_almost_equal(float(sol.cost), 1390.4256426188954)
    np.testing.assert_almost_equal(
        sol.decision_controls(to_merge=SolutionMerge.NODES)["last_pulse_width_BIClong"][0],
        np.array(
            [
                0.00017984,
                0.00025333,
                0.00035251,
                0.00060001,
                0.00060001,
                0.00060001,
                0.00018995,
                0.0001314,
                0.00027692,
                0.00060001,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        sol.decision_controls(to_merge=SolutionMerge.NODES)["last_pulse_width_TRIlong"][0],
        np.array(
            [
                0.00015681,
                0.0001314,
                0.0001314,
                0.0001314,
                0.0001314,
                0.0001314,
                0.0001314,
                0.0001314,
                0.0001314,
                0.00013141,
            ]
        ),
    )

    sol_states = sol.decision_states(to_merge=SolutionMerge.NODES)
    np.testing.assert_almost_equal(sol_states["q"][0][0], 0)
    np.testing.assert_almost_equal(sol_states["q"][0][-1], 0)
    np.testing.assert_almost_equal(sol_states["q"][1][0], 0.08722222222222223)
    np.testing.assert_almost_equal(sol_states["q"][1][-1], 2.0933333333333333)
    np.testing.assert_almost_equal(sol_states["F_BIClong"][0][-1], 45.49183822814395, decimal=4)
    np.testing.assert_almost_equal(sol_states["F_TRIlong"][0][-1], 8.064858860229864e-09, decimal=4)


def test_pulse_intensity_multi_muscle_fes_dynamics():
    model = FesMskModel(
        biorbd_path=biorbd_model_path,
        muscles_model=[
            DingModelPulseIntensityFrequencyWithFatigue(muscle_name="BIClong", sum_stim_truncation=10),
            DingModelPulseIntensityFrequencyWithFatigue(muscle_name="TRIlong", sum_stim_truncation=10),
        ],
        stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=True,
        activate_residual_torque=False,
    )

    ocp = ocp_module.prepare_ocp(
        model=model,
        final_time=1,
        max_bound=130,
        msk_info={
            "with_residual_torque": False,
            "bound_type": "start_end",
            "bound_data": [[0, 5], [0, 120]],
        },
        minimize_force=True,
        minimize_fatigue=False,
    )

    sol = ocp.solve(Solver.IPOPT(_max_iter=1000))

    np.testing.assert_almost_equal(float(sol.cost), 1904.7010473130113)
    np.testing.assert_almost_equal(
        sol.parameters["pulse_intensity_BIClong"],
        np.array(
            [
                31.347287,
                41.95028146,
                48.93343787,
                61.36861566,
                60.00744174,
                47.08608841,
                28.16105528,
                17.02855157,
                45.94329847,
                129.99999941,
            ]
        ),
        decimal=4,
    )
    np.testing.assert_almost_equal(
        sol.parameters["pulse_intensity_TRIlong"],
        np.array(
            [
                20.46766357,
                17.02854927,
                17.02854922,
                17.0285492,
                17.02854918,
                17.02854917,
                17.02854916,
                17.02854916,
                17.02854917,
                17.02854921,
            ]
        ),
        decimal=4,
    )

    sol_states = sol.decision_states(to_merge=SolutionMerge.NODES)
    np.testing.assert_almost_equal(sol_states["q"][0][0], 0)
    np.testing.assert_almost_equal(sol_states["q"][0][-1], 0)
    np.testing.assert_almost_equal(sol_states["q"][1][0], 0.08722, decimal=4)
    np.testing.assert_almost_equal(sol_states["q"][1][-1], 2.09333, decimal=4)
    np.testing.assert_almost_equal(sol_states["F_BIClong"][0][-1], 66.52666960616047, decimal=4)
    np.testing.assert_almost_equal(sol_states["F_TRIlong"][0][-1], 1.8134351446942753e-06, decimal=4)
