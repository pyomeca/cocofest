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
biorbd_model_path = biomodel_folder + "/Arm26/arm26_biceps_triceps.bioMod"


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

    np.testing.assert_almost_equal(float(sol.cost), 0.02348273125089339)
    np.testing.assert_almost_equal(
        sol.decision_controls(to_merge=SolutionMerge.NODES)["last_pulse_width_BIClong"][0],
        np.array(
            [
                0.00020004,
                0.00026119,
                0.00040928,
                0.00059994,
                0.00059996,
                0.00058914,
                0.00013638,
                0.00013516,
                0.00059938,
                0.00059996,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        sol.decision_controls(to_merge=SolutionMerge.NODES)["last_pulse_width_TRIlong"][0],
        np.array(
            [
                0.0001582,
                0.0001314,
                0.0001314,
                0.0001314,
                0.0001314,
                0.0001314,
                0.00013141,
                0.00013141,
                0.00013141,
                0.00013142,
            ]
        ),
    )

    sol_states = sol.decision_states(to_merge=SolutionMerge.NODES)
    np.testing.assert_almost_equal(sol_states["q"][0][0], 0)
    np.testing.assert_almost_equal(sol_states["q"][0][-1], 0)
    np.testing.assert_almost_equal(sol_states["q"][1][0], 0.08722222222222223)
    np.testing.assert_almost_equal(sol_states["q"][1][-1], 2.0933333333333333)
    np.testing.assert_almost_equal(sol_states["F_BIClong"][0][-1], 52.742804677557395, decimal=4)
    np.testing.assert_almost_equal(sol_states["F_TRIlong"][0][-1], 0.004764212593426825, decimal=4)


# Converge to optimal solution found locally but not on GitHub Actions
# def test_pulse_intensity_multi_muscle_fes_dynamics():
#     model = FesMskModel(
#         biorbd_path=biorbd_model_path,
#         muscles_model=[
#             DingModelPulseIntensityFrequencyWithFatigue(muscle_name="BIClong", sum_stim_truncation=10),
#             DingModelPulseIntensityFrequencyWithFatigue(muscle_name="TRIlong", sum_stim_truncation=10),
#         ],
#         stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#         activate_force_length_relationship=True,
#         activate_force_velocity_relationship=True,
#         activate_passive_force_relationship=True,
#         activate_residual_torque=False,
#     )
#
#     ocp = ocp_module.prepare_ocp(
#         model=model,
#         final_time=1,
#         max_bound=130,
#         msk_info={
#             "with_residual_torque": False,
#             "bound_type": "start_end",
#             "bound_data": [[0, 5], [0, 120]],
#         },
#         minimize_force=True,
#         minimize_fatigue=False,
#     )
#
#     sol = ocp.solve(Solver.IPOPT(_max_iter=1000))
#
#     np.testing.assert_almost_equal(float(sol.cost), 0.01955805559046117)
#     np.testing.assert_almost_equal(
#         sol.parameters["pulse_intensity_BIClong"],
#         np.array(
#             [
#                 34.12442806,
#                 42.49976813,
#                 50.80243479,
#                 62.85752027,
#                 58.91462545,
#                 44.5799096,
#                 24.7298089,
#                 17.51555338,
#                 59.54395547,
#                 129.86953413,
#             ]
#         ),
#         decimal=4,
#     )
#     np.testing.assert_almost_equal(
#         sol.parameters["pulse_intensity_TRIlong"],
#         np.array(
#             [
#                 21.26538345,
#                 17.04180387,
#                 17.0368087,
#                 17.03394909,
#                 17.03228404,
#                 17.03109576,
#                 17.03046418,
#                 17.03046743,
#                 17.03165201,
#                 17.04247682,
#             ]
#         ),
#         decimal=4,
#     )
#
#     sol_states = sol.decision_states(to_merge=SolutionMerge.NODES)
#     np.testing.assert_almost_equal(sol_states["q"][0][0], 0)
#     np.testing.assert_almost_equal(sol_states["q"][0][-1], 0)
#     np.testing.assert_almost_equal(sol_states["q"][1][0], 0.08722, decimal=4)
#     np.testing.assert_almost_equal(sol_states["q"][1][-1], 2.09333, decimal=4)
#     np.testing.assert_almost_equal(sol_states["F_BIClong"][0][-1], 74.55962836277212, decimal=4)
#     np.testing.assert_almost_equal(sol_states["F_TRIlong"][0][-1], 0.011387839148288692, decimal=4)
