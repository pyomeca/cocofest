import re
import pytest
import os

import numpy as np
from bioptim import (
    ObjectiveFcn,
    ObjectiveList,
    Solver,
    SolutionMerge,
)

from cocofest import (
    DingModelPulseWidthFrequencyWithFatigue,
    DingModelPulseIntensityFrequencyWithFatigue,
    OcpFesMsk,
    FesMskModel,
)

from cocofest.examples.msk_models import init as ocp_module

biomodel_folder = os.path.dirname(ocp_module.__file__)
biorbd_model_path = biomodel_folder + "/arm26_biceps_triceps.bioMod"


def test_pulse_width_multi_muscle_fes_dynamics():
    model = FesMskModel(
        biorbd_path=biorbd_model_path,
        muscles_model=[
            DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIClong", is_approximated=True),
            DingModelPulseWidthFrequencyWithFatigue(muscle_name="TRIlong", is_approximated=True),
        ],
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_residual_torque=True,
    )

    minimum_pulse_width = DingModelPulseWidthFrequencyWithFatigue().pd0
    ocp = OcpFesMsk.prepare_ocp(
        model=model,
        stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        final_time=1,
        pulse_width={
            "min": minimum_pulse_width,
            "max": 0.0006,
            "bimapping": False,
        },
        objective={"minimize_residual_torque": True},
        msk_info={
            "with_residual_torque": True,
            "bound_type": "start_end",
            "bound_data": [[0, 5], [0, 120]],
        },
        use_sx=False,
        n_threads=6,
    )

    sol = ocp.solve(Solver.IPOPT(_max_iter=1000))

    np.testing.assert_almost_equal(sol.cost, 864.1739, decimal=3)
    np.testing.assert_almost_equal(
        sol.parameters["pulse_width_BIClong"],
        np.array(
            [
                0.00013141,
                0.00060001,
                0.00060001,
                0.00060001,
                0.00060001,
                0.00060001,
                0.00055189,
                0.0001314,
                0.0001314,
                0.0001314,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        sol.parameters["pulse_width_TRIlong"],
        np.array(
            [
                0.0003657,
                0.0003657,
                0.0003657,
                0.0003657,
                0.0003657,
                0.0003657,
                0.0003657,
                0.0003657,
                0.0003657,
                0.0003657,
            ]
        ),
    )

    sol_states = sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])
    np.testing.assert_almost_equal(sol_states["q"][0][0], 0)
    np.testing.assert_almost_equal(sol_states["q"][0][-1], 0)
    np.testing.assert_almost_equal(sol_states["q"][1][0], 0.08722222222222223)
    np.testing.assert_almost_equal(sol_states["q"][1][-1], 2.0933333333333333)
    np.testing.assert_almost_equal(sol_states["F_BIClong"][0][-1], 14.604532949917337, decimal=4)
    np.testing.assert_almost_equal(sol_states["F_TRIlong"][0][-1], 3.9810503521165272, decimal=4)


def test_pulse_intensity_multi_muscle_fes_dynamics():
    model = FesMskModel(
        biorbd_path=biorbd_model_path,
        muscles_model=[
            DingModelPulseIntensityFrequencyWithFatigue(muscle_name="BIClong", is_approximated=True),
            DingModelPulseIntensityFrequencyWithFatigue(muscle_name="TRIlong", is_approximated=True),
        ],
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_residual_torque=True,
    )

    minimum_pulse_intensity = DingModelPulseIntensityFrequencyWithFatigue().min_pulse_intensity()

    ocp = OcpFesMsk.prepare_ocp(
        model=model,
        stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        final_time=1,
        pulse_intensity={
            "min": minimum_pulse_intensity,
            "max": 130,
            "bimapping": False,
        },
        objective={"minimize_residual_torque": True},
        msk_info={
            "with_residual_torque": True,
            "bound_type": "start_end",
            "bound_data": [[0, 5], [0, 120]],
        },
        use_sx=False,
        n_threads=6,
    )

    sol = ocp.solve(Solver.IPOPT(_max_iter=1000))

    np.testing.assert_almost_equal(sol.cost, 656.123, decimal=3)
    np.testing.assert_almost_equal(
        sol.parameters["pulse_intensity_BIClong"],
        np.array(
            [
                17.02855017,
                129.99999969,
                130.00000018,
                129.99999886,
                129.99999383,
                129.99997181,
                55.45255047,
                17.02854996,
                17.02854961,
                17.0285512,
            ]
        ),
        decimal=4,
    )
    np.testing.assert_almost_equal(
        sol.parameters["pulse_intensity_TRIlong"],
        np.array(
            [
                73.51427522,
                73.51427522,
                73.51427522,
                73.51427522,
                73.51427522,
                73.51427522,
                73.51427522,
                73.51427522,
                73.51427522,
                73.51427522,
            ]
        ),
        decimal=4,
    )

    sol_states = sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])
    np.testing.assert_almost_equal(sol_states["q"][0][0], 0)
    np.testing.assert_almost_equal(sol_states["q"][0][-1], 0)
    np.testing.assert_almost_equal(sol_states["q"][1][0], 0.08722, decimal=4)
    np.testing.assert_almost_equal(sol_states["q"][1][-1], 2.09333, decimal=4)
    np.testing.assert_almost_equal(sol_states["F_BIClong"][0][-1], 13.65894, decimal=4)
    np.testing.assert_almost_equal(sol_states["F_TRIlong"][0][-1], 4.832778, decimal=4)


def test_fes_models_inputs_sanity_check_errors():
    model = FesMskModel(
        biorbd_path=biorbd_model_path,
        muscles_model=[
            DingModelPulseIntensityFrequencyWithFatigue(muscle_name="BIClong"),
            DingModelPulseIntensityFrequencyWithFatigue(muscle_name="TRIlong"),
        ],
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_residual_torque=False,
    )

    with pytest.raises(
        ValueError,
        match=re.escape("bound_type should be a string and should be equal to start, end or start_end"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            model=model,
            stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            final_time=1,
            pulse_width={"min": 0.0003, "max": 0.0006},
            msk_info={
                "bound_type": "hello",
                "bound_data": [[0, 5], [0, 120]],
            },
        )

    with pytest.raises(
        TypeError,
        match=re.escape("bound_data should be a list"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            model=model,
            stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            final_time=1,
            pulse_width={"min": 0.0003, "max": 0.0006},
            msk_info={
                "bound_type": "start_end",
                "bound_data": "[[0, 5], [0, 120]]",
            },
        )

    with pytest.raises(
        ValueError,
        match=re.escape(f"bound_data should be a list of {2} elements"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            model=model,
            stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            final_time=1,
            pulse_width={"min": 0.0003, "max": 0.0006},
            msk_info={
                "bound_type": "start_end",
                "bound_data": [[0, 5, 7], [0, 120, 150]],
            },
        )

    with pytest.raises(
        TypeError,
        match=re.escape(f"bound_data should be a list of two list"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            model=model,
            stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            final_time=1,
            pulse_width={"min": 0.0003, "max": 0.0006},
            msk_info={
                "bound_type": "start_end",
                "bound_data": ["[0, 5]", [0, 120]],
            },
        )

    with pytest.raises(
        TypeError,
        match=re.escape(f"bound data index {1}: {5} and {'120'} should be an int or float"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            model=model,
            stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            final_time=1,
            pulse_width={"min": 0.0003, "max": 0.0006},
            msk_info={
                "bound_type": "start_end",
                "bound_data": [[0, 5], [0, "120"]],
            },
        )

    with pytest.raises(
        ValueError,
        match=re.escape(f"bound_data should be a list of {2} element"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            model=model,
            stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            final_time=1,
            pulse_width={"min": 0.0003, "max": 0.0006},
            msk_info={
                "bound_type": "start",
                "bound_data": [0, 5, 10],
            },
        )

    with pytest.raises(
        TypeError,
        match=re.escape(f"bound data index {1}: {'5'} should be an int or float"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            model=model,
            stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            final_time=1,
            pulse_width={"min": 0.0003, "max": 0.0006},
            msk_info={
                "bound_type": "end",
                "bound_data": [0, "5"],
            },
        )

    #
    # with pytest.raises(
    #     TypeError,
    #     match=re.escape(f"force_tracking index 1: {'[1, 2, 3]'} must be list type"),
    # ):
    #     ocp = OcpFesMsk.prepare_ocp(
    #         model=model,
    #         stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         n_shooting=100,
    #         final_time=1,
    #         pulse_width={"min": 0.0003, "max": 0.0006},
    #         msk_info={"bound_type": "start",
    #                   "bound_data": [0, 5],
    #                   },
    #         objective={"force_tracking": [np.array([1, 2, 3]), "[1, 2, 3]"]},
    #     )
    #
    # with pytest.raises(
    #     ValueError,
    #     match=re.escape(
    #         "force_tracking index 1 list must have the same size as the number of muscles in fes_muscle_models"
    #     ),
    # ):
    #     ocp = OcpFesMsk.prepare_ocp(
    #         model=model,
    #         stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         n_shooting=100,
    #         final_time=1,
    #         pulse_width={"min": 0.0003, "max": 0.0006},
    #         msk_info={"bound_type": "start",
    #                   "bound_data": [0, 5],
    #                   },
    #         objective={
    #             "force_tracking": [
    #                 np.array([1, 2, 3]),
    #                 [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
    #             ]
    #         },
    #     )
    #
    # with pytest.raises(
    #     ValueError,
    #     match=re.escape("force_tracking time and force argument must be the same length"),
    # ):
    #     ocp = OcpFesMsk.prepare_ocp(
    #         model=model,
    #         stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         n_shooting=100,
    #         final_time=1,
    #         pulse_width={"min": 0.0003, "max": 0.0006},
    #         msk_info={"bound_type": "start",
    #                   "bound_data": [0, 5],
    #                   },
    #         objective={"force_tracking": [np.array([1, 2, 3]), [[1, 2, 3], [1, 2]]]},
    #     )
    #
    # with pytest.raises(
    #     TypeError,
    #     match=re.escape(f"force_tracking: {'hello'} must be list type"),
    # ):
    #     ocp = OcpFesMsk.prepare_ocp(
    #         model=model,
    #         stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         n_shooting=100,
    #         final_time=1,
    #         pulse_width={"min": 0.0003, "max": 0.0006},
    #         msk_info={"bound_type": "start",
    #                   "bound_data": [0, 5],
    #                   },
    #         objective={"end_node_tracking": "hello"},
    #     )
    #
    # with pytest.raises(
    #     ValueError,
    #     match=re.escape("end_node_tracking list must have the same size as the number of muscles in fes_muscle_models"),
    # ):
    #     ocp = OcpFesMsk.prepare_ocp(
    #         model=model,
    #         stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         n_shooting=100,
    #         final_time=1,
    #         pulse_width={"min": 0.0003, "max": 0.0006},
    #         msk_info={"bound_type": "start",
    #                   "bound_data": [0, 5],
    #                   },
    #         objective={"end_node_tracking": [2, 3, 4]},
    #     )
    #
    # with pytest.raises(
    #     TypeError,
    #     match=re.escape(f"end_node_tracking index {1}: {'hello'} must be int or float type"),
    # ):
    #     ocp = OcpFesMsk.prepare_ocp(
    #         model=model,
    #         stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         n_shooting=100,
    #         final_time=1,
    #         pulse_width={"min": 0.0003, "max": 0.0006},
    #         msk_info={"bound_type": "start",
    #                   "bound_data": [0, 5],
    #                   },
    #         objective={"end_node_tracking": [2, "hello"]},
    #     )
    #
    # with pytest.raises(
    #     TypeError,
    #     match=re.escape("q_tracking should be a list of size 2"),
    # ):
    #     ocp = OcpFesMsk.prepare_ocp(
    #         model=model,
    #         stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         n_shooting=100,
    #         final_time=1,
    #         pulse_width={"min": 0.0003, "max": 0.0006},
    #         msk_info={"bound_type": "start",
    #                   "bound_data": [0, 5],
    #                   },
    #         objective={"q_tracking": "hello"},
    #     )
    #
    # with pytest.raises(
    #     ValueError,
    #     match=re.escape("q_tracking[0] should be a list"),
    # ):
    #     ocp = OcpFesMsk.prepare_ocp(
    #         model=model,
    #         stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         n_shooting=100,
    #         final_time=1,
    #         pulse_width={"min": 0.0003, "max": 0.0006},
    #         msk_info={"bound_type": "start",
    #                   "bound_data": [0, 5],
    #                   },
    #         objective={"q_tracking": ["hello", [1, 2, 3]]},
    #     )
    #
    # with pytest.raises(
    #     ValueError,
    #     match=re.escape("q_tracking[1] should have the same size as the number of generalized coordinates"),
    # ):
    #     ocp = OcpFesMsk.prepare_ocp(
    #         model=model,
    #         stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         n_shooting=100,
    #         final_time=1,
    #         pulse_width={"min": 0.0003, "max": 0.0006},
    #         msk_info={"bound_type": "start",
    #                   "bound_data": [0, 5],
    #                   },
    #         objective={"q_tracking": [[1, 2, 3], [1, 2, 3, 4]]},
    #     )
    #
    # with pytest.raises(
    #     ValueError,
    #     match=re.escape("q_tracking[0] and q_tracking[1] should have the same size"),
    # ):
    #     ocp = OcpFesMsk.prepare_ocp(
    #         model=model,
    #         stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         n_shooting=100,
    #         final_time=1,
    #         pulse_width={"min": 0.0003, "max": 0.0006},
    #         msk_info={"bound_type": "start",
    #                   "bound_data": [0, 5],
    #                   },
    #         objective={"q_tracking": [[1, 2, 3], [[1, 2, 3], [4, 5]]]},
    #     )
    #
    # with pytest.raises(
    #     TypeError,
    #     match=re.escape(f"{'with_residual_torque'} should be a boolean"),
    # ):
    #     ocp = OcpFesMsk.prepare_ocp(
    #         model=model,
    #         stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         n_shooting=100,
    #         final_time=1,
    #         pulse_width={"min": 0.0003, "max": 0.0006},
    #         msk_info={"bound_type": "start",
    #                   "bound_data": [0, 5],
    #                   "with_residual_torque": "hello"},
    #     )


#
# def test_fes_muscle_models_sanity_check_errors():
#     model = FesMskModel(biorbd_path=biorbd_model_path,
#                         muscles_model=[
#                             DingModelPulseIntensityFrequencyWithFatigue(muscle_name="BIClong"),
#                         ],
#                         activate_force_length_relationship=True,
#                         activate_force_velocity_relationship=True,
#                         activate_residual_torque=False,
#                         )
#     with pytest.raises(
#         ValueError,
#         match=re.escape(
#             f"The muscle {'TRIlong'} is not in the fes muscle model "
#             f"please add it into the fes_muscle_models list by providing the muscle_name ="
#             f" {'TRIlong'}"
#         ),
#     ):
#         ocp = OcpFesMsk.prepare_ocp(
#             model=model,
#             stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#             n_shooting=100,
#             final_time=1,
#             pulse_intensity={"min": 30, "max": 100},
#             msk_info={"with_residual_torque": False,
#                       "bound_type": "start",
#                       "bound_data": [0, 5],
#                       },
#         )
