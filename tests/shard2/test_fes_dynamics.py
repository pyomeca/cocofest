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
    DingModelPulseDurationFrequencyWithFatigue,
    DingModelIntensityFrequencyWithFatigue,
    OcpFesMsk,
)

from cocofest.examples.msk_models import init as ocp_module

biomodel_folder = os.path.dirname(ocp_module.__file__)
biorbd_model_path = biomodel_folder + "/arm26_biceps_triceps.bioMod"


def test_pulse_duration_multi_muscle_fes_dynamics():
    objective_functions = ObjectiveList()
    n_stim = 10
    for i in range(n_stim):
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="tau",
            weight=1,
            quadratic=True,
            phase=i,
        )

    minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0
    ocp = OcpFesMsk.prepare_ocp(
        biorbd_model_path=biorbd_model_path,
        bound_type="start_end",
        bound_data=[[0, 5], [0, 120]],
        fes_muscle_models=[
            DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
            DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
        ],
        n_stim=n_stim,
        n_shooting=10,
        final_time=1,
        pulse_duration={
            "min": minimum_pulse_duration,
            "max": 0.0006,
            "bimapping": False,
        },
        objective={"custom": objective_functions},
        with_residual_torque=True,
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        use_sx=False,
    )

    sol = ocp.solve(Solver.IPOPT(_max_iter=1000))

    np.testing.assert_almost_equal(sol.cost, 2.64645e-08)
    np.testing.assert_almost_equal(
        sol.parameters["pulse_duration_BIClong"],
        np.array(
            [
                0.00059643,
                0.00059543,
                0.00059475,
                0.00059325,
                0.0005899,
                0.00058005,
                0.00020145,
                0.0001394,
                0.00015499,
                0.00027584,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        sol.parameters["pulse_duration_TRIlong"],
        np.array(
            [
                0.0001635,
                0.00015817,
                0.00029651,
                0.00048692,
                0.00014608,
                0.0001396,
                0.00013858,
                0.00013775,
                0.000141,
                0.0001786,
            ]
        ),
    )

    sol_states = sol.decision_states(
        to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES]
    )
    np.testing.assert_almost_equal(sol_states["q"][0][0], 0)
    np.testing.assert_almost_equal(sol_states["q"][0][-1], 0)
    np.testing.assert_almost_equal(sol_states["q"][1][0], 0.08722222222222223)
    np.testing.assert_almost_equal(sol_states["q"][1][-1], 2.0933333333333333)
    np.testing.assert_almost_equal(
        sol_states["F_BIClong"][0][-1], 25.871305635093197, decimal=4
    )
    np.testing.assert_almost_equal(
        sol_states["F_TRIlong"][0][-1], 11.572282303524243, decimal=4
    )


def test_pulse_intensity_multi_muscle_fes_dynamics():
    n_stim = 10
    minimum_pulse_intensity = (
        DingModelIntensityFrequencyWithFatigue.min_pulse_intensity(
            DingModelIntensityFrequencyWithFatigue()
        )
    )
    track_forces = [
        np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
        [
            np.array([0, 10, 40, 90, 140, 80, 50, 10, 0, 0, 0]),
            np.array([0, 0, 0, 10, 40, 90, 140, 80, 50, 10, 0]),
        ],
    ]

    ocp = OcpFesMsk.prepare_ocp(
        biorbd_model_path=biorbd_model_path,
        bound_type="start",
        bound_data=[0, 5],
        fes_muscle_models=[
            DingModelIntensityFrequencyWithFatigue(muscle_name="BIClong"),
            DingModelIntensityFrequencyWithFatigue(muscle_name="TRIlong"),
        ],
        n_stim=n_stim,
        n_shooting=5,
        final_time=1,
        pulse_intensity={
            "min": minimum_pulse_intensity,
            "max": 130,
            "bimapping": False,
        },
        objective={"force_tracking": track_forces},
        with_residual_torque=False,
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        use_sx=False,
    )

    sol = ocp.solve(Solver.IPOPT(_max_iter=1000))

    np.testing.assert_almost_equal(sol.cost, 6666357.331403, decimal=6)
    np.testing.assert_almost_equal(
        sol.parameters["pulse_intensity_BIClong"],
        np.array(
            [
                130.00000125,
                130.00000126,
                130.00000128,
                130.00000128,
                130.00000121,
                50.86019267,
                17.02854916,
                17.02854916,
                17.02854917,
                17.0285492,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        sol.parameters["pulse_intensity_TRIlong"],
        np.array(
            [
                45.78076277,
                25.84044302,
                59.54858653,
                130.00000124,
                130.00000128,
                130.00000128,
                130.00000122,
                76.08547779,
                17.02854916,
                22.72956196,
            ]
        ),
    )

    sol_states = sol.decision_states(
        to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES]
    )
    np.testing.assert_almost_equal(sol_states["q"][0][0], 0)
    np.testing.assert_almost_equal(sol_states["q"][0][-1], -0.35378857156156907)
    np.testing.assert_almost_equal(sol_states["q"][1][0], 0.08722222222222223)
    np.testing.assert_almost_equal(sol_states["q"][1][-1], -7.097935277992009e-10)
    np.testing.assert_almost_equal(sol_states["F_BIClong"][0][-1], 47.408594, decimal=4)
    np.testing.assert_almost_equal(sol_states["F_TRIlong"][0][-1], 29.131785, decimal=4)


def test_fes_models_inputs_sanity_check_errors():
    with pytest.raises(
        TypeError,
        match=re.escape("biorbd_model_path should be a string"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=5,
            bound_type="start_end",
            bound_data=[[0, 5], [0, 120]],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "bound_type should be a string and should be equal to start, end or start_end"
        ),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="hello",
            bound_data=[[0, 5], [0, 120]],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
        )

    with pytest.raises(
        TypeError,
        match=re.escape("bound_data should be a list"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="start_end",
            bound_data="[[0, 5], [0, 120]]",
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
        )

    with pytest.raises(
        ValueError,
        match=re.escape(f"bound_data should be a list of {2} elements"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="start_end",
            bound_data=[[0, 5, 7], [0, 120, 150]],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
        )

    with pytest.raises(
        TypeError,
        match=re.escape(f"bound_data should be a list of two list"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="start_end",
            bound_data=["[0, 5]", [0, 120]],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
        )

    with pytest.raises(
        ValueError,
        match=re.escape(f"bound_data should be a list of {2} elements"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="start_end",
            bound_data=[[0, 5, 7], [0, 120, 150]],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
        )

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"bound data index {1}: {5} and {'120'} should be an int or float"
        ),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="start_end",
            bound_data=[[0, 5], [0, "120"]],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
        )

    with pytest.raises(
        ValueError,
        match=re.escape(f"bound_data should be a list of {2} element"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="start",
            bound_data=[0, 5, 10],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
        )

    with pytest.raises(
        TypeError,
        match=re.escape(f"bound data index {1}: {'5'} should be an int or float"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="end",
            bound_data=[0, "5"],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
        )

    with pytest.raises(
        TypeError,
        match="model must be a FesModel type",
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="start",
            bound_data=[0, 5],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                "DingModelPulseDurationFrequencyWithFatigue(muscle_name='TRIlong')",
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
        )

    with pytest.raises(
        TypeError,
        match=re.escape(f"force_tracking: {'hello'} must be list type"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="start",
            bound_data=[0, 5],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
            objective={"force_tracking": "hello"},
        )

    with pytest.raises(
        ValueError,
        match=re.escape("force_tracking must of size 2"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="end",
            bound_data=[0, 5],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
            objective={"force_tracking": ["hello"]},
        )

    with pytest.raises(
        TypeError,
        match=re.escape(f"force_tracking index 0: {'hello'} must be np.ndarray type"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="start",
            bound_data=[0, 5],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
            objective={"force_tracking": ["hello", [1, 2, 3]]},
        )

    with pytest.raises(
        TypeError,
        match=re.escape(f"force_tracking index 1: {'[1, 2, 3]'} must be list type"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="start",
            bound_data=[0, 5],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
            objective={"force_tracking": [np.array([1, 2, 3]), "[1, 2, 3]"]},
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "force_tracking index 1 list must have the same size as the number of muscles in fes_muscle_models"
        ),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="start",
            bound_data=[0, 5],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
            objective={
                "force_tracking": [
                    np.array([1, 2, 3]),
                    [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                ]
            },
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "force_tracking time and force argument must be the same length"
        ),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="start",
            bound_data=[0, 5],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
            objective={"force_tracking": [np.array([1, 2, 3]), [[1, 2, 3], [1, 2]]]},
        )

    with pytest.raises(
        TypeError,
        match=re.escape(f"force_tracking: {'hello'} must be list type"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="start",
            bound_data=[0, 5],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
            objective={"end_node_tracking": "hello"},
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "end_node_tracking list must have the same size as the number of muscles in fes_muscle_models"
        ),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="start",
            bound_data=[0, 5],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
            objective={"end_node_tracking": [2, 3, 4]},
        )

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"end_node_tracking index {1}: {'hello'} must be int or float type"
        ),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="start",
            bound_data=[0, 5],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
            objective={"end_node_tracking": [2, "hello"]},
        )

    with pytest.raises(
        TypeError,
        match=re.escape("q_tracking should be a list of size 2"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="start",
            bound_data=[0, 5],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
            objective={"q_tracking": "hello"},
        )

    with pytest.raises(
        ValueError,
        match=re.escape("q_tracking[0] should be a list"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="start",
            bound_data=[0, 5],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
            objective={"q_tracking": ["hello", [1, 2, 3]]},
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "q_tracking[1] should have the same size as the number of generalized coordinates"
        ),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="start",
            bound_data=[0, 5],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
            objective={"q_tracking": [[1, 2, 3], [1, 2, 3, 4]]},
        )

    with pytest.raises(
        ValueError,
        match=re.escape("q_tracking[0] and q_tracking[1] should have the same size"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="start",
            bound_data=[0, 5],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
            objective={"q_tracking": [[1, 2, 3], [[1, 2, 3], [4, 5]]]},
        )

    with pytest.raises(
        TypeError,
        match=re.escape(f"{'with_residual_torque'} should be a boolean"),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="start",
            bound_data=[0, 5],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong"),
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
            with_residual_torque="hello",
        )


def test_fes_muscle_models_sanity_check_errors():
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The muscle {'TRIlong'} is not in the fes muscle model "
            f"please add it into the fes_muscle_models list by providing the muscle_name ="
            f" {'TRIlong'}"
        ),
    ):
        ocp = OcpFesMsk.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            bound_type="start",
            bound_data=[0, 5],
            fes_muscle_models=[
                DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIClong")
            ],
            n_stim=1,
            n_shooting=10,
            final_time=1,
            pulse_duration={"min": 0.0003, "max": 0.0006},
        )
