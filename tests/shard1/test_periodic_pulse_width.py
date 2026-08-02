import importlib.util
import numpy as np
import pytest
import re
import warnings
from casadi import Function, SX, collocation_coeff, collocation_points
from pathlib import Path
from types import SimpleNamespace
from bioptim import (
    Bounds,
    BoundsList,
    InitialGuessList,
    InterpolationType,
    Node,
    OdeSolver,
    SolutionMerge,
    Solver,
)

import examples.fes_multibody.cycling.cycling_pulse_width_mhe_acados_periodic as periodic_example
import examples.fes_multibody.cycling.cycling_fes_solver_comparison as comparison_example
import examples.fes_multibody.cycling.cycling_pulse_width_mhe as mhe_example
from cocofest.optimization.receding_horizon_initial_guess import (
    audit_initial_guess,
    copy_container_values,
)
from cocofest.optimization.fes_nmpc_multibody import (
    CompactNmpcSolution,
    FesNmpcMsk,
)
from cocofest.optimization.fes_ocp_multibody import OcpFesMsk
from cocofest.models.ding2007.ding2007_with_fatigue_periodic import (
    DingModelPulseWidthFrequencyWithFatiguePeriodic,
)
from cocofest.models.ding2007.ding2007_with_fatigue_periodic_node import (
    DingModelPulseWidthFrequencyWithFatiguePeriodicNode,
)
from examples.fes_multibody.cycling.cycling_pulse_width_mhe import MyCyclicNMPC
from examples.fes_multibody.cycling.cycling_pulse_width_mhe_acados_periodic import (
    _copy_refinement_initial_guesses,
    pulse_width_initial_guess_summary,
    set_acados_unsafe_option,
    tile_one_cycle_solution_to_periodic_nmpc,
)

_benchmark_report_spec = importlib.util.spec_from_file_location(
    "cycling_benchmark_report",
    Path(__file__).resolve().parents[2]
    / ".github"
    / "scripts"
    / "summarize_cycling_benchmark.py",
)
benchmark_report = importlib.util.module_from_spec(_benchmark_report_spec)
_benchmark_report_spec.loader.exec_module(benchmark_report)


def _muscle_model():
    return DingModelPulseWidthFrequencyWithFatiguePeriodic(
        muscle_name="Biceps",
        stim_time=[0.0, 1 / 30],
    )


def test_full_contact_stabilization_constrains_every_shooting_node(
    monkeypatch,
):
    captured = []

    class FakeConstraintList:
        def add(self, constraint, **kwargs):
            captured.append((constraint, kwargs))

    monkeypatch.setattr(mhe_example, "ConstraintList", FakeConstraintList)
    model = SimpleNamespace(marker_index=lambda name: 7)

    mhe_example.set_constraints(
        model,
        enforce_start_constraints=False,
        enforce_contact_constraints_all_nodes=True,
    )

    assert len(captured) == 2
    assert all(kwargs["node"] == Node.ALL for _, kwargs in captured)
    assert captured[0][1]["marker_index"] == 7
    assert captured[1][1]["first_marker"] == "wheel_center"
    assert captured[1][1]["second_marker"] == "global_wheel_center"


def test_full_contact_terminal_stabilization_closes_only_rho_seam(monkeypatch):
    captured = []

    class FakeConstraintList:
        def add(self, constraint, **kwargs):
            captured.append((constraint, kwargs))

    monkeypatch.setattr(mhe_example, "ConstraintList", FakeConstraintList)
    model = SimpleNamespace(marker_index=lambda name: 7)

    mhe_example.set_constraints(
        model,
        enforce_start_constraints=True,
        enforce_contact_constraints_terminal=True,
    )

    assert [kwargs["node"] for _, kwargs in captured] == [
        Node.START,
        Node.START,
        Node.END,
        Node.END,
    ]
    assert captured[2][1]["marker_index"] == 7
    assert captured[3][1]["first_marker"] == "wheel_center"
    assert captured[3][1]["second_marker"] == "global_wheel_center"


def test_full_contact_terminal_position_avoids_velocity_redundancy(
    monkeypatch,
):
    captured = []

    class FakeConstraintList:
        def add(self, constraint, **kwargs):
            captured.append((constraint, kwargs))

    monkeypatch.setattr(mhe_example, "ConstraintList", FakeConstraintList)
    model = SimpleNamespace(marker_index=lambda name: 7)

    mhe_example.set_constraints(
        model,
        enforce_start_constraints=True,
        enforce_contact_position_terminal=True,
    )

    assert [kwargs["node"] for _, kwargs in captured] == [
        Node.START,
        Node.START,
        Node.END,
    ]
    assert captured[-1][1]["first_marker"] == "wheel_center"


def test_full_contact_position_stabilization_avoids_velocity_redundancy(
    monkeypatch,
):
    captured = []

    class FakeConstraintList:
        def add(self, constraint, **kwargs):
            captured.append((constraint, kwargs))

    monkeypatch.setattr(mhe_example, "ConstraintList", FakeConstraintList)
    model = SimpleNamespace(marker_index=lambda name: 7)

    mhe_example.set_constraints(
        model,
        enforce_start_constraints=False,
        enforce_contact_position_all_nodes=True,
        contact_position_tolerance_m=2e-5,
    )

    assert len(captured) == 1
    assert captured[0][1]["node"] == Node.ALL
    assert captured[0][1]["min_bound"] == -2e-5
    assert captured[0][1]["max_bound"] == 2e-5


def test_full_cadence_constraint_covers_collocation_stages_and_terminal(
    monkeypatch,
):
    captured = []

    class FakeConstraintList:
        def add(self, constraint, **kwargs):
            captured.append((constraint, kwargs))

    monkeypatch.setattr(mhe_example, "ConstraintList", FakeConstraintList)
    model = SimpleNamespace(marker_index=lambda name: 7)

    mhe_example.set_constraints(
        model,
        enforce_start_constraints=False,
        enforce_physical_crank_velocity_bounds=True,
    )

    assert len(captured) == 2
    assert captured[0][0] is (
        mhe_example.physical_crank_velocity_all_collocation_points_constraint
    )
    assert captured[0][1]["node"] == Node.ALL_SHOOTING
    assert captured[1][0] is mhe_example.physical_crank_velocity_constraint
    assert captured[1][1]["node"] == Node.END


def test_full_transfer_contact_projection_preserves_bound_crank_states():
    class Kinematics:
        @staticmethod
        def q(theta):
            return np.array([2.0 * theta, 3.0 * theta, theta])

        @staticmethod
        def tangent(_theta):
            return np.array([2.0, 3.0, 1.0])

        @staticmethod
        def project_generalized_trajectory(q, qdot):
            del qdot
            return np.array([[q[2, 0]]]), None, {}

    q = np.array([[0.0, 1.0], [0.0, 1.0], [4.0, 5.0]])
    qdot = np.array([[0.0, 1.0], [0.0, 1.0], [-6.0, -7.0]])
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={
                    "q": SimpleNamespace(init=q),
                    "qdot": SimpleNamespace(init=qdot),
                }
            )
        ],
        wheel_state_index=2,
        _cocofest_mechanical_equivalence_dynamics=SimpleNamespace(
            kinematics=Kinematics()
        ),
    )

    summary = mhe_example.project_full_first_node_initial_guess_to_contact(
        nmpc, project_velocity=True
    )

    assert summary["applied"] is True
    np.testing.assert_allclose(nmpc.nlp[0].x_init["q"].init[:, 0], [8.0, 12.0, 4.0])
    np.testing.assert_allclose(
        nmpc.nlp[0].x_init["qdot"].init[:, 0], [-12.0, -18.0, -6.0]
    )
    # Only the first node is projected; the shifted tail remains untouched.
    np.testing.assert_allclose(nmpc.nlp[0].x_init["q"].init[:, 1], [1.0, 1.0, 5.0])
    np.testing.assert_allclose(nmpc.nlp[0].x_init["qdot"].init[:, 1], [1.0, 1.0, -7.0])

    nmpc.nlp[0].x_init["q"].init[:, :] = np.array([[0.0, 1.0], [0.0, 1.0], [4.0, 5.0]])
    nmpc.nlp[0].x_init["qdot"].init[:, :] = np.array(
        [[0.0, 1.0], [0.0, 1.0], [-6.0, -7.0]]
    )
    summary = mhe_example.project_full_first_node_initial_guess_to_contact(nmpc)
    assert summary["mode"] == "position"
    np.testing.assert_allclose(nmpc.nlp[0].x_init["qdot"].init[:, 0], [0.0, 0.0, -6.0])

    summary = mhe_example.project_full_first_node_initial_guess_to_contact(
        nmpc, node=-1
    )
    assert summary["node"] == 1
    np.testing.assert_allclose(nmpc.nlp[0].x_init["q"].init[:, 1], [10.0, 15.0, 5.0])
    np.testing.assert_allclose(nmpc.nlp[0].x_init["qdot"].init[:, 1], [1.0, 1.0, -7.0])


def test_updating_full_model_preserves_every_force_relationship(monkeypatch):
    captured = {}

    def fake_fes_model(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(mhe_example, "FesMskModel", fake_fes_model)
    muscle = SimpleNamespace(stim_time=[0.0], previous_stim=None)
    model = SimpleNamespace(
        name="cycling",
        biorbd_path="cycling.bioMod",
        muscles_dynamics_model=[muscle],
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=True,
        activate_residual_torque=False,
        constant_external_torque=np.array([0.2]),
    )

    updated = mhe_example.updating_model(
        model,
        external_force_set="external",
        parameters="parameters",
    )

    assert updated.activate_force_length_relationship is True
    assert updated.activate_force_velocity_relationship is True
    assert updated.activate_passive_force_relationship is True
    assert captured["constant_external_torque"] is model.constant_external_torque


def test_full_qdot_envelope_contains_every_lifted_reduced_velocity():
    class Kinematics:
        theta_origin = 0.0
        direction = 1.0

        @staticmethod
        def tangent(theta):
            return np.array(
                [
                    0.4 * np.sin(theta),
                    0.5 * np.cos(theta),
                    1.0 + 0.45 * np.cos(theta),
                ]
            )

    reduced = SimpleNamespace(kinematics=Kinematics())
    (
        lower,
        upper,
    ) = mhe_example.full_coordinate_qdot_bounds_from_reduced_profile(
        reduced,
        physical_crank_velocity_target=-2.0 * np.pi,
        physical_crank_velocity_margin=3.0,
        sample_count=721,
        relative_padding=0.01,
    )

    for theta in np.linspace(0.0, 2.0 * np.pi, 721):
        tangent = reduced.kinematics.tangent(theta)
        for omega in (-2.0 * np.pi - 3.0, -2.0 * np.pi + 3.0):
            lifted_qdot = tangent * omega
            assert np.all(lifted_qdot >= lower)
            assert np.all(lifted_qdot <= upper)
    assert lower[2] < -13.4
    assert upper[2] > -1.9


@pytest.mark.parametrize(
    "overrides, message",
    (
        (
            {"nlp_solver_type": "SQP_WITH_FEASIBLE_QP", "ext_qp_res": True},
            "does not support",
        ),
        (
            {"search_direction_mode": "FEASIBILITY_QP"},
            "declares FEASIBILITY_QP but does not implement",
        ),
        (
            {
                "with_anderson_acceleration": True,
                "globalization": "FUNNEL_L1PEN_LINESEARCH",
            },
            "requires FIXED_STEP",
        ),
        (
            {"byrd_omojokon_slack_relaxation_factor": 0.99},
            "must be finite and >= 1",
        ),
    ),
)
def test_acados_v055_rejects_unsafe_option_combinations(overrides, message):
    options = {
        "nlp_solver_type": "SQP",
        "search_direction_mode": "NOMINAL_QP",
        "globalization": "FIXED_STEP",
        "ext_qp_res": False,
        "code_reuse_tolerance": 1e-12,
        "with_anderson_acceleration": False,
        "anderson_activation_threshold": 0.1,
        "byrd_omojokon_slack_relaxation_factor": 1.00001,
    }
    options.update(overrides)

    with pytest.raises(ValueError, match=message):
        periodic_example.validate_acados_v055_options(**options)


def _dynamics(model, pulse_width, numerical_timeseries):
    return np.asarray(
        model.system_dynamics(
            cn=0.5,
            cn_sum=1.0,
            f=0.0,
            a=model.a_scale,
            tau1=model.tau1_rest,
            km=model.km_rest,
            pulse_width=pulse_width,
            numerical_timeseries=numerical_timeseries,
        ),
        dtype=float,
    ).squeeze()


def test_periodic_calcium_forcing_uses_fixed_unit_intensity():
    model = _muscle_model()

    baseline = _dynamics(model, pulse_width=0.0002, numerical_timeseries=None)
    arbitrary_numerical_data = _dynamics(
        model,
        pulse_width=0.0002,
        numerical_timeseries=np.array([100.0]),
    )

    np.testing.assert_allclose(arbitrary_numerical_data, baseline)
    expected_gain = model.periodic_cn_sum_gain()
    np.testing.assert_allclose(baseline[1], -1.0 / model.tauc + expected_gain)


def test_pulse_width_changes_force_recruitment_not_calcium_forcing():
    model = _muscle_model()

    short_pulse = _dynamics(model, pulse_width=0.00015, numerical_timeseries=None)
    long_pulse = _dynamics(model, pulse_width=0.0006, numerical_timeseries=None)

    np.testing.assert_allclose(short_pulse[:2], long_pulse[:2])
    assert long_pulse[2] > short_pulse[2]


def test_periodic_node_amplitude_matches_truncated_historical_sum():
    model = DingModelPulseWidthFrequencyWithFatiguePeriodicNode(
        muscle_name="Biceps",
        stim_time=[0.0, 1 / 30],
        sum_stim_truncation=6,
    )
    decay = np.exp(-(1 / 30) / model.tauc)
    ri = 1.0 + (model.get_r0(model.km_rest) - 1.0) * decay
    expected = decay**5 + ri * sum(decay**age for age in range(5))

    np.testing.assert_allclose(model.post_stimulation_amplitude(), expected)


def test_periodic_node_data_reconstructs_exact_within_interval_decay():
    model = DingModelPulseWidthFrequencyWithFatiguePeriodicNode(
        muscle_name="Biceps",
        stim_time=[0.0, 1 / 30],
        sum_stim_truncation=6,
    )
    data, _ = model.get_numerical_data_time_series(3, 0.1)
    stage_data = data["periodic_calcium"][:, 0, 1]
    local_time = 0.012
    absolute_time = stage_data[1] + local_time

    observed = float(model.calcium_history(np.array([absolute_time]), stage_data))
    expected = stage_data[0] * np.exp(-local_time / model.tauc)

    np.testing.assert_allclose(observed, expected)


def _periodic_cn_collocation_fixed_point(model, degree: int) -> float:
    """Evaluate the same one-interval Radau map used by direct collocation."""

    interval = 1.0 / 30.0
    tau = model.tauc
    amplitude = model.post_stimulation_amplitude()
    points = collocation_points(degree, "radau")
    coefficients, continuity, _ = collocation_coeff(points)
    coefficients = np.asarray(coefficients, dtype=float)
    continuity = np.asarray(continuity, dtype=float).reshape(-1)

    stage_matrix = np.zeros((degree, degree))
    for equation in range(degree):
        for stage in range(1, degree + 1):
            stage_matrix[equation, stage - 1] = coefficients[stage, equation]
            if stage == equation + 1:
                stage_matrix[equation, stage - 1] += interval / tau

    def step(initial_cn: float) -> float:
        right_hand_side = np.array(
            [
                -coefficients[0, equation] * initial_cn
                + (interval / tau)
                * amplitude
                * np.exp(-points[equation] * interval / tau)
                for equation in range(degree)
            ]
        )
        stages = np.linalg.solve(stage_matrix, right_hand_side)
        return float(
            continuity[0] * initial_cn
            + np.dot(continuity[1:], stages)
        )

    affine_offset = step(0.0)
    affine_gain = step(1.0) - affine_offset
    return affine_offset / (1.0 - affine_gain)


def test_scientific_radau5_resolves_periodic_calcium_against_exact_solution():
    model = DingModelPulseWidthFrequencyWithFatiguePeriodicNode(
        muscle_name="Biceps",
        stim_time=[0.0, 1 / 30],
        sum_stim_truncation=6,
    )

    exact = model.periodic_cn_fixed_point()
    radau3 = _periodic_cn_collocation_fixed_point(model, 3)
    radau4 = _periodic_cn_collocation_fixed_point(model, 4)
    radau5 = _periodic_cn_collocation_fixed_point(model, 5)
    radau6 = _periodic_cn_collocation_fixed_point(model, 6)

    assert exact == pytest.approx(0.1629821583533315, abs=1e-14)
    assert abs(radau3 / exact - 1.0) > 0.06
    assert 4e-3 < abs(radau4 / exact - 1.0) < 5e-3
    assert abs(radau5 / exact - 1.0) < 1e-3
    assert abs(radau6 / exact - 1.0) < 1e-3
    assert abs(radau5 / radau6 - 1.0) < 1e-3


def test_periodic_node_pulse_width_does_not_change_calcium_derivative():
    model = DingModelPulseWidthFrequencyWithFatiguePeriodicNode(
        muscle_name="Biceps",
        stim_time=[0.0, 1 / 30],
    )
    state = model.standard_rest_values().reshape(-1)
    state[0] = 0.5
    data = np.array([model.post_stimulation_amplitude(), 0.0])

    short = np.asarray(
        model.system_dynamics(
            states=state,
            controls=np.array([0.00015]),
            time=np.array([0.005]),
            numerical_timeseries=data,
        ),
        dtype=float,
    ).reshape(-1)
    long = np.asarray(
        model.system_dynamics(
            states=state,
            controls=np.array([0.0006]),
            time=np.array([0.005]),
            numerical_timeseries=data,
        ),
        dtype=float,
    ).reshape(-1)

    np.testing.assert_allclose(short[0], long[0])
    assert long[1] > short[1]


def test_time_dependent_rk4_map_retains_local_time_inside_interval():
    state = SX.sym("state", 1)
    control = SX.sym("control", 0)
    parameters = SX.sym("parameters", 0)
    local_time = SX.sym("local_time", 1)
    discrete_map = periodic_example.build_time_dependent_rk4_map(
        rhs=local_time,
        state=state,
        control=control,
        stage_parameters=parameters,
        local_time=local_time,
        interval_duration=2.0,
        n_substeps=2,
    )

    observed = float(Function("discrete_map", [state], [discrete_map])(3.0))

    np.testing.assert_allclose(observed, 5.0)


def test_periodic_node_irk_converts_acados_local_stage_time_to_absolute_time():
    observed = periodic_example._periodic_node_dynamics_time(
        "IRK", acados_time=0.012, interval_start=0.5, interval_duration=1 / 30
    )

    np.testing.assert_allclose(observed, 0.512)


def test_periodic_node_erk_uses_absolute_interval_midpoint():
    observed = periodic_example._periodic_node_dynamics_time(
        "ERK", acados_time=None, interval_start=0.5, interval_duration=1 / 30
    )

    np.testing.assert_allclose(observed, 0.5 + 1 / 60)


def test_acados_rhs_is_converted_to_scaled_state_derivative():
    rhs = SX.sym("rhs", 3)
    scaled_rhs = periodic_example._scaled_acados_dynamics_rhs(
        rhs, state_scaling=np.array([2.0, 10.0]), n_parameters=1
    )
    function = Function("scaled_rhs", [rhs], [scaled_rhs])

    observed = np.asarray(function(np.array([0.0, 4.0, 30.0]))).reshape(-1)

    np.testing.assert_allclose(observed, np.array([0.0, 2.0, 3.0]))


def test_high_accuracy_integrator_diagnostic_handles_time_dependent_dynamics():
    class Variables(dict):
        def __init__(self, values, shape):
            super().__init__(values)
            self.shape = shape

    state_variables = Variables({"x": SimpleNamespace(index=[0])}, shape=1)
    control_variables = Variables({"u": SimpleNamespace(index=[0])}, shape=1)
    nlp = SimpleNamespace(
        states=state_variables,
        controls=control_variables,
        x_init={"x": SimpleNamespace(init=np.array([[0.0, 0.125, 0.5]]))},
        u_init={"u": SimpleNamespace(init=np.zeros((1, 2)))},
        numerical_data_timeseries=None,
        dynamics_func=lambda time, state, control, parameters, algebraic, data: np.array(
            [time[0]]
        ),
    )
    nmpc = SimpleNamespace(nlp=[nlp], cycle_duration=1.0, cycle_len=2)

    rows = periodic_example.high_accuracy_integrator_map_diagnostics(
        nmpc, nodes=(0, 1), rk4_substeps=2
    )

    assert [row["node"] for row in rows] == [0, 1]
    assert max(row["trajectory_vs_reference"] for row in rows) < 1e-12
    assert max(row["rk4_vs_reference"] for row in rows) < 1e-12


def test_high_accuracy_trace_rollout_reintegrates_controls_and_fatigue():
    class Variables(dict):
        def __init__(self, values, shape):
            super().__init__(values)
            self.shape = shape

    state_variables = Variables({"A_Test": SimpleNamespace(index=[0])}, shape=1)
    control_variables = Variables({"u": SimpleNamespace(index=[0])}, shape=1)
    nlp = SimpleNamespace(
        states=state_variables,
        controls=control_variables,
        numerical_data_timeseries=None,
        dynamics_func=lambda _time, _state, _control, _parameters, _algebraic, _data: np.array(
            [-1.0]
        ),
    )
    nmpc = SimpleNamespace(nlp=[nlp], cycle_duration=1.0, cycle_len=2)

    diagnostic = periodic_example.high_accuracy_trace_rollout_diagnostics(
        nmpc,
        {"A_Test": np.array([[10.0, 9.5, 9.0]])},
        {"u": np.zeros((1, 2))},
        cycle_count=1,
        capacity_scales={"A_Test": 10.0},
    )

    assert diagnostic["available"] is True
    assert diagnostic["interval_count"] == 2
    assert diagnostic["maximum_absolute_endpoint_error"] < 1e-11
    np.testing.assert_allclose(diagnostic["fatigue_auc_cycles"], 0.05, atol=1e-11)
    np.testing.assert_allclose(
        diagnostic["executed_fatigue_objective"], 100.0 / 3.0, atol=1e-9
    )
    np.testing.assert_allclose(
        diagnostic["muscle_fatigue"][0]["final_capacity_ratio"], 0.9
    )


def test_high_accuracy_trace_rollout_selects_collocation_shooting_nodes():
    class Variables(dict):
        def __init__(self, values, shape):
            super().__init__(values)
            self.shape = shape

    state_variables = Variables({"A_Test": SimpleNamespace(index=[0])}, shape=1)
    control_variables = Variables({"u": SimpleNamespace(index=[0])}, shape=1)
    nlp = SimpleNamespace(
        states=state_variables,
        controls=control_variables,
        numerical_data_timeseries=None,
        dynamics_func=lambda _time, _state, _control, _parameters, _algebraic, _data: np.array(
            [-1.0]
        ),
    )
    nmpc = SimpleNamespace(nlp=[nlp], cycle_duration=1.0, cycle_len=2)

    # Radau-5 stores six state columns per shooting interval: one endpoint
    # followed by five collocation stages.  Only columns 0, 6 and 12 are the
    # shooting nodes against which the independent rollout must be compared.
    diagnostic = periodic_example.high_accuracy_trace_rollout_diagnostics(
        nmpc,
        {"A_Test": np.linspace(10.0, 9.0, 13)[None, :]},
        {"u": np.zeros((1, 2))},
        cycle_count=1,
        capacity_scales={"A_Test": 10.0},
    )

    assert diagnostic["available"] is True
    assert diagnostic["state_node_stride"] == 6
    assert diagnostic["maximum_absolute_endpoint_error"] < 1e-11
    np.testing.assert_allclose(diagnostic["fatigue_auc_cycles"], 0.05, atol=1e-11)


def test_solver_comparison_cli_exposes_high_accuracy_trace_audit():
    parser = comparison_example.build_cli()

    assert parser.parse_args([]).validate_integrator_maps is False
    assert (
        parser.parse_args(["--validate-integrator-maps"]).validate_integrator_maps
        is True
    )


def test_wheel_periodicity_diagnostic_supports_reduced_theta_state():
    class Variables(dict):
        def __init__(self, values, shape):
            super().__init__(values)
            self.shape = shape

    states = Variables(
        {
            "theta": SimpleNamespace(index=[0]),
            "omega": SimpleNamespace(index=[1]),
        },
        shape=2,
    )
    controls = Variables({"u": SimpleNamespace(index=[0])}, shape=1)
    nlp = SimpleNamespace(
        states=states,
        controls=controls,
        x_init={
            "theta": SimpleNamespace(init=np.array([[0.4, 0.5, 0.6]])),
            "omega": SimpleNamespace(init=np.array([[-6.0, -6.1, -6.2]])),
        },
        u_init={"u": SimpleNamespace(init=np.zeros((1, 2)))},
        numerical_data_timeseries=None,
        dynamics_func=lambda _time, state, *_args: np.array(
            [state[1], np.sin(state[0])]
        ),
    )
    nmpc = SimpleNamespace(nlp=[nlp], cycle_duration=1.0, cycle_len=2)

    diagnostic = periodic_example.wheel_angle_periodicity_diagnostics(nmpc)

    assert diagnostic["max_abs_rhs_difference"] < 1e-12
    assert diagnostic["l2_rhs_difference"] < 1e-12


def test_solution_trace_comparisons_reports_scaled_differences():
    reference = periodic_example._WarmupSolutionAdapter(
        states={"x": np.array([[0.0, 1.0, 2.0]])},
        controls={"u": np.array([[1.0, 2.0]])},
    )
    candidate = periodic_example._WarmupSolutionAdapter(
        states={"x": np.array([[0.0, 1.5, 2.0]])},
        controls={"u": np.array([[1.0, 3.0]])},
    )

    state_row = periodic_example.solution_trace_comparisons(
        reference, candidate, controls=False
    )[0]
    control_row = periodic_example.solution_trace_comparisons(
        reference, candidate, controls=True
    )[0]

    np.testing.assert_allclose(state_row["rmse"], 0.5 / np.sqrt(3))
    np.testing.assert_allclose(state_row["normalized_rmse"], 0.25 / np.sqrt(3))
    np.testing.assert_allclose(control_row["max_abs_error"], 1.0)


def test_pulse_width_trust_region_keeps_nodewise_centers():
    bounds = SimpleNamespace(
        min=np.array([[0.1, 0.1, 0.1]]), max=np.array([[0.6, 0.6, 0.6]])
    )
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                u_init={
                    "last_pulse_width_Biceps": SimpleNamespace(
                        init=np.array([[0.2, 0.4, 0.5]])
                    )
                },
                u_bounds={"last_pulse_width_Biceps": bounds},
            )
        ]
    )

    periodic_example.apply_pulse_width_control_trust_region(nmpc, radius=0.01)
    lower, upper = nmpc._cocofest_nodewise_control_bounds["last_pulse_width_Biceps"]

    np.testing.assert_allclose(lower, np.array([[0.19, 0.39, 0.49]]))
    np.testing.assert_allclose(upper, np.array([[0.21, 0.41, 0.51]]))

    nmpc.nlp[0].u_init["last_pulse_width_Biceps"].init[:, :] = [0.3, 0.5, 0.55]
    periodic_example.apply_pulse_width_control_trust_region(nmpc, radius=0.01)
    lower, upper = nmpc._cocofest_nodewise_control_bounds["last_pulse_width_Biceps"]

    np.testing.assert_allclose(lower, np.array([[0.29, 0.49, 0.54]]))
    np.testing.assert_allclose(upper, np.array([[0.31, 0.51, 0.56]]))


def test_phase_aligned_active_set_guard_only_releases_transition_neighborhoods():
    center = np.array([[0.1, 0.1, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1]])
    bounds = SimpleNamespace(
        min=np.full(center.shape, 0.1),
        max=np.full(center.shape, 0.6),
    )
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                u_init={"last_pulse_width_Biceps": SimpleNamespace(init=center.copy())},
                u_bounds={"last_pulse_width_Biceps": bounds},
            )
        ]
    )

    periodic_example.apply_pulse_width_control_trust_region(nmpc, radius=0.01)
    summary = periodic_example.apply_phase_aligned_pulse_width_transition_guard(
        nmpc,
        radius=0.2,
        margin=1,
        activation_threshold=0.01,
    )
    lower, upper = nmpc._cocofest_nodewise_control_bounds["last_pulse_width_Biceps"]

    assert summary["last_pulse_width_Biceps"]["transition_nodes"] == [2, 5]
    assert summary["last_pulse_width_Biceps"]["released_nodes"] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    np.testing.assert_allclose(lower[:, [0, 7]], [[0.1, 0.1]])
    np.testing.assert_allclose(upper[:, [0, 7]], [[0.11, 0.11]])
    np.testing.assert_allclose(upper[:, [1, 5, 6]], [[0.3, 0.3, 0.3]])
    np.testing.assert_allclose(lower[:, [2, 3, 4]], [[0.2, 0.2, 0.2]])
    np.testing.assert_allclose(upper[:, [2, 3, 4]], [[0.6, 0.6, 0.6]])


def test_phase_aligned_active_set_guard_does_not_widen_uniform_recruitment():
    center = np.full((1, 4), 0.1)
    bounds = SimpleNamespace(
        min=np.full(center.shape, 0.1),
        max=np.full(center.shape, 0.6),
    )
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                u_init={"last_pulse_width_Biceps": SimpleNamespace(init=center.copy())},
                u_bounds={"last_pulse_width_Biceps": bounds},
            )
        ]
    )

    periodic_example.apply_pulse_width_control_trust_region(nmpc, radius=0.01)
    summary = periodic_example.apply_phase_aligned_pulse_width_transition_guard(
        nmpc, radius=0.2
    )
    lower, upper = nmpc._cocofest_nodewise_control_bounds["last_pulse_width_Biceps"]

    assert summary["last_pulse_width_Biceps"]["released_count"] == 0
    assert summary["last_pulse_width_Biceps"]["reason"] == "no_active_set_transition"
    np.testing.assert_allclose(lower, 0.1)
    np.testing.assert_allclose(upper, 0.11)


def test_phase_aligned_active_set_guard_wraps_circular_margin():
    center = np.array([[0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    bounds = SimpleNamespace(
        min=np.full(center.shape, 0.1),
        max=np.full(center.shape, 0.6),
    )
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                u_init={"last_pulse_width_Biceps": SimpleNamespace(init=center.copy())},
                u_bounds={"last_pulse_width_Biceps": bounds},
            )
        ]
    )

    periodic_example.apply_pulse_width_control_trust_region(nmpc, radius=0.01)
    summary = periodic_example.apply_phase_aligned_pulse_width_transition_guard(
        nmpc,
        radius=0.2,
        margin=1,
        activation_threshold=0.01,
    )["last_pulse_width_Biceps"]

    assert summary["transition_nodes"] == [0, 2]
    assert summary["released_nodes"] == [0, 1, 2, 3, 7]
    lower, upper = nmpc._cocofest_nodewise_control_bounds["last_pulse_width_Biceps"]
    np.testing.assert_allclose(lower[:, [4, 5, 6]], 0.1)
    np.testing.assert_allclose(upper[:, [4, 5, 6]], 0.11)
    np.testing.assert_allclose(upper[:, 7], 0.3)


def test_control_homotopy_radii_are_parsed_as_an_increasing_sequence():
    parser = periodic_example.build_argument_parser()
    args = parser.parse_args(
        [
            "--acados-control-homotopy-radii",
            "1e-8,1e-7,1e-6",
            "--acados-control-homotopy-keep-final-radius",
            "--acados-control-homotopy-each-window",
            "--acados-control-homotopy-window-growth",
            "1.25",
            "--acados-control-homotopy-window-max-radius",
            "1e-4",
            "--acados-control-homotopy-max-restarts",
            "3",
            "--acados-control-homotopy-stage-iterations",
            "40",
        ]
    )

    assert args.acados_control_homotopy_radii == (1e-8, 1e-7, 1e-6)
    assert args.acados_control_homotopy_keep_final_radius is True
    assert args.acados_control_homotopy_each_window is True
    assert args.acados_control_homotopy_window_growth == 1.25
    assert args.acados_control_homotopy_window_max_radius == 1e-4
    assert args.acados_control_homotopy_max_restarts == 3
    assert args.acados_control_homotopy_stage_iterations == 40


def test_comparison_cli_forwards_acados_hot_start_homotopy_options():
    args = comparison_example.build_cli().parse_args(
        [
            "--disable-acados-assisted-hot-start",
            "--acados-control-homotopy-radii",
            "1e-6,1e-5",
            "--acados-control-homotopy-tolerance",
            "2e-2",
            "--acados-control-homotopy-stage-iterations",
            "30",
            "--acados-control-homotopy-max-restarts",
            "2",
            "--acados-transfer-active-set-guard-radius",
            "5e-4",
            "--acados-transfer-active-set-guard-margin",
            "2",
            "--acados-transfer-active-set-threshold",
            "2e-6",
        ]
    )

    assert args.acados_assisted_hot_start is False
    assert args.acados_control_homotopy_radii == (1e-6, 1e-5)
    assert args.acados_control_homotopy_tolerance == 2e-2
    assert args.acados_control_homotopy_stage_iterations == 30
    assert args.acados_control_homotopy_max_restarts == 2
    assert args.acados_transfer_active_set_guard_radius == 5e-4
    assert args.acados_transfer_active_set_guard_margin == 2
    assert args.acados_transfer_active_set_threshold == 2e-6


def test_control_homotopy_window_radius_growth_respects_its_physical_cap():
    radius = 1e-7
    observed = []
    for _ in range(5):
        radius = periodic_example.next_acados_control_homotopy_radius(
            radius,
            growth=10.0,
            maximum_radius=1e-4,
        )
        observed.append(radius)

    np.testing.assert_allclose(observed, [1e-6, 1e-5, 1e-4, 1e-4, 1e-4])


def test_proximal_control_weights_are_parsed_as_a_decreasing_sequence():
    args = periodic_example.build_argument_parser().parse_args(
        [
            "--acados-proximal-control-weights",
            "1e6,1e5,1e4",
            "--acados-proximal-control-each-window",
            "--acados-proximal-control-try-next-weight-on-failure",
            "--continue-after-acados-transfer-failure",
            "--acados-terminal-wheel-q-homotopy-slacks",
            "0.2,0.1,0.02",
            "--acados-terminal-wheel-q-homotopy-each-window",
            "--acados-proximal-control-stage-iterations",
            "30",
            "--acados-proximal-control-restart-feasibility-factor",
            "5",
            "--acados-transfer-mechanical-restoration",
            "--acados-transfer-mechanical-control-radius",
            "7e-5",
            "--acados-transfer-mechanical-regularization",
            "0.02",
            "--acados-transfer-mechanical-substeps",
            "7",
        ]
    )

    assert args.acados_proximal_control_weights == (1e6, 1e5, 1e4)
    assert args.acados_proximal_control_each_window is True
    assert args.acados_proximal_control_try_next_weight_on_failure is True
    assert args.continue_after_acados_transfer_failure is True
    assert args.acados_proximal_control_stage_iterations == 30
    assert args.acados_proximal_control_restart_feasibility_factor == 5
    assert args.acados_transfer_mechanical_restoration is True
    assert args.acados_transfer_mechanical_control_radius == 7e-5
    assert args.acados_transfer_mechanical_regularization == 0.02
    assert args.acados_transfer_mechanical_substeps == 7


def test_terminal_wheel_bound_slacks_are_parsed_as_a_decreasing_sequence():
    args = periodic_example.build_argument_parser().parse_args(
        [
            "--acados-terminal-wheel-q-homotopy-slacks",
            "0.2,0.1,0.05,0.02",
            "--acados-terminal-wheel-q-homotopy-each-window",
        ]
    )

    assert args.acados_terminal_wheel_q_homotopy_slacks == (
        0.2,
        0.1,
        0.05,
        0.02,
    )
    assert args.acados_terminal_wheel_q_homotopy_each_window is True


def test_transfer_bound_homotopy_fractions_are_parsed():
    args = periodic_example.build_argument_parser().parse_args(
        [
            "--acados-transfer-irk-rollout",
            "--acados-transfer-bound-homotopy",
            "--acados-transfer-bound-homotopy-fractions",
            "0,0.5,1",
            "--acados-transfer-bound-homotopy-padding",
            "0.1",
            "--acados-transfer-bound-homotopy-iterations",
            "12",
            "--acados-transfer-bound-homotopy-solver-tolerance",
            "1e-5",
            "--acados-transfer-bound-homotopy-min-fraction-step",
            "0.001953125",
            "--acados-transfer-bound-homotopy-max-refinements",
            "16",
        ]
    )

    assert args.acados_transfer_bound_homotopy is True
    assert args.acados_transfer_bound_homotopy_fractions == (0.0, 0.5, 1.0)
    assert args.acados_transfer_bound_homotopy_padding == 0.1
    assert args.acados_transfer_bound_homotopy_iterations == 12
    assert args.acados_transfer_bound_homotopy_solver_tolerance == 1e-5
    assert args.acados_transfer_bound_homotopy_min_fraction_step == 0.001953125
    assert args.acados_transfer_bound_homotopy_max_refinements == 16


def test_transfer_sqp_restart_options_are_parsed():
    parser = periodic_example.build_argument_parser()
    args = parser.parse_args(
        [
            "--acados-transfer-sqp-restarts",
            "4",
            "--acados-transfer-sqp-restart-iterations",
            "2",
            "--acados-transfer-sqp-restart-feasibility-tolerance",
            "0.02",
        ]
    )

    assert args.acados_transfer_sqp_restarts == 4
    assert args.acados_transfer_sqp_restart_iterations == 2
    assert args.acados_transfer_sqp_restart_feasibility_tolerance == 0.02


def test_acados_iterate_storage_is_opt_in():
    parser = periodic_example.build_argument_parser()
    stored_args = parser.parse_args(["--acados-store-iterates"])

    assert parser.parse_args([]).acados_store_iterates is False
    assert stored_args.acados_store_iterates is True
    assert periodic_example._codegen_signature(
        parser.parse_args([])
    ) != periodic_example._codegen_signature(stored_args)


def test_acados_conditional_maxiter_retry_options_are_parsed():
    parser = periodic_example.build_argument_parser()
    args = parser.parse_args(
        [
            "--acados-store-iterates",
            "--acados-maxiter-retries",
            "1",
            "--acados-maxiter-retry-iterations",
            "20",
            "--acados-maxiter-retry-feasibility-tolerance",
            "0.0025",
        ]
    )

    assert args.acados_store_iterates is True
    assert args.acados_maxiter_retries == 1
    assert args.acados_maxiter_retry_iterations == 20
    assert args.acados_maxiter_retry_feasibility_tolerance == pytest.approx(0.0025)


def test_acados_maxiter_retry_candidate_requires_nearly_feasible_history():
    diagnostics = {
        "res_stat_all": np.array([20.0, 0.2, 0.3]),
        "res_eq_all": np.array([0.4, 0.0019, 0.003]),
        "res_ineq_all": np.array([0.01, 1e-8, 1e-7]),
        "res_comp_all": np.array([0.0, 1e-9, 1e-8]),
    }

    candidate = periodic_example._acados_maxiter_retry_candidate(
        2, diagnostics, feasibility_tolerance=0.0025
    )
    rejected = periodic_example._acados_maxiter_retry_candidate(
        2, diagnostics, feasibility_tolerance=0.001
    )

    assert candidate["eligible"] is True
    assert candidate["best_index"] == 1
    assert candidate["best_feasibility"] == pytest.approx(0.0019)
    assert rejected["eligible"] is False
    assert rejected["reason"] == "best_iterate_not_nearly_feasible"
    assert periodic_example._acados_maxiter_retry_candidate(
        0, diagnostics, feasibility_tolerance=0.0025
    ) == {"eligible": False, "reason": "status_not_maxiter"}


def test_acados_maxiter_retry_prefers_stationarity_within_feasible_candidates():
    diagnostics = {
        "res_stat_all": np.array([20.0, 0.4, 0.01]),
        "res_eq_all": np.array([0.4, 1e-5, 0.002]),
        "res_ineq_all": np.array([0.01, 1e-8, 1e-7]),
        "res_comp_all": np.array([0.0, 1e-9, 1e-8]),
    }

    candidate = periodic_example._acados_maxiter_retry_candidate(
        2, diagnostics, feasibility_tolerance=0.0025
    )

    assert candidate["eligible"] is True
    assert candidate["best_index"] == 2
    assert candidate["selection"] == "minimum_stationarity_within_feasibility"


def test_capture_and_queue_acados_stored_primal_dual_iterate():
    fields = ("x", "u", "pi", "lam", "sl", "su")

    class StoredIterate:
        def __init__(self):
            self.arrays = {
                field: np.full(2, fields.index(field) + 1.0) for field in fields
            }

        def flatten(self):
            return SimpleNamespace(**self.arrays)

    class Capsule:
        stored = StoredIterate()

        def get_iterate(self, index):
            assert index == 8
            return self.stored

        def get_dim_flat(self, field):
            if field == "z":
                return 0
            assert field in fields
            return 2

    class Interface:
        def __init__(self):
            self.ocp_solver = Capsule()
            self.queued = None

        def get_solver_state(self):
            return {
                "solver": "ACADOS",
                "format_version": 1,
                "n_horizon": 30,
                "iterate": {field: np.zeros(2) for field in fields},
            }

        def set_lagrange_multiplier(self, solution):
            self.queued = solution.solver_state

    interface = Interface()
    nmpc = SimpleNamespace(ocp_solver=interface)

    summary, state = periodic_example.capture_acados_stored_primal_dual_iterate(
        nmpc, 8
    )
    queued = periodic_example.queue_acados_primal_dual_solver_state(nmpc, state)
    interface.ocp_solver.stored.arrays["x"][:] = -99.0

    assert summary["captured"] is True
    assert summary["field_sizes"] == {field: 2 for field in fields}
    assert queued == {"queued": True, "reason": None}
    for index, field in enumerate(fields):
        np.testing.assert_allclose(interface.queued["iterate"][field], index + 1.0)


def test_capture_rejects_acados_algebraic_states():
    class Capsule:
        def get_iterate(self, index):
            return SimpleNamespace(flatten=lambda: SimpleNamespace())

        def get_dim_flat(self, field):
            return 1 if field == "z" else 0

    interface = SimpleNamespace(
        ocp_solver=Capsule(),
        get_solver_state=lambda: {"iterate": {}},
    )

    summary, state = periodic_example.capture_acados_stored_primal_dual_iterate(
        SimpleNamespace(ocp_solver=interface), 0
    )

    assert summary["captured"] is False
    assert summary["reason"] == "algebraic_states_not_supported"
    assert state is None


def test_conditional_maxiter_retry_ignores_successful_main_solve():
    class FakeAcadosInterface:
        status = -1
        real_time_to_optimize = 0.1

        def solve(self):
            self.status = 0
            return self.get_optimized_value()

        def get_optimized_value(self):
            return {
                "status": self.status,
                "solver_time_to_optimize": 0.1,
                "real_time_to_optimize": self.real_time_to_optimize,
                "iter": 3,
            }

    interface = FakeAcadosInterface()
    nmpc = SimpleNamespace(
        ocp_solver=interface,
        total_optimization_run=4,
        _cocofest_acados_main_window_retry_armed=True,
    )
    summaries = []

    periodic_example.install_acados_conditional_maxiter_retry(
        nmpc,
        max_retries=1,
        retry_iterations=20,
        feasibility_tolerance=0.0025,
        nominal_iterations=100,
        summaries=summaries,
        echo=False,
        residual_diagnostics_function=lambda current_nmpc: pytest.fail(
            "A successful solve must not inspect the residual history."
        ),
    )

    output = interface.solve()

    assert output["status"] == 0
    assert summaries == []


def test_conditional_maxiter_retry_replaces_solution_and_accounts_full_budget():
    class FakeAcadosInterface:
        def __init__(self):
            self.status = -1
            self.real_time_to_optimize = 0.0
            self.solve_index = -1
            self.statuses = [2, 0]
            self.solver_times = [1.25, 0.2]
            self.wall_times = [1.4, 0.25]
            self.iterations = [100, 7]

        def solve(self):
            self.solve_index += 1
            self.status = self.statuses[self.solve_index]
            self.real_time_to_optimize = self.wall_times[self.solve_index]
            return self.get_optimized_value()

        def get_optimized_value(self):
            return {
                "status": self.status,
                "solver_time_to_optimize": self.solver_times[self.solve_index],
                "real_time_to_optimize": self.real_time_to_optimize,
                "iter": self.iterations[self.solve_index],
            }

    interface = FakeAcadosInterface()
    nmpc = SimpleNamespace(
        ocp_solver=interface,
        total_optimization_run=13,
        _cocofest_acados_main_window_retry_armed=True,
    )
    summaries = []
    applied = []
    resets = []
    queued_states = []
    iteration_budgets = []

    def residual_diagnostics(current_nmpc):
        index = current_nmpc.ocp_solver.solve_index
        if index == 0:
            return {
                "sqp_iter": 100,
                "time_tot": 1.25,
                "res_stat_all": np.array([300.0, 0.2, 0.3]),
                "res_eq_all": np.array([0.4, 0.0019, 0.003]),
                "res_ineq_all": np.array([0.01, 1e-8, 1e-7]),
                "res_comp_all": np.array([0.0, 1e-9, 1e-8]),
            }
        return {
            "sqp_iter": 7,
            "time_tot": 0.2,
            "res_stat_all": np.array([1e-3]),
            "res_eq_all": np.array([1e-8]),
            "res_ineq_all": np.array([1e-9]),
            "res_comp_all": np.array([1e-10]),
        }

    def apply_primal(current_nmpc, iterate_index, require_stored_iterate):
        applied.append((iterate_index, require_stored_iterate))
        return {
            "applied": True,
            "reason": None,
            "source": "stored_iterate",
            "iterate_index": iterate_index,
        }

    installed = periodic_example.install_acados_conditional_maxiter_retry(
        nmpc,
        max_retries=1,
        retry_iterations=20,
        feasibility_tolerance=0.0025,
        nominal_iterations=100,
        summaries=summaries,
        echo=False,
        residual_diagnostics_function=residual_diagnostics,
        apply_primal_function=apply_primal,
        capture_solver_state_function=lambda current_nmpc, iterate_index: (
            {
                "captured": True,
                "reason": None,
                "iterate_index": iterate_index,
            },
            {"iterate": "stored"},
        ),
        queue_solver_state_function=lambda current_nmpc, state: (
            queued_states.append(state) or {"queued": True, "reason": None}
        ),
        clear_solver_state_function=lambda current_nmpc: {
            "cleared": True,
            "reason": None,
        },
        reset_memory_function=lambda current_nmpc: resets.append(True) or True,
        set_iterations_function=lambda current_nmpc, value: (
            iteration_budgets.append(value) or True
        ),
    )
    output = interface.solve()

    assert installed is True
    assert output["status"] == 0
    assert output["solver_time_to_optimize"] == pytest.approx(1.45)
    assert output["real_time_to_optimize"] >= 0.0
    assert output["iter"] == 107
    assert applied == [(1, True)]
    assert resets == [True]
    assert queued_states == [{"iterate": "stored"}]
    assert iteration_budgets == [20, 100]
    assert summaries[0]["window"] == 13
    assert summaries[0]["attempts"][0]["trigger_status"] == 2
    assert summaries[0]["attempts"][0]["retry_status"] == 0
    assert summaries[0]["final_status"] == 0
    assert summaries[0]["native_solver_wall_time_s"] == pytest.approx(1.65)
    assert summaries[0]["wall_time_s"] == pytest.approx(
        output["real_time_to_optimize"]
    )
    assert summaries[0]["iteration_budget_restored"] is True


def test_conditional_maxiter_retry_does_not_reset_when_primal_dual_queue_fails():
    class FakeAcadosInterface:
        status = -1
        real_time_to_optimize = 0.1

        def solve(self):
            self.status = 2
            return self.get_optimized_value()

        def get_optimized_value(self):
            return {
                "status": self.status,
                "solver_time_to_optimize": 0.1,
                "real_time_to_optimize": self.real_time_to_optimize,
                "iter": 100,
            }

    interface = FakeAcadosInterface()
    nmpc = SimpleNamespace(
        ocp_solver=interface,
        total_optimization_run=13,
        _cocofest_acados_main_window_retry_armed=True,
    )
    diagnostics = {
        "sqp_iter": 100,
        "time_tot": 0.1,
        "res_stat_all": np.array([1.0, 0.2]),
        "res_eq_all": np.array([0.1, 1e-4]),
        "res_ineq_all": np.array([0.0, 0.0]),
        "res_comp_all": np.array([0.0, 0.0]),
    }
    resets = []
    budgets = []
    summaries = []

    periodic_example.install_acados_conditional_maxiter_retry(
        nmpc,
        max_retries=1,
        retry_iterations=20,
        feasibility_tolerance=0.0025,
        nominal_iterations=100,
        summaries=summaries,
        echo=False,
        residual_diagnostics_function=lambda current_nmpc: diagnostics,
        apply_primal_function=lambda *args, **kwargs: {
            "applied": True,
            "reason": None,
            "bound_projection": {
                "state_max_change": 0.0,
                "control_max_change": 0.0,
            },
        },
        capture_solver_state_function=lambda current_nmpc, iterate_index: (
            {"captured": True, "reason": None},
            {"iterate": "stored"},
        ),
        queue_solver_state_function=lambda current_nmpc, state: {
            "queued": False,
            "reason": "test_queue_failure",
        },
        clear_solver_state_function=lambda current_nmpc: pytest.fail(
            "Nothing was queued, so no state should need clearing."
        ),
        reset_memory_function=lambda current_nmpc: resets.append(True) or True,
        set_iterations_function=lambda current_nmpc, value: budgets.append(value)
        or True,
    )

    output = interface.solve()

    assert output["status"] == 2
    assert resets == []
    assert budgets == [20, 100]
    assert summaries[0]["attempts"][0]["reason"] == "test_queue_failure"


def test_conditional_maxiter_retry_rejects_a_projected_stored_primal():
    class FakeAcadosInterface:
        status = -1
        real_time_to_optimize = 0.1

        def solve(self):
            self.status = 2
            return self.get_optimized_value()

        def get_optimized_value(self):
            return {
                "status": self.status,
                "solver_time_to_optimize": 0.1,
                "real_time_to_optimize": self.real_time_to_optimize,
                "iter": 100,
            }

    interface = FakeAcadosInterface()
    nmpc = SimpleNamespace(
        ocp_solver=interface,
        total_optimization_run=13,
        _cocofest_acados_main_window_retry_armed=True,
    )
    diagnostics = {
        "sqp_iter": 100,
        "time_tot": 0.1,
        "res_stat_all": np.array([1.0, 0.2]),
        "res_eq_all": np.array([0.1, 1e-4]),
        "res_ineq_all": np.array([0.0, 0.0]),
        "res_comp_all": np.array([0.0, 0.0]),
    }
    queues = []
    resets = []
    summaries = []

    periodic_example.install_acados_conditional_maxiter_retry(
        nmpc,
        max_retries=1,
        retry_iterations=20,
        feasibility_tolerance=0.0025,
        nominal_iterations=100,
        summaries=summaries,
        echo=False,
        residual_diagnostics_function=lambda current_nmpc: diagnostics,
        apply_primal_function=lambda *args, **kwargs: {
            "applied": True,
            "reason": None,
            "bound_projection": {
                "state_max_change": 1e-8,
                "control_max_change": 0.0,
            },
        },
        capture_solver_state_function=lambda current_nmpc, iterate_index: (
            {"captured": True, "reason": None},
            {"iterate": "stored"},
        ),
        queue_solver_state_function=lambda current_nmpc, state: (
            queues.append(state) or {"queued": True, "reason": None}
        ),
        reset_memory_function=lambda current_nmpc: resets.append(True) or True,
        set_iterations_function=lambda current_nmpc, value: True,
    )

    interface.solve()

    assert queues == []
    assert resets == []
    assert (
        summaries[0]["attempts"][0]["reason"]
        == "stored_primal_requires_bound_projection"
    )
    assert summaries[0]["attempts"][0]["bound_projection_tolerance"] == pytest.approx(
        1e-9
    )


def test_conditional_maxiter_retry_fails_if_nominal_budget_is_not_restored():
    class FakeAcadosInterface:
        def __init__(self):
            self.status = -1
            self.real_time_to_optimize = 0.1
            self.solve_index = -1

        def solve(self):
            self.solve_index += 1
            self.status = (2, 0)[self.solve_index]
            return self.get_optimized_value()

        def get_optimized_value(self):
            return {
                "status": self.status,
                "solver_time_to_optimize": 0.1,
                "real_time_to_optimize": self.real_time_to_optimize,
                "iter": (100, 3)[self.solve_index],
            }

    interface = FakeAcadosInterface()
    nmpc = SimpleNamespace(
        ocp_solver=interface,
        total_optimization_run=13,
        _cocofest_acados_main_window_retry_armed=True,
    )

    def residual_diagnostics(current_nmpc):
        if current_nmpc.ocp_solver.solve_index == 0:
            return {
                "sqp_iter": 100,
                "time_tot": 0.1,
                "res_stat_all": np.array([1.0, 0.2]),
                "res_eq_all": np.array([0.1, 1e-4]),
                "res_ineq_all": np.array([0.0, 0.0]),
                "res_comp_all": np.array([0.0, 0.0]),
            }
        return {
            "sqp_iter": 3,
            "time_tot": 0.1,
            "res_stat_all": np.array([1e-4]),
            "res_eq_all": np.array([1e-8]),
            "res_ineq_all": np.array([0.0]),
            "res_comp_all": np.array([0.0]),
        }

    periodic_example.install_acados_conditional_maxiter_retry(
        nmpc,
        max_retries=1,
        retry_iterations=20,
        feasibility_tolerance=0.0025,
        nominal_iterations=100,
        summaries=[],
        echo=False,
        residual_diagnostics_function=residual_diagnostics,
        apply_primal_function=lambda *args, **kwargs: {
            "applied": True,
            "reason": None,
            "source": "stored_iterate",
        },
        capture_solver_state_function=lambda current_nmpc, iterate_index: (
            {"captured": True, "reason": None},
            {"iterate": "stored"},
        ),
        queue_solver_state_function=lambda current_nmpc, state: {
            "queued": True,
            "reason": None,
        },
        clear_solver_state_function=lambda current_nmpc: {
            "cleared": True,
            "reason": None,
        },
        reset_memory_function=lambda current_nmpc: True,
        set_iterations_function=lambda current_nmpc, value: value == 20,
    )

    with pytest.raises(RuntimeError, match="did not restore"):
        interface.solve()


def test_conditional_maxiter_retry_does_not_mask_original_solver_exception():
    class FakeAcadosInterface:
        def __init__(self):
            self.status = -1
            self.real_time_to_optimize = 0.1
            self.solve_index = -1

        def solve(self):
            self.solve_index += 1
            if self.solve_index == 1:
                raise ValueError("original retry solve failure")
            self.status = 2
            return self.get_optimized_value()

        def get_optimized_value(self):
            return {
                "status": self.status,
                "solver_time_to_optimize": 0.1,
                "real_time_to_optimize": self.real_time_to_optimize,
                "iter": 100,
            }

    interface = FakeAcadosInterface()
    nmpc = SimpleNamespace(
        ocp_solver=interface,
        total_optimization_run=13,
        _cocofest_acados_main_window_retry_armed=True,
    )
    diagnostics = {
        "sqp_iter": 100,
        "time_tot": 0.1,
        "res_stat_all": np.array([1.0, 0.2]),
        "res_eq_all": np.array([0.1, 1e-4]),
        "res_ineq_all": np.array([0.0, 0.0]),
        "res_comp_all": np.array([0.0, 0.0]),
    }

    periodic_example.install_acados_conditional_maxiter_retry(
        nmpc,
        max_retries=1,
        retry_iterations=20,
        feasibility_tolerance=0.0025,
        nominal_iterations=100,
        summaries=[],
        echo=False,
        residual_diagnostics_function=lambda current_nmpc: diagnostics,
        apply_primal_function=lambda *args, **kwargs: {
            "applied": True,
            "reason": None,
            "source": "stored_iterate",
        },
        capture_solver_state_function=lambda current_nmpc, iterate_index: (
            {"captured": True, "reason": None},
            {"iterate": "stored"},
        ),
        queue_solver_state_function=lambda current_nmpc, state: {
            "queued": True,
            "reason": None,
        },
        clear_solver_state_function=lambda current_nmpc: {
            "cleared": True,
            "reason": None,
        },
        reset_memory_function=lambda current_nmpc: True,
        # Simulate a failed restoration. Even with warnings promoted to
        # exceptions, the original solver exception must remain visible.
        set_iterations_function=lambda current_nmpc, value: value == 20,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="original retry solve failure"):
            interface.solve()


def test_hpipm_tuning_options_are_parsed_and_affect_codegen():
    parser = periodic_example.build_argument_parser()
    tuned_args = parser.parse_args(
        ["--acados-qp-cond-n", "10", "--acados-hpipm-mode", "BALANCE"]
    )

    assert tuned_args.acados_qp_cond_n == 10
    assert tuned_args.acados_hpipm_mode == "BALANCE"
    assert periodic_example._codegen_signature(
        parser.parse_args([])
    ) != periodic_example._codegen_signature(tuned_args)


def test_compact_rho_output_is_opt_in():
    parser = periodic_example.build_argument_parser()

    assert parser.parse_args([]).compact_rho_output is False
    assert parser.parse_args(["--compact-rho-output"]).compact_rho_output is True


def test_acados_example_defaults_to_the_assisted_periodic_profile():
    args = periodic_example.build_argument_parser().parse_args([])
    periodic_example.apply_assisted_hot_start_defaults(args)
    torque = periodic_example.crank_torque_diagnostics(
        args.constant_crank_torque,
        args.wheel_qdot_regularization_target,
    )

    assert args.constant_crank_torque == -0.2
    assert torque["role"] == "driving"
    assert torque["assistance_nm"] == 0.2
    np.testing.assert_allclose(torque["expected_power_w"], 0.4 * np.pi)
    assert args.model_formulation == "periodic_node"
    assert args.cycles_per_window == 1
    assert args.n_windows == 3
    assert args.state_scaling == "full"
    assert args.acados_standard_warmup_transfer == "advance"
    assert args.acados_wheel_q_slack == 0.0
    assert args.acados_terminal_wheel_q_slack == 0.002
    assert args.warmup_ipopt_linear_solver == "mumps"
    assert args.periodic_ipopt_refinement is True
    assert (
        args.acados_control_homotopy_radii
        == periodic_example.DEFAULT_ASSISTED_CONTROL_HOMOTOPY_RADII
    )
    assert args.acados_control_homotopy_keep_final_radius is True
    assert args.acados_control_homotopy_stage_iterations == 100
    assert args.acados_control_homotopy_max_restarts == 1
    assert args.acados_cycle_boundary_homotopy_slacks is None

    ipopt_args = periodic_example.build_argument_parser().parse_args(
        ["--solver", "ipopt"]
    )
    periodic_example.apply_assisted_hot_start_defaults(ipopt_args)
    assert ipopt_args.acados_control_homotopy_radii is None
    assert ipopt_args.acados_control_homotopy_keep_final_radius is False


def test_common_target_seed_enables_the_robust_acados_reference_preparation():
    parser = periodic_example.build_argument_parser()
    args = parser.parse_args(["--common-initial-solution", "common.npz"])

    periodic_example.apply_assisted_hot_start_defaults(args)

    assert args.periodic_fes_warmup_projection_strategy == "rollout"
    assert args.full_dynamics_phase_one is True
    assert args.acados_transfer_full_dynamics_rollout is False
    assert args.acados_transfer_irk_rollout is True
    assert args.acados_bind_first_node_fes_states is True
    assert args.acados_dual_warm_start_mode == "reset"
    assert args.acados_reset_solver_before_solve is False
    assert args.acados_check_reuse_possible is False
    assert args.acados_code_reuse_tolerance == 1e-12
    assert args.acados_with_anderson_acceleration is False
    assert args.acados_anderson_activation_threshold == 0.1
    assert args.acados_byrd_omojokon_slack_relaxation_factor == 1.00001
    assert args.acados_nlp_solver_type == "SQP"
    assert args.acados_hessian_approx == "GAUSS_NEWTON"

    disabled = parser.parse_args(
        [
            "--common-initial-solution",
            "common.npz",
            "--disable-acados-assisted-hot-start",
        ]
    )
    periodic_example.apply_assisted_hot_start_defaults(disabled)
    assert disabled.periodic_fes_warmup_projection_strategy == "sequential"
    assert disabled.full_dynamics_phase_one is False
    assert disabled.acados_transfer_full_dynamics_rollout is False
    assert disabled.acados_transfer_irk_rollout is False
    assert disabled.acados_bind_first_node_fes_states is False


def test_solver_clis_distinguish_assistance_magnitude_from_signed_torque():
    periodic_parser = periodic_example.build_argument_parser()
    comparison_parser = comparison_example.build_cli()

    assert (
        periodic_parser.parse_args(["--crank-assistance", "0.2"]).constant_crank_torque
        == -0.2
    )
    assert (
        periodic_parser.parse_args(
            ["--signed-crank-torque", "0.2"]
        ).constant_crank_torque
        == 0.2
    )
    assert (
        comparison_parser.parse_args(["--crank-assistance", "0.2"]).resistive_torque
        == -0.2
    )
    assert (
        comparison_parser.parse_args(["--resistive-torque", "0.2"]).resistive_torque
        == 0.2
    )
    with np.testing.assert_raises(SystemExit):
        periodic_parser.parse_args(["--crank-assistance", "-0.2"])

    periodic_budgets = periodic_parser.parse_args(
        [
            "--max-ipopt-iterations",
            "2000",
            "--standard-warmup-max-iterations",
            "10000",
        ]
    )
    comparison_budgets = comparison_parser.parse_args(
        [
            "--ipopt-max-iter",
            "2000",
            "--standard-warmup-max-iter",
            "10000",
        ]
    )
    assert periodic_budgets.max_ipopt_iterations == 2000
    assert periodic_budgets.standard_warmup_max_iterations == 10000
    assert comparison_budgets.ipopt_max_iter == 2000
    assert comparison_budgets.standard_warmup_max_iter == 10000


def test_standard_warmup_cache_metadata_rejects_a_resistance_seed(tmp_path):
    args = periodic_example.build_argument_parser().parse_args([])
    solution = periodic_example._WarmupSolutionAdapter(
        states={
            "q": np.zeros((3, 121)),
            "qdot": np.zeros((3, 121)),
        },
        controls={
            "last_pulse_width_Biceps": np.full(
                (1, args.cycles_per_window * args.stimulations_per_cycle),
                0.0002,
            )
        },
    )
    assisted_path = tmp_path / "assisted_warmup.npz"
    metadata = periodic_example._standard_warmup_metadata(args)
    periodic_example._save_warmup_cache(
        assisted_path,
        solution,
        metadata=metadata,
    )

    loaded = periodic_example._load_warmup_cache(assisted_path)
    assert loaded.metadata == metadata
    periodic_example._validate_standard_warmup_seed(loaded, args, assisted_path)

    resistance_metadata = dict(metadata)
    resistance_metadata["signed_crank_torque_nm"] = 0.2
    resistance_metadata["crank_torque_role"] = "resistive"
    resistance_path = tmp_path / "resistance_warmup.npz"
    periodic_example._save_warmup_cache(
        resistance_path,
        solution,
        metadata=resistance_metadata,
    )
    resistance_seed = periodic_example._load_warmup_cache(resistance_path)

    with np.testing.assert_raises_regex(ValueError, "cannot initialize"):
        periodic_example._validate_standard_warmup_seed(
            resistance_seed,
            args,
            resistance_path,
        )
    periodic_example._validate_standard_warmup_seed(
        resistance_seed,
        args,
        resistance_path,
        allow_torque_continuation=True,
    )

    wrong_magnitude_metadata = dict(metadata)
    wrong_magnitude_metadata["signed_crank_torque_nm"] = -0.21
    wrong_magnitude_path = tmp_path / "wrong_magnitude_warmup.npz"
    periodic_example._save_warmup_cache(
        wrong_magnitude_path,
        solution,
        metadata=wrong_magnitude_metadata,
    )
    wrong_magnitude_seed = periodic_example._load_warmup_cache(wrong_magnitude_path)
    with np.testing.assert_raises_regex(ValueError, "signed crank torque"):
        periodic_example._validate_standard_warmup_seed(
            wrong_magnitude_seed,
            args,
            wrong_magnitude_path,
        )


def test_common_initial_solution_metadata_rejects_an_incompatible_horizon(
    tmp_path,
):
    args = SimpleNamespace(
        model_formulation="periodic_node",
        mechanical_formulation="reduced",
        cycles_per_window=1,
        stimulations_per_cycle=30,
        objective="fatigue",
        objective_shape="quadratic",
        constant_crank_torque=-0.2,
        torque_application="constant",
        enforce_start_constraints=False,
        acados_wheel_q_slack=0.0,
        acados_terminal_wheel_q_slack=0.002,
        terminal_wheel_q_reference_mode="absolute_initial",
        pulse_width_scaling=0.0025,
        pulse_width_active_set="none",
        ode_solver="collocation",
        nlp_ordering_strategy="time_major",
        solver="ipopt",
        warmup_cycles_consumed=1,
    )
    metadata = periodic_example._common_initial_solution_metadata(args)
    metadata["cycles_per_window"] = 2
    seed = periodic_example._WarmupSolutionAdapter({}, {}, metadata=metadata)

    with pytest.raises(ValueError, match="cycles_per_window"):
        periodic_example._validate_common_initial_solution_metadata(
            seed, args, tmp_path / "common.npz"
        )


def test_common_initial_solution_metadata_allows_a_transcription_change(
    tmp_path,
):
    args = SimpleNamespace(
        model_formulation="periodic_node",
        mechanical_formulation="full",
        cycles_per_window=1,
        stimulations_per_cycle=30,
        objective="fatigue",
        objective_shape="quadratic",
        constant_crank_torque=-0.2,
        torque_application="constant",
        enforce_start_constraints=False,
        acados_wheel_q_slack=0.0,
        acados_terminal_wheel_q_slack=0.002,
        terminal_wheel_q_reference_mode="absolute_initial",
        pulse_width_scaling=0.0025,
        pulse_width_active_set="none",
        ode_solver="rk4",
        nlp_ordering_strategy="time_major",
        solver="fatrop",
        warmup_cycles_consumed=1,
    )
    metadata = periodic_example._common_initial_solution_metadata(args)
    metadata["ode_solver"] = "collocation"
    metadata["producer_solver"] = "ipopt"
    seed = periodic_example._WarmupSolutionAdapter({}, {}, metadata=metadata)

    periodic_example._validate_common_initial_solution_metadata(
        seed, args, tmp_path / "common.npz"
    )


def test_common_initial_solution_metadata_allows_a_stricter_start_seed(
    tmp_path,
):
    args = SimpleNamespace(
        model_formulation="periodic_node",
        mechanical_formulation="reduced",
        cycles_per_window=1,
        stimulations_per_cycle=30,
        objective="fatigue",
        objective_shape="quadratic",
        constant_crank_torque=0.0,
        torque_application="constant",
        enforce_start_constraints=False,
        acados_wheel_q_slack=0.0,
        acados_terminal_wheel_q_slack=0.002,
        terminal_wheel_q_reference_mode="absolute_initial",
        pulse_width_scaling=0.0025,
        pulse_width_active_set="none",
        ode_solver="collocation",
        nlp_ordering_strategy="time_major",
        solver="acados",
        warmup_cycles_consumed=1,
    )
    metadata = periodic_example._common_initial_solution_metadata(args)
    metadata["enforce_start_constraints"] = True
    seed = periodic_example._WarmupSolutionAdapter({}, {}, metadata=metadata)

    periodic_example._validate_common_initial_solution_metadata(
        seed, args, tmp_path / "strict-common.npz"
    )


def test_common_initial_solution_metadata_rejects_a_looser_start_seed(
    tmp_path,
):
    args = SimpleNamespace(
        model_formulation="periodic_node",
        mechanical_formulation="reduced",
        cycles_per_window=1,
        stimulations_per_cycle=30,
        objective="fatigue",
        objective_shape="quadratic",
        constant_crank_torque=0.0,
        torque_application="constant",
        enforce_start_constraints=True,
        acados_wheel_q_slack=0.0,
        acados_terminal_wheel_q_slack=0.002,
        terminal_wheel_q_reference_mode="absolute_initial",
        pulse_width_scaling=0.0025,
        pulse_width_active_set="none",
        ode_solver="collocation",
        nlp_ordering_strategy="time_major",
        solver="ipopt",
        warmup_cycles_consumed=1,
    )
    metadata = periodic_example._common_initial_solution_metadata(args)
    metadata["enforce_start_constraints"] = False
    seed = periodic_example._WarmupSolutionAdapter({}, {}, metadata=metadata)

    with pytest.raises(ValueError, match="enforce_start_constraints"):
        periodic_example._validate_common_initial_solution_metadata(
            seed, args, tmp_path / "loose-common.npz"
        )


def test_common_initial_solution_metadata_allows_a_stricter_terminal_seed(
    tmp_path,
):
    args = SimpleNamespace(
        model_formulation="periodic_node",
        mechanical_formulation="full",
        cycles_per_window=1,
        stimulations_per_cycle=30,
        objective="fatigue",
        objective_shape="quadratic",
        constant_crank_torque=0.0,
        torque_application="constant",
        enforce_start_constraints=False,
        acados_wheel_q_slack=0.0,
        acados_terminal_wheel_q_slack=0.01,
        terminal_wheel_q_reference_mode="absolute_initial",
        pulse_width_scaling=0.0025,
        pulse_width_active_set="none",
        ode_solver="collocation",
        nlp_ordering_strategy="time_major",
        solver="acados",
        warmup_cycles_consumed=1,
    )
    metadata = periodic_example._common_initial_solution_metadata(args)
    metadata["terminal_wheel_q_slack"] = 0.002
    seed = periodic_example._WarmupSolutionAdapter({}, {}, metadata=metadata)

    periodic_example._validate_common_initial_solution_metadata(
        seed, args, tmp_path / "strict-terminal-common.npz"
    )


def test_common_initial_solution_metadata_records_the_terminal_homotopy_target():
    args = SimpleNamespace(
        model_formulation="periodic_node",
        mechanical_formulation="full",
        cycles_per_window=1,
        stimulations_per_cycle=30,
        objective="fatigue",
        objective_shape="quadratic",
        constant_crank_torque=0.0,
        torque_application="constant",
        enforce_start_constraints=False,
        acados_wheel_q_slack=0.0,
        acados_terminal_wheel_q_slack=0.01,
        acados_terminal_wheel_q_homotopy_slacks=(0.01, 0.005, 0.002),
        terminal_wheel_q_reference_mode="absolute_initial",
        pulse_width_scaling=0.0025,
        pulse_width_active_set="none",
        ode_solver="collocation",
        nlp_ordering_strategy="time_major",
        solver="acados",
        warmup_cycles_consumed=1,
    )

    metadata = periodic_example._common_initial_solution_metadata(args)

    assert metadata["terminal_wheel_q_slack"] == 0.002
    assert metadata["terminal_wheel_q_initial_slack"] == 0.01
    assert metadata["terminal_wheel_q_homotopy_slacks"] == [0.01, 0.005, 0.002]


def test_common_initial_solution_metadata_rejects_a_looser_terminal_seed(
    tmp_path,
):
    args = SimpleNamespace(
        model_formulation="periodic_node",
        mechanical_formulation="full",
        cycles_per_window=1,
        stimulations_per_cycle=30,
        objective="fatigue",
        objective_shape="quadratic",
        constant_crank_torque=0.0,
        torque_application="constant",
        enforce_start_constraints=False,
        acados_wheel_q_slack=0.0,
        acados_terminal_wheel_q_slack=0.002,
        terminal_wheel_q_reference_mode="absolute_initial",
        pulse_width_scaling=0.0025,
        pulse_width_active_set="none",
        ode_solver="collocation",
        nlp_ordering_strategy="time_major",
        solver="acados",
        warmup_cycles_consumed=1,
    )
    metadata = periodic_example._common_initial_solution_metadata(args)
    metadata["terminal_wheel_q_slack"] = 0.01
    seed = periodic_example._WarmupSolutionAdapter({}, {}, metadata=metadata)

    with pytest.raises(ValueError, match="terminal_wheel_q_slack"):
        periodic_example._validate_common_initial_solution_metadata(
            seed, args, tmp_path / "loose-terminal-common.npz"
        )


def test_common_initial_solution_metadata_allows_exact_reduced_to_full_lift(
    tmp_path,
):
    args = SimpleNamespace(
        model_formulation="periodic_node",
        mechanical_formulation="full",
        cycles_per_window=1,
        stimulations_per_cycle=30,
        objective="fatigue",
        objective_shape="quadratic",
        constant_crank_torque=-0.2,
        torque_application="constant",
        enforce_start_constraints=True,
        acados_wheel_q_slack=0.0,
        acados_terminal_wheel_q_slack=0.002,
        terminal_wheel_q_reference_mode="absolute_initial",
        pulse_width_scaling=0.0025,
        pulse_width_active_set="none",
        ode_solver="collocation",
        nlp_ordering_strategy="time_major",
        solver="ipopt",
        warmup_cycles_consumed=1,
    )
    metadata = periodic_example._common_initial_solution_metadata(args)
    metadata["mechanical_formulation"] = "reduced"
    seed = periodic_example._WarmupSolutionAdapter(
        {"theta": np.zeros((1, 2)), "omega": -np.ones((1, 2))},
        {},
        metadata=metadata,
    )

    periodic_example._validate_common_initial_solution_metadata(
        seed, args, tmp_path / "common-reduced.npz"
    )


def test_common_initial_solution_metadata_rejects_a_different_warmup_cycle(
    tmp_path,
):
    args = SimpleNamespace(
        model_formulation="periodic_node",
        mechanical_formulation="full",
        cycles_per_window=1,
        stimulations_per_cycle=30,
        objective="fatigue",
        objective_shape="quadratic",
        constant_crank_torque=-0.2,
        torque_application="constant",
        enforce_start_constraints=False,
        acados_wheel_q_slack=0.0,
        acados_terminal_wheel_q_slack=0.002,
        terminal_wheel_q_reference_mode="absolute_initial",
        pulse_width_scaling=0.0025,
        pulse_width_active_set="none",
        ode_solver="collocation",
        nlp_ordering_strategy="time_major",
        solver="madnlp",
        warmup_cycles_consumed=1,
    )
    metadata = periodic_example._common_initial_solution_metadata(args)
    metadata["warmup_cycles_consumed"] = 0
    seed = periodic_example._WarmupSolutionAdapter({}, {}, metadata=metadata)

    with pytest.raises(ValueError, match="warmup_cycles_consumed"):
        periodic_example._validate_common_initial_solution_metadata(
            seed, args, tmp_path / "common.npz"
        )


def test_common_initial_solution_can_explicitly_adopt_seed_warmup_chronology(
    tmp_path,
):
    args = SimpleNamespace(warmup_cycles_consumed=0)
    seed = periodic_example._WarmupSolutionAdapter(
        {},
        {},
        metadata={"warmup_cycles_consumed": 1},
    )

    adopted = periodic_example._adopt_common_initial_solution_warmup_cycles(
        seed, args, tmp_path / "rho-prefix.npz"
    )

    assert adopted == 1
    assert args.warmup_cycles_consumed == 1


@pytest.mark.parametrize("invalid_value", (None, -1, 1.5, True))
def test_common_initial_solution_rejects_invalid_adopted_warmup_chronology(
    tmp_path, invalid_value
):
    args = SimpleNamespace(warmup_cycles_consumed=0)
    seed = periodic_example._WarmupSolutionAdapter(
        {},
        {},
        metadata={"warmup_cycles_consumed": invalid_value},
    )

    with pytest.raises(ValueError, match="warmup_cycles_consumed"):
        periodic_example._adopt_common_initial_solution_warmup_cycles(
            seed, args, tmp_path / "rho-prefix.npz"
        )


def test_receding_horizon_solution_is_exported_as_one_multi_cycle_seed(
    tmp_path,
):
    args = periodic_example.build_argument_parser().parse_args([])
    args.single_shot = False
    args.cycles_per_window = 1
    args.n_windows = 2
    args.terminal_wheel_q_reference_mode = "absolute_initial"
    output_path = tmp_path / "two_cycle_rho_seed.npz"
    summary = {
        "success": True,
        "covered_cycles": 2,
        "state_traces": {
            "theta": np.array([[0.0, -2.0 * np.pi, -4.0 * np.pi]]),
            "omega": np.array([[-2.0 * np.pi] * 3]),
        },
        "control_traces": {"Biceps": np.array([[150e-6, 160e-6]])},
        "state_boundary_jumps": {
            "available": True,
            "boundary_count": 1,
            "by_state": {
                "theta": {"maximum_absolute_jump": 1e-10},
                "omega": {"maximum_absolute_jump": 2e-10},
            },
        },
    }

    periodic_example._save_receding_horizon_solution(output_path, summary, args)
    seed = periodic_example._load_warmup_cache(output_path)

    assert seed.metadata["cycles_per_window"] == 2
    assert seed.metadata["producer_mode"] == "receding_horizon_concatenation"
    assert seed.metadata["producer_cycles_per_window"] == 1
    assert seed.metadata["producer_requested_cycles"] == 2
    assert (
        seed.metadata["state_boundary_maximum_absolute_jump"]
        == pytest.approx(2e-10)
    )
    np.testing.assert_allclose(
        seed.decision_states()["theta"],
        summary["state_traces"]["theta"],
    )


def test_receding_horizon_solution_rejects_an_incomplete_physical_prefix(
    tmp_path,
):
    args = periodic_example.build_argument_parser().parse_args([])
    args.single_shot = False
    args.cycles_per_window = 1
    args.n_windows = 100
    summary = {
        "success": False,
        "covered_cycles": 99,
        "state_traces": {"theta": np.zeros((1, 2))},
        "control_traces": {"Biceps": np.zeros((1, 1))},
    }

    with pytest.raises(RuntimeError, match="99/100"):
        periodic_example._save_receding_horizon_solution(
            tmp_path / "invalid.npz", summary, args
        )


def test_receding_horizon_solution_can_export_an_explicit_partial_prefix(
    tmp_path,
):
    args = periodic_example.build_argument_parser().parse_args([])
    args.single_shot = False
    args.cycles_per_window = 1
    args.n_windows = 3
    args.allow_partial_receding_horizon_solution_output = True
    args.terminal_wheel_q_reference_mode = "absolute_initial"
    output_path = tmp_path / "two_of_three_cycle_rho_seed.npz"
    summary = {
        "success": False,
        "covered_cycles": 2,
        "state_traces": {
            "theta": np.array([[0.0, -2.0 * np.pi, -4.0 * np.pi]]),
            "omega": np.array([[-2.0 * np.pi] * 3]),
        },
        "control_traces": {"Biceps": np.array([[150e-6, 160e-6]])},
        "state_boundary_jumps": {
            "available": True,
            "boundary_count": 1,
            "by_state": {
                "theta": {"maximum_absolute_jump": 0.0},
                "omega": {"maximum_absolute_jump": 0.0},
            },
        },
    }

    periodic_example._save_receding_horizon_solution(output_path, summary, args)
    seed = periodic_example._load_warmup_cache(output_path)

    assert seed.metadata["cycles_per_window"] == 2
    assert seed.metadata["producer_requested_cycles"] == 3


def test_periodic_node_projection_uses_all_five_ding_states():
    keys = {
        "Cn_Biceps",
        "F_Biceps",
        "A_Biceps",
        "Tau1_Biceps",
        "Km_Biceps",
    }

    assert periodic_example._projection_state_keys(
        "Biceps", "all", available_keys=keys
    ) == (
        "Cn_Biceps",
        "F_Biceps",
        "A_Biceps",
        "Tau1_Biceps",
        "Km_Biceps",
    )


def test_phase_one_maps_reduced_mechanics_without_classifying_them_as_fes():
    class Variable:
        def __init__(self, index):
            self.index = index

    nlp = SimpleNamespace(
        states={
            "theta": Variable([0]),
            "omega": Variable([1]),
            "Cn_Biceps": Variable([2]),
            "F_Biceps": Variable([3]),
        }
    )

    blocks = periodic_example._phase_one_state_keys(nlp)

    assert blocks == {
        "q": ("theta",),
        "qdot": ("omega",),
        "fes": ("Cn_Biceps", "F_Biceps"),
    }
    scales = periodic_example._full_dynamics_defect_state_scales(
        nlp,
        np.array(
            [
                [-100.0, -106.0],
                [-6.0, -6.1],
                [0.2, 0.3],
                [5.0, 6.0],
            ]
        ),
    )
    np.testing.assert_allclose(scales[0], 2.0 * np.pi)


def test_legacy_warmup_requires_an_explicit_compatible_torque_assertion(
    tmp_path,
):
    args = periodic_example.build_argument_parser().parse_args([])
    legacy_path = tmp_path / "legacy_warmup.npz"
    solution = periodic_example._WarmupSolutionAdapter(
        states={
            "q": np.zeros((3, 241)),
            "qdot": np.zeros((3, 241)),
        },
        controls={
            "last_pulse_width_Biceps": np.full(
                (1, args.cycles_per_window * args.stimulations_per_cycle),
                0.0002,
            )
        },
    )
    periodic_example._save_warmup_cache(legacy_path, solution)
    loaded = periodic_example._load_warmup_cache(legacy_path)

    with np.testing.assert_raises_regex(ValueError, "no physical metadata"):
        periodic_example._validate_standard_warmup_seed(loaded, args, legacy_path)

    periodic_example._attach_declared_legacy_warmup_metadata(
        loaded,
        args,
        legacy_path,
        declared_signed_torque_nm=-0.2,
    )
    periodic_example._validate_standard_warmup_seed(loaded, args, legacy_path)

    loaded.metadata = None
    periodic_example._attach_declared_legacy_warmup_metadata(
        loaded,
        args,
        legacy_path,
        declared_signed_torque_nm=0.2,
    )
    with np.testing.assert_raises_regex(ValueError, "cannot initialize"):
        periodic_example._validate_standard_warmup_seed(loaded, args, legacy_path)


def test_legacy_warmup_can_be_truncated_to_a_shorter_integer_cycle_horizon():
    args = periodic_example.build_argument_parser().parse_args(
        ["--cycles-per-window", "1"]
    )
    solution = periodic_example._WarmupSolutionAdapter(
        states={"q": np.arange(241, dtype=float)[None, :]},
        controls={"last_pulse_width_Biceps": np.arange(60, dtype=float)[None, :]},
    )

    periodic_example._attach_declared_legacy_warmup_metadata(
        solution,
        args,
        Path("legacy_two_cycle_seed.npz"),
        declared_signed_torque_nm=-0.2,
    )

    assert solution.decision_controls()["last_pulse_width_Biceps"].shape == (
        1,
        30,
    )
    assert solution.decision_states()["q"].shape == (1, 121)
    assert solution.metadata["legacy_source_control_nodes"] == 60
    assert solution.metadata["legacy_truncated"] is True


def test_standard_warmup_cache_signature_separates_assistance_and_resistance(
    tmp_path,
):
    parser = periodic_example.build_argument_parser()
    assisted = parser.parse_args(["--crank-assistance", "0.2"])
    resistive = parser.parse_args(["--signed-crank-torque", "0.2"])
    model_path = tmp_path / "model.bioMod"
    model_path.write_text("version 4\n")
    conditions = {"scenario": "signed-torque"}
    cycling_info = {"resistive_torque": object()}

    assisted_signature = periodic_example._warmup_cache_signature(
        assisted,
        model_path,
        conditions,
        cycling_info,
    )
    resistive_signature = periodic_example._warmup_cache_signature(
        resistive,
        model_path,
        conditions,
        cycling_info,
    )

    assert assisted_signature != resistive_signature


def test_source_stamp_is_portable_and_content_addressed(tmp_path):
    first = tmp_path / "runner_a" / "model.bioMod"
    second = tmp_path / "runner_b" / "model.bioMod"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("version 4\n")
    second.write_text("version 4\n")
    first.touch()
    second.touch()

    assert periodic_example._source_stamp(first) == periodic_example._source_stamp(
        second
    )

    second.write_text("version 4\n// changed\n")
    assert periodic_example._source_stamp(first) != periodic_example._source_stamp(
        second
    )


def test_compact_rho_output_concatenates_trajectories_without_an_ocp():
    nmpc = object.__new__(FesNmpcMsk)
    nmpc._compact_solution_output = True
    nmpc.nlp = [
        SimpleNamespace(states={"q": None}, controls={"u": None}, parameters={})
    ]

    solution = nmpc._initialize_solution(
        dt=0.1,
        states=[
            {"q": np.array([[0.0, 1.0]])},
            {"q": np.array([[1.0, 2.0]])},
            {"q": np.array([[2.0]])},
        ],
        controls=[{"u": np.array([[3.0]])}, {"u": np.array([[4.0]])}],
        parameters=[],
    )

    assert isinstance(solution, CompactNmpcSolution)
    np.testing.assert_array_equal(solution.decision_states()["q"], [[0.0, 1.0, 2.0]])
    np.testing.assert_array_equal(solution.decision_controls()["u"], [[3.0, 4.0]])


def test_acados_cyclical_transfer_extrapolates_by_default():
    args = periodic_example.build_argument_parser().parse_args([])

    assert args.acados_cyclical_transfer_mode == "extrapolate"
    assert args.acados_control_homotopy_stage_iterations == 100


def test_periodic_ipopt_window_cache_paths_are_window_specific(tmp_path, monkeypatch):
    args = periodic_example.build_argument_parser().parse_args([])
    model_path = tmp_path / "cycling.bioMod"
    model_path.write_text("version 4\n")
    monkeypatch.setattr(periodic_example, "_cache_root", lambda: tmp_path)

    first = periodic_example._periodic_ipopt_window_refinement_cache_path(
        args, model_path, 1
    )
    second = periodic_example._periodic_ipopt_window_refinement_cache_path(
        args, model_path, 2
    )

    assert first.parent == tmp_path
    assert first.name.endswith("_window_0001.npz")
    assert second.name.endswith("_window_0002.npz")
    assert first != second


def test_strict_fes_continuity_uses_a_distinct_periodic_ipopt_cache(
    tmp_path, monkeypatch
):
    parser = periodic_example.build_argument_parser()
    relaxed_args = parser.parse_args([])
    strict_args = parser.parse_args(["--acados-bind-first-node-fes-states"])
    model_path = tmp_path / "cycling.bioMod"
    model_path.write_text("version 4\n")
    monkeypatch.setattr(periodic_example, "_cache_root", lambda: tmp_path)

    relaxed = periodic_example._periodic_ipopt_refinement_cache_path(
        relaxed_args, model_path
    )
    strict = periodic_example._periodic_ipopt_refinement_cache_path(
        strict_args, model_path
    )

    assert relaxed != strict


def test_target_integrator_uses_a_distinct_periodic_ipopt_cache(tmp_path, monkeypatch):
    parser = periodic_example.build_argument_parser()
    rk4_args = parser.parse_args(
        [
            "--periodic-ipopt-refinement-ode-solver",
            "target",
            "--ode-solver",
            "rk4",
        ]
    )
    collocation_args = parser.parse_args(
        [
            "--periodic-ipopt-refinement-ode-solver",
            "target",
            "--ode-solver",
            "collocation",
        ]
    )
    model_path = tmp_path / "cycling.bioMod"
    model_path.write_text("version 4\n")
    monkeypatch.setattr(periodic_example, "_cache_root", lambda: tmp_path)

    rk4 = periodic_example._periodic_ipopt_refinement_cache_path(rk4_args, model_path)
    collocation = periodic_example._periodic_ipopt_refinement_cache_path(
        collocation_args, model_path
    )

    assert rk4 != collocation


def test_reduced_mechanics_uses_a_distinct_periodic_ipopt_cache(tmp_path, monkeypatch):
    parser = periodic_example.build_argument_parser()
    full_args = parser.parse_args([])
    reduced_args = parser.parse_args(["--mechanical-formulation", "reduced"])
    model_path = tmp_path / "cycling.bioMod"
    model_path.write_text("version 4\n")
    monkeypatch.setattr(periodic_example, "_cache_root", lambda: tmp_path)

    full = periodic_example._periodic_ipopt_refinement_cache_path(full_args, model_path)
    reduced = periodic_example._periodic_ipopt_refinement_cache_path(
        reduced_args, model_path
    )

    assert full != reduced


def test_standard_warmup_projects_reduced_mechanics_and_clips_pw():
    class FakeKinematics:
        @staticmethod
        def project_generalized_trajectory(q, qdot):
            return (
                q[2:3, :],
                qdot[2:3, :],
                {"maximum_configuration_projection_error_rad": 0.0},
            )

    warmup = periodic_example._WarmupSolutionAdapter(
        states={
            "q": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -1.0, -2.0]]),
            "qdot": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]),
        },
        controls={"last_pulse_width_Biceps": np.array([[0.0, 0.0007]])},
    )
    target_model = SimpleNamespace(
        reduced_dynamics=SimpleNamespace(kinematics=FakeKinematics()),
        muscles_dynamics_model=[SimpleNamespace(muscle_name="Biceps", pd0=0.000131405)],
    )
    target = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                model=target_model,
                x_init={
                    "theta": SimpleNamespace(init=np.zeros((1, 3))),
                    "omega": SimpleNamespace(init=np.zeros((1, 3))),
                },
                u_init={
                    "last_pulse_width_Biceps": SimpleNamespace(init=np.zeros((1, 2)))
                },
            )
        ]
    )

    with pytest.warns(RuntimeWarning, match="physical Ding bounds"):
        adapted = periodic_example._adapt_warmup_solution_to_periodic_nodes(
            target, warmup
        )

    assert set(adapted.decision_states()) == {"theta", "omega"}
    np.testing.assert_allclose(adapted.decision_states()["theta"], [[0.0, -1.0, -2.0]])
    np.testing.assert_allclose(
        adapted.decision_controls()["last_pulse_width_Biceps"],
        [[0.000131405, 0.0006]],
    )


def test_reduced_common_seed_is_lifted_exactly_for_full_mechanics():
    class FakeKinematics:
        @staticmethod
        def lift_generalized_trajectory(theta, omega):
            return (
                np.vstack((theta, 2.0 * theta, 3.0 * theta)),
                np.vstack((omega, 2.0 * omega, 3.0 * omega)),
            )

    warmup = periodic_example._WarmupSolutionAdapter(
        states={
            "theta": np.array([[0.0, -1.0, -2.0]]),
            "omega": np.array([[-4.0, -5.0, -6.0]]),
        },
        controls={"last_pulse_width_Biceps": np.full((1, 2), 0.0002)},
    )
    target = SimpleNamespace(
        _cocofest_mechanical_equivalence_dynamics=SimpleNamespace(
            kinematics=FakeKinematics()
        ),
        nlp=[
            SimpleNamespace(
                model=SimpleNamespace(
                    muscles_dynamics_model=[
                        SimpleNamespace(muscle_name="Biceps", pd0=0.000131405)
                    ]
                ),
                x_init={
                    "q": SimpleNamespace(init=np.zeros((3, 3))),
                    "qdot": SimpleNamespace(init=np.zeros((3, 3))),
                },
                u_init={
                    "last_pulse_width_Biceps": SimpleNamespace(init=np.zeros((1, 2)))
                },
            )
        ],
    )

    adapted = periodic_example._adapt_warmup_solution_to_periodic_nodes(target, warmup)

    np.testing.assert_allclose(
        adapted.decision_states()["q"],
        [[0.0, -1.0, -2.0], [0.0, -2.0, -4.0], [0.0, -3.0, -6.0]],
    )
    np.testing.assert_allclose(
        adapted.decision_states()["qdot"],
        [[-4.0, -5.0, -6.0], [-8.0, -10.0, -12.0], [-12.0, -15.0, -18.0]],
    )


def test_warmup_state_resampling_refines_collocation_grid_and_keeps_endpoints():
    values = np.vstack(
        (
            np.linspace(0.0, 1.0, 121),
            np.linspace(-2.0, 3.0, 121),
        )
    )

    refined = periodic_example._resample_warmup_data(
        values, target_len=181, has_terminal_node=True
    )

    assert refined.shape == (2, 181)
    np.testing.assert_array_equal(refined[:, 0], values[:, 0])
    np.testing.assert_array_equal(refined[:, -1], values[:, -1])
    np.testing.assert_allclose(refined[:, 90], [0.5, 0.5])


def test_mechanical_equivalence_audit_rejects_off_manifold_full_motion():
    from cocofest.dynamics.reduced_cycling import ReducedCyclingKinematics

    theta = np.linspace(0.0, -2.0 * np.pi, 31)
    q_samples = np.vstack(
        (
            0.8 + 0.2 * np.cos(theta),
            1.4 + 0.3 * np.sin(theta),
            theta + 0.1 * np.sin(theta),
        )
    )
    kinematics = ReducedCyclingKinematics.fit(theta, q_samples, order=2)
    omega = -5.0 * np.ones_like(theta)
    q, qdot = kinematics.lift_generalized_trajectory(theta, omega)
    q[0, 10] += 0.05
    qdot[1, 15] += 0.5
    summary = {
        "state_traces": {"q": q, "qdot": qdot},
        "diagnostics": {"is_physical": True, "issues": []},
        "physical_success": True,
        "success": True,
    }

    periodic_example.attach_mechanical_equivalence_audit(
        summary,
        SimpleNamespace(kinematics=kinematics),
    )

    assert summary["mechanical_equivalence_audit"]["available"]
    assert not summary["mechanical_equivalence_audit"]["passes_tolerance"]
    assert not summary["physical_success"]
    assert summary["nlp_crank_diagnostics"] == {
        "is_physical": True,
        "issues": [],
    }
    assert (
        "mechanical_trajectory_off_reduced_manifold" in summary["diagnostics"]["issues"]
    )
    assert summary["nlp_crank_diagnostics"]["issues"] == []
    assert summary["physical_crank_angle_trace"].shape == theta.shape
    assert summary["physical_crank_velocity_trace"].shape == omega.shape


def test_control_homotopy_stops_on_failure_and_restores_bounds(monkeypatch):
    class FakeSolver:
        def set_maximum_iterations(self, value):
            self.max_iterations = value

        def set_convergence_tolerance(self, value):
            self.tolerance = value

    bounds = SimpleNamespace(
        min=np.array([[0.1, 0.1, 0.1]]), max=np.array([[0.6, 0.6, 0.6]])
    )
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                u_init={
                    "last_pulse_width_Biceps": SimpleNamespace(
                        init=np.array([[0.2, 0.4, 0.5]])
                    )
                },
                u_bounds={"last_pulse_width_Biceps": bounds},
            )
        ]
    )
    solutions = iter(
        [
            SimpleNamespace(
                status=2,
                residuals=np.array([4e-4, 1e-6, 0.0, 1e-6]),
                solver_time_to_optimize=1.0,
                real_time_to_optimize=1.1,
            ),
            SimpleNamespace(
                status=0,
                residuals=np.zeros(4),
                solver_time_to_optimize=2.0,
                real_time_to_optimize=2.1,
            ),
            SimpleNamespace(
                status=2,
                residuals=np.array([2.0, 0.1, 0.0, 0.0]),
                solver_time_to_optimize=3.0,
                real_time_to_optimize=3.1,
            ),
        ]
    )
    stage_bounds = []

    def solve_stage():
        nodewise = getattr(nmpc, "_cocofest_nodewise_control_bounds", {})
        stage_bounds.append(
            {
                key: (lower.copy(), upper.copy())
                for key, (lower, upper) in nodewise.items()
            }
        )
        return next(solutions)

    applied_statuses = []
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda solution: {"residuals": solution.residuals},
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda _nmpc, solution: applied_statuses.append(solution.status),
    )

    summaries = periodic_example.run_acados_control_homotopy(
        nmpc,
        FakeSolver(),
        radii=(0.01, 0.02),
        convergence_tolerance=5e-4,
        fixed_control_tolerance=1e-8,
        echo=False,
        solve_stage=solve_stage,
        max_restarts=0,
        stage_iterations=25,
    )

    assert [summary["accepted"] for summary in summaries] == [
        True,
        True,
        False,
    ]
    assert applied_statuses == [2, 0]
    assert stage_bounds[0] == {}
    np.testing.assert_allclose(
        stage_bounds[1]["last_pulse_width_Biceps"][0], [[0.19, 0.39, 0.49]]
    )
    np.testing.assert_allclose(bounds.min, [[0.1, 0.1, 0.1]])
    np.testing.assert_allclose(bounds.max, [[0.6, 0.6, 0.6]])
    assert nmpc._cocofest_fix_controls_to_warmup is False
    assert nmpc._cocofest_nodewise_control_bounds == {}
    assert summaries[0]["stage"] == 0


def test_control_homotopy_restarts_a_nearly_feasible_stage(monkeypatch):
    class FakeSolver:
        def set_convergence_tolerance(self, value):
            self.tolerance = value

        def set_maximum_iterations(self, value):
            self.max_iterations = value

    bounds = SimpleNamespace(min=np.array([[0.1, 0.1]]), max=np.array([[0.6, 0.6]]))
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                u_init={
                    "last_pulse_width_Biceps": SimpleNamespace(
                        init=np.array([[0.2, 0.4]])
                    )
                },
                u_bounds={"last_pulse_width_Biceps": bounds},
            )
        ]
    )
    solutions = iter(
        [
            SimpleNamespace(
                status=2,
                residuals=np.array([0.2, 1e-6, 0.0, 1e-5]),
                solver_time_to_optimize=1.0,
                real_time_to_optimize=1.1,
            ),
            SimpleNamespace(
                status=0,
                residuals=np.array([4e-4, 1e-7, 0.0, 1e-6]),
                solver_time_to_optimize=2.0,
                real_time_to_optimize=2.1,
            ),
            SimpleNamespace(
                status=0,
                residuals=np.zeros(4),
                solver_time_to_optimize=3.0,
                real_time_to_optimize=3.1,
            ),
        ]
    )
    applied_statuses = []
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda solution: {"residuals": solution.residuals},
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda _nmpc, solution: applied_statuses.append(solution.status),
    )

    summaries = periodic_example.run_acados_control_homotopy(
        nmpc,
        FakeSolver(),
        radii=(0.01,),
        convergence_tolerance=5e-4,
        fixed_control_tolerance=1e-8,
        max_restarts=1,
        stage_iterations=25,
        echo=False,
        solve_stage=lambda: next(solutions),
    )

    assert [(item["stage"], item["attempt"]) for item in summaries] == [
        (0, 0),
        (0, 1),
        (1, 0),
    ]
    assert [item["accepted"] for item in summaries] == [False, True, True]
    assert summaries[0]["restartable"] is True
    assert applied_statuses == [2, 0, 0]


def test_control_homotopy_can_relax_stationarity_without_relaxing_feasibility(
    monkeypatch,
):
    configured_tolerances = []

    class FakeSolver:
        def set_convergence_tolerance(self, value):
            configured_tolerances.append(("all", value))

        def set_nlp_solver_tol_stat(self, value):
            configured_tolerances.append(("stationarity", value))

    nmpc = SimpleNamespace(nlp=[SimpleNamespace(u_init={}, u_bounds={})])
    solution = SimpleNamespace(
        status=2,
        residuals=np.array([3.4e-3, 8e-9, 3e-11, 2e-5]),
        solver_time_to_optimize=1.0,
        real_time_to_optimize=1.1,
    )
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda result: {"residuals": result.residuals},
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda *_args: None,
    )

    summaries = periodic_example.run_acados_control_homotopy(
        nmpc,
        FakeSolver(),
        radii=(),
        convergence_tolerance=5e-4,
        stationarity_tolerance=5e-3,
        fixed_control_tolerance=1e-8,
        max_restarts=0,
        stage_iterations=None,
        echo=False,
        solve_stage=lambda: solution,
    )

    assert configured_tolerances == [("all", 5e-4), ("stationarity", 5e-3)]
    assert summaries[0]["accepted"] is True
    assert summaries[0]["feasibility_tolerance"] == 5e-4
    assert summaries[0]["stationarity_tolerance"] == 5e-3


def test_proximal_control_continuation_reduces_weight_without_changing_bounds(
    monkeypatch,
):
    class FakeSolver:
        def set_convergence_tolerance(self, value):
            self.tolerance = value

        def set_maximum_iterations(self, value):
            self.max_iterations = value

    solutions = iter(
        [
            SimpleNamespace(
                status=0,
                residuals=np.zeros(4),
                solver_time_to_optimize=1.0,
                real_time_to_optimize=1.1,
            )
            for _ in range(3)
        ]
    )
    bounds = SimpleNamespace(
        min=np.array([[0.1, 0.1]]),
        max=np.array([[0.6, 0.6]]),
    )
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                u_bounds={"last_pulse_width_Biceps": bounds},
            )
        ]
    )
    applied_weights = []
    applied_statuses = []
    monkeypatch.setattr(
        periodic_example,
        "set_acados_runtime_control_regularization_weight",
        lambda _nmpc, weight: (
            applied_weights.append(weight)
            or {"applied": True, "reason": None, "weight": weight}
        ),
    )
    monkeypatch.setattr(
        periodic_example,
        "set_acados_runtime_max_iterations",
        lambda _nmpc, _iterations: True,
    )
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda solution: {"residuals": solution.residuals},
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda _nmpc, solution: applied_statuses.append(solution.status),
    )

    summaries = periodic_example.run_acados_proximal_control_continuation(
        nmpc,
        FakeSolver(),
        weights=(1e6, 1e5, 1e4),
        convergence_tolerance=5e-4,
        max_restarts=0,
        stage_iterations=20,
        echo=False,
        solve_stage=lambda: next(solutions),
    )

    assert [summary["accepted"] for summary in summaries] == [True, True, True]
    assert applied_weights == [1e6, 1e5, 1e4, 1e4]
    assert applied_statuses == [0, 0, 0]
    assert nmpc._cocofest_dual_warm_start_mode == "preserve"
    np.testing.assert_allclose(bounds.min, [[0.1, 0.1]])
    np.testing.assert_allclose(bounds.max, [[0.6, 0.6]])


def test_proximal_control_continuation_restarts_from_best_failed_qp_iterate(
    monkeypatch,
):
    class FakeSolver:
        def set_convergence_tolerance(self, value):
            self.tolerance = value

    solutions = iter(
        [
            SimpleNamespace(
                status=4,
                solver_time_to_optimize=1.0,
                real_time_to_optimize=1.1,
            ),
            SimpleNamespace(
                status=0,
                solver_time_to_optimize=2.0,
                real_time_to_optimize=2.1,
            ),
        ]
    )
    nmpc = SimpleNamespace(nlp=[SimpleNamespace(u_bounds={})])
    restored_iterates = []
    reset_calls = []
    monkeypatch.setattr(
        periodic_example,
        "set_acados_runtime_control_regularization_weight",
        lambda _nmpc, weight: {"applied": True, "weight": weight},
    )
    monkeypatch.setattr(
        periodic_example,
        "set_acados_runtime_max_iterations",
        lambda _nmpc, _iterations: True,
    )
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda solution: (
            {
                "residuals": np.array([1e8, 10.0, 0.0, 1e6]),
                "res_stat_all": np.array([3.0, 0.2, 1e8]),
                "res_eq_all": np.array([1.0, 2e-3, 10.0]),
                "res_ineq_all": np.zeros(3),
                "res_comp_all": np.array([0.1, 1e-5, 1e6]),
            }
            if solution.status == 4
            else {"residuals": np.zeros(4)}
        ),
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_acados_capsule_primal_to_initial_guess",
        lambda _nmpc, iterate_index=None: (
            restored_iterates.append(iterate_index)
            or {"applied": True, "iterate_index": iterate_index}
        ),
    )
    monkeypatch.setattr(
        periodic_example,
        "reset_acados_solver_memory",
        lambda _nmpc: reset_calls.append(True) or True,
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda _nmpc, _solution: None,
    )

    summaries = periodic_example.run_acados_proximal_control_continuation(
        nmpc,
        FakeSolver(),
        weights=(1e6,),
        convergence_tolerance=5e-4,
        max_restarts=1,
        restart_feasibility_factor=5.0,
        stage_iterations=20,
        echo=False,
        solve_stage=lambda: next(solutions),
    )

    assert [summary["status"] for summary in summaries] == [4, 0]
    assert summaries[0]["restartable"] is True
    assert restored_iterates == [1]
    assert reset_calls == [True]


def test_proximal_control_continuation_can_fallback_to_a_lower_weight(
    monkeypatch,
):
    class FakeSolver:
        def set_convergence_tolerance(self, value):
            self.tolerance = value

    solutions = iter(
        [
            SimpleNamespace(
                status=4,
                residuals=np.array([1.0, 1.0, 0.0, 1e-4]),
                solver_time_to_optimize=1.0,
                real_time_to_optimize=1.1,
            ),
            SimpleNamespace(
                status=0,
                residuals=np.zeros(4),
                solver_time_to_optimize=2.0,
                real_time_to_optimize=2.1,
            ),
        ]
    )
    nmpc = SimpleNamespace(nlp=[SimpleNamespace(u_bounds={})])
    applied_weights = []
    monkeypatch.setattr(
        periodic_example,
        "set_acados_runtime_control_regularization_weight",
        lambda _nmpc, weight: (
            applied_weights.append(weight)
            or {"applied": True, "reason": None, "weight": weight}
        ),
    )
    monkeypatch.setattr(
        periodic_example,
        "set_acados_runtime_max_iterations",
        lambda _nmpc, _iterations: True,
    )
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda solution: {"residuals": solution.residuals},
    )
    monkeypatch.setattr(
        periodic_example,
        "reset_acados_solver_memory",
        lambda _nmpc: True,
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda _nmpc, _solution: None,
    )

    summaries = periodic_example.run_acados_proximal_control_continuation(
        nmpc,
        FakeSolver(),
        weights=(1e4, 1e3),
        convergence_tolerance=1e-3,
        max_restarts=0,
        stage_iterations=20,
        try_next_weight_on_failure=True,
        echo=False,
        solve_stage=lambda: next(solutions),
    )

    assert [summary["weight"] for summary in summaries] == [1e4, 1e3]
    assert [summary["accepted"] for summary in summaries] == [False, True]
    assert applied_weights == [1e4, 1e3, 1e3]
    assert nmpc._cocofest_dual_warm_start_mode == "preserve"


def test_ding_force_compensation_increases_width_when_capacity_drops():
    class FakeDingModel:
        muscle_name = "Biceps"

        def system_dynamics(
            self,
            cn,
            cn_sum,
            f,
            a,
            tau1,
            km,
            pulse_width,
        ):
            return np.array([0.0, 0.0, a * pulse_width - f, 0.0, 0.0, 0.0])

    state_values = {
        "Cn_Biceps": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Cn_sum_Biceps": [0.0, 0.0, 0.0, 0.0, 0.0],
        "F_Biceps": [2.0, 2.0, 2.0, 0.0, 0.0],
        "A_Biceps": [10.0, 10.0, 5.0, 5.0, 5.0],
        "Tau1_Biceps": [1.0, 1.0, 1.0, 1.0, 1.0],
        "Km_Biceps": [1.0, 1.0, 1.0, 1.0, 1.0],
    }
    control_key = "last_pulse_width_Biceps"
    controls = np.array([[0.2, 0.2, 0.2, 0.2]])
    nlp = SimpleNamespace(
        model=SimpleNamespace(muscles_dynamics_model=[FakeDingModel()]),
        x_init={
            key: SimpleNamespace(init=np.asarray([values], dtype=float))
            for key, values in state_values.items()
        },
        u_init={control_key: SimpleNamespace(init=controls)},
        u_bounds={
            control_key: SimpleNamespace(
                min=np.array([[0.1, 0.1, 0.1]]),
                max=np.array([[0.6, 0.6, 0.6]]),
            )
        },
    )
    nmpc = SimpleNamespace(nlp=[nlp], cycle_len=2, cycle_duration=2.0)

    summary = periodic_example.compensate_appended_pulse_widths_from_ding_force(
        nmpc,
        n_substeps=5,
        bisection_iterations=25,
    )

    assert summary["applied"] is True
    muscle_summary = summary["muscles"]["Biceps"]
    assert muscle_summary["gain_mean"] > 1.5
    assert (
        muscle_summary["compensated_force_rmse"] < muscle_summary["baseline_force_rmse"]
    )
    np.testing.assert_allclose(controls[0, 2:], 0.4, atol=1e-6)


def test_ding_force_compensation_uses_previous_solution_for_one_cycle_window():
    class FakeDingModel:
        muscle_name = "Biceps"

        def system_dynamics(self, cn, cn_sum, f, a, tau1, km, pulse_width):
            return np.array([0.0, 0.0, a * pulse_width - f, 0.0, 0.0, 0.0])

    state_values = {
        "Cn_Biceps": [0.0, 0.0, 0.0],
        "Cn_sum_Biceps": [0.0, 0.0, 0.0],
        "F_Biceps": [2.0, 0.0, 0.0],
        "A_Biceps": [5.0, 5.0, 5.0],
        "Tau1_Biceps": [1.0, 1.0, 1.0],
        "Km_Biceps": [1.0, 1.0, 1.0],
    }
    control_key = "last_pulse_width_Biceps"
    controls = np.array([[0.2, 0.2]])
    nlp = SimpleNamespace(
        model=SimpleNamespace(muscles_dynamics_model=[FakeDingModel()]),
        x_init={
            key: SimpleNamespace(init=np.asarray([values], dtype=float))
            for key, values in state_values.items()
        },
        u_init={control_key: SimpleNamespace(init=controls)},
        u_bounds={
            control_key: SimpleNamespace(
                min=np.array([[0.1, 0.1]]), max=np.array([[0.6, 0.6]])
            )
        },
    )
    previous_solution = SimpleNamespace(
        decision_states=lambda to_merge=None: {"F_Biceps": np.array([[2.0, 2.0, 2.0]])},
        decision_controls=lambda to_merge=None: {control_key: np.array([[0.2, 0.2]])},
    )
    nmpc = SimpleNamespace(nlp=[nlp], cycle_len=2, cycle_duration=2.0)

    summary = periodic_example.compensate_appended_pulse_widths_from_ding_force(
        nmpc,
        n_substeps=5,
        bisection_iterations=25,
        previous_solution=previous_solution,
    )

    assert summary["applied"] is True
    assert summary["previous_solution_used"] is True
    assert summary["start_node"] == 0
    np.testing.assert_allclose(controls, 0.4, atol=1e-6)


def test_ding_force_compensation_supports_periodic_node_calcium_forcing():
    class FakePeriodicNodeDingModel:
        muscle_name = "Biceps"

        @staticmethod
        def post_stimulation_amplitude():
            return 1.0

        def system_dynamics(self, states, controls, time, numerical_timeseries):
            cn, force, capacity, tau1, km = states
            return np.array([0.0, capacity * controls[0] - force, 0.0, 0.0, 0.0])

    state_values = {
        "Cn_Biceps": [0.0, 0.0, 0.0],
        "F_Biceps": [2.0, 0.0, 0.0],
        "A_Biceps": [5.0, 5.0, 5.0],
        "Tau1_Biceps": [1.0, 1.0, 1.0],
        "Km_Biceps": [1.0, 1.0, 1.0],
    }
    control_key = "last_pulse_width_Biceps"
    controls = np.array([[0.2, 0.2]])
    nlp = SimpleNamespace(
        model=SimpleNamespace(muscles_dynamics_model=[FakePeriodicNodeDingModel()]),
        x_init={
            key: SimpleNamespace(init=np.asarray([values], dtype=float))
            for key, values in state_values.items()
        },
        u_init={control_key: SimpleNamespace(init=controls)},
        u_bounds={
            control_key: SimpleNamespace(
                min=np.array([[0.1, 0.1]]), max=np.array([[0.6, 0.6]])
            )
        },
    )
    previous_solution = SimpleNamespace(
        decision_states=lambda to_merge=None: {"F_Biceps": np.array([[2.0, 2.0, 2.0]])},
        decision_controls=lambda to_merge=None: {control_key: np.array([[0.2, 0.2]])},
    )
    nmpc = SimpleNamespace(nlp=[nlp], cycle_len=2, cycle_duration=2.0)

    summary = periodic_example.compensate_appended_pulse_widths_from_ding_force(
        nmpc,
        n_substeps=5,
        bisection_iterations=25,
        previous_solution=previous_solution,
    )

    assert summary["applied"] is True
    np.testing.assert_allclose(controls, 0.4, atol=1e-6)


def test_terminal_wheel_bound_continuation_tightens_accepted_stages(
    monkeypatch,
):
    class FakeSolver:
        def set_convergence_tolerance(self, value):
            self.tolerance = value

    solutions = iter(
        [
            SimpleNamespace(
                status=0,
                residuals=np.array([1e-4, 1e-6, 0.0, 1e-5]),
                solver_time_to_optimize=0.1,
                real_time_to_optimize=0.2,
            )
            for _ in range(3)
        ]
    )
    applied_slacks = []
    applied_solutions = []
    nmpc = SimpleNamespace(
        _cocofest_terminal_wheel_q_slack_scale=0.5,
        nlp=[
            SimpleNamespace(
                x_bounds={
                    "q": SimpleNamespace(
                        min=np.array(
                            [
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, -1.2],
                            ]
                        ),
                        max=np.array(
                            [
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, -0.8],
                            ]
                        ),
                    )
                }
            )
        ]
    )
    monkeypatch.setattr(
        periodic_example,
        "set_terminal_wheel_q_bound_slack",
        lambda _nmpc, slack: applied_slacks.append(slack),
    )
    monkeypatch.setattr(
        periodic_example,
        "set_acados_runtime_max_iterations",
        lambda _nmpc, _iterations: True,
    )
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda solution: {"residuals": solution.residuals},
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda _nmpc, solution: applied_solutions.append(solution.status),
    )

    summaries = periodic_example.run_acados_terminal_wheel_bound_continuation(
        nmpc,
        FakeSolver(),
        slacks=(0.2, 0.1, 0.02),
        convergence_tolerance=1e-3,
        stage_iterations=20,
        echo=False,
        solve_stage=lambda: next(solutions),
    )

    assert [item["accepted"] for item in summaries] == [True, True, True]
    assert applied_slacks == [0.1, 0.05, 0.01, 0.01]
    assert [item["slack"] for item in summaries] == [0.2, 0.1, 0.02]
    assert [item["applied_wheel_q_slack"] for item in summaries] == [
        0.1,
        0.05,
        0.01,
    ]
    assert applied_solutions == [0, 0, 0]
    assert nmpc._cocofest_dual_warm_start_mode == "preserve"
    assert nmpc._cocofest_terminal_wheel_q_center == -1.0


def test_terminal_wheel_bound_continuation_restores_target_after_failure(
    monkeypatch,
):
    class FakeSolver:
        def set_convergence_tolerance(self, value):
            self.tolerance = value

    applied_slacks = []
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_bounds={
                    "q": SimpleNamespace(
                        min=np.array(
                            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.2]]
                        ),
                        max=np.array(
                            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -0.8]]
                        ),
                    )
                }
            )
        ]
    )
    monkeypatch.setattr(
        periodic_example,
        "set_terminal_wheel_q_bound_slack",
        lambda _nmpc, slack: applied_slacks.append(slack),
    )
    monkeypatch.setattr(
        periodic_example,
        "set_acados_runtime_max_iterations",
        lambda _nmpc, _iterations: True,
    )
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda solution: {"residuals": solution.residuals},
    )
    failed_solution = SimpleNamespace(
        status=2,
        residuals=np.array([1.0, 1e-2, 0.0, 0.0]),
        solver_time_to_optimize=0.1,
        real_time_to_optimize=0.2,
    )

    summaries = periodic_example.run_acados_terminal_wheel_bound_continuation(
        nmpc,
        FakeSolver(),
        slacks=(0.2, 0.1, 0.02),
        convergence_tolerance=1e-3,
        stage_iterations=20,
        echo=False,
        solve_stage=lambda: failed_solution,
    )

    assert summaries[-1]["accepted"] is False
    assert applied_slacks == [0.2, 0.02]
    assert not hasattr(nmpc, "_cocofest_dual_warm_start_mode")


def test_terminal_wheel_bound_is_recentered_before_a_new_continuation():
    sync_calls = []
    bounds = SimpleNamespace(
        min=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -7.0]]),
        max=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -6.8]]),
    )
    nmpc = SimpleNamespace(
        nlp=[SimpleNamespace(x_bounds={"q": bounds})],
        _sync_acados_state_bounds=lambda: sync_calls.append(True),
    )

    summary = periodic_example.recenter_terminal_wheel_q_bound_slack(nmpc, 0.2)

    assert summary == {
        "center": -6.9,
        "slack": 0.2,
        "lower": -7.1000000000000005,
        "upper": -6.7,
    }
    assert nmpc._cocofest_terminal_wheel_q_center == -6.9
    np.testing.assert_allclose(bounds.min[2, 2], -7.1)
    np.testing.assert_allclose(bounds.max[2, 2], -6.7)
    assert sync_calls == [True]


def test_terminal_wheel_bound_continuation_requires_the_final_target():
    slacks = (0.01, 0.005, 0.002)

    assert periodic_example.terminal_wheel_bound_continuation_reached_target(
        [
            {"slack": 0.01, "accepted": True},
            {"slack": 0.005, "accepted": True},
            {"slack": 0.002, "accepted": True},
        ],
        slacks,
    )
    assert not periodic_example.terminal_wheel_bound_continuation_reached_target(
        [
            {"slack": 0.01, "accepted": True},
            {"slack": 0.005, "accepted": False},
        ],
        slacks,
    )


def test_acados_residual_history_selects_one_feasible_iterate():
    diagnostics = {
        "res_stat_all": np.array([5.0, 0.2, 1e-4]),
        "res_eq_all": np.array([1.0, 2e-4, 1e-2]),
        "res_ineq_all": np.array([0.0, 0.0, 0.0]),
        "res_comp_all": np.array([0.5, 1e-5, 1e-6]),
    }

    summary = periodic_example._acados_residual_history_summary(diagnostics)

    assert summary["best_index"] == 1
    np.testing.assert_allclose(summary["best"], [0.2, 2e-4, 0.0, 1e-5])
    np.testing.assert_allclose(summary["componentwise_best"], [1e-4, 2e-4, 0.0, 1e-6])


def test_control_homotopy_does_not_restart_a_linesearch_failure(monkeypatch):
    reset_calls = []

    class FakeAcadosSolver:
        def reset(self, reset_qp_solver_mem):
            reset_calls.append(reset_qp_solver_mem)

    class FakeSolver:
        nlp_solver_max_iter = 50

        def set_convergence_tolerance(self, value):
            self.tolerance = value

    nmpc = SimpleNamespace(
        ocp_solver=SimpleNamespace(ocp_solver=FakeAcadosSolver()),
        nlp=[SimpleNamespace(u_init={}, u_bounds={})],
    )
    solution = SimpleNamespace(
        status=3,
        residuals=np.array([1.0, 1e-6, 0.0, 1e-6]),
        solver_time_to_optimize=1.0,
        real_time_to_optimize=1.1,
    )
    applied_statuses = []
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda result: {"residuals": result.residuals},
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda _nmpc, result: applied_statuses.append(result.status),
    )

    summaries = periodic_example.run_acados_control_homotopy(
        nmpc,
        FakeSolver(),
        radii=(),
        convergence_tolerance=1e-3,
        fixed_control_tolerance=1e-8,
        max_restarts=3,
        stage_iterations=50,
        echo=False,
        solve_stage=lambda: solution,
    )

    assert summaries[0]["restartable"] is False
    assert summaries[0]["solver_reset"] is True
    assert applied_statuses == []
    assert reset_calls == [1]


def test_control_homotopy_reuses_compiled_acados_options(monkeypatch):
    option_change_flags = []
    runtime_options = []

    class FakeAcadosSolver:
        def options_set(self, key, value):
            runtime_options.append((key, value))

    class FakeSolver:
        nlp_solver_max_iter = 100

        def set_convergence_tolerance(self, value):
            self.tolerance = value

        def set_maximum_iterations(self, value):
            self.nlp_solver_max_iter = value

        def set_only_first_options_has_changed(self, value):
            option_change_flags.append(value)

    nmpc = SimpleNamespace(
        ocp_solver=SimpleNamespace(ocp_solver=FakeAcadosSolver()),
        nlp=[SimpleNamespace(u_init={}, u_bounds={})],
    )
    solution = SimpleNamespace(
        status=0,
        residuals=np.zeros(4),
        solver_time_to_optimize=1.0,
        real_time_to_optimize=1.1,
    )
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda _solution: {"residuals": np.zeros(4)},
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda *_args: None,
    )

    periodic_example.run_acados_control_homotopy(
        nmpc,
        FakeSolver(),
        radii=(),
        convergence_tolerance=1e-3,
        fixed_control_tolerance=1e-8,
        stage_iterations=50,
        echo=False,
        solve_stage=lambda: solution,
    )

    assert option_change_flags == [False]
    assert runtime_options == [("nlp_solver_max_iter", 50)]


def _benchmark_result(statuses, solver_success=False, success=False):
    cycle_count = 3
    shooting_per_cycle = 2
    return {
        "args": SimpleNamespace(stimulations_per_cycle=shooting_per_cycle),
        "window_statuses": statuses,
        "solver_success": solver_success,
        "success": success,
        "covered_cycles": cycle_count if solver_success else 0,
        "wheel_angle_trace": np.arange(cycle_count * shooting_per_cycle + 1),
        "state_traces": {
            "A_Biceps": np.linspace(100.0, 80.0, 7)[np.newaxis, :],
            "Tau1_Biceps": np.linspace(0.05, 0.06, 7)[np.newaxis, :],
        },
        "fatigue_capacity_scales": {"A_Biceps": 100.0},
        "control_traces": {
            "last_pulse_width_Biceps": np.array(
                [[0.0002, 0.0006, 0.0003, 0.0004, 0.0005, 0.0006]]
            )
        },
        "control_bounds": {
            "last_pulse_width_Biceps": {"lower": 0.0001, "upper": 0.0006}
        },
    }


def test_benchmark_compares_only_the_successful_prefix():
    result = _benchmark_result([0, 0, 4])

    assert comparison_example._successful_prefix_length([0, 0, 4, 0]) == 2
    assert comparison_example._validated_cycle_count(result) == 2

    limited = comparison_example._truncate_result_to_cycles(result, 2)
    assert limited["wheel_angle_trace"].shape == (5,)
    assert limited["state_traces"]["A_Biceps"].shape == (1, 5)
    assert limited["control_traces"]["last_pulse_width_Biceps"].shape == (1, 4)


def test_benchmark_separates_attempted_and_validated_prefix_objectives():
    result = _benchmark_result([0, 1, 0])
    result.update(
        objective=103.0,
        window_objectives=[1.0, 2.0, 100.0],
        window_iterations=[10, 20, 30],
        window_solutions=[
            SimpleNamespace(
                status=status,
                solver_time_to_optimize=0.1,
                real_time_to_optimize=0.2,
            )
            for status in (0, 1, 0)
        ],
        window_feasibility=[
            {"passes_tolerance": True},
            {"passes_tolerance": True},
            {"passes_tolerance": True},
        ],
        attempted_windows=3,
        successful_windows=2,
        solver_time_s=0.3,
        wall_time_s=0.6,
        end_to_end_wall_time_s=0.6,
        initial_guess_preparation_time_s=0.0,
    )

    row = comparison_example.solver_overview_rows({"ipopt": result})[0]

    assert row["window_objective_sum"] == 103.0
    assert row["validated_prefix_window_objective_sum"] == 1.0


def test_benchmark_excludes_nlp_cycles_after_external_physical_failure():
    result = _benchmark_result([0, 0, 0], solver_success=True, success=False)
    result["physical_success"] = False
    result["window_solutions"] = [
        SimpleNamespace(
            status=0,
            solver_time_to_optimize=0.1,
            real_time_to_optimize=0.2,
        )
        for _ in range(3)
    ]
    result["window_feasibility"] = [
        {"passes_tolerance": True},
        {"passes_tolerance": True},
        {"passes_tolerance": True},
    ]

    performance = comparison_example._window_performance(result)

    assert performance["nlp_validated_cycles"] == 3
    assert performance["physically_validated_cycles"] == 0
    assert performance["validated_cycles"] == 0


def test_benchmark_keeps_the_physical_prefix_before_a_later_angle_failure():
    result = _benchmark_result([0, 0, 0], solver_success=True, success=False)
    result.update(
        physical_success=False,
        physical_crank_angle_trace=np.array(
            [
                0.0,
                -np.pi,
                -2 * np.pi,
                -2.5 * np.pi,
                -3 * np.pi,
                -3.5 * np.pi,
                -4 * np.pi,
            ]
        ),
        physical_crank_absolute_reference=0.0,
        physical_crank_diagnostics={
            "absolute_cycle_tolerance": 0.00201,
            "cycle_progress_tolerance": 0.00402,
        },
        mechanical_equivalence_audit={
            "available": True,
            "passes_tolerance": True,
        },
    )

    assert comparison_example._validated_cycle_count(result) == 3
    assert comparison_example._physically_validated_cycle_count(result) == 1


def test_first_failed_rho_preserves_a_later_solver_failure():
    windows = [{"rho": rho, "validated": rho < 81} for rho in range(1, 101)]

    assert comparison_example._first_failed_rho(windows, False) == 81
    assert (
        comparison_example._first_failed_rho(
            [{"rho": rho, "validated": True} for rho in range(1, 101)],
            False,
        )
        == 1
    )


def test_rho_boundary_jump_summary_keeps_both_sides_of_every_seam():
    class CycleSolution:
        def __init__(self, theta, omega, capacity):
            self._states = {
                "theta": np.asarray(theta, dtype=float)[np.newaxis, :],
                "omega": np.asarray(omega, dtype=float)[np.newaxis, :],
                "A_Biceps": np.asarray(capacity, dtype=float)[np.newaxis, :],
            }

        def decision_states(self, to_merge):
            assert to_merge == SolutionMerge.NODES
            return self._states

    cycles = [
        CycleSolution([0.0, -1.0], [-6.0, -6.1], [100.0, 99.0]),
        CycleSolution([-0.99, -2.0], [-6.0, -6.2], [98.5, 98.0]),
        CycleSolution([-2.02, -3.0], [-6.3, -6.4], [97.8, 97.0]),
    ]

    summary = periodic_example._state_boundary_jump_summary(cycles)

    assert summary["available"] is True
    assert summary["boundary_count"] == 2
    np.testing.assert_allclose(
        summary["by_state"]["theta"]["jump"][:, 0],
        [0.01, -0.02],
    )
    np.testing.assert_allclose(
        summary["by_state"]["omega"]["jump"][:, 0],
        [0.1, -0.1],
    )
    assert summary["by_state"]["A_Biceps"]["maximum_absolute_jump"] == 0.5


def test_benchmark_reports_hot_window_timing_separately():
    result = _benchmark_result([0, 0, 0], solver_success=True, success=True)
    result["window_solutions"] = [
        SimpleNamespace(
            status=0, solver_time_to_optimize=10.0, real_time_to_optimize=11.0
        ),
        SimpleNamespace(
            status=0, solver_time_to_optimize=2.0, real_time_to_optimize=2.2
        ),
        SimpleNamespace(
            status=0, solver_time_to_optimize=4.0, real_time_to_optimize=4.4
        ),
    ]
    result["window_feasibility"] = [
        {"passes_tolerance": True},
        {"passes_tolerance": True},
        {"passes_tolerance": True},
    ]

    performance = comparison_example._window_performance(result)

    assert performance["hot_window_count"] == 2
    assert performance["hot_solver_time_median_s"] == 3.0
    np.testing.assert_allclose(performance["hot_solver_time_p90_s"], 3.8)
    np.testing.assert_allclose(performance["hot_wall_time_median_s"], 3.3)


def test_stimulation_snapshots_use_one_based_cycles_and_real_crank_phase():
    cycle_count = 100
    shooting_per_cycle = 2
    result = {
        "args": SimpleNamespace(
            stimulations_per_cycle=shooting_per_cycle,
            cycles_per_window=1,
            ode_solver="rk4",
        ),
        "window_statuses": [0] * cycle_count,
        "window_feasibility": [{"passes_tolerance": True}] * cycle_count,
        "solver_success": True,
        "success": True,
        "covered_cycles": cycle_count,
        "exported_cycles": cycle_count,
        "wheel_angle_trace": np.linspace(
            0.0,
            -2.0 * np.pi * cycle_count,
            cycle_count * shooting_per_cycle + 1,
        ),
        "physical_crank_angle_trace": np.linspace(
            0.2,
            0.2 - 2.0 * np.pi * cycle_count,
            cycle_count * shooting_per_cycle + 1,
        ),
        "state_traces": {
            "qdot": np.vstack(
                (
                    np.zeros(cycle_count * shooting_per_cycle + 1),
                    np.zeros(cycle_count * shooting_per_cycle + 1),
                    -np.ones(cycle_count * shooting_per_cycle + 1),
                )
            )
        },
        "control_traces": {
            "last_pulse_width_Biceps": (
                1e-6 * np.repeat(np.arange(1, cycle_count + 1), shooting_per_cycle)
            )[np.newaxis, :]
        },
        "control_bounds": {"last_pulse_width_Biceps": {"lower": 0.0, "upper": 50e-6}},
    }

    snapshots = comparison_example.stimulation_pattern_snapshots(result)

    cycle_10 = snapshots["cycle_10"]
    assert cycle_10["available"] is True
    assert cycle_10["rho"] == 10
    np.testing.assert_allclose(
        cycle_10["muscles"]["Biceps"]["pulse_width_us"], [10.0, 10.0]
    )
    np.testing.assert_allclose(cycle_10["crank_phase_rad"], [0.0, np.pi])
    np.testing.assert_allclose(
        cycle_10["crank_angle_rad"],
        [0.2 - 18.0 * np.pi, 0.2 - 19.0 * np.pi],
    )
    np.testing.assert_allclose(cycle_10["crank_velocity_rad_s"], [-1.0, -1.0])
    np.testing.assert_allclose(
        snapshots["cycle_30"]["muscles"]["Biceps"]["pulse_width_us"],
        [30.0, 30.0],
    )
    np.testing.assert_allclose(
        snapshots["cycle_100"]["muscles"]["Biceps"]["pulse_width_us"],
        [100.0, 100.0],
    )


def test_stimulation_snapshot_rejects_cycle_outside_converged_prefix():
    result = _benchmark_result([0, 0, 1], solver_success=False, success=False)
    result["window_feasibility"] = [
        {"passes_tolerance": True},
        {"passes_tolerance": True},
        {"passes_tolerance": False},
    ]

    snapshot = comparison_example._stimulation_pattern_snapshot(result, 3)

    assert snapshot["available"] is False
    assert snapshot["reason"] == "only_2_cycles_belong_to_the_converged_prefix"


def test_stimulation_snapshot_rejects_nlp_cycle_without_mechanical_certificate():
    result = _benchmark_result([0], solver_success=True, success=False)
    result["mechanical_equivalence_audit"] = {
        "available": True,
        "passes_tolerance": False,
    }

    snapshot = comparison_example._stimulation_pattern_snapshot(result, 1)

    assert snapshot["available"] is False
    assert snapshot["reason"] == "only_0_cycles_belong_to_the_converged_prefix"


def test_isolated_checkpoint_preserves_terminal_fatigue_after_prefix_failure():
    class WindowSolution:
        def __init__(self, status, capacity, pulse_width):
            self.status = status
            self._capacity = capacity
            self._pulse_width = pulse_width

        def decision_states(self, to_merge):
            assert to_merge == SolutionMerge.NODES
            return {"A_Biceps": np.asarray([self._capacity], dtype=float)}

        def decision_controls(self, to_merge):
            assert to_merge == SolutionMerge.NODES
            return {
                "last_pulse_width_Biceps": np.asarray([self._pulse_width], dtype=float)
            }

    result = _benchmark_result([0, 1, 0], solver_success=False, success=False)
    result["window_solutions"] = [
        WindowSolution(0, [100.0, 99.0], [200e-6, 210e-6]),
        WindowSolution(1, [99.0, 97.0], [220e-6, 230e-6]),
        WindowSolution(0, [97.0, 96.0], [240e-6, 250e-6]),
    ]
    result["window_objectives"] = [1.0, 2.0, 3.0]
    result["window_feasibility"] = [
        {"passes_tolerance": True},
        {"passes_tolerance": False},
        {"passes_tolerance": True},
    ]
    result["fatigue_capacity_scales"] = {"A_Biceps": 100.0}

    checkpoint = comparison_example.isolated_window_checkpoint_snapshots(
        result, cycles=(3,)
    )["cycle_3"]

    assert checkpoint["available"] is True
    assert checkpoint["diagnostic_only"] is True
    assert checkpoint["belongs_to_strict_prefix"] is False
    assert checkpoint["objective"] == 3.0
    assert checkpoint["primal_feasible"] is True
    assert checkpoint["capacity_states"]["A_Biceps"]["terminal_ratio"] == 0.96
    np.testing.assert_allclose(checkpoint["pulse_width_us"]["Biceps"], [240.0, 250.0])


def test_pulse_width_cycle_variation_reports_aligned_transition_percentiles():
    result = _benchmark_result([0, 0, 0], solver_success=True, success=True)

    variation = comparison_example.pulse_width_cycle_variation(result, cycle_count=3)

    assert variation["available"] is True
    assert variation["transition_count"] == 2
    muscle = variation["muscles"][0]
    assert muscle["muscle"] == "Biceps"
    np.testing.assert_allclose(
        [row["mean_absolute_change_us"] for row in muscle["transitions"]],
        [150.0, 200.0],
    )
    assert variation["pooled_absolute_change_us"]["maximum"] == pytest.approx(200.0)


def test_benchmark_rejects_status_zero_window_above_feasibility_threshold():
    result = _benchmark_result([0, 0], solver_success=False, success=False)
    result["window_feasibility"] = [
        {"passes_tolerance": True},
        {"passes_tolerance": False},
    ]

    assert comparison_example._validated_cycle_count(result) == 1


def test_benchmark_extracts_collocation_shooting_nodes_without_interpolation():
    result = _benchmark_result([0, 0], solver_success=True, success=True)
    result["args"] = SimpleNamespace(
        stimulations_per_cycle=2,
        ode_solver="collocation",
        collocation_degree=3,
    )
    result["exported_cycles"] = 2
    collocation_values = np.arange(17, dtype=float)
    result["wheel_angle_trace"] = collocation_values
    result["state_traces"] = {"q": collocation_values[np.newaxis, :]}
    result["control_traces"] = {"u": np.arange(4, dtype=float)[np.newaxis, :]}

    limited = comparison_example._truncate_result_to_cycles(result, 2)

    np.testing.assert_array_equal(limited["wheel_angle_trace"], [0, 4, 8, 12, 16])
    np.testing.assert_array_equal(limited["state_traces"]["q"], [[0, 4, 8, 12, 16]])
    np.testing.assert_array_equal(limited["control_traces"]["u"], [[0, 1, 2, 3]])


def test_benchmark_accepts_a_shorter_certified_physical_prefix():
    result = _benchmark_result([0, 0, 4])
    result["physical_crank_angle_trace"] = np.array([0.0, -np.pi, -2 * np.pi])
    result["physical_crank_velocity_trace"] = np.array([-6.0, -6.1, -6.0])

    limited = comparison_example._truncate_result_to_cycles(result, 2)

    np.testing.assert_array_equal(
        limited["physical_crank_angle_trace"], [0.0, -np.pi, -2 * np.pi]
    )
    np.testing.assert_array_equal(
        limited["physical_crank_velocity_trace"], [-6.0, -6.1, -6.0]
    )


def test_benchmark_extracts_collocation_points_from_physical_crank_trace():
    result = _benchmark_result([0, 0], solver_success=True, success=True)
    result["args"] = SimpleNamespace(
        stimulations_per_cycle=2,
        ode_solver="collocation",
        collocation_degree=5,
    )
    result["exported_cycles"] = 2
    collocation_values = np.arange(25, dtype=float)
    result["wheel_angle_trace"] = collocation_values
    result["state_traces"] = {"q": collocation_values[np.newaxis, :]}
    result["control_traces"] = {"u": np.arange(4, dtype=float)[np.newaxis, :]}
    result["physical_crank_angle_trace"] = collocation_values
    result["physical_crank_velocity_trace"] = -collocation_values
    result["mechanical_equivalence_audit"] = {
        "cadence_audit_node_stride": 6,
        "passes_tolerance": True,
    }

    limited = comparison_example._truncate_result_to_cycles(result, 2)

    np.testing.assert_array_equal(
        limited["physical_crank_angle_trace"], [0, 6, 12, 18, 24]
    )
    np.testing.assert_array_equal(
        limited["physical_crank_velocity_trace"], [0, -6, -12, -18, -24]
    )


def test_state_comparison_aligns_wheel_turn_representation():
    reference = {
        "q": np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.0, -np.pi, -2 * np.pi],
            ]
        )
    }
    compared = {
        "q": np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [-2 * np.pi + 0.01, -3 * np.pi + 0.01, -4 * np.pi + 0.01],
            ]
        )
    }

    metrics = comparison_example._state_trace_comparisons(
        reference, compared, "reference", "compared"
    )
    wheel = next(metric for metric in metrics if metric["key"] == "q[2]")

    np.testing.assert_allclose(wheel["rmse"], 0.01)
    np.testing.assert_allclose(wheel["final_error"], 0.01)


def test_full_dynamics_q_defect_scale_is_independent_of_unwrapped_turns():
    class Variables(dict):
        pass

    nlp = SimpleNamespace(
        states=Variables(
            q=SimpleNamespace(index=[0, 1]),
            qdot=SimpleNamespace(index=[2]),
            F_Test=SimpleNamespace(index=[3]),
        )
    )
    states = np.array(
        [
            [0.0, 100.0 * 2.0 * np.pi],
            [1.0, 1.5],
            [-2.0, -3.0],
            [10.0, 20.0],
        ]
    )

    scales = periodic_example._full_dynamics_defect_state_scales(nlp, states)

    np.testing.assert_allclose(scales[[0, 1], :], 2.0 * np.pi)
    assert scales[2, 0] == 3.0
    assert scales[3, 0] == 20.0


def test_endurance_metrics_report_fatigue_and_control_saturation():
    result = _benchmark_result([0, 0, 4])

    fatigue = comparison_example._fatigue_metrics(result, cycle_count=2)
    executed_objective = comparison_example._executed_fatigue_objective(
        result, cycle_count=2
    )
    muscle_objectives = comparison_example._executed_fatigue_objective_by_muscle(
        result, cycle_count=2
    )
    saturation = comparison_example._control_saturation_metrics(result, cycle_count=2)

    a_row = next(row for row in fatigue if row["key"] == "A_Biceps")
    np.testing.assert_allclose(a_row["relative_final"], (100 - 4 * 20 / 6) / 100)
    assert a_row["mean_normalized_fatigue"] > 0
    assert a_row["fatigue_auc_cycles"] > 0
    np.testing.assert_allclose(
        comparison_example._minimum_a_capacity_ratio(fatigue),
        a_row["relative_final"],
    )
    assert comparison_example._format_a_capacity_by_muscle(fatigue) == (
        f"Biceps={a_row['relative_final']:.6f}"
    )
    assert executed_objective > 0
    assert muscle_objectives == [
        {
            "muscle": "Biceps",
            "state_key": "A_Biceps",
            "executed_fatigue_objective": executed_objective,
            "cumulative_normalized_fatigue_cycles": a_row["fatigue_auc_cycles"],
            "final_capacity_ratio": a_row["relative_final"],
        }
    ]
    assert saturation[0]["upper_fraction"] == 0.25


def test_external_crank_power_uses_generalized_power_sign():
    result = _benchmark_result([0, 0], solver_success=True, success=True)
    result["args"].constant_crank_torque = -0.2
    result["state_traces"]["qdot"] = np.vstack(
        (
            np.zeros(7),
            np.zeros(7),
            np.full(7, -2.0 * np.pi),
        )
    )

    driving = comparison_example._external_crank_power_metrics(result, cycle_count=3)
    result["args"].constant_crank_torque = 0.2
    resistive = comparison_example._external_crank_power_metrics(result, cycle_count=3)

    assert driving["role"] == "driving"
    assert resistive["role"] == "resistive"
    np.testing.assert_allclose(driving["mean_power_w"], 0.4 * np.pi)
    np.testing.assert_allclose(resistive["mean_power_w"], -0.4 * np.pi)

    result["args"].constant_crank_torque = -0.2
    result["state_traces"]["omega"] = result["state_traces"].pop("qdot")[2:3, :]
    reduced = comparison_example._external_crank_power_metrics(result, cycle_count=3)
    assert reduced["role"] == "driving"
    np.testing.assert_allclose(reduced["mean_power_w"], 0.4 * np.pi)


def test_cycle_boundary_wheel_angle_reports_turn_error():
    result = _benchmark_result([0, 0], solver_success=True, success=True)
    result["wheel_angle_trace"] = np.array(
        [
            0.0,
            -np.pi,
            -2.0 * np.pi + 0.01,
            -3.0 * np.pi,
            -4.0 * np.pi - 0.02,
            -5.0 * np.pi,
            -6.0 * np.pi + 0.03,
        ]
    )

    metrics = comparison_example._cycle_boundary_wheel_angle_metrics(
        result, cycle_count=3
    )

    np.testing.assert_allclose(metrics["signed_cycle_shift_rad"], -2.0 * np.pi)
    np.testing.assert_allclose(metrics["errors_rad"], [0.0, 0.01, -0.02, 0.03])
    np.testing.assert_allclose(metrics["maximum_absolute_error_rad"], 0.03)
    np.testing.assert_allclose(metrics["final_error_rad"], 0.03)
    np.testing.assert_allclose(
        metrics["cycle_progress_errors_rad"], [0.01, -0.03, 0.05]
    )
    np.testing.assert_allclose(metrics["maximum_cycle_progress_error_rad"], 0.05)


def test_cycle_boundary_wheel_angle_uses_fixed_absolute_reference():
    result = _benchmark_result([0, 0], solver_success=True, success=True)
    result["absolute_wheel_q_reference"] = 0.0
    result["wheel_angle_trace"] = np.array(
        [
            -0.002,
            -np.pi,
            -2.0 * np.pi + 0.002,
            -3.0 * np.pi,
            -4.0 * np.pi + 0.002,
            -5.0 * np.pi,
            -6.0 * np.pi + 0.002,
        ]
    )

    metrics = comparison_example._cycle_boundary_wheel_angle_metrics(
        result, cycle_count=3
    )

    assert metrics["absolute_reference_rad"] == 0.0
    np.testing.assert_allclose(metrics["errors_rad"], [-0.002, 0.002, 0.002, 0.002])
    np.testing.assert_allclose(metrics["maximum_absolute_error_rad"], 0.002)


def test_wheel_trace_diagnostic_rejects_wrong_rotation_direction():
    trace = np.linspace(0.0, 4.0 * np.pi, 5)

    diagnostics = periodic_example.diagnose_wheel_trace(
        trace,
        requested_windows=2,
        expected_cycle_shift=-2.0 * np.pi,
        cycle_progress_tolerance=0.01,
    )

    assert not diagnostics["is_physical"]
    assert "wheel_cycle_progress_out_of_bounds" in diagnostics["issues"]
    np.testing.assert_allclose(diagnostics["maximum_cycle_progress_error"], 4.0 * np.pi)


def test_wheel_trace_diagnostic_accepts_configured_cycle_slack():
    trace = np.array([0.0, -3.13, -6.28, -9.42, -12.56])

    diagnostics = periodic_example.diagnose_wheel_trace(
        trace,
        requested_windows=2,
        expected_cycle_shift=-2.0 * np.pi,
        cycle_progress_tolerance=0.01,
    )

    assert diagnostics["is_physical"]
    assert diagnostics["maximum_cycle_progress_error"] < 0.01
    assert diagnostics["maximum_absolute_cycle_error"] < 0.01
    np.testing.assert_allclose(
        diagnostics["final_absolute_cycle_error"],
        -12.56 - (-4.0 * np.pi),
    )


def test_wheel_trace_diagnostic_rejects_accumulated_same_sign_drift():
    per_cycle_error = 0.002
    cycle_boundaries = np.arange(101, dtype=float) * (-2.0 * np.pi + per_cycle_error)

    diagnostics = periodic_example.diagnose_wheel_trace(
        cycle_boundaries,
        requested_windows=100,
        expected_cycle_shift=-2.0 * np.pi,
        cycle_progress_tolerance=0.0021,
    )

    assert diagnostics["maximum_cycle_progress_error"] < 0.0021
    np.testing.assert_allclose(
        diagnostics["final_absolute_cycle_error"], 100 * per_cycle_error
    )
    assert diagnostics["maximum_absolute_cycle_error"] > 0.19
    assert "wheel_absolute_progress_out_of_bounds" in diagnostics["issues"]
    assert diagnostics["is_physical"] is False


def test_wheel_trace_diagnostic_uses_distinct_absolute_tolerance():
    trace = np.array(
        [
            -0.002,
            -2.0 * np.pi + 0.002,
            -4.0 * np.pi + 0.002,
        ]
    )

    diagnostics = periodic_example.diagnose_wheel_trace(
        trace,
        requested_windows=2,
        expected_cycle_shift=-2.0 * np.pi,
        cycle_progress_tolerance=0.0041,
        absolute_cycle_reference=0.0,
        absolute_cycle_tolerance=0.0021,
    )

    assert diagnostics["is_physical"] is True
    np.testing.assert_allclose(
        diagnostics["absolute_cycle_errors"], [-0.002, 0.002, 0.002]
    )
    np.testing.assert_allclose(diagnostics["maximum_absolute_cycle_error"], 0.002)
    assert diagnostics["maximum_cycle_progress_error"] == pytest.approx(0.004)


def test_wheel_cycle_diagnostic_tolerances_keep_absolute_slack():
    args = SimpleNamespace(
        solver="madnlp",
        acados_tolerance=None,
        nlp_tolerance=1e-8,
        primal_feasibility_threshold=1e-5,
        acados_terminal_wheel_q_slack=0.002,
        acados_wheel_q_slack=0.0,
    )

    progress, absolute = periodic_example._wheel_cycle_diagnostic_tolerances(
        args, wheel_q_scaling=2.0 * np.pi
    )

    assert progress == pytest.approx(0.004 + 4.0 * np.pi * 1e-5)
    assert absolute == pytest.approx(0.002 + 2.0 * np.pi * 1e-5)


def test_wheel_cycle_diagnostic_tolerances_keep_larger_first_node_slack():
    args = SimpleNamespace(
        solver="ipopt",
        acados_tolerance=None,
        nlp_tolerance=1e-6,
        primal_feasibility_threshold=1e-5,
        acados_terminal_wheel_q_slack=0.002,
        acados_wheel_q_slack=0.003,
    )

    progress, absolute = periodic_example._wheel_cycle_diagnostic_tolerances(args)

    assert progress == pytest.approx(0.00502)
    assert absolute == pytest.approx(0.00301)


def test_acados_wheel_audit_uses_the_public_absolute_feasibility_threshold():
    args = SimpleNamespace(
        solver="acados",
        acados_tolerance=1e-3,
        nlp_tolerance=1e-6,
        primal_feasibility_threshold=1e-5,
        acados_terminal_wheel_q_slack=0.002,
        acados_wheel_q_slack=0.0,
    )

    progress, absolute = periodic_example._wheel_cycle_diagnostic_tolerances(args)

    assert progress == pytest.approx(0.00402)
    assert absolute == pytest.approx(0.00201)


def test_acados_wheel_audit_uses_the_final_homotopy_slack():
    args = SimpleNamespace(
        solver="acados",
        acados_tolerance=1e-3,
        nlp_tolerance=1e-6,
        primal_feasibility_threshold=1e-5,
        acados_terminal_wheel_q_slack=0.01,
        acados_terminal_wheel_q_homotopy_slacks=(0.01, 0.005, 0.002),
        acados_wheel_q_slack=0.0,
    )

    progress, absolute = periodic_example._wheel_cycle_diagnostic_tolerances(args)

    assert progress == pytest.approx(0.00402)
    assert absolute == pytest.approx(0.00201)


def test_wheel_q_state_scaling_reads_crank_coordinate():
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_scaling={
                    "q": SimpleNamespace(
                        scaling=np.array([[1.0], [1.0], [2.0 * np.pi]])
                    )
                }
            )
        ]
    )

    assert periodic_example._wheel_q_state_scaling(nmpc) == pytest.approx(2.0 * np.pi)


def test_wheel_trace_absolute_reference_accounts_for_consumed_warmup():
    nmpc = SimpleNamespace(
        anchor_wheel_q_to_absolute_reference=True,
        absolute_wheel_q_reference=0.95,
        absolute_wheel_q_cycle_shift=-2.0 * np.pi,
        absolute_wheel_q_cycle_index=1,
    )

    (
        trace_reference,
        origin_reference,
        cycle_index,
    ) = periodic_example._wheel_trace_absolute_reference(nmpc)

    assert trace_reference == pytest.approx(0.95 - 2.0 * np.pi)
    assert origin_reference == pytest.approx(0.95)
    assert cycle_index == 1


def test_wheel_trace_absolute_reference_is_optional_without_absolute_anchor():
    assert periodic_example._wheel_trace_absolute_reference(
        SimpleNamespace(absolute_wheel_q_reference=0.95)
    ) == (None, None, 0)


def test_a_capacity_metrics_use_model_scale_instead_of_initial_state():
    result = _benchmark_result([0, 0])
    result["state_traces"]["A_Biceps"] = np.linspace(90.0, 80.0, 7)[np.newaxis, :]
    result["fatigue_capacity_scales"]["A_Biceps"] = 100.0

    a_row = next(
        row
        for row in comparison_example._fatigue_metrics(result, cycle_count=3)
        if row["key"] == "A_Biceps"
    )

    assert a_row["normalization_source"] == "a_scale"
    assert a_row["normalization_reference"] == 100.0
    np.testing.assert_allclose(a_row["relative_final"], 0.8)
    assert a_row["mean_normalized_fatigue"] > 0.1


def test_shared_capacity_limit_requires_two_independent_signals():
    ipopt = _benchmark_result([0, 0, 1])
    acados = _benchmark_result([0, 4])

    classification = comparison_example._shared_stop_classification(ipopt, acados)

    assert classification["label"] == "shared_capacity_limit_candidate"
    assert set(classification["evidence"]) == {
        "both_solvers_stop_at_similar_cycle",
        "pulse_width_upper_bound_active",
    }


def test_completed_endurance_horizon_uses_all_covered_cycles():
    result = _benchmark_result([0, 0], solver_success=True, success=True)

    assert comparison_example._validated_cycle_count(result) == 3
    assert comparison_example._stop_classification(result)["label"] == (
        "completed_requested_horizon"
    )


def test_receding_horizon_window_count_includes_single_cycle_horizons():
    assert periodic_example.receding_horizon_window_count(3, 1) == 3
    assert periodic_example.receding_horizon_window_count(30, 2) == 29

    with np.testing.assert_raises_regex(ValueError, "at least"):
        periodic_example.receding_horizon_window_count(1, 2)


def test_receding_horizon_objective_uses_source_windows_not_dummy_merged_cost():
    solutions = [
        SimpleNamespace(cost=np.array([[1.5]])),
        SimpleNamespace(cost=np.array([[2.0], [3.0]])),
        SimpleNamespace(cost=None),
    ]

    assert periodic_example._window_objective_values(solutions) == [
        1.5,
        5.0,
        None,
    ]


def test_endurance_cli_stops_on_failure_and_keeps_robust_irk_defaults():
    args = comparison_example.build_cli().parse_args([])

    assert args.max_consecutive_failing == 1
    assert args.n_threads == (comparison_example.os.cpu_count() or 1)
    assert args.acados_integrator_type == "IRK"
    assert args.acados_sim_stages == 4
    assert args.acados_sim_steps == 5
    assert args.acados_dual_warm_start_mode == "reset"
    assert args.acados_transfer_phase_one is False
    assert args.acados_cyclical_transfer_mode == "extrapolate"
    assert args.acados_transfer_phase_one_proximity_weight == 1.0
    assert args.acados_transfer_phase_one_defect_weight == 10.0
    assert args.acados_transfer_phase_one_substeps == 5
    assert args.acados_transfer_pulse_width_trust_radius is None
    assert args.acados_proximal_control_weights is None
    assert args.acados_proximal_control_try_next_weight_on_failure is False
    assert args.acados_proximal_control_restart_feasibility_factor == 1
    assert args.continue_after_acados_transfer_failure is False
    assert args.acados_transfer_mechanical_restoration is False
    assert args.periodic_ipopt_refinement_ode_solver == "target"


def test_solver_clis_accept_explicit_thread_count():
    periodic_args = periodic_example.build_argument_parser().parse_args(
        ["--n-threads", "8"]
    )
    mumps_args = comparison_example.build_cli().parse_args(
        ["--madnlp-linear-solver", "mumps"]
    )
    comparison_args = comparison_example.build_cli().parse_args(["--n-threads", "8"])

    assert periodic_args.n_threads == 8
    assert mumps_args.madnlp_linear_solver == "mumps"
    assert comparison_args.n_threads == 8


def test_solver_clis_expose_ipopt_fatrop_and_madnlp_compilation_options():
    periodic_args = periodic_example.build_argument_parser().parse_args(
        [
            "--ipopt-c-compile",
            "--ipopt-hsl-library",
            "/opt/coinhsl.dylib",
            "--warmup-ipopt-linear-solver",
            "mumps",
            "--ipopt-print-level",
            "5",
            "--ipopt-print-timing-statistics",
            "--ipopt-linear-system-scaling",
            "none",
            "--ipopt-ma57-automatic-scaling",
            "--ipopt-ma57-pivot-order",
            "2",
            "--madnlp-c-compile",
            "--fatrop-c-compile",
        ]
    )
    comparison_args = comparison_example.build_cli().parse_args(
        [
            "--ipopt-c-compile",
            "--ipopt-hsl-library",
            "/opt/coinhsl.dylib",
            "--warmup-ipopt-linear-solver",
            "mumps",
            "--ipopt-print-level",
            "5",
            "--ipopt-print-timing-statistics",
            "--ipopt-linear-system-scaling",
            "none",
            "--ipopt-ma57-automatic-scaling",
            "--ipopt-ma57-pivot-order",
            "2",
            "--madnlp-c-compile",
            "--fatrop-c-compile",
        ]
    )

    assert periodic_args.ipopt_c_compile is True
    assert comparison_args.ipopt_c_compile is True
    assert periodic_args.ipopt_hsl_library == "/opt/coinhsl.dylib"
    assert comparison_args.ipopt_hsl_library == "/opt/coinhsl.dylib"
    assert periodic_args.warmup_ipopt_linear_solver == "mumps"
    assert comparison_args.warmup_ipopt_linear_solver == "mumps"
    assert periodic_example._warmup_ipopt_linear_solver(periodic_args) == "mumps"
    assert periodic_args.ipopt_print_level == 5
    assert comparison_args.ipopt_print_level == 5
    assert periodic_args.ipopt_print_timing_statistics is True
    assert comparison_args.ipopt_print_timing_statistics is True
    assert periodic_args.ipopt_linear_system_scaling == "none"
    assert comparison_args.ipopt_linear_system_scaling == "none"
    assert periodic_args.ipopt_ma57_automatic_scaling is True
    assert comparison_args.ipopt_ma57_automatic_scaling is True
    assert periodic_args.ipopt_ma57_pivot_order == 2
    assert comparison_args.ipopt_ma57_pivot_order == 2
    assert periodic_example._ipopt_advanced_options(periodic_args) == {
        "print_timing_statistics": "yes",
        "linear_system_scaling": "none",
        "ma57_pivot_order": 2,
        "ma57_automatic_scaling": "yes",
    }
    assert periodic_args.madnlp_c_compile is True
    assert comparison_args.madnlp_c_compile is True
    assert periodic_args.fatrop_c_compile is True
    assert comparison_args.fatrop_c_compile is True


def test_warmup_cache_signature_is_independent_from_target_linear_solver(
    tmp_path,
):
    model_path = tmp_path / "model.bioMod"
    model_path.write_text("version 4\n")
    common = [
        "--warmup-ipopt-linear-solver",
        "mumps",
        "--objective",
        "fatigue",
    ]
    mumps_args = periodic_example.build_argument_parser().parse_args(
        [*common, "--ipopt-linear-solver", "mumps"]
    )
    ma57_args = periodic_example.build_argument_parser().parse_args(
        [*common, "--ipopt-linear-solver", "ma57"]
    )
    simulation_conditions = {"test": "shared-seed"}
    cycling_info = {"event": object()}

    mumps_signature = periodic_example._warmup_cache_signature(
        mumps_args,
        model_path,
        simulation_conditions,
        cycling_info,
    )
    ma57_signature = periodic_example._warmup_cache_signature(
        ma57_args,
        model_path,
        simulation_conditions,
        cycling_info,
    )

    assert ma57_signature == mumps_signature

    ma57_args.warmup_ipopt_linear_solver = "ma57"
    assert (
        periodic_example._warmup_cache_signature(
            ma57_args,
            model_path,
            simulation_conditions,
            cycling_info,
        )
        != mumps_signature
    )


def test_standard_warmup_conditions_ignore_target_active_set():
    baseline = {
        "pulse_width_active_set_mode": "none",
        "pulse_width_active_threshold": 0.01,
        "pulse_width_active_margin": 3,
        "state_scaling": "full",
    }
    masked = {
        **baseline,
        "pulse_width_active_set_mode": "historical",
        "pulse_width_active_threshold": 0.1,
        "pulse_width_active_margin": 5,
    }

    assert periodic_example._target_independent_warmup_conditions(
        masked
    ) == periodic_example._target_independent_warmup_conditions(baseline)


class _DualWarmStartSolver:
    def __init__(self, horizon, terminal_lam_size=1):
        self.values = {
            (stage, "lam"): np.array([stage + 1.0]) for stage in range(horizon + 1)
        }
        self.values[(horizon, "lam")] = np.full(terminal_lam_size, horizon + 1.0)
        self.values.update(
            {(stage, "pi"): np.array([10.0 + stage]) for stage in range(horizon)}
        )

    def get(self, stage, field):
        return self.values[(stage, field)].copy()

    def set(self, stage, field, values):
        self.values[(stage, field)] = np.asarray(values, dtype=float).copy()


def test_acados_dual_warm_start_can_reset_all_multipliers():
    solver = _DualWarmStartSolver(horizon=3)

    summary = periodic_example.apply_acados_dual_warm_start(
        solver, horizon=3, mode="reset", shift_stages=1
    )

    assert summary == {
        "mode": "reset",
        "shift_stages": 0,
        "zeroed_tail_stages": 4,
    }
    assert all(not np.any(values) for values in solver.values.values())


@pytest.mark.parametrize(
    ("status", "passes_tolerance", "expected_mode", "certified"),
    (
        (0, True, "preserve", True),
        (2, True, "reset", False),
        (0, False, "reset", False),
        (None, None, "reset", False),
    ),
)
def test_acados_duals_are_only_preserved_after_primal_dynamics_certification(
    status, passes_tolerance, expected_mode, certified
):
    feasibility = (
        None
        if passes_tolerance is None
        else {"passes_tolerance": passes_tolerance}
    )

    mode, was_certified = periodic_example.select_acados_dual_warm_start_mode(
        "preserve", status, feasibility
    )

    assert mode == expected_mode
    assert was_certified is certified


def test_acados_dual_warm_start_can_shift_one_cycle_and_zero_tail():
    solver = _DualWarmStartSolver(horizon=3)

    summary = periodic_example.apply_acados_dual_warm_start(
        solver, horizon=3, mode="shift", shift_stages=1
    )

    assert summary == {
        "mode": "shift",
        "shift_stages": 1,
        "zeroed_tail_stages": 1,
    }
    np.testing.assert_array_equal(
        [solver.values[(stage, "lam")][0] for stage in range(4)], [2, 3, 4, 0]
    )
    np.testing.assert_array_equal(
        [solver.values[(stage, "pi")][0] for stage in range(3)], [11, 12, 0]
    )


def test_acados_dual_shift_zeros_structurally_incompatible_terminal_multipliers():
    solver = _DualWarmStartSolver(horizon=3, terminal_lam_size=2)

    periodic_example.apply_acados_dual_warm_start(
        solver, horizon=3, mode="shift", shift_stages=1
    )

    np.testing.assert_array_equal(solver.values[(2, "lam")], [0.0])


def _ipopt_dual_warm_start_fixture():
    interface = SimpleNamespace(
        lam_g=None,
        lam_x=None,
        limits={"lbg": np.zeros(3), "x0": np.zeros(4)},
    )
    nmpc = SimpleNamespace(ocp_solver=interface, _is_warm_starting=False)
    solution = SimpleNamespace(
        lam_g=np.array([1.0, 2.0, 3.0]),
        lam_x=np.array([4.0, 5.0, 6.0, 7.0]),
    )
    return nmpc, solution


def test_ipopt_dual_warm_start_can_transfer_constraint_multipliers_only():
    nmpc, solution = _ipopt_dual_warm_start_fixture()

    summary = periodic_example.apply_ipopt_dual_warm_start(
        nmpc, solution, mode="constraints"
    )

    assert summary == {
        "mode": "constraints",
        "applied": True,
        "lam_g_size": 3,
        "lam_x_size": 0,
        "reason": None,
    }
    np.testing.assert_array_equal(nmpc.ocp_solver.lam_g, solution.lam_g)
    assert nmpc.ocp_solver.lam_x is None
    # Avoid Bioptim's aggressive set_warm_start_options(1e-10): the configured
    # IPOPT solver already accepts lam_g0 while retaining its robust mu_init.
    assert nmpc._is_warm_starting is False


def test_ipopt_dual_warm_start_can_include_bound_multipliers():
    nmpc, solution = _ipopt_dual_warm_start_fixture()

    summary = periodic_example.apply_ipopt_dual_warm_start(nmpc, solution, mode="all")

    assert summary["applied"] is True
    assert summary["lam_x_size"] == 4
    np.testing.assert_array_equal(nmpc.ocp_solver.lam_x, solution.lam_x)


def test_ipopt_dual_warm_start_can_transfer_bound_multipliers_only():
    nmpc, solution = _ipopt_dual_warm_start_fixture()

    summary = periodic_example.apply_ipopt_dual_warm_start(
        nmpc, solution, mode="bounds"
    )

    assert summary["applied"] is True
    assert summary["lam_g_size"] == 0
    assert summary["lam_x_size"] == 4
    assert nmpc.ocp_solver.lam_g is None
    np.testing.assert_array_equal(nmpc.ocp_solver.lam_x, solution.lam_x)


def test_ipopt_dual_warm_start_rejects_nonfinite_or_wrong_sized_duals():
    nmpc, solution = _ipopt_dual_warm_start_fixture()
    solution.lam_g = np.array([1.0, np.nan, 3.0])

    nonfinite = periodic_example.apply_ipopt_dual_warm_start(
        nmpc, solution, mode="constraints"
    )
    assert nonfinite["applied"] is False
    assert nonfinite["reason"] == "invalid_constraint_multipliers"

    solution.lam_g = np.ones(2)
    wrong_size = periodic_example.apply_ipopt_dual_warm_start(
        nmpc, solution, mode="constraints"
    )
    assert wrong_size["applied"] is False
    assert wrong_size["reason"] == "invalid_constraint_multipliers"


def test_optional_nlp_dual_warm_starts_preserve_shifted_primal():
    nmpc, solution = _ipopt_dual_warm_start_fixture()
    shifted_primal = np.array([[10.0, 11.0, 12.0]])
    nmpc.nlp = [
        SimpleNamespace(x_init={"q": SimpleNamespace(init=shifted_primal.copy())})
    ]

    madnlp = periodic_example.apply_nlp_dual_warm_start(
        nmpc, solution, solver_name="madnlp", mode="all"
    )

    assert madnlp["solver"] == "madnlp"
    assert madnlp["applied"] is True
    np.testing.assert_array_equal(nmpc.nlp[0].x_init["q"].init, shifted_primal)
    np.testing.assert_array_equal(nmpc.ocp_solver.lam_g, solution.lam_g)
    np.testing.assert_array_equal(nmpc.ocp_solver.lam_x, solution.lam_x)

    nmpc.ocp_solver.lam_g = None
    nmpc.ocp_solver.lam_x = None
    alpaqa = periodic_example.apply_nlp_dual_warm_start(
        nmpc, solution, solver_name="alpaqa", mode="constraints"
    )
    assert alpaqa["solver"] == "alpaqa"
    assert alpaqa["lam_g_size"] == 3
    assert alpaqa["lam_x_size"] == 0
    assert nmpc.ocp_solver.lam_x is None

    with np.testing.assert_raises_regex(ValueError, "only supports"):
        periodic_example.apply_nlp_dual_warm_start(
            nmpc, solution, solver_name="alpaqa", mode="bounds"
        )


def test_disabled_nlp_dual_warm_start_clears_stale_multipliers():
    nmpc, solution = _ipopt_dual_warm_start_fixture()
    nmpc.ocp_solver.lam_g = np.ones(3)
    nmpc.ocp_solver.lam_x = np.ones(2)

    summary = periodic_example.apply_nlp_dual_warm_start(
        nmpc, solution, solver_name="madnlp", mode="off"
    )

    assert summary["reason"] == "disabled"
    assert nmpc.ocp_solver.lam_g is None
    assert nmpc.ocp_solver.lam_x is None


def test_ipopt_dual_warm_start_cli_defaults_to_bound_multipliers():
    periodic_args = periodic_example.build_argument_parser().parse_args([])
    comparison_args = comparison_example.build_cli().parse_args([])

    assert periodic_args.ipopt_dual_warm_start_mode == "bounds"
    assert comparison_args.ipopt_dual_warm_start_mode == "bounds"
    assert periodic_args.fatrop_dual_warm_start_mode == "off"
    assert comparison_args.fatrop_dual_warm_start_mode == "off"
    assert periodic_args.madnlp_dual_warm_start_mode == "off"
    assert periodic_args.alpaqa_dual_warm_start_mode == "constraints"
    assert comparison_args.madnlp_dual_warm_start_mode == "off"
    assert comparison_args.alpaqa_dual_warm_start_mode == "constraints"
    assert periodic_args.primal_feasibility_threshold is None
    assert comparison_args.primal_feasibility_threshold is None


def test_full_contact_stabilization_is_opt_in_on_both_clis():
    periodic_parser = periodic_example.build_argument_parser()
    comparison_parser = comparison_example.build_cli()

    assert not periodic_parser.parse_args([]).full_contact_constraints_all_nodes
    assert not comparison_parser.parse_args([]).full_contact_constraints_all_nodes
    assert not periodic_parser.parse_args([]).full_contact_constraints_terminal
    assert not comparison_parser.parse_args([]).full_contact_constraints_terminal
    assert periodic_parser.parse_args(
        ["--full-contact-constraints-terminal"]
    ).full_contact_constraints_terminal
    assert comparison_parser.parse_args(
        ["--full-contact-constraints-terminal"]
    ).full_contact_constraints_terminal
    assert not periodic_parser.parse_args([]).full_contact_position_terminal
    assert not comparison_parser.parse_args([]).full_contact_position_terminal
    assert periodic_parser.parse_args(
        ["--full-contact-position-terminal"]
    ).full_contact_position_terminal
    assert comparison_parser.parse_args(
        ["--full-contact-position-terminal"]
    ).full_contact_position_terminal
    assert periodic_parser.parse_args(
        ["--full-contact-constraints-all-nodes"]
    ).full_contact_constraints_all_nodes
    assert comparison_parser.parse_args(
        ["--full-contact-constraints-all-nodes"]
    ).full_contact_constraints_all_nodes
    assert not periodic_parser.parse_args([]).full_contact_position_all_nodes
    assert not comparison_parser.parse_args([]).full_contact_position_all_nodes
    assert periodic_parser.parse_args(
        ["--full-contact-position-all-nodes"]
    ).full_contact_position_all_nodes
    assert comparison_parser.parse_args(
        ["--full-contact-position-all-nodes"]
    ).full_contact_position_all_nodes
    assert (
        periodic_parser.parse_args(
            ["--full-contact-position-tolerance", "2e-5"]
        ).full_contact_position_tolerance
        == 2e-5
    )
    assert (
        comparison_parser.parse_args(
            ["--full-contact-position-tolerance", "2e-5"]
        ).full_contact_position_tolerance
        == 2e-5
    )
    assert periodic_parser.parse_args(
        ["--transfer-contact-manifold-projection"]
    ).transfer_contact_manifold_projection
    assert comparison_parser.parse_args(
        ["--shared-transfer-contact-projection"]
    ).shared_transfer_contact_projection
    assert (
        comparison_parser.parse_args(
            ["--shared-transfer-contact-projection"]
        ).shared_transfer_contact_projection_mode
        == "position"
    )


def test_common_primal_threshold_is_independent_of_nlp_solver_tolerance():
    args = SimpleNamespace(
        solver="madnlp",
        nlp_tolerance=1e-8,
        primal_feasibility_threshold=1e-5,
    )

    tolerance = periodic_example._window_feasibility_tolerance(args)

    assert tolerance == pytest.approx(1e-6)


def test_nlp_solver_stats_snapshot_keeps_oracle_timing_without_iterations():
    stats = {
        "t_wall_total": 2.0,
        "t_wall_nlp_hess_l": 1.0,
        "t_proc_nlp_hess_l": 3.0,
        "n_call_nlp_hess_l": 4,
        "iter_count": 5,
        "success": True,
        "return_status": "Solve_Succeeded",
        "iterations": {"inf_pr": [1.0, 0.0]},
        "unrelated": object(),
    }
    nmpc = SimpleNamespace(
        ocp_solver=SimpleNamespace(
            shaked_ocp_solver=SimpleNamespace(stats=lambda: stats)
        )
    )

    snapshot = periodic_example.snapshot_nlp_solver_stats(nmpc)

    assert snapshot["t_wall_total"] == 2.0
    assert snapshot["n_call_nlp_hess_l"] == 4
    assert snapshot["return_status"] == "Solve_Succeeded"
    assert "iterations" not in snapshot
    assert "unrelated" not in snapshot


def test_standard_warmup_seed_resolves_repository_relative_path(tmp_path, monkeypatch):
    repository_root = tmp_path / "repository"
    example_root = repository_root / "examples" / "fes_multibody" / "cycling"
    seed = example_root / "result" / "cache" / "warmup.npz"
    seed.parent.mkdir(parents=True)
    seed.touch()
    fake_module = example_root / "cycling_pulse_width_mhe_acados_periodic.py"
    monkeypatch.setattr(periodic_example, "__file__", str(fake_module))
    monkeypatch.chdir(example_root)

    resolved = periodic_example._resolve_standard_warmup_seed(
        "examples/fes_multibody/cycling/result/cache/warmup.npz"
    )

    assert resolved == seed


def test_fatigue_benchmark_defaults_to_all_solvers_and_full_scaling():
    args = comparison_example.build_cli().parse_args([])
    periodic_args = periodic_example.build_argument_parser().parse_args([])

    assert args.solvers == ("ipopt", "acados", "fatrop", "madnlp")
    assert args.objective == "fatigue"
    assert periodic_args.objective == "fatigue"
    assert args.objective_shape == "quadratic"
    assert args.state_scaling == "full"
    assert periodic_example.build_cost_fun_weight({"fatigue"}) == [0, 1, 0]
    assert periodic_example.parse_objectives("") == {"fatigue"}


def test_fatigue_only_objective_disables_terminal_wheel_regularization(
    monkeypatch,
):
    class FakeObjectiveList:
        def __init__(self):
            self.entries = []

        def add(self, objective, **kwargs):
            self.entries.append((objective, kwargs))

    monkeypatch.setattr(mhe_example, "ObjectiveList", FakeObjectiveList)
    model = SimpleNamespace(muscles_dynamics_model=[])

    objectives = mhe_example.set_objective_functions(
        model=model,
        minimize_force=False,
        minimize_fatigue=True,
        minimize_control=False,
        cost_fun_weight=[0, 1, 0],
        target=2 * np.pi,
        objective_shape="quadratic",
        terminal_wheel_regularization_weight=0.0,
    )

    assert len(objectives.entries) == 1
    assert (
        objectives.entries[0][0]
        is mhe_example.CustomObjective.minimize_overall_muscle_fatigue
    )
    assert objectives.entries[0][1]["weight"] == 10000
    assert objectives.entries[0][1]["quadratic"] is True


def test_benchmark_json_summary_contains_comparable_fatigue_metrics(tmp_path):
    result = _benchmark_result([0, 0], solver_success=True, success=True)
    result.update(
        physical_success=True,
        status=0,
        objective=12.5,
        solver_time_s=2.0,
        wall_time_s=2.2,
        end_to_end_wall_time_s=2.5,
        attempted_windows=2,
        successful_windows=2,
        window_solutions=[],
        nlp_solver_stats=[
            {
                "window": 0,
                "iter_count": 12,
                "return_status": "Solve_Succeeded",
                "t_wall_total": 1.25,
                "t_wall_nlp_hess_l": 0.75,
            }
        ],
        compiled_nlp_reuse={
            "enabled": True,
            "compiled_library_build_count": 1,
            "compiled_library_reused": True,
            "graph_rebuild_detected": False,
        },
        state_boundary_jumps={
            "available": True,
            "boundary_count": 1,
            "by_state": {
                "omega": {
                    "terminal_left": [[-6.2]],
                    "initial_right": [[-6.2]],
                    "jump": [[0.0]],
                    "maximum_absolute_jump": 0.0,
                    "root_mean_square_jump": 0.0,
                }
            },
        },
    )
    result["args"].objective = "fatigue"
    result["args"].objective_shape = "quadratic"
    result["args"].solver = "madnlp"
    result["args"].constant_crank_torque = -0.24
    result["args"].max_consecutive_failing = 2
    result["args"].ipopt_dual_warm_start_mode = "all"
    result["args"].max_ipopt_iterations = 2000
    result["args"].wheel_qdot_regularization_target = -2.0 * np.pi
    result["args"].wheel_qdot_bound_margin = 3.0
    result["args"].benchmark_profile = "scientific-radau5"
    result["args"].profile_integrity = True
    result["args"].scientific_status = "candidate"
    result["args"].calcium_analytical_periodic_value = 0.1629821583533315
    result["args"].ding_sum_stim_truncation = 6
    result["args"].activate_passive_force_relationship = True
    result["args"].enforce_start_constraints = True
    result["acados_maxiter_retry_summaries"] = [
        {"window": 13, "retry_status": 2}
    ]

    output_path = comparison_example.write_benchmark_summary(
        tmp_path / "benchmark.json", {"madnlp": result}
    )
    payload = comparison_example.json.loads(output_path.read_text())

    assert payload["schema_version"] == 3
    assert payload["runtime"]["logical_cpu_count"] >= 1
    assert "OMP_NUM_THREADS" in payload["runtime"]["thread_environment"]
    assert payload["configurations"]["madnlp"]["objective"] == "fatigue"
    assert payload["configurations"]["madnlp"]["constant_crank_torque"] == -0.24
    assert payload["configurations"]["madnlp"]["max_consecutive_failing"] == 2
    assert payload["configurations"]["madnlp"]["ipopt_dual_warm_start_mode"] == "all"
    assert payload["configurations"]["madnlp"]["max_ipopt_iterations"] == 2000
    assert payload["configurations"]["madnlp"][
        "wheel_qdot_regularization_target"
    ] == pytest.approx(-2.0 * np.pi)
    assert payload["configurations"]["madnlp"]["wheel_qdot_bound_margin"] == 3.0
    assert payload["configurations"]["madnlp"]["benchmark_profile"] == (
        "scientific-radau5"
    )
    assert payload["configurations"]["madnlp"]["profile_integrity"] is True
    assert payload["configurations"]["madnlp"]["scientific_status"] == "candidate"
    assert payload["configurations"]["madnlp"][
        "calcium_analytical_periodic_value"
    ] == pytest.approx(0.1629821583533315)
    assert payload["configurations"]["madnlp"]["ding_sum_stim_truncation"] == 6
    assert payload["configurations"]["madnlp"][
        "activate_passive_force_relationship"
    ] is True
    assert payload["configurations"]["madnlp"]["enforce_start_constraints"] is True
    row = payload["results"][0]
    assert row["solver"] == "madnlp"
    assert row["success"] is True
    assert row["validated_cycles"] == 3
    assert row["min_A_capacity_ratio"] == 0.8
    assert row["max_mean_normalized_fatigue"] > 0
    assert row["fatigue_auc_cycles"] > 0
    assert row["window_objective_sum"] == 12.5
    assert row["executed_fatigue_objective"] > 0
    assert row["muscle_fatigue"][0]["muscle"] == "Biceps"
    assert (
        row["muscle_fatigue"][0]["executed_fatigue_objective"]
        == row["executed_fatigue_objective"]
    )
    assert row["pulse_width_cycle_variation"]["available"] is True
    assert "cycle_100" in row["isolated_window_checkpoints"]
    assert row["isolated_window_checkpoints"]["cycle_100"]["diagnostic_only"] is True
    assert row["external_crank_power"]["role"] == "unavailable"
    assert row["cycle_boundary_wheel_angle"]["maximum_absolute_error_rad"] is not None
    assert row["stop"]["label"] == "completed_requested_horizon"
    assert row["fatigue_by_state"]
    assert set(row["prefix_fatigue_checkpoints"]) == {
        "cycle_1",
        "cycle_2",
        "cycle_3",
    }
    assert row["control_saturation"][0]["upper_fraction"] > 0
    assert [window["status"] for window in row["windows"]] == [0, 0]
    assert [window["rho"] for window in row["windows"]] == [1, 2]
    assert row["windows"][0]["native_status"] == "Solve_Succeeded"
    assert row["nlp_solver_stats"][0]["t_wall_nlp_hess_l"] == 0.75
    assert row["compiled_nlp_reuse"]["compiled_library_build_count"] == 1
    assert row["acados_maxiter_retry_summaries"] == [
        {"window": 13, "retry_status": 2}
    ]
    assert row["state_boundary_jumps"]["boundary_count"] == 1
    assert row["state_boundary_jumps"]["by_state"]["omega"]["jump"] == [[0.0]]


def test_acados_dual_warm_start_summaries_survive_benchmark_serialization():
    result = _benchmark_result([0], solver_success=True, success=True)
    result["acados_dual_warm_start_summaries"] = [
        {
            "window": 0,
            "mode": "reset",
            "shift_stages": 0,
            "zeroed_tail_stages": 31,
        },
        {
            "window": 1,
            "mode": "preserve",
            "shift_stages": 0,
            "zeroed_tail_stages": 0,
        },
    ]

    row = comparison_example.solver_overview_rows({"acados": result})[0]

    assert row["warm_start"]["dual_summaries"] == [
        {
            "window": 0,
            "mode": "reset",
            "shift_stages": 0,
            "zeroed_tail_stages": 31,
        },
        {
            "window": 1,
            "mode": "preserve",
            "shift_stages": 0,
            "zeroed_tail_stages": 0,
        },
    ]
    assert [window["dual_warm_start_mode"] for window in row["windows"]] == [
        "reset"
    ]


def test_failed_rho_checkpoints_preserve_neighboring_pw_active_sets():
    lower = 131.405e-6
    upper = 600e-6

    class SnapshotSolution:
        def __init__(self, status, offset):
            self.status = status
            self.offset = offset

        def decision_states(self, to_merge=None):
            return {"A_Biceps": np.array([[3000.0, 2999.0 - self.offset]])}

        def decision_controls(self, to_merge=None):
            return {
                "last_pulse_width_Biceps": np.array(
                    [[lower, 300e-6 + self.offset * 1e-9, upper]]
                )
            }

    result = {
        "args": SimpleNamespace(cycles_per_window=1),
        "window_solutions": [
            SnapshotSolution(0, 0),
            SnapshotSolution(1, 1),
            SnapshotSolution(0, 2),
            SnapshotSolution(0, 3),
        ],
        "window_statuses": [0, 1, 0, 0],
        "window_objectives": [1.0, 2.0, 3.0, 4.0],
        "window_feasibility": [
            {"passes_tolerance": True},
            {"passes_tolerance": False},
            {"passes_tolerance": True},
            {"passes_tolerance": True},
        ],
        "solver_success": False,
        "covered_cycles": 4,
        "fatigue_capacity_scales": {"A_Biceps": 3000.0},
        "control_bounds": {
            "last_pulse_width_Biceps": {"lower": lower, "upper": upper}
        },
    }

    checkpoints = comparison_example.isolated_window_checkpoint_snapshots(result)

    assert {"cycle_1", "cycle_2", "cycle_3"}.issubset(checkpoints)
    assert checkpoints["cycle_1"]["belongs_to_strict_prefix"] is True
    assert checkpoints["cycle_2"]["primal_feasible"] is False
    active_set = checkpoints["cycle_2"]["pulse_width_active_set"]["Biceps"]
    assert active_set["lower_bound_us"] == pytest.approx(131.405)
    assert active_set["upper_bound_us"] == pytest.approx(600.0)
    assert active_set["lower_active_indices"] == [0]
    assert active_set["upper_active_indices"] == [2]


def test_independent_bound_violation_accepts_infinite_bounds():
    violation = periodic_example._maximum_bound_violation(
        values=[12.0, 0.5],
        lower_bounds=[-np.inf, 0.0],
        upper_bounds=[np.inf, 1.0],
    )

    assert violation == 0.0


def test_feasibility_rejects_solution_without_constraint_metric():
    class FakeSolution:
        constraints = None
        inf_pr = None
        vector = np.array([0.5])
        ocp = SimpleNamespace(
            ocp_solver=SimpleNamespace(
                limits={
                    "lbx": np.array([0.0]),
                    "ubx": np.array([1.0]),
                }
            )
        )

        @staticmethod
        def decision_states(to_merge=None):
            return {"q": np.zeros((3, 2))}

        @staticmethod
        def decision_controls(to_merge=None):
            return {"u": np.zeros((1, 1))}

    feasibility = periodic_example._solution_feasibility_summary(
        FakeSolution(), tolerance=1e-6
    )

    assert feasibility["decision_bound_violation"] == 0.0
    assert feasibility["decision_bound_violation_index"] is None
    assert feasibility["decision_bound_block"] is None
    assert feasibility["constraint_feasibility_available"] is False
    assert feasibility["passes_tolerance"] is False
    assert feasibility["failure_reason"] == "constraint_feasibility_unavailable"


def test_feasibility_uses_constraint_and_decision_bounds():
    class FakeSolution:
        constraints = np.array([1.0 + 2e-5, 0.0])
        inf_pr = None
        vector = np.array([0.5, 1.1])
        ocp = SimpleNamespace(
            ocp_solver=SimpleNamespace(
                limits={
                    "lbg": np.array([1.0, -np.inf]),
                    "ubg": np.array([1.0, np.inf]),
                    "lbx": np.array([0.0, 0.0]),
                    "ubx": np.array([1.0, 1.0]),
                }
            )
        )

        @staticmethod
        def decision_states(to_merge=None):
            return {"q": np.zeros((3, 2))}

        @staticmethod
        def decision_controls(to_merge=None):
            return {"u": np.zeros((1, 1))}

    feasibility = periodic_example._solution_feasibility_summary(
        FakeSolution(), tolerance=1e-6
    )

    np.testing.assert_allclose(feasibility["constraint_bound_violation"], 2e-5)
    np.testing.assert_allclose(feasibility["decision_bound_violation"], 0.1)
    assert feasibility["decision_bound_violation_index"] == 1
    assert feasibility["decision_bound_violation_value"] == 1.1
    assert feasibility["decision_bound_lower"] == 0.0
    assert feasibility["decision_bound_upper"] == 1.0
    np.testing.assert_allclose(feasibility["effective_primal_infeasibility"], 0.1)
    assert feasibility["passes_tolerance"] is False


def test_feasibility_recomputes_constraints_from_compiled_nlp():
    from casadi import MX

    x = MX.sym("x", 2)
    interface = SimpleNamespace(
        nlp={"x": x, "g": x[0] + 2 * x[1]},
        limits={
            "lbg": np.array([1.0]),
            "ubg": np.array([1.0]),
            "lbx": np.array([0.0, 0.0]),
            "ubx": np.array([1.0, 1.0]),
        },
    )

    class FakeCompiledSolution:
        constraints = None
        inf_pr = None
        vector = np.array([0.5, 0.25])
        ocp = SimpleNamespace(ocp_solver=interface)

        @staticmethod
        def decision_states(to_merge=None):
            return {"q": np.zeros((3, 2))}

        @staticmethod
        def decision_controls(to_merge=None):
            return {"u": np.zeros((1, 1))}

    feasibility = periodic_example._solution_feasibility_summary(
        FakeCompiledSolution(), tolerance=1e-6
    )

    assert feasibility["constraint_values_source"] == "recomputed_nlp"
    assert feasibility["constraint_bound_violation"] == 0.0
    assert feasibility["constraint_feasibility_available"] is True
    assert feasibility["passes_tolerance"] is True


def test_feasibility_snapshot_is_not_recomputed_with_next_window_bounds():
    class FakeSolution:
        constraints = np.array([0.0])
        inf_pr = np.array([0.0])
        vector = np.array([0.5])
        ocp = SimpleNamespace(
            ocp_solver=SimpleNamespace(
                limits={
                    "lbg": np.array([0.0]),
                    "ubg": np.array([0.0]),
                    "lbx": np.array([0.0]),
                    "ubx": np.array([1.0]),
                }
            )
        )

        @staticmethod
        def decision_states(to_merge=None):
            return {"q": np.zeros((3, 2))}

        @staticmethod
        def decision_controls(to_merge=None):
            return {"u": np.zeros((1, 1))}

    solution = FakeSolution()
    snapshot = periodic_example._solution_feasibility_summary(solution, tolerance=1e-6)
    solution._cocofest_feasibility_summary = snapshot
    solution.ocp.ocp_solver.limits["lbx"][:] = 10.0
    solution.ocp.ocp_solver.limits["ubx"][:] = 11.0

    feasibility = periodic_example._solution_feasibility_summary(
        solution, tolerance=1e-6
    )

    assert feasibility["decision_bound_violation"] == 0.0
    assert feasibility["passes_tolerance"] is True


def test_acados_shooting_residuals_are_part_of_the_physical_feasibility_audit():
    base = {
        "passes_tolerance": True,
        "failure_reason": None,
        "feasibility_threshold": 1e-5,
        "constraint_infeasibility": 1e-9,
        "effective_primal_infeasibility": 1e-9,
        "maximum_bound_violation": 1e-9,
    }

    rejected = periodic_example.augment_feasibility_with_acados_residuals(
        base,
        {"residuals": np.array([0.2, 0.13, 1e-12, 1e-8])},
    )
    accepted = periodic_example.augment_feasibility_with_acados_residuals(
        base,
        {"residuals": np.array([0.2, 2e-7, 3e-8, 1e-8])},
    )

    assert rejected["passes_tolerance"] is False
    assert rejected["failure_reason"] == "acados_primal_residual_above_threshold"
    np.testing.assert_allclose(rejected["acados_primal_residual"], 0.13)
    np.testing.assert_allclose(rejected["effective_primal_infeasibility"], 0.13)
    assert accepted["passes_tolerance"] is True
    np.testing.assert_allclose(accepted["acados_primal_residual"], 2e-7)


def test_acados_residuals_replace_missing_exported_constraint_feasibility():
    base = {
        "passes_tolerance": False,
        "failure_reason": "constraint_feasibility_unavailable",
        "feasibility_threshold": 1e-5,
        "trajectories_finite": True,
        "constraints_finite": True,
        "inf_pr_available": False,
        "final_inf_pr": None,
        "constraint_infeasibility": None,
        "effective_primal_infeasibility": None,
        "maximum_bound_violation": None,
    }

    audited = periodic_example.augment_feasibility_with_acados_residuals(
        base,
        {"residuals": np.array([3e-3, 2e-9, 1e-12, 1e-8])},
    )

    assert audited["passes_tolerance"] is True
    assert audited["failure_reason"] is None
    assert audited["constraint_feasibility_available"] is True
    np.testing.assert_allclose(audited["effective_primal_infeasibility"], 2e-9)


def test_acados_uses_the_same_absolute_primal_feasibility_threshold_as_nlps():
    args = periodic_example.build_argument_parser().parse_args(
        [
            "--solver",
            "acados",
            "--primal-feasibility-threshold",
            "1e-5",
        ]
    )

    assert periodic_example._window_feasibility_tolerance(args) == pytest.approx(1e-6)


def test_solver_comparison_forwards_the_physical_threshold_to_acados():
    ipopt_args = SimpleNamespace()
    acados_args = SimpleNamespace()

    comparison_example._set_common_primal_feasibility_threshold(
        (ipopt_args, acados_args),
        1e-5,
    )

    assert ipopt_args.primal_feasibility_threshold == pytest.approx(1e-5)
    assert acados_args.primal_feasibility_threshold == pytest.approx(1e-5)


def test_github_benchmark_report_compares_patterns_and_writes_csv(tmp_path):
    def entry(solver, pulse_width_us):
        return {
            "source": f"{solver}.json",
            "runtime": {
                "provenance": {
                    "BIOPTIM_BENCHMARK_COMMIT": (
                        "mad-commit" if solver != "alpaqa" else "alpaqa-commit"
                    )
                }
            },
            "configuration": {"n_windows": 30},
            "result": {
                "solver": solver,
                "success": True,
                "validated_cycles": 30,
                "end_to_end_wall_time_s": 4.0,
                "initial_guess_preparation_time_s": 1.0,
                "hot_wall_time_median_s": 0.1,
                "hot_wall_time_p90_s": 0.2,
                "stop": {"label": "completed_requested_horizon"},
                "windows": [
                    {
                        "rho": 1,
                        "status": 0,
                        "native_status": "success",
                        "solver_converged": True,
                        "primal_feasible": True,
                        "validated": True,
                        "iterations": 2,
                        "objective": 1.0,
                        "solver_time_s": 0.08,
                        "wall_time_s": 0.1,
                        "feasibility": {
                            "effective_primal_infeasibility": 1e-8,
                            "inf_pr_available": True,
                        },
                    }
                ],
                "stimulation_patterns": {
                    "cycle_10": {
                        "available": True,
                        "cycle": 10,
                        "rho": 10,
                        "phase_fraction": [0.0, 0.5],
                        "crank_angle_rad": [0.0, -np.pi],
                        "crank_phase_rad": [0.0, np.pi],
                        "crank_velocity_rad_s": [-1.0, -1.0],
                        "muscles": {
                            "Biceps": {
                                "pulse_width_s": [
                                    value * 1e-6 for value in pulse_width_us
                                ],
                                "pulse_width_us": pulse_width_us,
                                "normalized_to_bounds": [0.2, 0.4],
                                "minimum_s": min(pulse_width_us) * 1e-6,
                                "mean_s": np.mean(pulse_width_us) * 1e-6,
                                "maximum_s": max(pulse_width_us) * 1e-6,
                                "lower_bound_fraction": 0.0,
                                "upper_bound_fraction": 0.0,
                            }
                        },
                    }
                },
            },
        }

    entries = [entry("ipopt", [10.0, 20.0]), entry("alpaqa", [11.0, 21.0])]

    comparisons = benchmark_report.stimulation_comparisons(entries)
    benchmark_report.write_rho_csv(tmp_path / "rho.csv", entries)
    benchmark_report.write_stimulation_csv(tmp_path / "patterns.csv", entries)
    markdown = benchmark_report.render_markdown(
        entries, [], missing_solvers=("madnlp",)
    )

    np.testing.assert_allclose(comparisons[0]["root_mean_square_error_us"], 1.0)
    assert "branches d’intégration Bioptim différentes" in markdown
    assert "# Benchmark cyclage FES — 30 RHO" in markdown
    assert "INCOMPLÈTE" in markdown
    assert "MADNLP" in markdown
    assert "native_status" in (tmp_path / "rho.csv").read_text()
    assert "crank_phase_rad" in (tmp_path / "patterns.csv").read_text()


def test_github_benchmark_compares_physical_threshold_not_solver_tolerance():
    entries = [
        {
            "configuration": {
                "nlp_tolerance": 1e-6,
                "primal_feasibility_threshold": 1e-5,
            },
            "result": {"solver": "ipopt"},
        },
        {
            "configuration": {
                "nlp_tolerance": 1e-8,
                "primal_feasibility_threshold": 1e-5,
            },
            "result": {"solver": "madnlp"},
        },
    ]

    assert benchmark_report.configuration_mismatches(entries) == []
    entries[1]["configuration"]["primal_feasibility_threshold"] = 2e-5

    mismatches = benchmark_report.configuration_mismatches(entries)

    assert [item["field"] for item in mismatches] == ["primal_feasibility_threshold"]


def test_github_acados_runner_uses_reference_and_option_profiles_sequentially():
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "cycling_solver_benchmark_linux.yml"
    ).read_text(encoding="utf-8")

    assert "acados_smoke_rhos:" in workflow
    assert re.search(
        r"acados_smoke_rhos:\s+" r"description:.*\s+required: true\s+default: \"100\"",
        workflow,
    )
    assert "acados_option_rhos:" in workflow
    assert "compile_nlp_evaluators:" in workflow
    assert "refined_collocation_validation:" in workflow
    assert "refined_collocation_rhos:" in workflow
    assert "collocation_diagnostic_rhos:" in workflow
    assert "Run IPOPT reduced Radau degree 5" in workflow
    assert "Run MadNLP MUMPS reduced Radau degree 5" in workflow
    assert "Run IPOPT reduced Radau degree 4" in workflow
    assert "Run IPOPT reduced Radau degree 6" in workflow
    assert "Run MadNLP MUMPS reduced Radau degree 4" in workflow
    assert "Run MadNLP MUMPS reduced Radau degree 6" in workflow
    assert "sqp-irk-two-stage-fast-guard-2p6" in workflow
    assert "sqp-irk-two-stage-fast-guard-2p55" in workflow
    assert "--acados-wheel-qdot-fast-bound-margin 2.6" in workflow
    assert "--acados-wheel-qdot-fast-bound-margin 2.55" in workflow
    assert "ipopt-radau5-reduced" in workflow
    assert "madnlp-mumps-radau5-reduced" in workflow
    assert "ipopt-radau5-full" in workflow
    assert "madnlp-mumps-radau5-full" in workflow
    assert "Scientific collocation gate is not strict-successful" in workflow
    assert workflow.count("5 scientific-radau5") == 4
    assert '"$BENCHMARK_CYCLES" "${{ inputs.compile_nlp_evaluators }}"' in workflow
    assert (
        "run_cycling_benchmark_case.sh ipopt ipopt full mumps collocation "
        'benchmark-results "$BENCHMARK_CYCLES" false' in workflow
    )
    assert (
        "run_cycling_benchmark_case.sh madnlp-mumps madnlp full mumps collocation "
        'benchmark-results "$BENCHMARK_CYCLES" false' in workflow
    )
    assert (
        "run_cycling_benchmark_case.sh madnlp-mumps madnlp reduced mumps collocation "
        'benchmark-results "$BENCHMARK_CYCLES" "${{ inputs.compile_nlp_evaluators }}"'
        in workflow
    )
    assert (
        "run_cycling_benchmark_case.sh fatrop-collocation fatrop full fatrop collocation "
        'benchmark-results "$BENCHMARK_CYCLES" false sx none' in workflow
    )
    assert (
        "run_cycling_benchmark_case.sh fatrop-collocation fatrop reduced fatrop collocation "
        'benchmark-results "$BENCHMARK_CYCLES" "${{ inputs.compile_nlp_evaluators }}"'
        in workflow
    )
    assert (
        'if [[ "$BENCHMARK_SOLVER" == "madnlp" && '
        '"$result" == *"madnlp-mumps-full/"* ]]' in workflow
    )
    assert "specified structure of A does not correspond" not in workflow
    assert 'case_requires_compile="$COMPILE_NLP_EVALUATORS"' in workflow
    assert "case_requires_compile=false" in workflow
    assert '[[ "$result" == *-radau[456]-* ]]' in workflow
    assert ".compiled_nlp_reuse.compiled_library_build_count == 1" in workflow
    assert ".compiled_nlp_reuse.graph_rebuild_detected == false" in workflow
    assert ".compiled_nlp_reuse.runtime_bounds_changed == true" in workflow
    assert ".compiled_nlp_reuse.observed_solves == .attempted_windows" in workflow
    assert (
        ".compiled_nlp_reuse.compiled_source_observation_count == .attempted_windows"
        in workflow
    )
    assert "inputs.cycles != 'screen' && inputs.cycles != 'acados'" in workflow
    assert "prepare-acados-stack:" in workflow
    assert (
        "BIOPTIM_PRODUCTION_COMMIT: " "4179bf076b724fe6c4702739b3462e29ae4adef4"
    ) in workflow
    assert (
        workflow.count("bioptim_commit: 4179bf076b724fe6c4702739b3462e29ae4adef4") == 3
    )
    assert "a3499cab16d7605b8efa7255cf89f1af6a7c59c9" not in workflow
    assert "ACADOS_COMMIT: 59d93e17d2985fdd73fc58b8a83ed8f83a024171" in workflow
    assert (
        "ACADOS_INSTALL_SCRIPT_BLOB: 5ac8064ab613251e62560b5de8cbbb9550f5c5d0"
        in workflow
    )
    assert "for mechanics in full reduced" in workflow
    assert workflow.index("prepare_case reduced") < workflow.index("prepare_case full")
    assert (
        '--common-initial-solution "$GITHUB_WORKSPACE/benchmark-seed-result/common-reduced.npz"'
        in workflow
    )
    assert '--common-initial-solution "$common_seed"' in workflow
    assert "--experimental-reduced-acados" in workflow
    assert "--common-initial-solution" in workflow
    assert "run_case sqp-irk-reference" in workflow
    assert "run_case sqp-irk-reset-memory" in workflow
    assert "--acados-reset-solver-before-solve" in workflow
    assert "run_case sqp-irk-qp-hot" in workflow
    assert "--acados-qp-warm-start-level 2" in workflow
    assert "--acados-warm-start-first-qp-from-nlp" in workflow
    assert "run_case sqp-irk-fixed" in workflow
    assert "run_case sqp-irk-anderson" in workflow
    assert "--acados-with-anderson-acceleration" in workflow
    assert "--acados-anderson-activation-threshold 0.1" in workflow
    assert "--acados-check-reuse-possible" in workflow
    assert "--acados-byrd-omojokon-slack-relaxation-factor 1.00001" in workflow
    assert "run_case sqp-byrd-projected-selector-cadence-reg-1-irk" in workflow
    assert "--acados-transfer-select-projected-candidate" in workflow
    assert "run_case sqp-irk-two-stage" in workflow
    assert (
        "--acados-transfer-bound-homotopy-fractions "
        "0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1"
    ) in workflow
    assert "--acados-transfer-bound-homotopy-min-fraction-step 0.001953125" in workflow
    assert "--acados-transfer-bound-homotopy-max-refinements 16" in workflow
    assert "--acados-transfer-bound-homotopy-iterations 40" in workflow
    assert "--acados-transfer-bound-homotopy-solver-tolerance 1e-4" in workflow
    assert "run_case sqp-feasible-qp-irk" in workflow
    assert "run_case sqp-feasibility-qp-irk" not in workflow
    assert "--acados-search-direction-mode FEASIBILITY_QP" not in workflow
    assert "SQP_WITH_FEASIBLE_QP IRK" in workflow
    assert 'run_case sqp-feasible-qp-irk "$mechanics" "$ACADOS_SMOKE_RHOS"' in workflow
    assert "--acados-search-direction-mode BYRD_OMOJOKUN" in workflow
    assert "run_case sqp-rti-irk" in workflow
    assert "SQP_RTI IRK" in workflow
    assert "--acados-control-homotopy-radii 1e-6,1e-5" in workflow
    assert "--acados-control-homotopy-tolerance 2e-2" in workflow
    assert "--acados-qpscaling-scale-objective NO_OBJECTIVE_SCALING" in workflow
    assert "--acados-qpscaling-scale-constraints NO_CONSTRAINT_SCALING" in workflow
    assert "--acados-globalization FIXED_STEP" in workflow
    assert "--acados-control-homotopy-release-final-radius" in workflow
    assert "run_case sqp-erk" in workflow
    assert '--acados-stationarity-tolerance "$stationarity"' in workflow
    assert "--acados-control-homotopy-window-growth 10" in workflow
    assert "--acados-control-homotopy-window-max-radius 1e-5" in workflow
    assert "run_case sqp-irk-active-set-guard reduced" in workflow
    assert "--acados-transfer-active-set-guard-radius 5e-4" in workflow
    assert "--acados-transfer-active-set-guard-margin 1" in workflow
    assert "--acados-transfer-active-set-threshold 1e-6" in workflow
    assert "--max-consecutive-failing 2" in workflow
    assert "cycling-acados-smoke-${{ github.run_id }}" in workflow
    assert workflow.count("name: Save the MadNLP numerical stack") == 2
    assert (
        "matrix.solver == 'madnlp' && "
        "steps.benchmark-madnlp-stack-cache.outputs.cache-hit != 'true'"
    ) in workflow
    assert (
        'echo "${variant}-${mechanics}" >> acados-smoke-results/expected-cases.txt'
        in workflow
    )
    assert (
        "mapfile -t expected_cases < acados-smoke-results/expected-cases.txt"
        in workflow
    )
    assert (
        "mapfile -t reference_cases < acados-smoke-results/reference-cases.txt"
        in workflow
    )
    assert "expected_reference_count=2" in workflow
    assert "expected_reference_count=1" in workflow
    assert '--arg homotopy_only "$ACADOS_HOMOTOPY_ONLY"' in workflow
    assert '$extended == "true" and $homotopy_only != "true"' in workflow
    assert (
        'if [[ "$ACADOS_HOMOTOPY_ONLY" == "true" ]]; then' in workflow
    )
    assert 'run_case sqp-irk-contact-position full "$ACADOS_OPTION_RHOS"' in workflow
    assert 'run_case sqp-irk-reference full "$ACADOS_SMOKE_RHOS"' in workflow
    assert "sqp-irk-reference-full" in workflow
    assert 'result="acados-smoke-results/${case_name}/result.json"' in workflow
    assert "select_acados_case()" in workflow
    assert "sqp-irk-two-stage-cadence-reg-0p1-reduced/result.json" in workflow
    assert "sqp-irk-cadence-reg-1-best-retry-full/result.json" in workflow
    assert "sqp-irk-two-stage-cadence-reg-1-${mechanics}/result.json" in workflow
    assert ".results[0].success == true" in workflow
    assert "sqp-irk-two-stage-${mechanics}/result.json" in workflow
    assert "sqp-irk-reference-${mechanics}/result.json" in workflow
    assert workflow.index(
        "sqp-irk-two-stage-${mechanics}/result.json"
    ) < workflow.index("sqp-irk-reference-${mechanics}/result.json")
    assert "run_case sqp-irk-two-stage-adaptive" in workflow
    assert "--acados-transfer-bound-homotopy-fractions 0,1" in workflow
    assert "run_case sqp-irk-two-stage-cadence-guard" not in workflow
    assert "run_case sqp-irk-two-stage-cadence-reg-0p1 reduced" in workflow
    assert "--acados-wheel-qdot-regularization-weight 0.1" in workflow
    assert "run_case sqp-irk-two-stage-cadence-reg-1 full" in workflow
    assert "run_case sqp-irk-cadence-reg-1-best-retry full" in workflow
    assert "run_case sqp-byrd-omojokun-cadence-reg-1-irk full" in workflow
    assert "run_case sqp-byrd-terminal-homotopy-cadence-reg-1-irk full" in workflow
    assert "run_case sqp-byrd-rollout-accept-cadence-reg-1-irk full" in workflow
    assert "--acados-terminal-wheel-q-homotopy-slacks 0.01,0.005,0.002" in workflow
    assert "--acados-terminal-wheel-q-homotopy-each-window" in workflow
    assert (
        "$config.acados_terminal_wheel_q_homotopy_slacks | "
        "if . == null then null else tojson end"
    ) in workflow
    assert "--shared-transfer-rollout-max-bound-violation 12" in workflow
    assert "sqp-byrd-dual-preserve-cadence-reg-1-irk" not in workflow
    assert 'inputs.cycles == \'acados_homotopy\'' in workflow
    assert 'if [[ "$ACADOS_HOMOTOPY_ONLY" == "true" ]]; then' in workflow
    focused_campaign = workflow.split(
        'if [[ "$ACADOS_HOMOTOPY_ONLY" == "true" ]]; then', maxsplit=1
    )[1].split('elif [[ "$acados_long" == "true" ]]; then', maxsplit=1)[0]
    assert focused_campaign.count("run_case ") == 5
    assert "run_case sqp-irk-reference full 1" in focused_campaign
    assert "run_case sqp-byrd-omojokun-cadence-reg-1-irk full" in focused_campaign
    assert (
        "run_case sqp-byrd-terminal-homotopy-cadence-reg-1-irk full"
        in focused_campaign
    )
    assert (
        "run_case sqp-byrd-rollout-accept-cadence-reg-1-irk full"
        in focused_campaign
    )
    assert "--acados-store-iterates" in workflow
    assert "--acados-maxiter-retries 1" in workflow
    assert "--acados-maxiter-retry-iterations 20" in workflow
    assert "--acados-maxiter-retry-feasibility-tolerance 0.0025" in workflow
    assert "--shared-transfer-rollout-max-bound-violation 0.2" in workflow
    assert "run_case sqp-irk-two-stage-cadence-reg-1 reduced" in workflow
    assert "--acados-wheel-qdot-regularization-weight 1" in workflow
    assert "reference-full-feasible-seed.npz" in workflow
    assert "--common-initial-solution-output" in workflow
    assert '--common-initial-solution "$common_seed"' in workflow
    assert (
        "The full cadence-regularized case requires the preceding feasible "
        "reference seed."
    ) in workflow
    assert "if (( ACADOS_SMOKE_RHOS > 30 )); then" in workflow
    assert 'echo "$acados_long" > acados-smoke-results/long-campaign.txt' in workflow
    long_campaign = workflow.split(
        'if [[ "$acados_long" == "true" ]]; then', maxsplit=1
    )[1].split('elif [[ "$acados_extended" == "false" ]]; then', maxsplit=1)[0]
    assert long_campaign.count("run_case ") == 8
    assert "run_case sqp-irk-two-stage-adaptive reduced" in long_campaign
    assert "run_case sqp-irk-two-stage-cadence-reg-1 full" in long_campaign
    assert "run_case sqp-irk-cadence-reg-1-best-retry full" in long_campaign
    assert "run_case sqp-byrd-omojokun-cadence-reg-1-irk full" in long_campaign
    assert (
        "run_case sqp-byrd-terminal-homotopy-cadence-reg-1-irk full 20 "
        "SQP_WITH_FEASIBLE_QP IRK 5 5"
    ) in long_campaign
    assert "run_case sqp-irk-two-stage-cadence-reg-1 reduced" in long_campaign
    assert "sqp-irk-two-stage-cadence-reg-0p1" not in long_campaign
    assert "sqp-rti-irk" not in long_campaign
    assert "--shared-transfer-contact-projection" in workflow
    assert "--shared-transfer-contact-projection-mode position_velocity" in workflow
    assert "expected ${#expected_cases[@]} JSON files" in workflow
    assert "expected 12 JSON files" not in workflow
    assert ".configurations.acados.n_windows > 5" in workflow
    assert ".validated_cycles == $expected" not in workflow
    assert "case_slug: fatrop-collocation" in workflow
    assert "Run FATROP collocation full" in workflow
    assert "Run FATROP collocation reduced" in workflow
    assert "max-parallel: 3" in workflow
    assert "madnlp-pardiso" not in workflow
    assert "fatrop-rk4" not in workflow
    benchmark_runner = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "scripts"
        / "run_cycling_benchmark_case.sh"
    ).read_text(encoding="utf-8")
    assert "solver_options+=(--ipopt-c-compile)" in benchmark_runner
    assert "solver_options+=(--madnlp-c-compile)" in benchmark_runner
    assert (
        "initialization_options=(--no-optional-nlp-periodic-ipopt-hot-start)"
        in benchmark_runner
    )
    assert (
        "initialization_options=(--optional-nlp-periodic-ipopt-hot-start)"
        in benchmark_runner
    )
    assert '"${initialization_options[@]}"' in benchmark_runner
    assert '--fatrop-state-scaling "$fatrop_state_scaling"' in benchmark_runner
    assert 'collocation_degree="${11:-3}"' in benchmark_runner
    assert 'ipopt_profile="${12:-periodic_collocation}"' in benchmark_runner
    assert 'dual_warm_start="${13:-auto}"' in benchmark_runner
    assert 'target_refinement="${14:-auto}"' in benchmark_runner
    assert 'trajectory_options=()' in benchmark_runner
    assert '[[ "$ipopt_profile" =~ ^scientific[-_]radau[456]$ ]]' in benchmark_runner
    assert '--receding-horizon-solution-output' in benchmark_runner
    assert '"$case_dir/validated-rho-trajectory.npz"' in benchmark_runner
    assert '"${trajectory_options[@]}"' in benchmark_runner
    assert "BENCHMARK_MAX_ITER=5000 bash" not in workflow
    assert '--ipopt-collocation-degree "$collocation_degree"' in benchmark_runner
    assert '--ipopt-profile "$ipopt_profile"' in benchmark_runner
    assert ".benchmark_profile == $profile" in benchmark_runner
    assert ".activate_passive_force_relationship == true" in benchmark_runner
    assert ".control_decisions_per_cycle == 30" in benchmark_runner
    assert ".enforce_start_constraints == true" in benchmark_runner
    assert 'case "$graph_mode" in' in benchmark_runner
    assert "The endurance benchmark is SX-only" in benchmark_runner
    assert "libMAD WARNING: option linear_solver is of unknown type" in (
        benchmark_runner
    )
    assert ".configurations[$solver].use_sx == true" in benchmark_runner
    assert 'mktemp -d "$case_dir/codegen.XXXXXX"' in benchmark_runner
    assert 'pushd "$codegen_dir"' in benchmark_runner
    assert "--ipopt-enforce-start-constraints" in benchmark_runner
    assert "solver_options+=(--full-contact-position-tolerance 2e-5)" in (
        benchmark_runner
    )
    assert (
        'if [[ "$mechanics" != "reduced" && "$ode_solver" != "collocation" ]]'
        in benchmark_runner
    )
    assert "--shared-transfer-phase-one" in benchmark_runner
    assert "\n    --transfer-phase-one\n" not in benchmark_runner
    assert '--reduced-cycling-profile "$workspace/benchmark-seed/' in benchmark_runner
    assert (
        "Compare IPOPT interpreted and compiled evaluators over 5 RHO" not in workflow
    )
    assert "cycling-compile-ablation-" not in workflow
    assert "Compare MadNLP MUMPS interpreted and compiled evaluators" not in workflow
    assert "Compile and reuse reduced IPOPT/MadNLP" in workflow
    assert "Checkpoint IPOPT full" in workflow
    assert "Checkpoint MadNLP MUMPS reduced" in workflow
    assert "max-parallel: 2" in workflow
    assert re.search(r"\n  benchmark:.*?\n    needs: prepare-seed", workflow, re.DOTALL)
    assert re.search(
        r"- solver: ipopt\b.*?- solver: madnlp\b",
        workflow,
        flags=re.DOTALL,
    )
    assert " MX " not in workflow
    assert " pardiso_mkl " not in workflow
    assert "--ipopt-use-sx" in benchmark_runner
    assert "--ipopt-no-use-sx" not in benchmark_runner
    assert "--ipopt-use-sx" in workflow
    assert "--disable-periodic-ipopt-refinement" in workflow
    assert '"mumps": "MumpsSolver"' in workflow
    assert ".configurations.acados.use_sx == true" in workflow
    assert re.search(
        r"madnlp_common=\(\s+"
        r"--madnlp-max-iter 2000\s+"
        r"--madnlp-linear-solver mumps",
        workflow,
    )
    assert "unknown type mumps" not in workflow
    assert "merge-multiple: true" in workflow  # ACADOS remains a single artifact.

    mumps_installer = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "scripts"
        / "install_libmad_mumps_linux.sh"
    ).read_text(encoding="utf-8")
    assert '"$build_dir/no_hsl_example"' in mumps_installer
    assert '"$build_dir/basic_problem"' not in mumps_installer
    assert "option linear_solver is of unknown type" in mumps_installer


def test_benchmark_readme_tracks_every_active_bioptim_revision_and_patch():
    repository_root = Path(__file__).resolve().parents[2]
    workflow = (
        repository_root / ".github" / "workflows" / "cycling_solver_benchmark_linux.yml"
    ).read_text(encoding="utf-8")
    benchmark_readme = (
        repository_root / "docs" / "cycling_solver_benchmark" / "README.md"
    ).read_text(encoding="utf-8")

    pinned_revisions = set(
        re.findall(r"BIOPTIM_[A-Z_]+_COMMIT:\s*([0-9a-f]{40})", workflow)
    )
    pinned_revisions.update(re.findall(r"bioptim_commit:\s*([0-9a-f]{40})", workflow))
    applied_patches = set(
        re.findall(r'\.github/patches/(bioptim-[^"\s]+\.patch)', workflow)
    )

    assert pinned_revisions
    assert "mickaelbegon/BiorbdOptim" in benchmark_readme
    for revision in pinned_revisions:
        assert revision in benchmark_readme
    for patch_name in applied_patches:
        assert patch_name in benchmark_readme


def test_acados_transfer_restoration_time_is_attributed_to_the_next_rho():
    timing = comparison_example.acados_transfer_restoration_timing(
        {
            "transfer_bound_homotopy_summaries": [
                {
                    "window": 4,
                    "stages": [
                        {
                            "fraction": 0.0,
                            "attempt": 0,
                            "accepted": True,
                            "solver_time_s": 0.02,
                            "wall_time_s": 0.025,
                        },
                        {
                            "fraction": 1.0,
                            "attempt": 0,
                            "accepted": True,
                            "solver_time_s": 0.03,
                            "wall_time_s": 0.035,
                        },
                    ],
                }
            ],
            "terminal_wheel_bound_summaries": [
                {
                    "slack": 0.01,
                    "attempt": 0,
                    "accepted": True,
                    "solver_time_s": 0.005,
                    "wall_time_s": 0.01,
                }
            ],
            "inter_window_terminal_wheel_bound_summaries": [
                {
                    "window": 4,
                    "slack": 0.005,
                    "attempt": 0,
                    "accepted": True,
                    "solver_time_s": 0.04,
                    "wall_time_s": 0.05,
                }
            ],
        }
    )

    assert timing["available"] is True
    assert timing["total_wall_time_s"] == pytest.approx(0.12)
    assert timing["by_target_rho_wall_time_s"] == {
        1: pytest.approx(0.01),
        5: pytest.approx(0.11),
    }
    assert [row["kind"] for row in timing["stages"]] == [
        "transfer_bound_homotopy",
        "transfer_bound_homotopy",
        "terminal_wheel_bound_homotopy",
        "terminal_wheel_bound_homotopy",
    ]
    assert [stage["target_rho"] for stage in timing["stages"]] == [5, 5, 1, 5]


def test_single_shot_requires_solver_and_feasibility_success():
    class FakeSolution:
        status = 1
        iterations = 10
        cost = np.array([[1.0]])
        constraints = np.zeros(3)
        inf_pr = 1e-8
        solver_time_to_optimize = 0.1
        real_time_to_optimize = 0.2

        @staticmethod
        def decision_states(to_merge=None):
            return {"q": np.array([[0.0, 0.1], [0.0, 0.1], [0.0, -2.0 * np.pi]])}

        @staticmethod
        def decision_controls(to_merge=None):
            return {"u": np.array([[0.2]])}

    failed_status = periodic_example.build_single_shot_summary(
        FakeSolution(), feasibility_tolerance=1e-6
    )
    assert failed_status["physical_success"] is True
    assert failed_status["solver_success"] is False
    assert failed_status["success"] is False

    solution = FakeSolution()
    solution.status = 0
    solution.inf_pr = 1e-2
    failed_feasibility = periodic_example.build_single_shot_summary(
        solution, feasibility_tolerance=1e-6
    )
    assert failed_feasibility["window_feasibility"][0]["passes_tolerance"] is False
    assert failed_feasibility["success"] is False


def test_two_cycle_single_shot_checks_each_complete_turn():
    class FakeSolution:
        status = 0
        iterations = 10
        cost = np.array([[1.0]])
        constraints = np.zeros(3)
        inf_pr = 1e-8
        solver_time_to_optimize = 0.1
        real_time_to_optimize = 0.2

        @staticmethod
        def decision_states(to_merge=None):
            return {
                "q": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, -2.0 * np.pi, -4.0 * np.pi],
                    ]
                )
            }

        @staticmethod
        def decision_controls(to_merge=None):
            return {"u": np.array([[0.2, 0.2]])}

    summary = periodic_example.build_single_shot_summary(
        FakeSolution(),
        feasibility_tolerance=1e-6,
        cycle_count=2,
        cycle_progress_tolerance=1e-6,
    )

    assert summary["physical_success"] is True
    assert summary["requested_cycles"] == 2
    assert summary["exported_cycles"] == 2
    assert summary["covered_cycles"] == 2
    assert summary["validated_cycles"] == 2
    assert summary["physically_validated_cycles"] == 2
    np.testing.assert_allclose(
        summary["diagnostics"]["cycle_progress_errors"], [0.0, 0.0]
    )


def test_optional_nlp_config_clones_ipopt_transcription_exactly():
    reference = SimpleNamespace(
        solver="ipopt",
        model_formulation="standard",
        torque_application="external_forces",
        ode_solver="collocation",
        collocation_degree=3,
        collocation_method="radau",
        use_sx=False,
        state_scaling="full",
        objective="fatigue",
        objective_shape="quadratic",
    )

    madnlp = comparison_example._nlp_solver_config(
        "madnlp",
        reference,
        tolerance=1e-6,
        max_iterations=500,
        dual_warm_start_mode="bounds",
        madnlp_linear_solver="umfpack",
        periodic_ipopt_hot_start=True,
    )
    fatrop = comparison_example._nlp_solver_config(
        "fatrop",
        reference,
        tolerance=1e-7,
        max_iterations=600,
        dual_warm_start_mode="off",
        fatrop_structure_detection="auto",
        fatrop_bound_tightening_factor=2e-8,
        periodic_ipopt_hot_start=True,
    )
    alpaqa = comparison_example._nlp_solver_config(
        "alpaqa",
        reference,
        tolerance=1e-5,
        max_iterations=800,
        dual_warm_start_mode="constraints",
        alpaqa_alm_max_iterations=40,
        alpaqa_initial_tolerance=1e-3,
        alpaqa_penalty_update_factor=5.0,
        periodic_ipopt_hot_start=True,
    )

    for candidate in (fatrop, madnlp, alpaqa):
        assert candidate.model_formulation == reference.model_formulation
        assert candidate.torque_application == reference.torque_application
        assert candidate.ode_solver == reference.ode_solver
        assert candidate.collocation_degree == reference.collocation_degree
        assert candidate.collocation_method == reference.collocation_method
        assert candidate.use_sx == reference.use_sx
        assert candidate.state_scaling == reference.state_scaling
        assert candidate.objective == "fatigue"
        assert candidate.objective_shape == "quadratic"
        assert candidate.nlp_periodic_ipopt_hot_start is True

    assert madnlp.madnlp_linear_solver == "umfpack"
    assert fatrop.fatrop_structure_detection == "auto"
    assert fatrop.fatrop_bound_tightening_factor == 2e-8
    assert fatrop.max_fatrop_iterations == 600
    assert alpaqa.alpaqa_initial_tolerance == 1e-3
    assert alpaqa.alpaqa_penalty_update_factor == 5.0
    assert alpaqa.alpaqa_alm_max_iterations == 40


def test_regularized_mhe_cli_exposes_previous_window_targets_and_terminal_slack():
    args = periodic_example.build_argument_parser().parse_args(
        [
            "--control-regularization-target-source",
            "previous",
            "--terminal-qdot-regularization-weight",
            "0.1",
            "--terminal-qdot-regularization-target-source",
            "first_node",
        ]
    )

    assert args.control_regularization_target_source == "previous"
    assert args.terminal_qdot_regularization_weight == 0.1
    assert args.terminal_qdot_regularization_target_source == "first_node"
    assert args.acados_terminal_wheel_q_slack == 0.002
    assert args.wheel_qdot_bound_margin == 3.0
    assert args.acados_globalization == "FUNNEL_L1PEN_LINESEARCH"
    assert args.periodic_ipopt_refinement_each_window is False


def test_acados_internal_wheel_speed_guard_keeps_physical_audit_margin_separate():
    args = periodic_example.build_argument_parser().parse_args(
        [
            "--solver",
            "acados",
            "--wheel-qdot-bound-margin",
            "3.0",
            "--acados-wheel-qdot-fast-bound-margin",
            "2.55",
        ]
    )

    assert args.wheel_qdot_bound_margin == 3.0
    assert periodic_example._effective_wheel_qdot_bound_margins(args) == (
        2.55,
        3.0,
    )

    comparison_args = comparison_example.build_cli().parse_args(
        ["--acados-wheel-qdot-fast-bound-margin", "2.55"]
    )
    assert comparison_args.acados_wheel_qdot_fast_bound_margin == 2.55
    assert comparison_args.acados_wheel_qdot_slow_bound_margin is None


def test_reduced_wheel_speed_bounds_support_an_asymmetric_fast_guard(monkeypatch):
    monkeypatch.setattr(
        OcpFesMsk,
        "set_x_bounds_fes",
        staticmethod(lambda _model: (BoundsList(), InitialGuessList())),
    )
    x_init = InitialGuessList()
    x_init.add(
        "theta",
        np.array([[0.0, -2.0 * np.pi]]),
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_init.add(
        "omega",
        np.array([[-2.0 * np.pi, -2.0 * np.pi]]),
        interpolation=InterpolationType.EACH_FRAME,
    )

    bounds, _ = mhe_example.set_reduced_x_bounds(
        model=SimpleNamespace(),
        x_init=x_init,
        n_shooting=1,
        ode_solver=OdeSolver.RK4(),
        init_file_path=None,
        omega_fast_bound_margin=2.55,
        omega_slow_bound_margin=3.0,
    )

    np.testing.assert_allclose(bounds["omega"].min, -2.0 * np.pi - 2.55)
    np.testing.assert_allclose(bounds["omega"].max, -2.0 * np.pi + 3.0)


def test_full_wheel_speed_bounds_support_an_asymmetric_fast_guard(monkeypatch):
    monkeypatch.setattr(
        OcpFesMsk,
        "set_x_bounds_fes",
        staticmethod(lambda _model: (BoundsList(), InitialGuessList())),
    )

    class FakeModel:
        @staticmethod
        def bounds_from_ranges(key):
            return Bounds(
                key,
                min_bound=np.full((3, 3), -20.0),
                max_bound=np.full((3, 3), 20.0),
                interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
            )

    x_init = InitialGuessList()
    x_init.add(
        "q",
        np.array([[0.0, 0.0], [1.0, 1.0], [0.0, -2.0 * np.pi]]),
        interpolation=InterpolationType.EACH_FRAME,
    )
    x_init.add(
        "qdot",
        np.tile(np.array([[0.0], [0.0], [-2.0 * np.pi]]), (1, 2)),
        interpolation=InterpolationType.EACH_FRAME,
    )

    bounds, _ = mhe_example.set_x_bounds(
        model=FakeModel(),
        x_init=x_init,
        n_shooting=1,
        ode_solver=OdeSolver.RK4(),
        init_file_path=None,
        wheel_qdot_bound_margin=3.0,
        wheel_qdot_fast_bound_margin=2.55,
        wheel_qdot_slow_bound_margin=3.0,
    )

    np.testing.assert_allclose(bounds["qdot"].min[2], -2.0 * np.pi - 2.55)
    np.testing.assert_allclose(bounds["qdot"].max[2], -2.0 * np.pi + 3.0)


def test_optional_nlp_cli_exposes_cross_solver_hot_start_and_tuning():
    periodic_args = periodic_example.build_argument_parser().parse_args(
        [
            "--solver",
            "madnlp",
            "--nlp-periodic-ipopt-hot-start",
            "--madnlp-linear-solver",
            "umfpack",
        ]
    )
    comparison_args = comparison_example.build_cli().parse_args(
        [
            "--alpaqa-initial-tolerance",
            "1e-3",
            "--alpaqa-alm-max-iter",
            "40",
            "--alpaqa-penalty-update-factor",
            "5",
            "--alpaqa-maximum-penalty",
            "1e7",
            "--alpaqa-panoc-max-wall-time",
            "0.25",
            "--alpaqa-max-no-progress",
            "25",
        ]
    )

    assert periodic_args.nlp_periodic_ipopt_hot_start is True
    assert periodic_args.madnlp_linear_solver == "umfpack"
    assert comparison_args.optional_nlp_periodic_ipopt_hot_start is True
    assert comparison_args.alpaqa_alm_max_iter == 40
    assert comparison_args.alpaqa_initial_tolerance == 1e-3
    assert comparison_args.alpaqa_penalty_update_factor == 5
    assert comparison_args.alpaqa_maximum_penalty == 1e7
    assert comparison_args.alpaqa_panoc_max_wall_time == 0.25
    assert comparison_args.alpaqa_max_no_progress == 25

    fatrop_args = comparison_example.build_cli().parse_args(
        [
            "--solvers",
            "fatrop",
            "--fatrop-max-iter",
            "750",
            "--fatrop-structure-detection",
            "auto",
            "--fatrop-bound-tightening-factor",
            "2e-8",
            "--fatrop-state-scaling",
            "none",
            "--fatrop-dual-warm-start-mode",
            "off",
        ]
    )
    assert fatrop_args.solvers == ("fatrop",)
    assert fatrop_args.fatrop_max_iter == 750
    assert fatrop_args.fatrop_structure_detection == "auto"
    assert fatrop_args.fatrop_bound_tightening_factor == 2e-8
    assert fatrop_args.fatrop_state_scaling == "none"
    assert fatrop_args.fatrop_dual_warm_start_mode == "off"


def test_backend_independent_terminal_wheel_slack_cli():
    periodic_args = periodic_example.build_argument_parser().parse_args(
        ["--terminal-wheel-q-slack", "0.02"]
    )
    comparison_args = comparison_example.build_cli().parse_args(
        [
            "--terminal-wheel-q-slack",
            "0.02",
            "--first-node-wheel-q-slack",
            "0",
        ]
    )

    assert periodic_args.terminal_wheel_q_slack == 0.02
    assert comparison_args.acados_terminal_wheel_q_slack == 0.02
    assert comparison_args.first_node_wheel_q_slack == 0.0


def test_historical_pulse_width_active_set_is_periodic_and_keeps_a_guard_band():
    pd0 = 0.000131405
    reference = np.full(10, pd0)
    reference[0] = 0.0006

    active = mhe_example.periodic_pulse_width_activity_mask(
        reference,
        pd0=pd0,
        maximum=0.0006,
        cycles_per_window=2,
        relative_threshold=0.01,
        margin=1,
    )

    np.testing.assert_array_equal(
        active,
        [True, True, False, False, True, True, True, False, False, True],
    )


def test_pulse_width_activity_margin_only_adds_free_nodes():
    pd0 = 0.000131405
    reference = np.full(60, pd0)
    reference[[2, 17, 32, 47]] = 0.0006

    margin_three = mhe_example.periodic_pulse_width_activity_mask(
        reference,
        pd0=pd0,
        maximum=0.0006,
        cycles_per_window=2,
        relative_threshold=0.01,
        margin=3,
    )
    margin_four = mhe_example.periodic_pulse_width_activity_mask(
        reference,
        pd0=pd0,
        maximum=0.0006,
        cycles_per_window=2,
        relative_threshold=0.01,
        margin=4,
    )

    assert np.all(margin_three <= margin_four)
    assert np.count_nonzero(margin_four) > np.count_nonzero(margin_three)
    np.testing.assert_array_equal(margin_four[:30], margin_four[30:])


def test_generic_msk_pulse_width_initial_guess_uses_ding_pd0():
    model = _muscle_model()
    bio_model = SimpleNamespace(
        muscles_dynamics_model=[model],
        nb_tau=0,
    )

    _, initial_guesses = OcpFesMsk.set_u_bounds_msk(
        BoundsList(),
        InitialGuessList(),
        bio_model,
        with_residual_torque=False,
    )

    np.testing.assert_allclose(
        initial_guesses["last_pulse_width_Biceps"].init,
        [[model.pd0]],
    )


def test_reduced_theta_seed_is_recentered_on_absolute_cycle_targets():
    theta = np.array([[0.1, -3.0, -6.0, -9.0, -12.1]])
    omega = -np.ones_like(theta)

    (
        corrected,
        corrected_omega,
        audit,
    ) = mhe_example.recenter_reduced_theta_seed(
        theta,
        omega,
        nodes_per_cycle=2,
        cycles=2,
        node_time_grid_s=np.array([0.0, 0.3, 1.0, 1.2, 2.0]),
    )

    np.testing.assert_allclose(
        corrected[0, [0, 2, 4]],
        0.1 - np.arange(3) * 2.0 * np.pi,
    )
    assert audit["maximum_boundary_error_before_rad"] > 0.1
    assert audit["maximum_theta_change_rad"] > 0.1
    assert audit["maximum_omega_change_rad_s"] > 0.0
    assert not np.allclose(corrected_omega, omega)


def test_collocation_warm_start_uses_radau_abscissae():
    ode_solver = SimpleNamespace(
        is_direct_collocation=True,
        polynomial_degree=3,
        method="radau",
    )

    grid = mhe_example.state_initial_guess_time_grid(
        n_shooting=2,
        turn_number=1,
        ode_solver=ode_solver,
    )

    expected_local = np.array([0.0, 0.15505102572168222, 0.6449489742783179, 1.0])
    np.testing.assert_allclose(grid[:4], expected_local / 2.0, atol=1e-14)
    np.testing.assert_allclose(grid[4:8], (1.0 + expected_local) / 2.0, atol=1e-14)
    assert grid[3] == grid[4]
    assert grid[-1] == 1.0


def test_physical_crank_velocity_uses_center_to_hand_kinematics():
    class FakeBioModel:
        @staticmethod
        def marker_index(name):
            return {"hand": 0, "global_wheel_center": 1}[name]

        @staticmethod
        def marker(index):
            values = (
                np.array([1.0, 0.0, 0.0]) if index == 0 else np.array([0.0, 0.0, 0.0])
            )
            return lambda q, parameters: values

        @staticmethod
        def marker_velocity(index):
            values = (
                np.array([0.0, -5.0, 0.0]) if index == 0 else np.array([0.0, 0.0, 0.0])
            )
            return lambda q, qdot, parameters: values

    controller = SimpleNamespace(
        model=SimpleNamespace(bio_model=FakeBioModel()),
        states={
            "q": SimpleNamespace(cx=np.zeros(3)),
            "qdot": SimpleNamespace(cx=np.zeros(3)),
        },
        parameters=SimpleNamespace(cx=np.array([])),
    )

    omega = mhe_example.physical_crank_velocity_constraint(controller)

    assert omega == pytest.approx(-5.0)


def test_physical_crank_velocity_includes_all_collocation_points():
    class FakeBioModel:
        @staticmethod
        def marker_index(name):
            return {"hand": 0, "global_wheel_center": 1}[name]

        @staticmethod
        def marker(index):
            values = (
                np.array([1.0, 0.0, 0.0]) if index == 0 else np.array([0.0, 0.0, 0.0])
            )
            return lambda q, parameters: values

        @staticmethod
        def marker_velocity(index):
            if index == 0:
                return lambda q, qdot, parameters: np.array([0.0, qdot[0], 0.0])
            return lambda q, qdot, parameters: np.zeros(3)

    controller = SimpleNamespace(
        model=SimpleNamespace(bio_model=FakeBioModel()),
        get_nlp=SimpleNamespace(
            dynamics_type=SimpleNamespace(
                ode_solver=SimpleNamespace(is_direct_collocation=True)
            )
        ),
        states={
            "q": SimpleNamespace(
                cx_start=np.zeros(3),
                cx_intermediates_list=[np.zeros(3), np.zeros(3)],
            ),
            "qdot": SimpleNamespace(
                cx_start=np.array([-5.0, 0.0, 0.0]),
                cx_intermediates_list=[
                    np.array([-7.0, 0.0, 0.0]),
                    np.array([-9.0, 0.0, 0.0]),
                ],
            ),
        },
        parameters=SimpleNamespace(cx=np.array([])),
    )

    omega = mhe_example.physical_crank_velocity_all_collocation_points_constraint(
        controller
    )

    np.testing.assert_allclose(np.asarray(omega).reshape(-1), [-5.0, -7.0, -9.0])


def test_physical_crank_velocity_ignores_pseudo_stages_in_direct_shooting():
    class FakeBioModel:
        @staticmethod
        def marker_index(name):
            return {"hand": 0, "global_wheel_center": 1}[name]

        @staticmethod
        def marker(index):
            values = (
                np.array([1.0, 0.0, 0.0]) if index == 0 else np.array([0.0, 0.0, 0.0])
            )
            return lambda q, parameters: values

        @staticmethod
        def marker_velocity(index):
            if index == 0:
                return lambda q, qdot, parameters: np.array([0.0, qdot[0], 0.0])
            return lambda q, qdot, parameters: np.zeros(3)

    controller = SimpleNamespace(
        model=SimpleNamespace(bio_model=FakeBioModel()),
        get_nlp=SimpleNamespace(
            dynamics_type=SimpleNamespace(
                ode_solver=SimpleNamespace(is_direct_collocation=False)
            )
        ),
        states={
            "q": SimpleNamespace(
                cx_start=np.zeros(3),
                cx_intermediates_list=[np.full(3, np.nan)],
            ),
            "qdot": SimpleNamespace(
                cx_start=np.array([-5.0, 0.0, 0.0]),
                cx_intermediates_list=[np.full(3, np.nan)],
            ),
        },
        parameters=SimpleNamespace(cx=np.array([])),
    )

    omega = mhe_example.physical_crank_velocity_all_collocation_points_constraint(
        controller
    )

    np.testing.assert_allclose(np.asarray(omega).reshape(-1), [-5.0])


def test_mechanical_audit_rejects_collocation_only_cadence_violation():
    class FakeKinematics:
        @staticmethod
        def project_generalized_trajectory(q, qdot):
            return (
                np.asarray(q[:1], dtype=float),
                np.asarray(qdot[:1], dtype=float),
                {
                    "maximum_configuration_projection_error_rad": 0.0,
                    "maximum_tangent_velocity_residual_rad_s": 0.0,
                },
            )

    omega = np.array([[-6.0, -10.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0]])
    _, _, audit = periodic_example.audit_mechanical_trajectory(
        {
            "q": np.zeros((3, omega.shape[1])),
            "qdot": np.vstack((omega, omega, omega)),
        },
        SimpleNamespace(kinematics=FakeKinematics()),
        velocity_tolerance_rad_s=0.1,
        cadence_node_stride=4,
    )

    assert audit["maximum_physical_crank_velocity_bound_violation_rad_s"] == 0.0
    assert audit["maximum_shooting_node_crank_velocity_bound_violation_rad_s"] == 0.0
    assert audit["maximum_all_node_crank_velocity_bound_violation_rad_s"] > 0.7
    assert audit["passes_physical_crank_velocity_bounds"] is False
    assert audit["passes_tolerance"] is False


def test_mechanical_audit_rejects_hidden_acados_interval_cadence_violation():
    class FakeKinematics:
        @staticmethod
        def lift_generalized_trajectory(theta, omega):
            return np.asarray(theta, dtype=float), np.asarray(omega, dtype=float)

        @staticmethod
        def project_generalized_trajectory(q, qdot):
            return (
                np.asarray(q, dtype=float),
                np.asarray(qdot, dtype=float),
                {
                    "maximum_configuration_projection_error_rad": 0.0,
                    "maximum_tangent_velocity_residual_rad_s": 0.0,
                },
            )

    _, _, audit = periodic_example.audit_mechanical_trajectory(
        {
            "theta": np.array([[0.0, -0.322]]),
            "omega": np.array([[-6.0, -6.0]]),
        },
        SimpleNamespace(kinematics=FakeKinematics()),
        velocity_tolerance_rad_s=0.1,
        shooting_interval_duration_s=1.0 / 30.0,
    )

    assert audit["maximum_all_node_crank_velocity_bound_violation_rad_s"] == 0.0
    assert audit["interval_average_crank_velocity_available"] is True
    assert audit["maximum_interval_average_crank_velocity_bound_violation_rad_s"] > 0.37
    assert audit["passes_physical_crank_velocity_bounds"] is False
    assert audit["passes_tolerance"] is False


@pytest.mark.parametrize(
    ("absolute_offset", "absolute_reference", "expected_nlp_issues"),
    (
        (0.0, None, []),
        (0.01, 0.0, ["wheel_absolute_progress_out_of_bounds"]),
    ),
)
def test_acados_mechanical_audit_excludes_failed_tail_from_validated_prefix(
    absolute_offset,
    absolute_reference,
    expected_nlp_issues,
):
    class FakeKinematics:
        @staticmethod
        def lift_generalized_trajectory(theta, omega):
            return np.asarray(theta, dtype=float), np.asarray(omega, dtype=float)

        @staticmethod
        def project_generalized_trajectory(q, qdot):
            return (
                np.asarray(q, dtype=float),
                np.asarray(qdot, dtype=float),
                {
                    "maximum_configuration_projection_error_rad": 0.0,
                    "maximum_tangent_velocity_residual_rad_s": 0.0,
                },
            )

    summary = {
        "args": SimpleNamespace(
            solver="acados",
            ode_solver="rk4",
            stimulations_per_cycle=2,
            cycles_per_window=1,
            wheel_qdot_regularization_target=-2.0 * np.pi,
            wheel_qdot_bound_margin=3.0,
            acados_terminal_wheel_q_slack=0.002,
            primal_feasibility_threshold=1e-5,
        ),
        "mode": "rho",
        "state_traces": {
            "theta": np.array(
                [
                    [
                        absolute_offset,
                        absolute_offset - np.pi,
                        absolute_offset - 2.0 * np.pi,
                        absolute_offset - 20.0,
                    ]
                ]
            ),
            "omega": np.full((1, 4), -2.0 * np.pi),
        },
        "window_statuses": [0, 4],
        "window_feasibility": [
            {"passes_tolerance": True},
            {"passes_tolerance": False},
        ],
        "covered_cycles": 2,
        "diagnostics": {
            "is_physical": True,
            "issues": [],
            "absolute_cycle_reference": absolute_reference,
            "cycle_progress_tolerance": 0.00402,
            "absolute_cycle_tolerance": 0.00201,
        },
        "physical_success": True,
        "success": True,
    }

    periodic_example.attach_mechanical_equivalence_audit(
        summary,
        SimpleNamespace(kinematics=FakeKinematics()),
    )

    audit = summary["mechanical_equivalence_audit"]
    assert audit["audited_validated_cycles"] == 1
    assert audit["passes_tolerance"] is True
    assert summary["physical_crank_angle_trace"].shape == (3,)
    assert summary["nlp_crank_diagnostics"]["issues"] == expected_nlp_issues
    assert summary["nlp_crank_diagnostics"]["final_angle"] == (
        absolute_offset - 2.0 * np.pi
    )
    assert summary["physical_success"] is True


def test_historical_pulse_width_seed_is_clipped_with_bound_warning(tmp_path):
    model = _muscle_model()
    seed_path = tmp_path / "legacy-seed.pkl"
    seed_values = np.array(
        [0.0, model.pd0, 0.0002, 0.0008],
        dtype=float,
    )
    import pickle

    with seed_path.open("wb") as seed_file:
        pickle.dump({"last_pulse_width_Biceps": seed_values}, seed_file)

    with pytest.warns(
        RuntimeWarning,
        match=r"violates the physical Ding bounds.*below pd0.*above the maximum",
    ):
        _, initial_guesses, _ = mhe_example.set_u_bounds_and_init(
            SimpleNamespace(muscles_dynamics_model=[model]),
            n_shooting=4,
            init_file_path=seed_path,
        )

    np.testing.assert_allclose(
        initial_guesses["last_pulse_width_Biceps"].init,
        [[model.pd0, model.pd0, 0.0002, 0.0006]],
    )


def test_non_finite_historical_pulse_width_seed_is_rejected(tmp_path):
    model = _muscle_model()
    seed_path = tmp_path / "invalid-seed.pkl"
    import pickle

    with seed_path.open("wb") as seed_file:
        pickle.dump(
            {"last_pulse_width_Biceps": np.array([model.pd0, np.nan, model.pd0])},
            seed_file,
        )

    with pytest.raises(ValueError, match="non-finite"):
        mhe_example.set_u_bounds_and_init(
            SimpleNamespace(muscles_dynamics_model=[model]),
            n_shooting=3,
            init_file_path=seed_path,
        )


def test_pulse_width_active_set_cli_is_shared_by_benchmark():
    periodic_args = periodic_example.build_argument_parser().parse_args(
        [
            "--pulse-width-active-set",
            "historical",
            "--pulse-width-active-threshold",
            "0.02",
            "--pulse-width-active-margin",
            "4",
        ]
    )
    comparison_args = comparison_example.build_cli().parse_args(
        [
            "--pulse-width-active-set",
            "historical",
            "--pulse-width-active-threshold",
            "0.02",
            "--pulse-width-active-margin",
            "4",
        ]
    )

    for args in (periodic_args, comparison_args):
        assert args.pulse_width_active_set == "historical"
        assert args.pulse_width_active_threshold == 0.02
        assert args.pulse_width_active_margin == 4


def test_previous_control_and_terminal_velocity_targets_are_recentered():
    control_penalty = SimpleNamespace(
        extra_parameters={"key": "last_pulse_width_Biceps"},
        node_idx=[0, 1, 2],
        node=[Node.ALL],
        rows=np.array([7]),
        target=None,
    )
    terminal_penalty = SimpleNamespace(
        extra_parameters={"key": "qdot"},
        node_idx=[2],
        node=[Node.END],
        rows=np.array([10, 11, 12]),
        target=np.zeros((3, 1)),
    )
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                J=[control_penalty, terminal_penalty],
                u_init={
                    "last_pulse_width_Biceps": SimpleNamespace(
                        init=np.array([[0.0002, 0.0004]])
                    )
                },
            )
        ]
    )

    keys = periodic_example.apply_initial_guess_control_regularization_targets(nmpc)
    updated_terminal = periodic_example.apply_terminal_qdot_regularization_target(
        nmpc, [-1.0, -2.0, -6.5]
    )

    assert keys == ["last_pulse_width_Biceps"]
    np.testing.assert_allclose(control_penalty.target, [[0.0002, 0.0004, 0.0004]])
    assert updated_terminal is True
    np.testing.assert_allclose(terminal_penalty.target[:, 0], [-1.0, -2.0, -6.5])

    terminal_penalty.extra_parameters["key"] = "omega"
    updated_omega = periodic_example.apply_terminal_qdot_regularization_target(
        nmpc, [-6.25]
    )
    assert updated_omega is True
    np.testing.assert_allclose(terminal_penalty.target[:, 0], [-6.25])


def test_updated_targets_are_copied_to_acados_cached_yrefs():
    control_penalty = SimpleNamespace(
        extra_parameters={"key": "last_pulse_width_Biceps"},
        node_idx=[0, 1],
        node=[Node.ALL],
        rows=np.array([7]),
        target=np.array([[0.0002, 0.0004]]),
    )
    terminal_penalty = SimpleNamespace(
        extra_parameters={"key": "qdot"},
        node_idx=[2],
        node=[Node.END],
        rows=np.array([10, 11, 12]),
        target=np.array([[-1.0], [-2.0], [-6.5]]),
    )
    interface = SimpleNamespace(
        y_ref=[[np.zeros((1, 1)), np.zeros((1, 1))]],
        y_ref_end=[np.zeros((3, 1))],
    )
    nlp = SimpleNamespace(
        J=[control_penalty, terminal_penalty],
        controls={"last_pulse_width_Biceps": SimpleNamespace(index=[7])},
        states={"qdot": SimpleNamespace(index=[10, 11, 12])},
    )
    nmpc = SimpleNamespace(nlp=[nlp], ocp_solver=interface)

    periodic_example.refresh_acados_cached_objective_targets(nmpc)

    np.testing.assert_allclose(
        [item[0, 0] for item in interface.y_ref[0]], [0.0002, 0.0004]
    )
    np.testing.assert_allclose(interface.y_ref_end[0][:, 0], [-1.0, -2.0, -6.5])


def test_all_muscle_targets_are_copied_to_acados_cached_yrefs():
    keys = [
        "last_pulse_width_Biceps",
        "last_pulse_width_Delt_ant",
        "last_pulse_width_Delt_post",
        "last_pulse_width_Triceps",
    ]
    penalties = [
        SimpleNamespace(
            extra_parameters={"key": key},
            node_idx=[0, 1],
            node=[Node.ALL],
            rows=np.array([index]),
            target=np.array([[index + 0.1, index + 0.2]]),
        )
        for index, key in enumerate(keys)
    ]
    interface = SimpleNamespace(
        y_ref=[[np.zeros((1, 1)), np.zeros((1, 1))] for _ in penalties],
        y_ref_end=[],
    )
    nlp = SimpleNamespace(
        J=penalties,
        controls={
            key: SimpleNamespace(index=[index]) for index, key in enumerate(keys)
        },
        states={},
    )

    periodic_example.refresh_acados_cached_objective_targets(
        SimpleNamespace(nlp=[nlp], ocp_solver=interface)
    )

    for index, references in enumerate(interface.y_ref):
        np.testing.assert_allclose(
            [reference[0, 0] for reference in references],
            [index + 0.1, index + 0.2],
        )


def test_runtime_proximal_weight_updates_only_pulse_width_blocks():
    class LagrangeFunction:
        pass

    class MayerFunction:
        pass

    class FakeGeneratedSolver:
        def __init__(self):
            self.calls = []

        def cost_set(self, stage, field, value, api=None):
            self.calls.append((stage, field, np.array(value, copy=True)))

    def penalty(key, node, size, penalty_type):
        return SimpleNamespace(
            extra_parameters={"key": key},
            node=[node],
            function=[SimpleNamespace(numel_out=lambda: size)],
            type=SimpleNamespace(get_type=lambda: penalty_type),
        )

    penalties = [
        penalty(None, Node.ALL, 2, LagrangeFunction),
        penalty("last_pulse_width_Biceps", Node.ALL, 1, LagrangeFunction),
        penalty("last_pulse_width_Triceps", Node.ALL, 1, LagrangeFunction),
        penalty("qdot", Node.END, 3, MayerFunction),
    ]
    generated_solver = FakeGeneratedSolver()
    interface = SimpleNamespace(
        ocp_solver=generated_solver,
        acados_ocp=SimpleNamespace(solver_options=SimpleNamespace(N_horizon=3)),
        W=np.diag([7.0, 8.0, 9.0, 10.0]),
        W_0=np.diag([7.0, 8.0, 9.0, 10.0]),
        W_e=np.diag([7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]),
    )
    nmpc = SimpleNamespace(
        nlp=[SimpleNamespace(J=penalties)],
        ocp_solver=interface,
    )

    summary = periodic_example.set_acados_runtime_control_regularization_weight(
        nmpc, 100.0
    )

    assert summary["applied"] is True
    assert [call[0] for call in generated_solver.calls] == [0, 1, 2, 3]
    for _, field, matrix in generated_solver.calls:
        assert field == "W"
        np.testing.assert_allclose(np.diag(matrix)[:2], [7.0, 8.0])
        np.testing.assert_allclose(np.diag(matrix)[2:4], [100.0, 100.0])
    np.testing.assert_allclose(
        np.diag(generated_solver.calls[-1][2])[4:], [11.0, 12.0, 13.0]
    )


def test_terminal_wheel_slack_is_independent_from_first_node_slack():
    q_bounds = SimpleNamespace(
        min=np.zeros((3, 3)),
        max=np.zeros((3, 3)),
    )
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={
                    "q": SimpleNamespace(
                        init=np.array([[0, 0, 0], [0, 0, 0], [-5, -6, -7]])
                    )
                },
                x_bounds={"q": q_bounds},
            )
        ],
        _sync_acados_state_bounds=lambda: None,
    )

    periodic_example.set_terminal_wheel_q_bound_slack(nmpc, 0.2)

    assert q_bounds.min[2, 0] == 0.0
    assert q_bounds.max[2, 0] == 0.0
    np.testing.assert_allclose(q_bounds.min[2, 2], -7.2)
    np.testing.assert_allclose(q_bounds.max[2, 2], -6.8)


def test_absolute_terminal_reference_is_recentered_after_loading_same_formulation_seed():
    q_bounds = SimpleNamespace(
        min=np.full((3, 3), -100.0),
        max=np.full((3, 3), 100.0),
    )
    loaded_start = -6.3349272754445485
    nmpc = SimpleNamespace(
        anchor_wheel_q_to_absolute_reference=True,
        position_state_key="q",
        wheel_state_index=2,
        absolute_wheel_q_reference=-2.0 * np.pi,
        absolute_wheel_q_cycle_shift=-2.0 * np.pi,
        absolute_wheel_q_cycle_index=0,
        terminal_state_slack={"q": [0.0, 0.0, 0.002]},
        nlp=[
            SimpleNamespace(
                x_init={
                    "q": SimpleNamespace(
                        init=np.array([[0.0, 0.0], [0.0, 0.0], [loaded_start, -12.6]])
                    )
                },
                x_bounds={"q": q_bounds},
            )
        ],
        _sync_acados_state_bounds=lambda: None,
    )

    applied = periodic_example.recenter_absolute_wheel_q_reference_from_initial_guess(
        nmpc
    )

    assert applied is True
    np.testing.assert_allclose(nmpc.absolute_wheel_q_reference, loaded_start)
    target = loaded_start - 2.0 * np.pi
    np.testing.assert_allclose(nmpc._cocofest_terminal_wheel_q_center, target)
    np.testing.assert_allclose(q_bounds.min[2, 2], target - 0.002)
    np.testing.assert_allclose(q_bounds.max[2, 2], target + 0.002)


def test_absolute_terminal_reference_spans_the_complete_single_shot_horizon():
    theta_bounds = SimpleNamespace(
        min=np.full((1, 3), -100.0),
        max=np.full((1, 3), 100.0),
    )
    loaded_start = -2.5
    nmpc = SimpleNamespace(
        anchor_wheel_q_to_absolute_reference=True,
        position_state_key="theta",
        wheel_state_index=0,
        absolute_wheel_q_reference=0.0,
        absolute_wheel_q_cycle_shift=-2.0 * np.pi,
        absolute_wheel_q_cycle_index=0,
        _cocofest_cycles_per_window=2,
        terminal_state_slack={"theta": [0.002]},
        nlp=[
            SimpleNamespace(
                x_init={
                    "theta": SimpleNamespace(
                        init=np.array([[loaded_start, loaded_start - 4.0 * np.pi]])
                    )
                },
                x_bounds={"theta": theta_bounds},
            )
        ],
        _sync_acados_state_bounds=lambda: None,
    )

    periodic_example.recenter_absolute_wheel_q_reference_from_initial_guess(nmpc)

    target = loaded_start - 4.0 * np.pi
    np.testing.assert_allclose(nmpc._cocofest_terminal_wheel_q_center, target)
    np.testing.assert_allclose(theta_bounds.min[0, 2], target - 0.002)
    np.testing.assert_allclose(theta_bounds.max[0, 2], target + 0.002)


def test_final_seed_is_recentered_clipped_and_reprojected_after_last_seed_source(
    monkeypatch,
):
    q_bounds = SimpleNamespace(
        min=np.full((3, 3), -100.0),
        max=np.full((3, 3), 100.0),
    )
    q_init = SimpleNamespace(
        init=np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [-2.5, -2.5 - 2.0 * np.pi],
            ]
        )
    )
    projection_calls = []
    nmpc = SimpleNamespace(
        anchor_wheel_q_to_absolute_reference=True,
        position_state_key="q",
        wheel_state_index=2,
        absolute_wheel_q_reference=0.0,
        absolute_wheel_q_cycle_shift=-2.0 * np.pi,
        absolute_wheel_q_cycle_index=0,
        _cocofest_cycles_per_window=2,
        terminal_state_slack={"q": [0.0, 0.0, 0.002]},
        nlp=[SimpleNamespace(x_init={"q": q_init}, x_bounds={"q": q_bounds})],
        _sync_acados_state_bounds=lambda: None,
    )

    def correct_init_guess(*, corrected_input):
        assert corrected_input == "states"
        q_init.init[:, -1] = np.minimum(
            np.maximum(q_init.init[:, -1], q_bounds.min[:, 2]),
            q_bounds.max[:, 2],
        )

    nmpc._correct_init_guess_to_fit_bounds = correct_init_guess
    monkeypatch.setattr(
        periodic_example,
        "project_full_first_node_initial_guess_to_contact",
        lambda _nmpc, **kwargs: projection_calls.append(kwargs)
        or {
            "applied": True,
            "node": -1,
            "q_max_change": 0.0,
            "preserved_wheel_q": True,
        },
    )

    summary = periodic_example.finalize_absolute_wheel_q_initial_guess(
        nmpc,
        project_terminal_contact=True,
    )

    target = -2.5 - 4.0 * np.pi
    np.testing.assert_allclose(q_init.init[2, -1], target + 0.002)
    assert summary["terminal_wheel_clipped"] is True
    assert summary["maximum_state_bound_violation"] == 0.0
    assert projection_calls == [{"node": -1, "project_velocity": False}]


class _BoundComplementaritySolver:
    def get(self, stage, field):
        values = {
            "x": np.array([3.0]),
            "u": np.array([2.0]),
            "lam": np.array([0.1, 0.2, 0.3, 0.4]),
        }
        return values[field]

    def constraints_get(self, stage, field):
        values = {
            "lbu": np.array([0.0]),
            "ubu": np.array([4.0]),
            "lbx": np.array([1.0]),
            "ubx": np.array([5.0]),
        }
        return values[field]


def test_acados_bound_complementarity_identifies_largest_product():
    rows = periodic_example._acados_bound_complementarity_rows(
        _BoundComplementaritySolver(),
        n_stages=1,
        state_labels=["force"],
        control_labels=["pulse_width"],
    )

    assert rows[0] == {
        "stage": 0,
        "variable": "force",
        "side": "upper",
        "value": 3.0,
        "bound": 5.0,
        "distance": 2.0,
        "multiplier": 0.4,
        "product": 0.8,
    }


def test_control_bounds_summary_preserves_physical_units():
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                controls={"last_pulse_width_Biceps": SimpleNamespace()},
                u_bounds={
                    "last_pulse_width_Biceps": SimpleNamespace(
                        min=np.array([[0.000131405]]),
                        max=np.array([[0.0006]]),
                    )
                },
            )
        ]
    )

    assert periodic_example._control_bounds_summary(nmpc) == {
        "last_pulse_width_Biceps": {"lower": 0.000131405, "upper": 0.0006}
    }

    nmpc._cocofest_original_control_bounds = {
        "last_pulse_width_Biceps": (
            np.array([[0.0001]]),
            np.array([[0.0007]]),
        )
    }
    assert periodic_example._control_bounds_summary(nmpc) == {
        "last_pulse_width_Biceps": {"lower": 0.0001, "upper": 0.0007}
    }


def test_exact_initial_nlp_audit_is_enabled_and_attached_without_mutation():
    audits = [
        {
            "solver": "madnlp",
            "constraints": {"maximum_bound_violation": 0.25},
            "variables": {"maximum_bound_violation": 0.0},
            "evaluation_time_s": 0.01,
        }
    ]

    class FakeInterface:
        initial_nlp_audits = audits

        def enable_initial_nlp_audit(self):
            self.enabled = True

    nmpc = SimpleNamespace(ocp_solver=None)

    def set_ocp_solver(solver):
        nmpc.ocp_solver = solver

    nmpc.set_ocp_solver = set_ocp_solver
    interface = FakeInterface()

    periodic_example.enable_exact_initial_nlp_audit(nmpc, interface)
    summary = {}
    periodic_example.attach_exact_initial_nlp_audits(summary, nmpc)

    assert interface.enabled is True
    assert summary["exact_initial_nlp_audits"] == audits
    assert summary["exact_initial_nlp_audits"] is not audits


def test_continuation_source_inherits_requested_acados_tolerances(
    monkeypatch, tmp_path
):
    args = SimpleNamespace(
        cycles_per_window=2,
        n_windows=1,
        single_shot=False,
        acados_horizon_continuation=True,
        max_acados_iterations=100,
        acados_continuation_source_max_iterations=50,
        acados_tolerance=1e-4,
        acados_stationarity_tolerance=0.1,
        acados_diagnostics=True,
        codegen_tag="test",
    )
    observed = {}

    monkeypatch.setattr(
        periodic_example,
        "_continuation_cache_path",
        lambda _: tmp_path / "missing.npz",
    )

    def fake_solve_case(source_args, echo):
        observed["feasibility"] = source_args.acados_tolerance
        observed["stationarity"] = source_args.acados_stationarity_tolerance
        return {"status": 1, "solution": None}

    monkeypatch.setattr(periodic_example, "solve_case", fake_solve_case)

    with np.testing.assert_raises(RuntimeError):
        periodic_example.get_one_cycle_acados_continuation_source(args, echo=False)

    assert observed == {"feasibility": 1e-4, "stationarity": 0.1}


def test_proximal_phase_one_update_balances_reference_and_dynamics():
    observed = periodic_example._proximal_phase_one_update(
        reference=np.array([0.0, 10.0]),
        predicted=np.array([2.0, 20.0]),
        lower=np.array([-1.0, -1.0]),
        upper=np.array([1.0, 100.0]),
        proximity_weight=1.0,
        defect_weight=3.0,
    )

    np.testing.assert_allclose(observed, [1.0, 17.5])


def test_proximal_phase_one_rejects_collocation_layout():
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={"q": SimpleNamespace(init=np.zeros((3, 5)))},
                u_init={"u": SimpleNamespace(init=np.zeros((1, 1)))},
            )
        ]
    )

    with np.testing.assert_raises_regex(ValueError, "one state node"):
        periodic_example.project_full_dynamics_initial_guess(nmpc)


def test_proximal_phase_one_restores_initial_guess_when_defect_increases(
    monkeypatch,
):
    class Variables(dict):
        shape = 1

    defects = iter(
        [
            {"scaled_by_block": {"q": 1.0}, "absolute_by_block": {"q": 1.0}},
            {"scaled_by_block": {"q": 2.0}, "absolute_by_block": {"q": 2.0}},
        ]
    )
    sync_calls = []
    nmpc = SimpleNamespace(
        cycle_duration=1.0,
        cycle_len=1,
        nlp=[
            SimpleNamespace(
                x_init={"q": SimpleNamespace(init=np.array([[0.0, 1.0]]))},
                u_init={"u": SimpleNamespace(init=np.array([[0.0]]))},
                x_bounds={
                    "q": SimpleNamespace(
                        min=np.array([[-100.0, -100.0, -100.0]]),
                        max=np.array([[100.0, 100.0, 100.0]]),
                    )
                },
                states=Variables(q=SimpleNamespace(index=[0])),
                controls=Variables(u=SimpleNamespace(index=[0])),
            )
        ],
        _correct_init_guess_to_fit_bounds=lambda corrected_input: None,
        _sync_acados_state_bounds=lambda: sync_calls.append(True),
    )
    monkeypatch.setattr(
        periodic_example,
        "_full_dynamics_rollout_defect_details",
        lambda *_args, **_kwargs: next(defects),
    )
    monkeypatch.setattr(
        periodic_example,
        "_numerical_timeseries_at_node",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        periodic_example,
        "_rk4_full_dynamics_step",
        lambda *_args, **_kwargs: np.array([10.0]),
    )

    summary = periodic_example.project_full_dynamics_initial_guess(
        nmpc,
        proximity_weight=0.0,
        defect_weight=1.0,
        n_substeps=1,
        max_backtracking_steps=0,
    )

    assert summary["accepted"] is False
    assert summary["restored"] is True
    assert summary["scaled_defect_before"] == 1.0
    assert summary["candidate_scaled_defect_after"] == 2.0
    assert summary["scaled_defect_after"] == 1.0
    assert summary["max_state_change"] == 0.0
    np.testing.assert_allclose(nmpc.nlp[0].x_init["q"].init, [[0.0, 1.0]])
    assert len(sync_calls) == 2


def test_proximal_phase_one_backtracks_to_respect_state_change_limit(
    monkeypatch,
):
    class Variables(dict):
        shape = 1

    defects = iter(
        [
            {"scaled_by_block": {"q": 1.0}, "absolute_by_block": {"q": 1.0}},
            {"scaled_by_block": {"q": 0.7}, "absolute_by_block": {"q": 0.7}},
            {"scaled_by_block": {"q": 0.8}, "absolute_by_block": {"q": 0.8}},
            {"scaled_by_block": {"q": 0.9}, "absolute_by_block": {"q": 0.9}},
        ]
    )
    nmpc = SimpleNamespace(
        cycle_duration=1.0,
        cycle_len=1,
        nlp=[
            SimpleNamespace(
                x_init={"q": SimpleNamespace(init=np.array([[0.0, 1.0]]))},
                u_init={"u": SimpleNamespace(init=np.array([[0.0]]))},
                x_bounds={
                    "q": SimpleNamespace(
                        min=np.array([[-100.0, -100.0, -100.0]]),
                        max=np.array([[100.0, 100.0, 100.0]]),
                    )
                },
                states=Variables(q=SimpleNamespace(index=[0])),
                controls=Variables(u=SimpleNamespace(index=[0])),
            )
        ],
        _correct_init_guess_to_fit_bounds=lambda corrected_input: None,
        _sync_acados_state_bounds=lambda: None,
    )
    monkeypatch.setattr(
        periodic_example,
        "_full_dynamics_rollout_defect_details",
        lambda *_args, **_kwargs: next(defects),
    )
    monkeypatch.setattr(
        periodic_example,
        "_numerical_timeseries_at_node",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        periodic_example,
        "_rk4_full_dynamics_step",
        lambda *_args, **_kwargs: np.array([10.0]),
    )

    summary = periodic_example.project_full_dynamics_initial_guess(
        nmpc,
        proximity_weight=0.0,
        defect_weight=1.0,
        n_substeps=1,
        max_backtracking_steps=2,
        max_state_change_by_block={"q": 5.0},
    )

    assert summary["accepted"] is True
    assert summary["accepted_step"] == 0.5
    assert summary["candidate_scaled_defect_after"] == 0.7
    assert summary["scaled_defect_after"] == 0.8
    assert summary["max_state_change"] == 4.5
    assert summary["candidate_max_state_change"] == 9.0
    assert summary["state_change_by_block"]["q"] == 4.5
    np.testing.assert_allclose(nmpc.nlp[0].x_init["q"].init, [[0.0, 5.5]])


def test_proximal_phase_one_preserves_retained_nodes_and_immutable_blocks(
    monkeypatch,
):
    class StateVariables(dict):
        shape = 2

    class ControlVariables(dict):
        shape = 1

    defects = iter(
        [
            {
                "scaled_by_block": {"q": 1.0, "fes": 1.0},
                "absolute_by_block": {"q": 1.0, "fes": 1.0},
            },
            {
                "scaled_by_block": {"q": 0.5, "fes": 1.0},
                "absolute_by_block": {"q": 0.5, "fes": 1.0},
            },
        ]
    )
    q = np.array([[0.0, 1.0, 2.0]])
    force = np.array([[10.0, 11.0, 12.0]])
    nmpc = SimpleNamespace(
        cycle_duration=2.0,
        cycle_len=2,
        nlp=[
            SimpleNamespace(
                x_init={
                    "q": SimpleNamespace(init=q.copy()),
                    "F_Test": SimpleNamespace(init=force.copy()),
                },
                u_init={"u": SimpleNamespace(init=np.zeros((1, 2)))},
                x_bounds={
                    "q": SimpleNamespace(
                        min=np.full((1, 3), -100.0),
                        max=np.full((1, 3), 100.0),
                    ),
                    "F_Test": SimpleNamespace(
                        min=np.full((1, 3), -100.0),
                        max=np.full((1, 3), 100.0),
                    ),
                },
                states=StateVariables(
                    q=SimpleNamespace(index=[0]),
                    F_Test=SimpleNamespace(index=[1]),
                ),
                controls=ControlVariables(u=SimpleNamespace(index=[0])),
            )
        ],
        _correct_init_guess_to_fit_bounds=lambda corrected_input: None,
        _sync_acados_state_bounds=lambda: None,
    )
    monkeypatch.setattr(
        periodic_example,
        "_full_dynamics_rollout_defect_details",
        lambda *_args, **_kwargs: next(defects),
    )
    monkeypatch.setattr(
        periodic_example,
        "_numerical_timeseries_at_node",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        periodic_example,
        "_rk4_full_dynamics_step",
        lambda *_args, **_kwargs: np.array([10.0, 99.0]),
    )

    summary = periodic_example.project_full_dynamics_initial_guess(
        nmpc,
        proximity_weight=1.0,
        defect_weight=1.0,
        n_substeps=1,
        start_node=1,
        mutable_blocks=("q",),
        monotone_blocks=("q",),
    )

    assert summary["accepted"] is True
    assert summary["start_node"] == 1
    assert summary["mutable_blocks"] == ("q",)
    np.testing.assert_allclose(nmpc.nlp[0].x_init["q"].init, [[0.0, 1.0, 6.0]])
    np.testing.assert_allclose(nmpc.nlp[0].x_init["F_Test"].init, force)


def test_pulse_width_summary_preserves_ipopt_control_variation():
    pulse_widths = np.array([[0.00015, 0.0003, 0.0006]])
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                u_init={
                    "last_pulse_width_Biceps": SimpleNamespace(init=pulse_widths),
                }
            )
        ]
    )

    summary = pulse_width_initial_guess_summary(nmpc)

    assert summary == [
        {
            "key": "last_pulse_width_Biceps",
            "minimum": 0.00015,
            "mean": 0.00035,
            "maximum": 0.0006,
            "span": 0.00045,
        }
    ]


def test_one_cycle_solution_is_tiled_with_wheel_and_fatigue_drift():
    source = SimpleNamespace(
        decision_states=lambda to_merge=None: {
            "q": np.array(
                [
                    [1.0, 1.1, 1.1, 1.0],
                    [2.0, 2.1, 2.1, 2.0],
                    [0.0, -2.0, -4.0, -2 * np.pi],
                ]
            ),
            "qdot": np.array(
                [
                    [0.0, 0.1, -0.1, 0.0],
                    [0.0, 0.2, -0.2, 0.0],
                    [-2 * np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi],
                ]
            ),
            "F_Biceps": np.array([[0.0, 2.0, 1.0, 0.0]]),
            "A_Biceps": np.array([[10.0, 9.0, 8.0, 7.0]]),
        },
        decision_controls=lambda to_merge=None: {
            "last_pulse_width_Biceps": np.array([[0.0002, 0.0004, 0.0003]])
        },
    )

    def guess(shape):
        return SimpleNamespace(init=np.zeros(shape))

    def bounds(rows):
        return SimpleNamespace(
            min=np.full((rows, 3), -100.0),
            max=np.full((rows, 3), 100.0),
        )

    nlp = SimpleNamespace(
        x_init={
            "q": guess((3, 7)),
            "qdot": guess((3, 7)),
            "F_Biceps": guess((1, 7)),
            "A_Biceps": guess((1, 7)),
        },
        u_init={"last_pulse_width_Biceps": guess((1, 6))},
        x_bounds={"q": bounds(3), "qdot": bounds(3)},
    )
    nmpc = SimpleNamespace(
        nlp=[nlp],
        _correct_init_guess_to_fit_bounds=lambda corrected_input: None,
        _sync_acados_state_bounds=lambda: None,
    )

    summary = tile_one_cycle_solution_to_periodic_nmpc(nmpc, source)

    np.testing.assert_allclose(
        nlp.u_init["last_pulse_width_Biceps"].init,
        [[0.0002, 0.0004, 0.0003, 0.0002, 0.0004, 0.0003]],
    )
    np.testing.assert_allclose(
        nlp.x_init["q"].init[2],
        [
            0.0,
            -2.0,
            -4.0,
            -2 * np.pi,
            -2.0 - 2 * np.pi,
            -4.0 - 2 * np.pi,
            -4 * np.pi,
        ],
    )
    np.testing.assert_allclose(
        nlp.x_init["F_Biceps"].init,
        [[0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0]],
    )
    np.testing.assert_allclose(
        nlp.x_init["A_Biceps"].init,
        [[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0]],
    )
    assert summary["repeat_count"] == 2
    assert summary["max_transfer_seam_error"] == 0.0


def test_tiled_fes_states_are_rolled_out_causally():
    class Variables(dict):
        def __init__(self, values, shape):
            super().__init__(values)
            self.shape = shape

    states = Variables(
        {
            "q": SimpleNamespace(index=[0]),
            "F_Biceps": SimpleNamespace(index=[1]),
        },
        shape=2,
    )
    controls = Variables(
        {"last_pulse_width_Biceps": SimpleNamespace(index=[0])}, shape=1
    )
    nlp = SimpleNamespace(
        model=SimpleNamespace(),
        states=states,
        controls=controls,
        x_init={
            "q": SimpleNamespace(init=np.zeros((1, 5))),
            "F_Biceps": SimpleNamespace(init=np.array([[0.0, 0.5, 1.0, 9.0, 9.0]])),
        },
        u_init={"last_pulse_width_Biceps": SimpleNamespace(init=np.ones((1, 4)))},
        numerical_data_timeseries=None,
        dynamics_func=lambda time, state, control, parameters, algebraic, data: np.array(
            [0.0, control[0]]
        ),
    )
    nmpc = SimpleNamespace(nlp=[nlp], cycle_duration=1.0, cycle_len=2)

    summary = periodic_example._rollout_tiled_fes_states(
        nmpc, start_node=2, n_substeps=2
    )

    assert summary["applied"] is True
    assert summary["start_node"] == 2
    np.testing.assert_allclose(nlp.x_init["F_Biceps"].init, [[0.0, 0.5, 1.0, 1.5, 2.0]])


def test_unsafe_acados_option_is_initialized_on_each_solver_instance():
    first_solver = Solver.ACADOS()
    second_solver = Solver.ACADOS()

    set_acados_unsafe_option(first_solver, 0.5, "test_repeated_option")
    set_acados_unsafe_option(second_solver, 0.25, "test_repeated_option")

    assert first_solver._test_repeated_option == 0.5
    assert second_solver._test_repeated_option == 0.25


def test_acados_diagnostics_snapshot_is_independent_from_shared_solver(
    monkeypatch,
):
    live_diagnostics = {"status": 0, "residuals": np.array([1.0, 2.0])}
    monkeypatch.setattr(
        periodic_example,
        "collect_acados_diagnostics",
        lambda solution: live_diagnostics,
    )

    snapshot = periodic_example.snapshot_acados_diagnostics(SimpleNamespace())
    live_diagnostics["status"] = 2
    live_diagnostics["residuals"][0] = 99.0

    assert snapshot["status"] == 0
    np.testing.assert_allclose(snapshot["residuals"], [1.0, 2.0])


def test_acados_diagnostics_must_meet_strict_cache_tolerances():
    strict = {"residuals": np.array([0.1, 0.009, 0.0, 0.008])}
    relaxed_only = {"residuals": np.array([3.0, 0.009, 0.0, 0.008])}

    assert periodic_example.acados_diagnostics_meet_tolerances(
        strict, convergence_tolerance=1e-2, stationarity_tolerance=0.15
    )
    assert not periodic_example.acados_diagnostics_meet_tolerances(
        relaxed_only, convergence_tolerance=1e-2, stationarity_tolerance=0.15
    )


def test_codegen_signature_ignores_run_only_options():
    parser = periodic_example.build_argument_parser()
    reference = parser.parse_args([])
    longer_diagnostic_run = parser.parse_args(
        [
            "--n-windows",
            "20",
            "--acados-diagnostics",
            "--codegen-tag",
            "diagnostic",
        ]
    )

    assert periodic_example._codegen_signature(
        reference
    ) == periodic_example._codegen_signature(longer_diagnostic_run)
    assert periodic_example._horizon_seed_cache_signature(
        reference
    ) == periodic_example._horizon_seed_cache_signature(longer_diagnostic_run)

    longer_diagnostic_run.stimulations_per_cycle += 1
    assert periodic_example._codegen_signature(
        reference
    ) != periodic_example._codegen_signature(longer_diagnostic_run)
    assert periodic_example._horizon_seed_cache_signature(
        reference
    ) != periodic_example._horizon_seed_cache_signature(longer_diagnostic_run)


def test_codegen_names_normalize_user_tag_for_casadi():
    parser = periodic_example.build_argument_parser()
    unsafe = parser.parse_args(["--codegen-tag", "ci-reduced / smoke__test"])
    other = parser.parse_args(["--codegen-tag", "ci_reduced_smoke_test"])

    unsafe_model, unsafe_directory = periodic_example.build_codegen_names(unsafe)
    other_model, _ = periodic_example.build_codegen_names(other)

    assert re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*", unsafe_model)
    assert Path(unsafe_directory).name.startswith(
        "c_generated_code_ci_reduced_smoke_test_"
    )
    assert unsafe_model != other_model


def test_acados_qp_warm_start_cli_options_are_explicit():
    parser = periodic_example.build_argument_parser()
    default_args = parser.parse_args([])
    warm_args = parser.parse_args(
        [
            "--acados-qp-warm-start-level",
            "1",
            "--acados-warm-start-first-qp",
            "--acados-warm-start-first-qp-from-nlp",
        ]
    )

    assert default_args.acados_qp_warm_start_level == 0
    assert default_args.acados_warm_start_first_qp is False
    assert default_args.acados_warm_start_first_qp_from_nlp is False
    assert warm_args.acados_qp_warm_start_level == 1
    assert warm_args.acados_warm_start_first_qp is True
    assert warm_args.acados_warm_start_first_qp_from_nlp is True
    assert periodic_example._codegen_signature(
        default_args
    ) != periodic_example._codegen_signature(warm_args)


def test_signed_wheel_transfer_preserves_seam_and_terminal_turn():
    source = np.array(
        [
            0.0,
            -1.0,
            -2.0,
            -2 * np.pi,
            -2 * np.pi - 1.0,
            -2 * np.pi - 2.0,
            -4 * np.pi,
        ]
    )
    initial_guess = np.zeros((1, source.shape[0]))
    qdot_source = np.full_like(source, -6.5)
    qdot_initial_guess = np.zeros((1, source.shape[0]))
    nmpc = SimpleNamespace(
        nodes_per_cycle=3,
        cycle_duration=1.0,
        use_signed_wheel_shift=True,
        transfer_initial_guess_mode="anchored",
        nlp=[
            SimpleNamespace(
                x_init={
                    "q": SimpleNamespace(init=initial_guess),
                    "qdot": SimpleNamespace(init=qdot_initial_guess),
                }
            )
        ],
        _wheel_cycle_shift=lambda states: -2 * np.pi,
    )
    nmpc.set_init_cyclical = lambda data, key, index: MyCyclicNMPC.set_init_cyclical(
        nmpc, data, key, index
    )
    states = {"q": source[None, :], "qdot": qdot_source[None, :]}

    MyCyclicNMPC.set_init_cyclical_wheel(nmpc, states, "q", 0)
    MyCyclicNMPC.set_init_cyclical_wheel_velocity(nmpc, states, "qdot", 0)

    transferred = initial_guess[0]
    np.testing.assert_allclose(transferred[:4], source[3:])
    np.testing.assert_allclose(transferred[4:], source[4:] - 2 * np.pi)
    transferred_seam_increment = transferred[4] - transferred[3]
    np.testing.assert_allclose(transferred_seam_increment, source[4] - source[3])
    np.testing.assert_allclose(transferred[-1], source[-1] - 2 * np.pi)
    np.testing.assert_allclose(qdot_initial_guess[0], qdot_source)


def test_phase_shifted_warmup_shifts_reduced_theta_and_preserves_omega():
    theta = np.array([[0.0, -1.0, -2.0]])
    omega = np.array([[-6.0, -6.1, -6.2]])
    pulse_width = np.array([[0.0002, 0.0003]])
    theta_target = np.zeros_like(theta)
    omega_target = np.zeros_like(omega)
    pulse_width_target = np.zeros_like(pulse_width)
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={
                    "theta": SimpleNamespace(init=theta_target),
                    "omega": SimpleNamespace(init=omega_target),
                },
                u_init={
                    "last_pulse_width_Biceps": SimpleNamespace(init=pulse_width_target)
                },
            )
        ],
        _wheel_cycle_shift=lambda _states: -2.0 * np.pi,
        _correct_init_guess_to_fit_bounds=lambda **_kwargs: None,
    )
    warmup = periodic_example._WarmupSolutionAdapter(
        states={"theta": theta, "omega": omega},
        controls={"last_pulse_width_Biceps": pulse_width},
    )

    periodic_example.apply_phase_shifted_warmup_initial_guess(nmpc, warmup)

    np.testing.assert_allclose(theta_target, theta - 2.0 * np.pi)
    np.testing.assert_allclose(omega_target, omega)
    np.testing.assert_allclose(pulse_width_target, pulse_width)


def test_collocation_control_transfer_uses_shooting_nodes_not_state_subnodes():
    controls = {"last_pulse_width_Biceps": np.arange(60, dtype=float)[None, :]}
    target = np.zeros((1, 60))
    nmpc = SimpleNamespace(
        nodes_per_cycle=120,
        control_nodes_per_cycle=30,
        transfer_initial_guess_mode="anchored",
        nlp=[
            SimpleNamespace(
                u_init={"last_pulse_width_Biceps": SimpleNamespace(init=target)}
            )
        ],
    )

    MyCyclicNMPC.set_init_cyclical(
        nmpc,
        controls,
        "last_pulse_width_Biceps",
        0,
        state=False,
    )

    np.testing.assert_array_equal(target[0, :30], np.arange(30, 60))
    np.testing.assert_array_equal(target[0, 30:], np.arange(30, 60))


def test_one_cycle_transfer_preserves_profiles_instead_of_broadcasting_terminal_node():
    nodes_per_cycle = 3
    q_source = np.array([0.0, -1.0, -2.0, -2 * np.pi])
    cyclical_source = np.array([1.0, 2.0, 3.0, 1.5])
    fatigue_source = np.array([10.0, 10.5, 11.0, 12.0])
    nmpc = SimpleNamespace(
        nodes_per_cycle=nodes_per_cycle,
        use_signed_wheel_shift=True,
        transfer_initial_guess_mode="anchored",
        nlp=[
            SimpleNamespace(
                x_init={
                    "q": SimpleNamespace(init=np.zeros((1, nodes_per_cycle + 1))),
                    "F": SimpleNamespace(init=np.zeros((1, nodes_per_cycle + 1))),
                    "A": SimpleNamespace(init=np.zeros((1, nodes_per_cycle + 1))),
                }
            )
        ],
        _wheel_cycle_shift=lambda _states: -2 * np.pi,
    )
    states = {
        "q": q_source[None, :],
        "F": cyclical_source[None, :],
        "A": fatigue_source[None, :],
    }

    MyCyclicNMPC.set_init_cyclical_wheel(nmpc, states, "q", 0)
    MyCyclicNMPC.set_init_cyclical(nmpc, states, "F", 0)
    MyCyclicNMPC.set_init_continuous(nmpc, states, "A", 0)

    np.testing.assert_allclose(nmpc.nlp[0].x_init["q"].init[0], q_source - 2 * np.pi)
    np.testing.assert_allclose(nmpc.nlp[0].x_init["F"].init[0], cyclical_source)
    np.testing.assert_allclose(nmpc.nlp[0].x_init["A"].init[0], fatigue_source + 2.0)


def test_cyclical_transfer_keeps_complete_state_cycle_and_repeats_controls():
    nmpc = SimpleNamespace(
        nodes_per_cycle=3,
        control_nodes_per_cycle=3,
        transfer_initial_guess_mode="anchored",
        nlp=[
            SimpleNamespace(
                x_init={"state": SimpleNamespace(init=np.zeros((1, 7)))},
                u_init={"control": SimpleNamespace(init=np.zeros((1, 6)))},
            )
        ],
    )
    states = {"state": np.arange(7, dtype=float)[None, :]}
    controls = {"control": np.arange(6, dtype=float)[None, :]}

    MyCyclicNMPC.set_init_cyclical(nmpc, states, "state", 0)
    MyCyclicNMPC.set_init_cyclical(nmpc, controls, "control", 0, state=False)

    np.testing.assert_allclose(
        nmpc.nlp[0].x_init["state"].init[0], [3, 4, 5, 6, 7, 8, 9]
    )
    np.testing.assert_allclose(
        nmpc.nlp[0].u_init["control"].init[0], [3, 4, 5, 3, 4, 5]
    )


def test_historical_collocation_control_transfer_uses_control_cycle_length():
    source = np.arange(60, dtype=float)[None, :]
    nmpc = SimpleNamespace(
        nodes_per_cycle=120,
        control_nodes_per_cycle=30,
        transfer_initial_guess_mode="historical",
        nlp=[
            SimpleNamespace(u_init={"control": SimpleNamespace(init=np.zeros((1, 60)))})
        ],
    )

    MyCyclicNMPC.set_init_cyclical(nmpc, {"control": source}, "control", 0, state=False)

    np.testing.assert_allclose(
        nmpc.nlp[0].u_init["control"].init[0],
        np.concatenate((np.arange(30, 60), np.arange(30, 60))),
    )


def test_cyclical_transfer_can_repeat_states_without_extrapolating_cycle_delta():
    nmpc = SimpleNamespace(
        nodes_per_cycle=3,
        transfer_initial_guess_mode="anchored",
        repeat_cyclical_state_initial_guess=True,
        nlp=[SimpleNamespace(x_init={"state": SimpleNamespace(init=np.zeros((1, 7)))})],
    )
    states = {"state": np.arange(7, dtype=float)[None, :]}

    MyCyclicNMPC.set_init_cyclical(nmpc, states, "state", 0)

    np.testing.assert_allclose(
        nmpc.nlp[0].x_init["state"].init[0], [3, 4, 5, 6, 4, 5, 6]
    )


def test_continuous_transfer_extrapolates_cycle_delta_without_duplicate_seam():
    nmpc = SimpleNamespace(
        nodes_per_cycle=3,
        transfer_initial_guess_mode="anchored",
        nlp=[
            SimpleNamespace(x_init={"fatigue": SimpleNamespace(init=np.zeros((1, 7)))})
        ],
    )
    states = {"fatigue": np.array([[0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 14.0]])}

    MyCyclicNMPC.set_init_continuous(nmpc, states, "fatigue", 0)

    np.testing.assert_allclose(
        nmpc.nlp[0].x_init["fatigue"].init[0], [10, 11, 12, 14, 15, 16, 18]
    )


def test_window_cache_callback_runs_before_window_is_advanced():
    events = []
    solution = SimpleNamespace(
        decision_states=lambda to_merge=None: events.append("decision_states") or {}
    )
    nmpc = SimpleNamespace(
        before_window_advance=lambda current_nmpc, current_solution: events.append(
            "before_window_advance"
        ),
        debugg_bounds=False,
        transfer_debug=False,
        update_stim=lambda: None,
        _sync_acados_state_bounds=lambda: None,
    )

    MyCyclicNMPC.advance_window_bounds_states(nmpc, solution)

    assert events == ["before_window_advance", "decision_states"]


def test_compiled_nlp_tracker_accepts_runtime_bound_changes_without_graph_rebuild():
    state_bounds = SimpleNamespace(
        min=np.array([[0.0, -10.0, -2.0 * np.pi]]),
        max=np.array([[0.0, 10.0, -2.0 * np.pi]]),
    )
    control_bounds = SimpleNamespace(
        min=np.array([[131.405e-6, 131.405e-6]]),
        max=np.array([[600e-6, 600e-6]]),
    )
    constraint_bounds = SimpleNamespace(
        min=np.array([[-0.002]]),
        max=np.array([[0.002]]),
    )
    compiled_solver = object()
    nmpc = SimpleNamespace(
        ocp_solver=SimpleNamespace(shaked_ocp_solver=compiled_solver),
        nlp=[
            SimpleNamespace(
                x_bounds={"theta": state_bounds},
                u_bounds={"last_pulse_width": control_bounds},
                g=[SimpleNamespace(bounds=constraint_bounds)],
            )
        ],
    )
    tracker = periodic_example.CompiledNlpReuseTracker(enabled=True)

    tracker.record(nmpc, 0)
    state_bounds.min[0, 0] = -2.0 * np.pi
    state_bounds.max[0, 0] = -2.0 * np.pi
    state_bounds.min[0, 2] = -4.0 * np.pi - 0.002
    state_bounds.max[0, 2] = -4.0 * np.pi + 0.002
    constraint_bounds.min[0, 0] = -2.0 * np.pi - 0.002
    constraint_bounds.max[0, 0] = -2.0 * np.pi + 0.002
    tracker.record(nmpc, 1)

    summary = tracker.summary()
    assert summary["compiled_library_build_count"] == 1
    assert summary["compiled_library_reused"] is True
    assert summary["graph_rebuild_detected"] is False
    assert summary["unique_runtime_bound_vectors"] == 2
    assert summary["runtime_bounds_changed"] is True
    assert (
        "absolute_terminal_angle_via_terminal_state_bounds" in summary["runtime_inputs"]
    )


def test_compiled_nlp_tracker_detects_a_second_generated_solver():
    bounds = SimpleNamespace(min=np.zeros((1, 3)), max=np.ones((1, 3)))
    nmpc = SimpleNamespace(
        ocp_solver=SimpleNamespace(shaked_ocp_solver=object()),
        nlp=[SimpleNamespace(x_bounds={"theta": bounds}, u_bounds={}, g=[])],
    )
    tracker = periodic_example.CompiledNlpReuseTracker(enabled=True)

    tracker.record(nmpc, 0)
    nmpc.ocp_solver.shaked_ocp_solver = object()
    tracker.record(nmpc, 1)

    summary = tracker.summary()
    assert summary["compiled_library_build_count"] == 2
    assert summary["compiled_library_reused"] is False
    assert summary["graph_rebuild_detected"] is True


def test_compiled_nlp_tracker_requires_source_at_every_solve(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "nlp.c"
    source.write_text("/* generated */", encoding="utf-8")
    bounds = SimpleNamespace(min=np.zeros((1, 3)), max=np.ones((1, 3)))
    nmpc = SimpleNamespace(
        ocp_solver=SimpleNamespace(shaked_ocp_solver=object()),
        nlp=[SimpleNamespace(x_bounds={"theta": bounds}, u_bounds={}, g=[])],
    )
    tracker = periodic_example.CompiledNlpReuseTracker(enabled=True)

    tracker.record(nmpc, 0)
    source.unlink()
    tracker.record(nmpc, 1)

    summary = tracker.summary()
    assert summary["observed_solves"] == 2
    assert summary["compiled_source_observation_count"] == 1
    assert summary["unique_compiled_source_versions"] == 1
    assert summary["compiled_source_reused"] is False


def test_c_codegen_resolves_relative_runtime_inputs_before_changing_directory(
    monkeypatch, tmp_path
):
    observed = {}
    args = SimpleNamespace(
        ipopt_c_compile=True,
        fatrop_c_compile=False,
        madnlp_c_compile=False,
        standard_warmup_seed=Path("seeds/warmup.npz"),
        common_initial_solution="seeds/common.npz",
        common_initial_solution_output=Path("results/common.npz"),
        reduced_cycling_profile=None,
        periodic_ipopt_refinement_window_cache=None,
    )

    def fake_solve_case(current_args, echo):
        observed["cwd"] = Path.cwd()
        observed["seed"] = current_args.standard_warmup_seed
        observed["common"] = current_args.common_initial_solution
        observed["output"] = current_args.common_initial_solution_output
        return {"success": True}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(comparison_example, "solve_case", fake_solve_case)

    result = comparison_example._run_benchmark_case("ipopt", args, echo=False)

    assert observed["cwd"] != tmp_path
    assert observed["seed"] == tmp_path / "seeds/warmup.npz"
    assert observed["common"] == tmp_path / "seeds/common.npz"
    assert observed["output"] == tmp_path / "results/common.npz"
    assert result["success"] is True


def test_terminal_wheel_target_is_anchored_to_new_window_start():
    q = np.zeros((3, 3))
    q[2] = [0.0, -6.0, -12.0]
    qdot = np.zeros((3, 3))
    qdot[2] = [-6.0, -6.0, -6.0]
    solution = SimpleNamespace(
        decision_states=lambda to_merge=None: {"q": q, "qdot": qdot}
    )
    q_bounds = SimpleNamespace(
        min=np.full((3, 3), -100.0),
        max=np.full((3, 3), 100.0),
    )
    qdot_bounds = SimpleNamespace(
        min=np.full((3, 3), -100.0),
        max=np.full((3, 3), 100.0),
    )
    nmpc = SimpleNamespace(
        before_window_advance=None,
        nodes_per_cycle=1,
        n_cycles_simultaneous=2,
        debugg_bounds=False,
        transfer_debug=False,
        bound_first_node_all_states=True,
        bound_first_node_wheel_qdot=False,
        advance_wheel_q_bounds=True,
        anchor_terminal_wheel_to_first_node=True,
        wheel_q_path_margin=2.0,
        use_signed_wheel_shift=True,
        first_node_state_slack={"q": [0.0, 0.0, 0.02]},
        terminal_state_slack={"q": [0.0, 0.0, 0.01]},
        nlp=[SimpleNamespace(x_bounds={"q": q_bounds, "qdot": qdot_bounds})],
        _wheel_cycle_shift=lambda states: -2.0 * np.pi,
        _state_slack_for=lambda key, index: (
            0.02 if key == "q" and index == 2 else 0.0
        ),
        _terminal_state_slack_for=lambda key, index: (
            0.01 if key == "q" and index == 2 else 0.0
        ),
        update_stim=lambda: None,
        _sync_acados_state_bounds=lambda: None,
    )

    MyCyclicNMPC.advance_window_bounds_states(nmpc, solution, n_cycles_simultaneous=2)

    expected_terminal = -6.0 - 4.0 * np.pi
    np.testing.assert_allclose(q_bounds.min[2, 2], expected_terminal - 0.01)
    np.testing.assert_allclose(q_bounds.max[2, 2], expected_terminal + 0.01)


def test_terminal_wheel_target_uses_absolute_cycle_reference_without_drift():
    q_bounds = SimpleNamespace(
        min=np.full((3, 3), -100.0),
        max=np.full((3, 3), 100.0),
    )
    qdot_bounds = SimpleNamespace(
        min=np.full((3, 3), -100.0),
        max=np.full((3, 3), 100.0),
    )
    nmpc = SimpleNamespace(
        before_window_advance=None,
        nodes_per_cycle=1,
        cycle_len=30,
        time_idx_to_cycle=30,
        n_cycles_simultaneous=2,
        debugg_bounds=False,
        transfer_debug=False,
        bound_first_node_all_states=True,
        bound_first_node_wheel_qdot=False,
        advance_wheel_q_bounds=True,
        anchor_terminal_wheel_to_first_node=False,
        anchor_wheel_q_to_absolute_reference=True,
        absolute_wheel_q_reference=0.0,
        absolute_wheel_q_cycle_shift=-2.0 * np.pi,
        absolute_wheel_q_cycle_index=0,
        wheel_q_path_margin=2.0,
        use_signed_wheel_shift=True,
        first_node_state_slack={"q": [0.0, 0.0, 0.0]},
        terminal_state_slack={"q": [0.0, 0.0, 0.002]},
        nlp=[SimpleNamespace(x_bounds={"q": q_bounds, "qdot": qdot_bounds})],
        _wheel_cycle_shift=lambda states: -2.0 * np.pi,
        _state_slack_for=lambda key, index: 0.0,
        _terminal_state_slack_for=lambda key, index: (
            0.002 if key == "q" and index == 2 else 0.0
        ),
        update_stim=lambda: None,
        _sync_acados_state_bounds=lambda: None,
    )

    first_q = np.zeros((3, 3))
    first_q[2] = [
        0.0,
        -2.0 * np.pi + 0.005,
        -4.0 * np.pi + 0.010,
    ]
    qdot = np.zeros((3, 3))
    qdot[2] = -2.0 * np.pi
    first_solution = SimpleNamespace(
        decision_states=lambda to_merge=None: {"q": first_q, "qdot": qdot}
    )

    MyCyclicNMPC.advance_window_bounds_states(
        nmpc, first_solution, n_cycles_simultaneous=2
    )

    np.testing.assert_allclose(q_bounds.min[2, 2], -6.0 * np.pi - 0.002)
    np.testing.assert_allclose(q_bounds.max[2, 2], -6.0 * np.pi + 0.002)

    second_q = np.zeros((3, 3))
    second_q[2] = [
        -2.0 * np.pi + 0.005,
        -4.0 * np.pi + 0.010,
        -6.0 * np.pi + 0.015,
    ]
    second_solution = SimpleNamespace(
        decision_states=lambda to_merge=None: {"q": second_q, "qdot": qdot}
    )

    MyCyclicNMPC.advance_window_bounds_states(
        nmpc, second_solution, n_cycles_simultaneous=2
    )

    np.testing.assert_allclose(q_bounds.min[2, 2], -8.0 * np.pi - 0.002)
    np.testing.assert_allclose(q_bounds.max[2, 2], -8.0 * np.pi + 0.002)
    assert nmpc.absolute_wheel_q_cycle_index == 2


def test_reduced_theta_target_uses_absolute_cycle_reference_without_drift():
    theta_bounds = SimpleNamespace(
        min=np.full((1, 3), -100.0),
        max=np.full((1, 3), 100.0),
    )
    omega_bounds = SimpleNamespace(
        min=np.full((1, 3), -100.0),
        max=np.full((1, 3), 100.0),
    )
    nmpc = SimpleNamespace(
        before_window_advance=None,
        position_state_key="theta",
        velocity_state_key="omega",
        wheel_state_index=0,
        nodes_per_cycle=1,
        cycle_len=30,
        time_idx_to_cycle=30,
        n_cycles_simultaneous=2,
        debugg_bounds=False,
        transfer_debug=False,
        bound_first_node_all_states=True,
        bound_first_node_wheel_qdot=False,
        advance_wheel_q_bounds=True,
        anchor_terminal_wheel_to_first_node=False,
        anchor_wheel_q_to_absolute_reference=True,
        absolute_wheel_q_reference=0.0,
        absolute_wheel_q_cycle_shift=-2.0 * np.pi,
        absolute_wheel_q_cycle_index=0,
        wheel_q_path_margin=2.0,
        use_signed_wheel_shift=True,
        first_node_state_slack={"theta": [0.0]},
        terminal_state_slack={"theta": [0.002]},
        nlp=[SimpleNamespace(x_bounds={"theta": theta_bounds, "omega": omega_bounds})],
        _wheel_cycle_shift=lambda states: -2.0 * np.pi,
        _state_slack_for=lambda key, index: 0.0,
        _terminal_state_slack_for=lambda key, index: (
            0.002 if key == "theta" and index == 0 else 0.0
        ),
        update_stim=lambda: None,
        _sync_acados_state_bounds=lambda: None,
    )
    omega = np.full((1, 3), -2.0 * np.pi)

    for cycle_index in range(2):
        theta = np.array(
            [
                [
                    -2.0 * np.pi * cycle_index + 0.005 * cycle_index,
                    -2.0 * np.pi * (cycle_index + 1) + 0.005 * (cycle_index + 1),
                    -2.0 * np.pi * (cycle_index + 2) + 0.005 * (cycle_index + 2),
                ]
            ]
        )
        solution = SimpleNamespace(
            decision_states=lambda to_merge=None, theta=theta: {
                "theta": theta,
                "omega": omega,
            }
        )
        MyCyclicNMPC.advance_window_bounds_states(
            nmpc, solution, n_cycles_simultaneous=2
        )

    np.testing.assert_allclose(theta_bounds.min[0, 2], -8.0 * np.pi - 0.002)
    np.testing.assert_allclose(theta_bounds.max[0, 2], -8.0 * np.pi + 0.002)
    assert nmpc.absolute_wheel_q_cycle_index == 2


def test_wheel_cycle_boundary_constraint_is_recentered_with_unwrapped_angle():
    penalty = SimpleNamespace(
        extra_parameters={
            "boundary_cycle_index": 1,
            "wheel_cycle_boundary_slack": 0.02,
        },
        min_bound=0.0,
        max_bound=0.0,
        bounds=SimpleNamespace(
            min=np.zeros((1, 1)),
            max=np.zeros((1, 1)),
        ),
    )
    nmpc = SimpleNamespace(nlp=[SimpleNamespace(g=[penalty])])

    summary = MyCyclicNMPC._recenter_wheel_cycle_boundary_constraints(
        nmpc,
        first_wheel_q=100.0,
        cycle_shift=-2.0 * np.pi,
    )

    center = 100.0 - 2.0 * np.pi
    np.testing.assert_allclose(penalty.bounds.min, center - 0.02)
    np.testing.assert_allclose(penalty.bounds.max, center + 0.02)
    np.testing.assert_allclose(summary[0]["center"], center)


def test_acados_cycle_boundary_is_applied_as_scaled_stage_state_bound():
    calls = []
    path_lower = np.array([-10.0, -20.0, -30.0, -40.0])
    path_upper = np.array([10.0, 20.0, 30.0, 40.0])
    q_min = np.array(
        [
            [-5.0, -5.0, -5.0],
            [-5.0, -5.0, -5.0],
            [-1.0, -20.0, -20.0],
        ]
    )
    q_max = np.array(
        [
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
            [-1.0, 20.0, 20.0],
        ]
    )
    interface = SimpleNamespace(
        ocp=SimpleNamespace(
            _cocofest_wheel_cycle_boundary_slack=0.02,
            _cocofest_cycle_len=3,
            _cocofest_cycles_per_window=2,
            _cocofest_wheel_cycle_shift=-2 * np.pi,
            nlp=[
                SimpleNamespace(
                    x_bounds={"q": SimpleNamespace(min=q_min, max=q_max)},
                    states={"q": SimpleNamespace(index=np.array([0, 1, 2]))},
                    x_scaling={
                        "q": SimpleNamespace(scaling=np.array([[1.0], [1.0], [2.0]]))
                    },
                )
            ],
        ),
        ocp_solver=SimpleNamespace(
            constraints_set=lambda stage, field, values: calls.append(
                (stage, field, np.asarray(values, dtype=float).copy())
            )
        ),
        nparams=1,
        x_bound_min=np.column_stack((path_lower, path_lower, path_lower)),
        x_bound_max=np.column_stack((path_upper, path_upper, path_upper)),
        acados_ocp=SimpleNamespace(solver_options=SimpleNamespace(N_horizon=6)),
    )

    summary = periodic_example.apply_acados_wheel_cycle_boundary_bounds(interface)

    assert [item[:2] for item in calls] == [(3, "lbx"), (3, "ubx")]
    expected_center = -1.0 - 2 * np.pi
    np.testing.assert_allclose(calls[0][2][:3], path_lower[:3])
    np.testing.assert_allclose(calls[1][2][:3], path_upper[:3])
    np.testing.assert_allclose(calls[0][2][3], (expected_center - 0.02) / 2.0)
    np.testing.assert_allclose(calls[1][2][3], (expected_center + 0.02) / 2.0)
    assert summary[0]["cycle_index"] == 1
    assert summary[0]["stage"] == 3
    np.testing.assert_allclose(summary[0]["center"], expected_center)
    np.testing.assert_allclose(summary[0]["lower"], expected_center - 0.02)
    np.testing.assert_allclose(summary[0]["upper"], expected_center + 0.02)


def test_acados_cycle_boundary_bounds_cover_every_internal_seam():
    calls = []
    interface = SimpleNamespace(
        ocp=SimpleNamespace(
            _cocofest_wheel_cycle_boundary_slack=0.02,
            _cocofest_cycle_len=3,
            _cocofest_cycles_per_window=3,
            _cocofest_wheel_cycle_shift=-2 * np.pi,
            nlp=[
                SimpleNamespace(
                    x_bounds={
                        "q": SimpleNamespace(
                            min=np.array(
                                [[-5.0] * 3, [-5.0] * 3, [-1.0, -20.0, -20.0]]
                            ),
                            max=np.array([[5.0] * 3, [5.0] * 3, [-1.0, 20.0, 20.0]]),
                        )
                    },
                    states={"q": SimpleNamespace(index=np.array([0, 1, 2]))},
                    x_scaling={
                        "q": SimpleNamespace(scaling=np.array([[1.0], [1.0], [2.0]]))
                    },
                )
            ],
        ),
        ocp_solver=SimpleNamespace(
            constraints_set=lambda stage, field, values: calls.append(
                (stage, field, np.asarray(values, dtype=float).copy())
            )
        ),
        nparams=1,
        x_bound_min=np.tile(np.array([[-10.0], [-20.0], [-30.0], [-40.0]]), (1, 3)),
        x_bound_max=np.tile(np.array([[10.0], [20.0], [30.0], [40.0]]), (1, 3)),
        acados_ocp=SimpleNamespace(
            solver_options=SimpleNamespace(N_horizon=9),
            dims=SimpleNamespace(nbx=4),
            constraints=SimpleNamespace(idxbx=np.arange(4)),
        ),
    )

    summary = periodic_example.apply_acados_wheel_cycle_boundary_bounds(interface)

    assert [(item["cycle_index"], item["stage"]) for item in summary] == [
        (1, 3),
        (2, 6),
    ]
    assert [item[:2] for item in calls] == [
        (3, "lbx"),
        (3, "ubx"),
        (6, "lbx"),
        (6, "ubx"),
    ]
    assert all(stage != 9 for stage, _, _ in calls)


def test_cycle_boundary_schedule_contains_seed_and_densifies_large_steps():
    nmpc = SimpleNamespace(
        cycle_len=3,
        n_cycles_simultaneous=2,
        _cocofest_cycles_per_window=2,
        _cocofest_wheel_cycle_shift=-2 * np.pi,
        nlp=[
            SimpleNamespace(
                x_init={
                    "q": SimpleNamespace(
                        init=np.array(
                            [
                                [0.0] * 7,
                                [0.0] * 7,
                                [-1.0, -2.0, -3.0, -6.0, -7.0, -8.0, -9.0],
                            ]
                        )
                    )
                }
            )
        ],
    )

    slacks = periodic_example.resolve_cycle_boundary_homotopy_slacks(
        nmpc, (1.0, 0.5, 0.002), maximum_step=0.05
    )

    assert slacks[0] >= abs((-6.0) - (-1.0 - 2 * np.pi)) + 0.02
    assert slacks[-1] == 0.002
    assert max(a - b for a, b in zip(slacks, slacks[1:])) <= 0.05 + 1e-12


def test_periodic_refinement_requires_measured_primal_feasibility():
    high_violation = {
        "inf_pr_available": True,
        "passes_tolerance": False,
    }
    missing_measurement = {
        "inf_pr_available": False,
        "passes_tolerance": True,
    }
    feasible_nonconverged = {
        "inf_pr_available": True,
        "passes_tolerance": True,
    }

    assert (
        periodic_example.periodic_refinement_acceptance(0, high_violation)["accepted"]
        is False
    )
    assert (
        periodic_example.periodic_refinement_acceptance(0, missing_measurement)[
            "accepted"
        ]
        is False
    )
    provisional = periodic_example.periodic_refinement_acceptance(
        1, feasible_nonconverged
    )
    assert provisional["accepted"] is True
    assert provisional["provisional"] is True


def test_horizon_seed_recenters_kinematic_boundary_bounds():
    source = SimpleNamespace(
        decision_states=lambda to_merge=None: {
            "q": np.array([[-5.0, -6.0, -7.0]]),
            "qdot": np.array([[-1.0, -2.0, -3.0]]),
        },
        decision_controls=lambda to_merge=None: {
            "last_pulse_width_Biceps": np.array([[0.0002, 0.0003]])
        },
    )

    def guess(shape):
        return SimpleNamespace(init=np.zeros(shape))

    def bounds(half_width):
        return SimpleNamespace(
            min=np.array([[-half_width, -10.0, -half_width]]),
            max=np.array([[half_width, 10.0, half_width]]),
        )

    nlp = SimpleNamespace(
        x_init={"q": guess((1, 3)), "qdot": guess((1, 3))},
        u_init={"last_pulse_width_Biceps": guess((1, 2))},
        x_bounds={"q": bounds(0.1), "qdot": bounds(0.2)},
    )
    nmpc = SimpleNamespace(
        nlp=[nlp],
        _correct_init_guess_to_fit_bounds=lambda corrected_input: None,
        _sync_acados_state_bounds=lambda: None,
    )

    periodic_example.apply_solution_directly_to_periodic_nmpc_initial_guess(
        nmpc, source, recenter_kinematic_bounds=True
    )

    np.testing.assert_allclose(nlp.x_bounds["q"].min[:, [0, 2]], [[-5.1, -7.1]])
    np.testing.assert_allclose(nlp.x_bounds["q"].max[:, [0, 2]], [[-4.9, -6.9]])
    np.testing.assert_allclose(nlp.x_bounds["qdot"].min[:, [0, 2]], [[-1.2, -3.2]])
    np.testing.assert_allclose(nlp.x_bounds["qdot"].max[:, [0, 2]], [[-0.8, -2.8]])


def test_full_dynamics_transfer_rollout_reintegrates_appended_cycle():
    class Variables(dict):
        def __init__(self, *args, shape, **kwargs):
            super().__init__(*args, **kwargs)
            self.shape = shape

    state_variables = Variables(
        {
            "q": SimpleNamespace(index=[0]),
            "qdot": SimpleNamespace(index=[1]),
        },
        shape=2,
    )
    control_variables = Variables({"acceleration": SimpleNamespace(index=[0])}, shape=1)
    x_init = {
        "q": SimpleNamespace(init=np.array([[0.0, 0.5, 1.0, 9.0, 9.0]])),
        "qdot": SimpleNamespace(init=np.array([[1.0, 1.0, 1.0, 9.0, 9.0]])),
    }
    loose_bounds = SimpleNamespace(
        min=np.full((1, 3), -100.0), max=np.full((1, 3), 100.0)
    )
    nlp = SimpleNamespace(
        x_init=x_init,
        u_init={"acceleration": SimpleNamespace(init=np.zeros((1, 4)))},
        x_bounds={"q": loose_bounds, "qdot": loose_bounds},
        states=state_variables,
        controls=control_variables,
        numerical_data_timeseries=None,
        dynamics_func=lambda time, state, control, numerical, algebraic, parameters: np.array(
            [state[1], control[0]]
        ),
    )
    nmpc = SimpleNamespace(
        nlp=[nlp],
        nodes_per_cycle=2,
        control_nodes_per_cycle=2,
        cycle_duration=1.0,
        cycle_len=2,
    )

    summary = periodic_example.rollout_transferred_cycle_full_dynamics(
        nmpc, n_substeps=2
    )

    assert summary["applied"] is True
    assert summary["start_node"] == 2
    assert summary["max_bound_violation"] == 0.0
    np.testing.assert_allclose(x_init["q"].init, [[0.0, 0.5, 1.0, 1.5, 2.0]])
    np.testing.assert_allclose(x_init["qdot"].init, [[1.0, 1.0, 1.0, 1.0, 1.0]])

    x_init["q"].init[:, 3:] = 9.0
    x_init["qdot"].init[:, 3:] = 9.0
    nlp.x_bounds["q"] = SimpleNamespace(
        min=np.full((1, 3), -100.0), max=np.full((1, 3), 1.0)
    )
    rejected = periodic_example.rollout_transferred_cycle_full_dynamics(
        nmpc, n_substeps=2, max_allowed_bound_violation=0.1
    )

    assert rejected["applied"] is False
    assert rejected["max_bound_violation"] == 1.0
    np.testing.assert_allclose(x_init["q"].init[:, 3:], 9.0)


def test_appended_pulse_width_scaling_preserves_retained_cycle_and_clips():
    values = np.array([[0.1, 0.2, 0.2, 0.3]])
    nmpc = SimpleNamespace(
        nodes_per_cycle=2,
        control_nodes_per_cycle=2,
        nlp=[
            SimpleNamespace(
                u_init={"last_pulse_width_Biceps": SimpleNamespace(init=values)},
                u_bounds={
                    "last_pulse_width_Biceps": SimpleNamespace(
                        min=np.zeros((1, 3)), max=np.full((1, 3), 0.6)
                    )
                },
            )
        ],
    )

    summary = periodic_example.scale_appended_pulse_width_controls(nmpc, 2.5)

    np.testing.assert_allclose(values, [[0.1, 0.2, 0.5, 0.6]])
    assert summary["start_node"] == 2
    assert summary["controls"]["last_pulse_width_Biceps"]["clipped_count"] == 1


def test_acados_irk_transfer_rollout_uses_scaled_variables_and_stage_data():
    class Variables(dict):
        def __init__(self, *args, shape, **kwargs):
            super().__init__(*args, **kwargs)
            self.shape = shape

    class FakeSimulator:
        def __init__(self):
            self.acados_sim = SimpleNamespace(dims=SimpleNamespace(nx=2, nu=1))
            self.calls = []
            self.settings = []

        def set(self, field, value):
            assert field in ("T", "t0")
            self.settings.append((field, value.copy()))

        def simulate(self, x, u, p):
            self.calls.append((x.copy(), u.copy(), p.copy()))
            return x + np.array([u[0], p[0]])

        def get(self, field):
            assert field == "time_tot"
            return 0.001

    states = Variables(
        {"q": SimpleNamespace(index=[0]), "qdot": SimpleNamespace(index=[1])},
        shape=2,
    )
    controls = Variables({"acceleration": SimpleNamespace(index=[0])}, shape=1)
    x_init = {
        "q": SimpleNamespace(init=np.array([[0.0, 1.0, 2.0, 9.0, 9.0]])),
        "qdot": SimpleNamespace(init=np.array([[2.0, 2.0, 2.0, 9.0, 9.0]])),
    }
    scaling = lambda value: SimpleNamespace(  # noqa: E731 - compact test fixture.
        scaling=np.array([[value]])
    )
    loose_bounds = SimpleNamespace(
        min=np.full((1, 3), -100.0), max=np.full((1, 3), 100.0)
    )
    numerical_data = {"periodic_calcium": np.arange(1.0, 6.0).reshape((1, 1, 5))}
    nlp = SimpleNamespace(
        x_init=x_init,
        u_init={"acceleration": SimpleNamespace(init=np.full((1, 4), 6.0))},
        x_bounds={"q": loose_bounds, "qdot": loose_bounds},
        states=states,
        controls=controls,
        x_scaling={"q": scaling(2.0), "qdot": scaling(4.0)},
        u_scaling={"acceleration": scaling(3.0)},
        numerical_data_timeseries=numerical_data,
    )
    simulator = FakeSimulator()
    nmpc = SimpleNamespace(
        nlp=[nlp],
        nodes_per_cycle=2,
        control_nodes_per_cycle=2,
        cycle_duration=1.0,
        cycle_len=2,
        _cocofest_acados_sim_solver=simulator,
    )

    summary = periodic_example.rollout_transferred_cycle_acados_irk(nmpc)

    assert summary["applied"] is True
    assert summary["simulator_built"] is False
    np.testing.assert_allclose(summary["simulation_time_s"], 0.002)
    np.testing.assert_allclose(x_init["q"].init, [[0.0, 1.0, 2.0, 6.0, 10.0]])
    np.testing.assert_allclose(x_init["qdot"].init, [[2.0, 2.0, 2.0, 14.0, 30.0]])
    np.testing.assert_allclose(simulator.calls[0][0], [1.0, 0.5])
    np.testing.assert_allclose(simulator.calls[0][1], [2.0])
    np.testing.assert_allclose(simulator.calls[0][2], [3.0])
    assert [field for field, _ in simulator.settings] == ["T", "t0", "T", "t0"]
    np.testing.assert_allclose(simulator.settings[0][1], [0.5])
    np.testing.assert_allclose(simulator.settings[1][1], [1.0])
    assert summary["max_scaled_bound_violation_by_key"] == {"q": 0.0, "qdot": 0.0}


def test_projected_acados_transfer_selector_keeps_mechanically_better_rollout(
    monkeypatch,
):
    q = np.array([[0.0, 0.0]])
    qdot = np.array([[0.0, 0.0]])
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={
                    "q": SimpleNamespace(init=q),
                    "qdot": SimpleNamespace(init=qdot),
                },
                u_init={"u": SimpleNamespace(init=np.array([[0.0]]))},
            )
        ]
    )
    monkeypatch.setattr(
        periodic_example,
        "project_transferred_initial_guess_to_bounds",
        lambda _: {"state_max_change": 0.0, "control_max_change": 0.0},
    )

    def fake_rollout(candidate, max_allowed_bound_violation):
        assert max_allowed_bound_violation is None
        candidate.nlp[0].x_init["q"].init[:, :] = 1.0
        candidate.nlp[0].x_init["qdot"].init[:, :] = 1.0
        return {
            "applied": True,
            "max_bound_violation": 9.7,
            "max_bound_violation_by_key": {"q": 0.3, "qdot": 9.7, "F_Biceps": 2.0},
            "max_scaled_bound_violation_by_key": {
                "q": 0.05,
                "qdot": 0.8,
                "F_Biceps": 0.2,
            },
        }

    monkeypatch.setattr(
        periodic_example, "rollout_transferred_cycle_acados_irk", fake_rollout
    )

    def fake_defects(candidate):
        is_rollout = bool(candidate.nlp[0].x_init["qdot"].init[0, 0])
        return {
            "scaled_by_block": (
                {"q": 0.05, "qdot": 0.05}
                if is_rollout
                else {"q": 0.01, "qdot": 0.4}
            )
        }

    monkeypatch.setattr(
        periodic_example, "_full_dynamics_rollout_defect_details", fake_defects
    )

    summary = periodic_example.select_projected_acados_irk_transfer_candidate(nmpc)

    assert summary["applied"] is True
    assert summary["candidate_selection"]["selected"] == "rollout"
    assert summary["candidate_selection"]["shift_score"] == 4.0
    assert summary["candidate_selection"]["rollout_score"] == 0.5
    assert summary["candidate_selection"]["raw_bound_violations"] == {
        "q_rad": 0.3,
        "qdot_rad_s": 9.7,
        "other_scaled": 0.2,
    }
    np.testing.assert_allclose(q, 1.0)
    np.testing.assert_allclose(qdot, 1.0)


def test_projected_acados_transfer_selector_restores_shift_when_qdot_guard_fails(
    monkeypatch,
):
    q = np.array([[2.0, 3.0]])
    qdot = np.array([[4.0, 5.0]])
    controls = np.array([[6.0]])
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={
                    "q": SimpleNamespace(init=q),
                    "qdot": SimpleNamespace(init=qdot),
                },
                u_init={"u": SimpleNamespace(init=controls)},
            )
        ]
    )
    monkeypatch.setattr(
        periodic_example,
        "project_transferred_initial_guess_to_bounds",
        lambda _: {"state_max_change": 0.0, "control_max_change": 0.0},
    )

    def fake_rollout(candidate, max_allowed_bound_violation):
        candidate.nlp[0].x_init["q"].init[:, :] = 10.0
        candidate.nlp[0].x_init["qdot"].init[:, :] = 11.0
        candidate.nlp[0].u_init["u"].init[:, :] = 12.0
        return {
            "applied": True,
            "max_bound_violation": 13.0,
            "max_bound_violation_by_key": {"q": 0.2, "qdot": 13.0},
            "max_scaled_bound_violation_by_key": {"q": 0.03, "qdot": 1.0},
        }

    monkeypatch.setattr(
        periodic_example, "rollout_transferred_cycle_acados_irk", fake_rollout
    )
    monkeypatch.setattr(
        periodic_example,
        "_full_dynamics_rollout_defect_details",
        lambda candidate: {
            "scaled_by_block": (
                {"q": 0.01, "qdot": 0.01}
                if candidate.nlp[0].x_init["q"].init[0, 0] == 10.0
                else {"q": 0.05, "qdot": 0.5}
            )
        },
    )

    summary = periodic_example.select_projected_acados_irk_transfer_candidate(nmpc)

    assert summary["applied"] is False
    assert summary["candidate_selection"]["selected"] == "shift"
    assert summary["candidate_selection"]["checks"]["qdot_bound_guard"] is False
    assert "qdot_bound_guard" in summary["candidate_selection"]["reason"]
    np.testing.assert_allclose(q, [[2.0, 3.0]])
    np.testing.assert_allclose(qdot, [[4.0, 5.0]])
    np.testing.assert_allclose(controls, [[6.0]])


def test_projected_acados_transfer_selector_supports_reduced_theta_omega(
    monkeypatch,
):
    theta = np.zeros((1, 2))
    omega = np.zeros((1, 2))
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={
                    "theta": SimpleNamespace(init=theta),
                    "omega": SimpleNamespace(init=omega),
                },
                u_init={"u": SimpleNamespace(init=np.zeros((1, 1)))},
            )
        ]
    )
    monkeypatch.setattr(
        periodic_example,
        "project_transferred_initial_guess_to_bounds",
        lambda _: {"state_max_change": 0.0, "control_max_change": 0.0},
    )

    def fake_rollout(candidate, max_allowed_bound_violation):
        candidate.nlp[0].x_init["theta"].init[:, :] = 1.0
        candidate.nlp[0].x_init["omega"].init[:, :] = 1.0
        return {
            "applied": True,
            "max_bound_violation": 8.0,
            "max_bound_violation_by_key": {"theta": 0.4, "omega": 8.0},
            "max_scaled_bound_violation_by_key": {"theta": 0.06, "omega": 0.7},
        }

    monkeypatch.setattr(
        periodic_example, "rollout_transferred_cycle_acados_irk", fake_rollout
    )
    monkeypatch.setattr(
        periodic_example,
        "_full_dynamics_rollout_defect_details",
        lambda candidate: {
            "scaled_by_block": (
                {"q": 0.04, "qdot": 0.04}
                if candidate.nlp[0].x_init["theta"].init[0, 0]
                else {"q": 0.02, "qdot": 0.3}
            )
        },
    )

    summary = periodic_example.select_projected_acados_irk_transfer_candidate(nmpc)

    assert summary["applied"] is True
    assert summary["candidate_selection"]["raw_bound_violations"] == {
        "q_rad": 0.4,
        "qdot_rad_s": 8.0,
        "other_scaled": 0.0,
    }
    np.testing.assert_allclose(theta, 1.0)
    np.testing.assert_allclose(omega, 1.0)


def test_acados_irk_transfer_rejects_dimension_mismatch_without_mutating_guess():
    class Variables(dict):
        def __init__(self, *args, shape, **kwargs):
            super().__init__(*args, **kwargs)
            self.shape = shape

    simulator = SimpleNamespace(
        acados_sim=SimpleNamespace(dims=SimpleNamespace(nx=3, nu=1))
    )
    states = Variables({"q": SimpleNamespace(index=[0])}, shape=1)
    controls = Variables({"u": SimpleNamespace(index=[0])}, shape=1)
    guess = np.zeros((1, 3))
    nlp = SimpleNamespace(
        x_init={"q": SimpleNamespace(init=guess)},
        u_init={"u": SimpleNamespace(init=np.zeros((1, 2)))},
        x_bounds={"q": SimpleNamespace(min=np.full((1, 3), -1.0), max=np.ones((1, 3)))},
        states=states,
        controls=controls,
        x_scaling={"q": SimpleNamespace(scaling=np.ones((1, 1)))},
        u_scaling={"u": SimpleNamespace(scaling=np.ones((1, 1)))},
    )
    nmpc = SimpleNamespace(
        nlp=[nlp],
        nodes_per_cycle=1,
        control_nodes_per_cycle=1,
        _cocofest_acados_sim_solver=simulator,
    )

    with np.testing.assert_raises_regex(ValueError, "dimensions do not match"):
        periodic_example.rollout_transferred_cycle_acados_irk(nmpc)

    np.testing.assert_allclose(guess, 0.0)


def test_transfer_bound_homotopy_never_relaxes_first_node():
    bounds = SimpleNamespace(
        min=np.array([[-0.1, -1.0, -2.1]]),
        max=np.array([[0.1, 1.0, -1.9]]),
    )
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={"qdot": SimpleNamespace(init=np.array([[0.0, -4.0, -5.0]]))},
                x_bounds={"qdot": bounds},
            )
        ],
        _sync_acados_state_bounds=lambda: None,
    )
    original = periodic_example._copy_state_bounds(nmpc)
    relaxed, expansion = periodic_example.build_relaxed_transfer_state_bounds(
        nmpc, padding=0.1
    )

    np.testing.assert_allclose(relaxed["qdot"][0][:, 0], [-0.1])
    np.testing.assert_allclose(relaxed["qdot"][1][:, 0], [0.1])
    assert relaxed["qdot"][0][0, 1] < -4.0
    assert relaxed["qdot"][0][0, 2] < -5.0
    assert expansion["qdot"] > 0.0

    periodic_example.apply_transfer_state_bound_fraction(
        nmpc, original, relaxed, fraction=0.5
    )
    np.testing.assert_allclose(bounds.min[:, 0], [-0.1])
    np.testing.assert_allclose(bounds.max[:, 0], [0.1])
    assert relaxed["qdot"][0][0, 1] < bounds.min[0, 1] < -1.0


def test_qdot_projection_is_recomputed_from_q_and_clipped_to_bounds():
    qdot_init = SimpleNamespace(init=np.zeros((1, 3)))
    nmpc = SimpleNamespace(
        cycle_duration=1.0,
        cycle_len=2,
        nlp=[
            SimpleNamespace(
                x_init={
                    "q": SimpleNamespace(init=np.array([[0.0, -1.0, -3.0]])),
                    "qdot": qdot_init,
                },
                x_bounds={
                    "qdot": SimpleNamespace(
                        min=np.array([[-10.0, -3.0, -3.0]]),
                        max=np.array([[10.0, 3.0, 3.0]]),
                    )
                },
            )
        ],
        _sync_acados_state_bounds=lambda: None,
    )

    summary = periodic_example.project_qdot_initial_guess_from_q(nmpc)

    np.testing.assert_allclose(qdot_init.init, [[-2.0, -3.0, -3.0]])
    assert summary == {
        "applied": True,
        "start_node": 0,
        "accepted_step": 1.0,
        "scaled_defect_before": None,
        "scaled_defect_after": None,
        "max_change": 3.0,
        "clipped_count": 2,
    }


def test_qdot_projection_can_preserve_the_solved_cycle():
    qdot_init = SimpleNamespace(init=np.array([[1.0, 2.0, 3.0, 4.0]]))
    nmpc = SimpleNamespace(
        cycle_duration=1.0,
        cycle_len=2,
        nlp=[
            SimpleNamespace(
                x_init={
                    "q": SimpleNamespace(init=np.array([[0.0, -1.0, -2.0, -4.0]])),
                    "qdot": qdot_init,
                },
                x_bounds={
                    "qdot": SimpleNamespace(
                        min=np.array([[-10.0, -10.0, -10.0]]),
                        max=np.array([[10.0, 10.0, 10.0]]),
                    )
                },
            )
        ],
        _sync_acados_state_bounds=lambda: None,
    )

    summary = periodic_example.project_qdot_initial_guess_from_q(nmpc, start_node=2)

    np.testing.assert_allclose(qdot_init.init, [[1.0, 2.0, -4.0, -4.0]])
    assert summary["start_node"] == 2
    assert summary["max_change"] == 8.0


def test_qdot_projection_selects_the_best_dynamics_step(monkeypatch):
    qdot_init = SimpleNamespace(init=np.zeros((1, 3)))
    nmpc = SimpleNamespace(
        cycle_duration=1.0,
        cycle_len=2,
        nlp=[
            SimpleNamespace(
                x_init={
                    "q": SimpleNamespace(init=np.array([[0.0, -2.0, -4.0]])),
                    "qdot": qdot_init,
                },
                x_bounds={
                    "qdot": SimpleNamespace(
                        min=np.array([[-10.0, -10.0, -10.0]]),
                        max=np.array([[10.0, 10.0, 10.0]]),
                    )
                },
            )
        ],
        _sync_acados_state_bounds=lambda: None,
    )

    def fake_defects(candidate_nmpc, n_substeps):
        values = candidate_nmpc.nlp[0].x_init["qdot"].init
        score = float(np.max(np.abs(values + 2.0)))
        return {"scaled_by_block": {"qdot": score}}

    monkeypatch.setattr(
        periodic_example, "_full_dynamics_rollout_defect_details", fake_defects
    )

    summary = periodic_example.project_qdot_initial_guess_from_q(
        nmpc,
        select_by_dynamics=True,
    )

    np.testing.assert_allclose(qdot_init.init, [[-2.0, -2.0, -2.0]])
    assert summary["accepted_step"] == 0.5
    assert summary["scaled_defect_before"] == 2.0
    assert summary["scaled_defect_after"] == 0.0


def test_reduced_mechanical_restoration_adjusts_qdot_and_controls(monkeypatch):
    qdot_init = SimpleNamespace(init=np.zeros((1, 5)))
    pulse_width_init = SimpleNamespace(init=np.full((1, 4), 2e-4))
    nmpc = SimpleNamespace(
        cycle_duration=1.0,
        cycle_len=2,
        nlp=[
            SimpleNamespace(
                x_init={
                    "q": SimpleNamespace(
                        init=np.array([[0.0, -1.0, -2.0, -3.0, -4.0]])
                    ),
                    "qdot": qdot_init,
                },
                x_bounds={
                    "qdot": SimpleNamespace(
                        min=np.full((1, 5), -10.0),
                        max=np.full((1, 5), 10.0),
                    )
                },
                u_init={"last_pulse_width_Biceps": pulse_width_init},
                u_bounds={
                    "last_pulse_width_Biceps": SimpleNamespace(
                        min=np.full((1, 4), 1e-4),
                        max=np.full((1, 4), 6e-4),
                    )
                },
            )
        ],
        _sync_acados_state_bounds=lambda: None,
    )

    def fake_rollout(candidate_nmpc, start_node, n_substeps):
        qdot = candidate_nmpc.nlp[0].x_init["qdot"].init[0, -1]
        pulse_width = np.mean(
            candidate_nmpc.nlp[0].u_init["last_pulse_width_Biceps"].init[0, start_node:]
        )
        return np.array([qdot + 1.0, (pulse_width - 2.25e-4) / 5e-5])

    monkeypatch.setattr(
        periodic_example,
        "_appended_mechanical_rollout_residual",
        fake_rollout,
    )

    summary = periodic_example.restore_appended_cycle_mechanics(
        nmpc,
        start_node=2,
        control_radius=5e-5,
        regularization=0.0,
    )

    assert summary["applied"] is True
    np.testing.assert_allclose(qdot_init.init[:, :3], 0.0)
    np.testing.assert_allclose(qdot_init.init[:, 3:], -1.0)
    np.testing.assert_allclose(pulse_width_init.init[:, :2], 2e-4)
    np.testing.assert_allclose(pulse_width_init.init[:, 2:], 2.25e-4)
    np.testing.assert_allclose(summary["accepted_parameters"], [0.5, 0.5])
    assert summary["score_after"] < 1e-10


def test_transfer_bound_homotopy_only_relaxes_mechanical_states():
    qdot_bounds = SimpleNamespace(
        min=np.array([[-0.1, -1.0, -2.1]]),
        max=np.array([[0.1, 1.0, -1.9]]),
    )
    fatigue_bounds = SimpleNamespace(
        min=np.array([[0.0, 0.0, 0.0]]),
        max=np.array([[1.0, 1.0, 1.0]]),
    )
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={
                    "qdot": SimpleNamespace(init=np.array([[0.0, -4.0, -5.0]])),
                    "A_Triceps": SimpleNamespace(init=np.array([[1.0, 10.0, 20.0]])),
                },
                x_bounds={
                    "qdot": qdot_bounds,
                    "A_Triceps": fatigue_bounds,
                },
            )
        ]
    )

    relaxed, expansion = periodic_example.build_relaxed_transfer_state_bounds(
        nmpc, padding=0.1
    )

    assert expansion["qdot"] > 0.0
    assert expansion["A_Triceps"] == 0.0
    np.testing.assert_allclose(relaxed["A_Triceps"][0], fatigue_bounds.min)
    np.testing.assert_allclose(relaxed["A_Triceps"][1], fatigue_bounds.max)


def test_transfer_bound_homotopy_relaxes_reduced_mechanical_states():
    theta_bounds = SimpleNamespace(
        min=np.array([[-0.1, -1.0, -2.1]]),
        max=np.array([[0.1, 1.0, -1.9]]),
    )
    omega_bounds = SimpleNamespace(
        min=np.array([[-0.1, -2.0, -3.0]]),
        max=np.array([[0.1, 2.0, 3.0]]),
    )
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={
                    "theta": SimpleNamespace(init=np.array([[0.0, -4.0, -5.0]])),
                    "omega": SimpleNamespace(init=np.array([[0.0, -8.0, -9.0]])),
                },
                x_bounds={"theta": theta_bounds, "omega": omega_bounds},
            )
        ]
    )

    relaxed, expansion = periodic_example.build_relaxed_transfer_state_bounds(
        nmpc, padding=0.1
    )

    assert expansion["theta"] > 0.0
    assert expansion["omega"] > 0.0
    np.testing.assert_allclose(relaxed["theta"][0][:, 0], theta_bounds.min[:, 0])
    np.testing.assert_allclose(relaxed["omega"][1][:, 0], omega_bounds.max[:, 0])
    assert relaxed["theta"][0][0, 1] < -4.0
    assert relaxed["omega"][0][0, 1] < -8.0


def test_transfer_bound_homotopy_restores_physical_bounds(monkeypatch):
    class FakeSolver:
        def __init__(self):
            self.nlp_solver_max_iter = 100

        def set_convergence_tolerance(self, value):
            self.tolerance = value

        def set_maximum_iterations(self, value):
            self.nlp_solver_max_iter = value

    bounds = SimpleNamespace(
        min=np.array([[-0.1, -1.0, -2.1]]),
        max=np.array([[0.1, 1.0, -1.9]]),
    )
    original_min = bounds.min.copy()
    original_max = bounds.max.copy()
    nlp = SimpleNamespace(
        x_init={"qdot": SimpleNamespace(init=np.array([[0.0, -4.0, -5.0]]))},
        u_init={"u": SimpleNamespace(init=np.zeros((1, 2)))},
        x_bounds={"qdot": bounds},
    )
    nmpc = SimpleNamespace(
        nlp=[nlp],
        ocp_solver=None,
        _cocofest_fix_controls_to_warmup=True,
        _correct_init_guess_to_fit_bounds=lambda corrected_input: None,
        _sync_acados_state_bounds=lambda: None,
    )
    fixed_control_values = []
    solutions = [
        SimpleNamespace(
            status=0, solver_time_to_optimize=0.1, real_time_to_optimize=0.2
        ),
        SimpleNamespace(
            status=0, solver_time_to_optimize=0.1, real_time_to_optimize=0.2
        ),
    ]
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda solution: {"residuals": np.zeros(4)},
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda periodic_nmpc, solution: None,
    )

    summary = periodic_example.run_acados_transfer_bound_homotopy(
        nmpc,
        FakeSolver(),
        fractions=(0.0, 1.0),
        padding=0.1,
        convergence_tolerance=1e-4,
        stage_iterations=10,
        echo=False,
        solve_stage=lambda: (
            fixed_control_values.append(nmpc._cocofest_fix_controls_to_warmup)
            or solutions.pop(0)
        ),
    )

    assert summary["completed"] is True
    assert [stage["accepted"] for stage in summary["stages"]] == [True, True]
    np.testing.assert_allclose(bounds.min, original_min)
    np.testing.assert_allclose(bounds.max, original_max)
    assert nmpc._cocofest_fix_controls_to_warmup is True
    assert fixed_control_values == [False, False]


def test_transfer_bound_homotopy_accepts_last_finite_maxiter_nlp_iterate(
    monkeypatch,
):
    class FakeSolver:
        nlp_solver_max_iter = 100

        def set_convergence_tolerance(self, value):
            self.tolerance = value

        def set_maximum_iterations(self, value):
            self.nlp_solver_max_iter = value

    bounds = SimpleNamespace(
        min=np.array([[-0.1, -1.0, -2.1]]),
        max=np.array([[0.1, 1.0, -1.9]]),
    )
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={"qdot": SimpleNamespace(init=np.array([[0.0, -4.0, -5.0]]))},
                u_init={"u": SimpleNamespace(init=np.zeros((1, 2)))},
                x_bounds={"qdot": bounds},
            )
        ],
        ocp_solver=None,
        _cocofest_fix_controls_to_warmup=False,
        _correct_init_guess_to_fit_bounds=lambda corrected_input: None,
        _sync_acados_state_bounds=lambda: None,
    )
    solution = SimpleNamespace(
        status=2, solver_time_to_optimize=0.1, real_time_to_optimize=0.2
    )
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda _: {
            "residuals": np.array([1.0, 100.0, 0.0, 0.0]),
            "res_stat_all": np.array([0.02]),
            "res_eq_all": np.array([0.03]),
            "res_ineq_all": np.array([0.0]),
            "res_comp_all": np.array([0.01]),
        },
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda periodic_nmpc, accepted_solution: None,
    )
    monkeypatch.setattr(
        periodic_example,
        "reset_acados_solver_memory",
        lambda periodic_nmpc: True,
    )

    summary = periodic_example.run_acados_transfer_bound_homotopy(
        nmpc,
        FakeSolver(),
        fractions=(1.0,),
        padding=0.1,
        convergence_tolerance=0.05,
        stage_iterations=10,
        echo=False,
        solve_stage=lambda: solution,
    )

    assert summary["completed"] is True
    assert summary["stages"][0]["accepted"] is True
    assert summary["stages"][0]["residual_history_eligible"] is True
    assert summary["stages"][0]["accepted_from_residual_history"] is True
    assert summary["stages"][0]["solver_reset"] is True


def test_transfer_bound_homotopy_rejects_stale_history_after_qp_failure(
    monkeypatch,
):
    class FakeSolver:
        nlp_solver_max_iter = 100

        def set_convergence_tolerance(self, value):
            self.tolerance = value

        def set_maximum_iterations(self, value):
            self.nlp_solver_max_iter = value

    bounds = SimpleNamespace(
        min=np.array([[-0.1, -1.0, -2.1]]),
        max=np.array([[0.1, 1.0, -1.9]]),
    )
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={"qdot": SimpleNamespace(init=np.array([[0.0, -4.0, -5.0]]))},
                u_init={"u": SimpleNamespace(init=np.zeros((1, 2)))},
                x_bounds={"qdot": bounds},
            )
        ],
        ocp_solver=None,
        _cocofest_fix_controls_to_warmup=False,
        _correct_init_guess_to_fit_bounds=lambda corrected_input: None,
        _sync_acados_state_bounds=lambda: None,
    )
    solution = SimpleNamespace(
        status=4, solver_time_to_optimize=0.1, real_time_to_optimize=0.2
    )
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda _: {
            "residuals": np.array([100.0, 10.0, 1.0, 0.1]),
            "res_stat_all": np.array([1e-8]),
            "res_eq_all": np.array([1e-8]),
            "res_ineq_all": np.array([1e-8]),
            "res_comp_all": np.array([1e-8]),
        },
    )
    monkeypatch.setattr(
        periodic_example,
        "reset_acados_solver_memory",
        lambda periodic_nmpc: True,
    )

    summary = periodic_example.run_acados_transfer_bound_homotopy(
        nmpc,
        FakeSolver(),
        fractions=(1.0,),
        padding=0.1,
        convergence_tolerance=1e-4,
        stage_iterations=10,
        max_restarts=0,
        echo=False,
        solve_stage=lambda: solution,
    )

    assert summary["completed"] is False
    assert summary["termination_reason"] == "initial_stage_failed"
    assert summary["stages"][0]["accepted"] is False
    assert summary["stages"][0]["residual_history_eligible"] is False
    assert summary["stages"][0]["accepted_from_residual_history"] is False


def test_transfer_bound_homotopy_only_requires_stationarity_at_physical_stage(
    monkeypatch,
):
    class FakeSolver:
        nlp_solver_max_iter = 100

        def set_convergence_tolerance(self, value):
            self.tolerance = value

        def set_maximum_iterations(self, value):
            self.nlp_solver_max_iter = value

    bounds = SimpleNamespace(
        min=np.array([[-0.1, -1.0, -2.1]]),
        max=np.array([[0.1, 1.0, -1.9]]),
    )
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={"qdot": SimpleNamespace(init=np.array([[0.0, -4.0, -5.0]]))},
                u_init={"u": SimpleNamespace(init=np.zeros((1, 2)))},
                x_bounds={"qdot": bounds},
            )
        ],
        ocp_solver=None,
        _cocofest_fix_controls_to_warmup=False,
        _correct_init_guess_to_fit_bounds=lambda corrected_input: None,
        _sync_acados_state_bounds=lambda: None,
    )
    solutions = [
        SimpleNamespace(
            status=2, solver_time_to_optimize=0.1, real_time_to_optimize=0.2
        ),
        SimpleNamespace(
            status=0, solver_time_to_optimize=0.1, real_time_to_optimize=0.2
        ),
    ]
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda _: {
            "residuals": np.array([1e-2, 1e-5, 0.0, 1e-6]),
            "res_stat_all": np.array([1e-2]),
            "res_eq_all": np.array([1e-5]),
            "res_ineq_all": np.array([0.0]),
            "res_comp_all": np.array([1e-6]),
        },
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda periodic_nmpc, accepted_solution: None,
    )
    monkeypatch.setattr(
        periodic_example,
        "reset_acados_solver_memory",
        lambda periodic_nmpc: True,
    )

    summary = periodic_example.run_acados_transfer_bound_homotopy(
        nmpc,
        FakeSolver(),
        fractions=(0.0, 1.0),
        padding=0.1,
        convergence_tolerance=1e-4,
        stage_iterations=10,
        max_restarts=0,
        echo=False,
        solve_stage=lambda: solutions.pop(0),
    )

    assert summary["completed"] is False
    assert [stage["accepted"] for stage in summary["stages"]] == [True, False]
    assert summary["stages"][0]["accepted_as_intermediate_primal_feasible"] is True
    assert summary["stages"][1]["accepted_as_intermediate_primal_feasible"] is False


def test_transfer_bound_homotopy_restores_best_stored_intermediate_iterate(
    monkeypatch,
):
    class FakeSolver:
        nlp_solver_max_iter = 100

        def set_convergence_tolerance(self, value):
            self.tolerance = value

        def set_maximum_iterations(self, value):
            self.nlp_solver_max_iter = value

    bounds = SimpleNamespace(
        min=np.array([[-0.1, -1.0, -2.1]]),
        max=np.array([[0.1, 1.0, -1.9]]),
    )
    state_guess = SimpleNamespace(init=np.array([[0.0, -4.0, -5.0]]))
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={"qdot": state_guess},
                u_init={"u": SimpleNamespace(init=np.zeros((1, 2)))},
                x_bounds={"qdot": bounds},
            )
        ],
        ocp_solver=None,
        _cocofest_fix_controls_to_warmup=False,
        _correct_init_guess_to_fit_bounds=lambda corrected_input: None,
        _sync_acados_state_bounds=lambda: None,
    )
    solution = SimpleNamespace(
        status=2, solver_time_to_optimize=0.1, real_time_to_optimize=0.2
    )
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda _: {
            "residuals": np.array([4e-3, 4e-4, 0.0, 2e-6]),
            "res_stat_all": np.array([1.0, 2e-3, 4e-3]),
            "res_eq_all": np.array([1.0, 5e-6, 4e-4]),
            "res_ineq_all": np.array([0.0, 0.0, 0.0]),
            "res_comp_all": np.array([0.0, 2e-6, 2e-6]),
        },
    )

    def restore_stored_iterate(periodic_nmpc, iterate_index):
        assert iterate_index == 1
        state_guess.init[0, 0] = 7.0
        return {
            "applied": True,
            "source": "stored_iterate",
            "iterate_index": iterate_index,
        }

    monkeypatch.setattr(
        periodic_example,
        "apply_acados_capsule_primal_to_initial_guess",
        restore_stored_iterate,
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda *args: pytest.fail("The degraded final iterate must not be restored."),
    )
    monkeypatch.setattr(
        periodic_example,
        "reset_acados_solver_memory",
        lambda periodic_nmpc: True,
    )

    summary = periodic_example.run_acados_transfer_bound_homotopy(
        nmpc,
        FakeSolver(),
        fractions=(0.5,),
        padding=0.1,
        convergence_tolerance=1e-4,
        stage_iterations=40,
        max_restarts=0,
        echo=False,
        solve_stage=lambda: solution,
    )

    stage = summary["stages"][0]
    assert stage["accepted"] is True
    assert stage["accepted_from_best_stored_iterate"] is True
    assert stage["accepted_as_intermediate_primal_feasible"] is False
    assert stage["best_stored_primal"]["iterate_index"] == 1
    np.testing.assert_allclose(state_guess.init[0, 0], 7.0)


def test_transfer_bound_homotopy_backtracks_from_last_accepted_primal(
    monkeypatch,
):
    class FakeSolver:
        nlp_solver_max_iter = 100

        def set_convergence_tolerance(self, value):
            self.tolerance = value

        def set_maximum_iterations(self, value):
            self.nlp_solver_max_iter = value

    bounds = SimpleNamespace(
        min=np.array([[-0.1, -1.0, -2.1]]),
        max=np.array([[0.1, 1.0, -1.9]]),
    )
    state_guess = SimpleNamespace(init=np.array([[0.0, -4.0, -5.0]]))
    control_guess = SimpleNamespace(init=np.zeros((1, 2)))
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={"qdot": state_guess},
                u_init={"u": control_guess},
                x_bounds={"qdot": bounds},
            )
        ],
        ocp_solver=None,
        _cocofest_fix_controls_to_warmup=False,
        _correct_init_guess_to_fit_bounds=lambda corrected_input: None,
        _sync_acados_state_bounds=lambda: None,
    )

    def solution(name, status, residuals, expected_input, output_value):
        return SimpleNamespace(
            name=name,
            status=status,
            diagnostics=np.asarray(residuals, dtype=float),
            expected_input=expected_input,
            output_value=output_value,
            solver_time_to_optimize=0.1,
            real_time_to_optimize=0.2,
        )

    solutions = [
        solution("lambda_0", 0, [1e-5, 1e-6, 0.0, 1e-6], 0.0, 1.0),
        solution("lambda_0125_retry_0", 2, [1e-2, 1e-3, 0.0, 1e-6], 1.0, 99.0),
        solution("lambda_0125_retry_1", 2, [1e-2, 1e-3, 0.0, 1e-6], 99.0, 99.0),
        solution("lambda_00625", 0, [1e-5, 1e-6, 0.0, 1e-6], 1.0, 2.0),
        solution("lambda_0125", 0, [1e-5, 1e-6, 0.0, 1e-6], 2.0, 3.0),
        solution("lambda_1", 0, [1e-5, 1e-6, 0.0, 1e-6], 3.0, 4.0),
    ]

    def solve_stage():
        candidate = solutions.pop(0)
        np.testing.assert_allclose(state_guess.init[0, 0], candidate.expected_input)
        np.testing.assert_allclose(control_guess.init[0, 0], candidate.expected_input)
        return candidate

    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda candidate: {"residuals": candidate.diagnostics},
    )

    def apply_solution(periodic_nmpc, candidate):
        periodic_nmpc.nlp[0].x_init["qdot"].init[0, 0] = candidate.output_value
        periodic_nmpc.nlp[0].u_init["u"].init[0, 0] = candidate.output_value

    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        apply_solution,
    )
    monkeypatch.setattr(
        periodic_example,
        "reset_acados_solver_memory",
        lambda periodic_nmpc: True,
    )

    summary = periodic_example.run_acados_transfer_bound_homotopy(
        nmpc,
        FakeSolver(),
        fractions=(0.0, 0.125, 1.0),
        padding=0.1,
        convergence_tolerance=1e-4,
        stage_iterations=10,
        max_restarts=1,
        minimum_fraction_step=0.001953125,
        max_refinements=4,
        echo=False,
        solve_stage=solve_stage,
    )

    assert summary["completed"] is True
    assert summary["refinement_count"] == 1
    assert summary["scheduled_fractions"] == [0.0, 0.0625, 0.125, 1.0]
    assert summary["last_accepted_fraction"] == 1.0
    assert summary["termination_reason"] == "physical_fraction_accepted"
    assert summary["stages"][2]["refinement_inserted_fraction"] == 0.0625
    assert summary["stages"][2]["pre_rollback_state_distance"] == 98.0
    assert summary["stages"][2]["pre_rollback_control_distance"] == 98.0
    assert summary["stages"][2]["rollback_state_error"] == 0.0
    assert summary["stages"][2]["rollback_control_error"] == 0.0
    np.testing.assert_allclose(state_guess.init[0, 0], 4.0)
    np.testing.assert_allclose(control_guess.init[0, 0], 4.0)
    assert solutions == []


def test_transfer_sqp_restarts_from_nearly_feasible_iterate(monkeypatch):
    runtime_options = []
    reset_calls = []

    class FakeAcadosSolver:
        def options_set(self, key, value):
            runtime_options.append((key, value))

        def reset(self, reset_qp_solver_mem):
            reset_calls.append(reset_qp_solver_mem)

    class FakeSolver:
        nlp_solver_max_iter = 100

        def set_maximum_iterations(self, value):
            self.nlp_solver_max_iter = value

        def set_only_first_options_has_changed(self, value):
            self.options_changed = value

    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={"q": SimpleNamespace(init=np.zeros((1, 2)))},
                u_init={"u": SimpleNamespace(init=np.zeros((1, 1)))},
            )
        ],
        ocp_solver=SimpleNamespace(ocp_solver=FakeAcadosSolver()),
    )
    solutions = iter(
        [
            SimpleNamespace(
                status=4,
                residuals=np.array([0.7, 2.0, 0.0, 0.0]),
                solver_time_to_optimize=0.1,
                real_time_to_optimize=0.2,
            ),
            SimpleNamespace(
                status=0,
                residuals=np.array([1e-6, 1e-8, 0.0, 1e-7]),
                solver_time_to_optimize=0.3,
                real_time_to_optimize=0.4,
            ),
        ]
    )
    applied_statuses = []
    monkeypatch.setattr(
        periodic_example,
        "snapshot_acados_diagnostics",
        lambda solution: (
            {
                "residuals": solution.residuals,
                "res_stat_all": np.array([0.7, 0.6]),
                "res_eq_all": np.array([2.0, 3e-4]),
                "res_ineq_all": np.zeros(2),
                "res_comp_all": np.array([0.0, 1.4e-3]),
            }
            if solution.status == 4
            else {"residuals": solution.residuals}
        ),
    )
    monkeypatch.setattr(
        periodic_example,
        "apply_solution_directly_to_periodic_nmpc_initial_guess",
        lambda _nmpc, solution: applied_statuses.append(solution.status),
    )

    summary = periodic_example.run_acados_transfer_sqp_restarts(
        nmpc,
        FakeSolver(),
        max_restarts=3,
        stage_iterations=1,
        feasibility_tolerance=1e-2,
        echo=False,
        solve_stage=lambda: next(solutions),
    )

    assert summary["completed"] is True
    assert [item["status"] for item in summary["attempts"]] == [4, 0]
    np.testing.assert_allclose(
        summary["attempts"][0]["reported_residuals"], [0.7, 2.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        summary["attempts"][0]["residuals"], [0.6, 3e-4, 0.0, 1.4e-3]
    )
    assert applied_statuses == [4, 0]
    assert reset_calls == [1]
    assert runtime_options == [
        ("nlp_solver_max_iter", 1),
        ("nlp_solver_max_iter", 1),
        ("nlp_solver_max_iter", 100),
    ]


def test_failed_acados_capsule_primal_is_unscaled_into_initial_guess():
    class Variables(dict):
        def __init__(self, values, shape):
            super().__init__(values)
            self.shape = shape

    class FakeCapsule:
        def get(self, stage, field):
            values = {
                (0, "x"): np.array([1.0, 2.0]),
                (1, "x"): np.array([3.0, 4.0]),
                (0, "u"): np.array([5.0]),
            }
            return values[(stage, field)]

    states = Variables(
        {"q": SimpleNamespace(index=[0]), "qdot": SimpleNamespace(index=[1])},
        shape=2,
    )
    controls = Variables({"u": SimpleNamespace(index=[0])}, shape=1)
    nlp = SimpleNamespace(
        states=states,
        controls=controls,
        parameters=SimpleNamespace(shape=0),
        x_init={
            "q": SimpleNamespace(init=np.zeros((1, 2))),
            "qdot": SimpleNamespace(init=np.zeros((1, 2))),
        },
        u_init={"u": SimpleNamespace(init=np.zeros((1, 1)))},
        x_scaling={
            "q": SimpleNamespace(scaling=np.array([[2.0]])),
            "qdot": SimpleNamespace(scaling=np.array([[4.0]])),
        },
        u_scaling={"u": SimpleNamespace(scaling=np.array([[3.0]]))},
    )
    nmpc = SimpleNamespace(
        nlp=[nlp], ocp_solver=SimpleNamespace(ocp_solver=FakeCapsule())
    )

    summary = periodic_example.apply_acados_capsule_primal_to_initial_guess(nmpc)

    assert summary["applied"] is True
    np.testing.assert_allclose(nlp.x_init["q"].init, [[2.0, 6.0]])
    np.testing.assert_allclose(nlp.x_init["qdot"].init, [[8.0, 16.0]])
    np.testing.assert_allclose(nlp.u_init["u"].init, [[15.0]])


def test_transferred_guess_is_projected_after_bounds_move():
    state_guess = SimpleNamespace(init=np.array([[2.0, -3.0]]))
    control_guess = SimpleNamespace(init=np.array([[4.0]]))
    calls = []

    def correct(corrected_input):
        calls.append(corrected_input)
        if corrected_input == "states":
            state_guess.init[:] = np.clip(state_guess.init, -1.0, 1.0)
        else:
            control_guess.init[:] = np.clip(control_guess.init, 0.0, 2.0)

    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={"state": state_guess},
                u_init={"control": control_guess},
            )
        ],
        _correct_init_guess_to_fit_bounds=correct,
        _sync_acados_state_bounds=lambda: calls.append("sync"),
    )

    summary = periodic_example.project_transferred_initial_guess_to_bounds(nmpc)

    assert calls == ["states", "controls", "sync"]
    assert summary["state_max_change"] == 2.0
    assert summary["control_max_change"] == 2.0
    np.testing.assert_allclose(state_guess.init, [[1.0, -1.0]])
    np.testing.assert_allclose(control_guess.init, [[2.0]])


def test_initial_guess_audit_is_solver_independent_and_deterministic():
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={"q": SimpleNamespace(init=np.array([[1.0, 2.0]]))},
                u_init={"tau": SimpleNamespace(init=np.array([[3.0]]))},
            )
        ]
    )

    first = audit_initial_guess(nmpc)
    second = audit_initial_guess(nmpc)

    assert first["signature"] == second["signature"]
    assert first["finite"] is True
    assert first["state_shapes"] == {"q": (1, 2)}
    assert first["control_shapes"] == {"tau": (1, 1)}
    nmpc.nlp[0].x_init["q"].init[0, 1] += 1.0
    assert audit_initial_guess(nmpc)["signature"] != first["signature"]


def test_detailed_initial_guess_diagnostics_are_solver_independent(monkeypatch):
    state_bounds = SimpleNamespace(
        min=np.array([[-10.0, -10.0, -10.0]]),
        max=np.array([[10.0, 10.0, 10.0]]),
    )
    control_bounds = SimpleNamespace(min=np.array([[0.0]]), max=np.array([[1.0]]))
    nmpc = SimpleNamespace(
        cycle_duration=1.0,
        cycle_len=2,
        nlp=[
            SimpleNamespace(
                x_init={
                    "q": SimpleNamespace(init=np.array([[0.0, 1.0, 2.0]])),
                    "qdot": SimpleNamespace(init=np.array([[1.0, 1.0, 1.0]])),
                },
                u_init={"tau": SimpleNamespace(init=np.array([[0.5, 2.0]]))},
                x_bounds={"q": state_bounds, "qdot": state_bounds},
                u_bounds={"tau": control_bounds},
            )
        ],
    )
    fes = {
        "absolute_by_state": {"Cn": 0.25},
        "absolute_by_muscle": {"Biceps": 0.25},
        "scaled_by_state": {"Cn": 0.5},
    }
    full = {
        "absolute_by_block": {"q": 0.75},
        "scaled_by_block": {"q": 0.125},
        "top_keys": {"q": 0.75},
        "q_by_dof": {"wheel": 0.75},
        "qdot_by_dof": {},
        "worst_qdot_nodes": [],
    }
    monkeypatch.setattr(periodic_example, "_periodic_fes_rollout_defect_details", lambda _: fes)
    monkeypatch.setattr(periodic_example, "_full_dynamics_rollout_defect_details", lambda _: full)

    diagnostics = periodic_example.collect_initial_guess_diagnostics(nmpc)

    assert diagnostics["state_node_stride"] == 1
    assert diagnostics["q_kinematic"] == {
        "method": "forward_euler_endpoint_consistency",
        "maximum": 1.0,
        "per_dof": [1.0],
    }
    assert diagnostics["state_bound_violations"] == {}
    assert diagnostics["control_bound_violations"] == {"tau": 1.0}
    assert diagnostics["periodic_fes_rollout"] == fes
    assert diagnostics["full_dynamics_rk4_rollout"] == full


def test_initial_guess_diagnostics_locate_terminal_state_bound_violation(monkeypatch):
    bounds = SimpleNamespace(
        min=np.array([[-1.0, -2.0, -3.0]]),
        max=np.array([[1.0, 2.0, 3.0]]),
    )
    nmpc = SimpleNamespace(
        cycle_duration=1.0,
        cycle_len=2,
        nlp=[
            SimpleNamespace(
                x_init={
                    "q": SimpleNamespace(init=np.array([[0.0, 0.5, 4.25]])),
                    "qdot": SimpleNamespace(init=np.zeros((1, 3))),
                },
                u_init={"tau": SimpleNamespace(init=np.zeros((1, 2)))},
                x_bounds={"q": bounds, "qdot": bounds},
                u_bounds={
                    "tau": SimpleNamespace(
                        min=np.array([[-1.0]]), max=np.array([[1.0]])
                    )
                },
            )
        ],
    )
    monkeypatch.setattr(
        periodic_example, "_periodic_fes_rollout_defect_details", lambda _: {}
    )
    monkeypatch.setattr(
        periodic_example, "_full_dynamics_rollout_defect_details", lambda _: {}
    )

    diagnostics = periodic_example.collect_initial_guess_diagnostics(nmpc)

    assert diagnostics["state_bound_violations"] == {"q": 1.25}
    assert diagnostics["state_bound_violation_details"] == {
        "q": {
            "component": 0,
            "node": 2,
            "node_role": "terminal",
            "is_shooting_node": True,
            "value": 4.25,
            "lower": -3.0,
            "upper": 3.0,
            "violation": 1.25,
        }
    }


def test_collocation_initial_guess_uses_only_shooting_endpoints():
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                x_init={"q": SimpleNamespace(init=np.zeros((1, 9)))},
                u_init={"tau": SimpleNamespace(init=np.zeros((1, 2)))},
            )
        ]
    )

    indices, stride = periodic_example._initial_guess_shooting_node_indices(nmpc)

    np.testing.assert_array_equal(indices, [0, 4, 8])
    assert stride == 4


def test_collocation_endpoint_correction_preserves_internal_stage_offsets():
    original = np.array(
        [[0.0, 10.0, 20.0, 30.0, 100.0, 110.0, 120.0, 130.0, 200.0]]
    )
    projected_endpoints = np.array([[1.0, 102.0, 203.0]])

    updated = periodic_example._lift_shooting_endpoint_update_to_state_columns(
        original,
        projected_endpoints,
        np.array([0, 4, 8]),
    )

    np.testing.assert_allclose(updated[:, [0, 4, 8]], projected_endpoints)
    np.testing.assert_allclose(
        updated[0, 1:4] - original[0, 1:4], [4 / 3, 5 / 3, 2]
    )
    np.testing.assert_allclose(
        updated[0, 5:8] - original[0, 5:8], [7 / 3, 8 / 3, 3]
    )


def test_comparison_forwards_solver_neutral_seed_diagnostics(monkeypatch):
    captured = {}

    def fake_run(solver_name, args, **_):
        captured[solver_name] = args
        return {}

    monkeypatch.setattr(comparison_example, "_run_benchmark_case", fake_run)
    monkeypatch.setattr(comparison_example, "print_solver_overview", lambda _: None)

    comparison_example.main(
        solvers=("ipopt", "madnlp"),
        n_windows=1,
        initial_guess_diagnostics=True,
        exact_initial_nlp_audit=True,
    )

    assert captured["ipopt"].initial_guess_diagnostics is True
    assert captured["madnlp"].initial_guess_diagnostics is True
    assert captured["ipopt"].exact_initial_nlp_audit is True
    assert captured["madnlp"].exact_initial_nlp_audit is True
    assert captured["madnlp"].acados_diagnostics is False


def test_generic_initial_guess_copy_reports_incompatible_grids():
    source = {"q": SimpleNamespace(init=np.ones((1, 3)))}
    target = {"q": SimpleNamespace(init=np.zeros((1, 2)))}

    with np.testing.assert_raises_regex(ValueError, "shape"):
        copy_container_values(source, target, "init")


def test_shared_transfer_rollout_cli_is_available_to_ipopt():
    args = periodic_example.build_argument_parser().parse_args(
        [
            "--solver",
            "ipopt",
            "--exact-initial-nlp-audit",
            "--transfer-full-dynamics-rollout",
            "--transfer-phase-one",
            "--acados-transfer-phase-one-mode",
            "mechanical",
            "--acados-transfer-phase-one-lookback-nodes",
            "15",
            "--transfer-rollout-substeps",
            "7",
            "--full-dynamics-phase-one-max-state-change",
            "20",
            "--full-dynamics-phase-one-max-q-change",
            "1",
            "--full-dynamics-phase-one-max-qdot-change",
            "2",
            "--full-dynamics-phase-one-max-fes-change",
            "3",
        ]
    )
    comparison_args = comparison_example.build_cli().parse_args(
        [
            "--shared-transfer-full-dynamics-rollout",
            "--exact-initial-nlp-audit",
            "--compact-rho-output",
            "--shared-transfer-phase-one",
            "--acados-transfer-phase-one",
            "--acados-transfer-phase-one-mode",
            "mechanical",
            "--acados-transfer-phase-one-lookback-nodes",
            "15",
            "--acados-cyclical-transfer-mode",
            "repeat",
            "--acados-transfer-phase-one-proximity-weight",
            "0",
            "--acados-transfer-phase-one-defect-weight",
            "1",
            "--acados-transfer-phase-one-substeps",
            "10",
            "--acados-transfer-phase-one-max-state-change",
            "20",
            "--acados-transfer-phase-one-max-q-change",
            "1",
            "--acados-transfer-phase-one-max-qdot-change",
            "2",
            "--acados-transfer-phase-one-max-fes-change",
            "3",
            "--acados-transfer-bound-homotopy",
            "--acados-transfer-bound-homotopy-fractions",
            "0,1",
            "--acados-transfer-bound-homotopy-padding",
            "0.05",
            "--acados-transfer-bound-homotopy-iterations",
            "20",
            "--acados-transfer-bound-homotopy-tolerance",
            "1e-4",
            "--acados-transfer-bound-homotopy-solver-tolerance",
            "1e-4",
            "--acados-transfer-bound-homotopy-min-fraction-step",
            "0.001953125",
            "--acados-transfer-bound-homotopy-max-refinements",
            "16",
            "--shared-initial-phase-one",
            "--shared-transfer-rollout-substeps",
            "7",
            "--acados-transfer-select-projected-candidate",
            "--acados-transfer-selector-max-q-bound-violation-rad",
            "0.8",
            "--acados-transfer-selector-max-qdot-bound-violation-rad-s",
            "10",
            "--acados-transfer-selector-max-other-scaled-bound-violation",
            "0.9",
            "--acados-transfer-selector-max-scaled-q-defect",
            "0.08",
            "--acados-transfer-selector-max-scaled-qdot-defect",
            "0.09",
            "--acados-transfer-selector-improvement-ratio",
            "0.9",
            "--shared-transfer-ding-force-compensation",
            "--shared-transfer-ding-force-compensation-substeps",
            "6",
            "--acados-transfer-ding-force-compensation",
            "--acados-proximal-control-weights",
            "1e6,1e5",
            "--acados-proximal-control-each-window",
            "--acados-proximal-control-try-next-weight-on-failure",
            "--acados-transfer-sqp-restarts",
            "2",
            "--acados-transfer-sqp-restart-iterations",
            "1",
            "--acados-transfer-sqp-restart-feasibility-tolerance",
            "0.01",
            "--acados-terminal-wheel-q-homotopy-slacks",
            "0.2,0.1,0.02",
            "--acados-terminal-wheel-q-homotopy-each-window",
            "--acados-control-homotopy-release-final-radius",
            "--acados-control-homotopy-window-growth",
            "10",
            "--acados-control-homotopy-window-max-radius",
            "1e-4",
            "--acados-newton-iter",
            "3",
        ]
    )

    assert args.acados_transfer_full_dynamics_rollout is True
    assert args.exact_initial_nlp_audit is True
    assert args.acados_transfer_phase_one is True
    assert args.acados_transfer_phase_one_mode == "mechanical"
    assert args.acados_transfer_phase_one_lookback_nodes == 15
    assert args.acados_transfer_rollout_substeps == 7
    assert args.full_dynamics_phase_one_max_state_change == 20
    assert args.full_dynamics_phase_one_max_q_change == 1
    assert args.full_dynamics_phase_one_max_qdot_change == 2
    assert args.full_dynamics_phase_one_max_fes_change == 3
    assert comparison_args.shared_transfer_full_dynamics_rollout is True
    assert comparison_args.exact_initial_nlp_audit is True
    assert comparison_args.compact_rho_output is True
    assert comparison_args.shared_transfer_phase_one is True
    assert comparison_args.acados_transfer_phase_one is True
    assert comparison_args.acados_transfer_phase_one_mode == "mechanical"
    assert comparison_args.acados_transfer_phase_one_lookback_nodes == 15
    assert comparison_args.acados_cyclical_transfer_mode == "repeat"
    assert comparison_args.acados_transfer_phase_one_proximity_weight == 0
    assert comparison_args.acados_transfer_phase_one_defect_weight == 1
    assert comparison_args.acados_transfer_phase_one_substeps == 10
    assert comparison_args.acados_transfer_phase_one_max_state_change == 20
    assert comparison_args.acados_transfer_phase_one_max_q_change == 1
    assert comparison_args.acados_transfer_phase_one_max_qdot_change == 2
    assert comparison_args.acados_transfer_phase_one_max_fes_change == 3
    assert comparison_args.acados_transfer_bound_homotopy is True
    assert comparison_args.acados_transfer_bound_homotopy_fractions == (
        0.0,
        1.0,
    )
    assert comparison_args.acados_transfer_bound_homotopy_padding == 0.05
    assert comparison_args.acados_transfer_bound_homotopy_iterations == 20
    assert comparison_args.acados_transfer_bound_homotopy_tolerance == 1e-4
    assert comparison_args.acados_transfer_bound_homotopy_solver_tolerance == 1e-4
    assert (
        comparison_args.acados_transfer_bound_homotopy_min_fraction_step == 0.001953125
    )
    assert comparison_args.acados_transfer_bound_homotopy_max_refinements == 16
    assert comparison_args.shared_initial_phase_one is True
    assert comparison_args.shared_transfer_rollout_substeps == 7
    assert comparison_args.acados_transfer_select_projected_candidate is True
    assert comparison_args.acados_transfer_selector_max_q_bound_violation_rad == 0.8
    assert (
        comparison_args.acados_transfer_selector_max_qdot_bound_violation_rad_s == 10
    )
    assert (
        comparison_args.acados_transfer_selector_max_other_scaled_bound_violation
        == 0.9
    )
    assert comparison_args.acados_transfer_selector_max_scaled_q_defect == 0.08
    assert comparison_args.acados_transfer_selector_max_scaled_qdot_defect == 0.09
    assert comparison_args.acados_transfer_selector_improvement_ratio == 0.9
    assert comparison_args.shared_transfer_ding_force_compensation is True
    assert comparison_args.shared_transfer_ding_force_compensation_substeps == 6
    assert comparison_args.acados_transfer_ding_force_compensation is True
    assert comparison_args.acados_proximal_control_weights == (1e6, 1e5)
    assert comparison_args.acados_proximal_control_each_window is True
    assert comparison_args.acados_proximal_control_try_next_weight_on_failure is True
    assert comparison_args.acados_transfer_sqp_restarts == 2
    assert comparison_args.acados_transfer_sqp_restart_iterations == 1
    assert comparison_args.acados_transfer_sqp_restart_feasibility_tolerance == 0.01
    assert comparison_args.acados_terminal_wheel_q_homotopy_slacks == (
        0.2,
        0.1,
        0.02,
    )
    assert comparison_args.acados_terminal_wheel_q_homotopy_each_window is True
    assert comparison_args.acados_control_homotopy_keep_final_radius is False
    assert comparison_args.acados_control_homotopy_window_growth == 10
    assert comparison_args.acados_control_homotopy_window_max_radius == 1e-4
    assert comparison_args.acados_newton_iter == 3
    assert (
        comparison_example.IPOPT_PROFILE_DEFAULTS["acados_like"]["model_formulation"]
        == "periodic_node"
    )
    assert (
        comparison_example.IPOPT_PROFILE_DEFAULTS["acados_like"][
            "disable_periodic_fes_warmup_projection"
        ]
        is False
    )


def test_transfer_phase_one_only_runs_from_a_completed_rho_callback():
    seed_solution = object()

    assert (
        periodic_example._should_apply_transfer_phase_one(
            0,
            continue_solving=True,
            previous_solution=seed_solution,
            enabled=True,
        )
        is False
    )
    assert (
        periodic_example._should_apply_transfer_phase_one(
            1,
            continue_solving=True,
            previous_solution=seed_solution,
            enabled=True,
        )
        is True
    )
    assert (
        periodic_example._should_apply_transfer_phase_one(
            1,
            continue_solving=False,
            previous_solution=seed_solution,
            enabled=True,
        )
        is False
    )


def test_periodic_collocation_ipopt_profile_is_available():
    args = comparison_example.build_cli().parse_args(
        ["--ipopt-profile", "periodic-collocation"]
    )
    defaults = comparison_example.IPOPT_PROFILE_DEFAULTS["periodic_collocation"]

    assert comparison_example._normalize_ipopt_profile(args.ipopt_profile) == (
        "periodic_collocation"
    )
    assert defaults["model_formulation"] == "periodic_node"
    assert defaults["torque_application"] == "constant"
    assert defaults["ode_solver"] == "collocation"
    assert defaults["use_sx"] is True

    periodic_args = periodic_example.build_argument_parser().parse_args(
        ["--periodic-ipopt-refinement-ode-solver", "collocation"]
    )
    comparison_args = comparison_example.build_cli().parse_args(
        ["--periodic-ipopt-refinement-ode-solver", "collocation"]
    )
    assert periodic_args.periodic_ipopt_refinement_ode_solver == "collocation"
    assert comparison_args.periodic_ipopt_refinement_ode_solver == "collocation"


def test_scientific_radau5_profile_is_fixed_and_shared_by_nlp_solvers(monkeypatch):
    captured = {}

    def fake_run(solver_name, args, **_):
        captured[solver_name] = args
        return {}

    monkeypatch.setattr(comparison_example, "_run_benchmark_case", fake_run)
    monkeypatch.setattr(comparison_example, "print_solver_overview", lambda _: None)

    cli_args = comparison_example.build_cli().parse_args(
        ["--benchmark-profile", "scientific-radau5"]
    )
    assert cli_args.ipopt_profile == "scientific-radau5"

    comparison_example.main(
        solvers=("ipopt", "madnlp"),
        n_windows=1,
        ipopt_profile="scientific-radau5",
    )

    for solver_name in ("ipopt", "madnlp"):
        args = captured[solver_name]
        assert args.benchmark_profile == "scientific-radau5"
        assert args.transcription_profile == "scientific-radau5"
        assert args.profile_integrity is True
        assert args.scientific_status == "candidate"
        assert args.model_formulation == "periodic_node"
        assert args.ode_solver == "collocation"
        assert args.collocation_degree == 5
        assert args.collocation_method == "radau"
        assert args.use_sx is True
        assert args.enforce_start_constraints is True

    diagnostic_hashes = set()
    for profile, degree, status in (
        ("scientific-radau4", 4, "diagnostic"),
        ("scientific-radau6", 6, "diagnostic"),
    ):
        captured.clear()
        comparison_example.main(
            solvers=("ipopt", "madnlp"),
            n_windows=1,
            ipopt_profile=profile,
        )
        for solver_name in ("ipopt", "madnlp"):
            args = captured[solver_name]
            assert args.collocation_degree == degree
            assert args.profile_integrity is True
            assert args.scientific_status == status
            assert args.enforce_start_constraints is True
            diagnostic_hashes.add(args.profile_hash)
    assert len(diagnostic_hashes) == 2

    with pytest.raises(ValueError, match="fixed scientific contract"):
        comparison_example.main(
            solvers=("ipopt",),
            n_windows=1,
            ipopt_profile="scientific-radau5",
            ipopt_collocation_degree=3,
        )


def test_comparison_cli_accepts_acados_best_iterate_retry_options():
    args = comparison_example.build_cli().parse_args(
        [
            "--acados-store-iterates",
            "--acados-maxiter-retries",
            "1",
            "--acados-maxiter-retry-iterations",
            "20",
            "--acados-maxiter-retry-feasibility-tolerance",
            "0.0025",
        ]
    )

    assert args.acados_store_iterates is True
    assert args.acados_maxiter_retries == 1
    assert args.acados_maxiter_retry_iterations == 20
    assert args.acados_maxiter_retry_feasibility_tolerance == pytest.approx(0.0025)


def test_comparison_main_forwards_acados_best_iterate_retry_options(monkeypatch):
    captured = {}

    def fake_run(solver_name, args, **_):
        captured[solver_name] = args
        return {}

    monkeypatch.setattr(comparison_example, "_run_benchmark_case", fake_run)
    monkeypatch.setattr(comparison_example, "print_solver_overview", lambda _: None)

    comparison_example.main(
        solvers=("acados",),
        n_windows=1,
        acados_store_iterates=True,
        acados_maxiter_retries=1,
        acados_maxiter_retry_iterations=20,
        acados_maxiter_retry_feasibility_tolerance=0.0025,
    )

    args = captured["acados"]
    assert args.acados_store_iterates is True
    assert args.acados_maxiter_retries == 1
    assert args.acados_maxiter_retry_iterations == 20
    assert args.acados_maxiter_retry_feasibility_tolerance == pytest.approx(0.0025)


def test_refinement_initial_guess_expands_shooting_nodes_for_collocation():
    source = {
        "q": SimpleNamespace(init=np.array([[0.0, 1.0, 2.0]])),
    }
    target = {
        "q": SimpleNamespace(init=np.zeros((1, 9))),
    }

    _copy_refinement_initial_guesses(source, target, has_terminal_node=True)

    np.testing.assert_allclose(target["q"].init, np.linspace(0.0, 2.0, 9)[None, :])


def test_fes_nmpc_reports_incomplete_export_as_solver_failure():
    nmpc = object.__new__(FesNmpcMsk)
    nmpc.n_cycles_simultaneous = 1

    def fail_while_assembling_window(*_args, **_kwargs):
        raise IndexError("index 31 is out of bounds for axis 1 with size 31")

    nmpc.solve = fail_while_assembling_window
    with np.testing.assert_raises_regex(RuntimeError, "exported window"):
        nmpc.solve_fes_nmpc(
            update_functions=None,
            solver=SimpleNamespace(),
            total_cycles=1,
            external_force=None,
            cycle_solutions=SimpleNamespace(),
        )


def test_shared_initial_guess_comparison_detects_exact_and_biased_seeds():
    shared = {
        "initial_guess_state_traces": {"q": np.array([[1.0, 2.0]])},
        "initial_guess_control_traces": {"u": np.array([[3.0]])},
        "initial_guess_audits": [{"signature": "same"}],
    }
    exact = comparison_example._shared_initial_guess_comparison(shared, shared)
    changed = {
        **shared,
        "initial_guess_control_traces": {"u": np.array([[4.0]])},
        "initial_guess_audits": [{"signature": "different"}],
    }
    biased = comparison_example._shared_initial_guess_comparison(shared, changed)

    assert exact["comparable"] is True
    assert exact["exact"] is True
    assert exact["max_abs_error"] == 0.0
    assert biased["comparable"] is True
    assert biased["exact"] is False
    assert biased["max_abs_error"] == 1.0


def test_failed_first_window_keeps_initial_guess_for_backend_comparison():
    nmpc = SimpleNamespace(
        nlp=[
            SimpleNamespace(
                controls={"u": object()},
                u_bounds={
                    "u": SimpleNamespace(min=np.array([[0.0]]), max=np.array([[1.0]]))
                },
            )
        ]
    )
    states = {"q": np.array([[0.0, 0.0], [0.0, 0.0], [0.0, -1.0]])}
    controls = {"u": np.array([[0.5]])}

    summary = periodic_example.build_failed_solve_summary(
        nmpc,
        SimpleNamespace(n_windows=2),
        RuntimeError("no iterate"),
        states,
        controls,
    )

    assert summary["success"] is False
    assert summary["attempted_windows"] == 0
    assert summary["diagnostics"]["issues"] == ["no_solver_solution"]
    assert summary["initial_guess_state_traces"] is states
    assert summary["initial_guess_control_traces"] is controls
    assert summary["final_wheel_angle"] == -1.0


def test_full_dynamics_rhs_passes_numerical_timeseries_as_data():
    recorded = {}

    def dynamics(
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
    ):
        recorded["parameters"] = np.asarray(parameters)
        recorded["algebraic_states"] = np.asarray(algebraic_states)
        recorded["numerical_timeseries"] = np.asarray(numerical_timeseries)
        return states

    nlp = SimpleNamespace(dynamics_func=dynamics)
    numerical_timeseries = np.array([1.0, 2.0, 3.0])

    result = periodic_example._full_dynamics_rhs(
        nlp,
        time=0.0,
        dt=0.1,
        state=np.array([4.0, 5.0]),
        control=np.array([6.0]),
        numerical_timeseries=numerical_timeseries,
    )

    np.testing.assert_allclose(result, [4.0, 5.0])
    assert recorded["parameters"].size == 0
    assert recorded["algebraic_states"].size == 0
    np.testing.assert_allclose(recorded["numerical_timeseries"], numerical_timeseries)
