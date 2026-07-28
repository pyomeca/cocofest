from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.dynamics.reduced_cycling import (
    PeriodicFourierSeries,
    ReducedCyclingDynamics,
    ReducedCyclingKinematics,
    build_reduced_cycling_dynamics,
    validate_reduced_cycling_dynamics,
)
from cocofest.models.reduced_cycling_model import ReducedFesCyclingModel


def test_periodic_fourier_series_values_and_derivatives():
    phase = np.linspace(0.0, 2.0 * np.pi, 101)
    values = np.vstack(
        (
            1.2 + 0.4 * np.cos(phase) - 0.3 * np.sin(2.0 * phase),
            -0.2 + 0.7 * np.sin(phase),
        )
    )
    series = PeriodicFourierSeries.fit(phase, values, order=2)
    test_phase = 0.37

    np.testing.assert_allclose(
        series.evaluate(test_phase),
        np.array(
            [
                1.2
                + 0.4 * np.cos(test_phase)
                - 0.3 * np.sin(2.0 * test_phase),
                -0.2 + 0.7 * np.sin(test_phase),
            ]
        ),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        series.evaluate(test_phase, derivative=1),
        np.array(
            [
                -0.4 * np.sin(test_phase)
                - 0.6 * np.cos(2.0 * test_phase),
                0.7 * np.cos(test_phase),
            ]
        ),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        series.evaluate(test_phase, derivative=2),
        np.array(
            [
                -0.4 * np.cos(test_phase)
                + 1.2 * np.sin(2.0 * test_phase),
                -0.7 * np.sin(test_phase),
            ]
        ),
        atol=1e-12,
    )


def test_reduced_kinematics_retains_variable_crank_velocity():
    theta = np.linspace(0.0, -2.0 * np.pi, 101)
    q = np.vstack(
        (
            0.8 + 0.2 * np.cos(theta),
            1.4 + 0.3 * np.sin(theta),
            theta + 0.1 * np.sin(theta),
        )
    )
    kinematics = ReducedCyclingKinematics.fit(theta, q, order=2)

    theta_value = -0.7
    omega_slow = -3.0
    omega_fast = -8.0
    tangent = kinematics.tangent(theta_value)
    _, qdot_slow, _ = kinematics.generalized_kinematics(
        theta_value, omega_slow, 0.0
    )
    _, qdot_fast, _ = kinematics.generalized_kinematics(
        theta_value, omega_fast, 0.0
    )

    np.testing.assert_allclose(qdot_slow, tangent * omega_slow)
    np.testing.assert_allclose(qdot_fast, tangent * omega_fast)
    assert not np.allclose(qdot_slow, qdot_fast)
    np.testing.assert_allclose(
        kinematics.q(theta_value - 2.0 * np.pi)
        - kinematics.q(theta_value),
        np.array([0.0, 0.0, -2.0 * np.pi]),
        atol=1e-10,
    )

    theta_nodes = np.linspace(0.0, -4.0 * np.pi, 121)
    omega_nodes = -5.0 - 0.8 * np.sin(theta_nodes)
    full_q = kinematics.q(theta_nodes)
    full_qdot = kinematics.tangent(theta_nodes) * omega_nodes
    projected_theta, projected_omega, audit = (
        kinematics.project_generalized_trajectory(full_q, full_qdot)
    )
    np.testing.assert_allclose(projected_theta, theta_nodes[np.newaxis, :], atol=1e-10)
    np.testing.assert_allclose(projected_omega, omega_nodes[np.newaxis, :], atol=1e-10)
    assert audit["maximum_configuration_projection_error_rad"] < 1e-10


def test_wu_reduced_dynamics_matches_constrained_forward_dynamics(tmp_path):
    pytest.importorskip("biorbd")
    model_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "msk_models"
        / "Wu"
        / "Modified_Wu_Shoulder_Model_Cycling.bioMod"
    )
    reduced, build_audit = build_reduced_cycling_dynamics(
        model_path,
        sample_count=61,
        kinematic_order=12,
        dynamics_order=12,
    )
    assert build_audit["maximum_contact_phase_residual"] < 1e-10
    assert build_audit["maximum_cycle_closure_error"] < 1e-9
    assert build_audit["maximum_kinematic_fit_error_rad"] < 1e-7

    profile_path = reduced.save(tmp_path / "reduced-cycling.npz")
    loaded = ReducedCyclingDynamics.load(profile_path)
    theta = -1.23
    forces = np.array([50.0, 40.0, 150.0, 250.0])
    np.testing.assert_allclose(
        loaded.acceleration(
            theta, -6.0, forces, external_crank_torque=-0.2
        ),
        reduced.acceleration(
            theta, -6.0, forces, external_crank_torque=-0.2
        ),
        atol=1e-12,
    )

    validation = validate_reduced_cycling_dynamics(
        model_path,
        reduced,
        sample_count=30,
        seed=123,
    )
    assert validation["maximum_contact_position_error_m"] < 1e-7
    assert (
        validation["maximum_crank_acceleration_absolute_error_rad_s2"]
        < 0.02
    )
    assert validation["p95_crank_acceleration_relative_error"] < 5e-5
    assert validation["maximum_active_force_length_coefficient_error"] < 1e-7
    assert validation["maximum_force_velocity_coefficient_error"] < 1e-7
    assert validation["maximum_passive_force_coefficient_error"] < 1e-7


def test_stale_reduced_profile_is_rejected(tmp_path):
    profile_path = tmp_path / "stale-profile.npz"
    np.savez(profile_path, schema_version=np.array([1], dtype=int))

    with pytest.raises(ValueError, match="schema 1 is stale"):
        ReducedCyclingDynamics.load(profile_path)


def test_reduced_dynamics_exposes_smooth_casadi_expression():
    casadi = pytest.importorskip("casadi")
    theta = np.linspace(0.0, -2.0 * np.pi, 51)
    q = np.vstack(
        (
            0.8 + 0.2 * np.cos(theta),
            1.4 + 0.3 * np.sin(theta),
            theta,
        )
    )
    kinematics = ReducedCyclingKinematics.fit(theta, q, order=2)
    coefficient_phase = np.linspace(0.0, 2.0 * np.pi, 51)
    coefficients = PeriodicFourierSeries.fit(
        coefficient_phase,
        np.vstack(
            (
                np.full(51, 0.01),
                np.zeros(51),
                np.zeros(51),
                np.full(51, 0.02),
                np.full(51, -0.01),
                np.ones(51),
            )
        ),
        order=2,
    )
    reduced = ReducedCyclingDynamics(
        kinematics=kinematics,
        coefficients=coefficients,
        muscle_names=("flexor", "extensor"),
        crank_torque_dof_index=2,
    )
    theta_symbol = casadi.MX.sym("theta")
    omega_symbol = casadi.MX.sym("omega")
    forces_symbol = casadi.MX.sym("forces", 2)
    acceleration = reduced.casadi_acceleration(
        theta_symbol,
        omega_symbol,
        forces_symbol,
        -0.2,
    )
    function = casadi.Function(
        "reduced_acceleration",
        [theta_symbol, omega_symbol, forces_symbol],
        [acceleration],
    )

    expected = reduced.acceleration(
        -0.4,
        -5.0,
        [100.0, 50.0],
        external_crank_torque=-0.2,
    )
    np.testing.assert_allclose(
        float(function(-0.4, -5.0, [100.0, 50.0])),
        expected,
        atol=1e-12,
    )


def test_reduced_fes_model_rejects_more_than_twenty_ding_states():
    names = ("Delt_ant", "Delt_post", "Biceps", "Triceps")
    muscles = [
        SimpleNamespace(
            muscle_name=name,
            nb_state=6 if index == 0 else 5,
        )
        for index, name in enumerate(names)
    ]

    with pytest.raises(ValueError, match="exactly five Ding states per muscle"):
        ReducedFesCyclingModel(
            reduced_dynamics=SimpleNamespace(
                muscle_names=names,
                muscle_geometry=None,
            ),
            muscles_model=muscles,
            activate_force_length_relationship=False,
            activate_force_velocity_relationship=False,
            activate_passive_force_relationship=False,
        )


def test_reduced_fes_ocp_has_twenty_ding_states_and_theta_omega():
    pytest.importorskip("biorbd")
    from bioptim import OdeSolver
    from examples.fes_multibody.cycling.cycling_pulse_width_mhe import (
        prepare_nmpc,
        set_fes_model,
    )

    model_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "msk_models"
        / "Wu"
        / "Modified_Wu_Shoulder_Model_Cycling.bioMod"
    )
    reduced, _ = build_reduced_cycling_dynamics(
        model_path,
        sample_count=61,
        kinematic_order=12,
        dynamics_order=12,
    )
    stim_time = list(np.linspace(0.0, 1.0, 6, endpoint=False))
    full_model = set_fes_model(
        str(model_path),
        stim_time,
        periodic_node_forcing=True,
    )
    with pytest.warns(RuntimeWarning, match="projected onto the reduced"):
        nmpc = prepare_nmpc(
            full_model,
            {
                "cycle_duration": 1.0,
                "cycle_len": 6,
                "n_cycles_to_advance": 1,
                "n_cycles_simultaneous": 1,
                "ode_solver": OdeSolver.RK4(n_integration_steps=1),
                "use_sx": False,
                "n_threads": 1,
            },
            {
                "turn_number": 1,
                "pedal_config": {
                    "x_center": 0.35,
                    "y_center": 0.0,
                    "radius": 0.1,
                },
                "constant_crank_torque": -0.2,
                "enforce_start_constraints": False,
            },
            {
                "minimize_force": False,
                "minimize_fatigue": True,
                "minimize_control": False,
                "cost_fun_weight": (0.0, 1.0, 0.0),
                "objective_shape": "quadratic",
                "init_guess_file_path": None,
                "state_scaling": "full",
                "pulse_width_scaling": 0.0025,
                "mechanical_formulation": "reduced",
                "reduced_cycling_dynamics": reduced,
                "terminal_wheel_regularization_weight": 0.01,
            },
        )

    state_keys = list(nmpc.nlp[0].states.keys())
    assert state_keys[-2:] == ["theta", "omega"]
    assert len([key for key in state_keys if key not in ("theta", "omega")]) == 20
    assert nmpc.nlp[0].states.shape == 22
    assert nmpc.nlp[0].controls.shape == 4
    assert nmpc.nlp[0].dynamics_func.numel_out() == 22
