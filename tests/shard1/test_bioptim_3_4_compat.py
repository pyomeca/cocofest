from types import SimpleNamespace

import numpy as np
from casadi import DM
from bioptim import OdeSolver

from cocofest import ModelMaker
from cocofest.models.ding2003.ding2003 import DingModelFrequency
from cocofest.models.ding2007.ding2007 import DingModelPulseWidthFrequency
from cocofest.models.ding2007.ding2007_with_fatigue import DingModelPulseWidthFrequencyWithFatigue
from cocofest.models.hmed2018.hmed2018 import DingModelPulseIntensityFrequency
from examples.getting_started.optimization.pulse_width_optimization_mhe import prepare_mhe


def _build_fake_nlp(model, states_dot, ode_solver, dt=0.2):
    return SimpleNamespace(
        model=model,
        dt=dt,
        dynamics_type=SimpleNamespace(ode_solver=ode_solver),
        states_dot=SimpleNamespace(scaled=SimpleNamespace(cx=DM(states_dot))),
    )


def test_model_maker_instantiates_ding_models_on_bioptim_3_4():
    ding2003 = ModelMaker.create_model("ding2003_with_fatigue", stim_time=[0, 0.1], sum_stim_truncation=10)
    ding2007 = ModelMaker.create_model("ding2007_with_fatigue", stim_time=[0, 0.1], sum_stim_truncation=10)

    assert ding2003._with_fatigue is True
    assert ding2007._with_fatigue is True
    assert ding2003.name_dof == ["Cn", "F", "A", "Tau1", "Km"]
    assert ding2007.name_dof == ["Cn", "F", "A", "Tau1", "Km"]


def test_ding2007_collocation_dynamics_returns_expected_defects():
    model = DingModelPulseWidthFrequency(stim_time=[0, 0.1], sum_stim_truncation=10)
    states = DM([5, 100])
    controls = DM([0.0002])
    states_dot = DM([1.2, -3.4])
    nlp = _build_fake_nlp(model=model, states_dot=states_dot, ode_solver=OdeSolver.COLLOCATION())

    dynamics = model.dynamics(
        time=0.11,
        states=states,
        controls=controls,
        parameters=DM(),
        algebraic_states=DM(),
        numerical_timeseries=np.array([0, 0.1]),
        nlp=nlp,
    )

    expected_dxdt = model.system_dynamics(
        cn=states[0],
        f=states[1],
        t=0.11,
        t_stim_prev=np.array([0, 0.1]),
        pulse_width=controls[0],
        force_length_relationship=1,
        force_velocity_relationship=1,
        passive_force_relationship=0,
    )
    expected_defects = states_dot * nlp.dt - expected_dxdt * nlp.dt

    np.testing.assert_allclose(np.array(dynamics.dxdt).squeeze(), np.array(expected_dxdt).squeeze())
    np.testing.assert_allclose(np.array(dynamics.defects).squeeze(), np.array(expected_defects).squeeze())


def test_ding2007_with_fatigue_rk4_dynamics_keeps_defects_disabled():
    model = DingModelPulseWidthFrequencyWithFatigue(stim_time=[0, 0.1], sum_stim_truncation=10)
    nlp = _build_fake_nlp(model=model, states_dot=[1, 2, 3, 4, 5], ode_solver=OdeSolver.RK4())

    dynamics = model.dynamics(
        time=0.11,
        states=DM([5, 100, 4920, 0.060601, 0.137]),
        controls=DM([0.0002]),
        parameters=DM(),
        algebraic_states=DM(),
        numerical_timeseries=np.array([0, 0.1]),
        nlp=nlp,
    )

    assert dynamics.defects is None


def test_ding2003_collocation_dynamics_returns_expected_defects():
    model = DingModelFrequency(stim_time=[0, 0.1], sum_stim_truncation=10)
    states = DM([5, 100])
    states_dot = DM([1.2, -3.4])
    nlp = _build_fake_nlp(model=model, states_dot=states_dot, ode_solver=OdeSolver.COLLOCATION())

    dynamics = model.dynamics(
        time=0.11,
        states=states,
        controls=DM(),
        parameters=DM(),
        algebraic_states=DM(),
        numerical_timeseries=np.array([0, 0.1]),
        nlp=nlp,
    )

    expected_dxdt = model.system_dynamics(
        cn=states[0],
        f=states[1],
        t=0.11,
        t_stim_prev=np.array([0, 0.1]),
    )
    expected_defects = states_dot * nlp.dt - expected_dxdt * nlp.dt

    np.testing.assert_allclose(np.array(dynamics.dxdt).squeeze(), np.array(expected_dxdt).squeeze())
    np.testing.assert_allclose(np.array(dynamics.defects).squeeze(), np.array(expected_defects).squeeze())


def test_hmed2018_collocation_dynamics_returns_expected_defects():
    model = DingModelPulseIntensityFrequency(
        stim_time=[0, 0.1],
        previous_stim={"time": [0], "pulse_intensity": [60]},
        sum_stim_truncation=10,
    )
    states = DM([5, 100])
    controls = DM([60])
    states_dot = DM([1.2, -3.4])
    nlp = _build_fake_nlp(model=model, states_dot=states_dot, ode_solver=OdeSolver.COLLOCATION())

    numerical_timeseries = np.array([0.0, 0.1])
    dynamics = model.dynamics(
        time=0.11,
        states=states,
        controls=controls,
        parameters=DM(),
        algebraic_states=DM(),
        numerical_timeseries=numerical_timeseries,
        nlp=nlp,
    )

    expected_dxdt = model.system_dynamics(
        cn=states[0],
        f=states[1],
        t=0.11,
        t_stim_prev=numerical_timeseries,
        pulse_intensity=controls,
    )
    expected_defects = states_dot * nlp.dt - expected_dxdt * nlp.dt

    np.testing.assert_allclose(np.array(dynamics.dxdt).squeeze(), np.array(expected_dxdt).squeeze())
    np.testing.assert_allclose(np.array(dynamics.defects).squeeze(), np.array(expected_defects).squeeze())


def test_prepare_mhe_supports_collocation_with_ding2007_fatigue_model():
    cycle_duration = 1
    stimulation_frequency = 10
    stim_time = list(np.linspace(0, cycle_duration, stimulation_frequency + 1)[:-1])
    model = DingModelPulseWidthFrequencyWithFatigue(stim_time=stim_time, sum_stim_truncation=10)

    mhe = prepare_mhe(
        model=model,
        cycle_duration=cycle_duration,
        n_cycles_to_advance=1,
        n_cycles_simultaneous=1,
        max_pulse_width=0.0006,
        use_sx=False,
        minimize_force=True,
        minimize_fatigue=False,
        ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3, method="radau"),
    )

    assert mhe is not None
