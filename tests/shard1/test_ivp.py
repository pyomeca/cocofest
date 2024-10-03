import numpy as np
import pytest
import re

from cocofest import (
    IvpFes,
    DingModelFrequency,
    DingModelFrequencyWithFatigue,
    DingModelPulseDurationFrequency,
    DingModelPulseDurationFrequencyWithFatigue,
    DingModelIntensityFrequency,
    DingModelIntensityFrequencyWithFatigue,
)


@pytest.mark.parametrize("model", [DingModelFrequency(), DingModelFrequencyWithFatigue()])
def test_ding2003_ivp(model):
    fes_parameters = {"model": model, "n_stim": 3}
    ivp_parameters = {"n_shooting": 10, "final_time": 0.3, "use_sx": True}

    ivp = IvpFes(fes_parameters, ivp_parameters)

    # Integrating the solution
    result = ivp.integrate(return_time=False)

    if model._with_fatigue:
        np.testing.assert_almost_equal(result["F"][0][0], 0)
        np.testing.assert_almost_equal(result["F"][0][10], 92.06532561584642)
        np.testing.assert_almost_equal(result["F"][0][-1], 138.94556672277545)
    else:
        np.testing.assert_almost_equal(result["F"][0][0], 0)
        np.testing.assert_almost_equal(result["F"][0][10], 91.4098711524036)
        np.testing.assert_almost_equal(result["F"][0][-1], 130.3736693032713)


@pytest.mark.parametrize(
    "model",
    [DingModelPulseDurationFrequency(), DingModelPulseDurationFrequencyWithFatigue()],
)
@pytest.mark.parametrize("pulse_duration", [0.0003, [0.0003, 0.0004, 0.0005]])
def test_ding2007_ivp(model, pulse_duration):
    fes_parameters = {"model": model, "n_stim": 3, "pulse_duration": pulse_duration}
    ivp_parameters = {"n_shooting": 10, "final_time": 0.3, "use_sx": True}

    ivp = IvpFes(fes_parameters, ivp_parameters)

    # Integrating the solution
    result = ivp.integrate(return_time=False)

    if model._with_fatigue and isinstance(pulse_duration, list):
        np.testing.assert_almost_equal(result["F"][0][0], 0)
        np.testing.assert_almost_equal(result["F"][0][10], 28.3477940849177)
        np.testing.assert_almost_equal(result["F"][0][-1], 52.38870505209033)
    elif model._with_fatigue is False and isinstance(pulse_duration, list):
        np.testing.assert_almost_equal(result["F"][0][0], 0)
        np.testing.assert_almost_equal(result["F"][0][10], 28.116838973337046)
        np.testing.assert_almost_equal(result["F"][0][-1], 49.30316895125016)
    elif model._with_fatigue and isinstance(pulse_duration, float):
        np.testing.assert_almost_equal(result["F"][0][0], 0)
        np.testing.assert_almost_equal(result["F"][0][10], 28.3477940849177)
        np.testing.assert_almost_equal(result["F"][0][-1], 36.51217790065462)
    elif model._with_fatigue is False and isinstance(pulse_duration, float):
        np.testing.assert_almost_equal(result["F"][0][0], 0)
        np.testing.assert_almost_equal(result["F"][0][10], 28.116838973337046)
        np.testing.assert_almost_equal(result["F"][0][-1], 34.6369350284091)


@pytest.mark.parametrize("model", [DingModelIntensityFrequency(), DingModelIntensityFrequencyWithFatigue()])
@pytest.mark.parametrize("pulse_intensity", [50, [50, 60, 70]])
def test_hmed2018_ivp(model, pulse_intensity):
    fes_parameters = {"model": model, "n_stim": 3, "pulse_intensity": pulse_intensity}
    ivp_parameters = {"n_shooting": 10, "final_time": 0.3, "use_sx": True}

    ivp = IvpFes(fes_parameters, ivp_parameters)

    # Integrating the solution
    result = ivp.integrate(return_time=False)

    if model._with_fatigue and isinstance(pulse_intensity, list):
        np.testing.assert_almost_equal(result["F"][0][0], 0)
        np.testing.assert_almost_equal(result["F"][0][10], 42.18211764372109)
        np.testing.assert_almost_equal(result["F"][0][-1], 96.38882396648857)
    elif model._with_fatigue is False and isinstance(pulse_intensity, list):
        np.testing.assert_almost_equal(result["F"][0][0], 0)
        np.testing.assert_almost_equal(result["F"][0][10], 41.91914906078192)
        np.testing.assert_almost_equal(result["F"][0][-1], 92.23749672532881)
    elif model._with_fatigue and isinstance(pulse_intensity, float):
        np.testing.assert_almost_equal(result["F"][0][0], 0)
        np.testing.assert_almost_equal(result["F"][0][10], 42.18211764372109)
        np.testing.assert_almost_equal(result["F"][0][-1], 58.26448576796251)
    elif model._with_fatigue is False and isinstance(pulse_intensity, float):
        np.testing.assert_almost_equal(result["F"][0][0], 0)
        np.testing.assert_almost_equal(result["F"][0][10], 41.91914906078192)
        np.testing.assert_almost_equal(result["F"][0][-1], 55.57471909903151)


@pytest.mark.parametrize("pulse_mode", ["single", "doublet", "triplet"])
def test_pulse_mode_ivp(pulse_mode):
    n_stim = 3 if pulse_mode == "single" else 6 if pulse_mode == "doublet" else 9
    fes_parameters = {
        "model": DingModelFrequencyWithFatigue(),
        "n_stim": n_stim,
        "pulse_mode": pulse_mode,
    }
    ivp_parameters = {"n_shooting": 10, "final_time": 0.3, "use_sx": True}

    ivp = IvpFes(fes_parameters, ivp_parameters)

    # Integrating the solution
    result = ivp.integrate(return_time=False)

    if pulse_mode == "single":
        np.testing.assert_almost_equal(result["F"][0][0], 0)
        np.testing.assert_almost_equal(result["F"][0][10], 92.06532561584642)
        np.testing.assert_almost_equal(result["F"][0][-1], 138.94556672277545)
    elif pulse_mode == "doublet":
        np.testing.assert_almost_equal(result["F"][0][0], 0)
        np.testing.assert_almost_equal(result["F"][0][20], 107.1572156700596)
        np.testing.assert_almost_equal(result["F"][0][-1], 199.51123480749564)

    elif pulse_mode == "triplet":
        np.testing.assert_almost_equal(result["F"][0][0], 0)
        np.testing.assert_almost_equal(result["F"][0][30], 137.72706226851224)
        np.testing.assert_almost_equal(result["F"][0][-1], 236.04865519419803)


def test_ivp_methods():
    fes_parameters = {
        "model": DingModelFrequency(),
        "frequency": 30,
        "round_down": True,
    }
    ivp_parameters = {"n_shooting": 10, "final_time": 1.25, "use_sx": True}
    ivp = IvpFes.from_frequency_and_final_time(fes_parameters, ivp_parameters)

    fes_parameters = {"model": DingModelFrequency(), "n_stim": 3, "frequency": 10}
    ivp_parameters = {"n_shooting": 10, "use_sx": True}
    ivp = IvpFes.from_frequency_and_n_stim(fes_parameters, ivp_parameters)


def test_all_ivp_errors():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The number of stimulation needs to be integer within the final time t, set round down "
            "to True or set final_time * frequency to make the result an integer."
        ),
    ):
        IvpFes.from_frequency_and_final_time(
            fes_parameters={
                "model": DingModelFrequency(),
                "frequency": 30,
                "round_down": False,
            },
            ivp_parameters={"n_shooting": 1, "final_time": 1.25},
        )

    with pytest.raises(ValueError, match="Pulse mode not yet implemented"):
        IvpFes(
            fes_parameters={
                "model": DingModelFrequency(),
                "n_stim": 3,
                "pulse_mode": "Quadruplet",
            },
            ivp_parameters={"n_shooting": 10, "final_time": 0.3},
        )

    pulse_duration = 0.00001
    with pytest.raises(
        ValueError,
        match=re.escape("Pulse duration must be greater than minimum pulse duration"),
    ):
        IvpFes(
            fes_parameters={
                "model": DingModelPulseDurationFrequency(),
                "n_stim": 3,
                "pulse_duration": pulse_duration,
            },
            ivp_parameters={"n_shooting": 10, "final_time": 0.3},
        )

    with pytest.raises(ValueError, match="pulse_duration list must have the same length as n_stim"):
        IvpFes(
            fes_parameters={
                "model": DingModelPulseDurationFrequency(),
                "n_stim": 3,
                "pulse_duration": [0.0003, 0.0004],
            },
            ivp_parameters={"n_shooting": 10, "final_time": 0.3},
        )

    pulse_duration = [0.001, 0.0001, 0.003]
    with pytest.raises(
        ValueError,
        match=re.escape("Pulse duration must be greater than minimum pulse duration"),
    ):
        IvpFes(
            fes_parameters={
                "model": DingModelPulseDurationFrequency(),
                "n_stim": 3,
                "pulse_duration": pulse_duration,
            },
            ivp_parameters={"n_shooting": 10, "final_time": 0.3},
        )

    with pytest.raises(TypeError, match="pulse_duration must be int, float or list type"):
        IvpFes(
            fes_parameters={
                "model": DingModelPulseDurationFrequency(),
                "n_stim": 3,
                "pulse_duration": True,
            },
            ivp_parameters={"n_shooting": 10, "final_time": 0.3},
        )

    pulse_intensity = 0.1
    with pytest.raises(
        ValueError,
        match=re.escape("Pulse intensity must be greater than minimum pulse intensity"),
    ):
        IvpFes(
            fes_parameters={
                "model": DingModelIntensityFrequency(),
                "n_stim": 3,
                "pulse_intensity": pulse_intensity,
            },
            ivp_parameters={"n_shooting": 10, "final_time": 0.3},
        )

    with pytest.raises(ValueError, match="pulse_intensity list must have the same length as n_stim"):
        IvpFes(
            fes_parameters={
                "model": DingModelIntensityFrequency(),
                "n_stim": 3,
                "pulse_intensity": [20, 30],
            },
            ivp_parameters={"n_shooting": 10, "final_time": 0.3},
        )

    pulse_intensity = [20, 30, 0.1]
    with pytest.raises(
        ValueError,
        match=re.escape("Pulse intensity must be greater than minimum pulse intensity"),
    ):
        IvpFes(
            fes_parameters={
                "model": DingModelIntensityFrequency(),
                "n_stim": 3,
                "pulse_intensity": pulse_intensity,
            },
            ivp_parameters={"n_shooting": 10, "final_time": 0.3},
        )

    with pytest.raises(TypeError, match="pulse_intensity must be int, float or list type"):
        IvpFes(
            fes_parameters={
                "model": DingModelIntensityFrequency(),
                "n_stim": 3,
                "pulse_intensity": True,
            },
            ivp_parameters={"n_shooting": 10, "final_time": 0.3},
        )

    with pytest.raises(ValueError, match="ode_solver must be a OdeSolver type"):
        IvpFes(
            fes_parameters={"model": DingModelFrequency(), "n_stim": 3},
            ivp_parameters={"n_shooting": 10, "final_time": 0.3, "ode_solver": None},
        )

    with pytest.raises(ValueError, match="use_sx must be a bool type"):
        IvpFes(
            fes_parameters={"model": DingModelFrequency(), "n_stim": 3},
            ivp_parameters={"n_shooting": 10, "final_time": 0.3, "use_sx": None},
        )

    with pytest.raises(ValueError, match="n_thread must be a int type"):
        IvpFes(
            fes_parameters={"model": DingModelFrequency(), "n_stim": 3},
            ivp_parameters={"n_shooting": 10, "final_time": 0.3, "n_threads": None},
        )
