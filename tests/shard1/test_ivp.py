import numpy as np
import pytest
import re

from bioptim import OdeSolver

from cocofest import (
    IvpFes,
    ModelMaker,
)


ding2003_model = ModelMaker.create_model("ding2003", stim_time=[0, 0.1, 0.2], sum_stim_truncation=3)
ding2003_with_fatigue_model = ModelMaker.create_model(
    "ding2003_with_fatigue", stim_time=[0, 0.1, 0.2], sum_stim_truncation=3
)


@pytest.mark.parametrize("model", [ding2003_model, ding2003_with_fatigue_model])
def test_ding2003_ivp(model):
    fes_parameters = {"model": model}
    ivp_parameters = {"final_time": 0.3, "use_sx": True, "ode_solver": OdeSolver.RK4(n_integration_steps=10)}

    ivp = IvpFes(fes_parameters, ivp_parameters)

    # Integrating the solution
    result = ivp.integrate(return_time=False)

    if model._with_fatigue:
        np.testing.assert_almost_equal(
            result["F"][0],
            np.array(
                [
                    0.0,
                    15.85435155,
                    36.35204199,
                    54.97169718,
                    70.81575896,
                    83.41136615,
                    92.40789376,
                    97.57896902,
                    98.90445688,
                    96.65469613,
                    91.4120125,
                    99.09361882,
                    111.76148442,
                    123.22551071,
                    132.4700185,
                    138.93153923,
                    142.1847179,
                    141.94409218,
                    138.15150421,
                    131.06678227,
                    121.2908411,
                    125.7329513,
                    135.80039992,
                    144.91927299,
                    152.01179863,
                    156.47959033,
                    157.87252439,
                    155.88794042,
                    150.45844244,
                    141.84300151,
                    130.64878849,
                ]
            ),
        )
    else:
        np.testing.assert_almost_equal(
            result["F"][0],
            np.array(
                [
                    0.0,
                    15.85441782,
                    36.35269151,
                    54.97388526,
                    70.82053021,
                    83.41948674,
                    92.41942711,
                    97.5927868,
                    98.91780522,
                    96.66310621,
                    91.40987115,
                    99.10072926,
                    111.78356358,
                    123.2610033,
                    132.516354,
                    138.98486526,
                    142.23920967,
                    141.99114589,
                    138.17929522,
                    131.06091122,
                    121.23657189,
                    125.69535432,
                    135.79631287,
                    144.94318549,
                    152.05621732,
                    156.53496114,
                    157.92617846,
                    155.92291619,
                    150.45309627,
                    141.77256874,
                    130.48950127,
                ]
            ),
        )


ding2007_model = ModelMaker.create_model("ding2007", stim_time=[0, 0.1, 0.2], sum_stim_truncation=3)
ding2007_with_fatigue_model = ModelMaker.create_model(
    "ding2007_with_fatigue", stim_time=[0, 0.1, 0.2], sum_stim_truncation=3
)


@pytest.mark.parametrize(
    "model",
    [ding2007_model, ding2007_with_fatigue_model],
)
def test_ding2007_ivp(model):
    fes_parameters = {"model": model, "pulse_width": [0.0003, 0.0004, 0.0005]}
    ivp_parameters = {"final_time": 0.3, "use_sx": True, "ode_solver": OdeSolver.RK4(n_integration_steps=10)}

    ivp = IvpFes(fes_parameters, ivp_parameters)

    # Integrating the solution
    result = ivp.integrate(return_time=False)

    if model._with_fatigue:
        np.testing.assert_almost_equal(
            result["F"][0],
            np.array(
                [
                    0.0,
                    13.77407874,
                    30.42852732,
                    42.27898605,
                    48.64457727,
                    49.94831506,
                    47.54621724,
                    43.10082987,
                    37.93217388,
                    32.82193783,
                    28.13517032,
                    41.73854391,
                    59.64066732,
                    71.8805695,
                    77.49814818,
                    76.97127321,
                    71.99301134,
                    64.66027778,
                    56.63449125,
                    48.88819145,
                    41.86156497,
                    55.79791551,
                    74.8432786,
                    87.65932787,
                    93.12652796,
                    91.76219741,
                    85.46573453,
                    76.59704038,
                    67.02455461,
                    57.83809394,
                    49.52612787,
                ]
            ),
        )
    else:
        np.testing.assert_almost_equal(
            result["F"][0],
            np.array(
                [
                    0.0,
                    13.77411525,
                    30.42871244,
                    42.27924764,
                    48.64438024,
                    49.94664132,
                    47.54186407,
                    43.09290638,
                    37.92039497,
                    32.80657506,
                    28.11683897,
                    41.72514061,
                    59.63049607,
                    71.86902366,
                    77.48062902,
                    76.94318354,
                    71.95071726,
                    64.60234062,
                    56.56188609,
                    48.80354063,
                    41.76818425,
                    55.72032701,
                    74.77676255,
                    87.59145874,
                    93.04635121,
                    91.65999944,
                    85.33481641,
                    76.43578883,
                    66.83627994,
                    57.62907643,
                    49.30358905,
                ]
            ),
        )


hmed2018_model = ModelMaker.create_model("hmed2018", stim_time=[0, 0.1, 0.2], sum_stim_truncation=3)
hmed2018_with_fatigue_model = ModelMaker.create_model(
    "hmed2018_with_fatigue", stim_time=[0, 0.1, 0.2], sum_stim_truncation=3
)


@pytest.mark.parametrize("model", [hmed2018_model, hmed2018_with_fatigue_model])
def test_hmed2018_ivp(model):
    fes_parameters = {"model": model, "pulse_intensity": [50, 60, 70]}
    ivp_parameters = {"final_time": 0.3, "use_sx": True, "ode_solver": OdeSolver.RK4(n_integration_steps=10)}

    ivp = IvpFes(fes_parameters, ivp_parameters)

    # Integrating the solution
    result = ivp.integrate(return_time=False)

    if model._with_fatigue:
        np.testing.assert_almost_equal(
            result["F"][0],
            np.array(
                [
                    0.0,
                    8.9044621,
                    22.01573928,
                    33.76239359,
                    42.86842924,
                    48.90472654,
                    51.86726686,
                    52.06735882,
                    50.04399139,
                    46.44928894,
                    41.92750125,
                    48.76756334,
                    60.72872406,
                    71.46968401,
                    79.66898172,
                    84.77262716,
                    86.59667643,
                    85.27783783,
                    81.25361871,
                    75.18941517,
                    67.84992635,
                    73.8074438,
                    85.38080494,
                    95.82027285,
                    103.84589804,
                    108.853909,
                    110.53727837,
                    108.86843237,
                    104.13275282,
                    96.91362005,
                    87.99530615,
                ]
            ),
        )
    else:
        np.testing.assert_almost_equal(
            result["F"][0],
            np.array(
                [
                    0.0,
                    8.90449461,
                    22.0160478,
                    33.7634507,
                    42.87063964,
                    48.9081057,
                    51.87120621,
                    52.0705969,
                    50.04485602,
                    46.44613542,
                    41.91914906,
                    48.76576008,
                    60.73741959,
                    71.48783943,
                    79.69448892,
                    84.80217258,
                    86.62549723,
                    85.29982887,
                    81.26205602,
                    75.17823763,
                    67.81510144,
                    73.78860387,
                    85.3886438,
                    95.85080013,
                    103.89323931,
                    108.91031792,
                    110.59265526,
                    108.91024897,
                    104.14700474,
                    96.88702022,
                    87.91811759,
                ]
            ),
        )


@pytest.mark.parametrize("pulse_mode", ["single", "doublet", "triplet"])
def test_pulse_mode_ivp(pulse_mode):
    fes_parameters = {
        "model": ding2003_with_fatigue_model,
        "pulse_mode": pulse_mode,
    }
    ivp_parameters = {"final_time": 0.3, "use_sx": True, "ode_solver": OdeSolver.RK4(n_integration_steps=10)}

    ivp = IvpFes(fes_parameters, ivp_parameters)

    # Integrating the solution
    result = ivp.integrate(return_time=False)

    if pulse_mode == "single":
        np.testing.assert_almost_equal(
            result["F"][0],
            np.array(
                [
                    0.0,
                    15.85435155,
                    36.35204199,
                    54.97169718,
                    70.81575896,
                    83.41136615,
                    92.40789376,
                    97.57896902,
                    98.90445688,
                    96.65469613,
                    91.4120125,
                    99.09361882,
                    111.76148442,
                    123.22551071,
                    132.4700185,
                    138.93153923,
                    142.1847179,
                    141.94409218,
                    138.15150421,
                    131.06678227,
                    121.2908411,
                    125.7329513,
                    135.80039992,
                    144.91927299,
                    152.01179863,
                    156.47959033,
                    157.87252439,
                    155.88794042,
                    150.45844244,
                    141.84300151,
                    130.64878849,
                ]
            ),
        )
    elif pulse_mode == "doublet":
        np.testing.assert_almost_equal(
            result["F"][0][150:180],
            np.array(
                [
                    124.87618293,
                    125.18689644,
                    125.48798444,
                    125.77941427,
                    126.06115422,
                    126.33317358,
                    126.59544267,
                    126.8479329,
                    127.09061681,
                    127.32346812,
                    127.54646176,
                    127.75957395,
                    127.96278219,
                    128.15606537,
                    128.33940376,
                    128.51277911,
                    128.67617463,
                    128.8295751,
                    128.97296688,
                    129.10633793,
                    129.22967793,
                    129.34297824,
                    129.44623198,
                    129.53943411,
                    129.62258137,
                    129.69567244,
                    129.75870789,
                    129.81169025,
                    129.85462405,
                    129.88751587,
                ]
            ),
        )
    elif pulse_mode == "triplet":
        np.testing.assert_almost_equal(
            result["F"][0][350:380],
            np.array(
                [
                    220.086985,
                    220.22482724,
                    220.35632941,
                    220.48143445,
                    220.60008492,
                    220.71222302,
                    220.81779059,
                    220.91672912,
                    221.00897984,
                    221.09448364,
                    221.17318118,
                    221.24501288,
                    221.30991893,
                    221.36783933,
                    221.41871394,
                    221.46248248,
                    221.49908455,
                    221.5284597,
                    221.55054742,
                    221.56528722,
                    221.5726186,
                    221.57248114,
                    221.56481451,
                    221.54955853,
                    221.52665317,
                    221.49603862,
                    221.45765532,
                    221.41144399,
                    221.3573457,
                    221.29530191,
                ]
            ),
        )


def test_ivp_methods():
    fes_parameters = {
        "model": ding2003_model,
        "frequency": 30,
        "round_down": True,
    }
    ivp_parameters = {"final_time": 1.25, "use_sx": True}
    ivp = IvpFes.from_frequency_and_final_time(fes_parameters, ivp_parameters)

    fes_parameters = {"model": ding2003_model, "n_stim": 3, "frequency": 10}
    ivp_parameters = {"use_sx": True}
    ivp = IvpFes.from_frequency_and_n_stim(fes_parameters, ivp_parameters)


# def test_all_ivp_errors():
#     with pytest.raises(
#         ValueError,
#         match=re.escape(
#             "The number of stimulation needs to be integer within the final time t, set round down "
#             "to True or set final_time * frequency to make the result an integer."
#         ),
#     ):
#         IvpFes.from_frequency_and_final_time(
#             fes_parameters={
#                 "model": ding2003_model,
#                 "frequency": 30,
#                 "round_down": False,
#             },
#             ivp_parameters={"final_time": 1.25},
#         )
#
#     with pytest.raises(ValueError, match="Pulse mode not yet implemented"):
#         IvpFes(
#             fes_parameters={
#                 "model": ding2003_model,
#                 "stim_time": [0, 0.1, 0.2],
#                 "pulse_mode": "Quadruplet",
#             },
#             ivp_parameters={"final_time": 0.3},
#         )
#
#     pulse_width = 0.00001
#     with pytest.raises(
#         ValueError,
#         match=re.escape("pulse width must be greater than minimum pulse width"),
#     ):
#         IvpFes(
#             fes_parameters={
#                 "model": ding2007_model,
#                 "pulse_width": pulse_width,
#             },
#             ivp_parameters={"final_time": 0.3},
#         )
#
#     with pytest.raises(ValueError, match="pulse_width list must have the same length as n_stim"):
#         IvpFes(
#             fes_parameters={
#                 "model": ding2007_model,
#                 "stim_time": [0, 0.1, 0.2],
#             },
#             ivp_parameters={"final_time": 0.3},
#         )
#
#     pulse_width = [0.001, 0.0001, 0.003]
#     with pytest.raises(
#         ValueError,
#         match=re.escape("pulse width must be greater than minimum pulse width"),
#     ):
#         IvpFes(
#             fes_parameters={
#                 "model": ding2007_model,
#                 "stim_time": [0, 0.1, 0.2],
#                 "pulse_width": pulse_width,
#             },
#             ivp_parameters={"final_time": 0.3},
#         )
#
#     with pytest.raises(TypeError, match="pulse_width must be int, float or list type"):
#         IvpFes(
#             fes_parameters={
#                 "model": ding2007_model,
#                 "stim_time": [0, 0.1, 0.2],
#                 "pulse_width": True,
#             },
#             ivp_parameters={"final_time": 0.3},
#         )
#
#     pulse_intensity = 0.1
#     with pytest.raises(
#         ValueError,
#         match=re.escape("Pulse intensity must be greater than minimum pulse intensity"),
#     ):
#         IvpFes(
#             fes_parameters={
#                 "model": hmed2018_model,
#                 "stim_time": [0, 0.1, 0.2],
#                 "pulse_intensity": pulse_intensity,
#             },
#             ivp_parameters={"final_time": 0.3},
#         )
#
#     with pytest.raises(ValueError, match="pulse_intensity list must have the same length as n_stim"):
#         IvpFes(
#             fes_parameters={
#                 "model": hmed2018_model,
#                 "stim_time": [0, 0.1, 0.2],
#                 "pulse_intensity": [20, 30],
#             },
#             ivp_parameters={"final_time": 0.3},
#         )
#
#     pulse_intensity = [20, 30, 0.1]
#     with pytest.raises(
#         ValueError,
#         match=re.escape("Pulse intensity must be greater than minimum pulse intensity"),
#     ):
#         IvpFes(
#             fes_parameters={
#                 "model": hmed2018_model,
#                 "stim_time": [0, 0.1, 0.2],
#                 "pulse_intensity": pulse_intensity,
#             },
#             ivp_parameters={"final_time": 0.3},
#         )
#
#     with pytest.raises(TypeError, match="pulse_intensity must be int, float or list type"):
#         IvpFes(
#             fes_parameters={
#                 "model": hmed2018_model,
#                 "stim_time": [0, 0.1, 0.2],
#                 "pulse_intensity": True,
#             },
#             ivp_parameters={"final_time": 0.3},
#         )
#
#     with pytest.raises(ValueError, match="ode_solver must be a OdeSolver type"):
#         IvpFes(
#             fes_parameters={
#                 "model": ding2003_model,
#                 "stim_time": [0, 0.1, 0.2],
#             },
#             ivp_parameters={"final_time": 0.3, "ode_solver": None},
#         )
#
#     with pytest.raises(ValueError, match="use_sx must be a bool type"):
#         IvpFes(
#             fes_parameters={
#                 "model": ding2003_model,
#                 "stim_time": [0, 0.1, 0.2],
#             },
#             ivp_parameters={"final_time": 0.3, "use_sx": None},
#         )
#
#     with pytest.raises(ValueError, match="n_thread must be a int type"):
#         IvpFes(
#             fes_parameters={
#                 "model": ding2003_model,
#                 "stim_time": [0, 0.1, 0.2],
#             },
#             ivp_parameters={"final_time": 0.3, "n_threads": None},
#         )
