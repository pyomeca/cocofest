import numpy as np
import pytest
import re

from bioptim import OdeSolver

from cocofest import (
    IvpFes,
    ModelMaker,
)

stim_time = np.linspace(0, 1, 34).tolist()
tested_index = np.linspace(0, 330, 34, dtype=int)


def check_values(result, tested_values):
    for i in range(len(tested_values)):
        np.testing.assert_almost_equal(
            result[tested_index[i]],
            tested_values[i],
        )


ding2003_model = ModelMaker.create_model("ding2003", stim_time=stim_time, sum_stim_truncation=10)
ding2003_with_fatigue_model = ModelMaker.create_model(
    "ding2003_with_fatigue", stim_time=stim_time, sum_stim_truncation=10
)


@pytest.mark.parametrize("model", [ding2003_model, ding2003_with_fatigue_model])
def test_ding2003_ivp(model):
    fes_parameters = {"model": model}
    ivp_parameters = {"final_time": 1, "use_sx": True, "ode_solver": OdeSolver.RK4(n_integration_steps=10)}

    ivp = IvpFes(fes_parameters, ivp_parameters)

    # Integrating the solution
    result = ivp.integrate(return_time=False)

    if model._with_fatigue:
        tested_values = [
            0.0,
            56.30597484805479,
            107.85873390605241,
            148.1414604844786,
            178.67945759630845,
            201.59045442765145,
            218.716328484854,
            231.5056501251912,
            241.05844352897964,
            248.19916026990384,
            253.54276340362148,
            257.5472232923092,
            260.55346579154696,
            262.8153013014773,
            264.52173135724735,
            265.81354370376124,
            266.7956504424266,
            267.546260721029,
            268.12370121361846,
            268.57148892477477,
            268.9221053497975,
            269.19980544397566,
            269.4227090302129,
            269.6043585677413,
            269.75487991975274,
            269.8818476589967,
            269.99093039266296,
            270.08637223755454,
            270.17135220287156,
            270.24825255735533,
            270.318859318314,
            270.3845120962799,
            270.4462161371346,
            270.5047261349804,
        ]

    else:
        tested_values = [
            0.0,
            56.30834295477824,
            107.87317965806251,
            148.17485505144225,
            178.73035920955502,
            201.6499607806392,
            218.77104245557604,
            231.5405877835894,
            241.05914647704896,
            248.1529497509755,
            253.43931150149805,
            257.3786585294478,
            260.31420000999896,
            262.50171580794563,
            264.1318144174064,
            265.3465352160571,
            266.2517237252118,
            266.9262542210047,
            267.42890243490444,
            267.80346700703456,
            268.08258591117317,
            268.29058038911853,
            268.4455742007614,
            268.56107285190177,
            268.6471404069466,
            268.7112764289543,
            268.75906945986844,
            268.7946839824481,
            268.8212232956977,
            268.8409999232289,
            268.8557371167964,
            268.86671901317396,
            268.87490252840666,
            268.88100073943167,
        ]

    check_values(result["F"][0], tested_values)


ding2007_model = ModelMaker.create_model("ding2007", stim_time=stim_time, sum_stim_truncation=10)
ding2007_with_fatigue_model = ModelMaker.create_model(
    "ding2007_with_fatigue", stim_time=stim_time, sum_stim_truncation=10
)


@pytest.mark.parametrize(
    "model",
    [ding2007_model, ding2007_with_fatigue_model],
)
def test_ding2007_ivp(model):
    fes_parameters = {"model": model, "pulse_width": np.linspace(0.0002, 0.0006, 33).tolist()}
    ivp_parameters = {"final_time": 1, "use_sx": True, "ode_solver": OdeSolver.RK4(n_integration_steps=10)}

    ivp = IvpFes(fes_parameters, ivp_parameters)

    # Integrating the solution
    result = ivp.integrate(return_time=False)

    if model._with_fatigue:
        tested_values = [
            0.0,
            22.775556557690887,
            42.34881706753852,
            57.942068375807814,
            70.71989889120825,
            81.56613720628856,
            91.04899039889378,
            99.52831255841697,
            107.23470721327949,
            114.31911590039098,
            120.8832160386713,
            126.99800794594817,
            132.71518973956375,
            138.07413274182338,
            143.10617268440373,
            147.83726224635006,
            152.28962218508616,
            156.4827795389792,
            160.43422980792235,
            164.15986767084058,
            167.67427452696046,
            170.99091684886454,
            174.12228842083013,
            177.08001677524058,
            179.87494634772986,
            182.5172061083052,
            185.01626650936788,
            187.38098880238422,
            189.61966867417098,
            191.74007547380572,
            193.74948787884853,
            195.65472658522458,
            197.46218443790778,
            199.1778543122337,
        ]
    else:
        tested_values = [
            0.0,
            22.775617489789315,
            42.34758582223616,
            57.93428248212732,
            70.69670613158615,
            81.51629755778552,
            90.95988236550876,
            99.3865944648236,
            107.02671827640745,
            114.0311192223002,
            120.50155510568823,
            126.50921778793361,
            132.10608345273684,
            137.33187256130657,
            142.21832914607666,
            146.79186263536624,
            151.0751887631101,
            155.08835802815042,
            158.84940880752916,
            162.3747898807743,
            165.67964077705224,
            168.77798397631756,
            171.68286201506936,
            174.406439742243,
            176.96008415362874,
            179.35442945938723,
            181.59943212235214,
            183.70441882107167,
            185.6781291993707,
            187.52875459419036,
            189.26397352121296,
            190.89098444300765,
            192.4165361859368,
            193.8469562724926,
        ]

    check_values(result["F"][0], tested_values)


hmed2018_model = ModelMaker.create_model("hmed2018", stim_time=stim_time, sum_stim_truncation=10)
hmed2018_with_fatigue_model = ModelMaker.create_model(
    "hmed2018_with_fatigue", stim_time=stim_time, sum_stim_truncation=10
)


@pytest.mark.parametrize("model", [hmed2018_model, hmed2018_with_fatigue_model])
def test_hmed2018_ivp(model):
    fes_parameters = {"model": model, "pulse_intensity": np.linspace(20, 130, 33).tolist()}
    ivp_parameters = {"final_time": 1, "use_sx": True, "ode_solver": OdeSolver.RK4(n_integration_steps=10)}

    ivp = IvpFes(fes_parameters, ivp_parameters)

    # Integrating the solution
    result = ivp.integrate(return_time=False)

    if model._with_fatigue:
        tested_values = [
            0.0,
            2.931932242446827,
            10.343645642459522,
            21.554714949147243,
            35.418463733219085,
            50.975005471405844,
            67.48513662634976,
            84.37683464849633,
            101.20383555828413,
            117.62067137228574,
            133.36670927979108,
            148.25405437065473,
            162.15692823876867,
            175.00169268262854,
            186.75739715962365,
            197.42698485906618,
            207.0393355329798,
            215.64227880921993,
            223.29664133033677,
            230.07132353513578,
            236.03934890712742,
            241.27479297155196,
            245.8504798859795,
            249.83632799679455,
            253.2982286383877,
            256.297351415762,
            258.88978151425096,
            261.1264081920421,
            263.0529971475683,
            264.7103920628657,
            266.13480185431337,
            267.35813983910134,
            268.4083891506616,
            269.30997541225105,
        ]

    else:
        tested_values = [
            0.0,
            2.931941488405683,
            10.343849939688235,
            21.555913710024903,
            35.42243035947065,
            50.98451040876859,
            67.50362133733857,
            84.40784058933676,
            101.25033615102208,
            117.68445078882532,
            133.44789449421344,
            148.3508563909385,
            162.26560110715675,
            175.1166836815902,
            186.87164118892403,
            197.5322886229341,
            207.12679960435145,
            215.70271407635684,
            223.3209475243321,
            230.05080906740574,
            235.96598157007142,
            241.14137937396714,
            245.65077756197167,
            249.5650980594211,
            252.95123896492407,
            255.87134104972608,
            258.3823966517354,
            260.53611914415416,
            262.3790043291961,
            263.9525275727038,
            265.29343173891436,
            266.4340707760552,
            267.40278210028987,
            268.224267798157,
        ]

    check_values(result["F"][0], tested_values)


ding2003_with_fatigue_model = ModelMaker.create_model(
    "ding2003_with_fatigue", stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
)


@pytest.mark.parametrize("pulse_mode", ["single", "doublet", "triplet"])
def test_pulse_mode_ivp(pulse_mode):
    fes_parameters = {
        "model": ding2003_with_fatigue_model,
        "pulse_mode": pulse_mode,
    }
    ivp_parameters = {"final_time": 1, "use_sx": True, "ode_solver": OdeSolver.RK4(n_integration_steps=10)}

    ivp = IvpFes(fes_parameters, ivp_parameters)

    # Integrating the solution
    result = ivp.integrate(return_time=False)

    if pulse_mode == "single":
        tested_values = [
            0.0,
            91.41201249507682,
            121.29084110441761,
            130.64878849132887,
            133.6325306398771,
            134.63501242329733,
            135.01838738959748,
            135.2065215051987,
            135.33181981213994,
            135.43580597313056,
            135.5315562363452,
        ]

    elif pulse_mode == "doublet":
        tested_values = [
            0.0,
            6.811659796099999,
            17.836098653932126,
            29.55736919641,
            41.04200577764048,
            52.06499584872172,
            62.532496833034685,
            72.39128730499773,
            81.6025536882239,
            90.13179437212492,
            97.94429264124467,
            105.00306253311439,
            111.26822359129736,
            116.6974534531757,
            121.24742211246753,
            124.87618292671563,
            127.54646176449623,
            129.22967792906385,
            129.91037433984613,
            129.59056707624796,
            128.2933969321963,
            130.2820796006739,
            135.62676745353022,
            141.89351264990967,
            148.18010795280892,
            154.24004557311832,
            159.96083428619735,
            165.27342115086742,
            170.12491198575734,
            174.4678252728422,
            178.2551988671325,
            181.43833498379723,
            183.96608043102177,
            185.78526600963679,
        ]

    elif pulse_mode == "triplet":
        tested_values = [
            0.0,
            6.811659796099999,
            18.419341569107342,
            31.018933207521105,
            43.51388159780265,
            55.5996068206944,
            67.18132459057564,
            78.23205802958206,
            88.74309318549606,
            98.71107226716651,
            108.13310254845092,
            117.00434140759826,
            125.31651184893666,
            133.0568245159095,
            140.20712082300022,
            146.74319450865164,
            152.63432797007994,
            157.84313653511586,
            162.32586010475507,
            166.03327591175878,
            168.91241867862357,
            172.07412408967804,
            176.7299944839003,
            182.24856137475152,
            187.95780010310577,
            193.56606260351114,
            198.96016141251192,
            204.0961127202081,
            208.95003601617074,
            213.50462110419983,
            217.74384599976568,
            221.6503283192785,
            225.20368978421823,
            228.37937196951108,
        ]

    check_values(result["F"][0], tested_values)


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
