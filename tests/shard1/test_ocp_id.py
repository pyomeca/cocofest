import re
import pickle
import os

import numpy as np
import pytest

from bioptim import ControlType

from cocofest import (
    OcpFesId,
    IvpFes,
    DingModelFrequency,
    DingModelFrequencyIntegrate,
    DingModelPulseDurationFrequency,
    DingModelPulseDurationFrequencyIntegrate,
    DingModelIntensityFrequency,
    DingModelIntensityFrequencyIntegrate,
    DingModelFrequencyWithFatigue,
    DingModelFrequencyForceParameterIdentification,
    DingModelPulseDurationFrequencyForceParameterIdentification,
    DingModelPulseIntensityFrequencyForceParameterIdentification,
)

# model = DingModelFrequencyIntegrate
# stim = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# discontinuity = []
# n_shooting = [10, 10, 10, 10, 10, 10, 10, 10, 10]
# final_time_phase = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
force_at_node = [
    0.0,
    15.854417817548697,
    36.352691513848825,
    54.97388526038394,
    70.82053020534033,
    83.4194867436611,
    92.41942711195031,
    97.59278680323499,
    98.91780522472382,
    96.66310621259122,
    91.40987115240357,
    99.06445243586181,
    111.71243288393804,
    123.15693012605658,
    132.377564374587,
    138.80824696656194,
    142.02210466592646,
    141.73329382731202,
    137.8846391765497,
    130.73845520335917,
    120.89955520685814,
    125.38085346839195,
    135.5102503776543,
    144.68352158676947,
    151.82103024326696,
    156.32258873146037,
    157.7352209679825,
    155.7522062101244,
    150.30163417046595,
    141.63941942299235,
    130.37366930327116,
    133.8337968502193,
    143.14339960917405,
    151.5764058207702,
    158.03337418092343,
    161.90283357182437,
    162.7235878298402,
    160.18296049789234,
    154.20561770076475,
    145.04700073978324,
    133.31761023027764,
    136.45613333201518,
    145.51077095073057,
    153.71376988470408,
    159.9593525193574,
    163.63249745182017,
    164.26944493409266,
    161.5556743081453,
    155.41480228777354,
    146.102138099759,
    134.2289338756813,
    137.26784246878907,
    146.24355077243675,
    154.37534966572915,
    160.55549761772778,
    164.16787361651103,
    164.74792419884477,
    161.980557871029,
    155.7890666284376,
    146.42871894433975,
    134.51099963420057,
    137.5190757238638,
    146.47035441998682,
    154.58011604419679,
    160.74001118470105,
    164.33357848128765,
    164.89601880480728,
    162.11206398140135,
    155.90490550332385,
    146.52979923122618,
    134.5983019993197,
    137.59683509441382,
    146.54055256585733,
    154.64349341999517,
    160.7971200991939,
    164.3848659017574,
    164.94185566229874,
    162.15276652256654,
    155.9407588680354,
    146.56108465562548,
    134.62532301012072,
    137.6209024468235,
    146.56227963878422,
    154.66310939202043,
    160.8147959157575,
    164.4007399039886,
    164.95604265730248,
    162.16536439204623,
    155.9518558658154,
    146.5707678272839,
]

additional_key_settings = {
    "a_rest": {
        "initial_guess": 1000,
        "min_bound": 1,
        "max_bound": 10000,
        "function": DingModelFrequencyIntegrate.set_a_rest,
        "scaling": 1,
    },
    "km_rest": {
        "initial_guess": 0.5,
        "min_bound": 0.001,
        "max_bound": 1,
        "function": DingModelFrequencyIntegrate.set_km_rest,
        "scaling": 1,
    },
    "tau1_rest": {
        "initial_guess": 0.5,
        "min_bound": 0.0001,
        "max_bound": 1,
        "function": DingModelFrequencyIntegrate.set_tau1_rest,
        "scaling": 1,
    },
    "tau2": {
        "initial_guess": 0.5,
        "min_bound": 0.0001,
        "max_bound": 1,
        "function": DingModelFrequencyIntegrate.set_tau2,
        "scaling": 1,
    },
}


@pytest.mark.parametrize("model", [DingModelFrequency(), DingModelFrequencyIntegrate()])
def test_ocp_id_ding2003(model):
    # --- Creating the simulated data to identify on --- #
    # Building the Initial Value Problem
    ivp = IvpFes(
        fes_parameters={
            "model": DingModelFrequencyIntegrate(),
            "stim_time": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
        ivp_parameters={
            "n_shooting": 200,
            "final_time": 2,
            "use_sx": True,
        },
    )

    # Integrating the solution
    result, time = ivp.integrate()

    force = result["F"][0].tolist()
    time = [float(time) for time in time]

    # Saving the data in a pickle file
    dictionary = {
        "time": time,
        "force": force,
        "stim_time": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    }

    pickle_file_name = "temp_identification_simulation.pkl"
    with open(pickle_file_name, "wb") as file:
        pickle.dump(dictionary, file)

    # --- Identifying the model parameters --- #
    ocp = DingModelFrequencyForceParameterIdentification(
        model=model,
        n_shooting=200,
        final_time=2,
        data_path=[pickle_file_name],
        identification_method="full",
        double_step_identification=False,
        key_parameter_to_identify=["a_rest", "km_rest", "tau1_rest", "tau2"],
        additional_key_settings={},
        use_sx=True,
        n_threads=6,
        control_type=ControlType.LINEAR_CONTINUOUS,
    )

    identification_result = ocp.force_model_identification()

    # --- Delete the temp file ---#
    os.remove(f"temp_identification_simulation.pkl")

    if isinstance(model, DingModelFrequencyIntegrate):
        np.testing.assert_almost_equal(identification_result["a_rest"], DingModelFrequencyIntegrate().a_rest, decimal=0)
        np.testing.assert_almost_equal(
            identification_result["km_rest"], DingModelFrequencyIntegrate().km_rest, decimal=3
        )
        np.testing.assert_almost_equal(
            identification_result["tau1_rest"], DingModelFrequencyIntegrate().tau1_rest, decimal=3
        )
        np.testing.assert_almost_equal(identification_result["tau2"], DingModelFrequencyIntegrate().tau2, decimal=3)
    else:
        np.testing.assert_almost_equal(identification_result["a_rest"], 2920, decimal=0)
        np.testing.assert_almost_equal(identification_result["km_rest"], 0.1108, decimal=3)
        np.testing.assert_almost_equal(identification_result["tau1_rest"], 0.0525, decimal=3)
        np.testing.assert_almost_equal(identification_result["tau2"], 0.0534, decimal=3)


@pytest.mark.parametrize("model", [DingModelPulseDurationFrequency(), DingModelPulseDurationFrequencyIntegrate()])
def test_ocp_id_ding2007(model):
    # --- Creating the simulated data to identify on --- #
    # Building the Initial Value Problem
    ivp = IvpFes(
        fes_parameters={
            "model": DingModelPulseDurationFrequencyIntegrate(),
            "stim_time": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "pulse_duration": [0.003] * 10,
        },
        ivp_parameters={
            "n_shooting": 200,
            "final_time": 2,
            "use_sx": True,
        },
    )

    # Integrating the solution
    result, time = ivp.integrate()

    force = result["F"][0].tolist()
    time = [float(time) for time in time]

    # Saving the data in a pickle file
    dictionary = {
        "time": time,
        "force": force,
        "stim_time": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "pulse_duration": [0.003] * 10,
    }

    pickle_file_name = "temp_identification_simulation.pkl"
    with open(pickle_file_name, "wb") as file:
        pickle.dump(dictionary, file)

    # --- Identifying the model parameters --- #
    ocp = DingModelPulseDurationFrequencyForceParameterIdentification(
        model=model,
        n_shooting=200,
        final_time=2,
        data_path=[pickle_file_name],
        identification_method="full",
        double_step_identification=False,
        key_parameter_to_identify=["tau1_rest", "tau2", "km_rest", "a_scale", "pd0", "pdt"],
        additional_key_settings={},
        use_sx=True,
        n_threads=6,
    )

    identification_result = ocp.force_model_identification()

    # --- Delete the temp file ---#
    os.remove(f"temp_identification_simulation.pkl")

    if isinstance(model, DingModelPulseDurationFrequencyIntegrate):
        np.testing.assert_almost_equal(
            identification_result["tau1_rest"], DingModelPulseDurationFrequencyIntegrate().tau1_rest, decimal=3
        )
        np.testing.assert_almost_equal(
            identification_result["tau2"], DingModelPulseDurationFrequencyIntegrate().tau2, decimal=3
        )
        np.testing.assert_almost_equal(
            identification_result["km_rest"], DingModelPulseDurationFrequencyIntegrate().km_rest, decimal=3
        )
        np.testing.assert_almost_equal(
            identification_result["a_scale"], DingModelPulseDurationFrequencyIntegrate().a_scale, decimal=-2
        )
        np.testing.assert_almost_equal(
            identification_result["pd0"], DingModelPulseDurationFrequencyIntegrate().pd0, decimal=3
        )
        np.testing.assert_almost_equal(
            identification_result["pdt"], DingModelPulseDurationFrequencyIntegrate().pdt, decimal=3
        )
    else:
        np.testing.assert_almost_equal(identification_result["tau1_rest"], 0.0644, decimal=3)
        np.testing.assert_almost_equal(identification_result["tau2"], 0.0493, decimal=3)
        np.testing.assert_almost_equal(identification_result["km_rest"], 0.424, decimal=3)
        np.testing.assert_almost_equal(identification_result["a_scale"], 6093, decimal=-2)
        np.testing.assert_almost_equal(identification_result["pd0"], 0.0003, decimal=3)
        np.testing.assert_almost_equal(identification_result["pdt"], 0.0003, decimal=3)


@pytest.mark.parametrize("model", [DingModelIntensityFrequency(), DingModelIntensityFrequencyIntegrate()])
def test_ocp_id_hmed2018(model):
    # --- Creating the simulated data to identify on --- #
    # Building the Initial Value Problem
    ivp = IvpFes(
        fes_parameters={
            "model": DingModelIntensityFrequencyIntegrate(),
            "stim_time": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "pulse_intensity": list(np.linspace(30, 130, 11))[:-1],
        },
        ivp_parameters={
            "n_shooting": 200,
            "final_time": 2,
            "use_sx": True,
        },
    )

    # Integrating the solution
    result, time = ivp.integrate()

    force = result["F"][0].tolist()
    time = [float(time) for time in time]

    # Saving the data in a pickle file
    dictionary = {
        "time": time,
        "force": force,
        "stim_time": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "pulse_intensity": list(np.linspace(30, 130, 11))[:-1],
    }

    pickle_file_name = "temp_identification_simulation.pkl"
    with open(pickle_file_name, "wb") as file:
        pickle.dump(dictionary, file)

    # --- Identifying the model parameters --- #
    ocp = DingModelPulseIntensityFrequencyForceParameterIdentification(
        model=model,
        n_shooting=200,
        final_time=2,
        data_path=[pickle_file_name],
        identification_method="full",
        double_step_identification=False,
        key_parameter_to_identify=[
            "a_rest",
            "km_rest",
            "tau1_rest",
            "tau2",
            "ar",
            "bs",
            "Is",
            "cr",
        ],
        additional_key_settings={},
        use_sx=True,
        n_threads=6,
    )

    identification_result = ocp.force_model_identification()

    # --- Delete the temp file ---#
    os.remove(f"temp_identification_simulation.pkl")

    if isinstance(model, DingModelIntensityFrequencyIntegrate):
        np.testing.assert_almost_equal(
            identification_result["tau1_rest"], DingModelIntensityFrequencyIntegrate().tau1_rest, decimal=3
        )
        np.testing.assert_almost_equal(
            identification_result["tau2"], DingModelIntensityFrequencyIntegrate().tau2, decimal=3
        )
        np.testing.assert_almost_equal(identification_result["km_rest"], 0.174, decimal=3)
        np.testing.assert_almost_equal(identification_result["ar"], 0.9913, decimal=3)
        np.testing.assert_almost_equal(
            identification_result["bs"], DingModelIntensityFrequencyIntegrate().bs, decimal=3
        )
        np.testing.assert_almost_equal(
            identification_result["Is"], DingModelIntensityFrequencyIntegrate().Is, decimal=1
        )
        np.testing.assert_almost_equal(
            identification_result["cr"], DingModelIntensityFrequencyIntegrate().cr, decimal=3
        )
    else:
        np.testing.assert_almost_equal(identification_result["tau1_rest"], 0.0499, decimal=3)
        np.testing.assert_almost_equal(identification_result["tau2"], 0.0284, decimal=3)
        np.testing.assert_almost_equal(identification_result["km_rest"], 0.004, decimal=3)
        np.testing.assert_almost_equal(identification_result["ar"], 0.0228, decimal=3)
        np.testing.assert_almost_equal(identification_result["bs"], 0.026, decimal=3)
        np.testing.assert_almost_equal(identification_result["Is"], 68.732, decimal=3)
        np.testing.assert_almost_equal(identification_result["cr"], 0.865, decimal=3)


def test_all_ocp_id_errors():
    n_shooting = "10"
    with pytest.raises(
        TypeError,
        match=re.escape(f"n_shooting must be a positive int type"),
    ):
        OcpFesId.prepare_ocp(model=DingModelFrequency(), n_shooting=n_shooting)

    n_shooting = -10
    with pytest.raises(TypeError, match=re.escape(f"n_shooting must be a positive int type")):
        OcpFesId.prepare_ocp(model=DingModelFrequency(), n_shooting=n_shooting)

    final_time_phase = "0.1"
    with pytest.raises(
        TypeError,
        match=re.escape(f"final_time must be a positive int or float type"),
    ):
        OcpFesId.prepare_ocp(
            model=DingModelFrequency(),
            n_shooting=100,
            final_timee=final_time_phase,
        )

    force_tracking = "10"
    with pytest.raises(
        TypeError,
        match=re.escape(
            f"force_tracking must be list type," f" currently force_tracking is {type(force_tracking)}) type."
        ),
    ):
        OcpFesId.prepare_ocp(
            model=DingModelFrequency(),
            n_shooting=100,
            final_time=1,
            objective={"force_tracking": force_tracking},
        )

    force_tracking = [10, 10, 10, 10, 10, 10, 10, 10, "10"]
    with pytest.raises(TypeError, match=re.escape(f"force_tracking must be list of int or float type.")):
        OcpFesId.prepare_ocp(
            model=DingModelFrequency(),
            n_shooting=100,
            final_time=1,
            objective={"force_tracking": force_tracking},
        )


def test_all_id_program_errors():
    key_parameter_to_identify = ["a_rest", "km_rest", "tau1_rest", "tau2"]
    with pytest.raises(
        ValueError,
        match="The given model is not valid and should not be including the fatigue equation in the model",
    ):
        DingModelFrequencyForceParameterIdentification(
            model=DingModelFrequencyWithFatigue(),
            key_parameter_to_identify=key_parameter_to_identify,
        )

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"In the given list, all model_data_path must be str type," f" path index n°{0} is not str type"
        ),
    ):
        DingModelFrequencyForceParameterIdentification(
            model=DingModelFrequency(),
            data_path=[5],
            key_parameter_to_identify=key_parameter_to_identify,
        )

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"In the given list, all model_data_path must be pickle type and end with .pkl,"
            f" path index n°{0} is not ending with .pkl"
        ),
    ):
        DingModelFrequencyForceParameterIdentification(
            model=DingModelFrequency(),
            data_path=["test"],
            key_parameter_to_identify=key_parameter_to_identify,
        )

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"In the given list, all model_data_path must be pickle type and end with .pkl,"
            f" path index is not ending with .pkl"
        ),
    ):
        DingModelFrequencyForceParameterIdentification(
            model=DingModelFrequency(),
            data_path="test",
            key_parameter_to_identify=key_parameter_to_identify,
        )

    data_path = 5
    with pytest.raises(
        TypeError,
        match=re.escape(
            f"In the given path, model_data_path must be str or list[str] type, the input is {type(data_path)} type"
        ),
    ):
        DingModelFrequencyForceParameterIdentification(
            model=DingModelFrequency(),
            data_path=data_path,
            key_parameter_to_identify=key_parameter_to_identify,
        )

    data_path = ["test.pkl"]
    identification_method = "empty"
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The given model identification method is not valid,"
            f"only 'full', 'average' and 'sparse' are available,"
            f" the given value is {identification_method}"
        ),
    ):
        DingModelFrequencyForceParameterIdentification(
            model=DingModelFrequency(),
            identification_method=identification_method,
            data_path=data_path,
            key_parameter_to_identify=key_parameter_to_identify,
        )

    identification_method = "full"
    double_step_identification = "True"
    with pytest.raises(
        TypeError,
        match=re.escape(
            f"The given double_step_identification must be bool type,"
            f" the given value is {type(double_step_identification)} type"
        ),
    ):
        DingModelFrequencyForceParameterIdentification(
            model=DingModelFrequency(),
            identification_method=identification_method,
            data_path=data_path,
            double_step_identification=double_step_identification,
            key_parameter_to_identify=key_parameter_to_identify,
        )

    key_parameter_to_identify = ["a_rest", "test"]
    default_keys = ["a_rest", "km_rest", "tau1_rest", "tau2"]
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The given key_parameter_to_identify is not valid,"
            f" the given value is {key_parameter_to_identify[1]},"
            f" the available values are {default_keys}"
        ),
    ):
        DingModelFrequencyForceParameterIdentification(
            model=DingModelFrequency(),
            identification_method=identification_method,
            data_path=data_path,
            key_parameter_to_identify=key_parameter_to_identify,
        )

    key_parameter_to_identify = "a_rest"
    with pytest.raises(
        TypeError,
        match=re.escape(
            f"The given key_parameter_to_identify must be list type,"
            f" the given value is {type(key_parameter_to_identify)} type"
        ),
    ):
        DingModelFrequencyForceParameterIdentification(
            model=DingModelFrequency(),
            identification_method=identification_method,
            data_path=data_path,
            key_parameter_to_identify=key_parameter_to_identify,
        )

    model = DingModelFrequency()
    key_parameter_to_identify = ["a_rest", "km_rest", "tau1_rest", "tau2"]
    additional_key_settings = {
        "test": {
            "initial_guess": 1000,
            "min_bound": 1,
            "max_bound": 10000,
            "function": model.set_a_rest,
            "scaling": 1,
        }
    }
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The given additional_key_settings is not valid,"
            f" the given value is {'test'},"
            f" the available values are {default_keys}"
        ),
    ):
        DingModelFrequencyForceParameterIdentification(
            model=DingModelFrequency(),
            identification_method=identification_method,
            data_path=data_path,
            key_parameter_to_identify=key_parameter_to_identify,
            additional_key_settings=additional_key_settings,
        )

    additional_key_settings = {
        "a_rest": {
            "test": 1000,
            "min_bound": 1,
            "max_bound": 10000,
            "function": model.set_a_rest,
            "scaling": 1,
        }
    }
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The given additional_key_settings is not valid,"
            f" the given value is {'test'},"
            f" the available values are ['initial_guess', 'min_bound', 'max_bound', 'function', 'scaling']"
        ),
    ):
        DingModelFrequencyForceParameterIdentification(
            model=DingModelFrequency(),
            identification_method=identification_method,
            data_path=data_path,
            key_parameter_to_identify=key_parameter_to_identify,
            additional_key_settings=additional_key_settings,
        )

    additional_key_settings = {
        "a_rest": {
            "initial_guess": "test",
            "min_bound": 1,
            "max_bound": 10000,
            "function": model.set_a_rest,
            "scaling": 1,
        }
    }
    with pytest.raises(
        TypeError,
        match=re.escape(
            f"The given additional_key_settings value is not valid,"
            f" the given value is <class 'str'>,"
            f" the available type is <class 'int'>"
        ),
    ):
        DingModelFrequencyForceParameterIdentification(
            model=DingModelFrequency(),
            identification_method=identification_method,
            data_path=data_path,
            key_parameter_to_identify=key_parameter_to_identify,
            additional_key_settings=additional_key_settings,
        )

    additional_key_settings = "test"
    with pytest.raises(
        TypeError,
        match=re.escape(
            f"The given additional_key_settings must be dict type,"
            f" the given value is {type(additional_key_settings)} type"
        ),
    ):
        DingModelFrequencyForceParameterIdentification(
            model=DingModelFrequency(),
            identification_method=identification_method,
            data_path=data_path,
            key_parameter_to_identify=key_parameter_to_identify,
            additional_key_settings=additional_key_settings,
        )

    additional_key_settings = {}
    n_shooting = "10"
    with pytest.raises(
        TypeError,
        match=re.escape(f"The given n_shooting must be int type," f" the given value is {type(n_shooting)} type"),
    ):
        DingModelFrequencyForceParameterIdentification(
            model=DingModelFrequency(),
            identification_method=identification_method,
            data_path=data_path,
            key_parameter_to_identify=key_parameter_to_identify,
            additional_key_settings=additional_key_settings,
            n_shooting=n_shooting,
        )
