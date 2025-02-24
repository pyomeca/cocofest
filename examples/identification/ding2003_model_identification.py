"""
THIS IS NOT CONVERGING TO THE EXPECTED RESULT - SEE ISSUE #11
--
This example demonstrates the way of identifying the Ding 2003 model parameter using noisy simulated data.
First we integrate the model with a given parameter set. Then we add noise to the previously calculated force output.
Finally, we use the noisy data to identify the model parameters. It is possible to lock a_rest to an arbitrary value but
you need to remove it from the key_parameter_to_identify.
"""

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

from bioptim import SolutionMerge

from cocofest import (
    DingModelFrequency,
    DingModelFrequencyForceParameterIdentification,
    IvpFes,
)
from cocofest.identification.identification_method import full_data_extraction


def simulate_data(final_time=2, stim_time=None):
    ivp_model = DingModelFrequency(stim_time=stim_time, sum_stim_truncation=10)
    fes_parameters = {"model": ivp_model}
    ivp_parameters = {"final_time": final_time, "use_sx": True}

    # --- Creating the simulated data to identify on --- #
    # Building the Initial Value Problem
    ivp = IvpFes(fes_parameters, ivp_parameters)

    # Integrating the solution
    result, time = ivp.integrate()

    # Adding noise to the force
    noise = np.random.normal(0, 5, len(result["F"][0]))
    force = result["F"][0]  # + noise

    # Saving the data in a pickle file
    dictionary = {"time": time, "force": force, "stim_time": stim_time}

    pickle_file_name = "../data/temp_identification_simulation.pkl"
    with open(pickle_file_name, "wb") as file:
        pickle.dump(dictionary, file)


def prepare_ocp():
    # --- Identifying the model parameters --- #
    final_time = 2
    stim_time = list(np.linspace(0, 1, 11)[:-1])
    pickle_file_name = simulate_data(final_time=final_time, stim_time=stim_time)

    ocp_model = DingModelFrequency(stim_time=stim_time)
    return (
        DingModelFrequencyForceParameterIdentification(
            model=ocp_model,
            final_time=final_time,
            control_max_bound=None,
            data_path=[pickle_file_name],
            identification_method="full",
            double_step_identification=False,
            key_parameter_to_identify=["a_rest", "km_rest", "tau1_rest", "tau2"],
            additional_key_settings={},
            use_sx=True,
            n_threads=6,
        ),
        pickle_file_name,
    )


def main():
    ocp, pickle_file_name = prepare_ocp()
    identified_parameters = ocp.force_model_identification()
    force_ocp = ocp.force_identification_result.stepwise_states(to_merge=SolutionMerge.NODES)["F"][0]
    time_ocp = ocp.force_identification_result.stepwise_time(to_merge=SolutionMerge.NODES).T[0]
    print(identified_parameters)

    (
        pickle_time_data,
        pickle_stim_apparition_time,
        pickle_muscle_data,
        pickle_discontinuity_phase_list,
    ) = full_data_extraction([pickle_file_name])

    result_dict = {
        "a_rest": [identified_parameters["a_rest"], DingModelFrequency().a_rest],
        "km_rest": [identified_parameters["km_rest"], DingModelFrequency().km_rest],
        "tau1_rest": [identified_parameters["tau1_rest"], DingModelFrequency().tau1_rest],
        "tau2": [identified_parameters["tau2"], DingModelFrequency().tau2],
    }

    # Plotting the identification result
    plt.title("Force state result")
    plt.plot(pickle_time_data, pickle_muscle_data, color="blue", label="simulated")
    plt.plot(time_ocp, force_ocp, color="red", label="identified")

    plt.xlabel("time (s)")
    plt.ylabel("force (N)")

    y_pos = 0.85
    for key, value in result_dict.items():
        plt.annotate(f"{key} : ", xy=(0.7, y_pos), xycoords="axes fraction", color="black")
        plt.annotate(str(round(value[0], 5)), xy=(0.78, y_pos), xycoords="axes fraction", color="red")
        plt.annotate(str(round(value[1], 5)), xy=(0.85, y_pos), xycoords="axes fraction", color="blue")
        y_pos -= 0.05

    # --- Delete the temp file ---#
    os.remove(f"../data/temp_identification_simulation.pkl")

    plt.legend()
    plt.show()
