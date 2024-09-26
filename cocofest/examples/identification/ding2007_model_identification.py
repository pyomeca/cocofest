"""
This example demonstrates the way of identifying the Ding 2007 model parameter using noisy simulated data.
First we integrate the model with a given parameter set. Then we add noise to the previously calculated force output.
Finally, we use the noisy data to identify the model parameters.
"""

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

from bioptim import SolutionMerge

from cocofest import (
    DingModelPulseDurationFrequency,
    DingModelPulseDurationFrequencyIntegrate,
    DingModelPulseDurationFrequencyForceParameterIdentification,
    IvpFes,
)
from cocofest.identification.identification_method import full_data_extraction


# --- Setting simulation parameters --- #
stim_time = np.round(np.linspace(0, 1, 11)[:-1], 2)
pulse_duration = np.random.uniform(0.0002, 0.0006, 10).tolist()

n_shooting = 200
final_time = 2
ivp_model = DingModelPulseDurationFrequencyIntegrate()
fes_parameters = {"model": ivp_model, "stim_time": stim_time, "pulse_duration": pulse_duration}
ivp_parameters = {"n_shooting": n_shooting, "final_time": final_time, "use_sx": True}

# --- Creating the simulated data to identify on --- #
# Building the Initial Value Problem
ivp = IvpFes(fes_parameters, ivp_parameters)

# Integrating the solution
result, time = ivp.integrate()

# Adding noise to the force
noise = np.random.normal(0, 5, len(result["F"][0]))
force = result["F"][0] + noise

# Saving the data in a pickle file
dictionary = {"time": time, "force": force, "stim_time": stim_time, "pulse_duration": pulse_duration}

pickle_file_name = "../data/temp_identification_simulation.pkl"
with open(pickle_file_name, "wb") as file:
    pickle.dump(dictionary, file)

# --- Identifying the model parameters --- #
ocp_model = DingModelPulseDurationFrequency()
ocp = DingModelPulseDurationFrequencyForceParameterIdentification(
    model=ocp_model,
    data_path=[pickle_file_name],
    identification_method="full",
    double_step_identification=False,
    key_parameter_to_identify=["tau1_rest", "tau2", "km_rest", "a_scale", "pd0", "pdt"],
    additional_key_settings={},
    n_shooting=n_shooting,
    final_time=final_time,
    use_sx=True,
    n_threads=6,
)

identified_parameters = ocp.force_model_identification()
force_ocp = ocp.force_identification_result.decision_states(to_merge=SolutionMerge.NODES)["F"][0]
print(identified_parameters)

(
    pickle_time_data,
    pickle_stim_apparition_time,
    pickle_muscle_data,
    pickle_discontinuity_phase_list,
) = full_data_extraction([pickle_file_name])

result_dict = {"tau1_rest": [identified_parameters["tau1_rest"],DingModelPulseDurationFrequency().tau1_rest],
               "tau2": [identified_parameters["tau2"], DingModelPulseDurationFrequency().tau2],
               "km_rest": [identified_parameters["km_rest"], DingModelPulseDurationFrequency().km_rest],
               "a_scale": [identified_parameters["a_scale"], DingModelPulseDurationFrequency().a_scale],
               "pd0": [identified_parameters["pd0"], DingModelPulseDurationFrequency().pd0],
               "pdt": [identified_parameters["pdt"], DingModelPulseDurationFrequency().pdt]}

# Plotting the identification result
plt.title("Force state result")
plt.plot(pickle_time_data, pickle_muscle_data, "-.", color="blue", label="simulated")
plt.plot(pickle_time_data, force_ocp, color="red", label="identified")

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
