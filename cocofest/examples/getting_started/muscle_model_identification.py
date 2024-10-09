import pickle
import os

import numpy as np

from bioptim import SolutionMerge

from cocofest import (
    DingModelIntensityFrequency,
    DingModelIntensityFrequencyIntegrate,
    DingModelPulseIntensityFrequencyForceParameterIdentification,
    IvpFes,
)
from cocofest.identification.identification_method import full_data_extraction

import matplotlib.pyplot as plt

# Example n°5 : Identification of the parameters of the Ding model with the pulse intensity method for simulated data
# --- Simulating data --- #
# This problem was build to be integrated and has no objectives nor parameter to optimize.
n_shooting = 500
n_stim = 50
final_time = 5

stim_time = list(np.round(np.linspace(0, final_time, n_stim + 1), 2))[:-1]
pulse_intensity_values = [20, 20, 30, 40, 50, 60, 70, 80, 90, 100] * int((n_stim / 10))

fes_parameters = {
    "model": DingModelIntensityFrequencyIntegrate(),
    "pulse_intensity": pulse_intensity_values,
    "stim_time": stim_time,
}
ivp_parameters = {"n_shooting": n_shooting, "final_time": final_time, "use_sx": True}
ivp = IvpFes(fes_parameters, ivp_parameters)

# Integrating the solution
result, time = ivp.integrate()
force = result["F"][0]

pulse_intensity = pulse_intensity_values

dictionary = {"time": time, "force": force, "stim_time": stim_time, "pulse_intensity": pulse_intensity}

pickle_file_name = "../data/temp_identification_simulation.pkl"
with open(pickle_file_name, "wb") as file:
    pickle.dump(dictionary, file)

ocp = DingModelPulseIntensityFrequencyForceParameterIdentification(
    model=DingModelIntensityFrequency(sum_stim_truncation=10),
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
    n_shooting=n_shooting,
    final_time=final_time,
    use_sx=True,
    n_threads=6,
)

identified_parameters = ocp.force_model_identification()
force_ocp = ocp.force_identification_result.decision_states(to_merge=SolutionMerge.NODES)["F"][0]
a_rest = identified_parameters["a_rest"] if "a_rest" in identified_parameters else DingModelIntensityFrequency().a_rest
km_rest = (
    identified_parameters["km_rest"] if "km_rest" in identified_parameters else DingModelIntensityFrequency().km_rest
)
tau1_rest = (
    identified_parameters["tau1_rest"]
    if "tau1_rest" in identified_parameters
    else DingModelIntensityFrequency().tau1_rest
)
tau2 = identified_parameters["tau2"] if "tau2" in identified_parameters else DingModelIntensityFrequency().tau2
ar = identified_parameters["ar"] if "ar" in identified_parameters else DingModelIntensityFrequency().ar
bs = identified_parameters["bs"] if "bs" in identified_parameters else DingModelIntensityFrequency().bs
Is = identified_parameters["Is"] if "Is" in identified_parameters else DingModelIntensityFrequency().Is
cr = identified_parameters["cr"] if "cr" in identified_parameters else DingModelIntensityFrequency().cr
print(
    "a_rest : ",
    a_rest,
    "km_rest : ",
    km_rest,
    "tau1_rest : ",
    tau1_rest,
    "tau2 : ",
    tau2,
    "ar : ",
    ar,
    "bs : ",
    bs,
    "Is : ",
    Is,
    "cr : ",
    cr,
)

(
    pickle_time_data,
    pickle_stim_apparition_time,
    pickle_muscle_data,
    pickle_discontinuity_phase_list,
) = full_data_extraction([pickle_file_name])

# Plotting the identification result
plt.title("Force state result")
plt.plot(pickle_time_data, pickle_muscle_data, color="blue", label="simulated")
plt.plot(pickle_time_data, force_ocp, color="green", label="identified")
plt.xlabel("time (s)")
plt.ylabel("force (N)")

plt.annotate("a_rest : ", xy=(0.7, 0.4), xycoords="axes fraction", color="black")
plt.annotate("km_rest : ", xy=(0.7, 0.35), xycoords="axes fraction", color="black")
plt.annotate("tau1_rest : ", xy=(0.7, 0.3), xycoords="axes fraction", color="black")
plt.annotate("tau2 : ", xy=(0.7, 0.25), xycoords="axes fraction", color="black")
plt.annotate("ar : ", xy=(0.7, 0.2), xycoords="axes fraction", color="black")
plt.annotate("bs : ", xy=(0.7, 0.15), xycoords="axes fraction", color="black")
plt.annotate("Is : ", xy=(0.7, 0.1), xycoords="axes fraction", color="black")
plt.annotate("cr : ", xy=(0.7, 0.05), xycoords="axes fraction", color="black")

plt.annotate(str(round(a_rest, 5)), xy=(0.78, 0.4), xycoords="axes fraction", color="red")
plt.annotate(str(round(km_rest, 5)), xy=(0.78, 0.35), xycoords="axes fraction", color="red")
plt.annotate(str(round(tau1_rest, 5)), xy=(0.78, 0.3), xycoords="axes fraction", color="red")
plt.annotate(str(round(tau2, 5)), xy=(0.78, 0.25), xycoords="axes fraction", color="red")
plt.annotate(str(round(ar, 5)), xy=(0.78, 0.2), xycoords="axes fraction", color="red")
plt.annotate(str(round(bs, 5)), xy=(0.78, 0.15), xycoords="axes fraction", color="red")
plt.annotate(str(round(Is, 5)), xy=(0.78, 0.1), xycoords="axes fraction", color="red")
plt.annotate(str(round(cr, 5)), xy=(0.78, 0.05), xycoords="axes fraction", color="red")

plt.annotate(str(DingModelIntensityFrequency().a_rest), xy=(0.85, 0.4), xycoords="axes fraction", color="blue")
plt.annotate(str(DingModelIntensityFrequency().km_rest), xy=(0.85, 0.35), xycoords="axes fraction", color="blue")
plt.annotate(str(DingModelIntensityFrequency().tau1_rest), xy=(0.85, 0.3), xycoords="axes fraction", color="blue")
plt.annotate(str(DingModelIntensityFrequency().tau2), xy=(0.85, 0.25), xycoords="axes fraction", color="blue")
plt.annotate(str(DingModelIntensityFrequency().ar), xy=(0.85, 0.2), xycoords="axes fraction", color="blue")
plt.annotate(str(DingModelIntensityFrequency().bs), xy=(0.85, 0.15), xycoords="axes fraction", color="blue")
plt.annotate(str(DingModelIntensityFrequency().Is), xy=(0.85, 0.1), xycoords="axes fraction", color="blue")
plt.annotate(str(DingModelIntensityFrequency().cr), xy=(0.85, 0.05), xycoords="axes fraction", color="blue")

# --- Delete the temp file ---#
os.remove(f"../data/temp_identification_simulation.pkl")

plt.legend()
plt.show()


# Example n°1 : Identification of the parameters of the Ding model with the frequency method for experimental data
"""
ocp = DingModelFrequencyParameterIdentification(
    model=DingModelFrequency,
    force_model_data_path=["data/biceps_force.pkl"],
    force_model_identification_method="full",
    force_model_double_step_identification=True,
    use_sx=True,
)

a_rest, km_rest, tau1_rest, tau2 = ocp.force_model_identification()
print("a_rest : ", a_rest, "km_rest : ", km_rest, "tau1_rest : ", tau1_rest, "tau2 : ", tau2)
"""

# # Example n°2 : Identification of the parameters of the Ding model with the frequency method for simulated data
# # --- Simulating data --- #
# # This problem was build to be integrated and has no objectives nor parameter to optimize.

# ivp = IvpFes(
#     model=DingModelFrequency(),
#     n_stim=10,
#     n_shooting=10,
#     final_time=1,
#     use_sx=True,
# )
#
# # Creating the solution from the initial guess
# sol_from_initial_guess = Solution.from_initial_guess(ivp, [ivp.x_init, ivp.u_init, ivp.p_init, ivp.s_init])
#
# # Integrating the solution
# result = sol_from_initial_guess.integrate(
#     shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP, merge_phases=True
# )
#
# force = result.states["F"].tolist()
# time = [result.time.tolist()]
# stim_temp = [0 if i == 0 else result.ocp.nlp[i].tf for i in range(len(result.ocp.nlp))]
# stim = [sum(stim_temp[: i + 1]) for i in range(len(stim_temp))]
#
# dictionary = {
#     "time": time,
#     "biceps": force,
#     "stim_time": stim,
# }
#
# pickle_file_name = "../data/temp_identification_simulation.pkl"
# with open(pickle_file_name, "wb") as file:
#     pickle.dump(dictionary, file)
#
# # ocp = DingModelFrequencyParameterIdentification(
# #     model=DingModelFrequency(),
# #     force_model_data_path=[pickle_file_name],
# #     force_model_identification_method="full",
# #     force_model_double_step_identification=False,
# #     n_shooting=100,
# #     use_sx=True,
# # )
#
# ocp = DingModelFrequencyForceParameterIdentification(
#     model=DingModelFrequency(),
#     data_path=[pickle_file_name],
#     identification_method="full",
#     double_step_identification=False,
#     key_parameter_to_identify=["km_rest", "tau1_rest", "tau2"],
#     additional_key_settings={},
#     n_shooting=10,
#     a_rest=2500,
#     use_sx=True,
# )
#
# identified_parameters = ocp.force_model_identification()
# a_rest = identified_parameters["a_rest"] if "a_rest" in identified_parameters else 2500  #TODO : correct this
# km_rest = identified_parameters["km_rest"] if "km_rest" in identified_parameters else DingModelFrequency().km_rest
# tau1_rest = identified_parameters["tau1_rest"] if "tau1_rest" in identified_parameters else DingModelFrequency().tau1_rest
# tau2 = identified_parameters["tau2"] if "tau2" in identified_parameters else DingModelFrequency().tau2
# print("a_rest : ", a_rest, "km_rest : ", km_rest, "tau1_rest : ", tau1_rest, "tau2 : ", tau2)
#
# identified_model = DingModelFrequency()
# identified_model.a_rest = a_rest
# identified_model.km_rest = km_rest
# identified_model.tau1_rest = tau1_rest
# identified_model.tau2 = tau2
#
# identified_force_list = []
# identified_time_list = []
#
# ivp_from_identification = IvpFes(
#     model=identified_model,
#     n_stim=10,
#     n_shooting=100,
#     final_time=1,
#     use_sx=True,
# )
#
# # Creating the solution from the initial guess
# identified_sol_from_initial_guess = Solution.from_initial_guess(
#     ivp_from_identification,
#     [
#         ivp_from_identification.x_init,
#         ivp_from_identification.u_init,
#         ivp_from_identification.p_init,
#         ivp_from_identification.s_init,
#     ],
# )
#
# # Integrating the solution
# identified_result = identified_sol_from_initial_guess.integrate(
#     shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP, merge_phases=True
# )
#
# identified_time = identified_result.time.tolist()
# identified_force = identified_result.states["F"][0]
#
# (
#     pickle_time_data,
#     pickle_stim_apparition_time,
#     pickle_muscle_data,
#     pickle_discontinuity_phase_list,
# ) = DingModelFrequencyForceParameterIdentification.full_data_extraction([pickle_file_name])
#
# # Plotting the identification result
# plt.title("Force state result")
# plt.plot(pickle_time_data, pickle_muscle_data, color="blue", label="simulated")
# plt.plot(identified_time, identified_force, color="red", label="identified")
# plt.xlabel("time (s)")
# plt.ylabel("force (N)")
#
# plt.annotate("a_rest : ", xy=(0.7, 0.25), xycoords="axes fraction", color="black")
# plt.annotate("km_rest : ", xy=(0.7, 0.20), xycoords="axes fraction", color="black")
# plt.annotate("tau1_rest : ", xy=(0.7, 0.15), xycoords="axes fraction", color="black")
# plt.annotate("tau2 : ", xy=(0.7, 0.10), xycoords="axes fraction", color="black")
#
# plt.annotate(str(round(a_rest, 5)), xy=(0.78, 0.25), xycoords="axes fraction", color="red")
# plt.annotate(str(round(km_rest, 5)), xy=(0.78, 0.20), xycoords="axes fraction", color="red")
# plt.annotate(str(round(tau1_rest, 5)), xy=(0.78, 0.15), xycoords="axes fraction", color="red")
# plt.annotate(str(round(tau2, 5)), xy=(0.78, 0.10), xycoords="axes fraction", color="red")
#
# plt.annotate(str(3009), xy=(0.85, 0.25), xycoords="axes fraction", color="blue")
# plt.annotate(str(0.103), xy=(0.85, 0.20), xycoords="axes fraction", color="blue")
# plt.annotate(str(0.050957), xy=(0.85, 0.15), xycoords="axes fraction", color="blue")
# plt.annotate(str(0.060), xy=(0.85, 0.10), xycoords="axes fraction", color="blue")
#
# # --- Delete the temp file ---#
# os.remove(f"../data/temp_identification_simulation.pkl")
#
# plt.legend()
# plt.show()


# Example n°3 : Identification of the fatigue model parameters based on the Ding model
# with the frequency method for simulated data
# /!\ This example is not working yet because it is too heavy to compute /!\
"""
# --- Simulating data --- #
ocp = DingModelFrequencyParameterIdentification(
    model=DingModelFrequency,
    fatigue_model_data_path=["../data/simulated_fatigue_trial.pkl"],
    a_rest=DingModelFrequency().a_rest,
    km_rest=DingModelFrequency().km_rest,
    tau1_rest=DingModelFrequency().tau1_rest,
    tau2=DingModelFrequency().tau2,
    n_shooting=5,
    use_sx=True,
)

alpha_a, alpha_km, alpha_tau1, tau_fat = ocp.fatigue_model_identification()
print("alpha_a : ", alpha_a, "alpha_km : ", alpha_km, "alpha_tau1 : ", alpha_tau1, "tau_fat : ", tau_fat)
"""

#
# # Example n°4 : Identification of the parameters of the Ding model with the pulse duration method for simulated data
# # --- Simulating data --- #
# # This problem was build to be integrated and has no objectives nor parameter to optimize.
# pulse_duration_values = [0.000180, 0.0002, 0.000250, 0.0003, 0.000350, 0.0004, 0.000450, 0.0005, 0.000550, 0.0006]
# ivp = IvpFes(
#     model=DingModelPulseDurationFrequency(),
#     n_stim=10,
#     n_shooting=10,
#     final_time=1,
#     use_sx=True,
#     pulse_duration=pulse_duration_values,
# )
#
# # Creating the solution from the initial guess
# sol_from_initial_guess = Solution.from_initial_guess(ivp, [ivp.x_init, ivp.u_init, ivp.p_init, ivp.s_init])
#
# # Integrating the solution
# result = sol_from_initial_guess.integrate(
#     shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP, merge_phases=True
# )
#
# force = result.states["F"].tolist()
# time = [result.time.tolist()]
# stim_temp = [0 if i == 0 else result.ocp.nlp[i].tf for i in range(len(result.ocp.nlp))]
# stim = [sum(stim_temp[: i + 1]) for i in range(len(stim_temp))]
# pulse_duration = pulse_duration_values
#
# dictionary = {
#     "time": time,
#     "biceps": force,
#     "stim_time": stim,
#     "pulse_duration": pulse_duration,
# }
#
# pickle_file_name = "../data/temp_identification_simulation.pkl"
# with open(pickle_file_name, "wb") as file:
#     pickle.dump(dictionary, file)
#
# ocp = DingModelPulseDurationFrequencyForceParameterIdentification(
#     model=DingModelPulseDurationFrequency(),
#     data_path=[pickle_file_name],
#     identification_method="full",
#     double_step_identification=False,
#     key_parameter_to_identify=["tau1_rest", "tau2", "km_rest", "a_scale", "pd0", "pdt"],
#     additional_key_settings={},
#     n_shooting=10,
#     use_sx=True,
# )
#
# identified_parameters = ocp.force_model_identification()
# a_scale = identified_parameters["a_scale"] if "a_scale" in identified_parameters else DingModelPulseDurationFrequency().a_scale
# pd0 = identified_parameters["pd0"] if "pd0" in identified_parameters else DingModelPulseDurationFrequency().pd0
# pdt = identified_parameters["pdt"] if "pdt" in identified_parameters else DingModelPulseDurationFrequency().pdt
# km_rest = identified_parameters["km_rest"] if "km_rest" in identified_parameters else DingModelPulseDurationFrequency().km_rest
# tau1_rest = identified_parameters["tau1_rest"] if "tau1_rest" in identified_parameters else DingModelPulseDurationFrequency().tau1_rest
# tau2 = identified_parameters["tau2"] if "tau2" in identified_parameters else DingModelPulseDurationFrequency().tau2
# print("a_scale : ", a_scale, "pd0 : ", pd0, "pdt : ", pdt,  "km_rest : ", km_rest, "tau1_rest : ", tau1_rest, "tau2 : ", tau2)
#
# identified_model = DingModelPulseDurationFrequency()
# identified_model.a_scale = a_scale
# identified_model.km_rest = km_rest
# identified_model.tau1_rest = tau1_rest
# identified_model.tau2 = tau2
# identified_model.pd0 = pd0
# identified_model.pdt = pdt
#
# identified_force_list = []
# identified_time_list = []
#
# ivp_from_identification = IvpFes(
#     model=identified_model,
#     n_stim=10,
#     n_shooting=10,
#     final_time=1,
#     use_sx=True,
#     pulse_duration=[0.000184, 0.0002, 0.000250, 0.0003, 0.000350, 0.0004, 0.000450, 0.0005, 0.000550, 0.0006],
# )
#
# # Creating the solution from the initial guess
# identified_sol_from_initial_guess = Solution.from_initial_guess(
#     ivp_from_identification,
#     [
#         ivp_from_identification.x_init,
#         ivp_from_identification.u_init,
#         ivp_from_identification.p_init,
#         ivp_from_identification.s_init,
#     ],
# )
#
# # Integrating the solution
# identified_result = identified_sol_from_initial_guess.integrate(
#     shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP, merge_phases=True
# )
#
# identified_time = identified_result.time.tolist()
# identified_force = identified_result.states["F"][0]
#
# (
#     pickle_time_data,
#     pickle_stim_apparition_time,
#     pickle_muscle_data,
#     pickle_discontinuity_phase_list,
# ) = DingModelPulseDurationFrequencyForceParameterIdentification.full_data_extraction([pickle_file_name])
#
# # Plotting the identification result
# plt.title("Force state result")
# plt.plot(pickle_time_data, pickle_muscle_data, color="blue", label="simulated")
# plt.plot(identified_time, identified_force, color="red", label="identified")
# plt.xlabel("time (s)")
# plt.ylabel("force (N)")
#
# plt.annotate("a_scale : ", xy=(0.7, 0.25), xycoords="axes fraction", color="black")
# plt.annotate("km_rest : ", xy=(0.7, 0.20), xycoords="axes fraction", color="black")
# plt.annotate("tau1_rest : ", xy=(0.7, 0.15), xycoords="axes fraction", color="black")
# plt.annotate("tau2 : ", xy=(0.7, 0.10), xycoords="axes fraction", color="black")
# plt.annotate("pd0 : ", xy=(0.7, 0.05), xycoords="axes fraction", color="black")
# plt.annotate("pdt : ", xy=(0.7, 0.0), xycoords="axes fraction", color="black")
#
#
# plt.annotate(str(round(a_scale, 5)), xy=(0.78, 0.25), xycoords="axes fraction", color="red")
# plt.annotate(str(round(km_rest, 5)), xy=(0.78, 0.20), xycoords="axes fraction", color="red")
# plt.annotate(str(round(tau1_rest, 5)), xy=(0.78, 0.15), xycoords="axes fraction", color="red")
# plt.annotate(str(round(tau2, 5)), xy=(0.78, 0.10), xycoords="axes fraction", color="red")
# plt.annotate(str(round(pd0, 9)), xy=(0.78, 0.05), xycoords="axes fraction", color="red")
# plt.annotate(str(round(pdt, 9)), xy=(0.78, 0.0), xycoords="axes fraction", color="red")
#
# plt.annotate(str(4920), xy=(0.85, 0.25), xycoords="axes fraction", color="blue")
# plt.annotate(str(0.137), xy=(0.85, 0.20), xycoords="axes fraction", color="blue")
# plt.annotate(str(0.060601), xy=(0.85, 0.15), xycoords="axes fraction", color="blue")
# plt.annotate(str(0.001), xy=(0.85, 0.10), xycoords="axes fraction", color="blue")
# plt.annotate(str(0.000131405), xy=(0.85, 0.05), xycoords="axes fraction", color="blue")
# plt.annotate(str(0.000194138), xy=(0.85, 0.0), xycoords="axes fraction", color="blue")
#
# # --- Delete the temp file ---#
# os.remove(f"../data/temp_identification_simulation.pkl")
#
# plt.legend()
# plt.show()
