import time
import pickle

from cocofest import (
    DingModelFrequencyWithFatigue,
    IvpFes,
)

# This is a sensitivity analysis, the associated graphs are available in the summation_truncation_graph example.

counter = 0
min_stim = 1
max_stim = 101
repetition = 100
modes = ["single", "doublet", "triplet"]
nb = (
    int((max_stim - min_stim) ** 2 / 2 + (max_stim - min_stim) / 2)
    * len(modes)
    * repetition
)
node_shooting = 1000
final_time = 1
for mode in modes:
    print("currently mode: " + mode)
    force_total_results = []
    calcium_total_results = []
    a_total_results = []
    km_total_results = []
    tau1_total_results = []
    computations_time = []
    computations_time_avg = []
    creation_ocp_time = []
    parameter_list = []
    if mode == "single":
        coefficient = 1
    elif mode == "doublet":
        coefficient = 2
    elif mode == "triplet":
        coefficient = 3
    else:
        raise RuntimeError("Mode not recognized")
    for i in range(min_stim, max_stim):
        force_results_per_frequency = []
        calcium_results_per_frequency = []
        a_results_per_frequency = []
        km_results_per_frequency = []
        tau1_results_per_frequency = []
        print("currently stimulation: " + str(i))
        n_stim = i * coefficient
        for j in range(1, i + 1):
            time_computation = []
            force = 0
            calcium = 0
            a = 0
            km = 0
            tau1 = 0
            temp_node_shooting = int(node_shooting / n_stim)

            ocp_start_time = time.time()
            fes_parameters = {
                "model": DingModelFrequencyWithFatigue(sum_stim_truncation=j),
                "n_stim": n_stim,
                "pulse_mode": mode,
            }
            ivp_parameters = {
                "n_shooting": temp_node_shooting,
                "final_time": 1,
                "use_sx": True,
            }
            ivp = IvpFes(
                fes_parameters=fes_parameters,
                ivp_parameters=ivp_parameters,
            )
            ocp_end_time = time.time()

            for k in range(repetition):
                start_time = time.time()

                # Integrating the solution
                result = ivp.integrate(return_time=False)

                time_computation.append(time.time() - start_time)

                if k == 0:
                    force = result["F"][0][-1]
                    calcium = result["Cn"][0][-1]
                    a = result["A"][0][-1]
                    km = result["Km"][0][-1]
                    tau1 = result["Tau1"][0][-1]

                counter += 1
                print(
                    "currently : "
                    + str(counter)
                    + "/"
                    + str(nb)
                    + " in "
                    + str(round(time_computation[-1], 4))
                    + "s"
                )

            time_computation_for_n_rep = sum(time_computation)
            time_computation_mean = sum(time_computation) / len(time_computation)
            time_ocp_creation = ocp_end_time - ocp_start_time

            computations_time.append(time_computation_for_n_rep)
            computations_time_avg.append(time_computation_mean)
            creation_ocp_time.append(time_ocp_creation)
            force_results_per_frequency.append(force)
            calcium_results_per_frequency.append(calcium)
            a_results_per_frequency.append(a)
            km_results_per_frequency.append(km)
            tau1_results_per_frequency.append(tau1)
            parameter_list.append([i, j])

        force_total_results.append(force_results_per_frequency)
        calcium_total_results.append(calcium_results_per_frequency)
        a_total_results.append(a_results_per_frequency)
        km_total_results.append(km_results_per_frequency)
        tau1_total_results.append(tau1_results_per_frequency)

    dictionary = {
        "parameter_list": parameter_list,
        "force_total_results": force_total_results,
        "calcium_total_results": calcium_total_results,
        "a_total_results": a_total_results,
        "km_total_results": km_total_results,
        "tau1_total_results": tau1_total_results,
        "computations_time": computations_time,
        "computations_time_avg": computations_time_avg,
        "creation_ocp_time": creation_ocp_time,
        "repetition": repetition,
    }

    if mode == "single":
        with open("truncation_single.pkl", "wb") as file:
            pickle.dump(dictionary, file)
    elif mode == "doublet":
        with open("truncation_doublet.pkl", "wb") as file:
            pickle.dump(dictionary, file)
    elif mode == "triplet":
        with open("truncation_triplet.pkl", "wb") as file:
            pickle.dump(dictionary, file)
