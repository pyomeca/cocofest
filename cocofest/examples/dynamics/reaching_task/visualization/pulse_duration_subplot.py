"""
This script is used to make the graph of the muscle force and fatigue for the reaching task.
The data used to make the graph is from the result file of the optimization.
The available graphs are: duration
"""

import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

pickle_path = [
    r"../result_file/pulse_duration_minimize_muscle_force.pkl",
    r"../result_file/pulse_duration_minimize_muscle_fatigue.pkl",
]

with open(pickle_path[0], "rb") as f:
    data_minimize_force = pickle.load(f)

with open(pickle_path[1], "rb") as f:
    data_minimize_fatigue = pickle.load(f)

pulse_duration_keys = list(data_minimize_fatigue["parameters"].keys())
muscle_names = ["BIClong", "BICshort", "TRIlong", "TRIlat", "TRImed", "BRA"]
nb_stim = len(data_minimize_fatigue["parameters"][pulse_duration_keys[0]])
width = round(data_minimize_fatigue["time"][-1], 2) / nb_stim

pw_data_list = [data_minimize_force["parameters"], data_minimize_fatigue["parameters"]]

pw_list = []
for j in range(2):
    pw_list.append([pw_data_list[j][pulse_duration_keys[i]] * 1000000 for i in range(len(pulse_duration_keys))])

plasma = cm = plt.get_cmap("plasma")
cNorm = colors.Normalize(vmin=100, vmax=600)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plasma)


def plot_graph(datas):
    fig, axs = plt.subplots(6, 1, figsize=(5, 3), constrained_layout=True)

    for i in range(len(pulse_duration_keys)):
        axs[i].set_xlim(left=0, right=1.5)
        plt.setp(
            axs[i],
            xticks=[0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4],
            xticklabels=[],
        )
        if i == len(pulse_duration_keys) - 1:
            plt.setp(
                axs[i],
                xticks=[0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4],
                xticklabels=[0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4],
            )
            axs[i].set_xlabel("Time (s)")

        for j in range(nb_stim):
            value = datas[i][j]

            color = scalarMap.to_rgba(value)
            axs[i].barh(muscle_names[i], width, left=j * width, height=0.5, color=color)

    fig.colorbar(scalarMap, ax=axs, orientation="vertical", label="Pulse duration (us)")
    plt.show()


for data in pw_list:
    plot_graph(data)
