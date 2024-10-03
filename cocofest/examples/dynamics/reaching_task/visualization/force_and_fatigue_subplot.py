"""
This script is used to make the graph of the muscle force and fatigue for the reaching task.
The data used to make the graph is from the result file of the optimization.
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

pickle_path = [
    r"../result_file/pulse_duration_minimize_muscle_force.pkl",
    r"../result_file/pulse_duration_minimize_muscle_fatigue.pkl",
]

with open(pickle_path[0], "rb") as f:
    data_minimize_force = pickle.load(f)

with open(pickle_path[1], "rb") as f:
    data_minimize_fatigue = pickle.load(f)

force_muscle_keys = [
    "F_BIClong",
    "F_BICshort",
    "F_TRIlong",
    "F_TRIlat",
    "F_TRImed",
    "F_BRA",
]
fatigue_muscle_keys = [
    "A_BIClong",
    "A_BICshort",
    "A_TRIlong",
    "A_TRIlat",
    "A_TRImed",
    "A_BRA",
]
muscle_names = ["BIClong", "BICshort", "TRIlong", "TRIlat", "TRImed", "BRA"]

# Force graph
fig, axs = plt.subplots(3, 2, figsize=(6, 7))
fig.suptitle("Muscle force", fontsize=20, fontweight="bold", fontname="Times New Roman")
index = 0

for j in range(2):
    for i in range(3):
        axs[i][j].set_xlim(left=0, right=1.5)
        axs[i][j].set_ylim(bottom=0, top=250)
        if j == 0:
            plt.setp(
                axs[i][j],
                xticks=[0, 0.5, 1, 1.5],
                xticklabels=[],
                yticks=[0, 75, 150, 225],
                yticklabels=[0, 75, 150, 225],
            )

        else:
            plt.setp(
                axs[i][j],
                xticks=[0, 0.5, 1, 1.5],
                xticklabels=[],
                yticks=[0, 75, 150, 225],
                yticklabels=[],
            )

        if i == 2:
            plt.setp(
                axs[i][j],
                xticks=[0, 0.5, 1, 1.5],
                xticklabels=[0, 0.5, 1, 1.5],
            )

        axs[i][j].plot(
            data_minimize_force["time"],
            data_minimize_force["states"][force_muscle_keys[index]],
            lw=5,
        )
        axs[i][j].plot(
            data_minimize_fatigue["time"],
            data_minimize_fatigue["states"][force_muscle_keys[index]],
            lw=5,
        )
        axs[i][j].text(
            0.5,
            0.9,
            f"{muscle_names[index]}",
            transform=axs[i][j].transAxes,
            ha="center",
            va="center",
            fontsize=14,
            weight="bold",
            font="Times New Roman",
        )

        labels = axs[i][j].get_xticklabels() + axs[i][j].get_yticklabels()
        [label.set_fontname("Times New Roman") for label in labels]
        [label.set_fontsize(14) for label in labels]

        index += 1

fig.text(
    0.5,
    0.02,
    "Time (s)",
    ha="center",
    va="center",
    fontsize=18,
    weight="bold",
    font="Times New Roman",
)
fig.text(
    0.025,
    0.5,
    "Force (N)",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=18,
    weight="bold",
    font="Times New Roman",
)
fig.legend(
    ["Force", "Fatigue"],
    loc="upper right",
    ncol=1,
    prop={"family": "Times New Roman", "size": 14, "weight": "bold"},
)
plt.show()

# Joint angle graph
joint_keys = ["Shoulder", "Elbow"]
fig, axs = plt.subplots(2, 1, figsize=(3, (2 / 3) * 7))
fig.suptitle("Joint angle", fontsize=20, fontweight="bold", fontname="Times New Roman")

for i in range(2):
    axs[i].set_xlim(left=0, right=1.5)

    if i == 1:
        plt.setp(
            axs[i],
            xticks=[0, 0.5, 1, 1.5],
            xticklabels=[0, 0.5, 1, 1.5],
        )
    else:
        plt.setp(
            axs[i],
            xticks=[0, 0.5, 1, 1.5],
            xticklabels=[],
        )

    force_angles = data_minimize_force["states"]["q"][i] * 180 / 3.14
    fatigue_angles = data_minimize_fatigue["states"]["q"][i] * 180 / 3.14

    axs[i].plot(data_minimize_force["time"], force_angles, lw=5)
    axs[i].plot(data_minimize_fatigue["time"], fatigue_angles, lw=5)
    axs[i].text(
        0.05,
        0.9,
        f"{joint_keys[i]}",
        transform=axs[i].transAxes,
        ha="left",
        va="center",
        fontsize=14,
        weight="bold",
        font="Times New Roman",
    )
    labels = axs[i].get_xticklabels() + axs[i].get_yticklabels()
    [label.set_fontname("Times New Roman") for label in labels]
    [label.set_fontsize(14) for label in labels]

fig.text(
    0.05,
    0.5,
    "Joint angle (°)",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=18,
    weight="bold",
    font="Times New Roman",
)
axs[1].text(
    0.75,
    -25,
    "Time (s)",
    ha="center",
    va="center",
    fontsize=18,
    weight="bold",
    font="Times New Roman",
)
fig.legend(
    ["Force", "Fatigue"],
    loc="upper right",
    ncol=1,
    prop={"family": "Times New Roman", "size": 14, "weight": "bold"},
)
plt.show()

# Fatigue graph
a_list = ["A_BIClong", "A_BICshort", "A_TRIlong", "A_TRIlat", "A_TRImed", "A_BRA"]
a_sum_base_line = 0
a_force_sum_list = []
a_fatigue_sum_list = []
for key_a in a_list:
    a_sum_base_line += data_minimize_force["states"][key_a][0]
for i in range(len(data_minimize_force["time"])):
    a_force_sum = 0
    a_fatigue_sum = 0
    for key_a in a_list:
        a_force_sum += data_minimize_force["states"][key_a][i]
        a_fatigue_sum += data_minimize_fatigue["states"][key_a][i]

    a_force_sum_list.append(a_force_sum)
    a_fatigue_sum_list.append(a_fatigue_sum)

a_force_diff_list = []
a_fatigue_diff_list = []
fatigue_minimization_percentage_gain_list = []
for i in range(len(data_minimize_force["time"])):
    a_force_diff_list.append((a_force_sum_list[i] - a_force_sum_list[0]) * 1000)
    a_fatigue_diff_list.append((a_fatigue_sum_list[i] - a_fatigue_sum_list[0]) * 1000)

    fatigue_minimization_percentage_gain_list.append(
        (a_fatigue_sum_list[i] - a_force_sum_list[i]) / (a_force_sum_list[0] - a_force_sum_list[-1]) * 100
    )

fig, axs = plt.subplots(1, 1, figsize=(3, (1 / 3) * 7))
fig.suptitle("Muscle fatigue", fontsize=20, fontweight="bold")

axs.set_xlim(left=0, right=1.5)
plt.setp(
    axs,
    xticks=[0, 0.5, 1, 1.5],
    xticklabels=[0, 0.5, 1, 1.5],
)

a_force_sum_percentage = (np.array(a_force_sum_list) / a_sum_base_line) * 100
a_fatigue_sum_percentage = (np.array(a_fatigue_sum_list) / a_sum_base_line) * 100

axs.plot(
    data_minimize_force["time"],
    a_force_sum_percentage,
    lw=5,
    label="Minimize force production",
)
axs.plot(
    data_minimize_force["time"],
    a_fatigue_sum_percentage,
    lw=5,
    label="Maximize muscle capacity",
)

axs.set_xlim(left=0, right=1.5)

plt.setp(
    axs,
    xticks=[0, 0.5, 1, 1.5],
    xticklabels=[0, 0.5, 1, 1.5],
)

labels = axs.get_xticklabels() + axs.get_yticklabels()
fig.text(
    0.05,
    0.5,
    "Muscle capacity (%)",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=18,
    weight="bold",
)
axs.text(0.75, 96.3, "Time (s)", ha="center", va="center", fontsize=18, weight="bold")
axs.legend(title="Cost function", fontsize="medium", loc="upper right", ncol=1)
plt.show()
