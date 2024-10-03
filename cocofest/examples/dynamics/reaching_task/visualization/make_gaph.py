"""
This script is used to make the graph of the muscle force and fatigue for the reaching task.
The data used to make the graph is from the result file of the optimization.
The available graphs are: duration
"""

import pickle
import matplotlib.pyplot as plt

chosen_graph_to_plot = "duration"

duration_path = [
    r"../result_file/pulse_duration_minimize_muscle_force.pkl",
    r"../result_file/pulse_duration_minimize_muscle_fatigue.pkl",
]

chosen_graph_to_plot_path = duration_path if chosen_graph_to_plot == "duration" else None


if chosen_graph_to_plot_path is None:
    raise ValueError("The chosen graph to plot is not available")

with open(chosen_graph_to_plot_path[0], "rb") as f:
    data_minimize_force = pickle.load(f)

with open(chosen_graph_to_plot_path[1], "rb") as f:
    data_minimize_fatigue = pickle.load(f)

force_muscle_keys = [
    "F_BIClong",
    "F_BICshort",
    "F_TRIlong",
    "F_TRIlat",
    "F_TRImed",
    "F_BRA",
]
muscle_names = ["BIClong", "BICshort", "TRIlong", "TRIlat", "TRImed", "BRA"]
muscle_title_x_postiton = [0.55, 0.5, 0.56, 0.62, 0.55, 0.73]
fig, axs = plt.subplots(3, 3, figsize=(5, 3), constrained_layout=True)
index = 0

# Force across time
for i in range(2):
    for j in range(3):
        axs[i][j].set_xlim(left=0, right=1.5)
        axs[i][j].set_ylim(bottom=0, top=250)

        axs[i][j].text(
            muscle_title_x_postiton[index],
            0.975,
            f"{muscle_names[index]}",
            transform=axs[i][j].transAxes,
            ha="left",
            va="top",
            weight="bold",
            font="Times New Roman",
        )

        labels = axs[i][j].get_xticklabels() + axs[i][j].get_yticklabels()
        [label.set_fontname("Times New Roman") for label in labels]
        [label.set_fontsize(14) for label in labels]

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

        if i == 0 and j == 0:
            axs[i][j].plot(
                data_minimize_force["time"],
                data_minimize_force["states"][force_muscle_keys[index]],
                ms=4,
                linewidth=5.0,
                label="Minimizing force",
            )
            axs[i][j].plot(
                data_minimize_fatigue["time"],
                data_minimize_fatigue["states"][force_muscle_keys[index]],
                ms=4,
                linewidth=5.0,
                label="Minimizing fatigue",
            )
        else:
            axs[i][j].plot(
                data_minimize_force["time"],
                data_minimize_force["states"][force_muscle_keys[index]],
                ms=4,
                linewidth=5.0,
            )
            axs[i][j].plot(
                data_minimize_fatigue["time"],
                data_minimize_fatigue["states"][force_muscle_keys[index]],
                ms=4,
                linewidth=5.0,
            )
        index += 1

# Joint angle across time
q_names = ["q_arm", "q_elbow"]
q_x_position = [0.65, 0.57]
for i in range(2):
    axs[2][i].set_xlim(left=0, right=1.5)
    axs[2][i].set_ylim(bottom=-1, top=2.5)

    axs[2][i].text(
        q_x_position[i],
        0.975,
        f"{q_names[i]}",
        transform=axs[2][i].transAxes,
        ha="left",
        va="top",
        weight="bold",
        font="Times New Roman",
    )

    labels = axs[2][i].get_xticklabels() + axs[2][i].get_yticklabels()
    [label.set_fontname("Times New Roman") for label in labels]
    [label.set_fontsize(14) for label in labels]

    if i == 0:
        plt.setp(
            axs[2][i],
            xticks=[0, 0.5, 1, 1.5],
            xticklabels=[0, 0.5, 1, 1.5],
            yticks=[-2, -1, 0, 1, 2],
            yticklabels=[-2, -1, 0, 1, 2],
        )

    else:
        plt.setp(
            axs[2][i],
            xticks=[0, 0.5, 1, 1.5],
            xticklabels=[0, 0.5, 1, 1.5],
            yticks=[-1, 0, 1, 2],
            yticklabels=[],
        )

    axs[2][i].plot(
        data_minimize_force["time"],
        data_minimize_force["states"]["q"][i],
        ms=4,
        linewidth=5.0,
    )
    axs[2][i].plot(
        data_minimize_fatigue["time"],
        data_minimize_fatigue["states"]["q"][i],
        ms=4,
        linewidth=5.0,
    )

# fatigue across time
axs[2][2].set_xlim(left=0, right=1.5)

axs[2][2].text(
    0.3,
    0.975,
    f"{'Scaling factor'}",
    transform=axs[2][2].transAxes,
    ha="left",
    va="top",
    weight="bold",
    font="Times New Roman",
)

labels = axs[2][2].get_xticklabels() + axs[2][2].get_yticklabels()
[label.set_fontname("Times New Roman") for label in labels]
[label.set_fontsize(14) for label in labels]

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

axs[2][2].plot(
    data_minimize_force["time"],
    fatigue_minimization_percentage_gain_list,
    ms=4,
    linewidth=5.0,
    color="green",
)

axs[2][2].text(
    0,
    1.15,
    f"{'%'}",
    transform=axs[2][2].transAxes,
    ha="left",
    va="top",
    fontsize=10,
)

plt.setp(
    axs[2][2],
    xticks=[0, 0.5, 1, 1.5],
    xticklabels=[0, 0.5, 1, 1.5],
)

# Figure labels
axs[1][0].text(
    -0.5,
    1.85,
    f"{'Force (N)'}",
    transform=axs[1][0].transAxes,
    ha="left",
    va="top",
    weight="bold",
    font="Times New Roman",
    fontsize=14,
    rotation=90,
)

axs[2][0].text(
    -0.5,
    1.18,
    "Joint angle \n     (rad)",
    transform=axs[2][0].transAxes,
    ha="left",
    va="top",
    weight="bold",
    font="Times New Roman",
    fontsize=14,
    rotation=90,
)

axs[2][1].text(
    0.225,
    -0.4,
    "Time (s)",
    transform=axs[2][1].transAxes,
    ha="left",
    va="top",
    weight="bold",
    font="Times New Roman",
    fontsize=14,
)

fig.tight_layout(pad=4.0)
plt.subplots_adjust(left=0.130, right=0.975, top=0.975, bottom=0.155)
plt.show()
