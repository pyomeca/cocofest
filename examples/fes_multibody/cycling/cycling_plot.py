import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation


def process_pickle_data(pickle_file_path_list):
    data_list = []
    for i in range(len(pickle_file_path_list)):
        # Initialize the dictionary with the cost function as name
        data_dictionary = {"name": 0}

        # Open the pickle file and load the data
        with open(pickle_file_path_list[i], "rb") as f:
            data = pickle.load(f)

        # Setting the first pedal position q as 0 radian
        q = (data["states"]["q"][-1] - data["states"]["q"][-1][0]) * -1
        data_dictionary["q"] = q

        # Extract the muscle forces
        muscle_list = [key[2:] for key in data["states"].keys() if key.startswith("F_")]
        force_keys = [key for key in data["states"].keys() if key.startswith("F_")]
        data_force_list = [data["states"][force_keys[j]][0] for j in range(len(muscle_list))]
        data_dictionary["force"] = {muscle_list[j]: data_force_list[j] for j in range(len(muscle_list))}

        # Get A
        a_keys = [key for key in data["states"].keys() if key.startswith("A_")]
        data_a_list = [data["states"][a_keys[j]][0] for j in range(len(muscle_list))]
        data_dictionary["a"] = {muscle_list[j]: data_a_list[j] for j in range(len(muscle_list))}

        # Get Km
        km_keys = [key for key in data["states"].keys() if key.startswith("Km_")]
        data_km_list = [data["states"][km_keys[j]][0] for j in range(len(muscle_list))]
        data_dictionary["km"] = {muscle_list[j]: data_km_list[j] for j in range(len(muscle_list))}

        # Get Tau1
        tau1_keys = [key for key in data["states"].keys() if key.startswith("Tau1_")]
        data_tau1_list = [data["states"][tau1_keys[j]][0] for j in range(len(muscle_list))]
        data_dictionary["tau1"] = {muscle_list[j]: data_tau1_list[j] for j in range(len(muscle_list))}

        # Get the index to slice for each cycle
        data_dictionary["slice_index"] = data["n_shooting_per_cycle"] * (data["polynomial_order"] + 1)
        data_dictionary["number_of_turns_before_failing"] = data["number_of_turns_before_failing"]
        data_dictionary["solving_time_per_ocp"] = data["solving_time_per_ocp"]

        data_list.append(data_dictionary)

    return data_list

def plot_data(data_list, conditions, update_dict=None):
    """
    Plots the data from the processed pickle files.
    """
    # Create layout
    layout = create_layout(data_list)
    color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    if update_dict is not None and "fig" in update_dict and "axd" in update_dict:
        fig, axd, frame = update_dict["fig"], update_dict["axd"], update_dict["frames"]
    else:
        # Create the figure and subplots
        fig, axd = plt.subplot_mosaic(mosaic=layout, figsize=(24, 16))
        frame = 0

    temp_layout = list({s for row in layout for s in row})
    for layout_index in temp_layout:
        axd[str(layout_index)].clear()

    # Plot circular force plot
    axd = create_circular_subplots(data_list, axd, fig, layout, color_list, frame=frame)

    # Plot muscle fatigue
    axd = create_bar_subplots(data_list, axd, key="a", layout_index=2, color_list=color_list, conditions=conditions, frame=frame)

    # Plot Tau1
    axd = create_bar_subplots(data_list, axd, key="tau1", layout_index=3, color_list=color_list, conditions=conditions, frame=frame)

    # Plot Km
    axd = create_bar_subplots(data_list, axd, key="km", layout_index=4, color_list=color_list, conditions=conditions, frame=frame)

    # Plot number of turns before failing
    axd = create_n_cycle_before_failure_plot(data_list, axd, layout_index=layout[-1][-1], index=frame, color_list=color_list)

    # Set legend
    axd = set_legend(data_list, axd, conditions, color_list, layout_index=layout, frame=frame)

    plt.subplots_adjust(hspace=0.6)

    if isinstance(update_dict, dict):
        return fig, axd
    else:
        plt.show()

def create_layout(data_list):
    """
    Creates a layout for the plots based on the data list.
    """
    number_of_muscle = len(data_list[0]["force"])
    plot_layout = [1, 1, 2, 3, 4]
    cycling_info_layout = [1, 2, 2, 2, 2]

    layout = [
             [val + i * 4 for val in plot_layout]
              + [cycling_info_layout[i] + number_of_muscle * 4]
         for i in range(number_of_muscle)
         for _ in range(2)
        ]

    # layout.append([val + number_of_muscle * 4 for val in cycling_info_layout])
    layout_str = [[str(n) for n in row] for row in layout]
    return layout_str


def create_circular_force_plot(data, axis, fig, index_update=0, f_max=0, color=None, frame=0):
    layout_index = 1
    data_slicing_min = frame  * data[0]["slicing"]
    data_slicing_min = (frame + 1) * data[0]["slicing"]

    ticks = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ticks = np.delete(ticks, [2, 6])
    for key in data["force"].keys():
        axis[str(layout_index)].set_xticks(ticks)
        axis[str(layout_index)].set_yticklabels([])
        axis[str(layout_index)].set_rlim(0, f_max)
        axis[str(layout_index)].set_ylabel(key, fontsize=20, labelpad=80, fontweight='bold')

        # Slice
        q_slice = data["q"][data_slicing_min:data_slicing_max]
        F_slice = data["force"][key][data_slicing_min:data_slicing_max]
        # Plot
        axis[str(layout_index)].plot(q_slice, F_slice, color=color, lw=2)
        axis[str(layout_index)].fill_between(q_slice, F_slice, color=color, alpha=0.3)

        layout_index += 4
    return axis

def create_circular_subplots(data_list, axis, fig, layout, color_list, frame):
    f_max = max([max(np.concatenate(list(data["force"].values()))) for data in data_list])
    # Init graphs
    axis = init_circular_graphs(axis, fig, keys=data_list[0]["force"].keys())
    for i in range(len(data_list)):
        axis = create_circular_force_plot(data=data_list[i], axis=axis, fig=fig, f_max=f_max, color=color_list[i], frame=frame)
    axis["1"].set_title("Force" + " (max = " + str(int(f_max)) + "N)", fontsize=15, fontweight='bold', pad=20)
    return axis

def init_circular_graphs(axis, fig, keys):
    layout_index = 1
    for key in keys:
        bbox = axis[str(layout_index)].get_position()
        fig.delaxes(axis[str(layout_index)])
        axis[str(layout_index)] = fig.add_axes(bbox, projection='polar')
        axis[str(layout_index)].set_theta_zero_location('E')
        axis[str(layout_index)].set_theta_direction(-1)
        layout_index += 4
    return axis

def create_bar_subplots(data_list, axis, key, layout_index, color_list, conditions, frame=0):
    max_val = max([max(np.concatenate(list(data[key].values()))) for data in data_list])
    for i in range(len(data_list)):
        axis = create_bar_plot(data=data_list[i][key],
                               axis=axis,
                               index=i,
                               color=color_list[i],
                               layout_index=layout_index,
                               max_y=max_val,
                               key=key,
                               conditions=conditions,
                               frame=frame)
    return axis


def create_bar_plot(data, axis, index, color, layout_index, max_y, key, conditions, frame):
    muscle_names = list(data.keys())
    slicing = frame  * data[0]["slicing"]
    for i in range(len(muscle_names)):
        axis[str(layout_index)].set_ylim([0, max_y + 0.05 * max_y])
        axis[str(layout_index)].bar(index, data[muscle_names[i]][slicing], color=color)
        axis[str(layout_index)].set_xticks([])
        if i == 0:
            title = "Force scaling factor A (N/s)" if key == "a" else "Time force decay Tau1 (s)" if key == "tau1" else "Cross-bridges to calcium Km (-)"
            axis[str(layout_index)].set_title(title, fontsize=13, fontweight='bold', pad=20)
        if i == len(muscle_names) - 1:
            axis[str(layout_index)].set_xticks(range(len(data_list)))
            axis[str(layout_index)].set_xticklabels(conditions, fontsize=12, rotation=65)
        layout_index += 4
    return axis

def create_n_cycle_before_failure_plot(data_list, axis, layout_index, index, color_list):
    axis[str(layout_index)].set_xticks([])
    axis[str(layout_index)].set_title("Turns before failing", fontsize=13, fontweight='bold', pad=20)
    axis[str(layout_index)].set_xticks(range(len(data_list)))
    axis[str(layout_index)].set_xticklabels(conditions, fontsize=12, rotation=65)
    axis[str(layout_index)].set_ylim([0, max(data_list[i]["number_of_turns_before_failing"] for i in range(len(data_list))) + 1])
    for i in range(len(data_list)):
        height = index + 1 if data_list[i]["number_of_turns_before_failing"] > index + 1 else data_list[i]["number_of_turns_before_failing"]
        axis[str(layout_index)].bar(i, height, color=color_list[i])
    return axis

def set_legend(data_list, axis, conditions, colors, layout_index, frame):
    """
    Creates a legend for the plot.
    """
    cycle = frame
    index = layout_index[0][-1]
    axis[index].axis('off')

    square_size = 0.08
    start_y = 0.85
    y_step = 0.15
    text_offset = 0.03
    x0 = 0.10
    value_x = 0.92

    axis[index].text(value_x, start_y + 0.2, f"Time to solve cycle {cycle} (s)",
            transform=axis[index].transAxes, ha="right", va="bottom",
            fontsize=12, fontweight="bold")

    # Draw entries
    for i, (label, color) in enumerate(zip(conditions, colors[:len(conditions)])):
        y = start_y - i * y_step

        if i < len(data_list) and "solving_time_per_ocp" in data_list[i]:
            value = float(data_list[i]["solving_time_per_ocp"][cycle]) if cycle < len(data_list[i]["solving_time_per_ocp"]) else float(data_list[i]["solving_time_per_ocp"][-1])
        else:
            value = float("nan")

        # Square
        square = mpatches.Rectangle((x0, y), square_size, square_size,
                                    transform=axis[index].transAxes,
                                    facecolor=color, edgecolor='black',
                                    clip_on=False)
        axis[index].add_patch(square)

        # Text
        axis[index].text(x0 + square_size + text_offset, y + square_size / 2, label,
                transform=axis[index].transAxes, va='center', ha='left', fontsize=12)

        # Value
        axis[index].text(value_x, y + square_size / 2, f"{value:.2f}",
                transform=axis[index].transAxes, va="center", ha="right", fontsize=12)

    return axis


def animate_results(data_list, conditions, frames, save_path):
    update_dict = {}
    fig, axis = plot_data(data_list, conditions, update_dict=update_dict)
    update_dict["fig"] = fig
    update_dict["axd"] = axis

    def update(frames):
        update_dict["frames"] = frames
        plot_data(data_list, conditions, update_dict=update_dict)
        return []

    anim = animation.FuncAnimation(fig, update,
                                   frames=len(frames),
                                   interval=1,
                                   blit=False)
    anim.save(save_path, writer='pillow')
    plt.close(fig)

if __name__ == "__main__":
    data_list = process_pickle_data(["result/2_cycle/2_min_100_force_collocation_3_radau_with_init.pkl",
                                     # "result/2_cycle/2_min_100_force_collocation_3_radau_with_init.pkl",
                                     ])
    conditions = ["min force", "min fatigue"]

    plt.plot(data_list["time"], data_list)

    # plot_data(data_list, conditions)

    animate_results(
        data_list=data_list,
        conditions=conditions,
        frames=np.arange(0, 2, 1),
        save_path="results_animation.gif"
    )



