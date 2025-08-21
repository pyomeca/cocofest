import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation


def _safe_cycle_slice(total_len, slice_index, frame):
    """
    Returns a safe [start, end) slice for the requested frame.
    If frame is past available data, returns the last full/partial slice.
    """
    if total_len <= 0 or slice_index <= 0:
        return 0, 0
    s = frame * slice_index
    e = s + slice_index
    if s < total_len:
        return s, min(e, total_len)
    last_start = max(0, (total_len - 1) // slice_index * slice_index)
    return last_start, total_len

def _safe_cycle_index(length, slice_index, frame):
    """
    Returns a safe index for a scalar read at the beginning of a cycle.
    If frame is past available data, returns the last valid index.
    """
    if length == 0:
        return 0
    idx = frame * slice_index
    return idx if idx < length else length - 1

def _condition_alpha(data_entry, frame, active_alpha=1.0, inactive_alpha=0.25):
    """Lower alpha once the condition has failed (i.e., its cycles are exhausted)."""
    n_ok = int(data_entry.get("number_of_turns_before_failing", 0))
    return active_alpha if frame < n_ok else inactive_alpha


def process_data(file_path_list):
    data_list = []
    for i in range(len(file_path_list)):
        # Initialize the dictionary with the cost function as name
        data_dictionary = {"name": 0}

        if file_path_list[i].endswith(".pkl"):
            with open(file_path_list[i], "rb") as f:
                data = pickle.load(f)
        else:
            data = np.load(file_path_list[i])

        data_dictionary["time"] = data["time"]

        # Setting the first pedal position q as 0 radian
        q = (data["q"][-1] - data["q"][-1][0]) * -1
        data_dictionary["q"] = q
        data_dictionary["qdot"] = data["qdot"]

        # Extract the muscle forces
        muscle_list = [key[2:] for key in data.keys() if key.startswith("F_")]
        force_keys = [key for key in data.keys() if key.startswith("F_")]
        data_force_list = [data[force_keys[j]][0] for j in range(len(muscle_list))]
        data_dictionary["force"] = {muscle_list[j]: data_force_list[j] for j in range(len(muscle_list))}

        # Get A
        a_keys = [key for key in data.keys() if key.startswith("A_")]
        data_a_list = [data[a_keys[j]][0] for j in range(len(muscle_list))]
        data_dictionary["a"] = {muscle_list[j]: data_a_list[j] for j in range(len(muscle_list))}

        # Get Km
        km_keys = [key for key in data.keys() if key.startswith("Km_")]
        data_km_list = [data[km_keys[j]][0] for j in range(len(muscle_list))]
        data_dictionary["km"] = {muscle_list[j]: data_km_list[j] for j in range(len(muscle_list))}

        # Get Tau1
        tau1_keys = [key for key in data.keys() if key.startswith("Tau1_")]
        data_tau1_list = [data[tau1_keys[j]][0] for j in range(len(muscle_list))]
        data_dictionary["tau1"] = {muscle_list[j]: data_tau1_list[j] for j in range(len(muscle_list))}

        # Get the index to slice for each cycle
        # data_dictionary["slice_index"] = data["n_shooting_per_cycle"] * (data["polynomial_order"] + 1)
        data_dictionary["slice_index"] = int(data["total_n_shooting"]/(len(data["solving_time_per_ocp"]))) * (data["polynomial_order"] + 1)  # Temporary fix
        data_dictionary["number_of_turns_before_failing"] = int(data["number_of_turns_before_failing"])
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

    layout_str = [[str(n) for n in row] for row in layout]
    return layout_str


def create_circular_force_plot(data, axis, fig, index_update=0, f_max=0, color=None, frame=0, fmax=None, alpha=1.0):
    layout_index = 1

    s_min, s_max = _safe_cycle_slice(len(data["q"]), data["slice_index"], frame)

    ticks = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ticks = np.delete(ticks, [2, 6])
    for key in data["force"].keys():
        axis[str(layout_index)].set_xticks(ticks)
        axis[str(layout_index)].set_yticklabels([])
        axis[str(layout_index)].set_rlim(0, fmax[key])
        axis[str(layout_index)].set_ylabel(key + "\n" + "Fmax = " + str(int(fmax[key])) + "N", fontsize=18, labelpad=80, fontweight='bold')

        # Slice safely
        q_slice = data["q"][s_min:s_max]
        F_slice = data["force"][key][s_min:s_max]

        # If no data, skip plotting this muscle to avoid errors
        if len(q_slice) == 0 or len(F_slice) == 0:
            layout_index += 4
            continue

        # Plot with fading alpha when failed
        axis[str(layout_index)].plot(q_slice, F_slice, color=color, lw=2, alpha=alpha)
        axis[str(layout_index)].fill_between(q_slice, F_slice, color=color, alpha=0.3 * alpha)

        layout_index += 4
    return axis

def create_circular_subplots(data_list, axis, fig, layout, color_list, frame):
    f_max_dict = get_max_per_key(data_list, "force")
    axis = init_circular_graphs(axis, fig, keys=data_list[0]["force"].keys())
    for i in range(len(data_list)):
        a = _condition_alpha(data_list[i], frame)
        axis = create_circular_force_plot(data=data_list[i], axis=axis, fig=fig, color=color_list[i], frame=frame, fmax=f_max_dict, alpha=a)
    axis["1"].set_title("Force", fontsize=15, fontweight='bold', pad=20)
    return axis

def get_max_per_key(data_list, tag):
    temp_list = []
    for i in range(len(data_list)):
        temp_list.append({})
        for key in data_list[i][tag].keys():
            fmax = round(max(data_list[i][tag][key]) + 0.01 * max(data_list[i][tag][key]), 3)
            temp_list[i][key] = fmax

    max_dict = {key: max([temp_list[i][key] for i in range(len(data_list))]) for key in temp_list[0].keys()}
    return max_dict

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
    max_dict = get_max_per_key(data_list, key)
    for i in range(len(data_list)):
        alpha = _condition_alpha(data_list[i], frame)
        axis = create_bar_plot(data=data_list[i][key],
                               axis=axis,
                               index=i,
                               color=color_list[i],
                               layout_index=layout_index,
                               max_y=max_dict,
                               key=key,
                               conditions=conditions,
                               slice_index=data_list[i]["slice_index"],
                               frame=frame,
                               alpha=alpha)
    return axis

def create_bar_plot(data, axis, index, color, layout_index, max_y, key, conditions, slice_index, frame, alpha=1.0):
    muscle_names = list(data.keys())
    for i in range(len(muscle_names)):
        axis[str(layout_index)].set_ylim([0, max_y[muscle_names[i]] + 0.05 * max_y[muscle_names[i]]])

        arr = data[muscle_names[i]]
        safe_idx = _safe_cycle_index(len(arr), slice_index, frame)
        height = float(arr[safe_idx]) if len(arr) > 0 else 0.0

        axis[str(layout_index)].bar(index, height, color=color, alpha=alpha)
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
        alpha = _condition_alpha(data_list[i], index)
        axis[str(layout_index)].bar(i, height, color=color_list[i], alpha=alpha)
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

        alpha = _condition_alpha(data_list[i], frame)

        # Square
        square = mpatches.Rectangle((x0, y), square_size, square_size,
                                    transform=axis[index].transAxes,
                                    facecolor=color, edgecolor='black',
                                    clip_on=False, alpha=alpha)  # NEW
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

    def update(fr):
        update_dict["frames"] = fr
        plot_data(data_list, conditions, update_dict=update_dict)
        print("Frame:", fr)
        return []

    anim = animation.FuncAnimation(fig, update,
                                   frames=frames,
                                   interval=1,
                                   blit=False)
    anim.save(save_path, writer='pillow')
    plt.close(fig)

if __name__ == "__main__":
    data_list = process_data(["result/o/2_min_100_force_collocation_3_radau_with_init.npz",
                              "result/o/2_min_100_fatigue_collocation_3_radau_with_init.npz",
                              "result/o/2_min_100_control_collocation_3_radau_with_init.npz",
                              ])
    conditions = ["min force", "min fatigue", "min control"]

    max_cycle = max([data["number_of_turns_before_failing"] for data in data_list])

    animate_results(
        data_list=data_list,
        conditions=conditions,
        frames=np.arange(0, max_cycle, 1),
        save_path="results_animation1.gif"
    )

