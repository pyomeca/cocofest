import pickle
import numpy as np


def full_data_extraction(model_data_path):
    """
    Extracts full data from the provided path.

    Parameters
    ----------
    model_data_path : list
        List of paths to the data files.

    Returns
    -------
    tuple
        A tuple containing the time data, stim apparition time, muscle data, and discontinuity phase list.
    """

    global_model_muscle_data = []
    global_model_stim_apparition_time = []
    global_model_time_data = []

    discontinuity_phase_list = []
    for i in range(len(model_data_path)):
        with open(model_data_path[i], "rb") as f:
            data = pickle.load(f)
        model_data = data["force"]

        # Arranging the data to have the beginning time starting at 0 second for all data
        model_stim_apparition_time = (
            data["stim_time"]
            if data["stim_time"][0] == 0
            else [stim_time - data["stim_time"][0] for stim_time in data["stim_time"]]
        )

        model_time_data = (
            data["time"]
            if data["stim_time"][0] == 0
            else [
                [(time - data["stim_time"][0]) for time in row] for row in data["time"]
            ]
        )

        # model_data = [item for sublist in model_data for item in sublist]
        # model_time_data = [item for sublist in model_time_data for item in sublist]

        # Indexing the current data time on the previous one to ensure time continuity
        if i != 0:
            discontinuity_phase_list.append(
                len(global_model_stim_apparition_time[-1])
                if discontinuity_phase_list == []
                else discontinuity_phase_list[-1]
                + len(global_model_stim_apparition_time[-1])
            )

            model_stim_apparition_time = [
                stim_time + global_model_time_data[i - 1][-1]
                for stim_time in model_stim_apparition_time
            ]

            model_time_data = [
                (time + global_model_time_data[i - 1][-1]) for time in model_time_data
            ]
            model_stim_apparition_time = [
                (time + global_model_time_data[i - 1][-1])
                for time in model_stim_apparition_time
            ]

        # Storing data into global lists
        global_model_muscle_data.append(model_data)
        global_model_stim_apparition_time.append(model_stim_apparition_time)
        global_model_time_data.append(model_time_data)
    # Expending global lists
    global_model_muscle_data = [
        item for sublist in global_model_muscle_data for item in sublist
    ]
    global_model_stim_apparition_time = [
        item for sublist in global_model_stim_apparition_time for item in sublist
    ]
    global_model_time_data = [
        item for sublist in global_model_time_data for item in sublist
    ]
    return (
        global_model_time_data,
        global_model_stim_apparition_time,
        global_model_muscle_data,
        discontinuity_phase_list,
    )


def average_data_extraction(model_data_path):
    """
    Extracts average data from the provided path.

    Parameters
    ----------
    model_data_path : list
        List of paths to the data files.

    Returns
    -------
    tuple
        A tuple containing the time data, stim apparition time, muscle data, and discontinuity phase list.
    """

    global_model_muscle_data = []
    global_model_stim_apparition_time = []
    global_model_time_data = []

    discontinuity_phase_list = []
    for i in range(len(model_data_path)):
        with open(model_data_path[i], "rb") as f:
            data = pickle.load(f)
        model_data = data["force"]

        temp_stimulation_instant = []
        stim_threshold = data["stim_time"][1] - data["stim_time"][0]
        for j in range(1, len(data["stim_time"])):
            stim_interval = data["stim_time"][j] - data["stim_time"][j - 1]
            if stim_interval < stim_threshold * 1.5:
                temp_stimulation_instant.append(
                    data["stim_time"][j] - data["stim_time"][j - 1]
                )
        stimulation_temp_frequency = round(1 / np.mean(temp_stimulation_instant), 0)

        model_time_data = (
            data["time"]
            if data["stim_time"][0] == 0
            else [
                [(time - data["stim_time"][0]) for time in row] for row in data["time"]
            ]
        )

        # Average on each force curve
        smallest_list = 0
        for j in range(len(model_data)):
            if j == 0:
                smallest_list = len(model_data[j])
            if len(model_data[j]) < smallest_list:
                smallest_list = len(model_data[j])

        model_data = np.mean(
            [row[:smallest_list] for row in model_data], axis=0
        ).tolist()
        model_time_data = [item for sublist in model_time_data for item in sublist]

        model_time_data = model_time_data[:smallest_list]
        train_duration = 1

        average_stim_apparition = np.linspace(
            0, train_duration, int(stimulation_temp_frequency * train_duration) + 1
        )[:-1]
        average_stim_apparition = [time for time in average_stim_apparition]
        if i == len(model_data_path) - 1:
            average_stim_apparition = np.append(
                average_stim_apparition, model_time_data[-1]
            ).tolist()

        # Indexing the current data time on the previous one to ensure time continuity
        if i != 0:
            discontinuity_phase_list.append(
                len(global_model_stim_apparition_time[-1])
                if discontinuity_phase_list == []
                else discontinuity_phase_list[-1]
                + len(global_model_stim_apparition_time[-1])
            )

            model_time_data = [
                (time + global_model_time_data[i - 1][-1]) for time in model_time_data
            ]
            average_stim_apparition = [
                (time + global_model_time_data[i - 1][-1])
                for time in average_stim_apparition
            ]

        # Storing data into global lists
        global_model_muscle_data.append(model_data)
        global_model_stim_apparition_time.append(average_stim_apparition)
        global_model_time_data.append(model_time_data)

    # Expending global lists
    global_model_muscle_data = [
        item for sublist in global_model_muscle_data for item in sublist
    ]
    global_model_stim_apparition_time = [
        item for sublist in global_model_stim_apparition_time for item in sublist
    ]
    global_model_time_data = [
        item for sublist in global_model_time_data for item in sublist
    ]
    return (
        global_model_time_data,
        global_model_stim_apparition_time,
        global_model_muscle_data,
        discontinuity_phase_list,
    )


def sparse_data_extraction(model_data_path, force_curve_number=5):
    """
    Extracts sparse data from the provided path.

    Parameters
    ----------
    model_data_path : list
        List of paths to the data files.
    force_curve_number : int
        Number of force curves to extract, by default 5.

    Returns
    -------
    tuple
        A tuple containing the time data, stim apparition time, muscle data, and discontinuity phase list.
    """

    raise NotImplementedError("This method has not been tested yet")

    # global_model_muscle_data = []
    # global_model_stim_apparition_time = []
    # global_model_time_data = []
    #
    # discontinuity_phase_list = []
    # for i in range(len(model_data_path)):
    #     with open(model_data_path[i], "rb") as f:
    #         data = pickle.load(f)
    #     model_data = data["force"]
    #
    #     # Arranging the data to have the beginning time starting at 0 second for all data
    #     model_stim_apparition_time = (
    #         data["stim_time"]
    #         if data["stim_time"][0] == 0
    #         else [stim_time - data["stim_time"][0] for stim_time in data["stim_time"]]
    #     )
    #
    #     model_time_data = (
    #         data["time"]
    #         if data["stim_time"][0] == 0
    #         else [[(time - data["stim_time"][0]) for time in row] for row in data["time"]]
    #     )
    #
    #     # TODO : check this part
    #     model_data = model_data[0:force_curve_number] + model_data[:-force_curve_number]
    #     model_time_data = model_time_data[0:force_curve_number] + model_time_data[:-force_curve_number]
    #
    #     # TODO correct this part
    #     model_stim_apparition_time = (
    #         model_stim_apparition_time[0:force_curve_number] + model_stim_apparition_time[:-force_curve_number]
    #     )
    #
    #     model_data = [item for sublist in model_data for item in sublist]
    #     model_time_data = [item for sublist in model_time_data for item in sublist]
    #
    #     # Indexing the current data time on the previous one to ensure time continuity
    #     if i != 0:
    #         discontinuity_phase_list.append(
    #             len(global_model_stim_apparition_time[-1]) - 1
    #             if discontinuity_phase_list == []
    #             else discontinuity_phase_list[-1] + len(global_model_stim_apparition_time[-1])
    #         )
    #
    #         model_stim_apparition_time = [
    #             stim_time + global_model_time_data[i - 1][-1] for stim_time in model_stim_apparition_time
    #         ]
    #
    #         model_time_data = [(time + global_model_time_data[i - 1][-1]) for time in model_time_data]
    #         model_stim_apparition_time = [
    #             (time + global_model_time_data[i - 1][-1]) for time in model_stim_apparition_time
    #         ]
    #
    #     # Storing data into global lists
    #     global_model_muscle_data.append(model_data)
    #     global_model_stim_apparition_time.append(model_stim_apparition_time)
    #     global_model_time_data.append(model_time_data)
    # # Expending global lists
    # global_model_muscle_data = [item for sublist in global_model_muscle_data for item in sublist]
    # global_model_stim_apparition_time = [item for sublist in global_model_stim_apparition_time for item in sublist]
    # global_model_time_data = [item for sublist in global_model_time_data for item in sublist]
    #
    # return (
    #     global_model_time_data,
    #     global_model_stim_apparition_time,
    #     global_model_muscle_data,
    #     discontinuity_phase_list,
    # )


def force_at_node_in_ocp(time, force, n_shooting, final_time, sparse=None):
    """
    Interpolates the force at each node in the optimal control problem (OCP).

    Parameters
    ----------
    time : list
        List of time data.
    force : list
        List of force data.
    n_shooting : int
        List of number of shooting points for each phase.
    sparse : int, optional
        Number of sparse points, by default None.

    Returns
    -------
    list
        List of force at each node in the OCP.
    """

    temp_time = np.linspace(0, final_time, int((n_shooting / final_time) + 1))
    force_at_node = list(np.interp(temp_time, time, force))
    # if sparse:  # TODO check this part
    #     force_at_node = force_at_node[0:sparse] + force_at_node[:-sparse]
    return force_at_node


def node_shooting_list_creation(stim, stimulated_n_shooting):
    """
    Creates a list of node shooting points.

    Parameters
    ----------
    stim : list
        List of stimulation times.
    stimulated_n_shooting : int
        Number of shooting points for stimulated phase.

    Returns
    -------
    tuple
        A tuple containing the list of number of shooting points for each phase and the final time for each phase.
    """

    first_final_time = stim[1] if stim[0] == 0 else stim[0]
    final_time_phase = (first_final_time,)
    for i in range(1, len(stim)):
        final_time_phase = final_time_phase + (stim[i] - stim[i - 1],)

    threshold_stimulation_interval = np.mean(final_time_phase)
    stimulation_interval_average_without_rest_time = np.delete(
        np.array(final_time_phase),
        np.where(
            np.logical_or(
                final_time_phase > threshold_stimulation_interval,
                np.array(final_time_phase) == 0,
            )
        ),
    )
    stimulation_interval_average = np.mean(
        stimulation_interval_average_without_rest_time
    )
    n_shooting = []

    for i in range(len(final_time_phase)):
        if final_time_phase[i] > threshold_stimulation_interval:
            temp_final_time = final_time_phase[i]
            rest_n_shooting = int(
                (temp_final_time / stimulation_interval_average) * stimulated_n_shooting
            )
            n_shooting.append(rest_n_shooting)
        else:
            n_shooting.append(stimulated_n_shooting)

    return n_shooting, final_time_phase
