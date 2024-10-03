import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from copy import deepcopy
import heapq
import pickle

from pyomeca import Analogs


class ExtractAnalogForceFromC3D:
    """
    Perfect data for identification is no force at the beginning and a force release between each stimulation train.
    This will enable data slicing of the force response to stimulation.

    It is assumed that V1, V2, V3, V4, V5, V6 are the 6D sensor data and last input is stimulation signal
    if not give an input list of the keys position "V1", "V2", "V3", "V4", "V5", "V6", "stim"
    in the c3d file (default is [0, 1, 2, 3, 4, 5, 6]).
    """

    def __init__(
        self,
        c3d_path: str | list[str] = None,
        calibration_matrix_path: str = None,
        for_identification: bool = False,
        saving_pickle_path: str | list[str] = None,
        down_sample: bool = True,
        **kwargs,
    ):
        c3d_path_list = [c3d_path] if isinstance(c3d_path, str) else c3d_path
        saving_pickle_path_list = [saving_pickle_path] if isinstance(saving_pickle_path, str) else saving_pickle_path
        if saving_pickle_path_list:
            if len(saving_pickle_path_list) != 1 and len(saving_pickle_path_list) != len(c3d_path_list):
                raise ValueError(
                    "The number of saving_pickle_path must be the same as the number of c3d_path."
                    "If you entered only one path, the file name will be iterated."
                )

        for i in range(len(c3d_path_list)):
            c3d_path = c3d_path_list[i]
            if not isinstance(c3d_path, str):
                raise TypeError("c3d_path must be a str or a list of str.")
            raw_data = Analogs.from_c3d(c3d_path)

            order = kwargs["order"] if "order" in kwargs else 1
            cutoff = kwargs["cutoff"] if "cutoff" in kwargs else 2
            if not isinstance(order, int | None) or not isinstance(cutoff, int | None):
                raise TypeError("window_length and order must be either None or int type")
            if type(order) != type(cutoff):
                raise TypeError("window_length and order must be both None or int type")

            time = raw_data.time.values.tolist()
            filtered_data = (
                np.array(raw_data.meca.low_pass(order=order, cutoff=cutoff, freq=raw_data.rate))
                if order and cutoff
                else raw_data
            )

            if "input_channel" in kwargs:
                filtered_data = self.reindex_2d_list(filtered_data, kwargs["input_channel"])

            if calibration_matrix_path:
                self.calibration_matrix = self.read_text_file_to_matrix(calibration_matrix_path)
                filtered_6d_force = self.calibration_matrix @ filtered_data[:6]

            else:
                if "already_calibrated" in kwargs:
                    if kwargs["already_calibrated"] is True:
                        filtered_6d_force = filtered_data[:6]
                    else:
                        raise ValueError("already_calibrated must be either True or False")
                else:
                    raise ValueError(
                        "Please specify if the data is already calibrated or not with already_calibrated input."
                        "If not, please provide a calibration matrix path"
                    )

            filtered_6d_force = self.set_zero_level(filtered_6d_force, average_on=[1000, 3000])
            if for_identification:
                check_stimulation = kwargs["check_stimulation"] if "check_stimulation" in kwargs else None

                if "average_time_difference" in kwargs and "frequency_acquisition" in kwargs:
                    stimulation_time, peaks = self.stimulation_detection(
                        time,
                        raw_data[6].data,
                        average_time_difference=kwargs["average_time_difference"],
                        frequency_acquisition=kwargs["frequency_acquisition"],
                        check_stimulation=check_stimulation,
                    )  # detect the stimulation time
                else:
                    stimulation_time, peaks = self.stimulation_detection(
                        time, raw_data[6].data, check_stimulation=check_stimulation
                    )  # detect the stimulation time
                sliced_time, sliced_data = self.slice_data(
                    time, filtered_6d_force, peaks
                )  # slice the data into different stimulation
                temp_time = deepcopy(sliced_time)
                temp_data = deepcopy(sliced_data)
                if down_sample:
                    for j in range(len(sliced_data[0])):
                        for k in range(len(sliced_data)):
                            remove_list = []
                            counter = 0
                            for m in range(1, len(sliced_data[k][j])):
                                if counter == 9:
                                    counter = 0
                                else:
                                    remove_list.append(m)
                                    counter += 1
                            remove_list.reverse()
                            for l in remove_list:
                                sliced_data[k][j].pop(l)

                        remove_list = []
                        counter = 0
                        for k in range(1, len(sliced_time[j])):
                            if counter == 9:
                                counter = 0
                            else:
                                remove_list.append(k)
                                counter += 1
                        remove_list.reverse()
                        for l in remove_list:
                            sliced_time[j].pop(l)

                if "plot" in kwargs:
                    if kwargs["plot"]:
                        for k in range(len(sliced_time)):
                            plt.plot(sliced_time[k], sliced_data[0][k])
                        for k in range(len(peaks)):
                            plt.plot(time[peaks[k]], filtered_6d_force[0][peaks[k]], "x")
                        if down_sample:
                            for k in range(len(sliced_time)):
                                plt.plot(temp_time[k], temp_data[0][k])
                        plt.show()

                if saving_pickle_path_list:
                    if len(saving_pickle_path_list) == 1:
                        if saving_pickle_path_list[:-4] == ".pkl":
                            save_pickle_path = saving_pickle_path_list[:-4] + "_" + str(i) + ".pkl"
                        else:
                            save_pickle_path = saving_pickle_path_list[0] + "_" + str(i) + ".pkl"
                    else:
                        save_pickle_path = saving_pickle_path_list[i]

                    dictionary = {
                        "time": sliced_time,
                        "x": sliced_data[0],
                        "y": sliced_data[1],
                        "z": sliced_data[2],
                        "mx": sliced_data[3],
                        "my": sliced_data[4],
                        "mz": sliced_data[5],
                        "stim_time": stimulation_time,
                    }
                    with open(save_pickle_path, "wb") as file:
                        pickle.dump(dictionary, file)
            else:
                if saving_pickle_path_list[:-4] == ".pkl":
                    save_pickle_path = saving_pickle_path_list[:-4] + "_" + str(i) + ".pkl"
                else:
                    save_pickle_path = saving_pickle_path_list[0] + "_" + str(i) + ".pkl"

                if down_sample:
                    for j in range(len(filtered_6d_force[0])):
                        for k in range(len(filtered_6d_force)):
                            remove_list = []
                            counter = 0
                            for m in range(1, len(filtered_6d_force[k][j])):
                                if counter == 9:
                                    counter = 0
                                else:
                                    remove_list.append(m)
                                    counter += 1
                            remove_list.reverse()
                            for l in remove_list:
                                filtered_6d_force[k][j].pop(l)

                        remove_list = []
                        counter = 0
                        for k in range(1, len(time[j])):
                            if counter == 9:
                                counter = 0
                            else:
                                remove_list.append(k)
                                counter += 1
                        remove_list.reverse()
                        for l in remove_list:
                            time[j].pop(l)
                            raw_data[7].pop(l)
                dictionary = {
                    "time": time,
                    "x": filtered_6d_force[0],
                    "y": filtered_6d_force[1],
                    "z": filtered_6d_force[2],
                    "mx": filtered_6d_force[3],
                    "my": filtered_6d_force[4],
                    "mz": filtered_6d_force[5],
                    "stim_time": raw_data[7],
                }
                with open(save_pickle_path, "wb") as file:
                    pickle.dump(dictionary, file)

    @staticmethod
    def reindex_2d_list(data, new_indices):
        # Ensure the new_indices list is not out of bounds
        if max(new_indices) >= len(data) or min(new_indices) < 0:
            raise ValueError("Invalid new_indices list. Out of bounds.")

        # Create a new 2D list with re-ordered elements
        new_data = [[data[i][j] for j in range(len(data[i]))] for i in new_indices]

        return new_data

    def read_text_file_to_matrix(self, file_path):
        try:
            # Read the text file and split lines
            with open(file_path, "r") as file:
                lines = file.readlines()
            # Initialize an empty list to store the rows
            data = []
            # Iterate through the lines, split by tabs, and convert to float
            for line in lines:
                row = [float(value) for value in line.strip().split("\t")]
                data.append(row)
            # Convert the list of lists to a NumPy matrix
            matrix = np.array(data)
            return matrix
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

    @staticmethod
    def set_zero_level(data: np.array, average_length: int = 1000, average_on: list[int, int] = None):
        """
        Set the zero level of the data by averaging the first 1000 points
        :param data: The data to set the zero level
        :param average_length: The number of points to average
        :return: The data with the zero level set
        """
        if len(data.shape) == 1:
            return (
                data - np.mean(data[average_on[0] : average_on[1]])
                if average_on
                else data - np.mean(data[:average_length])
            )
        else:
            for i in range(data.shape[0]):
                data[i] = (
                    data[i] - np.mean(data[i][average_on[0] : average_on[1]])
                    if average_on
                    else data[i] - np.mean(data[i][:average_length])
                )
            return data

    def slice_data(self, time, data, stimulation_peaks, main_axis=0):
        sliced_time = []
        temp_stimulation_peaks = stimulation_peaks
        x = []
        y = []
        z = []
        mx = []
        my = []
        mz = []

        while len(temp_stimulation_peaks) != 0:
            substact_to_zero = data[:, temp_stimulation_peaks[0]]
            for i in range(len(data[:, temp_stimulation_peaks[0]])):
                data[:, temp_stimulation_peaks[0] :][i] = data[:, temp_stimulation_peaks[0] :][i] - substact_to_zero[i]

            first = temp_stimulation_peaks[0]
            last = next(x for x, val in enumerate(-data[main_axis, first:]) if val < 0) + first

            x.append(data[0, first:last].tolist())
            y.append(data[1, first:last].tolist())
            z.append(data[2, first:last].tolist())
            mx.append(data[3, first:last].tolist())
            my.append(data[4, first:last].tolist())
            mz.append(data[5, first:last].tolist())

            sliced_time.append(time[first:last])

            temp_stimulation_peaks = [peaks for peaks in temp_stimulation_peaks if peaks > last]
        sliced_data = [x, y, z, mx, my, mz]
        return sliced_time, sliced_data

    def stimulation_detection(
        self,
        time,
        stimulation_signal,
        average_time_difference: float = None,
        frequency_acquisition: int = None,
        check_stimulation: bool = False,
    ):
        # # --- Cleaning artefact from fatigue data set --- #
        #
        # stimulation_signal[:46000] = 0
        # stimulation_signal[56000:66000] = 0
        # stimulation_signal[76000:86000] = 0
        # stimulation_signal[96000:106000] = 0
        # stimulation_signal[116000:126000] = 0
        # stimulation_signal[136000:146000] = 0
        # stimulation_signal[156000:169200] = 0
        # stimulation_signal[176000:186000] = 0
        # stimulation_signal[200000:210000] = 0
        # stimulation_signal[220500:229000] = 0
        # stimulation_signal[241000:251700] = 0
        # stimulation_signal[260000:272550] = 0
        # stimulation_signal[282000:292000] = 0
        # stimulation_signal[303000:313200] = 0
        # stimulation_signal[324000:333000] = 0
        # stimulation_signal[344000:354000] = 0
        # stimulation_signal[364000:374000] = 0
        # stimulation_signal[385000:394000] = 0
        # stimulation_signal[385000:394000] = 0
        # stimulation_signal[426000:433000] = 0
        # stimulation_signal[448000:457500] = 0
        # stimulation_signal[468000:476500] = 0
        # stimulation_signal[490000:498500] = 0
        # stimulation_signal[510000:518000] = 0
        # stimulation_signal[550000:558500] = 0

        # stimulation_signal = np.where(stimulation_signal > 0.75, 0, stimulation_signal)
        # stimulation_signal = np.where(stimulation_signal < -0.75, 0, stimulation_signal)

        if average_time_difference:
            if not isinstance(average_time_difference, float):
                raise TypeError("average_time_difference must be a float.")
            if not frequency_acquisition:
                raise ValueError("Please specify the acquisition frequency when average_time_difference is entered.")
            if not isinstance(frequency_acquisition, int):
                raise TypeError("frequency_acquisition must be an integer.")
            if abs(average_time_difference) < 1 / frequency_acquisition:
                raise ValueError(
                    "average_time_difference must be bigger than the inverse of the acquisition frequency."
                )

        threshold_positive = np.mean(heapq.nlargest(200, stimulation_signal)) / 2
        threshold_negative = np.mean(heapq.nsmallest(200, stimulation_signal)) / 2
        positive = np.where(stimulation_signal > threshold_positive)
        negative = np.where(stimulation_signal < threshold_negative)

        if negative[0][0] < positive[0][0]:
            stimulation_signal = -stimulation_signal  # invert the signal if the first peak is negative
            threshold = -threshold_negative
        else:
            threshold = threshold_positive
        peaks, _ = find_peaks(stimulation_signal, distance=10, height=threshold)
        time_peaks = []
        for i in range(len(peaks)):
            time_peaks.append(time[peaks[i]])

        if check_stimulation:
            for k in range(len(time_peaks)):
                plt.plot(time_peaks[k], stimulation_signal[peaks[k]], "x")
            plt.plot(time, stimulation_signal)
            plt.show()

        if average_time_difference:
            time_peaks = np.array(time_peaks) + average_time_difference
            peaks = np.array(peaks) + int(average_time_difference * frequency_acquisition)

        if isinstance(time_peaks, np.ndarray):
            time_peaks = time_peaks.tolist()
        if isinstance(peaks, np.ndarray):
            peaks = peaks.tolist()

        return time_peaks, peaks


if __name__ == "__main__":
    ExtractAnalogForceFromC3D(
        # c3d_path=f"D:\These\Experiences\Ergometre_isocinetique\With_FES\Data_with_fes_26_09_2023\Biceps_90deg_30mA_300us_33Hz_essai_fatigue.c3d",
        c3d_path=[
            f"D:\These\Experiences\Ergometre_isocinetique\With_FES\Data_with_fes_26_09_2023\Biceps_90deg_30mA_300us_33Hz_essai1.c3d",
            f"D:\These\Experiences\Ergometre_isocinetique\With_FES\Data_with_fes_26_09_2023\Biceps_90deg_30mA_300us_33Hz_essai2.c3d",
            f"D:\These\Experiences\Ergometre_isocinetique\With_FES\Data_with_fes_26_09_2023\Biceps_90deg_30mA_300us_33Hz_essai3.c3d",
            f"D:\These\Experiences\Ergometre_isocinetique\With_FES\Data_with_fes_26_09_2023\Biceps_70deg_30mA_300us_33Hz_essai1.c3d",
            f"D:\These\Experiences\Ergometre_isocinetique\With_FES\Data_with_fes_26_09_2023\Biceps_70deg_30mA_300us_33Hz_essai2.c3d",
            f"D:\These\Experiences\Ergometre_isocinetique\With_FES\Data_with_fes_26_09_2023\Biceps_70deg_30mA_300us_33Hz_essai3.c3d",
            f"D:\These\Experiences\Ergometre_isocinetique\With_FES\Data_with_fes_26_09_2023\Biceps_110deg_30mA_300us_33Hz_essai1.c3d",
            f"D:\These\Experiences\Ergometre_isocinetique\With_FES\Data_with_fes_26_09_2023\Biceps_110deg_30mA_300us_33Hz_essai2.c3d",
            f"D:\These\Experiences\Ergometre_isocinetique\With_FES\Data_with_fes_26_09_2023\Biceps_110deg_30mA_300us_33Hz_essai3.c3d",
        ],
        calibration_matrix_path="D:\These\Experiences\Ergometre_isocinetique\Capteur 6D\G_201602A1-P (matrice etalonnage).txt",
        for_identification=True,
        average_time_difference=-0.0015,
        frequency_acquisition=10000,
        saving_pickle_path=[
            "identification_data_Biceps_90deg_30mA_300us_33Hz_essai1.pkl",
            "identification_data_Biceps_90deg_30mA_300us_33Hz_essai2.pkl",
            "identification_data_Biceps_90deg_30mA_300us_33Hz_essai3.pkl",
            "identification_data_Biceps_70deg_30mA_300us_33Hz_essai1.pkl",
            "identification_data_Biceps_70deg_30mA_300us_33Hz_essai2.pkl",
            "identification_data_Biceps_70deg_30mA_300us_33Hz_essai3.pkl",
            "identification_data_Biceps_110deg_30mA_300us_33Hz_essai1.pkl",
            "identification_data_Biceps_110deg_30mA_300us_33Hz_essai2.pkl",
            "identification_data_Biceps_110deg_30mA_300us_33Hz_essai3.pkl",
        ],
        # saving_pickle_path="identification_data_Biceps_90deg_30mA_300us_33Hz_essai_fatigue.pkl",
        plot=True,
        check_stimulation=True,
        down_sample=True,
    )
    # input_channel=[6, 0, 1, 2, 3, 4, 5])
