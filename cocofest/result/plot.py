import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from bioptim import Solution
import pickle


class PlotCyclingResult:
    def __init__(self, solution: Solution | str):
        self.sol = solution

    def plot(self, starting_location: str = None, show_rehastim=False):
        """
        Plot the muscle stimulation angle of a cycling motion
        Parameters
        ----------
        starting_location: str
            The starting polar location of the cycling motion. Default is West (W)
        show_rehastim

        Returns
        -------

        """

        extracted_data = self.extract_data_from_sol(self.sol) if isinstance(self.sol, Solution) else None
        is_pickle = True if isinstance(self.sol, str) and self.sol.endswith(".pkl") else False
        extracted_data = self.extract_data_from_pickle(self.sol) if is_pickle else extracted_data

        if extracted_data is None:
            raise ValueError("The solution must be a Solution object or a pickle file")

        fig = plt.figure(figsize=(8, 8))
        fig.suptitle("Muscle stimulation angle of a cycling motion")
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        ax.set_theta_zero_location(starting_location)
        ax.set_theta_direction(-1)
        ax.set_yticklabels([])
        ticks = np.linspace(0, 1, len(extracted_data) + 1)
        ax.set_rticks(ticks)  # To configure radial ticks
        color = ["b", "g", "r", "c", "m", "y"]

        counter = 0
        for muscle in extracted_data:
            if muscle == "empty":
                continue

            bars = ax.bar(
                extracted_data[muscle]["theta"],
                extracted_data[muscle]["radii"],
                width=extracted_data[muscle]["width"],
                bottom=extracted_data[muscle]["bottom"],
                label=extracted_data[muscle]["label"],
                color=color[counter],
                edgecolor="black",
                linewidth=2,
            )

            for i in range(len(bars)):
                bars[i].set_alpha(extracted_data[muscle]["opacity"][i])
            counter += 1

        empty_bar = ax.bar(
            extracted_data["empty"]["theta"],
            extracted_data["empty"]["radii"],
            width=extracted_data["empty"]["width"],
            bottom=extracted_data["empty"]["bottom"],
            label=extracted_data["empty"]["label"],
            linewidth=0,
            color="w",
        )
        for i in range(len(empty_bar)):
            empty_bar[i].set_alpha(0)

        leg = plt.legend()
        for lh in leg.legend_handles:
            lh.set_alpha(1)
        plt.show()

    def plot_rehastim(self):

        triceps_brachii = {
            "theta": np.radians(100),
            "radii": 1 / 5,
            "width": np.radians(160),
            "bottom": 4 / 5,
        }
        biceps_brachii = {
            "theta": np.radians(295),
            "radii": 1 / 5,
            "width": np.radians(150),
            "bottom": 3 / 5,
        }
        deltoideus_anterior = {
            "theta": np.radians(100),
            "radii": 1 / 5,
            "width": np.radians(160),
            "bottom": 2 / 5,
        }
        deltoideus_posterior = {
            "theta": np.radians(295),
            "radii": 1 / 5,
            "width": np.radians(150),
            "bottom": 1 / 5,
        }
        empty = {"theta": 1, "radii": 1 / 5, "width": 1, "bottom": 0}
        stimulated_muscles = {
            "triceps_brachii": triceps_brachii,
            "biceps_brachii": biceps_brachii,
            "deltoideus_anterior": deltoideus_anterior,
            "deltoideus_posterior": deltoideus_posterior,
            "": empty,
        }
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        ax.set_theta_zero_location("W")
        ax.set_theta_direction(-1)
        ax.set_yticklabels([])
        ticks = np.linspace(0, 1, 6)
        ax.set_rticks(ticks)  # Less radial ticks
        theta = []
        radii = []
        width = []
        bottom = []
        for muscle in stimulated_muscles:
            theta.append(stimulated_muscles[muscle]["theta"])
            radii.append(stimulated_muscles[muscle]["radii"])
            width.append(stimulated_muscles[muscle]["width"])
            bottom.append(stimulated_muscles[muscle]["bottom"])
        bars = ax.bar(
            theta,
            radii,
            width=width,
            bottom=bottom,
            label=stimulated_muscles.keys(),
            edgecolor="black",
            linewidth=2,
        )
        color = ["b", "g", "r", "c", "w"]
        for i in range(len(bars)):
            bars[i].set_facecolor(color[i])
            bars[i].set_alpha(0.5)
        bars[-1].set_linewidth(0)
        plt.legend()
        plt.show()

    def extract_data_from_sol(self, solution: Solution):
        data = {}
        n_phase = solution.ocp.n_phases
        width = 2 * np.pi / n_phase
        radii = 1 / (solution.ocp.nlp[0].model.nb_muscles + 1)
        if "pulse_apparition_time" in solution.ocp.parameters.keys():
            final_time = sum(solution.ocp.phase_time)
            pulse_apparition_time = solution.parameters["pulse_apparition_time"]
            theta = np.array([(pulse_apparition_time[i] / final_time) * 2 * np.pi + width / 2 for i in range(n_phase)])
        else:
            theta = np.linspace(0, 2 * np.pi, n_phase + 1)[:-1] + width / 2

        intensity = (
            True
            if sum(["pulse_intensity" in parameter_key for parameter_key in solution.ocp.parameters.keys()]) > 0
            else False
        )
        pulse_duration = (
            True
            if sum(["pulse_duration" in parameter_key for parameter_key in solution.ocp.parameters.keys()]) > 0
            else False
        )
        parameter = "pulse_intensity" if intensity else "pulse_duration" if pulse_duration else None
        if parameter is None:
            raise ValueError(
                "The solution must contain either a pulse intensity or a pulse duration parameter to be plotted with the PlotCyclingResult class"
            )

        counter = 0
        for muscle in solution.ocp.nlp[0].model.muscle_names:
            opacity = []
            parameter_key = parameter + "_" + muscle
            min = solution.ocp.parameter_bounds[0][parameter_key].min[0][0]
            max = solution.ocp.parameter_bounds[0][parameter_key].max[0][0]
            parameter_range = max - min
            for i in range(n_phase):
                value = solution.parameters[parameter_key][i] - min
                opacity_percentage = value / parameter_range
                opacity_percentage = (
                    1 if opacity_percentage > 1 else 0 if opacity_percentage < 0 else opacity_percentage
                )
                opacity.append(opacity_percentage)

            data[muscle] = {
                "theta": theta,
                "radii": radii,
                "width": width,
                "bottom": (counter + 1) / (solution.ocp.nlp[0].model.nb_muscles + 1),
                "opacity": opacity,
                "label": muscle,
            }
            counter += 1

        data = self.add_empty_muscle(data)

        return data

    def extract_data_from_pickle(self, solution: str):
        with open(solution, "rb") as f:
            pickle_data = pickle.load(f)
        data = {}

        n_phase = pickle_data["parameters"][next(iter(pickle_data["parameters"]))].shape[0]
        width = 2 * np.pi / n_phase
        pulse_apparition_time_as_parameter = 1 if "pulse_apparition_time" in pickle_data["parameters"] else 0
        nb_muscle = len(pickle_data["parameters"].keys()) - pulse_apparition_time_as_parameter
        radii = 1 / (nb_muscle + 1)

        if pulse_apparition_time_as_parameter:
            final_time = pickle_data["time"][-1]
            pulse_apparition_time = pickle_data["parameters"]["pulse_apparition_time"]
            theta = np.array([(pulse_apparition_time[i] / final_time) * 2 * np.pi + width / 2 for i in range(n_phase)])
        else:
            theta = np.linspace(0, 2 * np.pi, n_phase + 1)[:-1] + width / 2

        intensity = (
            True
            if sum(["pulse_intensity" in parameter_key for parameter_key in pickle_data["parameters"]]) > 0
            else False
        )
        pulse_duration = (
            True
            if sum(["pulse_duration" in parameter_key for parameter_key in pickle_data["parameters"]]) > 0
            else False
        )
        parameter = "pulse_intensity" if intensity else "pulse_duration" if pulse_duration else None
        if parameter is None:
            raise ValueError(
                "The solution must contain either a pulse intensity or a pulse duration parameter to be plotted with the PlotCyclingResult class"
            )

        counter = 0
        muscle_name_list = list(pickle_data["parameters"].keys())
        (muscle_name_list.remove("pulse_apparition_time") if pulse_apparition_time_as_parameter else None)
        muscle_name_list = [s.replace(parameter + "_", "", 1) for s in muscle_name_list]

        for muscle in muscle_name_list:
            opacity = []
            parameter_key = parameter + "_" + muscle
            min = pickle_data["parameters_bounds"][parameter_key][0]
            max = pickle_data["parameters_bounds"][parameter_key][1]
            parameter_range = max - min
            for i in range(n_phase):
                value = pickle_data["parameters"][parameter_key][i] - min
                opacity_percentage = value / parameter_range
                opacity_percentage = (
                    1 if opacity_percentage > 1 else 0 if opacity_percentage < 0 else opacity_percentage
                )
                opacity.append(opacity_percentage)

            data[muscle] = {
                "theta": theta,
                "radii": radii,
                "width": width,
                "bottom": (counter + 1) / (nb_muscle + 1),
                "opacity": opacity,
                "label": muscle,
            }
            counter += 1

        data = self.add_empty_muscle(data)

        return data

    @staticmethod
    def add_empty_muscle(data):
        empty = {
            "theta": 1,
            "radii": 1 / len(data),
            "width": 1,
            "bottom": 0,
            "opacity": 0,
            "label": "",
        }
        data["empty"] = empty
        return data

    @staticmethod
    def rehamove_data():
        triceps_brachii = {
            "theta": np.radians(100),
            "radii": 1 / 5,
            "width": np.radians(160),
            "bottom": 4 / 5,
        }
        biceps_brachii = {
            "theta": np.radians(295),
            "radii": 1 / 5,
            "width": np.radians(150),
            "bottom": 3 / 5,
        }
        deltoideus_anterior = {
            "theta": np.radians(100),
            "radii": 1 / 5,
            "width": np.radians(160),
            "bottom": 2 / 5,
        }
        deltoideus_posterior = {
            "theta": np.radians(295),
            "radii": 1 / 5,
            "width": np.radians(150),
            "bottom": 1 / 5,
        }
        empty = {"theta": 1, "radii": 1 / 5, "width": 1, "bottom": 0}
        stimulated_muscles = {
            "triceps_brachii": triceps_brachii,
            "biceps_brachii": biceps_brachii,
            "deltoideus_anterior": deltoideus_anterior,
            "deltoideus_posterior": deltoideus_posterior,
            "": empty,
        }
        return stimulated_muscles
