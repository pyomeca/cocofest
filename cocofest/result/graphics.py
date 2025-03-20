from bioptim import Solution, SolutionMerge, InterpolationType
import matplotlib.pyplot as plt
import numpy as np
from ..models.ding2007 import DingModelPulseWidthFrequency
from ..models.dynamical_model import FesMskModel


class FES_plot:
    def __init__(self, data: Solution | dict):
        self.data = data

        # --- Default Values for Annotation ---
        Y_STEP = 0.05

        self.y_step = Y_STEP
        self.identifiable_parameters = [
            "a_rest",
            "km_rest",
            "tau1_rest",
            "tau2",
            "pd0",
            "pdt",
            "a_scale",
            "ar",
            "bs",
            "Is",
            "cr",
        ]
        self.default_decimal_values = {
            "a_rest": 0,
            "km_rest": 3,
            "tau1_rest": 6,
            "tau2": 3,
            "pd0": 9,
            "pdt": 9,
            "a_scale": 0,
            "ar": 3,
            "bs": 3,
            "Is": 1,
            "cr": 3,
        }

    def plot(
        self,
        title: str = None,
        show_stim: bool = False,
        show_bounds: bool = False,
        tracked_data=None,
        default_model=None,
    ):
        if isinstance(self.data, Solution):
            if isinstance(self.data.ocp.nlp[0].model, FesMskModel):
                self.msk_plot(title, show_stim, show_bounds)
            elif any(parameter in self.data.parameters.keys() for parameter in self.identifiable_parameters):
                self.id_plot(title=title, tracked_data=tracked_data, default_model=default_model)
            else:
                self.ocp_plot(title, show_stim, show_bounds)

        elif isinstance(self.data, dict):
            if show_stim or show_bounds:
                raise ValueError("Cannot show stim or bounds with data dictionary type")
            self.ivp_plot(title)
        else:
            raise ValueError("Data must be a Solution or a dictionary")

    def get_data(self, solution: Solution):
        states = solution.stepwise_states(to_merge=SolutionMerge.NODES)
        controls = solution.stepwise_controls(to_merge=SolutionMerge.NODES)
        time = solution.stepwise_time(to_merge=SolutionMerge.NODES).T[0]
        force = states[self.force_keys[0]]
        for i in range(1, len(self.force_keys)):
            force = np.append(force, states[self.force_keys[i]], axis=0)
        q = states["q"] if "q" in states else None
        qdot = states["qdot"] if "qdot" in states else None
        tau = controls["tau"] if "tau" in controls else None
        return q, qdot, tau, force, time

    def get_msk_bounds(self, solution: Solution):
        q_bounds = [solution.ocp.nlp[0].x_bounds["q"].min[0], solution.ocp.nlp[0].x_bounds["q"].max[0]]
        qdot_bounds = [solution.ocp.nlp[0].x_bounds["qdot"].min[0], solution.ocp.nlp[0].x_bounds["qdot"].max[0]]
        tau_bounds = (
            [solution.ocp.nlp[0].u_bounds["tau"].min[0], solution.ocp.nlp[0].u_bounds["tau"].max[0]]
            if "tau" in solution.ocp.nlp[0].u_bounds
            else None
        )
        force_bounds = {}
        for i in range(len(self.force_keys)):
            force_bounds[self.force_keys[i]] = [
                solution.ocp.nlp[0].x_bounds[self.force_keys[i]].min[0],
                solution.ocp.nlp[0].x_bounds[self.force_keys[i]].max[0],
            ]
        bounds = {"q": q_bounds, "qdot": qdot_bounds, "tau": tau_bounds, "force": force_bounds}
        return bounds

    def get_bounds(self, solution: Solution):
        if solution.ocp.nlp[0].x_bounds.type == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            solution_bounds = solution.ocp.nlp[0].x_bounds
            cn_bounds = [list(solution_bounds["Cn"].min[0]), list(solution_bounds["Cn"].max[0])]
            force_bounds = [list(solution_bounds["F"].min[0]), list(solution_bounds["F"].max[0])]
            bounds = {"cn": cn_bounds, "force": force_bounds}
            if solution.ocp.nlp[0].model._with_fatigue:
                a_bounds = [list(solution_bounds["A"].min[0]), list(solution_bounds["A"].max[0])]
                tau1_bounds = [list(solution_bounds["Tau1"].min[0]), list(solution_bounds["Tau1"].max[0])]
                km_bounds = [list(solution_bounds["Km"].min[0]), list(solution_bounds["Km"].max[0])]
                bounds = {"cn": cn_bounds, "force": force_bounds, "a": a_bounds, "tau1": tau1_bounds, "km": km_bounds}

            temp_time = solution.decision_time(to_merge=SolutionMerge.NODES).T[0]
            bound_time = [temp_time[0], temp_time[1], temp_time[-1]]

        elif solution.ocp.nlp[0].x_bounds.type == InterpolationType.EACH_FRAME:
            solution_bounds = solution.ocp.nlp[0].x_bounds
            cn_bounds = [list(solution_bounds["Cn"].min[0]), list(solution_bounds["Cn"].max[0])]
            force_bounds = [list(solution_bounds["F"].min[0]), list(solution_bounds["F"].max[0])]
            bounds = {"cn": cn_bounds, "force": force_bounds}

            if solution.ocp.nlp[0].model._with_fatigue:
                a_bounds = [list(solution_bounds["A"].min[0]), list(solution_bounds["A"].max[0])]
                tau1_bounds = [list(solution_bounds["Tau1"].min[0]), list(solution_bounds["Tau1"].max[0])]
                km_bounds = [list(solution_bounds["Km"].min[0]), list(solution_bounds["Km"].max[0])]
                bounds = {"cn": cn_bounds, "force": force_bounds, "a": a_bounds, "tau1": tau1_bounds, "km": km_bounds}

            bound_time = solution.decision_time(to_merge=SolutionMerge.NODES).T[0]

        else:
            raise NotImplementedError("Bounds type not implemented")

        return bounds, bound_time

    def axes_settings(self, axes_list):
        for i in range(len(axes_list)):
            offset = -0.2 * i if len(axes_list) > 1 else -0.05
            axes_list[i].spines["left"].set_position(("axes", offset))
            axes_list[i].set_frame_on(True)
            for spine in axes_list[i].spines.values():
                spine.set_visible(False)
            axes_list[i].spines["left"].set_visible(True)

    def build_several_y_axis(self, axis, time, values, labels: list = None):
        n = values.shape[0]
        cmap = plt.get_cmap("tab20", n)
        colors = [cmap(i) for i in range(n)]
        axes_list = [axis] + [axis.twinx() for _ in range(values.shape[0] - 1)]

        lines_list, labels_list = [], []
        for i in range(len(axes_list)):
            axes_list[i].spines["left"].set_color(colors[i])
            axes_list[i].yaxis.set_label_position("left")
            axes_list[i].yaxis.set_ticks_position("left")
            axes_list[i].plot(time, values[i], color=colors[i], label=labels[i], lw=3)
            axes_list[i].set_ylabel(labels[i])
            lines, fig_labels = axes_list[i].get_legend_handles_labels()
            lines_list.append(lines)
            labels_list.append(fig_labels)

        self.axes_settings(axes_list)
        axis.legend(lines_list, labels_list)
        axis.set_xlabel("Time [s]")

    def build_several_y_axis_FES(
        self, axis, time, values, labels: list = None, stim_time=None, stim_values=None, axes_title=None
    ):
        n = len(labels)
        cmap = plt.get_cmap("tab20b", n)
        colors = [cmap(i) for i in range(n)]
        axes_list = [axis, axis.twinx()]
        axes_colors = ["red", "green"]
        stim_values_keys = list(stim_values.keys())

        lines_list, labels_list = [], []
        for i in range(len(axes_list)):
            axes_list[i].spines["left"].set_color(axes_colors[i])
            axes_list[i].yaxis.set_label_position("left")
            axes_list[i].yaxis.set_ticks_position("left")
            axes_list[i].set_ylabel(axes_title[i])
            lines, fig_labels = axes_list[i].get_legend_handles_labels()
            lines_list.append(lines)
            labels_list.append(fig_labels)
            if i == 0:
                [axes_list[i].plot(time, values[j], color=colors[j], label=labels[j], lw=3) for j in range(n)]
            if i == 1:
                stim_offset = (
                    [(i - n // 2) * 0.01 for i in range(n)]
                    if n % 2 != 0
                    else [-(n / 2) * 0.01 + 0.01 * i for i in range(n)]
                )
                for j in range(n):
                    [
                        axes_list[i].axvline(
                            x=stim_time[k] + stim_offset[j],
                            ymin=0,
                            ymax=stim_values[stim_values_keys[j]][k],
                            color=colors[j],
                            linestyle="--",
                            lw=1,
                        )
                        for k in range(len(stim_time))
                    ]

        self.axes_settings(axes_list)
        axis.legend(lines_list, labels_list)
        axis.set_xlabel("Time [s]")

    def create_twin_axes(self, ax, time, data, label, color, lw, ylabel, tick_color):
        """Create a twin y-axis on ax with custom styling."""
        twin_ax = ax.twinx()
        twin_ax.spines["left"].set_color(tick_color)
        twin_ax.yaxis.set_label_position("left")
        twin_ax.yaxis.set_ticks_position("left")
        (line,) = twin_ax.plot(time, data, color=color, label=label, lw=lw)
        twin_ax.set_ylabel(ylabel)
        self.axes_settings([twin_ax])
        return twin_ax, line

    def ocp_plot(self, title: str = None, show_stim: bool = True, show_bounds: bool = True):
        solution = self.data
        states = solution.stepwise_states(to_merge=SolutionMerge.NODES)
        time = solution.stepwise_time(to_merge=SolutionMerge.NODES).T[0]
        bounds, bounds_time = self.get_bounds(solution) if show_bounds else None, None
        stim_time = solution.ocp.nlp[0].model.stim_time if show_stim else None

        # Extract data from solution
        cn = states["Cn"][0]
        force = states["F"][0]
        if solution.ocp.nlp[0].model._with_fatigue:
            a = states["A"][0]
            km = states["Km"][0]
            tau1 = states["Tau1"][0]

        fatigue_model_used = solution.ocp.nlp[0].model._with_fatigue
        # Create subplots
        nrows = 2 if fatigue_model_used else 1
        fig, axs = plt.subplots(nrows, 1, figsize=(12, 9))

        if nrows == 1:
            axs = [axs]

        # --- Force Model --- #
        ax1 = axs[0]
        # Plot Cn
        (line_cn,) = ax1.plot(time, cn, color="royalblue", label="Cn", lw=2)
        ax1.set_ylabel("Cn (-)", fontsize=12, color="royalblue")
        ax1.set_xlabel("Time (s)", fontsize=12)
        ax1.set_title("Force model", fontsize=14, fontweight="bold")
        ax1.tick_params(axis="y", colors="royalblue")
        ax1.grid(True, linestyle="--", alpha=0.7)

        # Create twin for Force
        ax1_force, line_force = self.create_twin_axes(
            ax1, time, force, label="Force", color="darkred", lw=3, ylabel="Force (N)", tick_color="darkred"
        )

        if show_stim:
            for stim in stim_time:
                ax1.axvline(stim, color="goldenrod", linestyle="--", lw=1)
            offset = 0.01 * (stim_time[-1] - stim_time[0]) / (stim_time[-1] - stim_time[0])
            ax1.text(stim_time[0] - offset, 0, "Stimulation", rotation=90, color="goldenrod", fontsize=10)
        if show_bounds:
            ax1.fill_between(bounds_time, bounds["cn"][0], bounds["cn"][1], color="royalblue", alpha=0.2)
            ax1_force.fill_between(bounds_time, bounds["force"][0], bounds["force"][1], color="darkred", alpha=0.2)

        # Combine legends from ax1 and its twin
        lines = [line_cn, line_force]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, fontsize=10, fancybox=True, shadow=True)

        # --- Fatigue Model --- #
        if fatigue_model_used:
            ax2 = axs[1]
            # Plot A on ax2
            (line_a,) = ax2.plot(time, a, color="forestgreen", label="A", lw=3)
            ax2.set_ylabel("A (-)", fontsize=12, color="forestgreen")
            ax2.set_xlabel("Time (s)", fontsize=12)
            ax2.set_title("Fatigue model", fontsize=14, fontweight="bold")
            ax2.tick_params(axis="y", colors="forestgreen")
            ax2.grid(True, linestyle="--", alpha=0.7)

            # Create twin for Km and Tau1
            ax2_twin = ax2.twinx()
            ax2_twin.spines["left"].set_color("crimson")
            ax2_twin.yaxis.set_label_position("left")
            ax2_twin.yaxis.set_ticks_position("left")
            (line_km,) = ax2_twin.plot(time, km, color="crimson", label="Km", lw=3)
            (line_tau1,) = ax2_twin.plot(time, tau1, color="purple", label="Tau1", lw=3)
            ax2_twin.set_ylabel("Km (-) & Tau1 (s)", fontsize=12, color="crimson")
            ax2_twin.tick_params(axis="y", colors="crimson")
            self.axes_settings([ax2_twin])

            # Combine legends for the fatigue model subplot
            lines2 = [line_a, line_km, line_tau1]
            labels2 = [line.get_label() for line in lines2]
            ax2.legend(lines2, labels2, fontsize=10, fancybox=True, shadow=True)

            if show_bounds:
                ax2.fill_between(bounds_time, bounds["a"][0], bounds["a"][1], color="forestgreen", alpha=0.2)
                ax2_twin.fill_between(bounds_time, bounds["km"][0], bounds["km"][1], color="crimson", alpha=0.2)
                ax2_twin.fill_between(bounds_time, bounds["tau1"][0], bounds["tau1"][1], color="purple", alpha=0.2)

        fig.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()

    def ivp_plot(self, title: str = None):
        """
        Plot the force and (optionally) fatigue models.

        Parameters:
            title (str, optional): Title for the figure.
        """
        result = self.data
        time = result["time"]
        # Extract data from result
        cn = result["Cn"][0]
        force = result["F"][0]
        fatigue_model_used = "A" in result

        if fatigue_model_used:
            a = result["A"][0]
            km = result["Km"][0]
            tau1 = result["Tau1"][0]

        # Create subplots
        nrows = 2 if fatigue_model_used else 1
        fig, axs = plt.subplots(nrows, 1, figsize=(12, 9))

        if nrows == 1:
            axs = [axs]

        # --- Force Model --- #
        ax1 = axs[0]
        # Plot Cn
        (line_cn,) = ax1.plot(time, cn, color="royalblue", label="Cn", lw=2)
        ax1.set_ylabel("Cn (-)", fontsize=12, color="royalblue")
        ax1.set_xlabel("Time (s)", fontsize=12)
        ax1.set_title("Force model", fontsize=14, fontweight="bold")
        ax1.tick_params(axis="y", colors="royalblue")
        ax1.grid(True, linestyle="--", alpha=0.7)

        # Create twin for Force
        ax1_force, line_force = self.create_twin_axes(
            ax1, time, force, label="Force", color="darkred", lw=3, ylabel="Force (N)", tick_color="darkred"
        )

        # Combine legends from ax1 and its twin
        lines = [line_cn, line_force]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, fontsize=10, fancybox=True, shadow=True)

        # --- Fatigue Model --- #
        if fatigue_model_used:
            ax2 = axs[1]
            # Plot A on ax2
            (line_a,) = ax2.plot(time, a, color="forestgreen", label="A", lw=3)
            ax2.set_ylabel("A (-)", fontsize=12, color="forestgreen")
            ax2.set_xlabel("Time (s)", fontsize=12)
            ax2.set_title("Fatigue model", fontsize=14, fontweight="bold")
            ax2.tick_params(axis="y", colors="forestgreen")
            ax2.grid(True, linestyle="--", alpha=0.7)

            # Create twin for Km and Tau1
            ax2_twin = ax2.twinx()
            ax2_twin.spines["left"].set_color("crimson")
            ax2_twin.yaxis.set_label_position("left")
            ax2_twin.yaxis.set_ticks_position("left")
            (line_km,) = ax2_twin.plot(time, km, color="crimson", label="Km", lw=3)
            (line_tau1,) = ax2_twin.plot(time, tau1, color="purple", label="Tau1", lw=3)
            ax2_twin.set_ylabel("Km (-) & Tau1 (s)", fontsize=12, color="crimson")
            ax2_twin.tick_params(axis="y", colors="crimson")
            self.axes_settings([ax2_twin])

            # Combine legends for the fatigue model subplot
            lines2 = [line_a, line_km, line_tau1]
            labels2 = [line.get_label() for line in lines2]
            ax2.legend(lines2, labels2, fontsize=10, fancybox=True, shadow=True)

        fig.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()

    def msk_plot(self, title: str = None, show_stim: bool = True, show_bounds: bool = True):
        solution = self.data
        self.force_keys = [key for key in solution.stepwise_states().keys() if key.startswith("F_")]
        q, qdot, tau, force, time = self.get_data(solution)
        bounds = self.get_msk_bounds(solution) if show_bounds else None

        fig, axs = plt.subplots(2, 2, figsize=(12, 9))

        # Q states
        q_keys = solution.ocp.nlp[0].model.bio_model.name_dof
        q_keys = ["q_" + key for key in q_keys]
        self.build_several_y_axis(axs[0, 0], time, q, labels=q_keys)
        # Set titles/labels
        axs[0, 0].set_title("Joint angles (rad)")

        # Qdot states
        qdot_keys = solution.ocp.nlp[0].model.bio_model.name_dof
        qdot_keys = ["qdot_" + key for key in qdot_keys]
        self.build_several_y_axis(axs[0, 1], time, qdot, labels=qdot_keys)
        # Set titles/labels
        axs[0, 1].set_title("Joint velocity (rad/s)")

        # Tau
        tau_keys = solution.ocp.nlp[0].model.bio_model.name_dof
        tau_keys = ["tau_" + key for key in tau_keys]
        tau_time = solution.decision_time(to_merge=SolutionMerge.NODES).T[0][:-1]
        self.build_several_y_axis(axs[1, 0], tau_time, tau, labels=tau_keys)
        # Set titles/labels
        axs[1, 0].set_title("Torque (N.m)")

        # Forces
        stim_time = np.array(solution.ocp.nlp[0].model.muscles_dynamics_model[0].stim_time)
        stim_values = solution.parameters
        # Todo normalize stim_value to yaxis scale

        force_label = "FES forces (N)"

        fes_parameter_label = (
            "Pulse width (us)"
            if isinstance(solution.ocp.nlp[0].model.muscles_dynamics_model[0], DingModelPulseWidthFrequency)
            else "Pulse intensity (mA)"
        )
        axes_title = [force_label, fes_parameter_label]
        self.build_several_y_axis_FES(
            axs[1, 1],
            time,
            force,
            labels=self.force_keys,
            stim_time=stim_time,
            stim_values=stim_values,
            axes_title=axes_title,
        )

        plt.legend()
        plt.tight_layout()
        plt.show()

    def extract_identified_parameters(self, identified):
        """
        For each parameter in keys, use the identified value if available.
        Returns a dictionary mapping parameter names to their values.
        """
        solution = self.data
        return {key: identified.parameters[key][0] for key in solution.parameters.keys()}

    def annotate_parameters(self, ax, identified_params, default_model=None):
        """
        Annotate the plot with parameter names, the identified values, and default values.
        The names are annotated in black, identified values in red, and default values in blue.
        """
        for i, key in enumerate(identified_params.keys()):
            y = 0.99 - i * self.y_step
            ax.annotate(f"{key} :", xy=(0.7, y), xycoords="axes fraction", color="black", ha="right", va="top")
            ax.annotate(
                f"{round(identified_params[key], min(self.default_decimal_values[key], 6))}",
                xy=(0.99, y),
                xycoords="axes fraction",
                color="red",
                ha="right",
                va="top",
            )

            if default_model:
                ax.annotate(
                    f"{round(getattr(default_model, key), min(self.default_decimal_values[key], 6))}",
                    xy=(0.85, y),
                    xycoords="axes fraction",
                    color="blue",
                    ha="right",
                    va="top",
                )

    def id_plot(
        self, title: str = None, show_stim: bool = True, show_bounds: bool = True, tracked_data=None, default_model=None
    ):
        solution = self.data
        identified_params = self.extract_identified_parameters(solution)

        print("Identified parameters:")
        for key, value in identified_params.items():
            print(f"  {key}: {value}")

        sol_time = solution.stepwise_time(to_merge=SolutionMerge.NODES).T[0]
        sol_force = solution.stepwise_states(to_merge=SolutionMerge.NODES)["F"][0]

        # Plot the simulation and identification results
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("force (N)")

        ax.plot(sol_time, sol_force, color="red", label="identified")

        if tracked_data:
            tracked_data_time = tracked_data["time"]
            tracked_data_force = tracked_data["force"]

            ax.plot(tracked_data_time, tracked_data_force, color="blue", label="simulated")

        self.annotate_parameters(ax, identified_params, default_model)

        ax.legend()
        plt.show()
