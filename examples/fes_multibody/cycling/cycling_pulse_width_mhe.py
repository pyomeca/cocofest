"""
This example will perform an optimal control program moving time horizon for a hand cycling motion driven by FES.
"""

import os
import pickle
from sys import platform
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.ma.extras import average

import cocofest._matplotlib_compat  # Temporary fix, see cocofest/_matplotlib_compat.py
from bioptim import (
    Axis,
    BiorbdModel,
    BoundsList,
    ConstraintList,
    ConstraintFcn,
    CostType,
    ExternalForceSetTimeSeries,
    InitialGuessList,
    InterpolationType,
    MultiCyclicCycleSolutions,
    MultiCyclicNonlinearModelPredictiveControl,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    PhaseDynamics,
    ParameterObjectiveList,
    SolutionMerge,
    Solution,
    Solver,
    ParameterList,
    Node,
    VariableScalingList,
    VariableScaling,
)
from cocofest import (
    DingModelPulseWidthFrequencyWithFatigue,
    FesMskModel,
    inverse_kinematics_cycling,
    OcpFesMsk,
    FesMheMsk,
)
from examples.fes_multibody.cycling.cost_functions import CustomCostFunctions


# --------------------#
#    MHE functions    #
# --------------------#
class MyCyclicMHE(FesMheMsk):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nodes_per_cycle = self.cycle_len * (
            self.nlp[0].dynamics_type.ode_solver.polynomial_degree + 1
            if isinstance(self.nlp[0].dynamics_type.ode_solver, OdeSolver.COLLOCATION)
            else 1
        )
        self.pedal_turn_in_one_cycle = 2 * np.pi  # One mhe cycle simulates on pedal turn
        self.debugg_bounds = False
        self.previous_bounds = None

    def advance_window_bounds_states(self, sol, n_cycles_simultaneous=None, **extra):
        # --- Get states results --- #
        states = sol.decision_states(to_merge=SolutionMerge.NODES)
        states_keys = states.keys()
        # --- Store previous state bounds for debugg purpose --- #
        if self.debugg_bounds:
            self.previous_bounds = {}
            for key in states_keys:
                xb = self.nlp[0].x_bounds[key]
                self.previous_bounds[key] = {
                    "min": xb.min[:, : self.nodes_per_cycle].copy(),
                    "max": xb.max[:, : self.nodes_per_cycle].copy(),
                }

        # --- States are bounded to match the last node of the cycle to ensure continuity between window --- #
        for key in states_keys:
            for i in range(states[key].shape[0]):
                # --- Only doing wheel to prevent over constraining the system --- #
                if key == "q" or key == "qdot":
                    if i == 2:
                        self.nlp[0].x_bounds[key].min[i, 0] = states[key][i][self.nodes_per_cycle]
                        self.nlp[0].x_bounds[key].max[i, 0] = states[key][i][self.nodes_per_cycle]
                    if key == "q" and i == 2:
                        self.nlp[0].x_bounds[key].min[i, 0] = (
                            self.nlp[0].x_bounds["q"].min[i, 0] + self.pedal_turn_in_one_cycle
                        )
                        self.nlp[0].x_bounds[key].max[i, 0] = (
                            self.nlp[0].x_bounds["q"].max[i, 0] + self.pedal_turn_in_one_cycle
                        )
                else:
                    self.nlp[0].x_bounds[key].min[i, 0] = states[key][i][self.nodes_per_cycle]
                    self.nlp[0].x_bounds[key].max[i, 0] = states[key][i][self.nodes_per_cycle]
        # --- Inform the past cycle stimulation time into the new one --- #
        self.update_stim()
        return True

    def advance_window_initial_guess_states(self, sol, n_cycles_simultaneous=None):
        # --- Get states results --- #
        states = sol.decision_states(to_merge=SolutionMerge.NODES)
        states_keys = states.keys()
        cyclical_keys = [s for s in states if any(s.startswith(prefix) for prefix in ("Cn_", "F_", "q", "qdot"))]
        continuous_keys = [s for s in states if any(s.startswith(prefix) for prefix in ("A_", "Tau1_", "Km_"))]
        # --- Set initial guesses for cyclical and continuous states --- #
        for key in states_keys:
            for i in range(states[key].shape[0]):
                if key in cyclical_keys:
                    if key == "q" and i == states[key].shape[0] - 1:
                        # Special case for the wheel position
                        self.set_init_cyclical_wheel(states, key, i)
                    else:
                        self.set_init_cyclical(states, key, i)
                elif key in continuous_keys:
                    self.set_init_continuous(states, key, i)
        self._correct_init_guess_to_fit_bounds(
            corrected_input="states"
        )  # This function is called to move init guess within the bounds if not in bounds

        # --- Print bounds and initial guesses for debugg purpose --- #
        if self.debugg_bounds:
            for key in states.keys():
                self.plot_initial_guess(
                    data=self.nlp[0].x_init[key].init,
                    current_bounds=self.nlp[0].x_bounds[key],
                    past_bounds=self.previous_bounds[key],
                    key=key,
                )
        return True

    def advance_window_initial_guess_controls(self, sol, n_cycles_simultaneous=None):
        # --- Get control results --- #
        controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
        controls_keys = controls.keys()

        # --- Store previous control bounds for debugg purpose --- #
        if self.debugg_bounds:
            self.previous_bounds = {}
            for key in controls_keys:
                ub = self.nlp[0].u_bounds[key]
                self.previous_bounds[key] = {
                    "min": ub.min[:, : self.nodes_per_cycle].copy(),
                    "max": ub.max[:, : self.nodes_per_cycle].copy(),
                }

        # --- Set initial guess for controls --- #
        for key in controls.keys():
            self.set_init_cyclical(controls, key, 0, False)
        self._correct_init_guess_to_fit_bounds(
            corrected_input="controls"
        )  # This function is called to move init guess within the bounds if not in bounds

        # --- Print bounds and initial guesses for debugg purpose --- #
        if self.debugg_bounds:
            for key in controls_keys:
                self.plot_initial_guess(
                    data=self.nlp[0].u_init[key].init,
                    current_bounds=self.nlp[0].u_bounds[key],
                    past_bounds=self.previous_bounds[key],
                    key=key,
                )
        return True

    def set_init_continuous(self, states, key, i):
        n_plus_one_cycles = states[key][i][self.nodes_per_cycle : -1]
        last_cycle = states[key][i][-self.nodes_per_cycle - 1 :]
        delta = n_plus_one_cycles[-1] - last_cycle[0]
        shifted_last_cycle = states[key][i][-self.nodes_per_cycle - 1 :] + delta
        values = np.concatenate((n_plus_one_cycles, shifted_last_cycle))
        self.nlp[0].x_init[key].init[:, :] = values
        return True

    def set_init_cyclical(self, data, key, i, state=True):
        n_plus_one_cycles = data[key][i][self.nodes_per_cycle : -1]
        last_cycle = data[key][i][-self.nodes_per_cycle - 1 :]
        values = np.concatenate((n_plus_one_cycles, last_cycle))
        if state:
            self.nlp[0].x_init[key].init[i, :] = values
        else:
            self.nlp[0].u_init[key].init[i, :] = values
        return True

    def set_init_cyclical_wheel(self, states, key, i):
        shifted_n_plus_one_cycles = states[key][i][self.nodes_per_cycle : -1] + self.pedal_turn_in_one_cycle
        last_cycle = states[key][i][-self.nodes_per_cycle - 1 :]
        values = np.concatenate((shifted_n_plus_one_cycles, last_cycle))
        self.nlp[0].x_init[key].init[i, :] = values
        return True

    def _correct_init_guess_to_fit_bounds(self, corrected_input="states"):
        corrected_data_input = (
            self.nlp[0].x_init
            if corrected_input == "states"
            else self.nlp[0].u_init if corrected_input == "controls" else None
        )
        corrected_bound_input = (
            self.nlp[0].x_bounds
            if corrected_input == "states"
            else self.nlp[0].u_bounds if corrected_input == "controls" else None
        )
        if corrected_data_input is None or corrected_bound_input is None:
            raise ValueError("Input must be either 'states' or 'controls'.")
        # This function is called to move init guess within the bounds if not in bounds
        for key in corrected_data_input.keys():
            data = corrected_data_input[key].init
            bounds = corrected_bound_input[key]
            for i in range(data.shape[0]):
                if bounds.min.shape == data.shape:
                    min_bounds = bounds.min[:, :][i]
                    max_bounds = bounds.max[:, :][i]
                else:
                    min_bounds = [bounds.min[i][0], *[bounds.min[i][1]] * (data.shape[1] - 2), bounds.min[i][2]]
                    max_bounds = [bounds.max[i][0], *[bounds.max[i][1]] * (data.shape[1] - 2), bounds.max[i][2]]

                for j in range(data.shape[1]):
                    if data[:, :][i][j] < min_bounds[j]:
                        corrected_data_input[key].init[i, j] = min_bounds[j]
                    if data[:, :][i][j] > max_bounds[j]:
                        corrected_data_input[key].init[i, j] = max_bounds[j]

    def plot_initial_guess(self, data, current_bounds, past_bounds, key):
        for i in range(data.shape[0]):
            if current_bounds.min.shape == data.shape:
                current_min_bounds = current_bounds.min[:, :][i]
                current_max_bounds = current_bounds.max[:, :][i]
            else:
                current_min_bounds = [
                    current_bounds.min[i][0],
                    *[current_bounds.min[i][1]] * (data.shape[1] - 2),
                    current_bounds.min[i][2],
                ]
                current_max_bounds = [
                    current_bounds.max[i][0],
                    *[current_bounds.max[i][1]] * (data.shape[1] - 2),
                    current_bounds.max[i][2],
                ]

            if past_bounds["min"].shape[1] == self.nodes_per_cycle:
                past_min_bounds = past_bounds["min"][i]
                past_max_bounds = past_bounds["max"][i]
            else:
                past_min_bounds = [past_bounds["min"][i][0], *[past_bounds["min"][i][1]] * (self.nodes_per_cycle - 1)]
                past_max_bounds = [past_bounds["max"][i][0], *[past_bounds["max"][i][1]] * (self.nodes_per_cycle - 1)]

            fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [4, 1]})
            fig.suptitle("Bounds and initial guess of " + key + " " + "index n°" + str(i), size=14, weight="bold")

            current_time_index = list(np.linspace(0, self.n_cycles_simultaneous, data[:, :][i].shape[0]))
            axs[0].plot(current_time_index, data[:, :][i], label="Initial guess", color="black", lw=3)
            axs[0].plot(
                current_time_index, current_min_bounds, linestyle="-", label="Current bound", color="grey", lw=1
            )
            axs[0].plot(current_time_index, current_max_bounds, linestyle="-", color="grey", lw=1)

            past_time_index = np.linspace(-1, 0, self.nodes_per_cycle)
            axs[0].plot(
                past_time_index, past_min_bounds, linestyle="-", label="Previous bound", color="lightcoral", lw=1
            )
            axs[0].plot(past_time_index, past_max_bounds, linestyle="-", color="lightcoral", lw=1)

            labeled = False
            for j in range(data.shape[1]):
                if data[:, :][i][j] < current_min_bounds[j] or data[:, :][i][j] > current_max_bounds[j]:
                    axs[0].scatter(
                        current_time_index[j],
                        data[:, :][i][j],
                        color="red",
                        s=10,
                        label="out of bounds" if not labeled else None,
                    )
                    labeled = True
            axs[0].legend()

            axs[1].plot(past_time_index, past_max_bounds, linestyle="-", color="lightcoral", lw=1)
            axs[1].set_ylim([0, 1])
            axs[1].axvspan(-1, 0, color="lightcoral", alpha=0.5)
            axs[1].text(-0.5, 0.5, "Cycle n-1", ha="center", va="center", size=15, weight="bold")

            for j in range(self.n_cycles_simultaneous):
                axs[1].axvspan(j, j + 1, color="lightgreen", alpha=0.5 - 0.05 * j)
                axs[1].text(
                    j + 0.5, 0.5, f'Cycle n{f"+{j}" if j > 0 else ""}', ha="center", va="center", size=15, weight="bold"
                )

            axs[1].set_ylim(0, 1)
            axs[1].set_yticks([])
            axs[1].set_xlabel("Time (s)", size=15, weight="bold")

            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()


# --------------------#
#    OCP functions    #
# --------------------#
def prepare_mhe(
    model: BiorbdModel | FesMskModel,
    mhe_info: dict,
    cycling_info: dict,
    simulation_conditions: dict,
):
    # --- Initialize parameters from dictionaries --- #
    # --- MHE info --- #
    cycle_duration = mhe_info["cycle_duration"]
    cycle_len = mhe_info["cycle_len"]
    n_cycles_to_advance = mhe_info["n_cycles_to_advance"]
    n_cycles_simultaneous = mhe_info["n_cycles_simultaneous"]
    ode_solver = mhe_info["ode_solver"]
    use_sx = mhe_info["use_sx"]

    # --- Initial guess file info --- #
    initial_guess_path = simulation_conditions["init_guess_file_path"]

    window_n_shooting = cycle_len * n_cycles_simultaneous
    window_cycle_duration = cycle_duration * n_cycles_simultaneous

    # --- Cycling info --- #
    turn_number = cycling_info["turn_number"]
    pedal_config = cycling_info["pedal_config"]
    external_force = cycling_info["resistive_torque"]

    # --- Cost function info --- #
    objective_fun_dict = {
        "cost_fun_key": simulation_conditions["cost_fun_key"],
        "cost_fun_weight": (
            0.1
            if "weight" in simulation_conditions["cost_fun_key"][0]
            else 1 if "bayesian" in simulation_conditions["cost_fun_key"][0] else 10000
        ),
        "individual_quadratic": True,
    }

    # --- Set dynamics --- #
    # --- External force numerical time series --- #
    numerical_time_series, external_force_set = set_external_forces(
        n_shooting=window_n_shooting, external_force_dict=external_force, force_name="external_torque"
    )
    # --- Stimulation instant numerical time series --- #
    numerical_data_time_series, stim_idx_at_node_list = model.muscles_dynamics_model[0].get_numerical_data_time_series(
        window_n_shooting, window_cycle_duration
    )
    numerical_time_series.update(numerical_data_time_series)
    # --- Dynamics --- #
    dynamics_options = set_dynamics_options(numerical_time_series=numerical_time_series, ode_solver=ode_solver)

    # --- Set states --- #
    # --- Set q (position and speed) initial guesses --- #
    x_init = set_q_qdot_init(
        n_shooting=window_n_shooting,
        pedal_config=pedal_config,
        turn_number=turn_number,
        ode_solver=ode_solver,
        init_file_path=initial_guess_path,
    )

    # --- Set bounds and FES initial guesses --- #
    x_bounds, x_init = set_x_bounds(
        model=model,
        x_init=x_init,
        n_shooting=window_n_shooting,
        ode_solver=ode_solver,
        init_file_path=initial_guess_path,
    )

    # --- Set controls --- #
    u_bounds, u_init, u_scaling = set_u_bounds_and_init(model, window_n_shooting, init_file_path=initial_guess_path)

    # --- Set objective --- #
    objective_fun_dict["target"] = x_init["q"].init[2][-1]
    objective_functions = set_objective_functions(
        objective_fun_dict=objective_fun_dict,
    )

    # --- Set parameters (for minmax cost_function) --- #
    if objective_fun_dict["cost_fun_key"] in [
        ["minimize_peak_force"],
        ["minimize_peak_activation"],
        ["minimize_peak_muscle_stress"],
        ["minimize_peak_fatigue"],
        ["minimize_peak_fatigue_decay"],
    ]:
        parameters, parameters_bounds, parameters_init, parameters_objectives = set_parameters(
            ns=cycle_len, cycle=n_cycles_simultaneous, use_sx=use_sx
        )
    else:
        parameters = ParameterList(use_sx=use_sx)
        parameters_bounds = BoundsList()
        parameters_init = InitialGuessList()
        parameters_objectives = ParameterObjectiveList()

    # --- Set constraints --- #
    constraints = set_constraints(
        model, x_init["q"].init[2][0] - 2 * np.pi, cycle_len, n_cycles_simultaneous, objective_fun_dict["cost_fun_key"]
    )

    # --- Update model for resistive torque --- #
    model = updating_model(model=model, external_force_set=external_force_set, parameters=parameters)

    return MyCyclicMHE(
        bio_model=[model],
        dynamics=dynamics_options,
        cycle_len=cycle_len,
        cycle_duration=cycle_duration,
        n_cycles_simultaneous=n_cycles_simultaneous,
        n_cycles_to_advance=n_cycles_to_advance,
        common_objective_functions=objective_functions,
        constraints=constraints,
        x_bounds=x_bounds,
        x_init=x_init,
        u_bounds=u_bounds,
        u_init=u_init,
        u_scaling=u_scaling,
        parameters=parameters,
        parameter_init=parameters_init,
        parameter_bounds=parameters_bounds,
        parameter_objectives=parameters_objectives,
        n_threads=48,
        use_sx=use_sx,
    )


def set_external_forces(n_shooting, external_force_dict, force_name):
    external_force_set = ExternalForceSetTimeSeries(nb_frames=n_shooting)
    external_force_array = np.array(external_force_dict["torque"])
    reshape_values_array = np.tile(external_force_array[:, np.newaxis], (1, n_shooting))
    external_force_set.add_torque(
        segment=external_force_dict["Segment_application"], values=reshape_values_array, force_name=force_name
    )
    numerical_time_series = {"external_forces": external_force_set.to_numerical_time_series()}
    return numerical_time_series, external_force_set


def _existing_init_file_path(init_file_path):
    """
    Fall back to no initial guess when the given init_file_path does not exist.
    """
    if init_file_path and not Path(init_file_path).exists():
        print(f"[warning] init_file_path '{init_file_path}' not found, running without an initial guess.")
        return None
    return init_file_path


def set_dynamics_options(numerical_time_series, ode_solver):
    dynamics_options = OcpFesMsk.declare_dynamics_options(
        numerical_time_series=numerical_time_series, ode_solver=ode_solver
    )
    return dynamics_options


def set_q_qdot_init(
    n_shooting: int, pedal_config: dict, turn_number: int, ode_solver: OdeSolver, init_file_path: str
) -> InitialGuessList:
    init_file_path = _existing_init_file_path(init_file_path)
    x_init = InitialGuessList()
    if init_file_path:
        with open(init_file_path, "rb") as file:
            data = pickle.load(file)
        q_guess = data["q"]
        qdot_guess = data["qdot"]
        x_init.add("q", q_guess, interpolation=InterpolationType.ALL_POINTS)
        x_init.add("qdot", qdot_guess, interpolation=InterpolationType.ALL_POINTS)
    else:
        # --- Chose the biorbd model to init the inverse kinematics --- #
        biorbd_model_path = str(
            Path(__file__).resolve().parent.parent.parent
            / "msk_models"
            / "Wu"
            / "Modified_Wu_Shoulder_Model_Cycling_for_IK.bioMod"
        )
        n_shooting = (
            n_shooting * (ode_solver.polynomial_degree + 1)
            if isinstance(ode_solver, OdeSolver.COLLOCATION)
            else n_shooting
        )
        # --- Run inverse kinematics --- #
        q_guess, qdot_guess, qddot_guess = inverse_kinematics_cycling(
            biorbd_model_path,
            n_shooting,
            x_center=pedal_config["x_center"],
            y_center=pedal_config["y_center"],
            radius=pedal_config["radius"],
            ik_method="trf",
            cycling_number=turn_number,
        )
        # --- Set q and qdot initial guesses values obtained by inverse kinematics --- #
        if isinstance(ode_solver, OdeSolver.COLLOCATION):
            x_init.add("q", q_guess, interpolation=InterpolationType.ALL_POINTS)
            x_init.add("qdot", qdot_guess, interpolation=InterpolationType.ALL_POINTS)
        elif isinstance(ode_solver, OdeSolver.RK1 | OdeSolver.RK2 | OdeSolver.RK4):
            x_init.add("q", q_guess, interpolation=InterpolationType.EACH_FRAME)
            x_init.add("qdot", qdot_guess, interpolation=InterpolationType.EACH_FRAME)
        else:
            raise RuntimeError("ode_solver must be COLLOCATION or RK4")

    return x_init


def set_x_bounds(
    model, x_init: InitialGuessList, n_shooting: int, ode_solver: OdeSolver, init_file_path: str
) -> tuple[BoundsList, InitialGuessList]:
    init_file_path = _existing_init_file_path(init_file_path)
    # --- Set interpolation type according to ode_solver type --- #
    interpolation_type = InterpolationType.EACH_FRAME
    if isinstance(ode_solver, OdeSolver.COLLOCATION):
        n_shooting = n_shooting * (ode_solver.polynomial_degree + 1)
        interpolation_type = InterpolationType.ALL_POINTS

    # --- Initialize default FES bounds and initial guess --- #
    x_bounds, x_init_fes = OcpFesMsk.set_x_bounds_fes(model)

    # --- Getting initial guesses from initialization file if entered --- #
    if init_file_path:
        with open(init_file_path, "rb") as file:
            data = pickle.load(file)

    # --- Setting FES initial guesses --- #
    for key in x_init_fes.keys():
        initial_guess = data[key] if init_file_path else np.array([[x_init_fes[key].init[0][0]] * (n_shooting + 1)])
        x_init.add(key=key, initial_guess=initial_guess, phase=0, interpolation=interpolation_type)

    # --- Setting q bounds --- #
    q_x_bounds = model.bounds_from_ranges("q")

    # --- First: enter general bound values in radiant --- #
    arm_q = [0, 1.5]  # Arm min_max q bound in radiant
    forearm_q = [0.5, 2.5]  # Forearm min_max q bound in radiant
    slack = 0.05  # Wheel rotation slack
    wheel_q = [x_init["q"].init[2][-1] - slack, x_init["q"].init[2][0] + slack]  # Wheel min_max q bound in radiant

    # --- Second: set general bound values in radiant, CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT mandatory for qdot --- #
    q_x_bounds.min[0] = [arm_q[0]] * 3
    q_x_bounds.max[0] = [arm_q[1]] * 3
    q_x_bounds.min[1] = [forearm_q[0]] * 3
    q_x_bounds.max[1] = [forearm_q[1]] * 3
    q_x_bounds.min[2] = [x_init["q"].init[2][0], wheel_q[0] - 2, x_init["q"].init[2][-1] - slack]
    q_x_bounds.max[2] = [x_init["q"].init[2][0], wheel_q[1] + 3, x_init["q"].init[2][-1] + slack]

    x_bounds.add(
        key="q", bounds=q_x_bounds, phase=0, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT
    )

    # --- Setting qdot bounds --- #
    qdot_x_bounds = model.bounds_from_ranges("qdot")

    # --- First: enter general bound values in radiant --- #
    arm_qdot = [-10, 10]  # Arm min_max qdot bound in radiant
    forearm_qdot = [-14, 10]  # Forearm min_max qdot bound in radiant
    wheel_qdot = [-2 * np.pi - 3, -2 * np.pi + 3]  # Wheel min_max qdot bound in radiant

    # --- Second: set general bound values in radiant, CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT mandatory for qdot --- #
    qdot_x_bounds.min[0] = [arm_qdot[0]] * 3
    qdot_x_bounds.max[0] = [arm_qdot[1]] * 3
    qdot_x_bounds.min[1] = [forearm_qdot[0]] * 3
    qdot_x_bounds.max[1] = [forearm_qdot[1]] * 3
    qdot_x_bounds.min[2] = [wheel_qdot[0]] * 3
    qdot_x_bounds.max[2] = [wheel_qdot[1]] * 3

    x_bounds.add(
        key="qdot",
        bounds=qdot_x_bounds,
        phase=0,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    return x_bounds, x_init


def set_u_bounds_and_init(bio_model, n_shooting, init_file_path):
    init_file_path = _existing_init_file_path(init_file_path)
    u_bounds, u_init = OcpFesMsk.set_u_bounds_fes(bio_model)
    u_init = InitialGuessList()  # Controls initial guess
    models = bio_model.muscles_dynamics_model
    if init_file_path:
        with open(init_file_path, "rb") as file:
            data = pickle.load(file)

    for model in models:
        key = "last_pulse_width_" + str(model.muscle_name)
        if init_file_path:
            initial_guess = data[key]
        else:
            initial_guess = np.array([[model.pd0] * n_shooting])
        u_init.add(
            key=key,
            initial_guess=initial_guess,
            phase=0,
            interpolation=InterpolationType.EACH_FRAME,
        )
    u_scaling = VariableScalingList()
    for model in bio_model.muscles_dynamics_model:
        key = "last_pulse_width_" + str(model.muscle_name)
        u_scaling.add(key=key, scaling=[1 / 400])

    return (
        u_bounds,
        u_init,
        u_scaling,
    )


def set_constraints(bio_model, one_cycle_bound, cycle_len, n_simultaneous, objective_function_key=None):
    constraints = ConstraintList()
    # --- Constraining wheel center position to a fix position --- #
    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        node=Node.START,
        marker_index=bio_model.marker_index("wheel_center"),
        axes=[Axis.X, Axis.Y],
    )
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        first_marker="wheel_center",
        second_marker="global_wheel_center",
        node=Node.START,
        axes=[Axis.X, Axis.Y],
    )

    angle_slack = 0.174533 / 4  # 10 degrees in radiant
    for i in range(n_simultaneous - 1):
        constraints.add(
            ConstraintFcn.BOUND_STATE,
            key="q",
            node=cycle_len * (i + 1),
            index=2,
            min_bound=np.array([one_cycle_bound - (2 * np.pi * i) - angle_slack]),
            max_bound=np.array([one_cycle_bound - (2 * np.pi * i) + angle_slack]),
        )

    if objective_function_key in [
        ["minimize_peak_force"],
        ["minimize_peak_activation"],
        ["minimize_peak_muscle_stress"],
        ["minimize_peak_fatigue"],
        ["minimize_peak_fatigue_decay"],
    ]:
        for i in range(4):
            constraints.add(
                CustomCostFunctions.constraints_minmax,
                obj_fun_key=objective_function_key,
                node=Node.ALL,
                param_index=i,
                min_bound=0,
                max_bound=np.inf,
            )

    return constraints


def set_objective_functions(objective_fun_dict, recalculate=False):
    objective_functions = ObjectiveList()
    custom_objective_functions = CustomCostFunctions().dict_functions
    weights = objective_fun_dict["cost_fun_weight"]
    keys = objective_fun_dict["cost_fun_key"]

    # --- Set cost function for initial_guess ocp --- #
    if keys[0] == "initial_guess":
        target = objective_fun_dict["target"]
        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_STATE,
            key="q",
            index=2,
            node=Node.END,
            weight=1e6,
            target=target,
            quadratic=True,
        )

    # --- Set main cost function --- #
    else:
        for i in range(len(keys)):
            if (
                keys[i]
                in [
                    "minimize_peak_force",
                    "minimize_peak_activation",
                    "minimize_peak_muscle_stress",
                    "minimize_peak_fatigue",
                    "minimize_peak_fatigue_decay",
                ]
                and recalculate is False
            ):
                objective_functions.add(
                    custom_objective_functions["minimize_peak"]["function"],
                    custom_type=ObjectiveFcn.Lagrange,
                    node=Node.ALL,
                    weight=weights,
                    quadratic=False,
                )
            else:
                objective_functions.add(
                    custom_objective_functions[keys[i]]["function"],
                    custom_type=ObjectiveFcn.Lagrange,
                    node=Node.ALL,
                    weight=weights,
                    quadratic=False,
                )

    return objective_functions


def set_parameters(ns: int, cycle: int, use_sx: bool):
    parameters = ParameterList(use_sx=use_sx)
    parameter_bounds = BoundsList()
    parameter_init = InitialGuessList()
    parameter_objectives = ParameterObjectiveList()
    ns = ns * cycle + 1

    parameters.add("minmax_param", None, size=ns, scaling=VariableScaling("minmax_param", [1] * ns))
    parameter_init["minmax_param"] = [0] * ns
    parameter_bounds.add(
        "minmax_param", min_bound=[0] * ns, max_bound=[1000000] * ns, interpolation=InterpolationType.CONSTANT
    )

    return parameters, parameter_bounds, parameter_init, parameter_objectives


def updating_model(model: FesMskModel, external_force_set, parameters=None) -> FesMskModel:
    model = FesMskModel(
        name=model.name,
        biorbd_path=model.biorbd_path,
        muscles_model=model.muscles_dynamics_model,
        stim_time=model.muscles_dynamics_model[0].stim_time,
        previous_stim=model.muscles_dynamics_model[0].previous_stim,
        activate_force_length_relationship=model.activate_force_length_relationship,
        activate_force_velocity_relationship=model.activate_force_velocity_relationship,
        activate_residual_torque=model.activate_residual_torque,
        parameters=parameters,
        external_force_set=external_force_set,
        with_contact=True,
    )
    return model


# --------------------------#
#   Simulation functions   #
# --------------------------#
def set_fes_model(model_path, stim_time):
    # Set FES model (set to Ding et al. 2007 + fatigue, for now)
    dummy_biomodel = BiorbdModel(model_path)
    muscle_name_list = dummy_biomodel.muscle_names
    muscles_model = [
        DingModelPulseWidthFrequencyWithFatigue(muscle_name=muscle, sum_stim_truncation=6)
        for muscle in muscle_name_list
    ]

    # --- Muscle parameter scaling --- #
    # Values from Ding et al. 2007 + Ding et al. 2003 for fatigue, based on the rectus femoris (RF) muscle
    # Note: these values were scaled on PCSA and fiber proportion to match biceps, triceps, and deltoids muscles

    # ------------------------------------------------------ #
    # Muscle         |  PCSA (cm²) | Fiber proportion (I/II) |
    # ------------------------------------------------------ #
    # Rectus femoris |    10.88    |          35/65          |
    # Biceps         |    7.33     |          38/62          |
    # Triceps        |    10.87    |          44/56          |
    # Delt_ant       |    2.54     |          47/53          |
    # Delt_post      |    2.73     |          56/44          |
    # ------------------------------------------------------ #
    # Table 1: PCSA and fiber type proportion from Peterson et al. (2011) and Kumazaki et al. (2022)

    # The scaling was done as follows (a_scale_RF=4920; alpha_a_RF=-4.0*10e-2;: tau_fat_RF=127):
    # a_scale = a_scale_RF * PCSA_muscle / PCSA_RF
    # alpha_a = (alpha_a_RF * Fiber_prop_II_muscle / Fiber_prop_II_RF) * (a_scale_RF / a_scale_muscle)
    # tau_fat = (tau_fat_RF * Fiber_prop_II_muscle / Fiber_prop_II_RF) * (a_scale_RF / a_scale_muscle)

    # Maximal force Fmax was obtained during a simulation of a 30 Hz stimulation train of one second at 600us with
    # scaled muscle properties

    parameter_dict = {
        "Biceps": {"Fmax": 149, "a_scale": 3314.7, "alpha_a": -5.6 * 10e-2, "tau_fat": 179.6, "pcsa": 7.33},
        "Triceps": {"Fmax": 262, "a_scale": 4915.5, "alpha_a": -3.4 * 10e-2, "tau_fat": 109.1, "pcsa": 10.87},
        "Delt_ant": {"Fmax": 48, "a_scale": 1148.6, "alpha_a": -1.4 * 10e-1, "tau_fat": 445.5, "pcsa": 2.54},
        "Delt_post": {"Fmax": 51, "a_scale": 1234.5, "alpha_a": -1.1 * 10e-1, "tau_fat": 342.7, "pcsa": 2.73},
    }

    for model in muscles_model:
        muscle_name = model.muscle_name
        model.a_scale = parameter_dict[muscle_name]["a_scale"]
        model.a_rest = parameter_dict[muscle_name]["a_scale"]
        model.fmax = parameter_dict[muscle_name]["Fmax"]
        model.alpha_a = parameter_dict[muscle_name]["alpha_a"]
        model.tau_fat = parameter_dict[muscle_name]["tau_fat"]
        model.pcsa = parameter_dict[muscle_name]["pcsa"]

    # Create MSK FES-driven model
    fes_model = FesMskModel(
        name=None,
        biorbd_path=model_path,
        muscles_model=muscles_model,
        stim_time=stim_time,
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=True,
        activate_residual_torque=False,
        external_force_set=None,  # External forces will be added later (resistive_torque)
        with_contact=True,
    )
    return fes_model


def create_simulation_list(
    n_cycles_simultaneous: list[int],
    stimulation: list[int],
    cost_fun_dict: dict,
    ode_solver: OdeSolver(),
) -> list[dict]:

    def make_file_paths(
        num_cycles: int,
        index: list,
        solver_type: OdeSolver,
    ) -> tuple[str, str]:

        parts = ["cost_fun_index"]
        for i in range(len(index)):
            parts.append(f"{index[i]}")
        weight_suffix = "_".join(parts)

        if isinstance(solver_type, OdeSolver.COLLOCATION):
            solver_suffix = f"collocation_{solver_type.polynomial_degree}_{solver_type.method}"
        elif isinstance(solver_type, OdeSolver.RK4):
            solver_suffix = f"rk4_{solver_type.n_integration_steps}"
        else:
            raise RuntimeError("ode_solver must be COLLOCATION or RK4")

        full_suffix = f"{weight_suffix}_{solver_suffix}_with_init"
        pkl = str(Path("result/test") / f"{num_cycles}_cycle" / f"{num_cycles}_min_{full_suffix}.pkl")
        init = str(Path("result/initial_guess") / f"{num_cycles}_initial_guess_{solver_suffix}.pkl")
        init = init if os.path.exists(init) else None
        if init is None:
            print("No initial guess file for n_cycle: " + str(num_cycles) + " and solver: " + str(solver_suffix))
        return pkl, init

    sims = []
    custom_cost_function_dict = CustomCostFunctions().dict_functions
    for (n_cycles, stim), cost_fun_key in product(
        zip(n_cycles_simultaneous, stimulation),
        cost_fun_dict["optimized_function"],
    ):
        index = [custom_cost_function_dict[key]["index"] for key in cost_fun_key]
        pkl_path, init_path = make_file_paths(n_cycles, index, ode_solver)
        sims.append(
            {
                "n_cycles_simultaneous": n_cycles,
                "stimulation": stim,
                "cost_fun_key": cost_fun_key,
                "pickle_file_path": pkl_path,
                "init_guess_file_path": init_path,
            }
        )
    return sims


def save_sol_in_pkl(sol, simulation_conditions, mhe, is_initial_guess=False, torque=None):
    solution = sol[0] if not is_initial_guess else sol[1][0]
    time = solution.stepwise_time(to_merge=[SolutionMerge.NODES]).T[0]
    states = solution.stepwise_states(to_merge=[SolutionMerge.NODES])
    controls = solution.stepwise_controls(to_merge=[SolutionMerge.NODES])
    stim_time = solution.ocp.nlp[0].model.muscles_dynamics_model[0].stim_time
    solving_time_per_ocp = [sol[1][i].solver_time_to_optimize for i in range(len(sol[1]))]
    objective_values_per_ocp = [float(sol[1][i].cost) for i in range(len(sol[1]))]
    objective_values_per_kept_cycle = [
        float(sol[2][i].cost) for i in range(len(sol[2]) - (simulation_conditions["n_cycles_simultaneous"] - 1))
    ]
    iter_per_ocp = [sol[1][i].iterations for i in range(len(sol[1]))]
    average_solving_time_per_iter_list = [solving_time_per_ocp[i] / (iter_per_ocp[i] + 1) for i in range(len(sol[1]))]
    total_average_solving_time_per_iter = average(average_solving_time_per_iter_list)
    number_of_turns_before_failing = len(sol[2])
    convergence_status = [sol[1][i].status for i in range(len(sol[1]))]
    cost_function = np.array(simulation_conditions["cost_fun_key"], dtype=np.str_)

    # --- Convert all data into lists for compatibility across Python versions --- #
    time = time.tolist()
    states = {key: value.tolist() for key, value in states.items()}
    controls = {key: value.tolist() for key, value in controls.items()}

    dictionary = {
        "time": time,
        "stim_time": stim_time,
        "solving_time_per_ocp": solving_time_per_ocp,
        "objective_values_per_ocp": objective_values_per_ocp,
        "objective_values_per_kept_cycle": objective_values_per_kept_cycle,
        "number_of_turns_before_failing": number_of_turns_before_failing,
        "convergence_status": convergence_status,
        "iter_per_ocp": iter_per_ocp,
        "average_solving_time_per_iter_list": average_solving_time_per_iter_list,
        "total_average_solving_time_per_iter": total_average_solving_time_per_iter,
        "total_n_shooting": solution.ocp.n_shooting,
        "n_shooting_per_cycle": int(solution.ocp.n_shooting / len(sol[1])),
        "polynomial_order": solution.ocp.nlp[0].dynamics_type.ode_solver.polynomial_degree,
        "applied_torque": torque,
        "cost_function": cost_function,
    }

    recalculate_objective = False
    if recalculate_objective:
        recalculate_objective_dict = recalculate_objective_fun(sol[1], mhe, sim_cond=simulation_conditions)
        similar_cost_values = [
            True if objective_values_per_kept_cycle == recalculate_objective_dict[key] else False
            for key in recalculate_objective_dict
        ]
        if any(similar_cost_values):
            dictionary["similar_cost_values"] = True
            print("Recalculated cost function matches the original one.")
        else:
            dictionary["similar_cost_values"] = False
            print("Recalculated cost function does not match the original one.")

        for key in recalculate_objective_dict.keys():
            dictionary[key] = recalculate_objective_dict[key]

    for key in states.keys():
        dictionary[key] = states[key]
    for key in controls.keys():
        dictionary[key] = controls[key]

    pickle_file_name = simulation_conditions["pickle_file_path"]
    # with open(pickle_file_name, "wb") as file:
    #     pickle.dump(dictionary, file)

    np.savez_compressed(str(pickle_file_name)[:-4] + ".npz", **dictionary)
    print(simulation_conditions["pickle_file_path"])


def recalculate_objective_fun(cycle_solutions: list[Solution], mhe, sim_cond) -> dict:
    import time

    recalculated_cost_functions = {}
    custom_cost_functions = CustomCostFunctions().dict_functions  # will recalculate all cost functions (long process!)
    recalculated_cost_functions_keys = list(custom_cost_functions.keys())[:-1]

    for key in recalculated_cost_functions_keys:
        initial_time = time.time()
        obj_fun_dict = {
            "cost_fun_key": [key],
            "cost_fun_weight": sim_cond["cost_fun_weight"],
        }
        objective = set_objective_functions(obj_fun_dict, recalculate=True)
        mhe.common_objective_functions = objective
        cost_function_values = []
        for i in range(len(cycle_solutions)):
            _states, _controls, _parameters = mhe.export_cycles(cycle_solutions[i])
            dt = float(cycle_solutions[i].t_span()[0][-1])
            solution = mhe._initialize_one_cycle(dt, _states, _controls, _parameters)
            cost = float(solution.cost)
            cost_function_values.append(cost)

        recalculated_cost_functions[key + "_cost"] = cost_function_values
        print(f"Recalculating {key} took {time.time() - initial_time:.2f} seconds")

    return recalculated_cost_functions


def run_initial_guess(mhe_info, cycling_info, model_path, stimulation, n_cycles_simultaneous, save_sol=True):
    init_guess_mhe_info = {
        "cycle_duration": mhe_info["cycle_duration"],
        "n_cycles_to_advance": mhe_info["n_cycles_to_advance"],
        "n_cycles": 1,
        "ode_solver": mhe_info["ode_solver"],
        "use_sx": mhe_info["use_sx"],
    }

    ode_solver = mhe_info["ode_solver"]
    rk_name = {
        OdeSolver.RK1: "rk1",
        OdeSolver.RK2: "rk2",
        OdeSolver.RK4: "rk4",
    }.get(type(ode_solver))
    if isinstance(ode_solver, OdeSolver.COLLOCATION):
        solver_suffix = f"collocation_{ode_solver.polynomial_degree}_{ode_solver.method}"
    elif rk_name is not None:
        solver_suffix = f"{rk_name}_{ode_solver.n_integration_steps}"
    else:
        raise RuntimeError("ode_solver must either be COLLOCATION or RK")

    for i in range(len(n_cycles_simultaneous)):
        simulation_conditions = {
            "n_cycles_simultaneous": n_cycles_simultaneous[i],
            "stimulation": stimulation[i],
            "cost_fun_key": ["initial_guess"],
            "cost_fun_weight": [10000],
            "pickle_file_path": Path("result")
            / "initial_guess"
            / f"{n_cycles_simultaneous[i]}_initial_guess_{solver_suffix}.pkl",
            "init_guess_file_path": None,
        }

        run_optim(
            mhe_info=init_guess_mhe_info,
            cycling_info=cycling_info,
            simulation_conditions=simulation_conditions,
            model_path=model_path,
            save_sol=save_sol,
            is_initial_guess=True,
        )


def run_optim(mhe_info, cycling_info, simulation_conditions, model_path, save_sol, is_initial_guess=False):
    # --- Set FES model --- #
    stim_time = list(
        np.linspace(
            0,
            mhe_info["cycle_duration"] * simulation_conditions["n_cycles_simultaneous"],
            simulation_conditions["stimulation"],
            endpoint=False,
        )
    )
    model = set_fes_model(model_path, stim_time)

    mhe_info["cycle_len"] = int(len(stim_time) / simulation_conditions["n_cycles_simultaneous"])
    mhe_info["n_cycles_simultaneous"] = simulation_conditions["n_cycles_simultaneous"]
    cycling_info["turn_number"] = simulation_conditions["n_cycles_simultaneous"]  # One turn per cycle

    mhe = prepare_mhe(
        model=model,
        mhe_info=mhe_info,
        cycling_info=cycling_info,
        simulation_conditions=simulation_conditions,
    )
    mhe.n_cycles_simultaneous = simulation_conditions["n_cycles_simultaneous"]

    def update_functions(_mhe: MultiCyclicNonlinearModelPredictiveControl, cycle_idx: int, _sol: Solution):
        if _sol:
            print("Optimized window n°" + str(cycle_idx))
            result_sol = _sol.decision_states(to_merge=SolutionMerge.NODES)
            muscles = _sol.ocp.nlp[0].model.muscle_names
            a_rest_list = [_sol.ocp.nlp[0].model.muscles_dynamics_model[i].a_rest for i in range(len(muscles))]
            a_t_list = [result_sol["A_" + muscles[i]][0][-1] for i in range(len(muscles))]
            fatigue_percentage = [
                round(((a_rest_list[i] - a_t_list[i]) / (a_rest_list[i])) * 100, 2) for i in range(len(muscles))
            ]
            for i in range(len(muscles)):
                print(muscles[i] + " fatigue = " + str(fatigue_percentage[i]) + "%")
        return cycle_idx < mhe_info["n_cycles"]  # True if there are still some cycle to perform

    # Add the penalty cost function plot
    mhe.add_plot_penalty(CostType.ALL)

    # Set solver for the optimal control problem
    solver = Solver.IPOPT(show_online_optim=False, _max_iter=2000, show_options=dict(show_bounds=True))
    linear_solver = "ma57" if platform == "linux" else "mumps"
    solver.set_linear_solver(linear_solver)

    # Solve the optimal control problem
    sol = mhe.solve_fes_mhe(
        update_functions,
        solver=solver,
        total_cycles=mhe_info["n_cycles"],
        external_force=cycling_info["resistive_torque"],
        cycle_solutions=MultiCyclicCycleSolutions.ALL_CYCLES,
        get_all_iterations=True,
        cyclic_options={"states": {}},
        max_consecutive_failing=1,
    )

    result_show = False
    if result_show:
        sol[0].animate(viewer="pyorerun")
        sol[0].graphs(show_bounds=True)

    # Saving the data in a pickle file
    if save_sol:
        save_sol_in_pkl(
            sol,
            simulation_conditions,
            mhe=mhe,
            is_initial_guess=is_initial_guess,
            torque=cycling_info["resistive_torque"]["torque"][-1],
        )


def main(
    stimulation_frequency, n_total_cycle, n_cycles_simultaneous, resistive_torque, cost_fun_dict, init_guess, save
):
    # --- Simulation configuration --- #
    save_sol = save
    get_initial_guess = init_guess

    # --- Model choice --- #
    model_path = str(
        Path(__file__).resolve().parent.parent.parent
        / "msk_models"
        / "Wu"
        / "Modified_Wu_Shoulder_Model_Cycling.bioMod"
    )

    # --- MHE parameters --- #
    ode_solver = OdeSolver.COLLOCATION(polynomial_degree=3, method="radau")
    mhe_info = {
        "cycle_duration": 1,
        "n_cycles_to_advance": 1,
        "n_cycles": n_total_cycle,
        "ode_solver": ode_solver,
        "use_sx": False,
    }

    # --- Bike parameters --- #
    cycling_info = {
        "pedal_config": {"x_center": 0.35, "y_center": 0.0, "radius": 0.1},
        "resistive_torque": {"Segment_application": "wheel", "torque": np.array([0, 0, resistive_torque])},
    }

    # --- Build simulation list --- #
    stimulation = [stimulation_frequency * i for i in n_cycles_simultaneous]

    # --- Build the simulation conditions list --- #
    simulation_conditions_list = create_simulation_list(
        n_cycles_simultaneous=n_cycles_simultaneous,
        stimulation=stimulation,
        cost_fun_dict=cost_fun_dict,
        ode_solver=mhe_info["ode_solver"],
    )

    # --- Run the initial guess optimization --- #
    if get_initial_guess:
        run_initial_guess(
            mhe_info=mhe_info,
            cycling_info=cycling_info,
            model_path=model_path,
            stimulation=stimulation,
            n_cycles_simultaneous=n_cycles_simultaneous,
            save_sol=save_sol,
        )

    # --- Run the optimization --- #
    for i in range(len(simulation_conditions_list)):
        run_optim(
            mhe_info=mhe_info,
            cycling_info=cycling_info,
            simulation_conditions=simulation_conditions_list[i],
            model_path=model_path,
            save_sol=save_sol,
        )


if __name__ == "__main__":
    main(
        stimulation_frequency=30,
        n_total_cycle=10000,
        n_cycles_simultaneous=[2],  # [2, 3, 4, 5],
        resistive_torque=-0.20,  # (N.m)
        cost_fun_dict={
            "optimized_function": [
                # --- UNWEIGHTED --- #
                # --- Pulse width --- #
                # ["minimize_average_activation"],
                ["minimize_root_mean_square_activation"],
                # ["minimize_cubic_average_activation"],
                # ["minimize_peak_activation"],
                # --- Force --- #
                # ["minimize_average_force"],
                # ["minimize_root_mean_square_force"],
                # ["minimize_cubic_average_force"],
                # ["minimize_peak_force"],
                # --- Stress --- #
                # ["minimize_average_muscle_stress"],
                # ["minimize_root_mean_square_muscle_stress"],
                # ["minimize_cubic_average_muscle_stress"],
                # ["minimize_peak_muscle_stress"],
                # --- Fatigue --- #
                # ["minimize_average_fatigue"],
                # ["minimize_root_mean_square_fatigue"],
                # ["minimize_cubic_average_fatigue"],
                # ["minimize_peak_fatigue"],
                # --- Power --- #
                # ["minimize_root_mean_square_muscle_power"],
                # --- BAYESIAN --- # (Only RMS)
                # --- Pulse width --- #
                # ["minimize_root_mean_square_activation_bayesian"],
                # --- Force --- #
                # ["minimize_root_mean_square_force_bayesian"],
                # --- Stress --- #
                # ["minimize_root_mean_square_muscle_stress_bayesian"],
                # --- Fatigue --- #
                # ["minimize_root_mean_square_fatigue_bayesian"],
                # --- WEIGHTED --- # (Only RMS)
                # --- Pulse width --- #
                # ["minimize_root_mean_square_activation_weight"],
                # --- Force --- #
                # ["minimize_root_mean_square_force_weight"],
                # --- Stress --- #
                # ["minimize_root_mean_square_muscle_stress_weight"],
                # --- Fatigue --- #
                # ["minimize_root_mean_square_fatigue_weight"],
            ]
        },
        init_guess=False,  # Compute initial guess
        save=True,
    )
