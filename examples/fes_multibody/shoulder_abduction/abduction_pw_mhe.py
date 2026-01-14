"""
This example will perform an optimal control program moving time horizon for an abduction/adduction motion driven by FES.
"""

import pickle
from sys import platform
from itertools import product
from pathlib import Path

import numpy as np
from numpy.ma.extras import average

from bioptim import (
    BiorbdModel,
    BoundsList,
    ConstraintList,
    ConstraintFcn,
    ContactType,
    CostType,
    DynamicsList,
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
)
from cocofest import (
    DingModelPulseWidthFrequencyWithFatigue,
    FesMskModel,
    OcpFesMsk,
    FesNmpcMsk,
)
from examples.fes_multibody.cycling.cost_functions import CustomCostFunctions


class MyCyclicNMPC(FesNmpcMsk):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nodes_per_cycle = self.cycle_len * (
            self.nlp[0].dynamics_type.ode_solver.polynomial_degree + 1
            if isinstance(self.nlp[0].dynamics_type.ode_solver, OdeSolver.COLLOCATION)
            else 1
        )
        self.debugg_bounds = False
        self.previous_bounds = None

    def advance_window_bounds_states(self, sol, n_cycles_simultaneous=None, **extra):
        # --- Get states results --- #
        states = sol.decision_states(to_merge=SolutionMerge.NODES)
        states_keys = states.keys()
        # --- States are bounded to match the last node of the cycle to ensure continuity between window --- #
        for key in states_keys:
            for i in range(states[key].shape[0]):
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
                    self.set_init_cyclical(states, key, i)
                elif key in continuous_keys:
                    self.set_init_continuous(states, key, i)
        self._correct_init_guess_to_fit_bounds(
            corrected_input="states"
        )  # This function is called to move init guess within the bounds if not in bounds

        return True

    def advance_window_initial_guess_controls(self, sol, n_cycles_simultaneous=None):
        # --- Get control results --- #
        controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

        # --- Set initial guess for controls --- #
        for key in controls.keys():
            self.set_init_cyclical(controls, key, 0, False)
        self._correct_init_guess_to_fit_bounds(
            corrected_input="controls"
        )  # This function is called to move init guess within the bounds if not in bounds

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


# --------------------#
#    OCP functions    #
# --------------------#
def prepare_nmpc(
    model: BiorbdModel | FesMskModel,
    mhe_info: dict,
    abduction_info: dict,
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
    window_n_shooting = cycle_len * n_cycles_simultaneous
    window_cycle_duration = cycle_duration * n_cycles_simultaneous

    # --- Abduction info --- #
    abd_range_config = abduction_info["range_config"]
    external_force = abduction_info["resistive_torque"]

    # --- Cost function info --- #
    objective_fun_dict = {"cost_fun_key": simulation_conditions["cost_fun_key"],
                          "cost_fun_weight": simulation_conditions["cost_fun_weight"],
                          "individual_quadratic": True}

    # --- Set dynamics --- #
    # --- External force numerical time series --- #
    numerical_time_series, external_force_set = set_external_forces(
        n_shooting=window_n_shooting, external_force_dict=external_force, force_name="external_torque"
    ) # No external torque applied, except to provide FES supplementary assistance
    # --- Stimulation instant numerical time series --- #
    numerical_data_time_series, stim_idx_at_node_list = model.muscles_dynamics_model[0].get_numerical_data_time_series(
        window_n_shooting, window_cycle_duration
    )
    numerical_time_series.update(numerical_data_time_series)
    # --- Dynamics --- #
    dynamics = set_dynamics(model=model, numerical_time_series=numerical_time_series, ode_solver=ode_solver)

    # --- Set states --- #
    x_bounds, x_init = set_x_bounds(
        model=model,
        n_shooting=window_n_shooting,
        abduction_range=abd_range_config,
        ode_solver=ode_solver,
    )

    # --- Set controls --- #
    u_bounds, u_init, u_scaling = set_u_bounds_and_init(model, window_n_shooting)

    # --- Set objective --- #
    objective_functions = set_objective_functions(
        objective_fun_dict=objective_fun_dict,
    )

    # --- Set parameters (for minmax cost_function) --- #
    parameters = ParameterList(use_sx=use_sx)
    parameters_bounds = BoundsList()
    parameters_init = InitialGuessList()
    parameters_objectives = ParameterObjectiveList()

     # --- Set constraints --- #
    constraints = set_constraints(abd_range_config, cycle_len, n_cycles_simultaneous)

    # --- Update model for resistive torque --- #
    model = updating_model(model=model, external_force_set=external_force_set, parameters=parameters)

    return MyCyclicNMPC(
        bio_model=[model],
        dynamics=dynamics,
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
    )  # warning forloop different force name
    numerical_time_series = {"external_forces": external_force_set.to_numerical_time_series()}
    return numerical_time_series, external_force_set


def set_dynamics(model, numerical_time_series, ode_solver):
    dynamics = DynamicsList()
    dynamics.add(
        dynamics_type=model.declare_model_variables,
        dynamic_function=model.muscle_dynamic,
        expand_dynamics=True,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        numerical_data_timeseries=numerical_time_series,
        contact_type=[ContactType.RIGID_EXPLICIT],
        phase=0,
        ode_solver=ode_solver,
    )
    return dynamics


def set_x_bounds(model, n_shooting: int, abduction_range: dict, ode_solver: OdeSolver) -> tuple[BoundsList, InitialGuessList]:
    # --- Set interpolation type according to ode_solver type --- #
    interpolation_type = InterpolationType.EACH_FRAME
    if isinstance(ode_solver, OdeSolver.COLLOCATION):
        n_shooting = n_shooting * (ode_solver.polynomial_degree + 1)
        interpolation_type = InterpolationType.ALL_POINTS

    # --- Initialize default FES bounds and initial guess --- #
    x_bounds, x_init_fes = OcpFesMsk.set_x_bounds_fes(model)

    # --- Appending FES initial guesses to main list --- #
    x_init = InitialGuessList()
    for key in x_init_fes.keys():
        initial_guess = np.array([[x_init_fes[key].init[0][0]] * (n_shooting + 1)])
        x_init.add(key=key, initial_guess=initial_guess, phase=0, interpolation=interpolation_type)

    x_init.add(key="q", initial_guess=np.array([[0.523599] * (n_shooting + 1)]), phase=0, interpolation=interpolation_type)
    x_init.add(key="qdot", initial_guess=np.array([[0.0] * (n_shooting + 1)]), phase=0, interpolation=interpolation_type)

    # --- Setting q bounds --- #
    q_x_bounds = model.bounds_from_ranges("q")

    # --- First: enter general bound values in radiant --- #
    humerus_q_min = [abduction_range["min"], abduction_range["min"], abduction_range["min"]]  # Arm min_max q bound in radiant
    humerus_q_max = [abduction_range["min"], abduction_range["max"], abduction_range["min"]]  # Arm min_max q bound in radiant
    slack = 0.0872665  # 5° slack
    humerus_q_min[1] -= slack
    humerus_q_min[2] -= slack
    humerus_q_max[1] += slack
    humerus_q_max[2] += slack

    # --- Second: set general bound values in radiant, CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT mandatory for qdot --- #
    q_x_bounds.min[0] = np.array(humerus_q_min)
    q_x_bounds.max[0] = np.array(humerus_q_max)

    x_bounds.add(
        key="q", bounds=q_x_bounds, phase=0, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT
    )

    # --- Setting qdot bounds --- #
    qdot_x_bounds = model.bounds_from_ranges("qdot")
    x_bounds.add(
        key="qdot",
        bounds=qdot_x_bounds,
        phase=0,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    return x_bounds, x_init


def set_u_bounds_and_init(bio_model, n_shooting):
    # --- Initialize bounds and initial guess --- #
    u_init = InitialGuessList()  # Controls initial guess
    u_bounds = BoundsList()  # Controls bounds
    u_scaling = VariableScalingList()
    models = bio_model.muscles_dynamics_model

    for model in models:
        key = "last_pulse_width_" + str(model.muscle_name)

        # --- Set pulse width bounds for abduction adduction --- #
        min_pw_val = model.pd0
        max_pw_val = 0.0006
        u_bounds.add(
            key=key,
            min_bound=[min_pw_val],
            max_bound=[max_pw_val],
            interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        )

        # --- Set pulse width initial guess --- #
        initial_guess = np.array([[model.pd0] * n_shooting])
        u_init.add(
            key=key,
            initial_guess=initial_guess,
            phase=0,
            interpolation=InterpolationType.EACH_FRAME,
        )

    # --- Initialize tau bounds and initial guess --- #
    min_tau_val = -10
    max_tau_val = 10
    u_bounds.add(key="tau",
                 min_bound=[min_tau_val],
                 max_bound=[max_tau_val],
                 interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT, )
    u_init.add(key="tau",
               initial_guess=np.array([[0] * n_shooting]),
               phase=0,
               interpolation=InterpolationType.EACH_FRAME, )

    # --- Set control scaling --- #
    for model in bio_model.muscles_dynamics_model:
        key = "last_pulse_width_" + str(model.muscle_name)
        u_scaling.add(key=key, scaling=[1 / 400])
    u_scaling.add(key="tau", scaling=[1])

    return (
        u_bounds,
        u_init,
        u_scaling,
    )


def set_constraints(abduction_range, cycle_len, n_simultaneous):
    constraints = ConstraintList()
    angle_slack = np.deg2rad(5)
    abduction_height = abduction_range["max"]

    for i in range(n_simultaneous):
        constraints.add(
            ConstraintFcn.BOUND_STATE,
            key="q",
            node=int((cycle_len/2) + cycle_len * i),
            index=0,
            min_bound=np.array([abduction_height - angle_slack]),
            max_bound=np.array([abduction_height + angle_slack]),
        )

        for j in range(int(cycle_len/2) + 1):
            constraints.add(
                ConstraintFcn.BOUND_CONTROL,
                key="tau",
                node=int(j + i * cycle_len),
                min_bound=np.array([0]),
                max_bound=np.array([0]),
            )

    return constraints


def set_objective_functions(objective_fun_dict):
    objective_functions = ObjectiveList()
    custom_objective_functions = CustomCostFunctions().dict_functions
    weights = objective_fun_dict["cost_fun_weight"]
    keys = objective_fun_dict["cost_fun_key"]

    # --- Set main cost function --- #
    for i in range(len(keys)):
            objective_functions.add(
                custom_objective_functions[keys[i]]["function"],
                custom_type=ObjectiveFcn.Lagrange,
                node=Node.ALL,
                weight=weights[i],
                quadratic=False,
            )

    # --- Set supplementary cost function for adduction tau --- #
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
        key="tau",
        weight=1,
        quadratic=True,
    )

    return objective_functions


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
    # Values from Ding et al. 2007 + Ding et al. 2003 for fatigue, based on the rectus femoris muscle
    # Note: these values were scaled on PCSA and fiber proportion to match deltoids muscles

    # ------------------------------------------------------ #
    # Muscle         |  PCSA (cm²) | Fiber proportion (I/II) |
    # ------------------------------------------------------ #
    # Rectus femoris |    10.88    |          35/65          |
    # Delt_ant       |    2.54     |          47/53          |
    # Delt_cent      |    11.18    |          47/53          |
    # Delt_post      |    2.73     |          56/44          |
    # ------------------------------------------------------ #

    # The scaling was done as follows (a_scale_RF=4920; alpha_a_RF=-4.0*10e-2;: tau_fat_RF=127):
    # a_scale = a_scale_RF * PCSA_muscle / PCSA_RF
    # alpha_a = (alpha_a_RF * Fiber_prop_II_muscle / Fiber_prop_II_RF) * (a_scale_RF / a_scale_muscle)
    # tau_fat = (tau_fat_RF * Fiber_prop_II_muscle / Fiber_prop_II_RF) * (a_scale_RF / a_scale_muscle)

    parameter_dict = {
        "DeltoideusClavicle_A": {"Fmax": 60, "a_scale": 1148.6, "alpha_a": -1.4 * 10e-1, "tau_fat": 445.5, "pcsa": 2.54},
        "DeltoideusScapula_M": {"Fmax": 264, "a_scale": 5055.7, "alpha_a": -3.2 * 10e-2, "tau_fat": 101.20, "pcsa": 11.18},
        "DeltoideusScapula_P": {"Fmax": 65, "a_scale": 1234.5, "alpha_a": -1.1 * 10e-1, "tau_fat": 342.7, "pcsa": 2.73},
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
        activate_residual_torque=True,  # For voluntary torque activation in the adduction phase where FES is not applied
        external_force_set=None,  # External forces will be added later (resistive_torque)
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
    ) -> str:

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
        pkl = str(Path("result") / f"{num_cycles}_cycle" / f"{num_cycles}_min_{full_suffix}.pkl")
        return pkl

    sims = []
    custom_cost_function_dict = CustomCostFunctions().dict_functions
    for (n_cycles, stim), (cost_fun_key, weight) in product(zip(n_cycles_simultaneous, stimulation), zip(cost_fun_dict["optimized_function"], cost_fun_dict["weight"])):
        index = [custom_cost_function_dict[key]["index"] for key in cost_fun_key]
        pkl_path = make_file_paths(n_cycles, index, ode_solver)
        sims.append(
            {
                "n_cycles_simultaneous": n_cycles,
                "stimulation": stim,
                "cost_fun_key": cost_fun_key,
                "cost_fun_weight": weight,
                "pickle_file_path": pkl_path,
            }
        )
    return sims


def save_sol_in_pkl(sol, simulation_conditions, is_initial_guess=False, torque=None):
    solution = sol[0] if not is_initial_guess else sol[1][0]
    time = solution.stepwise_time(to_merge=[SolutionMerge.NODES]).T[0]
    states = solution.stepwise_states(to_merge=[SolutionMerge.NODES])
    controls = solution.stepwise_controls(to_merge=[SolutionMerge.NODES])
    stim_time = solution.ocp.nlp[0].model.muscles_dynamics_model[0].stim_time
    solving_time_per_ocp = [sol[1][i].solver_time_to_optimize for i in range(len(sol[1]))]
    objective_values_per_ocp = [float(sol[1][i].cost) for i in range(len(sol[1]))]
    objective_values_per_kept_cycle = [float(sol[2][i].cost) for i in range(len(sol[2])-(simulation_conditions["n_cycles_simultaneous"]-1))]
    iter_per_ocp = [sol[1][i].iterations for i in range(len(sol[1]))]
    average_solving_time_per_iter_list = [solving_time_per_ocp[i] / iter_per_ocp[i] for i in range(len(sol[1]))]
    total_average_solving_time_per_iter = average(average_solving_time_per_iter_list)
    number_of_turns_before_failing = len(sol[2])
    convergence_status = [sol[1][i].status for i in range(len(sol[1]))]
    cost_function = np.array(simulation_conditions["cost_fun_key"], dtype=np.str_)
    cost_function_weight = simulation_conditions["cost_fun_weight"]

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
        "cost_function_weight": cost_function_weight,
    }

    for key in states.keys():
        dictionary[key] = states[key]
    for key in controls.keys():
        dictionary[key] = controls[key]

    pickle_file_name = simulation_conditions["pickle_file_path"]
    with open(pickle_file_name, "wb") as file:
        pickle.dump(dictionary, file)

    np.savez_compressed(str(pickle_file_name)[:-4] + ".npz", **dictionary)
    print(simulation_conditions["pickle_file_path"])


def run_optim(mhe_info, abduction_info, simulation_conditions, model_path, save_sol, is_initial_guess=False):
    # --- Set FES model --- #
    stim_time = list(
        np.linspace(
            0,
            mhe_info["n_cycles"] * simulation_conditions["n_cycles_simultaneous"],
            simulation_conditions["stimulation"],
            endpoint=False,
        )
    )
    model = set_fes_model(model_path, stim_time)

    mhe_info["cycle_len"] = int(len(stim_time) / simulation_conditions["n_cycles_simultaneous"])
    mhe_info["n_cycles_simultaneous"] = simulation_conditions["n_cycles_simultaneous"]

    nmpc = prepare_nmpc(
        model=model,
        mhe_info=mhe_info,
        abduction_info=abduction_info,
        simulation_conditions=simulation_conditions,
    )
    nmpc.n_cycles_simultaneous = simulation_conditions["n_cycles_simultaneous"]

    def update_functions(_nmpc: MultiCyclicNonlinearModelPredictiveControl, cycle_idx: int, _sol: Solution):
        print("Optimized window n°" + str(cycle_idx))
        return cycle_idx < mhe_info["n_cycles"]  # True if there are still some cycle to perform

    # Add the penalty cost function plot
    nmpc.add_plot_penalty(CostType.ALL)

    # Set solver for the optimal control problem
    solver = Solver.IPOPT(show_online_optim=False, _max_iter=2000, show_options=dict(show_bounds=True))
    linear_solver = "ma57" if platform == "linux" else "mumps"
    solver.set_linear_solver(linear_solver)

    # Solve the optimal control problem
    sol = nmpc.solve_fes_nmpc(
        update_functions,
        solver=solver,
        total_cycles=mhe_info["n_cycles"],
        external_force=abduction_info["resistive_torque"],
        cycle_solutions=MultiCyclicCycleSolutions.ALL_CYCLES,
        get_all_iterations=True,
        cyclic_options={"states": {}},
        max_consecutive_failing=1,
    )

    result_show = True
    if result_show:
        sol[0].animate(viewer="pyorerun")
        sol[0].graphs(show_bounds=True)

    # Saving the data in a pickle file
    if save_sol:
        save_sol_in_pkl(
            sol,
            simulation_conditions,
            is_initial_guess=is_initial_guess,
            torque=abduction_info["resistive_torque"]["torque"][-1],
        )


def main(
    stimulation_frequency, n_total_cycle, n_cycles_simultaneous, resistive_torque, cost_fun_dict, save
):
    # --- Simulation configuration --- #
    save_sol = save

    # --- Model choice --- #
    model_path = "../../msk_models/Seth/Seth_abd.bioMod"

    # --- MHE parameters --- #
    ode_solver = OdeSolver.COLLOCATION(polynomial_degree=3, method="radau")
    mhe_info = {
        "cycle_duration": 6,
        "n_cycles_to_advance": 1,
        "n_cycles": n_total_cycle,
        "ode_solver": ode_solver,
        "use_sx": False,
    }

    # --- Bike parameters --- #
    abduction_info = {
        "range_config": {"min": np.deg2rad(30), "max": np.deg2rad(110)},  # in radian
        "resistive_torque": {"Segment_application": "humerus", "torque": np.array([0, 0, resistive_torque])},
    }

    # --- Build simulation list --- #
    stimulation = [stimulation_frequency * mhe_info["cycle_duration"] * i for i in n_cycles_simultaneous]

    # --- Build the simulation conditions list --- #
    simulation_conditions_list = create_simulation_list(
        n_cycles_simultaneous=n_cycles_simultaneous,
        stimulation=stimulation,
        cost_fun_dict=cost_fun_dict,
        ode_solver=mhe_info["ode_solver"],
    )

    # --- Run the optimization --- #
    for i in range(len(simulation_conditions_list)):
        run_optim(
            mhe_info=mhe_info,
            abduction_info=abduction_info,
            simulation_conditions=simulation_conditions_list[i],
            model_path=model_path,
            save_sol=save_sol,
        )


if __name__ == "__main__":
    main(
        stimulation_frequency=50,
        n_total_cycle=2,
        n_cycles_simultaneous=[2],
        resistive_torque=0,  # (N.m)
        cost_fun_dict={"optimized_function": [
            ["minimize_root_mean_square_activation"],
            ["minimize_root_mean_square_force"],
            ["minimize_root_mean_square_muscle_stress"],
            ["minimize_root_mean_square_fatigue"],
            ],
            "weight": [
                       [10000],
                       [10000],
                       [10000],
                       [10000],
                       ],
        },
        save=False,
    )
