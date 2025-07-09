"""
This example will perform an optimal control program moving time horizon for a hand cycling motion driven by FES.
"""

import pickle

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.ma.extras import average

from casadi import MX, vertcat

from bioptim import (
    Axis,
    BiorbdModel,
    ConstraintList,
    ConstraintFcn,
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
    SolutionMerge,
    Solution,
    Solver,
    ParameterList,
    Node,
    VariableScalingList,
    PenaltyController,
    ContactType,
)
from cocofest import (
    CustomObjective,
    DingModelPulseWidthFrequencyWithFatigue,
    FesMskModel,
    inverse_kinematics_cycling,
    OcpFesMsk,
    FesNmpcMsk,
)


class MyCyclicNMPC(FesNmpcMsk):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def advance_window_bounds_states(self, sol, n_cycles_simultaneous=None, **extra):
        states = sol.decision_states(to_merge=SolutionMerge.NODES)
        states_keys = states.keys()
        polynomial_order = (
            self.nlp[0].dynamics_type.ode_solver.polynomial_degree + 1
            if isinstance(self.nlp[0].dynamics_type.ode_solver, OdeSolver.COLLOCATION)
            else 1
        )
        for key in states_keys:
            if key == "q" or key == "qdot":
                pass
            else:
                self.nlp[0].x_bounds[key].min[0, 0] = states[key][0][self.cycle_len * polynomial_order]
                self.nlp[0].x_bounds[key].max[0, 0] = states[key][0][self.cycle_len * polynomial_order]
        self.update_stim()
        return True

    def advance_window_initial_guess_states(self, sol, n_cycles_simultaneous=None):
        # Reimplementation of the advance_window method so the rotation of the wheel restart at -pi
        states = sol.decision_states(to_merge=SolutionMerge.NODES)
        fes_state_key = ["A_", "Tau1_", "Km_"]
        polynomial_order = (
            self.nlp[0].dynamics_type.ode_solver.polynomial_degree + 1
            if isinstance(self.nlp[0].dynamics_type.ode_solver, OdeSolver.COLLOCATION)
            else 1
        )
        for key in states.keys():
            if any(x in key for x in fes_state_key):
                if self.nlp[0].x_init[key].init.shape[1] == self.cycle_len + 1:
                    self.nlp[0].x_init[key].init[:, :] = np.array([[states[key][0, -1]] * (self.cycle_len + 1)])
                else:
                    self.nlp[0].x_init[key].init[
                        :, : self.cycle_len * polynomial_order * (self.n_cycles_simultaneous - 1)
                    ] = states[key][:, self.cycle_len * polynomial_order + 1 :]
                    self.nlp[0].x_init[key].init[
                        :, self.cycle_len * polynomial_order * (self.n_cycles_simultaneous - 1) :
                    ] = np.array([[states[key][0, -1]] * (self.cycle_len * polynomial_order + 1)])
            else:
                self.nlp[0].x_init[key].init[:, :] = states[key][:, :]
        return True

    def advance_window_initial_guess_controls(self, sol, n_cycles_simultaneous=None):
        controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
        for key in controls.keys():
            self.nlp[0].u_init[key].init[:, :] = controls[key][:, :]
        return True

# WIP
# def task_performance_coefficient_cost_fun(controller: PenaltyController):
#     F0 = vertcat(
#             *[controller.model.muscles_dynamics_model[i].fmax for i in range(len(controller.model.muscles_dynamics_model))]
#         )
#     force_state = vertcat(
#         *[controller.states["F_" + controller.model.muscles_dynamics_model[i].muscle_name].cx for i in
#           range(len(controller.model.muscles_dynamics_model))]
#     )
#     fatigue_state = vertcat(
#             *[controller.states["A_" + controller.model.muscles_dynamics_model[i].muscle_name].cx for i in range(len(controller.model.muscles_dynamics_model))]
#         )
#     rested_state = vertcat(
#             *[controller.model.muscles_dynamics_model[i].a_scale for i in range(len(controller.model.muscles_dynamics_model))]
#         )
#
#     return (1-(fatigue_state/rested_state)) - (force_state/F0)

def prepare_nmpc(
    model: BiorbdModel | FesMskModel,
    mhe_info: dict,
    cycling_info: dict,
    simulation_conditions: dict,
):
    cycle_duration = mhe_info["cycle_duration"]
    cycle_len = mhe_info["cycle_len"]
    n_cycles_to_advance = mhe_info["n_cycles_to_advance"]
    n_cycles_simultaneous = mhe_info["n_cycles_simultaneous"]
    ode_solver = mhe_info["ode_solver"]
    use_sx = mhe_info["use_sx"]

    turn_number = cycling_info["turn_number"]
    pedal_config = cycling_info["pedal_config"]
    external_force = cycling_info["resistive_torque"]

    minimize_force = simulation_conditions["minimize_force"]
    minimize_fatigue = simulation_conditions["minimize_fatigue"]
    minimize_control = simulation_conditions["minimize_control"]
    cost_fun_weight = simulation_conditions["cost_fun_weight"]
    initial_guess_path = simulation_conditions["init_guess_file_path"]

    window_n_shooting = cycle_len * n_cycles_simultaneous
    window_cycle_duration = cycle_duration * n_cycles_simultaneous
    # Dynamics
    numerical_time_series, external_force_set = set_external_forces(
        n_shooting=window_n_shooting, external_force_dict=external_force, force_name="external_torque"
    )
    numerical_data_time_series, stim_idx_at_node_list = model.muscles_dynamics_model[0].get_numerical_data_time_series(
        window_n_shooting, window_cycle_duration
    )
    numerical_time_series.update(numerical_data_time_series)
    dynamics = set_dynamics(model=model, numerical_time_series=numerical_time_series, ode_solver=ode_solver)
    # Initial q guess
    x_init = set_x_init(window_n_shooting, pedal_config, turn_number, ode_solver=ode_solver, init_file_path=initial_guess_path)
    # Path constraint
    x_bounds, x_init = set_bounds(
        model=model,
        x_init=x_init,
        n_shooting=window_n_shooting,
        turn_number=turn_number,
        ode_solver=ode_solver,
        init_file_path = initial_guess_path,
    )
    # Control path constraint
    u_bounds, u_init, u_scaling = set_u_bounds_and_init(model, window_n_shooting, init_file_path=initial_guess_path)
    objective_functions = set_objective_functions(minimize_force, minimize_fatigue, minimize_control, cost_fun_weight, x_init["q"].init[2][-1])
    # Constraints
    constraints = set_constraints(model, window_n_shooting, turn_number, x_init)
    # Update model
    model = updating_model(model=model, external_force_set=external_force_set, parameters=ParameterList(use_sx=use_sx))
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
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        n_threads=32,
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


def updating_model(model, external_force_set, parameters=None):
    if isinstance(model, FesMskModel):
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
    else:
        model = BiorbdModel(model.path, external_force_set=external_force_set)
    return model


def set_dynamics(model, numerical_time_series, ode_solver):
    dynamics = DynamicsList()
    dynamics.add(
        dynamics_type=model.declare_model_variables,
        dynamic_function=model.muscle_dynamic,
        expand_dynamics=True,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        numerical_data_timeseries=numerical_time_series,
        contact_type=[ContactType.RIGID_EXPLICIT],  # empty list for no contact
        phase=0,
        ode_solver=ode_solver,
    )
    return dynamics


def set_objective_functions(minimize_force, minimize_fatigue, minimize_control, cost_fun_weight, target):
    objective_functions = ObjectiveList()
    if minimize_force:
        objective_functions.add(
            CustomObjective.minimize_overall_muscle_force_production,
            custom_type=ObjectiveFcn.Lagrange,
            node=Node.ALL,
            weight=10000 * cost_fun_weight[0],
            quadratic=True,
        )
    if minimize_fatigue:
        objective_functions.add(
            CustomObjective.minimize_overall_muscle_fatigue,
            custom_type=ObjectiveFcn.Lagrange,
            node=Node.ALL,
            weight=10000 * cost_fun_weight[1],
            quadratic=True,
        )
    if minimize_control:
        objective_functions.add(
            CustomObjective.minimize_overall_stimulation_charge,
            custom_type=ObjectiveFcn.Lagrange,
            node=Node.ALL,
            weight=10000 * cost_fun_weight[2],
            quadratic=True,
        )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        key="q",
        index=2,
        node=Node.END,
        # weight=1e-3,
        weight=10000,
        target=target,
        quadratic=True,
    )

    # WIP
    # objective_functions.add(
    #     task_performance_coefficient_cost_fun,
    #     custom_type=ObjectiveFcn.Mayer,
    #     node=Node.ALL,
    #     weight=1e-3,
    #     quadratic=True,
    # )


    return objective_functions


def set_x_init(n_shooting, pedal_config, turn_number, ode_solver, init_file_path):
    if isinstance(ode_solver, OdeSolver.COLLOCATION):
        n_shooting = n_shooting * (ode_solver.polynomial_degree + 1)
    x_init = InitialGuessList()

    if init_file_path:
        with open(init_file_path, "rb") as file:
            data = pickle.load(file)
        q_guess = data["states"]["q"]
        qdot_guess = data["states"]["qdot"]
        x_init.add("q", q_guess, interpolation=InterpolationType.ALL_POINTS)
        x_init.add("qdot", qdot_guess, interpolation=InterpolationType.ALL_POINTS)
    else:
        # biorbd_model_path = "../../model_msk/simplified_UL_Seth_2D_cycling_for_inverse_kinematics.bioMod"
        biorbd_model_path = "../../model_msk/Wu_Shoulder_Model_mod_kev_inverse_dyn.bioMod"
        q_guess, qdot_guess, qddotguess = inverse_kinematics_cycling(
            biorbd_model_path,
            n_shooting,
            x_center=pedal_config["x_center"],
            y_center=pedal_config["y_center"],
            radius=pedal_config["radius"],
            ik_method="lm",
            cycling_number=turn_number,
        )
        if isinstance(ode_solver, OdeSolver.COLLOCATION):
            x_init.add("q", q_guess, interpolation=InterpolationType.ALL_POINTS)
            x_init.add("qdot", qdot_guess, interpolation=InterpolationType.ALL_POINTS)
        else:
            x_init.add("q", q_guess, interpolation=InterpolationType.EACH_FRAME)
            x_init.add("qdot", qdot_guess, interpolation=InterpolationType.EACH_FRAME)

    return x_init


def set_u_bounds_and_init(bio_model, n_shooting, init_file_path):
    u_bounds, u_init = OcpFesMsk.set_u_bounds_fes(bio_model)
    u_init = InitialGuessList()  # Controls initial guess
    models = bio_model.muscles_dynamics_model
    if init_file_path:
        with open(init_file_path, "rb") as file:
            data = pickle.load(file)
        u_guess = data["controls"]
    else:
        u_guess = None
    for model in models:
        key = "last_pulse_width_" + str(model.muscle_name)
        if u_guess:
            initial_guess = data["controls"][key]
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
        u_scaling.add(key=key, scaling=[10000])
    return (
        u_bounds,
        u_init,
        u_scaling,
    )


def set_bounds(model, x_init, n_shooting, turn_number, ode_solver, init_file_path):
    interpolation_type = InterpolationType.EACH_FRAME
    if isinstance(ode_solver, OdeSolver.COLLOCATION):
        n_shooting = n_shooting * (ode_solver.polynomial_degree + 1)
        interpolation_type = InterpolationType.ALL_POINTS
    x_bounds, x_init_fes = OcpFesMsk.set_x_bounds_fes(model)

    states = None
    if init_file_path:
        with open(init_file_path, "rb") as file:
            data = pickle.load(file)
            states = data["states"]

    for key in x_init_fes.keys():
        initial_guess = np.array([[x_init_fes[key].init[0][0]] * (n_shooting + 1)])
        if states:
            initial_guess = states[key]
        x_init.add(
            key=key,
            initial_guess=initial_guess,
            phase=0,
            interpolation=interpolation_type,
        )

    # Retrieve default bounds from the model for positions and velocities
    q_x_bounds = model.bounds_from_ranges("q")
    x_min_bound = []
    x_max_bound = []
    for i in range(q_x_bounds.min.shape[0]):
        x_min_bound.append([q_x_bounds.min[i][0]] * (n_shooting + 1))
        x_max_bound.append([q_x_bounds.max[i][0]] * (n_shooting + 1))
    slack = 0.2
    for i in range(len(x_min_bound[0])):
        x_min_bound[0][i] = -1
        x_max_bound[0][i] = 2.5
        x_min_bound[1][i] = 0
        x_max_bound[1][i] = 2.5
        x_min_bound[2][i] = x_init["q"].init[2][-1] - slack
        x_max_bound[2][i] = x_init["q"].init[2][0] + slack
    # Adjust bounds at cardinal nodes for a specific coordinate (e.g., index 2)
    cardinal_node_list = [
        i * (n_shooting / ((n_shooting / (n_shooting / turn_number)) * 1))
        for i in range(int((n_shooting / (n_shooting / turn_number)) * 1 + 1))
    ]
    cardinal_node_list = [int(cardinal_node_list[i]) for i in range(len(cardinal_node_list))]
    for i in range(0, 2):
        x_max_bound[2][cardinal_node_list[i]] = x_init["q"].init[2][cardinal_node_list[i]]
        x_min_bound[2][cardinal_node_list[i]] = x_init["q"].init[2][cardinal_node_list[i]]
    for i in range(2, len(cardinal_node_list)):
        x_max_bound[2][cardinal_node_list[i]] = x_init["q"].init[2][cardinal_node_list[i]] + slack
        x_min_bound[2][cardinal_node_list[i]] = x_init["q"].init[2][cardinal_node_list[i]] - slack
    x_bounds.add(key="q", min_bound=x_min_bound, max_bound=x_max_bound, phase=0, interpolation=interpolation_type)
    init_rotation_speed = x_init["qdot"].init[2][0]
    qdot_x_bounds = model.bounds_from_ranges("qdot")
    # qdot_x_bounds.max[2] = [init_rotation_speed, -0.01, -0.01]
    # qdot_x_bounds.min[2] = [init_rotation_speed, -20, -20]

    qdot_x_bounds.max[0] = [10, 10, 10]
    qdot_x_bounds.min[0] = [-10, -10, -10]
    qdot_x_bounds.max[1] = [10, 10, 10]
    qdot_x_bounds.min[1] = [-15, -15, -15]
    qdot_x_bounds.max[2] = [-1, -1, -1]
    qdot_x_bounds.min[2] = [-15, -15, -15]

    x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=0)
    return x_bounds, x_init


def set_constraints(bio_model, n_shooting, turn_number, x_init=None):
    constraints = ConstraintList()
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
    # Adjust bounds at cardinal nodes for a specific coordinate (e.g., index 2)
    cardinal_node_list = [
        i * (n_shooting / ((n_shooting / (n_shooting / turn_number)) * 1))
        for i in range(int((n_shooting / (n_shooting / turn_number)) * 1 + 1))
    ]
    cardinal_node_list = [int(cardinal_node_list[i]) for i in range(len(cardinal_node_list))]
    constraints.add(
        ConstraintFcn.TRACK_STATE,
        key="qdot",
        index=2,
        node=cardinal_node_list[1],
        target=x_init["qdot"].init[2][0],
    )
    return constraints


def set_fes_model(model_path, stim_time):
    dummy_biomodel = BiorbdModel(model_path)
    muscle_name_list = dummy_biomodel.muscle_names
    muscles_model = [DingModelPulseWidthFrequencyWithFatigue(
                    muscle_name=muscle,
                    sum_stim_truncation=10
                ) for muscle in muscle_name_list]

    fes_model = FesMskModel(
        name=None,
        biorbd_path=model_path,
        muscles_model=muscles_model,
        stim_time=stim_time,
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=True,
        activate_residual_torque=False,
        external_force_set=None,  # External forces will be added later
    )

    return fes_model

def create_simulation_list(
    n_cycles_simultaneous: list[int],
    stimulation:           list[int],
    cost_fun_weight:       list[tuple[float, float, float]],
    ode_solver:            OdeSolver(),
) -> list[dict]:

    def make_file_paths(
        n_cycles: int,
        w_force:  float,
        w_fatigue: float,
        w_control: float,
        ode_solver: OdeSolver,
    ) -> tuple[str, str]:

        parts = []
        if w_force:   parts.append(f"{int(w_force*100)}_force")
        if w_fatigue: parts.append(f"{int(w_fatigue*100)}_fatigue")
        if w_control: parts.append(f"{int(w_control*100)}_control")
        weight_suffix = "_".join(parts)

        if isinstance(ode_solver, OdeSolver.COLLOCATION):
            solver_suffix = f"collocation_{ode_solver.polynomial_degree}_{ode_solver.method}"
        elif isinstance(ode_solver, OdeSolver.RK4):
            solver_suffix = f"rk4_{ode_solver.n_integration_steps}"
        else:
            raise RuntimeError("ode_solver must be COLLOCATION or RK4")

        full_suffix = f"{weight_suffix}_{solver_suffix}_with_init"
        pkl = Path("result") / f"{n_cycles}_cycle" / f"{n_cycles}_min_{full_suffix}.pkl"
        init = Path("result/initial_guess") / f"{n_cycles}_initial_guess_{solver_suffix}.pkl"
        return str(pkl), str(init)

    sims = []
    for n_cycles, stim, (w_f, w_fat, w_c) in product(
        n_cycles_simultaneous, stimulation, cost_fun_weight
    ):
        pkl_path, init_path = make_file_paths(n_cycles, w_f, w_fat, w_c, ode_solver)
        sims.append({
            "n_cycles_simultaneous": n_cycles,
            "stimulation":           stim,
            "minimize_force":        bool(w_f),
            "minimize_fatigue":      bool(w_fat),
            "minimize_control":      bool(w_c),
            "cost_fun_weight":       [w_f, w_fat, w_c],
            "pickle_file_path":      pkl_path,
            "init_guess_file_path":  init_path,
        })
    return sims

def save_sol_in_pkl(sol, simulation_conditions):
    time = sol[0].stepwise_time(to_merge=[SolutionMerge.NODES]).T[0]
    states = sol[0].stepwise_states(to_merge=[SolutionMerge.NODES])
    controls = sol[0].stepwise_controls(to_merge=[SolutionMerge.NODES])
    stim_time = sol[0].ocp.nlp[0].model.muscles_dynamics_model[0].stim_time
    solving_time_per_ocp = [sol[1][i].solver_time_to_optimize for i in range(len(sol[1]))]
    objective_values_per_ocp = [float(sol[1][i].cost) for i in range(len(sol[1]))]
    iter_per_ocp = [sol[1][i].iterations for i in range(len(sol[1]))]
    average_solving_time_per_iter_list = [solving_time_per_ocp[i] / iter_per_ocp[i] for i in range(len(sol[1]))]
    total_average_solving_time_per_iter = average(average_solving_time_per_iter_list)
    number_of_turns_before_failing = len(sol[2]) - 1 + simulation_conditions["n_cycles_simultaneous"]
    convergence_status = [sol[1][i].status for i in range(len(sol[1]))]
    dictionary = {
        "time": time,
        "states": states,
        "controls": controls,
        "stim_time": stim_time,
        "solving_time_per_ocp": solving_time_per_ocp,
        "objective_values_per_ocp": objective_values_per_ocp,
        "number_of_turns_before_failing": number_of_turns_before_failing,
        "convergence_status": convergence_status,
        "iter_per_ocp": iter_per_ocp,
        "average_solving_time_per_iter_list": average_solving_time_per_iter_list,
        "total_average_solving_time_per_iter": total_average_solving_time_per_iter
    }
    pickle_file_name = simulation_conditions["pickle_file_path"]
    with open(pickle_file_name, "wb") as file:
        pickle.dump(dictionary, file)
    print(simulation_conditions["pickle_file_path"])

def run_optim(mhe_info, cycling_info, simulation_conditions, model_path, save_sol):
    # --- Set FES model --- #
    stim_time = list(
        np.linspace(
            0,
            mhe_info["cycle_duration"] * simulation_conditions["n_cycles_simultaneous"],
            simulation_conditions["stimulation"] + 1,
        )[:-1]
    )
    model = set_fes_model(model_path, stim_time)

    mhe_info["cycle_len"] = (int((len(stim_time)) / simulation_conditions["n_cycles_simultaneous"])) * 2
    mhe_info["n_cycles_simultaneous"] = simulation_conditions["n_cycles_simultaneous"]

    cycling_info["turn_number"] = simulation_conditions["n_cycles_simultaneous"]

    nmpc = prepare_nmpc(
        model=model,
        mhe_info=mhe_info,
        cycling_info=cycling_info,
        simulation_conditions=simulation_conditions,
    )
    nmpc.n_cycles_simultaneous = simulation_conditions["n_cycles_simultaneous"]

    def update_functions(_nmpc: MultiCyclicNonlinearModelPredictiveControl, cycle_idx: int, _sol: Solution):
        print("Optimized window n°" + str(cycle_idx))
        return cycle_idx < mhe_info["n_cycles"]  # True if there are still some cycle to perform

    # Add the penalty cost function plot
    nmpc.add_plot_penalty(CostType.ALL)
    # Solve the optimal control problem
    sol = nmpc.solve_fes_nmpc(
        update_functions,
        solver=Solver.IPOPT(show_online_optim=False, _max_iter=10000, show_options=dict(show_bounds=True)),
        # , _linear_solver="ma57"),
        total_cycles=mhe_info["n_cycles"],
        external_force=cycling_info["resistive_torque"],
        cycle_solutions=MultiCyclicCycleSolutions.ALL_CYCLES,
        get_all_iterations=True,
        cyclic_options={"states": {}},
        max_consecutive_failing=1,
    )
    # sol[0].animate(viewer="pyorerun")
    sol[0].graphs(show_bounds=True)
    # Saving the data in a pickle file
    if save_sol:
        save_sol_in_pkl(sol, simulation_conditions)

def main():
    # --- Configuration --- #
    save_sol = False
    run_initial_guess = True
    # Chosen MSK model
    # model_path = "../../model_msk/simplified_UL_Seth_2D_cycling.bioMod"
    model_path = "../../model_msk/Wu_Shoulder_Model_mod_kev_v2.bioMod"

    # MHE parameters
    mhe_info = {
        "cycle_duration": 1,
        "n_cycles_to_advance": 1,
        "n_cycles": 10,
        "ode_solver": OdeSolver.COLLOCATION(polynomial_degree=5, method="radau"),
        # "ode_solver": OdeSolver.RK4(n_integration_steps=10),
        "use_sx": False
    }

    # Bike parameters
    cycling_info = {"pedal_config": {"x_center": 0.35, "y_center": 0.0, "radius": 0.1},
                    "resistive_torque": {"Segment_application": "wheel", "torque": np.array([0, 0, -1])}}

    # Build cost function parameters
    cost_function_weight = [
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (0.75, 0.25, 0), (0.5, 0.5, 0), (0.25, 0.75, 0),
        (0.75, 0, 0.25), (0.5, 0, 0.5), (0.25, 0, 0.75),
        (0, 0.75, 0.25), (0, 0.5, 0.5), (0, 0.25, 0.75),
        (1/3, 1/3, 1/3),
    ]  # (min_force, min_fatigue, min_control)

    # Build simulation list
    stimulation_frequency = 50
    n_cycles_simultaneous = [2, 3, 4, 5]
    stimulation = [stimulation_frequency * i for i in n_cycles_simultaneous]
    simulation_conditions_list = create_simulation_list(n_cycles_simultaneous=n_cycles_simultaneous,
                                                        stimulation=stimulation,
                                                        cost_fun_weight=cost_function_weight,
                                                        ode_solver=mhe_info["ode_solver"])

    # --- Run the initial guess optimization --- #
    if run_initial_guess:
        init_guess_mhe_info = {
            "cycle_duration": mhe_info["cycle_duration"],
            "n_cycles_to_advance": mhe_info["n_cycles_to_advance"],
            "n_cycles": 1,
            "ode_solver": mhe_info["ode_solver"],
            "use_sx": mhe_info["use_sx"]
        }

        ode_solver = mhe_info["ode_solver"]
        if isinstance(ode_solver, OdeSolver.COLLOCATION):
            solver_suffix = f"collocation_{ode_solver.polynomial_degree}_{ode_solver.method}"
        elif isinstance(ode_solver, OdeSolver.RK4):
            solver_suffix = f"rk4_{ode_solver.n_integration_steps}"
        else:
            raise RuntimeError("ode_solver must be COLLOCATION or RK4")

        for i in range(len(n_cycles_simultaneous)):
            simulation_conditions = {
                "n_cycles_simultaneous": n_cycles_simultaneous[i],
                "stimulation": stimulation[i],
                "minimize_force": False,
                "minimize_fatigue": False,
                "minimize_control": False,
                "cost_fun_weight": [0, 0, 0],
                "pickle_file_path": Path("result") / "initial_guess" / f"{n_cycles_simultaneous[i]}_initial_guess_{solver_suffix}.pkl",
                "init_guess_file_path": None,
            }

            run_optim(mhe_info=init_guess_mhe_info,
                      cycling_info=cycling_info,
                      simulation_conditions=simulation_conditions,
                      model_path=model_path,
                      save_sol=False)

    # --- Run the optimization --- #
    for i in range(len(simulation_conditions_list)):
        run_optim(mhe_info=mhe_info,
                  cycling_info=cycling_info,
                  simulation_conditions=simulation_conditions_list[i],
                  model_path=model_path,
                  save_sol=save_sol)

if __name__ == "__main__":
    main()
