"""
This example demonstrates how long hand cycling could be maintained at 30 Hz with standard clinical settings (fixed
pulse width and defined stimulation angles). The crank torque will be optimized to achieve the target cadence (60 rpm).
Then, after each cycle, the muscle fatigue will be updated based on the activation history and tested with an optimal
control problem. If an optimal solution is found, the standard clinical settings simulation continues considering a
cycle could be performed. When an optimal solution is not found, this ends meaning it reached muscle failure.
"""

import pickle
import numpy as np

from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    ParameterList,
    BoundsList,
    InitialGuessList,
    ParameterObjectiveList,
    Node,
    Axis,
    OdeSolver,
    VariableScalingList,
    InterpolationType,
    SolutionMerge,
    CostType,
)

from cycling_pulse_width_mhe import (
    set_fes_model,
    set_external_forces,
    set_dynamics,
    set_q_qdot_init,
    set_x_bounds,
    updating_model,
    set_u_bounds_and_init,
)
from cost_functions import CustomCostFunctions

from collections import defaultdict

MUSCLES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")
STATE_VARS = ("A", "F", "Km", "Tau1")
CONTROL_VARS = ("last_pulse_width",)


def prepare_standard_fes_cycling(optim_info, cycling_info, model, previous_problem=None, previous_sol=None):
    # --- Optimization info --- #
    phase_time = optim_info["phase_time"]
    n_shooting = optim_info["n_shooting"]
    ode_solver = optim_info["ode_solver"]
    use_sx = optim_info["use_sx"]

    # --- Cycling info --- #
    turn_number = cycling_info["turn_number"]
    pedal_config = cycling_info["pedal_config"]
    external_force = cycling_info["resistive_torque"]

    # --- Set dynamics --- #
    # --- External force numerical time series --- #
    numerical_time_series, external_force_set = set_external_forces(
        n_shooting=n_shooting, external_force_dict=external_force, force_name="external_torque"
    )
    # --- Stimulation instant numerical time series --- #
    numerical_data_time_series, stim_idx_at_node_list = model.muscles_dynamics_model[0].get_numerical_data_time_series(
        n_shooting, phase_time
    )
    numerical_time_series.update(numerical_data_time_series)
    # --- Dynamics --- #
    dynamics = set_dynamics(model=model, numerical_time_series=numerical_time_series, ode_solver=ode_solver)

    # --- Set states --- #
    if previous_problem is None:
        # --- Set q (position and speed) initial guesses --- #
        x_init = set_q_qdot_init(
            n_shooting=n_shooting,
            pedal_config=pedal_config,
            turn_number=turn_number,
            ode_solver=ode_solver,
            init_file_path=None,
        )

        # --- Set bounds and FES initial guesses --- #
        x_bounds, x_init = set_x_bounds(
            model=model,
            x_init=x_init,
            n_shooting=n_shooting,
            ode_solver=ode_solver,
            init_file_path=None,
        )
    else:
        x_bounds = previous_problem.nlp[0].x_bounds
        x_init = previous_problem.nlp[0].x_init

        prev_state_result = previous_sol.decision_states(to_merge=SolutionMerge.NODES)
        for key in x_bounds.keys():
            if key not in ("q", "qdot"):
                x_bounds[key].min[:,0] = prev_state_result[key][:, -1]
                x_bounds[key].max[:,0] = prev_state_result[key][:, -1]

    # --- Set controls --- #
    if previous_problem is None:
        u_bounds, u_init, u_scaling = set_standard_u_bounds_and_init(
            bio_model=model,
            n_shooting=n_shooting
        )
    else:
        u_bounds = previous_problem.nlp[0].u_bounds
        u_init = previous_problem.nlp[0].u_init
        u_init["tau"].init[:, :] = previous_sol.decision_controls(to_merge=SolutionMerge.NODES)["tau"]  # Previous solution as initial guess
        u_scaling = previous_problem.nlp[0].u_scaling

    # --- Set objective --- #
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        key="qdot",
        index=2,
        node=Node.ALL,
        weight=1e6,
        target=-2 * np.pi * turn_number / phase_time,
        quadratic=True,
    )

    # --- Set parameters --- #
    parameters = ParameterList(use_sx=use_sx)
    parameters_bounds = BoundsList()
    parameters_init = InitialGuessList()
    parameters_objectives = ParameterObjectiveList()

    # --- Set constraints --- #
    constraints = ConstraintList()
    # --- Constraining wheel center position to a fix position --- #
    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        node=Node.START,
        marker_index=model.marker_index("wheel_center"),
        axes=[Axis.X, Axis.Y],
    )
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        first_marker="wheel_center",
        second_marker="global_wheel_center",
        node=Node.START,
        axes=[Axis.X, Axis.Y],
    )

    # --- Update model for resistive torque --- #
    model.activate_residual_torque=True
    model = updating_model(model=model, external_force_set=external_force_set, parameters=parameters)

    return OptimalControlProgram(
        bio_model=[model],
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=phase_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        constraints=constraints,
        parameters=parameters,
        parameter_init=parameters_init,
        parameter_bounds=parameters_bounds,
        parameter_objectives=parameters_objectives,
        u_scaling=u_scaling,
        n_threads=48,
        use_sx=use_sx,
    )

def set_pw_dictionary(model, n_shooting):
    pulse_width_dictionary = {}
    angle_vector = np.linspace(0, 360, n_shooting, endpoint=False)

    # --- Muscle stimulation ranges based on the RehaMove arm cycling bike default settings --- #
    delt_ant_range = [20, 180]
    delt_post_range = [220, 10]
    biceps_range = [220, 10]
    triceps_range = [20, 180]

    delt_ant_pw_vector = np.where(
        (angle_vector >= delt_ant_range[0]) & (angle_vector <= delt_ant_range[1]),
        0.0003,
        model[0].pd0,
    )
    delt_post_pw_vector = np.where(
        (angle_vector >= delt_post_range[0]) | (angle_vector <= delt_post_range[1]),
        0.0003,
        model[0].pd0,
    )
    biceps_pw_vector = np.where(
        (angle_vector >= biceps_range[0]) | (angle_vector <= biceps_range[1]),
        0.0003,
        model[0].pd0,
    )
    triceps_pw_vector = np.where(
        (angle_vector >= triceps_range[0]) & (angle_vector <= triceps_range[1]),
        0.0003,
        model[0].pd0,
    )

    for muscle in model:
        if muscle.muscle_name == "Delt_ant":
            pulse_width_dictionary[muscle.muscle_name] = delt_ant_pw_vector
        elif muscle.muscle_name == "Delt_post":
            pulse_width_dictionary[muscle.muscle_name] = delt_post_pw_vector
        elif muscle.muscle_name == "Biceps":
            pulse_width_dictionary[muscle.muscle_name] = biceps_pw_vector
        elif muscle.muscle_name == "Triceps":
            pulse_width_dictionary[muscle.muscle_name] = triceps_pw_vector
    return pulse_width_dictionary


def set_standard_u_bounds_and_init(bio_model, n_shooting):
    # --- Stimulation controls --- #
    u_bounds = BoundsList()
    u_init = InitialGuessList()
    models = bio_model.muscles_dynamics_model

    pulse_width_dictionary = set_pw_dictionary(models, n_shooting)
    for model in models:
        key = "last_pulse_width_" + str(model.muscle_name)
        reshaped_bounds = np.array([list(pulse_width_dictionary[model.muscle_name])])
        u_init.add(key=key, initial_guess=reshaped_bounds, phase=0,
                   interpolation=InterpolationType.EACH_FRAME)
        u_bounds.add(key=key, min_bound=reshaped_bounds,
                     max_bound=reshaped_bounds, phase=0,
                     interpolation=InterpolationType.EACH_FRAME)

    # --- Pedal assistance control --- #
    u_init.add(key="tau", initial_guess=np.array([[0]*n_shooting]*3), phase=0,
               interpolation=InterpolationType.EACH_FRAME)
    u_bounds.add(key="tau", min_bound=np.array([0, 0, -10]),
                 max_bound=np.array([0, 0, 10]), phase=0)

    u_scaling = VariableScalingList()
    for model in bio_model.muscles_dynamics_model:
        key = "last_pulse_width_" + str(model.muscle_name)
        u_scaling.add(key=key, scaling=[1 / 400])
    u_scaling.add(key="tau", scaling=[1, 1, 1])

    return (
        u_bounds,
        u_init,
        u_scaling,
    )

def prepare_ocp_fes_cycling(optim_info, cycling_info, model, previous_problem=None, previous_sol=None):
    # --- Optimization info --- #
    phase_time = optim_info["phase_time"]
    n_shooting = optim_info["n_shooting"]
    ode_solver = optim_info["ode_solver"]
    use_sx = optim_info["use_sx"]

    # --- Cycling info --- #
    turn_number = cycling_info["turn_number"]
    pedal_config = cycling_info["pedal_config"]
    external_force = cycling_info["resistive_torque"]

    # --- Set dynamics --- #
    # --- External force numerical time series --- #
    numerical_time_series, external_force_set = set_external_forces(
        n_shooting=n_shooting, external_force_dict=external_force, force_name="external_torque"
    )
    # --- Stimulation instant numerical time series --- #
    numerical_data_time_series, stim_idx_at_node_list = model.muscles_dynamics_model[0].get_numerical_data_time_series(
        n_shooting, phase_time
    )
    numerical_time_series.update(numerical_data_time_series)
    # --- Dynamics --- #
    dynamics = set_dynamics(model=model, numerical_time_series=numerical_time_series, ode_solver=ode_solver)

    # --- Set states --- #
    # --- Set q (position and speed) initial guesses --- #
    x_init = set_q_qdot_init(
        n_shooting=n_shooting,
        pedal_config=pedal_config,
        turn_number=turn_number,
        ode_solver=ode_solver,
        init_file_path=None,
    )

    # --- Set bounds and FES initial guesses --- #
    x_bounds, x_init = set_x_bounds(
        model=model,
        x_init=x_init,
        n_shooting=n_shooting,
        ode_solver=ode_solver,
        init_file_path=None,
    )
    init_shape = x_init["qdot"].init[-1].shape[0]
    x_init["qdot"].init[-1] = np.array([-2*np.pi] * init_shape)

    if previous_sol:
        prev_state_result = previous_sol.decision_states(to_merge=SolutionMerge.NODES)  # stewise_states
        for key in x_bounds.keys():
            if key == "A_" + "Delt_ant" or key == "A_" + "Delt_post" or key == "A_" + "Biceps" or key == "A_" + "Triceps":
                x_bounds[key].min[0, 0] = prev_state_result[key][0][-1]
                x_bounds[key].max[0, 0] = prev_state_result[key][0][-1]
            if key == "Km_" + "Delt_ant" or key == "Km_" + "Delt_post" or key == "Km_" + "Biceps" or key == "Km_" + "Triceps":
                x_bounds[key].min[0, 0] = prev_state_result[key][0][-1]
                x_bounds[key].max[0, 0] = prev_state_result[key][0][-1]
            if key == "Tau1_" + "Delt_ant" or key == "Tau1_" + "Delt_post" or key == "Tau1_" + "Biceps" or key == "Tau1_" + "Triceps":
                x_bounds[key].min[0, 0] = prev_state_result[key][0][-1]
                x_bounds[key].max[0, 0] = prev_state_result[key][0][-1]

    # --- Set controls --- #
    if previous_sol is None:
        u_bounds, u_init, u_scaling = set_u_bounds_and_init(
            bio_model=model,
            n_shooting=n_shooting,
            init_file_path=None,
        )
    else:
        u_bounds = previous_problem.nlp[0].u_bounds
        u_init = InitialGuessList()
        prev_control_result = previous_sol.decision_controls(to_merge=SolutionMerge.NODES)
        for muscle_model in model.muscles_dynamics_model:
            key = "last_pulse_width_" + str(muscle_model.muscle_name)
            u_init.add(key=key, initial_guess=prev_control_result[key], phase=0,
                       interpolation=InterpolationType.EACH_FRAME)
        u_scaling = previous_problem.nlp[0].u_scaling

    # --- Set objective --- #
    objective_functions = ObjectiveList()
    objective_functions.add(
        CustomCostFunctions.minimize_root_mean_square_activation,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        weight=1e6,
        quadratic=False,
    )

    # --- Set parameters --- #
    parameters = ParameterList(use_sx=use_sx)
    parameters_bounds = BoundsList()
    parameters_init = InitialGuessList()
    parameters_objectives = ParameterObjectiveList()

    # --- Set constraints --- #
    constraints = ConstraintList()
    # --- Constraining wheel center position to a fix position --- #
    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        node=Node.START,
        marker_index=model.marker_index("wheel_center"),
        axes=[Axis.X, Axis.Y],
    )
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        first_marker="wheel_center",
        second_marker="global_wheel_center",
        node=Node.START,
        axes=[Axis.X, Axis.Y],
    )

    # --- Update model for resistive torque --- #
    model.activate_residual_torque=False
    model = updating_model(model=model, external_force_set=external_force_set, parameters=parameters)

    return OptimalControlProgram(
        bio_model=[model],
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=phase_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        constraints=constraints,
        parameters=parameters,
        parameter_init=parameters_init,
        parameter_bounds=parameters_bounds,
        parameter_objectives=parameters_objectives,
        u_scaling=u_scaling,
        n_threads=48,
        use_sx=use_sx,
    )

def solution_to_dict(solution, muscles=MUSCLES):
    time = solution.stepwise_time(to_merge=[SolutionMerge.NODES]).T[0]
    states = solution.stepwise_states(to_merge=[SolutionMerge.NODES])
    controls = solution.stepwise_controls(to_merge=[SolutionMerge.NODES])

    out = {"time": time}

    for m in muscles:
        for v in STATE_VARS:
            key = f"{v}_{m}"
            out[key] = states[key]

    for m in muscles:
        for v in CONTROL_VARS:
            key = f"{v}_{m}"
            out[key] = controls[key]

    if "tau" in controls:
        out["tau"] = controls["tau"][2, :]

    return out

def append_cycle(store_dict_of_lists, cycle_dict):
    for k, v in cycle_dict.items():
        store_dict_of_lists[k].append(v)


def main():
    # --- Optimization info --- #
    optim_info = {
        "phase_time": 1,  # 1 second per turn (60 rpm)
        "n_shooting": 30,  # Corresponding to 30 Hz
        "ode_solver": OdeSolver.COLLOCATION(),
        "use_sx": False,
    }

    # --- Cycling info --- #
    resistive_torque = -0.20  # Nm
    cycling_info = {
        "turn_number": 1,
        "pedal_config": {"x_center": 0.35, "y_center": 0.0, "radius": 0.1},
        "resistive_torque": {"Segment_application": "wheel", "torque": np.array([0, 0, resistive_torque])},
    }

    fes_ocp_solution_found = True
    cycle_to_failure = 0

    # --- Set FES model --- #
    stim_time = list(
        np.linspace(
            0,
            optim_info["phase_time"],
            optim_info["n_shooting"],
            endpoint=False,
        )
    )
    model = set_fes_model(model_path = "../../msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod",
                          stim_time=stim_time)

    standard_cycling_problem = None
    standard_cycling_sol = None
    fes_ocp_problem = None
    fes_ocp_sol = None

    standard_store = defaultdict(list)
    optim_store = defaultdict(list)

    while fes_ocp_solution_found:
        standard_cycling_problem = prepare_standard_fes_cycling(optim_info=optim_info,
                                                                cycling_info=cycling_info,
                                                                model=model,
                                                                previous_problem=standard_cycling_problem,
                                                                previous_sol=standard_cycling_sol,
        )
        standard_cycling_sol = standard_cycling_problem.solve()

        fes_ocp_problem = prepare_ocp_fes_cycling(optim_info=optim_info,
                                                  cycling_info=cycling_info,
                                                  model=model,
                                                  previous_problem=fes_ocp_problem,
                                                  previous_sol=fes_ocp_sol,
        )
        fes_ocp_problem.add_plot_penalty(CostType.ALL)
        fes_ocp_sol = fes_ocp_problem.solve()

        fes_ocp_solution_found = fes_ocp_sol.status == 0
        cycle_to_failure += 1 if fes_ocp_solution_found else 0

        # --- Append results --- #
        standard_cycle = solution_to_dict(standard_cycling_sol)
        optim_cycle = solution_to_dict(fes_ocp_sol)

        append_cycle(standard_store, standard_cycle)
        append_cycle(optim_store, optim_cycle)

        print(f"Cycle {cycle_to_failure} completed.")
    print(f"Muscle failure reached after {cycle_to_failure} complete cycles.")

    # --- Save results --- #
    dictionary = (
            {f"standard_{k}": v for k, v in standard_store.items()}
            | {f"optim_{k}": v for k, v in optim_store.items()}
            | {"cycle_to_failure": cycle_to_failure}
    )

    pickle_file_name = "standard_cycling_to_failure.pkl"
    pickle_dict = {"standard": dict(standard_store), "optim": dict(optim_store), "cycle_to_failure": cycle_to_failure}
    with open(pickle_file_name, "wb") as file:
        pickle.dump(pickle_dict, file)

    np.savez_compressed(str(pickle_file_name)[:-4] + ".npz", **dictionary)

if __name__ == "__main__":
    main()
