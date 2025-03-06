"""
This example will do an optimal control program of a 100 steps tracking a hand cycling motion with a torque driven and
a torque resistance at the handle.
"""

import pickle
import numpy as np

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
        min_pedal_bound = self.nlp[0].x_bounds["q"].min[-1, 0]
        max_pedal_bound = self.nlp[0].x_bounds["q"].max[-1, 0]
        super(MyCyclicNMPC, self).advance_window_bounds_states(sol)
        self.nlp[0].x_bounds["q"].min[-1, 0] = min_pedal_bound
        self.nlp[0].x_bounds["q"].max[-1, 0] = max_pedal_bound
        return True

    def advance_window_initial_guess_states(self, sol, n_cycles_simultaneous=None):
        # Reimplementation of the advance_window method so the rotation of the wheel restart at -pi
        super(MyCyclicNMPC, self).advance_window_initial_guess_states(sol)
        # q = sol.decision_states(to_merge=SolutionMerge.NODES)["q"]
        # self.nlp[0].x_init["q"].init[:, :] = q[:, :]  # Keep the previously found value for the wheel
        return True


def prepare_nmpc(
    model: BiorbdModel | FesMskModel,
    cycle_duration: int | float,
    cycle_len: int,
    n_cycles_to_advance: int,
    n_cycles_simultaneous: int,
    turn_number: int,
    pedal_config: dict,
    external_force: dict,
    minimize_force: bool = True,
    minimize_fatigue: bool = False,
    use_sx: bool = False,
):

    window_n_shooting = cycle_len * n_cycles_simultaneous
    window_cycle_duration = cycle_duration * n_cycles_simultaneous

    # Dynamics
    numerical_time_series, external_force_set = set_external_forces(
        n_shooting=window_n_shooting, external_force_dict=external_force
    )
    numerical_data_time_series, stim_idx_at_node_list = model.muscles_dynamics_model[0].get_numerical_data_time_series(
        window_n_shooting, window_cycle_duration
    )
    numerical_time_series.update(numerical_data_time_series)

    dynamics = set_dynamics(model=model, numerical_time_series=numerical_time_series)

    # Define objective functions
    objective_functions = set_objective_functions(minimize_force, minimize_fatigue)

    # Initial q guess
    x_init = set_x_init(window_n_shooting, pedal_config, turn_number)

    # Path constraint
    x_bounds, x_init = set_bounds(
        model=model,
        x_init=x_init,
        n_shooting=window_n_shooting,
        turn_number=turn_number,
    )

    # Control path constraint
    u_bounds, u_init, u_scaling = set_u_bounds_and_init(model)

    # Constraints
    constraints = set_constraints(model, window_n_shooting, turn_number)

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
        # ode_solver=OdeSolver.RK4(n_integration_steps=10),
        ode_solver=OdeSolver.COLLOCATION(polynomial_degree=2, method="radau"),
        n_threads=20,
        use_sx=use_sx,
    )


def set_external_forces(n_shooting, external_force_dict):
    external_force_set = ExternalForceSetTimeSeries(nb_frames=n_shooting)
    external_force_array = np.array(external_force_dict["torque"])
    reshape_values_array = np.tile(external_force_array[:, np.newaxis], (1, n_shooting))
    external_force_set.add_torque(segment=external_force_dict["Segment_application"], values=reshape_values_array)
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


def set_dynamics(model, numerical_time_series):
    dynamics = DynamicsList()
    dynamics.add(
        dynamics_type=model.declare_model_variables,
        dynamic_function=model.muscle_dynamic,
        expand_dynamics=True,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        numerical_data_timeseries=numerical_time_series,
        with_contact=True,
        phase=0,
    )
    return dynamics


def set_objective_functions(minimize_force, minimize_fatigue):
    objective_functions = ObjectiveList()

    if minimize_force:
        objective_functions.add(
            CustomObjective.minimize_multibody_muscle_force_norm,
            custom_type=ObjectiveFcn.Lagrange,
            weight=1,
            quadratic=True,
        )

    if minimize_fatigue:
        objective_functions.add(
            CustomObjective.minimize_multibody_muscle_fatigue,
            custom_type=ObjectiveFcn.Lagrange,
            weight=1,
            quadratic=True,
        )

    return objective_functions


def set_x_init(n_shooting, pedal_config, turn_number):
    x_init = InitialGuessList()

    biorbd_model_path = "../../model_msk/simplified_UL_Seth_2D_cycling_for_inverse_kinematics.bioMod"
    q_guess, qdot_guess, qddotguess = inverse_kinematics_cycling(
        biorbd_model_path,
        n_shooting,
        x_center=pedal_config["x_center"],
        y_center=pedal_config["y_center"],
        radius=pedal_config["radius"],
        ik_method="lm",
        cycling_number=turn_number,
    )
    x_init.add("q", q_guess, interpolation=InterpolationType.EACH_FRAME)
    x_init.add("qdot", [0, 0, -6], interpolation=InterpolationType.CONSTANT)
    # x_init.add("qdot", qdot_guess, interpolation=InterpolationType.EACH_FRAME)

    return x_init


def set_u_bounds_and_init(bio_model):
    u_bounds, u_init = OcpFesMsk.set_u_bounds_fes(bio_model)
    u_scaling = VariableScalingList()
    for model in bio_model.muscles_dynamics_model:
        key = "last_pulse_width_" + str(model.muscle_name)
        u_scaling.add(key=key, scaling=[10000])
    return (
        u_bounds,
        u_init,
        u_scaling,
    )


def set_bounds(model, x_init, n_shooting, turn_number):
    x_bounds, x_init_fes = OcpFesMsk.set_x_bounds_fes(model)
    for key in x_init_fes.keys():
        x_init[key] = x_init_fes[key]

    # Retrieve default bounds from the model for positions and velocities
    q_x_bounds = model.bounds_from_ranges("q")

    x_min_bound = []
    x_max_bound = []
    for i in range(q_x_bounds.min.shape[0]):
        x_min_bound.append([q_x_bounds.min[i][0]] * (n_shooting + 1))
        x_max_bound.append([q_x_bounds.max[i][0]] * (n_shooting + 1))

    slack = 0.2
    for i in range(len(x_min_bound[0])):
        x_min_bound[0][i] = -0.5
        x_max_bound[0][i] = 1.5
        x_min_bound[1][i] = 1
        x_min_bound[2][i] = x_init["q"].init[2][-1] - slack
        x_max_bound[2][i] = x_init["q"].init[2][0] + slack

    # Adjust bounds at cardinal nodes for a specific coordinate (e.g., index 2)
    cardinal_node_list = [
        i * (n_shooting / ((n_shooting / (n_shooting / turn_number)) * 1))
        for i in range(int((n_shooting / (n_shooting / turn_number)) * 1 + 1))
    ]
    cardinal_node_list = [int(cardinal_node_list[i]) for i in range(len(cardinal_node_list))]

    x_min_bound[2][0] = x_init["q"].init[2][0]
    x_max_bound[2][0] = x_init["q"].init[2][0]
    x_min_bound[2][-1] = x_init["q"].init[2][-1]
    x_max_bound[2][-1] = x_init["q"].init[2][-1]

    for i in range(1, len(cardinal_node_list) - 1):
        x_max_bound[2][cardinal_node_list[i]] = x_init["q"].init[2][cardinal_node_list[i]] + slack
        x_min_bound[2][cardinal_node_list[i]] = x_init["q"].init[2][cardinal_node_list[i]] - slack

    x_bounds.add(
        key="q", min_bound=x_min_bound, max_bound=x_max_bound, phase=0, interpolation=InterpolationType.EACH_FRAME
    )

    # Modify bounds for velocities (e.g., setting maximum pedal speed to 0 to prevent the pedal to go backward)
    qdot_x_bounds = model.bounds_from_ranges("qdot")
    qdot_x_bounds.max[0] = [4, 4, 4]
    qdot_x_bounds.min[0] = [-4, -4, -4]
    qdot_x_bounds.max[1] = [4, 4, 4]
    qdot_x_bounds.min[1] = [-4, -4, -4]
    qdot_x_bounds.max[2] = [-2, -2, -2]
    qdot_x_bounds.min[2] = [-15, -15, -15]
    x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=0)
    return x_bounds, x_init


def set_constraints(bio_model, n_shooting, turn_number):
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

    for i in range(1, len(cardinal_node_list) - 1):
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            first_marker="wheel_center",
            second_marker="global_wheel_center",
            node=cardinal_node_list[i],
            axes=[Axis.X, Axis.Y],
        )

    return constraints


def set_model(model_path, stim_time):
    # Define muscle dynamics for the FES-driven model
    DeltoideusClavicle_A = DingModelPulseWidthFrequencyWithFatigue(
        muscle_name="DeltoideusClavicle_A", sum_stim_truncation=10
    )
    DeltoideusScapula_P = DingModelPulseWidthFrequencyWithFatigue(
        muscle_name="DeltoideusScapula_P", sum_stim_truncation=10
    )
    TRIlong = DingModelPulseWidthFrequencyWithFatigue(muscle_name="TRIlong", sum_stim_truncation=10)
    BIC_long = DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIC_long", sum_stim_truncation=10)
    BIC_brevis = DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIC_brevis", sum_stim_truncation=10)

    muscles_model = [DeltoideusClavicle_A, DeltoideusScapula_P, TRIlong, BIC_long, BIC_brevis]

    model = FesMskModel(
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
    return model


def main():
    """
    Main function to configure and solve the optimal control problem.
    """
    # --- Configuration --- #
    model_path = "../../model_msk/simplified_UL_Seth_2D_cycling.bioMod"

    # NMPC parameters
    cycle_duration = 1
    n_cycles_to_advance = 1
    n_cycles = 3

    # Bike parameters
    pedal_config = {"x_center": 0.35, "y_center": 0.0, "radius": 0.1}

    resistive_torque = {
        "Segment_application": "wheel",
        "torque": np.array([0, 0, -1]),
    }

    simulation_conditions_1 = {
        "n_cycles_simultaneous": 3,
        "stimulation": 99,
        "minimize_force": True,
        "minimize_fatigue": False,
        "pickle_file_path": "3_min_force.pkl",
    }
    simulation_conditions_2 = {
        "n_cycles_simultaneous": 3,
        "stimulation": 99,
        "minimize_force": False,
        "minimize_fatigue": True,
        "pickle_file_path": "3_min_fatigue.pkl",
    }
    simulation_conditions_3 = {
        "n_cycles_simultaneous": 10,
        "stimulation": 330,
        "minimize_force": True,
        "minimize_fatigue": False,
        "pickle_file_path": "10_min_force.pkl",
    }
    simulation_conditions_4 = {
        "n_cycles_simultaneous": 10,
        "stimulation": 330,
        "minimize_force": False,
        "minimize_fatigue": True,
        "pickle_file_path": "10_min_fatigue.pkl",
    }
    simulation_conditions_list = [
        simulation_conditions_1,
        simulation_conditions_2,
        simulation_conditions_3,
        simulation_conditions_4,
    ]

    for i in range(len(simulation_conditions_list)):
        # --- Set FES model --- #
        stim_time = list(
            np.linspace(
                0,
                cycle_duration * simulation_conditions_list[i]["n_cycles_simultaneous"],
                simulation_conditions_list[i]["stimulation"] + 1,
            )[:-1]
        )
        model = set_model(model_path, stim_time)

        # Adjust n_shooting based on the stimulation time
        cycle_len = int(len(stim_time) / simulation_conditions_list[i]["n_cycles_simultaneous"])
        turn_number = simulation_conditions_list[i]["n_cycles_simultaneous"]
        nmpc = prepare_nmpc(
            model=model,
            cycle_duration=cycle_duration,
            cycle_len=cycle_len,
            n_cycles_to_advance=n_cycles_to_advance,
            n_cycles_simultaneous=simulation_conditions_list[i]["n_cycles_simultaneous"],
            minimize_force=simulation_conditions_list[i]["minimize_force"],
            minimize_fatigue=simulation_conditions_list[i]["minimize_fatigue"],
            turn_number=turn_number,
            pedal_config=pedal_config,
            external_force=resistive_torque,
        )
        nmpc.n_cycles_simultaneous = simulation_conditions_list[i]["n_cycles_simultaneous"]

        def update_functions(_nmpc: MultiCyclicNonlinearModelPredictiveControl, cycle_idx: int, _sol: Solution):
            return cycle_idx < n_cycles  # True if there are still some cycle to perform

        # Add the penalty cost function plot
        nmpc.add_plot_penalty(CostType.ALL)
        # Solve the optimal control problem
        sol = nmpc.solve_fes_nmpc(
            update_functions,
            solver=Solver.IPOPT(show_online_optim=False, _max_iter=100000, show_options=dict(show_bounds=True)),
            total_cycles=n_cycles,
            external_force=resistive_torque,
            cycle_solutions=MultiCyclicCycleSolutions.ALL_CYCLES,
            get_all_iterations=True,
            cyclic_options={"states": {}},
            max_consecutive_failing=3,
        )

        # Saving the data in a pickle file
        time = sol[0].stepwise_time(to_merge=[SolutionMerge.NODES]).T[0]
        states = sol[0].stepwise_states(to_merge=[SolutionMerge.NODES])
        controls = sol[0].stepwise_controls(to_merge=[SolutionMerge.NODES])
        stim_time = sol[0].ocp.nlp[0].model.muscles_dynamics_model[0].stim_time
        solving_time_per_ocp = [sol[1][i].solver_time_to_optimize for i in range(len(sol[1]))]
        objective_values_per_ocp = [float(sol[1][i].cost) for i in range(len(sol[1]))]
        number_of_turns_before_failing = len(sol[2])
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
        }

        pickle_file_name = simulation_conditions_list[i]["pickle_file_path"]
        with open(pickle_file_name, "wb") as file:
            pickle.dump(dictionary, file)


if __name__ == "__main__":
    main()
