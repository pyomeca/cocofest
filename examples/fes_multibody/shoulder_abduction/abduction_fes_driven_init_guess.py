"""
This example will perform an optimal control program moving time horizon for an abduction/adduction motion driven by FES.
"""

import pickle
from sys import platform
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.ma.extras import average

from bioptim import (
    BiorbdModel,
    BoundsList,
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
    ParameterObjectiveList,
    SolutionMerge,
    Solution,
    Solver,
    ParameterList,
    Node,
    VariableScalingList,
    OptimalControlProgram,
)
from cocofest import (
    DingModelPulseWidthFrequencyWithFatigue,
    FesMskModel,
    OcpFesMsk,
)
from examples.fes_multibody.cycling.cost_functions import CustomCostFunctions


# --------------------#
#    OCP functions    #
# --------------------#
def prepare_ocp(
    model: BiorbdModel | FesMskModel,
    abduction_info: dict,
    simulation_conditions: dict,
    initial_guess_data,
):
    # --- Data from initial guess --- #
    q = initial_guess_data["q"]
    qdot = initial_guess_data["qdot"]
    tau = initial_guess_data["tau"]
    force = initial_guess_data["muscles_force"]

    # --- Initialize parameters from dictionaries --- #
    # --- Abduction info --- #
    abd_range_config = abduction_info["range_config"]
    external_force = abduction_info["resistive_torque"]
    n_shooting = simulation_conditions["stimulation"]

    ode_solver = OdeSolver.COLLOCATION(polynomial_degree=3, method="radau")

    # --- Set dynamics --- #
    # --- External force numerical time series --- #
    numerical_time_series, external_force_set = set_external_forces(
        n_shooting=n_shooting, external_force_dict=external_force, force_name="external_torque"
    ) # No external torque applied, except to provide FES supplementary assistance
    # --- Stimulation instant numerical time series --- #
    numerical_data_time_series, stim_idx_at_node_list = model.muscles_dynamics_model[0].get_numerical_data_time_series(
        n_shooting, 6
    )
    numerical_time_series.update(numerical_data_time_series)
    # --- Dynamics --- #
    dynamics = set_dynamics(model=model, numerical_time_series=numerical_time_series, ode_solver=ode_solver)

    # --- Set states --- #
    x_bounds, x_init = set_x_bounds(
        model=model,
        n_shooting=n_shooting,
        abduction_range=abd_range_config,
        ode_solver=ode_solver,
        q = q,
        qdot = qdot,
    )

    # --- Set controls --- #
    u_bounds, u_init, u_scaling = set_u_bounds_and_init(model, n_shooting, tau=tau)

    # --- Set objective --- #
    objective_functions = set_objective_functions(
        model=model,
        n_shooting=n_shooting,
        force=force,
    )

    # --- Set parameters (for minmax cost_function) --- #
    parameters = ParameterList(use_sx=False)
    parameters_bounds = BoundsList()
    parameters_init = InitialGuessList()
    parameters_objectives = ParameterObjectiveList()

     # --- Set constraints --- #
    constraints = ConstraintList()

    # --- Update model for resistive torque --- #
    model = updating_model(model=model, external_force_set=external_force_set, parameters=parameters)

    return OptimalControlProgram(
        bio_model=[model],
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=6,
        objective_functions=objective_functions,
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
        use_sx=False,
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
        phase=0,
        ode_solver=ode_solver,
    )
    return dynamics


def set_x_bounds(model, n_shooting: int, abduction_range: dict, ode_solver, q, qdot) -> tuple[BoundsList, InitialGuessList]:
    n_shooting = n_shooting * (ode_solver.polynomial_degree + 1)
    interpolation_type = InterpolationType.ALL_POINTS

    # --- Initialize default FES bounds and initial guess --- #
    x_bounds, x_init_fes = OcpFesMsk.set_x_bounds_fes(model)

    # --- Appending FES initial guesses to main list --- #
    x_init = InitialGuessList()
    for key in x_init_fes.keys():
        initial_guess = np.array([[x_init_fes[key].init[0][0]] * (n_shooting + 1)])
        x_init.add(key=key, initial_guess=initial_guess, phase=0, interpolation=interpolation_type)

    x_init.add(key="q", initial_guess=q, phase=0, interpolation=interpolation_type)
    x_init.add(key="qdot", initial_guess=qdot, phase=0, interpolation=interpolation_type)

    # --- Setting q bounds --- #
    slack = 0.349066  # 10° slack
    humerus_q_min = np.array([abduction_range["min"]] * (n_shooting + 1))
    humerus_q_max = np.array([abduction_range["max"]] * (n_shooting + 1))
    humerus_q_min -= slack
    humerus_q_max += slack

    # linear bounds around the turning point to help the solver with the change of direction
    humerus_q_max[:n_shooting // 2 - 100] = np.linspace(abduction_range["min"] + slack,
                                                        abduction_range["max"] + slack / 4, n_shooting // 2 - 100)
    humerus_q_min[:n_shooting // 2] = np.linspace(abduction_range["min"], abduction_range["max"], n_shooting // 2)
    # exponential bounds at the beginning to help the solver
    val = ((abduction_range["max"] - abduction_range["min"]) / 3) / 2 + abduction_range["min"]
    t = np.linspace(0, 1, n_shooting // 12)
    s = (np.exp(6 * t) - 1) / (np.exp(6) - 1)
    exp_bound = abduction_range["min"] + (val - abduction_range["min"]) * s
    humerus_q_min[:n_shooting // 12] = exp_bound

    humerus_q_max[n_shooting // 2 - 100:n_shooting // 2 + 100] = np.linspace(abduction_range["max"] + slack / 4,
                                                                             abduction_range["max"] + slack / 4, 200)

    humerus_q_max[n_shooting // 2 + 100:] = np.linspace(abduction_range["max"] + slack / 4,
                                                        abduction_range["min"] + slack, n_shooting // 2 - 99)
    humerus_q_min[n_shooting // 2:] = np.linspace(abduction_range["max"], abduction_range["min"], n_shooting // 2 + 1)
    humerus_q_min[0] = abduction_range["min"]
    humerus_q_max[0] = abduction_range["min"]

    humerus_q_max[-1] = abduction_range["min"] + slack / 5

    x_bounds.add(
        key="q", min_bound=np.array([humerus_q_min]), max_bound=np.array([humerus_q_max]), phase=0,
        interpolation=InterpolationType.ALL_POINTS
    )

    # --- Setting qdot bounds --- #
    humerus_qdot_min = np.zeros(n_shooting + 1, dtype=float)
    humerus_qdot_max = np.zeros(n_shooting + 1, dtype=float)

    humerus_qdot_min -= 4
    humerus_qdot_max += 4

    humerus_qdot_min[0] = 0
    humerus_qdot_max[0] = 0
    humerus_qdot_min[:n_shooting // 2 - 1] = 0
    humerus_qdot_max[n_shooting // 2 + 1:] = 0
    humerus_qdot_min[-1] = 0

    x_bounds.add(
        key="qdot",
        min_bound=np.array([humerus_qdot_min]),
        max_bound=np.array([humerus_qdot_max]),
        phase=0,
        interpolation=InterpolationType.ALL_POINTS,
    )

    return x_bounds, x_init


def set_u_bounds_and_init(bio_model, n_shooting, tau):
    # --- Initialize bounds and initial guess --- #
    u_init = InitialGuessList()  # Controls initial guess
    u_bounds = BoundsList()  # Controls bounds
    u_scaling = VariableScalingList()
    models = bio_model.muscles_dynamics_model

    for model in models:
        key = "last_pulse_width_" + str(model.muscle_name)

        # --- Set pulse width bounds for abduction adduction --- #
        min_pw_val = [model.pd0] * n_shooting
        max_pw_val = [0.0006] * n_shooting
        max_pw_val[int(n_shooting/2):] = [model.pd0] * int(n_shooting/2)

        u_bounds.add(
            key=key,
            min_bound=[min_pw_val],
            max_bound=[max_pw_val],
            interpolation=InterpolationType.EACH_FRAME,
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
    tau_min, tau_max, tau_init = -100.0, 100.0, 0.0
    control_tau_min = np.array([tau_min] * (n_shooting))
    control_tau_max = np.array([tau_max] * (n_shooting))
    control_tau_min[:(n_shooting // 2)] = 0
    control_tau_max[:(n_shooting // 2)] = 0
    control_tau_max[:(n_shooting // 24)] = tau_max
    control_tau_min[:(n_shooting // 24)] = tau_min

    control_tau_init = np.array([tau_init] * (n_shooting))
    control_tau_init[:(n_shooting // 2)] = 0

    u_bounds.add(
        key="tau", min_bound=np.array([control_tau_min]), max_bound=np.array([control_tau_max]), phase=0,
        interpolation=InterpolationType.ALL_POINTS
    )
    u_init.add(
        key="tau", initial_guess=tau, phase=0, interpolation=InterpolationType.ALL_POINTS
    )

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


def set_objective_functions(model, n_shooting, force):
    objective_functions = ObjectiveList()

    # --- Set main cost function --- #
    muscle_names = [model.muscles_dynamics_model[i].muscle_name for i in range(len(model.muscles_dynamics_model))]
    # force_states = ["F_" + muscle_name for muscle_name in muscle_names]
    control_keys = ["last_pulse_width_" + muscle_name for muscle_name in muscle_names]

    # for i in range(3):
    #     objective_functions.add(
    #         ObjectiveFcn.Lagrange.MINIMIZE_STATE,
    #         key=force_states[i],
    #         target=np.array([force[i]]),
    #         node=Node.ALL_SHOOTING,
    #         weight=1,
    #         quadratic=True,
    #     )

    # --- Set supplementary cost function for adduction tau --- #
    # objective_functions.add(
    #     ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
    #     key="tau",
    #     weight=1e3,
    #     quadratic=True,
    #     node=Node.ALL_SHOOTING,
    # )
    for i in range(3):
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key=control_keys[i],
            weight=1e6,
            quadratic=True,
            node=Node.ALL_SHOOTING,
        )

    for i in range(int(n_shooting/24)):
        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_CONTROL, key="tau", weight=1e6, node=i, quadratic=True,
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

    [parameter_dict[muscle_name].update({"a_scale": parameter_dict[muscle_name]["a_scale"] * 4}) for muscle_name in parameter_dict.keys()]
    [parameter_dict[muscle_name].update({"Fmax": parameter_dict[muscle_name]["Fmax"] * 4}) for muscle_name in
     parameter_dict.keys()]

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


def save_sol_in_pkl(sol, simulation_conditions, torque=None):
    solution = sol
    time = solution.stepwise_time(to_merge=[SolutionMerge.NODES]).T[0]
    states = solution.stepwise_states(to_merge=[SolutionMerge.NODES])
    controls = solution.stepwise_controls(to_merge=[SolutionMerge.NODES])
    stim_time = solution.ocp.nlp[0].model.muscles_dynamics_model[0].stim_time
    solving_time_per_ocp = solution.solver_time_to_optimize
    objective_values_per_ocp = solution.cost
    iter_per_ocp = solution.iterations
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
        "iter_per_ocp": iter_per_ocp,
        "total_n_shooting": solution.ocp.n_shooting,
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


def run_optim(abduction_info, simulation_conditions, model_path, initial_guess_path, save_sol):
    # --- Set FES model --- #
    stim_time = list(
        np.linspace(
            0,
            abduction_info["stimulation_duration"],
            simulation_conditions["stimulation"],
            endpoint=False,
        )
    )
    model = set_fes_model(model_path, stim_time)

    initial_guess_data = np.load(initial_guess_path, allow_pickle=True)
    ocp = prepare_ocp(
        model=model,
        abduction_info=abduction_info,
        simulation_conditions=simulation_conditions,
        initial_guess_data=initial_guess_data,
    )
    ocp.add_plot_penalty()

    # Set solver for the optimal control problem
    solver = Solver.IPOPT(show_online_optim=False, _max_iter=1000, show_options=dict(show_bounds=True))
    linear_solver = "ma57" if platform == "linux" else "mumps"
    solver.set_linear_solver(linear_solver)

    # Solve the optimal control problem
    sol = ocp.solve(solver=solver)

    result_show = True
    if result_show:
        # sol.animate(viewer="pyorerun")
        sol.graphs(show_bounds=False)
        import matplotlib.pyplot as plt

        time = sol.stepwise_time(to_merge=[SolutionMerge.NODES]).T[0]
        state_keys = ["q", "qdot"]
        for key in state_keys:
            state = sol.stepwise_states(to_merge=[SolutionMerge.NODES])[key][0]
            bounds = sol.ocp.nlp[0].x_bounds[key]
            min_bound = bounds.min[0]
            max_bound = bounds.max[0]
            plt.plot(time, state, label=key, color="black")
            plt.plot(time, min_bound, label="min_bound", linestyle="--", color="blue")
            plt.plot(time, max_bound, label="max_bound", linestyle="--", color="red")
            plt.legend()
            plt.show()

        control_keys = ["last_pulse_width_DeltoideusClavicle_A", "last_pulse_width_DeltoideusScapula_M",
                        "last_pulse_width_DeltoideusScapula_P", "tau"]
        for key in control_keys:
            control = sol.stepwise_controls(to_merge=[SolutionMerge.NODES])[key][0]
            bounds = sol.ocp.nlp[0].u_bounds[key]
            min_bound = bounds.min[0]
            max_bound = bounds.max[0]
            plt.plot(time[:-1][::4], control, label=key, color="black")
            plt.plot(time[:-1][::4], min_bound, label="min_bound", linestyle="--", color="blue")
            plt.plot(time[:-1][::4], max_bound, label="max_bound", linestyle="--", color="red")
            plt.legend()
            plt.show()

        sol.animate(viewer="pyorerun")

    # Saving the data in a pickle file
    if save_sol:
        save_sol_in_pkl(
            sol,
            simulation_conditions,
            torque=abduction_info["resistive_torque"]["torque"][-1],
        )


def main(
    stimulation_frequency, initial_guess_path, save
):
    # --- Simulation configuration --- #
    save_sol = save

    # --- Model choice --- #
    model_path = "../../msk_models/Seth/Seth_abd.bioMod"

    # --- Abduction parameters --- #
    resistive_torque = 0
    stimulation_duration = 6  # in seconds, total duration of the movement (abduction + adduction)
    abduction_info = {
        "stimulation_frequency": stimulation_frequency,  # in Hz
        "stimulation_duration": stimulation_duration,  # in s
        "range_config": {"min": np.deg2rad(30), "max": np.deg2rad(110)},  # in radian
        "resistive_torque": {"Segment_application": "humerus", "torque": np.array([0, 0, resistive_torque])},
    }

    # --- Build simulation list --- #
    stimulation = stimulation_frequency * stimulation_duration

    # --- Build the simulation conditions --- #
    simulation_conditions = {
            "stimulation": stimulation,
            "pickle_file_path": "results/abduction_fes_driven_init_guess_" + str(stimulation_frequency) + "Hz.pkl",
        }

    # --- Run the optimization --- #
    run_optim(
        abduction_info=abduction_info,
        simulation_conditions=simulation_conditions,
        model_path=model_path,
        initial_guess_path=initial_guess_path,
        save_sol=save_sol,
    )


if __name__ == "__main__":
    main(
        stimulation_frequency=50,
        initial_guess_path="results/abduction_motion_muscle_driven_solution.npz",
        save=True,
    )
