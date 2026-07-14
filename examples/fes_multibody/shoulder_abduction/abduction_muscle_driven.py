"""
This example will perform an optimal control program moving time horizon for an abduction/adduction motion driven by FES.
"""

import pickle
from sys import platform

import matplotlib.pyplot as plt
import numpy as np
import biorbd

from bioptim import (
    BoundsList,
    ConstraintList,
    DynamicsOptionsList,
    DynamicsOptions,
    ExternalForceSetTimeSeries,
    InitialGuessList,
    InterpolationType,
    MusclesBiorbdModel,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    PhaseDynamics,
    ParameterObjectiveList,
    SolutionMerge,
    Solver,
    ParameterList,
    Node,
    OptimalControlProgram,
)


# --------------------#
#    OCP functions    #
# --------------------#
def prepare_ocp(
    model: MusclesBiorbdModel,
    abduction_info: dict,
):
    # --- Initialize parameters from dictionaries --- #
    # --- Abduction info --- #
    abd_range_config = abduction_info["range_config"]
    n_shooting = 300
    final_time = 6
    ode_solver = OdeSolver.COLLOCATION(polynomial_degree=3, method="radau")

    # --- Set dynamics --- #
    dynamics_options = DynamicsOptionsList()
    dynamics_options.add(
        DynamicsOptions(
            expand_dynamics=True,
            phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
            numerical_data_timeseries=None,
            ode_solver=ode_solver,
        )
    )

    # --- Set states --- #
    x_bounds, x_init = set_x_bounds(
        n_shooting=n_shooting,
        abduction_range=abd_range_config,
        ode_solver=ode_solver,
    )

    # --- Set controls --- #
    u_bounds, u_init = set_u_bounds_and_init(n_shooting)

    # --- Set objective --- #
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
        key="muscles",
        weight=1e6,
        quadratic=True,
        node=Node.ALL_SHOOTING,
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
        key="tau",
        weight=1e-8,
        quadratic=True,
        node=Node.ALL_SHOOTING,
    )

    for i in range(int(n_shooting/24)):
        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_CONTROL, key="tau", weight=1e10, node=i, quadratic=True,
        )


    # --- Set parameters (for minmax cost_function) --- #
    parameters = ParameterList(use_sx=False)
    parameters_bounds = BoundsList()
    parameters_init = InitialGuessList()
    parameters_objectives = ParameterObjectiveList()

     # --- Set constraints --- #
    constraints = ConstraintList()

    return OptimalControlProgram(
        bio_model=[model],
        dynamics=dynamics_options,
        n_shooting=n_shooting,
        phase_time=final_time,
        objective_functions=objective_functions,
        constraints=constraints,
        x_bounds=x_bounds,
        x_init=x_init,
        u_bounds=u_bounds,
        u_init=u_init,
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


def set_x_bounds(n_shooting: int, abduction_range: dict, ode_solver) -> tuple[BoundsList, InitialGuessList]:
    n_shooting = n_shooting * (ode_solver.polynomial_degree + 1)
    interpolation_type = InterpolationType.ALL_POINTS

    # --- Initialize initial guess --- #
    x_init = InitialGuessList()

    # q from 0.523599 rad (30°) to 1.91986 rad (110°) in 3s than back to 0.523599 rad (30°) in 3s
    q_init = np.concatenate(
        (np.linspace(abduction_range["min"], abduction_range["max"], int(n_shooting / 2) + 1), np.linspace(abduction_range["max"], abduction_range["min"], int(n_shooting / 2) + 1)[1:]), axis=0
    )
    x_init.add(key="q", initial_guess=np.array([q_init]), phase=0, interpolation=interpolation_type)

    # qdot initial guess: linearly increasing from 0.0 rad/s to 4.0 rad/s at 1.5 s, then decreasing to -4 rad/s at 4.5 s, back to 0 rad/s at 6 s.
    qdot_init = np.concatenate(
        (np.linspace(0.0, 4.0, int(n_shooting / 4) + 1), np.linspace(4.0, -4.0,
            int(n_shooting / 2) + 1)[1:]), axis=0
    )
    qdot_init = np.concatenate((qdot_init, np.linspace(-4.0, 0.0, int(n_shooting / 4) + 1)[1:]), axis=0)
    x_init.add(key="qdot", initial_guess=np.array([qdot_init]), phase=0, interpolation=interpolation_type)

    # --- Initialize bounds --- #
    x_bounds = BoundsList()

    # --- Setting q bounds --- #
    slack = 0.349066 # 10° slack
    humerus_q_min = np.array([abduction_range["min"]] * (n_shooting + 1))
    humerus_q_max = np.array([abduction_range["max"]] * (n_shooting + 1))
    humerus_q_min -= slack
    humerus_q_max += slack

    # linear bounds around the turning point to help the solver with the change of direction
    humerus_q_max[:n_shooting // 2 - 100] = np.linspace(abduction_range["min"] + slack, abduction_range["max"] + slack/4, n_shooting // 2 - 100)
    humerus_q_min[:n_shooting // 2] = np.linspace(abduction_range["min"], abduction_range["max"], n_shooting // 2)
    # exponential bounds at the beginning to help the solver
    val = ((abduction_range["max"] - abduction_range["min"]) / 3) / 2 + abduction_range["min"]
    t = np.linspace(0, 1, n_shooting // 12)
    s = (np.exp(6 * t) - 1) / (np.exp(6) - 1)
    exp_bound = abduction_range["min"] + (val - abduction_range["min"]) * s
    humerus_q_min[:n_shooting // 12] = exp_bound

    humerus_q_max[n_shooting // 2 - 100:n_shooting // 2 + 100] = np.linspace(abduction_range["max"] + slack/4, abduction_range["max"] + slack/4, 200)

    humerus_q_max[n_shooting // 2 + 100:] = np.linspace(abduction_range["max"] + slack/4, abduction_range["min"] + slack, n_shooting // 2 - 99)
    humerus_q_min[n_shooting // 2:] = np.linspace(abduction_range["max"], abduction_range["min"], n_shooting // 2 + 1)
    humerus_q_min[0] = abduction_range["min"]
    humerus_q_max[0] = abduction_range["min"]

    humerus_q_max[-1] = abduction_range["min"] + slack/5

    x_bounds.add(
        key="q", min_bound=np.array([humerus_q_min]), max_bound=np.array([humerus_q_max]), phase=0, interpolation=InterpolationType.ALL_POINTS
    )

    # --- Setting qdot bounds --- #
    humerus_qdot_min = np.zeros(n_shooting + 1, dtype=float)
    humerus_qdot_max = np.zeros(n_shooting + 1, dtype=float)

    humerus_qdot_min -= 4
    humerus_qdot_max += 4

    humerus_qdot_min[0] = 0
    humerus_qdot_max[0] = 0
    humerus_qdot_min[:n_shooting // 2-1] = 0
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


def set_u_bounds_and_init(n_shooting):
    # --- Initialize bounds and initial guess --- #
    u_init = InitialGuessList()
    u_bounds = BoundsList()

    muscle_min, muscle_max, muscle_init = 0.0, 1.0, 0.5
    tau_min, tau_max, tau_init = -100.0, 100.0, 0.0

    control_muscle_min = np.full((3, n_shooting), muscle_min)
    control_muscle_max = np.full((3, n_shooting), muscle_max)
    control_muscle_init = np.full((3, n_shooting), muscle_init)
    control_muscle_min[:, -(n_shooting // 2):] = 0
    control_muscle_max[:, -(n_shooting // 2):] = 0
    control_muscle_init[:, -(n_shooting // 2):] = 0

    control_tau_min = np.array([tau_min] * (n_shooting))
    control_tau_max = np.array([tau_max] * (n_shooting))
    control_tau_min[:(n_shooting // 2)] = 0
    control_tau_max[:(n_shooting // 2)] = 0
    control_tau_max[:(n_shooting // 24)] = tau_max
    control_tau_min[:(n_shooting // 24)] = tau_min

    control_tau_init = np.array([tau_init] * (n_shooting))
    control_tau_init[:(n_shooting // 2)] = 0

    u_bounds.add(
        key="muscles", min_bound=control_muscle_min, max_bound=control_muscle_max, phase=0,
        interpolation=InterpolationType.ALL_POINTS
    )
    u_bounds.add(
        key="tau", min_bound=np.array([control_tau_min]), max_bound=np.array([control_tau_max]), phase=0,
        interpolation=InterpolationType.ALL_POINTS
    )
    u_init.add(
        key="muscles", initial_guess=control_muscle_init, phase=0, interpolation=InterpolationType.ALL_POINTS
    )
    u_init.add(
        key="tau", initial_guess=np.array([control_tau_init]), phase=0, interpolation=InterpolationType.ALL_POINTS
    )

    return (
        u_bounds,
        u_init,
    )


def save_sol_in_pkl(sol):
    solution = sol
    time = solution.stepwise_time(to_merge=[SolutionMerge.NODES]).T[0]
    states = solution.stepwise_states(to_merge=[SolutionMerge.NODES])
    controls = solution.stepwise_controls(to_merge=[SolutionMerge.NODES])
    solving_time_per_ocp = solution.solver_time_to_optimize
    objective_values_per_ocp = solution.cost
    iter_per_ocp = solution.iterations

    # --- Convert all data into lists for compatibility across Python versions --- #
    time = time.tolist()
    states_dict = {key: value.tolist() for key, value in states.items()}
    controls_dict = {key: value.tolist() for key, value in controls.items()}
    forces_dict = {key: value.tolist() for key, value in controls.items()}

    bio_path = sol.ocp.nlp[0].model.path  # path to the .bioMod
    m = biorbd.Model(bio_path)
    f_iso_max = np.array(
        [m.muscle(i).characteristics().forceIsoMax() for i in range(m.nbMuscles())],
        dtype=float,
    )

    for key in controls.keys():
        if key == "muscles":
            for i in range(3):
                forces_dict[key][i] = (np.array(forces_dict[key][i]) * f_iso_max[i]).tolist()

    dictionary = {
        "time": time,
        "solving_time_per_ocp": solving_time_per_ocp,
        "objective_values_per_ocp": objective_values_per_ocp,
        "iter_per_ocp": iter_per_ocp,
        "total_n_shooting": solution.ocp.n_shooting,
        "polynomial_order": solution.ocp.nlp[0].dynamics_type.ode_solver.polynomial_degree,
    }

    for key in states.keys():
        dictionary[key] = states_dict[key]
    for key in controls.keys():
        dictionary[key] = controls_dict[key]
    for key in forces_dict.keys():
        dictionary[key + "_force"] = forces_dict[key]

    pickle_file_name = "results/abduction_motion_muscle_driven_solution.pkl"
    with open(pickle_file_name, "wb") as file:
        pickle.dump(dictionary, file)

    np.savez_compressed(str(pickle_file_name)[:-4] + ".npz", **dictionary)
    print(pickle_file_name)


def plot_results(sol):
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

    control_keys = ["muscles", "tau"]
    for key in control_keys:
        if key == "muscles":
            for i in range(3):
                control = sol.stepwise_controls(to_merge=[SolutionMerge.NODES])[key][i]
                bounds = sol.ocp.nlp[0].u_bounds[key]
                min_bound = bounds.min[i]
                max_bound = bounds.max[i]
                plt.plot(time[:-1][::4], control, label=key + "_" + str(i), color="black")
                plt.plot(time[:-1][::4], min_bound, label="min_bound", linestyle="--", color="blue")
                plt.plot(time[:-1][::4], max_bound, label="max_bound", linestyle="--", color="red")
                plt.legend()
                plt.show()
        else:
            control = sol.stepwise_controls(to_merge=[SolutionMerge.NODES])[key][0]
            bounds = sol.ocp.nlp[0].u_bounds[key]
            min_bound = bounds.min[0]
            max_bound = bounds.max[0]
            plt.plot(time[:-1][::4], control, label=key, color="black")
            plt.plot(time[:-1][::4], min_bound, label="min_bound", linestyle="--", color="blue")
            plt.plot(time[:-1][::4], max_bound, label="max_bound", linestyle="--", color="red")
            plt.legend()
            plt.show()


def run_optim(abduction_info, model_path, save_sol):

    # --- Prepare the optimal control problem --- #
    model = MusclesBiorbdModel(model_path, with_residual_torque=True)
    ocp = prepare_ocp(
        model=model,
        abduction_info=abduction_info,
    )

    # Set solver for the optimal control problem
    solver = Solver.IPOPT(show_online_optim=False, _max_iter=5000, show_options=dict(show_bounds=True))
    linear_solver = "ma57" if platform == "linux" else "mumps"
    solver.set_linear_solver(linear_solver)

    # Solve the optimal control problem
    sol = ocp.solve(solver=solver)

    result_show = True
    if result_show:
        plot_results(sol)
        # sol.animate(viewer="pyorerun")
        # sol.graphs(show_bounds=False)

    # Saving the data in a pickle file
    if save_sol:
        save_sol_in_pkl(
            sol,
        )


def main(save=False):
    # --- Simulation configuration --- #
    save_sol = save

    # --- Model choice --- #
    model_path = "../../msk_models/Seth/Seth_abd.bioMod"

    # --- Abduction parameters --- #
    abduction_info = {
        "range_config": {"min": np.deg2rad(30), "max": np.deg2rad(110)},  # in radian
    }

    # --- Run the optimization --- #
    run_optim(
        abduction_info=abduction_info,
        model_path=model_path,
        save_sol=save_sol,
    )


if __name__ == "__main__":
    main(save=True)
