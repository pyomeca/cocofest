"""
This example will perform a bayesian optimization on an optimal control program moving time horizon for a hand cycling
 motion driven by FES to find the best weight to increase number of cycling before failure.
"""

from pathlib import Path
from sys import platform

import pickle
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from casadi import MX, vertcat

from bioptim import (
    ObjectiveList,
    ObjectiveFcn,
    Node,
    CostType,
    MultiCyclicCycleSolutions,
    Solver,
    OdeSolver,
    PenaltyController,
)

import cycling_pulse_width_mhe as base


def minimize_muscle_fatigue(controller: PenaltyController, key) -> MX:
    """
    Minimize the muscle fatigue.

    Parameters
    ----------
    controller: PenaltyController
        The penalty node elements

    Returns
    -------
    The force scaling factor difference
    """
    muscle_name_list = controller.model.bio_model.muscle_names
    muscle_model = controller.model.muscles_dynamics_model
    muscle_index = muscle_name_list.index(key[2:])  # key is like "A_muscleName"
    muscle_a_scale_rest = muscle_model[muscle_index].a_scale
    muscle_fatigue = vertcat(1 - (controller.states[key].cx / muscle_a_scale_rest))
    return muscle_fatigue


def set_objective_functions(muscle_fatigue_key, cost_fun_weight, target):
    objective_functions = ObjectiveList()

    # Normalize weights to per-muscle list
    if isinstance(cost_fun_weight, (int, float)):
        weights = [float(cost_fun_weight)] * len(muscle_fatigue_key)
    else:
        if len(cost_fun_weight) == 1:
            weights = [float(cost_fun_weight[0])] * len(muscle_fatigue_key)
        elif len(cost_fun_weight) == len(muscle_fatigue_key):
            weights = list(map(float, cost_fun_weight))
        else:
            raise ValueError(
                f"cost_fun_weight must be length 1 or {len(muscle_fatigue_key)}, got {len(cost_fun_weight)}"
            )

    for key, w in zip(muscle_fatigue_key, weights):
        objective_functions.add(
            minimize_muscle_fatigue,
            custom_type=ObjectiveFcn.Lagrange,
            key=key,
            node=Node.ALL,
            weight=10000 * w,
            quadratic=True,
        )

    # Regulation function
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        key="q",
        index=2,
        node=Node.END,
        weight=1e-2,
        target=target,
        quadratic=True,
    )

    return objective_functions


def prepare_nmpc_bo(
    model,
    mhe_info: dict,
    cycling_info: dict,
    sim_cond: dict,
):
    # --- Unpack / window sizes --- #
    cycle_duration = mhe_info["cycle_duration"]
    cycle_len = mhe_info["cycle_len"]
    n_cycles_to_advance = mhe_info["n_cycles_to_advance"]
    n_cycles_simultaneous = mhe_info["n_cycles_simultaneous"]
    ode_solver = mhe_info["ode_solver"]
    use_sx = mhe_info["use_sx"]

    window_n_shooting = cycle_len * n_cycles_simultaneous
    window_cycle_duration = cycle_duration * n_cycles_simultaneous

    # --- External forces & numerical series --- #
    numerical_time_series, external_force_set = base.set_external_forces(
        n_shooting=window_n_shooting,
        external_force_dict=cycling_info["resistive_torque"],
        force_name="external_torque",
    )
    time_series2, _ = model.muscles_dynamics_model[0].get_numerical_data_time_series(
        window_n_shooting, window_cycle_duration
    )
    numerical_time_series.update(time_series2)

    # --- Dynamics & states --- #
    dynamics_options = base.set_dynamics_options(numerical_time_series=numerical_time_series,
                                                 ode_solver=OdeSolver.RK4(n_integration_steps=10))

    x_init = base.set_q_qdot_init(
        n_shooting=window_n_shooting,
        pedal_config=cycling_info["pedal_config"],
        turn_number=cycling_info["turn_number"],
        ode_solver=ode_solver,
        init_file_path=sim_cond.get("init_guess_file_path"),
    )
    x_bounds, x_init = base.set_x_bounds(
        model=model,
        x_init=x_init,
        n_shooting=window_n_shooting,
        ode_solver=ode_solver,
        init_file_path=sim_cond.get("init_guess_file_path"),
    )
    u_bounds, u_init, u_scaling = base.set_u_bounds_and_init(
        model, window_n_shooting, init_file_path=sim_cond.get("init_guess_file_path")
    )
    constraints = base.set_constraints(model)

    # --- Per-muscle fatigue objective --- #
    muscle_fatigue_keys = [f"A_{m.muscle_name}" for m in model.muscles_dynamics_model]
    objective_functions = set_objective_functions(
        muscle_fatigue_keys, sim_cond["cost_fun_weight"], target=x_init["q"].init[2][-1]
    )

    # --- Update model with forces / params --- #
    model = base.updating_model(model=model, external_force_set=external_force_set)

    return base.MyCyclicNMPC(
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
        n_threads=48,
        use_sx=use_sx,
    )


def run_optim_bo(
    mhe_info, cycling_info, sim_cond, model_path, save_sol=False, return_metric=False, return_solution=False
):
    # --- Build FES model --- #
    stim_time = list(
        np.linspace(
            0,
            mhe_info["cycle_duration"] * sim_cond["n_cycles_simultaneous"],
            sim_cond["stimulation"],
            endpoint=False,
        )
    )
    model = base.set_fes_model(model_path, stim_time)

    # --- Update MHE window sizes --- #
    mhe_info = dict(mhe_info)  # don’t mutate caller
    mhe_info["cycle_len"] = int(len(stim_time) / sim_cond["n_cycles_simultaneous"])
    mhe_info["n_cycles_simultaneous"] = sim_cond["n_cycles_simultaneous"]
    cycling_info = dict(cycling_info)
    cycling_info["turn_number"] = sim_cond["n_cycles_simultaneous"]  # 1 turn / cycle

    # --- Build NMPC with per-muscle objective --- #
    nmpc = prepare_nmpc_bo(model, mhe_info, cycling_info, sim_cond)

    # --- IPOPT settings --- #
    solver = Solver.IPOPT(show_online_optim=False, _max_iter=1000, show_options=dict(show_bounds=True))
    solver.set_warm_start_init_point("yes")
    solver.set_mu_init(1e-2)
    solver.set_tol(1e-6)
    solver.set_dual_inf_tol(1e-6)
    solver.set_constr_viol_tol(1e-6)
    solver.set_linear_solver("ma57" if platform == "linux" else "mumps")

    # Plot penalties
    nmpc.add_plot_penalty(CostType.ALL)

    # --- Solve window by window --- #
    def _update(_nmpc, cycle_idx, _sol):
        print("Optimized window n°", cycle_idx)
        return cycle_idx < mhe_info["n_cycles"]

    sol = nmpc.solve_fes_nmpc(
        _update,
        solver=solver,
        total_cycles=mhe_info["n_cycles"],
        external_force=cycling_info["resistive_torque"],
        cycle_solutions=MultiCyclicCycleSolutions.ALL_CYCLES,
        get_all_iterations=True,
        cyclic_options={"states": {}},
        max_consecutive_failing=1,
    )

    # Metric: number of turns before failing
    metric = len(sol[1]) - 1 + sim_cond["n_cycles_simultaneous"]

    if save_sol:
        Path("result/bo").mkdir(parents=True, exist_ok=True)
        base.save_sol_in_pkl(
            sol, sim_cond, is_initial_guess=False, torque=cycling_info["resistive_torque"]["torque"][-1]
        )

    if return_metric or return_solution:
        out = (metric,)
        if return_solution:
            out += (sol,)
        return out if len(out) > 1 else out[0]


def bayes_optimize_weights(
    mhe_info,
    cycling_info,
    model_path,
    stimulation_frequency=30,
    n_cycles_simultaneous_for_bo=3,
    n_calls=20,
    n_initial_points=6,
    random_state=42,
    weight_bounds_log=(1e-4, 1e2),
    use_simplex=False,
    init_guess_file_path=None,
    fixed_weights=None,
):
    """
    Bayesian optimize per-muscle weights on A_* (fatigue/activation states).
    Maximizes #turns before failing (we minimize its negative).

    If use_simplex=True, weights are softmax(x) and sum to 1 (relative importance).
    'fixed_weights' lets you pin certain muscles to a given value (e.g., {"delt_ant": 1.0}).
    NOTE: With use_simplex=True, fixed weights reduce the free mass (1 - sum_fixed).
    """

    fixed_weights = dict(fixed_weights or {})

    # Build a temp model to get muscle order
    stim_time_tmp = list(
        np.linspace(
            0,
            mhe_info["cycle_duration"] * n_cycles_simultaneous_for_bo,
            stimulation_frequency * n_cycles_simultaneous_for_bo,
            endpoint=False,
        )
    )
    tmp_model = base.set_fes_model(model_path, stim_time_tmp)
    muscle_names = [m.muscle_name for m in tmp_model.muscles_dynamics_model]

    # Validate fixed keys
    invalid = [k for k in fixed_weights.keys() if k not in muscle_names]
    if invalid:
        raise ValueError(f"fixed_weights contains unknown muscles: {invalid}. " f"Known muscles: {muscle_names}")

    # Split free vs fixed
    free_names = [n for n in muscle_names if n not in fixed_weights]
    fixed_sum = float(sum(fixed_weights.values()))

    if use_simplex and fixed_sum >= 1.0 - 1e-12:
        print("[BO] Warning: use_simplex=True but sum of fixed weights >= 1. " "Free weights will be (almost) zero.")

    log_dir = Path("result/bo")
    log_dir.mkdir(parents=True, exist_ok=True)
    bo_log = {}  # dict[index] = {'metric': float, muscle_1: w1, ...}

    def _save_logs_snapshot():
        """Persist logs after each iteration (both pickle + npz)."""
        with open(log_dir / "bo_iter_log.pkl", "wb") as f:
            pickle.dump(bo_log, f)
        np.savez_compressed(log_dir / "bo_iter_log.npz", bo_log=np.array([bo_log], dtype=object))
        if bo_log:
            idx = np.array(sorted(bo_log.keys()))
            metrics = np.array([float(bo_log[i]["metric"]) for i in idx], dtype=float)
            weights_mat = np.array([[float(bo_log[i][name]) for name in muscle_names] for i in idx], dtype=float)
            np.savez_compressed(
                log_dir / "bo_iter_arrays.npz",
                iteration=idx,
                muscle_names=np.array(muscle_names),
                weights=weights_mat,
                metric=metrics,
            )

    # --- Define search space only over FREE muscles --- #
    if use_simplex:
        space = [Real(-5.0, 5.0, name=f"z_{n}") for n in free_names]
        free_param_names = [f"z_{n}" for n in free_names]
    else:
        space = [
            Real(weight_bounds_log[0], weight_bounds_log[1], prior="log-uniform", name=f"w_{n}") for n in free_names
        ]
        free_param_names = [f"w_{n}" for n in free_names]

    # Template simulation conditions for each BO eval
    stim_count = stimulation_frequency * n_cycles_simultaneous_for_bo
    sim_cond_template = {
        "n_cycles_simultaneous": n_cycles_simultaneous_for_bo,
        "stimulation": stim_count,
        "cost_fun_weight": None,
        "pickle_file_path": Path("result/bo/bo_tmp.pkl"),
        "init_guess_file_path": init_guess_file_path,
    }

    def _compose_full_weights(free_vec):
        """
        Turn a vector 'free_vec' (ordered as free_names) into the full ordered
        weight list aligned with 'muscle_names', inserting fixed weights.
        Handles simplex case by allocating remaining mass to the free muscles.
        """
        if use_simplex:
            # softmax over free variables, then scale by remaining mass
            z = np.array(free_vec) - (np.max(free_vec) if len(free_vec) else 0.0)
            w_free = np.exp(z) if len(z) else np.array([])
            w_free = w_free / (np.sum(w_free) + 1e-12) if len(w_free) else w_free
            remaining = max(1e-12, 1.0 - fixed_sum)  # remaining simplex mass
            w_free = w_free * remaining
        else:
            w_free = np.array(free_vec, dtype=float)

        full = []
        j = 0
        for name in muscle_names:
            if name in fixed_weights:
                full.append(float(fixed_weights[name]))
            else:
                full.append(float(w_free[j]))
                j += 1
        return full

    _cache = {}

    @use_named_args(space)
    def objective(**kwargs):
        # Extract free vector in the consistent order
        x_free = [kwargs[k] for k in free_param_names] if free_param_names else []
        weights = _compose_full_weights(x_free)

        # Cache by full weight tuple to avoid duplicate work
        key = tuple([round(float(v), 8) for v in weights])
        if key in _cache:
            loss = _cache[key]
            i = len(bo_log)
            entry = {name: float(w) for name, w in zip(muscle_names, weights)}
            entry["metric"] = float(-loss) if np.isfinite(loss) else float("nan")
            bo_log[i] = entry
            _save_logs_snapshot()
            return loss

        sim_cond = dict(sim_cond_template)
        sim_cond["cost_fun_weight"] = weights

        try:
            print(f"[BO] Running NMPC with weights: {weights} for muscles: {muscle_names}")
            metric = run_optim_bo(
                mhe_info=mhe_info,
                cycling_info=cycling_info,
                sim_cond=sim_cond,
                model_path=model_path,
                save_sol=False,
                return_metric=True,
            )
            loss = -float(metric)  # maximize metric
        except Exception as e:
            print(f"[BO] NMPC failed for weights={weights} -> {e}")
            loss = 1e6
            metric = float("nan")

        _cache[key] = loss

        i = len(bo_log)
        entry = {name: float(w) for name, w in zip(muscle_names, weights)}
        entry["metric"] = float(metric) if np.isfinite(metric) else float("nan")
        bo_log[i] = entry
        _save_logs_snapshot()

        return loss

    print(f"[BO] Optimizing {len(free_names)} free weights (of {len(muscle_names)}) over {n_calls} evaluations...")
    res = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=n_calls,
        n_initial_points=min(n_initial_points, max(1, len(space))),  # guard when few frees
        acq_func="EI",
        random_state=random_state,
        verbose=True,
    )

    # Compose full best weights (include fixed)
    best_full_w = _compose_full_weights(res.x if len(space) else [])
    best_metric = -res.fun
    best_w_dict = {name: w for name, w in zip(muscle_names, best_full_w)}

    print("\n[BO] Done.")
    print("[BO] Best metric (turns before failing):", best_metric)
    for k, v in best_w_dict.items():
        print(f"   {k:>12s}: {v:.6g}")

    # Confirm best with a saved final run
    final_sim_cond = dict(sim_cond_template)
    final_sim_cond["cost_fun_weight"] = best_full_w
    final_sim_cond["pickle_file_path"] = Path("result/bo/bo_best.pkl")

    final_metric, final_sol = run_optim_bo(
        mhe_info=mhe_info,
        cycling_info=cycling_info,
        sim_cond=final_sim_cond,
        model_path=model_path,
        save_sol=True,
        return_metric=True,
        return_solution=True,
    )
    print(f"[BO] Confirmed best metric after final run: {final_metric}")
    return best_w_dict, best_metric, res


def main_bayes():
    model_path = "../../msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod"
    ode_solver = OdeSolver.COLLOCATION(polynomial_degree=3, method="radau")

    mhe_info = {
        "cycle_duration": 1,
        "n_cycles_to_advance": 1,
        "n_cycles": 3000,
        "ode_solver": ode_solver,
        "use_sx": False,
    }

    resistive_torque = -0.263
    cycling_info = {
        "pedal_config": {"x_center": 0.35, "y_center": 0.0, "radius": 0.1},
        "resistive_torque": {"Segment_application": "wheel", "torque": np.array([0, 0, resistive_torque])},
    }

    n_cycle_simultaneous = 2
    init_guess = f"result/initial_guess/{n_cycle_simultaneous}_initial_guess_collocation_3_radau.pkl"

    best_w, best_metric, _ = bayes_optimize_weights(
        mhe_info=mhe_info,
        cycling_info=cycling_info,
        model_path=model_path,
        stimulation_frequency=30,
        n_cycles_simultaneous_for_bo=n_cycle_simultaneous,
        n_calls=100,
        n_initial_points=6,
        random_state=42,
        weight_bounds_log=(1e-4, 1e2),
        use_simplex=False,
        init_guess_file_path=init_guess,
        fixed_weights={"Delt_ant": 1.0},
    )

    print("\nSuggested BO weights:")
    for k, v in best_w.items():
        print(f"{k}: {v:.6g}")


def read_pickle_file(file_type="pkl", is_numeric=False):
    bo = None
    if file_type == "pkl":
        bo = pickle.load(open(Path("result/bo/bo_iter_log.pkl"), "rb"))

    if file_type == "npz" and not is_numeric:
        bo = np.load("result/bo/bo_iter_log.npz", allow_pickle=True)["bo_log"].item()

    if file_type == "npz" and is_numeric:
        arr = np.load("result/bo/bo_iter_arrays.npz", allow_pickle=False)
        iteration = arr["iteration"]
        muscle_names = [s for s in arr["muscle_names"]]
        weights = arr["weights"]  # shape (n_iter, n_muscles)
        metric = arr["metric"]  # shape (n_iter,)
        bo = {}
        for k, iter_id in enumerate(iteration):
            entry = {"metric": float(metric[k])}
            entry.update({name: float(weights[k, j]) for j, name in enumerate(muscle_names)})
            bo[int(iter_id)] = entry
    print(bo)


if __name__ == "__main__":
    main_bayes()
    read_pickle_file(file_type="pkl", is_numeric=False)
