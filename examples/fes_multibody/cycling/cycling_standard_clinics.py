"""
This example demonstrates how long hand cycling could be maintained at 30 Hz with standard clinical settings
(fixed pulse width and defined stimulation angles). The crank torque is optimized to achieve the target cadence
(60 rpm). Then, after each cycle, muscle fatigue is updated based on the activation history and tested with an
optimal control problem. If an optimal solution is found, the standard clinical settings simulation continues.
If no optimal solution is found, the simulation stops and muscle failure is considered reached.

Important note about the feasibility logic
------------------------------------------
The OCP used to assess feasibility intentionally freezes some FES states using prev_state_result[key][0][0].
This is kept unchanged because the goal is to test whether the clinical condition remains feasible before moving
one window forward.
"""

from sys import platform
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
    Solver,
    CostType,
)

from cycling.cycling_pulse_width_mhe import (
    set_fes_model,
    set_external_forces,
    set_dynamics,
    set_q_qdot_init,
    set_x_bounds,
    updating_model,
    set_u_bounds_and_init,
)
from cycling.cost_functions import CustomCostFunctions


# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------
TARGET_RPM = 60
PHASE_TIME = 60 / TARGET_RPM  # 1 second per revolution at 60 rpm
N_SHOOTING = 30  # 30 Hz for a 1-second cycle
N_THREADS = 48

MODEL_PATH = "../../msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod"
RESULT_FILE_NAME = "temp/standard_and_fes_ocp_cycles.pkl"

MUSCLES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")
STATE_VARS = ("A", "F", "Km", "Tau1")
CONTROL_VARS = ("last_pulse_width",)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def target_crank_velocity(turn_number: int, phase_time: float) -> float:
    """Return the target crank angular velocity in rad/s."""
    return -2 * np.pi * turn_number / phase_time

def compute_net_work(time: np.ndarray, torque: np.ndarray) -> float:
    """
    Compute net mechanical work at 60 rpm.

    Parameters
    ----------
    """
    time = np.asarray(time, dtype=float).squeeze()
    torque = np.asarray(torque, dtype=float).squeeze()

    if time.ndim != 1:
        raise ValueError("time must be a 1D array.")
    if torque.ndim != 1:
        raise ValueError("torque must be a 1D array.")
    if time.size < 2:
        raise ValueError("time must contain at least two points.")

    omega = 2 * np.pi * TARGET_RPM / 60

    if torque.size == time.size:
        return float(np.trapezoid(torque * omega, time))

    if torque.size == time.size - 1:
        dt = np.diff(time)
        return float(np.sum(torque * omega * dt))

    if (time.size - 1) % torque.size == 0:
        step = (time.size - 1) // torque.size
        time_boundaries = time[::step]

        if time_boundaries.size != torque.size + 1:
            raise ValueError(
                f"Could not build interval boundaries. "
                f"len(time_boundaries)={time_boundaries.size}, expected {torque.size + 1}."
            )

        dt = np.diff(time_boundaries)
        return float(np.sum(torque * omega * dt))

    raise ValueError(
        f"Cannot match time and torque automatically. "
        f"Got len(time)={time.size}, len(torque)={torque.size}."
    )

def compute_constant_torque_work(time: np.ndarray, constant_torque: float) -> float:
    """Compute net work for a constant torque over the provided time vector."""
    time = np.asarray(time, dtype=float).squeeze()
    torque = np.full(time.shape, constant_torque, dtype=float)
    return compute_net_work(time=time, torque=torque)

def prepare_solver(max_iter: int = 10):
    """Prepare the IPOPT solver."""
    solver = Solver.IPOPT(
        show_online_optim=False,
        _max_iter=max_iter,
        show_options=dict(show_bounds=True),
    )

    linear_solver = "ma57" if platform == "linux" else "mumps"
    solver.set_linear_solver(linear_solver)

    return solver


# -----------------------------------------------------------------------------
# Pulse-width settings for the clinical condition
# -----------------------------------------------------------------------------
def is_angle_in_range(angle_vector: np.ndarray, start: float, end: float) -> np.ndarray:
    """
    Return a Boolean vector indicating whether each angle is inside a stimulation range.

    Handles both normal ranges, e.g. 20 deg to 180 deg,
    and wrap-around ranges, e.g. 220 deg to 10 deg.
    """
    if start <= end:
        return (angle_vector >= start) & (angle_vector <= end)

    return (angle_vector >= start) | (angle_vector <= end)

def set_pw_dictionary(muscle_models, n_shooting: int, active_pw: float = 0.0003):
    """Create the fixed pulse-width dictionary for the standard clinical stimulation pattern."""
    angle_vector = np.linspace(0, 360, n_shooting, endpoint=False)

    stimulation_ranges = {
        "Delt_ant": (20, 180),
        "Delt_post": (220, 10),
        "Biceps": (220, 10),
        "Triceps": (20, 180),
    }

    pulse_width_dictionary = {}

    for muscle in muscle_models:
        muscle_name = muscle.muscle_name

        if muscle_name not in stimulation_ranges:
            raise ValueError(f"No stimulation range is defined for muscle: {muscle_name}")

        start_angle, end_angle = stimulation_ranges[muscle_name]
        active_zone = is_angle_in_range(angle_vector, start_angle, end_angle)

        pulse_width_dictionary[muscle_name] = np.where(
            active_zone,
            active_pw,
            muscle.pd0,
        )

    return pulse_width_dictionary

def set_standard_u_bounds_and_init(bio_model, n_shooting: int):
    """Set fixed clinical pulse-width controls and optimized residual crank torque."""
    u_bounds = BoundsList()
    u_init = InitialGuessList()
    u_scaling = VariableScalingList()

    muscle_models = bio_model.muscles_dynamics_model
    pulse_width_dictionary = set_pw_dictionary(muscle_models=muscle_models, n_shooting=n_shooting)

    # Fixed stimulation controls
    for muscle_model in muscle_models:
        key = f"last_pulse_width_{muscle_model.muscle_name}"
        pulse_width = np.array([pulse_width_dictionary[muscle_model.muscle_name]])

        u_init.add(
            key=key,
            initial_guess=pulse_width,
            phase=0,
            interpolation=InterpolationType.EACH_FRAME,
        )
        u_bounds.add(
            key=key,
            min_bound=pulse_width,
            max_bound=pulse_width,
            phase=0,
            interpolation=InterpolationType.EACH_FRAME,
        )
        u_scaling.add(key=key, scaling=[1 / 400])

    # Residual crank torque control
    u_init.add(
        key="tau",
        initial_guess=np.zeros((3, n_shooting)),
        phase=0,
        interpolation=InterpolationType.EACH_FRAME,
    )
    u_bounds.add(
        key="tau",
        min_bound=np.array([0, 0, -10]),
        max_bound=np.array([0, 0, 10]),
        phase=0,
    )
    u_scaling.add(key="tau", scaling=[1, 1, 1])

    return u_bounds, u_init, u_scaling


# -----------------------------------------------------------------------------
# Shared OCP preparation helpers
# -----------------------------------------------------------------------------
def prepare_common_dynamics(model, n_shooting: int, phase_time: float, ode_solver, external_force):
    """Prepare external force time series and FES numerical data time series."""
    numerical_time_series, external_force_set = set_external_forces(
        n_shooting=n_shooting,
        external_force_dict=external_force,
        force_name="external_torque",
    )

    numerical_data_time_series, _ = model.muscles_dynamics_model[0].get_numerical_data_time_series(
        n_shooting,
        phase_time,
    )
    numerical_time_series.update(numerical_data_time_series)

    dynamics = set_dynamics(
        model=model,
        numerical_time_series=numerical_time_series,
        ode_solver=ode_solver,
    )

    return dynamics, external_force_set

def prepare_empty_parameters(use_sx: bool):
    """Prepare empty parameter containers."""
    return (
        ParameterList(use_sx=use_sx),
        BoundsList(),
        InitialGuessList(),
        ParameterObjectiveList(),
    )

def prepare_wheel_center_constraints(model):
    """Constrain the wheel center to remain fixed at the beginning of the cycle."""
    constraints = ConstraintList()

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

    return constraints

def prepare_standard_objective(turn_number: int, phase_time: float):
    """Objective for the standard condition: match the target crank velocity."""
    objective_functions = ObjectiveList()

    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        key="qdot",
        index=2,
        node=Node.ALL,
        weight=1e6,
        target=target_crank_velocity(turn_number=turn_number, phase_time=phase_time),
        quadratic=True,
    )

    return objective_functions

def prepare_fes_objective():
    """Objective for the FES feasibility OCP."""
    objective_functions = ObjectiveList()

    objective_functions.add(
        CustomCostFunctions.minimize_root_mean_square_activation,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        weight=1e6,
        quadratic=False,
    )

    return objective_functions

def build_ocp(
    model,
    dynamics,
    n_shooting: int,
    phase_time: float,
    x_bounds,
    u_bounds,
    x_init,
    u_init,
    objective_functions,
    constraints,
    parameters,
    parameters_init,
    parameters_bounds,
    parameters_objectives,
    u_scaling,
    use_sx: bool,
):
    """Build a one-phase OptimalControlProgram."""
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
        n_threads=N_THREADS,
        use_sx=use_sx,
    )


# -----------------------------------------------------------------------------
# OCP preparation functions
# -----------------------------------------------------------------------------
def prepare_standard_fes_cycling(optim_info, cycling_info, model, previous_problem=None, previous_sol=None):
    """Prepare the standard clinical FES cycling problem."""
    phase_time = optim_info["phase_time"]
    n_shooting = optim_info["n_shooting"]
    ode_solver = optim_info["ode_solver"]
    use_sx = optim_info["use_sx"]

    turn_number = cycling_info["turn_number"]
    pedal_config = cycling_info["pedal_config"]
    external_force = cycling_info["resistive_torque"]

    dynamics, external_force_set = prepare_common_dynamics(
        model=model,
        n_shooting=n_shooting,
        phase_time=phase_time,
        ode_solver=ode_solver,
        external_force=external_force,
    )

    # States
    if previous_problem is None:
        x_init = set_q_qdot_init(
            n_shooting=n_shooting,
            pedal_config=pedal_config,
            turn_number=turn_number,
            ode_solver=ode_solver,
            init_file_path=None,
        )
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
                x_bounds[key].min[:, 0] = prev_state_result[key][:, -1]
                x_bounds[key].max[:, 0] = prev_state_result[key][:, -1]

    # Controls
    if previous_problem is None:
        u_bounds, u_init, u_scaling = set_standard_u_bounds_and_init(
            bio_model=model,
            n_shooting=n_shooting,
        )
    else:
        u_bounds = previous_problem.nlp[0].u_bounds
        u_init = previous_problem.nlp[0].u_init
        u_scaling = previous_problem.nlp[0].u_scaling

        previous_controls = previous_sol.decision_controls(to_merge=SolutionMerge.NODES)
        u_init["tau"].init[:, :] = previous_controls["tau"]

    objective_functions = prepare_standard_objective(
        turn_number=turn_number,
        phase_time=phase_time,
    )

    parameters, parameters_bounds, parameters_init, parameters_objectives = prepare_empty_parameters(use_sx=use_sx)
    constraints = prepare_wheel_center_constraints(model=model)

    # Activate residual crank torque for the standard condition
    model.activate_residual_torque = True
    model = updating_model(model=model, external_force_set=external_force_set, parameters=parameters)

    return build_ocp(
        model=model,
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
        parameters_init=parameters_init,
        parameters_bounds=parameters_bounds,
        parameters_objectives=parameters_objectives,
        u_scaling=u_scaling,
        use_sx=use_sx,
    )

def prepare_ocp_fes_cycling(optim_info, cycling_info, model, previous_problem=None, previous_sol=None):
    """Prepare the FES feasibility OCP."""
    phase_time = optim_info["phase_time"]
    n_shooting = optim_info["n_shooting"]
    ode_solver = optim_info["ode_solver"]
    use_sx = optim_info["use_sx"]

    turn_number = cycling_info["turn_number"]
    pedal_config = cycling_info["pedal_config"]
    external_force = cycling_info["resistive_torque"]

    dynamics, external_force_set = prepare_common_dynamics(
        model=model,
        n_shooting=n_shooting,
        phase_time=phase_time,
        ode_solver=ode_solver,
        external_force=external_force,
    )

    # States
    x_init = set_q_qdot_init(
        n_shooting=n_shooting,
        pedal_config=pedal_config,
        turn_number=turn_number,
        ode_solver=ode_solver,
        init_file_path=None,
    )
    x_bounds, x_init = set_x_bounds(
        model=model,
        x_init=x_init,
        n_shooting=n_shooting,
        ode_solver=ode_solver,
        init_file_path=None,
    )

    x_bounds["qdot"].min[2, 0] = -8.390
    x_bounds["qdot"].max[2, 0] = -8.390

    init_shape = x_init["qdot"].init[-1].shape[0]
    x_init["qdot"].init[-1] = np.array([-2 * np.pi] * init_shape)

    if previous_sol is not None:
        prev_state_result = previous_sol.decision_states(to_merge=SolutionMerge.NODES)

        # Kept intentionally unchanged.
        # The feasibility test uses the first value of the previous solution window.
        for muscle in MUSCLES:
            for state_var in ("A", "Km", "Tau1"):
                key = f"{state_var}_{muscle}"
                x_bounds[key].min[0, 0] = prev_state_result[key][0][0]
                x_bounds[key].max[0, 0] = prev_state_result[key][0][0]

    # Controls
    if previous_sol is None or previous_problem is None:
        u_bounds, u_init, u_scaling = set_u_bounds_and_init(
            bio_model=model,
            n_shooting=n_shooting,
            init_file_path=None,
        )
    else:
        u_bounds = previous_problem.nlp[0].u_bounds
        u_init = InitialGuessList()
        u_scaling = previous_problem.nlp[0].u_scaling

        previous_controls = previous_sol.decision_controls(to_merge=SolutionMerge.NODES)
        for muscle_model in model.muscles_dynamics_model:
            key = f"last_pulse_width_{muscle_model.muscle_name}"
            u_init.add(
                key=key,
                initial_guess=previous_controls[key],
                phase=0,
                interpolation=InterpolationType.EACH_FRAME,
            )

    objective_functions = prepare_fes_objective()

    parameters, parameters_bounds, parameters_init, parameters_objectives = prepare_empty_parameters(use_sx=use_sx)
    constraints = prepare_wheel_center_constraints(model=model)

    # No residual crank torque in the FES feasibility OCP
    model.activate_residual_torque = False
    model = updating_model(model=model, external_force_set=external_force_set, parameters=parameters)

    return build_ocp(
        model=model,
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
        parameters_init=parameters_init,
        parameters_bounds=parameters_bounds,
        parameters_objectives=parameters_objectives,
        u_scaling=u_scaling,
        use_sx=use_sx,
    )

# -----------------------------------------------------------------------------
# Solution extraction and saving
# -----------------------------------------------------------------------------
def solution_to_dict(solution, muscles=MUSCLES):
    """Extract the relevant states and controls from a Bioptim solution."""
    time = solution.stepwise_time(to_merge=[SolutionMerge.NODES]).T[0]
    states = solution.stepwise_states(to_merge=[SolutionMerge.NODES])
    controls = solution.stepwise_controls(to_merge=[SolutionMerge.NODES])

    out = {"time": time}

    for muscle in muscles:
        for state_var in STATE_VARS:
            key = f"{state_var}_{muscle}"
            out[key] = states[key]

    for muscle in muscles:
        for control_var in CONTROL_VARS:
            key = f"{control_var}_{muscle}"
            if key in controls:
                out[key] = controls[key]

    if "tau" in controls:
        out["tau"] = controls["tau"][2, :]

    out["q"] = states["q"]
    out["qdot"] = states["qdot"]

    return out

def safe_solution_to_dict(solution):
    """
    Try to extract a solution dictionary.

    This prevents the saving step from crashing if a failed OCP solution cannot be fully extracted.
    """
    if solution is None:
        return None

    try:
        return solution_to_dict(solution)
    except Exception as error:
        return {"extraction_error": repr(error)}

def solution_status(solution):
    """Return the solver status if available."""
    if solution is None:
        return None

    return int(solution.status)

def compute_cycle_work(solution_dict, cycling_info):
    """
    Compute work-related information for one cycle.

    - residual_tau_net_work_J is computed from the optimized residual crank torque when available.
    - external_resistive_work_J is computed from the constant external resistive torque.
    """
    if solution_dict is None or "extraction_error" in solution_dict:
        return {
            "residual_tau_net_work_J": None,
            "external_resistive_work_J": None,
        }

    time = solution_dict["time"]
    external_torque = float(cycling_info["resistive_torque"]["torque"][2])

    work = {
        "external_resistive_work_J": compute_constant_torque_work(
            time=time,
            constant_torque=external_torque,
        )
    }

    if "tau" in solution_dict:
        work["residual_tau_net_work_J"] = compute_net_work(
            time=time,
            torque=solution_dict["tau"],
        )
    else:
        work["residual_tau_net_work_J"] = None

    return work

def make_cycle_record(cycle_index, standard_sol, fes_ocp_sol, cycling_info):
    """Create one common record containing standard and FES-OCP information for one cycle."""
    standard_data = safe_solution_to_dict(standard_sol)
    fes_ocp_data = safe_solution_to_dict(fes_ocp_sol)

    return {
        "cycle_index": cycle_index,
        "standard": {
            "status": solution_status(standard_sol),
            "converged": solution_status(standard_sol) == 0,
            "data": standard_data,
            "work": compute_cycle_work(standard_data, cycling_info),
        },
        "fes_ocp": {
            "status": solution_status(fes_ocp_sol),
            "converged": solution_status(fes_ocp_sol) == 0,
            "data": fes_ocp_data,
            "work": compute_cycle_work(fes_ocp_data, cycling_info),
        },
    }

def save_common_results(
    file_name,
    cycle_records,
    cycle_to_failure,
    optim_info,
    cycling_info,
    model_path,
    failure_reason,
):
    """
    Save all standard and FES-OCP information in one common pickle file.
    """
    results = {
        "metadata": {
            "target_rpm": TARGET_RPM,
            "model_path": model_path,
            "optim_info": {
                "phase_time": optim_info["phase_time"],
                "n_shooting": optim_info["n_shooting"],
                "ode_solver": str(optim_info["ode_solver"]),
                "use_sx": optim_info["use_sx"],
            },
            "cycling_info": cycling_info,
        },
        "cycle_to_failure": cycle_to_failure,
        "failure_reason": failure_reason,
        "cycles": cycle_records,
    }

    with open(file_name, "wb") as file:
        pickle.dump(results, file)

    print(f"Results saved in {file_name}")

def load_common_results(file_name=RESULT_FILE_NAME):
    """Load the common result file."""
    with open(file_name, "rb") as file:
        return pickle.load(file)


# -----------------------------------------------------------------------------
# Main simulation
# -----------------------------------------------------------------------------
def main():
    optim_info = {
        "phase_time": PHASE_TIME,
        "n_shooting": N_SHOOTING,
        "ode_solver": OdeSolver.COLLOCATION(),
        "use_sx": False,
    }

    resistive_torque = -0.20  # N.m
    cycling_info = {
        "turn_number": 1,
        "pedal_config": {
            "x_center": 0.35,
            "y_center": 0.0,
            "radius": 0.1,
        },
        "resistive_torque": {
            "Segment_application": "wheel",
            "torque": np.array([0, 0, resistive_torque]),
        },
    }

    stim_time = list(
        np.linspace(
            0,
            optim_info["phase_time"],
            optim_info["n_shooting"],
            endpoint=False,
        )
    )

    model = set_fes_model(
        model_path=MODEL_PATH,
        stim_time=stim_time,
    )

    solver = prepare_solver()

    standard_cycling_problem = None
    standard_cycling_sol = None
    fes_ocp_problem = None

    cycle_to_failure = 0
    attempted_cycle = 0
    cycle_records = []
    failure_reason = "No failure detected."

    while True:
        attempted_cycle += 1

        standard_cycling_problem = prepare_standard_fes_cycling(
            optim_info=optim_info,
            cycling_info=cycling_info,
            model=model,
            previous_problem=standard_cycling_problem,
            previous_sol=standard_cycling_sol,
        )
        standard_cycling_sol = standard_cycling_problem.solve(solver=solver)

        if standard_cycling_sol.status != 0:
            cycle_record = make_cycle_record(
                cycle_index=attempted_cycle,
                standard_sol=standard_cycling_sol,
                fes_ocp_sol=None,
                cycling_info=cycling_info,
            )
            cycle_records.append(cycle_record)
            failure_reason = "Standard clinical cycling problem failed."
            print(failure_reason)
            break

        fes_ocp_problem = prepare_ocp_fes_cycling(
            optim_info=optim_info,
            cycling_info=cycling_info,
            model=model,
            previous_problem=fes_ocp_problem,
            previous_sol=standard_cycling_sol,
        )
        fes_ocp_problem.add_plot_penalty(CostType.ALL)
        fes_ocp_sol = fes_ocp_problem.solve(solver=solver)

        cycle_record = make_cycle_record(
            cycle_index=attempted_cycle,
            standard_sol=standard_cycling_sol,
            fes_ocp_sol=fes_ocp_sol,
            cycling_info=cycling_info,
        )
        cycle_records.append(cycle_record)

        if fes_ocp_sol.status != 0:
            failure_reason = "FES-OCP feasibility problem failed. Muscle failure reached."
            print(failure_reason)
            break

        cycle_to_failure += 1
        print(f"Cycle {cycle_to_failure} completed.")

    print(f"Muscle failure reached after {cycle_to_failure} complete cycles.")

    save_common_results(
        file_name=RESULT_FILE_NAME,
        cycle_records=cycle_records,
        cycle_to_failure=cycle_to_failure,
        optim_info=optim_info,
        cycling_info=cycling_info,
        model_path=MODEL_PATH,
        failure_reason=failure_reason,
    )


if __name__ == "__main__":
    main()
