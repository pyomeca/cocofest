"""
This example will do a pulse width optimization to either minimize overall muscle force or muscle fatigue
for a reaching task. Those ocp were build to move from starting position (arm: 0°, elbow: 5°) to a target position
defined in the bioMod file. At the end of the simulation 2 files will be created, one for each optimization.
The files will contain the time, states, controls and parameters of the ocp.
"""

import numpy as np

import bioptim
from bioptim import (
    Axis,
    ConstraintFcn,
    ConstraintList,
    Solver,
    OdeSolver,
    ObjectiveList,
    ObjectiveFcn,
    OptimalControlProgram,
    ControlType,
    Node,
    InitialGuessList,
)

from cocofest import (
    DingModelPulseWidthFrequency,
    OcpFesMsk,
    FesMskModel,
    CustomObjective,
)


def initialize_model(final_time, model_path):
    # Build the functional electrical stimulation models according
    # to number and name of muscle in the musculoskeletal model used
    fes_muscle_models = [
        DingModelPulseWidthFrequency(muscle_name="BIClong", sum_stim_truncation=6),
        DingModelPulseWidthFrequency(muscle_name="BICshort", sum_stim_truncation=6),
        DingModelPulseWidthFrequency(muscle_name="TRIlong", sum_stim_truncation=6),
        DingModelPulseWidthFrequency(muscle_name="TRIlat", sum_stim_truncation=6),
        DingModelPulseWidthFrequency(muscle_name="TRImed", sum_stim_truncation=6),
        DingModelPulseWidthFrequency(muscle_name="BRA", sum_stim_truncation=6),
    ]

    stim_time = list(np.linspace(0, final_time, 30, endpoint=False))
    model = FesMskModel(
        name=None,
        biorbd_path=model_path,
        stim_time=stim_time,
        muscles_model=fes_muscle_models,
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=True,
        activate_residual_torque=False,
        with_contact=False,
    )
    return model


def prepare_ocp(
    model: FesMskModel,
    final_time: float,
    max_bound: float,
):
    muscle_model = model.muscles_dynamics_model[0]
    n_shooting = muscle_model.get_n_shooting(final_time)
    numerical_data_time_series, stim_idx_at_node_list = muscle_model.get_numerical_data_time_series(
        n_shooting, final_time
    )
    dynamics_options = OcpFesMsk.declare_dynamics_options(
        numerical_time_series=numerical_data_time_series,
        ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3, method="radau"),
    )

    # --- Initialize default FES bounds and initial guess --- #
    x_bounds, x_init_fes = OcpFesMsk.set_x_bounds_fes(model)

    # --- Setting q bounds --- #
    q_x_bounds = model.bounds_from_ranges("q")

    arm_q = [-0.5, 3.14]
    forearm_q = [0, 3.15]

    q_x_bounds.min[0] = [0, arm_q[0], arm_q[0]]
    q_x_bounds.max[0] = [0, arm_q[1], arm_q[1]]
    q_x_bounds.min[1] = [forearm_q[0], forearm_q[0], forearm_q[0]]
    q_x_bounds.max[1] = [forearm_q[0], forearm_q[1], forearm_q[1]]

    x_bounds.add(key="q", bounds=q_x_bounds, phase=0)

    # --- Setting u bounds --- #
    u_bounds = bioptim.BoundsList()
    u_init = InitialGuessList()

    models = model.muscles_dynamics_model
    for individual_model in models:
        key = "last_pulse_width_" + str(individual_model.muscle_name)
        u_bounds.add(
            key=key,
            min_bound=[individual_model.pd0],
            max_bound=[max_bound],
            phase=0,
        )

    objective_functions = ObjectiveList()
    objective_functions.add(
        CustomObjective.minimize_overall_muscle_force_production,
        custom_type=ObjectiveFcn.Lagrange,
        weight=1,
        quadratic=True,
    )

    constraint = ConstraintList()
    constraint.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        first_marker="COM_hand",
        second_marker="reaching_target",
        phase=0,
        node=Node.END,
        axes=[Axis.X, Axis.Y],
    )

    return OptimalControlProgram(
        bio_model=[model],
        dynamics=dynamics_options,
        n_shooting=n_shooting,
        phase_time=final_time,
        objective_functions=objective_functions,
        x_init=x_init_fes,
        x_bounds=x_bounds,
        u_init=u_init,
        u_bounds=u_bounds,
        constraints=constraint,
        control_type=ControlType.CONSTANT,
        use_sx=False,
        n_threads=20,
    )


def main(plot=True, model_path="../../msk_models/Arm26/arm26_with_reaching_target.bioMod"):
    final_time = 1
    model = initialize_model(final_time, model_path)
    ocp = prepare_ocp(model=model, final_time=final_time, max_bound=0.0006)

    sol = ocp.solve(Solver.IPOPT(_max_iter=10000))
    if plot:
        sol.animate(viewer="pyorerun")
        sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
