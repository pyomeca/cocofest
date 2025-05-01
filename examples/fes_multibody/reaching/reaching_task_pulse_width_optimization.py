"""
This example will do a pulse width optimization to either minimize overall muscle force or muscle fatigue
for a reaching task. Those ocp were build to move from starting position (arm: 0°, elbow: 5°) to a target position
defined in the bioMod file. At the end of the simulation 2 files will be created, one for each optimization.
The files will contain the time, states, controls and parameters of the ocp.
"""

import numpy as np

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
)

from cocofest import (
    DingModelPulseWidthFrequencyWithFatigue,
    OcpFesMsk,
    FesMskModel,
    CustomObjective,
)


def initialize_model():
    # Scaling alpha_a and a_scale parameters for each muscle proportionally to the muscle PCSA and fiber type 2 proportion
    # Fiber type proportion from [1]
    biceps_fiber_type_2_proportion = 0.607
    triceps_fiber_type_2_proportion = 0.465
    brachioradialis_fiber_type_2_proportion = 0.457
    alpha_a_proportion_list = [
        biceps_fiber_type_2_proportion,
        biceps_fiber_type_2_proportion,
        triceps_fiber_type_2_proportion,
        triceps_fiber_type_2_proportion,
        triceps_fiber_type_2_proportion,
        brachioradialis_fiber_type_2_proportion,
    ]

    # PCSA (cm²) from [2]
    triceps_pcsa = 28.3
    biceps_pcsa = 12.7
    brachioradialis_pcsa = 11.6
    triceps_a_scale_proportion = 1
    biceps_a_scale_proportion = biceps_pcsa / triceps_pcsa
    brachioradialis_a_scale_proportion = brachioradialis_pcsa / triceps_pcsa
    a_scale_proportion_list = [
        biceps_a_scale_proportion,
        biceps_a_scale_proportion,
        triceps_a_scale_proportion,
        triceps_a_scale_proportion,
        triceps_a_scale_proportion,
        brachioradialis_a_scale_proportion,
    ]

    # Build the functional electrical stimulation models according
    # to number and name of muscle in the musculoskeletal model used
    fes_muscle_models = [
        DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIClong", sum_stim_truncation=10),
        DingModelPulseWidthFrequencyWithFatigue(muscle_name="BICshort", sum_stim_truncation=10),
        DingModelPulseWidthFrequencyWithFatigue(muscle_name="TRIlong", sum_stim_truncation=10),
        DingModelPulseWidthFrequencyWithFatigue(muscle_name="TRIlat", sum_stim_truncation=10),
        DingModelPulseWidthFrequencyWithFatigue(muscle_name="TRImed", sum_stim_truncation=10),
        DingModelPulseWidthFrequencyWithFatigue(muscle_name="BRA", sum_stim_truncation=10),
    ]

    # Applying the scaling
    for i in range(len(fes_muscle_models)):
        fes_muscle_models[i].alpha_a = fes_muscle_models[i].alpha_a * alpha_a_proportion_list[i]
        fes_muscle_models[i].a_scale = fes_muscle_models[i].a_scale * a_scale_proportion_list[i]

    stim_time = list(np.round(np.linspace(0, 1.5, 61), 3))[:-1]
    model = FesMskModel(
        name=None,
        biorbd_path="../../model_msk/arm26.bioMod",
        stim_time=stim_time,
        muscles_model=fes_muscle_models,
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=True,
        activate_residual_torque=False,
    )
    return model


def prepare_ocp(
    model: FesMskModel,
    final_time: float,
    max_bound: float,
    msk_info: dict,
    minimize_force: bool = False,
    minimize_fatigue: bool = False,
):
    muscle_model = model.muscles_dynamics_model[0]
    n_shooting = muscle_model.get_n_shooting(final_time)
    numerical_data_time_series, stim_idx_at_node_list = muscle_model.get_numerical_data_time_series(
        n_shooting, final_time
    )

    dynamics = OcpFesMsk.declare_dynamics(
        model,
        numerical_time_series=numerical_data_time_series,
        ode_solver=OdeSolver.RK4(n_integration_steps=5),
        with_contact=False,
    )

    x_bounds, x_init = OcpFesMsk.set_x_bounds(model, msk_info)
    u_bounds, u_init = OcpFesMsk.set_u_bounds(model, msk_info["with_residual_torque"], max_bound=max_bound)

    objective_functions = ObjectiveList()
    if minimize_force:
        objective_functions.add(
            CustomObjective.minimize_overall_muscle_force_production,
            custom_type=ObjectiveFcn.Lagrange,
            weight=1,
            quadratic=True,
        )
    if minimize_fatigue:
        objective_functions.add(
            CustomObjective.minimize_overall_muscle_fatigue,
            custom_type=ObjectiveFcn.Lagrange,
            weight=1,
            quadratic=True,
        )

    # Step time of 1ms -> 1sec / (40Hz * 25) = 0.001s
    constraint = ConstraintList()
    constraint.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        first_marker="COM_hand",
        second_marker="reaching_target",
        phase=0,
        node=40,
        axes=[Axis.X, Axis.Y],
    )

    return OptimalControlProgram(
        bio_model=[model],
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=final_time,
        objective_functions=objective_functions,
        x_init=x_init,
        x_bounds=x_bounds,
        u_init=u_init,
        u_bounds=u_bounds,
        control_type=ControlType.CONSTANT,
        use_sx=True,
        n_threads=20,
    )


def main():
    model = initialize_model()
    for i in range(2):
        ocp = prepare_ocp(
            model=model,
            final_time=1.5,
            max_bound=0.0006,
            msk_info={
                "with_residual_torque": False,
                "bound_type": "start_end",
                "bound_data": [[0, 5], [0, 5]],
            },
            minimize_force=(i == 0),
            minimize_fatigue=(i == 1),
        )

        sol = ocp.solve(Solver.IPOPT(_max_iter=10000))
        sol.graphs(show_bounds=False)
        sol.animate(viewer="pyorerun")


if __name__ == "__main__":
    main()

# [1] Dahmane, R., Djordjevič, S., Šimunič, B., & Valenčič, V. (2005).
# Spatial fiber type distribution in normal human muscle: histochemical and tensiomyographical evaluation.
# Journal of biomechanics, 38(12), 2451-2459.

# [2] Klein, C. S., Allman, B. L., Marsh, G. D., & Rice, C. L. (2002).
# Muscle size, strength, and bone geometry in the upper limbs of young and old men.
# The Journals of Gerontology Series A: Biological Sciences and Medical Sciences, 57(7), M455-M459.
