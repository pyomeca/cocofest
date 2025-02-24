"""
/!\ This example is not functional yet. /!\
/!\ It is a work in progress muscles can not be stimulated seperatly /!\

This example will do a pulse apparition optimization to either minimize overall muscle force or muscle fatigue
for a reaching task. Those ocp were build to move from starting position (arm: 0°, elbow: 5°) to a target position
defined in the bioMod file. At the end of the simulation 2 files will be created, one for each optimization.
The files will contain the time, states, controls and parameters of the ocp.
"""

import numpy as np

from bioptim import Axis, ConstraintFcn, ConstraintList, Node, Solver

from cocofest import DingModelFrequencyWithFatigue, OcpFesMsk, FesMskModel, SolutionToPickle

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
biceps_pcsa = 12.7
triceps_pcsa = 28.3
brachioradialis_pcsa = 11.6

biceps_a_rest_proportion = 12.7 / 28.3
triceps_a_rest_proportion = 1
brachioradialis_a_rest_proportion = 11.6 / 28.3
a_rest_proportion_list = [
    biceps_a_rest_proportion,
    biceps_a_rest_proportion,
    triceps_a_rest_proportion,
    triceps_a_rest_proportion,
    triceps_a_rest_proportion,
    brachioradialis_a_rest_proportion,
]

fes_muscle_models = [
    DingModelFrequencyWithFatigue(muscle_name="BIClong"),
    DingModelFrequencyWithFatigue(muscle_name="BICshort"),
    DingModelFrequencyWithFatigue(muscle_name="TRIlong"),
    DingModelFrequencyWithFatigue(muscle_name="TRIlat"),
    DingModelFrequencyWithFatigue(muscle_name="TRImed"),
    DingModelFrequencyWithFatigue(muscle_name="BRA"),
]

for i in range(len(fes_muscle_models)):
    fes_muscle_models[i].alpha_a = fes_muscle_models[i].alpha_a * alpha_a_proportion_list[i]
    fes_muscle_models[i].a_rest = fes_muscle_models[i].a_rest * a_rest_proportion_list[i]

model = FesMskModel(
    name=None,
    biorbd_path="../../model_msk/arm26.bioMod",
    muscles_model=fes_muscle_models,
    activate_force_length_relationship=True,
    activate_force_velocity_relationship=True,
    activate_residual_torque=False,
)

pickle_file_list = ["minimize_muscle_fatigue.pkl", "minimize_muscle_force.pkl"]
stim_time = list(np.round(np.linspace(0, 1, 41), 3))[:-1]

constraint = ConstraintList()
constraint.add(
    ConstraintFcn.SUPERIMPOSE_MARKERS,
    first_marker="COM_hand",
    second_marker="reaching_target",
    phase=0,
    node=Node.END,
    axes=[Axis.X, Axis.Y],
)

for i in range(len(pickle_file_list)):
    time = []
    states = []
    controls = []
    parameters = []

    ocp = OcpFesMsk.prepare_ocp(
        model=model,
        stim_time=stim_time,
        final_time=1,
        pulse_event={"min": 0.01, "max": 0.1, "bimapping": False},
        objective={
            "minimize_fatigue": (True if pickle_file_list[i] == "minimize_muscle_fatigue.pkl" else False),
            "minimize_force": (True if pickle_file_list[i] == "minimize_muscle_force.pkl" else False),
        },
        msk_info={
            "with_residual_torque": False,
            "bound_type": "start",
            "bound_data": [0, 5],
            "custom_constraint": constraint,
        },
        n_threads=5,
        use_sx=False,
    )

    sol = ocp.solve(Solver.IPOPT(_max_iter=10000))
    SolutionToPickle(sol, "pulse_intensity_" + pickle_file_list[i], "result_file/").pickle()


# [1] Dahmane, R., Djordjevič, S., Šimunič, B., & Valenčič, V. (2005).
# Spatial fiber type distribution in normal human muscle: histochemical and tensiomyographical evaluation.
# Journal of biomechanics, 38(12), 2451-2459.

# [2] Klein, C. S., Allman, B. L., Marsh, G. D., & Rice, C. L. (2002).
# Muscle size, strength, and bone geometry in the upper limbs of young and old men.
# The Journals of Gerontology Series A: Biological Sciences and Medical Sciences, 57(7), M455-M459.
