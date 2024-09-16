"""
This example will do an optimal control program of a 40 stimulation example with Ding's 2007 pulse duration model.
Those ocp were build to produce a cycling motion.
The stimulation frequency will be set to 40Hz and pulse duration will be optimized to satisfy the motion meanwhile
reducing residual torque.
"""

import pickle

import numpy as np

from bioptim import CostType, SolutionMerge, Solver

import biorbd

from pyorerun import BiorbdModel, PhaseRerun

from cocofest import DingModelPulseDurationFrequencyWithFatigue, OcpFesMsk, PlotCyclingResult, SolutionToPickle, FesMskModel


def main():
    n_shooting = 100
    minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0

    model = FesMskModel(name=None,
                        biorbd_path="../../msk_models/simplified_UL_Seth.bioMod",
                        muscles_model=[DingModelPulseDurationFrequencyWithFatigue(muscle_name="DeltoideusClavicle_A"),
                                       DingModelPulseDurationFrequencyWithFatigue(muscle_name="DeltoideusScapula_P"),
                                       DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong"),
                                       DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIC_long"),
                                       DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIC_brevis")],
                        activate_force_length_relationship=True,
                        activate_force_velocity_relationship=True,
                        )

    ocp = OcpFesMsk.prepare_ocp(
        model=model,
        stim_time=list(np.round(np.linspace(0, 1, 11), 3))[:-1],
        n_shooting=n_shooting,
        final_time=1,
        pulse_duration={
            "min": minimum_pulse_duration,
            "max": 0.0006,
            "bimapping": False,
        },
        msk_info={"with_residual_torque": True},
        objective={
            "cycling": {"x_center": 0.35, "y_center": 0, "radius": 0.1, "target": "marker"},
            "minimize_residual_torque": True,
        },
        warm_start=False,
        n_threads=5,
    )
    ocp.add_plot_penalty(CostType.ALL)
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=10000))
    SolutionToPickle(sol, "cycling_fes_driven_min_residual_torque.pkl", "").pickle()

    biorbd_model = biorbd.Model("../../msk_models/simplified_UL_Seth_full_mesh.bioMod")
    prr_model = BiorbdModel.from_biorbd_object(biorbd_model)

    nb_frames = 440
    nb_seconds = 1
    t_span = np.linspace(0, nb_seconds, nb_frames)

    viz = PhaseRerun(t_span)
    q = sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"]
    viz.add_animated_model(prr_model, q)
    viz.rerun("msk_model")

    sol.graphs(show_bounds=False)
    PlotCyclingResult(sol).plot(starting_location="E")


if __name__ == "__main__":
    main()
