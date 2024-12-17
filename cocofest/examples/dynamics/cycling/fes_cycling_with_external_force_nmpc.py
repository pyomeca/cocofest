"""
This example will do an optimal control program of a 100 steps tracking a hand cycling motion with a torque driven and
a torque resistance at the handle.
"""

import platform
import numpy as np

from bioptim import (
    Axis,
    BiorbdModel,
    BoundsList,
    ConstraintList,
    ConstraintFcn,
    CostType,
    DynamicsFcn,
    DynamicsList,
    ExternalForceSetTimeSeries,
    InitialGuessList,
    InterpolationType,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    PhaseDynamics,
    Solver,
    PhaseTransitionList,
    PhaseTransitionFcn,
    MultiCyclicNonlinearModelPredictiveControl,
    Dynamics,
    Objective,
    Solution,
    SolutionMerge,
    MultiCyclicCycleSolutions,
    ControlType,
)

from cocofest import (
    get_circle_coord,
    inverse_kinematics_cycling,
    inverse_dynamics_cycling,
)

from cocofest import NmpcFes, OcpFes, FesMskModel, DingModelPulseWidthFrequencyWithFatigue, SolutionToPickle, PickleAnimate, NmpcFesMsk


def main():
    model = FesMskModel(
        name=None,
        biorbd_path="../../msk_models/simplified_UL_Seth_pedal_aligned_test.bioMod",
        muscles_model=[
            DingModelPulseWidthFrequencyWithFatigue(muscle_name="DeltoideusClavicle_A", is_approximated=False),
            DingModelPulseWidthFrequencyWithFatigue(muscle_name="DeltoideusScapula_P", is_approximated=False),
            DingModelPulseWidthFrequencyWithFatigue(muscle_name="TRIlong", is_approximated=False),
            DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIC_long", is_approximated=False),
            DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIC_brevis", is_approximated=False),
        ],
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_residual_torque=True,
        external_force_set=None,  # External forces will be added
    )

    minimum_pulse_width = DingModelPulseWidthFrequencyWithFatigue().pd0
    n_cycles = 2
    nmpc_fes_msk = NmpcFesMsk
    nmpc = nmpc_fes_msk.prepare_nmpc_for_cycling(
        model=model,
        stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        cycle_duration=1,
        n_cycles_to_advance=1,
        n_cycles_simultaneous=3,
        n_total_cycles=n_cycles,
        pulse_width={
            "min": minimum_pulse_width,
            "max": 0.0006,
            "bimapping": False,
        },
        objective={"cycling": {"x_center": 0.35, "y_center": 0, "radius": 0.1},
                   "minimize_residual_torque": True},
        msk_info={"with_residual_torque": True},
        external_forces={"torque": [0, 0, -1], "Segment_application": "wheel"}
    )

    def update_functions(_nmpc: MultiCyclicNonlinearModelPredictiveControl, cycle_idx: int, _sol: Solution):
        return cycle_idx < n_cycles  # True if there are still some cycle to perform

    sol = nmpc.solve(
        update_functions,
        solver=Solver.IPOPT(),
        cyclic_options={"states": {}},
    )

    sol.print_cost()
    sol.graphs(show_bounds=True)
    sol.animate(n_frames=200, show_tracked_markers=True)


if __name__ == "__main__":
    main()




