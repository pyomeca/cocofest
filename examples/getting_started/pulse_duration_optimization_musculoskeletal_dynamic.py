"""
This example will do a 33 stimulation example with Ding's 2007 frequency model.
This ocp was build to produce an elbow motion from 5 to 120 degrees.
The stimulation frequency is fixed at 33hz and the pulse width is optimized to satisfy the flexion while minimizing
elbow residual torque control.
"""

import numpy as np

from bioptim import Solver, OdeSolver
from cocofest import DingModelPulseWidthFrequencyWithFatigue, OcpFesMsk, FesMskModel


def prepare_ocp(stim_time: list, final_time: float, external_force: bool):
    model = FesMskModel(
        name=None,
        biorbd_path="../model_msk/arm26_biceps_1dof.bioMod",
        muscles_model=[DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIClong")],
        stim_time=stim_time,
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=True,
        activate_residual_torque=True,
        external_force_set=None,  # External forces will be added later
    )

    minimum_pulse_width = DingModelPulseWidthFrequencyWithFatigue().pd0
    resistive_torque = {
        "Segment_application": "r_ulna_radius_hand",
        "torque": np.array([0, 0, -1]),
        "with_contact": False,
    }
    return OcpFesMsk.prepare_ocp(
        model=model,
        final_time=final_time,
        pulse_width={
            "min": minimum_pulse_width,
            "max": 0.0006,
            "bimapping": False,
        },
        objective={"minimize_residual_torque": True},
        msk_info={
            "bound_type": "start_end",
            "bound_data": [[5], [120]],
            "with_residual_torque": True,
        },
        use_sx=True,
        n_threads=10,
        ode_solver=OdeSolver.RK4(n_integration_steps=10),
        external_forces=resistive_torque if external_force else None,
    )


def main():
    simulation_ending_time = 1
    ocp = prepare_ocp(
        stim_time=list(np.linspace(0, simulation_ending_time, 34)[:-1]),
        final_time=simulation_ending_time,
        external_force=True,
    )

    sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=2000))

    sol.animate(viewer="pyorerun")
    sol.graphs(show_bounds=False)


if __name__ == "__main__":
    main()
