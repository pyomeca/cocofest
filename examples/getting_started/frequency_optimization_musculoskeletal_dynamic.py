"""
This example will do a 10 stimulation example with Ding's 2003 frequency model.
This ocp was build to produce a elbow motion from 5 to 120 degrees.
The stimulation frequency is fixed at 10 Hz and the elbow torque control is optimized to satisfy the flexion.
"""

import numpy as np

from cocofest import DingModelFrequencyWithFatigue, OcpFesMsk, FesMskModel

from bioptim import OdeSolver


def prepare_ocp():
    model = FesMskModel(
        name=None,
        biorbd_path="../msk_models/arm26_biceps_1dof.bioMod",
        muscles_model=[DingModelFrequencyWithFatigue(muscle_name="BIClong")],
        stim_time=list(np.linspace(0, 1, 11)[:-1]),
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=True,
        activate_residual_torque=True,
    )

    return OcpFesMsk.prepare_ocp(
        model=model,
        final_time=1,
        objective={"minimize_residual_torque": True},
        msk_info={
            "with_residual_torque": True,
            "bound_type": "start_end",
            "bound_data": [[5], [120]],
        },
        use_sx=True,
        n_threads=5,
        ode_solver=OdeSolver.RK4(n_integration_steps=10),
    )


def main():
    ocp = prepare_ocp()

    sol = ocp.solve()

    sol.animate(viewer="pyorerun", n_frames=1000)
    sol.graphs(show_bounds=False)


if __name__ == "__main__":
    main()
