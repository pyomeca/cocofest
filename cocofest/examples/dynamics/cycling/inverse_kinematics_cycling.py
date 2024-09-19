"""
This example will do an inverse kinematics and dynamics of a 100 steps hand cycling motion.
"""

import numpy as np
import matplotlib.pyplot as plt

import biorbd
from pyorerun import BiorbdModel, PhaseRerun

from cocofest import get_circle_coord


def main(show_plot=True, animate=True):
    # Load a predefined model
    model = biorbd.Model("../../msk_models/simplified_UL_Seth.bioMod")
    n_frames = 1000

    # Define the marker target to match
    z = model.markers(np.array([0, 0]))[0].to_array()[2]
    get_circle_coord_list = np.array(
        [
            get_circle_coord(theta, 0.35, 0, 0.1, z)
            for theta in np.linspace(0, -2 * np.pi, n_frames)
        ]
    )
    target_q = np.array(
        [
            [get_circle_coord_list[:, 0]],
            [get_circle_coord_list[:, 1]],
            [get_circle_coord_list[:, 2]],
        ]
    )

    # Perform the inverse kinematics
    ik = biorbd.InverseKinematics(model, target_q)
    ik_q = ik.solve(method="trf")
    ik_qdot = np.array(
        [np.gradient(ik_q[i], (1 / n_frames)) for i in range(ik_q.shape[0])]
    )
    ik_qddot = np.array(
        [np.gradient(ik_qdot[i], (1 / (n_frames))) for i in range(ik_qdot.shape[0])]
    )

    # Perform the inverse dynamics
    tau_shape = (model.nbQ(), ik_q.shape[1] - 1)
    tau = np.zeros(tau_shape)
    for i in range(tau.shape[1]):
        tau_i = model.InverseDynamics(ik_q[:, i], ik_qdot[:, i], ik_qddot[:, i])
        tau[:, i] = tau_i.to_array()

    # Plot the results
    if show_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title("Q")
        ax1.plot(np.linspace(0, 1, n_frames), ik_q[0], color="orange", label="shoulder")
        ax1.plot(np.linspace(0, 1, n_frames), ik_q[1], color="blue", label="elbow")
        ax1.set(xlabel="Time (s)", ylabel="Angle (rad)")
        ax2.set_title("Tau")
        ax2.plot(
            np.linspace(0, 1, n_frames - 1), tau[0], color="orange", label="shoulder"
        )
        ax2.plot(np.linspace(0, 1, n_frames - 1), tau[1], color="blue", label="elbow")
        ax2.set(xlabel="Time (s)", ylabel="Torque (N.m)")
        plt.legend()
        plt.show()

    # pyorerun animation
    if animate:
        biorbd_model = biorbd.Model(
            "../../msk_models/simplified_UL_Seth_full_mesh.bioMod"
        )
        prr_model = BiorbdModel.from_biorbd_object(biorbd_model)

        nb_seconds = 1
        t_span = np.linspace(0, nb_seconds, n_frames)

        viz = PhaseRerun(t_span)
        viz.add_animated_model(prr_model, ik_q)
        viz.rerun("msk_model")


if __name__ == "__main__":
    main(show_plot=True, animate=True)
