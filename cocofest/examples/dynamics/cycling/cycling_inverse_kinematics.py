"""
This example will do an inverse kinematics and dynamics of a 100 steps hand cycling motion.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import biorbd
from pyorerun import BiorbdModel, PhaseRerun

from cocofest import get_circle_coord


def main(show_plot=True, animate=True):
    n_shooting = 1000
    cycling_number = 1

    # Define the circle parameters
    x_center = 0.35
    y_center = 0
    radius = 0.1

    # Load a predefined model
    model = biorbd.Model("../../msk_models/simplified_UL_Seth_pedal_aligned_for_inverse_kinematics.bioMod")

    # Define the marker target to match
    z = model.markers(np.array([0] * model.nbQ()))[0].to_array()[2]
    if z != model.markers(np.array([np.pi / 2] * model.nbQ()))[0].to_array()[2]:
        print("The model not strictly 2d. Warm start not optimal.")

    f = interp1d(np.linspace(0, -360 * cycling_number, 360 * cycling_number + 1),
                 np.linspace(0, -360 * cycling_number, 360 * cycling_number + 1), kind="linear")
    x_new = f(np.linspace(0, -360 * cycling_number, n_shooting + 1))
    x_new_rad = np.deg2rad(x_new)

    x_y_z_coord = np.array(
        [
            get_circle_coord(theta, x_center, y_center, radius)
            for theta in x_new_rad
        ]
    ).T

    target_q_hand = x_y_z_coord.reshape((3, 1, n_shooting + 1))  # Hand marker_target
    wheel_center_x_y_z_coord = np.array([x_center, y_center, z])
    target_q_wheel_center = np.tile(wheel_center_x_y_z_coord[:, np.newaxis, np.newaxis],
                                    (1, 1, n_shooting + 1))  # Wheel marker_target
    target_q = np.concatenate((target_q_hand, target_q_wheel_center), axis=1)

    # Perform the inverse kinematics
    ik = biorbd.InverseKinematics(model, target_q)
    ik_q = ik.solve(method="lm")
    ik_qdot = np.array([np.gradient(ik_q[i], (1 / n_shooting)) for i in range(ik_q.shape[0])])
    ik_qddot = np.array([np.gradient(ik_qdot[i], (1 / n_shooting)) for i in range(ik_qdot.shape[0])])

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
        ax1.plot(np.linspace(0, 1, n_shooting + 1), ik_q[0], color="orange", label="shoulder")
        ax1.plot(np.linspace(0, 1, n_shooting + 1), ik_q[1], color="blue", label="elbow")
        ax1.set(xlabel="Time (s)", ylabel="Angle (rad)")
        ax2.set_title("Tau")
        ax2.plot(np.linspace(0, 1, n_shooting)[2:-1], tau[0][2:-1], color="orange", label="shoulder")
        ax2.plot(np.linspace(0, 1, n_shooting)[2:-1], tau[1][2:-1], color="blue", label="elbow")
        ax2.set(xlabel="Time (s)", ylabel="Torque (N.m)")
        plt.legend()
        plt.show()

    # pyorerun animation
    if animate:
        biorbd_model = biorbd.Model("../../msk_models/simplified_UL_Seth_pedal_aligned.bioMod")
        prr_model = BiorbdModel.from_biorbd_object(biorbd_model)

        nb_seconds = 1
        t_span = np.linspace(0, nb_seconds, n_shooting+1)

        viz = PhaseRerun(t_span)
        viz.add_animated_model(prr_model, ik_q)
        viz.rerun("msk_model")


if __name__ == "__main__":
    main(show_plot=True, animate=True)
