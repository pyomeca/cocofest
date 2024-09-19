import numpy as np
import matplotlib.pyplot as plt
import pickle

from biorbd import Model


class ForceSensorToMuscleForce:  # TODO : Enable several muscles (biceps, triceps, deltoid, etc.)
    """
    This class is used to convert the force sensor data into muscle force.
    This program was built to use the arm26 upper limb model.
    """

    def __init__(
        self,
        pickle_path: str | list[str] = None,
        muscle_name: str | list[str] = None,
        forearm_angle: int | float | list[int] | list[float] = None,
        out_pickle_path: str | list[str] = None,
    ):
        if pickle_path is None:
            raise ValueError("Please provide a path to the pickle file(s).")
        if not isinstance(pickle_path, str) and not isinstance(pickle_path, list):
            raise TypeError(
                "Please provide a pickle_path list of str type or a str type path."
            )
        if not isinstance(out_pickle_path, str) and not isinstance(
            out_pickle_path, list
        ):
            raise TypeError(
                "Please provide a out_pickle_path list of str type or a str type path."
            )
        if out_pickle_path is not None:
            if isinstance(out_pickle_path, str):
                out_pickle_path = [out_pickle_path]
            if len(out_pickle_path) != 1:
                if len(out_pickle_path) != len(pickle_path):
                    raise ValueError(
                        "If not str type, out_pickle_path must be the same length as pickle_path."
                    )

        self.path = pickle_path
        self.plot = plot
        self.time = None
        self.stim_time = None
        self.t_local = None
        self.model = None
        self.Q = None
        self.Qdot = None
        self.Qddot = None
        self.biceps_moment_arm = None
        self.biceps_force_vector = None

        pickle_path_list = (
            [pickle_path] if isinstance(pickle_path, str) else pickle_path
        )

        for i in range(len(pickle_path_list)):
            self.t_local = self.load_data(pickle_path_list[i], forearm_angle)
            self.load_model(forearm_angle)
            self.get_muscle_force(local_torque_force_vector=self.t_local)
            if out_pickle_path:
                if len(out_pickle_path) == 1:
                    if out_pickle_path[:-4] == ".pkl":
                        save_pickle_path = out_pickle_path[:-4] + "_" + str(i) + ".pkl"
                    else:
                        save_pickle_path = out_pickle_path[0] + "_" + str(i) + ".pkl"
                else:
                    save_pickle_path = out_pickle_path[i]

                muscle_name = (
                    muscle_name
                    if isinstance(muscle_name, str)
                    else muscle_name[i] if isinstance(muscle_name, list) else "biceps"
                )
                dictionary = {
                    "time": self.time,
                    muscle_name: self.all_biceps_force_vector,
                    "stim_time": self.stim_time,
                }
                with open(save_pickle_path, "wb") as file:
                    pickle.dump(dictionary, file)

    def load_data(self, pickle_path, forearm_angle):
        # --- Retrieving pickle data --- #
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
            sensor_data = []
            dict_name_list = ["mx", "my", "mz", "x", "y", "z"]
            for name in dict_name_list:
                sensor_data.append(data[name])

        self.time = data["time"]
        self.stim_time = data["stim_time"]

        # return self.local_to_global(sensor_data, forearm_angle)
        return self.local_sensor_to_local_hand(sensor_data)

    @staticmethod
    def local_sensor_to_local_hand(sensor_data: np.array) -> np.array:
        """
        This function is used to convert the sensor data from the local axis to the local hand axis.
        fx_global = -fx_local
        fy_global = fz_local
        fz_global = fy_local

        Parameters
        ----------
        sensor_data

        Returns
        -------

        """
        # TODO Might be an error when not at 90°
        hand_local_force_data = [
            [[-x for x in data] for data in sensor_data[0]],
            sensor_data[2],
            sensor_data[1],
            [[-x for x in data] for data in sensor_data[3]],
            sensor_data[5],
            sensor_data[4],
        ]
        return hand_local_force_data

    @staticmethod
    def local_to_global(sensor_data: np.array, forearm_angle: int | float) -> np.array:
        """
        This function is used to convert the sensor data from the local axis to the global axis.
        When the forearm position is at 90° :
        fx_global = -fz_local
        fy_global = -fx_local
        fz_global = fy_local

        Parameters
        ----------
        sensor_data
        forearm_angle

        Returns
        -------

        """
        rotation_1_rad = np.radians(-90) + np.radians(forearm_angle - 90)
        rotation_2_rad = np.radians(90)

        rotation_matrix_1 = np.array(
            [
                [np.cos(rotation_1_rad), 0, np.sin(rotation_1_rad)],
                [0, 1, 0],
                [-np.sin(rotation_1_rad), 0, np.cos(rotation_1_rad)],
            ]
        )
        rotation_matrix_2 = np.array(
            [
                [1, 0, 0],
                [0, np.cos(rotation_2_rad), -np.sin(rotation_2_rad)],
                [0, np.sin(rotation_2_rad), np.cos(rotation_2_rad)],
            ]
        )

        rotation_matrix = rotation_matrix_2 @ rotation_matrix_1

        global_orientation_sensor_data = []
        for i in range(len(sensor_data[0])):
            sensor_data_temp_torque = (
                rotation_matrix
                @ np.array([sensor_data[0][i], sensor_data[1][i], sensor_data[2][i]])
            ).tolist()
            sensor_data_temp_force = (
                rotation_matrix
                @ np.array([sensor_data[3][i], sensor_data[4][i], sensor_data[5][i]])
            ).tolist()
            global_orientation_sensor_data.append(
                [
                    sensor_data_temp_torque[0],
                    sensor_data_temp_torque[1],
                    sensor_data_temp_torque[2],
                    sensor_data_temp_force[0],
                    sensor_data_temp_force[1],
                    sensor_data_temp_force[2],
                ]
            )

        return global_orientation_sensor_data

    def load_model(self, forearm_angle: int | float):
        # Load a predefined model
        self.model = Model("model/arm26_unmesh.bioMod")
        # Get number of q, qdot, qddot
        nq = self.model.nbQ()
        nqdot = self.model.nbQdot()
        nqddot = self.model.nbQddot()

        # Choose a position/velocity/acceleration to compute dynamics from
        if nq != 2:
            raise ValueError("The number of degrees of freedom has changed.")  # 0
        self.Q = np.array(
            [0.0, np.radians(forearm_angle)]
        )  # "0" arm along body and "1.57" 90° forearm position  |__.
        self.Qdot = np.zeros((nqdot,))  # speed null
        self.Qddot = np.zeros((nqddot,))  # acceleration null

        # Biceps moment arm
        self.model.musclesLengthJacobian(self.Q).to_array()
        if self.model.muscleNames()[1].to_string() != "BIClong":
            raise ValueError(
                "Biceps muscle index as changed."
            )  # biceps is index 1 in the model
        self.biceps_moment_arm = self.model.musclesLengthJacobian(self.Q).to_array()[1][
            1
        ]

        # Expressing the external force array [Mx, My, Mz, Fx, Fy, Fz]
        # experimentally applied at the hand into the last joint
        if (
            self.model.segments()[15].name().to_string()
            != "r_ulna_radius_hand_r_elbow_flex"
        ):
            raise ValueError("r_ulna_radius_hand_r_elbow_flex index as changed.")

        if self.model.markerNames()[3].to_string() != "r_ulna_radius_hand":
            raise ValueError("r_ulna_radius_hand marker index as changed.")

        if self.model.markerNames()[4].to_string() != "hand":
            raise ValueError("hand marker index as changed.")

    def get_muscle_force(self, local_torque_force_vector):
        self.all_biceps_force_vector = []
        for i in range(len(local_torque_force_vector)):
            self.biceps_force_vector = []
            for j in range(len(local_torque_force_vector[i][0])):
                # a = self.model.markers(self.Q)[4].to_array()
                # b = self.model.markers(Q)[3].to_array()  # [0, 0, 0]
                # the 'b' point is not used for calculation as 'a' is expressed in 'b' local coordinates
                # t_global = self.force_transport(
                #     [
                #         local_torque_force_vector[i][0][j],
                #         local_torque_force_vector[i][1][j],
                #         local_torque_force_vector[i][2][j],
                #         local_torque_force_vector[i][3][j],
                #         local_torque_force_vector[i][4][j],
                #         local_torque_force_vector[i][5][j],
                #     ],
                #     a,
                # )  # TODO make it a list : local_torque_force_vector

                hand_local_vector = [
                    local_torque_force_vector[i][0][j],
                    local_torque_force_vector[i][1][j],
                    local_torque_force_vector[i][2][j],
                    local_torque_force_vector[i][3][j],
                    local_torque_force_vector[i][4][j],
                    local_torque_force_vector[i][5][j],
                ]

                #
                # external_forces = np.array(t_global)[:, np.newaxis]
                # external_forces_v = biorbd.to_spatial_vector(external_forces)

                external_forces_vector = np.array(hand_local_vector)
                external_forces_set = self.model.externalForceSet()
                external_forces_set.addInSegmentReferenceFrame(
                    segmentName="r_ulna_radius_hand",
                    vector=external_forces_vector,
                    pointOfApplication=np.array([0, 0, 0]),
                )

                # tau = self.model.InverseDynamics(self.Q, self.Qdot, self.Qddot, f_ext=external_forces_v).to_array()[1]
                tau = self.model.InverseDynamics(
                    self.Q, self.Qdot, self.Qddot, external_forces_set
                ).to_array()
                tau = tau[1]
                biceps_force = tau / self.biceps_moment_arm
                self.biceps_force_vector.append(biceps_force)
            hack = (
                self.biceps_force_vector + self.biceps_force_vector[0]
                if self.biceps_force_vector[0] > 0
                else self.biceps_force_vector - self.biceps_force_vector[0]
            )
            self.all_biceps_force_vector.append(
                self.biceps_force_vector
                # hack.tolist()
            )  # TODO: This is an hack, find why muscle force is sometimes negative when it shouldn't

        # --- Plotting the biceps force --- #
        if self.plot:
            for i in range(len(self.all_biceps_force_vector)):
                plt.plot(self.time[i], self.all_biceps_force_vector[i])
            # plt.plot(self.time[0], self.all_biceps_force_vector[0])
            plt.show()

    @staticmethod
    def force_transport(f, a, b: list = None):
        if b is None:
            b = [0, 0, 0]
        vector_ba = a[:3] - b[:3]
        new_f = np.array(f)
        new_f[:3] = np.array(f[:3]) + np.cross(vector_ba, np.array(f[3:6]))
        return new_f.tolist()

    def plot(self):
        for i in range(len(self.all_biceps_force_vector)):
            plt.plot(self.time[i], self.all_biceps_force_vector[i])
        plt.show()


if __name__ == "__main__":
    ForceSensorToMuscleForce(
        pickle_path="D:\These\Programmation\Modele_Musculaire\optistim\data_process\identification_data_Biceps_90deg_30mA_300us_33Hz_essai1.pkl",
        muscle_name="biceps",
        forearm_angle=90,
        out_pickle_path="biceps_force",
    )

    # with open("D:/These/Programmation/Modele_Musculaire/optistim/data_process/biceps_force_0.pkl", 'rb') as f:
    #     data = pickle.load(f)
