from casadi import MX, vertcat, sum1, mmax, fabs, sign
from bioptim import PenaltyController
from cocofest.models.ding2007.ding2007 import DingModelPulseWidthFrequency


class CustomCostFunctions:
    def __init__(self):
        self.dict_functions = {
            "minimize_average_activation": {
                "function": self.minimize_average_activation,
                "index": 1,
                "latex": r"\phi_1 = \frac{1}{n_m}\sum_{j=1}^{n_m} a^{j}, \quad a^{j}=\frac{f^{j}-f^{j}_{\min}}{f^{j}_{\max}-f^{j}_{\min}}",
                "description": "Minimize the average muscle activation",
                "power": "1",
                "state": "pw",
            },
            "minimize_root_mean_square_activation": {
                "function": self.minimize_root_mean_square_activation,
                "index": 2,
                "latex": r"\phi_2 = \left(\frac{1}{n_m}\sum_{j=1}^{n_m} (a^{j})^{2}\right)^{\tfrac{1}{2}}, \quad a^{j}=\frac{f^{j}-f^{j}_{\min}}{f^{j}_{\max}-f^{j}_{\min}}",
                "description": "Minimize the root mean square of muscle activation",
                "power": "2",
                "state": "pw",
            },
            "minimize_cubic_average_activation": {
                "function": self.minimize_cubic_average_activation,
                "index": 3,
                "latex": r"\phi_3 = \left(\frac{1}{n_m}\sum_{j=1}^{n_m} (a^{j})^{3}\right)^{\tfrac{1}{3}}, \quad a^{j}=\frac{f^{j}-f^{j}_{\min}}{f^{j}_{\max}-f^{j}_{\min}}",
                "description": "Minimize the cubic average of muscle activation",
                "power": "3",
                "state": "pw",
            },
            "minimize_peak_activation": {
                "function": self.minimize_peak_activation,
                "index": 4,
                "latex": r"\phi_4 = \max_{j=1,\ldots,n_m} \; a^{j}, \quad a^{j}=\frac{f^{j}-f^{j}_{\min}}{f^{j}_{\max}-f^{j}_{\min}}",
                "description": "Minimize the peak of muscle activation",
                "power": "max",
                "state": "pw",
            },
            "minimize_average_force": {
                "function": self.minimize_average_force,
                "index": 5,
                "latex": r"\phi_5 = \frac{1}{n_m}\sum_{j=1}^{n_m} f^{j}",
                "description": "Minimize the average muscle force",
                "power": "1",
                "state": "f",
            },
            "minimize_root_mean_square_force": {
                "function": self.minimize_root_mean_square_force,
                "index": 6,
                "latex": r"\phi_6 = \left(\frac{1}{n_m}\sum_{j=1}^{n_m} (f^{j})^{2}\right)^{\tfrac{1}{2}}",
                "description": "Minimize the root mean square of muscle force",
                "power": "2",
                "state": "f",
            },
            "minimize_cubic_average_force": {
                "function": self.minimize_cubic_average_force,
                "index": 7,
                "latex": r"\phi_7 = \left(\frac{1}{n_m}\sum_{j=1}^{n_m} (f^{j})^{3}\right)^{\tfrac{1}{3}}",
                "description": "Minimize the cubic average of muscle force",
                "power": "3",
                "state": "f",
            },
            "minimize_peak_force": {
                "function": self.minimize_peak_force,
                "index": 8,
                "latex": r"\phi_8 = \max_{j=1,\ldots,n_m} \; f^{j}",
                "description": "Minimize the peak muscle force",
                "power": "max",
                "state": "f",
            },
            "minimize_average_muscle_stress": {
                "function": self.minimize_average_muscle_stress,
                "index": 9,
                "latex": r"\phi_9 = \frac{1}{n_m}\sum_{j=1}^{n_m} \frac{f^{j}}{S^{j}}",
                "description": "Minimize the average muscle stress",
                "power": "1",
                "state": "str",
            },
            "minimize_root_mean_square_muscle_stress": {
                "function": self.minimize_root_mean_square_muscle_stress,
                "index": 10,
                "latex": r"\phi_{10} = \left(\frac{1}{n_m}\sum_{j=1}^{n_m} \left(\frac{f^{j}}{S^{j}}\right)^{2}\right)^{\tfrac{1}{2}}",
                "description": "Minimize the root mean square of muscle stress",
                "power": "2",
                "state": "str",
            },
            "minimize_cubic_average_muscle_stress": {
                "function": self.minimize_cubic_average_muscle_stress,
                "index": 11,
                "latex": r"\phi_{11} = \left(\frac{1}{n_m}\sum_{j=1}^{n_m} \left(\frac{f^{j}}{S^{j}}\right)^{3}\right)^{\tfrac{1}{3}}",
                "description": "Minimize the cubic average of muscle stress",
                "power": "3",
                "state": "str",
            },
            "minimize_peak_muscle_stress": {
                "function": self.minimize_peak_muscle_stress,
                "index": 12,
                "latex": r"\phi_{12} = \max_{j=1,\ldots,n_m} \; \frac{f^{j}}{S^{j}}",
                "description": "Minimize the peak muscle stress",
                "power": "max",
                "state": "str",
            },
            "minimize_average_muscle_power": {
                "function": self.minimize_average_muscle_power,
                "index": 13,
                "latex": r"\phi_{13} = \frac{1}{n_m}\sum_{j=1}^{n_m} f^{j} v^{j}",
                "description": "Minimize the average muscle power",
                "power": "1",
                "state": "pow",
            },
            "minimize_root_mean_square_muscle_power": {
                "function": self.minimize_root_mean_square_muscle_power,
                "index": 14,
                "latex": r"\phi_{14} = \left(\frac{1}{n_m}\sum_{j=1}^{n_m} (f^{j} v^{j})^{2}\right)^{\tfrac{1}{2}}",
                "description": "Minimize the root mean square of muscle power",
                "power": "2",
                "state": "pow",
            },
            "minimize_cubic_average_muscle_power": {
                "function": self.minimize_cubic_average_muscle_power,
                "index": 15,
                "latex": r"\phi_{15} = \left(\frac{1}{n_m}\sum_{j=1}^{n_m} (f^{j} v^{j})^{3}\right)^{\tfrac{1}{3}}",
                "description": "Minimize the cubic average of muscle power",
                "power": "3",
                "state": "pow",
            },
            "minimize_peak_muscle_power": {
                "function": self.minimize_peak_muscle_power,
                "index": 16,
                "latex": r"\phi_{16} = \max_{j=1,\ldots,n_m} \; f^{j} v^{j}}",
                "description": "Minimize the peak muscle power",
                "power": "max",
                "state": "pow",
            },
            "minimize_average_fatigue": {
                "function": self.minimize_average_fatigue,
                "index": 17,
                "latex": r"\phi_{17} = \frac{1}{n_m}\sum_{j=1}^{n_m} \mathcal{F}^{j}",
                "description": "Minimize the average muscle fatigue",
                "power": "1",
                "state": "fat",
            },
            "minimize_root_mean_square_fatigue": {
                "function": self.minimize_root_mean_square_fatigue,
                "index": 18,
                "latex": r"\phi_{18} = \left(\frac{1}{n_m}\sum_{j=1}^{n_m} (\mathcal{F}^{j})^{2}\right)^{\tfrac{1}{2}}",
                "description": "Minimize the root mean square of muscle fatigue",
                "power": "2",
                "state": "fat",
            },
            "minimize_cubic_average_fatigue": {
                "function": self.minimize_cubic_average_fatigue,
                "index": 19,
                "latex": r"\phi_{19} = \left(\frac{1}{n_m}\sum_{j=1}^{n_m} (\mathcal{F}^{j})^{3}\right)^{\tfrac{1}{3}}",
                "description": "Minimize the cubic average of muscle fatigue",
                "power": "3",
                "state": "fat",
            },
            "minimize_peak_fatigue": {
                "function": self.minimize_peak_fatigue,
                "index": 20,
                "latex": r"\phi_{20} = \max_{j=1,\ldots,n_m} \; \mathcal{F}^{j}",
                "description": "Minimize the peak muscle fatigue",
                "power": "max",
                "state": "fat",
            },
            "minimize_peak": {
                "function": self.minimize_peak,
                "index": 99,
                "latex": r"\phi_{99} = \max_{j=1,\ldots,n_m} \; \mathcal{Var}^{j}",
                "description": "Minimize the peak of a variable",
            },
        }

    # --- Electrical stimulation cost functions --- #
    @staticmethod
    def minimize_average_activation(controller: PenaltyController) -> MX:
        """
        Minimize the average muscle activation.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The average of muscle activation
        """
        muscle_name_list = controller.model.bio_model.muscle_names
        if isinstance(controller.model.muscles_dynamics_model[0], DingModelPulseWidthFrequency):
            stim_charge = vertcat(
                *[
                    (
                        controller.controls["last_pulse_width_" + muscle_name_list[x]].cx
                        - controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].min[0][0]
                    )
                    / (
                        controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].max[0][0]
                        - controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].min[0][0]
                    )
                    for x in range(len(muscle_name_list))
                ]
            )
        else:
            raise NotImplementedError(
                "Minimizing average activation is only implemented for DingModelPulseWidthFrequency."
            )

        return sum1(stim_charge) / len(muscle_name_list)

    @staticmethod
    def minimize_root_mean_square_activation(controller: PenaltyController) -> MX:
        """
        Minimize the root-mean-square of muscle activation.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The root-mean-square of muscle activation
        """
        eps = 1e-8
        muscle_name_list = controller.model.bio_model.muscle_names
        if isinstance(controller.model.muscles_dynamics_model[0], DingModelPulseWidthFrequency):
            stim_charge = vertcat(
                *[
                    (
                        (
                            controller.controls["last_pulse_width_" + muscle_name_list[x]].cx
                            - controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].min[0][0]
                        )
                        / (
                            controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].max[0][0]
                            - controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].min[0][0]
                        )
                    )
                    ** 2
                    for x in range(len(muscle_name_list))
                ]
            )
        else:
            raise NotImplementedError(
                "Minimizing average activation is only implemented for DingModelPulseWidthFrequency."
            )

        rms_activation = (sum1(stim_charge) / len(muscle_name_list) + eps) ** 0.5
        return rms_activation

    @staticmethod
    def minimize_cubic_average_activation(controller: PenaltyController) -> MX:
        """
        Minimize the cubic average of muscle activation.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The cubic average of muscle activation
        """
        eps = 1e-8
        muscle_name_list = controller.model.bio_model.muscle_names
        if isinstance(controller.model.muscles_dynamics_model[0], DingModelPulseWidthFrequency):
            stim_charge = vertcat(
                *[
                    (
                        (
                            controller.controls["last_pulse_width_" + muscle_name_list[x]].cx
                            - controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].min[0][0]
                        )
                        / (
                            controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].max[0][0]
                            - controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].min[0][0]
                        )
                    )
                    ** 3
                    for x in range(len(muscle_name_list))
                ]
            )
        else:
            raise NotImplementedError(
                "Minimizing average activation is only implemented for DingModelPulseWidthFrequency."
            )

        x = sum1(stim_charge) / len(muscle_name_list)
        cubic_avg_activation = sign(x) * (fabs(x) + eps) ** (1 / 3)
        # cubic_avg_activation = (sum1(stim_charge) / len(muscle_name_list) + eps) ** (1/3)
        return cubic_avg_activation

    @staticmethod
    def minimize_peak_activation(controller: PenaltyController) -> MX:
        """
        Minimize the peak muscle activation.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The peak of muscle activation
        """
        muscle_name_list = controller.model.bio_model.muscle_names
        stim_activation = vertcat(
            *[
                (
                    controller.controls["last_pulse_width_" + muscle_name_list[x]].cx
                    - controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].min[0][0]
                )
                / (
                    controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].max[0][0]
                    - controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].min[0][0]
                )
                for x in range(len(muscle_name_list))
            ]
        )
        max_activation = mmax(stim_activation)
        return max_activation

    # --- Muscle force cost functions --- #
    @staticmethod
    def minimize_average_force(controller: PenaltyController) -> MX:
        """
        Minimize the average muscle force production.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The average of produced force
        """
        muscle_name_list = controller.model.bio_model.muscle_names
        muscle_force = vertcat(
            *[controller.states["F_" + muscle_name_list[x]].cx for x in range(len(muscle_name_list))]
        )
        return sum1(muscle_force) / len(muscle_name_list)

    @staticmethod
    def minimize_root_mean_square_force(controller: PenaltyController) -> MX:
        """
        Minimize the root-mean-square of muscle force production.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The root-mean-square of produced force
        """
        eps = 1e-8
        muscle_name_list = controller.model.bio_model.muscle_names
        muscle_force = vertcat(
            *[controller.states["F_" + muscle_name_list[x]].cx ** 2 for x in range(len(muscle_name_list))]
        )
        rms_force = (sum1(muscle_force) / len(muscle_name_list) + eps) ** 0.5
        return rms_force

    @staticmethod
    def minimize_cubic_average_force(controller: PenaltyController) -> MX:
        """
        Minimize the cubic average of muscle force production.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The cubic average of produced force
        """
        eps = 1e-8
        muscle_name_list = controller.model.bio_model.muscle_names
        muscle_force = vertcat(
            *[controller.states["F_" + muscle_name_list[x]].cx ** 3 for x in range(len(muscle_name_list))]
        )
        cubic_avg_force = (sum1(muscle_force) / len(muscle_name_list) + eps) ** (1 / 3)
        return cubic_avg_force

    @staticmethod
    def minimize_peak_force(controller: PenaltyController) -> MX:
        """
        Minimize the peak muscle force production.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The peak of produced force
        """
        muscle_name_list = controller.model.bio_model.muscle_names
        muscle_force = vertcat(
            *[controller.states["F_" + muscle_name_list[x]].cx for x in range(len(muscle_name_list))]
        )
        max_force = mmax(muscle_force)
        return max_force

    # --- Muscle stress cost functions --- #
    @staticmethod
    def minimize_average_muscle_stress(controller: PenaltyController) -> MX:
        """
        Minimize the average muscle stress.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The average of muscle stress
        """
        muscle_name_list = controller.model.bio_model.muscle_names
        muscle_stress = vertcat(
            *[
                controller.states["F_" + muscle_name_list[x]].cx / controller.model.muscles_dynamics_model[x].pcsa
                for x in range(len(muscle_name_list))
            ]
        )
        return sum1(muscle_stress) / len(muscle_name_list)

    @staticmethod
    def minimize_root_mean_square_muscle_stress(controller: PenaltyController) -> MX:
        """
        Minimize the root-mean-square of muscle stress.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The root-mean-square of muscle stress
        """
        eps = 1e-8
        muscle_name_list = controller.model.bio_model.muscle_names
        muscle_stress = vertcat(
            *[
                (controller.states["F_" + muscle_name_list[x]].cx / controller.model.muscles_dynamics_model[x].pcsa)
                ** 2
                for x in range(len(muscle_name_list))
            ]
        )
        rms_stress = (sum1(muscle_stress) / len(muscle_name_list) + eps) ** 0.5
        return rms_stress

    @staticmethod
    def minimize_cubic_average_muscle_stress(controller: PenaltyController) -> MX:
        """
        Minimize the cubic average of muscle stress.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The cubic average of muscle stress
        """
        eps = 1e-8
        muscle_name_list = controller.model.bio_model.muscle_names
        muscle_stress = vertcat(
            *[
                (controller.states["F_" + muscle_name_list[x]].cx / controller.model.muscles_dynamics_model[x].pcsa)
                ** 3
                for x in range(len(muscle_name_list))
            ]
        )
        cubic_avg_stress = (sum1(muscle_stress) / len(muscle_name_list) + eps) ** (1 / 3)
        return cubic_avg_stress

    @staticmethod
    def minimize_peak_muscle_stress(controller: PenaltyController) -> MX:
        """
        Minimize the peak muscle stress.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The peak of muscle stress
        """
        muscle_name_list = controller.model.bio_model.muscle_names
        muscle_stress = vertcat(
            *[
                controller.states["F_" + muscle_name_list[x]].cx / controller.model.muscles_dynamics_model[x].pcsa
                for x in range(len(muscle_name_list))
            ]
        )
        max_stress = mmax(muscle_stress)
        return max_stress

    # --- Muscle power cost functions --- #
    @staticmethod
    def minimize_average_muscle_power(controller: PenaltyController) -> MX:
        """
        Minimize the average of muscle power.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The average of muscle power
        """
        muscle_name_list = controller.model.bio_model.muscle_names
        muscle_velocity = controller.model.muscle_velocity()(
            controller.states["q"].cx, controller.states["qdot"].cx, controller.parameters.cx
        )
        muscle_power = vertcat(
            *[
                (controller.states["F_" + muscle_name_list[x]].cx * muscle_velocity[x])
                for x in range(len(muscle_name_list))
            ]
        )
        return sum1(muscle_power) / len(muscle_name_list)

    @staticmethod
    def minimize_root_mean_square_muscle_power(controller: PenaltyController) -> MX:
        """
        Minimize the root-mean-square of muscle power.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The root-mean-square of muscle power
        """
        eps = 1e-8
        muscle_name_list = controller.model.bio_model.muscle_names
        muscle_velocity = controller.model.muscle_velocity()(
            controller.states["q"].cx, controller.states["qdot"].cx, controller.parameters.cx
        )
        muscle_power = vertcat(
            *[
                (controller.states["F_" + muscle_name_list[x]].cx * muscle_velocity[x]) ** 2
                for x in range(len(muscle_name_list))
            ]
        )
        rms_power = (sum1(muscle_power) / len(muscle_name_list) + eps) ** 0.5
        return rms_power

    @staticmethod
    def minimize_cubic_average_muscle_power(controller: PenaltyController) -> MX:
        """
        Minimize the cubic average of muscle power.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The cubic average of muscle power
        """
        eps = 1e-8
        muscle_name_list = controller.model.bio_model.muscle_names
        muscle_velocity = controller.model.muscle_velocity()(
            controller.states["q"].cx, controller.states["qdot"].cx, controller.parameters.cx
        )
        muscle_power = vertcat(
            *[
                ((controller.states["F_" + muscle_name_list[x]].cx * muscle_velocity[x]) ** 2 + 1e-6) ** 1.5
                for x in range(len(muscle_name_list))
            ]
        )
        cubic_avg_power = ((sum1(muscle_power)) / len(muscle_name_list) + eps) ** (1 / 3)
        return cubic_avg_power

    @staticmethod
    def minimize_peak_muscle_power(controller: PenaltyController) -> MX:
        """
        Minimize the peak muscle power.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The peak of muscle power
        """
        muscle_name_list = controller.model.bio_model.muscle_names
        muscle_velocity = controller.model.muscle_velocity()(
            controller.states["q"].cx, controller.states["qdot"].cx, controller.parameters.cx
        )
        muscle_power = vertcat(
            *[
                (controller.states["F_" + muscle_name_list[x]].cx * muscle_velocity[x])
                for x in range(len(muscle_name_list))
            ]
        )
        max_power = mmax(muscle_power)
        return max_power

    # --- Muscle fatigue cost functions --- #
    @staticmethod
    def minimize_average_fatigue(controller: PenaltyController) -> MX:
        """
        Minimize the average muscle fatigue.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The average of muscle fatigue
        """
        muscle_name_list = controller.model.bio_model.muscle_names
        muscle_fatigue = vertcat(
            *[
                controller.model.muscles_dynamics_model[x].a_scale - controller.states["A_" + muscle_name_list[x]].cx
                for x in range(len(muscle_name_list))
            ]
        )
        return sum1(muscle_fatigue) / len(muscle_name_list)

    @staticmethod
    def minimize_root_mean_square_fatigue(controller: PenaltyController) -> MX:
        """
        Minimize the root-mean-square of muscle fatigue.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The root-mean-square of muscle fatigue
        """
        eps = 1e-8
        muscle_name_list = controller.model.bio_model.muscle_names
        muscle_fatigue = vertcat(
            *[
                (controller.model.muscles_dynamics_model[x].a_scale - controller.states["A_" + muscle_name_list[x]].cx)
                ** 2
                for x in range(len(muscle_name_list))
            ]
        )
        rms_fatigue = (sum1(muscle_fatigue) / len(muscle_name_list) + eps) ** 0.5
        return rms_fatigue

    @staticmethod
    def minimize_cubic_average_fatigue(controller: PenaltyController) -> MX:
        """
        Minimize the cubic average of muscle fatigue.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The cubic average of muscle fatigue
        """
        eps = 1e-8
        muscle_name_list = controller.model.bio_model.muscle_names
        muscle_fatigue = vertcat(
            *[
                (controller.model.muscles_dynamics_model[x].a_scale - controller.states["A_" + muscle_name_list[x]].cx)
                ** 3
                for x in range(len(muscle_name_list))
            ]
        )
        cubic_avg_fatigue = (sum1(muscle_fatigue) / len(muscle_name_list) + eps) ** (1 / 3)
        return cubic_avg_fatigue

    @staticmethod
    def minimize_peak_fatigue(controller: PenaltyController) -> MX:
        """
        Minimize the peak muscle fatigue.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The peak of muscle fatigue
        """
        muscle_name_list = controller.model.bio_model.muscle_names
        muscle_fatigue = vertcat(
            *[
                controller.model.muscles_dynamics_model[x].a_scale - controller.states["A_" + muscle_name_list[x]].cx
                for x in range(len(muscle_name_list))
            ]
        )
        max_fatigue = mmax(muscle_fatigue)
        return max_fatigue

    # --- Peak cost function and constraint used in OCP --- #
    @staticmethod
    def minimize_peak(controller: PenaltyController) -> MX:
        """
        Minimize the peak of a variable.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The peak of a variable
        """
        return controller.parameters["minmax_param"].cx

    @staticmethod
    def constraints_minmax(controller: PenaltyController, obj_fun_key: str, param_index: int) -> MX:
        muscle_name_list = controller.model.bio_model.muscle_names

        if obj_fun_key == ["minimize_peak_force"]:
            value = controller.states["F_" + muscle_name_list[param_index]].cx

        elif obj_fun_key == ["minimize_peak_activation"]:
            value = (
                controller.controls["last_pulse_width_" + muscle_name_list[param_index]].cx
                - controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[param_index]].min[0][0]
            ) / (
                controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[param_index]].max[0][0]
                - controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[param_index]].min[0][0]
            )

        elif obj_fun_key == ["minimize_peak_muscle_stress"]:
            value = (
                controller.states["F_" + muscle_name_list[param_index]].cx
                / controller.model.muscles_dynamics_model[param_index].pcsa
            )

        elif obj_fun_key == ["minimize_peak_fatigue"]:
            value = (
                controller.model.muscles_dynamics_model[param_index].a_scale
                - controller.states["A_" + muscle_name_list[param_index]].cx
            )

        else:
            raise NotImplementedError(f"The cost function {obj_fun_key}, is not implementend in minmax")

        return controller.parameters["minmax_param"].cx[controller.node_index] - value
