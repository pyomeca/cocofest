from casadi import MX, vertcat, sum1, mmax, fabs, sign, tanh
from bioptim import PenaltyController
from cocofest.models.ding2007.ding2007 import DingModelPulseWidthFrequency

class CustomCostFunctions:
    def __init__(self):
        self.dict_functions = {
            "minimize_average_activation": {
                "function": self.minimize_average_activation,
                "index": 1,
                "latex": r"\phi_1 = \frac{1}{M}\sum_{m=1}^{M} a^{m}, \quad a^{m}=\frac{f^{m}-f^{m}_{\min}}{f^{m}_{\max}-f^{m}_{\min}}",
                "description": "Minimize the average muscle activation",
                "power": "1",
                "state": "pw",
            },
            "minimize_root_mean_square_activation": {
                "function": self.minimize_root_mean_square_activation,
                "index": 2,
                "latex": r"\phi_2 = \left(\frac{1}{M}\sum_{m=1}^{M} (a^{m})^{2}\right)^{\tfrac{1}{2}}, \quad a^{m}=\frac{f^{m}-f^{m}_{\min}}{f^{m}_{\max}-f^{m}_{\min}}",
                "description": "Minimize the root mean square of muscle activation",
                "power": "2",
                "state": "pw",
            },
            "minimize_cubic_average_activation": {
                "function": self.minimize_cubic_average_activation,
                "index": 3,
                "latex": r"\phi_3 = \left(\frac{1}{M}\sum_{m=1}^{M} (a^{m})^{3}\right)^{\tfrac{1}{3}}, \quad a^{m}=\frac{f^{m}-f^{m}_{\min}}{f^{m}_{\max}-f^{m}_{\min}}",
                "description": "Minimize the cubic average of muscle activation",
                "power": "3",
                "state": "pw",
            },
            "minimize_peak_activation": {
                "function": self.minimize_peak_activation,
                "index": 4,
                "latex": r"\phi_4 = \max_{m=1,\ldots,M} \; a^{m}, \quad a^{m}=\frac{f^{m}-f^{m}_{\min}}{f^{m}_{\max}-f^{m}_{\min}}",
                "description": "Minimize the peak of muscle activation",
                "power": "max",
                "state": "pw",
            },
            "minimize_average_force": {
                "function": self.minimize_average_force,
                "index": 5,
                "latex": r"\phi_5 = \frac{1}{M}\sum_{m=1}^{M} f^{m}",
                "description": "Minimize the average muscle force",
                "power": "1",
                "state": "f",
            },
            "minimize_root_mean_square_force": {
                "function": self.minimize_root_mean_square_force,
                "index": 6,
                "latex": r"\phi_6 = \left(\frac{1}{M}\sum_{m=1}^{M} (f^{m})^{2}\right)^{\tfrac{1}{2}}",
                "description": "Minimize the root mean square of muscle force",
                "power": "2",
                "state": "f",
            },
            "minimize_cubic_average_force": {
                "function": self.minimize_cubic_average_force,
                "index": 7,
                "latex": r"\phi_7 = \left(\frac{1}{M}\sum_{m=1}^{M} (f^{m})^{3}\right)^{\tfrac{1}{3}}",
                "description": "Minimize the cubic average of muscle force",
                "power": "3",
                "state": "f",
            },
            "minimize_peak_force": {
                "function": self.minimize_peak_force,
                "index": 8,
                "latex": r"\phi_8 = \max_{m=1,\ldots,M} \; f^{m}",
                "description": "Minimize the peak muscle force",
                "power": "max",
                "state": "f",
            },
            "minimize_average_muscle_stress": {
                "function": self.minimize_average_muscle_stress,
                "index": 9,
                "latex": r"\phi_9 = \frac{1}{M}\sum_{m=1}^{M} \frac{f^{m}}{S^{m}}",
                "description":"Minimize the average muscle stress",
                "power": "1",
                "state": "str",
            },
            "minimize_root_mean_square_muscle_stress": {
                "function": self.minimize_root_mean_square_muscle_stress,
                "index": 10,
                "latex": r"\phi_{10} = \left(\frac{1}{M}\sum_{m=1}^{M} \left(\frac{f^{m}}{S^{m}}\right)^{2}\right)^{\tfrac{1}{2}}",
                "description":"Minimize the root mean square of muscle stress",
                "power": "2",
                "state": "str",
            },
            "minimize_cubic_average_muscle_stress": {
                "function": self.minimize_cubic_average_muscle_stress,
                "index": 11,
                "latex": r"\phi_{11} = \left(\frac{1}{M}\sum_{m=1}^{M} \left(\frac{f^{m}}{S^{m}}\right)^{3}\right)^{\tfrac{1}{3}}",
                "description":"Minimize the cubic average of muscle stress",
                "power": "3",
                "state": "str",
            },
            "minimize_peak_muscle_stress": {
                "function": self.minimize_peak_muscle_stress,
                "index": 12,
                "latex": r"\phi_{12} = \max_{m=1,\ldots,M} \; \frac{f^{m}}{S^{m}}",
                "description":"Minimize the peak muscle stress",
                "power": "max",
                "state": "str",
            },
            "minimize_average_fatigue": {
                "function": self.minimize_average_fatigue,
                "index": 13,
                "latex": r"\phi_{13} = \frac{1}{M}\sum_{m=1}^{M} \mathcal{F}^{m}",
                "description": "Minimize the average muscle fatigue",
                "power": "1",
                "state": "fat",
            },
            "minimize_root_mean_square_fatigue": {
                "function": self.minimize_root_mean_square_fatigue,
                "index": 14,
                "latex": r"\phi_{14} = \left(\frac{1}{M}\sum_{m=1}^{M} (\mathcal{F}^{m})^{2}\right)^{\tfrac{1}{2}}",
                "description": "Minimize the root mean square of muscle fatigue",
                "power": "2",
                "state": "fat",
            },
            "minimize_cubic_average_fatigue": {
                "function": self.minimize_cubic_average_fatigue,
                "index": 15,
                "latex": r"\phi_{15} = \left(\frac{1}{M}\sum_{m=1}^{M} (\mathcal{F}^{m})^{3}\right)^{\tfrac{1}{3}}",
                "description": "Minimize the cubic average of muscle fatigue",
                "power": "3",
                "state": "fat",
            },
            "minimize_peak_fatigue": {
                "function": self.minimize_peak_fatigue,
                "index": 16,
                "latex": r"\phi_{16} = \max_{m=1,\ldots,M} \; \mathcal{F}^{m}",
                "description": "Minimize the peak muscle fatigue",
                "power": "max",
                "state": "fat",
            },
            "minimize_root_mean_square_muscle_power": {
                "function": self.minimize_root_mean_square_muscle_power,
                "index": 17,
                "latex": r"\phi_{17} = \left(\frac{1}{M}\sum_{m=1}^{M} (f^{m} v^{m})^{2}\right)^{\tfrac{1}{2}}",
                "description": "Minimize the root mean square of muscle power",
                "power": "2",
                "state": "pow",
            },

            # --- Custom cost functions --- #
            "minimize_root_mean_square_scalable_fatigue_decay": {
                "function": self.minimize_root_mean_square_scalable_fatigue_decay,
                "index": 20,
                "latex": r"\phi_{20} = \left(\frac{1}{M}\sum_{t=1}^{M}\left(\frac{A_{t+1} - A_t}{A_t - A_{\text{end}}}\right)^{2}\right)^{\tfrac{1}{2}}",
                "description": "Minimize the root mean square of scalable muscle fatigue decay",
                "power": "2",
                "state": "fats",
            },
            "minimize_root_mean_square_fatigue_decay": {
                "function": self.minimize_root_mean_square_fatigue_decay,
                "index": 21,
                "latex": r"\phi_{21} = \left(\frac{1}{M}\sum_{t=1}^{M}\left(\frac{A_{t+1} - A_t}{-A_{\text{rest}} / alpha_a}\right)^{2}\right)^{\tfrac{1}{2}}",
                "description": "Minimize the root mean square of muscle fatigue decay",
                "power": "2",
                "state": "fatd",
            },
            "minimize_peak_fatigue_decay": {
                "function": self.minimize_peak_fatigue_decay,
                "index": 22,
                "latex": r"\phi_{22} = \max_{m=1,\ldots,M} \left( \frac{A_{t+1}^{(m)} - A_t^{(m)}}{-A_{\text{rest}}^{(m)} / \alpha_a} \right)",
                "description": "Minimize the peak muscle fatigue decay",
                "power": "max",
                "state": "fatd",
            },

            "minimize_root_mean_square_tanh_fatigue_decay": {
                "function": self.minimize_root_mean_square_tanh_fatigue_decay,
                "index": 23,
                "latex": r"\phi_{23} = \left(1,\ \left(\frac{1}{n_m}\sum_{t=1}^{n_m}\left(\frac{1+\tanh\!\left(A_{m,t}-A_{m,t+1}\right)}{A_{m,\text{rest}}/(-\alpha_{A_m})}\right)^{2}\right)^{\tfrac{1}{2}}\right)",
                "description": "Minimize the root mean square of scalable muscle fatigue decay",
                "power": "2",
                "state": "fatdtanh",
            },

            "minimize_root_mean_square_weighted_tanh_fatigue_decay": {
                "function": self.minimize_root_mean_square_weighted_tanh_fatigue_decay,
                "index": 24,
                "latex": r"\phi_{24} = \left(1,\ \left(\frac{1}{n_m}\sum_{t=1}^{n_m}\omega_{A_m}\left(1+\tanh\!\left(A_{m,t}-A_{m,t+1}\right)\right)^{2}\right)^{\tfrac{1}{2}}\right)",
                "description": "Minimize the root mean square of scalable muscle fatigue decay",
                "power": "2",
                "state": "fatdwtanh",
            },

            "minimize_peak": {
                "function": self.minimize_peak,
                "index": 99,
                "latex": r"\phi_{99} = \max_{m=1,\ldots,M} \; \mathcal{Var}^{m}",
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
                    (controller.controls["last_pulse_width_" + muscle_name_list[x]].cx - controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].min[0][0])
                    / (controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].max[0][0] - controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].min[0][0])
                    for x in range(len(muscle_name_list))
                ]
            )
        else:
            raise NotImplementedError("Minimizing average activation is only implemented for DingModelPulseWidthFrequency.")

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
                    ((controller.controls["last_pulse_width_" + muscle_name_list[x]].cx -
                     controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].min[0][0])
                    / (controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].max[0][0] -
                       controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].min[0][0])) ** 2
                    for x in range(len(muscle_name_list))
                ]
            )
        else:
            raise NotImplementedError(
                "Minimizing average activation is only implemented for DingModelPulseWidthFrequency.")

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
                    ((controller.controls["last_pulse_width_" + muscle_name_list[x]].cx -
                      controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].min[0][0])
                     / (controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].max[0][0] -
                        controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].min[0][0])) ** 3
                    for x in range(len(muscle_name_list))
                ]
            )
        else:
            raise NotImplementedError(
                "Minimizing average activation is only implemented for DingModelPulseWidthFrequency.")

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
                (controller.controls["last_pulse_width_" + muscle_name_list[x]].cx -
                 controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].min[0][0])
                / (controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].max[0][0] -
                   controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].min[0][0])
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
            *[
                controller.states["F_" + muscle_name_list[x]].cx
                for x in range(len(muscle_name_list))
            ]
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
            *[
                controller.states["F_" + muscle_name_list[x]].cx ** 2
                for x in range(len(muscle_name_list))
            ]
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
            *[
                controller.states["F_" + muscle_name_list[x]].cx ** 3
                for x in range(len(muscle_name_list))
            ]
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
            *[
                controller.states["F_" + muscle_name_list[x]].cx
                for x in range(len(muscle_name_list))
            ]
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
                (controller.states["F_" + muscle_name_list[x]].cx / controller.model.muscles_dynamics_model[x].pcsa) ** 2
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
                (controller.states["F_" + muscle_name_list[x]].cx / controller.model.muscles_dynamics_model[
                    x].pcsa) ** 3
                for x in range(len(muscle_name_list))
            ]
        )
        cubic_avg_stress = (sum1(muscle_stress) / len(muscle_name_list) + eps) ** (1/3)
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
                (controller.model.muscles_dynamics_model[x].a_scale - controller.states["A_" + muscle_name_list[x]].cx) ** 2
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
                (controller.model.muscles_dynamics_model[x].a_scale - controller.states["A_" + muscle_name_list[x]].cx) ** 3
                for x in range(len(muscle_name_list))
            ]
        )
        cubic_avg_fatigue = (sum1(muscle_fatigue) / len(muscle_name_list) + eps) ** (1/3)
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

    # --- Muscle power cost functions --- #
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

    # --- Custom cost functions --- #
    @staticmethod
    def minimize_root_mean_square_scalable_fatigue_decay(controller: PenaltyController) -> MX:
        """
        Minimize the root-mean-square fatigue decay in a scalable way.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The root-mean-square fatigue decay
        """
        muscle_name_list = controller.model.bio_model.muscle_names
        A_t = vertcat(*[controller.states["A_" + muscle_name_list[x]].cx for x in range(len(muscle_name_list))])
        A_t_plus_one = CustomCostFunctions.compute_A_t_plus_one(controller)

        # Optimized elsewhere
        A_end = [297.59, 226.84, 1191.58, 89.46]

        eps = 1e-8
        muscle_fatigue_decay = vertcat(
            *[
                ((A_t[x] - A_t_plus_one[x]) / (1 + ((A_t[x] - A_end[x])/A_end[x]))) ** 2
                for x in range(len(muscle_name_list))
            ]
        )
        rms_fatigue = (sum1(muscle_fatigue_decay) / len(muscle_name_list) + eps) ** 0.5
        return rms_fatigue

    @staticmethod
    def minimize_root_mean_square_fatigue_decay(controller: PenaltyController) -> MX:
        """
        Minimize the root-mean-square fatigue decay.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The root-mean-square fatigue decay
        """
        muscle_name_list = controller.model.bio_model.muscle_names
        A_rest = vertcat(*[controller.model.muscles_dynamics_model[x].a_scale for x in range(len(muscle_name_list))])
        alpha_a = vertcat(*[controller.model.muscles_dynamics_model[x].alpha_a for x in range(len(muscle_name_list))])
        A_t = vertcat(*[controller.states["A_" + muscle_name_list[x]].cx for x in range(len(muscle_name_list))])
        A_t_plus_one = CustomCostFunctions.compute_A_t_plus_one(controller)

        eps = 1e-8
        muscle_fatigue_decay = vertcat(
            *[
                ((1 + (A_t[x] - A_t_plus_one[x])) / (1/(-(A_rest[x] / alpha_a[x])))) ** 2
                for x in range(len(muscle_name_list))
            ]
        )
        rms_fatigue = (sum1(muscle_fatigue_decay) / len(muscle_name_list) + eps) ** 0.5
        return rms_fatigue

    @staticmethod
    def minimize_peak_fatigue_decay(controller: PenaltyController) -> MX:
        """
        Minimize the peak fatigue decay.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The peak fatigue decay
        """
        muscle_name_list = controller.model.bio_model.muscle_names
        A_rest = vertcat(*[controller.model.muscles_dynamics_model[x].a_scale for x in range(len(muscle_name_list))])
        alpha_a = vertcat(*[controller.model.muscles_dynamics_model[x].alpha_a for x in range(len(muscle_name_list))])
        A_t = vertcat(*[controller.states["A_" + muscle_name_list[x]].cx for x in range(len(muscle_name_list))])
        A_t_plus_one = CustomCostFunctions.compute_A_t_plus_one(controller)

        muscle_fatigue_decay = vertcat(
            *[
                ((1 + (A_t[x] - A_t_plus_one[x])) / (1/(-(A_rest[x] / alpha_a[x]))))
                for x in range(len(muscle_name_list))
            ]
        )
        max_fatigue_decay = mmax(muscle_fatigue_decay)

        return max_fatigue_decay

    @staticmethod
    def minimize_root_mean_square_tanh_fatigue_decay(controller: PenaltyController) -> MX:
        """
        Minimize the root-mean-square fatigue decay in a hyperbolic tangential way.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The root-mean-square fatigue decay in a hyperbolic tangential way
        """
        muscle_name_list = controller.model.bio_model.muscle_names
        A_t = vertcat(*[controller.states["A_" + muscle_name_list[x]].cx for x in range(len(muscle_name_list))])
        A_t_plus_one = CustomCostFunctions.compute_A_t_plus_one(controller)

        eps = 1e-8
        muscle_fatigue_decay = vertcat(
            *[
                (1 + tanh(10 * (A_t[x] - A_t_plus_one[x]))) ** 2
                for x in range(len(muscle_name_list))
            ]
        )
        rms_fatigue = (sum1(muscle_fatigue_decay) / len(muscle_name_list) + eps) ** 0.5
        return rms_fatigue

    @staticmethod
    def minimize_root_mean_square_weighted_tanh_fatigue_decay(controller: PenaltyController) -> MX:
        """
        Minimize the root-mean-square fatigue decay in a hyperbolic tangential way weighted by A_rest and alpha_A.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The root-mean-square fatigue decay in a hyperbolic tangential way weighted by A_rest and alpha_A
        """
        muscle_name_list = controller.model.bio_model.muscle_names
        A_rest = vertcat(*[controller.model.muscles_dynamics_model[x].a_scale for x in range(len(muscle_name_list))])
        alpha_a = vertcat(*[controller.model.muscles_dynamics_model[x].alpha_a for x in range(len(muscle_name_list))])
        A_t = vertcat(*[controller.states["A_" + muscle_name_list[x]].cx for x in range(len(muscle_name_list))])
        A_t_plus_one = CustomCostFunctions.compute_A_t_plus_one(controller)

        eps = 1e-8
        muscle_fatigue_decay = vertcat(
            *[
                ((1 + tanh(10 * (A_t[x] - A_t_plus_one[x]))) / (1/(-(A_rest[x] / alpha_a[x])))) ** 2
                for x in range(len(muscle_name_list))
            ]
        )
        rms_fatigue = (sum1(muscle_fatigue_decay) / len(muscle_name_list) + eps) ** 0.5
        return rms_fatigue


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
    def constraints_minmax(controller: PenaltyController, obj_fun_key: str, param_index:int) -> MX:
        muscle_name_list = controller.model.bio_model.muscle_names

        if obj_fun_key == ["minimize_peak_force"]:
            value = controller.states["F_" + muscle_name_list[param_index]].cx

        elif obj_fun_key == ["minimize_peak_activation"]:
            value = ((controller.controls["last_pulse_width_" + muscle_name_list[param_index]].cx -
                     controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[param_index]].min[0][0])
            / (controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[param_index]].max[0][0] -
               controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[param_index]].min[0][0]))

        elif obj_fun_key == ["minimize_peak_muscle_stress"]:
            value = controller.states["F_" + muscle_name_list[param_index]].cx / controller.model.muscles_dynamics_model[param_index].pcsa

        elif obj_fun_key == ["minimize_peak_fatigue"]:
            value = controller.model.muscles_dynamics_model[param_index].a_scale - controller.states["A_" + muscle_name_list[param_index]].cx

        elif obj_fun_key == ["minimize_peak_fatigue_decay"]:
            A_rest = controller.model.muscles_dynamics_model[param_index].a_scale
            tau_fat = controller.model.muscles_dynamics_model[param_index].tau_fat
            alpha_a = controller.model.muscles_dynamics_model[param_index].alpha_a

            # At time t or t+1
            A_t = controller.states["A_" + muscle_name_list[param_index]].cx
            F_t = controller.states["F_" + muscle_name_list[param_index]].cx
            A_t_plus_one = A_t - (((A_t - A_rest) / tau_fat) + (alpha_a * F_t))

            value = (A_t - A_t_plus_one) / -(A_rest / alpha_a)

        else:
            raise NotImplementedError(f"The cost function {obj_fun_key}, is not implementend in minmax")

        return controller.parameters["minmax_param"].cx[controller.node_index] - value

    # --- A_t+1 cost function used in OCP --- #
    @staticmethod
    def compute_A_t_plus_one(controller: PenaltyController) -> MX:
        """
        Compute A_t+1 based on the fatigue model.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The value of A_t+1
        """
        muscle_name_list = controller.model.bio_model.muscle_names

        # Known form model
        A_rest = vertcat(
            *[controller.model.muscles_dynamics_model[x].a_scale for x in range(len(muscle_name_list))])
        tau_fat = vertcat(
            *[controller.model.muscles_dynamics_model[x].tau_fat for x in range(len(muscle_name_list))])
        alpha_a = vertcat(
            *[controller.model.muscles_dynamics_model[x].alpha_a for x in range(len(muscle_name_list))])

        # At time t
        A_t = vertcat(*[controller.states["A_" + muscle_name_list[x]].cx for x in range(len(muscle_name_list))])
        F_t = vertcat(*[controller.states["F_" + muscle_name_list[x]].cx for x in range(len(muscle_name_list))])

        A_t_plus_one = vertcat(
            *[A_t[x] - ((A_t[x] - A_rest[x]) / tau_fat[x] + alpha_a[x] * F_t[x]) for x in
              range(len(muscle_name_list))])

        return A_t_plus_one