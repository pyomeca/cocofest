from casadi import MX, vertcat, sum1, mmax, fabs, sign, tanh, if_else, log, exp, DM, dot
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
                "power": r"\infty",
                "state": "pw",
            },
            "minimize_average_force": {
                "function": self.minimize_average_force,
                "index": 5,
                "latex": r"\phi_5 = \frac{1}{M}\sum_{m=1}^{M} f^{m}",
                "description": "Minimize the average muscle force",
                "power": "1",
                "state": r"F^{m}",
            },
            "minimize_root_mean_square_force": {
                "function": self.minimize_root_mean_square_force,
                "index": 6,
                "latex": r"\phi_6 = \left(\frac{1}{M}\sum_{m=1}^{M} (f^{m})^{2}\right)^{\tfrac{1}{2}}",
                "description": "Minimize the root mean square of muscle force",
                "power": "2",
                "state": r"F^{m}",
            },
            "minimize_cubic_average_force": {
                "function": self.minimize_cubic_average_force,
                "index": 7,
                "latex": r"\phi_7 = \left(\frac{1}{M}\sum_{m=1}^{M} (f^{m})^{3}\right)^{\tfrac{1}{3}}",
                "description": "Minimize the cubic average of muscle force",
                "power": "3",
                "state": r"F^{m}",
            },
            "minimize_peak_force": {
                "function": self.minimize_peak_force,
                "index": 8,
                "latex": r"\phi_8 = \max_{m=1,\ldots,M} \; f^{m}",
                "description": "Minimize the peak muscle force",
                "power": r"\infty",
                "state": r"F^{m}",
            },
            "minimize_average_muscle_stress": {
                "function": self.minimize_average_muscle_stress,
                "index": 9,
                "latex": r"\phi_9 = \frac{1}{M}\sum_{m=1}^{M} \frac{f^{m}}{S^{m}}",
                "description":"Minimize the average muscle stress",
                "power": "1",
                "state": r"\sigma",
            },
            "minimize_root_mean_square_muscle_stress": {
                "function": self.minimize_root_mean_square_muscle_stress,
                "index": 10,
                "latex": r"\phi_{10} = \left(\frac{1}{M}\sum_{m=1}^{M} \left(\frac{f^{m}}{S^{m}}\right)^{2}\right)^{\tfrac{1}{2}}",
                "description":"Minimize the root mean square of muscle stress",
                "power": "2",
                "state": r"\sigma",
            },
            "minimize_cubic_average_muscle_stress": {
                "function": self.minimize_cubic_average_muscle_stress,
                "index": 11,
                "latex": r"\phi_{11} = \left(\frac{1}{M}\sum_{m=1}^{M} \left(\frac{f^{m}}{S^{m}}\right)^{3}\right)^{\tfrac{1}{3}}",
                "description":"Minimize the cubic average of muscle stress",
                "power": "3",
                "state": r"\sigma",
            },
            "minimize_peak_muscle_stress": {
                "function": self.minimize_peak_muscle_stress,
                "index": 12,
                "latex": r"\phi_{12} = \max_{m=1,\ldots,M} \; \frac{f^{m}}{S^{m}}",
                "description":"Minimize the peak muscle stress",
                "power": r"\infty",
                "state": r"\sigma",
            },
            "minimize_average_fatigue": {
                "function": self.minimize_average_fatigue,
                "index": 13,
                "latex": r"\phi_{13} = \frac{1}{M}\sum_{m=1}^{M} \mathcal{F}^{m}",
                "description": "Minimize the average muscle fatigue",
                "power": "1",
                "state": "A",
            },
            "minimize_root_mean_square_fatigue": {
                "function": self.minimize_root_mean_square_fatigue,
                "index": 14,
                "latex": r"\phi_{14} = \left(\frac{1}{M}\sum_{m=1}^{M} (\mathcal{F}^{m})^{2}\right)^{\tfrac{1}{2}}",
                "description": "Minimize the root mean square of muscle fatigue",
                "power": "2",
                "state": "A",
            },
            "minimize_cubic_average_fatigue": {
                "function": self.minimize_cubic_average_fatigue,
                "index": 15,
                "latex": r"\phi_{15} = \left(\frac{1}{M}\sum_{m=1}^{M} (\mathcal{F}^{m})^{3}\right)^{\tfrac{1}{3}}",
                "description": "Minimize the cubic average of muscle fatigue",
                "power": "3",
                "state": "A",
            },
            "minimize_peak_fatigue": {
                "function": self.minimize_peak_fatigue,
                "index": 16,
                "latex": r"\phi_{16} = \max_{m=1,\ldots,M} \; \mathcal{F}^{m}",
                "description": "Minimize the peak muscle fatigue",
                "power": r"\infty",
                "state": "A",
            },
            "minimize_root_mean_square_muscle_power": {
                "function": self.minimize_root_mean_square_muscle_power,
                "index": 17,
                "latex": r"\phi_{17} = \left(\frac{1}{M}\sum_{m=1}^{M} (f^{m} v^{m})^{2}\right)^{\tfrac{1}{2}}",
                "description": "Minimize the root mean square of muscle power",
                "power": "2",
                "state": "W",
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
            "minimize_fatigue_decay": {
                "function": self.minimize_fatigue_decay,
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

            "minimize_failure_point": {
                "function": self.minimize_failure_point,
                "index": 25,
                # "latex": r"\phi_{25} = \left(1,\ \left(\frac{1}{n_m}\sum_{t=1}^{n_m}\omega_{A_m}\left(1+\tanh\!\left(A_{m,t}-A_{m,t+1}\right)\right)^{2}\right)^{\tfrac{1}{2}}\right)",
                "description": "Minimize the barrier task failure point",
                "power": "1",
                "state": "barrier",
            },


            "minimize_root_mean_square_tanh_fatigue_decay_norm": {
                "function": self.minimize_root_mean_square_tanh_fatigue_decay_norm,
                "index": 26,
                "latex": r"\phi_{23} = \left(1,\ \left(\frac{1}{n_m}\sum_{t=1}^{n_m}\left(\frac{1+\tanh\!\left(A_{m,t}-A_{m,t+1}\right)}{A_{m,\text{rest}}/(-\alpha_{A_m})}\right)^{2}\right)^{\tfrac{1}{2}}\right)",
                "description": "Minimize the root mean square of scalable muscle fatigue decay",
                "power": "2",
                "state": "fatdtanhmul",
            },

            "minimize_average_tanh_fatigue_decay": {
                "function": self.minimize_average_tanh_fatigue_decay,
                "index": 27,
                "latex": r"\phi_{23} = \left(1,\ \left(\frac{1}{n_m}\sum_{t=1}^{n_m}\left(\frac{1+\tanh\!\left(A_{m,t}-A_{m,t+1}\right)}{A_{m,\text{rest}}/(-\alpha_{A_m})}\right)^{2}\right)^{\tfrac{1}{2}}\right)",
                "description": "Minimize the root mean square of scalable muscle fatigue decay",
                "power": "2",
                "state": "fatdtanhmul",
            },

            "minimize_rms_tanh_fatigue_decay": {
                "function": self.minimize_rms_tanh_fatigue_decay,
                "index": 28,
                "latex": r"\phi_{23} = \left(1,\ \left(\frac{1}{n_m}\sum_{t=1}^{n_m}\left(\frac{1+\tanh\!\left(A_{m,t}-A_{m,t+1}\right)}{A_{m,\text{rest}}/(-\alpha_{A_m})}\right)^{2}\right)^{\tfrac{1}{2}}\right)",
                "description": "Minimize the root mean square of scalable muscle fatigue decay",
                "power": "2",
                "state": "fatdtanhmul",
            },

            "minimize_rms_tanh_fatigue_decay_new": {
                "function": self.minimize_rms_tanh_fatigue_decay_new,
                "index": 29,
                "latex": r"\phi_{23} = \left(1,\ \left(\frac{1}{n_m}\sum_{t=1}^{n_m}\left(\frac{1+\tanh\!\left(A_{m,t}-A_{m,t+1}\right)}{A_{m,\text{rest}}/(-\alpha_{A_m})}\right)^{2}\right)^{\tfrac{1}{2}}\right)",
                "description": "Minimize the root mean square of scalable muscle fatigue decay",
                "power": "2",
                "state": "fatdtanhmul",
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
        dA = CustomCostFunctions.calculate_dA(controller)

        # Optimized elsewhere
        # A_end = [297.59, 226.84, 1191.58, 89.46]
        # A_end = [128.45242911, 175.68692114, 889.89511914, 0]  # 4438.59666597
        A_end = [128.45242911, 175.68692114, 889.89511914]

        eps = 1e-8
        muscle_fatigue_decay = vertcat(
            *[
                (dA[x] / (1 + ((A_t[x] - A_end[x])/A_end[x]))) ** 2
                for x in range(len(muscle_name_list)-1)
            ]
        )
        rms_fatigue = (sum1(muscle_fatigue_decay) / (len(muscle_name_list)-1) + eps) ** 0.5
        return rms_fatigue

    @staticmethod
    def minimize_fatigue_decay(controller: PenaltyController) -> MX:
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
        dA = CustomCostFunctions.calculate_dA(controller)

        muscle_fatigue_decay = vertcat(
            *[
                -dA[x]
                for x in range(len(muscle_name_list))
            ]
        )
        avg_fatigue = sum1(muscle_fatigue_decay) / len(muscle_name_list)
        return avg_fatigue

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
        dA = CustomCostFunctions.calculate_dA(controller)

        muscle_fatigue_decay = vertcat(
            *[
                (-dA[x])
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
        dA = CustomCostFunctions.calculate_dA(controller)

        eps = 1e-8
        muscle_fatigue_decay = vertcat(
            *[
                (1 + tanh(0.01 * (-dA[x]))) ** 2
                for x in range(len(muscle_name_list)-1)
            ]
        )

        rms_fatigue = (sum1(muscle_fatigue_decay) / (len(muscle_name_list)-1) + eps) ** 0.5
        return rms_fatigue

    @staticmethod
    def minimize_root_mean_square_tanh_fatigue_decay_norm(controller: PenaltyController) -> MX:
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
        dA = CustomCostFunctions.calculate_dA(controller)

        max_dA_fatigue = [72.2, 61.2, 85.7, 92.3]
        max_dA_recovery = [2.3, 3.0, 14.8, 35.6]
        dA_nomalized = vertcat(
            *[
                if_else(dA[x] < 0, dA[x] / max_dA_fatigue[x], dA[x] / max_dA_recovery[x])
                for x in range(len(muscle_name_list))
            ]
        )

        muscle_fatigue_decay = vertcat(
            *[
                (1 + tanh(1 * (-dA_nomalized[x]))) ** 2
                for x in range(len(muscle_name_list)-1)
            ]
        )

        eps = 1e-8
        rms_fatigue = (sum1(muscle_fatigue_decay) / (len(muscle_name_list)-1)+ eps) ** 0.5
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
        # alpha_a = vertcat(*[controller.model.muscles_dynamics_model[x].alpha_a for x in range(len(muscle_name_list))])
        dA = CustomCostFunctions.calculate_dA(controller)

        # fatigability = vertcat(*[1 / (A_rest[x] / (-alpha_a[x])) for x in range(len(muscle_name_list))])
        # l1 = sum1(fatigability)
        # weights = 1 + (fatigability / l1)

        # weights = [1.4 * 10e-1 * 342.7, 1.1 * 10e-1 * 445.5, 5.6 * 10e-2 * 179.6, 3.4 * 10e-2 * 109.1]
        max_dA_fatigue = [72.2, 61.2, 85.7, 92.3]
        max_dA_recovery = [2.3, 3.0, 14.8, 35.6]

        dA_nomalized = vertcat(
            *[
                if_else(dA[x] < 0, dA[x] / max_dA_fatigue[x], dA[x] / max_dA_recovery[x])
                for x in range(len(muscle_name_list)-1)
            ]
        )

        A_t = vertcat(*[controller.states["A_" + muscle_name_list[x]].cx for x in range(len(muscle_name_list))])
        fatigue = [((A_rest[i]-A_t[i])/A_rest[i] * 100)**2 for i in range(A_rest.shape[0])]

        muscle_fatigue_decay = vertcat(
            *[
                fatigue[x] * (1 + tanh(4 * (-dA_nomalized[x]))) ** 2
                for x in range(len(muscle_name_list)-1)
            ]
        )
        eps = 1e-8
        rms_fatigue = (sum1(muscle_fatigue_decay) / (len(muscle_name_list)-1) + eps) ** 0.5
        return rms_fatigue

    @staticmethod
    def minimize_average_tanh_fatigue_decay(controller: PenaltyController) -> MX:
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
        dA = CustomCostFunctions.calculate_dA(controller)
        A_rest = [controller.model.muscles_dynamics_model[x].a_scale for x in range(len(muscle_name_list))]
        normed_a = [A / max(A_rest) for A in A_rest]

        max_dA_fatigue = [72.2, 61.2, 85.7, 92.3]
        max_dA_recovery = [2.3, 3.0, 14.8, 35.6]
        A_min = [41, 70, 379, 932]

        with_triceps = True
        muscle_range = 4 if with_triceps else 3

        dA_nomalized = vertcat(
            *[
                if_else(dA[x] < 0, dA[x] / max_dA_fatigue[x], dA[x] / max_dA_recovery[x])
                for x in range(muscle_range)
            ]
        )

        A_t = vertcat(*[controller.states["A_" + muscle_name_list[x]].cx for x in range(len(muscle_name_list))])
        fatigue = [((A_rest[i] - A_t[i]) / (A_rest[i]-A_min[i])) for i in range(muscle_range)]

        muscle_fatigue_decay = vertcat(
            *[
                normed_a[x] * (100 * fatigue[x]) * (1 + tanh(-dA_nomalized[x]))
                for x in range(muscle_range)
            ]
        )

        avg_fatigue = sum1(muscle_fatigue_decay) / muscle_range
        return avg_fatigue

    @staticmethod
    def minimize_rms_tanh_fatigue_decay(controller: PenaltyController) -> MX:
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
        dA = CustomCostFunctions.calculate_dA(controller)

        max_dA_fatigue = [72.2, 61.2, 85.7, 92.3]
        max_dA_recovery = [2.3, 3.0, 14.8, 35.6]
        A_min = [41, 70, 379, 932]

        with_triceps = True
        muscle_range = 4 if with_triceps else 3

        dA_nomalized = vertcat(
            *[
                if_else(dA[x] < 0, dA[x] / max_dA_fatigue[x], dA[x] / max_dA_recovery[x])
                for x in range(muscle_range)
            ]
        )

        A_t = vertcat(*[controller.states["A_" + muscle_name_list[x]].cx for x in range(len(muscle_name_list))])
        fatigue = [((A_rest[i] - A_t[i]) / (A_rest[i] - A_min) * 100) for i in range(muscle_range)]

        muscle_fatigue_decay = vertcat(
            *[
                (fatigue[x] * (1 + tanh(-dA_nomalized[x])))**2
                for x in range(muscle_range)
            ]
        )

        eps = 1e-8
        rms_fatigue = (sum1(muscle_fatigue_decay) / muscle_range + eps)**0.5
        return rms_fatigue

    @staticmethod
    def minimize_rms_tanh_fatigue_decay_new(controller: PenaltyController) -> MX:
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
        dA = CustomCostFunctions.calculate_dA(controller)

        max_dA_fatigue = [72.2, 61.2, 85.7, 92.3]
        max_dA_recovery = [2.3, 3.0, 14.8, 35.6]

        with_triceps = True
        muscle_range = 4 if with_triceps else 3

        dA_nomalized = vertcat(
            *[
                if_else(dA[x] < 0, dA[x] / max_dA_fatigue[x], dA[x] / max_dA_recovery[x])
                for x in range(muscle_range)
            ]
        )

        fatigue = vertcat(
            *[
                controller.model.muscles_dynamics_model[x].a_scale - controller.states["A_" + muscle_name_list[x]].cx
                for x in range(len(muscle_name_list))
            ]
        )

        muscle_fatigue_decay = vertcat(
            *[
                (fatigue[x] * (1 + tanh(-dA_nomalized[x]))) ** 2
                for x in range(muscle_range)
            ]
        )

        eps = 1e-8
        rms_fatigue = (sum1(muscle_fatigue_decay) / muscle_range + eps)**0.5
        return rms_fatigue




    @staticmethod
    def minimize_failure_point(controller: PenaltyController) -> MX:
        muscle_name_list = controller.model.bio_model.muscle_names
        barrier_model = controller.model.barrier_model
        A_rest = vertcat(*[controller.model.muscles_dynamics_model[x].a_scale for x in range(len(muscle_name_list))])
        A_t = vertcat(*[controller.states["A_" + muscle_name_list[x]].cx for x in range(len(muscle_name_list))])

        w = DM(barrier_model["w"])
        b = float(barrier_model["b"])
        s = A_t / A_rest
        u = 1 - s
        I = dot(w, u) + b
        k = float(barrier_model["kappa"])

        return log(1 + exp(k * (I - 1.0)) / k)


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
            
            value = -((A_t - A_rest) / tau_fat) + (alpha_a * F_t)

        else:
            raise NotImplementedError(f"The cost function {obj_fun_key}, is not implementend in minmax")

        return controller.parameters["minmax_param"].cx[controller.node_index] - value

    # --- A_t+1 cost function used in OCP --- #
    @staticmethod
    def calculate_dA(controller: PenaltyController) -> MX:
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
        dA = -((A_t - A_rest) / tau_fat) + (alpha_a * F_t)

        return dA