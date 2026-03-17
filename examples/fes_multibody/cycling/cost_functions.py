from casadi import MX, vertcat, sum1, fabs, sign, tanh, if_else, log, exp, DM, dot, mmax, mmin, cos, sin
from bioptim import PenaltyController
from cocofest.models.ding2007.ding2007 import DingModelPulseWidthFrequency
from cocofest.models.hill_coefficients import (muscle_force_length_coefficient,
                                               muscle_force_velocity_coefficient,
                                               muscle_passive_force_coefficient)


class CustomCostFunctions:
    def __init__(self):
        self.dict_functions = {
            # --- Pulse width --- #
            "minimize_average_activation": {
                "function": self.minimize_average_activation,
                "index": 1,
                "description": "Minimize the average fes activation",
                "power": "1",
                "state": "pw",
            },
            "minimize_root_mean_square_activation": {
                "function": self.minimize_root_mean_square_activation,
                "index": 2,
                "description": "Minimize the root mean square of fes activation",
                "power": "2",
                "state": "pw",
            },
            "minimize_cubic_average_activation": {
                "function": self.minimize_cubic_average_activation,
                "index": 3,
                "description": "Minimize the cubic average of fes activation",
                "power": "3",
                "state": "pw",
            },
            "minimize_peak_activation": {
                "function": self.minimize_peak_activation,
                "index": 4,
                "description": "Minimize the peak of fes activation",
                "power": r"\infty",
                "state": "pw",
            },
            # --- Force --- #
            "minimize_average_force": {
                "function": self.minimize_average_force,
                "index": 5,
                "description": "Minimize the average muscle force",
                "power": "1",
                "state": r"F^{m}",
            },
            "minimize_root_mean_square_force": {
                "function": self.minimize_root_mean_square_force,
                "index": 6,
                "description": "Minimize the root mean square of muscle force",
                "power": "2",
                "state": r"F^{m}",
            },
            "minimize_cubic_average_force": {
                "function": self.minimize_cubic_average_force,
                "index": 7,
                "description": "Minimize the cubic average of muscle force",
                "power": "3",
                "state": r"F^{m}",
            },
            "minimize_peak_force": {
                "function": self.minimize_peak_force,
                "index": 8,
                "description": "Minimize the peak muscle force",
                "power": r"\infty",
                "state": r"F^{m}",
            },
            # --- Stress --- #
            "minimize_average_muscle_stress": {
                "function": self.minimize_average_muscle_stress,
                "index": 9,
                "description":"Minimize the average muscle stress",
                "power": "1",
                "state": r"\sigma",
            },
            "minimize_root_mean_square_muscle_stress": {
                "function": self.minimize_root_mean_square_muscle_stress,
                "index": 10,
                "description":"Minimize the root mean square of muscle stress",
                "power": "2",
                "state": r"\sigma",
            },
            "minimize_cubic_average_muscle_stress": {
                "function": self.minimize_cubic_average_muscle_stress,
                "index": 11,
                "description":"Minimize the cubic average of muscle stress",
                "power": "3",
                "state": r"\sigma",
            },
            "minimize_peak_muscle_stress": {
                "function": self.minimize_peak_muscle_stress,
                "index": 12,
                "description":"Minimize the peak muscle stress",
                "power": r"\infty",
                "state": r"\sigma",
            },
            # --- Fatigue --- #
            "minimize_average_fatigue": {
                "function": self.minimize_average_fatigue,
                "index": 13,
                "description": "Minimize the average muscle fatigue",
                "power": "1",
                "state": "A",
            },
            "minimize_root_mean_square_fatigue": {
                "function": self.minimize_root_mean_square_fatigue,
                "index": 14,
                "description": "Minimize the root mean square of muscle fatigue",
                "power": "2",
                "state": "A",
            },
            "minimize_cubic_average_fatigue": {
                "function": self.minimize_cubic_average_fatigue,
                "index": 15,
                "description": "Minimize the cubic average of muscle fatigue",
                "power": "3",
                "state": "A",
            },
            "minimize_peak_fatigue": {
                "function": self.minimize_peak_fatigue,
                "index": 16,
                "description": "Minimize the peak muscle fatigue",
                "power": r"\infty",
                "state": "A",
            },
            # --- Power --- #
            "minimize_root_mean_square_muscle_power": {
                "function": self.minimize_root_mean_square_muscle_power,
                "index": 17,
                "latex": r"\phi_{17} = \left(\frac{1}{M}\sum_{m=1}^{M} (f^{m} v^{m})^{2}\right)^{\tfrac{1}{2}}",
                "description": "Minimize the root mean square of muscle power",
                "power": "2",
                "state": "W",
            },

            # --- Custom cost functions --- #
            "minimize_average_fatigue_and_recovery": {
                "function": self.minimize_average_fatigue_and_recovery,
                "index": 20,
                "description": "Minimize the average fatigue and recovery",
                "power": "1",
                "state": "A_recovery",
            },
            "minimize_useful_torque_fatigue_tradeoff": {
                "function": self.minimize_useful_torque_fatigue_tradeoff,
                "index": 21,
                "description": "Minimize the average fatigue and recovery based on torque efficiency",
                "power": "1",
                "state": "tau_eff",
            },

            "minimize_average_fatigue_and_recovery_2": {
                "function": self.minimize_average_fatigue_and_recovery_2,
                "index": 22,
                "description": "Minimize the average fatigue and recovery",
                "power": "1",
                "state": "A_recovery",
            },

            "minimize_average_fatigue_and_recovery_3": {
                "function": self.minimize_average_fatigue_and_recovery_3,
                "index": 23,
                "description": "Minimize the average fatigue and recovery",
                "power": "1",
                "state": "A_recovery",
            },

            "minimize_average_fatigue_and_recovery_4": {
                "function": self.minimize_average_fatigue_and_recovery_4,
                "index": 24,
                "description": "Minimize the average fatigue and recovery",
                "power": "1",
                "state": "A_recovery",
            },

            "minimize_average_fatigue_and_recovery_5": {
                "function": self.minimize_average_fatigue_and_recovery_5,
                "index": 25,
                "description": "Minimize the average fatigue and recovery",
                "power": "1",
                "state": "A_recovery",
            },

            # "minimize_torque_deficit_area": {
            #     "function": self.minimize_torque_deficit_area,
            #     "index": 25,
            #     "latex": r"\phi_{24}=\left(\tau_{\mathrm{thr}}-\sum_{m=1}^{M}[\gamma^m(\theta)F^m]_+\right)_+|\dot{\theta}|",
            #     "description": "Minimize the crank-angle area where useful torque is below threshold",
            #     "power": "1",
            #     "state": r"\tau_{\mathrm{def}}",
            # },
            #
            #
            #
            #

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
        Minimize the average fes activation.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The average of fes activation
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
        Minimize the root-mean-square of fes activation.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The root-mean-square of fes activation
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
        Minimize the cubic average of fes activation.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The cubic average of fes activation
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
        Minimize the peak fes activation.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The peak of fes activation
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
    def minimize_average_fatigue_and_recovery(controller: PenaltyController) -> MX:
        """
        Minimize the average fatigue and recuperation.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The average fatigue and recuperation
        """
        muscle_name_list = controller.model.bio_model.muscle_names
        dA = CustomCostFunctions.calculate_dA(controller)
        A_rest = [controller.model.muscles_dynamics_model[x].a_scale for x in range(len(muscle_name_list))]
        tau_fat = [controller.model.muscles_dynamics_model[x].tau_fat for x in range(len(muscle_name_list))]
        alpha_a = [controller.model.muscles_dynamics_model[x].alpha_a for x in range(len(muscle_name_list))]
        fmax = [controller.model.muscles_dynamics_model[x].fmax for x in range(len(muscle_name_list))]
        normed_a = [A / max(A_rest) for A in A_rest]

        max_dA_recovery = [A_rest[x]/tau_fat[x] for x in range(len(muscle_name_list))]
        max_dA_fatigue = [-(alpha_a[x] * fmax[x]) for x in range(len(muscle_name_list))]

        with_triceps = True
        muscle_range = 4 if with_triceps else 3

        dA_nomalized = vertcat(
            *[
                if_else(dA[x] < 0, dA[x] / max_dA_fatigue[x], dA[x] / max_dA_recovery[x])
                for x in range(muscle_range)
            ]
        )

        A_t = vertcat(*[controller.states["A_" + muscle_name_list[x]].cx for x in range(len(muscle_name_list))])
        fatigue = [((A_rest[i] - A_t[i]) / (A_rest[i])) for i in range(muscle_range)]

        muscle_fatigue_decay = vertcat(
            *[
                normed_a[x] * (10 ** (4 * fatigue[x])) * (1 + tanh(-dA_nomalized[x]))
                for x in range(muscle_range)
            ]
        )

        avg_fatigue = sum1(muscle_fatigue_decay) / muscle_range
        return avg_fatigue

    @staticmethod
    def minimize_average_fatigue_and_recovery_2(controller: PenaltyController) -> MX:
        """
        Minimize the average fatigue and recuperation.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The average fatigue and recuperation
        """
        # --- Get all information --- #
        muscle_names, q, qdot, F, A, A_rest, tau_fat, alpha_a, fmax, dA = CustomCostFunctions.get_muscle_quantities(
            controller)
        gamma = CustomCostFunctions.useful_gain_from_angle(q[2])
        max_dA_recovery = [A_rest[x] / tau_fat[x] for x in range(F.shape[0])]
        max_dA_fatigue = [-(alpha_a[x] * fmax[x]) for x in range(F.shape[0])]
        dA_nomalized = vertcat(
            *[
                if_else(dA[x] < 0, dA[x] / max_dA_fatigue[x], dA[x] / max_dA_recovery[x])
                for x in range(F.shape[0])
            ]
        )

        # --- Contribution --- #
        weight_contribution = (F / fmax) / (mmax(F / fmax) + 1e-8)
        cost_contribution = [(1 + tanh(-gamma[i] * 10)) for i in range(F.shape[0])]

        # --- Fatigue --- #
        weight_fatigue = vertcat([74.425, 151.963, 93.339, 42.114]) / mmax(vertcat([74.425, 151.963, 93.339, 42.114]))
        fatigue = [1 - ((A_rest[i] - A[i]) / (A_rest[i])) for i in range(F.shape[0])]
        cost_fatigue = [(10**4) / (10 ** (4 * fatigue[i])) for i in range(F.shape[0])]

        # --- Recovery --- #
        weight_recovery = (alpha_a / A_rest) / mmin(alpha_a / A_rest)
        cost_recovery = [(1 + tanh(-dA_nomalized[i])) for i in range(F.shape[0])]

        # --- Cost function --- #
        cost = vertcat(
            *[
                weight_contribution[i] * cost_contribution[i] * weight_fatigue[i] * cost_fatigue[i] * weight_recovery[i] * cost_recovery[i]
                for i in range(F.shape[0])
            ]
        )
        avg_cost= sum1(cost) / F.shape[0]
        return avg_cost

    @staticmethod
    def minimize_average_fatigue_and_recovery_3(controller: PenaltyController) -> MX:
        """
        Minimize the average fatigue and recuperation.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The average fatigue and recuperation
        """

        # --- Get all information --- #
        muscle_names, q, qdot, F, A, A_rest, tau_fat, alpha_a, fmax, dA = CustomCostFunctions.get_muscle_quantities(
            controller)
        # A_min = [41, 70, 379, 932]

        # --- Fatigue --- #
        weight_fatigue = vertcat([100.0, 1, 5, 0.0])
        # cost_fatigue = [((A_rest[i] - A[i]) / (A_rest[i] - A_min[i] + 1e-8)) ** 2 for i in range(F.shape[0])]
        cost_fatigue = [(A_rest[i] - A[i]) ** 2 for i in range(F.shape[0])]

        # --- Cost function --- #
        cost = vertcat(*[weight_fatigue[i] * cost_fatigue[i] for i in range(F.shape[0])])
        rms_cost = (sum1(cost) / F.shape[0] + 1e-8) ** (1/2)

        # muscle_weights = vertcat([100.0, 1, 5, 0.0])
        # eps = 1e-8
        # muscle_name_list = controller.model.bio_model.muscle_names
        # muscle_fatigue = vertcat(
        #     *[
        #         muscle_weights[x]
        #         * (controller.model.muscles_dynamics_model[x].a_scale - controller.states[
        #             "A_" + muscle_name_list[x]].cx) ** 2
        #         for x in range(len(muscle_name_list))
        #     ]
        # )
        # rms_fatigue = (sum1(muscle_fatigue) / len(muscle_name_list) + eps) ** 0.5
        return rms_cost

    @staticmethod
    def minimize_average_fatigue_and_recovery_4(controller: PenaltyController) -> MX:
        """
        Minimize the average fatigue and recuperation.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The average fatigue and recuperation
        """

        # # --- Get all information --- #
        # muscle_names, q, qdot, F, A, A_rest, tau_fat, alpha_a, fmax, dA = CustomCostFunctions.get_muscle_quantities(
        #     controller)
        # A_min = [41, 70, 379, 932]
        # max_dA_recovery = [A_rest[x] / tau_fat[x] for x in range(F.shape[0])]
        # max_dA_fatigue = [-(alpha_a[x] * fmax[x]) for x in range(F.shape[0])]
        #
        # # --- Fatigue --- #
        # weight_fatigue = [1.0, 0.6104101922170402, 0.754209800965523, 0.0]
        # # cost_fatigue = [((A_rest[i] - A[i]) / (A_rest[i] - A_min[i] + 1e-8)) ** 2 for i in range(F.shape[0])]
        #
        # # --- Recovery --- #
        # dA_nomalized = vertcat(
        #     *[
        #         if_else(dA[x] < 0, dA[x] / max_dA_fatigue[x], dA[x] / max_dA_recovery[x])
        #         for x in range(F.shape[0])
        #     ]
        # )
        # cost_recovery = [(1 + tanh(-dA_nomalized[i])) for i in range(F.shape[0])]
        #
        # # --- Cost function --- #
        # # cost = vertcat(*[weight_fatigue[i] * cost_fatigue[i] * cost_recovery[i] for i in range(F.shape[0])])
        # cost = vertcat(*[weight_fatigue[i] * cost_recovery[i] for i in range(F.shape[0])])
        # rms_cost = (sum1(cost) / F.shape[0] + 1e-8) ** (1 / 2)

        muscle_weights = vertcat([1e5, 1560, 4665, 1e-5])
        eps = 1e-8
        muscle_name_list = controller.model.bio_model.muscle_names
        muscle_fatigue = vertcat(
            *[
                muscle_weights[x]
                * (controller.model.muscles_dynamics_model[x].a_scale - controller.states[
                    "A_" + muscle_name_list[x]].cx) ** 2
                for x in range(len(muscle_name_list))
            ]
        )
        rms_fatigue = (sum1(muscle_fatigue) / len(muscle_name_list) + eps) ** 0.5
        return rms_fatigue


    @staticmethod
    def minimize_average_fatigue_and_recovery_5(controller: PenaltyController) -> MX:
        """
        Minimize the average fatigue and recuperation.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The average fatigue and recuperation
        """

        muscle_weights = vertcat([1e5, 1560, 4665, 0])
        eps = 1e-8
        muscle_name_list = controller.model.bio_model.muscle_names
        muscle_fatigue = vertcat(
            *[
                muscle_weights[x]
                * (controller.model.muscles_dynamics_model[x].a_scale - controller.states[
                    "A_" + muscle_name_list[x]].cx) ** 2
                for x in range(len(muscle_name_list))
            ]
        )
        rms_fatigue = (sum1(muscle_fatigue) / len(muscle_name_list) + eps) ** 0.5
        return rms_fatigue


    # @staticmethod
    # def minimize_useful_torque_fatigue_tradeoff(
    #         controller: PenaltyController,
    # ) -> MX:
    #     """
    #     Minimize the fatigue and recuperation based on torque efficiency.
    #
    #     Parameters
    #     ----------
    #     controller: PenaltyController
    #         The penalty node elements
    #
    #     Returns
    #     -------
    #     The average fatigue and recuperation based on torque efficiency
    #     """
    #     # --- Weights --- #
    #     lambda_fatigue = 1.0
    #     lambda_dA = 1.0
    #     kappa_recovery = 1.0
    #
    #     # --- Get all information --- #
    #     muscle_names, q, qdot, F, A, A_rest, tau_fat, alpha_a, fmax, dA = CustomCostFunctions.get_muscle_quantities(controller)
    #
    #     # --- Calculate useful torque based on force and muscle efficiency --- #
    #     gamma = CustomCostFunctions.useful_gain_from_angle(q[2])
    #
    #     gamma_tanh = vertcat(
    #         *[(1 + tanh(10*gamma[x]))
    #           for x in range(F.shape[0])
    #           ]
    #     )
    #
    #
    #     # tau_use = vertcat(*[
    #     #     if_else(gamma[i] * F[i] > 0, gamma[i] * F[i], 0)
    #     #     for i in range(F.shape[0])
    #     # ])
    #
    #     muscle_motion_contrib = [0.16259391348727756, 0.32644352775103314, 0.06473043181533218, 0.07310528260037236]
    #     muscle_motion_contrib_norm = [muscle_contrib/max(muscle_motion_contrib) for muscle_contrib in muscle_motion_contrib]
    #
    #     muscle_fatigability = [-alpha_a[i]/A_rest[i] for i in range(F.shape[0])]
    #     muscle_fatigability_norm = [muscle_fat/max(muscle_fatigability) for muscle_fat in muscle_fatigability]
    #
    #     muscle_importance_weight = [muscle_motion_contrib_norm[i] * muscle_fatigability_norm[i] for i in range(F.shape[0])]
    #
    #     torque_distribution = [(muscle_importance_weight[i] / (gamma_tanh[i] + 1e-8)) * F[i] for i in range(F.shape[0])]
    #
    #
    #     # --- Fatigue and recovery --- #
    #     cap = A / A_rest
    #     fatigue = 1 - cap
    #     dA_norm = CustomCostFunctions.normalized_dA(dA, A_rest, tau_fat, alpha_a, fmax)
    #
    #     # --- Build the adaptive weight --- #
    #     adaptive_weight = vertcat(*[
    #         1
    #         + fatigue[i]
    #         + kappa_recovery * if_else(dA_norm[i] < 0, -dA_norm[i], 0)
    #         for i in range(F.shape[0])
    #     ])
    #
    #     # --- Calculate the cost function --- #
    #     # use_term = vertcat(*[
    #     #     adaptive_weight[i] * (tau_use[i] / fmax[i]) ** 2
    #     #     for i in range(F.shape[0])
    #     # ])
    #
    #     # fatigue_term = vertcat(*[
    #     #     lambda_fatigue * fatigue[i] ** 2
    #     #     for i in range(F.shape[0])
    #     # ])
    #     fatigue_term = vertcat(*[
    #         lambda_fatigue * (10 ** (4 * fatigue[i]))
    #         for i in range(F.shape[0])
    #     ])
    #
    #     # dA_term = vertcat(*[
    #     #     lambda_dA * if_else(dA_norm[i] < 0, (-dA_norm[i]) ** 2, 0)
    #     #     for i in range(F.shape[0])
    #     # ])
    #     dA_term = vertcat(
    #         *[(1 + tanh(-dA_norm[x]))
    #             for x in range(F.shape[0])
    #         ]
    #     )
    #
    #     # return sum1(use_term + fatigue_term + dA_term) / len(muscle_names)
    #
    #     cost_fun = vertcat(
    #         *[torque_distribution[i] * fatigue_term[i] * dA_term[i]
    #             for i in range(F.shape[0])
    #         ]
    #     )
    #     return sum1(cost_fun) / len(muscle_names)

    @staticmethod
    def minimize_useful_torque_fatigue_tradeoff(controller: PenaltyController) -> MX:
        kappa_recovery = 1
        lambda_fatigue = 1
        lambda_dA = 1
        muscle_names, q, qdot, F, A, A_rest, tau_fat, alpha_a, fmax, dA = CustomCostFunctions.get_muscle_quantities(controller)

        gamma = CustomCostFunctions.useful_gain_from_angle(q[2])
        tau_use = vertcat(*[
            if_else(gamma[i] * F[i] > 0, gamma[i] * F[i], 0)
            for i in range(F.shape[0])
        ])

        cap = A / A_rest
        fatigue = 1 - cap

        dA_norm = CustomCostFunctions.normalized_dA(dA, A_rest, tau_fat, alpha_a, fmax)

        adaptive_weight = vertcat(*[
            1 + fatigue[i] + kappa_recovery * if_else(dA_norm[i] < 0, -dA_norm[i], 0)
            for i in range(F.shape[0])
        ])

        use_term = vertcat(*[
            adaptive_weight[i] * (tau_use[i] / (fmax[i])) ** 2
            for i in range(F.shape[0])
        ])

        fatigue_term = vertcat(*[
            lambda_fatigue * fatigue[i] ** 2
            for i in range(F.shape[0])
        ])

        dA_term = vertcat(*[
            lambda_dA * if_else(dA_norm[i] < 0, (-dA_norm[i]) ** 2, 0)
            for i in range(F.shape[0])
        ])

        return sum1(use_term + fatigue_term + dA_term) / len(muscle_names)


    @staticmethod
    def minimize_terminal_fatigue_reserve(controller: PenaltyController) -> MX:
        muscle_names = controller.model.bio_model.muscle_names
        A = vertcat(*[controller.states[f"A_{name}"].cx for name in muscle_names])
        A_rest = vertcat(*[
            controller.model.muscles_dynamics_model[i].a_scale
            for i in range(len(muscle_names))
        ])
        reserve_loss = 1 - A / A_rest
        return sum1(reserve_loss ** 2) / len(muscle_names)


    @staticmethod
    def minimize_torque_deficit_area(
            controller: PenaltyController,
    ) -> MX:
        """
        Minimize the crank-angle area where useful torque is below a threshold.

        This implements:
            integral( max(threshold - tau_use(theta), 0) dtheta )

        Since the OCP is integrated over time, the node-wise Lagrange term is:
            max(threshold - tau_use, 0) * abs(theta_dot)

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        MX
            Instantaneous deficit-area density
        """
        muscle_names, q, qdot, F, A, A_rest, tau_fat, alpha_a, fmax, dA = \
            CustomCostFunctions.get_muscle_quantities(controller)

        moment_arm = CustomCostFunctions.get_moment_arm_from_angle(q[2])
        tau_use_per_muscle = vertcat(*[moment_arm[i] * F[i] for i in range(F.shape[0])])

        tau_use_total = sum1(tau_use_per_muscle)
        torque_deficit = 0.20 - tau_use_total
        positive_deficit = fabs(torque_deficit)

        angular_weight = fabs(qdot[2])  # Convert time integral into crank-angle integral: dtheta = |theta_dot| dt
        cost = positive_deficit * angular_weight

        return cost


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

    @staticmethod
    def get_muscle_quantities(controller: PenaltyController):
        muscle_names = controller.model.bio_model.muscle_names

        F = vertcat(*[controller.states[f"F_{name}"].cx for name in muscle_names])
        A = vertcat(*[controller.states[f"A_{name}"].cx for name in muscle_names])

        A_rest = vertcat(*[
            controller.model.muscles_dynamics_model[i].a_scale
            for i in range(len(muscle_names))
        ])
        tau_fat = vertcat(*[
            controller.model.muscles_dynamics_model[i].tau_fat
            for i in range(len(muscle_names))
        ])
        alpha_a = vertcat(*[
            controller.model.muscles_dynamics_model[i].alpha_a
            for i in range(len(muscle_names))
        ])
        fmax = vertcat(*[
            controller.model.muscles_dynamics_model[i].fmax
            for i in range(len(muscle_names))
        ])

        q = controller.states["q"].cx
        qdot = controller.states["qdot"].cx
        dA = CustomCostFunctions.calculate_dA(controller)

        return muscle_names, q, qdot, F, A, A_rest, tau_fat, alpha_a, fmax, dA

    @staticmethod
    def useful_gain_from_angle(theta):
        gains = []
        coeffs = [
            [0.003796394160508141, 0.07527209994759597, -0.0012680063435009572, 0.0061414695027003875, -0.010001029394988675],
            [-0.0035496461377487435, -0.168625205597484, -0.00715498920622323, -0.0021333488227879933, 0.01389008809979322],
            [0.002167840747823907, 0.025718492411623006, -0.01056494736328612, -0.008221893404768942, -0.0004271720304807771],
            [0.004142036625458454, -0.020814164180226594, -0.02166684165948551, 0.005969951747631583, 0.004676535739979978],
        ]

        for i in range(len(coeffs)):
            a0, a1, b1, a2, b2 = coeffs[i]
            g = (
                    a0
                    + a1 * cos(theta)
                    + b1 * sin(theta)
                    + a2 * cos(2 * theta)
                    + b2 * sin(2 * theta)
            )
            gains.append(g)
        return vertcat(*gains)

    @staticmethod
    def get_moment_arm_from_angle(theta):
        gains = []
        coeffs = [
            [0.004161487758942413, 0.1073083139877185, -0.0015211682786977012, 0.00824887333313916,
             -0.004961689555209104],
            [-0.005257685768086165, -0.17937397449870182, -0.006646755380076929, -0.003783181999089678,
             0.02065945437924536],
            [0.0025016818882907517, 0.03586617223381479, -0.014120195346288443, -0.01071484060626549,
             0.001829055148093859],
            [0.004146157359059304, -0.021729857362885213, -0.022603532576381705, 0.007384693394067062,
             0.004975602705321599],
        ]

        for i in range(len(coeffs)):
            a0, a1, b1, a2, b2 = coeffs[i]
            g = (
                    a0
                    + a1 * cos(theta)
                    + b1 * sin(theta)
                    + a2 * cos(2 * theta)
                    + b2 * sin(2 * theta)
            )
            gains.append(g)
        return vertcat(*gains)

    @staticmethod
    def normalized_dA(dA, A_rest, tau_fat, alpha_a, fmax):
        max_dA_recovery = A_rest / tau_fat
        max_dA_fatigue = -(alpha_a * fmax)

        return vertcat(*[
            if_else(
                dA[i] < 0,
                dA[i] / max_dA_fatigue[i],
                dA[i] / max_dA_recovery[i]
            )
            for i in range(dA.shape[0])
        ])