from casadi import MX, vertcat, sum1, mmax
from bioptim import PenaltyController
from cocofest.models.ding2007.ding2007 import DingModelPulseWidthFrequency

BAYESIAN_WEIGHT = vertcat([191790, 31609, 117259, 1])
PHYSIOLOGICAL_WEIGHT = vertcat([191790, 31609, 117259, 1])


class CustomCostFunctions:
    def __init__(self):
        self.dict_functions = {
            # --- UNWEIGHTED --- #
            # --- Pulse width --- #
            "minimize_average_activation": {
                "function": self.minimize_average_activation,
                "index": "1",
                "description": "Minimize the average fes activation",
                "power": "1",
                "state": "pw",
            },
            "minimize_root_mean_square_activation": {
                "function": self.minimize_root_mean_square_activation,
                "index": "2",
                "description": "Minimize the root mean square of fes activation",
                "power": "2",
                "state": "pw",
            },
            "minimize_cubic_average_activation": {
                "function": self.minimize_cubic_average_activation,
                "index": "3",
                "description": "Minimize the cubic average of fes activation",
                "power": "3",
                "state": "pw",
            },
            "minimize_peak_activation": {
                "function": self.minimize_peak_activation,
                "index": "4",
                "description": "Minimize the peak of fes activation",
                "power": r"\infty",
                "state": "pw",
            },
            # --- Force --- #
            "minimize_average_force": {
                "function": self.minimize_average_force,
                "index": "5",
                "description": "Minimize the average muscle force",
                "power": "1",
                "state": r"F^{m}",
            },
            "minimize_root_mean_square_force": {
                "function": self.minimize_root_mean_square_force,
                "index": "6",
                "description": "Minimize the root mean square of muscle force",
                "power": "2",
                "state": r"F^{m}",
            },
            "minimize_cubic_average_force": {
                "function": self.minimize_cubic_average_force,
                "index": "7",
                "description": "Minimize the cubic average of muscle force",
                "power": "3",
                "state": r"F^{m}",
            },
            "minimize_peak_force": {
                "function": self.minimize_peak_force,
                "index": "8",
                "description": "Minimize the peak muscle force",
                "power": r"\infty",
                "state": r"F^{m}",
            },
            # --- Stress --- #
            "minimize_average_muscle_stress": {
                "function": self.minimize_average_muscle_stress,
                "index": "9",
                "description": "Minimize the average muscle stress",
                "power": "1",
                "state": r"\sigma",
            },
            "minimize_root_mean_square_muscle_stress": {
                "function": self.minimize_root_mean_square_muscle_stress,
                "index": "10",
                "description": "Minimize the root mean square of muscle stress",
                "power": "2",
                "state": r"\sigma",
            },
            "minimize_cubic_average_muscle_stress": {
                "function": self.minimize_cubic_average_muscle_stress,
                "index": "11",
                "description": "Minimize the cubic average of muscle stress",
                "power": "3",
                "state": r"\sigma",
            },
            "minimize_peak_muscle_stress": {
                "function": self.minimize_peak_muscle_stress,
                "index": "12",
                "description": "Minimize the peak muscle stress",
                "power": r"\infty",
                "state": r"\sigma",
            },
            # --- Fatigue --- #
            "minimize_average_fatigue": {
                "function": self.minimize_average_fatigue,
                "index": "13",
                "description": "Minimize the average muscle fatigue",
                "power": "1",
                "state": "A",
            },
            "minimize_root_mean_square_fatigue": {
                "function": self.minimize_root_mean_square_fatigue,
                "index": "14",
                "description": "Minimize the root mean square of muscle fatigue",
                "power": "2",
                "state": "A",
            },
            "minimize_cubic_average_fatigue": {
                "function": self.minimize_cubic_average_fatigue,
                "index": "15",
                "description": "Minimize the cubic average of muscle fatigue",
                "power": "3",
                "state": "A",
            },
            "minimize_peak_fatigue": {
                "function": self.minimize_peak_fatigue,
                "index": "16",
                "description": "Minimize the peak muscle fatigue",
                "power": r"\infty",
                "state": "A",
            },
            # --- Power --- #
            "minimize_root_mean_square_muscle_power": {
                "function": self.minimize_root_mean_square_muscle_power,
                "index": "17",
                "latex": r"\phi_{17} = \left(\frac{1}{M}\sum_{m=1}^{M} (f^{m} v^{m})^{2}\right)^{\tfrac{1}{2}}",
                "description": "Minimize the root mean square of muscle power",
                "power": "2",
                "state": "W",
            },
            # --- BAYESIAN --- #
            # --- Pulse width --- #
            "minimize_root_mean_square_activation_bayesian": {
                "function": self.minimize_root_mean_square_activation_bayesian,
                "index": "2_bayesian",
                "description": "Minimize the root mean square of fes activation with bayesian weight",
                "power": "2",
                "state": "pw",
            },
            # --- Force --- #
            "minimize_root_mean_square_force_bayesian": {
                "function": self.minimize_root_mean_square_force_bayesian,
                "index": "6_bayesian",
                "description": "Minimize the root mean square of muscle force with bayesian weight",
                "power": "2",
                "state": r"F^{m}",
            },
            # --- Stress --- #
            "minimize_root_mean_square_muscle_stress_bayesian": {
                "function": self.minimize_root_mean_square_muscle_stress_bayesian,
                "index": "10_bayesian",
                "description": "Minimize the root mean square of muscle stress with bayesian weight",
                "power": "2",
                "state": r"\sigma",
            },
            # --- Fatigue --- #
            "minimize_root_mean_square_fatigue_bayesian": {
                "function": self.minimize_root_mean_square_fatigue_bayesian,
                "index": "14_bayesian",
                "description": "Minimize the root mean square of muscle fatigue with bayesian weight",
                "power": "2",
                "state": "A",
            },
            # --- Power --- #
            "minimize_root_mean_square_muscle_power_bayesian": {
                "function": self.minimize_root_mean_square_muscle_power_bayesian,
                "index": "17_bayesian",
                "latex": r"\phi_{17} = \left(\frac{1}{M}\sum_{m=1}^{M} (f^{m} v^{m})^{2}\right)^{\tfrac{1}{2}}",
                "description": "Minimize the root mean square of muscle power with bayesian weight",
                "power": "2",
                "state": "W",
            },
            # --- WEIGHTED --- #
            # --- Pulse width --- #
            "minimize_root_mean_square_activation_weight": {
                "function": self.minimize_root_mean_square_activation_weight,
                "index": "2_weight",
                "description": "Minimize the root mean square of fes activation with weight",
                "power": "2",
                "state": "pw",
            },
            # --- Force --- #
            "minimize_root_mean_square_force_weight": {
                "function": self.minimize_root_mean_square_force_weight,
                "index": "6_weight",
                "description": "Minimize the root mean square of muscle force with weight",
                "power": "2",
                "state": r"F^{m}",
            },
            # --- Stress --- #
            "minimize_root_mean_square_muscle_stress_weight": {
                "function": self.minimize_root_mean_square_muscle_stress_weight,
                "index": "10_weight",
                "description": "Minimize the root mean square of muscle stress with weight",
                "power": "2",
                "state": r"\sigma",
            },
            # --- Fatigue --- #
            "minimize_root_mean_square_fatigue_weight": {
                "function": self.minimize_root_mean_square_fatigue_weight,
                "index": "14_weight",
                "description": "Minimize the root mean square of muscle fatigue with weight",
                "power": "2",
                "state": "A",
            },
            # --- Power --- #
            "minimize_root_mean_square_muscle_power_weight": {
                "function": self.minimize_root_mean_square_muscle_power_weight,
                "index": "17_weight",
                "latex": r"\phi_{17} = \left(\frac{1}{M}\sum_{m=1}^{M} (f^{m} v^{m})^{2}\right)^{\tfrac{1}{2}}",
                "description": "Minimize the root mean square of muscle power with weight",
                "power": "2",
                "state": "W",
            },
            # --- HELPER --- #
            "minimize_peak": {
                "function": self.minimize_peak,
                "index": 99,
                "latex": r"\phi_{99} = \max_{m=1,\ldots,M} \; \mathcal{Var}^{m}",
                "description": "Minimize the peak of a variable",
            },
        }

    # --- Electrical stimulation cost functions --- #
    # --- UNWEIGHTED --- #
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
        cubic_avg_activation = (sum1(stim_charge) / len(muscle_name_list) + eps) ** (1 / 3)
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

    # --- BAYESIAN --- #
    # --- Pulse width --- #
    @staticmethod
    def minimize_root_mean_square_activation_bayesian(controller: PenaltyController) -> MX:
        """
        Minimize the root-mean-square of fes activation with bayesian weights.

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
        weight = vertcat([10000, 1560, 4665, 0.0001])
        if isinstance(controller.model.muscles_dynamics_model[0], DingModelPulseWidthFrequency):
            stim_charge = vertcat(
                *[
                    weight[x]
                    * (
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

    # --- Force --- #
    @staticmethod
    def minimize_root_mean_square_force_bayesian(controller: PenaltyController) -> MX:
        """
        Minimize the root-mean-square of muscle force production with bayesian weights.

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
        weight = vertcat([10000, 1560, 4665, 0.0001])
        muscle_force = vertcat(
            *[weight[x] * controller.states["F_" + muscle_name_list[x]].cx ** 2 for x in range(len(muscle_name_list))]
        )
        rms_force = (sum1(muscle_force) / len(muscle_name_list) + eps) ** 0.5
        return rms_force

    # --- Stress --- #
    @staticmethod
    def minimize_root_mean_square_muscle_stress_bayesian(controller: PenaltyController) -> MX:
        """
        Minimize the root-mean-square of muscle stress with bayesian weights.

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
        weight = vertcat([10000, 1560, 4665, 0.0001])
        muscle_stress = vertcat(
            *[
                weight[x]
                * (controller.states["F_" + muscle_name_list[x]].cx / controller.model.muscles_dynamics_model[x].pcsa)
                ** 2
                for x in range(len(muscle_name_list))
            ]
        )
        rms_stress = (sum1(muscle_stress) / len(muscle_name_list) + eps) ** 0.5
        return rms_stress

    # --- Fatigue --- #
    @staticmethod
    def minimize_root_mean_square_fatigue_bayesian(controller: PenaltyController) -> MX:
        """
        Minimize the root-mean-square of muscle fatigue with bayesian weights.

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
        weight = vertcat([10000, 1560, 4665, 0.0001])
        muscle_fatigue = vertcat(
            *[
                weight[x]
                * (
                    controller.model.muscles_dynamics_model[x].a_scale
                    - controller.states["A_" + muscle_name_list[x]].cx
                )
                ** 2
                for x in range(len(muscle_name_list))
            ]
        )
        rms_fatigue = (sum1(muscle_fatigue) / len(muscle_name_list) + eps) ** 0.5
        return rms_fatigue

    # --- Power --- #
    @staticmethod
    def minimize_root_mean_square_muscle_power_bayesian(controller: PenaltyController) -> MX:
        """
        Minimize the root-mean-square of muscle power with bayesian weights.

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
        weight = vertcat([10000, 1560, 4665, 0.0001])
        muscle_velocity = controller.model.muscle_velocity()(
            controller.states["q"].cx, controller.states["qdot"].cx, controller.parameters.cx
        )
        muscle_power = vertcat(
            *[
                weight[x] * (controller.states["F_" + muscle_name_list[x]].cx * muscle_velocity[x]) ** 2
                for x in range(len(muscle_name_list))
            ]
        )
        rms_power = (sum1(muscle_power) / len(muscle_name_list) + eps) ** 0.5
        return rms_power

    # --- WEIGHTED --- #
    # --- Pulse width --- #
    @staticmethod
    def minimize_root_mean_square_activation_weight(controller: PenaltyController) -> MX:
        """
        Minimize the root-mean-square of fes activation with muscle weights.

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
                    PHYSIOLOGICAL_WEIGHT[x]
                    * (
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

    # --- Force --- #
    @staticmethod
    def minimize_root_mean_square_force_weight(controller: PenaltyController) -> MX:
        """
        Minimize the root-mean-square of muscle force production with muscle weights.

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
                PHYSIOLOGICAL_WEIGHT[x] * controller.states["F_" + muscle_name_list[x]].cx ** 2
                for x in range(len(muscle_name_list))
            ]
        )
        rms_force = (sum1(muscle_force) / len(muscle_name_list) + eps) ** 0.5
        return rms_force

    # --- Stress --- #
    @staticmethod
    def minimize_root_mean_square_muscle_stress_weight(controller: PenaltyController) -> MX:
        """
        Minimize the root-mean-square of muscle stress with muscle weights.

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
                PHYSIOLOGICAL_WEIGHT[x]
                * (controller.states["F_" + muscle_name_list[x]].cx / controller.model.muscles_dynamics_model[x].pcsa)
                ** 2
                for x in range(len(muscle_name_list))
            ]
        )
        rms_stress = (sum1(muscle_stress) / len(muscle_name_list) + eps) ** 0.5
        return rms_stress

    # --- Fatigue --- #
    @staticmethod
    def minimize_root_mean_square_fatigue_weight(controller: PenaltyController) -> MX:
        """
        Minimize the root-mean-square of muscle fatigue with muscle weights.

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
                PHYSIOLOGICAL_WEIGHT[x]
                * (
                    controller.model.muscles_dynamics_model[x].a_scale
                    - controller.states["A_" + muscle_name_list[x]].cx
                )
                ** 2
                for x in range(len(muscle_name_list))
            ]
        )
        rms_fatigue = (sum1(muscle_fatigue) / len(muscle_name_list) + eps) ** 0.5
        return rms_fatigue

    # --- Power --- #
    @staticmethod
    def minimize_root_mean_square_muscle_power_weight(controller: PenaltyController) -> MX:
        """
        Minimize the root-mean-square of muscle power with muscle weights.

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
                PHYSIOLOGICAL_WEIGHT[x] * (controller.states["F_" + muscle_name_list[x]].cx * muscle_velocity[x]) ** 2
                for x in range(len(muscle_name_list))
            ]
        )
        rms_power = (sum1(muscle_power) / len(muscle_name_list) + eps) ** 0.5
        return rms_power

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
