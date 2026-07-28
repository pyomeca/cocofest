"""
This custom objective class regroups all the custom objectives that are used in the optimization problem.
"""

from casadi import MX, vertcat
from bioptim import PenaltyController
from .models.fes_model import FesModel
from .models.ding2007.ding2007 import DingModelPulseWidthFrequency
from .models.hmed2018.hmed2018 import DingModelPulseIntensityFrequency


class CustomObjective:
    @staticmethod
    def _muscle_names(controller: PenaltyController) -> list[str]:
        if hasattr(controller.model, "muscles_dynamics_model"):
            return [
                str(model.muscle_name)
                for model in controller.model.muscles_dynamics_model
            ]
        return list(controller.model.bio_model.muscle_names)

    @staticmethod
    def minimize_overall_muscle_fatigue(controller: PenaltyController) -> MX:
        """
        Minimize the overall muscle fatigue.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The sum of each force scaling factor
        """
        muscle_name_list = CustomObjective._muscle_names(controller)
        muscle_model = controller.model.muscles_dynamics_model
        muscle_fatigue = vertcat(
            *[
                1 - (controller.states["A_" + muscle_name_list[x]].cx / muscle_model[x].a_scale)
                for x in range(len(muscle_name_list))
            ]
        )
        return muscle_fatigue

    @staticmethod
    def minimize_overall_muscle_force_production(controller: PenaltyController) -> MX:
        """
        Minimize the overall muscle force production.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The sum of each force
        """
        muscle_name_list = CustomObjective._muscle_names(controller)
        muscle_model = controller.model.muscles_dynamics_model
        muscle_force = vertcat(
            *[
                controller.states["F_" + muscle_name_list[x]].cx / muscle_model[x].fmax
                for x in range(len(muscle_name_list))
            ]
        )
        return muscle_force

    @staticmethod
    def minimize_overall_stimulation_charge(controller: PenaltyController) -> MX:
        """
        Minimize the overall stimulation charge.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements

        Returns
        -------
        The sum of each stimulation control
        """
        if hasattr(controller.model, "muscles_dynamics_model"):
            muscle_name_list = CustomObjective._muscle_names(controller)
            if isinstance(controller.model.muscles_dynamics_model[0], DingModelPulseWidthFrequency):
                stim_charge = vertcat(
                    *[
                        controller.controls["last_pulse_width_" + muscle_name_list[x]].cx
                        / controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name_list[x]].max[0][0]
                        for x in range(len(muscle_name_list))
                    ]
                )
            else:
                stim_charge = vertcat(
                    *[
                        controller.controls["pulse_intensity_" + muscle_name_list[x]].cx
                        for x in range(len(muscle_name_list))
                    ]
                )
        else:
            stim_charge = controller.controls.cx

        return stim_charge

    @staticmethod
    def minimize_muscle_fatigue_normalized(controller: PenaltyController, fes_model: FesModel, power: int = 1) -> MX:
        """
        Minimize the normalized muscle fatigue.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements
        fes_model: FesModel
            The fes model to use
        power: int
            The power to use for the cost function

        Returns
        -------
        The force scaling factor normalized by the rested capacity
        """
        muscle_name = fes_model.muscle_name
        muscle_fatigue = 1 - (controller.states["A_" + muscle_name].cx / fes_model.a_scale)
        return muscle_fatigue**power

    @staticmethod
    def minimize_muscle_force_production_normalized(
        controller: PenaltyController, fes_model: FesModel, power: int = 1
    ) -> MX:
        """
        Minimize the normalized muscle force production.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements
        fes_model: FesModel
            The fes model to use
        power: int
            The power to use for the cost function

        Returns
        -------
        The force normalized by the maximal isometric force
        """
        muscle_name = fes_model.muscle_name
        muscle_force = controller.states["F_" + muscle_name].cx / fes_model.fmax
        return muscle_force**power

    @staticmethod
    def minimize_stimulation_charge_normalized(
        controller: PenaltyController, fes_model: FesModel, power: int = 1
    ) -> MX:
        """
        Minimize the normalized stimulation charge.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements
        fes_model: FesModel
            The fes model to use
        power: int
            The power to use for the cost function

        Returns
        -------
        The stimulation control normalized by its maximal value
        """
        muscle_name = fes_model.muscle_name
        if isinstance(fes_model, DingModelPulseWidthFrequency):
            stim_charge = (
                controller.controls["last_pulse_width_" + muscle_name].cx
                / controller.ocp.nlp[0].u_bounds["last_pulse_width_" + muscle_name].max[0][0]
            )
        elif isinstance(fes_model, DingModelPulseIntensityFrequency):
            stim_charge = (
                controller.controls["pulse_intensity_" + muscle_name].cx
                / controller.ocp.nlp[0].u_bounds["pulse_intensity_" + muscle_name].max[0][0]
            )
        else:
            raise ValueError(
                "For now, only the DingModelPulseWidthFrequency and DingModelPulseIntensityFrequency are"
                " supported for this cost function."
            )

        return stim_charge**power
