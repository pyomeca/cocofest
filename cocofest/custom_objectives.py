"""
This custom objective class regroups all the custom objectives that are used in the optimization problem.
"""

from casadi import MX, vertcat
from bioptim import PenaltyController
from .models.ding2007.ding2007 import DingModelPulseWidthFrequency
from .models.dynamical_model import FesMskModel


class CustomObjective:
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
        muscle_name_list = controller.model.muscle_names
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
        muscle_name_list = controller.model.muscle_names
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
        if isinstance(controller.model, FesMskModel):
            muscle_name_list = controller.model.muscle_names
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
