"""
This custom objective class regroups all the custom objectives that are used in the optimization problem.
"""

from casadi import MX, vertcat
from bioptim import PenaltyController


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
        muscle_name_list = controller.model.bio_model.muscle_names
        muscle_fatigue = vertcat(
            *[controller.states["A_" + muscle_name_list[x]].cx for x in range(len(muscle_name_list))]
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
        muscle_name_list = controller.model.bio_model.muscle_names
        muscle_force = vertcat(
            *[controller.states["F_" + muscle_name_list[x]].cx for x in range(len(muscle_name_list))]
        )
        return muscle_force
