"""
This custom objective is to enable the tracking of a curve by a state at all node. Used for sample data control problems
such as functional electro stimulation
"""

import numpy as np
from casadi import MX, SX, sum1, horzcat

from bioptim import PenaltyController
from .fourier_approx import FourierSeries


class CustomObjective:
    @staticmethod
    def track_state_from_time(
        controller: PenaltyController, fourier_coeff: np.ndarray, key: str
    ) -> MX | SX:
        """
        Minimize the states variables.
        By default, this function is quadratic, meaning that it minimizes towards the target.
        Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements
        fourier_coeff: np.ndarray
            The values to aim for
        key: str
            The name of the state to minimize

        Returns
        -------
        The difference between the two keys
        """
        # get the approximated force value from the fourier series at the node time
        value_from_fourier = (
            FourierSeries().fit_func_by_fourier_series_with_real_coeffs(
                controller.ocp.node_time(
                    phase_idx=controller.phase_idx, node_idx=controller.t[0]
                ),
                fourier_coeff,
                mode="casadi",
            )
        )
        return value_from_fourier - controller.states[key].cx

    @staticmethod
    def track_state_from_time_interpolate(
        controller: PenaltyController,
        force: np.ndarray,
        key: str,
        minimization_type: str = "least square",
    ) -> MX:
        """
        Minimize the states variables.
        This function least square.
        Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements
        force: np.ndarray
            The force vector
        key: str
            The name of the state to minimize
        minimization_type: str
            The type of minimization to perform. Either "least square" or "best fit"

        Returns
        -------
        The difference between the two keys
        """
        if minimization_type == "least square":
            return force - controller.states[key].cx
        elif minimization_type == "best fit":
            return 1 - (force / controller.states[key].cx)
        else:
            raise RuntimeError(f"Minimization type {minimization_type} not implemented")

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
        muscle_fatigue_rest = horzcat(
            *[
                controller.model.bio_stim_model[x].a_rest
                for x in range(1, len(muscle_name_list) + 1)
            ]
        )
        muscle_fatigue = horzcat(
            *[
                controller.states["A_" + muscle_name_list[x]].cx
                for x in range(len(muscle_name_list))
            ]
        )
        return sum1(muscle_fatigue_rest) / sum1(muscle_fatigue)

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
        muscle_force = horzcat(
            *[
                controller.states["F_" + muscle_name_list[x]].cx
                for x in range(len(muscle_name_list))
            ]
        )
        return sum1(muscle_force)
