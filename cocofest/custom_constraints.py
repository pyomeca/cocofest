"""
This class regroups all the custom constraints that are used in the optimization problem.
"""

from casadi import MX, SX

from bioptim import PenaltyController


class CustomConstraint:
    @staticmethod
    def equal_to_first_pulse_interval_time(controller: PenaltyController) -> MX | SX:
        if controller.ocp.n_phases <= 1:
            RuntimeError("There is only one phase, the bimapping constraint is not possible")

        first_phase_tf = controller.ocp.node_time(0, controller.ocp.nlp[controller.phase_idx].ns)
        current_phase_tf = controller.ocp.nlp[controller.phase_idx].node_time(
            controller.ocp.nlp[controller.phase_idx].ns
        )

        return first_phase_tf - current_phase_tf

    @staticmethod
    def equal_to_first_pulse_duration(controller: PenaltyController) -> MX | SX:
        if controller.ocp.n_phases <= 1:
            RuntimeError("There is only one phase, the bimapping constraint is not possible")
        return (
            controller.parameters["pulse_duration"].cx[0]
            - controller.parameters["pulse_duration"].cx[controller.phase_idx]
        )

    @staticmethod
    def equal_to_first_pulse_intensity(controller: PenaltyController) -> MX | SX:
        if controller.ocp.n_phases <= 1:
            RuntimeError("There is only one phase, the bimapping constraint is not possible")
        return (
            controller.parameters["pulse_intensity"].cx[0]
            - controller.parameters["pulse_intensity"].cx[controller.phase_idx]
        )
