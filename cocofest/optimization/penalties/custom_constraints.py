"""
This class regroups constraints that are not available through Bioptim and can be used in the optimization problem.
By adding definitions to this class, you can create your own custom constraints.
"""

from casadi import MX, SX, vertcat
from bioptim import PenaltyController

from ...models.dynamical_model import FesMskModel


class CustomConstraint:
    """
    Custom constraint functions not available in bioptim, usable as extra constraints in an optimal control
    program.
    """

    @staticmethod
    def pulse_intensity_sliding_window_constraint(
        controller: PenaltyController, last_stim_idx: int, muscle_name: str = ""
    ) -> MX | SX:
        """
        Constrain the pulse intensity control to match the truncated window of identified/optimized parameters.

        Parameters
        ----------
        controller: PenaltyController
            The penalty node elements
        last_stim_idx: int
            The index of the last stimulation considered at this node
        muscle_name: str
            The muscle name, if any

        Returns
        -------
        MX | SX
            The difference between the pulse intensity control and the truncated parameter window (constrained to 0)
        """
        key = "pulse_intensity" + "_" + str(muscle_name) if muscle_name else "pulse_intensity"
        parameters = [controller.parameters[key].cx[i] for i in range(last_stim_idx + 1)]
        if isinstance(controller.model, FesMskModel):
            model = controller.model.muscles_dynamics_model[0]
        else:
            model = controller.model

        while len(parameters) < controller.controls[key].cx.shape[0]:
            min_intensity = model.min_pulse_intensity() if isinstance(model.min_pulse_intensity(), int | float) else 0
            parameters.insert(0, min_intensity)
        if len(parameters) > controller.controls[key].cx.shape[0]:
            size_diff = len(parameters) - controller.controls[key].cx.shape[0]
            parameters = parameters[size_diff:]

        return controller.controls[key].cx - vertcat(*parameters)
