"""
This class regroups all the custom constraints that are used in the optimization problem.
"""

from casadi import MX, SX, vertcat
from bioptim import PenaltyController

from .models.dynamical_model import FesMskModel


class CustomConstraint:
    @staticmethod
    def pulse_intensity_sliding_window_constraint(
        controller: PenaltyController, last_stim_idx: int, muscle_name: str = ""
    ) -> MX | SX:
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
