"""
This class regroups all the custom constraints that are used in the optimization problem.
"""

from casadi import MX, SX

from bioptim import PenaltyController

from .models.hmed2018 import DingModelIntensityFrequency


class CustomConstraint:
    @staticmethod
    def equal_to_first_pulse_interval_time(controller: PenaltyController) -> MX | SX:
        if controller.ocp.n_phases <= 1:
            RuntimeError(
                "There is only one phase, the bimapping constraint is not possible"
            )

        first_phase_tf = controller.ocp.node_time(
            0, controller.ocp.nlp[controller.phase_idx].ns
        )
        current_phase_tf = controller.ocp.nlp[controller.phase_idx].node_time(
            controller.ocp.nlp[controller.phase_idx].ns
        )

        return first_phase_tf - current_phase_tf

    @staticmethod
    def equal_to_first_pulse_duration(controller: PenaltyController) -> MX | SX:
        if controller.ocp.n_phases <= 1:
            RuntimeError(
                "There is only one phase, the bimapping constraint is not possible"
            )
        return (
            controller.parameters["pulse_duration"].cx[0]
            - controller.parameters["pulse_duration"].cx[controller.phase_idx]
        )

    @staticmethod
    def equal_to_first_pulse_intensity(controller: PenaltyController) -> MX | SX:
        if controller.ocp.n_phases <= 1:
            RuntimeError(
                "There is only one phase, the bimapping constraint is not possible"
            )
        return (
            controller.parameters["pulse_intensity"].cx[0]
            - controller.parameters["pulse_intensity"].cx[controller.phase_idx]
        )

    @staticmethod
    def cn_sum(controller: PenaltyController, stim_time: list) -> MX | SX:
        intensity_in_model = True if isinstance(controller.model, DingModelIntensityFrequency) else False
        lambda_i = [controller.model.lambda_i_calculation(controller.parameters["pulse_intensity"].cx[i]) for i in range(len(stim_time))] if intensity_in_model else [1 for _ in range(len(stim_time))]
        km = controller.states["Km"].cx if controller.model._with_fatigue else controller.model.km_rest
        r0 = km + controller.model.r0_km_relationship
        return controller.controls["Cn_sum"].cx - controller.model.cn_sum_fun(r0=r0, t=controller.time.cx, t_stim_prev=stim_time, lambda_i=lambda_i)

    @staticmethod
    def a_calculation(controller: PenaltyController, last_stim_index: int) -> MX | SX:
        a = controller.states["A"].cx if controller.model._with_fatigue else controller.model.a_scale
        last_stim_index = 0 if controller.parameters["pulse_duration"].cx.shape == (1, 1) else last_stim_index
        a_calculation = controller.model.a_calculation(
            a_scale=a,
            impulse_time=controller.parameters["pulse_duration"].cx[last_stim_index],
        )
        return controller.controls["A_calculation"].cx - a_calculation
