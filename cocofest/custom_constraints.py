"""
This class regroups all the custom constraints that are used in the optimization problem.
"""

from casadi import MX, SX, vertcat

from bioptim import PenaltyController

from .models.hmed2018 import DingModelIntensityFrequency


class CustomConstraint:
    @staticmethod
    def cn_sum(controller: PenaltyController, stim_time: list, stim_index: list) -> MX | SX:
        intensity_in_model = True if isinstance(controller.model, DingModelIntensityFrequency) else False
        lambda_i = (
            [controller.model.lambda_i_calculation(controller.parameters["pulse_intensity"].cx[i]) for i in stim_index]
            if intensity_in_model
            else [1 for _ in range(len(stim_time))]
        )
        km = controller.states["Km"].cx if controller.model._with_fatigue else controller.model.km_rest
        r0 = km + controller.model.r0_km_relationship
        return controller.controls["Cn_sum"].cx - controller.model.cn_sum_fun(
            r0=r0, t=controller.time.cx, t_stim_prev=stim_time, lambda_i=lambda_i
        )

    @staticmethod
    def cn_sum_msk(controller: PenaltyController, stim_time: list, model_idx: int, sum_truncation: int) -> MX | SX:
        model = controller.model.muscles_dynamics_model[model_idx]
        muscle_name = model.muscle_name
        stim_time = model.stim_prev + stim_time
        stim_time = stim_time[-sum_truncation:]

        if isinstance(model, DingModelIntensityFrequency):
            intensity_list = vertcat(
                model.stim_pulse_intensity_prev, controller.parameters["pulse_intensity_" + muscle_name].cx
            )
            sum_truncation = sum_truncation if sum_truncation < intensity_list.shape[0] else intensity_list.shape[0]
            intensity_list = intensity_list[-sum_truncation:]
            lambda_i = [model.lambda_i_calculation(intensity_list[i]) for i in range(intensity_list.shape[0])]
        else:
            lambda_i = [1 for _ in range(len(stim_time))]
        km = controller.states["Km_" + muscle_name].cx if model.with_fatigue else model.km_rest
        r0 = km + model.r0_km_relationship
        return controller.controls["Cn_sum_" + muscle_name].cx - model.cn_sum_fun(
            r0=r0, t=controller.time.cx, t_stim_prev=stim_time, lambda_i=lambda_i
        )

    @staticmethod
    def cn_sum_identification(controller: PenaltyController, stim_time: list, stim_index: list) -> MX | SX:
        intensity_in_model = True if isinstance(controller.model, DingModelIntensityFrequency) else False
        ar, bs, Is, cr = None, None, None, None
        if intensity_in_model:
            ar = controller.parameters["ar"].cx if "ar" in controller.parameters.keys() else controller.model.ar
            bs = controller.parameters["bs"].cx if "bs" in controller.parameters.keys() else controller.model.bs
            Is = controller.parameters["Is"].cx if "Is" in controller.parameters.keys() else controller.model.Is
            cr = controller.parameters["cr"].cx if "cr" in controller.parameters.keys() else controller.model.cr
        lambda_i = (
            [
                controller.model.lambda_i_calculation_identification(
                    controller.parameters["pulse_intensity"].cx[i], ar, bs, Is, cr
                )
                for i in stim_index
            ]
            if intensity_in_model
            else [1 for _ in range(len(stim_time))]
        )
        km = (
            controller.parameters["km_rest"].cx
            if "km_rest" in controller.parameters.keys()
            else controller.model.km_rest
        )
        r0 = km + controller.model.r0_km_relationship
        return controller.controls["Cn_sum"].cx - controller.model.cn_sum_fun(
            r0=r0, t=controller.time.cx, t_stim_prev=stim_time, lambda_i=lambda_i
        )

    @staticmethod
    def a_calculation(controller: PenaltyController, last_stim_index: int) -> MX | SX:
        a = controller.states["A"].cx if controller.model.with_fatigue else controller.model.a_scale
        last_stim_index = 0 if controller.parameters["pulse_duration"].cx.shape == (1, 1) else last_stim_index
        a_calculation = controller.model.a_calculation(
            a_scale=a,
            impulse_time=controller.parameters["pulse_duration"].cx[last_stim_index],
        )
        return controller.controls["A_calculation"].cx - a_calculation

    @staticmethod
    def a_calculation_msk(controller: PenaltyController, last_stim_index: int, model_idx: int) -> MX | SX:
        model = controller.model.muscles_dynamics_model[model_idx]
        muscle_name = model.muscle_name
        a = controller.states["A_" + muscle_name].cx if model.with_fatigue else model.a_scale
        last_stim_index = (
            0 if controller.parameters["pulse_duration_" + muscle_name].cx.shape == (1, 1) else last_stim_index
        )
        a_calculation = model.a_calculation(
            a_scale=a,
            impulse_time=controller.parameters["pulse_duration_" + muscle_name].cx[last_stim_index],
        )
        return controller.controls["A_calculation_" + muscle_name].cx - a_calculation

    @staticmethod
    def a_calculation_identification(controller: PenaltyController, last_stim_index: int) -> MX | SX:
        a = (
            controller.parameters["a_scale"].cx
            if "a_scale" in controller.parameters.keys()
            else controller.model.a_scale
        )
        pd0 = controller.parameters["pd0"].cx if "pd0" in controller.parameters.keys() else controller.model.pd0
        pdt = controller.parameters["pdt"].cx if "pdt" in controller.parameters.keys() else controller.model.pdt
        last_stim_index = 0 if controller.parameters["pulse_duration"].cx.shape == (1, 1) else last_stim_index
        a_calculation = controller.model.a_calculation_identification(
            a_scale=a,
            impulse_time=controller.parameters["pulse_duration"].cx[last_stim_index],
            pd0=pd0,
            pdt=pdt,
        )
        return controller.controls["A_calculation"].cx - a_calculation
