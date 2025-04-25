import numpy as np

from bioptim import (
    BoundsList,
    InitialGuessList,
    InterpolationType,
    ParameterList,
    VariableScaling,
)

from ..models.fes_model import FesModel
from ..models.ding2003 import DingModelFrequency
from ..models.ding2007 import DingModelPulseWidthFrequency
from ..models.hmed2018 import DingModelPulseIntensityFrequency
from ..optimization.fes_ocp import OcpFes


class OcpFesId(OcpFes):
    def __init__(self):
        super(OcpFesId, self).__init__()

    @staticmethod
    def set_x_bounds(
        model: FesModel = None,
        force_tracking=None,
    ):
        # ---- STATE BOUNDS REPRESENTATION ---- #
        #
        #                    |‾‾‾‾‾‾‾‾‾‾x_max_middle‾‾‾‾‾‾‾‾‾‾‾‾x_max_end‾
        #                    |          max_bounds              max_bounds
        #    x_max_start     |
        #   _starting_bounds_|
        #   ‾starting_bounds‾|
        #    x_min_start     |
        #                    |          min_bounds              min_bounds
        #                     ‾‾‾‾‾‾‾‾‾‾x_min_middle‾‾‾‾‾‾‾‾‾‾‾‾x_min_end‾

        # Sets the bound for all the phases
        x_bounds = BoundsList()
        variable_bound_list = model.name_dof
        starting_bounds, min_bounds, max_bounds = (
            model.standard_rest_values(),
            model.standard_rest_values(),
            model.standard_rest_values(),
        )

        for i in range(len(variable_bound_list)):
            if variable_bound_list[i] == "Cn":
                max_bounds[i] = 10
            elif variable_bound_list[i] == "F":
                max_bounds[i] = 500
            elif variable_bound_list[i] == "Tau1" or variable_bound_list[i] == "Km":
                max_bounds[i] = 1
            elif variable_bound_list[i] == "A":
                min_bounds[i] = 0

        starting_bounds_min = np.concatenate((starting_bounds, min_bounds, min_bounds), axis=1)
        starting_bounds_max = np.concatenate((starting_bounds, max_bounds, max_bounds), axis=1)

        for j in range(len(variable_bound_list)):
            x_bounds.add(
                variable_bound_list[j],
                min_bound=np.array([starting_bounds_min[j]]),
                max_bound=np.array([starting_bounds_max[j]]),
                phase=0,
                interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
            )

        x_init = InitialGuessList()

        x_init.add(
            "F",
            np.array([force_tracking]),
            phase=0,
            interpolation=InterpolationType.EACH_FRAME,
        )
        x_init.add("Cn", [0], phase=0, interpolation=InterpolationType.CONSTANT)
        if model._with_fatigue:
            for j in range(len(variable_bound_list)):
                if variable_bound_list[j] == "F" or variable_bound_list[j] == "Cn":
                    pass
                else:
                    x_init.add(variable_bound_list[j], model.standard_rest_values()[j])

        return x_bounds, x_init

    @staticmethod
    def set_parameters(
        parameter_to_identify,
        parameter_setting,
        use_sx,
    ):
        parameters = ParameterList(use_sx=use_sx)
        parameters_bounds = BoundsList()
        parameters_init = InitialGuessList()

        for param in parameter_to_identify:
            parameters.add(
                name=param,
                function=parameter_setting[param]["function"],
                size=1,
                scaling=VariableScaling(
                    param,
                    [parameter_setting[param]["scaling"]],
                ),
            )
            parameters_bounds.add(
                param,
                min_bound=np.array([parameter_setting[param]["min_bound"]]),
                max_bound=np.array([parameter_setting[param]["max_bound"]]),
                interpolation=InterpolationType.CONSTANT,
            )
            parameters_init.add(
                key=param,
                initial_guess=np.array([parameter_setting[param]["initial_guess"]]),
            )

        return parameters, parameters_bounds, parameters_init

    @staticmethod
    def set_u_bounds(model, control_value: list, stim_idx_at_node_list: list, n_shooting: int):
        u_bounds = BoundsList()  # Controls bounds
        u_init = InitialGuessList()  # Controls initial guess
        if isinstance(model, DingModelPulseWidthFrequency):
            if len(control_value) != 1:
                last_stim_idx = [stim_idx_at_node_list[i][-1] for i in range(len(stim_idx_at_node_list) - 1)]
                control_bounds = [control_value[last_stim_idx[i]] for i in range(len(last_stim_idx))]
            else:
                control_bounds = [control_value] * n_shooting
            u_init.add(key="last_pulse_width", initial_guess=[0], phase=0)
            u_bounds.add(
                "last_pulse_width",
                min_bound=np.array([control_bounds]),
                max_bound=np.array([control_bounds]),
                interpolation=InterpolationType.EACH_FRAME,
            )

        if isinstance(model, DingModelPulseIntensityFrequency):
            padded = [
                [control_value[stim_idx_at_node_list[i][0]]]
                * (model.sum_stim_truncation - len(stim_idx_at_node_list[i]))
                + [control_value[idx] for idx in stim_idx_at_node_list[i]]
                for i in range(n_shooting)
            ]
            control_list = [list(row) for row in zip(*padded)]
            u_init.add(key="pulse_intensity", initial_guess=np.array(control_list)[:, 0], phase=0)
            u_bounds.add(
                "pulse_intensity",
                min_bound=np.array(control_list),
                max_bound=np.array(control_list),
                interpolation=InterpolationType.EACH_FRAME,
            )

        return u_bounds, u_init

    @staticmethod
    def set_default_values(model):
        """
        Sets the default values for the identified parameters (initial guesses, bounds, scaling and function).
        If the user does not provide additional_key_settings for a specific parameter, the default value will be used.

        Parameters
        ----------
        model: FesModel
            The model to use for the OCP.

        Returns
        -------
        dict
            A dictionary of default values for the identified parameters.
        """
        if isinstance(model, DingModelPulseWidthFrequency):
            return {
                "tau1_rest": {
                    "initial_guess": 0.5,
                    "min_bound": 0.0001,
                    "max_bound": 1,
                    "function": model.set_tau1_rest,
                    "scaling": 1,  # 10000
                },
                "tau2": {
                    "initial_guess": 0.5,
                    "min_bound": 0.0001,
                    "max_bound": 1,
                    "function": model.set_tau2,
                    "scaling": 1,  # 10000
                },
                "km_rest": {
                    "initial_guess": 0.5,
                    "min_bound": 0.001,
                    "max_bound": 1,
                    "function": model.set_km_rest,
                    "scaling": 1,  # 10000
                },
                "a_scale": {
                    "initial_guess": 5000,
                    "min_bound": 1,
                    "max_bound": 10000,
                    "function": model.set_a_scale,
                    "scaling": 1,
                },
                "pd0": {
                    "initial_guess": 1e-4,
                    "min_bound": 1e-4,
                    "max_bound": 6e-4,
                    "function": model.set_pd0,
                    "scaling": 1,  # 1000
                },
                "pdt": {
                    "initial_guess": 1e-4,
                    "min_bound": 1e-4,
                    "max_bound": 6e-4,
                    "function": model.set_pdt,
                    "scaling": 1,  # 1000
                },
            }
        elif isinstance(model, DingModelPulseIntensityFrequency):
            return {
                "a_rest": {
                    "initial_guess": 1000,
                    "min_bound": 1,
                    "max_bound": 10000,
                    "function": model.set_a_rest,
                    "scaling": 1,
                },
                "km_rest": {
                    "initial_guess": 0.5,
                    "min_bound": 0.001,
                    "max_bound": 1,
                    "function": model.set_km_rest,
                    "scaling": 1,  # 1000
                },
                "tau1_rest": {
                    "initial_guess": 0.5,
                    "min_bound": 0.0001,
                    "max_bound": 1,
                    "function": model.set_tau1_rest,
                    "scaling": 1,  # 1000
                },
                "tau2": {
                    "initial_guess": 0.5,
                    "min_bound": 0.0001,
                    "max_bound": 1,
                    "function": model.set_tau2,
                    "scaling": 1,  # 1000
                },
                "ar": {
                    "initial_guess": 0.5,
                    "min_bound": 0.01,
                    "max_bound": 1,
                    "function": model.set_ar,
                    "scaling": 1,
                },  # 100
                "bs": {
                    "initial_guess": 0.05,
                    "min_bound": 0.001,
                    "max_bound": 0.1,
                    "function": model.set_bs,
                    "scaling": 1,  # 1000
                },
                "Is": {
                    "initial_guess": 50,
                    "min_bound": 1,
                    "max_bound": 150,
                    "function": model.set_Is,
                    "scaling": 1,
                },
                "cr": {
                    "initial_guess": 1,
                    "min_bound": 0.01,
                    "max_bound": 2,
                    "function": model.set_cr,
                    "scaling": 1,
                },  # 100
            }
        elif isinstance(model, DingModelFrequency):
            return {
                "a_rest": {
                    "initial_guess": 1000,
                    "min_bound": 1,
                    "max_bound": 10000,
                    "function": model.set_a_rest,
                    "scaling": 1,
                },
                "km_rest": {
                    "initial_guess": 0.5,
                    "min_bound": 0.001,
                    "max_bound": 1,
                    "function": model.set_km_rest,
                    "scaling": 1,
                },
                "tau1_rest": {
                    "initial_guess": 0.5,
                    "min_bound": 0.0001,
                    "max_bound": 1,
                    "function": model.set_tau1_rest,
                    "scaling": 1,
                },
                "tau2": {
                    "initial_guess": 0.5,
                    "min_bound": 0.0001,
                    "max_bound": 1,
                    "function": model.set_tau2,
                    "scaling": 1,
                },
            }
