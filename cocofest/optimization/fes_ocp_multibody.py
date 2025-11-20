import numpy as np

from bioptim import (
    BoundsList,
    ConstraintList,
    ExternalForceSetTimeSeries,
    InitialGuessList,
    InterpolationType,
    ParameterList,
    VariableScaling,
)

from ..models.ding2007.ding2007 import DingModelPulseWidthFrequency
from ..models.dynamical_model import FesMskModel
from ..models.hmed2018.hmed2018 import DingModelPulseIntensityFrequency
from ..optimization.fes_ocp import OcpFes
from ..custom_constraints import CustomConstraint


class OcpFesMsk(OcpFes):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_numerical_time_series_for_external_forces(n_shooting, external_force_dict):
        external_force_set = ExternalForceSetTimeSeries(nb_frames=n_shooting)
        external_force_array = np.array(external_force_dict["torque"])
        reshape_values_array = np.tile(external_force_array[:, np.newaxis], (1, n_shooting))
        external_force_set.add_torque(
            segment=external_force_dict["Segment_application"],
            values=reshape_values_array,
            force_name="resistive_torque",
        )

        numerical_time_series = {"external_forces": external_force_set.to_numerical_time_series()}

        return numerical_time_series, external_force_set

    @staticmethod
    def build_parameters(
        model: FesMskModel,
        max_pulse_intensity: int,
        use_sx: bool = True,
    ):
        parameters = ParameterList(use_sx=use_sx)
        parameters_bounds = BoundsList()
        parameters_init = InitialGuessList()

        n_stim = len(model.muscles_dynamics_model[0].stim_time)

        for i in range(len(model.muscles_dynamics_model)):
            if isinstance(model.muscles_dynamics_model[i], DingModelPulseIntensityFrequency):
                parameter_name = "pulse_intensity" + "_" + model.muscles_dynamics_model[i].muscle_name

                parameters_bounds.add(
                    parameter_name,
                    min_bound=[model.muscles_dynamics_model[i].min_pulse_intensity()],
                    max_bound=[max_pulse_intensity],
                    interpolation=InterpolationType.CONSTANT,
                )
                intensity_avg = (model.muscles_dynamics_model[i].min_pulse_intensity() + max_pulse_intensity) / 2
                parameters_init[parameter_name] = np.array([intensity_avg] * n_stim)
                parameters.add(
                    name=parameter_name,
                    function=DingModelPulseIntensityFrequency.set_impulse_intensity,
                    size=n_stim,
                    scaling=VariableScaling(parameter_name, [1] * n_stim),
                )

        return parameters, parameters_bounds, parameters_init

    @staticmethod
    def set_constraints(models, n_shooting, stim_idx_at_node_list, custom_constraint=None):
        constraints = ConstraintList()
        for model in models.muscles_dynamics_model:
            if isinstance(model, DingModelPulseIntensityFrequency):
                for i in range(n_shooting):
                    last_stim_idx = stim_idx_at_node_list[i][-1]
                    constraints.add(
                        CustomConstraint.pulse_intensity_sliding_window_constraint,
                        last_stim_idx=last_stim_idx,
                        muscle_name=model.muscle_name,
                        node=i,
                    )

        if custom_constraint:
            for i in range(len(custom_constraint)):
                if custom_constraint[i]:
                    for j in range(len(custom_constraint[i])):
                        constraints.add(custom_constraint[i][j])

        return constraints

    @staticmethod
    def set_x_bounds_fes(bio_models):
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
        x_init = InitialGuessList()
        for model in bio_models.muscles_dynamics_model:
            muscle_name = model.muscle_name
            variable_bound_list = [model.name_dof[i] + "_" + muscle_name for i in range(len(model.name_dof))]

            starting_bounds, min_bounds, max_bounds = (
                model.standard_rest_values(),
                model.standard_rest_values(),
                model.standard_rest_values(),
            )

            for i in range(len(variable_bound_list)):
                if variable_bound_list[i] == "Cn_" + muscle_name:
                    max_bounds[i] = 10
                elif variable_bound_list[i] == "F_" + muscle_name:
                    max_bounds[i] = model.fmax
                elif variable_bound_list[i] == "Tau1_" + muscle_name or variable_bound_list[i] == "Km_" + muscle_name:
                    max_bounds[i] = 1
                elif variable_bound_list[i] == "A_" + muscle_name:
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

            for j in range(len(variable_bound_list)):
                x_init.add(variable_bound_list[j], model.standard_rest_values()[j], phase=0)

        return x_bounds, x_init

    @staticmethod
    def set_x_bounds_msk(x_bounds, x_init, bio_models, msk_info):
        if msk_info["bound_type"] == "start_end":
            start_bounds = []
            end_bounds = []
            for i in range(bio_models.nb_q):
                start_bounds.append(
                    3.14 / (180 / msk_info["bound_data"][0][i]) if msk_info["bound_data"][0][i] != 0 else 0
                )
                end_bounds.append(
                    3.14 / (180 / msk_info["bound_data"][1][i]) if msk_info["bound_data"][1][i] != 0 else 0
                )

        elif msk_info["bound_type"] == "start":
            start_bounds = []
            for i in range(bio_models.nb_q):
                start_bounds.append(3.14 / (180 / msk_info["bound_data"][i]) if msk_info["bound_data"][i] != 0 else 0)

        elif msk_info["bound_type"] == "end":
            end_bounds = []
            for i in range(bio_models.nb_q):
                end_bounds.append(3.14 / (180 / msk_info["bound_data"][i]) if msk_info["bound_data"][i] != 0 else 0)

        q_x_bounds = bio_models.bounds_from_ranges("q")
        qdot_x_bounds = bio_models.bounds_from_ranges("qdot")

        if msk_info["bound_type"] == "start_end":
            for j in range(bio_models.nb_q):
                q_x_bounds[j, [0]] = start_bounds[j]
                q_x_bounds[j, [-1]] = end_bounds[j]
        elif msk_info["bound_type"] == "start":
            for j in range(bio_models.nb_q):
                q_x_bounds[j, [0]] = start_bounds[j]
        elif msk_info["bound_type"] == "end":
            for j in range(bio_models.nb_q):
                q_x_bounds[j, [-1]] = end_bounds[j]
        qdot_x_bounds[:, [0]] = 0  # Start without any velocity

        x_bounds.add(key="q", bounds=q_x_bounds, phase=0)
        x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=0)

        return x_bounds, x_init

    @staticmethod
    def set_x_bounds(bio_models, msk_info):
        x_bounds, x_init = OcpFesMsk.set_x_bounds_fes(bio_models)
        x_bounds, x_init = OcpFesMsk.set_x_bounds_msk(x_bounds, x_init, bio_models, msk_info)
        return x_bounds, x_init

    @staticmethod
    def set_u_bounds_fes(bio_models):
        u_bounds = BoundsList()  # Controls bounds
        u_init = InitialGuessList()  # Controls initial guess
        models = bio_models.muscles_dynamics_model
        if isinstance(models[0], DingModelPulseWidthFrequency):
            for model in models:
                key = "last_pulse_width_" + str(model.muscle_name)
                u_init.add(key=key, initial_guess=[model.pd0], phase=0)
                u_bounds.add(key=key, min_bound=[model.pd0], max_bound=[0.0006], phase=0)

        if isinstance(models[0], DingModelPulseIntensityFrequency):
            for model in models:
                key = "pulse_intensity_" + str(model.muscle_name)
                u_init.add(key=key, initial_guess=[0] * model.sum_stim_truncation, phase=0)
                min_pulse_intensity = (
                    model.min_pulse_intensity() if isinstance(model.min_pulse_intensity(), int | float) else 0
                )
                u_bounds.add(
                    key=key,
                    min_bound=[min_pulse_intensity] * model.sum_stim_truncation,
                    max_bound=[130] * model.sum_stim_truncation,
                    interpolation=InterpolationType.CONSTANT,
                )
        return u_bounds, u_init

    @staticmethod
    def set_u_bounds_msk(u_bounds, u_init, bio_models, with_residual_torque, max_bound=None):
        if with_residual_torque:  # TODO : ADD SEVERAL INDIVIDUAL FIXED RESIDUAL TORQUE FOR EACH JOINT
            nb_tau = bio_models.nb_tau
            tau_min, tau_max, tau_init = [-200] * nb_tau, [200] * nb_tau, [0] * nb_tau
            u_bounds.add(
                key="tau", min_bound=tau_min, max_bound=tau_max, phase=0, interpolation=InterpolationType.CONSTANT
            )
            u_init.add(key="tau", initial_guess=tau_init, phase=0)

        models = bio_models.muscles_dynamics_model
        if isinstance(models[0], DingModelPulseWidthFrequency):
            max_bound = max_bound if max_bound else 0.0006
            for model in models:
                key = "last_pulse_width_" + str(model.muscle_name)
                u_init.add(key=key, initial_guess=[0], phase=0)
                u_bounds.add(key=key, min_bound=[model.pd0], max_bound=[max_bound], phase=0)

        if isinstance(models[0], DingModelPulseIntensityFrequency):
            max_bound = max_bound if max_bound else 130
            for model in models:
                key = "pulse_intensity_" + str(model.muscle_name)
                u_init.add(key=key, initial_guess=[0] * model.sum_stim_truncation, phase=0)
                u_bounds.add(
                    key=key,
                    min_bound=[model.min_pulse_intensity()] * model.sum_stim_truncation,
                    max_bound=[max_bound] * model.sum_stim_truncation,
                    phase=0,
                )

        return u_bounds, u_init

    @staticmethod
    def set_u_bounds(bio_models, with_residual_torque, max_bound=None):
        u_bounds, u_init = OcpFesMsk.set_u_bounds_fes(bio_models)
        u_bounds, u_init = OcpFesMsk.set_u_bounds_msk(u_bounds, u_init, bio_models, with_residual_torque, max_bound)
        return u_bounds, u_init

    @staticmethod
    def update_model(model, parameters, external_force_set):
        # rebuilding model for the OCP
        return FesMskModel(
            name=model.name,
            biorbd_path=model.biorbd_path,
            muscles_model=model.muscles_dynamics_model,
            stim_time=model.muscles_dynamics_model[0].stim_time,
            previous_stim=model.muscles_dynamics_model[0].previous_stim,
            activate_force_length_relationship=model.activate_force_length_relationship,
            activate_force_velocity_relationship=model.activate_force_velocity_relationship,
            activate_passive_force_relationship=model.activate_passive_force_relationship,
            activate_residual_torque=model.activate_residual_torque,
            parameters=parameters,
            external_force_set=external_force_set,
        )
