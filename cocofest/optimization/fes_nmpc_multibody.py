import numpy as np
from casadi import SX
from bioptim import (
    BiorbdModel,
    InitialGuessList,
    InterpolationType,
    ParameterList,
    VariableScaling,
    Solution,
    OptimalControlProgram,
    Solver,
    MultiCyclicCycleSolutions,
    ExternalForceSetTimeSeries,
)
from .fes_nmpc import FesNmpc
from ..models.dynamical_model import FesMskModel


class FesNmpcMsk(FesNmpc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_new_model(self, model, previous_stim_time):
        new_model = FesMskModel(
            name=model.name,
            biorbd_path=model.biorbd_path,
            muscles_model=model.muscles_dynamics_model,
            stim_time=model.muscles_dynamics_model[0].stim_time,
            previous_stim=previous_stim_time,
            activate_force_length_relationship=model.activate_force_length_relationship,
            activate_force_velocity_relationship=model.activate_force_velocity_relationship,
            activate_residual_torque=model.activate_residual_torque,
            parameters=self.nlp[0].parameters,
            external_force_set=model.external_force_set,
        )
        return new_model

    def update_stim(self):
        if isinstance(self.nlp[0].model, FesMskModel):
            muscle_model = self.nlp[0].model.muscles_dynamics_model[0]
            truncation_term = muscle_model.sum_stim_truncation
            solution_stimulation_time = muscle_model.stim_time[-truncation_term:]
            previous_stim_time = [x - self.phase_time[0] for x in solution_stimulation_time]
            previous_stim = {"time": previous_stim_time}
            new_model = self.build_new_model(model=self.nlp[0].model, previous_stim_time=previous_stim)
            if self.first_run:
                numerical_data_timeseries, _ = new_model.muscles_dynamics_model[0].get_numerical_data_time_series(
                    self.n_shooting, self.phase_time[0]
                )
                self.nlp[0].numerical_data_timeseries["stim_time"] = numerical_data_timeseries["stim_time"]
                self.first_run = False

            self.nlp[0].model = new_model
            self.all_models.append(new_model)

    def _initialize_solution(self, dt: float, states: list, controls: list, parameters: list):
        combine_model = False if isinstance(self.nlp[0].model, BiorbdModel) else True
        combined_model = self.create_model_from_list(self.all_models) if combine_model else self.nlp[0].model
        x_init = InitialGuessList()
        for key in self.nlp[0].states.keys():
            x_init.add(
                key,
                np.concatenate([state[key][:, :-1] for state in states] + [states[-1][key][:, -1:]], axis=1),
                interpolation=InterpolationType.EACH_FRAME,
                phase=0,
            )

        u_init = InitialGuessList()
        for key in self.nlp[0].controls.keys():
            controls_tp = np.concatenate([control[key] for control in controls], axis=1)
            u_init.add(key, controls_tp, interpolation=InterpolationType.EACH_FRAME, phase=0)

        p_init = InitialGuessList()
        if combine_model:
            stimulation_per_cycle = int(len(self.nlp[0].model.stim_time) / self.n_cycles)
            for key in self.nlp[0].parameters.keys():
                combined_parameters = (
                    [[parameters[i][key][0]] * stimulation_per_cycle for i in range(len(parameters))]
                    if self.bimapped_param
                    else [list(parameter[key][:stimulation_per_cycle]) for parameter in parameters]
                )
                combined_parameters = [val for sublist in combined_parameters for val in sublist]
                p_init[key] = combined_parameters

            parameters = ParameterList(use_sx=self.cx == SX)
            for key in self.nlp[0].parameters.keys():
                parameters.add(
                    name=key,
                    function=self.nlp[0].parameters[key].function,
                    size=len(combined_parameters),
                    scaling=VariableScaling(key, [1] * len(combined_parameters)),
                )
        else:
            parameters = ParameterList(use_sx=self.cx == SX)

        solution_ocp = OptimalControlProgram(
            bio_model=[combined_model],
            dynamics=self.nlp[0].dynamics_type,
            n_shooting=self.total_optimization_run * self.cycle_len,
            phase_time=self.total_optimization_run * self.cycle_len * dt,
            x_init=x_init,
            u_init=u_init,
            use_sx=self.cx == SX,
            parameters=parameters,
            parameter_init=p_init,
            ode_solver=self.nlp[0].ode_solver,
        )
        a_init = InitialGuessList()
        return Solution.from_initial_guess(solution_ocp, [np.array([dt]), x_init, u_init, p_init, a_init])

    def create_model_from_list(self, models: list):
        if isinstance(models[0], BiorbdModel):
            return models[0]

        stimulation_per_cycle = int(len(self.nlp[0].model.stim_time) / self.n_cycles)
        stim_time = []
        for i in range(len(models)):
            stim_time.append(list(np.array(models[0].stim_time[:stimulation_per_cycle]) + (i * self.cycle_duration)))
        stim_time = [val for sublist in stim_time for val in sublist]

        combined_model = FesMskModel(
            name=self.nlp[0].model.name,
            biorbd_path=self.nlp[0].model.biorbd_path,
            muscles_model=self.nlp[0].model.muscles_dynamics_model,
            stim_time=stim_time,
            previous_stim={},
            activate_force_length_relationship=self.nlp[0].model.activate_force_length_relationship,
            activate_force_velocity_relationship=self.nlp[0].model.activate_force_velocity_relationship,
            activate_residual_torque=self.nlp[0].model.activate_residual_torque,
            parameters=self.nlp[0].model.parameters,
            external_force_set=self.nlp[0].model.external_force_set,
        )

        return combined_model

    def get_stim_time_from_all_models(self):
        stim_time = []
        offset = 0.0
        for model in self.all_models:
            current_stim = model.muscles_dynamics_model[0].stim_time
            current_stim = [val for val in current_stim if val < self.cycle_duration]
            shifted_stim = [t + offset for t in current_stim]
            stim_time.extend(shifted_stim)
            offset = shifted_stim[-1] + (current_stim[1] - current_stim[0])
        return stim_time

    def solve_fes_nmpc(
        self,
        update_functions,
        solver: Solver.IPOPT,
        total_cycles: int,
        external_force: dict,
        cycle_solutions: MultiCyclicCycleSolutions,
        get_all_iterations: bool = True,
        cyclic_options: dict = None,
        max_consecutive_failing: int = 3,
    ):

        sol = self.solve(
            update_functions,
            solver=solver,
            cycle_solutions=cycle_solutions,
            get_all_iterations=get_all_iterations,
            cyclic_options=cyclic_options,
            n_cycles_simultaneous=self.n_cycles_simultaneous,
            max_consecutive_failing=max_consecutive_failing,
        )
        model = self.nlp[0].model

        total_external_forces_frame = total_cycles * self.cycle_len
        total_nmpc_duration = self.cycle_duration * total_cycles
        total_nmpc_shooting_len = self.cycle_len * self.n_cycles

        external_force_set = ExternalForceSetTimeSeries(nb_frames=total_external_forces_frame)
        external_force_array = np.array(external_force["torque"])
        reshape_values_array = np.tile(external_force_array[:, np.newaxis], (1, total_external_forces_frame))
        external_force_set.add_torque(segment=external_force["Segment_application"], values=reshape_values_array)
        numerical_time_series = {"external_forces": external_force_set.to_numerical_time_series()}

        if isinstance(model, FesMskModel):
            all_stim_time = self.get_stim_time_from_all_models()
            self.nlp[0].model.muscles_dynamics_model[0].stim_time = all_stim_time
            numerical_data_time_series, stim_idx_at_node_list = model.muscles_dynamics_model[
                0
            ].get_numerical_data_time_series(total_nmpc_shooting_len, total_nmpc_duration)
            numerical_time_series.update(numerical_data_time_series)

        sol[0].ocp.nlp[0].numerical_data_timeseries = numerical_time_series

        return sol
