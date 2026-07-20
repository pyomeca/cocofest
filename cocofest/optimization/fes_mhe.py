import numpy as np

from casadi import SX
from bioptim import (
    MultiCyclicNonlinearModelPredictiveControl,
    MultiCyclicCycleSolutions,
    OptimalControlProgram,
    InitialGuessList,
    ParameterList,
    VariableScaling,
    Solution,
    Solver,
)

from cocofest.models.ding2007.ding2007_with_fatigue import DingModelPulseWidthFrequencyWithFatigue


class FesNmpc(MultiCyclicNonlinearModelPredictiveControl):
    def __init__(self, **kwargs):
        super(FesNmpc, self).__init__(**kwargs)
        self.all_models = []
        self.cycle_duration = kwargs["cycle_duration"]
        self.n_cycles_simultaneous = kwargs["n_cycles_simultaneous"]

        self.first_run = True
        self.use_sx = kwargs["use_sx"]

    def advance_window_bounds_states(self, sol, n_cycles_simultaneous=None, **extra):
        super(FesNmpc, self).advance_window_bounds_states(sol)
        self.update_stim()
        return True

    def advance_window_initial_guess_states(self, sol, n_cycles_simultaneous=None):
        super(FesNmpc, self).advance_window_initial_guess_states(sol)
        return True

    def advance_window_bounds_controls(self, sol, n_cycles_simultaneous=None, **extra):
        bound_have_changed = super(FesNmpc, self).advance_window_bounds_controls(sol)
        return bound_have_changed

    @staticmethod
    def build_new_model(model, previous_stim):
        new_model = DingModelPulseWidthFrequencyWithFatigue(
            previous_stim=previous_stim, stim_time=model.stim_time, sum_stim_truncation=model.sum_stim_truncation
        )
        return new_model

    def update_stim(self):
        truncation_term = self.nlp[0].model.sum_stim_truncation
        solution_stimulation_time = self.nlp[0].model.stim_time[-truncation_term:]
        previous_stim_time = [x - self.phase_time[0] for x in solution_stimulation_time]
        previous_stim = {"time": previous_stim_time}
        new_model = self.build_new_model(model=self.nlp[0].model, previous_stim=previous_stim)
        if self.first_run:
            self.nlp[0].numerical_data_timeseries, _ = new_model.get_numerical_data_time_series(
                self.n_shooting, self.phase_time[0]
            )
            self.first_run = False

        self.nlp[0].model = new_model
        self.all_models.append(new_model)

    def _initialize_solution(self, dt: float, states: list, controls: list, parameters: list):
        combined_model = self.create_model_from_list(self.all_models)
        x_init = InitialGuessList()
        for key in self.nlp[0].states.keys():
            x_init.add(
                key,
                np.concatenate([state[key][:, :-1] for state in states] + [states[-1][key][:, -1:]], axis=1),
                interpolation=self.nlp[0].x_init.type,
                phase=0,
            )

        u_init = InitialGuessList()
        for key in self.nlp[0].controls.keys():
            controls_tp = np.concatenate([control[key] for control in controls], axis=1)
            u_init.add(key, controls_tp, interpolation=self.nlp[0].u_init.type, phase=0)

        p_init = InitialGuessList()
        parameters = ParameterList(use_sx=self.cx == SX)
        stimulation_per_cycle = int(len(self.nlp[0].model.stim_time) / self.n_cycles)
        for key in self.nlp[0].parameters.keys():
            combined_parameters = [list(parameter[key][:stimulation_per_cycle]) for parameter in parameters]
            combined_parameters = [val for sublist in combined_parameters for val in sublist]
            p_init[key] = combined_parameters

            parameters.add(
                name=key,
                function=self.nlp[0].parameters[key].function,
                size=len(combined_parameters),
                scaling=VariableScaling(key, [1] * len(combined_parameters)),
            )

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
        )
        a_init = InitialGuessList()
        return Solution.from_initial_guess(solution_ocp, [np.array([dt]), x_init, u_init, p_init, a_init])

    def create_model_from_list(self, models: list):
        stimulation_per_cycle = int(len(self.nlp[0].model.stim_time) / self.n_cycles)
        stim_time = []
        for i in range(len(models)):
            stim_time.append(list(np.array(models[0].stim_time[:stimulation_per_cycle]) + (i * self.cycle_duration)))
        stim_time = [val for sublist in stim_time for val in sublist]

        combined_model = DingModelPulseWidthFrequencyWithFatigue(
            stim_time=stim_time, sum_stim_truncation=self.nlp[0].model.sum_stim_truncation
        )
        return combined_model

    def solve_fes_nmpc(
        self,
        update_functions,
        solver: Solver.IPOPT,
        total_cycles: int,
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

        total_nmpc_duration = self.cycle_duration * total_cycles
        stim_time = self.nlp[0].model.stim_time
        step = stim_time[1] - stim_time[0]
        stim_interval = int(1 / step) + 1
        all_stim_time = [
            val
            for start in range(0, total_nmpc_duration, 2)
            for val in np.linspace(start, start + 1, stim_interval)[:-1]
        ]
        self.nlp[0].model.stim_time = all_stim_time
        total_nmpc_shooting_len = int(
            self.nlp[0].model.get_n_shooting(self.cycle_duration * self.n_cycles_simultaneous)
            / self.n_cycles_simultaneous
            * total_cycles
        )

        numerical_data_time_series, stim_idx_at_node_list = model.get_numerical_data_time_series(
            total_nmpc_shooting_len, total_nmpc_duration, all_stim_time
        )
        sol[0].ocp.nlp[0].numerical_data_timeseries = numerical_data_time_series

        return sol
