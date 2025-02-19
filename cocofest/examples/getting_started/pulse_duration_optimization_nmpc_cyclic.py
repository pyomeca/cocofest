"""
This example showcases a moving time horizon simulation problem of cyclic muscle force tracking at last node.
The FES model used here is Ding's 2007 pulse width and frequency model with fatigue.
Only the pulse width is optimized to minimize muscle force production, frequency is fixed at 33 Hz.
The nmpc cyclic problem stops once the last cycle is reached.
"""
import pickle
import numpy as np

from bioptim import (
    ConstraintList,
    CostType,
    ConstraintFcn,
    InitialGuessList,
    InterpolationType,
    MultiCyclicCycleSolutions,
    MultiCyclicNonlinearModelPredictiveControl,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    SolutionMerge,
    VariableScaling,
    Solution,
    Solver,
    ParameterList,
    OptimalControlProgram,
    ControlType
)

from casadi import SX

from cocofest import (
    DingModelPulseWidthFrequencyWithFatigue,
    OcpFes,
    FesModel,
    FES_plot,
    ModifiedOdeSolverRK4
)


class MyCyclicNMPC(MultiCyclicNonlinearModelPredictiveControl):
    def __init__(self, **kwargs):
        super(MyCyclicNMPC, self).__init__(**kwargs)
        self.all_models = []
        self.cycle_duration = kwargs["cycle_duration"]
        self.n_cycles_simultaneous = kwargs["n_cycles_simultaneous"]
        self.bimapped_param = True if self.parameters.shape == self.n_cycles_simultaneous else False
        self.initial_guess_param_index = []
        self.initial_guess_frames = []
        self.initialize_frames_and_parameters(kwargs["bio_model"].stim_time)

    def initialize_frames_and_parameters(self, stim_time):
        for i in range(2):
            self.initial_guess_frames.extend(
                list(range((self.n_cycles_to_advance + i) * self.cycle_len, self.n_cycles * self.cycle_len)))
        self.initial_guess_frames.append(self.n_cycles * self.cycle_len)

        for i in range(2):
            self.initial_guess_param_index.extend(
                list(range((int(len(stim_time) / self.n_cycles) * (1 + i)),
                           len(stim_time))))

    def advance_window_bounds_states(self, sol, n_cycles_simultaneous=None, **extra):
        super(MyCyclicNMPC, self).advance_window_bounds_states(sol)
        self.update_stim(sol)
        return True

    def advance_window_initial_guess_states(self, sol, n_cycles_simultaneous=None):
        super(MyCyclicNMPC, self).advance_window_initial_guess_states(sol)
        # self.ocp_solver.ocp.parameter_init["pulse_width"].init[:, :] = sol.parameters["pulse_width"][:, None]
        # TODO : Add parameters initial guess
        return True

    def advance_window_initial_guess_parameters(self, sol, **advance_options):
        parameters = sol.parameters
        for key in parameters.keys():
            if self.parameter_init[key].type != InterpolationType.EACH_FRAME:
                self.parameter_init.add(
                    key, np.ndarray(parameters[key].shape), interpolation=InterpolationType.CONSTANT, phase=0
                )
                self.parameter_init[key].check_and_adjust_dimensions(len(self.nlp[0].parameters[key]), self.nlp[0].ns)
            reshaped_parameter = parameters[key][:, None]
            if self.bimapped_param:
                self.parameter_init[key].init[:, :] = np.concatenate(
                    (reshaped_parameter[:, 1:], reshaped_parameter[:, -1][:, np.newaxis]), axis=1
                )
            else:
                self.parameter_init[key].init[:, :] = reshaped_parameter[self.initial_guess_param_index, :]

    def update_stim(self, sol):
        truncation_term = self.nlp[0].model._sum_stim_truncation
        solution_stimulation_time = self.nlp[0].model.stim_time[-truncation_term:]
        previous_stim_time = [x - self.phase_time[0] for x in solution_stimulation_time]
        stimulation_per_cycle = int(len(self.nlp[0].model.stim_time) / self.n_cycles)
        previous_pw_time = list([sol.parameters["pulse_width"][0]]) * len(previous_stim_time) if self.bimapped_param else list(sol.parameters["pulse_width"][:stimulation_per_cycle][-truncation_term:])
        previous_stim = {"time": previous_stim_time, "pulse_width": previous_pw_time}
        new_model = DingModelPulseWidthFrequencyWithFatigue(
            previous_stim=previous_stim,
            stim_time=self.nlp[0].model.stim_time,
            sum_stim_truncation=truncation_term
        )
        self.nlp[0].model = new_model
        self.all_models.append(new_model)

    def _initialize_solution(self, dt: float, states: list, controls: list, parameters: list):
        combined_model = self.create_model_from_list(self.all_models)
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
        stimulation_per_cycle = int(len(self.nlp[0].model.stim_time) / self.n_cycles)
        for key in self.nlp[0].parameters.keys():
            combined_parameters = [[parameters[i][key][0]] * stimulation_per_cycle for i in range(len(parameters))] if self.bimapped_param else [list(parameter[key][:stimulation_per_cycle]) for parameter in parameters]
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
            # parameter_bounds=self.parameter_bounds,
            ode_solver=self.nlp[0].ode_solver,
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
            stim_time=stim_time,
            sum_stim_truncation=self.nlp[0].model._sum_stim_truncation
        )
        return combined_model


def prepare_nmpc(
    model: FesModel,
    cycle_duration: int | float,
    n_cycles_to_advance: int,
    n_cycles_simultaneous: int,
    pulse_width: dict,
    use_sx: bool = False,
):
    total_cycle_len = OcpFes.prepare_n_shooting(model.stim_time, cycle_duration * n_cycles_simultaneous)
    total_cycle_duration = cycle_duration * n_cycles_simultaneous
    cycle_len = total_cycle_len / n_cycles_simultaneous

    numerical_data_time_series, stim_idx_at_node_list = OcpFes.set_all_stim_time_for_numerical_data_time_series(model,
                                                                                         total_cycle_len,
                                                                                         total_cycle_duration)

    dynamics = OcpFes._declare_dynamics(model, numerical_data_time_series)

    x_bounds, x_init = OcpFes._set_bounds(model)
    u_bounds, u_init = OcpFes._set_u_bounds(model)

    constraints = OcpFes._build_constraints(
        model,
        total_cycle_len,
        total_cycle_duration,
        stim_idx_at_node_list,
        ControlType.CONSTANT,
        pulse_width["bimapping"],
        n_cycles_simultaneous=n_cycles_simultaneous,
    )
    constraints_node_list = [33 * (1 + 2 * i) for i in range(int(cycle_len * n_cycles_simultaneous) // (2 * 33))]
    for i in constraints_node_list:
        constraints.add(ConstraintFcn.TRACK_STATE, key="F", node=i, min_bound=200, max_bound=210)

    # Define objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        key="F",
        weight=1,
        quadratic=True)

    # x_bounds, x_init = OcpFes._set_bounds(model)

    (parameters, parameters_bounds, parameters_init, parameter_objectives) = OcpFes._build_parameters(
        model=model,
        pulse_width=pulse_width,
        pulse_intensity=None,
        use_sx=use_sx,
        n_cycles_simultaneous=n_cycles_simultaneous,
    )
    OcpFes.update_model_param(model, parameters)

    return MyCyclicNMPC(
        bio_model=model,
        dynamics=dynamics,
        cycle_len=int(cycle_len),
        cycle_duration=cycle_duration,
        n_cycles_simultaneous=n_cycles_simultaneous,
        n_cycles_to_advance=n_cycles_to_advance,
        common_objective_functions=objective_functions,
        constraints=constraints,
        x_bounds=x_bounds,
        x_init=x_init,
        u_bounds=u_bounds,
        u_init=u_init,
        parameters=parameters,
        parameter_bounds=parameters_bounds,
        parameter_init=parameters_init,
        parameter_objectives=parameter_objectives,
        ode_solver=OdeSolver.RK2(n_integration_steps=10),
        # ode_solver=ModifiedOdeSolverRK4(n_integration_steps=10),
        n_threads=20,
        use_sx=use_sx,
    )


def main():
    """
    Main function to configure and solve the optimal control problem.
    """
    # --- Build nmpc cyclic --- #
    cycle_duration = 2
    n_cycles_simultaneous = 3
    n_cycles_to_advance = 1
    n_cycles = 70

    # --- Set stimulation time apparition --- #
    final_time = 2 * n_cycles_simultaneous
    stim_time = [val for start in range(0, final_time, 2) for val in np.linspace(start, start + 1, 34)[:-1]]
    # --- Build FES model --- #
    fes_model = DingModelPulseWidthFrequencyWithFatigue(stim_time=stim_time, sum_stim_truncation=10)
    minimum_pulse_width = fes_model.pd0

    nmpc = prepare_nmpc(
        model=fes_model,
        cycle_duration=cycle_duration,
        n_cycles_to_advance=n_cycles_to_advance,
        n_cycles_simultaneous=n_cycles_simultaneous,
        # t_total_n_cycles=n_cycles,
        # all_stim_time=all_stim_time,
        pulse_width={"min": minimum_pulse_width,
                     "max": 0.0006,
                     "bimapping": False,
                     "fixed": False},
        use_sx=False,
    )

    def update_functions(_nmpc: MultiCyclicNonlinearModelPredictiveControl, cycle_idx: int, _sol: Solution):
        return cycle_idx < n_cycles  # True if there are still some cycle to perform

    # Add the penalty cost function plot
    nmpc.add_plot_penalty(CostType.ALL)
    # Solve the optimal control problem
    sol = nmpc.solve(
        update_functions,
        solver=Solver.IPOPT(show_online_optim=False, _max_iter=1000, show_options=dict(show_bounds=True)),
        cycle_solutions=MultiCyclicCycleSolutions.ALL_CYCLES,
        get_all_iterations=True,
        cyclic_options={"states": {}},
        n_cycles_simultaneous=n_cycles_simultaneous,
        max_consecutive_failing=3,
    )

    total_nmpc_duration = cycle_duration * n_cycles
    all_stim_time = [val for start in range(0, total_nmpc_duration, 2) for val in np.linspace(start, start + 1, 34)[:-1]]
    total_nmpc_shooting_len = int(OcpFes.prepare_n_shooting(fes_model.stim_time,
                                                        cycle_duration * n_cycles_simultaneous) / n_cycles_simultaneous * n_cycles)

    numerical_data_time_series, stim_idx_at_node_list = OcpFes.set_all_stim_time_for_numerical_data_time_series(fes_model,
                                                                                                                total_nmpc_shooting_len,
                                                                                                                total_nmpc_duration,
                                                                                                                all_stim_time)
    sol[0].ocp.nlp[0].numerical_data_timeseries = numerical_data_time_series

    result = sol[0].stepwise_states(to_merge=[SolutionMerge.NODES])
    time = sol[0].stepwise_time(to_merge=[SolutionMerge.NODES]).T[0]
    result["time"] = time
    result["pulse_width"] = sol[0].parameters["pulse_width"]
    with open("results_pulse_duration_optimization_nmpc_cyclic_2.pkl", "wb") as file:
        pickle.dump(result, file)

    # Plotting the force state result
    FES_plot(data=result).plot(title="NMPC FES model optimization")
    for i in range(sol[0].parameters["pulse_width"].shape[0]):
        print(sol[0].parameters["pulse_width"][i])


if __name__ == "__main__":
    main()
