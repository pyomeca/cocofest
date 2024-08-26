import math

import numpy as np
from bioptim import SolutionMerge, ObjectiveList, ObjectiveFcn, OdeSolver, Node, OptimalControlProgram, ControlType, TimeAlignment
from cocofest import DingModelPulseDurationFrequencyWithFatigue, OcpFes, CustomObjective, FesModel
# TODO relative import


class OcpFesMhe:
    def __init__(self,
                 model: FesModel = None,
                 n_stim: int = None,
                 n_shooting: int = None,
                 final_time: int | float = None,
                 pulse_event: dict = None,
                 pulse_duration: dict = None,
                 pulse_intensity: dict = None,
                 n_total_cycles: int = None,
                 n_simultaneous_cycles: int = None,
                 n_cycle_to_advance: int = None,
                 cycle_to_keep: str = None,
                 objective: dict = None,
                 use_sx: bool = True,
                 ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
                 n_threads: int = 1,
    ):

        super(OcpFesMhe, self).__init__()
        self.model = model
        self.n_stim = n_stim
        self.n_shooting = n_shooting
        self.final_time = final_time
        self.pulse_event = pulse_event
        self.pulse_duration = pulse_duration
        self.pulse_intensity = pulse_intensity
        self.objective = objective
        self.n_total_cycles = n_total_cycles
        self.n_simultaneous_cycles = n_simultaneous_cycles
        self.n_cycle_to_advance = n_cycle_to_advance
        self.cycle_to_keep = cycle_to_keep
        self.use_sx = use_sx
        self.ode_solver = ode_solver
        self.n_threads = n_threads
        self.ocp = None
        self._mhe_sanity_check()
        # self.time = []
        self.states = []
        self.parameters = []
        self.previous_stim = []
        # self.result_states = {}
        # self.result_parameters = {}
        # self.result_time = {}
        self.result = {"time": {}, "states": {}, "parameters": {}}

    def prepare_mhe(self):
        (pulse_event, pulse_duration, pulse_intensity, objective) = OcpFes._fill_dict(
            self.pulse_event, self.pulse_duration, self.pulse_intensity, self.objective
        )

        time_min = pulse_event["min"]
        time_max = pulse_event["max"]
        time_bimapping = pulse_event["bimapping"]
        frequency = pulse_event["frequency"]
        round_down = pulse_event["round_down"]
        pulse_mode = pulse_event["pulse_mode"]

        fixed_pulse_duration = pulse_duration["fixed"]
        pulse_duration_min = pulse_duration["min"]
        pulse_duration_max = pulse_duration["max"]
        pulse_duration_bimapping = pulse_duration["bimapping"]

        fixed_pulse_intensity = pulse_intensity["fixed"]
        pulse_intensity_min = pulse_intensity["min"]
        pulse_intensity_max = pulse_intensity["max"]
        pulse_intensity_bimapping = pulse_intensity["bimapping"]

        force_tracking = objective["force_tracking"]
        end_node_tracking = objective["end_node_tracking"]
        custom_objective = objective["custom"]

        OcpFes._sanity_check(
            model=self.model,
            n_stim=self.n_stim,
            n_shooting=self.n_shooting,
            final_time=self.final_time,
            pulse_mode=pulse_mode,
            frequency=frequency,
            time_min=time_min,
            time_max=time_max,
            time_bimapping=time_bimapping,
            fixed_pulse_duration=fixed_pulse_duration,
            pulse_duration_min=pulse_duration_min,
            pulse_duration_max=pulse_duration_max,
            pulse_duration_bimapping=pulse_duration_bimapping,
            fixed_pulse_intensity=fixed_pulse_intensity,
            pulse_intensity_min=pulse_intensity_min,
            pulse_intensity_max=pulse_intensity_max,
            pulse_intensity_bimapping=pulse_intensity_bimapping,
            force_tracking=force_tracking,
            end_node_tracking=end_node_tracking,
            custom_objective=custom_objective,
            use_sx=self.use_sx,
            ode_solver=self.ode_solver,
            n_threads=self.n_threads,
        )

        OcpFes._sanity_check_frequency(n_stim=self.n_stim, final_time=self.final_time, frequency=frequency, round_down=round_down)

        n_stim, final_time = OcpFes._build_phase_parameter(
            n_stim=self.n_stim * self.n_simultaneous_cycles, final_time=self.final_time * self.n_simultaneous_cycles, frequency=frequency, pulse_mode=pulse_mode, round_down=round_down
        )

        force_fourier_coefficient = (
            None if force_tracking is None else OcpFes._build_fourier_coefficient(force_tracking)
        )

        models = [self.model] * self.n_stim * self.n_simultaneous_cycles

        final_time_phase = OcpFes._build_phase_time(
            final_time=self.final_time * self.n_simultaneous_cycles,
            n_stim=self.n_stim * self.n_simultaneous_cycles,
            pulse_mode=pulse_mode,
            time_min=time_min,
            time_max=time_max,
        )
        parameters, parameters_bounds, parameters_init, parameter_objectives, constraints = OcpFes._build_parameters(
            model=self.model,
            n_stim=self.n_stim * self.n_simultaneous_cycles,
            time_min=time_min,
            time_max=time_max,
            time_bimapping=time_bimapping,
            fixed_pulse_duration=fixed_pulse_duration,
            pulse_duration_min=pulse_duration_min,
            pulse_duration_max=pulse_duration_max,
            pulse_duration_bimapping=pulse_duration_bimapping,
            fixed_pulse_intensity=fixed_pulse_intensity,
            pulse_intensity_min=pulse_intensity_min,
            pulse_intensity_max=pulse_intensity_max,
            pulse_intensity_bimapping=pulse_intensity_bimapping,
            use_sx=self.use_sx,
        )

        if len(constraints) == 0 and len(parameters) == 0:
            raise ValueError(
                "This is not an optimal control problem,"
                " add parameter to optimize or use the IvpFes method to build your problem"
            )

        dynamics = OcpFes._declare_dynamics(models, self.n_stim * self.n_simultaneous_cycles)
        x_bounds, x_init = OcpFes._set_bounds(self.model, self.n_stim * self.n_simultaneous_cycles)
        one_cycle_shooting = [self.n_shooting] * self.n_stim
        objective_functions = self._set_objective(
            self.n_stim, one_cycle_shooting, force_fourier_coefficient, end_node_tracking, custom_objective, time_min, time_max, self.n_simultaneous_cycles
        )
        all_cycle_n_shooting = [self.n_shooting] * self.n_stim * self.n_simultaneous_cycles
        self.ocp = OptimalControlProgram(
            bio_model=models,
            dynamics=dynamics,
            n_shooting=all_cycle_n_shooting,
            phase_time=final_time_phase,
            objective_functions=objective_functions,
            x_init=x_init,
            x_bounds=x_bounds,
            constraints=constraints,
            parameters=parameters,
            parameter_bounds=parameters_bounds,
            parameter_init=parameters_init,
            parameter_objectives=parameter_objectives,
            control_type=ControlType.CONSTANT,
            use_sx=self.use_sx,
            ode_solver=self.ode_solver,
            n_threads=self.n_threads,
        )

        return self.ocp

    # def update_mhe(self, previous_sol):
    #     sol_states = sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])
    #     cn_results.append(list(sol_states["Cn"][0]))
    #     f_results.append(list(sol_states["F"][0]))
    #     a_results.append(list(sol_states["A"][0]))
    #     tau1_results.append(list(sol_states["Tau1"][0]))
    #     km_results.append(list(sol_states["Km"][0]))
    #     sol_time = sol.decision_time(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])
    #     sol_time = list(sol_time.reshape(sol_time.shape[0]))
    #
    #     sol_time_stim_parameters = sol.decision_parameters()["pulse_apparition_time"]
    #
    #     if previous_stim:
    #         # stim_prev = list(sol_time_stim_parameters - sol_time[-1])
    #         update_previous_stim = list(np.array(previous_stim) - sol_time[-1])
    #         previous_stim = update_previous_stim + stim_prev
    #     else:
    #         stim_prev = list(sol_time_stim_parameters - sol_time[-1])
    #         previous_stim = stim_prev
    #
    #     if i != 0:
    #         sol_time = [x + time[-1][-1] for x in sol_time]
    #
    #     time.append(sol_time)
    #     keys = list(sol_states.keys())
    #
    #     for key in keys:
    #         ocp.nlp[0].x_bounds[key].max[0][0] = sol_states[key][-1][-1]
    #         ocp.nlp[0].x_bounds[key].min[0][0] = sol_states[key][-1][-1]
    #     for j in range(len(ocp.nlp)):
    #         ocp.nlp[j].model = DingModelPulseDurationFrequencyWithFatigue(sum_stim_truncation=10,
    #                                                                       stim_prev=previous_stim)
    #         for key in keys:
    #             ocp.nlp[j].x_init[key].init[0][0] = sol_states[key][-1][-1]

    def update_time(self, sol, index):
        if index == 0:
            sol_time = sol.decision_time(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])
            sol_time = list(sol_time.reshape(sol_time.shape[0]))
            sol_time = [t - sol_time[0] for t in sol_time]



            # self.time.append(sol_time)
        else:
            sol_time = sol.decision_time(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])
            sol_time = list(sol_time.reshape(sol_time.shape[0]))


        sol_time = sol.decision_time(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])
        sol_time = list(sol_time.reshape(sol_time.shape[0]))
        final_time = sol_time[-1] + self.time[-1][-1]

        if index != 0:
            sol_time = [x + self.time[-1][-1] for x in sol_time]
        self.time.append(sol_time)

    def update_states_bounds(self, sol_states, state_keys):
        for key in state_keys:
            self.ocp.nlp[0].x_bounds[key].max[0][0] = sol_states[self.n_stim//self.n_simultaneous_cycles - 1][key][0][-1]
            self.ocp.nlp[0].x_bounds[key].min[0][0] = sol_states[self.n_stim//self.n_simultaneous_cycles - 1][key][0][-1]
            for j in range(self.n_stim//self.n_simultaneous_cycles - 1, len(self.ocp.nlp)):
                self.ocp.nlp[j].x_init[key].init[0][0] = sol_states[j][key][0][0]

    def update_parameters(self, sol, parameters_keys):
        sol_parameters = sol.decision_parameters()
        sol_parameters_dict = {key: list(sol_parameters[key][0]) for key in sol_parameters.keys()}
        return

    def update_stim(self, sol):
        previous_time = self.time[0]
        if "pulse_apparition_time" in sol.decision_parameters():
            sol_time_stim_parameters = sol.decision_parameters()["pulse_apparition_time"]

            if self.previous_stim:
                stim_prev = list(sol_time_stim_parameters - self.time[-1])
                update_previous_stim = list(np.array(self.previous_stim) - self.time[-1])
                self.previous_stim = update_previous_stim + stim_prev
                # TODO: check if correct did some modification since last time
                # TODO: wrong, will take the last previous_stim from n_simultaneous_cycles at each iteration
                # TODO: wrong, time is pushed to many times
            else:
                stim_prev = list(sol_time_stim_parameters - self.time[-1])
                self.previous_stim = stim_prev

            list(self.previous_stim)

    def store_results(self, sol_time, sol_states, sol_parameters, index, merge=False):
        if self.cycle_to_keep == "middle":
            # Get the middle phase index to keep
            phase_to_keep = int(math.ceil(self.n_simultaneous_cycles / 2))
            first_node_in_phase = self.n_stim * (phase_to_keep - 1)
            last_node_in_phase = self.n_stim * phase_to_keep

            # Initialize the dict if it's the first iteration
            if index == 0:
                self.result["time"] = [None]*self.n_total_cycles
                [self.result["states"].update({state_key: [None]*self.n_total_cycles}) for state_key in list(sol_states[0].keys())]
                [self.result["parameters"].update({key_parameter: [None]*self.n_total_cycles}) for key_parameter in list(sol_parameters.keys())]

            # Store the results
            phase_size = np.array(sol_time).shape[0]
            node_size = np.array(sol_time).shape[1]
            sol_time = list(np.array(sol_time).reshape(phase_size*node_size))[first_node_in_phase*node_size:last_node_in_phase*node_size]
            if index == 0:
                sol_time = [t - sol_time[0] for t in sol_time]
            else:
                sol_time = [t - sol_time[0] + self.result["time"][index-1][-1] for t in sol_time]
                # time_diff = sol_time[first_node_in_phase]
                # sol_time[first_node_in_phase:last_node_in_phase] = sol_time[first_node_in_phase:last_node_in_phase] - self.time[-1][-1]
            self.result["time"][index] = sol_time  # Todo remove last node

            for state_key in list(sol_states[0].keys()):
                middle_states_values = sol_states[first_node_in_phase:last_node_in_phase]
                middle_states_values = [list(middle_states_values[i][state_key][0]) for i in range(len(middle_states_values))]
                middle_states_values = [j for sub in middle_states_values for j in sub]
                self.result["states"][state_key][index] = middle_states_values# Todo might be wrong

            for key_parameter in list(sol_parameters.keys()):
                self.result["parameters"][key_parameter][index] = sol_parameters[key_parameter][first_node_in_phase:last_node_in_phase]
        return

    def solve(self):
        state_keys = list(self.ocp.nlp[0].states.keys())
        parameters_keys = list(self.ocp.nlp[0].parameters.keys())
        for i in range(self.n_total_cycles // self.n_cycle_to_advance):
            sol = self.ocp.solve()

            sol_states = sol.decision_states(to_merge=[SolutionMerge.NODES])
            self.update_states_bounds(sol_states, state_keys)

            sol_time = sol.decision_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.STATES)
            # sol_time = self.update_time(sol, index=i)

            sol_parameters = sol.decision_parameters()
            # self.update_parameters(sol, parameters_keys)
            # self.update_stim(sol)

            # sol.graphs()

            self.store_results(sol_time, sol_states, sol_parameters, i)
        # return self.ocp.solve()

    @staticmethod
    def _set_objective(n_stim, n_shooting, force_fourier_coefficient, end_node_tracking, custom_objective, time_min, time_max,
                       n_simultaneous_cycles):
        # Creates the objective for our problem
        objective_functions = ObjectiveList()
        if custom_objective:
            if len(custom_objective) != n_stim:
                raise ValueError("The number of custom objective must be equal to the stimulation number of a single cycle")
            for i in range(len(custom_objective)):
                for j in range(n_simultaneous_cycles):
                    objective_functions.add(custom_objective[i + j * n_stim][0])

        if force_fourier_coefficient is not None:
            for phase in range(n_stim):
                for i in range(n_shooting[phase]):
                    for j in range(n_simultaneous_cycles):
                        objective_functions.add(
                            CustomObjective.track_state_from_time,
                            custom_type=ObjectiveFcn.Mayer,
                            node=i,
                            fourier_coeff=force_fourier_coefficient,
                            key="F",
                            quadratic=True,
                            weight=1,
                            phase=phase + j * n_stim,
                        )

        if end_node_tracking:
            if isinstance(end_node_tracking, int | float):
                for i in range(n_simultaneous_cycles):
                    objective_functions.add(
                        ObjectiveFcn.Mayer.MINIMIZE_STATE,
                        node=Node.END,
                        key="F",
                        quadratic=True,
                        weight=1,
                        target=end_node_tracking,
                        phase=n_stim - 1 + i * n_stim,
                    )

        if time_min and time_max:
            for i in range(n_stim):
                for j in range(n_simultaneous_cycles):
                    objective_functions.add(
                        ObjectiveFcn.Mayer.MINIMIZE_TIME,
                        weight=0.001 / n_shooting[i],
                        min_bound=time_min,
                        max_bound=time_max,
                        quadratic=True,
                        phase=i + j * n_stim,
                    )

        return objective_functions

    def _mhe_sanity_check(self):
        if self.n_total_cycles is None:
            raise ValueError("n_total_cycles must be set")
        if self.n_simultaneous_cycles is None:
            raise ValueError("n_simultaneous_cycles must be set")
        if self.n_cycle_to_advance is None:
            raise ValueError("n_cycle_to_advance must be set")
        if self.cycle_to_keep is None:
            raise ValueError("cycle_to_keep must be set")

        if self.n_total_cycles % self.n_cycle_to_advance != 0:
            raise ValueError("The number of n_total_cycles must be a multiple of the number n_cycle_to_advance")

        if self.n_cycle_to_advance > self.n_simultaneous_cycles:
            raise ValueError("The number of n_simultaneous_cycles must be higher than the number of n_cycle_to_advance")

        if self.cycle_to_keep not in ["first", "middle", "last"]:
            raise ValueError("cycle_to_keep must be either 'first', 'middle' or 'last'")

        # if self.cycle_to_keep == "middle" and self.n_simultaneous_cycles % 2 == 0:
        #     raise ValueError("The number of n_total_cycles must be an odd number if cycle_to_keep is 'middle'")