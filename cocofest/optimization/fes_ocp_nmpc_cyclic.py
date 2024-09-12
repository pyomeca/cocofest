import numpy as np
from casadi import SX

from bioptim import (
    OdeSolver,
    CyclicNonlinearModelPredictiveControl,
    ControlType,
    Solution,
    BoundsList,
    InitialGuessList,
    ParameterList,
    InterpolationType,
    VariableScaling,
)

from .fes_ocp import OcpFes
from ..models.fes_model import FesModel
from ..models.ding2003 import DingModelFrequency
from ..models.ding2007 import DingModelPulseDurationFrequency


class NmpcFes(CyclicNonlinearModelPredictiveControl):
    def advance_window_bounds_states(self, sol, **extra):
        # Reimplementation of the advance_window method so the rotation of the wheel restart at -pi
        super(NmpcFes, self).advance_window_bounds_states(sol, **extra)

        # self.nlp[0].x_bounds["Cn"][0, 0] = 0
        # self.nlp[0].x_bounds["F"][0, 0] = 0

        # self.nlp[0].parameters, self.nlp[0].parameter_bounds, self.nlp[0].parameter_init = self.update_stim(sol)
        # TODO

        return True

    def update_stim(self, sol):

        stimulation_time = sol.decision_parameters()["pulse_apparition_time"]
        stim_prev = np.array(stimulation_time) - sol.ocp.phase_time[0]
        current_stim = np.array(sol.ocp.parameter_bounds["pulse_apparition_time"].min).reshape(sol.ocp.parameter_bounds["pulse_apparition_time"].min.shape[0])
        updated_stim = np.append(stim_prev[:-1], current_stim)

        previous_pulse_duration = list(sol.parameters["pulse_duration"][:-1])
        pulse_duration_bound_min = previous_pulse_duration + [sol.ocp.parameter_bounds["pulse_duration"].min[0][0]] * len(sol.decision_parameters()["pulse_apparition_time"])
        pulse_duration_bound_max = previous_pulse_duration + [sol.ocp.parameter_bounds["pulse_duration"].min[0][0]] * len(sol.decision_parameters()["pulse_apparition_time"])

        parameters, parameters_bounds, parameters_init = self._build_parameters(
            model=self.nlp[0].model,
            n_stim=len(updated_stim),
            use_sx=self.cx == SX,
            stim_time=updated_stim,
            pulse_duration_min=pulse_duration_bound_min,
            pulse_duration_max=pulse_duration_bound_max,
        )

        return parameters, parameters_bounds, parameters_init

    @staticmethod
    def _build_parameters(
            model,
            n_stim,
            time_min=None,
            time_max=None,
            time_bimapping=None,
            fixed_pulse_duration=None,
            pulse_duration_min=None,
            pulse_duration_max=None,
            pulse_duration_bimapping=None,
            fixed_pulse_intensity=None,
            pulse_intensity_min=None,
            pulse_intensity_max=None,
            pulse_intensity_bimapping=None,
            use_sx=None,
            stim_time=None,
    ):
        parameters = ParameterList(use_sx=use_sx)
        parameters_bounds = BoundsList()
        parameters_init = InitialGuessList()

        parameters.add(
            name="pulse_apparition_time",
            function=DingModelFrequency.set_pulse_apparition_time,
            size=n_stim,
            scaling=VariableScaling("pulse_apparition_time", [1] * n_stim),
        )

        parameters_init["pulse_apparition_time"] = np.array(stim_time)

        parameters_bounds.add(
            "pulse_apparition_time",
            min_bound=stim_time,
            max_bound=stim_time,
            interpolation=InterpolationType.CONSTANT,
        )

        if isinstance(model, DingModelPulseDurationFrequency):
            if pulse_duration_min and pulse_duration_max:
                parameters_bounds.add(
                    "pulse_duration",
                    min_bound=pulse_duration_min,
                    max_bound=pulse_duration_max,
                    interpolation=InterpolationType.CONSTANT,
                )
                pulse_duration_init = [(pulse_duration_min[i] + pulse_duration_max[i]) / 2 for i in range(len(pulse_duration_max))]
                parameters_init["pulse_duration"] = np.array(pulse_duration_init)
                parameters.add(
                    name="pulse_duration",
                    function=DingModelPulseDurationFrequency.set_impulse_duration,
                    size=n_stim,
                    scaling=VariableScaling("pulse_duration", [1] * n_stim),
                )


        #     if fixed_pulse_duration:
        #         parameters.add(
        #             name="pulse_duration",
        #             function=DingModelPulseDurationFrequency.set_impulse_duration,
        #             size=n_stim,
        #             scaling=VariableScaling("pulse_duration", [1] * n_stim),
        #         )
        #         if isinstance(fixed_pulse_duration, list):
        #             parameters_bounds.add(
        #                 "pulse_duration",
        #                 min_bound=np.array(fixed_pulse_duration),
        #                 max_bound=np.array(fixed_pulse_duration),
        #                 interpolation=InterpolationType.CONSTANT,
        #             )
        #             parameters_init.add(key="pulse_duration", initial_guess=np.array(fixed_pulse_duration))
        #         else:
        #             parameters_bounds.add(
        #                 "pulse_duration",
        #                 min_bound=np.array([fixed_pulse_duration] * n_stim),
        #                 max_bound=np.array([fixed_pulse_duration] * n_stim),
        #                 interpolation=InterpolationType.CONSTANT,
        #             )
        #             parameters_init["pulse_duration"] = np.array([fixed_pulse_duration] * n_stim)
        #
        #     elif pulse_duration_min is not None and pulse_duration_max is not None:
        #         parameters_bounds.add(
        #             "pulse_duration",
        #             min_bound=[pulse_duration_min],
        #             max_bound=[pulse_duration_max],
        #             interpolation=InterpolationType.CONSTANT,
        #         )
        #         parameters_init["pulse_duration"] = np.array([0] * n_stim)
        #         parameters.add(
        #             name="pulse_duration",
        #             function=DingModelPulseDurationFrequency.set_impulse_duration,
        #             size=n_stim,
        #             scaling=VariableScaling("pulse_duration", [1] * n_stim),
        #         )
        #
        #     if pulse_duration_bimapping is True:
        #         for i in range(1, n_stim):
        #             constraints.add(CustomConstraint.equal_to_first_pulse_duration, node=Node.START, target=0,
        #                             phase=i)
        #
        # if isinstance(model, DingModelIntensityFrequency):
        #     if fixed_pulse_intensity:
        #         parameters.add(
        #             name="pulse_intensity",
        #             function=DingModelIntensityFrequency.set_impulse_intensity,
        #             size=n_stim,
        #             scaling=VariableScaling("pulse_intensity", [1] * n_stim),
        #         )
        #         if isinstance(fixed_pulse_intensity, list):
        #             parameters_bounds.add(
        #                 "pulse_intensity",
        #                 min_bound=np.array(fixed_pulse_intensity),
        #                 max_bound=np.array(fixed_pulse_intensity),
        #                 interpolation=InterpolationType.CONSTANT,
        #             )
        #             parameters_init.add(key="pulse_intensity", initial_guess=np.array(fixed_pulse_intensity))
        #         else:
        #             parameters_bounds.add(
        #                 "pulse_intensity",
        #                 min_bound=np.array([fixed_pulse_intensity] * n_stim),
        #                 max_bound=np.array([fixed_pulse_intensity] * n_stim),
        #                 interpolation=InterpolationType.CONSTANT,
        #             )
        #             parameters_init["pulse_intensity"] = np.array([fixed_pulse_intensity] * n_stim)
        #
        #     elif pulse_intensity_min is not None and pulse_intensity_max is not None:
        #         parameters_bounds.add(
        #             "pulse_intensity",
        #             min_bound=[pulse_intensity_min],
        #             max_bound=[pulse_intensity_max],
        #             interpolation=InterpolationType.CONSTANT,
        #         )
        #         intensity_avg = (pulse_intensity_min + pulse_intensity_max) / 2
        #         parameters_init["pulse_intensity"] = np.array([intensity_avg] * n_stim)
        #         parameters.add(
        #             name="pulse_intensity",
        #             function=DingModelIntensityFrequency.set_impulse_intensity,
        #             size=n_stim,
        #             scaling=VariableScaling("pulse_intensity", [1] * n_stim),
        #         )
        #
        #     if pulse_intensity_bimapping is True:
        #         for i in range(1, n_stim):
        #             constraints.add(CustomConstraint.equal_to_first_pulse_intensity, node=Node.START, target=0,
        #                             phase=i)

        return parameters, parameters_bounds, parameters_init


    @staticmethod
    def prepare_nmpc(
        model: FesModel = None,
        n_stim: int = None,
        n_shooting: int = None,
        final_time: int | float = None,
        pulse_event: dict = None,
        pulse_duration: dict = None,
        pulse_intensity: dict = None,
        objective: dict = None,
        use_sx: bool = True,
        ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
        n_threads: int = 1,
        stim_time: list = None,
        cycle_len: int = None,
        cycle_duration: int | float = None,
    ):
        (pulse_event, pulse_duration, pulse_intensity, objective) = OcpFes._fill_dict(
            pulse_event, pulse_duration, pulse_intensity, objective
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
            model=model,
            n_stim=n_stim,
            n_shooting=n_shooting,
            final_time=final_time,
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
            use_sx=use_sx,
            ode_solver=ode_solver,
            n_threads=n_threads,
        )

        OcpFes._sanity_check_frequency(n_stim=n_stim, final_time=final_time, frequency=frequency, round_down=round_down)

        force_fourier_coefficient = (
            None if force_tracking is None else OcpFes._build_fourier_coefficient(force_tracking)
        )

        parameters, parameters_bounds, parameters_init, parameter_objectives, constraints = OcpFes._build_parameters(
            model=model,
            n_stim=n_stim,
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
            use_sx=use_sx,
            stim_time=stim_time,
        )

        if len(constraints) == 0 and len(parameters) == 0:
            raise ValueError(
                "This is not an optimal control problem,"
                " add parameter to optimize or use the IvpFes method to build your problem"
            )

        dynamics = OcpFes._declare_dynamics(model)
        x_bounds, x_init = OcpFes._set_bounds(model)
        objective_functions = OcpFes._set_objective(
            n_stim, n_shooting, force_fourier_coefficient, end_node_tracking, custom_objective, time_min, time_max
        )

        return NmpcFes(
            bio_model=[model],
            dynamics=dynamics,
            cycle_len=cycle_len,
            cycle_duration=cycle_duration,
            common_objective_functions=objective_functions,
            x_init=x_init,
            x_bounds=x_bounds,
            constraints=constraints,
            parameters=parameters,
            parameter_bounds=parameters_bounds,
            parameter_init=parameters_init,
            parameter_objectives=parameter_objectives,
            control_type=ControlType.CONSTANT,
            use_sx=use_sx,
            ode_solver=ode_solver,
            n_threads=n_threads,
        )

    @staticmethod
    def update_functions(_nmpc: CyclicNonlinearModelPredictiveControl, cycle_idx: int, _sol: Solution):
        return cycle_idx < 2  # True if there are still some cycle to perform
