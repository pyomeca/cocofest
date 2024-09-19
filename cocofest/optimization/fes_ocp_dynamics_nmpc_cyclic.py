import numpy as np
from casadi import SX

from bioptim import (
    OdeSolver,
    CyclicNonlinearModelPredictiveControl,
    ControlType,
    Solution,
)

from .fes_ocp_dynamics import OcpFesMsk
from ..models.dynamical_model import FesMskModel


class NmpcFesMsk:
    def __init__(self):
        self.n_cycles = 1

    def advance_window_bounds_states(self, sol, **extra):
        # Reimplementation of the advance_window method so the rotation of the wheel restart at -pi
        CyclicNonlinearModelPredictiveControl.advance_window_bounds_states(sol, **extra)
        # TODO

        return True

    def update_stim(self, sol):

        stimulation_time = sol.decision_parameters()["pulse_apparition_time"]
        stim_prev = np.array(stimulation_time) - sol.ocp.phase_time[0]
        current_stim = np.array(
            sol.ocp.parameter_bounds["pulse_apparition_time"].min
        ).reshape(sol.ocp.parameter_bounds["pulse_apparition_time"].min.shape[0])
        updated_stim = np.append(stim_prev[:-1], current_stim)

        previous_pulse_duration = list(sol.parameters["pulse_duration"][:-1])
        pulse_duration_bound_min = previous_pulse_duration + [
            sol.ocp.parameter_bounds["pulse_duration"].min[0][0]
        ] * len(sol.decision_parameters()["pulse_apparition_time"])
        pulse_duration_bound_max = previous_pulse_duration + [
            sol.ocp.parameter_bounds["pulse_duration"].min[0][0]
        ] * len(sol.decision_parameters()["pulse_apparition_time"])

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
    def prepare_nmpc(
        model: FesMskModel = None,
        stim_time: list = None,
        cycle_len: int = None,
        cycle_duration: int | float = None,
        pulse_event: dict = None,
        pulse_duration: dict = None,
        pulse_intensity: dict = None,
        objective: dict = None,
        msk_info: dict = None,
        warm_start: bool = False,
        use_sx: bool = True,
        ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
        n_threads: int = 1,
    ):

        input_dict = {
            "model": model,
            "stim_time": stim_time,
            "n_shooting": cycle_len,
            "final_time": cycle_duration,
            "pulse_event": pulse_event,
            "pulse_duration": pulse_duration,
            "pulse_intensity": pulse_intensity,
            "objective": objective,
            "msk_info": msk_info,
            "warm_start": warm_start,
            "use_sx": use_sx,
            "ode_solver": ode_solver,
            "n_threads": n_threads,
        }

        optimization_dict = OcpFesMsk._prepare_optimization_problem(input_dict)

        return CyclicNonlinearModelPredictiveControl(
            bio_model=[optimization_dict["model"]],
            dynamics=optimization_dict["dynamics"],
            cycle_len=cycle_len,
            cycle_duration=cycle_duration,
            common_objective_functions=optimization_dict["objective_functions"],
            x_init=optimization_dict["x_init"],
            x_bounds=optimization_dict["x_bounds"],
            constraints=optimization_dict["constraints"],
            parameters=optimization_dict["parameters"],
            parameter_bounds=optimization_dict["parameters_bounds"],
            parameter_init=optimization_dict["parameters_init"],
            parameter_objectives=optimization_dict["parameter_objectives"],
            control_type=ControlType.CONSTANT,
            use_sx=optimization_dict["use_sx"],
            ode_solver=optimization_dict["ode_solver"],
            n_threads=optimization_dict["n_threads"],
        )

    # @staticmethod
    def update_functions(
        self,
        _nmpc: CyclicNonlinearModelPredictiveControl,
        cycle_idx: int,
        _sol: Solution,
    ):
        return (
            cycle_idx < self.n_cycles
        )  # True if there are still some cycle to perform
