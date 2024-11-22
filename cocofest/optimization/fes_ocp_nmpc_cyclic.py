import numpy as np
from casadi import SX

from bioptim import (
    OdeSolver,
    CyclicNonlinearModelPredictiveControl,
    ControlType,
    Solution,
)

from .fes_ocp import OcpFes
from ..models.fes_model import FesModel


class NmpcFes(CyclicNonlinearModelPredictiveControl):
    def advance_window_bounds_states(self, sol, **extra):
        super(NmpcFes, self).advance_window_bounds_states(sol)
        self.update_stim(sol)

    def update_stim(self, sol):
        stimulation_time = sol.decision_parameters()["pulse_apparition_time"]
        stim_prev = list(np.round(np.array(stimulation_time) - sol.ocp.phase_time[0], 3))

        self.nlp[0].model.stim_prev = stim_prev
        if "pulse_intensity" in sol.parameters.keys():
            self.nlp[0].model.stim_pulse_intensity_prev = list(sol.parameters["pulse_intensity"])

    @staticmethod
    def prepare_nmpc(
        model: FesModel = None,
        stim_time: list = None,
        cycle_duration: int | float = None,
        n_cycles_simultaneous: int = None,
        n_cycles_to_advance: int = None,
        pulse_event: dict = None,
        pulse_width: dict = None,
        pulse_intensity: dict = None,
        objective: dict = None,
        use_sx: bool = True,
        ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
        n_threads: int = 1,
        control_type: ControlType = ControlType.CONSTANT,
    ):

        input_dict = {
            "model": model,
            "stim_time": stim_time,
            "n_shooting": OcpFes.prepare_n_shooting(stim_time, cycle_duration),
            "final_time": cycle_duration,
            "n_cycles_simultaneous": n_cycles_simultaneous,
            "n_cycles_to_advance": n_cycles_to_advance,
            "pulse_event": pulse_event,
            "pulse_width": pulse_width,
            "pulse_intensity": pulse_intensity,
            "objective": objective,
            "use_sx": use_sx,
            "ode_solver": ode_solver,
            "n_threads": n_threads,
            "control_type": control_type,
        }

        optimization_dict = OcpFes._prepare_optimization_problem(input_dict)

        return NmpcFes(
            bio_model=[optimization_dict["model"]],
            dynamics=optimization_dict["dynamics"],
            cycle_len=optimization_dict["n_shooting"],
            cycle_duration=cycle_duration,
            n_cycles_simultaneous=n_cycles_simultaneous,
            n_cycles_to_advance=n_cycles_to_advance,
            common_objective_functions=optimization_dict["objective_functions"],
            x_init=optimization_dict["x_init"],
            x_bounds=optimization_dict["x_bounds"],
            constraints=optimization_dict["constraints"],
            parameters=optimization_dict["parameters"],
            parameter_bounds=optimization_dict["parameters_bounds"],
            parameter_init=optimization_dict["parameters_init"],
            parameter_objectives=optimization_dict["parameter_objectives"],
            control_type=control_type,
            use_sx=optimization_dict["use_sx"],
            ode_solver=optimization_dict["ode_solver"],
            n_threads=optimization_dict["n_threads"],
        )
