import numpy as np

from bioptim import (
    OdeSolver,
    MultiCyclicNonlinearModelPredictiveControl,
    ControlType,
    SolutionMerge,
    Solution,
)
from .fes_ocp import OcpFes
from .fes_ocp_dynamics import OcpFesMsk
from ..models.dynamical_model import FesMskModel
from ..models.ding2007_with_fatigue import DingModelPulseWidthFrequencyWithFatigue


class NmpcFesMsk(MultiCyclicNonlinearModelPredictiveControl):
    def advance_window_bounds_states(self, sol, n_cycles_simultaneous=None, **extra):
        super(NmpcFesMsk, self).advance_window_bounds_states(sol)
        self.update_stim(sol)
        if self.nlp[0].model.for_cycling:
            self.nlp[0].x_bounds["q"].min[-1, :] = (
                self.nlp[0].model.bounds_from_ranges("q").min[-1] * n_cycles_simultaneous
            )
            self.nlp[0].x_bounds["q"].max[-1, :] = (
                self.nlp[0].model.bounds_from_ranges("q").max[-1] * n_cycles_simultaneous
            )
        return True

    def advance_window_initial_guess_states(self, sol, n_cycles_simultaneous=None):
        # Reimplementation of the advance_window method so the rotation of the wheel restart at -pi
        super(NmpcFesMsk, self).advance_window_initial_guess_states(sol)
        # if cycling else pass
        if self.nlp[0].model.for_cycling:
            q = sol.decision_states(to_merge=SolutionMerge.NODES)["q"]
            self.nlp[0].x_init["q"].init[-1, :] = q[-1, :]  # Keep the previously found value for the wheel
        return True

    def update_stim(self, sol):
        # only keep the last 10 stimulation times
        previous_stim_time = [
            round(x - self.phase_time[0], 2) for x in self.nlp[0].model.muscles_dynamics_model[0].stim_time[-10:]
        ]  # TODO fix this (keep the middle window)
        for i in range(len(self.nlp[0].model.muscles_dynamics_model)):
            self.nlp[0].model.muscles_dynamics_model[i].previous_stim = (
                {}
                if self.nlp[0].model.muscles_dynamics_model[i].previous_stim is None
                else self.nlp[0].model.muscles_dynamics_model[i].previous_stim
            )
            self.nlp[0].model.muscles_dynamics_model[i].previous_stim["time"] = previous_stim_time
            if isinstance(self.nlp[0].model.muscles_dynamics_model[i], DingModelPulseWidthFrequencyWithFatigue):
                self.nlp[0].model.muscles_dynamics_model[i].previous_stim["pulse_width"] = list(
                    sol.parameters["pulse_width_" + self.nlp[0].model.muscles_dynamics_model[i].muscle_name][-10:]
                )
            self.nlp[0].model.muscles_dynamics_model[i].all_stim = (
                self.nlp[0].model.muscles_dynamics_model[i].previous_stim["time"]
                + self.nlp[0].model.muscles_dynamics_model[i].stim_time
            )

    @staticmethod
    def prepare_nmpc(
        model: FesMskModel = None,
        cycle_duration: int | float = None,
        n_cycles_simultaneous: int = None,
        n_cycles_to_advance: int = None,
        n_total_cycles: int = None,
        pulse_width: dict = None,
        pulse_intensity: dict = None,
        objective: dict = None,
        msk_info: dict = None,
        external_forces: dict = None,
        initial_guess_warm_start: bool = False,
        use_sx: bool = True,
        ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
        n_threads: int = 1,
        control_type: ControlType = ControlType.CONSTANT,
    ):

        input_dict = {
            "model": model,
            "n_shooting": OcpFes.prepare_n_shooting(model.muscles_dynamics_model[0].stim_time, cycle_duration),
            "final_time": cycle_duration,
            "n_cycles_simultaneous": n_cycles_simultaneous,
            "n_cycles_to_advance": n_cycles_to_advance,
            "n_total_cycles": n_total_cycles,
            "pulse_width": pulse_width,
            "pulse_intensity": pulse_intensity,
            "objective": objective,
            "msk_info": msk_info,
            "external_forces": external_forces,
            "initial_guess_warm_start": initial_guess_warm_start,
            "use_sx": use_sx,
            "ode_solver": ode_solver,
            "n_threads": n_threads,
            "control_type": control_type,
        }

        optimization_dict = OcpFesMsk._prepare_optimization_problem(input_dict)

        return NmpcFesMsk(
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
            use_sx=optimization_dict["use_sx"],
            ode_solver=optimization_dict["ode_solver"],
            n_threads=optimization_dict["n_threads"],
            control_type=optimization_dict["control_type"],
        )

    @staticmethod
    def prepare_nmpc_for_cycling(
        model: FesMskModel = None,
        cycle_duration: int | float = None,
        n_cycles_simultaneous: int = None,
        n_cycles_to_advance: int = None,
        n_total_cycles: int = None,
        pulse_width: dict = None,
        pulse_intensity: dict = None,
        objective: dict = None,
        msk_info: dict = None,
        external_forces: dict = None,
        initial_guess_warm_start: bool = False,
        use_sx: bool = True,
        ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
        control_type: ControlType = ControlType.CONSTANT,
        n_threads: int = 1,
    ):
        input_dict = {
            "model": model,
            "n_shooting": OcpFes.prepare_n_shooting(
                model.muscles_dynamics_model[0].stim_time, cycle_duration * n_cycles_simultaneous
            ),
            "final_time": cycle_duration,
            "n_cycles_simultaneous": n_cycles_simultaneous,
            "n_cycles_to_advance": n_cycles_to_advance,
            "n_total_cycles": n_total_cycles,
            "pulse_width": pulse_width,
            "pulse_intensity": pulse_intensity,
            "objective": objective,
            "msk_info": msk_info,
            "initial_guess_warm_start": initial_guess_warm_start,
            "use_sx": use_sx,
            "ode_solver": ode_solver,
            "n_threads": n_threads,
            "control_type": control_type,
            "external_forces": external_forces,
        }

        optimization_dict = OcpFesMsk._prepare_optimization_problem(input_dict)
        optimization_dict_for_cycling = OcpFesMsk._prepare_optimization_problem_for_cycling(
            optimization_dict, input_dict
        )

        return NmpcFesMsk(
            bio_model=[optimization_dict["model"]],
            dynamics=optimization_dict["dynamics"],
            cycle_len=optimization_dict["n_shooting"],
            cycle_duration=cycle_duration,
            n_cycles_simultaneous=n_cycles_simultaneous,
            n_cycles_to_advance=n_cycles_to_advance,
            common_objective_functions=optimization_dict["objective_functions"],
            x_init=optimization_dict_for_cycling["x_init"],
            x_bounds=optimization_dict_for_cycling["x_bounds"],
            u_init=optimization_dict_for_cycling["u_init"],
            u_bounds=optimization_dict_for_cycling["u_bounds"],
            constraints=optimization_dict_for_cycling["constraints"],
            parameters=optimization_dict["parameters"],
            parameter_bounds=optimization_dict["parameters_bounds"],
            parameter_init=optimization_dict["parameters_init"],
            parameter_objectives=optimization_dict["parameter_objectives"],
            control_type=control_type,
            use_sx=optimization_dict["use_sx"],
            ode_solver=optimization_dict["ode_solver"],
            n_threads=optimization_dict["n_threads"],
        )
