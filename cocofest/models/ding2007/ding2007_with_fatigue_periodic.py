from typing import Callable

import numpy as np
from casadi import MX, vertcat, exp

from bioptim import (
    ConfigureProblem,
    DynamicsEvaluation,
    NonLinearProgram,
    OptimalControlProgram,
)

from cocofest.models.ding2007.ding2007_with_fatigue import (
    DingModelPulseWidthFrequencyWithFatigue,
)
from cocofest.models.state_configure import StateConfigure


class DingModelPulseWidthFrequencyWithFatiguePeriodic(
    DingModelPulseWidthFrequencyWithFatigue
):
    """
    ACADOS-friendly periodic-stimulation surrogate of the Ding 2007 pulse-width model with fatigue.

    Instead of injecting a truncated list of stimulation times into the dynamics, this variant introduces an
    auxiliary state `Cn_sum` whose dynamics approximate the calcium summation under a fixed stimulation period.
    The default mode keeps the historical constant-intensity approximation. The stimulation-intensity mode lets
    the calcium summation gain vary from one shooting interval to the next through numerical timeseries data.
    """

    def __init__(
        self,
        model_name: str = "ding_2007_with_fatigue_periodic",
        muscle_name: str = None,
        stim_time: list[float] = None,
        previous_stim: dict = None,
        sum_stim_truncation: int = 20,
        stim_interval: float | None = None,
        cn_sum_stimulation_mode: str = "mean",
        stimulation_intensity_index: int = 0,
    ):
        super().__init__(
            model_name=model_name,
            muscle_name=muscle_name,
            stim_time=stim_time,
            previous_stim=previous_stim,
            sum_stim_truncation=sum_stim_truncation,
        )
        self._stim_interval = (
            stim_interval
            if stim_interval is not None
            else self._infer_stim_interval(stim_time)
        )
        if cn_sum_stimulation_mode not in ("mean", "stimulation_intensity"):
            raise ValueError(
                "cn_sum_stimulation_mode must be 'mean' or 'stimulation_intensity'."
            )
        self.cn_sum_stimulation_mode = cn_sum_stimulation_mode
        self.stimulation_intensity_index = stimulation_intensity_index

    @staticmethod
    def _infer_stim_interval(stim_time: list[float] | None) -> float | None:
        if stim_time is None or len(stim_time) < 2:
            return None

        deltas = np.diff(np.array(stim_time, dtype=float))
        reference = float(deltas[0])
        if not np.allclose(deltas, reference):
            raise ValueError(
                "The periodic Ding approximation requires evenly spaced stimulation times."
            )
        return reference

    @property
    def name_dof(self, with_muscle_name: bool = False) -> list[str]:
        muscle_name = (
            "_" + self.muscle_name if self.muscle_name and with_muscle_name else ""
        )
        return [
            "Cn" + muscle_name,
            "Cn_sum" + muscle_name,
            "F" + muscle_name,
            "A" + muscle_name,
            "Tau1" + muscle_name,
            "Km" + muscle_name,
        ]

    @property
    def nb_state(self) -> int:
        return 6

    def standard_rest_values(self) -> np.array:
        return np.array(
            [[0], [0], [0], [self.a_scale], [self.tau1_rest], [self.km_rest]]
        )

    def serialize(self) -> tuple[Callable, dict]:
        _, base_dict = super().serialize()
        base_dict["stim_interval"] = self._stim_interval
        base_dict["cn_sum_stimulation_mode"] = self.cn_sum_stimulation_mode
        base_dict["stimulation_intensity_index"] = self.stimulation_intensity_index
        return DingModelPulseWidthFrequencyWithFatiguePeriodic, base_dict

    def stimulation_decay_factor(self) -> float:
        if self._stim_interval is None:
            raise ValueError(
                "stim_interval could not be inferred. Provide evenly spaced stim_time or pass stim_interval."
            )
        return float(np.exp(-self._stim_interval / self.tauc))

    def periodic_cn_sum_gain(self, stimulation_intensity: MX | float = 1) -> MX | float:
        decay = self.stimulation_decay_factor()
        ri = 1 + (self.get_r0(self.km_rest) - 1) * decay
        return stimulation_intensity * ri / (self.tauc * (1 - decay))

    def cn_sum_dot_fun(self, cn_sum: MX, stimulation_intensity: MX | float = 1) -> MX:
        return -cn_sum / self.tauc + self.periodic_cn_sum_gain(stimulation_intensity)

    def stimulation_intensity_from_numerical_timeseries(
        self, numerical_timeseries: MX | None
    ) -> MX | float:
        if self.cn_sum_stimulation_mode == "mean":
            return 1
        if numerical_timeseries is None:
            return 1
        if (
            not hasattr(numerical_timeseries, "shape")
            or numerical_timeseries.shape[0] == 0
        ):
            return 1
        if numerical_timeseries.shape[0] == 1:
            return numerical_timeseries[0]
        if self.stimulation_intensity_index >= numerical_timeseries.shape[0]:
            raise RuntimeError(
                f"stimulation_intensity_index={self.stimulation_intensity_index} is out of range for "
                f"numerical_timeseries with {numerical_timeseries.shape[0]} rows."
            )
        return numerical_timeseries[self.stimulation_intensity_index]

    def system_dynamics(
        self,
        cn: MX = None,
        cn_sum: MX = None,
        f: MX = None,
        a: MX = None,
        tau1: MX = None,
        km: MX = None,
        pulse_width: MX = None,
        states: MX = None,
        controls: MX = None,
        time: MX = None,
        numerical_timeseries: MX = None,
        force_length_relationship: MX | float = 1,
        force_velocity_relationship: MX | float = 1,
        passive_force_relationship: MX | float = 0,
        **_,
    ) -> MX:
        if states is not None:
            cn = states[0]
            cn_sum = states[1]
            f = states[2]
            a = states[3]
            tau1 = states[4]
            km = states[5]
        if controls is not None:
            pulse_width = controls
        if isinstance(pulse_width, (list, tuple)):
            pulse_width = pulse_width[0]

        stimulation_intensity = self.stimulation_intensity_from_numerical_timeseries(
            numerical_timeseries
        )
        cn_sum_dot = self.cn_sum_dot_fun(cn_sum, stimulation_intensity)
        cn_dot = self.cn_dot_fun(cn, cn_sum)
        a_scale = self.a_calculation(a_scale=a, pulse_width=pulse_width)
        f_dot = self.f_dot_fun(
            cn,
            f,
            a_scale,
            tau1,
            km,
            force_length_relationship=force_length_relationship,
            force_velocity_relationship=force_velocity_relationship,
            passive_force_relationship=passive_force_relationship,
        )
        a_dot = self.a_dot_fun(a, f)
        tau1_dot = self.tau1_dot_fun(tau1, f)
        km_dot = self.km_dot_fun(km, f)
        return vertcat(cn_dot, cn_sum_dot, f_dot, a_dot, tau1_dot, km_dot)

    @staticmethod
    def dynamics(
        time: MX,
        states: MX,
        controls: MX,
        parameters: MX,
        algebraic_states: MX,
        numerical_timeseries: MX,
        nlp: NonLinearProgram,
        fes_model=None,
        force_length_relationship: MX | float = 1,
        force_velocity_relationship: MX | float = 1,
        passive_force_relationship: MX | float = 0,
    ) -> DynamicsEvaluation:
        model = fes_model if fes_model else nlp.model
        dxdt = model.system_dynamics(
            states=states,
            controls=controls,
            time=time,
            numerical_timeseries=numerical_timeseries,
            force_length_relationship=force_length_relationship,
            force_velocity_relationship=force_velocity_relationship,
            passive_force_relationship=passive_force_relationship,
        )
        return DynamicsEvaluation(
            dxdt=dxdt, defects=model._collocation_defects(nlp, model, dxdt)
        )

    def declare_ding_variables(
        self,
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        numerical_data_timeseries: dict[str, np.ndarray] = None,
        contact_type: list = (),
    ):
        StateConfigure().configure_all_fes_model_states(ocp, nlp, fes_model=self)
        StateConfigure().configure_last_pulse_width(ocp, nlp)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, dyn_func=self.dynamics)

    def get_numerical_data_time_series(
        self, n_shooting, final_time, all_stim_time=None
    ):
        return {}, [[] for _ in range(n_shooting + 1)]
