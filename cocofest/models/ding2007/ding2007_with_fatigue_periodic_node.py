from typing import Callable

import numpy as np
from casadi import MX, exp, vertcat

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


class DingModelPulseWidthFrequencyWithFatiguePeriodicNode(
    DingModelPulseWidthFrequencyWithFatigue
):
    """Fixed-frequency Ding model with an exact within-interval calcium history.

    Each shooting interval starts at a stimulation. Numerical data provide the
    steady truncated post-stimulation calcium-history amplitude and the interval
    start time. The history then decays exactly inside the interval; pulse width
    only affects force recruitment and the stimulation intensity remains fixed
    to one. This formulation targets receding-horizon windows after their warmup
    cycle; it does not model the initial buildup from an unstimulated muscle.
    """

    def __init__(
        self,
        model_name: str = "ding_2007_with_fatigue_periodic_node",
        muscle_name: str = None,
        stim_time: list[float] = None,
        previous_stim: dict = None,
        sum_stim_truncation: int = 20,
        stim_interval: float | None = None,
    ):
        super().__init__(
            model_name=model_name,
            muscle_name=muscle_name,
            stim_time=stim_time,
            previous_stim=previous_stim,
            sum_stim_truncation=sum_stim_truncation,
        )
        self._stim_interval = (
            float(stim_interval)
            if stim_interval is not None
            else self._infer_stim_interval(stim_time)
        )

    @staticmethod
    def _infer_stim_interval(stim_time: list[float] | None) -> float | None:
        if stim_time is None or len(stim_time) < 2:
            return None
        intervals = np.diff(np.asarray(stim_time, dtype=float))
        if not np.allclose(intervals, intervals[0]):
            raise ValueError(
                "The periodic-node Ding model requires evenly spaced stimulations."
            )
        return float(intervals[0])

    def serialize(self) -> tuple[Callable, dict]:
        return DingModelPulseWidthFrequencyWithFatiguePeriodicNode, {
            "model_name": self.model_name,
            "muscle_name": self.muscle_name,
            "stim_time": self.stim_time,
            "previous_stim": self.previous_stim,
            "sum_stim_truncation": self.sum_stim_truncation,
            "stim_interval": self._stim_interval,
        }

    def stimulation_increment(self) -> float:
        if self._stim_interval is None:
            raise ValueError(
                "stim_interval is required for periodic-node calcium forcing."
            )
        decay = np.exp(-self._stim_interval / self.tauc)
        return float(1.0 + (self.get_r0(self.km_rest) - 1.0) * decay)

    def post_stimulation_amplitude(
        self, retained_stimulations: int | None = None
    ) -> float:
        """Return the history sum immediately after a stimulation.

        This reproduces the historical truncated sum: the oldest retained pulse
        has coefficient one and subsequent pulses use Ding's ``R_i`` factor.
        """

        if self._stim_interval is None:
            raise ValueError(
                "stim_interval is required for periodic-node calcium forcing."
            )
        count = (
            self.sum_stim_truncation
            if retained_stimulations is None
            else retained_stimulations
        )
        if count < 1:
            return 0.0
        count = min(int(count), self.sum_stim_truncation)
        decay = float(np.exp(-self._stim_interval / self.tauc))
        if count == 1:
            return 1.0
        recent_sum = sum(decay**age for age in range(count - 1))
        return float(decay ** (count - 1) + self.stimulation_increment() * recent_sum)

    def calcium_history(self, time: MX, numerical_timeseries: MX) -> MX:
        post_stimulation_amplitude = numerical_timeseries[0]
        interval_start_time = numerical_timeseries[1]
        current_time = time[0] if time.shape[0] > 1 else time
        return post_stimulation_amplitude * exp(
            -(current_time - interval_start_time) / self.tauc
        )

    def system_dynamics(
        self,
        cn: MX = None,
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
            cn, f, a, tau1, km = (states[index] for index in range(5))
        if controls is not None:
            pulse_width = controls[0]
        elif isinstance(pulse_width, (list, tuple)):
            pulse_width = pulse_width[0]
        if numerical_timeseries is None or numerical_timeseries.shape[0] < 2:
            raise ValueError(
                "Periodic-node dynamics require amplitude and interval-start numerical data."
            )

        cn_sum = self.calcium_history(time, numerical_timeseries)
        cn_dot = self.cn_dot_fun(cn, cn_sum)
        effective_a = self.a_calculation(a_scale=a, pulse_width=pulse_width)
        f_dot = self.f_dot_fun(
            cn,
            f,
            effective_a,
            tau1,
            km,
            force_length_relationship=force_length_relationship,
            force_velocity_relationship=force_velocity_relationship,
            passive_force_relationship=passive_force_relationship,
        )
        return vertcat(
            cn_dot,
            f_dot,
            self.a_dot_fun(a, f),
            self.tau1_dot_fun(tau1, f),
            self.km_dot_fun(km, f),
        )

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
        if n_shooting < 1:
            raise ValueError("n_shooting must be strictly positive.")
        dt = float(final_time) / n_shooting
        if self._stim_interval is None:
            self._stim_interval = dt
        if not np.isclose(dt, self._stim_interval):
            raise ValueError(
                "Each shooting interval must contain exactly one stimulation."
            )

        node_times = np.arange(n_shooting + 1, dtype=float) * dt
        amplitudes = np.full(n_shooting + 1, self.post_stimulation_amplitude())
        values = np.stack((amplitudes, node_times), axis=0)[:, np.newaxis, :]
        return {"periodic_calcium": values}, [[] for _ in range(n_shooting + 1)]
