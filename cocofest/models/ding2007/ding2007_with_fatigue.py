from typing import Callable

from casadi import MX, vertcat
import numpy as np

from cocofest.models.ding2007.ding2007 import DingModelPulseWidthFrequency


class DingModelPulseWidthFrequencyWithFatigue(DingModelPulseWidthFrequency):
    """
    This is a custom models that inherits from bioptim. CustomModel.
    As CustomModel is an abstract class, some methods are mandatory and must be implemented.
    Such as serialize, name_dof, nb_state.

    This is the Ding 2007 model using the stimulation frequency and pulse width in input.

    Ding, J., Chou, L. W., Kesar, T. M., Lee, S. C., Johnston, T. E., Wexler, A. S., & Binder‐Macleod, S. A. (2007).
    Mathematical model that predicts the force–intensity and force–frequency relationships after spinal cord injuries.
    Muscle & Nerve: Official Journal of the American Association of Electrodiagnostic Medicine, 36(2), 214-222.
    """

    def __init__(
        self,
        model_name: str = "ding_2007_with_fatigue",
        muscle_name: str = None,
        stim_time: list[float] = None,
        previous_stim: dict = None,
        sum_stim_truncation: int = 20,
    ):
        super(DingModelPulseWidthFrequencyWithFatigue, self).__init__(
            model_name=model_name,
            muscle_name=muscle_name,
            stim_time=stim_time,
            previous_stim=previous_stim,
            sum_stim_truncation=sum_stim_truncation,
        )
        self._with_fatigue = True
        self.stim_time = stim_time
        self.fmax = 248

        # --- Default values --- #
        ALPHA_A_DEFAULT = -4.0 * 10e-2  # Value from Ding's experimentation [1] (s^-2)
        TAU_FAT_DEFAULT = 127  # Value from Ding's experimentation [1] (s)
        ALPHA_TAU1_DEFAULT = 2.1 * 10e-6  # Value from Ding's experimentation [1] (N^-1)
        ALPHA_KM_DEFAULT = 1.9 * 10e-6  # Value from Ding's experimentation [1] (s^-1.N^-1)

        # ---- Fatigue models ---- #
        self.alpha_a = ALPHA_A_DEFAULT
        self.alpha_tau1 = ALPHA_TAU1_DEFAULT
        self.tau_fat = TAU_FAT_DEFAULT
        self.alpha_km = ALPHA_KM_DEFAULT

    # ---- Absolutely needed methods ---- #
    @property
    def name_dofs(self) -> list[str]:
        muscle_name = "_" + self.muscle_name if self.muscle_name is not None else ""
        return [
            "Cn" + muscle_name,
            "F" + muscle_name,
            "A" + muscle_name,
            "Tau1" + muscle_name,
            "Km" + muscle_name,
        ]

    @property
    def nb_state(self) -> int:
        return 5

    @property
    def identifiable_parameters(self):
        return {
            "a_scale": self.a_scale,
            "tau1_rest": self.tau1_rest,
            "km_rest": self.km_rest,
            "tau2": self.tau2,
            "pd0": self.pd0,
            "pdt": self.pdt,
            "alpha_a": self.alpha_a,
            "alpha_tau1": self.alpha_tau1,
            "alpha_km": self.alpha_km,
            "tau_fat": self.tau_fat,
        }

    def standard_rest_values(self) -> np.array:
        """
        Returns
        -------
        The rested values of Cn, F, A, Tau1, Km
        """
        return np.array([[0], [0], [self.a_scale], [self.tau1_rest], [self.km_rest]])

    def serialize(self) -> tuple[Callable, dict]:
        # This is where you can serialize your models
        # This is useful if you want to save your models and load it later

        # Check commented for nmpc test but might introduce a non continuity from personalized model

        return (
            DingModelPulseWidthFrequencyWithFatigue,
            {
                # "tauc": self.tauc,
                # "a_rest": self.a_rest,
                # "tau1_rest": self.tau1_rest,
                # "km_rest": self.km_rest,
                # "tau2": self.tau2,
                # "alpha_a": self.alpha_a,
                # "alpha_tau1": self.alpha_tau1,
                # "alpha_km": self.alpha_km,
                # "tau_fat": self.tau_fat,
                # "a_scale": self.a_scale,
                # "pd0": self.pd0,
                # "pdt": self.pdt,
                # "stim_time": self.stim_time,
                # "previous_stim": self.previous_stim,
            },
        )

    def system_dynamics(
        self,
        time: MX,
        states: MX,
        controls: MX,
        numerical_timeseries: MX,
    ) -> MX:
        """
        The system dynamics is the function that describes the models.

        Parameters
        ----------
        time: MX
            The system's current node time
        states: MX
            The state of the system CN, F, A, Tau1, Km
        controls: MX
            The controls of the system, pulse_width
        numerical_timeseries: MX
            The numerical timeseries of the system

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        t = time
        cn = states[0]
        f = states[1]
        a = states[2]
        tau1 = states[3]
        km = states[4]
        pulse_width = controls[0]
        t_stim_prev = numerical_timeseries

        cn_dot = self.calculate_cn_dot(cn, t, t_stim_prev)
        a_scale = self.a_calculation(a_scale=a, pulse_width=pulse_width)

        f_dot = self.f_dot_fun(
            cn,
            f,
            a_scale,
            tau1,
            km,
        )  # Equation n°2 from Ding's 2003 article

        a_dot = self.a_dot_fun(a, f)
        tau1_dot = self.tau1_dot_fun(tau1, f)  # Equation n°9 from Ding's 2003 article
        km_dot = self.km_dot_fun(km, f)  # Equation n°11 from Ding's 2003 article
        return vertcat(cn_dot, f_dot, a_dot, tau1_dot, km_dot)

    def a_dot_fun(self, a: MX, f: MX) -> MX | float:
        """
        Parameters
        ----------
        a: MX
            The previous step value of scaling factor (unitless)
        f: MX
            The previous step value of force (N)

        Returns
        -------
        The value of the derivative scaling factor (unitless)
        """
        return -(a - self.a_scale) / self.tau_fat + self.alpha_a * f  # Equation n°5

    def tau1_dot_fun(self, tau1: MX, f: MX) -> MX | float:
        """
        Parameters
        ----------
        tau1: MX
            The previous step value of time_state_force_no_cross_bridge (ms)
        f: MX
            The previous step value of force (N)

        Returns
        -------
        The value of the derivative time_state_force_no_cross_bridge (ms)
        """
        return -(tau1 - self.tau1_rest) / self.tau_fat + self.alpha_tau1 * f  # Equation n°9

    def km_dot_fun(self, km: MX, f: MX) -> MX | float:
        """
        Parameters
        ----------
        km: MX
            The previous step value of cross_bridges (unitless)
        f: MX
            The previous step value of force (N)

        Returns
        -------
        The value of the derivative cross_bridges (unitless)
        """
        return -(km - self.km_rest) / self.tau_fat + self.alpha_km * f  # Equation n°11
