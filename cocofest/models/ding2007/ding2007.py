from typing import Callable, List

from casadi import MX, vertcat, exp

from bioptim import States

from cocofest.models.ding2003.ding2003 import DingModelFrequency
from cocofest.models.state_configure import StateConfigure


class DingModelPulseWidthFrequency(DingModelFrequency):
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
        model_name: str = "ding_2007",
        muscle_name: str = None,
        stim_time: list[float] = None,
        previous_stim: dict = None,
        sum_stim_truncation: int = 20,
    ):
        super(DingModelPulseWidthFrequency, self).__init__(
            model_name=model_name,
            muscle_name=muscle_name,
            stim_time=stim_time,
            previous_stim=previous_stim,
            sum_stim_truncation=sum_stim_truncation,
        )
        self._with_fatigue = False
        self.pulse_width = None
        self.previous_stim = previous_stim if previous_stim else {"time": []}
        self.stim_time = stim_time

        # --- Default values --- #
        A_SCALE_DEFAULT = 4920  # Value from Ding's 2007 article (N/s)
        PD0_DEFAULT = 0.000131405  # Value from Ding's 2007 article (s)
        PDT_DEFAULT = 0.000194138  # Value from Ding's 2007 article (s)
        TAU1_REST_DEFAULT = 0.060601  # Value from Ding's 2003 article (s)
        TAU2_DEFAULT = 0.001  # Value from Ding's 2007 article (s)
        KM_REST_DEFAULT = 0.137  # Value from Ding's 2007 article (unitless)
        TAUC_DEFAULT = 0.011  # Value from Ding's 2007 article (s)

        # ---- Custom values for the example ---- #
        # ---- Force models ---- #
        self.a_scale = A_SCALE_DEFAULT
        self.pd0 = PD0_DEFAULT
        self.pdt = PDT_DEFAULT
        self.tau1_rest = TAU1_REST_DEFAULT
        self.tau2 = TAU2_DEFAULT
        self.km_rest = KM_REST_DEFAULT
        self.tauc = TAUC_DEFAULT
        self.fmax = 248  # Maximum force (N) at 100 Hz and 600 us

    @property
    def control_configuration_functions(self) -> List[States | Callable]:
        return [StateConfigure().configure_last_pulse_width]

    @property
    def identifiable_parameters(self):
        return {
            "a_scale": self.a_scale,
            "tau1_rest": self.tau1_rest,
            "km_rest": self.km_rest,
            "tau2": self.tau2,
            "pd0": self.pd0,
            "pdt": self.pdt,
        }

    def set_a_scale(self, model, a_scale: MX | float):
        # models is required for bioptim compatibility
        self.a_scale = a_scale

    def set_pd0(self, model, pd0: MX | float):
        self.pd0 = pd0

    def set_pdt(self, model, pdt: MX | float):
        self.pdt = pdt

    def serialize(self) -> tuple[Callable, dict]:
        # This is where you can serialize your models
        # This is useful if you want to save your models and load it later
        return (
            DingModelPulseWidthFrequency,
            {
                "tauc": self.tauc,
                "a_rest": self.a_rest,
                "tau1_rest": self.tau1_rest,
                "km_rest": self.km_rest,
                "tau2": self.tau2,
                "a_scale": self.a_scale,
                "pd0": self.pd0,
                "pdt": self.pdt,
                "stim_time": self.stim_time,
                "previous_stim": self.previous_stim,
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
            The state of the system CN, F
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
        pulse_width = controls[0]
        t_stim_prev = numerical_timeseries

        cn_dot = self.calculate_cn_dot(cn, t, t_stim_prev)
        a_scale = self.a_calculation(a_scale=self.a_scale, pulse_width=pulse_width)
        f_dot = self.f_dot_fun(
            cn,
            f,
            a_scale,
            self.tau1_rest,
            self.km_rest,
        )  # Equation n°2 from Ding's 2003 article
        return vertcat(cn_dot, f_dot)

    def a_calculation(
        self,
        a_scale: float | MX,
        pulse_width: MX,
    ) -> MX:
        """
        Parameters
        ----------
        a_scale: float | MX
            The scaling factor of the current stimulation (unitless)
        pulse_width: MX
            The pulsation duration of the current stimulation (s)
        Returns
        -------
        The value of scaling factor (unitless)
        """
        return a_scale * (1 - exp(-(pulse_width - self.pd0) / self.pdt))

    def a_calculation_identification(
        self,
        a_scale: float | MX,
        pulse_width: MX,
        pd0: float | MX,
        pdt: float | MX,
    ) -> MX:
        """
        Parameters
        ----------
        a_scale: float | MX
            The scaling factor of the current stimulation (unitless)
        pulse_width: MX
            The pulsation duration of the current stimulation (s)
        pd0: float | MX
            The pd0 value (s)
        pdt: float | MX
            The pdt value (s)

        Returns
        -------
        The value of scaling factor (unitless)
        """
        return a_scale * (1 - exp(-(pulse_width - pd0) / pdt))

    def set_impulse_width(self, value: list[MX]):
        """
        Sets the pulse width for each pulse (phases) according to the ocp parameter "pulse_width"

        Parameters
        ----------
        value: list[MX]
            The pulsation duration list (s)
        """
        self.pulse_width = value
