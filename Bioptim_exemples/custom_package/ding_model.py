"""
This script implements several custom model to work with bioptim. Bioptim has a deep connection with biorbd,
but it is possible to use bioptim without biorbd. This is an example of how to use bioptim with different custom models.
"""
from typing import Callable

import numpy as np
from casadi import MX, SX, exp, vertcat, Function, sum1, horzcat, tanh, fmin, fmax, if_else

from bioptim import (
    OptimalControlProgram,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsEvaluation,
    DynamicsFunctions,
)


class DingModelFrequency:
    """
    This is a custom model that inherits from bioptim. CustomModel.
    As CustomModel is an abstract class, some methods must be implemented.
    Such as serialize, name_dof, nb_state and name.

    This is the Ding 2003 model using the stimulation frequency in input.
    """

    def __init__(self, name: str = None):
        self._name = name
        # ---- Custom values for the example ---- #
        self.tauc = 0.020  # Value from Ding's experimentation [1] (s)
        self.r0_km_relationship = 1.04  # (unitless)
        # ---- Different values for each person ---- #
        self.alpha_a = -4.0 * 10e-7  # Value from Ding's experimentation [1] (s^-2)
        self.alpha_tau1 = 2.1 * 10e-5  # Value from Ding's experimentation [1] (N^-1)
        self.tau2 = 0.060  # Close value from Ding's experimentation [2] (s)
        self.tau_fat = 127  # Value from Ding's experimentation [1] (s)
        self.alpha_km = 1.9 * 10e-8  # Value from Ding's experimentation [1] (s^-1.N^-1)
        self.a_rest = 3009  # Value from Ding's experimentation [1] (N.s-1)
        self.tau1_rest = 0.050957  # Value from Ding's experimentation [1] (s)
        self.km_rest = 0.103  # Value from Ding's experimentation [1] (unitless)

    def standard_rest_values(self) -> np.array:
        """
        Returns
        -------
        The rested values of Cn, F, A, Tau1, Km
        """
        return np.array([[0], [0], [self.a_rest], [self.tau1_rest], [self.km_rest]])

    # ---- Absolutely needed methods ---- #
    def serialize(self) -> tuple[Callable, dict]:
        # This is where you can serialize your model
        # This is useful if you want to save your model and load it later
        return DingModelFrequency, {"tauc": self.tauc, "a_rest": self.a_rest, "tau1_rest": self.tau1_rest,
                                    "km_rest": self.km_rest, "tau2": self.tau2, "alpha_a": self.alpha_a,
                                    "alpha_tau1": self.alpha_tau1, "alpha_km": self.alpha_km, "tau_fat": self.tau_fat}

    # ---- Needed for the example ---- #
    @property
    def name_dof(self):
        return ["cn", "f", "a", "tau1", "km"]

    @property
    def nb_state(self):
        return 5

    @property
    def name(self):
        return self._name

    # ---- Model's dynamics ---- #
    def system_dynamics(
        self, cn: MX, f: MX, a: MX, tau1: MX, km: MX, t: MX, t_stim_prev: list[MX], impulse_time: None,
        intensity_stim: None,
    ) -> MX:
        """
        The system dynamics is the function that describes the model.

        Parameters
        ----------
        cn: MX
            The value of the ca_troponin_complex (unitless)
        f: MX
            The value of the force (N)
        a: MX
            The value of the scaling factor (unitless)
        tau1: MX
            The value of the time_state_force_no_cross_bridge (ms)
        km: MX
            The value of the cross_bridges (unitless)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_prev: list[MX]
            The list of the time of the previous stimulations (ms)
        impulse_time: None
            The pulsation duration of the current stimulation (ms). Not used for this model
        intensity_stim: None
            The pulsation intensity of the current stimulation (mA). Not used for this model

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        r0 = km + MX(self.r0_km_relationship)  # Simplification
        cn_dot = self.cn_dot_fun(cn, r0, t, t_stim_prev)  # Equation n°1
        f_dot = self.f_dot_fun(cn, f, a, tau1, km)  # Equation n°2
        a_dot = self.a_dot_fun(a, f)  # Equation n°5
        tau1_dot = self.tau1_dot_fun(tau1, f)  # Equation n°9
        km_dot = self.km_dot_fun(km, f)  # Equation n°11

        return vertcat(cn_dot, f_dot, a_dot, tau1_dot, km_dot)

    def exp_time_fun(self, t: MX, t_stim_i: MX):
        """
        Parameters
        ----------
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_i: MX
            Time when the stimulation i occurred (ms)

        Returns
        -------
        A part of the n°1 equation
        """
        return exp(-(t - t_stim_i) / self.tauc)  # Eq from [1]

    def ri_fun(self, r0: MX, time_between_stim: MX):
        """
        Parameters
        ----------
        r0: MX
            Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli (unitless)
        time_between_stim: MX
            Time between the last stimulation i and the current stimulation i (ms)

        Returns
        -------
        A part of the n°1 equation
        """
        return 1 + (r0 - 1) * exp(time_between_stim / self.tauc)  # Eq from [1]

    def cn_sum_fun(self, r0: MX, t: MX, t_stim_prev: list[MX], intensity_stim: None):
        """
        Parameters
        ----------
        r0: MX
            Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli (unitless)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_prev: list[MX]
            The list of the time of the previous stimulations (ms)
        intensity_stim: None
            The pulsation intensity of the current stimulation (mA). Not used for this model

        Returns
        -------
        A part of the n°1 equation
        """
        sum_multiplier = 0

        for i in range(len(t_stim_prev)):  # Eq from [1]
            if i == 0 and len(t_stim_prev) == 1:  # Eq from Bakir et al.
                ri = 1
            # elif i == 0 and len(t_stim_prev) > 1:
            #     previous_phase_time = t_stim_prev[i+1] - t_stim_prev[i]
            #     ri = 0
            else:
                previous_phase_time = t_stim_prev[i] - t_stim_prev[i - 1]
                ri = self.ri_fun(r0, previous_phase_time)

            exp_time = self.exp_time_fun(t, t_stim_prev[i])
            # if i in [10, 11, 12, 13, 14]:
            #     ri = 0
            sum_multiplier = ri * exp_time
        return sum_multiplier

    def cn_dot_fun(self, cn: MX , r0: MX, t: MX, t_stim_prev: list[MX]):
        """
        Parameters
        ----------
        cn: MX
            The previous step value of ca_troponin_complex (unitless)
        r0: MX
            Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli (unitless)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_prev: list[MX]
            The list of the time of the previous stimulations (ms)

        Returns
        -------
        The value of the derivative ca_troponin_complex (unitless)
        """
        sum_multiplier = self.cn_sum_fun(r0, t, t_stim_prev, None)

        return (1 / self.tauc) * sum_multiplier - (cn / self.tauc)  # Eq(1)

    def f_dot_fun(self, cn: MX, f: MX, a: MX, tau1: MX, km: MX):
        """
        Parameters
        ----------
        cn: MX
            The previous step value of ca_troponin_complex (unitless)
        f: MX
            The previous step value of force (N)
        a: MX
            The previous step value of scaling factor (unitless)
        tau1: MX
            The previous step value of time_state_force_no_cross_bridge (ms)
        km: MX
            The previous step value of cross_bridges (unitless)

        Returns
        -------
        The value of the derivative force (N)
        """
        return a * (cn / (km + cn)) - (f / (tau1 + self.tau2 * (cn / (km + cn))))  # Eq(2)

    def a_dot_fun(self, a: MX, f: MX):
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
        return -(a - self.a_rest) / self.tau_fat + self.alpha_a * f  # Eq(5)

    def tau1_dot_fun(self, tau1: MX, f: MX):
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
        return -(tau1 - self.tau1_rest) / self.tau_fat + self.alpha_tau1 * f  # Eq(9)

    def km_dot_fun(self, km: MX, f: MX):
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
        return -(km - self.km_rest) / self.tau_fat + self.alpha_km * f  # Eq(11)

    @staticmethod
    def get_time_parameters(nlp_parameters) -> MX | SX:
        time_parameters = vertcat()
        for j in range(nlp_parameters.cx.shape[0]):
            if "time" in str(nlp_parameters.cx[j]):
                time_parameters = vertcat(time_parameters, nlp_parameters.cx[j])
        return time_parameters


class DingModelPulseDurationFrequency(DingModelFrequency):
    def __init__(self):
        super().__init__()
        self.impulse_time = None
        self.a_scale = 492
        self.pd0 = 0.000131405
        self.pdt = 0.000194138
        self.tau1_rest = 0.060601  # Value from Ding's experimentation [1] (s)
        self.tau2 = 0.001
        self.km = 0.137
        self.tauc = 0.011

    # ---- Absolutely needed methods ---- #
    @property
    def name_dof(self):
        return ["cn", "f", "tau1", "km"]

    @property
    def nb_state(self):
        return 4

    def standard_rest_values(self) -> np.array:
        """
        Returns
        -------
        The rested values of Cn, F, A, Tau1, Km
        """
        return np.array([[0], [0], [self.tau1_rest], [self.km_rest]])

    def serialize(self) -> tuple[Callable, dict]:
        # This is where you can serialize your model
        # This is useful if you want to save your model and load it later
        return DingModelPulseDurationFrequency, {"tauc": self.tauc, "a_rest": self.a_rest,
                                                 "tau1_rest": self.tau1_rest, "km_rest": self.km_rest,
                                                 "tau2": self.tau2, "alpha_a": self.alpha_a,
                                                 "alpha_tau1": self.alpha_tau1, "alpha_km": self.alpha_km,
                                                 "tau_fat": self.tau_fat, "a_scale": self.a_scale,
                                                 "pd0": self.pd0, "pdt": self.pdt}

    def system_dynamics(
        self, cn: MX, f: MX, tau1: MX, km: MX, t: MX, t_stim_prev: list[MX], impulse_time: MX,
        intensity_stim: None,
    ) -> MX:
        """
        The system dynamics is the function that describes the model.

        Parameters
        ----------
        cn: MX
            The value of the ca_troponin_complex (unitless)
        f: MX
            The value of the force (N)
        tau1: MX
            The value of the time_state_force_no_cross_bridge (ms)
        km: MX
            The value of the cross_bridges (unitless)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_prev: list[MX]
            The list of the time of the previous stimulations (ms)
        impulse_time: MX
            The pulsation duration of the current stimulation (ms)
        intensity_stim: None
            The pulsation intensity of the current stimulation (mA). Not used for this model

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        # from Ding's 2003 article
        r0 = MX(5)  # Simplification
        # r0 = km + MX(self.r0_km_relationship)  # Simplification
        cn_dot = self.cn_dot_fun(cn, r0, t, t_stim_prev)  # Equation n°1
        a = self.a_calculation(impulse_time)
        f_dot = self.f_dot_fun(cn, f, a, tau1, km)  # Equation n°2
        tau1_dot = self.tau1_dot_fun(tau1, f)  # Equation n°9
        km_dot = self.km_dot_fun(km, f)  # Equation n°11

        return vertcat(cn_dot, f_dot, tau1_dot, km_dot)

    def a_calculation(self, impulse_time: MX):
        """
        Parameters
        ----------
        impulse_time: MX
            The pulsation duration of the current stimulation (ms)

        Returns
        -------
        The value of scaling factor (unitless)
        """
        # new equation to include impulse time [2]
        return self.a_scale * (1 - np.exp(-(impulse_time - self.pd0) / self.pdt))

    def set_impulse_duration(self, value: MX):
        self.impulse_time = value

    @staticmethod
    def get_pulse_duration_parameters(nlp_parameters):
        pulse_duration_parameters = vertcat()
        for j in range(nlp_parameters.mx.shape[0]):
            if "pulse_duration_" in nlp_parameters.mx[j].name():
                pulse_duration_parameters = vertcat(pulse_duration_parameters, nlp_parameters.mx[j])
        return pulse_duration_parameters




class DingModelIntensityFrequency(DingModelFrequency):
    def __init__(self):
        super().__init__()
        self.ar = 0.586  # (-) Translation of axis coordinates.
        self.bs = 0.026  # (-) Fiber muscle recruitment constant identification.
        self.Is = 63.1  # (mA) Muscle saturation intensity.
        self.cr = 0.833  # (-) Translation of axis coordinates.
        self.impulse_intensity = None

    # ---- Absolutely needed methods ---- #
    def serialize(self) -> tuple[Callable, dict]:
        # This is where you can serialize your model
        # This is useful if you want to save your model and load it later
        return DingModelIntensityFrequency, {"tauc": self.tauc, "a_rest": self.a_rest,
                                                             "tau1_rest": self.tau1_rest, "km_rest": self.km_rest,
                                                             "tau2": self.tau2, "alpha_a": self.alpha_a,
                                                             "alpha_tau1": self.alpha_tau1, "alpha_km": self.alpha_km,
                                                             "tau_fat": self.tau_fat, "ar": self.ar,
                                                             "bs": self.bs, "Is": self.Is, "cr": self.cr}

    def system_dynamics(
        self, cn: MX | SX, f: MX | SX, a: MX | SX, tau1: MX | SX, km: MX | SX, t: MX | SX,
            t_stim_prev: list[MX] | list [SX], impulse_time: None, intensity_stim: list[MX] | list [SX],
    ) -> MX | SX:
        """
        The system dynamics is the function that describes the model.

        Parameters
        ----------
        cn: MX
            The value of the ca_troponin_complex (unitless)
        f: MX
            The value of the force (N)
        a: MX
            The value of the scaling factor (unitless)
        tau1: MX
            The value of the time_state_force_no_cross_bridge (ms)
        km: MX
            The value of the cross_bridges (unitless)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_prev: list[MX]
            The list of the time of the previous stimulations (ms)
        impulse_time: MX
            The pulsation duration of the current stimulation (ms)
        intensity_stim: MX
            The pulsation intensity of the current stimulation (mA)

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        # from Ding's 2003 article
        r0 = km + self.r0_km_relationship  # Simplification
        cn_dot = self.cn_dot_fun(cn, r0, t, t_stim_prev, intensity_stim)  # Equation n°1
        f_dot = self.f_dot_fun(cn, f, a, tau1, km)  # Equation n°2
        a_dot = self.a_dot_fun(a, f)  # Equation n°5
        tau1_dot = self.tau1_dot_fun(tau1, f)  # Equation n°9
        km_dot = self.km_dot_fun(km, f)  # Equation n°11

        return vertcat(cn_dot, f_dot, a_dot, tau1_dot, km_dot)

    def cn_dot_fun(self, cn: MX | SX, r0: MX | SX, t: MX | SX, t_stim_prev: list[MX] | list [SX], intensity_stim: list[MX] | list[SX]):
        """
        Parameters
        ----------
        cn: MX
            The previous step value of ca_troponin_complex (unitless)
        r0: MX
            Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli (unitless)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_prev: list[MX]
            The list of the time of the previous stimulations (ms)
        intensity_stim: list[MX]
            The list of stimulation intensity (mA)

        Returns
        -------
        The value of the derivative ca_troponin_complex (unitless)
        """
        sum_multiplier = self.cn_sum_fun(r0, t, t_stim_prev, intensity_stim)

        return (1 / self.tauc) * sum_multiplier - (cn / self.tauc)  # Eq(1)

    def cn_sum_fun(self, r0: MX, t: MX, t_stim_prev: list[MX], intensity_stim: list[MX]):
        """
        Parameters
        ----------
        r0: MX
            Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli (unitless)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_prev: list[MX]
            The list of the time of the previous stimulations (ms)
        intensity_stim: None
            The pulsation intensity of the current stimulation (mA). Not used for this model

        Returns
        -------
        A part of the n°1 equation
        """
        sum_multiplier = 0
        for i in range(len(t_stim_prev)):  # Eq from [1]
            if i == 0 and len(t_stim_prev) == 1:  # Eq from Bakir et al.
                ri = 1
            elif i == 0 and len(t_stim_prev) != 1:
                previous_phase_time = t_stim_prev[i+1] - t_stim_prev[i]
                ri = self.ri_fun(r0, previous_phase_time)
            else:
                previous_phase_time = t_stim_prev[i] - t_stim_prev[i - 1]
                ri = self.ri_fun(r0, previous_phase_time)

            exp_time = self.exp_time_fun(t, t_stim_prev[i])

            lambda_i = self.lambda_i_calculation(intensity_stim[i])

            sum_multiplier += lambda_i * ri * exp_time

        return sum_multiplier

    def lambda_i_calculation(self, intensity_stim: MX):
        lambda_i = self.ar * (tanh(self.bs * (intensity_stim - self.Is)) + self.cr)  # equation include intensity
        lambda_i = if_else(lambda_i < 0, 0, lambda_i)
        lambda_i = if_else(lambda_i > 1, 1, lambda_i)
        return lambda_i

    def set_impulse_intensity(self, value: list[MX]):
        self.impulse_intensity = []
        for i in range(value.shape[0]):
            self.impulse_intensity.append(value[i])
        # self.impulse_intensity = value

    @staticmethod
    def get_intensity_parameters(nlp_parameters) -> MX | SX:
        intensity_parameters = vertcat()
        for j in range(nlp_parameters.cx.shape[0]):
            if "pulse_intensity" in str(nlp_parameters.cx[j]):
                intensity_parameters = vertcat(intensity_parameters, nlp_parameters.cx[j])
        return intensity_parameters


"""
This script implements a custom dynamics to work with bioptim. Bioptim has a deep connection with biorbd,
but it is possible to use bioptim without biorbd.
This is an example of how to use bioptim with a custom dynamics.
"""


class CustomDynamicsFrequency:
    def __init__(self):
        self.t0_phase_in_ocp = 0
        self.dynamics_eval_horzcat = None

    @staticmethod
    def custom_dynamics(
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        nlp: NonLinearProgram,
        t=None,
    ) -> DynamicsEvaluation:
        """
        Functional electrical stimulation dynamic

        Parameters
        ----------
        states: MX | SX
            The state of the system CN, F, A, Tau1, Km
        controls: MX | SX
            The controls of the system, none
        parameters: MX | SX
            The parameters acting on the system, final time of each phase
        nlp: NonLinearProgram
            A reference to the phase
        t: MX
            Current node time, this t is used to set the dynamics and as to be a symbolic
        Returns
        -------
        The derivative of the states in the tuple[MX | SX]] format
        """

        t_stim_prev = []  # Every stimulation instant before the current phase, i.e.: the beginning of each phase

        if nlp.parameters.mx.shape[0] == 1:  # check if time is bimapped
            for i in range(nlp.phase_idx+1):
                t_stim_prev.append(nlp.parameters.mx*i)
        else:
            for i in range(nlp.phase_idx+1):
                t_stim_prev.append(sum1(nlp.parameters.mx[0: i]))

        return DynamicsEvaluation(
            dxdt=nlp.model.system_dynamics(
                cn=states[0],
                f=states[1],
                a=states[2],
                tau1=states[3],
                km=states[4],
                t=t,
                t_stim_prev=t_stim_prev,
                impulse_time=None,
                intensity_stim=None
            ),
            defects=None,
        )

    @staticmethod
    def custom_configure_dynamics_function(ocp, nlp, **extra_params):
        """
        Configure the dynamics of the system

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        **extra_params:
            t: MX
                Current node time
        """

        nlp.parameters = ocp.v.parameters_in_list
        DynamicsFunctions.apply_parameters(nlp.parameters.mx, nlp)

        # Gets the t0 time for the current phase

        if nlp.parameters.mx.shape[0] != 1:  # check if time is not bimapped
            CustomDynamicsFrequency.t0_phase_in_ocp = sum1(nlp.parameters.mx[0: nlp.phase_idx])

        # Gets every time node for the current phase
        for i in range(nlp.ns):
            if nlp.parameters.mx.shape[0] == 1:  # check if time is bimapped
                t_node_in_phase = nlp.parameters.mx * nlp.phase_idx / (nlp.ns + 1) * i
                t_node_in_ocp = nlp.parameters.mx * nlp.phase_idx + t_node_in_phase
                extra_params["t"] = t_node_in_ocp
            else:
                t_node_in_phase = nlp.parameters.mx[nlp.phase_idx] / (nlp.ns + 1) * i
                t_node_in_ocp = CustomDynamicsFrequency.t0_phase_in_ocp + t_node_in_phase
                extra_params["t"] = t_node_in_ocp

            dynamics_eval = CustomDynamicsFrequency.custom_dynamics(
                nlp.states["scaled"].mx_reduced, nlp.controls["scaled"].mx_reduced, nlp.parameters.mx, nlp,
                **extra_params
            )

            CustomDynamicsFrequency.dynamics_eval_horzcat = horzcat(dynamics_eval.dxdt) if i == 0 else horzcat(CustomDynamicsFrequency.dynamics_eval_horzcat, dynamics_eval.dxdt)

        nlp.dynamics_func = Function(
            "ForwardDyn",
            [nlp.states["scaled"].mx_reduced, nlp.controls["scaled"].mx_reduced, nlp.parameters.mx],
            [CustomDynamicsFrequency.dynamics_eval_horzcat],
            ["x", "u", "p"],
            ["xdot"],
        )

    @staticmethod
    def declare_ding_variables(ocp: OptimalControlProgram, nlp: NonLinearProgram):
        """
        Tell the program which variables are states and controls.
        The user is expected to use the ConfigureProblem.configure_xxx functions.
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """
        CustomDynamicsFrequency.configure_ca_troponin_complex(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        CustomDynamicsFrequency.configure_force(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        CustomDynamicsFrequency.configure_scaling_factor(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        CustomDynamicsFrequency.configure_time_state_force_no_cross_bridge(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        CustomDynamicsFrequency.configure_cross_bridges(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)

        t = MX.sym("t")  # t needs a symbolic value to start computing in custom_configure_dynamics_function

        CustomDynamicsFrequency.custom_configure_dynamics_function(ocp, nlp, t=t)

    @staticmethod
    def configure_ca_troponin_complex(ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False):
        """
        Configure a new variable of the Ca+ troponin complex (unitless)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized coordinates should be a state
        as_controls: bool
            If the generalized coordinates should be a control
        as_states_dot: bool
            If the generalized velocities should be a state_dot
        """
        name = "Cn"
        name_cn = [name]
        ConfigureProblem.configure_new_variable(
            name,
            name_cn,
            ocp,
            nlp,
            as_states,
            as_controls,
            as_states_dot,
        )

    @staticmethod
    def configure_force(ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False):
        """
        Configure a new variable of the force (N)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized coordinates should be a state
        as_controls: bool
            If the generalized coordinates should be a control
        as_states_dot: bool
            If the generalized velocities should be a state_dot
        """
        name = "F"
        name_f = [name]
        ConfigureProblem.configure_new_variable(
            name,
            name_f,
            ocp,
            nlp,
            as_states,
            as_controls,
            as_states_dot,
        )

    @staticmethod
    def configure_scaling_factor(ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False):
        """
        Configure a new variable of the scaling factor (N/ms)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized coordinates should be a state
        as_controls: bool
            If the generalized coordinates should be a control
        as_states_dot: bool
            If the generalized velocities should be a state_dot
        """
        name = "A"
        name_a = [name]
        ConfigureProblem.configure_new_variable(
            name,
            name_a,
            ocp,
            nlp,
            as_states,
            as_controls,
            as_states_dot,
        )

    @staticmethod
    def configure_time_state_force_no_cross_bridge(
        ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False
    ):
        """
        Configure a new variable for time constant of force decline at the absence of strongly bound cross-bridges (ms)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized coordinates should be a state
        as_controls: bool
            If the generalized coordinates should be a control
        as_states_dot: bool
            If the generalized velocities should be a state_dot
        """
        name = "Tau1"
        name_tau1 = [name]
        ConfigureProblem.configure_new_variable(
            name,
            name_tau1,
            ocp,
            nlp,
            as_states,
            as_controls,
            as_states_dot,
        )

    @staticmethod
    def configure_cross_bridges(ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False):
        """
        Configure a new variable for sensitivity of strongly bound cross-bridges to Cn (unitless)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized coordinates should be a state
        as_controls: bool
            If the generalized coordinates should be a control
        as_states_dot: bool
            If the generalized velocities should be a state_dot
        """
        name = "Km"
        name_km = [name]
        ConfigureProblem.configure_new_variable(
            name,
            name_km,
            ocp,
            nlp,
            as_states,
            as_controls,
            as_states_dot,
        )


class CustomDynamicsPulseDurationFrequency(CustomDynamicsFrequency):
    def __init__(self):
        super().__init__()

    @staticmethod
    def custom_dynamics(
        states: MX,
        controls: MX,
        parameters: MX,
        nlp: NonLinearProgram,
        t=None,
    ) -> DynamicsEvaluation:
        """
        Functional electrical stimulation dynamic

        Parameters
        ----------
        states: MX | SX
            The state of the system CN, F, A, Tau1, Km
        controls: MX | SX
            The controls of the system, none
        parameters: MX | SX
            The parameters acting on the system, final time of each phase
        nlp: NonLinearProgram
            A reference to the phase
        t: MX
            Current node time, this t is used to set the dynamics and as to be a symbolic
        Returns
        -------
        The derivative of the states in the tuple[MX | SX]] format
        """

        t_stim_prev = []  # Every stimulation instant before the current phase, i.e.: the beginning of each phase
        time_parameters = nlp.model.get_time_parameters(nlp.parameters)
        pulse_duration_parameters = nlp.model.get_pulse_duration_parameters(nlp.parameters)

        if time_parameters.shape[0] == 1:  # check if time is bimapped
            for i in range(nlp.phase_idx+1):
                t_stim_prev.append(time_parameters[0]*i)
        else:
            for i in range(nlp.phase_idx+1):
                t_stim_prev.append(sum1(time_parameters[0: i]))

        if pulse_duration_parameters.shape[0] == 1:  # check if pulse duration is bimapped
            impulse_time = pulse_duration_parameters[0]
        else:
            impulse_time = pulse_duration_parameters[nlp.phase_idx]

        return DynamicsEvaluation(
            dxdt=nlp.model.system_dynamics(
                cn=states[0],
                f=states[1],
                tau1=states[2],
                km=states[3],
                t=t,
                t_stim_prev=t_stim_prev,
                impulse_time=impulse_time,
                intensity_stim=None,
            ),
            defects=None,
        )

    @staticmethod
    def custom_configure_dynamics_function(ocp, nlp, **extra_params):
        """
        Configure the dynamics of the system

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        **extra_params:
            t: MX
                Current node time
        """

        nlp.parameters = ocp.v.parameters_in_list
        DynamicsFunctions.apply_parameters(nlp.parameters.mx, nlp)
        time_parameters = nlp.model.get_time_parameters(nlp.parameters)

        # Gets the t0 time for the current phase
        if 'time' not in ocp.parameter_mappings.keys():  # check if time is not bimapped
            CustomDynamicsPulseDurationFrequency.t0_phase_in_ocp = sum1(time_parameters[0: nlp.phase_idx])

        # Gets every time node for the current phase
        for i in range(nlp.ns):
            if 'time' in ocp.parameter_mappings.keys():
                t_node_in_phase = time_parameters[0] * nlp.phase_idx / (nlp.ns + 1) * i
                t_node_in_ocp = time_parameters[0] * nlp.phase_idx + t_node_in_phase
                extra_params["t"] = t_node_in_ocp
            else:
                t_node_in_phase = time_parameters[nlp.phase_idx] / (nlp.ns + 1) * i
                t_node_in_ocp = CustomDynamicsPulseDurationFrequency.t0_phase_in_ocp + t_node_in_phase
                extra_params["t"] = t_node_in_ocp

            dynamics_eval = CustomDynamicsPulseDurationFrequency.custom_dynamics(
                nlp.states["scaled"].mx_reduced, nlp.controls["scaled"].mx_reduced, nlp.parameters.mx, nlp,
                **extra_params
            )

            CustomDynamicsPulseDurationFrequency.dynamics_eval_horzcat = horzcat(dynamics_eval.dxdt) if i == 0 else horzcat(CustomDynamicsPulseDurationFrequency.dynamics_eval_horzcat, dynamics_eval.dxdt)

        nlp.dynamics_func = Function(
            "ForwardDyn",
            [nlp.states["scaled"].mx_reduced, nlp.controls["scaled"].mx_reduced, nlp.parameters.mx],
            [CustomDynamicsPulseDurationFrequency.dynamics_eval_horzcat],
            ["x", "u", "p"],
            ["xdot"],
        )

    @staticmethod
    def declare_ding_variables(ocp: OptimalControlProgram, nlp: NonLinearProgram):
        """
        Tell the program which variables are states and controls.
        The user is expected to use the ConfigureProblem.configure_xxx functions.
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """
        CustomDynamicsFrequency.configure_ca_troponin_complex(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        CustomDynamicsFrequency.configure_force(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        CustomDynamicsFrequency.configure_time_state_force_no_cross_bridge(ocp=ocp, nlp=nlp, as_states=True,
                                                                           as_controls=False)
        CustomDynamicsFrequency.configure_cross_bridges(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)

        t = MX.sym("t")  # t needs a symbolic value to start computing in custom_configure_dynamics_function

        CustomDynamicsPulseDurationFrequency.custom_configure_dynamics_function(ocp, nlp, t=t)


class CustomDynamicsIntensityFrequency(CustomDynamicsFrequency):
    def __init__(self):
        super().__init__()

    @staticmethod
    def custom_dynamics(
            states: MX | SX,
            controls: MX | SX,
            parameters: MX | SX,
            nlp: NonLinearProgram,
            t=None,
    ) -> DynamicsEvaluation:
        """
        Functional electrical stimulation dynamic

        Parameters
        ----------
        states: MX | SX
            The state of the system CN, F, A, Tau1, Km
        controls: MX | SX
            The controls of the system, none
        parameters: MX | SX
            The parameters acting on the system, final time of each phase
        nlp: NonLinearProgram
            A reference to the phase
        t: MX | SX
            Current node time, this t is used to set the dynamics and as to be a symbolic
        Returns
        -------
        The derivative of the states in the tuple[MX | SX]] format
        """
        t_stim_prev = []  # Every stimulation instant before the current phase, i.e.: the beginning of each phase
        intensity_stim_prev = []

        time_parameters = nlp.model.get_time_parameters(nlp.parameters)
        intensity_parameters = nlp.model.get_intensity_parameters(nlp.parameters)

        if time_parameters.shape[0] == 1:  # check if time is bimapped
            for i in range(nlp.phase_idx+1):
                t_stim_prev.append(time_parameters[0]*i)
        else:
            for i in range(nlp.phase_idx+1):
                t_stim_prev.append(sum1(time_parameters[0: i]))

        if intensity_parameters.shape[0] == 1:  # check if pulse duration is bimapped
            for i in range(nlp.phase_idx+1):
                intensity_stim_prev.append(intensity_parameters[0])
        else:
            for i in range(nlp.phase_idx+1):
                intensity_stim_prev.append(intensity_parameters[i])

        return DynamicsEvaluation(
            dxdt=nlp.model.system_dynamics(
                cn=states[0],
                f=states[1],
                a=states[2],
                tau1=states[3],
                km=states[4],
                t=t,
                t_stim_prev=t_stim_prev,
                impulse_time=None,
                intensity_stim=intensity_stim_prev
            ),
            defects=None,
        )

    @staticmethod
    def custom_configure_dynamics_function(ocp, nlp, **extra_params):
        """
        Configure the dynamics of the system

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        **extra_params:
            t: MX | SX
                Current node time
        """

        nlp.parameters = ocp.v.parameters_in_list
        DynamicsFunctions.apply_parameters(nlp.parameters.cx, nlp)

        # Gets every time node for the current phase
        for i in range(nlp.ns):
            extra_params["t"] = ocp.time(phase_idx=nlp.phase_idx, node_idx=i)

            dynamics_eval = CustomDynamicsIntensityFrequency.custom_dynamics(
                nlp.states["scaled"].cx, nlp.controls["scaled"].cx, nlp.parameters.cx, nlp,
                **extra_params
            )

            CustomDynamicsIntensityFrequency.dynamics_eval_horzcat = horzcat(
                dynamics_eval.dxdt) if i == 0 else horzcat(CustomDynamicsIntensityFrequency.dynamics_eval_horzcat,
                                                           dynamics_eval.dxdt)

        nlp.dynamics_func = Function(
            "ForwardDyn",
            [nlp.states["scaled"].cx, nlp.controls["scaled"].cx, nlp.parameters.cx],
            [CustomDynamicsIntensityFrequency.dynamics_eval_horzcat],
            ["x", "u", "p"],
            ["xdot"],
        )

    @staticmethod
    def declare_ding_variables(ocp: OptimalControlProgram, nlp: NonLinearProgram):
        """
        Tell the program which variables are states and controls.
        The user is expected to use the ConfigureProblem.configure_xxx functions.
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """
        CustomDynamicsFrequency.configure_ca_troponin_complex(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        CustomDynamicsFrequency.configure_force(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        CustomDynamicsFrequency.configure_scaling_factor(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)
        CustomDynamicsFrequency.configure_time_state_force_no_cross_bridge(ocp=ocp, nlp=nlp, as_states=True,
                                                                           as_controls=False)
        CustomDynamicsFrequency.configure_cross_bridges(ocp=ocp, nlp=nlp, as_states=True, as_controls=False)

        t = MX.sym("t") if nlp.cx.type_name() == 'MX' else SX.sym("t")  # t needs a symbolic value to start computing in custom_configure_dynamics_function

        CustomDynamicsIntensityFrequency.custom_configure_dynamics_function(ocp, nlp, t=t)





