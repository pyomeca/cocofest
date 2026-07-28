"""
Abstract interface every FES model must implement to be usable within bioptim.
"""

from abc import ABC, abstractmethod

from casadi import MX
from bioptim import NonLinearProgram


class FesModel(ABC):
    """
    Abstract base class defining the interface every FES model (Ding2003, Ding2007, Hmed2018, Marion2009/2013,
    Veltink1992, ...) must implement to be usable within bioptim.
    """

    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.stim_time = None
        self.previous_stim = None
        self._name = name

    @property
    def name(self):
        """The model's name."""
        return self._name

    @abstractmethod
    def set_a_rest(self, model, a_rest: MX | float):
        """
        Set the rest value of the force-scaling parameter A.

        Parameters
        ----------
        model
            The model to update (required by bioptim's parameter callback signature)
        a_rest: MX | float
            The new rest value of A
        """

    @abstractmethod
    def set_km_rest(self, model, km_rest: MX | float):
        """
        Set the rest value of the cross-bridges sensitivity parameter Km.

        Parameters
        ----------
        model
            The model to update (required by bioptim's parameter callback signature)
        km_rest: MX | float
            The new rest value of Km
        """

    @abstractmethod
    def set_tau1_rest(self, model, tau1_rest: MX | float):
        """
        Set the rest value of the force decline time constant Tau1.

        Parameters
        ----------
        model
            The model to update (required by bioptim's parameter callback signature)
        tau1_rest: MX | float
            The new rest value of Tau1
        """

    @abstractmethod
    def set_tau2(self, model, tau2: MX | float):
        """
        Set the time constant of force decline due to cross-bridges Tau2.

        Parameters
        ----------
        model
            The model to update (required by bioptim's parameter callback signature)
        tau2: MX | float
            The new value of Tau2
        """

    @abstractmethod
    def standard_rest_values(self):
        """
        The model's states at rest.

        Returns
        -------
        The rested (initial) values of the model's states
        """

    @abstractmethod
    def serialize(self):
        """
        Serialize the model's parameters for later saving/reloading.

        Returns
        -------
        A tuple of the model's class and a dict of its parameters, used to save/reload the model
        """

    @abstractmethod
    def nb_state(self):
        """
        The number of states of the model.

        Returns
        -------
        int
            The number of states of the model
        """

    @abstractmethod
    def model_name(self):
        """
        The model's name.

        Returns
        -------
        str
            The model's name
        """

    @abstractmethod
    def muscle_name(self):
        """
        The muscle's name.

        Returns
        -------
        str
            The muscle's name
        """

    @abstractmethod
    def with_fatigue(self):
        """
        If the model includes fatigue dynamics.

        Returns
        -------
        bool
            If the model includes fatigue dynamics
        """

    @abstractmethod
    def system_dynamics(
        self,
        time: MX,
        states: MX,
        controls: MX,
        numerical_timeseries: MX,
    ):
        """
        The function describing the model's dynamics.

        Parameters
        ----------
        time: MX
            The system's current node time
        states: MX
            The state of the system
        controls: MX
            The controls of the system
        numerical_timeseries: MX
            The numerical timeseries of the system

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """

    @abstractmethod
    def exp_time_fun(self, t: MX, t_stim_i: MX):
        """
        Compute the exponential decay term of the calcium-troponin complex since a given stimulation.

        Parameters
        ----------
        t: MX
            The current time at which the dynamics is evaluated (s)
        t_stim_i: MX
            Time when the stimulation i occurred (s)

        Returns
        -------
        A part of the Cn (calcium-troponin complex) equation
        """

    @abstractmethod
    def ri_fun(self, r0: MX | float, time_between_stim: MX):
        """
        Compute the magnitude of enhancement in the calcium-troponin complex from the following stimuli.

        Parameters
        ----------
        r0: MX | float
            Mathematical term characterizing the magnitude of enhancement in Cn from the following stimuli (unitless)
        time_between_stim: MX
            Time between the last stimulation i and the current stimulation i (s)

        Returns
        -------
        A part of the Cn (calcium-troponin complex) equation
        """

    @abstractmethod
    def cn_sum_fun(self, r0: MX | float, t: MX, t_stim_prev: list[MX], lambda_i: list[MX]):
        """
        Compute the calcium-troponin complex summation over the previous stimulations.

        Parameters
        ----------
        r0: MX | float
            Mathematical term characterizing the magnitude of enhancement in Cn from the following stimuli (unitless)
        t: MX
            The current time at which the dynamics is evaluated (s)
        t_stim_prev: list[MX]
            The time list of the previous stimulations (s)
        lambda_i: list[MX]
            A list of force-pulse amplitude relationship (unitless)

        Returns
        -------
        The calcium-troponin complex summation over the previous stimulations
        """

    @abstractmethod
    def cn_dot_fun(self, cn: MX, cn_sum: MX):
        """
        Compute the derivative of the calcium-troponin complex.

        Parameters
        ----------
        cn: MX
            The previous step value of the calcium-troponin complex (unitless)
        cn_sum: MX
            The previous calculated calcium-troponin complex summation

        Returns
        -------
        The value of the derivative of the calcium-troponin complex (unitless)
        """

    @abstractmethod
    def f_dot_fun(
        self,
        cn: MX,
        f: MX,
        a: MX | float,
        tau1: MX | float,
        km: MX | float,
    ):
        """
        Compute the derivative of the force.

        Parameters
        ----------
        cn: MX
            The current value of the calcium-troponin complex (unitless)
        f: MX
            The current force value (N)
        a: MX | float
            The current force-scaling factor
        tau1: MX | float
            The current force decline time constant
        km: MX | float
            The current cross-bridges sensitivity to Cn

        Returns
        -------
        The value of the derivative of the force (N/s)
        """

    @abstractmethod
    def dynamics(
        self,
        time: MX,
        states: MX,
        controls: MX,
        parameters: MX,
        algebraic_states: MX,
        numerical_timeseries: MX,
        nlp: NonLinearProgram,
    ):
        """
        The bioptim-compatible wrapper around system_dynamics, used to declare the model's dynamics to the ocp.

        Parameters
        ----------
        time: MX
            The system's current node time
        states: MX
            The state of the system
        controls: MX
            The controls of the system
        parameters: MX
            The parameters of the system
        algebraic_states: MX
            The algebraic states of the system
        numerical_timeseries: MX
            The numerical timeseries of the system
        nlp: NonLinearProgram
            A reference to the phase

        Returns
        -------
        DynamicsEvaluation
            The derivative of the states
        """

    @abstractmethod
    def get_numerical_data_time_series(self, total_cycle_len, total_cycle_duration):
        """
        Build the numerical time series data used by the dynamics.

        Parameters
        ----------
        total_cycle_len
            The number of shooting points of the full cycle
        total_cycle_duration
            The duration of the full cycle

        Returns
        -------
        The numerical time series data used by the dynamics (e.g. previous stimulation times)
        """

    @abstractmethod
    def get_n_shooting(self, final_time):
        """
        Prepare the n_shooting for the ocp in order to have a time step that is a multiple of the stimulation time.

        Parameters
        ----------
        final_time
            The final time of the ocp

        Returns
        -------
        int
            The number of shooting points matching the stimulation times
        """
