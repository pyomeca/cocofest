from abc import ABC, abstractmethod
import numpy as np

from casadi import MX
from bioptim import NonLinearProgram, OptimalControlProgram


class FesModel(ABC):
    def __init__(self):
        self.stim_time = None

    @abstractmethod
    def set_a_rest(self, model, a_rest: MX | float):
        """

        Returns
        -------

        """

    @abstractmethod
    def set_km_rest(self, model, km_rest: MX | float):
        """

        Returns
        -------

        """

    @abstractmethod
    def set_tau1_rest(self, model, tau1_rest: MX | float):
        """

        Returns
        -------

        """

    @abstractmethod
    def set_tau2(self, model, tau2: MX | float):
        """

        Returns
        -------

        """

    @abstractmethod
    def standard_rest_values(self):
        """

        Returns
        -------

        """

    @abstractmethod
    def serialize(self):
        """

        Returns
        -------

        """

    @abstractmethod
    def name_dof(self):
        """

        Returns
        -------

        """

    @abstractmethod
    def nb_state(self):
        """

        Returns
        -------

        """

    @abstractmethod
    def model_name(self):
        """

        Returns
        -------

        """

    @abstractmethod
    def muscle_name(self):
        """

        Returns
        -------

        """

    @abstractmethod
    def with_fatigue(self):
        """

        Returns
        -------

        """

    @abstractmethod
    def system_dynamics(
        self,
        cn: MX,
        f: MX,
        cn_sum: MX,
        force_length_relationship: MX | float,
        force_velocity_relationship: MX | float,
        passive_force_relationship: MX | float,
    ):
        """

        Returns
        -------

        """

    @abstractmethod
    def exp_time_fun(self, t: MX, t_stim_i: MX):
        """

        Returns
        -------

        """

    @abstractmethod
    def ri_fun(self, r0: MX | float, time_between_stim: MX):
        """

        Returns
        -------

        """

    @abstractmethod
    def cn_sum_fun(self, r0: MX | float, t: MX, t_stim_prev: list[MX], lambda_i: list[MX]):
        """
        Returns
        -------

        """

    @abstractmethod
    def cn_dot_fun(self, cn: MX, cn_sum: MX):
        """

        Returns
        -------

        """

    @abstractmethod
    def f_dot_fun(
        self,
        cn: MX,
        f: MX,
        a: MX | float,
        tau1: MX | float,
        km: MX | float,
        force_length_relationship: MX | float,
        force_velocity_relationship: MX | float,
        passive_force_relationship: MX | float,
    ):
        """

        Returns
        -------

        """

    @staticmethod
    @abstractmethod
    def dynamics(
        time: MX,
        states: MX,
        controls: MX,
        parameters: MX,
        algebraic_states: MX,
        numerical_data_timeseries: MX,
        nlp: NonLinearProgram,
        fes_model,
        force_length_relationship: MX | float,
        force_velocity_relationship: MX | float,
        passive_force_relationship: MX | float,
    ):
        """

        Returns
        -------

        """

    @abstractmethod
    def declare_ding_variables(
        self,
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        numerical_data_timeseries: dict[str, np.ndarray] = None,
        contact_type: tuple = (),
    ):
        """

        Returns
        -------

        """

    @abstractmethod
    def get_numerical_data_time_series(self, total_cycle_len, total_cycle_duration):
        """

        Returns
        -------

        """

    @abstractmethod
    def get_n_shooting(self, final_time):
        """

        Returns
        -------

        """
