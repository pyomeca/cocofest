from abc import ABC, abstractmethod


class ParameterIdentification(ABC):
    @abstractmethod
    def _set_default_values(self, model):
        """

        Returns
        -------

        """

    @abstractmethod
    def _set_default_parameters_list(self):
        """

        Returns
        -------

        """

    @abstractmethod
    def input_sanity(
        self,
        model,
        data_path,
        identification_method,
        double_step_identification,
        key_parameter_to_identify,
        additional_key_settings,
    ):
        """

        Returns
        -------

        """

    @abstractmethod
    def key_setting_to_dictionary(self, key_settings):
        """

        Returns
        -------

        """

    @staticmethod
    @abstractmethod
    def check_experiment_force_format(data_path):
        """

        Returns
        -------

        """

    @staticmethod
    @abstractmethod
    def _set_model_parameters(model, model_parameters_value):
        """

        Returns
        -------

        """

    @abstractmethod
    def _force_model_identification_for_initial_guess(self):
        """

        Returns
        -------

        """

    @abstractmethod
    def force_model_identification(self):
        """

        Returns
        -------

        """
