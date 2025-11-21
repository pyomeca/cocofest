from typing import Callable
from casadi import MX, vertcat
import numpy as np

from cocofest.models.veltink1992.veltink1992 import VeltinkModelPulseIntensity


class VeltinkRienerModelPulseIntensityWithFatigue(VeltinkModelPulseIntensity):
    """
    Extension of VeltinkModelPulseIntensity that includes fatigue effects from:

    Veltink, P. H., Chizeck, H. J., Crago, P. E., & El-Bialy, A. (1992).
    Nonlinear joint angle control for artificially stimulated muscle.
    IEEE Transactions on Biomedical Engineering, 39(4), 368-380.

    Riener, R., & Veltink, P. H. (1998).
    A model of muscle fatigue during electrical stimulation.
    IEEE Transactions on Biomedical Engineering, 45(1), 105-113.
    """

    def __init__(
        self,
        model_name: str = "veltink_riener",
        muscle_name: str = None,
        Ta: float = None,
        I_threshold: float = None,
        I_saturation: float = None,
        mu_min: float = None,
        T_fat: float = None,
        T_rec: float = None,
    ):
        super().__init__(
            model_name=model_name,
            muscle_name=muscle_name,
            Ta=Ta,
            I_threshold=I_threshold,
            I_saturation=I_saturation,
        )
        self._with_fatigue = True

        # Default fatigue parameter values
        MU_MIN_DEFAULT = 0.2  # Minimum fatigue level (unitless)
        T_FAT_DEFAULT = 30.0  # Fatigue time constant (s)
        T_REC_DEFAULT = 50.0  # Recovery time constant (s)

        # Fatigue model parameters
        self.mu_min = mu_min if mu_min is not None else MU_MIN_DEFAULT
        self.T_fat = T_fat if T_fat is not None else T_FAT_DEFAULT
        self.T_rec = T_rec if T_rec is not None else T_REC_DEFAULT

    @property
    def name_dofs(self, with_muscle_name: bool = False) -> list[str]:
        muscle_name = ("_" + self.muscle_name if self.muscle_name is not None else "")
        return [
            "a" + muscle_name,  # Muscle activation
            "mu" + muscle_name,  # Fatigue state
        ]

    @property
    def nb_state(self) -> int:
        return 2

    @property
    def identifiable_parameters(self):
        params = super().identifiable_parameters
        params.update(
            {
                "mu_min": self.mu_min,
                "T_fat": self.T_fat,
                "T_rec": self.T_rec,
            }
        )
        return params

    def standard_rest_values(self) -> np.array:
        """
        Returns
        -------
        The rested values of muscle activation and fatigue state
        """
        return np.array([[0], [1]])  # a = 0, mu = 1 (no fatigue)

    def serialize(self) -> tuple[Callable, dict]:
        base_dict = super().serialize()[1]
        base_dict.update(
            {
                "mu_min": self.mu_min,
                "T_fat": self.T_fat,
                "T_rec": self.T_rec,
            }
        )
        return (VeltinkRienerModelPulseIntensityWithFatigue, base_dict)

    def get_mu_dot(self, a: MX, mu: MX) -> MX:
        """
        The fatigue dynamics.

        Parameters
        ----------
        a: MX
            Muscle activation state (unitless)
        mu: MX
            Fatigue state (unitless)

        Returns
        -------
        The fatigue state (unitless)
        """
        return ((self.mu_min - mu) * a) / self.T_fat + ((1 - mu) * (1 - a)) / self.T_rec

    def system_dynamics(
        self,
        time: MX,
        states: MX,
        controls: MX,
        numerical_timeseries: MX,
    ) -> MX:
        """
        The system dynamics including fatigue effects.

        Parameters
        ----------
        time: MX
            The system's current node time
        states: MX
            The state of the system a, mu
        controls: MX
            The controls of the system, I
        numerical_timeseries: MX
            The numerical timeseries of the system

        Returns
        -------
        The value of the derivative of each state
        """
        a = states[0]
        mu = states[1]
        I = controls[0]
        u = self.normalize_current(I)
        a_dot = self.get_muscle_activation(a=a, u=u)
        mu_dot = self.get_mu_dot(a=a, mu=mu)

        return vertcat(a_dot, mu_dot)
