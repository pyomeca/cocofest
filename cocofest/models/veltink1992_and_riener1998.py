from typing import Callable
from casadi import MX, vertcat
import numpy as np

from bioptim import (
    ConfigureProblem,
    DynamicsEvaluation,
    NonLinearProgram,
    OptimalControlProgram,
)

from .veltink1992 import VeltinkModel1992
from .state_configure import StateConfigure


class VeltinkRienerModel(VeltinkModel1992):
    """
    Extension of VeltinkModel1992 that includes fatigue effects from:
    
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
    def name_dof(self, with_muscle_name: bool = False) -> list[str]:
        muscle_name = "_" + self.muscle_name if self.muscle_name and with_muscle_name else ""
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
        params.update({
            "mu_min": self.mu_min,
            "T_fat": self.T_fat,
            "T_rec": self.T_rec,
        })
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
        base_dict.update({
            "mu_min": self.mu_min,
            "T_fat": self.T_fat,
            "T_rec": self.T_rec,
        })
        return (VeltinkRienerModel, base_dict)

    def system_dynamics(
        self,
        a: MX,
        mu: MX,
        I: MX = None,
    ) -> MX:
        """
        The system dynamics including fatigue effects.

        Parameters
        ----------
        a: MX
            Muscle activation state (unitless)
        mu: MX
            Fatigue state (unitless)
        I: MX
            Stimulation current amplitude (mA)

        Returns
        -------
        The value of the derivative of each state
        """
        # Get activation dynamics from parent model
        a_dot = super().system_dynamics(a, I)
        
        # Fatigue dynamics (equation 6)
        mu_dot = ((self.mu_min - mu) * a) / self.T_fat + ((1 - mu) * (1 - a)) / self.T_rec
        
        return vertcat(a_dot, mu_dot)

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
    ) -> DynamicsEvaluation:
        """
        Parameters
        ----------
        time: MX
            The system's current node time
        states: MX
            The state of the system (muscle activation, fatigue)
        controls: MX
            The controls of the system (stimulation current)
        parameters: MX
            The parameters acting on the system
        algebraic_states: MX
            The stochastic variables of the system
        numerical_timeseries: MX
            The numerical timeseries of the system
        nlp: NonLinearProgram
            A reference to the phase
        fes_model: VeltinkRienerModel
            The current phase fes model

        Returns
        -------
        The derivative of the states
        """
        model = fes_model if fes_model else nlp.model
        dxdt_fun = model.system_dynamics

        return DynamicsEvaluation(
            dxdt=dxdt_fun(
                a=states[0],
                mu=states[1],
                I=controls[0] if controls.shape[0] > 0 else 0.0,
            ),
            defects=None,
        )

    def declare_model_variables(
        self,
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        numerical_data_timeseries: dict[str, np.ndarray] = None,
    ):
        """
        Tell the program which variables are states and controls.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node
        """
        # TODO: Warning, not the same as ding, modify and check if their is an error
        StateConfigure().configure_all_fes_model_states(ocp, nlp, fes_model=self)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, dyn_func=self.dynamics) 