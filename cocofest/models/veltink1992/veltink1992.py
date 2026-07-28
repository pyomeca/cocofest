"""
Veltink1992 model: muscle activation dynamics driven by stimulation intensity.

Veltink, P. H., Chizeck, H. J., Crago, P. E., & El-Bialy, A. (1992). Nonlinear joint angle control
for artificially stimulated muscle. IEEE Transactions on Biomedical Engineering, 39(4), 368-380.
"""

from __future__ import annotations

from typing import Callable, List
from casadi import MX, vertcat
import numpy as np

from bioptim import StateDynamics, States, OdeSolver, DynamicsEvaluation, NonLinearProgram, DynamicsFunctions

from cocofest.models.state_configure import StateConfigure
from cocofest.models.fes_model import FesModel


class VeltinkModelPulseIntensity(StateDynamics):
    """
    This is a custom model implementing the muscle activation dynamics from:

    Veltink, P. H., Chizeck, H. J., Crago, P. E., & El-Bialy, A. (1992).
    Nonlinear joint angle control for artificially stimulated muscle.
    IEEE Transactions on Biomedical Engineering, 39(4), 368-380.
    """

    def __init__(
        self,
        model_name: str = "veltink_1992",
        muscle_name: str = None,
        Ta: float = None,
        I_threshold: float = None,
        I_saturation: float = None,
    ):
        super().__init__()
        self._model_name = model_name
        self._muscle_name = muscle_name
        self._with_fatigue = False

        # Default values
        TA_DEFAULT = 0.26  # Activation time constant (s)
        I_THRESHOLD_DEFAULT = 20.0  # Threshold current (mA)
        I_SATURATION_DEFAULT = 60.0  # Saturation current (mA)

        # Model parameters
        self.Ta = Ta if Ta is not None else TA_DEFAULT
        self.I_threshold = I_threshold if I_threshold is not None else I_THRESHOLD_DEFAULT
        self.I_saturation = I_saturation if I_saturation is not None else I_SATURATION_DEFAULT

        self.contact_types = ()

    @property
    def name(self):
        """The model's name."""
        return self._name

    # --- Configure variables --- #
    @property
    def state_configuration_functions(self) -> List[States | Callable]:
        """The state configuration functions used to declare the muscle activation state to bioptim."""
        return [StateConfigure().configure_all_muscle_states]

    @property
    def control_configuration_functions(self) -> List[States | Callable]:
        """The control configuration functions used to declare the stimulation intensity control to bioptim."""
        return [StateConfigure().configure_intensity]

    @property
    def algebraic_configuration_functions(self) -> List[States | Callable]:
        """The algebraic state configuration functions used to declare the model's algebraic states to bioptim (none)."""
        return []

    @property
    def extra_configuration_functions(self) -> List[States | Callable]:
        """Extra configuration functions used to declare additional variables to bioptim (none)."""
        return []

    @property
    def name_dofs(self, with_muscle_name: bool = False) -> list[str]:
        """The state's name (a), suffixed with the muscle name if any."""
        muscle_name = "_" + self.muscle_name if self.muscle_name is not None else ""
        return ["a" + muscle_name]  # Only muscle activation state

    @property
    def nb_state(self) -> int:
        """The number of states of the model (a)."""
        return 1

    @property
    def model_name(self) -> None | str:
        """The model's name."""
        return self._model_name

    @property
    def muscle_name(self) -> None | str:
        """The muscle's name."""
        return self._muscle_name

    @property
    def with_fatigue(self):
        """If the model includes fatigue dynamics (False for this model)."""
        return self._with_fatigue

    @property
    def identifiable_parameters(self):
        """The model's parameters that can be identified from experimental data."""
        return {
            "Ta": self.Ta,
            "I_threshold": self.I_threshold,
            "I_saturation": self.I_saturation,
        }

    def standard_rest_values(self) -> np.array:
        """
        The model's state at rest.

        Returns
        -------
        The rested value of muscle activation
        """
        return np.array([[0]])

    def serialize(self) -> tuple[Callable, dict]:
        """
        Serialize the model's parameters for later saving/reloading.

        Returns
        -------
        A tuple of the model's class and a dict of its parameters, used to save/reload the model
        """
        return (
            VeltinkModelPulseIntensity,
            {
                "Ta": self.Ta,
                "I_threshold": self.I_threshold,
                "I_saturation": self.I_saturation,
            },
        )

    def normalize_current(self, I: MX) -> MX:
        """
        Normalize stimulation current according to equation (5)

        Parameters
        ----------
        I: MX
            Stimulation current amplitude (mA)

        Returns
        -------
        Normalized stimulation between 0 and 1
        """
        # Piecewise function for current normalization
        u = (I - self.I_threshold) / (self.I_saturation - self.I_threshold)

        return u

    def get_muscle_activation(self, a: MX, u: MX) -> MX:
        """
        Get the muscle activation from the state variable.

        Parameters
        ----------
        a: MX
            Muscle activation state (unitless)
        u: MX
            Normalized stimulation (unitless)

        Returns
        -------
        The muscle activation value
        """
        return (-a + u) / self.Ta

    def system_dynamics(
        self,
        time: MX,
        states: MX,
        controls: MX,
        numerical_timeseries: MX,
    ) -> MX:
        """
        The system dynamics implementing equation (4) for muscle activation.

        Parameters
        ----------
        time: MX
            The system's current node time
        states: MX
            The state of the system a
        controls: MX
            The controls of the system, I
        numerical_timeseries: MX
            The numerical timeseries of the system

        Returns
        -------
        The derivative of muscle activation state
        """
        a = states[0]
        I = controls[0]
        u = self.normalize_current(I)
        a_dot = self.get_muscle_activation(a=a, u=u)

        return vertcat(a_dot)

    def dynamics(
        self,
        time: MX,
        states: MX,
        controls: MX,
        parameters: MX,
        algebraic_states: MX,
        numerical_timeseries: MX,
        nlp: NonLinearProgram,
    ) -> DynamicsEvaluation:
        """
        Functional electrical stimulation dynamic

        Parameters
        ----------
        time: MX
            The system's current node time
        states: MX
            The state of the system CN, F, A, Tau1, Km
        controls: MX
            The controls of the system, none
        parameters: MX
            The parameters acting on the system, final time of each phase
        algebraic_states: MX
            The stochastic variables of the system, none
        numerical_timeseries: MX
            The numerical timeseries of the system
        nlp: NonLinearProgram
            A reference to the phase
        Returns
        -------
        The derivative of the states in the tuple[MX] format
        """
        dxdt_fun = nlp.model.system_dynamics
        dxdt = dxdt_fun(
            time=time,
            states=states,
            controls=controls,
            numerical_timeseries=numerical_timeseries,
        )

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):
            states_dot_list = []
            for key in nlp.model.name_dofs:
                states_dot_list.append(DynamicsFunctions.get(nlp.states_dot[key], nlp.states_dot.scaled.cx))
            defects = vertcat(*states_dot_list) - dxdt

        return DynamicsEvaluation(
            dxdt=dxdt,
            defects=defects,
        )
