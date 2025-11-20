from bioptim import (
    ConfigureVariables,
    NonLinearProgram,
    OptimalControlProgram,
)


class StateConfigure:
    def __init__(self):
        self.state_dictionary = {
            "Cn": self.configure_ca_troponin_complex,  # Ding model
            "F": self.configure_force,  # Ding model
            "A": self.configure_scaling_factor,  # Ding model
            "Tau1": self.configure_time_state_force_no_cross_bridge,  # Ding model
            "Km": self.configure_cross_bridges,  # Ding model
            "a": self.configure_muscle_activation,  # Veltink model
            "mu": self.configure_fatigue_state,  # Veltink model
            "theta": self.configure_angle,  # Marion model
            "dtheta_dt": self.configure_angular_velocity,  # Marion model
        }

    @staticmethod
    def configure_ca_troponin_complex(
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        as_states: bool,
        as_controls: bool,
        as_states_dot: bool = False,
        muscle_name: str = None,
    ):
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
        muscle_name: str
            The muscle name
        """
        muscle_name = "_" + muscle_name if muscle_name else ""
        name = "Cn" + muscle_name
        name_cn = [name]
        ConfigureVariables.configure_new_variable(
            name,
            name_cn,
            ocp,
            nlp,
            as_states,
            as_controls,
            as_states_dot,
        )

    @staticmethod
    def configure_force(
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        as_states: bool,
        as_controls: bool,
        as_states_dot: bool = False,
        muscle_name: str = None,
    ):
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
        muscle_name: str
            The muscle name
        """
        muscle_name = "_" + muscle_name if muscle_name else ""
        name = "F" + muscle_name
        name_f = [name]
        ConfigureVariables.configure_new_variable(
            name,
            name_f,
            ocp,
            nlp,
            as_states,
            as_controls,
            as_states_dot,
        )

    @staticmethod
    def configure_scaling_factor(
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        as_states: bool,
        as_controls: bool,
        as_states_dot: bool = False,
        muscle_name: str = None,
    ):
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
        muscle_name: str
            The muscle name
        """
        muscle_name = "_" + muscle_name if muscle_name else ""
        name = "A" + muscle_name
        name_a = [name]
        return ConfigureVariables.configure_new_variable(
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
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        as_states: bool,
        as_controls: bool,
        as_states_dot: bool = False,
        muscle_name: str = None,
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
        muscle_name: str
            The muscle name
        """
        muscle_name = "_" + muscle_name if muscle_name else ""
        name = "Tau1" + muscle_name
        name_tau1 = [name]
        return ConfigureVariables.configure_new_variable(
            name,
            name_tau1,
            ocp,
            nlp,
            as_states,
            as_controls,
            as_states_dot,
        )

    @staticmethod
    def configure_cross_bridges(
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        as_states: bool,
        as_controls: bool,
        as_states_dot: bool = False,
        muscle_name: str = None,
    ):
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
        muscle_name: str
            The muscle name
        """
        muscle_name = "_" + muscle_name if muscle_name else ""
        name = "Km" + muscle_name
        name_km = [name]
        return ConfigureVariables.configure_new_variable(
            name,
            name_km,
            ocp,
            nlp,
            as_states,
            as_controls,
            as_states_dot,
        )

    @staticmethod
    def configure_cn_sum(ocp, nlp, muscle_name: str = None):
        """
        Configure the calcium summation control

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        muscle_name: str
            The muscle name
        """
        muscle_name = "_" + muscle_name if muscle_name else ""
        name = "Cn_sum" + muscle_name
        name_cn_sum = [name]
        return ConfigureVariables.configure_new_variable(name, name_cn_sum, ocp, nlp, as_states=False, as_controls=True)

    @staticmethod
    def configure_a_calculation(ocp, nlp, muscle_name: str = None):
        """
        Configure the force scaling factor calculation

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        muscle_name: str
            The muscle name
        """
        muscle_name = "_" + muscle_name if muscle_name else ""
        name = "A_calculation" + muscle_name
        name_cn_sum = [name]
        return ConfigureVariables.configure_new_variable(name, name_cn_sum, ocp, nlp, as_states=False, as_controls=True)

    @staticmethod
    def configure_muscle_activation(
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        as_states: bool,
        as_controls: bool,
        as_states_dot: bool = False,
        muscle_name: str = None,
    ):
        """
        Configure a new variable for muscle activation (unitless)

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
        muscle_name: str
            The muscle name
        """
        muscle_name = "_" + muscle_name if muscle_name else ""
        name = "a" + muscle_name
        name_a = [name]
        ConfigureVariables.configure_new_variable(
            name,
            name_a,
            ocp,
            nlp,
            as_states,
            as_controls,
            as_states_dot,
        )

    @staticmethod
    def configure_fatigue_state(
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        as_states: bool,
        as_controls: bool,
        as_states_dot: bool = False,
        muscle_name: str = None,
    ):
        """
        Configure a new variable for fatigue state (unitless)

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
        muscle_name: str
            The muscle name
        """
        muscle_name = "_" + muscle_name if muscle_name else ""
        name = "mu" + muscle_name
        name_mu = [name]
        ConfigureVariables.configure_new_variable(
            name,
            name_mu,
            ocp,
            nlp,
            as_states,
            as_controls,
            as_states_dot,
        )

    @staticmethod
    def configure_angle(
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        as_states: bool,
        as_controls: bool,
        as_states_dot: bool = False,
        muscle_name: str = None,
    ):
        """
        Configure a new variable for angle (rad)

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
        muscle_name: str
            The muscle name
        """
        muscle_name = "_" + muscle_name if muscle_name else ""
        name = "theta" + muscle_name
        name_theta = [name]
        ConfigureVariables.configure_new_variable(
            name,
            name_theta,
            ocp,
            nlp,
            as_states,
            as_controls,
            as_states_dot,
        )

    @staticmethod
    def configure_angular_velocity(
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        as_states: bool,
        as_controls: bool,
        as_states_dot: bool = False,
        muscle_name: str = None,
    ):
        """
        Configure a new variable for angular velocity (rad/s)

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
        muscle_name: str
            The muscle name
        """
        muscle_name = "_" + muscle_name if muscle_name else ""
        name = "dtheta_dt" + muscle_name
        name_dtheta_dt = [name]
        ConfigureVariables.configure_new_variable(
            name,
            name_dtheta_dt,
            ocp,
            nlp,
            as_states,
            as_controls,
            as_states_dot,
        )

    @staticmethod
    def configure_last_pulse_width(ocp, nlp, muscle_name: str = None, as_states=True, as_controls=False, as_algebraic_states=False):
        """
        Configure the last pulse width control

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """
        muscle_name = "_" + muscle_name if muscle_name else ""
        name = "last_pulse_width" + muscle_name
        last_pulse_width = [name]
        return ConfigureVariables.configure_new_variable(
            name, last_pulse_width, ocp, nlp, as_states=False, as_controls=True
        )

    @staticmethod
    def configure_pulse_intensity(ocp, nlp, muscle_name: str = None, truncation: int = None, as_states=True, as_controls=False, as_algebraic_states=False):
        """
        Configure the pulse intensity control for the Ding model

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """
        muscle_name = nlp.model.muscle_name if muscle_name is None else muscle_name
        truncation = nlp.model.sum_stim_truncation if truncation is None else truncation
        muscle_name = "_" + muscle_name if muscle_name else ""
        name = "pulse_intensity" + muscle_name
        pulse_intensity = [str(i) for i in range(truncation)]
        return ConfigureVariables.configure_new_variable(
            name, pulse_intensity, ocp, nlp, as_states=False, as_controls=True
        )

    @staticmethod
    def configure_intensity(ocp, nlp, as_states=False, as_controls=True, as_algebraic_states=False):
        """
        Configure the intensity control for the Veltink1992 model

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """
        muscle_name = nlp.model.muscle_name
        muscle_name = "_" + muscle_name if muscle_name else ""
        name = "I" + muscle_name
        pulse_intensity = [name]
        return ConfigureVariables.configure_new_variable(
            name, pulse_intensity, ocp, nlp, as_states=False, as_controls=True
        )

    @staticmethod
    def configure_all_muscle_states(ocp, nlp, as_states=True, as_controls=False, as_algebraic_states=False):
        for state_key in nlp.model.name_dof:
            if state_key in StateConfigure().state_dictionary.keys():
                StateConfigure().state_dictionary[state_key](
                    ocp=ocp,
                    nlp=nlp,
                    as_states=True,
                    as_controls=False,
                    muscle_name=nlp.model.muscle_name,
                )

    @staticmethod
    def configure_all_muscle_msk_states(ocp, nlp, as_states=True, as_controls=False, as_algebraic_states=False):
        for muscle_dynamics_model in nlp.model.muscles_dynamics_model:
            for state_key in muscle_dynamics_model.name_dof:
                if state_key in StateConfigure().state_dictionary.keys():
                    StateConfigure().state_dictionary[state_key](
                        ocp=ocp,
                        nlp=nlp,
                        as_states=True,
                        as_controls=False,
                        muscle_name=muscle_dynamics_model.muscle_name,
                    )

    def configure_all_fes_model_states(self, ocp, nlp, fes_model):
        for state_key in fes_model.name_dof:
            if state_key in self.state_dictionary.keys():
                self.state_dictionary[state_key](
                    ocp=ocp,
                    nlp=nlp,
                    as_states=True,
                    as_controls=False,
                    muscle_name=fes_model.muscle_name,
                )
