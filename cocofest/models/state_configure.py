from bioptim import (
    ConfigureProblem,
    NonLinearProgram,
    OptimalControlProgram,
)


class StateConfigure:
    def __init__(self):
        self.state_dictionary = {
            "Cn": self.configure_ca_troponin_complex,
            "F": self.configure_force,
            "A": self.configure_scaling_factor,
            "Tau1": self.configure_time_state_force_no_cross_bridge,
            "Km": self.configure_cross_bridges,
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
        return ConfigureProblem.configure_new_variable(
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
        return ConfigureProblem.configure_new_variable(
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
        return ConfigureProblem.configure_new_variable(
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
        return ConfigureProblem.configure_new_variable(name, name_cn_sum, ocp, nlp, as_states=False, as_controls=True)

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
        return ConfigureProblem.configure_new_variable(name, name_cn_sum, ocp, nlp, as_states=False, as_controls=True)

    @staticmethod
    def configure_last_pulse_width(ocp, nlp, muscle_name: str = None):
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
        return ConfigureProblem.configure_new_variable(name, last_pulse_width, ocp, nlp, as_states=False, as_controls=True)

    @staticmethod
    def configure_pulse_intensity(ocp, nlp, muscle_name: str = None, truncation: int = 20):
        """
        Configure the pulse intensity control

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """
        muscle_name = "_" + muscle_name if muscle_name else ""
        name = "pulse_intensity" + muscle_name
        pulse_intensity = [str(i) for i in range(truncation)]
        return ConfigureProblem.configure_new_variable(name, pulse_intensity, ocp, nlp, as_states=False, as_controls=True)

    def configure_all_muscle_states(self, muscles_dynamics_model, ocp, nlp):
        state_name_list = []
        for muscle_dynamics_model in muscles_dynamics_model:
            for state_key in muscle_dynamics_model.name_dof:
                if state_key in self.state_dictionary.keys():
                    self.state_dictionary[state_key](
                        ocp=ocp,
                        nlp=nlp,
                        as_states=True,
                        as_controls=False,
                        muscle_name=muscle_dynamics_model.muscle_name,
                    )
                    state_name_list.append(state_key + "_" + muscle_dynamics_model.muscle_name)

        return state_name_list

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
