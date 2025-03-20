import numpy as np
from bioptim import (
    ControlType,
    DynamicsList,
    InitialGuessList,
    OdeSolver,
    OptimalControlProgram,
    ParameterList,
    PhaseDynamics,
    BoundsList,
    InterpolationType,
    VariableScaling,
    Solution,
    Shooting,
    SolutionIntegrator,
    SolutionMerge,
)

from ..optimization.fes_ocp import OcpFes
from ..models.fes_model import FesModel
from ..models.ding2003 import DingModelFrequency
from ..models.ding2007 import DingModelPulseWidthFrequency
from ..models.ding2007_with_fatigue import DingModelPulseWidthFrequencyWithFatigue
from ..models.hmed2018 import DingModelPulseIntensityFrequency
from ..models.hmed2018_with_fatigue import DingModelPulseIntensityFrequencyWithFatigue


class IvpFes:
    """
    The main class to define an ivp. This class prepares the ivp and gives all
    the needed parameters to integrate a functional electrical stimulation problem.

    Methods
    -------
    from_frequency_and_final_time(self, frequency: int | float, final_time: float, round_down: bool)
        Calculates the number of stim (phases) for the ocp from frequency and final time
    from_frequency_and_n_stim(self, frequency: int | float, n_stim: int)
        Calculates the final ocp time from frequency and stimulation number
    """

    def __init__(
        self,
        fes_parameters: dict = None,
        ivp_parameters: dict = None,
    ):
        """
        Enables the creation of an ivp problem

        Parameters
        ----------
        fes_parameters: dict
            The parameters for the fes configuration including :
            model (FesModel type), stim_time (list), pulse_width (float type), pulse_intensity (int | float type), pulse_mode (str type), frequency (int | float type), round_down (bool type)
        ivp_parameters: dict
            The parameters for the ivp problem including :
            final_time (int | float type), ode_solver (OdeSolver type), n_threads (int type)
        """

        self._fill_fes_dict(fes_parameters)
        self._fill_ivp_dict(ivp_parameters)
        self.dictionaries_check()

        self.model = self.fes_parameters["model"]
        self.stim_time = self.model.stim_time
        self.n_stim = len(self.stim_time)
        self.pulse_width = self.fes_parameters["pulse_width"]
        self.pulse_intensity = self.fes_parameters["pulse_intensity"]

        self.parameter_mappings = None
        self.parameters = None

        self.final_time = self.ivp_parameters["final_time"]
        self.n_shooting = self.model.get_n_shooting(self.final_time)

        self.pulse_mode = self.fes_parameters["pulse_mode"]
        self._pulse_mode_settings()  # Update stim_time and n_stim, n_shooting can also be updated depending on the mode
        self.dt = np.array([self.final_time / self.n_shooting])

        parameters = ParameterList(use_sx=False)
        parameters_init = InitialGuessList()
        parameters_bounds = BoundsList()

        self.controls_keys = None
        if isinstance(self.model, DingModelPulseWidthFrequency):
            self.controls_keys = ["last_pulse_width"]
        if isinstance(self.model, DingModelPulseIntensityFrequency):
            self.controls_keys = ["pulse_intensity"]

        self.parameters = parameters
        self.parameters_init = parameters_init
        self.parameters_bounds = parameters_bounds

        numerical_data_time_series, stim_idx_at_node_list = self.model.get_numerical_data_time_series(
            self.n_shooting, self.final_time
        )

        self._declare_dynamics(numerical_data_time_series)

        (
            self.x_init,
            self.u_init,
            self.p_init,
            self.s_init,
        ) = self.build_initial_guess_from_ocp(self, stim_idx_at_node_list=stim_idx_at_node_list)

        self.ode_solver = self.ivp_parameters["ode_solver"]
        self.use_sx = False
        self.n_threads = self.ivp_parameters["n_threads"]

        self.fake_ocp = self._prepare_fake_ocp()
        self.initial_guess_solution = self._build_solution_from_initial_guess()

    def _fill_fes_dict(self, fes_parameters: dict):
        """
        Parameters
        ----------
        fes_parameters : dict
            Contains FES parameters

        Returns
        -------
        A dictionary with all needed FES parameters such as model, stimulation time, pulse width, intensity and mode (if not specified, default parameter is used)
        """

        default_fes_dict = {
            "model": FesModel,
            "stim_time": None,
            "pulse_width": 0.0003,
            "pulse_intensity": 50,
            "pulse_mode": "single",
        }

        if fes_parameters is None:
            fes_parameters = {}

        for key in default_fes_dict:
            if key not in fes_parameters:
                fes_parameters[key] = default_fes_dict[key]

        self.fes_parameters = fes_parameters

    def _fill_ivp_dict(self, ivp_parameters: dict):
        """

        Parameters
        ----------
        ivp_parameters: dict
            Contains IVP parameters

        Returns
        -------
        A dictionary with all needed IVP parameters such as final time, the solver used and the number of threads (if not specified, default parameter is used)

        """
        default_ivp_dict = {
            "final_time": None,
            "ode_solver": OdeSolver.RK4(n_integration_steps=10),
            "n_threads": 1,
        }

        if ivp_parameters is None:
            ivp_parameters = {}

        for key in default_ivp_dict:
            if key not in ivp_parameters:
                ivp_parameters[key] = default_ivp_dict[key]

        self.ivp_parameters = ivp_parameters

    def dictionaries_check(self):
        if not isinstance(self.fes_parameters, dict):
            raise ValueError("fes_parameters must be a dictionary")

        if not isinstance(self.ivp_parameters, dict):
            raise ValueError("ivp_parameters must be a dictionary")

        if not isinstance(self.fes_parameters["model"], FesModel):
            raise TypeError("model must be a FesModel type")

        if isinstance(
            self.fes_parameters["model"],
            DingModelPulseWidthFrequency | DingModelPulseWidthFrequencyWithFatigue,
        ):
            pulse_width_format = (
                isinstance(self.fes_parameters["pulse_width"], int | float | list)
                if not isinstance(self.fes_parameters["pulse_width"], bool)
                else False
            )
            pulse_width_format = (
                all([isinstance(pulse_width, int) for pulse_width in self.fes_parameters["pulse_width"]])
                if pulse_width_format == list
                else pulse_width_format
            )

            if pulse_width_format is False:
                raise TypeError("pulse_width must be int, float or list type")

            minimum_pulse_width = self.fes_parameters["model"].pd0
            min_pulse_width_check = (
                all([pulse_width >= minimum_pulse_width for pulse_width in self.fes_parameters["pulse_width"]])
                if isinstance(self.fes_parameters["pulse_width"], list)
                else self.fes_parameters["pulse_width"] >= minimum_pulse_width
            )

            if min_pulse_width_check is False:
                raise ValueError("pulse width must be greater than minimum pulse width")

        if isinstance(
            self.fes_parameters["model"],
            DingModelPulseIntensityFrequency | DingModelPulseIntensityFrequencyWithFatigue,
        ):
            pulse_intensity_format = (
                isinstance(self.fes_parameters["pulse_intensity"], int | float | list)
                if not isinstance(self.fes_parameters["pulse_intensity"], bool)
                else False
            )
            pulse_intensity_format = (
                all([isinstance(pulse_intensity, int) for pulse_intensity in self.fes_parameters["pulse_intensity"]])
                if pulse_intensity_format == list
                else pulse_intensity_format
            )

            if pulse_intensity_format is False:
                raise TypeError("pulse_intensity must be int, float or list type")

            minimum_pulse_intensity = (
                all(
                    [
                        pulse_width >= self.fes_parameters["model"].min_pulse_intensity()
                        for pulse_width in self.fes_parameters["pulse_intensity"]
                    ]
                )
                if isinstance(self.fes_parameters["pulse_intensity"], list)
                else bool(self.fes_parameters["pulse_intensity"] >= self.fes_parameters["model"].min_pulse_intensity())
            )

            if minimum_pulse_intensity is False:
                raise ValueError("Pulse intensity must be greater than minimum pulse intensity")

        if not isinstance(self.fes_parameters["pulse_mode"], str):
            raise ValueError("pulse_mode must be a string type")

        if not isinstance(self.ivp_parameters["final_time"], int | float):
            raise ValueError("final_time must be an int or float type")

        if not isinstance(
            self.ivp_parameters["ode_solver"],
            (OdeSolver.RK1, OdeSolver.RK2, OdeSolver.RK4, OdeSolver.COLLOCATION),
        ):
            raise ValueError("ode_solver must be a OdeSolver type")

        if not isinstance(self.ivp_parameters["n_threads"], int):
            raise ValueError("n_thread must be a int type")

    def _pulse_mode_settings(self):
        if self.pulse_mode == "single":
            pass
        elif self.pulse_mode == "doublet":
            doublet_step = 0.005
            stim_time_doublet = [round(stim_time + doublet_step, 3) for stim_time in self.stim_time]
            self.stim_time = self.stim_time + stim_time_doublet
            self.stim_time.sort()
            self.model.stim_time = self.stim_time
            self.n_stim = len(self.stim_time)
            self.model.stim_time = self.stim_time
            self.n_shooting = self.model.get_n_shooting(self.final_time)

        elif self.pulse_mode == "triplet":
            doublet_step = 0.005
            triplet_step = 0.01
            stim_time_doublet = [round(stim_time + doublet_step, 3) for stim_time in self.stim_time]
            stim_time_triplet = [round(stim_time + triplet_step, 3) for stim_time in self.stim_time]
            self.stim_time = self.stim_time + stim_time_doublet + stim_time_triplet
            self.stim_time.sort()
            self.model.stim_time = self.stim_time
            self.n_stim = len(self.stim_time)
            self.model.stim_time = self.stim_time
            self.n_shooting = self.model.get_n_shooting(self.final_time)

        else:
            raise ValueError("Pulse mode not yet implemented")

    def _prepare_fake_ocp(self):
        """This function creates the initial value problem by hacking Bioptim's OptimalControlProgram.
        It is not the normal use of bioptim, but it enables a simplified ivp construction.
        """

        return OptimalControlProgram(
            bio_model=[self.model],
            dynamics=self.dynamics,
            n_shooting=self.n_shooting,
            phase_time=self.final_time,
            x_init=self.x_init,
            u_init=self.u_init,
            ode_solver=self.ode_solver,
            control_type=ControlType.CONSTANT,
            use_sx=False,
            parameters=self.parameters,
            parameter_init=self.parameters_init,
            parameter_bounds=self.parameters_bounds,
            n_threads=self.n_threads,
        )

    def _build_solution_from_initial_guess(self):
        return Solution.from_initial_guess(self.fake_ocp, [self.dt, self.x_init, self.u_init, self.p_init, self.s_init])

    def integrate(
        self,
        shooting_type=Shooting.SINGLE,
        integrator=SolutionIntegrator.OCP,
        to_merge=None,
        return_time=True,
        duplicated_times=False,
    ):
        to_merge = [SolutionMerge.NODES, SolutionMerge.PHASES] if to_merge is None else to_merge
        return self.initial_guess_solution.integrate(
            shooting_type=shooting_type,
            integrator=integrator,
            to_merge=to_merge,
            return_time=return_time,
            duplicated_times=duplicated_times,
        )

    def _declare_dynamics(self, numerical_data_time_series=None):

        self.dynamics = DynamicsList()
        self.dynamics.add(
            self.model.declare_ding_variables,
            dynamic_function=self.model.dynamics,
            expand_dynamics=True,
            expand_continuity=False,
            phase=0,
            phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
            numerical_data_timeseries=numerical_data_time_series,
        )

    def build_initial_guess_from_ocp(self, ocp, stim_idx_at_node_list=None):
        """
        Build a state, control, parameters and stochastic initial guesses for each phases from a given ocp
        """
        x = InitialGuessList()
        u = InitialGuessList()
        p = InitialGuessList()
        s = InitialGuessList()

        muscle_name = "_" + ocp.model.muscle_name if ocp.model.muscle_name else ""
        for j in range(len(self.model.name_dof)):
            key = ocp.model.name_dof[j] + muscle_name
            x.add(key=key, initial_guess=ocp.model.standard_rest_values()[j], phase=0)

        if ocp.controls_keys:
            for key in ocp.controls_keys:
                if "pulse_intensity" in ocp.controls_keys:
                    if isinstance(self.pulse_intensity, list) and len(self.pulse_intensity) != 1:
                        padded = [
                            [self.pulse_intensity[stim_idx_at_node_list[i][0]]]
                            * (self.model.sum_stim_truncation - len(stim_idx_at_node_list[i]))
                            + [self.pulse_intensity[idx] for idx in stim_idx_at_node_list[i]]
                            for i in range(self.n_shooting)
                        ]

                        initial_guess_list = [list(row) for row in zip(*padded)]
                    else:
                        pi = self.pulse_intensity[0] if isinstance(self.pulse_intensity, list) else self.pulse_width
                        initial_guess_list = [[pi] * self.model.sum_stim_truncation] * self.n_shooting

                    u.add(key, initial_guess=initial_guess_list, phase=0, interpolation=InterpolationType.EACH_FRAME)

                if "last_pulse_width" in ocp.controls_keys:
                    if isinstance(self.pulse_width, list) and len(self.pulse_width) != 1:
                        last_stim_idx = [stim_idx_at_node_list[i][-1] for i in range(len(stim_idx_at_node_list) - 1)]
                        initial_guess = [self.pulse_width[last_stim_idx[i]] for i in range(len(last_stim_idx))]
                    else:
                        pw = self.pulse_width[0] if isinstance(self.pulse_width, list) else self.pulse_width
                        initial_guess = [pw] * self.n_shooting
                    u.add(key, initial_guess=[initial_guess], phase=0, interpolation=InterpolationType.EACH_FRAME)

        if len(ocp.parameters) != 0:
            for key in ocp.parameters.keys():
                p.add(key=key, initial_guess=ocp.parameters_init[key])

        return x, u, p, s

    @classmethod
    def from_frequency_and_final_time(
        cls,
        fes_parameters: dict = None,
        ivp_parameters: dict = None,
    ):
        """
        Enables the creation of an ivp problem from final time and frequency information instead of the stimulation
        number. The frequency indication is mandatory and round_down state must be set to True if the stim number is
        expected to not be even.

        Parameters
        ----------
        fes_parameters: dict
           The parameters for the fes configuration including :
           model, pulse_width, pulse_intensity, pulse_mode, frequency, round_down
        ivp_parameters: dict
           The parameters for the ivp problem including :
           final_time, ode_solver, n_threads
        """

        frequency = fes_parameters["frequency"]
        if not isinstance(frequency, int):
            raise ValueError("Frequency must be an int")
        round_down = fes_parameters["round_down"]
        if not isinstance(round_down, bool):
            raise ValueError("Round down must be a bool")
        final_time = ivp_parameters["final_time"]
        if not isinstance(final_time, int | float):
            raise ValueError("Final time must be an int or float")

        fes_parameters["n_stim"] = final_time * frequency

        if round_down or fes_parameters["n_stim"].is_integer():
            fes_parameters["n_stim"] = int(fes_parameters["n_stim"])
        else:
            raise ValueError(
                "The number of stimulation needs to be integer within the final time t, set round down "
                "to True or set final_time * frequency to make the result an integer."
            )
        fes_parameters["stim_time"] = list(
            np.round([i * 1 / fes_parameters["frequency"] for i in range(fes_parameters["n_stim"])], 3)
        )
        return cls(
            fes_parameters,
            ivp_parameters,
        )

    @classmethod
    def from_frequency_and_n_stim(
        cls,
        fes_parameters: dict = None,
        ivp_parameters: dict = None,
    ):
        """
        Enables the creation of an ivp problem from stimulation number and frequency information instead of the final
        time.

        Parameters
        ----------
        fes_parameters: dict
           The parameters for the fes configuration including :
           model, n_stim, pulse_width, pulse_intensity, pulse_mode
        ivp_parameters: dict
           The parameters for the ivp problem including :
           final_time, ode_solver, n_threads
        """

        n_stim = fes_parameters["n_stim"]
        if not isinstance(n_stim, int):
            raise ValueError("n_stim must be an int")
        frequency = fes_parameters["frequency"]
        if not isinstance(frequency, int):
            raise ValueError("Frequency must be an int")

        ivp_parameters["final_time"] = n_stim / frequency

        fes_parameters["stim_time"] = list(
            np.round([i * 1 / fes_parameters["frequency"] for i in range(fes_parameters["n_stim"])], 3)
        )

        return cls(
            fes_parameters,
            ivp_parameters,
        )
