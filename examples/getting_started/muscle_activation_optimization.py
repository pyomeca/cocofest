from cocofest import (
VeltinkModelPulseIntensity,
VeltinkRienerModelPulseIntensityWithFatigue,
OcpFes,
)

from bioptim import (
    OptimalControlProgram,
    DynamicsEvaluation,
    NonLinearProgram,
    ConfigureProblem,
    PhaseDynamics,
    DynamicsList,
    Solver,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    Node,
)

def prepare_ocp(model, final_time, n_shooting, fmax):
    # --- Set dynamics --- #
    dynamics = DynamicsList()
    dynamics.add(
        model.declare_model_variables,
        dynamic_function=model.dynamics,
        expand_dynamics=True,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        numerical_data_timeseries=None,
    )

    # --- Set initial guesses and bounds for states and controls --- #
    x_bounds = BoundsList()
    x_bounds.add("a", min_bound=0, max_bound=1)

    u_bounds = BoundsList()
    u_bounds.add("I", min_bound=model.I_threshold, max_bound=model.I_saturation)

    # --- Set objective functions --- #
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="a", weight=1, quadratic=True)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="a", node=Node.END, target=200/fmax, weight=1e5, quadratic=True
    )

    return OptimalControlProgram(
        bio_model=[model],
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=final_time,
        objective_functions=objective_functions,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        use_sx=True,
        n_threads=20,
    )


def main(with_fatigue=True, plot=True):
    final_time = 1
    fmax = 350
    n_shooting = 100

    model = VeltinkRienerModelPulseIntensityWithFatigue() if with_fatigue else VeltinkModelPulseIntensity()
    model.I_saturation = 70
    model.I_threshold = 20
    ocp = prepare_ocp(model=model, final_time=final_time, n_shooting=n_shooting, fmax=fmax)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT())
    # --- Plot the results --- #
    if plot:
        sol.graphs()


if __name__ == "__main__":
    main()
