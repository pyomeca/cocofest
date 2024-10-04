import numpy as np

from bioptim import Solver, SolutionMerge
from cocofest import NmpcFes, DingModelPulseDurationFrequencyWithFatigue


def test_nmpc_cyclic():
    # --- Build target force --- #
    target_time = np.linspace(0, 1, 100)
    target_force = abs(np.sin(target_time * np.pi)) * 50
    force_tracking = [target_time, target_force]

    # --- Build nmpc cyclic --- #
    cycles_len = 100
    cycle_duration = 1

    minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0
    fes_model = DingModelPulseDurationFrequencyWithFatigue()
    fes_model.alpha_a = -4.0 * 10e-1  # Increasing the fatigue rate to make the fatigue more visible

    nmpc = NmpcFes.prepare_nmpc(
        model=fes_model,
        stim_time=list(np.round(np.linspace(0, 1, 11), 2))[:-1],
        cycle_len=cycles_len,
        cycle_duration=cycle_duration,
        pulse_duration={
            "min": minimum_pulse_duration,
            "max": 0.0006,
            "bimapping": False,
        },
        objective={"force_tracking": force_tracking},
        use_sx=True,
        n_threads=6,
    )

    n_cycles_total = 6

    def update_functions(_nmpc, cycle_idx, _sol):
        return cycle_idx < n_cycles_total  # True if there are still some cycle to perform

    sol = nmpc.solve(
        update_functions,
        solver=Solver.IPOPT(),
        cyclic_options={"states": {}},
        get_all_iterations=True,
    )
    sol_merged = sol[0].decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])

    time = sol[0].decision_time(to_merge=SolutionMerge.KEYS, continuous=True)
    time = [float(j) for j in time]
    fatigue = sol_merged["A"][0]
    force = sol_merged["F"][0]

    np.testing.assert_almost_equal(time[0], 0.0, decimal=4)
    np.testing.assert_almost_equal(fatigue[0], 4920.0, decimal=4)
    np.testing.assert_almost_equal(force[0], 0, decimal=4)

    np.testing.assert_almost_equal(time[300], 3.0, decimal=4)
    np.testing.assert_almost_equal(fatigue[300], 4550.2883, decimal=4)
    np.testing.assert_almost_equal(force[300], 4.1559, decimal=4)

    np.testing.assert_almost_equal(time[-1], 6.0, decimal=4)
    np.testing.assert_almost_equal(fatigue[-1], 4184.9710, decimal=4)
    np.testing.assert_almost_equal(force[-1], 5.3672, decimal=4)
