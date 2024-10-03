import pytest
import re
import numpy as np

from bioptim import OdeSolver
from cocofest import OcpFesNmpcCyclic, DingModelPulseDurationFrequencyWithFatigue


def test_nmpc_cyclic():
    # --- Build target force --- #
    target_time = np.linspace(0, 1, 100)
    target_force = abs(np.sin(target_time * np.pi)) * 50
    force_tracking = [target_time, target_force]

    # --- Build nmpc cyclic --- #
    n_total_cycles = 6
    n_stim = 10
    n_shooting = 5

    minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0
    fes_model = DingModelPulseDurationFrequencyWithFatigue(sum_stim_truncation=10)
    fes_model.alpha_a = -4.0 * 10e-1  # Increasing the fatigue rate to make the fatigue more visible

    nmpc = OcpFesNmpcCyclic(
        model=fes_model,
        n_stim=n_stim,
        n_shooting=n_shooting,
        final_time=1,
        pulse_duration={
            "min": minimum_pulse_duration,
            "max": 0.0006,
            "bimapping": False,
        },
        objective={"force_tracking": force_tracking},
        n_total_cycles=n_total_cycles,
        n_simultaneous_cycles=3,
        n_cycle_to_advance=1,
        cycle_to_keep="middle",
        use_sx=True,
        ode_solver=OdeSolver.COLLOCATION(),
    )

    nmpc.prepare_nmpc()
    nmpc.solve()

    # --- Show results --- #
    time = [j for sub in nmpc.result["time"] for j in sub]
    fatigue = [j for sub in nmpc.result["states"]["A"] for j in sub]
    force = [j for sub in nmpc.result["states"]["F"] for j in sub]

    np.testing.assert_almost_equal(
        len(time),
        n_total_cycles * n_stim * n_shooting * (nmpc.ode_solver.polynomial_degree + 1),
    )
    np.testing.assert_almost_equal(len(fatigue), len(time))
    np.testing.assert_almost_equal(len(force), len(time))

    np.testing.assert_almost_equal(time[0], 0.0, decimal=4)
    np.testing.assert_almost_equal(fatigue[0], 4796.3120, decimal=4)
    np.testing.assert_almost_equal(force[0], 3.0948, decimal=4)

    np.testing.assert_almost_equal(time[750], 3.0, decimal=4)
    np.testing.assert_almost_equal(fatigue[750], 4427.2596, decimal=4)
    np.testing.assert_almost_equal(force[750], 4.5089, decimal=4)

    np.testing.assert_almost_equal(time[-1], 5.9986, decimal=4)
    np.testing.assert_almost_equal(fatigue[-1], 4063.8504, decimal=4)
    np.testing.assert_almost_equal(force[-1], 5.6615, decimal=4)


def test_all_nmpc_errors():
    with pytest.raises(
        TypeError,
        match=re.escape("n_total_cycles must be an integer"),
    ):
        OcpFesNmpcCyclic(
            model=DingModelPulseDurationFrequencyWithFatigue(sum_stim_truncation=10),
            n_stim=10,
            n_shooting=5,
            final_time=1,
            pulse_duration={
                "min": 0.0003,
                "max": 0.0006,
                "bimapping": False,
            },
            n_total_cycles=None,
        )

    with pytest.raises(
        TypeError,
        match=re.escape("n_simultaneous_cycles must be an integer"),
    ):
        OcpFesNmpcCyclic(
            model=DingModelPulseDurationFrequencyWithFatigue(sum_stim_truncation=10),
            n_stim=10,
            n_shooting=5,
            final_time=1,
            pulse_duration={
                "min": 0.0003,
                "max": 0.0006,
                "bimapping": False,
            },
            n_total_cycles=5,
        )

    with pytest.raises(
        TypeError,
        match=re.escape("n_cycle_to_advance must be an integer"),
    ):
        OcpFesNmpcCyclic(
            model=DingModelPulseDurationFrequencyWithFatigue(sum_stim_truncation=10),
            n_stim=10,
            n_shooting=5,
            final_time=1,
            pulse_duration={
                "min": 0.0003,
                "max": 0.0006,
                "bimapping": False,
            },
            n_total_cycles=5,
            n_simultaneous_cycles=3,
        )

    with pytest.raises(
        TypeError,
        match=re.escape("cycle_to_keep must be a string"),
    ):
        OcpFesNmpcCyclic(
            model=DingModelPulseDurationFrequencyWithFatigue(sum_stim_truncation=10),
            n_stim=10,
            n_shooting=5,
            final_time=1,
            pulse_duration={
                "min": 0.0003,
                "max": 0.0006,
                "bimapping": False,
            },
            n_total_cycles=5,
            n_simultaneous_cycles=3,
            n_cycle_to_advance=1,
        )

    with pytest.raises(
        ValueError,
        match=re.escape("The number of n_simultaneous_cycles must be higher than the number of n_cycle_to_advance"),
    ):
        OcpFesNmpcCyclic(
            model=DingModelPulseDurationFrequencyWithFatigue(sum_stim_truncation=10),
            n_stim=10,
            n_shooting=5,
            final_time=1,
            pulse_duration={
                "min": 0.0003,
                "max": 0.0006,
                "bimapping": False,
            },
            n_total_cycles=5,
            n_simultaneous_cycles=3,
            n_cycle_to_advance=6,
            cycle_to_keep="middle",
        )

    with pytest.raises(
        ValueError,
        match=re.escape("The number of n_total_cycles must be a multiple of the number n_cycle_to_advance"),
    ):
        OcpFesNmpcCyclic(
            model=DingModelPulseDurationFrequencyWithFatigue(sum_stim_truncation=10),
            n_stim=10,
            n_shooting=5,
            final_time=1,
            pulse_duration={
                "min": 0.0003,
                "max": 0.0006,
                "bimapping": False,
            },
            n_total_cycles=5,
            n_simultaneous_cycles=3,
            n_cycle_to_advance=2,
            cycle_to_keep="middle",
        )

    with pytest.raises(
        ValueError,
        match=re.escape("cycle_to_keep must be either 'first', 'middle' or 'last'"),
    ):
        OcpFesNmpcCyclic(
            model=DingModelPulseDurationFrequencyWithFatigue(sum_stim_truncation=10),
            n_stim=10,
            n_shooting=5,
            final_time=1,
            pulse_duration={
                "min": 0.0003,
                "max": 0.0006,
                "bimapping": False,
            },
            n_total_cycles=5,
            n_simultaneous_cycles=3,
            n_cycle_to_advance=1,
            cycle_to_keep="between",
        )

    with pytest.raises(
        NotImplementedError,
        match=re.escape("Only 'middle' cycle_to_keep is implemented"),
    ):
        OcpFesNmpcCyclic(
            model=DingModelPulseDurationFrequencyWithFatigue(sum_stim_truncation=10),
            n_stim=10,
            n_shooting=5,
            final_time=1,
            pulse_duration={
                "min": 0.0003,
                "max": 0.0006,
                "bimapping": False,
            },
            n_total_cycles=5,
            n_simultaneous_cycles=3,
            n_cycle_to_advance=1,
            cycle_to_keep="first",
        )

    with pytest.raises(
        NotImplementedError,
        match=re.escape("Only 3 simultaneous cycles are implemented yet work in progress"),
    ):
        OcpFesNmpcCyclic(
            model=DingModelPulseDurationFrequencyWithFatigue(sum_stim_truncation=10),
            n_stim=10,
            n_shooting=5,
            final_time=1,
            pulse_duration={
                "min": 0.0003,
                "max": 0.0006,
                "bimapping": False,
            },
            n_total_cycles=5,
            n_simultaneous_cycles=6,
            n_cycle_to_advance=1,
            cycle_to_keep="middle",
        )
