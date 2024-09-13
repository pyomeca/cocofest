"""
This example showcases a moving time horizon simulation problem of cyclic muscle force tracking.
The FES model used here is Ding's 2007 pulse duration and frequency model with fatigue.
Only the pulse duration is optimized, frequency is fixed.
The nmpc cyclic problem is composed of 3 cycles and will move forward 1 cycle at each step.
Only the middle cycle is kept in the optimization problem, the nmpc cyclic problem stops once the last 6th cycle is reached.
"""
import numpy as np
from bioptim import Solver
from cocofest import NmpcFes, DingModelPulseDurationFrequencyWithFatigue


# --- Building force to track ---#
time = np.linspace(0, 1, 101)
force = abs(np.sin(time * 5) + np.random.normal(scale=0.1, size=len(time))) * 100
force_tracking = [time, force]

# --- Build nmpc cyclic --- #
cycles_len = 100
cycle_duration = 1
n_cycles = 3

minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0
fes_model = DingModelPulseDurationFrequencyWithFatigue()
fes_model.alpha_a = -4.0 * 10e-1  # Increasing the fatigue rate to make the fatigue more visible
nmpc = NmpcFes.prepare_nmpc(
    model=fes_model,
    n_stim=10,
    n_shooting=cycles_len,
    final_time=cycle_duration,
    pulse_duration={
        "min": minimum_pulse_duration,
        "max": 0.0006,
        "bimapping": False,
    },
    objective={"force_tracking": force_tracking},
    cycle_len=cycles_len,
    cycle_duration=cycle_duration,
    # n_cycles=n_cycles,
    use_sx=True,
    stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],

)


sol = nmpc.solve(nmpc.update_functions, solver=Solver.IPOPT(), cyclic_options={"states": {}})
sol.graphs()
