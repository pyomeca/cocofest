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


# --- Build nmpc cyclic --- #
cycles_len = 100
cycle_duration = 0.25
n_cycles = 3

minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0
fes_model = DingModelPulseDurationFrequencyWithFatigue()
fes_model.alpha_a = -4.0 * 10e-1  # Increasing the fatigue rate to make the fatigue more visible
nmpc = NmpcFes.prepare_nmpc(
    model=fes_model,
    n_stim=11,
    n_shooting=cycles_len,
    final_time=0.5,
    pulse_duration={
        "min": minimum_pulse_duration,
        "max": 0.0006,
        "bimapping": False,
    },
    # objective={"force_tracking": force_tracking},
    objective={"end_node_tracking": 100},
    cycle_len=cycles_len,
    cycle_duration=cycle_duration,
    # n_cycles=n_cycles,
    use_sx=True,
    stim_time=list(np.linspace(0, 0.5, 11)),

)


sol = nmpc.solve(nmpc.update_functions, solver=Solver.IPOPT())
sol.graphs()
