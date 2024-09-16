"""
This example will do a 10 stimulation example with Ding's 2003 frequency model associated to Hmed's 2018 intensity work
This ocp was build to match a force curve across all optimization.
"""

import matplotlib.pyplot as plt
import numpy as np

from bioptim import SolutionMerge
from cocofest import (
    DingModelIntensityFrequency,
    FourierSeries,
    OcpFes,
)

# --- Building force to track ---#
time = np.linspace(0, 1, 101)
force = abs(np.sin(time * 5) + np.random.normal(scale=0.1, size=len(time))) * 100
force_tracking = [time, force]

# --- Build ocp --- #
# This ocp was build to track a force curve along the problem.
# The stimulation won't be optimized and is already set to one pulse every 0.1 seconds (n_stim/final_time).
# Plus the pulsation intensity will be optimized between 0 and 130 mA and are not the same across the problem.
model = DingModelIntensityFrequency()
minimum_pulse_intensity = model.min_pulse_intensity()

ocp = OcpFes().prepare_ocp(
    model=model,
    n_shooting=200,
    final_time=1,
    pulse_intensity={
        "min": minimum_pulse_intensity,
        "max": 130,
        "bimapping": False,
    },
    objective={"force_tracking": force_tracking},
    use_sx=True,
    stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
)

# --- Solve the program --- #
sol = ocp.solve()

# --- Show the optimization results --- #
sol.graphs()

# --- Show results from solution --- #
sol_merged = sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])

fourier_fun = FourierSeries()
fourier_coef = fourier_fun.compute_real_fourier_coeffs(time, force, 50)
y_approx = FourierSeries().fit_func_by_fourier_series_with_real_coeffs(time, fourier_coef)
plt.title("Comparison between given and simulated force after parameter optimization")
plt.plot(time, force, color="red", label="force from file")
plt.plot(time, y_approx, color="orange", label="force after fourier transform")

solution_time = sol.decision_time(to_merge=SolutionMerge.KEYS, continuous=True)
solution_time = [float(j) for j in solution_time]

plt.plot(solution_time, sol_merged["F"].squeeze(), color="blue", label="force from optimized stimulation")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.legend()
plt.show()
