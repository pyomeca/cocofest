"""
This example showcases a moving time horizon simulation problem of cyclic muscle force tracking.
The FES model used here is Ding's 2007 pulse duration and frequency model with fatigue.
Only the pulse duration is optimized, frequency is fixed.
The nmpc cyclic problem is composed of 3 cycles and will move forward 1 cycle at each step.
Only the middle cycle is kept in the optimization problem, the nmpc cyclic problem stops once the last 6th cycle is reached.
"""
import matplotlib.pyplot as plt
import numpy as np
from bioptim import Solver, SolutionMerge
from cocofest import NmpcFes, DingModelPulseDurationFrequencyWithFatigue, FourierSeries


# --- Building force to track ---#
target_time = np.linspace(0, 1, 100)
target_force = abs(np.sin(target_time * np.pi)) * 200
force_tracking = [target_time, target_force]

# --- Build nmpc cyclic --- #
cycles_len = 100
cycle_duration = 1
n_cycles = 8

minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0
fes_model = DingModelPulseDurationFrequencyWithFatigue()
fes_model.alpha_a = -4.0 * 10e-1  # Increasing the fatigue rate to make the fatigue more visible
nmpc = NmpcFes.prepare_nmpc(
    model=fes_model,
    n_stim=30,
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
    use_sx=True,
    stim_time=list(np.round(np.linspace(0, 1, 31), 3))[:-1],
    n_threads=4,
)

sol = nmpc.solve(nmpc.update_functions, solver=Solver.IPOPT(), cyclic_options={"states": {}}, get_all_iterations=True)

sol_merged = sol[0].decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])


time = sol[0].decision_time(to_merge=SolutionMerge.KEYS, continuous=True)
time = [float(j) for j in time]
fatigue = sol_merged["A"][0]
force = sol_merged["F"][0]

ax1 = plt.subplot(221)
ax1.plot(time, fatigue, label="A", color="green")
ax1.set_title("Fatigue", weight="bold")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Force scaling factor (-)")
plt.legend()

ax2 = plt.subplot(222)
ax2.plot(time, force, label="F", color="red", linewidth=4)
for i in range(n_cycles):
    if i == 0:
        ax2.plot(target_time, target_force, label="Target", color="purple")
    else:
        ax2.plot(target_time + i, target_force, color="purple")
ax2.set_title("Force", weight="bold")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Force (N)")
plt.legend()

barWidth = 0.25  # set width of bar
cycles = [sol[1][i].parameters["pulse_duration"] for i in range(len(sol[1]))]  # set height of bar
bar = []  # Set position of bar on X axis
for i in range(n_cycles):
    if i == 0:
        br = [barWidth * (x + 1) for x in range(len(cycles[i]))]
    else:
        br = [bar[-1][-1] + barWidth * (x + 1) for x in range(len(cycles[i]))]
    bar.append(br)

ax3 = plt.subplot(212)
for i in range(n_cycles):
    ax3.bar(bar[i], cycles[i], width=barWidth, edgecolor="grey", label=f"cycle n°{i+1}")
ax3.set_xticks([np.mean(r) for r in bar], [str(i + 1) for i in range(n_cycles)])
ax3.set_xlabel("Cycles")
ax3.set_ylabel("Pulse duration (s)")
plt.legend()
ax3.set_title("Pulse duration", weight="bold")
plt.show()
