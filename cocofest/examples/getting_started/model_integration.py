import matplotlib.pyplot as plt
from cocofest import (
    DingModelFrequencyWithFatigueIntegrate,
    IvpFes,
)

# --- Build ocp --- #
# This problem was build to be integrated and has no objectives nor parameter to optimize.

fes_parameters = {
    "model": DingModelFrequencyWithFatigueIntegrate(),
    "stim_time": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}
ivp_parameters = {"n_shooting": 100, "final_time": 1, "use_sx": True}

ivp = IvpFes(fes_parameters, ivp_parameters)

result, time = ivp.integrate()

# Plotting the force state result
plt.title("Force state result")

plt.plot(time, result["F"][0], color="blue", label="force")
plt.xlabel("time (s)")
plt.ylabel("force (N)")
plt.legend()
plt.show()
