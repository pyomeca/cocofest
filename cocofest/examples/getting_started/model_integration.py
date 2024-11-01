import matplotlib.pyplot as plt
from cocofest import IvpFes, ModelMaker

# --- Build ocp --- #
# This problem was build to be integrated and has no objectives nor parameter to optimize.
model = ModelMaker.create_model("ding2003_with_fatigue", is_approximated=False)  # Can not approximate this model in ivp
fes_parameters = {
    "model": model,
    "stim_time": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}
ivp_parameters = {"final_time": 1, "use_sx": True}

ivp = IvpFes(fes_parameters, ivp_parameters)

result, time = ivp.integrate()

# Plotting the force state result
plt.title("Force state result")

plt.plot(time, result["F"][0], color="blue", label="force")
plt.xlabel("time (s)")
plt.ylabel("force (N)")
plt.legend()
plt.show()
