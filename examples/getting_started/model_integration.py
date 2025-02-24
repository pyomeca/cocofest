"""
This example was build to explain of to integrate a solution and has no objectives nor parameter to optimize.
The uncommented model used is the DingModelFrequencyWithFatigue, but you can change it to any other model.
The model is integrated for 300 seconds and the stimulation will be on for 1 second at 33 Hz and of for a second.
The effect of the fatigue will be visible and the force state result will decrease over time.
"""

from cocofest import (
    IvpFes,
    DingModelFrequencyWithFatigue,
    DingModelPulseIntensityFrequencyWithFatigue,
    DingModelPulseWidthFrequencyWithFatigue,
    FES_plot,
)
import numpy as np
from bioptim import OdeSolver


def main(model_name="Ding2003"):
    # --- Set stimulation time apparition --- #
    final_time = 3
    stim_time = [val for start in range(0, final_time, 2) for val in np.linspace(start, start + 1, 34)[:-1]]

    if model_name == "Ding2003":
        model = DingModelFrequencyWithFatigue(stim_time=stim_time, sum_stim_truncation=10)
        fes_parameters = {"model": model}

    if model_name == "Ding2007":
        pulse_width = np.random.uniform(0.00019, 0.0006, len(stim_time)).tolist()
        model = DingModelPulseWidthFrequencyWithFatigue(stim_time=stim_time, sum_stim_truncation=10)
        fes_parameters = {"model": model, "pulse_width": pulse_width}

    if model_name == "Hmed2018":
        pulse_intensity = np.random.randint(20, 130, len(stim_time)).tolist()
        model = DingModelPulseIntensityFrequencyWithFatigue(stim_time=stim_time, sum_stim_truncation=10)
        fes_parameters = {"model": model, "pulse_intensity": pulse_intensity}

    # --- Build ivp --- #
    ivp_parameters = {"final_time": final_time, "ode_solver": OdeSolver.RK1(n_integration_steps=10)}
    ivp = IvpFes(fes_parameters=fes_parameters, ivp_parameters=ivp_parameters)

    result, time = ivp.integrate()
    result["time"] = time

    # Plotting the force state result
    FES_plot(data=result).plot(title="FES model integration")


if __name__ == "__main__":
    main()
