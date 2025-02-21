import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from bioptim import OdeSolver

from cocofest import (
    ModelMaker,
    DingModelPulseIntensityFrequency,
    DingModelPulseIntensityFrequencyForceParameterIdentification,
    IvpFes,
)
from cocofest.identification.identification_method import full_data_extraction


def simulate_data(n_stim: int, final_time: int, pulse_intensity_values: list, n_integration_steps):
    """
    Simulate the data using the pulse intensity method.
    Returns a dictionary with time, force, stim_time, and pulse_intensity.
    """
    # Create stimulation times and model
    stim_time = list(np.linspace(0, final_time, n_stim + 1)[:-1])
    model = ModelMaker.create_model("hmed2018", stim_time=stim_time)

    fes_parameters = {"model": model, "pulse_intensity": pulse_intensity_values}
    ivp_parameters = {
        "final_time": final_time,
        "use_sx": True,
        "ode_solver": OdeSolver.RK4(n_integration_steps=n_integration_steps),
    }
    ivp = IvpFes(fes_parameters, ivp_parameters)

    # Integrate to simulate the force response
    result, time = ivp.integrate()
    data = {
        "time": time,
        "force": result["F"][0],
        "stim_time": stim_time,
        "pulse_intensity": pulse_intensity_values,
    }
    return data


def save_simulation(data, filename):
    """Save the simulation data to a pickle file."""
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def extract_identified_parameters(identified, keys):
    """
    For each parameter in keys, use the identified value if available;
    otherwise fall back to the default from DingModelPulseIntensityFrequency.
    Returns a dictionary mapping parameter names to their values.
    """
    default_model = DingModelPulseIntensityFrequency()
    return {key: identified.get(key, getattr(default_model, key)) for key in keys}


def run_identification(data_path, final_time=5, n_integration_steps=10, n_threads=6):
    """
    Run the parameter identification. The model is recreated using the stim_time
    stored in the simulation file.
    """
    # Load simulation data to retrieve stim_time
    with open(data_path, "rb") as f:
        sim_data = pickle.load(f)
    stim_time = sim_data["stim_time"]
    model = ModelMaker.create_model("hmed2018", stim_time=stim_time)

    ocp = DingModelPulseIntensityFrequencyForceParameterIdentification(
        model=model,
        data_path=[data_path],
        identification_method="full",
        double_step_identification=False,
        key_parameter_to_identify=["a_rest", "km_rest", "tau1_rest", "tau2", "ar", "bs", "Is", "cr"],
        additional_key_settings={},
        final_time=final_time,
        use_sx=True,
        n_threads=n_threads,
        ode_solver=OdeSolver.RK4(n_integration_steps=n_integration_steps),
    )
    return ocp.force_model_identification()


def annotate_parameters(ax, identified_params, default_model, start_x=0.7, y_start=0.4, y_step=0.05):
    """
    Annotate the plot with parameter names, the identified values, and default values.
    The names are annotated in black, identified values in red, and default values in blue.
    """
    for i, key in enumerate(identified_params.keys()):
        y = y_start - i * y_step
        ax.annotate(f"{key} :", xy=(start_x, y), xycoords="axes fraction", color="black")
        ax.annotate(f"{round(identified_params[key], 5)}", xy=(start_x + 0.08, y), xycoords="axes fraction", color="red")
        ax.annotate(f"{getattr(default_model, key)}", xy=(start_x + 0.15, y), xycoords="axes fraction", color="blue")


def main():
    # Parameters for simulation and identification
    n_stim = 50
    final_time = 5
    integration_steps = 10
    pulse_intensity_values = list(np.random.randint(40, 130, n_stim))
    pickle_file = "../data/temp_identification_simulation.pkl"

    # Simulate data and save it
    sim_data = simulate_data(n_stim, final_time, pulse_intensity_values=pulse_intensity_values, n_integration_steps=integration_steps)
    save_simulation(sim_data, pickle_file)

    # Run identification and extract parameters of interest
    identified_results = run_identification(pickle_file, final_time, integration_steps)
    param_keys = ["a_rest", "km_rest", "tau1_rest", "tau2", "ar", "bs", "Is", "cr"]
    identified_params = extract_identified_parameters(identified_results, param_keys)

    print("Identified parameters:")
    for key, value in identified_params.items():
        print(f"  {key}: {value}")

    # Extract simulation data for plotting
    pickle_time_data, _, pickle_muscle_data, _ = full_data_extraction([pickle_file])

    # Plot the simulation and identification results
    fig, ax = plt.subplots()
    ax.set_title("Force state result")
    ax.plot(pickle_time_data, pickle_muscle_data, color="blue", label="simulated")
    ax.plot(identified_results["time"], identified_results["force"], color="green", label="identified")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("force (N)")

    default_model = DingModelPulseIntensityFrequency()
    annotate_parameters(ax, identified_params, default_model)

    ax.legend()
    plt.show()

    # Cleanup temporary file
    os.remove(pickle_file)


if __name__ == "__main__":
    main()
