import matplotlib.pyplot as plt
from cocofest import IvpFes, ModelMaker


def launch_simulation():
    final_time = 1
    model = ModelMaker.create_model("ding2003_with_fatigue", stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ivp_parameters = {"final_time": final_time, "use_sx": True}

    pulse_modes = ["single", "doublet", "triplet"]
    results = {}

    for pulse_mode in pulse_modes:
        fes_parameters = {
            "model": model,
            "pulse_mode": pulse_mode,
        }
        ivp = IvpFes(
            fes_parameters,
            ivp_parameters,
        )
        result, time = ivp.integrate()
        results[pulse_mode] = {
            "force": result["F"][0],
            "time": time,
            "stimulation": ivp.stim_time,
        }
    return results, pulse_modes


def main(plot=True):
    results, pulse_modes = launch_simulation()

    if plot:
        # --- Show results --- #
        plt.title("Force state result for single, doublet and triplet")

        colors = {"single": "blue", "doublet": "red", "triplet": "green"}
        for pulse_mode in pulse_modes:
            plt.plot(
                results[pulse_mode]["time"],
                results[pulse_mode]["force"],
                color=colors[pulse_mode],
                label=f"force {pulse_mode}",
            )
            plt.vlines(
                x=results[pulse_mode]["stimulation"],
                ymin=max(results[pulse_mode]["force"]) - 30,
                ymax=max(results[pulse_mode]["force"]),
                colors=colors[pulse_mode],
                ls="-." if pulse_mode == "single" else ":" if pulse_mode == "doublet" else "--",
                lw=2,
                label=f"stimulation {pulse_mode}",
            )

        plt.xlabel("time (s)")
        plt.ylabel("force (N)")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
