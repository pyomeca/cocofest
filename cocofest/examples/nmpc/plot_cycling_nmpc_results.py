from matplotlib import pyplot as plt
import numpy.ma as ma
import numpy as np
import pickle
import biorbd
from cocofest import PickleAnimate


# --- Load results --- #
with open("results/cycling_fes_driven_nmpc_full_force.pkl", "rb") as file:
    full_nmpc_result_force_optim = pickle.load(file)

with open("results/cycling_fes_driven_nmpc_full_fatigue.pkl", "rb") as file:
    full_nmpc_result_fatigue_optim = pickle.load(file)

all_pulse_duration_force = []
all_pulse_duration_fatigue = []
for i in range(8):
    with open("results/cycling_fes_driven_nmpc_" + str(i) + "_force.pkl", "rb") as file:
        temp_nmpc_result_force_optim = pickle.load(file)

    with open("results/cycling_fes_driven_nmpc_" + str(i) + "_fatigue.pkl", "rb") as file:
        temp_nmpc_result_fatigue_optim = pickle.load(file)

    all_pulse_duration_force.append(temp_nmpc_result_force_optim["parameters"])
    all_pulse_duration_fatigue.append(temp_nmpc_result_fatigue_optim["parameters"])

barWidth = 0.25 * (2 / 5)  # set width of bar
cycles = []
pulse_duration_force_optim_dict = {
    "DeltoideusClavicle_A": [],
    "DeltoideusScapula_P": [],
    "TRIlong": [],
    "BIC_long": [],
    "BIC_brevis": [],
}

pulse_duration_fatigue_optim_dict = {
    "DeltoideusClavicle_A": [],
    "DeltoideusScapula_P": [],
    "TRIlong": [],
    "BIC_long": [],
    "BIC_brevis": [],
}

for i in range(8):
    [
        pulse_duration_force_optim_dict[f].append(all_pulse_duration_force[i]["pulse_duration_" + f])
        for f in pulse_duration_force_optim_dict.keys()
    ]
    [
        pulse_duration_fatigue_optim_dict[f].append(all_pulse_duration_fatigue[i]["pulse_duration_" + f])
        for f in pulse_duration_fatigue_optim_dict.keys()
    ]


for key in pulse_duration_force_optim_dict.keys():
    pulse_duration_force_optim_dict[key] = np.array([x for xs in pulse_duration_force_optim_dict[key] for x in xs])
    pulse_duration_fatigue_optim_dict[key] = np.array([x for xs in pulse_duration_fatigue_optim_dict[key] for x in xs])

bar = np.array([barWidth * (x + 0.5) for x in range(80)])

# --- Plot results --- #
# --- First subplot for position and residual torque --- #
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].set_title("Shoulder Q")
axs[1, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Angle (rad)")
axs[0, 1].set_title("Elbow Q")
axs[1, 1].set_xlabel("Time (s)")

axs[1, 0].set_title("Shoulder Tau")
axs[1, 0].set_ylabel("Torque (Nm)")
axs[1, 1].set_title("Elbow Tau")

for i in range(8):
    axs[0, 0].plot(
        full_nmpc_result_force_optim["time"][:101],
        full_nmpc_result_force_optim["states"]["q"][0][100 * i : 100 * (i + 1) + 1],
        label="force optim cycle " + str(i),
    )
    axs[0, 0].plot(
        full_nmpc_result_fatigue_optim["time"][:101],
        full_nmpc_result_fatigue_optim["states"]["q"][0][100 * i : 100 * (i + 1) + 1],
        label="fatigue optim cycle " + str(i),
    )

    axs[0, 1].plot(
        full_nmpc_result_force_optim["time"][:101],
        full_nmpc_result_force_optim["states"]["q"][1][100 * i : 100 * (i + 1) + 1],
        label="force optim cycle " + str(i),
    )
    axs[0, 1].plot(
        full_nmpc_result_fatigue_optim["time"][:101],
        full_nmpc_result_fatigue_optim["states"]["q"][1][100 * i : 100 * (i + 1) + 1],
        label="fatigue optim cycle " + str(i),
    )

axs[1, 0].plot(
    full_nmpc_result_force_optim["time"][:-1],
    full_nmpc_result_force_optim["control"]["tau"][0],
    label="force optim",
)
axs[1, 0].plot(
    full_nmpc_result_fatigue_optim["time"][:-1],
    full_nmpc_result_fatigue_optim["control"]["tau"][0],
    label="fatigue optim",
)

axs[1, 1].plot(
    full_nmpc_result_force_optim["time"][:-1],
    full_nmpc_result_force_optim["control"]["tau"][1],
    label="force optim",
)
axs[1, 1].plot(
    full_nmpc_result_fatigue_optim["time"][:-1],
    full_nmpc_result_fatigue_optim["control"]["tau"][1],
    label="fatigue optim",
)

axs[1, 1].legend()
plt.show()

# --- Second subplot for pulse duration and muscle force / fatigue --- #
fig, axs = plt.subplots(3, 2, figsize=(10, 10))
axs[0, 0].set_title("DeltoideusClavicle_A")
axs[0, 0].plot(
    full_nmpc_result_force_optim["time"],
    full_nmpc_result_force_optim["states"]["F_DeltoideusClavicle_A"],
    label="force optim",
)
axs[0, 0].plot(
    full_nmpc_result_force_optim["time"],
    full_nmpc_result_fatigue_optim["states"]["F_DeltoideusClavicle_A"],
    label="fatigue optim",
)
axs[0, 0].set_ylabel("Force (N)")
axs[0, 0].set_ylim(bottom=-20)
axs00 = axs[0, 0].twinx()
axs00.set_ylim(top=0.003)
axs00.set_yticks([0, 0.0003, 0.0006], [0, 300, 600])
mask1 = ma.where(
    pulse_duration_force_optim_dict["DeltoideusClavicle_A"] >= pulse_duration_fatigue_optim_dict["DeltoideusClavicle_A"]
)
mask2 = ma.where(
    pulse_duration_fatigue_optim_dict["DeltoideusClavicle_A"] >= pulse_duration_force_optim_dict["DeltoideusClavicle_A"]
)
axs00.bar(
    bar[mask1],
    pulse_duration_force_optim_dict["DeltoideusClavicle_A"][mask1],
    color="tab:blue",
    edgecolor="grey",
    width=barWidth,
    label=f"force optim pw",
)
axs00.bar(
    bar,
    pulse_duration_fatigue_optim_dict["DeltoideusClavicle_A"],
    color="tab:orange",
    edgecolor="grey",
    width=barWidth,
    label=f"force optim pw",
)
axs00.bar(
    bar[mask2],
    pulse_duration_force_optim_dict["DeltoideusClavicle_A"][mask2],
    color="tab:blue",
    edgecolor="grey",
    width=barWidth,
    label=f"force optim pw",
)

axs[0, 1].set_title("DeltoideusScapula_P")
axs[0, 1].plot(
    full_nmpc_result_force_optim["time"],
    full_nmpc_result_force_optim["states"]["F_DeltoideusScapula_P"],
    label="force optim",
)
axs[0, 1].plot(
    full_nmpc_result_force_optim["time"],
    full_nmpc_result_fatigue_optim["states"]["F_DeltoideusScapula_P"],
    label="fatigue optim",
)
axs[0, 1].set_ylim(bottom=-20)
axs01 = axs[0, 1].twinx()
axs01.set_ylim(top=0.003)
axs01.set_yticks([0, 0.0003, 0.0006], [0, 300, 600])
axs01.set_ylabel("Pulse duration (us)")
mask1 = ma.where(
    pulse_duration_force_optim_dict["DeltoideusScapula_P"] >= pulse_duration_fatigue_optim_dict["DeltoideusScapula_P"]
)
mask2 = ma.where(
    pulse_duration_fatigue_optim_dict["DeltoideusScapula_P"] >= pulse_duration_force_optim_dict["DeltoideusScapula_P"]
)
axs01.bar(
    bar[mask1],
    pulse_duration_force_optim_dict["DeltoideusScapula_P"][mask1],
    color="tab:blue",
    edgecolor="grey",
    width=barWidth,
    label=f"force optim pw",
)
axs01.bar(
    bar,
    pulse_duration_fatigue_optim_dict["DeltoideusScapula_P"],
    color="tab:orange",
    edgecolor="grey",
    width=barWidth,
    label=f"force optim pw",
)
axs01.bar(
    bar[mask2],
    pulse_duration_force_optim_dict["DeltoideusScapula_P"][mask2],
    color="tab:blue",
    edgecolor="grey",
    width=barWidth,
    label=f"force optim pw",
)

axs[1, 0].set_title("TRIlong")
axs[1, 0].plot(
    full_nmpc_result_force_optim["time"],
    full_nmpc_result_force_optim["states"]["F_TRIlong"],
    label="force optim",
)
axs[1, 0].plot(
    full_nmpc_result_force_optim["time"],
    full_nmpc_result_fatigue_optim["states"]["F_TRIlong"],
    label="fatigue optim",
)
axs[1, 0].set_ylabel("Force (N)")
axs[1, 0].set_ylim(bottom=-20)
axs10 = axs[1, 0].twinx()
axs10.set_ylim(top=0.003)
axs10.set_yticks([0, 0.0003, 0.0006], [0, 300, 600])
axs10.bar(
    bar,
    pulse_duration_force_optim_dict["TRIlong"],
    width=barWidth,
    edgecolor="grey",
    label=f"force optim pw",
)
axs10.bar(
    bar,
    pulse_duration_fatigue_optim_dict["TRIlong"],
    width=barWidth,
    edgecolor="grey",
    label=f"force optim pw",
)
mask1 = ma.where(pulse_duration_force_optim_dict["TRIlong"] >= pulse_duration_fatigue_optim_dict["TRIlong"])
mask2 = ma.where(pulse_duration_fatigue_optim_dict["TRIlong"] >= pulse_duration_force_optim_dict["TRIlong"])
axs10.bar(
    bar[mask1],
    pulse_duration_force_optim_dict["TRIlong"][mask1],
    color="tab:blue",
    edgecolor="grey",
    width=barWidth,
    label=f"force optim pw",
)
axs10.bar(
    bar,
    pulse_duration_fatigue_optim_dict["TRIlong"],
    color="tab:orange",
    edgecolor="grey",
    width=barWidth,
    label=f"force optim pw",
)
axs10.bar(
    bar[mask2],
    pulse_duration_force_optim_dict["TRIlong"][mask2],
    color="tab:blue",
    edgecolor="grey",
    width=barWidth,
    label=f"force optim pw",
)

axs[1, 1].set_title("BIC_long")
axs[1, 1].plot(
    full_nmpc_result_force_optim["time"],
    full_nmpc_result_force_optim["states"]["F_BIC_long"],
    label="force optim",
)
axs[1, 1].plot(
    full_nmpc_result_force_optim["time"],
    full_nmpc_result_fatigue_optim["states"]["F_BIC_long"],
    label="fatigue optim",
)
axs[1, 1].set_ylim(bottom=-20)
axs11 = axs[1, 1].twinx()
axs11.set_ylim(top=0.003)
axs11.set_yticks([0, 0.0003, 0.0006], [0, 300, 600])
axs11.set_ylabel("Pulse duration (us)")
mask1 = ma.where(pulse_duration_force_optim_dict["BIC_long"] >= pulse_duration_fatigue_optim_dict["BIC_long"])
mask2 = ma.where(pulse_duration_fatigue_optim_dict["BIC_long"] >= pulse_duration_force_optim_dict["BIC_long"])
axs11.bar(
    bar[mask1],
    pulse_duration_force_optim_dict["BIC_long"][mask1],
    color="tab:blue",
    edgecolor="grey",
    width=barWidth,
    label=f"force optim pw",
)
axs11.bar(
    bar,
    pulse_duration_fatigue_optim_dict["BIC_long"],
    color="tab:orange",
    edgecolor="grey",
    width=barWidth,
    label=f"force optim pw",
)
axs11.bar(
    bar[mask2],
    pulse_duration_force_optim_dict["BIC_long"][mask2],
    color="tab:blue",
    edgecolor="grey",
    width=barWidth,
    label=f"force optim pw",
)

axs[2, 0].set_title("BIC_brevis")
axs[2, 0].plot(
    full_nmpc_result_force_optim["time"],
    full_nmpc_result_force_optim["states"]["F_BIC_brevis"],
    label="force optim",
)
axs[2, 0].plot(
    full_nmpc_result_force_optim["time"],
    full_nmpc_result_fatigue_optim["states"]["F_BIC_brevis"],
    label="fatigue optim",
)
axs[2, 0].set_xlabel("Time (s)")
axs[2, 0].set_ylabel("Force (N)")
axs[2, 0].set_ylim(bottom=-20)
axs20 = axs[2, 0].twinx()
axs20.set_ylim(top=0.003)
axs20.set_yticks([0, 0.0003, 0.0006], [0, 300, 600])
mask1 = ma.where(pulse_duration_force_optim_dict["BIC_brevis"] >= pulse_duration_fatigue_optim_dict["BIC_brevis"])
mask2 = ma.where(pulse_duration_fatigue_optim_dict["BIC_brevis"] >= pulse_duration_force_optim_dict["BIC_brevis"])
axs20.bar(
    bar[mask1],
    pulse_duration_force_optim_dict["BIC_brevis"][mask1],
    color="tab:blue",
    edgecolor="grey",
    width=barWidth,
    label=f"force optim pw",
)
axs20.bar(
    bar,
    pulse_duration_fatigue_optim_dict["BIC_brevis"],
    color="tab:orange",
    edgecolor="grey",
    width=barWidth,
    label=f"force optim pw",
)
axs20.bar(
    bar[mask2],
    pulse_duration_force_optim_dict["BIC_brevis"][mask2],
    color="tab:blue",
    edgecolor="grey",
    width=barWidth,
    label=f"force optim pw",
)

axs[2, 1].set_title("General muscle fatigue")
full_nmpc_result_force_optim_fatigue = np.sum(
    [full_nmpc_result_force_optim["states"][f] for f in full_nmpc_result_force_optim["states"] if "A_" in f],
    axis=0,
)
full_nmpc_result_fatigue_optim_fatigue = np.sum(
    [full_nmpc_result_fatigue_optim["states"][f] for f in full_nmpc_result_fatigue_optim["states"] if "A_" in f],
    axis=0,
)
axs[2, 1].plot(
    full_nmpc_result_force_optim["time"],
    full_nmpc_result_force_optim_fatigue,
    label="force optim",
)
axs[2, 1].plot(
    full_nmpc_result_force_optim["time"],
    full_nmpc_result_fatigue_optim_fatigue,
    label="fatigue optim",
)
axs[2, 1].set_xlabel("Time (s)")
axs[2, 1].set_ylabel("Force scaling factor (N/s)")
axs21 = axs[2, 1].twinx()
axs21.set_yticks([0], ["      "])
axs21.set_ylabel("Pulse duration (us)")
axs[2, 1].legend()

plt.show()


# --- Pyorerun animation --- #
biorbd_model = biorbd.Model("../msk_models/simplified_UL_Seth_full_mesh.bioMod")
PickleAnimate("results/cycling_fes_driven_nmpc_full_force.pkl").animate(model=biorbd_model)

biorbd_model = biorbd.Model("../msk_models/simplified_UL_Seth_full_mesh.bioMod")
PickleAnimate("results/cycling_fes_driven_nmpc_full_fatigue.pkl").animate(model=biorbd_model)
