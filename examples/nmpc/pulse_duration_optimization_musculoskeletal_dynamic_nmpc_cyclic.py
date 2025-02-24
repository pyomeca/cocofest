"""
This example will do a nmpc of 10 stimulation example with Ding's 2007 frequency model.
This ocp was build to produce a elbow motion from 5 to 120 degrees.
The pulse width between minimal sensitivity threshold and 600us to satisfy the flexion and minimizing required elbow
torque control.
"""
import numpy as np
import biorbd
from bioptim import Solver, MultiCyclicNonlinearModelPredictiveControl, Solution, ObjectiveList, ObjectiveFcn, MultiCyclicCycleSolutions
from cocofest import (
    DingModelPulseWidthFrequencyWithFatigue,
    NmpcFesMsk,
    FesMskModel,
)

model = FesMskModel(
    name=None,
    biorbd_path="../msk_models/arm26_biceps_1dof.bioMod",
    muscles_model=[DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIClong")],
    stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
    activate_force_length_relationship=True,
    activate_force_velocity_relationship=True,
    activate_residual_torque=True,
    external_force_set=None,
)

minimum_pulse_width = DingModelPulseWidthFrequencyWithFatigue().pd0

objective_functions = ObjectiveList()
for i in [0, 100]:
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        key="q",
        weight=100000,
        index=[0],
        target=np.array([[3.14 / (180 / 5)]]).T,
        node=i,
        phase=0,
        quadratic=True,
    )
for i in [50]:
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        key="q",
        weight=100000,
        index=[0],
        target=np.array([[3.14 / (180 / 120)]]).T,
        node=i,
        phase=0,
        quadratic=True,
    )

nmpc_fes_msk = NmpcFesMsk
nmpc = nmpc_fes_msk.prepare_nmpc(
    model=model,
    cycle_duration=1,
    n_cycles_to_advance=1,
    n_cycles_simultaneous=3,
    pulse_width={
        "min": minimum_pulse_width,
        "max": 0.0006,
        "bimapping": False,
    },
    objective={"minimize_residual_torque": True,
               "custom": objective_functions},
    msk_info={
        # "bound_type": "start_end",
        # "bound_data": [[5], [120]],
        "with_residual_torque": True,
    },
    use_sx=False,
)

n_cycles = 2


def update_functions(_nmpc: MultiCyclicNonlinearModelPredictiveControl, cycle_idx: int, _sol: Solution):
    return cycle_idx < n_cycles  # True if there are still some cycle to perform


sol = nmpc.solve(
    update_functions,
    solver=Solver.IPOPT(),
    cycle_solutions=MultiCyclicCycleSolutions.ALL_CYCLES,
    get_all_iterations=True,
    cyclic_options={"states": {}},
)

biorbd_model = biorbd.Model("../msk_models/arm26_biceps_1dof.bioMod")
# sol.print_cost()

# from matplotlib import pyplot as plt
# from bioptim import SolutionMerge

# solution_time = sol[1][0].decision_time(to_merge=SolutionMerge.KEYS, continuous=True)
# solution_time = [float(j) for j in solution_time]
# solution_time_full = sol[0].decision_time(to_merge=SolutionMerge.KEYS, continuous=True)
# solution_time_full = [float(j) for j in solution_time_full]
#
# plt.plot(solution_time, sol[1][0].decision_states(to_merge=SolutionMerge.NODES)["Cn_BIClong"].squeeze(), label="CN1")
# plt.plot(solution_time, sol[1][0].decision_states(to_merge=SolutionMerge.NODES)["F_BIClong"].squeeze(), label="F1")
# plt.plot(solution_time, sol[1][1].decision_states(to_merge=SolutionMerge.NODES)["Cn_BIClong"].squeeze(), label="CN2")
# plt.plot(solution_time, sol[1][1].decision_states(to_merge=SolutionMerge.NODES)["F_BIClong"].squeeze(), label="F2")
# plt.plot(solution_time_full, sol[0].decision_states(to_merge=SolutionMerge.NODES)["Cn_BIClong"].squeeze(), label="CNfull")
# plt.plot(solution_time_full, sol[0].decision_states(to_merge=SolutionMerge.NODES)["F_BIClong"].squeeze(), label="Ffull")
# plt.legend()
# plt.show()

sol[1][0].graphs(show_bounds=True)
sol[1][1].graphs(show_bounds=True)
sol[1][0].animate(n_frames=100)
sol[0].graphs(show_bounds=True)
sol[0].animate(n_frames=100)

# sol.graphs(show_bounds=True)
# sol.animate(n_frames=200, show_tracked_markers=True)
