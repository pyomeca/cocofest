"""
This example is used to compare the effect of the muscle force-length, force-velocity, and passive-force relationships.
"""

import numpy as np

import matplotlib.pyplot as plt

from bioptim import (
    SolutionMerge,
    OdeSolver,
    ObjectiveList,
    ObjectiveFcn,
    ParameterList,
    OptimalControlProgram,
    ControlType,
    ConstraintFcn,
    ConstraintList,
    Node,
)

from cocofest import DingModelPulseWidthFrequencyWithFatigue, OcpFesMsk, FesMskModel, CustomObjective, OcpFes


def prepare_ocp(model: FesMskModel, final_time: float, msk_info: dict, fixed_pw):

    muscle_model = model.muscles_dynamics_model[0]
    n_shooting = muscle_model.get_n_shooting(final_time)
    numerical_data_time_series, stim_idx_at_node_list = muscle_model.get_numerical_data_time_series(
        n_shooting, final_time
    )
    dynamics_options = OcpFes.declare_dynamics_options(
        numerical_time_series=numerical_data_time_series, ode_solver=OdeSolver.RK4(n_integration_steps=10)
    )

    x_bounds, x_init = OcpFesMsk.set_x_bounds(model, msk_info)
    u_bounds, u_init = OcpFesMsk.set_u_bounds(model, msk_info["with_residual_torque"], max_bound=0.0006)

    objective_functions = ObjectiveList()
    objective_functions.add(
        CustomObjective.minimize_overall_muscle_force_production,
        custom_type=ObjectiveFcn.Lagrange,
        weight=1,
        quadratic=True,
    )

    model = OcpFesMsk.update_model(model, parameters=ParameterList(use_sx=True), external_force_set=None)

    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.BOUND_CONTROL,
        key="last_pulse_width_BIClong",
        node=Node.ALL_SHOOTING,
        min_bound=np.array([fixed_pw]),
        max_bound=np.array([fixed_pw]),
    )

    return OptimalControlProgram(
        bio_model=[model],
        dynamics=dynamics_options,
        n_shooting=n_shooting,
        phase_time=final_time,
        objective_functions=objective_functions,
        x_init=x_init,
        x_bounds=x_bounds,
        u_init=u_init,
        u_bounds=u_bounds,
        constraints=constraints,
        control_type=ControlType.CONSTANT,
        use_sx=True,
        n_threads=20,
    )


# --- Main --- #
minimum_pulse_width = DingModelPulseWidthFrequencyWithFatigue().pd0

sol_list = []
sol_time = []

relationship_activation_dict = {
    "no_relationship": [False, False, False],
    "length": [True, False, False],
    "velocity": [False, True, False],
    "passive_force": [False, False, True],
    "length_velocity": [True, True, False],
    "length_passive_force": [True, False, True],
    "velocity_passive_force": [False, True, True],
    "all_relationship": [True, True, True],
}

keys = list(relationship_activation_dict.keys())
for key in keys:
    stim_time = np.linspace(0, 1, 11)[:-1]
    model = FesMskModel(
        name=None,
        biorbd_path="../../msk_models/Arm26/arm26_biceps_1dof.bioMod",
        muscles_model=[DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIClong")],
        stim_time=list(stim_time),
        activate_force_length_relationship=relationship_activation_dict[key][0],
        activate_force_velocity_relationship=relationship_activation_dict[key][1],
        activate_passive_force_relationship=relationship_activation_dict[key][2],
        activate_residual_torque=False,
    )

    ocp = prepare_ocp(
        model=model,
        final_time=1,
        msk_info={
            "bound_type": "start",
            "bound_data": [0],
            "with_residual_torque": False,
        },
        fixed_pw=0.00025,
    )

    sol = ocp.solve()
    sol_list.append(sol.stepwise_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES]))
    sol_time.append(sol.stepwise_time(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES]))

for i in range(len(sol_time)):
    key = keys[i]
    plt.plot(sol_time[i], np.degrees(sol_list[i]["q"][0]), label=key)

    joint_error = np.degrees(sol_list[-1]["q"][0][-1]) - np.degrees(sol_list[i]["q"][0][-1])
    print(f"Joint error for {key}: {joint_error} degrees")

plt.xlabel("Time (s)")
plt.ylabel("Angle (°)")
plt.legend()
plt.show()
