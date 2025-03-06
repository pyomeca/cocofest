"""
This example will do a 10 stimulation example with Ding's 2003 frequency model.
This ocp was build to produce a elbow motion from 5 to 120 degrees.
The stimulation frequency is fixed at 10 Hz and the elbow torque control is optimized to satisfy the flexion.
"""

import numpy as np

from cocofest import DingModelFrequencyWithFatigue, OcpFesMsk, FesMskModel

from bioptim import OdeSolver, ObjectiveList, ObjectiveFcn, OptimalControlProgram, ControlType


def prepare_ocp(model, final_time: float, resistive_torque, msk_info):
    muscle_model = model.muscles_dynamics_model[0]
    n_shooting = muscle_model.get_n_shooting(final_time=final_time)
    numerical_time_series, external_force_set = OcpFesMsk.get_numerical_time_series_for_external_forces(
        n_shooting, resistive_torque
    )
    numerical_data_time_series, stim_idx_at_node_list = muscle_model.get_numerical_data_time_series(
        n_shooting, final_time
    )
    numerical_time_series.update(numerical_data_time_series)

    dynamics = OcpFesMsk.declare_dynamics(
        model, numerical_time_series=numerical_time_series, with_contact=resistive_torque["with_contact"]
    )

    # --- Set initial guesses and bounds for states and controls --- #
    x_bounds, x_init = OcpFesMsk.set_x_bounds(model, msk_info)
    u_bounds, u_init = OcpFesMsk.set_u_bounds(model, with_residual_torque=msk_info["with_residual_torque"])

    # --- Set objective functions --- #
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, quadratic=True)

    # rebuilding model for the OCP
    model = FesMskModel(
        name=model.name,
        biorbd_path=model.biorbd_path,
        muscles_model=model.muscles_dynamics_model,
        stim_time=model.muscles_dynamics_model[0].stim_time,
        previous_stim=model.muscles_dynamics_model[0].previous_stim,
        activate_force_length_relationship=model.activate_force_length_relationship,
        activate_force_velocity_relationship=model.activate_force_velocity_relationship,
        activate_residual_torque=model.activate_residual_torque,
        external_force_set=external_force_set,
    )

    return OptimalControlProgram(
        bio_model=[model],
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=final_time,
        objective_functions=objective_functions,
        x_init=x_init,
        x_bounds=x_bounds,
        u_init=u_init,
        u_bounds=u_bounds,
        control_type=ControlType.CONSTANT,
        use_sx=True,
        ode_solver=OdeSolver.RK4(n_integration_steps=10),
        n_threads=20,
    )


def main(plot=True, biorbd_path="../model_msk/arm26_biceps_1dof.bioMod"):
    simulation_ending_time = 1
    model = FesMskModel(
        name=None,
        biorbd_path=biorbd_path,
        muscles_model=[DingModelFrequencyWithFatigue(muscle_name="BIClong")],
        stim_time=list(np.linspace(0, 1, 11)[:-1]),
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=True,
        activate_residual_torque=True,
    )

    resistive_torque = {
        "Segment_application": "r_ulna_radius_hand",
        "torque": np.array([0, 0, -1]),
        "with_contact": False,
    }

    msk_info = {
        "bound_type": "start_end",
        "bound_data": [[5], [120]],
        "with_residual_torque": True,
    }

    ocp = prepare_ocp(
        model=model, final_time=simulation_ending_time, resistive_torque=resistive_torque, msk_info=msk_info
    )
    sol = ocp.solve()

    if plot:
        sol.animate(viewer="pyorerun", n_frames=1000)
        sol.graphs(show_bounds=False)


if __name__ == "__main__":
    main()
