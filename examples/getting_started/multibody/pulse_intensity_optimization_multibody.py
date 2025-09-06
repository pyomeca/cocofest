"""
This example will do a 10 stimulation example with Ding's 2003 frequency model associated to Hmed's 2018 work.
This ocp was build to produce a elbow motion from 5 to 120 degrees.
The stimulation frequency will be optimized between 10 and 100 Hz and pulse intensity between minimal sensitivity
threshold and 130mA to satisfy the flexion and minimizing required elbow torque control.
"""

import numpy as np

from bioptim import (
    Solver,
    OdeSolver,
    ObjectiveList,
    ObjectiveFcn,
    OptimalControlProgram,
    ControlType,
)
from cocofest import DingModelPulseIntensityFrequencyWithFatigue, OcpFesMsk, FesMskModel, CustomObjective


def prepare_ocp(model: FesMskModel, final_time: float, external_force: dict, msk_info: dict, pi_max: int = 130):
    muscle_model = model.muscles_dynamics_model[0]
    n_shooting = muscle_model.get_n_shooting(final_time)
    numerical_time_series, external_force_set = OcpFesMsk.get_numerical_time_series_for_external_forces(
        n_shooting=n_shooting, external_force_dict=external_force
    )

    numerical_data_time_series, stim_idx_at_node_list = muscle_model.get_numerical_data_time_series(
        n_shooting, final_time
    )

    if external_force:
        numerical_time_series.update(numerical_data_time_series)
    else:
        numerical_time_series = numerical_data_time_series

    dynamics = OcpFesMsk.declare_dynamics(
        model,
        numerical_time_series=numerical_time_series,
        ode_solver=OdeSolver.RK4(n_integration_steps=10),
        contact_type=[],
    )

    x_bounds, x_init = OcpFesMsk.set_x_bounds(model, msk_info)
    u_bounds, u_init = OcpFesMsk.set_u_bounds(model, msk_info["with_residual_torque"], max_bound=pi_max)

    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
        key="tau",
        weight=1000,
        quadratic=True,
    )
    objective_functions.add(
        CustomObjective.minimize_overall_muscle_force_production,
        custom_type=ObjectiveFcn.Lagrange,
        weight=1,
        quadratic=True,
    )

    # --- Set parameters (required for intensity models) --- #
    use_sx = True
    parameters, parameters_bounds, parameters_init = OcpFesMsk.build_parameters(
        model=model,
        max_pulse_intensity=pi_max,
        use_sx=use_sx,
    )

    # --- Set constraints (required for intensity models) --- #
    constraints = OcpFesMsk.set_constraints(
        model,
        n_shooting,
        stim_idx_at_node_list,
    )

    model = OcpFesMsk.update_model(model, parameters=parameters, external_force_set=external_force_set)

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
        parameters=parameters,
        parameter_bounds=parameters_bounds,
        parameter_init=parameters_init,
        constraints=constraints,
        control_type=ControlType.CONSTANT,
        use_sx=True,
        n_threads=20,
    )


def main(plot=True, biorbd_path="../../msk_models/Arm26/arm26_biceps_1dof.bioMod"):
    simulation_ending_time = 1
    model = FesMskModel(
        name=None,
        biorbd_path=biorbd_path,
        muscles_model=[DingModelPulseIntensityFrequencyWithFatigue(muscle_name="BIClong")],
        stim_time=list(np.linspace(0, simulation_ending_time, 34)[:-1]),
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=True,
        activate_residual_torque=True,
        external_force_set=None,  # External forces will be added later
    )

    resistive_torque = {
        "Segment_application": "r_ulna_radius_hand",
        "torque": np.array([0, 0, -1]),
        "with_contact": False,
    }

    msk_info = {
        "with_residual_torque": True,
        "bound_type": "start_end",
        "bound_data": [[5], [120]],
    }

    ocp = prepare_ocp(
        model=model,
        pi_max=130,
        final_time=simulation_ending_time,
        external_force=resistive_torque,
        msk_info=msk_info,
    )

    sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=2000))

    if plot:
        sol.animate(viewer="pyorerun")
        sol.graphs(show_bounds=False)


if __name__ == "__main__":
    main()
