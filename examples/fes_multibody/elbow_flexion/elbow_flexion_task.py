"""
Those ocp were build to move the elbow from 0 to 90 degrees angle.
The stimulation frequency will be set to 10Hz and pulse width will be optimized to satisfy the motion and to minimize the overall muscle fatigue.
Intensity can be optimized from sensitivity threshold to 600us. No residual torque is allowed.
"""

import numpy as np

from bioptim import (
    Node,
    ObjectiveFcn,
    ObjectiveList,
    ParameterList,
    OdeSolver,
    OptimalControlProgram,
    ControlType,
    BoundsList,
    InitialGuessList,
    ConstraintList,
)

from cocofest import (
    OcpFesMsk,
    FesMskModel,
    DingModelPulseIntensityFrequencyWithFatigue,
    ModelMaker,
    CustomObjective,
)


def prepare_ocp(
    model,
    final_time: float,
    max_bound: float,
    msk_info: dict = None,
    minimize_force: bool = False,
    minimize_fatigue: bool = False,
):

    muscle_model = model.muscles_dynamics_model[0]
    n_shooting = muscle_model.get_n_shooting(final_time)
    numerical_data_time_series, stim_idx_at_node_list = muscle_model.get_numerical_data_time_series(
        n_shooting, final_time
    )
    dynamics_options = OcpFesMsk.declare_dynamics_options(
        numerical_time_series=numerical_data_time_series, ode_solver=OdeSolver.RK4(n_integration_steps=10)
    )

    x_bounds, x_init = OcpFesMsk.set_x_bounds(model, msk_info)
    u_bounds, u_init = OcpFesMsk.set_u_bounds(model, msk_info["with_residual_torque"], max_bound=max_bound)

    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        key="qdot",
        index=[0, 1],
        node=Node.END,
        target=np.array([[0, 0]]).T,
        weight=100,
        quadratic=True,
        phase=0,
    )
    if minimize_force:
        objective_functions.add(
            CustomObjective.minimize_overall_muscle_force_production,
            custom_type=ObjectiveFcn.Lagrange,
            weight=1,
            quadratic=True,
        )
    if minimize_fatigue:
        objective_functions.add(
            CustomObjective.minimize_overall_muscle_fatigue,
            custom_type=ObjectiveFcn.Lagrange,
            weight=1,
            quadratic=True,
        )

    # --- Set parameters (required for intensity models) --- #
    use_sx = False
    if isinstance(muscle_model, DingModelPulseIntensityFrequencyWithFatigue):
        parameters, parameters_bounds, parameters_init = OcpFesMsk.build_parameters(
            model=model,
            max_pulse_intensity=int(max_bound),
            use_sx=use_sx,
        )
        constraints = OcpFesMsk.set_constraints(
            model,
            n_shooting,
            stim_idx_at_node_list,
        )
        model = OcpFesMsk.update_model(model, parameters=parameters, external_force_set=None)
    else:
        parameters, parameters_bounds, parameters_init = ParameterList(use_sx=use_sx), BoundsList(), InitialGuessList()
        constraints = ConstraintList()

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
        parameters=parameters,
        parameter_bounds=parameters_bounds,
        parameter_init=parameters_init,
        constraints=constraints,
        control_type=ControlType.CONSTANT,
        use_sx=use_sx,
        n_threads=20,
    )


def main(plot=True, model_path="../../msk_models/Arm26/arm26_biceps_triceps.bioMod"):
    # --- Define the fes model --- #
    fes_model_type = "ding2007_with_fatigue"
    stim_time = list(np.linspace(0, 1, 11))[:-1]
    biceps = ModelMaker.create_model(fes_model_type, muscle_name="BIClong", stim_time=stim_time, sum_stim_truncation=10)
    triceps = ModelMaker.create_model(
        fes_model_type, muscle_name="TRIlong", stim_time=stim_time, sum_stim_truncation=10
    )
    max_bound = (
        0.0006
        if fes_model_type == "ding2007_with_fatigue"
        else 130 if fes_model_type == "hmed2018_with_fatigue" else None
    )
    model = FesMskModel(
        name=None,
        biorbd_path=model_path,
        muscles_model=[biceps, triceps],
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=True,
        activate_residual_torque=False,
        stim_time=stim_time,
        with_contact=False,
    )

    final_time = 1
    ocp = prepare_ocp(
        model,
        final_time,
        max_bound,
        msk_info={
            "with_residual_torque": False,
            "bound_type": "start_end",
            "bound_data": [[0, 5], [0, 90]],
        },
        minimize_force=False,
        minimize_fatigue=True,
    )
    sol = ocp.solve()

    # --- Show results from solution --- #
    if plot:
        sol.graphs(show_bounds=True)
        sol.animate(viewer="pyorerun")


if __name__ == "__main__":
    main()
