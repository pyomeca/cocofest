"""
This example will do a 10 stimulation example with Ding's 2007 pulse width and frequency model.
The stimulation will optimized the pulsation width between 0 and 0.0006 seconds to match a force value of 200N at the
end of the last node while minimizing the muscle force state.
"""

import numpy as np
from bioptim import Solver, OdeSolver, ObjectiveList, ObjectiveFcn
from cocofest import OcpFes, ModelMaker, FES_plot


def prepare_ocp():
    # --- Build ocp --- #
    final_time = 0.2
    model = ModelMaker.create_model(
        "ding2007_with_fatigue",
        sum_stim_truncation=10,
        stim_time=list(np.linspace(0, final_time, 11)[:-1]),
        previous_stim={"time": [-0.15, -0.10, -0.05], "pulse_width": [0.0005, 0.0005, 0.0005]},
    )

    # --- Minimize muscle force objective function --- #
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="F", weight=1, quadratic=True)

    # --- Prepare ocp --- #
    minimum_pulse_width = model.pd0
    return OcpFes().prepare_ocp(
        model=model,
        final_time=final_time,
        pulse_width={
            "min": minimum_pulse_width,
            "max": 0.0006,
            "bimapping": False,
        },
        objective={"end_node_tracking": 200, "custom": objective_functions},
        use_sx=True,
        n_threads=8,
        ode_solver=OdeSolver.RK4(n_integration_steps=10),
    )


def main():
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT())

    # --- Show results --- #
    FES_plot(data=sol).plot(title="Optimize pulse width", show_bounds=False, show_stim=False)


if __name__ == "__main__":
    main()
