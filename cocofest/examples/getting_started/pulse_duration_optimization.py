"""
This example will do a 10 stimulation example with Ding's 2007 pulse width and frequency model.
This ocp was build to match a force value of 200N at the end of the last node.
"""
import numpy as np
from bioptim import Solver, OdeSolver
from cocofest import OcpFes, ModelMaker, ModifiedOdeSolverRK4, FES_plot

# --- Build ocp --- #
# This ocp was build to match a force value of 200N at the end of the last node.
# The stimulation will optimized the pulsation width between 0 and 0.0006 seconds and the pulse width will be
# similar across the problem "bimapping": True,".

final_time = 1
model = ModelMaker.create_model("ding2007_with_fatigue", is_approximated=False, sum_stim_truncation=10,
                                stim_time=list(np.linspace(0, final_time, 34)[:-1]),
                                previous_stim={"time": [-0.15, -0.10, -0.05],
                                               "pulse_width": [0.0005, 0.0005, 0.0005]})

minimum_pulse_width = model.pd0
ocp = OcpFes().prepare_ocp(
    model=model,
    final_time=final_time,
    pulse_width={
        "min": minimum_pulse_width,
        "max": 0.0006,
        "bimapping": True,
    },
    objective={"end_node_tracking": 200},
    use_sx=True,
    n_threads=5,
    ode_solver=ModifiedOdeSolverRK4(n_integration_steps=10),
    # ode_solver=OdeSolver.RK1(n_integration_steps=10),
)

# --- Solve the program --- #
sol = ocp.solve(Solver.IPOPT(_hessian_approximation="limited-memory"))

# --- Show results --- #
FES_plot(data=sol).plot(title="Optimize pulse width", show_bounds=False, show_stim=False)
