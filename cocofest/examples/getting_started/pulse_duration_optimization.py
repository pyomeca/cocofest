"""
This example will do a 10 stimulation example with Ding's 2007 pulse width and frequency model.
This ocp was build to match a force value of 200N at the end of the last node.
"""

from bioptim import Solver, OdeSolver
from cocofest import OcpFes, ModelMaker

# --- Build ocp --- #
# This ocp was build to match a force value of 200N at the end of the last node.
# The stimulation will be optimized between 0.01 to 0.1 seconds and are equally spaced (a fixed frequency).
# Plus the pulsation width will be optimized between 0 and 0.0006 seconds and are not the same across the problem.
# The flag with_fatigue is set to True by default, this will include the fatigue model

model = ModelMaker.create_model("ding2007_with_fatigue", is_approximated=False,
                                stim_time=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
                                previous_stim={"time": [-0.15, -0.10, -0.05],
                                               "pulse_width": [0.0005, 0.0005, 0.0005]})

minimum_pulse_width = model.pd0
ocp = OcpFes().prepare_ocp(
    model=model,
    final_time=0.5,
    pulse_width={
        "min": minimum_pulse_width,
        "max": 0.0006,
        "bimapping": True,
    },
    objective={"end_node_tracking": 100},
    use_sx=True,
    n_threads=5,
    ode_solver=OdeSolver.RK1(n_integration_steps=5),
)

# --- Solve the program --- #
sol = ocp.solve(Solver.IPOPT(_hessian_approximation="limited-memory"))
# --- Show results --- #
sol.graphs()
