"""
This example will do a 10 stimulation example with Ding's 2003 frequency model associated to Hmed's 2018 work.
This ocp was build to match a force value of 200N at the end of the last node.
"""

from cocofest import DingModelIntensityFrequency, OcpFes

# --- Build ocp --- #
# This ocp was build to match a force value of 200N at the end of the last node.
# The stimulation won't be optimized and is already set to one pulse every 0.1 seconds (n_stim/final_time).
# Plus the pulsation intensity will be optimized between 0 and 130 mA and are not the same across the problem.
minimum_pulse_intensity = DingModelIntensityFrequency.min_pulse_intensity(DingModelIntensityFrequency())
ocp = OcpFes().prepare_ocp(
    model=DingModelIntensityFrequency(),
    stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    n_shooting=100,
    final_time=1,
    pulse_intensity={
        "min": minimum_pulse_intensity,
        "max": 130,
        "bimapping": False,
    },
    objective={"end_node_tracking": 130},
    use_sx=True,
    n_threads=5,
)

# --- Solve the program --- #
sol = ocp.solve()

# --- Show results --- #
sol.graphs()
