import biorbd
from cocofest import PickleAnimate

biorbd_model = biorbd.Model("../../../msk_models/arm26.bioMod")
PickleAnimate("../result_file/pulse_duration_minimize_muscle_fatigue.pkl").animate(
    model=biorbd_model
)
PickleAnimate(
    "../result_file/pulse_duration_minimize_muscle_fatigue.pkl"
).multiple_animations(
    ["../result_file/pulse_duration_minimize_muscle_force.pkl"], model=biorbd_model
)
