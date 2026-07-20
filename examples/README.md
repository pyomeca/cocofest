# Cocofest examples

Every script below is directly runnable, e.g.:
```bash
python examples/getting_started/optimization/pulse_width_optimization.py
```
Each script anchors its own default asset paths (`.bioMod` files, etc.) to its own location on disk (`Path(__file__)`), so they can be run from anywhere, not just from inside their own folder.

## [`getting_started/`](getting_started)
Minimal, single-muscle examples. Start here.

- **`optimization/`** — Optimal control (OCP) examples using the [Ding2007](../README.md#available-fes-models) model unless noted:
  - `force_tracking.py` — pulse width optimization to track a target force curve.
  - `frequency_optimization.py` — Ding2003 model, matches a target end force with a fixed 1 Hz stimulation (no optimization on stimulation itself).
  - `pulse_intensity_optimization.py` — Hmed2018 model, pulse intensity optimized to match a target end force.
  - `pulse_width_optimization.py` — pulse width optimized to match a target end force while minimizing muscle force.
  - `pulse_width_optimization_mhe.py` — same problem solved with a **moving time horizon (MHE)**, using [`FesMhe`](../cocofest/optimization/fes_mhe.py).
- **`initial_value_problem/`** — Forward simulation (no optimization) via [`IvpFes`](../cocofest/integration/ivp_fes.py):
  - `model_integration.py` — integrates `DingModelFrequencyWithFatigue` for 300 s to show fatigue's effect on force over time.
  - `pulse_mode_example.py` — compares `single`/`doublet`/`triplet` pulse modes on the same model.
- **`identification/`**:
  - `muscle_model_id.py` — simulates then identifies a Hmed2018 pulse-intensity model's parameters using [`OcpFesId`](../cocofest/optimization/fes_id_ocp.py) and [`DataExtraction`](../cocofest/identification/identification_method.py).
- **`multibody/`** — Same pulse width/intensity optimizations as above, but driving a musculoskeletal arm model through [`FesMskModel`](../cocofest/models/dynamical_model.py):
  - `pulse_width_optimization_multibody.py`, `pulse_intensity_optimization_multibody.py`.

## [`identification/`](identification)
- **`force_model/`** — Parameter identification (simulate then re-identify) for `ding2003_model_id.py`, `ding2007_model_id.py`, `hmed2018_model_id.py`.
- **`fatigue_model/WIP.py`** — placeholder; fatigue model identification is **not implemented yet** (work in progress, see [contributing guide](../docs/contributing.md) if you want to pick this up).

## [`other_fes_models/`](other_fes_models)
One self-contained example per FES model not already covered above: `marion2009_example.py`, `marion2013_example.py`, `veltink1992_example.py` (also demonstrates the combined `VeltinkRienerModelPulseIntensityWithFatigue`).

## [`fes_multibody/`](fes_multibody)
Full musculoskeletal, FES-driven motion examples:
- **`reaching/reaching_task.py`** — Arm26 reaching task (see [README](../README.md#musculoskeletal-model-driven-by-fes)).
- **`elbow_flexion/`** — `elbow_flexion_task.py` (pulse width) and `frequency_optimization_multibody.py` (frequency + pulse intensity).
- **`shoulder_abduction/`** — `abduction_fes_driven.py` vs `abduction_muscle_driven.py`: same task, FES-driven vs. classic muscle-driven for comparison.
- **`cycling/`** — hand-cycling task, the most involved example set:
  - `cycling_inverse_kinematics.py` — warm-starts the cycling motion with inverse kinematics/dynamics.
  - `cycling_with_different_driven_methods.py` — same cycling task, switchable between `torque_driven`/`muscle_driven`/`fes_driven` dynamics.
  - `cycling_pulse_width_mhe.py` — hand-cycling MHE, uses [`FesMheMsk`](../cocofest/optimization/fes_mhe_multibody.py) (see [README](../README.md#moving-time-horizons)).
  - `cycling_bayesian_mhe.py` — Bayesian optimization of per-muscle cost-function weights on top of `cycling_pulse_width_mhe.py`, requires `scikit-optimize`.
  - `cycling_standard_clinics.py` — simulates how long a standard (non-optimized) clinical stimulation protocol can be sustained before fatigue failure.
  - `physiological_weight_calculation.py`, `cost_functions.py` — helper modules (muscle scaling, custom cost functions) imported by the scripts above, not meant to be run directly.
  
## [`sensitivity/`](sensitivity)
- `force_length_velocity/muscle_relationships_comparison.py` — compares enabling/disabling the force-length, force-velocity and passive-force relationships (see [README](../README.md#musculoskeletal-model-driven-by-fes)).

## [`msk_models/`](msk_models)
- `model_viewer.py` — live-animates a `.bioMod` model with slider-controlled joint angles.
- `Arm26/`, `Seth/`, `Wu/` — the `.bioMod` musculoskeletal model files used as defaults across the examples above.
