<div align="center"><a name="readme"></a>

<p align="center"> 
  <img src="https://i.imgur.com/Gsj0JtN.jpeg" alt="">
</p>

# Cocofest

An Open-Source Python Package for Functional Electrical Stimulation (FES) Optimization in Optimal Control.<br/>
Supports predictive musculoskeletal simulation driven by FES, moving time horizon, and model identification.<br/>
"Prototype today’s FES to power tomorrow’s rehab."

[![Made-with-python](https://img.shields.io/badge/Made%20with-Python%203.11-1f425f.svg?style=for-the-badge)](https://www.python.org/)
[![OS](https://img.shields.io/badge/OS-windows%20%7C%20linux-blue?style=for-the-badge)]()
[![Last commit](https://img.shields.io/github/last-commit/pyomeca/cocofest.svg?style=for-the-badge)]()<br/>
[![Coverage](https://img.shields.io/codecov/c/github/pyomeca/cocofest?style=for-the-badge&logo=codecov&branch=main&token=GPBRI2U4CO)](https://codecov.io/gh/pyomeca/cocofest)
[![Maintainability](https://img.shields.io/badge/Maintainability-B-green?style=for-the-badge&logo=codeclimate)](https://qlty.sh/gh/pyomeca/projects/cocofest)
[![Tests](https://img.shields.io/github/actions/workflow/status/pyomeca/cocofest/run_tests.yml?branch=main&style=for-the-badge&label=Tests&logo=githubactions)](https://github.com/pyomeca/cocofest/actions/workflows/run_tests.yml)
[![Docs](https://img.shields.io/readthedocs/cocofest?style=for-the-badge&label=Docs&logo=readthedocs)](https://cocofest.readthedocs.io/en/latest/)<br/>
[![Discord](https://img.shields.io/discord/1340640457327247460.svg?label=chat&logo=discord&color=7289DA&style=for-the-badge)](https://discord.gg/s3g2ATpvDs)
[![Licence](https://img.shields.io/github/license/pyomeca/cocofest.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![JOSS](https://img.shields.io/badge/JOSS-10.21105%2Fjoss.10669-brightgreen?style=for-the-badge)](https://doi.org/10.21105/joss.10669)<br/>
</div>

<details>
  <summary><kbd>Table of contents</kbd></summary>

- [About](#about)
- [Installation](#installation)
- [Features](#features)
  - [Available FES models](#available-fes-models)
  - [Musculoskeletal model driven by FES](#musculoskeletal-model-driven-by-fes)
  - [Moving time horizons](#moving-time-horizons)
  - [Initial value problem](#initial-value-problem)
  - [Identification](#identification)
  - [Summation truncation](#summation-truncation)
- [Other](#other)
  - [Want to contribute?](#want-to-contribute)
  - [Collaborators and contributors](#collaborators-and-contributors)
  - [Citing](#citing)
  - [Cited in](#cited-in)
  - [Other related projects](#other-related-projects)
  - [Acknowledgements](#acknowledgements)

</details>

<a id="about"></a>
<h1 align="center">
  <img src="docs/assets/cocofest_logo.png" width="150" alt=""
       style="vertical-align: middle; margin-right: .5rem;" />
  About
</h1>

Functional electrical stimulation (FES) is a neurorehabilitation technique that promotes motor recovery after neurological injury.
By delivering coordinated electrical pulses to targeted muscles, FES elicits functional movements such as walking, reaching, and grasping.
Because responses to stimulation vary across individuals and muscle groups, most FES protocols still rely on empirically tuned parameters.
These settings can cause over stimulation, early muscle fatigue on-set, and reduce therapeutic gains.

Advanced control approaches like optimal control-driven FES can improve FES rehabilitation efficiency by personalizing stimulation parameters to a specific task and patient.
Therefore, we designed `Cocofest` (Custom Optimal COntrol for Functional Electrical STimulation) an open-source Python package for optimal control-driven FES.
`Cocofest` relies on [bioptim](https://github.com/pyomeca/bioptim), an optimal control program framework for biomechanics.
[bioptim](https://github.com/pyomeca/bioptim) uses [biorbd](https://github.com/pyomeca/biorbd) a biomechanics library, benefits from powerful algorithmic differentiation provided by [CasADi](https://web.casadi.org/)
and robust solver like [Ipopt](https://github.com/coin-or/Ipopt).
</br>

> \[!IMPORTANT]
>
> `Cocofest` has no clinical clearance and should not be used for rehabilitation purposes. </br>
> Don't forget to <a href="https://github.com/pyomeca/cocofest/stargazers"><img src="https://media2.dev.to/dynamic/image/width=1000,height=420,fit=cover,gravity=auto,format=auto/https%3A%2F%2Fthepracticaldev.s3.amazonaws.com%2Fi%2F2nn6mhp57inp6rdxarzt.png" align="center" width="80" alt="Star the repository"> </a>
> the repository to show your support and help us grow the community!

<p align="center"> 
  <img src="https://i.imgur.com/zXE9tC6.png" alt="">
</p>

<a id="installation"></a>
<h1 align="center">
  <img src="https://i.imgur.com/BEyEL43.png" width="100" alt=""
       style="vertical-align: middle; margin-right: .5rem;" />
  Installation
</h1>

> \[!IMPORTANT]
>
> `Cocofest` is currently **not available on Anaconda/PyPI**. The installation must be done from the sources,
> so **cloning the repository is the required first step** before running any command below:
> ```bash
> git clone https://github.com/pyomeca/cocofest.git
> cd cocofest
> ```

### ⚡ Quick installation
The repository contains an [`environment.yml`](environment.yml) file with all of `Cocofest`'s dependencies and a
[`pyproject.toml`](pyproject.toml) file for the package itself. The setup requires the following 3 commands from the
repository root:
```bash
conda env create -f environment.yml
conda activate cocofest
pip install -e .
```
`conda env create` reads `environment.yml` and creates an environment named `cocofest` with every
dependency in one go, and `pip install -e .` registers `Cocofest` as a regular importable package, so
`import cocofest` works from anywhere (not just from inside the `examples` folder), without any relative-pathing
issue.

### Manual installation
If you would rather pick your own environment name or dependency versions, follow these steps instead.
</br>
First, create a new conda environment
```bash
conda create -n YOUR_ENV_NAME python=3.11
```

Then, activate the environment
```bash
conda activate YOUR_ENV_NAME
```

After, install the dependencies
```bash
conda install numpy matplotlib casadi biorbd pyorerun scikit-optimize bioptim==3.4.0 -c conda-forge
```

Finally, from the root of the cloned repository, install `Cocofest` itself in editable mode
```bash
pip install -e .
```

You are now ready to use `Cocofest`!

> \[!NOTE]
> `pytest` is not part of `environment.yml`. If you want to run the tests or contribute to `Cocofest`, install the
> dev tools with `pip install -e ".[dev]"` or `conda install -c conda-forge pytest black codecov`.

### ▶️ Running an example in Cocofest
Every script in the [`examples`](examples) folder is directly runnable. For instance, the pulse width optimization
script under `getting_started` can be run with:
```bash
python examples/getting_started/optimization/pulse_width_optimization.py
```
See [`examples/README.md`](examples/README.md) for an index of every example folder and what each script demonstrates.

<p align="center"> 
  <img src="https://i.imgur.com/zXE9tC6.png" alt="">
</p>

<a id="features"></a>
<h1 align="center">
  <img src="https://i.imgur.com/OFVqYpy.png" width="100" alt=""
       style="vertical-align: middle; margin-right: .5rem;" />
  Features
</h1>

<a id="available-fes-models"></a>
## 📊 Available FES models
All models are implemented at the muscle actuator level, making them applicable to a wide range of problems regardless of the specific optimal control problem.

| **Model Name**  | **Citation**                                                                                                                                                                                                                                    | **Description / Focus**                                             |
| --------------- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| ------------------------------------------------------------------- |
| **Veltink1992** | Veltink, P. H., Chizeck, H. J., Crago, P. E., & El-Bialy, A. (1992). *Nonlinear joint angle control for artificially stimulated muscle*. IEEE Transactions on Biomedical Engineering, 39(4), 368–380.                                           | Nonlinear control of joint angles via electrical stimulation.       |
| **Veltink1992 + Riener1996** | Veltink et al. (1992), combined with the fatigue prediction from Riener, R., Quintern, J., & Schmidt, G. (1996). *Biomechanical model of the human knee evaluated by neuromuscular stimulation*. Journal of Biomechanics, 29(9), 1157–1167.  | Veltink1992 joint angle control, extended with Riener1996 fatigue prediction. |
| **Ding2003**    | Ding, J., Wexler, A. S., & Binder-Macleod, S. A. (2003). *Mathematical models for fatigue minimization during functional electrical stimulation*. Journal of Electromyography and Kinesiology, 13(6), 575–588.                                  | Focus on mathematical models for minimising fatigue.                |
| **Ding2007**    | Ding, J., Chou, L. W., Kesar, T. M., et al. (2007). *Mathematical model that predicts the force–intensity and force–frequency relationships after spinal cord injuries*. Muscle & Nerve, 36(2), 214–222.                                        | Predicts force–intensity and force–frequency responses post-SCI.    |
| **Marion2009**  | Marion, M. S., Wexler, A. S., Hull, M. L., & Binder-Macleod, S. A. (2009). *Predicting the effect of muscle length on fatigue during electrical stimulation*. Muscle & Nerve, 40(4), 573–581.                                                   | Examines muscle length impact on fatigue under stimulation.         |
| **Marion2013**  | Marion, M. S., Wexler, A. S., & Hull, M. L. (2013). *Predicting non-isometric fatigue induced by electrical stimulation pulse trains as a function of pulse duration*. Journal of NeuroEngineering and Rehabilitation, 10, 1–16.                | Predicts non-isometric fatigue based on pulse duration.             |
| **Hmed2018**    | Hmed, A. B., Bakir, T., Garnier, Y. M., Sakly, A., Lepers, R., & Binczak, S. (2018). An approach to a muscle force model with force-pulse amplitude relationship of human quadriceps muscles. Computers in Biology and Medicine, 101, 218-228.  | Models the relationship between pulse amplitude and force.          |

> \[!NOTE]
>
> Riener1996 alone is not implemented as a standalone model: only its fatigue prediction is available, combined with Veltink1992 (see `VeltinkRienerModelPulseIntensityWithFatigue`).

`Cocofest` gives access to every model above through the [`ModelMaker`](cocofest/models/model_maker.py) factory, using the following string keys. Each base key also has a `_with_fatigue` variant (adds fatigue dynamics) and, for Marion2009/Marion2013, a `_modified` variant (pulse width instead of frequency as control):

| **Key**                                   | **Class**                                            | **Control(s)**             |
| ------------------------------------------ | ----------------------------------------------------- | --------------------------- |
| `ding2003` / `ding2003_with_fatigue`        | `DingModelFrequency(WithFatigue)`                      | Frequency                   |
| `ding2007` / `ding2007_with_fatigue`        | `DingModelPulseWidthFrequency(WithFatigue)`            | Frequency, pulse width      |
| `hmed2018` / `hmed2018_with_fatigue`        | `DingModelPulseIntensityFrequency(WithFatigue)`        | Frequency, pulse intensity  |
| `marion2009` / `marion2009_with_fatigue`    | `Marion2009ModelFrequency(WithFatigue)`                | Frequency                   |
| `marion2009_modified` / `marion2009_modified_with_fatigue` | `Marion2009ModelPulseWidthFrequency(WithFatigue)` | Frequency, pulse width |
| `marion2013` / `marion2013_with_fatigue`    | `Marion2013ModelFrequency(WithFatigue)`                | Frequency                   |
| `marion2013_modified` / `marion2013_modified_with_fatigue` | `Marion2013ModelPulseWidthFrequency(WithFatigue)` | Frequency, pulse width |
| `veltink1992`                                | `VeltinkModelPulseIntensity`                          | Pulse intensity             |
| `veltink_and_riener1998`                     | `VeltinkRienerModelPulseIntensityWithFatigue`         | Pulse intensity             |

```python
from cocofest import ModelMaker

model = ModelMaker.create_model("ding2007_with_fatigue", stim_time=[0, 0.1, 0.2])
```

> \[!NOTE]
>
> It is possible to implement more FES models into Cocofest.
> Adventurous enough to code it by yourself, we are looking forward to read your [pull request](docs/contributing.md).
> Feel free to reach out on discord or submit an issue if you need help.

<a id="musculoskeletal-model-driven-by-fes"></a>
## 🦴 Musculoskeletal model driven by FES

In conventional Hill-type muscle model, muscle force ($F_m$) is the product of $a$ the muscle activation, $F_{max}$ the maximal
isometric muscle force, $f_l$ the force-length, $f_v$ the force-velocity and $f_{pas}$ the passive force-length relationship:
```math
F_m(t) = a(t)\, F_{\max}\, f_l(\tilde{l}_m)\, f_v(\tilde{v}_m) + f_{pas}(\tilde{l}_m)
```

`Cocofest` replaces $a(t)$ × $F_{max}$ by the force obtained using [FES models](#available-fes-models).
This approach allows motions driven-FES simulations, meanwhile benefiting from musculoskeletal model properties (e.g., muscle insertion, weight, inertial).

> \[!NOTE]
>
> Used force-length ($f_l$), force-velocity ($f_v$) and passive force-length ($f_{pas}$) are those published by [De Groot et al., (2016)](https://link.springer.com/article/10.1007/s10439-016-1591-9).
> Those relationships can be activated or not via the `activate_force_length_relationship`, `activate_force_velocity_relationship` and `activate_passive_force_relationship` arguments of the `FesMskModel` constructor (all default to `False`). Modification to the following [file](cocofest/models/hill_coefficients.py) can be done to have more/different relationships. 

### 💻 A short musculoskeletal FES-driven example
The following example displays a reaching task using the [Arm26](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53090607/Musculoskeletal+Models) model driven by the [Ding2007](https://onlinelibrary.wiley.com/doi/full/10.1002/mus.20806) FES model.

```math
\begin{aligned}
\min_{x(\cdot),\,u(\cdot)} \quad 
& \int_{0}^{T} \sum_{i=1}^{n} F_{m,i}(t)\,dt \\[4pt]
\text{s.t.:}\quad
& q_{\text{arm}}(t) \in [-0.5,\, 3.14], && \forall t \in [0,T],\\
& q_{\text{forearm}}(t) \in [0,\, 3.14], && \forall t \in [0,T],\\
& \text{(last node)}\;\; \|p_{\text{hand\_marker}}(T) - p_{\text{target\_marker}}\| \le \varepsilon,\\
& u(t) = \begin{bmatrix}
  \mathrm{pw}_{1}(t)\\ \mathrm{pw}_{2}(t)\\ \vdots\\ \mathrm{pw}_{n}(t)
\end{bmatrix}
\;\; \text{(pulse widths per muscle)}.
\end{aligned}
```

<br>

<p align="center">
  <img width="800" src=docs/assets/reaching.gif> <br>
  Figure 1: Motion performed for the reaching task and associated muscle force production.
</p>

> \[!NOTE]
>
> Solved in 6.7 second, computer with an AMD Ryzen Threadripper PRO 7965WXs x 48 processor. <br>
> Additional information: frequency = 30Hz, n_shooting = 30, step = 0.033s, final time = 1s, integration = Collocation radau method, polynomial_order = 3, solver = IPOPT.

You can find more examples of musculoskeletal model driven by FES in the following [file](https://github.com/pyomeca/cocofest/tree/main/examples/fes_multibody).

<a id="moving-time-horizons"></a>
## ⏳ Moving time horizons

For longer time span simulation and apprehend muscle fatigue apparition, `Cocofest` implements moving time horizons (MHE)
through the [`FesMhe`](cocofest/optimization/fes_mhe.py) (single muscle) and [`FesMheMsk`](cocofest/optimization/fes_mhe_multibody.py)
(musculoskeletal) classes. Each window is solved, then the horizon slides forward by re-using the previous stimulation
history to keep fatigue state continuous across windows.

```python
from cocofest import DingModelPulseWidthFrequencyWithFatigue, OcpFes, FesMhe

model = DingModelPulseWidthFrequencyWithFatigue(stim_time=[...], sum_stim_truncation=10)
dynamics_options = OcpFes.declare_dynamics_options(...)  # see full example for every argument

mhe = FesMhe(
    bio_model=model,
    dynamics=dynamics_options,
    cycle_len=cycle_len,
    cycle_duration=cycle_duration,
    n_cycles_simultaneous=n_cycles_simultaneous,
    n_cycles_to_advance=1,
    ...,
)


def update_functions(_mhe, cycle_idx, _sol):
    return cycle_idx < n_cycles  # keep sliding the window until n_cycles is reached


sol = mhe.solve_fes_mhe(update_functions, solver=..., total_cycles=n_cycles, cycle_solutions=...)
```

> \[!NOTE]
>
> See the full, runnable version in [`examples/getting_started/optimization/pulse_width_optimization_mhe.py`](examples/getting_started/optimization/pulse_width_optimization_mhe.py)
> (single muscle) or [`examples/fes_multibody/cycling/cycling_pulse_width_mhe.py`](examples/fes_multibody/cycling/cycling_pulse_width_mhe.py) (musculoskeletal hand-cycling, below).

### 💻 A short MHE hand cycling FES-driven example

```math
\begin{aligned}
\min_{x(\cdot),\,u(\cdot)} \quad 
& \int_{0}^{T} \sum_{i=1}^{n} F_{m,i}(t)\,dt \\[4pt]
\text{s.t.:}\quad
& q_{\text{arm}}(t) \in [0,\, 1.5], && \forall\, t \in [0,T],\\
& q_{\text{forearm}}(t) \in [0.5,\, 2.5], && \forall\, t \in [0,T],\\
& q_{\text{pedal}}(t) \in [0,\, 6.28], && \forall\, t \in [0,T],\\
& \text{(first node)}\;\; \left\|\,\mathrm{center}_{\text{wheel}}(t_{0}) -
\begin{bmatrix} 0.35 \\ 0 \end{bmatrix}\right\| \le \varepsilon,\\
& \text{(first node)}\;\; \left\|\,\dot{\mathrm{center}}_{\text{wheel}}(t_{0}) - 0\right\| \le \varepsilon,\\
& \text{(last node)}\;\; \left\|\,q_{\text{pedal}}(T) - 6.28\right\| \le \varepsilon,\\
& u(t) = \begin{bmatrix} \mathrm{pw}_{1}(t)\\ \mathrm{pw}_{2}(t)\\ \vdots\\ \mathrm{pw}_{n}(t) \end{bmatrix}
\;\; \text{(pulse widths per muscle)}.
\end{aligned}
```

<br>

<p align="center">
  <img width="800" src=docs/assets/cycling.gif> <br>
  Figure 2: Motion performed for the cycling task, muscle force contribution per section inspired by
  <a href="https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1581301/full">
    Quittmann et al. (2025)
  </a>
  and muscle force production above 10% of the maximal force.
</p>

> \[!NOTE]
>
> Solved in 1.02 second, computer with an AMD Ryzen Threadripper PRO 7965WXs x 48 processor. <br>
> Additional information: frequency = 30Hz, n_shooting = 60, step = 0.033s, final time = 2s, integration = Collocation radau method, polynomial_order = 3, solver = IPOPT, simultaneous turn per optimization = 2. 

<a id="initial-value-problem"></a>
## 🎯 Initial value problem

The initial value problem feature enables forward nonlinear dynamic integration to simulate the model’s behavior from given initial state and controls (i.e., series of pulse trains). 
This also permits comparison between FES models without using optimal control methods.

For that, the IvpFes class is used to build the problem.

```python
from cocofest import IvpFes, DingModelFrequencyWithFatigue

stim_time = list(range(10))  # 10 stimulations, one every second
fes_parameters = {"model": DingModelFrequencyWithFatigue(stim_time=stim_time)}
ivp_parameters = {"final_time": 10}

ivp = IvpFes(fes_parameters, ivp_parameters)

result, time = ivp.integrate()
```

<a id="identification"></a>
## 🔎 Identification

To personalize FES models to simulated or experimental force, `Cocofest` supports model identification using optimal
control, through the [`OcpFesId`](cocofest/optimization/fes_id_ocp.py) class. Model parameters are treated as
optimization parameters and identified by minimizing the difference between a tracked force (simulated with
[`IvpFes`](cocofest/integration/ivp_fes.py), or your own experimental data) and the model's predicted force.

```python
from cocofest import ModelMaker, OcpFesId, OcpFes
from cocofest.identification.identification_method import DataExtraction

model = ModelMaker.create_model("hmed2018", stim_time=stim_time, sum_stim_truncation=10)

# force_tracking: simulated (via IvpFes) or experimental (time, force) data to fit
force_at_node = DataExtraction.force_at_node_in_ocp(time, force, n_shooting, final_time)

x_bounds, x_init = OcpFesId.set_x_bounds(model=model, force_tracking=force_at_node)
u_bounds, u_init = OcpFesId.set_u_bounds(model=model, control_value=pulse_intensity_values, ...)

additional_key_settings = OcpFesId.set_default_values(model)
parameters, parameters_bounds, parameters_init = OcpFesId.set_parameters(
    parameter_to_identify=["a_rest", "km_rest", "tau1_rest", "tau2"],
    parameter_setting=additional_key_settings,
    use_sx=True,
)
OcpFesId.update_model_param(model, parameters)

ocp = OptimalControlProgram(bio_model=[model], x_bounds=x_bounds, x_init=x_init, u_bounds=u_bounds, u_init=u_init,
                             parameters=parameters, parameter_bounds=parameters_bounds, parameter_init=parameters_init, ...)
sol = ocp.solve()
identified_a_rest = sol.parameters["a_rest"][0]
```

> \[!NOTE]
>
> See the full, runnable version in [`examples/getting_started/identification/muscle_model_id.py`](examples/getting_started/identification/muscle_model_id.py),
> or [`examples/identification/force_model/`](examples/identification/force_model) for the Ding2003/Ding2007/Hmed2018 variants.

### 💻 A short model identification example

```math
\begin{aligned}
\min_{x(\cdot),\,p(\cdot)} \quad 
& \int_{0}^{T} \bigl(F(t) - F_{\text{sim/exp}}(t)\bigr)\,dt \\[4pt]
\text{s.t.:}\quad
& u(t) = \begin{bmatrix} \mathrm{pw}_{1}(t)\\ \mathrm{pw}_{2}(t)\\ \vdots\\ \mathrm{pw}_{n}(t) \end{bmatrix}
= \begin{bmatrix} \mathrm{pw}_{\text{exp}1}(t)\\ \mathrm{pw}_{\text{exp}2}(t)\\ \vdots\\ \mathrm{pw}_{\text{exp}n}(t) \end{bmatrix}.
\end{aligned}
```

<p align="center">
  <img width="800" src=docs/assets/identification.gif> <br>
  Figure 3: Model identification optimization
</p>

> \[!NOTE]
>
> Solved in 0.343 second, computer with an AMD Ryzen Threadripper PRO 7965WXs x 48 processor. <br>
> Additional information: frequency = 33Hz, n_shooting = 66, step = 0.03s, final time = 2s, integration = Runge-Kutta 4, integration steps = 10, solver = IPOPT. 

<a id="summation-truncation"></a>
## ✂️ Summation truncation

`Cocofest` also incorporates the recent numerical truncation method to speed up convergence.
This method limits the number of past stimulations considered in the dynamics to reduce the dependency on time-varying states.

```python
model = ModelMaker.create_model("ding2007", stim_time=stim_time, sum_stim_truncation=10)
```

> \[!TIP]
>
> To determine the value to use for `sum_stim_truncation`, you can refer to [Tiago et al. (2025)](https://inria.hal.science/hal-05165086/) or [Co et al. (2024)](https://umontreal.scholaris.ca/items/a94c280a-98dc-48ad-888c-e11c7c6a0d55).


<p align="center"> 
  <img src="https://i.imgur.com/zXE9tC6.png" alt="">
</p>

<a id="other"></a>
<h1 align="center">
  <img src="https://i.imgur.com/ksk1C7F.png" width="100" alt=""
       style="vertical-align: middle; margin-right: .5rem;" />
  Other
</h1>

<a id="want-to-contribute"></a>
## 🙌 Want to contribute?

We are always looking for new contributors to help us improve `Cocofest`. <br>
Feel free to check our [contributing guidelines](docs/contributing.md) to get started, and please read our
[code of conduct](docs/code_of_conduct.md) beforehand.

Don't know where to start? [Issues](https://github.com/pyomeca/cocofest/issues) tagged with "Good first issues" are a great place to begin!

<a id="collaborators-and-contributors"></a>
## 🤝 Collaborators and contributors

<a href="https://github.com/Kev1CO"><img src="https://avatars.githubusercontent.com/u/78259038?v=4" title="Kev1CO" width="50" height="50"></a>
<a href="https://github.com/Ipuch"><img src="https://avatars.githubusercontent.com/u/40755537?v=4" title="Ipuch" width="50" height="50"></a>
<a href="https://github.com/fmoissenet"><img src="https://avatars.githubusercontent.com/u/14890048?v=4" title="fmoissenet" width="50" height="50"></a>
<a href="https://github.com/mickaelbegon"><img src="https://avatars.githubusercontent.com/u/37242672?v=4" title="mickaelbegon" width="50" height="50"></a>
<a href="https://github.com/Florine353"><img src="https://avatars.githubusercontent.com/u/112490846?v=4" title="Florine353" width="50" height="50"></a>

<a id="citing"></a>
## 📝 Citing
If you use `Cocofest` in your research, please cite the associated paper:

```
Co et al., (2026). Cocofest: an Open-Source Python Package for Optimal-Control-Driven Functional Electrical Stimulation. Journal of Open Source Software, 11(124), 10669, https://doi.org/10.21105/joss.10669
```

<details>
  <summary><kbd>BibTeX</kbd></summary>

```bibtex
@article{Co2026,
  doi = {10.21105/joss.10669},
  url = {https://doi.org/10.21105/joss.10669},
  year = {2026},
  publisher = {The Open Journal},
  volume = {11},
  number = {124},
  pages = {10669},
  author = {Kevin Co and Pierre Puchaud and Florent Moissenet and Micka{\"e}l Begon},
  title = {Cocofest: an Open-Source Python Package for Optimal-Control-Driven Functional Electrical Stimulation},
  journal = {Journal of Open Source Software}
}
```
</details>

<a id="cited-in"></a>
## 📚 Cited in
> \[!NOTE]
> If you used `Cocofest` in your research, please let us know by submitting an issue or a pull request to add your publication to this list.

<a id="other-related-projects"></a>
## <img src="https://avatars.githubusercontent.com/u/36738416?s=200&v=4" width="25">  Other related projects <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRV1sVJMT3ICWt09iYCfGOXjQhhW1ryMehoBw&s" width="25"> 

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/pyomeca/bioptim"><img src="https://raw.githubusercontent.com/pyomeca/biorbd_design/main/logo_png/bioptim_full.png" alt="Bioptim" height="56">
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/pyomeca/biorbd"><img src="https://raw.githubusercontent.com/pyomeca/biorbd_design/main/logo_png/biorbd_full.png" alt="Biorbd" height="56"> 
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/pyomeca/pyorerun"><img src="docs/assets/Pyorerun.gif" alt="Pyomeca" height="56"> 
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/pyomeca/biobuddy"><img src="docs/assets/BioBuddy.gif" alt="Biobuddy" height="56">
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/s2mLab/pyScienceMode">
        <img src="https://raw.githubusercontent.com/s2mLab/pyScienceMode/main/docs/how_to_use_pysciencemode.png" alt="pyScienceMode" height="56">
      </a>
    </td>
  </tr>
</table>

<a id="acknowledgements"></a>
## 🙏 Acknowledgements

### 🌱 Funding
<a href="https://regroupementinter.com/fr/mandat/160-optistim/"><img src="https://regroupementinter.com/wp-content/uploads/2023/11/INTER_logo-600x208.png" title="INTER" width="100"></a>
&nbsp;&nbsp;&nbsp;
<a href="https://frq.gouv.qc.ca"><img src="https://cmqtr.qc.ca/wp-content/uploads/2022/09/FRQNT-logo-frqnt-COULEURS.png" title="FRQNT" width="100"></a>

### <img src="docs/assets/cocofest_logo.png" width="30"> Logo and assets design
<a href="https://www.instagram.com/maxmvpainting/"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQrO9f_QqjajojjcNJo8FDUdROLwLx-SQ7uVw&s" title="MaxMV" width="65"></a>

<p align="center"> 
  <img src="https://i.imgur.com/zXE9tC6.png" alt="">
</p>
