<img align="right" width="400" src="docs/cocofest_logo.png">

# COCOFEST

`Cocofest` : Custom Optimal Control Optimization for Functional Electrical STimulation, is an optimal control program (OCP) package for functional electrical stimulation (FES).
It is based on the [bioptim](https://github.com/pyomeca/bioptim) framework for the optimal control construction.
Bioptim uses [biorbd](https://github.com/pyomeca/biorbd) a biomechanics library and benefits from the powerful algorithmic diff provided by [CasADi](https://web.casadi.org/).
To solve the OCP, the robust solver [Ipopt](https://github.com/coin-or/Ipopt) has been implemented.

## Status

| Type          | Status                                                                                                                                                                  |
|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| License       | <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-success" alt="License"/></a>                                           |
| Code coverage | [![codecov](https://codecov.io/github/Kev1CO/cocofest/graph/badge.svg?token=85XHAQGLWQ)](https://codecov.io/github/Kev1CO/cocofest)                                     |
| Code climate  | <a href="https://codeclimate.com/github/pyomeca/cocofest/maintainability"><img src="https://api.codeclimate.com/v1/badges/68cd9855a07be8bfb8c5/maintainability" /></a>  |

# Table of Contents 

[How to install Cocofest](#how-to-install)

<details>
<summary><a href="#available-fes-models">Available FES models</a></summary>

- [Ding2003](#ding2003)
- [Ding2007](#ding2007)
- [Hmed2018](#hmed2018)

</details>

[Create your own FES OCP](#create-your-own-fes-ocp)

[Examples](#examples)
- [Musculoskeletal model driven by FES models](#musculoskeletal-model-driven-by-FES-models)

<details>
<summary><a href="#other-functionalities">Other functionalities</a></summary>

- [Initial value problem](#initital-value-problem)
- [Summation truncation](#summation-truncation)

</details>

[Citing](#citing)


# How to install 
Currently, no anaconda installation is available. The installation must be done from the sources.
Cloning the repository is the first step to be able to use the package.

## Dependencies
`Cocofest` relies on several libraries. 
Follows the steps to install everything you need to use `Cocofest`.
</br>
First, you need to create a new conda environment
```bash
conda create -n YOUR_ENV_NAME python=3.11
```

Then, activate the environment
```bash
conda activate YOUR_ENV_NAME
```

This step will allow you to install the dependencies in the environment much quicker
```bash:
conda install -cconda-forge conda-libmamba-solver
```

After, install the dependencies
```bash
conda install numpy matplotlib pytest casadi biorbd pyorerun -cconda-forge --solver=libmamba
```

Finally, install the bioptim setup.py file located in your cocofest/external/bioptim folder
```bash
cd <path_up_to_cocofest_file>/external/bioptim
python setup.py install
```

You are now ready to use `Cocofest`!


# Available FES models
The available FES models are likely to increase so stay tune.
## Ding2003
Ding, J., Wexler, A. S., & Binder-Macleod, S. A. (2003).
Mathematical models for fatigue minimization during functional electrical stimulation.
Journal of Electromyography and Kinesiology, 13(6), 575-588.

## Ding2007
Ding, J., Chou, L. W., Kesar, T. M., Lee, S. C., Johnston, T. E., Wexler, A. S., & Binder‐Macleod, S. A. (2007).
Mathematical model that predicts the force–intensity and force–frequency relationships after spinal cord injuries.
Muscle & Nerve: Official Journal of the American Association of Electrodiagnostic Medicine, 36(2), 214-222.

## Hmed2018
Hmed, A. B., Bakir, T., Sakly, A., & Binczak, S. (2018, July).
A new mathematical force model that predicts the force-pulse amplitude relationship of human skeletal muscle.
In 2018 40th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC) (pp. 3485-3488). IEEE.


# Create your own FES OCP
You can create your own FES OCP by following the steps below:
1. Create a new python file
2. Import the desired model from `Cocofest` (e.g. Ding2003) and the fes_ocp class

```python
from cocofest import DingModelFrequency, OcpFes
```

3. Create your own optimal control problem by adding the stimulation pulse number, the number of shooting points,
the final simulation time, the objective function
(for this example, the force at the end of the simulation must be the closest to 100N), 
the minimum and maximum time between two stimulation pulse, the time bimapping
(If True, will act like a frequency at constant pulse interval).

```python
ocp = OcpFes().prepare_ocp(...,
                         n_stim=10,
                         n_shooting=20,
                         ...,)
```

4. Solve you OCP

```python
result = ocp.solve()
```

# Examples
You can find all the available examples in the [examples](https://github.com/pyomeca/cocofest/tree/main/examples) file.
## Musculoskeletal model driven by FES models
The following example is a musculoskeletal model driven by the Ding2007 FES model.
The objective function is to reach a 90° forearm position and 0° arm position at the movement end.
The stimulation last 1s and the stimulation frequency is 10Hz.
The optimized parameter are each stimulation pulse width.

<p align="center">
  <img width="500" src=docs/arm_flexion.gif>
</p>



# Other functionalities

## Initital value problem
You can also compute the models form initial value problem.
For that, use the IvpFes class to build the computed problem.

```python
from cocofest import IvpFes, DingModelFrequencyWithFatigue

fes_parameters = {"model": DingModelFrequencyWithFatigue(), "n_stim": 10}
ivp_parameters = {"n_shooting": 20, "final_time": 1}

ivp = IvpFes(fes_parameters, ivp_parameters)

result, time = ivp.integrate()
```

## Summation truncation
The summation truncation is an integer parameter that can be added to the model.
It will truncate the stimulation apparition list used for the calcium summation.
The integer number defines the stimulation number to keep prior this summation calculation (e.g only the 5 past stimulation will be included).

```python
ocp = OcpFes().prepare_ocp(model=DingModelFrequency(sum_stim_truncation=5))
```


# Citing
`Cocofest` is not yet published in a journal.
But if you use `Cocofest` in your research, please kindly cite this zenodo link [10.5281/zenodo.10427934](https://doi.org/10.5281/zenodo.10427934).

# Acknowledgements
The software development was supported by Ingénierie de technologies interactives en réadaptation [INTER #160 OptiStim](https://regroupementinter.com/fr/mandat/160-optistim/).
The Cocofest [logo](docs/cocofest_logo.png) has been design by [MaxMV](https://www.instagram.com/max_mv3/)
