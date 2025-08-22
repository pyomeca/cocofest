<div align="center"><a name="readme-top"></a>

<p align="center"> 
  <img src="docs/image2.svg" alt="HAR Logo">
</p>

# Cocofest

An Open-Source Python Package for Functional Electrical Stimulation Optimization in Optimal Control.<br/> 
Supports predictive musculoskeletal simulation driven by FES, initial value problems, and FES model identification.

[![Made-with-python](https://img.shields.io/badge/Made%20with-Python%203.11-1f425f.svg?style=for-the-badge)](https://www.python.org/)
[![OS](https://img.shields.io/badge/OS-windows%20%7C%20linux-blue?style=for-the-badge)]()
[![Last commit](https://img.shields.io/github/last-commit/pyomeca/cocofest.svg?style=for-the-badge)]()

[![Coverage](https://img.shields.io/codecov/c/github/pyomeca/cocofest?style=for-the-badge&logo=codecov&branch=main&token=GPBRI2U4CO)](https://codecov.io/gh/pyomeca/cocofest)
[![Maintainability](https://img.shields.io/badge/Maintainability-B-green?style=for-the-badge&logo=codeclimate)](https://qlty.sh/gh/pyomeca/projects/cocofest)
[![Tests](https://img.shields.io/github/actions/workflow/status/pyomeca/cocofest/run_tests_win.yml?branch=main&style=for-the-badge&label=Tests&logo=githubactions)](https://github.com/pyomeca/cocofest/actions/workflows/run_tests_win.yml)

[![Discord](https://img.shields.io/discord/1340640457327247460.svg?label=chat&logo=discord&color=7289DA&style=for-the-badge)](https://discord.gg/Ux7BkdjQFW)
[![Licence](https://img.shields.io/github/license/pyomeca/cocofest.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

# About Cocofest

<img src="docs/cocofest_logo.png" align="right" alt="" width="300">

`Cocofest` : Custom Optimal Control Optimization for Functional Electrical STimulation, is an optimal control program (OCP) package for functional electrical stimulation (FES).
It is based on the [bioptim](https://github.com/pyomeca/bioptim) framework for the optimal control construction.
Bioptim uses [biorbd](https://github.com/pyomeca/biorbd) a biomechanics library and benefits from the powerful algorithmic diff provided by [CasADi](https://web.casadi.org/).
To solve the OCP, the robust solver [Ipopt](https://github.com/coin-or/Ipopt) has been implemented.


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
## Veltink1992
Veltink, P. H., Chizeck, H. J., Crago, P. E., & El-Bialy, A. (1992).
Nonlinear joint angle control for artificially stimulated muscle.
IEEE Transactions on Biomedical Engineering, 39(4), 368-380.

## Riener1998
Riener, R., & Veltink, P. H. (1998).
A model of muscle fatigue during electrical stimulation.
IEEE Transactions on Biomedical Engineering, 45(1), 105-113.

## Ding2003
Ding, J., Wexler, A. S., & Binder-Macleod, S. A. (2003).
Mathematical models for fatigue minimization during functional electrical stimulation.
Journal of Electromyography and Kinesiology, 13(6), 575-588.

## Ding2007
Ding, J., Chou, L. W., Kesar, T. M., Lee, S. C., Johnston, T. E., Wexler, A. S., & Binder‐Macleod, S. A. (2007).
Mathematical model that predicts the force–intensity and force–frequency relationships after spinal cord injuries.
Muscle & Nerve: Official Journal of the American Association of Electrodiagnostic Medicine, 36(2), 214-222.

## Marion2009
Marion, M. S., Wexler, A. S., Hull, M. L., & Binder‐Macleod, S. A. (2009).
Predicting the effect of muscle length on fatigue during electrical stimulation.
Muscle & Nerve: Official Journal of the American Association of Electrodiagnostic Medicine, 40(4), 573-581.

## Marion2013
Marion, M. S., Wexler, A. S., & Hull, M. L. (2013).
Predicting non-isometric fatigue induced by electrical stimulation pulse trains as a function of pulse duration.
Journal of neuroengineering and rehabilitation, 10, 1-16.

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


## Contributors

[//]: contributor-picture

<a href="https://github.com/Kev1CO"><img src="https://avatars.githubusercontent.com/u/78259038?v=4" title="Kev1CO" width="50" height="50"></a>
<a href="https://github.com/Ipuch"><img src="https://avatars.githubusercontent.com/u/40755537?v=4" title="Ipuch" width="50" height="50"></a>
<a href="https://github.com/Florine353"><img src="https://avatars.githubusercontent.com/u/112490846?v=4" title="Florine353" width="50" height="50"></a>

[//]: contributor-picture

# Citing
`Cocofest` is not yet published in a journal.
But if you use `Cocofest` in your research, please kindly cite this zenodo link [10.5281/zenodo.10427934](https://doi.org/10.5281/zenodo.10427934).



# <img src="https://avatars.githubusercontent.com/u/36738416?s=200&v=4" width="25">  Other Pyomeca projects

[//]: other-projects

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/pyomeca/bioptim">
        <img src="https://raw.githubusercontent.com/pyomeca/biorbd_design/main/logo_png/bioptim_full.png" alt="Bioptim logo" height="56"><br>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/pyomeca/biorbd">
        <img src="https://raw.githubusercontent.com/pyomeca/biorbd_design/main/logo_png/biorbd_full.png" alt="Biorbd logo" height="56"><br>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/pyomeca/pyorerun">
        <video
          src="https://github.com/Kev1CO/cocofest/simulation_processing/docs/assets/pyorerun.mp4"
          width="120"
          autoplay
          muted
          loop
          playsinline
          preload="metadata">
          <!-- Fallback text -->
          Your browser does not support the video tag.
        </video><br>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/pyomeca/biobuddy">
        <img src="https://github.com/Kev1CO/cocofest/blob/simulation_processing/docs/assets/BioBuddy.gif" alt="Biobuddy logo" height="56"><br>
      </a>
    </td>
  </tr>
</table>

[//]: other-projects

# Acknowledgements
The software development was supported by Ingénierie de technologies interactives en réadaptation [INTER #160 OptiStim](https://regroupementinter.com/fr/mandat/160-optistim/).
The Cocofest [logo](docs/cocofest_logo.png) has been design by [MaxMV](https://www.instagram.com/maxmvpainting/)
