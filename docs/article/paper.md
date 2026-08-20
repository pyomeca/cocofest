---
title: "Cocofest: an Open-Source Python Package for Optimal-Control-Driven Functional Electrical Stimulation"

tags:
  - python
  - functional electrical stimulation
  - optimal control
  - moving time horizon
  - electrical pulse trains
  - biomedical engineering

authors:
  - name: Kevin Co
    orcid: 0009-0009-0248-3548
    affiliation: "1"
  - name: Pierre Puchaud
    orcid: 0000-0002-9335-630X
    affiliation: "1, 2"
  - name: Florent Moissenet
    orcid: 0000-0001-6479-1911
    affiliation: "1, 3, 4"
  - name: Mickaël Begon
    orcid: 0000-0002-4107-9160
    affiliation: "1"

affiliations:
  - name: Laboratoire de Simulation et Modélisation du Mouvement, Université de Montréal, Montréal, Québec, Canada 
    index: 1
  - name: Auctus, Inria, Centre de l’Université de Bordeaux, Talence, France
    index: 2
  - name: Biomechanics Laboratory, Geneva University Hospitals and University of Geneva, Geneva, Switzerland
    index: 3
  - name: Kinesiology Laboratory, Geneva University Hospitals and University of Geneva, Geneva, Switzerland
    index: 4

date: 06 september 2025
bibliography: paper.bib
---

# Summary

Functional electrical stimulation (FES) is a rehabilitation method intended to promote motor recovery notably after
neurological impairment. Applying coordinated electrical pulses to muscles elicits functional movements like walking,
reaching, and grasping. FES rehabilitation mostly relies on empirical settings, as responses to stimulation vary across
populations and muscles. Empirical settings often cause overstimulation and premature fatigue [@ibitoye2016strategies],
shortening rehabilitation sessions and diminishing therapeutic benefit. Consequently, advanced control approaches like
optimal-control-driven FES are gaining interest in personalizing and improving FES rehabilitation efficiency, meanwhile
delaying muscle fatigue [@co2025optimal]. To address this need, we designed `Cocofest` (Custom Optimal COntrol for
Functional Electrical STimulation), an open-source Python package for optimal-control-driven FES. `Cocofest` provides a
framework to generate personalized pulse trains (Fig. 1) based on nonlinear dynamics models for FES, for
several musculoskeletal models and motor tasks. The package includes over 10 examples, covering optimization of
FES-related pulse-train parameters (including frequency, pulse width, pulse intensity), FES model parameters
identification from in-vivo measurements, and long-duration predictive simulations.

![Pulse train parameters that can be optimized in Cocofest](pulse_train.png){ width=90% }


# Statement of Need

Since the pioneer study on optimal-control-driven FES [@hunt1997feedback], no code has been shared in the field,
limiting objective comparison and replicability across studies. The lack of open-source practice led to an absence of
consensus on how to choose nonlinear dynamics for FES, and which cost functions to use for dedicated clinical needs,
hindering standardization and cumulative progress [@co2025optimal]. To address these challenges and support collective
scientific progress, `Cocofest` fulfills the following four needs:

Firstly, the relationship between the pulse-train parameters (e.g., frequency, pulse width and intensity; Fig. 1) and
the resulting muscle force, joint torque, and muscle fatigue can be modeled with different nonlinear dynamics
[@ding2003mathematical; @veltink1992nonlinear]. Gathering them within a unified package would facilitate comparison for
more informed modeling choices.

Secondly, no study has compared different optimal control problem (OCP) formulations applied to FES, due to OCP
implementation challenges [@co2025optimal]. Easily customizable OCP formulation, involving objective functions, models,
and transcriptions is required to provide an adequate research framework. Having the possibility to switch between
various OCP transcriptions (e.g., direct collocation or direct multiple shooting) is essential when dealing with stiff
differential equations [@puchaud2023direct], often embedded in FES models. Muscle fatigue is the primary challenge in
FES. Enabling the development and comparison of different OCP formulations could help address research questions, yield
novel stimulation patterns and enhance fatigue reduction. Moreover, using receding-horizon estimation for longer 
simulations reduces the computational complexity [@qu2009computation] associated with time-varying dynamics (e.g., fatigue)
[@ding2003mathematical]. 

Thirdly, predictive simulations of FES-driven or FES-assisted motions (e.g., walking, cycling, reaching, and grasping)
require the coupling of FES models with the equations of motion as well as adequate muscle force-length-velocity
relationships. Predictive simulations are usually actuated through Hill-type muscle models [@wakeling2023review].
A package capable of replacing muscle actuation by FES models in multibody musculoskeletal models will allow us to
simulate realistic FES-driven tasks.

Fourthly, personalized rehabilitation strategy is required to facilitate the motor recovery. Therefore, identifying the
patient-specific muscle response to FES is a crucial step. Unfortunately, current complex identification methods are a
barrier to clinical translation [@le2010identification]. Providing a robust and customizable framework for the
development of more patient-friendly protocols would help to overcome this barrier.

Despite its potential, optimal-control-driven FES remains unadopted in clinical practice due to its low technology
readiness level [@co2025optimal]. `Cocofest` is a comprehensive package designed to bridge the gaps and foster clinical
adoption. It integrates nonlinear muscle dynamics dedicated to FES, manages muscle fatigue, interfaces FES with
musculoskeletal models, supports customizable cost functions and parameter identification routines. With the goal of
bringing this technology to patient care, we believe this package will contribute to the open-science effort. `Cocofest`
is expected to accelerate the increase of technology readiness level by strengthening knowledge foundation.


# State of the Field

Several open-source toolkits support optimal control computations for musculoskeletal biomechanics, such as:

- `OpenSim Moco` [@dembia2020opensim], a C++ OpenSim extension that enables motion tracking and prediction using efficient
direct-collocation formulations coupled to nonlinear programming solvers. 
- `SCONE` [@geijtenbeek2019], a C++/C predictive-simulation environment for human and animal motion that optimizes
neuromusculoskeletal controllers to achieve task-level objectives (e.g., stable walking at a target speed).
- `Bioptim` [@michaud2022bioptim], a Python optimal-control framework for biomechanics that supports both direct collocation
and multiple shooting, with flexible interfaces to nonlinear programming solvers.

However, these toolkits are not tailored for FES. They control muscle activation as a piecewise-linear/constant
excitation, whereas FES requires optimizing deliverable stimulation patterns under device and safety constraints. As a
result, they lack reusable, validated components for the stimulation-to-force pathway and fatigue/recovery dynamics,
limiting reproducible comparison of FES models and slowing translation to practical stimulation design. `Cocofest`
addresses this gap by implementing published FES models that can drive musculoskeletal models. This design supports
reproducible comparisons of FES modeling assumptions and accelerates prototyping of patient- and task-specific
stimulation optimization. `Cocofest` also includes utilities for model identification and receding-horizon optimization
to support FES research workflows.


# Software Design

`Cocofest` is a Python library that relies on Biorbd, a musculoskeletal physics engine [@michaud2021biorbd], and
Bioptim, an open-source optimization framework for biomechanical problems [@michaud2022bioptim]. Specifically, Bioptim
enables easy OCP customization including cost functions, bounds, constraints, transcription methods (e.g., direct
collocation), integration methods, and solving methods (e.g., full- and receding-horizon OCPs).

In a conventional Hill-type muscle model, muscle force ($F_m$) is the product of $a$ the muscle activation, $F_{\mathrm{max}}$ the
maximal isometric muscle force, $f_l$ the force-length, $f_v$ the force-velocity and $f_{\mathrm{pas}}$ the passive force-length
relationship: $F_m(t) = a(t)\, F_{\mathrm{max}}\, f_l(\tilde{l}_m)\, f_v(\tilde{v}_m) + f_{\mathrm{pas}}(\tilde{l}_m)$. `Cocofest`
replaces $a(t)$ × $F_{\mathrm{max}}$ by the force obtained using FES models. This approach allows motions driven–FES simulations,
meanwhile benefiting from musculoskeletal model properties (e.g., muscle insertion, inertial parameters).

`Cocofest` was developed to maintain a consistent structure between classes and functions to facilitate the OCP
customization and new FES model implementation. This shared interface promotes reproducible work and comparisons of
optimal-control-driven FES strategies.


# Research Impact Statement

`Cocofest` was developed to address several gaps in the literature, including the lack of systematic comparisons of FES
models and OCP formulations, accessible tools for FES model identification, and open-source software for reproducible 
research. It enables researchers to generate personalized stimulation patterns, compare alternative OCP formulations,
and simulate realistic FES-driven tasks. By providing a consistent software structure and clear documentation,
`Cocofest` aims to streamline research workflows and support translation toward FES rehabilitation applications.
Although the project is new and targets a niche domain, it already offers a shared, reproducible environment that can
foster discussion, collaboration, and broader adoption of open-source practices within the FES community, which is an
important step toward clinical translation of this technique [@co2025optimal].


# AI Usage Disclosure

The authors used ChatGPT only to improve the manuscript clarity and readability.
After using this tool/service, the authors reviewed and edited the content as needed and took full responsibility for
the content of the publication.

GitHub Copilot and ChatGPT were used to assist in code refactoring and documentation.
Authors made all the core design and architectural decisions.


# Acknowledgements

The package development was supported by the Fonds de recherche du Québec – Nature et technologies (FRQNT, Grant 341023)
and by the FRQ strategic group in Ingénierie de technologies interactives en réadaptation (INTER #160 OptiStim).

# References