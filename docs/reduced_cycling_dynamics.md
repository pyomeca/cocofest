# Reduced cycling mechanics with variable crank speed

## Scope

The Wu cycling model has three mechanical generalized coordinates and two
holonomic constraints fixing the crank centre in the horizontal plane. Its
admissible mechanical manifold therefore has one degree of freedom.

The implementation in `cocofest/dynamics/reduced_cycling.py` parameterizes that
manifold with the physical, unwrapped crank angle `theta`. It does **not**
prescribe constant crank speed:

```text
theta_dot = omega
omega_dot = reduced_acceleration(theta, omega, muscle_forces, external_torque)
```

The existing `wheel_rotation_RotZ` coordinate is a relative joint coordinate
in the segment chain. It completes one turn when the arm posture is periodic,
but its instantaneous derivative is not exactly the physical angular velocity
of the crank. The reduced model obtains physical crank phase from the vector
between `global_wheel_center` and `hand`.

## Offline construction

For each crank angle, the builder solves three equations:

1. horizontal `wheel_center - global_wheel_center`;
2. vertical `wheel_center - global_wheel_center`;
3. alignment of the hand-to-centre vector with the requested crank phase.

Continuation from one angle to the next preserves the elbow branch. The full
configuration is fitted as

```text
q(theta) = winding * progress(theta) + periodic_Fourier_residual(theta)
```

This representation preserves the unwrapped wheel coordinate while keeping
the arm coordinates periodic. It also provides analytical first and second
derivatives:

```text
qdot = dq/dtheta * omega
qddot = dq/dtheta * omega_dot + d2q/dtheta2 * omega^2
```

The constrained multibody equations are projected on the admissible tangent
`T = dq/dtheta`:

```text
M_eff = T' M T

omega_dot =
    (
        sum(r_eff_m(theta) * F_m)
        + r_ext(theta) * tau_ext
        - g_eff(theta)
        - c_eff(theta) * omega^2
    ) / M_eff(theta)
```

The effective inertia, muscle effectiveness, external-torque effectiveness,
gravity and velocity-quadratic terms are fitted with smooth periodic Fourier
series. The same profile can be evaluated with NumPy or emitted as a CasADi
expression for IPOPT, MadNLP, Fatrop or ACADOS.

## Validation command

```bash
python examples/fes_multibody/cycling/validate_reduced_cycling_dynamics.py \
  --samples 181 \
  --kinematic-order 12 \
  --dynamics-order 12 \
  --validation-samples 200 \
  --external-crank-torque -0.2 \
  --casadi-profile \
  --casadi-repeats 1000 \
  --output-profile result/reduced-cycling.npz \
  --output-json result/reduced-cycling-validation.json
```

The profile is stored as plain NumPy arrays without pickled biorbd or solver
objects.

## Initial numerical result

On the Wu model, using 181 construction samples and Fourier order 12:

- maximum contact-manifold residual: approximately `2.8e-13`;
- maximum cycle-closure error: approximately `2.1e-12 rad`;
- maximum fitted configuration error: approximately `5.5e-9 rad`;
- maximum crank-acceleration error over 100 random validation points:
  approximately `1.1e-2 rad/s^2`;
- P95 relative crank-acceleration error: approximately `1.0e-5`;
- isolated numerical mechanical-kernel speedup: approximately `20x`.

For a like-for-like scalar CasADi kernel computing crank acceleration, its
Jacobian and its Hessian:

- full constrained expression: approximately `35,408` CasADi instructions;
- reduced expression: approximately `3,126` instructions;
- interpreted derivative evaluation: approximately `10x` faster;
- reduced JIT evaluation: approximately `25x` faster than the interpreted
  full constrained expression on the development Mac.

The speedup is not an OCP speedup. It excludes the Ding states and does not
measure nonlinear iterations or sparse factorization. The derivative-kernel
comparison does include the mechanical Jacobian and Hessian, but not the
remaining muscle-state transcription.

The Fourier profile now also contains normalized muscle lengths and
muscle-velocity-per-crank-speed. The OCP evaluates the original De Groote
force-length, force-velocity and passive-force equations from these profiles.
With 181 construction samples and order 12, their maximum coefficient errors
over 200 random validation samples were respectively about `2.8e-9`, `5.3e-9`
and `5.2e-10`.

## OCP integration

`ReducedFesCyclingModel` exposes exactly 22 states:

- the existing five Ding states for each of four muscles (20 states);
- unwrapped physical crank angle `theta`;
- variable physical crank velocity `omega`.

Historical full-mechanics warm starts are projected onto the contact manifold.
The projection correction is reported as a warning when it exceeds `1e-4 rad`.
Cycle-boundary drift in the projected theta seed is removed before solving,
using absolute `2*pi` targets. A loaded full-mechanics seed requiring more than
`0.01 rad` of projection is considered mechanically incompatible: its
theta/omega part is rejected while its Ding states and bounded PW are retained.
Pulse-width seeds are independently validated and clipped to
`[pd0, 600 microseconds]`, with a warning reporting every correction.

The reduced formulation is intentionally guarded for ACADOS. It is available
only with `--experimental-reduced-acados` while the one-cycle IPOPT-derived
seed, five-state Ding projection and dynamics rollout are validated. ACADOS
keeps its native stage-wise ordering; the generic time-major NLP ordering must
not be imposed on it.

## Thirty-RHO comparison

The backend benchmark uses one cycle per OCP and 30 requested windows, hence
exactly 30 RHO. A two-cycle horizon is retained as a separate robustness study
for delayed muscle-state memory, not as the default timing case. For each
mechanical formulation, the workflow uses the historical file only to let
IPOPT construct and certify a new solution on the assisted target. Every
compared solver then starts from that same target primal solution.

Run the full reference:

```bash
python examples/fes_multibody/cycling/cycling_fes_solver_comparison.py \
  --solvers ipopt,madnlp \
  --mechanical-formulation full \
  --ipopt-profile periodic_collocation \
  --cycles-per-window 1 \
  --n-windows 30 \
  --stimulations-per-cycle 30 \
  --objective fatigue \
  --standard-warmup-seed .github/benchmark-seeds/legacy-resistive-0p22-warmup.npz \
  --legacy-standard-warmup-seed-signed-torque 0.22 \
  --standard-warmup-seed-continuation \
  --common-initial-solution result/common-full.npz \
  --ipopt-disable-historical-initial-guess \
  --compact-rho-output \
  --output-json result/full-mechanics-30-rho.json
```

Run the reduced formulation from the same historical family of seeds:

```bash
python examples/fes_multibody/cycling/cycling_fes_solver_comparison.py \
  --solvers ipopt,madnlp \
  --mechanical-formulation reduced \
  --ipopt-profile periodic_collocation \
  --cycles-per-window 1 \
  --n-windows 30 \
  --stimulations-per-cycle 30 \
  --objective fatigue \
  --standard-warmup-seed .github/benchmark-seeds/legacy-resistive-0p22-warmup.npz \
  --legacy-standard-warmup-seed-signed-torque 0.22 \
  --standard-warmup-seed-continuation \
  --common-initial-solution result/common-reduced.npz \
  --ipopt-disable-historical-initial-guess \
  --compact-rho-output \
  --output-json result/reduced-mechanics-30-rho.json
```

The Linux GitHub Actions benchmark runs IPOPT, Fatrop-RK4,
MadNLP-PARDISO and MadNLP-MUMPS on the full mechanics, and IPOPT plus both
MadNLP linear solvers on the reduced mechanics. Its default input is 30 RHO.
All sparse NLP cases use time-major ordering. The aggregate report separates
solver/formulation/backend pairs and compares pulse patterns at cycles 10 and
30, including a full-versus-reduced comparison for each supported solver.

ACADOS is now evaluated by two separate CI smoke jobs, one for each mechanical
formulation. They default to five RHO so that a failure of the experimental
reduced transcription does not hide the production NLP benchmark. The RHO
count is controlled by `acados_smoke_rhos` and can be raised to 30 after the
short jobs are repeatable. They run only for the one-cycle benchmark; a
two-cycle dispatch skips ACADOS because its common seeds have a different
horizon. Both jobs start from the corresponding common
one-cycle solution, map variables by name, project the five Ding states per
muscle, perform a complete-dynamics phase I, and advance shifted windows with
the generated ACADOS IRK map. The phase-I mechanical blocks map to `q/qdot`
in the full model and `theta/omega` in the reduced model. The absolute
terminal angle is retained.

The reference remains full SQP with zeroed duals, Gauss--Newton, IRK
Gauss--Legendre (four stages, five integration steps) and the robust HPIPM
profile. RTI and faster integrator/QP variants are tested only after repeated
full-SQP convergence. ACADOS results are uploaded separately and are not yet
included in the Kevin aggregate table.

The reduced formulation is experimental until muscle force, fatigue,
stimulation patterns and terminal progress have been compared over the same
30- and 100-RHO trajectories.
