# Cycling RHO study: IPOPT/MA57 protocol and contribution sequence

This document preregisters the computational comparisons for a separate cycling
study. It does not change the published Cocofest example or claim that the
proposed reductions have already been validated on this branch. The starting
point is `Co_et_al_2026_Cycling` at commit `31e064f4`.

## Research questions

1. Which independently validated reductions bring a bilateral, isokinetic
   receding-horizon optimization (RHO) below the duration of a crank cycle?
2. At identical task and model settings, does minimizing integrated fatigue
   plus a terminal capacity-reserve term improve outcomes relative to
   minimizing integrated fatigue alone?
3. How much wall time and compute does Bayesian optimization (BO) need to find
   useful muscle weights, including initialization and unsuccessful trials?
4. How close is the resulting executed RHO policy to the best full-horizon
   optimization (FHO) solution found under the same task constraints?

Only IPOPT with MA57 is in scope. ACADOS, MadNLP, MUMPS recovery, changes in
stimulation frequency, and mechanically coupled bilateral OCPs are excluded
from the primary comparisons.

## Fixed problem and provenance

The baseline uses the Wu cycling model, four stimulated muscles per side,
30 equally spaced stimulation opportunities per crank cycle, and Radau IIA
collocation with five stages. The crank angle and angular speed are prescribed
for the primary isokinetic experiment. Both arms share the crank phase, but
their muscle states, pulse widths, and OCP solves remain independent. The
resistive load, cadence, cycle duration, left/right load allocation, muscle
parameters, stimulation bounds, and initial state must be recorded in each
run manifest rather than inferred from a directory name.

The initial numerical reference must reproduce the full-state model before
any reduction is enabled. A preparation run may generate a cycle-1 seed, but
its cost is reported separately from warm controller latency. No FHO result
may be used to select RHO weights or initialize the BO search.

Each archived run records the base and implementation commits, Python,
Bioptim, CasADi, IPOPT and MA57/HSL identities, CPU affinity and thread limits,
the complete configuration, solver statuses, physical constraint residuals,
and a hash of the model. The presence of `ma57` as an option is not a runtime
check: a preflight solve must verify that IPOPT actually loads it.

The existing cycling example uses Radau-3/MX and chooses MA57 only on Linux.
It is a starting point, not evidence for the proposed Radau-5/SX results.

## Timings and real-time criterion

For warm cycle `k`, measure the interval from availability of the incoming
state to availability of both arms' admissible pulse-width commands:

    T_pair,k = T_prepare,k + T_solve_pair,k + T_reconstruct,k + T_gate,k.

Report median, P90, P95, maximum, successful-cycle count and timing-scope
definitions. Report IPOPT internal time, complete per-arm RHO time, and pair
latency separately; these numbers must not be interchanged. Cold graph
construction, code compilation and BO are outside `T_pair`, but their wall
times and amortization assumptions must also be reported. At least 100
autonomous consecutive cycles, each feeding its own output into the next
window, are required for the timing comparison.

The primary real-time pass criterion is P90(`T_pair`) below the physical cycle
duration, with no missed commands and all 100 cycles passing the same physical
audit. This is a computational criterion, not a claim of clinical safety.
Specify the margin to the deadline and the CPU allocation for every result.

## Objective comparison

Let `a_i(t) = A_i(t) / A_rest,i` be the normalized capacity of muscle `i` and
`phi_i(t) = 1 - a_i(t)` its normalized fatigue. For `M` muscles and cycle
duration `T`, the reference objective is

    J_F = (1 / (M T)) sum_i integral_0^T phi_i(t)^2 dt.

The second objective uses the same admissible set and adds a smooth penalty
on the lowest terminal capacity:

    R_tau(a(T)) = -tau log((1/M) sum_i exp(-a_i(T)/tau)),
    J_F+R = J_F + lambda_R [1 - R_tau(a(T))].

State the numerical `tau`, `lambda_R`, objective normalization and quadrature
before inspecting outcomes. This is **capacity reserve**, not the maximum
mechanical work or torque achievable in a future cycle. A mechanical-reserve
objective would require a separate validation and is not silently substituted.

Compare the objectives at matched initial states, loads, bounds, grids,
warm-start rules and muscle weights. Report integrated fatigue, terminal
capacity per muscle, stimulation, work and mechanical tracking, solver time,
and certified executed cycles. An IPOPT failure alone is not a physiological
exhaustion label; a frozen-state feasibility check and an independent rollout
are required before making that interpretation.

## BO and FHO evaluation

The BO domain is a normalized, bounded set of positive muscle weights. The
primary study shares one four-muscle weight vector between symmetric sides;
side-specific weights are a separate sensitivity analysis. The objective
variant and BO budget are fixed before the search. Record every candidate,
seed, censoring limit, completed cycles, wall time, CPU time and failure
reason. Report time to the first candidate reaching a prespecified target,
not only the best score after the full budget. Re-evaluate finalists from a
fresh initial state and on held-out loads or parameter perturbations.

The FHO comparison uses the same dynamics, transcription, work requirement,
bounds and initial state. Evaluate both returned policies with a common
physical metric and independent dense rollout; weighted NLP objective values
are not comparable when weights differ. Use multiple FHO initializations and
report the best feasible solution found, its time and its optimality residuals.
Because the OCP is nonconvex, a small RHO--FHO gap is an *empirical proximity*
to that FHO reference, not a certificate of global quasi-optimality.

## Small-PR sequence and gates

Each PR should change one mechanism, normally within 200--300 changed lines
including tests. A larger slice is split before review. A PR is merged only
after its stated numerical equivalence or physical audit passes; later PRs
are based on the preceding accepted slice.

| Order | Isolated change | Required gate |
| --- | --- | --- |
| 01 | Fixed baseline and warm timing contract | Explicit Radau-5/SX, IPOPT/MA57, manifest and smoke |
| 02 | Wu sagittal contact/phase checks | Dense closure and branch tests |
| 03 | Reduced isokinetic moment/work balance | Full-model torque and work agreement |
| 04 | Reduced mechanics in the unilateral RHO | Matched full/reduced trajectories and constraints |
| 05 | Prescribed crank kinematics | Same physical task; state-count and timing ablation |
| 06--07 | Discrete local Ding map, then bound/dual reconstruction | Original-NLP residual and Radau certificate |
| 08--09 | Reduced Hessian compilation, then numeric dispatch cache | Identical fixed-input solutions; isolated timing A/B |
| 10--11 | Left/right model symmetry, then independent process pair | Two autonomous 100-cycle audits and full pair latency |
| 12 | Capacity-reserve objective | Matched objective ablation and gradient checks |
| 13--14 | Bounded BO, then independent finalist validation | Search cost and no FHO data leakage |
| 15--16 | FHO reference, then common replay/comparison plots | Feasible multi-start FHO and matched physical metrics |

The optional analytic four-bar solver belongs in an independent *offline*
geometry-preparation PR after contact geometry is validated. The warm RHO
already consumes fitted kinematic profiles, so its derivation alone does not
justify a claim of reduced controller latency.

Large experimental commits from another branch are source material, not
cherry-pick units. Port only reviewed functions and tests, record their source
commit in each PR description, and test against the dependencies actually
installed for this branch. In particular, compiled callback benchmarks must
not be attributed to upstream Bioptim until their fork-specific dependency is
cleanly pinned or contributed upstream.
