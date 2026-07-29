# Fatrop, MadNLP and Alpaqa for the cycling fatigue problem

The cycling MHE accepts `ipopt`, `fatrop`, `madnlp`, `alpaqa`, and `acados` as
solver backends. The production endurance matrix contains IPOPT, Fatrop and
MadNLP. Alpaqa remains available only as a documented diagnostic because it
did not validate a single RHO in the completed screens.

```bash
python examples/fes_multibody/cycling/cycling_fes_solver_comparison.py \
  --solvers ipopt,fatrop,madnlp \
  --objective fatigue \
  --objective-shape quadratic \
  --ipopt-profile periodic-collocation \
  --cycles-per-window 1 \
  --stimulations-per-cycle 30 \
  --n-windows 3 \
  --n-threads 8 \
  --state-scaling full \
  --fatrop-state-scaling none \
  --fatrop-bound-tightening-factor 1e-8 \
  --fatrop-max-iter 1000 \
  --madnlp-max-iter 300 \
  --max-consecutive-failing 1 \
  --output-json /tmp/cocofest-fatigue-solvers.json
```

## Cross-solver hot-start and tuning

All sparse NLP solvers use a time-major decision-variable order, the same
absolute terminal crank bound, and the same shifted primal MHE trajectory.
The Linux workflow first solves one assisted target OCP with IPOPT for each
mechanical formulation and stores the converged primal trajectory as a
metadata-bearing common solution. IPOPT, MadNLP-PARDISO, MadNLP-MUMPS and
Fatrop then load that exact target trajectory; the historical resistive file is
only a bridge used to construct it. ACADOS keeps its native stage-wise
`(x_k, u_k)` layout instead of receiving a generic NLP ordering.

Fatrop runs the target problem with RK4 direct shooting (five integration
steps), time-major ordering and no state scaling. IPOPT and both MadNLP
backends use Radau collocation and full state scaling. The physical target and
starting trajectory are common, but the report deliberately marks the
transcription and scaling differences because they affect conditioning and
timing.

The common target seed is committed only when IPOPT exposes a measured primal
infeasibility below the acceptance threshold. Its metadata records how many
continuation warm-up cycles were consumed. A consumer must replay the same
continuation before loading the seed, otherwise its first-node fatigue bounds
describe a different absolute cycle and the seed is rejected. Since the common
seed is already a certified target solution, the production benchmark uses
`--no-optional-nlp-periodic-ipopt-hot-start` and does not pay for a redundant
solver-specific IPOPT refinement.

The CasADi build used for the assisted endurance benchmark must expose both
`ipopt` and the target plugin. A local diagnostic build containing only
MadNLP/Alpaqa cannot execute either the standard IPOPT warmup or the certified
periodic seed. In that special case, both
`--ipopt-disable-standard-warmup` and
`--no-optional-nlp-periodic-ipopt-hot-start` are required, but the resulting
cold solve is not representative of the intended endurance workflow.

The useful transfers are not identical for every backend:

| Mechanism | IPOPT | Fatrop | MadNLP | Alpaqa | ACADOS |
|---|---|---|---|---|---|
| Shifted primal trajectory | yes | yes | yes | yes | yes |
| Common assisted IPOPT solution | producer/consumer | consumer | consumer | diagnostic consumer | consumer before projection/rollout |
| Variable order | time-major | time-major | time-major | time-major | native stage-wise |
| State scaling | full | none, pending interface support | full | full | full |
| Constraint multiplier reuse | optional | off pending shifted-block validation | experimental | supported | reset/rebuilt |
| Bound multiplier reuse | supported | off pending shifted-block validation | experimental | unsupported by plugin | backend-specific |
| Linear solver choice | MUMPS/MA57 | structured Riccati factorization | MadNLP runtime types | not applicable to PANOC | QP solver |
| Native C evaluator | optional | experimental | experimental | unsupported by plugin | generated code |
| Time budget | IPOPT option | 1000 native iterations | outer process/job timeout | ALM and PANOC budgets | backend timeout |

### Fatrop-specific safeguards

Fatrop is exposed by the official CasADi 3.7.2 wheel. It exploits the OCP
structure only when Bioptim uses `OrderingStrategy.TIME_MAJOR`. A previous
100-RHO CI validated 100/100 one-cycle collocation windows in that layout. The
later two-cycle collocation profile failed before its first RHO because the
automatic detector classified collocation variables as controls. The
production benchmark therefore returns to one-cycle windows and uses RK4
direct shooting with five integration steps. It also keeps
`--fatrop-state-scaling none`, because scaled gap equations do not retain the
identity next-state coefficient required by automatic structure detection.
This preserves the physical model but is not a pure backend-only timing
comparison with the collocation IPOPT/MadNLP NLP.

CasADi/Fatrop also applies an effective relative relaxation of approximately
`1e-8` to decision bounds. This matters for the unscaled fatigue capacities,
whose magnitude reaches about 7000: the returned point can otherwise exceed a
physical bound by about `7e-5`, above the common `1e-5` audit threshold. The
pinned Bioptim interface compensates by tightening non-fixed solver-call
bounds by `1e-8`, while retaining the original physical bounds for the
independent post-solve audit. This is not a relaxation of Cocofest's acceptance
criterion.

On the local five-RHO assisted smoke test, Fatrop validated every window. The
maximum constraint violation was `3.25e-9`, the physical decision-bound
violation remained zero, iterations stayed between 81 and 88, and solver time
was `4.80--5.93 s` with a `5.24 s` median. This is encouraging but too short
to establish endurance robustness; the Linux 100-RHO job is the actual
benchmark.

Relevant MadNLP screens include:

```bash
--madnlp-linear-solver mumps
--madnlp-linear-solver umfpack
--madnlp-linear-solver lapack_cpu
--madnlp-linear-solver pardiso_mkl
```

The MadNLP C runtimes accept `mumps`, `umfpack`, `lapack_cpu`, and three GPU
backends. The public x86-64 Linux benchmark now builds libMad commit
`5529f23a6bff33c566ad954da38d352f1f172356` directly from
`mickaelbegon/libMad`'s `codex/pardiso-mkl` branch and maps
`pardiso_mkl` to its native `PardisoMKLSolver` type. This runtime embeds
`MadNLPPardiso` and Intel oneMKL without depending on HSL. It is unavailable
on ARM, including Apple Silicon, because oneMKL does not provide that native
backend there. CoinHSL remains loaded dynamically only by IPOPT through
`--ipopt-hsl-library`.

The pinned libMad runtime rejects `mu_init`, `dual_initialized`,
`max_wall_time`, `nlp_scaling`, and the `acceptable_*` options. Cocofest
therefore does not expose or send them. Its MadNLP hot start is the shifted and
projected primal state/control trajectory after the certified periodic IPOPT
refinement; multiplier transfer remains off. The GitHub job timeout is the
safety limit.
The current libMad runtime embeds MadNLP 0.9.2, whose `LogLevels` enum ranges
from `TRACE=1` to `ERROR=6` and rejects the usual IPOPT quiet value `0`.
Cocofest therefore bypasses Bioptim's generic mapping and sends numeric
`print_level=6`, preventing verbose logging from contaminating timings.

Relevant Alpaqa screens include:

```bash
--alpaqa-lbfgs-memory 20
--alpaqa-max-iter 1000
--alpaqa-alm-max-iter 50
--alpaqa-initial-penalty 0
--alpaqa-initial-tolerance 1e-3
--alpaqa-penalty-update-factor 5
--alpaqa-maximum-penalty 1e7
--alpaqa-max-wall-time 600
--alpaqa-panoc-max-wall-time 60
--alpaqa-max-no-progress 100
```

Alpaqa is especially sensitive to constraint scaling, its initial penalty, and
the ratio between the outer ALM and inner PANOC budgets. Test these one factor
at a time. Its CasADi plugin cannot reconstruct the derivatives required after
the current `nlp.c` import path, so C compilation remains explicitly
unsupported.

With the default quadratic shape, the fatigue objective is the Lagrange
integral of `(1 - A_muscle / A_scale)^2` for every muscle. Its weight vector is
`[0, 1, 0]`; force, stimulation-charge, terminal wheel-angle, control, and
wheel-speed objectives are therefore disabled.

## ACADOS assisted example

The standalone ACADOS example now defaults to a `0.2 N.m` crank assistance:

```bash
python examples/fes_multibody/cycling/cycling_pulse_width_mhe_acados_periodic.py \
  --crank-assistance 0.2 \
  --n-windows 50 \
  --compact-rho-output \
  --max-consecutive-failing 2 \
  --codegen-tag assistance_0p2
```

The crank turns in the negative direction (`qdot ~= -2*pi rad/s`), so the
assistance magnitude is converted internally to a signed `-0.2 N.m` torque.
Its nominal external power is therefore positive, about `+1.257 W`. Use
`--signed-crank-torque 0.2` only for the opposite, genuinely resistive,
experiment.

The assisted defaults use the exact `periodic_node` forcing, a one-cycle
horizon, full state scaling, exact first-node crank-angle continuity, and a
`0.002 rad` terminal angle tolerance. The one-cycle horizon is intentional:
it fixes the crank phase every cycle without introducing an internal seam,
and halves the NLP size relative to the former two-cycle profile. The
initialization pipeline is:

1. solve the standard Ding collocation problem with the same assisted torque
   using IPOPT-MUMPS;
2. advance the solution by one executed cycle, including controls and the
   continuous `A`, `Tau1`, and `Km` fatigue states;
3. project the trajectory onto the periodic FES dynamics;
4. run and cache one one-cycle periodic IPOPT refinement, accepting it only
   when IPOPT reports a measured primal infeasibility below the threshold;
5. solve ACADOS first with controls fixed to the IPOPT reference, then release
   them over the measured radii `1e-8` and `1e-7 s`;
6. retain the largest converged radius, then recenter that trust region after
   every RHO shift.

On the local one-cycle assisted test, the periodic IPOPT bridge converged with
`inf_pr=4.73e-9` in `62.5 s` and was then cached. ACADOS reproduced its cost
(`3.69957`) in `1.40 s`. A five-cycle MHE smoke test subsequently completed
all five physical cycles: four windows converged in 3--7 SQP iterations and
one non-consecutive window reached the 100-iteration limit; total ACADOS
solver time was `37.4 s`, including `29.8 s` for that outlier. This is why
`--max-consecutive-failing 2` remains useful for endurance runs.

For explicit two-cycle experiments, ACADOS now receives stage-wise crank-angle
bounds at every internal cycle seam. The old underconstrained formulation
allowed the first and second cycle to cover about `0.793` and `1.207` turns,
respectively, even though the total was two turns. The strict bridge exposes
the resulting `~1.30 rad` phase error instead of silently optimizing a
different problem. An experimental seam continuation is available through
`--acados-cycle-boundary-homotopy-slacks`, but it is not the default: on the
macOS test it had not reached the strict `0.002 rad` seam within the bounded
continuation budget.

The warm-start preparation time is reported separately from the ACADOS solve
times, and `warmup_cycles_consumed=1` makes the initial fatigue advance
explicit. Automatic caches include the signed torque. Explicit warmup seeds
must also contain matching physical metadata, so an old `+0.2 N.m` resistance
seed is rejected instead of being reused silently.

A legacy seed created before metadata was introduced can only be used with an
explicit sign assertion:

```bash
--standard-warmup-seed /absolute/path/to/legacy_warmup.npz \
--legacy-standard-warmup-seed-signed-torque -0.2
```

The loader also checks that its control grid matches the requested horizon.
The assertion is runtime-only; it does not rewrite the legacy file.

## Bioptim and CasADi requirements

The production implementations currently live in this Bioptim 3.5 branch:

- Fatrop/MadNLP: `codex/fatrop-cocofest-benchmark` at
  `3523f1745e315f07761159d7e06bd2d876026704`.

The inactive Alpaqa diagnostic remains on its separate branch:

- Alpaqa: `codex/alpaqa-integration` at `d84e7e43534360fc048e0be26a3bd69a2abc2d77`.

Install the applicable branch and a CasADi build that contains the corresponding
`nlpsol` plugin. A normal Cocofest installation remains valid: unavailable
optional solvers produce an actionable benchmark failure instead of preventing
Cocofest from importing or aborting the rest of the solver matrix.

The production branch exposes Fatrop and MadNLP together. Alpaqa remains
separate; run its diagnostic alone with `--solvers alpaqa` when reproducing
the archived screen.

Plugin availability can be checked before a long solve:

```python
import casadi as ca
from bioptim import Solver

print("MadNLP:", hasattr(Solver, "MADNLP") and ca.has_nlpsol("madnlp"))
print("Fatrop:", hasattr(Solver, "FATROP") and ca.has_nlpsol("fatrop"))
print("Alpaqa:", hasattr(Solver, "ALPAQA") and ca.has_nlpsol("alpaqa"))
```

## NLP formulation and warm-start policy

Fatrop, MadNLP and Alpaqa clone the complete physical IPOPT-side NLP selected
by `--ipopt-profile`. All sparse NLP backends now request time-major ordering.
MadNLP and Alpaqa preserve IPOPT's full state scaling; Fatrop uses the RK4,
unscaled compatibility reparameterization described above, so it remains
physically comparable but not a backend-only numerical ablation:

- `historical` uses the standard formulation, segment-level external torque,
  and `COLLOCATION(3, radau)`;
- `periodic-collocation` uses `periodic_node`, a constant generalized crank
  torque, and the same robust collocation scheme.

The periodic profile is substantially faster for MadNLP in the macOS
experiments below. It is still a different formulation from the historical
profile and from the ACADOS RK transcription, so results must not be compared
across those profiles as if they were the same NLP.

The first horizon reuses the accumulated robust initialization pipeline:

1. load and validate the historical collocation solution using a source-aware
   cache signature;
2. resample it exactly onto the active grid;
3. project state and pulse-width values into their scaled bounds;
4. preserve the FES rollout and accumulated fatigue states;
5. audit finiteness and record a reproducible initial-guess signature.

Between horizons, Cocofest advances one pedalling cycle, shifts the
state/control trajectory, applies the signed wheel-angle shift, keeps fatigue
states continuous, synchronizes fixed first-node values with the new bounds,
and projects the result again.

Direct collocation degree three stores 120 state samples but only 30 controls
per cycle. These two strides are now kept separate when shifting a window. An
earlier implementation incorrectly used the state stride for controls, so a
60-control two-cycle vector was never shifted; all option screens made before
this correction are labelled preliminary below.

The shifted primal must not be replaced by Bioptim's generic warm-start helper.
Multipliers can be assigned directly after size and finiteness checks:

- MadNLP now defaults to no multiplier transfer. Its primal state/control
  hot-start remains enabled.
- Bound and constraint multiplier transfer remains available as an explicit
  ablation, but those blocks are not yet shifted structurally by cycle.
- Alpaqa accepts only constraint multipliers (`lam_g`) through CasADi.

The shifted primal and first-node synchronization are therefore the reliable
warm start. Multiplier transfer should not become the default until its blocks
are transformed consistently with the receding horizon.

The number of Bioptim/CasADi workers is configurable with `--n-threads` and
defaults to the logical CPU count. This replaces the historical hard-coded
value of 48, which oversubscribed smaller machines. When these workers are
enabled, keep nested BLAS/OpenMP pools at one thread unless a separate timing
experiment demonstrates a benefit.

For a same-direction torque continuation in `0.02 N.m` increments, a
neighbouring solved warmup can be reused without rebuilding it:

```bash
--standard-warmup-seed /absolute/path/to/warmup_previous_torque.npz
```

This bypasses only the standard IPOPT seed solve. The target OCP and its
constant torque are still rebuilt at the requested new load, then MadNLP
optimizes the complete target NLP. The seed path is stored in benchmark JSON
for provenance. The cache metadata must match the horizon, stimulation grid,
torque application, and mechanical role. A metadata-bearing resistance seed
is therefore rejected for an assisted target. The public Linux workflow uses
a separate, explicit legacy-continuation path: it asserts that the old seed
was created at signed torque `+0.22 N.m`, uses it only as a primal initial
guess, then solves and certifies a new IPOPT seed on the assisted
`-0.20 N.m` target before any compared solver starts. The legacy file itself
is never presented as an assisted solution.

## Linux GitHub Actions endurance benchmark

The manually triggered
[`cycling_solver_benchmark_linux.yml`](../.github/workflows/cycling_solver_benchmark_linux.yml)
now runs IPOPT-MUMPS, Fatrop-RK4, MadNLP-PARDISO and MadNLP-MUMPS in parallel
on separate Linux runners. IPOPT and both MadNLP backends are evaluated on the
full and reduced mechanical formulations; Fatrop currently covers the full
formulation.
Alpaqa was removed from the endurance matrix after the option screen and the
30-RHO confirmation: it validated no RHO, consumed two 600-second limits, and
its second shifted window reached `4.57e-2` infeasibility. Its integration,
historical results and explicit `cycles=screen` diagnostic remain documented
for reproducibility, but no further endurance compute is allocated to it.

The production experiment uses one-cycle RHO windows by default and stops
after two consecutive failed windows. A two-cycle input remains available as
a separate memory/robustness study: it can expose benefits from the delayed
Ding states and reduce terminal-boundary myopia, but doubles the horizon,
creates an internal seam and has already broken Fatrop's collocation structure
detection. It is therefore not the backend timing reference. The benchmark uses
the assisted physical case (`-0.20 N.m` signed crank torque), 30 stimulations
per cycle, the fatigue-only objective, and a `0.002 rad` absolute terminal
crank-angle slack. The combined report exposes Fatrop's RK4/no-state-scaling
mode as a configuration difference.

A preliminary IPOPT job builds one physically certified assisted solution for
the full formulation and one for the reduced formulation. Every solver in a
formulation downloads exactly the same immutable target solution and disables
its solver-specific IPOPT warmup. The seed-construction time remains separate
from the measured benchmark jobs.

Each job determines its effective CPU allocation with `nproc` and passes that
value to `--n-threads`. Nested OpenMP, BLAS, NumExpr, and Julia pools stay at
one thread so that CasADi owns the runner-wide parallelism instead of
oversubscribing every worker. The same value is used to compile the
source-built CasADi-compatible biorbd dependency.

The default GitHub-hosted runner can be replaced at dispatch time by the label
of a larger Linux or self-hosted runner. For example:

```bash
gh workflow run cycling_solver_benchmark_linux.yml \
  --repo mickaelbegon/cocofest \
  --ref codex/acados-pr-refresh \
  -f runner_label=ubuntu-24.04 \
  -f cycles=100 \
  -f cycles_per_window=1 \
  -f crank_assistance_nm=0.20 \
  -f terminal_wheel_q_slack=0.002 \
  -f solver_max_iterations=2000
```

The production action pins the combined Fatrop/MadNLP Bioptim integration
commit, so IPOPT, Fatrop and MadNLP use the same Bioptim revision. Each job
first asserts that IPOPT and its target plugin coexist in the same CasADi
runtime. The MadNLP job additionally pins and builds the libMad PARDISO/MKL
runtime, exercises the selected PARDISO/MKL or MUMPS backend through libMad's
C example, then repeats that check through CasADi before starting the OCP.
The libMad option is `PardisoMKLSolver` for PARDISO and the lowercase `mumps`
identifier for MUMPS. The official CasADi 3.7.2
wheel still targets the obsolete `libmadnlp_c.so` ABI, so the MadNLP job builds
the pinned post-release CasADi 3.7.2 source revision that targets
`libMad.so`; a symbolic link between these libraries would not be ABI-safe.
The installed runtime is cached by the runner image, architecture, exact Julia
version, and exact libMad and JuliaC commits. Julia 1.12's compiled runtime
requires the `GCC_13.0.0` symbol in `libgcc`;
the workflow therefore defaults to Ubuntu 24.04 and checks the resolved
library before the roughly 20-minute JuliaC build, including on cache hits.
`MKL_NUM_THREADS` receives the
runner's complete CPU allocation for this job, while the unrelated BLAS and
OpenMP pools remain at one thread to prevent nested oversubscription. The
archived Alpaqa screen still builds CasADi 3.7.2 with the pinned compatibility
fork declared by that release (`jgillis/alpaqa` at
`bf9f87d59640501ea72f94aa6e2d4e62b20c677b`); this path is no longer part of
the endurance matrix.

Each solver artifact contains its JSON and complete log. A final report job
adds a side-by-side Markdown summary, `rho-timings.csv`, the raw stimulation
profiles in `stimulation-patterns.csv`, and one combined JSON. The checkpoints
called “10”, “30” and “100” are the executed cycles/RHO of the same run, not
OCPs containing that many simultaneous cycles. Patterns are exported only
when the checkpoint belongs to the strictly converged prefix, and include the
real crank phase and velocity so that a kinematic phase shift is not mistaken
for a stimulation-strategy difference.

Solver non-convergence is a benchmark result: the JSON preserves all attempted
windows, while `validated_cycles` remains the strict prefix ending before the
first failed or infeasible RHO. Status-zero count, independent bound
violations, native status, and both failed attempts remain available for
diagnosis.

The focused Linux option screen
[`30292129183`](https://github.com/mickaelbegon/cocofest/actions/runs/30292129183)
applied one common independent `1e-5` feasibility threshold while keeping each
solver's internal tolerance visible:

| Variant | Common valid prefix | Maximum infeasibility | Median RHO wall |
|---|---:|---:|---:|
| MadNLP, tolerance `1e-8`, default linear solver | 4/4 | `1.69e-8` | 6.68 s |
| MadNLP, tolerance `1e-7`, default linear solver | 4/4 | `6.66e-7` | 6.71 s |
| MadNLP, tolerance `1e-7`, explicit MUMPS | 4/4 | `6.66e-7` | 6.78 s |
| MadNLP, tolerance `1e-7`, UMFPACK | 4/4 | `9.58e-7` | 11.0 s |
| MadNLP, tolerance `1e-6`, default linear solver | 3/4 | `1.28e-5` | 5.60 s |
| Alpaqa, automatic penalty, 20 s inner PANOC budget | 0/1 | `4.09e-4` | 60.0 s |
| Alpaqa, default penalty, 20 s inner PANOC budget | 0/1 | `2.91e-2` | 60.0 s |

The identical `1e-7` default and explicit-MUMPS trajectories confirm that
MUMPS is the pinned MadNLP runtime's default for this case. UMFPACK is about
64 percent slower here. Tightening MadNLP to `1e-8` costs essentially nothing
over four RHO and avoids the physical-threshold miss seen at `1e-6`; it is now
the endurance default. For Alpaqa, the automatic initial penalty improves the
one-minute residual by about 71 times over the tested default, but remains
about 41 times above the acceptance threshold. It is the least-bad tested
setting, not a demonstrated convergent configuration.

The final Linux run
[`30297904541`](https://github.com/mickaelbegon/cocofest/actions/runs/30297904541)
uses four available CPU cores and applies the screened settings on the
corrected one-cycle assisted formulation:

| Solver | Validated / requested RHO | Attempted RHO | Preparation | Hot median | Hot P90 | End-to-end |
|---|---:|---:|---:|---:|---:|---:|
| IPOPT-MUMPS, internal tolerance `1e-6` | 30 / 30 | 30 | 23.58 s | 6.232 s | 8.243 s | 236.40 s |
| MadNLP-MUMPS, internal tolerance `1e-8` | 30 / 30 | 30 | 56.60 s | 5.853 s | 7.806 s | 252.69 s |
| Alpaqa, automatic penalty | 0 / 30 | 2 | 58.76 s | — | — | 1260.25 s |

All 30 IPOPT and MadNLP windows pass the same independent `1e-5` feasibility
screen. Their maximum independently reconstructed violations are respectively
`9.64e-7` and `2.57e-7`; MadNLP's tighter result is partly explained by its
100-times tighter internal tolerance. Compared with the former `1e-6`
MadNLP run, this confirms that the earlier 3-RHO strict prefix was a
termination-accuracy problem, not an intrinsic inability to continue the MHE.

In this single run, MadNLP's hot median, P90 and attempted-RHO sum are about
6, 5 and 8 percent lower than IPOPT's. Its additional certified periodic IPOPT
refinement increases preparation by about 33 seconds, so IPOPT remains
16.3 seconds faster end-to-end at 30 RHO. The observed per-RHO saving would
amortize the preparation difference only around 60 RHO. This crossover is an
estimate, not a stable claim: at least three paired repetitions on a
controlled runner are required.

At cycle 10, the angle-aligned pulse-width RMSE between MadNLP and IPOPT is
below `0.8 us` for every muscle. At cycle 30, biceps and triceps diverge to
`100.09 us` and `149.05 us` angle-aligned RMSE, even though the executed
fatigue objectives differ by only 0.24 percent (`162.720` versus `163.110`).
The divergence survives interpolation onto IPOPT's real crank-angle grid, so
it is not just the observed `0.163 rad` intra-cycle phase difference. With no
control or cadence regularization, the most defensible interpretation is
distinct local muscle-sharing strategies with nearly equivalent scalar
fatigue. Seed exchanges, seed perturbations and inspection of crank speed,
muscle forces and fatigue states are required before assigning physiological
meaning to either pattern.

The 100-RHO extension
[`30304318862`](https://github.com/mickaelbegon/cocofest/actions/runs/30304318862)
uses the same formulation, solver options and four-core runner:

| Solver | Certified windows | Strict prefix | Preparation | Attempted-RHO sum | Hot median | Hot P90 | End-to-end |
|---|---:|---:|---:|---:|---:|---:|---:|
| IPOPT-MUMPS | 100 / 100 | 100 / 100 | 23.03 s | 818.74 s | 8.241 s | 11.130 s | 866.22 s |
| MadNLP-MUMPS | 99 / 100 | 85 / 100 | 45.72 s | 815.33 s | 5.623 s | 7.745 s | 881.48 s |

MadNLP reaches its 2000-iteration limit on RHO 86 after 187.12 seconds and
ends that window at `4.85e-2` independently reconstructed infeasibility.
IPOPT solves the corresponding window in 76 iterations and 8.59 seconds.
MadNLP then recovers: RHO 87 through 100 all return success and pass the
common physical threshold, so there is no pair of consecutive failures. The
strict prefix nevertheless remains 85 by design, because the executed
trajectory cannot skip the uncertified window. The event is therefore a
local robustness/outlier problem, not evidence that the physiological fatigue
limit occurs at cycle 86.

Over the first 85 paired windows, MadNLP's median wall-time ratio relative to
IPOPT is `0.824`. Its normal hot path is faster, but the single 187-second
outlier and its extra preparation erase that advantage: IPOPT is 15.26 seconds
faster end-to-end. The relevant deployment conclusion is consequently about
tail latency and robustness, not only median throughput.

IPOPT ends 100 validated cycles at `min(A/A_scale)=0.92495`; the physiological
failure point is still not reached. Its maximum per-cycle crank-progress error
is only `0.002006 rad`, but the terminal slack is selected with a persistent
sign and accumulates to `0.1587 rad` after 100 cycles. A future scientific
endurance run should screen exact terminal progression and a much tighter
slack (for example `0` and `1e-4 rad`) before extending toward 1000 cycles.
The cycle-100 stimulation pattern is reported only for IPOPT; the MadNLP
pattern is deliberately suppressed because it follows the break in the strict
prefix.

The three-solver extension
[`30309452077`](https://github.com/mickaelbegon/cocofest/actions/runs/30309452077)
repeats 100 RHO with Fatrop included and the common Fatrop-capable Bioptim
revision:

| Solver | Certified windows | Strict prefix | Preparation | Attempted-RHO sum | Hot median | Hot P90 | End-to-end |
|---|---:|---:|---:|---:|---:|---:|---:|
| IPOPT-MUMPS | 100 / 100 | 100 / 100 | 23.82 s | 830.20 s | 8.311 s | 11.190 s | 878.90 s |
| Fatrop | 100 / 100 | 100 / 100 | 42.96 s | 1148.17 s | 11.979 s | 14.016 s | 1209.02 s |
| MadNLP-MUMPS | 100 / 100 | 100 / 100 | 54.90 s | 799.46 s | 6.252 s | 9.278 s | 877.35 s |

All three strict prefixes reach 100 and pass the independent physical audit.
Their maximum effective infeasibilities are respectively `9.64e-7`,
`9.55e-7`, and `1.00e-6`, below the common `1e-5` threshold. Fatrop's
independently reconstructed decision-bound violation is exactly zero after
compensating for its native relative bound relaxation.

Fatrop is 44 percent slower than IPOPT in hot median and 38 percent slower
end-to-end in this run. Its effort also grows with the continued horizon:
RHO 1--10 average 77.4 iterations and have a 7.60-second median, whereas
RHO 91--100 average 121.7 iterations and have a 13.33-second median. This is
not a pure backend-only timing comparison. Fatrop's automatic gap-structure
detection currently requires time-major ordering and unscaled state
variables; full state scaling changes the collocation gap form and is rejected
as a structure mismatch. IPOPT and MadNLP retain full state scaling, and the
combined report exposes that mismatch explicitly.

Fatrop remains very close to IPOPT physiologically and numerically. Their
cycle-10 biceps/triceps pulse-width RMSE values are `0.10/0.49 us`, and their
cycle-30 values are `1.84/5.21 us`; the corresponding correlations remain
near one. Their minimum final capacity ratios are also close:
`0.92453` for Fatrop and `0.92495` for IPOPT.

MadNLP is fastest in the normal hot path, but a 969-iteration, 99.61-second
RHO 99 outlier makes its end-to-end time essentially tied with IPOPT. The fact
that it converges on all 100 windows in this repetition, after failing RHO 86
in the preceding run, reinforces the tail-latency and repeatability concern.
MadNLP also reaches a distinct lower-fatigue local solution:
`min(A/A_scale)=0.94777`, an executed fatigue objective about 34 percent below
IPOPT, and cycle-30 biceps/triceps pulse-width RMSE values of
`100/140 us` with near-zero correlations. With no control regularization,
forces, velocities, kinematics, and physiological plausibility must be checked
before interpreting this basin as superior.

No solver reaches physiological fatigue failure over 100 cycles: the lowest
remaining capacity is still about 92.5 percent. This run compares
continuation robustness, tail latency, and stimulation basins; it does not
identify the fatigue-to-failure cycle.

### Absolute-angle 100-RHO benchmark with MadNLP-PARDISO

Run
[`30363688991`](https://github.com/mickaelbegon/cocofest/actions/runs/30363688991)
uses the fixed absolute crank target, a common assisted seed, the production
Bioptim commit, and the libMad PARDISO/MKL runtime. The seed, all three
100-RHO jobs, and the aggregate report succeed:

| Solver | Certified windows | Strict prefix | Preparation | Attempted-RHO sum | Hot median | Hot P90 | End-to-end |
|---|---:|---:|---:|---:|---:|---:|---:|
| IPOPT-MUMPS | 100 / 100 | 100 / 100 | 23.76 s | 730.71 s | 7.110 s | 9.434 s | 778.99 s |
| Fatrop | 100 / 100 | 100 / 100 | 56.86 s | 1230.43 s | 11.719 s | 17.607 s | 1312.02 s |
| MadNLP-PARDISO | 100 / 100 | 100 / 100 | 51.60 s | 1060.00 s | 8.679 s | 10.827 s | 1135.93 s |

The maximum independently reconstructed primal infeasibilities are
`9.71e-7`, `9.43e-7`, and `1.00e-6`, respectively. All are below the common
`1e-5` acceptance threshold. The corresponding maximum absolute crank-angle
errors are `0.002005`, `0.002000`, and `0.002006 rad`. The audit reference is
the fixed problem origin shifted by the one warmup cycle consumed before the
exported RHO trace. The scaled decision-bound tolerance is converted back to
physical radians; this accepts solver-level numerical error without allowing
the terminal band to drift between cycles.

IPOPT is fastest in this run. Fatrop is 65 percent slower in hot median and
68 percent slower end-to-end. MadNLP-PARDISO is 22 percent slower in hot
median and 46 percent slower end-to-end. MadNLP's tail contains three
important events: RHO 75 takes 1177 iterations and 141.22 seconds, RHO 82
takes 533 iterations and 63.55 seconds, and RHO 60 takes 209 iterations and
24.61 seconds. Every event still converges and passes the physical audit.

The immediately preceding PARDISO run produced nearly identical timings and
the same outliers, so the tail behavior is reproducible. The older MUMPS run
was faster, but it still used the relative terminal-angle formulation that
could drift and reached a different fatigue trajectory. It is therefore not a
controlled linear-solver comparison. The current evidence shows no PARDISO
speed advantage for this OCP, but a paired MUMPS/PARDISO experiment on the
same absolute formulation and repeated runners is required before attributing
the difference to the factorization backend.

At cycle 10, all three stimulation patterns remain in the same basin. By
cycle 30, Fatrop and MadNLP have moved together to a second muscle-sharing
solution: their biceps/triceps means are approximately `134/174 us`, versus
`159/136 us` for IPOPT. Against IPOPT, the phase-aligned biceps and triceps
RMSE values are about `108 us` and `144--145 us`; Fatrop and MadNLP differ
from each other by only `0.49 us` and `1.05 us`. The deltoid patterns remain
almost unchanged. The executed fatigue objectives are `2177.52` for IPOPT,
`1751.06` for Fatrop, and `1696.11` for MadNLP. With no control
regularization, this lower scalar objective identifies a distinct local
minimum, not automatically a more plausible physiological strategy.

Remaining minimum capacity is still between 93.7 and 94.2 percent. The
absolute-angle run therefore remains a solver throughput and continuation
benchmark rather than a fatigue-to-failure experiment.

The source-built CasADi used by Alpaqa has `WITH_THREAD=ON`, and the workflow
rejects a runtime whose compiler flags do not expose that feature. Its two
600-second windows report roughly 2100 seconds of aggregate CPU time each,
confirming use of about 3.5 of the four available cores. With the screened
automatic initial penalty, the first limited candidate reaches
`3.51e-6` infeasibility and an IPOPT-like objective. It is physically feasible
at the common threshold, but remains invalidated because Alpaqa returns
`SOLVER_RET_LIMITED` and does not certify convergence or stationarity. The
second shifted-primal attempt degrades to `4.57e-2` infeasibility. Both perform
about 48,000 `psi` evaluations.

Constraint multiplier reuse is disabled in this final Alpaqa run. The second
RHO therefore reuses only the shifted/projected primal trajectory, not duals
from the first limited solve. The feasible-but-limited first candidate shows
that automatic penalty selection matters, while the failed continuation shows
that tuning alone has not made this collocation MHE usable. Further work
should prioritize constraint scaling and a smaller, less redundant
multiple-shooting or reduced formulation before another endurance run. These
results apply to CasADi 3.7.2's compatibility fork, not to current upstream
Alpaqa.

At RHO 30, IPOPT still has `min(A/A_scale)=0.98334`. This 30-cycle job is a
solver-throughput and stimulation-pattern benchmark, not a fatigue-to-failure
experiment. Hundreds of RHO, plausibly close to the requested 1000, remain
necessary to identify the physiological failure point.

MA57 is deliberately not part of the portable public-runner matrix because
CoinHSL cannot be redistributed with a public action. IPOPT-MUMPS is therefore
the Linux reference. A private runner can add MA57 once its licensed CoinHSL
path is provisioned locally.

## IPOPT with MA57 and compiled NLP evaluators

IPOPT can load CoinHSL without copying a dynamic library into the active Conda
environment:

```bash
--ipopt-linear-solver ma57 \
--ipopt-hsl-library \
  /Users/mickaelbegon/miniconda3/envs/Dev_bioptim/lib/libcoinhsl.2.dylib
```

The selected library is ARM64, contains the MA57 symbols, and has been
validated with IPOPT 3.14.19. Supplying the absolute path is safer than copying
it because its Homebrew runtime dependencies remain resolved from the original
installation. Do not use the small Julia `libhsl.dylib` shim shipped with the
MadNLP environment as an IPOPT CoinHSL replacement.

`--ipopt-c-compile` and the experimental `--madnlp-c-compile` generate native C
evaluators for the objective, constraints, gradient, Jacobian, and Lagrangian
Hessian. They do not compile IPOPT, MadNLP, or MA57 themselves. Each benchmark
uses a temporary build directory because Bioptim currently generates the fixed
filename `nlp.c`.

The JSON reports three distinct costs:

- `end_to_end_wall_time_s` includes OCP setup and C generation/compilation;
- the first window includes the initial solver setup;
- `hot_solver_time_*` excludes the first successful window and is the primary
  metric for repeated solves after the OCP has been constructed.

It also exports `nlp_solver_stats` for every window. These are the CasADi/IPOPT
`t_wall_*`, `t_proc_*`, and `n_call_*` counters. In particular, subtracting
objective, constraint, gradient, Hessian, and Jacobian evaluation wall times
from `t_wall_total` gives a useful, although still aggregate, estimate of the
IPOPT/linear-algebra remainder. It is not a pure factorization timer because it
also retains line-search and other IPOPT work. It nevertheless prevents
variable derivative-evaluation time from being attributed incorrectly to
MUMPS or MA57.

For a receding horizon, `window_objective_sum` is only the sum of the solved
subproblems. Horizons overlap whenever they contain more than one cycle but
advance by one cycle, so this sum is then not the cost of the executed
trajectory. `executed_fatigue_objective` re-evaluates
`10000 * integral(sum((1 - A / a_scale)^2))` with a common trapezoidal
quadrature on the unique exported cycles. The fatigue AUC, final
`A / a_scale`, and pulse-width saturation are reported alongside it.

On the tested Apple Silicon Mac, C compilation did not help either backend. For
the same three-cycle periodic problem, IPOPT-MA57 increased from 10.97 s to
15.72 s hot and from 20.64 s to 91.15 s end-to-end. MadNLP increased from
9.49 s to 12.41 s hot and from 29.61 s to 110.74 s end-to-end. The MadNLP
CasADi 3.8.0 generator also required a Clang workaround for invalid null-pointer
arithmetic, so compiled mode remains experimental and is disabled by default.

## Historical-profile fatigue benchmark

The following multi-window measurement was run on macOS on 2026-07-24 with
Python 3.11, CasADi 3.8.0, the two Bioptim integrations combined, two cycles per
horizon, three requested cycles, 30 stimulations per cycle, full state scaling,
the historical collocation seed, tolerance `1e-6`, and one solver process:

| Solver | Outcome | Validated cycles | Solver time | Iterations by window | Minimum final `A/A_scale` | Maximum mean normalized fatigue | Summed fatigue AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| MadNLP | success | 3 | 70.756 s | 125, 84 | 0.993626 | 0.003101 | 0.014412 muscle·cycle |
| Alpaqa | `SOLVER_RET_LIMITED` | 0 | 60.031 s | not exposed | not scored | not scored | not scored |

Alpaqa produced a physically plausible diagnostic trajectory, but its solver
status was non-zero when the configured 60 s wall-time limit was reached. It is
intentionally excluded from validated fatigue and per-cycle timing metrics; the
native status recorded in the JSON distinguishes a time limit from another
solver termination reason.

MadNLP's shifted second window took 26.899 s versus 43.857 s for the first
window, with 84 versus 125 iterations. This exercises the complete inter-window
pipeline and supports the warm-start choice, but it does not isolate the
individual contributions of the shifted primal and the bound multipliers.

The JSON summary records the model `a_scale` values, final primal infeasibility
against a `10 × tolerance` acceptance threshold, global state/control
finiteness, per-window status, objective, iterations and timing, warm-start
signatures and multiplier transfers, native solver status, and runtime
versions.

Both MadNLP windows pass the `1e-5` feasibility threshold: their final primal
infeasibilities are `3.36e-9` and `8.56e-8`. Alpaqa finishes with
`SOLVER_RET_LIMITED`, a primal infeasibility of `30.97`, and is rejected before
any fatigue metric is calculated.

## Periodic-collocation option screen

The following preliminary six-cycle screen used the same `periodic_node`,
constant `-0.2 N.m` torque,
Radau degree-three collocation, historical seed, full scaling, eight CasADi
workers, `1e-6` tolerance, and sequential solver processes:

| Backend | Dual transfer | Outcome | Hot median | Hot P90 | Window-objective sum | Minimum `A/A_scale` |
|---|---|---:|---:|---:|---:|---:|
| IPOPT-MA57 candidate | bounds | 6/6 cycles | 50.82 s | 66.76 s | 5.028 | 0.98883 |
| IPOPT-MUMPS | bounds | 6/6 cycles | 43.60 s | 47.10 s | 5.016 | 0.98877 |
| MadNLP | all | 6/6 cycles | 9.23 s | 9.77 s | 5.119 | 0.98873 |
| MadNLP | bounds | 6/6 cycles | 9.73 s | 10.22 s | 5.050 | 0.98863 |

The two IPOPT rows are a contemporary sequential pair. The MadNLP rows come
from the earlier low-load screen and must not be used to form a direct speedup
ratio against the later IPOPT seconds.

These absolute rows also predate the corrected 30-control cycle shift and the
explicit internal cycle-boundary constraint. They are retained only to explain
the MA57 investigation and must not be used as the final solver ranking.

The IPOPT rows use the same MUMPS-built warm-up seed. An earlier cache key
included the target linear solver, which gave MA57 and MUMPS different initial
trajectories and invalidated the apparent solution-quality advantage of MA57.
The corrected first window converges to the same objective
(`0.232408982275...`) in 99 iterations for both solvers. Tiny numerical
differences are subsequently amplified by the shifted primal/dual warm start
of this nonconvex receding horizon.

The MA57 candidate uses `ma57_pre_alloc=2`, `ma57_block_size=64`, and
`ma57_node_amalgamation=32`. The larger pre-allocation removes a factor-memory
reallocation; the other two values are architecture- and sparsity-dependent
screening candidates, not general optima.

A corrected `+0.22 N.m` paired run over ten cycles produced exactly the same
iteration sequence for MUMPS and MA57 (`69, 55, 73, 67, 70, 76, 72, 69, 77`),
the same window-objective sum to `1.4e-10`, and the same executed-fatigue
objective to `1e-10`. The apparent hot median nevertheless favoured MUMPS
(`8.09 s` versus `10.53 s`). The IPOPT timing output showed why that total was
not a clean linear-solver comparison: Hessian and Jacobian evaluation wall
times, which do not depend on the selected sparse factorizer, were much larger
during the MA57 process.

Two subsequent short repetitions recorded those oracle timings in JSON. At
the same 69 and 55 iterations, the median wall-time remainder after subtracting
all NLP evaluations was about 27--32 percent smaller with MA57. The total
ranking varied with the much larger multithreaded Hessian/Jacobian cost. This
remainder is consistent with a MA57 advantage but does not prove its cause,
because it is not an isolated factorization timer. The interpretation is
therefore:

- the ARM64 CoinHSL/MA57 installation is valid and there is no evidence of a
  wrong library or a different solution branch;
- total macOS wall time is noisy enough to hide or reverse that gain;
- compare repeated interleaved runs and the exported remainder, not a single
  total time;
- keep MUMPS as the operational baseline on this Mac because the longer
  ten-cycle total favoured it, and retain MA57 as the controlled-performance
  candidate when CoinHSL is available.

For MadNLP, transferring all multipliers was only about five percent faster in
this preliminary screen. Because neither constraint nor bound multiplier
blocks are structurally shifted, the corrected endurance configuration keeps
multiplier transfer `off`. The pinned runtime also rejects `mu_init`, so the
only enabled MadNLP hot start is the shifted/projected primal trajectory;
multiplier modes remain explicit performance ablations.

Absolute late-run wall times were affected by concurrent macOS indexing,
Defender, and desktop processes. The JSON stores every window separately and
records the threading environment. The table supports the configuration
choice, but it is not a hardware-independent speed claim.

## Endurance, torque sign, and terminal crank angle

A 30-cycle MadNLP run at the historical signed torque of `-0.22 N.m`
converged for all 29 overlapping windows. Its hot median and P90 were 30.91 s
and 41.52 s, its maximum final primal infeasibility was `3.67e-7`, and its
executed fatigue objective was 291.82. The minimum final `A/A_scale` was still
0.95362, so this run did not reach a physiological fatigue failure.

That result must not be interpreted as propulsion against a `0.22 N.m`
resistance. The crank coordinate and velocity decrease in the current model.
The constant torque is added directly to the generalized wheel torque, so its
mechanical power is

```text
P_external = tau_external * qdot_crank
```

With `tau=-0.22 N.m` and `qdot<0`, this power is positive: the external torque
drives the crank while the muscles regulate or brake the motion. A physically
resistive load for this rotation convention has positive torque. Benchmark
JSON now records `external_crank_power` and classifies it as `driving`,
`resistive`, or `neutral`; the endurance-to-failure sweep must use that
diagnostic instead of relying on the option's historical name.

The corrected positive-torque run at `+0.20 N.m` converged for all 29 windows
and 30 exported cycles. MadNLP's hot median and P90 were 9.37 s and 13.06 s,
the executed fatigue objective was 12094.87, and the minimum final
`A/A_scale` reached 0.67676. Mean external crank power was -1.249 W, confirming
that the load was resistive. This level still did not cause non-convergence;
the load sweep therefore continues upward in `0.02 N.m` steps with two
consecutive failures allowed.

At the next continuation point, `+0.22 N.m`, both interior-point solvers reach
the same failure region:

| Backend | Validated cycles before stop | Last valid-window iterations | Two failed attempts: iterations | Failed primal infeasibilities |
|---|---:|---:|---:|---:|
| MadNLP | 9 | 123 | 1306, 1323 | 0.01068, 0.01413 |
| IPOPT-MUMPS | 10 | 180 | 889, 837 | 0.01159, 0.01444 |

This is substantially stronger evidence than a 1000-iteration cutoff: MadNLP
was allowed 2000 iterations and still failed twice. The common, increasing
infeasibility after a long feasible prefix supports a fatigue-dependent
task-capacity boundary, and IPOPT is one cycle more robust in this experiment.
It is not a mathematical proof that the NLP is infeasible: the last accepted
capacity ratios are still approximately 0.846 for MadNLP and 0.831 for IPOPT,
and both are local nonlinear solvers. A formal fatigue-failure statement would
require a feasibility-restoration problem or a prescribed-cadence reduced
model showing that no admissible stimulation can supply the required crank
work.

The crank angle can be tightened independently of the solver with
`--first-node-wheel-q-slack` and `--terminal-wheel-q-slack`. The production
screen uses exact inter-window continuity and `0.002 rad` at internal and final
cycle boundaries. The runs reported above used a terminal center reconstructed
from the preceding window. At 30 cycles the maximum local turn-progress error
was 0.002002 rad, but its same-sign accumulation reached 0.0380 rad.

The current implementation instead anchors every internal and terminal center
to the original unwrapped crank angle:

```text
q_target(k) = q_initial + k * signed(2*pi)
```

The executed terminal state still becomes the exact first-node state of the
next RHO, but it no longer defines the next terminal target. Consequently,
`0.002 rad` remains an absolute tolerance around the requested cycle count and
cannot accumulate from one RHO to the next. The window diagnostic now reports
both local turn errors and absolute cycle-count errors, and rejects a
same-sign cumulative drift even when every individual turn is locally within
tolerance. Periodic-refinement, continuation, horizon-seed, and generated-code
cache signatures were advanced so a relative-reference seed cannot be
silently reused.

`physical_success` now checks progress against the problem's expected
`-2*pi rad/cycle` direction. A finite but reversed or incomplete rotation is
therefore rejected instead of being accepted merely because it contains no
large angle jump.

The internal seam constraint is currently added to IPOPT, MadNLP, and Alpaqa
periodic NLPs. ACADOS still uses its own terminal-bound/diagnostic path and
does not receive that same custom seam constraint. ACADOS-versus-NLP timings
are therefore not a strictly paired transcription until an equivalent stage
constraint is added to the ACADOS problem; the shared progress diagnostic at
least rejects a wrong or incomplete turn.

## Experimental removal of inactive pulse-width controls

Zero stimulation is not a zero pulse width in the Ding model. An inactive
control must be fixed to the rheobase-like lower bound `pd0` (approximately
`0.000131405 s` here), because recruitment depends on `PW-pd0`. The
experimental `--pulse-width-active-set warmup` option therefore:

1. obtains the target-independent IPOPT warm-up;
2. unions each muscle's active phase over the two cycles;
3. circularly expands that phase by a configurable number of stimulation
   nodes to account for delayed force production;
4. fixes only the remaining pulse widths to `pd0`.

The first mask table was invalidated by the control-shift bug described above.
After correcting the shift, imposing an internal seam, fixing inter-window
crank continuity exactly, using `0.02 rad` seam/terminal slack, disabling dual
transfer, and reusing the same warmup, the six-cycle screen gave:

| Formulation | Free pulse widths | Hot median / P90 | Iterations by window | Executed fatigue objective | Fatigue AUC |
|---|---:|---:|---:|---:|---:|
| full NLP | 240/240 | 7.96 / 15.32 s | 98, 90, 124, 107, 272 | 1.77384 | 0.047784 |
| warm-up mask, margin 3 | 94/240 | 7.51 / 8.65 s | 102, 95, 99, 121, 120 | 1.82693 | 0.047422 |
| warm-up mask, margin 4 | 110/240 | 6.17 / 6.21 s | 93, 88, 84, 93, 83 | 1.86633 | 0.048075 |

Margin 3 removes the late 272-iteration outlier and reduces total solver time
by 21 percent, but changes the executed fatigue objective by 2.99 percent and
worsens minimum capacity. Margin 4 is faster in this one run but changes the
executed fatigue objective by 5.21 percent. Neither meets the one-percent
quality criterion, and one repetition is insufficient to claim a stable
speedup.

Consequently, the fixed-node mask is an experimental diagnostic, not the
production formulation. A mask should be accepted only if, on the same seed
and terminal constraints, it stays within one percent of the full-NLP executed
fatigue objective and AUC, preserves the worst final capacity, and does not
create an iteration outlier. A safer use is to obtain a masked candidate and
then unfreeze and polish with the complete NLP.

The current mask uses nodewise equality bounds and does not remove symbols from
the CasADi graph. Larger structural gains require one of the following:

- parameterize every muscle's periodic pulse train with a small cyclic spline
  or Fourier basis, then polish in node space if necessary;
- reduce the constrained arm-crank mechanics to the single crank coordinate
  and precompute muscle effectiveness versus crank angle;
- prescribe cadence when the scientific question is stimulation scheduling,
  which makes terminal crank accuracy exact by construction and removes
  mechanical states and contact constraints;
- eliminate the analytically forced periodic calcium-sum state, then evaluate
  the slow fatigue states on a coarser grid.

For a geometry-derived muscle mask, usefulness must be evaluated in the
one-dimensional null space of the contact Jacobian. The relevant sign is the
future crank power produced after the electromechanical delay, not a raw
muscle moment arm or the instantaneous sign at the stimulation node.

## Interpretation

IPOPT-MUMPS remains the recommended reference. It certifies every assisted
RHO in all 100-cycle runs, is substantially faster than Fatrop in the current
compatibility mode, and is both faster and less exposed to tail latency than
MadNLP-PARDISO on the absolute-angle problem.

MadNLP remains relevant but is not ready to replace IPOPT. The MUMPS runs
showed a fast normal path but also a 2000-iteration failure and a converged
969-iteration outlier. The absolute-angle PARDISO runs certify every RHO, but
their normal path is slower than IPOPT and reproducibly contains
1177-iteration/141-second and 533-iteration/64-second outliers. The older
MUMPS timings and current PARDISO timings are not paired because the angular
formulation changed. A same-code, same-seed, repeated MUMPS/PARDISO screen is
required before concluding that the linear solver itself causes the
difference. The related Bioptim Linux benchmark still indicates that MadNLP
can be useful for long exact-Hessian collocation problems, but that indication
does not override the current OCP evidence.

Fatrop is now a functional independent structured solver for this benchmark:
it certifies all 100 RHO and respects the physical bounds. It does not provide
a speed advantage yet. The present time-major, unscaled-state compatibility
mode is 38 to 68 percent slower end-to-end than IPOPT across the available
100-RHO runs. On the relative-angle run it followed IPOPT closely; on the
absolute-angle run it instead follows MadNLP's lower-fatigue
biceps/triceps-sharing basin. A fair performance reevaluation requires either
preserving Fatrop's explicit gap structure under state scaling or adding
normalized gap constraints upstream; until then, the observed penalty cannot
be attributed to Fatrop alone.

Alpaqa is not usable for the present collocation MHE. Automatic penalty
selection can produce a
physically feasible first candidate after 600 seconds, but the solver still
returns a time-limit status and the next shifted window is strongly
infeasible. The integration-branch benchmarks also show that Alpaqa is slower
than IPOPT on a small cube problem and misses a 0.5 s deadline on shifted NMPC
windows. Its augmented-Lagrangian/PANOC method appears much more sensitive to
scaling, redundant collocation constraints, and shifted feasibility than the
interior-point solvers.

Recommended use:

- keep IPOPT as the reference solver;
- use MadNLP at `1e-8` as an experimental alternative and independent
  local-minimum check, reporting its linear solver, preparation, hot
  execution, outliers and strict-prefix length separately;
- do not prefer PARDISO over MUMPS for this OCP without a paired
  absolute-angle comparison;
- use Fatrop as an independent structured feasibility and local-minimum check,
  while reporting its time-major ordering and absence of state scaling;
- leave Alpaqa out of production and endurance matrices. Retain only the
  explicit diagnostic path until a dedicated multiple-shooting or less
  redundant formulation demonstrates reliable multi-window convergence.
