#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 5 || "$#" -gt 14 ]]; then
  echo "usage: $0 CASE SOLVER MECHANICS BACKEND ODE [ROOT] [WINDOWS] [COMPILE] [GRAPH] [FATROP_SCALING] [COLLOCATION_DEGREE] [IPOPT_PROFILE] [DUAL_WARM_START] [TARGET_REFINEMENT]" >&2
  exit 2
fi

case_slug="$1"
solver="$2"
mechanics="$3"
backend="$4"
ode_solver="$5"
case_root="${6:-benchmark-results}"
case_windows="${7:-${BENCHMARK_CYCLES:?BENCHMARK_CYCLES is required}}"
compile_mode="${8:-false}"
graph_mode="${9:-sx}"
fatrop_state_scaling="${10:-none}"
collocation_degree="${11:-3}"
ipopt_profile="${12:-periodic_collocation}"
dual_warm_start="${13:-auto}"
target_refinement="${14:-auto}"
workspace="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
case_dir="${workspace}/${case_root}/${case_slug}-${mechanics}"
result="$case_dir/result.json"
solver_options=()
initialization_options=(--no-optional-nlp-periodic-ipopt-hot-start)
solver_tolerance=1e-6

if ! [[ "$collocation_degree" =~ ^[2-9]$ ]]; then
  echo "COLLOCATION_DEGREE must be an integer between 2 and 9, got '$collocation_degree'." >&2
  exit 2
fi
case "$ipopt_profile" in
  historical|periodic_collocation|periodic-collocation|scientific_radau4|scientific-radau4|scientific_radau5|scientific-radau5|scientific_radau6|scientific-radau6|acados_like|acados-like) ;;
  *) echo "IPOPT_PROFILE is not supported: '$ipopt_profile'." >&2; exit 2 ;;
esac
case "$dual_warm_start" in
  auto|off|constraints|bounds|all) ;;
  *) echo "DUAL_WARM_START must be auto, off, constraints, bounds, or all; got '$dual_warm_start'." >&2; exit 2 ;;
esac
case "$target_refinement" in
  auto|true|false) ;;
  *) echo "TARGET_REFINEMENT must be auto, true, or false; got '$target_refinement'." >&2; exit 2 ;;
esac

if [[ "$target_refinement" == "true" ]]; then
  initialization_options=(--optional-nlp-periodic-ipopt-hot-start)
fi

mkdir -p "$case_dir"
# CasADi emits fixed filenames such as nlp.c/nlp.so in the current directory.
# Give every solver/mechanics process a fresh directory so a full result cannot
# certify a reduced run (or vice versa) through stale generated files.
codegen_dir="$(mktemp -d "$case_dir/codegen.XXXXXX")"

case "$graph_mode" in
  sx) solver_options+=(--ipopt-use-sx) ;;
  *)
    echo "The endurance benchmark is SX-only; GRAPH must be 'sx', got '$graph_mode'." >&2
    exit 2
    ;;
esac

if [[ "$solver" == "ipopt" ]]; then
  if [[ "$dual_warm_start" == "auto" ]]; then
    dual_warm_start="bounds"
  fi
  solver_options+=(
    --ipopt-max-iter "$BENCHMARK_MAX_ITER"
    --ipopt-dual-warm-start-mode "$dual_warm_start"
  )
  if [[ "$compile_mode" == "true" ]]; then
    solver_options+=(--ipopt-c-compile)
  fi
elif [[ "$solver" == "fatrop" ]]; then
  if [[ "$dual_warm_start" == "auto" ]]; then
    dual_warm_start="off"
  fi
  case "$fatrop_state_scaling" in
    none|full) ;;
    *) echo "FATROP_SCALING must be 'none' or 'full', got '$fatrop_state_scaling'." >&2; exit 2 ;;
  esac
  solver_options+=(
    --fatrop-max-iter 1000
    --fatrop-structure-detection auto
    --fatrop-bound-tightening-factor 1e-8
    --fatrop-state-scaling "$fatrop_state_scaling"
    --fatrop-dual-warm-start-mode "$dual_warm_start"
  )
  if [[ "$compile_mode" == "true" ]]; then
    solver_options+=(--fatrop-c-compile)
  fi
  if [[ "$ode_solver" == "rk4" ]]; then
    solver_options+=(--ipopt-ode-solver rk4 --ipopt-rk-steps 5)
  else
    solver_options+=(
      --ipopt-ode-solver collocation
      --ipopt-collocation-degree "$collocation_degree"
      --ipopt-collocation-method radau
    )
  fi
elif [[ "$solver" == "madnlp" ]]; then
  if [[ "$dual_warm_start" == "auto" ]]; then
    dual_warm_start="off"
  fi
  solver_tolerance=1e-8
  # The common seed is intentionally solver-independent, but the non-convex
  # one-cycle IPOPT validation can select a PW branch that is difficult for
  # MadNLP. Refine that same seed once with the target transcription before
  # timing the compiled MadNLP RHO loop. This setup cost remains reported in
  # initial_guess_preparation_time_s and does not rebuild the MadNLP graph.
  if [[ "$target_refinement" == "auto" ]]; then
    initialization_options=(--optional-nlp-periodic-ipopt-hot-start)
  fi
  solver_options+=(
    --madnlp-max-iter "$BENCHMARK_MAX_ITER"
    --madnlp-linear-solver "$backend"
    --madnlp-dual-warm-start-mode "$dual_warm_start"
  )
  if [[ "$compile_mode" == "true" ]]; then
    solver_options+=(--madnlp-c-compile)
  fi
fi
if [[ "$solver" != "fatrop" && "$ode_solver" == "collocation" ]]; then
  solver_options+=(
    --ipopt-ode-solver collocation
    --ipopt-collocation-degree "$collocation_degree"
    --ipopt-collocation-method radau
  )
fi
if [[ "$mechanics" == "reduced" ]]; then
  solver_options+=(--mechanical-formulation reduced)
else
  # The exact full contact equality stalls both interior-point solvers on the
  # same 10-20 µm seam residual. This explicit 20 µm band is still only
  # 0.02 % of the 0.1 m crank radius and is tighter than the angular endpoint
  # tolerance used by the benchmark.
  solver_options+=(--full-contact-position-tolerance 2e-5)
fi
if [[ "$mechanics" != "reduced" && "$ode_solver" != "collocation" ]]; then
  # A one-cycle horizon has no future tail to shift. The exact terminal state
  # therefore replaces node 0 while nodes 1..N retain the previous cycle's
  # shape, which can create a large full-dynamics defect. The current phase-I
  # projector can repair shooting grids, but intentionally rejects direct
  # collocation because its internal state nodes need a dedicated projection.
  solver_options+=(
    --shared-transfer-phase-one
    --acados-transfer-phase-one-mode all
  )
fi

# Keep the standard bridge enabled: the certified common seed records one
# consumed warmup cycle and rejects consumers configured with zero.
set +e
set -o pipefail
pushd "$codegen_dir" >/dev/null
heartbeat() {
  while sleep 45; do
    echo "benchmark heartbeat: ${case_slug}/${mechanics} is still running"
  done
}
heartbeat &
heartbeat_pid=$!
python "$workspace/examples/fes_multibody/cycling/cycling_fes_solver_comparison.py" \
  --solvers "$solver" \
  --objective fatigue \
  --ipopt-profile "$ipopt_profile" \
  --ipopt-enforce-start-constraints \
  --cycles-per-window "$BENCHMARK_CYCLES_PER_WINDOW" \
  --stimulations-per-cycle 30 \
  --n-windows "$case_windows" \
  --n-threads "$BENCHMARK_THREADS" \
  --crank-assistance "$BENCHMARK_ASSISTANCE" \
  --nlp-tolerance "$solver_tolerance" \
  --primal-feasibility-threshold 1e-5 \
  --max-consecutive-failing 2 \
  --standard-warmup-seed "$workspace/.github/benchmark-seeds/legacy-resistive-0p22-warmup.npz" \
  --legacy-standard-warmup-seed-signed-torque 0.22 \
  --standard-warmup-seed-continuation \
  --common-initial-solution "$workspace/benchmark-seed/common-reduced.npz" \
  "${initialization_options[@]}" \
  --warmup-ipopt-linear-solver mumps \
  --ipopt-linear-solver mumps \
  --ipopt-disable-historical-initial-guess \
  --reduced-cycling-profile "$workspace/benchmark-seed/reduced-cycling-fourier12.npz" \
  --state-scaling full \
  --first-node-wheel-q-slack 0 \
  --terminal-wheel-q-slack "$BENCHMARK_Q_SLACK" \
  --compact-rho-output \
  --print-traces \
  --output-json "$result" \
  "${solver_options[@]}" \
  2>&1 | tee "$case_dir/solver.log"
solver_exit="${PIPESTATUS[0]}"
kill "$heartbeat_pid" 2>/dev/null || true
wait "$heartbeat_pid" 2>/dev/null || true
set -e
popd >/dev/null
echo "$solver_exit" > "$case_dir/process-exit-code.txt"

# A libMad type warning means that the requested backend was not applied even
# when MadNLP subsequently converges with its default.  Treat this as an
# infrastructure/configuration failure so a mislabeled benchmark cannot pass.
if grep -Fq "libMAD WARNING: option linear_solver is of unknown type" "$case_dir/solver.log"; then
  echo "libMad rejected the requested linear_solver type." >&2
  exit 1
fi

if [[ -f "$result" ]] && ! jq -e --arg solver "$solver" \
  '.configurations[$solver].use_sx == true' "$result" >/dev/null
then
  echo "The generated result is not SX even though the benchmark is SX-only." >&2
  exit 1
fi

if [[ -f "$result" && "$ipopt_profile" =~ ^scientific[-_]radau[456]$ ]]
then
  normalized_profile="${ipopt_profile//_/-}"
  scientific_status="diagnostic"
  if [[ "$collocation_degree" == "5" ]]; then
    scientific_status="candidate"
  fi
  if ! jq -e \
    --arg solver "$solver" \
    --arg profile "$normalized_profile" \
    --arg status "$scientific_status" \
    --argjson degree "$collocation_degree" '
    .configurations[$solver] |
    (.benchmark_profile == $profile) and
    (.profile_integrity == true) and
    (.scientific_status == $status) and
    (.collocation_degree == $degree) and
    (.enforce_start_constraints == true) and
    (.activate_passive_force_relationship == true) and
    (.control_decisions_per_cycle == 30)
  ' "$result" >/dev/null
  then
    echo "The $normalized_profile result does not satisfy its serialized contract." >&2
    exit 1
  fi
fi

# Numerical non-convergence belongs in result.json and must not prevent later
# cases or the immediate artifact checkpoint from running on the same machine.
exit 0
