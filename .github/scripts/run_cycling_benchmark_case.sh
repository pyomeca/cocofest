#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 5 || "$#" -gt 9 ]]; then
  echo "usage: $0 CASE SOLVER MECHANICS BACKEND ODE [ROOT] [WINDOWS] [COMPILE] [GRAPH]" >&2
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
graph_mode="${9:-}"
case_dir="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}/${case_root}/${case_slug}-${mechanics}"
result="$case_dir/result.json"
solver_options=()
solver_tolerance=1e-6

mkdir -p "$case_dir"

if [[ -z "$graph_mode" ]]; then
  # The Linux 30-RHO screen shows a ~60% hot-solve reduction for IPOPT and
  # MadNLP with SX. Fatrop remains MX until its independent screen completes.
  if [[ "$solver" == "fatrop" ]]; then
    graph_mode=mx
  else
    graph_mode=sx
  fi
fi
case "$graph_mode" in
  sx) solver_options+=(--ipopt-use-sx) ;;
  mx) solver_options+=(--ipopt-no-use-sx) ;;
  *) echo "GRAPH must be 'sx' or 'mx', got '$graph_mode'." >&2; exit 2 ;;
esac

if [[ "$solver" == "ipopt" ]]; then
  solver_options+=(
    --ipopt-max-iter "$BENCHMARK_MAX_ITER"
    --ipopt-dual-warm-start-mode bounds
  )
elif [[ "$solver" == "fatrop" ]]; then
  solver_options+=(
    --fatrop-max-iter 1000
    --fatrop-structure-detection auto
    --fatrop-bound-tightening-factor 1e-8
    --fatrop-state-scaling none
    --fatrop-dual-warm-start-mode off
  )
  if [[ "$compile_mode" == "true" ]]; then
    solver_options+=(--fatrop-c-compile)
  fi
  if [[ "$ode_solver" == "rk4" ]]; then
    solver_options+=(--ipopt-ode-solver rk4 --ipopt-rk-steps 5)
  else
    solver_options+=(
      --ipopt-ode-solver collocation
      --ipopt-collocation-degree 3
      --ipopt-collocation-method radau
    )
  fi
elif [[ "$solver" == "madnlp" ]]; then
  solver_tolerance=1e-8
  solver_options+=(
    --madnlp-max-iter "$BENCHMARK_MAX_ITER"
    --madnlp-linear-solver "$backend"
    --madnlp-dual-warm-start-mode off
  )
fi
if [[ "$mechanics" == "reduced" ]]; then
  solver_options+=(--mechanical-formulation reduced)
fi

# Keep the standard bridge enabled: the certified common seed records one
# consumed warmup cycle and rejects consumers configured with zero.
set +e
set -o pipefail
python examples/fes_multibody/cycling/cycling_fes_solver_comparison.py \
  --solvers "$solver" \
  --objective fatigue \
  --ipopt-profile periodic_collocation \
  --cycles-per-window "$BENCHMARK_CYCLES_PER_WINDOW" \
  --stimulations-per-cycle 30 \
  --n-windows "$case_windows" \
  --n-threads "$BENCHMARK_THREADS" \
  --crank-assistance "$BENCHMARK_ASSISTANCE" \
  --nlp-tolerance "$solver_tolerance" \
  --primal-feasibility-threshold 1e-5 \
  --max-consecutive-failing 2 \
  --standard-warmup-seed .github/benchmark-seeds/legacy-resistive-0p22-warmup.npz \
  --legacy-standard-warmup-seed-signed-torque 0.22 \
  --standard-warmup-seed-continuation \
  --common-initial-solution "benchmark-seed/common-${mechanics}.npz" \
  --no-optional-nlp-periodic-ipopt-hot-start \
  --warmup-ipopt-linear-solver mumps \
  --ipopt-linear-solver mumps \
  --ipopt-disable-historical-initial-guess \
  --state-scaling full \
  --first-node-wheel-q-slack 0 \
  --terminal-wheel-q-slack "$BENCHMARK_Q_SLACK" \
  --compact-rho-output \
  --print-traces \
  --output-json "$result" \
  "${solver_options[@]}" \
  2>&1 | tee "$case_dir/solver.log"
solver_exit="${PIPESTATUS[0]}"
set -e
echo "$solver_exit" > "$case_dir/process-exit-code.txt"

# Numerical non-convergence belongs in result.json and must not prevent later
# cases or the immediate artifact checkpoint from running on the same machine.
exit 0
