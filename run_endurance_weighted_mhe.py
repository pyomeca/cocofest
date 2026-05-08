from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="Run cycling MHE with fixed endurance weights.")
    parser.add_argument(
        "--cycles",
        type=int,
        default=10000,
        help="Total simulated cycles. Default is intentionally long so the run stops on failure rather than at 20 cycles.",
    )
    parser.add_argument("--simultaneous", type=int, default=3, help="Number of simultaneous cycles in the MHE window.")
    parser.add_argument("--frequency", type=int, default=30, help="Stimulation frequency in Hz.")
    parser.add_argument("--torque", type=float, default=-0.20, help="Resistive crank torque in Nm.")
    parser.add_argument(
        "--objective",
        type=str,
        default="minimize_endurance_1500_weighted_fatigue",
        choices=[
            "minimize_weighted_root_mean_square_fatigue",
            "minimize_weighted_square_fatigue",
            "minimize_endurance_1500_weighted_fatigue",
            "minimize_endurance_fixed_weight_risk_to_failure",
            "minimize_endurance_adaptive_weight_risk_to_failure",
        ],
        help="Custom cycling endurance objective to use in the MHE.",
    )
    parser.add_argument(
        "--linear-solver",
        type=str,
        default=None,
        help="Optional IPOPT linear solver override. Default is ma57 on Linux/macOS, mumps otherwise.",
    )
    parser.add_argument(
        "--solver-config",
        type=str,
        default="baseline",
        choices=["baseline", "exact_jit", "lm", "two_stage"],
        help="Ipopt/CasADi configuration to compare inside the existing MHE workflow.",
    )
    parser.add_argument(
        "--lm-iter",
        type=int,
        default=20,
        help="Number of limited-memory IPOPT iterations in two-stage mode.",
    )
    parser.add_argument("--max-iter", type=int, default=2000, help="Maximum IPOPT iterations per window.")
    parser.add_argument(
        "--n-threads", type=int, default=4, help="Thread count passed to the NMPC backend and BLAS/OpenMP."
    )
    parser.add_argument("--hsllib", type=str, default=None, help="Optional path to an HSL shared library for IPOPT.")
    parser.add_argument("--save", action="store_true", help="Save the solution pickle.")
    parser.add_argument("--with-init-guess", action="store_true", help="Generate initial guesses if needed.")
    args = parser.parse_args()

    # Keep matplotlib headless in this environment.
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cache")
    os.environ["OMP_NUM_THREADS"] = str(args.n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.n_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.n_threads)

    repo_root = Path(__file__).resolve().parent
    cycling_dir = repo_root / "examples" / "fes_multibody" / "cycling"
    if not cycling_dir.exists():
        fallback_root = repo_root / "cocofest"
        fallback_cycling_dir = fallback_root / "examples" / "fes_multibody" / "cycling"
        if fallback_cycling_dir.exists():
            repo_root = fallback_root
            cycling_dir = fallback_cycling_dir
        else:
            raise FileNotFoundError(f"Could not locate cycling example directory from {repo_root}")
    os.chdir(cycling_dir)

    from examples.fes_multibody.cycling.cycling_pulse_width_mhe import main as run_main

    linear_solver = args.linear_solver
    if linear_solver is None:
        linear_solver = "ma57" if (sys.platform.startswith("linux") or sys.platform == "darwin") else "mumps"

    run_main(
        stimulation_frequency=args.frequency,
        n_total_cycle=args.cycles,
        n_cycles_simultaneous=[args.simultaneous],
        resistive_torque=args.torque,
        cost_fun_dict={"optimized_function": [[args.objective]]},
        init_guess=args.with_init_guess,
        save=args.save,
        n_threads=args.n_threads,
        ipopt_linear_solver=linear_solver,
        ipopt_max_iter=args.max_iter,
        ipopt_hsllib=args.hsllib,
        solver_config=args.solver_config,
        two_stage_lm_iter=args.lm_iter,
    )


if __name__ == "__main__":
    main()
