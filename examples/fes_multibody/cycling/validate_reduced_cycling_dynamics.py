"""Build and validate the one-DoF cycling mechanical reduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cocofest import (
    benchmark_reduced_casadi_mechanical_kernel,
    build_reduced_cycling_dynamics,
    validate_reduced_cycling_dynamics,
)


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = (
    EXAMPLE_DIR
    / "../../msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod"
).resolve()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=181)
    parser.add_argument("--kinematic-order", type=int, default=12)
    parser.add_argument("--dynamics-order", type=int, default=12)
    parser.add_argument("--validation-samples", type=int, default=200)
    parser.add_argument("--external-crank-torque", type=float, default=-0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--casadi-profile", action="store_true")
    parser.add_argument("--casadi-repeats", type=int, default=1000)
    parser.add_argument(
        "--output-profile",
        type=Path,
        default=EXAMPLE_DIR / "result/reduced_cycling_dynamics.npz",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main() -> dict:
    args = build_argument_parser().parse_args()
    reduced, build_audit = build_reduced_cycling_dynamics(
        args.model,
        sample_count=args.samples,
        kinematic_order=args.kinematic_order,
        dynamics_order=args.dynamics_order,
    )
    profile_path = reduced.save(args.output_profile)
    validation = validate_reduced_cycling_dynamics(
        args.model,
        reduced,
        sample_count=args.validation_samples,
        external_crank_torque=args.external_crank_torque,
        seed=args.seed,
    )
    report = {
        "model": str(args.model.resolve()),
        "profile": str(profile_path.resolve()),
        "muscle_names": list(reduced.muscle_names),
        "state_definition": {
            "theta": "unwrapped physical crank angle (rad)",
            "omega": "physical crank angular velocity (rad/s), not constrained to be constant",
        },
        "build": build_audit,
        "validation": validation,
    }
    if args.casadi_profile:
        report["casadi_profile"] = benchmark_reduced_casadi_mechanical_kernel(
            args.model,
            reduced,
            repeats=args.casadi_repeats,
            external_crank_torque=args.external_crank_torque,
        )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    return report


if __name__ == "__main__":
    main()
