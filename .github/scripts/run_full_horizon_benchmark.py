#!/usr/bin/env python3
"""Run an RSS-bounded RHO-seeded full-horizon MadNLP size sweep.

The reduced one-cycle RHO is solved once up to the requested ceiling.  Its
concatenated trajectory is then sliced into solver-neutral seeds for
full-mechanics, single-shot horizons of increasing size. Each problem is
warm-started from the matching RHO prefix rather than from the previous
full-horizon solution, keeping the comparison paired while progressively
increasing the NLP size.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Iterable

import numpy as np

GIB = 1024**3
SMALL_RUNNER_RSS_LIMIT_GIB = 12.5
LARGE_RUNNER_RSS_LIMIT_GIB = 97.5


def horizon_sweep_targets(max_cycles: int) -> list[int]:
    """Return the adaptive sparse/5/10-cycle size ladder."""

    if max_cycles < 1:
        raise ValueError("max_cycles must be strictly positive.")
    targets = [
        value
        for value in (1, 2, 3, 5, 10, 15, 20, 25, 30)
        if value <= max_cycles
    ]
    if max_cycles > 30:
        targets.extend(range(35, min(max_cycles, 60) + 1, 5))
    if max_cycles > 60:
        targets.extend(range(70, max_cycles + 1, 10))
    if targets[-1] != max_cycles:
        targets.append(max_cycles)
    return targets


def refinement_targets(last_success: int, first_failure: int) -> list[int]:
    """Fill the final coarse interval without retrying either endpoint."""

    if last_success < 0 or first_failure <= last_success:
        raise ValueError("The refinement interval must be ordered.")
    return list(range(last_success + 1, first_failure))


def automatic_rss_limit_gib(total_memory_bytes: int) -> float:
    """Choose a conservative RSS cap for 16 GiB and 128 GiB runners."""

    if total_memory_bytes <= 0:
        raise ValueError("total_memory_bytes must be strictly positive.")
    total_gib = total_memory_bytes / GIB
    if total_gib <= 32.0:
        return min(SMALL_RUNNER_RSS_LIMIT_GIB, 0.80 * total_gib)
    if total_gib >= 96.0:
        return min(LARGE_RUNNER_RSS_LIMIT_GIB, 0.80 * total_gib)
    # Intermediate machines are not benchmark targets, but retaining roughly
    # 22 % headroom is safer than extrapolating the 128 GiB absolute cap.
    return 0.78 * total_gib


def _read_positive_integer(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw.isdigit():
        return None
    value = int(raw)
    return value if value > 0 else None


def available_memory_bytes() -> int:
    """Return the tighter physical/cgroup memory allocation on Linux."""

    physical = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                physical = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        pass

    cgroup_limits = [
        _read_positive_integer(Path("/sys/fs/cgroup/memory.max")),
        _read_positive_integer(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")),
    ]
    candidates = [value for value in (physical, *cgroup_limits) if value]
    if not candidates:
        raise RuntimeError("Cannot determine the runner memory allocation.")
    # Some cgroup v1 hosts expose an effectively infinite sentinel.
    finite = [value for value in candidates if value < (1 << 60)]
    return min(finite or candidates)


def _child_pids(pid: int) -> tuple[int, ...]:
    children_path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        return tuple(int(value) for value in children_path.read_text().split())
    except (OSError, ValueError):
        return ()


def process_tree_pids(root_pid: int) -> set[int]:
    pending = [root_pid]
    observed: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in observed:
            continue
        observed.add(pid)
        pending.extend(_child_pids(pid))
    return observed


def _process_rss_bytes(pid: int) -> int:
    try:
        lines = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    for line in lines:
        if line.startswith("VmRSS:"):
            try:
                return int(line.split()[1]) * 1024
            except (ValueError, IndexError):
                return 0
    return 0


def process_tree_rss_bytes(root_pid: int) -> int:
    return sum(_process_rss_bytes(pid) for pid in process_tree_pids(root_pid))


@dataclass
class MonitoredRun:
    command: list[str]
    return_code: int
    peak_rss_bytes: int
    elapsed_s: float
    memory_limit_exceeded: bool
    timed_out: bool
    log_path: str


def _terminate_process_group(process: subprocess.Popen, grace_s: float = 10.0) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_s
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_monitored(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    rss_limit_bytes: int,
    poll_interval_s: float = 0.5,
    timeout_s: float | None = None,
) -> MonitoredRun:
    """Run one solver process and stop its whole process group at the RSS cap."""

    if rss_limit_bytes <= 0:
        raise ValueError("rss_limit_bytes must be strictly positive.")
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError("timeout_s must be strictly positive when provided.")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    peak_rss = 0
    memory_limit_exceeded = False
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        next_heartbeat = start + 30.0
        while process.poll() is None:
            rss = process_tree_rss_bytes(process.pid)
            peak_rss = max(peak_rss, rss)
            if rss >= rss_limit_bytes:
                memory_limit_exceeded = True
                message = (
                    f"RSS limit reached: {rss / GIB:.3f} GiB >= "
                    f"{rss_limit_bytes / GIB:.3f} GiB"
                )
                print(message, flush=True)
                log.write(message + "\n")
                log.flush()
                _terminate_process_group(process)
                break
            now = time.monotonic()
            if timeout_s is not None and now - start >= timeout_s:
                timed_out = True
                message = f"Attempt timeout reached after {now - start:.1f} s"
                print(message, flush=True)
                log.write(message + "\n")
                log.flush()
                _terminate_process_group(process)
                break
            if now >= next_heartbeat:
                print(
                    f"full-horizon heartbeat: pid={process.pid} "
                    f"rss={rss / GIB:.3f} GiB peak={peak_rss / GIB:.3f} GiB",
                    flush=True,
                )
                next_heartbeat = now + 30.0
            time.sleep(poll_interval_s)
        return_code = process.wait()
        peak_rss = max(peak_rss, process_tree_rss_bytes(process.pid))

    return MonitoredRun(
        command=command,
        return_code=return_code,
        peak_rss_bytes=peak_rss,
        elapsed_s=time.monotonic() - start,
        memory_limit_exceeded=memory_limit_exceeded,
        timed_out=timed_out,
        log_path=str(log_path),
    )


def _load_metadata(data) -> dict:
    if "metadata__json" not in data.files:
        raise ValueError("The RHO seed has no metadata__json entry.")
    return json.loads(str(data["metadata__json"].item()))


def write_rho_seed_prefix(
    source_path: Path, output_path: Path, target_cycles: int
) -> dict:
    """Slice a concatenated RHO seed to an exact multi-cycle prefix."""

    if target_cycles < 1:
        raise ValueError("target_cycles must be strictly positive.")
    with np.load(source_path, allow_pickle=False) as data:
        metadata = _load_metadata(data)
        source_cycles = int(metadata["cycles_per_window"])
        if target_cycles > source_cycles:
            raise ValueError(
                f"Cannot extract {target_cycles} cycles from {source_cycles}."
            )
        payload: dict[str, np.ndarray] = {}
        for key in data.files:
            if key == "metadata__json":
                continue
            values = np.asarray(data[key])
            if key.startswith("states__"):
                intervals, remainder = divmod(values.shape[-1] - 1, source_cycles)
                if remainder:
                    raise ValueError(
                        f"State seed '{key}' cannot be divided into "
                        f"{source_cycles} cycles."
                    )
                payload[key] = values[..., : target_cycles * intervals + 1]
            elif key.startswith("controls__"):
                nodes, remainder = divmod(values.shape[-1], source_cycles)
                if remainder:
                    raise ValueError(
                        f"Control seed '{key}' cannot be divided into "
                        f"{source_cycles} cycles."
                    )
                payload[key] = values[..., : target_cycles * nodes]
            else:
                payload[key] = values
    metadata.update(
        {
            "cycles_per_window": target_cycles,
            "producer_mode": "receding_horizon_prefix",
            "producer_source_cycles": source_cycles,
        }
    )
    payload["metadata__json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return metadata


def _benchmark_success(
    result_path: Path,
    *,
    expected_mode: str,
    expected_cycles: int,
    expected_solver: str,
) -> bool:
    """Require a solver and physical certificate for the complete horizon."""

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = payload["results"][0]
        configuration = payload["configurations"][expected_solver]
        expected_mechanics = "full" if expected_mode == "single_shot" else "reduced"
        expected_use_sx = expected_solver == "ipopt"
        return bool(
            result["success"]
            and result.get("solver_success") is True
            and result.get("physical_success") is True
            and result.get("solver") == expected_solver
            and result.get("mode") == expected_mode
            and int(result.get("covered_cycles") or 0) == expected_cycles
            and int(result.get("physically_validated_cycles") or 0) == expected_cycles
            and configuration.get("single_shot")
            is (expected_mode == "single_shot")
            and configuration.get("mechanical_formulation") == expected_mechanics
            and int(configuration.get("cycles_per_window") or 0)
            == (expected_cycles if expected_mode == "single_shot" else 1)
            and int(configuration.get("n_windows") or 0) == expected_cycles
            and configuration.get("use_sx") is expected_use_sx
            and configuration.get(f"{expected_solver}_linear_solver") == "mumps"
        )
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False


def _benchmark_payload_is_readable(result_path: Path) -> bool:
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = payload["results"][0]
        return isinstance(result, dict) and result.get("error") is None
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False


def _log_has_unknown_mumps_warning(log_path: str | Path) -> bool:
    log_path = Path(log_path)
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return (
        "libMAD WARNING: option linear_solver is of unknown type mumps" in text
    )


def _benchmark_validated_cycles(
    result_path: Path,
    *,
    expected_mode: str,
    expected_solver: str,
    expected_requested_cycles: int,
) -> int:
    """Return the complete solver/physical prefix reported for one backend."""

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = payload["results"][0]
        configuration = payload["configurations"][expected_solver]
        if (
            result.get("solver") != expected_solver
            or result.get("mode") != expected_mode
            or configuration.get("single_shot")
            is not (expected_mode == "single_shot")
            or configuration.get("mechanical_formulation") != "reduced"
            or int(configuration.get("cycles_per_window") or 0) != 1
            or int(configuration.get("n_windows") or 0)
            != expected_requested_cycles
            or configuration.get("use_sx") is not True
            or configuration.get(f"{expected_solver}_linear_solver") != "mumps"
        ):
            return 0
        covered = int(result.get("covered_cycles") or 0)
        physical = int(result.get("physically_validated_cycles") or 0)
        return min(covered, physical)
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return 0


def _seed_cycle_count(seed_path: Path) -> int:
    try:
        with np.load(seed_path, allow_pickle=False) as data:
            return int(_load_metadata(data)["cycles_per_window"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def _common_solver_options(args: argparse.Namespace) -> list[str]:
    return [
        "--objective",
        "fatigue",
        "--ipopt-profile",
        "periodic_collocation",
        "--ipopt-enforce-start-constraints",
        "--stimulations-per-cycle",
        "30",
        "--n-threads",
        str(args.n_threads),
        "--crank-assistance",
        str(args.crank_assistance),
        "--nlp-tolerance",
        "1e-8",
        "--primal-feasibility-threshold",
        "1e-5",
        "--standard-warmup-seed",
        str(
            args.workspace / ".github/benchmark-seeds/legacy-resistive-0p22-warmup.npz"
        ),
        "--legacy-standard-warmup-seed-signed-torque",
        "0.22",
        "--standard-warmup-seed-continuation",
        "--warmup-ipopt-linear-solver",
        "mumps",
        "--ipopt-linear-solver",
        "mumps",
        "--madnlp-linear-solver",
        "mumps",
        "--madnlp-max-iter",
        str(args.max_iterations),
        "--ipopt-disable-historical-initial-guess",
        "--reduced-cycling-profile",
        str(args.seed_dir / "reduced-cycling-fourier12.npz"),
        "--state-scaling",
        "full",
        "--first-node-wheel-q-slack",
        "0",
        "--terminal-wheel-q-slack",
        str(args.terminal_wheel_q_slack),
        "--compact-rho-output",
        "--print-traces",
    ]


def _rho_command(
    args: argparse.Namespace, result_path: Path, seed_path: Path
) -> list[str]:
    return [
        args.python,
        str(
            args.workspace
            / "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py"
        ),
        "--solvers",
        "ipopt",
        *_common_solver_options(args),
        "--ipopt-use-sx",
        "--no-optional-nlp-periodic-ipopt-hot-start",
        "--ipopt-max-iter",
        str(args.max_iterations),
        "--cycles-per-window",
        "1",
        "--n-windows",
        str(args.max_cycles),
        "--max-consecutive-failing",
        "1",
        "--mechanical-formulation",
        "reduced",
        "--common-initial-solution",
        str(args.seed_dir / "common-reduced.npz"),
        "--receding-horizon-solution-output",
        str(seed_path),
        "--allow-partial-receding-horizon-solution-output",
        "--output-json",
        str(result_path),
    ]


def _full_horizon_command(
    args: argparse.Namespace,
    cycles: int,
    seed_path: Path,
    result_path: Path,
    solution_path: Path,
) -> list[str]:
    command = [
        args.python,
        str(
            args.workspace
            / "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py"
        ),
        "--solvers",
        "madnlp",
        *_common_solver_options(args),
        "--ipopt-no-use-sx",
    ]
    if cycles >= 3:
        # The historical bridge has 60 controls and can initialize at most two
        # 30-stimulation cycles. Larger horizons consume the certified RHO
        # chronology directly instead of loading an incompatible warmup.
        command.extend(
            [
                "--ipopt-disable-standard-warmup",
                "--adopt-common-initial-solution-warmup-cycles",
            ]
        )
    command.extend(
        [
        "--optional-nlp-periodic-ipopt-hot-start",
        "--periodic-ipopt-refinement-use-sx",
        "--periodic-ipopt-refinement-iterations",
        str(args.max_iterations),
        "--single-shot",
        "--cycles-per-window",
        str(cycles),
        "--n-windows",
        str(cycles),
        "--mechanical-formulation",
        "full",
        "--full-contact-position-tolerance",
        "2e-5",
        "--common-initial-solution",
        str(seed_path),
        "--common-initial-solution-output",
        str(solution_path),
        "--output-json",
        str(result_path),
        ]
    )
    return command


def _attempt_record(
    cycles: int,
    phase: str,
    monitored: MonitoredRun,
    result_path: Path,
) -> dict:
    unknown_mumps_warning = _log_has_unknown_mumps_warning(
        Path(monitored.log_path)
    )
    certificate = _benchmark_success(
        result_path,
        expected_mode="single_shot",
        expected_cycles=cycles,
        expected_solver="madnlp",
    )
    infrastructure_error = bool(
        not monitored.memory_limit_exceeded
        and not monitored.timed_out
        and (
            monitored.return_code != 0
            or not _benchmark_payload_is_readable(result_path)
            or unknown_mumps_warning
        )
    )
    success = bool(
        certificate
        and monitored.return_code == 0
        and not monitored.memory_limit_exceeded
        and not monitored.timed_out
    )
    failure_kind = (
        None
        if success
        else "memory_limit"
        if monitored.memory_limit_exceeded
        else "timeout"
        if monitored.timed_out
        else "infrastructure_error"
        if infrastructure_error
        else "solver_failure"
    )
    return {
        "cycles": cycles,
        "phase": phase,
        "success": success,
        "failure_kind": failure_kind,
        "certificate_valid": certificate,
        "infrastructure_error": infrastructure_error,
        "unknown_mumps_warning": unknown_mumps_warning,
        "result_path": str(result_path),
        "peak_rss_bytes": monitored.peak_rss_bytes,
        "peak_rss_gib": monitored.peak_rss_bytes / GIB,
        "return_code": monitored.return_code,
        "elapsed_s": monitored.elapsed_s,
        "memory_limit_exceeded": monitored.memory_limit_exceeded,
        "timed_out": monitored.timed_out,
        "log_path": monitored.log_path,
    }


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# RHO reduced vs full-horizon independent size sweep",
        "",
        f"- Limite RSS : `{report['rss_limit_gib']:.3f} GiB`",
        f"- Plafond demandé : `{report['max_cycles']} cycles`",
        f"- Préfixe RHO disponible : `{report['rho_available_cycles']} cycles`",
        f"- Plus grand full horizon validé : `{report['largest_successful_cycles']}`",
        f"- Trous de convergence : `{report.get('solver_gap_cycles', [])}`",
        "- Initialisation de chaque taille : `préfixe RHO reduced indépendant`",
        (
            "- RHO reduced concaténé : "
            f"`{'succès' if report['rho']['success'] else 'échec'}`, "
            f"pic RSS `{report['rho']['peak_rss_gib']:.3f} GiB`, "
            f"temps `{report['rho']['elapsed_s']:.1f} s`"
        ),
        f"- Arrêt : `{report['stop_reason']}`",
        "",
        "| Cycles | Phase | Chance | Succès | Échec | Pic RSS (GiB) | Temps (s) |",
        "|---:|:---|---:|:---:|:---|---:|---:|",
    ]
    for attempt in report["full_horizon_attempts"]:
        lines.append(
            f"| {attempt['cycles']} | {attempt['phase']} | "
            f"{attempt.get('chance', 1)} | "
            f"{'oui' if attempt['success'] else 'non'} | "
            f"{attempt.get('failure_kind') or '—'} | "
            f"{attempt['peak_rss_gib']:.3f} | {attempt['elapsed_s']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_memory_limit(raw: str, total_memory: int) -> float:
    if raw.lower() == "auto":
        return automatic_rss_limit_gib(total_memory)
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("--memory-limit-gib must be 'auto' or a positive number.")
    if value * GIB >= total_memory:
        raise ValueError(
            "--memory-limit-gib must leave headroom below the detected allocation."
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cycles", type=int, required=True)
    parser.add_argument("--memory-limit-gib", default="auto")
    parser.add_argument("--n-threads", type=int, required=True)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--crank-assistance", type=float, default=0.0)
    parser.add_argument("--terminal-wheel-q-slack", type=float, default=0.002)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-interval-s", type=float, default=0.5)
    parser.add_argument(
        "--attempt-timeout-s",
        type=float,
        default=1800.0,
        help="Wall-time cap for each independent solver attempt.",
    )
    return parser


def _run_horizon_attempt(
    args: argparse.Namespace,
    *,
    rho_seed: Path,
    cycles: int,
    phase: str,
    rss_limit_bytes: int,
) -> dict:
    case_dir = args.output_dir / f"full-horizon-{cycles:04d}"
    seed_path = case_dir / "rho-reduced-prefix.npz"
    result_path = case_dir / "result.json"
    solution_path = case_dir / "full-solution.npz"
    for stale_path in (result_path, solution_path):
        stale_path.unlink(missing_ok=True)
    write_rho_seed_prefix(rho_seed, seed_path, cycles)
    monitored = run_monitored(
        _full_horizon_command(args, cycles, seed_path, result_path, solution_path),
        cwd=args.workspace,
        log_path=case_dir / "solver.log",
        rss_limit_bytes=rss_limit_bytes,
        poll_interval_s=args.poll_interval_s,
        timeout_s=args.attempt_timeout_s,
    )
    return _attempt_record(cycles, phase, monitored, result_path)


def run(args: argparse.Namespace) -> int:
    args.workspace = args.workspace.resolve()
    args.seed_dir = args.seed_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.max_cycles < 1:
        raise ValueError("--max-cycles must be strictly positive.")
    if args.n_threads < 1:
        raise ValueError("--n-threads must be strictly positive.")
    if args.poll_interval_s <= 0:
        raise ValueError("--poll-interval-s must be strictly positive.")
    if args.attempt_timeout_s <= 0:
        raise ValueError("--attempt-timeout-s must be strictly positive.")

    total_memory = available_memory_bytes()
    rss_limit_gib = _parse_memory_limit(args.memory_limit_gib, total_memory)
    rss_limit_bytes = int(rss_limit_gib * GIB)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "full-horizon-report.json"
    markdown_path = args.output_dir / "full-horizon-report.md"
    report = {
        "schema": "cocofest-full-horizon-sweep-v1",
        "max_cycles": args.max_cycles,
        "rho_available_cycles": 0,
        "total_memory_bytes": total_memory,
        "total_memory_gib": total_memory / GIB,
        "rss_limit_bytes": rss_limit_bytes,
        "rss_limit_gib": rss_limit_gib,
        "rho_graph": "SX",
        "full_horizon_graph": "MX",
        "rho_solver": "ipopt",
        "full_horizon_solver": "madnlp",
        "linear_solver": "mumps",
        "initialization": "independent_reduced_rho_prefix",
        "rho": None,
        "full_horizon_attempts": [],
        "largest_successful_cycles": 0,
        "stop_reason": "not_started",
    }
    _write_report(report_path, report)

    rho_dir = args.output_dir / "rho-reduced"
    rho_result_path = rho_dir / "result.json"
    rho_seed_path = rho_dir / "concatenated-solution.npz"
    for stale_path in (rho_result_path, rho_seed_path):
        stale_path.unlink(missing_ok=True)
    rho_monitored = run_monitored(
        _rho_command(args, rho_result_path, rho_seed_path),
        cwd=args.workspace,
        log_path=rho_dir / "solver.log",
        rss_limit_bytes=rss_limit_bytes,
        poll_interval_s=args.poll_interval_s,
        timeout_s=args.attempt_timeout_s,
    )
    report["rho"] = {
        **asdict(rho_monitored),
        "peak_rss_gib": rho_monitored.peak_rss_bytes / GIB,
        "result_path": str(rho_result_path),
        "seed_path": str(rho_seed_path),
    }
    rho_result_cycles = _benchmark_validated_cycles(
        rho_result_path,
        expected_mode="rho",
        expected_solver="ipopt",
        expected_requested_cycles=args.max_cycles,
    )
    rho_seed_cycles = _seed_cycle_count(rho_seed_path)
    rho_available_cycles = (
        rho_result_cycles if rho_result_cycles == rho_seed_cycles else 0
    )
    report["rho_available_cycles"] = rho_available_cycles
    report["rho"].update(
        {
            "success": (
                rho_available_cycles > 0
                and rho_monitored.return_code == 0
                and not rho_monitored.memory_limit_exceeded
                and not rho_monitored.timed_out
                and not _log_has_unknown_mumps_warning(rho_monitored.log_path)
            ),
            "certificate_valid": rho_result_cycles > 0,
            "unknown_mumps_warning": _log_has_unknown_mumps_warning(
                rho_monitored.log_path
            ),
            "validated_cycles": rho_result_cycles,
            "seed_cycles": rho_seed_cycles,
            "requested_ceiling_reached": rho_available_cycles == args.max_cycles,
        }
    )
    if not report["rho"]["success"]:
        rho_infrastructure_error = bool(
            not rho_monitored.memory_limit_exceeded
            and not rho_monitored.timed_out
            and (
                rho_monitored.return_code != 0
                or not _benchmark_payload_is_readable(rho_result_path)
            )
        )
        report["stop_reason"] = (
            "rho_memory_limit"
            if rho_monitored.memory_limit_exceeded
            else "rho_timeout"
            if rho_monitored.timed_out
            else "rho_infrastructure_error"
            if rho_infrastructure_error
            else "rho_solver_failure"
        )
        _write_report(report_path, report)
        _write_markdown(markdown_path, report)
        return 3 if rho_infrastructure_error else 2

    def run_with_two_solver_chances(cycles: int, phase: str) -> dict:
        last_attempt = None
        for chance in (1, 2):
            attempt = _run_horizon_attempt(
                args,
                rho_seed=rho_seed_path,
                cycles=cycles,
                phase=phase,
                rss_limit_bytes=rss_limit_bytes,
            )
            attempt["chance"] = chance
            report["full_horizon_attempts"].append(attempt)
            _write_report(report_path, report)
            last_attempt = attempt
            if (
                attempt["success"]
                or attempt["memory_limit_exceeded"]
                or attempt["infrastructure_error"]
            ):
                break
        return last_attempt

    first_memory_failure = None
    last_success = 0
    solver_gap_cycles = []
    for cycles in horizon_sweep_targets(rho_available_cycles):
        attempt = run_with_two_solver_chances(cycles, "coarse")
        if attempt["infrastructure_error"]:
            report["stop_reason"] = "infrastructure_error"
            _write_report(report_path, report)
            _write_markdown(markdown_path, report)
            return 3
        if attempt["success"]:
            last_success = cycles
            report["largest_successful_cycles"] = cycles
            if cycles == rho_available_cycles:
                report["stop_reason"] = (
                    "requested_ceiling_reached"
                    if rho_available_cycles == args.max_cycles
                    else "rho_prefix_ceiling_reached"
                )
            else:
                report["stop_reason"] = "running"
            continue
        if attempt["memory_limit_exceeded"]:
            first_memory_failure = cycles
            report["stop_reason"] = "memory_limit"
            break
        solver_gap_cycles.append(cycles)
        report["stop_reason"] = attempt["failure_kind"]
        if cycles == 1:
            report["stop_reason"] = "one_cycle_bridge_not_certified"
            _write_report(report_path, report)
            _write_markdown(markdown_path, report)
            return 2

    if (
        first_memory_failure is not None
        and first_memory_failure - last_success > 1
    ):
        for cycles in refinement_targets(last_success, first_memory_failure):
            attempt = run_with_two_solver_chances(cycles, "refinement")
            if attempt["infrastructure_error"]:
                report["stop_reason"] = "infrastructure_error"
                _write_report(report_path, report)
                _write_markdown(markdown_path, report)
                return 3
            if attempt["memory_limit_exceeded"]:
                report["stop_reason"] = "memory_limit_bracketed"
                break
            if attempt["success"]:
                last_success = cycles
                report["largest_successful_cycles"] = cycles
            else:
                solver_gap_cycles.append(cycles)

    report["solver_gap_cycles"] = sorted(set(solver_gap_cycles))
    if first_memory_failure is None:
        if report["largest_successful_cycles"] == rho_available_cycles:
            report["stop_reason"] = (
                "requested_ceiling_reached"
                if rho_available_cycles == args.max_cycles
                else "rho_prefix_ceiling_reached"
            )
        else:
            report["stop_reason"] = "sweep_completed_with_solver_gaps"
    _write_markdown(markdown_path, report)
    _write_report(report_path, report)
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
