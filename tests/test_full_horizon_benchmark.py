from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "run_full_horizon_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("run_full_horizon_benchmark", SCRIPT_PATH)
full_horizon = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = full_horizon
SPEC.loader.exec_module(full_horizon)


@pytest.mark.parametrize(
    ("maximum", "expected_tail"),
    (
        (3, [1, 2, 3]),
        (32, [25, 30, 32]),
        (60, [50, 55, 60]),
        (67, [55, 60, 67]),
        (100, [80, 90, 100]),
    ),
)
def test_horizon_sweep_targets_follow_adaptive_steps_and_arbitrary_maximum(
    maximum, expected_tail
):
    targets = full_horizon.horizon_sweep_targets(maximum)

    assert targets == sorted(set(targets))
    assert targets[-1] == maximum
    assert targets[-len(expected_tail) :] == expected_tail


def test_refinement_fills_only_the_last_coarse_interval():
    assert full_horizon.refinement_targets(60, 70) == list(range(61, 70))
    assert full_horizon.refinement_targets(12, 13) == []


def test_automatic_rss_limits_match_the_two_ci_machine_classes():
    assert full_horizon.automatic_rss_limit_gib(16 * full_horizon.GIB) == 12.5
    assert full_horizon.automatic_rss_limit_gib(128 * full_horizon.GIB) == 97.5


def test_rho_seed_prefix_preserves_complete_cycle_layout_and_metadata(
    tmp_path,
):
    source = tmp_path / "rho.npz"
    output = tmp_path / "prefix.npz"
    metadata = {
        "schema": "cocofest-common-periodic-initial-solution-v2",
        "cycles_per_window": 4,
        "mechanical_formulation": "reduced",
    }
    np.savez(
        source,
        states__theta=np.arange(17, dtype=float).reshape(1, 17),
        states__A_Biceps=np.arange(34, dtype=float).reshape(2, 17),
        controls__last_pulse_width_Biceps=np.arange(16, dtype=float).reshape(1, 16),
        metadata__json=np.asarray(json.dumps(metadata)),
    )

    written_metadata = full_horizon.write_rho_seed_prefix(source, output, 3)

    with np.load(output, allow_pickle=False) as data:
        assert data["states__theta"].shape == (1, 13)
        assert data["states__A_Biceps"].shape == (2, 13)
        assert data["controls__last_pulse_width_Biceps"].shape == (1, 12)
        persisted_metadata = json.loads(str(data["metadata__json"].item()))

    assert written_metadata == persisted_metadata
    assert persisted_metadata["cycles_per_window"] == 3
    assert persisted_metadata["producer_source_cycles"] == 4
    assert persisted_metadata["producer_mode"] == "receding_horizon_prefix"


def test_rho_seed_prefix_rejects_non_integral_cycle_layout(tmp_path):
    source = tmp_path / "rho.npz"
    np.savez(
        source,
        states__theta=np.zeros((1, 10)),
        controls__last_pulse_width_Biceps=np.zeros((1, 8)),
        metadata__json=np.asarray(json.dumps({"cycles_per_window": 4})),
    )

    with pytest.raises(ValueError, match="State seed"):
        full_horizon.write_rho_seed_prefix(source, tmp_path / "prefix.npz", 2)


def test_benchmark_success_requires_the_complete_physical_horizon(tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "success": True,
                        "solver_success": True,
                        "physical_success": True,
                        "solver": "madnlp",
                        "mode": "single_shot",
                        "covered_cycles": 30,
                        "physically_validated_cycles": 30,
                    }
                ],
                "configurations": {
                    "madnlp": {
                        "single_shot": True,
                        "mechanical_formulation": "full",
                        "cycles_per_window": 30,
                        "n_windows": 30,
                        "use_sx": False,
                        "madnlp_linear_solver": "mumps",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert full_horizon._benchmark_success(
        result_path,
        expected_mode="single_shot",
        expected_cycles=30,
        expected_solver="madnlp",
    )
    assert not full_horizon._benchmark_success(
        result_path,
        expected_mode="single_shot",
        expected_cycles=100,
        expected_solver="madnlp",
    )
    assert not full_horizon._benchmark_success(
        result_path,
        expected_mode="rho",
        expected_cycles=30,
        expected_solver="madnlp",
    )


def test_validated_cycles_retains_a_shorter_rho_prefix(tmp_path):
    result_path = tmp_path / "rho-result.json"
    result_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "solver": "ipopt",
                        "mode": "rho",
                        "success": False,
                        "covered_cycles": 43,
                        "physically_validated_cycles": 42,
                    }
                ],
                "configurations": {
                    "ipopt": {
                        "single_shot": False,
                        "mechanical_formulation": "reduced",
                        "cycles_per_window": 1,
                        "n_windows": 100,
                        "use_sx": True,
                        "ipopt_linear_solver": "mumps",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert (
        full_horizon._benchmark_validated_cycles(
            result_path,
            expected_mode="rho",
            expected_solver="ipopt",
            expected_requested_cycles=100,
        )
        == 42
    )


def test_unknown_mumps_warning_accepts_monitored_string_path(tmp_path):
    log_path = tmp_path / "solver.log"
    log_path.write_text(
        "libMAD WARNING: option linear_solver is of unknown type mumps, ignoring\n",
        encoding="utf-8",
    )

    assert full_horizon._log_has_unknown_mumps_warning(str(log_path))


def test_solver_chances_keep_independent_logs_and_results(tmp_path, monkeypatch):
    observed = {}
    args = SimpleNamespace(
        output_dir=tmp_path / "output",
        workspace=tmp_path,
        poll_interval_s=0.5,
        attempt_timeout_s=30.0,
    )
    monkeypatch.setattr(
        full_horizon,
        "write_rho_seed_prefix",
        lambda source, destination, cycles: destination.parent.mkdir(
            parents=True, exist_ok=True
        ),
    )
    monkeypatch.setattr(
        full_horizon,
        "_full_horizon_command",
        lambda *command_args, **command_kwargs: ["solver"],
    )

    def fake_run_monitored(command, **kwargs):
        observed["log_path"] = kwargs["log_path"]
        return full_horizon.MonitoredRun(
            command=command,
            return_code=1,
            peak_rss_bytes=0,
            elapsed_s=1.0,
            memory_limit_exceeded=False,
            timed_out=False,
            log_path=str(kwargs["log_path"]),
        )

    monkeypatch.setattr(full_horizon, "run_monitored", fake_run_monitored)

    attempt = full_horizon._run_horizon_attempt(
        args,
        rho_seed=tmp_path / "rho.npz",
        cycles=2,
        phase="coarse",
        chance=2,
        rss_limit_bytes=1024,
    )

    expected_dir = tmp_path / "output" / "full-horizon-0002" / "chance-2"
    assert observed["log_path"] == expected_dir / "solver.log"
    assert attempt["result_path"] == str(expected_dir / "result.json")


def test_workflow_has_an_isolated_mx_mumps_full_horizon_mode():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "cycling_solver_benchmark_linux.yml"
    ).read_text(encoding="utf-8")
    full_job = workflow.split("\n  full-horizon:", maxsplit=1)[1].split(
        "\n  screen-report:", maxsplit=1
    )[0]

    assert "inputs.cycles == 'full_horizon'" in full_job
    assert "--single-shot" in SCRIPT_PATH.read_text(encoding="utf-8")
    assert '"--madnlp-linear-solver"' in SCRIPT_PATH.read_text(encoding="utf-8")
    assert '"mumps"' in SCRIPT_PATH.read_text(encoding="utf-8")
    assert '"--ipopt-no-use-sx"' in SCRIPT_PATH.read_text(encoding="utf-8")
    assert '"--optional-nlp-periodic-ipopt-hot-start"' in SCRIPT_PATH.read_text(
        encoding="utf-8"
    )
    assert "--memory-limit-gib" in full_job
    assert "full_horizon_max_cycles" in workflow
    assert "cycling-full-horizon-${{ github.run_id }}" in full_job


def test_rho_and_full_horizon_use_the_intended_solver_contract(tmp_path):
    args = SimpleNamespace(
        python="python",
        workspace=tmp_path,
        seed_dir=tmp_path / "seed",
        n_threads=4,
        crank_assistance=0.0,
        max_iterations=2000,
        terminal_wheel_q_slack=0.002,
        max_cycles=100,
    )

    rho = full_horizon._rho_command(args, tmp_path / "rho.json", tmp_path / "rho.npz")
    full = full_horizon._full_horizon_command(
        args,
        60,
        tmp_path / "prefix.npz",
        tmp_path / "full.json",
        tmp_path / "full.npz",
    )
    one_cycle_full = full_horizon._full_horizon_command(
        args,
        1,
        tmp_path / "one-cycle-prefix.npz",
        tmp_path / "one-cycle-full.json",
        tmp_path / "one-cycle-full.npz",
    )
    paired_reduced = full_horizon._full_horizon_command(
        args,
        2,
        tmp_path / "two-cycle-prefix.npz",
        tmp_path / "two-cycle-reduced.json",
        tmp_path / "two-cycle-reduced.npz",
        mechanical_formulation="reduced",
    )

    assert rho[rho.index("--solvers") + 1] == "ipopt"
    assert full[full.index("--solvers") + 1] == "madnlp"
    assert "--single-shot" not in rho
    assert "--allow-partial-receding-horizon-solution-output" in rho
    assert "--ipopt-use-sx" in rho
    assert "--ipopt-no-use-sx" not in rho
    assert rho[rho.index("--ipopt-max-iter") + 1] == "2000"
    assert "--single-shot" in full
    assert (
        paired_reduced[paired_reduced.index("--mechanical-formulation") + 1]
        == "reduced"
    )
    assert "--ipopt-no-use-sx" in paired_reduced
    assert full[full.index("--madnlp-linear-solver") + 1] == "mumps"
    assert "--ipopt-no-use-sx" in full
    assert "--ipopt-disable-standard-warmup" in full
    assert "--adopt-common-initial-solution-warmup-cycles" in full
    assert "--ipopt-disable-standard-warmup" not in one_cycle_full
    assert (
        "--adopt-common-initial-solution-warmup-cycles" not in one_cycle_full
    )
    assert "--optional-nlp-periodic-ipopt-hot-start" in full
    assert "--initial-guess-diagnostics" in full
    assert "--acados-diagnostics" not in full
    assert "--periodic-ipopt-refinement-use-sx" in full
    assert full[full.index("--periodic-ipopt-refinement-iterations") + 1] == "2000"
