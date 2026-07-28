import importlib.util
import math
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "summarize_cycling_benchmark.py"
)
_SPEC = importlib.util.spec_from_file_location("cycling_benchmark_summary", _SCRIPT)
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)


def test_phase_alignment_reduces_a_pure_sampling_phase_error():
    sample_count = 64
    phase_shift = 0.1
    reference_phase = [
        2.0 * math.pi * index / sample_count for index in range(sample_count)
    ]
    compared_phase = [
        (phase + phase_shift) % (2.0 * math.pi) for phase in reference_phase
    ]
    reference_values = [200e-6 + 50e-6 * math.cos(phase) for phase in reference_phase]
    compared_values = [200e-6 + 50e-6 * math.cos(phase) for phase in compared_phase]
    reference = {"pulse_width_s": reference_values}
    compared = {"pulse_width_s": compared_values}

    raw = summary._pattern_comparison(reference, compared)
    aligned = summary._phase_aligned_pattern_comparison(
        reference,
        compared,
        reference_phase,
        compared_phase,
    )

    assert aligned is not None
    assert raw is not None
    assert aligned["root_mean_square_error_us"] < 0.1
    assert aligned["root_mean_square_error_us"] < raw["root_mean_square_error_us"]
    assert aligned["correlation"] > raw["correlation"]


def test_periodic_interpolation_wraps_across_zero():
    phases = [0.1, 1.0, 3.0, 5.0]
    values = [1.0, 2.0, 3.0, 4.0]

    interpolated = summary._periodic_linear_interpolation(
        phases,
        values,
        [0.0, 2.0 * math.pi],
    )

    assert interpolated is not None
    assert math.isclose(interpolated[0], interpolated[1])
    assert 1.0 < interpolated[0] < 4.0


def test_small_tolerances_are_rendered_in_scientific_notation():
    assert summary._fmt_scientific(1e-8) == "1.0e-08"
    assert summary._fmt_scientific(1e-5) == "1.0e-05"


def test_stimulation_comparison_exports_raw_and_phase_aligned_metrics():
    def entry(solver, phases, values):
        return {
            "result": {
                "solver": solver,
                "stimulation_patterns": {
                    "cycle_10": {
                        "available": True,
                        "cycle": 10,
                        "crank_phase_rad": phases,
                        "muscles": {
                            "Biceps": {"pulse_width_s": values},
                        },
                    }
                },
            }
        }

    phases = [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi]
    entries = [
        entry("ipopt", phases, [100e-6, 200e-6, 100e-6, 100e-6]),
        entry("madnlp", phases, [101e-6, 201e-6, 101e-6, 101e-6]),
    ]

    comparison = summary.stimulation_comparisons(entries)[0]

    assert math.isclose(comparison["root_mean_square_error_us"], 1.0)
    assert math.isclose(
        comparison["phase_aligned_root_mean_square_error_us"],
        1.0,
    )
    assert comparison["crank_phase_root_mean_square_error_rad"] == 0.0


def test_markdown_distinguishes_successful_windows_from_strict_prefix():
    entry = {
        "runtime": {"provenance": {"BIOPTIM_BENCHMARK_COMMIT": "abc"}},
        "configuration": {
            "n_windows": 100,
            "nlp_tolerance": 1e-8,
            "primal_feasibility_threshold": 1e-5,
        },
        "result": {
            "solver": "madnlp",
            "success": False,
            "successful_windows": 99,
            "attempted_windows": 100,
            "validated_cycles": 85,
            "first_failed_rho": 86,
            "end_to_end_wall_time_s": 881.48,
            "initial_guess_preparation_time_s": 45.72,
            "hot_wall_time_median_s": 5.62,
            "hot_wall_time_p90_s": 7.75,
            "stop": {"label": "solver_failure_after_valid_cycles"},
            "windows": [],
            "stimulation_patterns": {},
        },
    }

    markdown = summary.render_markdown([entry], [])

    assert "RHO résolus" in markdown
    assert "Préfixe strict" in markdown
    assert (
        "| MADNLP/FULL | 1.0e-08 | 1.0e-05 | non | 99/100 | 85/100 | 86 |"
        in markdown
    )
    assert "même si les fenêtres suivantes récupèrent" in markdown


def test_markdown_states_that_internal_solver_tolerances_are_backend_specific():
    entry = {
        "runtime": {},
        "configuration": {
            "mechanical_formulation": "full",
            "nlp_tolerance": 1e-8,
            "primal_feasibility_threshold": 1e-5,
        },
        "result": {
            "solver": "madnlp",
            "windows": [],
            "stimulation_patterns": {},
        },
    }

    markdown = summary.render_markdown([entry], [])

    assert "Comparabilité du problème et des critères physiques" in markdown
    assert "tolérances internes sont propres à chaque backend" in markdown
    assert "même seuil de faisabilité physique" in markdown


def test_configuration_comparability_is_scoped_by_mechanical_formulation():
    def entry(solver, mechanics, n_windows):
        return {
            "configuration": {
                "mechanical_formulation": mechanics,
                "n_windows": n_windows,
            },
            "result": {"solver": solver},
        }

    entries = [
        entry("ipopt", "full", 31),
        entry("madnlp", "full", 31),
        entry("ipopt", "reduced", 31),
        entry("madnlp", "reduced", 31),
    ]

    assert summary.configuration_mismatches(entries) == []
    entries[-1]["configuration"]["n_windows"] = 30
    mismatches = summary.configuration_mismatches(entries)

    assert len(mismatches) == 1
    assert mismatches[0]["case"] == "madnlp/reduced"
    assert mismatches[0]["reference_case"] == "ipopt/reduced"


def test_requested_rho_count_accounts_for_multi_cycle_window():
    entry = {
        "configuration": {"n_windows": 31, "cycles_per_window": 2},
        "result": {"solver": "ipopt"},
    }

    assert summary._requested_rho_count(entry) == 30


def test_mechanical_pattern_comparison_uses_full_as_reference():
    def entry(mechanics, values):
        return {
            "configuration": {"mechanical_formulation": mechanics},
            "result": {
                "solver": "ipopt",
                "stimulation_patterns": {
                    "cycle_10": {
                        "available": True,
                        "cycle": 10,
                        "crank_phase_rad": [0.0, 1.0],
                        "muscles": {"Biceps": {"pulse_width_s": values}},
                    }
                },
            },
        }

    comparisons = summary.mechanical_stimulation_comparisons(
        [entry("full", [100e-6, 200e-6]), entry("reduced", [101e-6, 201e-6])]
    )

    assert len(comparisons) == 1
    assert comparisons[0]["reference_case"] == "ipopt/full"
    assert comparisons[0]["case"] == "ipopt/reduced"
    assert math.isclose(comparisons[0]["root_mean_square_error_us"], 1.0)
