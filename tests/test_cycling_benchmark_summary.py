import csv
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
            "use_sx": True,
        },
        "result": {
            "solver": "madnlp",
            "success": False,
            "successful_windows": 99,
            "attempted_windows": 100,
            "validated_windows": 99,
            "validated_cycles": 85,
            "physically_validated_cycles": 85,
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
        "| MADNLP/FULL | SX | 1.0e-08 | 1.0e-05 | non | 99/100 | 85/100 | 86 |"
        in markdown
    )
    assert "même si les fenêtres suivantes récupèrent" in markdown


def test_markdown_strict_prefix_prefers_the_physical_certificate():
    entry = {
        "runtime": {"provenance": {"BIOPTIM_BENCHMARK_COMMIT": "abc"}},
        "configuration": {
            "mechanical_formulation": "full",
            "n_windows": 5,
            "use_sx": True,
        },
        "result": {
            "solver": "acados",
            "successful_windows": 1,
            "attempted_windows": 3,
            "validated_windows": 1,
            "validated_cycles": 0,
            "physically_validated_cycles": 0,
            "windows": [],
            "stimulation_patterns": {},
        },
    }

    markdown = summary.render_markdown([entry], [])

    assert "| ACADOS/FULL | SX | — | — | non | 1/3 | 0/5 |" in markdown


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


def test_markdown_reports_total_and_per_muscle_fatigue_metrics():
    entry = {
        "runtime": {},
        "configuration": {"mechanical_formulation": "reduced"},
        "result": {
            "solver": "acados",
            "executed_fatigue_objective": 12.5,
            "fatigue_auc_cycles": 0.75,
            "min_A_capacity_ratio": 0.91,
            "muscle_fatigue": [
                {
                    "muscle": "Biceps",
                    "executed_fatigue_objective": 8.0,
                    "cumulative_normalized_fatigue_cycles": 0.5,
                    "final_capacity_ratio": 0.91,
                }
            ],
            "windows": [],
            "stimulation_patterns": {},
        },
    }

    markdown = summary.render_markdown([entry], [])

    assert "| ACADOS/REDUCED | 12.500 | 0.750 | 0.910000 |" in markdown
    assert "| ACADOS/REDUCED | Biceps | 8.000 | 0.500 | 0.910000 |" in markdown


def test_markdown_attributes_acados_restoration_time_to_effective_rho_time():
    entry = {
        "runtime": {},
        "configuration": {"mechanical_formulation": "reduced"},
        "result": {
            "solver": "acados",
            "hot_wall_time_median_s": 0.06,
            "hot_effective_wall_time_median_s": 0.14,
            "hot_effective_wall_time_p90_s": 0.18,
            "feasibility_restoration": {
                "available": True,
                "total_wall_time_s": 0.8,
                "stages": [{}, {}],
            },
            "windows": [
                {
                    "rho": 2,
                    "wall_time_s": 0.06,
                    "feasibility_restoration_wall_time_s": 0.08,
                    "effective_wall_time_s": 0.14,
                }
            ],
            "stimulation_patterns": {},
        },
    }

    markdown = summary.render_markdown([entry], [])

    assert "Effectif/RHO médian (s)" in markdown
    assert "| ACADOS/REDUCED | 0.800 | 2 |" in markdown
    assert "| ACADOS/REDUCED | 2 |" in markdown
    assert "| 0.060 | 0.080 | 0.140 |" in markdown


def test_fatrop_internal_timings_are_normalized_and_exported(tmp_path):
    entry = {
        "runtime": {},
        "configuration": {
            "mechanical_formulation": "reduced",
            "ode_solver": "collocation",
        },
        "result": {
            "solver": "fatrop",
            "nlp_solver_stats": [
                {
                    "window": 0,
                    "iter_count": 5,
                    "fatrop": {
                        "iterations_count": 5,
                        "time_total": 2.0,
                        "eval_hess_time": 0.8,
                        "eval_jac_time": 0.4,
                        "eval_cv_time": 0.1,
                        "compute_sd_time": 0.2,
                        "eval_hess_count": 4,
                        "eval_jac_count": 5,
                    },
                },
                {
                    "window": 1,
                    "iter_count": 10,
                    "fatrop": {
                        "iterations_count": 10,
                        "time_total": 3.0,
                        "eval_hess_time": 1.2,
                        "eval_jac_time": 0.6,
                        "eval_cv_time": 0.15,
                        "compute_sd_time": 0.3,
                        "eval_hess_count": 8,
                        "eval_jac_count": 10,
                    },
                },
            ],
            "windows": [],
            "stimulation_patterns": {},
        },
    }

    rows = summary.fatrop_internal_timing_rows(entry)
    aggregate = summary.fatrop_internal_timing_summary(entry)

    assert len(rows) == 2
    assert rows[0]["case"] == "fatrop-collocation/reduced"
    assert rows[0]["rho"] == 1
    assert math.isclose(rows[0]["total_wall_time_per_iteration_s"], 0.4)
    assert math.isclose(rows[0]["hessian_wall_time_per_evaluation_s"], 0.2)
    assert math.isclose(rows[0]["derivative_wall_time_fraction"], 0.6)
    assert aggregate is not None
    assert aggregate["rho_count"] == 2
    assert math.isclose(aggregate["mean_iterations"], 7.5)
    assert math.isclose(aggregate["total_wall_time_s"], 5.0)
    assert math.isclose(aggregate["structure_detection_wall_time_s"], 0.5)
    assert math.isclose(aggregate["derivative_wall_time_fraction"], 0.6)

    csv_path = tmp_path / "fatrop-internal-timings.csv"
    summary.write_fatrop_internal_timing_csv(csv_path, [entry])
    with csv_path.open(newline="", encoding="utf-8") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 2
    assert exported[1]["rho"] == "2"
    assert exported[1]["iterations"] == "10"

    markdown = summary.render_markdown([entry], [])
    assert "## Décomposition interne Fatrop" in markdown
    assert "| FATROP-COLLOCATION/REDUCED | 2 | 7.50 |" in markdown


def test_fatrop_internal_timings_ignore_other_solvers():
    entry = {
        "configuration": {
            "mechanical_formulation": "full",
            "ode_solver": "collocation",
        },
        "result": {
            "solver": "ipopt",
            "nlp_solver_stats": [{"window": 0, "iter_count": 3}],
        },
    }

    assert summary.fatrop_internal_timing_rows(entry) == []
    assert summary.fatrop_internal_timing_summary(entry) is None


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


def test_numerical_transcription_choices_are_reported_as_non_comparable():
    def entry(solver, ode_solver, state_scaling):
        return {
            "configuration": {
                "mechanical_formulation": "full",
                "n_windows": 100,
                "ode_solver": ode_solver,
                "state_scaling": state_scaling,
            },
            "result": {"solver": solver},
        }

    entries = [
        entry("ipopt", "collocation", "full"),
        entry("fatrop", "rk4", "none"),
    ]

    mismatches = summary.configuration_mismatches(entries)

    assert [item["field"] for item in mismatches] == ["ode_solver"]


def test_calcium_and_passive_force_contracts_are_required_for_comparability():
    def entry(solver, calcium_formulation, passive_force):
        return {
            "configuration": {
                "mechanical_formulation": "reduced",
                "calcium_forcing_formulation": calcium_formulation,
                "activate_passive_force_relationship": passive_force,
            },
            "result": {"solver": solver},
        }

    mismatches = summary.configuration_mismatches(
        [
            entry("ipopt", "exact_exponential_periodic_node", True),
            entry("madnlp", "continuous_periodic_surrogate", False),
        ]
    )

    assert [item["field"] for item in mismatches] == [
        "calcium_forcing_formulation",
        "activate_passive_force_relationship",
    ]


def test_graph_type_is_a_required_comparability_field():
    def entry(solver, use_sx):
        return {
            "configuration": {
                "mechanical_formulation": "full",
                "use_sx": use_sx,
            },
            "result": {"solver": solver},
        }

    mismatches = summary.configuration_mismatches(
        [entry("ipopt", True), entry("madnlp", False)]
    )

    assert [item["field"] for item in mismatches] == ["use_sx"]


def test_requested_rho_count_accounts_for_multi_cycle_window():
    entry = {
        "configuration": {"n_windows": 31, "cycles_per_window": 2},
        "result": {"solver": "ipopt"},
    }

    assert summary._requested_rho_count(entry) == 30


def test_madnlp_cases_include_the_linear_solver_backend():
    def entry(linear_solver):
        return {
            "configuration": {
                "mechanical_formulation": "full",
                "madnlp_linear_solver": linear_solver,
            },
            "result": {"solver": "madnlp"},
        }

    assert summary._entry_case(entry("pardiso_mkl")) == "madnlp-pardiso/full"
    assert summary._entry_case(entry("mumps")) == "madnlp-mumps/full"


def test_ipopt_and_madnlp_cases_include_compilation_mode():
    ipopt = {
        "configuration": {
            "mechanical_formulation": "full",
            "ipopt_c_compile": True,
        },
        "result": {"solver": "ipopt"},
    }
    madnlp = {
        "configuration": {
            "mechanical_formulation": "reduced",
            "madnlp_linear_solver": "mumps",
            "madnlp_c_compile": True,
        },
        "result": {"solver": "madnlp"},
    }

    assert summary._entry_case(ipopt) == "ipopt-compiled/full"
    assert summary._entry_case(madnlp) == "madnlp-mumps-compiled/reduced"
    assert summary._entry_base_case(ipopt) == "ipopt/full"
    assert summary._entry_base_case(madnlp) == "madnlp-mumps/reduced"


def test_refined_collocation_cases_include_radau_degree():
    ipopt = {
        "configuration": {
            "mechanical_formulation": "reduced",
            "collocation_degree": 5,
            "ipopt_c_compile": False,
        },
        "result": {"solver": "ipopt"},
    }
    madnlp = {
        "configuration": {
            "mechanical_formulation": "reduced",
            "collocation_degree": 5,
            "madnlp_linear_solver": "mumps",
            "madnlp_c_compile": False,
        },
        "result": {"solver": "madnlp"},
    }

    assert summary._entry_case(ipopt) == "ipopt-radau5/reduced"
    assert summary._entry_case(madnlp) == "madnlp-mumps-radau5/reduced"


def test_fatrop_cases_include_transcription_and_compilation():
    entry = {
        "configuration": {
            "mechanical_formulation": "reduced",
            "ode_solver": "rk4",
            "fatrop_c_compile": True,
        },
        "result": {"solver": "fatrop"},
    }

    assert summary._entry_case(entry) == "fatrop-rk4-compiled/reduced"


def test_mechanical_comparison_keeps_madnlp_backends_separate():
    def entry(mechanics, linear_solver, value):
        return {
            "configuration": {
                "mechanical_formulation": mechanics,
                "madnlp_linear_solver": linear_solver,
            },
            "result": {
                "solver": "madnlp",
                "stimulation_patterns": {
                    "cycle_10": {
                        "available": True,
                        "cycle": 10,
                        "crank_phase_rad": [0.0, 1.0],
                        "muscles": {"Biceps": {"pulse_width_s": [value, value]}},
                    }
                },
            },
        }

    comparisons = summary.mechanical_stimulation_comparisons(
        [
            entry("full", "pardiso_mkl", 100e-6),
            entry("reduced", "pardiso_mkl", 101e-6),
            entry("full", "mumps", 200e-6),
            entry("reduced", "mumps", 202e-6),
        ]
    )

    assert {row["case"] for row in comparisons} == {
        "madnlp-pardiso/reduced",
        "madnlp-mumps/reduced",
    }


def test_mechanical_pattern_comparison_uses_full_as_reference():
    def entry(mechanics, values, *, compiled=False):
        return {
            "configuration": {
                "mechanical_formulation": mechanics,
                "ipopt_c_compile": compiled,
            },
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
        [
            entry("full", [100e-6, 200e-6]),
            entry("reduced", [101e-6, 201e-6], compiled=True),
        ]
    )

    assert len(comparisons) == 1
    assert comparisons[0]["reference_case"] == "ipopt/full"
    assert comparisons[0]["case"] == "ipopt-compiled/reduced"
    assert math.isclose(comparisons[0]["root_mean_square_error_us"], 1.0)
