import importlib.util
import json
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "summarize_solver_option_screen.py"
)
_SPEC = importlib.util.spec_from_file_location("solver_option_screen", _SCRIPT)
screen = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(screen)


def _write_result(
    path: Path,
    *,
    solver: str,
    tolerance: float,
    infeasibilities: list[float],
    converged: list[bool],
    wall_times: list[float],
):
    windows = [
        {
            "rho": index + 1,
            "solver_converged": solver_converged,
            "objective": 10.0 + index,
            "iterations": 20 + index,
            "wall_time_s": wall_time,
            "solver_time_s": wall_time - 0.1,
            "native_status": "SOLVE_SUCCEEDED",
            "feasibility": {
                "effective_primal_infeasibility": infeasibility,
            },
        }
        for index, (infeasibility, solver_converged, wall_time) in enumerate(
            zip(infeasibilities, converged, wall_times)
        )
    ]
    payload = {
        "configurations": {
            solver: {
                "nlp_tolerance": tolerance,
                "madnlp_linear_solver": ("umfpack" if "umfpack" in path.stem else None),
                "alpaqa_panoc_max_wall_time": (20.0 if solver == "alpaqa" else None),
            }
        },
        "results": [
            {
                "solver": solver,
                "success": all(converged),
                "validated_cycles": sum(converged),
                "attempted_windows": len(windows),
                "first_failed_rho": None,
                "end_to_end_wall_time_s": sum(wall_times) + 2.0,
                "initial_guess_preparation_time_s": 2.0,
                "stop": {"label": "completed_requested_horizon"},
                "windows": windows,
                "nlp_solver_stats": [
                    {
                        "window": index,
                        "n_call_nlp_f": 3,
                        "n_call_nlp_g": 4,
                    }
                    for index in range(len(windows))
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_screen_uses_a_common_feasibility_threshold_and_ranks(tmp_path):
    alpaqa = tmp_path / "alpaqa-panoc20-default.json"
    mad_default = tmp_path / "madnlp-tol1e-6-default.json"
    mad_umfpack = tmp_path / "madnlp-tol1e-7-umfpack.json"
    _write_result(
        alpaqa,
        solver="alpaqa",
        tolerance=1e-6,
        infeasibilities=[2e-5],
        converged=[True],
        wall_times=[20.0],
    )
    _write_result(
        mad_default,
        solver="madnlp",
        tolerance=1e-6,
        infeasibilities=[8e-6, 9e-6],
        converged=[True, True],
        wall_times=[5.0, 7.0],
    )
    _write_result(
        mad_umfpack,
        solver="madnlp",
        tolerance=1e-7,
        infeasibilities=[5e-6, 2e-5],
        converged=[True, True],
        wall_times=[4.0, 4.0],
    )

    rows = screen.ranked_rows(
        screen.load_rows(
            [alpaqa, mad_default, mad_umfpack],
            common_feasibility_threshold=1e-5,
        )
    )

    by_variant = {row["variant"]: row for row in rows}
    assert by_variant[alpaqa.stem]["common_validated_prefix"] == 0
    assert by_variant[mad_default.stem]["common_validated_prefix"] == 2
    assert by_variant[mad_default.stem]["rank_within_solver"] == 1
    assert by_variant[mad_default.stem]["rho_wall_median_s"] == 6.0
    assert by_variant[mad_default.stem]["total_oracle_evaluations"] == 14
    assert by_variant[mad_umfpack.stem]["common_validated_prefix"] == 1
    markdown = screen.render_markdown(rows, 1e-5, ("missing-case",))
    assert "même seuil physique indépendant `1.0e-05`" in markdown
    assert "Matrice incomplète" in markdown
