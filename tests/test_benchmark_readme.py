import re
from pathlib import Path


BENCHMARK_DOCS = Path(__file__).resolve().parents[1] / "docs" / "cycling_solver_benchmark"
MARKDOWN_DOCUMENTS = (
    BENCHMARK_DOCS / "README.md",
    BENCHMARK_DOCS / "development_history.md",
    BENCHMARK_DOCS / "resume_and_todo.md",
)


def test_display_equations_use_balanced_github_math_fences():
    for document in MARKDOWN_DOCUMENTS:
        lines = document.read_text(encoding="utf-8").splitlines()
        in_math = False
        math_blocks = 0

        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            assert stripped != "$$", (
                f"Display-math delimiter $$ at {document.name}:{line_number} can be "
                "split by Markdown constructs inside a multiline equation."
            )
            if stripped == "```math":
                assert not in_math, f"Nested math fence at {document.name}:{line_number}."
                in_math = True
                math_blocks += 1
            elif stripped == "```" and in_math:
                in_math = False

        assert math_blocks > 0, f"No display math found in {document.name}."
        assert not in_math, f"Unclosed GitHub math fence in {document.name}."


def test_benchmark_readme_separates_current_method_from_history():
    readme = (BENCHMARK_DOCS / "README.md").read_text(encoding="utf-8")
    history = (BENCHMARK_DOCS / "development_history.md").read_text(encoding="utf-8")

    assert "development_history.md" in readme
    assert "README.md" in history
    assert "resume_and_todo.md" in readme
    assert "resume_and_todo.md" in history
    assert "continuation_prompt.md" in readme
    assert "linux_32core_setup.md" in readme
    assert (BENCHMARK_DOCS / "continuation_prompt.md").is_file()
    assert (BENCHMARK_DOCS / "linux_32core_setup.md").is_file()
    assert "corriger plutôt que reproduire" in readme
    assert "ne doit jamais être utilisée comme oracle" in readme


def test_linux_32core_setup_tracks_workflow_solver_pins():
    setup = (BENCHMARK_DOCS / "linux_32core_setup.md").read_text(encoding="utf-8")
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "cycling_solver_benchmark_linux.yml"
    ).read_text(encoding="utf-8")

    pinned_variables = (
        "CASADI_VERSION",
        "CASADI_MADNLP_COMMIT",
        "BIOPTIM_PRODUCTION_COMMIT",
        "ACADOS_COMMIT",
        "LIBMAD_COMMIT",
        "JULIAC_COMMIT",
        "JULIA_VERSION",
    )
    for variable in pinned_variables:
        match = re.search(rf"^  {variable}:\s*[\"']?([^\"'\s]+)", workflow, re.MULTILINE)
        assert match, f"Missing {variable} in the benchmark workflow."
        assert match.group(1) in setup, f"The Linux setup does not document {variable}."

    assert "cocofest-rho32" in setup
    assert "cocofest-madnlp32" in setup
    assert "OMP_NUM_THREADS=1" in setup
    assert "CMAKE_BUILD_PARALLEL_LEVEL=32" in setup
