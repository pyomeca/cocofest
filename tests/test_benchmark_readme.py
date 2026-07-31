from pathlib import Path


README = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "cycling_solver_benchmark"
    / "README.md"
)


def test_display_equations_use_balanced_github_math_fences():
    lines = README.read_text(encoding="utf-8").splitlines()
    in_math = False
    math_blocks = 0

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        assert stripped != "$$", (
            f"Display-math delimiter $$ at line {line_number} can be split by "
            "Markdown constructs inside a multiline equation."
        )
        if stripped == "```math":
            assert not in_math, f"Nested math fence at line {line_number}."
            in_math = True
            math_blocks += 1
        elif stripped == "```" and in_math:
            in_math = False

    assert math_blocks > 0
    assert not in_math, "Unclosed GitHub math fence in the benchmark README."
