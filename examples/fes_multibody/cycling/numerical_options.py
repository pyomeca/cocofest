"""Numerical settings shared by the historical cycling MHE and study runs."""

from sys import platform


def build_cycling_mhe_info(
    total_cycles: int,
    *,
    radau_stages: int = 3,
    use_sx: bool = False,
    linear_solver: str | None = None,
) -> dict:
    """Build the existing MHE options without changing its default transcription.

    The paper protocol can request Radau-5/SX/MA57 explicitly. This function
    configures the OCP; it does not assert that a requested solver is installed.
    """
    if isinstance(radau_stages, bool) or not isinstance(radau_stages, int) or radau_stages < 1:
        raise ValueError("radau_stages must be a positive integer")
    if isinstance(total_cycles, bool) or not isinstance(total_cycles, int) or total_cycles < 1:
        raise ValueError("total_cycles must be a positive integer")
    if linear_solver is not None and (not isinstance(linear_solver, str) or not linear_solver.strip()):
        raise ValueError("linear_solver must be a nonempty name or None")

    from bioptim import OdeSolver

    return {
        "cycle_duration": 1,
        "n_cycles_to_advance": 1,
        "n_cycles": total_cycles,
        "ode_solver": OdeSolver.COLLOCATION(polynomial_degree=radau_stages, method="radau"),
        "use_sx": use_sx,
        "linear_solver": linear_solver,
    }


def resolve_ipopt_linear_solver(configured: str | None, system_platform: str = platform) -> str:
    """Preserve the example's old platform default unless explicitly overridden."""
    return configured if configured is not None else ("ma57" if system_platform.startswith("linux") else "mumps")
