"""Optional nonlinear solver backends used by Cocofest benchmarks.

MadNLP and Alpaqa are exposed by their Bioptim integration branches through
CasADi's ``nlpsol`` API.  They are optional on purpose: importing Cocofest must
continue to work with a standard Bioptim/CasADi installation.
"""

from __future__ import annotations

from typing import Any

NLP_SOLVER_NAMES = ("ipopt", "madnlp", "alpaqa")


class SolverBackendUnavailable(RuntimeError):
    """Raised when an optional Bioptim solver or CasADi plugin is unavailable."""


def nlp_solver_availability(
    solver_name: str,
    *,
    solver_namespace=None,
    casadi_module=None,
) -> tuple[bool, str | None]:
    """Check both the Bioptim factory and the underlying CasADi plugin."""

    solver_name = solver_name.lower()
    if solver_name not in NLP_SOLVER_NAMES:
        raise ValueError(f"Unsupported NLP solver '{solver_name}'.")

    if solver_namespace is None:
        from bioptim import Solver as solver_namespace

    factory_name = solver_name.upper()
    if not hasattr(solver_namespace, factory_name):
        return (
            False,
            f"Bioptim does not expose Solver.{factory_name}; install its "
            f"{solver_name} integration branch.",
        )

    if casadi_module is None:
        import casadi as casadi_module

    has_nlpsol = getattr(casadi_module, "has_nlpsol", None)
    if has_nlpsol is None:
        return False, "The installed CasADi does not expose has_nlpsol()."
    try:
        available = bool(has_nlpsol(solver_name))
    except RuntimeError as error:
        return False, str(error)
    if not available:
        return (
            False,
            f"CasADi nlpsol plugin '{solver_name}' is unavailable in this build.",
        )
    return True, None


def configure_nlp_solver(
    solver_name: str,
    *,
    max_iterations: int,
    tolerance: float = 1e-6,
    print_level: int = 0,
    ipopt_linear_solver: str = "ma57",
    ipopt_hsl_library: str | None = None,
    ipopt_c_compile: bool = False,
    ipopt_options: dict[str, Any] | None = None,
    madnlp_c_compile: bool = False,
    madnlp_linear_solver: str | None = None,
    madnlp_max_wall_time: float | None = None,
    madnlp_nlp_scaling: bool | None = None,
    madnlp_acceptable_tolerance: float | None = None,
    madnlp_acceptable_iterations: int | None = None,
    alpaqa_alm_max_iterations: int | None = None,
    alpaqa_lbfgs_memory: int = 20,
    alpaqa_max_wall_time: float | None = None,
    alpaqa_initial_penalty: float | None = None,
    alpaqa_initial_tolerance: float | None = None,
    alpaqa_penalty_update_factor: float | None = None,
    alpaqa_maximum_penalty: float | None = None,
    alpaqa_panoc_max_wall_time: float | None = None,
    alpaqa_max_no_progress: int | None = None,
    madnlp_mu_init: float = 1e-2,
    solver_namespace=None,
    check_availability: bool = True,
) -> Any:
    """Create a consistently configured Bioptim NLP solver.

    The common tolerance and iteration budget make solver comparisons
    interpretable.  Solver-specific settings only cover features that have
    been validated by the corresponding Bioptim integration branch.
    """

    solver_name = solver_name.lower()
    if max_iterations < 1:
        raise ValueError("max_iterations must be strictly positive.")
    if tolerance <= 0:
        raise ValueError("tolerance must be strictly positive.")
    if alpaqa_lbfgs_memory < 1:
        raise ValueError("alpaqa_lbfgs_memory must be strictly positive.")
    if madnlp_mu_init <= 0:
        raise ValueError("madnlp_mu_init must be strictly positive.")
    if madnlp_max_wall_time is not None and madnlp_max_wall_time <= 0:
        raise ValueError("madnlp_max_wall_time must be strictly positive.")
    if (
        madnlp_acceptable_tolerance is not None
        and madnlp_acceptable_tolerance <= 0
    ):
        raise ValueError(
            "madnlp_acceptable_tolerance must be strictly positive."
        )
    if (
        madnlp_acceptable_iterations is not None
        and madnlp_acceptable_iterations < 1
    ):
        raise ValueError(
            "madnlp_acceptable_iterations must be strictly positive."
        )
    if alpaqa_initial_tolerance is not None and alpaqa_initial_tolerance <= 0:
        raise ValueError("alpaqa_initial_tolerance must be strictly positive.")
    if (
        alpaqa_penalty_update_factor is not None
        and alpaqa_penalty_update_factor <= 1
    ):
        raise ValueError("alpaqa_penalty_update_factor must be greater than one.")
    if alpaqa_maximum_penalty is not None and alpaqa_maximum_penalty <= 0:
        raise ValueError("alpaqa_maximum_penalty must be strictly positive.")
    if (
        alpaqa_panoc_max_wall_time is not None
        and alpaqa_panoc_max_wall_time <= 0
    ):
        raise ValueError("alpaqa_panoc_max_wall_time must be strictly positive.")
    if alpaqa_max_no_progress is not None and alpaqa_max_no_progress < 1:
        raise ValueError("alpaqa_max_no_progress must be strictly positive.")

    if solver_namespace is None:
        from bioptim import Solver as solver_namespace

    if check_availability:
        available, reason = nlp_solver_availability(
            solver_name, solver_namespace=solver_namespace
        )
        if not available:
            raise SolverBackendUnavailable(reason)

    factory_name = solver_name.upper()
    factory = getattr(solver_namespace, factory_name, None)
    if factory is None:
        raise SolverBackendUnavailable(
            f"Bioptim does not expose Solver.{factory_name}."
        )

    if solver_name == "ipopt":
        solver = factory(
            show_online_optim=False,
            _max_iter=max_iterations,
            show_options={"show_bounds": True},
        )
        solver.set_warm_start_init_point("yes")
        solver.set_mu_init(1e-2)
        solver.set_tol(tolerance)
        solver.set_dual_inf_tol(tolerance)
        solver.set_constr_viol_tol(tolerance)
        solver.set_linear_solver(ipopt_linear_solver)
        solver.set_print_level(print_level)
        if ipopt_hsl_library is not None:
            solver.set_option_unsafe(ipopt_hsl_library, "hsllib")
        for name, value in (ipopt_options or {}).items():
            solver.set_option_unsafe(value, name)
        solver.set_c_compile(ipopt_c_compile)
        return solver

    solver = factory()
    solver.set_convergence_tolerance(tolerance)
    solver.set_constraint_tolerance(tolerance)
    solver.set_maximum_iterations(max_iterations)

    if solver_name == "madnlp":
        solver.set_print_level("ERROR" if print_level == 0 else print_level)
        # The pinned madnlp_c runtime rejects both ``dual_initialized`` and
        # ``mu_init``.  The reliable hot start is therefore the shifted,
        # projected primal trajectory supplied by Cocofest, without a
        # solver-specific barrier or multiplier initialization.
        for name, value in (
            ("linear_solver", madnlp_linear_solver),
            ("max_wall_time", madnlp_max_wall_time),
            ("nlp_scaling", madnlp_nlp_scaling),
            ("acceptable_tol", madnlp_acceptable_tolerance),
            ("acceptable_iter", madnlp_acceptable_iterations),
        ):
            if value is not None:
                solver.set_option_unsafe(value, name)
        solver.set_c_compile(madnlp_c_compile)
        return solver

    solver.set_print_level(print_level)
    solver.set_alm_maximum_iterations(
        max_iterations
        if alpaqa_alm_max_iterations is None
        else alpaqa_alm_max_iterations
    )
    solver.set_lbfgs_memory(alpaqa_lbfgs_memory)
    if alpaqa_max_wall_time is not None:
        if alpaqa_max_wall_time <= 0:
            raise ValueError("alpaqa_max_wall_time must be strictly positive.")
        solver.set_maximum_wall_time(alpaqa_max_wall_time)
    if alpaqa_initial_penalty is not None:
        if alpaqa_initial_penalty < 0:
            raise ValueError("alpaqa_initial_penalty must be non-negative.")
        solver.set_initial_penalty(alpaqa_initial_penalty)
    if alpaqa_initial_tolerance is not None:
        solver.set_initial_tolerance(alpaqa_initial_tolerance)
    if alpaqa_penalty_update_factor is not None:
        solver.set_penalty_update_factor(alpaqa_penalty_update_factor)
    if alpaqa_maximum_penalty is not None:
        solver.set_maximum_penalty(alpaqa_maximum_penalty)
    if alpaqa_panoc_max_wall_time is not None:
        solver.set_option_unsafe(
            f"{alpaqa_panoc_max_wall_time}s", "panoc.max_time"
        )
    if alpaqa_max_no_progress is not None:
        solver.set_option_unsafe(alpaqa_max_no_progress, "panoc.max_no_progress")
    return solver
