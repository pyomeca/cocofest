"""Optional nonlinear solver backends used by Cocofest benchmarks.

Fatrop, MadNLP and Alpaqa are exposed by compatible Bioptim revisions through
CasADi's ``nlpsol`` API.  They are optional on purpose: importing Cocofest must
continue to work with a standard Bioptim/CasADi installation.
"""

from __future__ import annotations

from typing import Any

NLP_SOLVER_NAMES = ("ipopt", "fatrop", "madnlp", "alpaqa")
MADNLP_LINEAR_SOLVER_NAMES = (
    "mumps",
    "umfpack",
    "lapack_cpu",
    "pardiso_mkl",
    "cudss",
    "lapack_gpu",
    "cucholesky",
)
MADNLP_LINEAR_SOLVER_RUNTIME_NAMES = {
    "pardiso_mkl": "PardisoMKLSolver",
}
# MadNLP 0.9.2 defines LogLevels as TRACE=1 through ERROR=6.  The libMad
# interface transports that enum as an integer and rejects IPOPT's conventional
# quiet value 0.
MADNLP_QUIET_PRINT_LEVEL = 6


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
    fatrop_c_compile: bool = False,
    fatrop_structure_detection: str = "auto",
    fatrop_bound_tightening_factor: float = 1e-8,
    madnlp_c_compile: bool = False,
    madnlp_linear_solver: str | None = None,
    alpaqa_alm_max_iterations: int | None = None,
    alpaqa_lbfgs_memory: int = 20,
    alpaqa_max_wall_time: float | None = None,
    alpaqa_initial_penalty: float | None = None,
    alpaqa_initial_tolerance: float | None = None,
    alpaqa_penalty_update_factor: float | None = None,
    alpaqa_maximum_penalty: float | None = None,
    alpaqa_panoc_max_wall_time: float | None = None,
    alpaqa_max_no_progress: int | None = None,
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
    if solver_name == "fatrop" and max_iterations > 1000:
        raise ValueError(
            "Fatrop's native max_iter option is bounded to 1000 in CasADi 3.7.2."
        )
    if tolerance <= 0:
        raise ValueError("tolerance must be strictly positive.")
    if fatrop_structure_detection not in {"auto", "manual"}:
        raise ValueError("fatrop_structure_detection must be 'auto' or 'manual'.")
    if fatrop_bound_tightening_factor < 0:
        raise ValueError("fatrop_bound_tightening_factor must be non-negative.")
    if alpaqa_lbfgs_memory < 1:
        raise ValueError("alpaqa_lbfgs_memory must be strictly positive.")
    if (
        madnlp_linear_solver is not None
        and madnlp_linear_solver not in MADNLP_LINEAR_SOLVER_NAMES
    ):
        raise ValueError(
            "madnlp_linear_solver must be one of "
            f"{', '.join(MADNLP_LINEAR_SOLVER_NAMES)}."
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

    if solver_name == "fatrop":
        solver = factory(
            _structure_detection=fatrop_structure_detection,
            _c_compile=fatrop_c_compile,
        )
    else:
        solver = factory()
    solver.set_convergence_tolerance(tolerance)
    solver.set_constraint_tolerance(tolerance)
    solver.set_maximum_iterations(max_iterations)

    if solver_name == "fatrop":
        solver.set_print_level(print_level)
        # Fatrop relaxes each bound relatively. For large fatigue capacity
        # states (~7000), the native 1e-8 relaxation permits about 7e-5
        # absolute overshoot. The Bioptim integration tightens only the solver
        # call bounds while retaining the physical limits for the audit.
        set_bound_tightening = getattr(
            solver, "set_bound_tightening_factor", None
        )
        if set_bound_tightening is None:
            raise SolverBackendUnavailable(
                "This Fatrop benchmark requires the Bioptim interface with "
                "set_bound_tightening_factor()."
            )
        set_bound_tightening(fatrop_bound_tightening_factor)
        solver.set_c_compile(fatrop_c_compile)
        return solver

    if solver_name == "madnlp":
        # Bypass Bioptim's generic print-level mapping. The pinned libMad
        # runtime embeds MadNLP 0.9.2, whose LogLevels enum accepts only 1..6
        # and uses ERROR=6 as the quiet benchmark setting.
        madnlp_print_level = (
            MADNLP_QUIET_PRINT_LEVEL
            if int(print_level) == 0
            else max(1, min(int(print_level), MADNLP_QUIET_PRINT_LEVEL))
        )
        solver.set_option_unsafe(madnlp_print_level, "print_level")
        # The pinned madnlp_c runtime rejects both ``dual_initialized`` and
        # ``mu_init``.  The reliable hot start is therefore the shifted,
        # projected primal trajectory supplied by Cocofest, without a
        # solver-specific barrier or multiplier initialization.
        if madnlp_linear_solver is not None:
            runtime_linear_solver = MADNLP_LINEAR_SOLVER_RUNTIME_NAMES.get(
                madnlp_linear_solver, madnlp_linear_solver
            )
            solver.set_option_unsafe(runtime_linear_solver, "linear_solver")
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
