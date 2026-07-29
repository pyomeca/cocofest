from types import SimpleNamespace

import pytest

from cocofest.optimization.solver_backends import (
    MADNLP_QUIET_PRINT_LEVEL,
    SolverBackendUnavailable,
    configure_nlp_solver,
    nlp_solver_availability,
)


class _FakeSolver:
    def __init__(self, **constructor_options):
        self.constructor_options = constructor_options
        self.calls = []

    def __getattr__(self, name):
        if not name.startswith("set_"):
            raise AttributeError(name)

        def record(*args):
            self.calls.append((name, *args))

        return record


class _Factory:
    def __init__(self):
        self.instances = []

    def __call__(self, **kwargs):
        instance = _FakeSolver(**kwargs)
        self.instances.append(instance)
        return instance


def _solver_namespace(*names):
    return SimpleNamespace(**{name: _Factory() for name in names})


def test_solver_availability_checks_bioptim_factory_and_casadi_plugin():
    namespace = _solver_namespace("IPOPT", "MADNLP")
    plugins = SimpleNamespace(has_nlpsol=lambda name: name == "madnlp")

    assert nlp_solver_availability(
        "madnlp", solver_namespace=namespace, casadi_module=plugins
    ) == (True, None)
    available, reason = nlp_solver_availability(
        "alpaqa", solver_namespace=namespace, casadi_module=plugins
    )
    assert available is False
    assert "Solver.ALPAQA" in reason


def test_configure_madnlp_uses_supported_primal_hot_start():
    namespace = _solver_namespace("MADNLP")

    solver = configure_nlp_solver(
        "madnlp",
        max_iterations=321,
        tolerance=2e-6,
        madnlp_linear_solver="umfpack",
        solver_namespace=namespace,
        check_availability=False,
    )

    assert ("set_convergence_tolerance", 2e-6) in solver.calls
    assert ("set_constraint_tolerance", 2e-6) in solver.calls
    assert ("set_maximum_iterations", 321) in solver.calls
    assert MADNLP_QUIET_PRINT_LEVEL == 6
    assert (
        "set_option_unsafe",
        MADNLP_QUIET_PRINT_LEVEL,
        "print_level",
    ) in solver.calls
    assert not any(call[0] == "set_print_level" for call in solver.calls)
    assert not any(
        call[0] == "set_option_unsafe" and call[-1] == "mu_init"
        for call in solver.calls
    )
    assert not any(call[0] == "set_warm_start_options" for call in solver.calls)
    assert ("set_option_unsafe", "umfpack", "linear_solver") in solver.calls
    assert ("set_c_compile", False) in solver.calls


@pytest.mark.parametrize(
    ("requested_level", "runtime_level"),
    [(0, 6), (-1, 1), (1, 1), (5, 5), (6, 6), (9, 6)],
)
def test_configure_madnlp_maps_print_level_to_runtime_enum(
    requested_level, runtime_level
):
    solver = configure_nlp_solver(
        "madnlp",
        max_iterations=10,
        print_level=requested_level,
        solver_namespace=_solver_namespace("MADNLP"),
        check_availability=False,
    )

    assert ("set_option_unsafe", runtime_level, "print_level") in solver.calls


def test_configure_madnlp_maps_pardiso_mkl_to_native_libmad_type():
    namespace = _solver_namespace("MADNLP")

    solver = configure_nlp_solver(
        "madnlp",
        max_iterations=321,
        tolerance=2e-6,
        madnlp_linear_solver="pardiso_mkl",
        solver_namespace=namespace,
        check_availability=False,
    )

    assert (
        "set_option_unsafe",
        "PardisoMKLSolver",
        "linear_solver",
    ) in solver.calls


def test_configure_madnlp_keeps_mumps_runtime_name():
    solver = configure_nlp_solver(
        "madnlp",
        max_iterations=321,
        madnlp_linear_solver="mumps",
        solver_namespace=_solver_namespace("MADNLP"),
        check_availability=False,
    )

    assert ("set_option_unsafe", "mumps", "linear_solver") in solver.calls


def test_configure_fatrop_uses_time_structured_native_options():
    namespace = _solver_namespace("FATROP")

    solver = configure_nlp_solver(
        "fatrop",
        max_iterations=432,
        tolerance=3e-7,
        print_level=2,
        fatrop_c_compile=True,
        fatrop_structure_detection="auto",
        solver_namespace=namespace,
        check_availability=False,
    )

    assert solver.constructor_options == {
        "_structure_detection": "auto",
        "_c_compile": True,
    }
    assert ("set_convergence_tolerance", 3e-7) in solver.calls
    assert ("set_constraint_tolerance", 3e-7) in solver.calls
    assert ("set_maximum_iterations", 432) in solver.calls
    assert ("set_print_level", 2) in solver.calls
    assert ("set_bound_tightening_factor", 1e-8) in solver.calls
    assert ("set_c_compile", True) in solver.calls
    assert not any(call[0] == "set_warm_start_options" for call in solver.calls)


def test_configure_alpaqa_sets_both_iteration_budgets_and_lbfgs():
    namespace = _solver_namespace("ALPAQA")

    solver = configure_nlp_solver(
        "alpaqa",
        max_iterations=500,
        tolerance=1e-5,
        alpaqa_alm_max_iterations=80,
        alpaqa_lbfgs_memory=30,
        alpaqa_max_wall_time=0.75,
        alpaqa_initial_penalty=5.0,
        alpaqa_initial_tolerance=1e-3,
        alpaqa_penalty_update_factor=5.0,
        alpaqa_maximum_penalty=1e7,
        alpaqa_panoc_max_wall_time=0.25,
        alpaqa_max_no_progress=25,
        solver_namespace=namespace,
        check_availability=False,
    )

    assert ("set_maximum_iterations", 500) in solver.calls
    assert ("set_alm_maximum_iterations", 80) in solver.calls
    assert ("set_lbfgs_memory", 30) in solver.calls
    assert ("set_maximum_wall_time", 0.75) in solver.calls
    assert ("set_initial_penalty", 5.0) in solver.calls
    assert ("set_initial_tolerance", 1e-3) in solver.calls
    assert ("set_penalty_update_factor", 5.0) in solver.calls
    assert ("set_maximum_penalty", 1e7) in solver.calls
    assert ("set_option_unsafe", "0.25s", "panoc.max_time") in solver.calls
    assert ("set_option_unsafe", 25, "panoc.max_no_progress") in solver.calls


def test_configure_ipopt_retains_robust_cocofest_settings():
    namespace = _solver_namespace("IPOPT")

    solver = configure_nlp_solver(
        "ipopt",
        max_iterations=1000,
        tolerance=1e-6,
        ipopt_linear_solver="mumps",
        ipopt_hsl_library="/opt/coinhsl/libcoinhsl.dylib",
        ipopt_c_compile=True,
        ipopt_options={
            "linear_system_scaling": "none",
            "ma57_automatic_scaling": "yes",
        },
        solver_namespace=namespace,
        check_availability=False,
    )

    assert solver.constructor_options["_max_iter"] == 1000
    assert ("set_warm_start_init_point", "yes") in solver.calls
    assert ("set_mu_init", 1e-2) in solver.calls
    assert ("set_tol", 1e-6) in solver.calls
    assert ("set_dual_inf_tol", 1e-6) in solver.calls
    assert ("set_constr_viol_tol", 1e-6) in solver.calls
    assert ("set_linear_solver", "mumps") in solver.calls
    assert ("set_print_level", 0) in solver.calls
    assert (
        "set_option_unsafe",
        "/opt/coinhsl/libcoinhsl.dylib",
        "hsllib",
    ) in solver.calls
    assert ("set_option_unsafe", "none", "linear_system_scaling") in solver.calls
    assert ("set_option_unsafe", "yes", "ma57_automatic_scaling") in solver.calls
    assert ("set_c_compile", True) in solver.calls


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_iterations": 0}, "max_iterations"),
        ({"max_iterations": 1, "tolerance": 0}, "tolerance"),
        (
            {"max_iterations": 1, "alpaqa_lbfgs_memory": 0},
            "alpaqa_lbfgs_memory",
        ),
        (
            {"max_iterations": 1, "madnlp_linear_solver": "MumpsSolver"},
            "madnlp_linear_solver",
        ),
        (
            {"max_iterations": 1, "fatrop_structure_detection": "none"},
            "fatrop_structure_detection",
        ),
        (
            {"max_iterations": 1, "fatrop_bound_tightening_factor": -1e-8},
            "fatrop_bound_tightening_factor",
        ),
        (
            {"max_iterations": 1, "alpaqa_penalty_update_factor": 1},
            "alpaqa_penalty_update_factor",
        ),
        (
            {"max_iterations": 1, "alpaqa_panoc_max_wall_time": 0},
            "alpaqa_panoc_max_wall_time",
        ),
    ],
)
def test_solver_configuration_rejects_invalid_common_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        configure_nlp_solver(
            "alpaqa",
            solver_namespace=_solver_namespace("ALPAQA"),
            check_availability=False,
            **kwargs,
        )


def test_fatrop_rejects_iteration_budget_above_native_limit():
    with pytest.raises(ValueError, match="bounded to 1000"):
        configure_nlp_solver(
            "fatrop",
            max_iterations=1001,
            solver_namespace=_solver_namespace("FATROP"),
            check_availability=False,
        )


def test_missing_optional_solver_has_actionable_error():
    with pytest.raises(SolverBackendUnavailable, match="Solver.MADNLP"):
        configure_nlp_solver(
            "madnlp",
            max_iterations=10,
            solver_namespace=_solver_namespace("IPOPT"),
            check_availability=False,
        )
