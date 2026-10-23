"""Small configuration tests that do not construct a cycling OCP."""

import pytest

from examples.fes_multibody.cycling.numerical_options import build_cycling_mhe_info, resolve_ipopt_linear_solver


def test_historical_defaults_are_unchanged():
    info = build_cycling_mhe_info(10)
    assert info["n_cycles"] == 10
    assert info["ode_solver"].polynomial_degree == 3
    assert info["ode_solver"].method == "radau"
    assert info["use_sx"] is False
    assert info["linear_solver"] is None
    assert resolve_ipopt_linear_solver(None, "linux") == "ma57"
    assert resolve_ipopt_linear_solver(None, "darwin") == "mumps"


def test_paper_numerics_are_explicit():
    info = build_cycling_mhe_info(100, radau_stages=5, use_sx=True, linear_solver="ma57")
    assert info["ode_solver"].polynomial_degree == 5
    assert info["use_sx"] is True
    assert info["linear_solver"] == "ma57"
    assert resolve_ipopt_linear_solver(info["linear_solver"], "darwin") == "ma57"


@pytest.mark.parametrize("stages", [0, -1, 2.5, True])
def test_invalid_radau_stage_count(stages):
    with pytest.raises(ValueError, match="radau_stages"):
        build_cycling_mhe_info(1, radau_stages=stages)


@pytest.mark.parametrize("cycles", [0, -1, 1.5, False])
def test_invalid_cycle_count(cycles):
    with pytest.raises(ValueError, match="total_cycles"):
        build_cycling_mhe_info(cycles)


def test_empty_linear_solver_name_is_rejected():
    with pytest.raises(ValueError, match="linear_solver"):
        build_cycling_mhe_info(1, linear_solver=" ")
