"""
CLI-friendly ACADOS example for the periodic Ding pulse-width cycling MHE with a constant crank torque.
"""

import argparse
import ctypes
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from sys import platform as sys_platform

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bioptim import MultiCyclicCycleSolutions, OdeSolver, SolutionMerge, Solver
from bioptim.optimization.receding_horizon_optimization import (
    RecedingHorizonOptimization,
)

try:
    from .cycling_pulse_width_mhe import prepare_nmpc, set_fes_model
except ImportError:
    from cycling_pulse_width_mhe import prepare_nmpc, set_fes_model

OBJECTIVE_TO_WEIGHT_INDEX = {"force": 0, "fatigue": 1, "control": 2}
ACADOS_STATUS_NAMES = {
    0: "ACADOS_SUCCESS",
    1: "ACADOS_NAN_DETECTED",
    2: "ACADOS_MAXITER",
    3: "ACADOS_MINSTEP",
    4: "ACADOS_QP_FAILURE",
    5: "ACADOS_READY",
    6: "ACADOS_UNBOUNDED",
    7: "ACADOS_TIMEOUT",
    8: "ACADOS_QPSCALING_BOUNDS_NOT_SATISFIED",
}


def build_time_dependent_rk4_map(
    rhs,
    state,
    control,
    stage_parameters,
    local_time,
    interval_duration: float,
    n_substeps: int,
):
    """Build a node-to-node RK4 map while retaining the local time dependence."""
    from casadi import Function

    if n_substeps < 1:
        raise ValueError("n_substeps must be strictly positive.")
    rhs_function = Function(
        "cocofest_discrete_rhs",
        [state, control, stage_parameters, local_time],
        [rhs],
    )
    step = float(interval_duration) / n_substeps
    next_state = state
    for substep in range(n_substeps):
        time = substep * step
        k1 = rhs_function(next_state, control, stage_parameters, time)
        k2 = rhs_function(
            next_state + step * k1 / 2,
            control,
            stage_parameters,
            time + step / 2,
        )
        k3 = rhs_function(
            next_state + step * k2 / 2,
            control,
            stage_parameters,
            time + step / 2,
        )
        k4 = rhs_function(
            next_state + step * k3,
            control,
            stage_parameters,
            time + step,
        )
        next_state = next_state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return next_state


def _periodic_node_interval_duration(ocp) -> float:
    muscle_models = getattr(ocp.nlp[0].model, "muscles_dynamics_model", ())
    intervals = {
        float(model._stim_interval)
        for model in muscle_models
        if getattr(model, "_stim_interval", None) is not None
    }
    if len(intervals) != 1:
        raise RuntimeError(
            "The periodic-node DISCRETE map requires one shared stimulation interval."
        )
    return intervals.pop()


def _periodic_node_dynamics_time(
    integrator_type: str, acados_time, interval_start, interval_duration: float
):
    """Return the absolute time used by the periodic-node Ding dynamics."""

    if integrator_type == "ERK":
        # The bundled explicit-ODE generator does not expose the stage time.
        return interval_start + interval_duration / 2
    if integrator_type == "IRK":
        # OCP shooting intervals initialize the IRK simulator at t0=0.
        return acados_time + interval_start
    raise ValueError(f"Unsupported time-dependent integrator: {integrator_type}")


def _scaled_acados_dynamics_rhs(rhs, state_scaling: np.ndarray, n_parameters: int):
    """Convert a physical derivative to the derivative of ACADOS scaled states."""

    from casadi import DM, vertcat

    state_scaling = np.asarray(state_scaling, dtype=float).reshape(-1)
    complete_scaling = np.concatenate((np.ones(n_parameters), state_scaling))
    if rhs.shape[0] != complete_scaling.size:
        raise ValueError(
            "ACADOS dynamics and scaling dimensions differ "
            f"({rhs.shape[0]} != {complete_scaling.size})."
        )
    return rhs / vertcat(*DM(complete_scaling).elements())


def patch_bioptim_acados_interface() -> None:
    """
    Patch the Bioptim 3.4 ACADOS interface for this example.

    The installed Bioptim interface currently exports control bounds as
    ``lbu=max`` and ``ubu=min`` during code generation, then later updates the
    solver with unscaled control bounds. The ACADOS variables are scaled, so both
    steps make the pulse-width QP infeasible. The interface also passes
    ``numerical_data_timeseries`` to the exported dynamics without declaring it
    as an ACADOS model parameter, which leaves stage-varying inputs as free
    CasADi symbols. This local patch keeps the example self-contained until the
    upstream interface is fixed.
    """
    from bioptim.interfaces.acados_interface import AcadosInterface

    if getattr(AcadosInterface, "_cocofest_interface_patch", False):
        return

    from bioptim.interfaces.interface_utils import get_numerical_timeseries

    original_export_model = AcadosInterface._AcadosInterface__acados_export_model
    original_set_constraints = AcadosInterface._AcadosInterface__set_constraints
    original_update_solver = AcadosInterface._AcadosInterface__update_solver

    def patched_export_model(self, ocp):
        original_export_model(self, ocp)
        state_scaling = np.ones(ocp.nlp[0].states.shape)
        for key in ocp.nlp[0].states.keys():
            state_scaling[ocp.nlp[0].states[key].index] = np.asarray(
                ocp.nlp[0].x_scaling[key].scaling[:, 0], dtype=float
            )
        self.acados_model.f_expl_expr = _scaled_acados_dynamics_rhs(
            self.acados_model.f_expl_expr,
            state_scaling,
            ocp.nlp[0].parameters.shape,
        )
        self.acados_model.f_impl_expr = (
            self.acados_model.xdot - self.acados_model.f_expl_expr
        )
        numerical_timeseries = ocp.nlp[0].numerical_timeseries.cx_start
        if numerical_timeseries.shape[0] == 0:
            return

        from casadi import SX, substitute

        original_time = ocp.nlp[0].time_cx
        local_time = SX.sym("cocofest_local_time", 1, 1)
        is_periodic_node = "periodic_calcium" in str(
            ocp.nlp[0].numerical_data_timeseries
        )
        interval_duration = (
            _periodic_node_interval_duration(ocp) if is_periodic_node else None
        )
        if is_periodic_node and self.opts.integrator_type == "DISCRETE":
            timed_rhs = substitute(
                self.acados_model.f_expl_expr,
                original_time,
                local_time + numerical_timeseries[-1],
            )
            self.acados_model.disc_dyn_expr = build_time_dependent_rk4_map(
                rhs=timed_rhs,
                state=self.acados_model.x,
                control=self.acados_model.u,
                stage_parameters=numerical_timeseries,
                local_time=local_time,
                interval_duration=interval_duration,
                n_substeps=getattr(ocp, "_cocofest_discrete_substeps", 5),
            )
        else:
            if is_periodic_node and self.opts.integrator_type in ("ERK", "IRK"):
                dynamics_time = _periodic_node_dynamics_time(
                    self.opts.integrator_type,
                    local_time,
                    numerical_timeseries[-1],
                    interval_duration,
                )
                if self.opts.integrator_type == "IRK":
                    self.acados_model.t = local_time
            elif is_periodic_node:
                dynamics_time = local_time
                self.acados_model.t = local_time
            else:
                dynamics_time = original_time
            self.acados_model.f_expl_expr = substitute(
                self.acados_model.f_expl_expr, original_time, dynamics_time
            )
            self.acados_model.f_impl_expr = substitute(
                self.acados_model.f_impl_expr, original_time, dynamics_time
            )
        self.acados_model.p = numerical_timeseries
        initial_numerical_data = np.asarray(
            get_numerical_timeseries(ocp, 0, 0, slice(None)), dtype=float
        ).reshape(-1)
        self.acados_ocp.parameter_values = initial_numerical_data

    def scaled_control_bounds(interface) -> tuple[np.ndarray, np.ndarray]:
        lower = np.empty(interface.acados_ocp.dims.nu)
        upper = np.empty(interface.acados_ocp.dims.nu)
        for key in interface.ocp.nlp[0].controls.keys():
            control_bounds = (
                interface.ocp.nlp[0]
                .u_bounds[key]
                .scale(interface.ocp.nlp[0].u_scaling[key].scaling)
            )
            index = interface.ocp.nlp[0].controls[key].index
            lower[index] = np.asarray(control_bounds.min[:, 0], dtype=float)
            upper[index] = np.asarray(control_bounds.max[:, 0], dtype=float)

        if np.any(lower > upper):
            raise RuntimeError(
                "Scaled ACADOS control bounds are inconsistent "
                f"(lower={lower}, upper={upper})."
            )
        return lower, upper

    def patched_set_constraints(self, ocp):
        original_set_constraints(self, ocp)
        lower, upper = scaled_control_bounds(self)
        self.acados_ocp.constraints.lbu = lower.reshape((-1, 1))
        self.acados_ocp.constraints.ubu = upper.reshape((-1, 1))

    def patched_update_solver(self):
        original_update_solver(self)
        if self.ocp_solver is None:
            return

        dual_summary = apply_acados_dual_warm_start(
            self.ocp_solver,
            horizon=self.acados_ocp.solver_options.N_horizon,
            mode=getattr(self.ocp, "_cocofest_dual_warm_start_mode", "preserve"),
            shift_stages=getattr(self.ocp, "_cocofest_dual_shift_stages", 0),
        )
        self.ocp._cocofest_last_dual_warm_start_summary = dual_summary

        if self.ocp.nlp[0].numerical_timeseries.shape:
            terminal_node = self.acados_ocp.solver_options.N_horizon
            for stage in range(terminal_node + 1):
                stage_values = np.asarray(
                    get_numerical_timeseries(self.ocp, 0, stage, slice(None)),
                    dtype=float,
                ).reshape(-1)
                self.ocp_solver.set(stage, "p", stage_values)

        param_init = []
        for key in self.ocp.nlp[0].parameters.keys():
            scaled_init = self.ocp.parameter_init[key].scale(
                self.ocp.parameters[key].scaling.scaling
            )
            param_init = np.concatenate((param_init, scaled_init.init[:, 0]))

        terminal_node = self.acados_ocp.solver_options.N_horizon
        terminal_x = np.empty((self.ocp.nlp[0].states.shape))
        for key in self.ocp.nlp[0].states.keys():
            index = self.ocp.nlp[0].states[key].index
            self.ocp.nlp[0].x_init[key].check_and_adjust_dimensions(
                self.ocp.nlp[0].states[key].shape, self.ocp.nlp[0].ns
            )
            terminal_x[index] = (
                self.ocp.nlp[0].x_init[key].init.evaluate_at(terminal_node)
                / self.ocp.nlp[0].x_scaling[key].scaling[:, 0]
            )
        self.ocp_solver.set(
            terminal_node, "x", np.concatenate((param_init, terminal_x))
        )

        lower, upper = scaled_control_bounds(self)
        nodewise_bounds = getattr(self.ocp, "_cocofest_nodewise_control_bounds", {})
        for stage in range(self.acados_ocp.solver_options.N_horizon):
            if getattr(self.ocp, "_cocofest_fix_controls_to_warmup", False):
                stage_control = np.empty(self.ocp.nlp[0].controls.shape)
                for key in self.ocp.nlp[0].controls.keys():
                    index = self.ocp.nlp[0].controls[key].index
                    stage_control[index] = (
                        self.ocp.nlp[0].u_init[key].init.evaluate_at(stage)
                        / self.ocp.nlp[0].u_scaling[key].scaling[:, 0]
                    )
                tolerance = getattr(self.ocp, "_cocofest_fixed_control_tolerance", 1e-5)
                self.ocp_solver.constraints_set(stage, "lbu", stage_control - tolerance)
                self.ocp_solver.constraints_set(stage, "ubu", stage_control + tolerance)
            else:
                stage_lower = lower.copy()
                stage_upper = upper.copy()
                for key, (key_lower, key_upper) in nodewise_bounds.items():
                    index = self.ocp.nlp[0].controls[key].index
                    scaling = self.ocp.nlp[0].u_scaling[key].scaling[:, 0]
                    stage_lower[index] = key_lower[:, stage] / scaling
                    stage_upper[index] = key_upper[:, stage] / scaling
                self.ocp_solver.constraints_set(stage, "lbu", stage_lower)
                self.ocp_solver.constraints_set(stage, "ubu", stage_upper)

    AcadosInterface._AcadosInterface__acados_export_model = patched_export_model
    AcadosInterface._AcadosInterface__set_constraints = patched_set_constraints
    AcadosInterface._AcadosInterface__update_solver = patched_update_solver
    AcadosInterface._cocofest_interface_patch = True


def apply_acados_dual_warm_start(
    acados_solver, horizon: int, mode: str, shift_stages: int
) -> dict[str, int | str]:
    if mode not in {"preserve", "reset", "shift"}:
        raise ValueError(f"Unsupported ACADOS dual warm-start mode '{mode}'.")
    if shift_stages < 0:
        raise ValueError("ACADOS dual shift must be non-negative.")
    if mode == "preserve":
        return {"mode": mode, "shift_stages": 0, "zeroed_tail_stages": 0}

    lam = [
        np.asarray(acados_solver.get(stage, "lam"), dtype=float)
        for stage in range(horizon + 1)
    ]
    pi = [
        np.asarray(acados_solver.get(stage, "pi"), dtype=float)
        for stage in range(horizon)
    ]
    if mode == "reset":
        for stage, values in enumerate(lam):
            acados_solver.set(stage, "lam", np.zeros_like(values))
        for stage, values in enumerate(pi):
            acados_solver.set(stage, "pi", np.zeros_like(values))
        return {
            "mode": mode,
            "shift_stages": 0,
            "zeroed_tail_stages": horizon + 1,
        }

    shift = min(shift_stages, horizon)
    for stage in range(horizon + 1):
        source = stage + shift
        values = (
            lam[source]
            if source <= horizon and lam[source].shape == lam[stage].shape
            else np.zeros_like(lam[stage])
        )
        acados_solver.set(stage, "lam", values)
    for stage in range(horizon):
        source = stage + shift
        values = pi[source] if source < horizon else np.zeros_like(pi[stage])
        acados_solver.set(stage, "pi", values)
    return {
        "mode": mode,
        "shift_stages": shift,
        "zeroed_tail_stages": shift,
    }


def parse_objectives(raw_objective: str) -> set[str]:
    values = {item.strip().lower() for item in raw_objective.split(",") if item.strip()}
    allowed = set(OBJECTIVE_TO_WEIGHT_INDEX) | {"none"}
    invalid = values - allowed
    if invalid:
        raise ValueError(f"Unsupported objectives: {', '.join(sorted(invalid))}")
    if "none" in values and len(values) > 1:
        raise ValueError("'none' cannot be combined with other objectives.")
    return values or {"force"}


def build_cost_fun_weight(objectives: set[str]) -> list[int]:
    weights = [0, 0, 0]
    for objective in objectives:
        if objective in OBJECTIVE_TO_WEIGHT_INDEX:
            weights[OBJECTIVE_TO_WEIGHT_INDEX[objective]] = 1
    return weights


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-windows",
        type=int,
        default=2,
        help="Number of successive MHE windows to solve.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=("acados", "ipopt"),
        default="acados",
        help="Solver backend used for the MHE windows.",
    )
    parser.add_argument(
        "--single-shot",
        action="store_true",
        help="Solve only one window directly, without the receding-horizon loop.",
    )
    parser.add_argument(
        "--model-formulation",
        type=str,
        choices=("periodic_node", "periodic", "standard"),
        default="periodic",
        help=(
            "Use exact node-wise periodic calcium forcing, the continuous periodic surrogate, "
            "or the historical finite stimulation history."
        ),
    )
    parser.add_argument(
        "--torque-application",
        type=str,
        choices=("constant", "external_forces"),
        default="constant",
        help="Apply the resistive crank torque directly on wheel_rotation_RotZ or through external_forces.",
    )
    parser.add_argument(
        "--cycles-per-window",
        type=int,
        default=1,
        help="Number of pedaling cycles simultaneously optimized in each MHE window.",
    )
    parser.add_argument(
        "--ode-solver",
        type=str,
        choices=("rk4", "rk8", "irk", "collocation"),
        default="rk4",
        help="Integration scheme used to transcribe the window dynamics.",
    )
    parser.add_argument(
        "--rk-steps",
        type=int,
        default=5,
        help="Number of RK4 integration steps per shooting interval when --ode-solver=rk4.",
    )
    parser.add_argument(
        "--collocation-degree",
        type=int,
        default=3,
        help="Polynomial degree when --ode-solver=collocation.",
    )
    parser.add_argument(
        "--collocation-method",
        type=str,
        default="radau",
        help="Collocation method name when --ode-solver=collocation.",
    )
    parser.add_argument(
        "--stimulations-per-cycle",
        type=int,
        default=30,
        help="Number of stimulation events per pedaling cycle.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="force",
        help="Objective(s) to minimize: force, fatigue, control, none, or comma-separated combinations.",
    )
    parser.add_argument(
        "--objective-shape",
        type=str,
        choices=("quadratic", "linear"),
        default="quadratic",
        help="Shape of the objective terms passed to bioptim.",
    )
    parser.add_argument(
        "--control-regularization-weight",
        type=float,
        default=0.0,
        help="Optional quadratic MINIMIZE_CONTROL weight on each pulse-width control.",
    )
    parser.add_argument(
        "--control-regularization-target",
        type=float,
        default=None,
        help="Optional pulse-width target, in seconds, for the control regularization.",
    )
    parser.add_argument(
        "--control-regularization-target-source",
        choices=("constant", "warmup"),
        default="constant",
        help="Use a constant pulse-width target or the standard IPOPT warmup controls as the target.",
    )
    parser.add_argument(
        "--wheel-qdot-regularization-weight",
        type=float,
        default=0.0,
        help="Optional quadratic MINIMIZE_STATE weight on the crank/wheel qdot.",
    )
    parser.add_argument(
        "--wheel-qdot-regularization-target",
        type=float,
        default=-float(2 * np.pi),
        help="Target wheel angular velocity, in rad/s, for qdot regularization.",
    )
    parser.add_argument(
        "--state-scaling",
        type=str,
        choices=("none", "fes", "full"),
        default="none",
        help="Scale optimization states: none, FES-only, or FES plus q/qdot.",
    )
    parser.add_argument(
        "--pulse-width-scaling",
        type=float,
        default=1 / 400,
        help="Scaling divisor, in seconds, for pulse-width controls.",
    )
    parser.add_argument(
        "--acados-pulse-width-trust-radius",
        type=float,
        default=None,
        help=(
            "Optional pulse-width control trust-region radius, in seconds, around the current "
            "ACADOS initial guess range."
        ),
    )
    parser.add_argument(
        "--acados-transfer-pulse-width-trust-radius",
        type=float,
        default=None,
        help=(
            "Optional pulse-width trust-region radius after the first solved window. "
            "Defaults to --acados-pulse-width-trust-radius."
        ),
    )
    parser.add_argument(
        "--acados-fes-state-trust-radius",
        type=float,
        default=None,
        help=(
            "Optional normalized trust-region radius around the current ACADOS FES state "
            "initial guess range. Applies to Cn, Cn_sum, F, A, Tau1 and Km states."
        ),
    )
    parser.add_argument(
        "--acados-fatigue-warmstart-mode",
        choices=("continuous", "cyclical"),
        default="continuous",
        help=(
            "How A_, Tau1_ and Km_ fatigue states are shifted from the standard IPOPT warmup "
            "to the ACADOS initial guess."
        ),
    )
    parser.add_argument(
        "--acados-standard-warmup-transfer",
        choices=("advance", "phase_shift"),
        default="phase_shift",
        help=(
            "Transfer the last IPOPT cycle with the historical receding-window rule, or preserve "
            "the complete IPOPT trajectory and shift only the absolute crank angle by one turn."
        ),
    )
    parser.add_argument(
        "--acados-horizon-continuation",
        action="store_true",
        help=(
            "Solve and cache a one-cycle ACADOS problem, then tile that solution as the "
            "initial guess for a longer horizon."
        ),
    )
    parser.add_argument(
        "--acados-continuation-source-max-iterations",
        type=int,
        default=500,
        help="Maximum SQP iterations used to obtain the one-cycle ACADOS continuation source.",
    )
    parser.add_argument(
        "--constant-crank-torque",
        type=float,
        default=-0.2,
        help="Resistive torque magnitude in N.m, used either as a constant crank torque or as external_forces.",
    )
    parser.add_argument(
        "--max-acados-iterations",
        type=int,
        default=100,
        help="Maximum number of ACADOS SQP iterations per window.",
    )
    parser.add_argument(
        "--acados-tolerance",
        type=float,
        default=None,
        help="Optional ACADOS NLP convergence tolerance applied to stationarity, constraints and complementarity.",
    )
    parser.add_argument(
        "--acados-stationarity-tolerance",
        type=float,
        default=None,
        help="Optional stationarity-only tolerance; unlike --acados-tolerance it does not relax QP feasibility.",
    )
    parser.add_argument(
        "--acados-first-window-tolerance",
        type=float,
        default=None,
        help="Optional relaxed feasibility tolerance used only on the first ACADOS window.",
    )
    parser.add_argument(
        "--acados-first-window-stationarity-tolerance",
        type=float,
        default=None,
        help="Optional relaxed stationarity tolerance used only on the first ACADOS window.",
    )
    parser.add_argument(
        "--acados-qp-solver",
        choices=(
            "PARTIAL_CONDENSING_HPIPM",
            "FULL_CONDENSING_HPIPM",
            "FULL_CONDENSING_QPOASES",
        ),
        default="PARTIAL_CONDENSING_HPIPM",
        help="QP solver backend used by ACADOS.",
    )
    parser.add_argument(
        "--acados-integrator-type",
        choices=("ERK", "IRK", "DISCRETE"),
        default="IRK",
        help=(
            "ACADOS integrator type. DISCRETE uses a time-aware RK4 node map "
            "for the periodic-node formulation."
        ),
    )
    parser.add_argument(
        "--acados-collocation-type",
        type=str,
        choices=("GAUSS_LEGENDRE", "GAUSS_RADAU_IIA", "EXPLICIT_RUNGE_KUTTA"),
        default="GAUSS_LEGENDRE",
        help="Collocation tableau used by the ACADOS integrator.",
    )
    parser.add_argument(
        "--acados-sim-stages",
        type=int,
        default=4,
        help="Number of ACADOS integration stages per integration step.",
    )
    parser.add_argument(
        "--acados-sim-steps",
        type=int,
        default=None,
        help="Number of ACADOS integration substeps per shooting interval. Defaults to max(3, --rk-steps).",
    )
    parser.add_argument(
        "--acados-newton-iter",
        type=int,
        default=5,
        help="Number of Newton iterations used by the ACADOS implicit integrator.",
    )
    parser.add_argument(
        "--acados-newton-tol",
        type=float,
        default=None,
        help="Optional Newton tolerance for the ACADOS implicit integrator.",
    )
    parser.add_argument(
        "--acados-jac-reuse",
        type=int,
        choices=(0, 1),
        default=0,
        help="Reuse the ACADOS implicit integrator Jacobian across Newton iterations.",
    )
    parser.add_argument(
        "--acados-hessian-approx",
        choices=("GAUSS_NEWTON", "EXACT"),
        default="GAUSS_NEWTON",
        help="Hessian approximation used by ACADOS.",
    )
    parser.add_argument(
        "--acados-nlp-solver-type",
        choices=("SQP", "SQP_WITH_FEASIBLE_QP"),
        default="SQP",
        help="ACADOS NLP solver type.",
    )
    parser.add_argument(
        "--acados-search-direction-mode",
        choices=("NOMINAL_QP", "BYRD_OMOJOKUN", "FEASIBILITY_QP"),
        default="NOMINAL_QP",
        help="Search direction mode used by ACADOS, mainly for SQP_WITH_FEASIBLE_QP.",
    )
    parser.add_argument(
        "--acados-use-constraint-hessian-in-feas-qp",
        action="store_true",
        help="Use constraint Hessians in the feasibility QP of SQP_WITH_FEASIBLE_QP.",
    )
    parser.add_argument(
        "--acados-disable-direction-mode-switch-to-nominal",
        action="store_true",
        help="Keep ACADOS in the selected non-nominal search direction mode.",
    )
    parser.add_argument(
        "--acados-regularize-method",
        choices=(
            "NO_REGULARIZE",
            "MIRROR",
            "PROJECT",
            "PROJECT_REDUC_HESS",
            "CONVEXIFY",
            "GERSHGORIN_LEVENBERG_MARQUARDT",
        ),
        default="GERSHGORIN_LEVENBERG_MARQUARDT",
        help="ACADOS Hessian regularization method.",
    )
    parser.add_argument(
        "--acados-levenberg-marquardt",
        type=float,
        default=0.0,
        help="Additional Levenberg-Marquardt diagonal regularization for ACADOS.",
    )
    parser.add_argument(
        "--acados-globalization",
        choices=("FIXED_STEP", "MERIT_BACKTRACKING", "FUNNEL_L1PEN_LINESEARCH"),
        default="MERIT_BACKTRACKING",
        help="ACADOS globalization strategy.",
    )
    parser.add_argument(
        "--acados-fixed-step-length",
        type=float,
        default=1.0,
        help="Step length used when --acados-globalization=FIXED_STEP.",
    )
    parser.add_argument(
        "--acados-nlp-qp-tol-strategy",
        choices=("FIXED_QP_TOL", "ADAPTIVE_CURRENT_RES_JOINT", "ADAPTIVE_QPSCALING"),
        default="ADAPTIVE_QPSCALING",
        help="Strategy used by ACADOS to set QP tolerances inside SQP.",
    )
    parser.add_argument(
        "--acados-qp-iter-max",
        type=int,
        default=50,
        help="Maximum number of HPIPM QP iterations.",
    )
    parser.add_argument(
        "--acados-qp-warm-start-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=0,
        help="HPIPM QP warm-start level (0 disables it).",
    )
    parser.add_argument(
        "--acados-warm-start-first-qp",
        action="store_true",
        help="Warm-start the first QP of each ACADOS NLP solve.",
    )
    parser.add_argument(
        "--acados-warm-start-first-qp-from-nlp",
        action="store_true",
        help="Initialize the first QP warm start from the current NLP iterate.",
    )
    parser.add_argument(
        "--acados-dual-warm-start-mode",
        choices=("preserve", "reset", "shift"),
        default="reset",
        help=(
            "Treatment of ACADOS inequality and dynamics multipliers between MHE windows. "
            "Reset is the robust default; shift is experimental."
        ),
    )
    parser.add_argument(
        "--acados-qpscaling-scale-objective",
        choices=("NO_OBJECTIVE_SCALING", "OBJECTIVE_GERSHGORIN"),
        default="OBJECTIVE_GERSHGORIN",
        help="Objective scaling strategy used by the ACADOS QP scaling module.",
    )
    parser.add_argument(
        "--acados-qpscaling-scale-constraints",
        choices=("NO_CONSTRAINT_SCALING", "INF_NORM"),
        default="INF_NORM",
        help="Constraint scaling strategy used by the ACADOS QP scaling module.",
    )
    parser.add_argument(
        "--acados-ext-qp-res",
        action="store_true",
        help="Ask ACADOS to log extended QP residuals in the statistics table.",
    )
    parser.add_argument(
        "--acados-diagnostics",
        action="store_true",
        help="Print ACADOS residuals, SQP/QP stats and finite-value checks after the solve.",
    )
    parser.add_argument(
        "--acados-print-level",
        type=int,
        default=0,
        help="Verbosity passed to ACADOS.",
    )
    parser.add_argument(
        "--acados-wheel-q-slack",
        type=float,
        default=0.02,
        help="First and terminal node slack, in rad, for the ACADOS wheel/crank angle transfer bounds.",
    )
    parser.add_argument(
        "--acados-wheel-qdot-slack",
        type=float,
        default=0.5,
        help="First node slack, in rad/s, for the ACADOS wheel/crank velocity transfer bound when enabled.",
    )
    parser.add_argument(
        "--acados-wheel-q-path-margin",
        type=float,
        default=2.0,
        help="Path margin, in rad, around the transferred ACADOS wheel/crank angle interval.",
    )
    parser.add_argument(
        "--acados-project-qdot-from-q",
        action="store_true",
        help="Project the ACADOS qdot initial guess from finite differences of q before solving.",
    )
    parser.add_argument(
        "--acados-transfer-full-dynamics-rollout",
        action="store_true",
        help="Reintegrate the appended cycle with the complete dynamics after each window transfer.",
    )
    parser.add_argument(
        "--acados-transfer-phase-one",
        action="store_true",
        help=(
            "Apply the bounded proximal complete-dynamics phase I after each MHE window "
            "transfer. Uses the --full-dynamics-phase-one-* settings."
        ),
    )
    parser.add_argument(
        "--acados-transfer-rollout-substeps",
        type=int,
        default=5,
        help="RK4 substeps used by the full-dynamics inter-window rollout.",
    )
    parser.add_argument(
        "--acados-transfer-rollout-max-bound-violation",
        type=float,
        default=1.0,
        help="Reject the inter-window rollout when a state exceeds its bounds by more than this value.",
    )
    parser.add_argument(
        "--warmup-state-comparison-limit",
        type=int,
        default=12,
        help="Number of warmup-vs-initial-guess state rows to print when ACADOS diagnostics are enabled.",
    )
    parser.add_argument(
        "--max-ipopt-iterations",
        type=int,
        default=2000,
        help="Maximum number of IPOPT iterations per window.",
    )
    parser.add_argument(
        "--full-dynamics-phase-one",
        action="store_true",
        help=(
            "Apply a bounded proximal projection of the complete dynamics before solving. "
            "This is a feasibility diagnostic, not a replacement for solver convergence."
        ),
    )
    parser.add_argument(
        "--full-dynamics-phase-one-proximity-weight",
        type=float,
        default=1.0,
        help="Weight retaining the IPOPT reference trajectory during the phase-I projection.",
    )
    parser.add_argument(
        "--full-dynamics-phase-one-defect-weight",
        type=float,
        default=10.0,
        help="Weight reducing one-step complete-dynamics defects during phase I.",
    )
    parser.add_argument(
        "--full-dynamics-phase-one-substeps",
        type=int,
        default=10,
        help="RK4 substeps used by the complete-dynamics phase-I projection.",
    )
    parser.add_argument(
        "--check-wheel-periodicity",
        action="store_true",
        help="Compare the complete dynamics at q_crank and q_crank + 2*pi before solving.",
    )
    parser.add_argument(
        "--validate-integrator-maps",
        action="store_true",
        help=(
            "Compare selected shooting intervals with a high-accuracy DOP853 integration "
            "before solving."
        ),
    )
    parser.add_argument(
        "--acados-fix-controls-to-warmup",
        action="store_true",
        help="Fix every ACADOS pulse-width control to its node-wise IPOPT warmup value.",
    )
    parser.add_argument(
        "--acados-fixed-control-tolerance",
        type=float,
        default=1e-5,
        help="Half-width of fixed ACADOS control bounds in scaled control units.",
    )
    parser.add_argument(
        "--ipopt-linear-solver",
        type=str,
        default="ma57",
        help="Linear solver used by IPOPT for direct IPOPT runs and the standard warmup.",
    )
    parser.add_argument(
        "--max-consecutive-failing",
        type=int,
        default=10,
        help="Maximum number of consecutive failing MHE windows tolerated before stopping.",
    )
    parser.add_argument(
        "--codegen-tag",
        type=str,
        default=None,
        help="Optional suffix added to the generated ACADOS code folder and model name.",
    )
    parser.add_argument(
        "--acados-seed-cache-tag",
        type=str,
        default=None,
        help=(
            "Load a successful ACADOS solution with this tag before solving and overwrite "
            "the cache when the current single-shot solve succeeds."
        ),
    )
    parser.add_argument(
        "--disable-standard-ipopt-warmup",
        action="store_true",
        help="Skip the one-window IPOPT warmup with the standard Ding formulation before periodic MHE.",
    )
    parser.add_argument(
        "--disable-periodic-fes-warmup-projection",
        action="store_true",
        help="Do not project periodic Ding FES initial guesses with a local rollout before ACADOS.",
    )
    parser.add_argument(
        "--periodic-ipopt-refinement",
        action="store_true",
        help=(
            "Run a one-window IPOPT refinement on the periodic formulation "
            "before handing the initial guess to ACADOS."
        ),
    )
    parser.add_argument(
        "--disable-periodic-ipopt-refinement",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--periodic-ipopt-refinement-iterations",
        type=int,
        default=300,
        help="Maximum IPOPT iterations for the periodic warmstart refinement.",
    )
    parser.add_argument(
        "--periodic-ipopt-refinement-use-sx",
        action="store_true",
        help=(
            "Build the auxiliary periodic IPOPT refinement with SX graphs. "
            "By default it uses MX to reduce memory pressure."
        ),
    )
    parser.add_argument(
        "--periodic-ipopt-refinement-ode-solver",
        choices=("target", "rk4", "irk"),
        default="target",
        help=(
            "Integrator used by the periodic IPOPT bridge. 'target' preserves "
            "the main Bioptim transcription; 'irk' can match an ACADOS IRK map."
        ),
    )
    parser.add_argument(
        "--periodic-fes-warmup-projection-weight",
        type=float,
        default=1.0,
        help="Blend weight between the original warmup FES states and the projected periodic Ding states.",
    )
    parser.add_argument(
        "--periodic-fes-warmup-projection-mode",
        choices=(
            "calcium",
            "all",
            "all_except_force",
            "all_force_blend",
            "all_force_adaptive_blend",
        ),
        default="all",
        help=(
            "Project only the periodic calcium states, all Ding fatigue states, "
            "all states except F to preserve the multibody force trajectory, "
            "all states with a separate blend weight for F, or all states with "
            "an automatically reduced F blend weight constrained by a qdot defect limit."
        ),
    )
    parser.add_argument(
        "--periodic-fes-warmup-force-projection-weight",
        type=float,
        default=0.25,
        help=(
            "Blend weight used only for F states when "
            "--periodic-fes-warmup-projection-mode is all_force_blend or "
            "all_force_adaptive_blend."
        ),
    )
    parser.add_argument(
        "--periodic-fes-warmup-force-qdot-defect-limit",
        type=float,
        default=3.0,
        help=(
            "Maximum full-dynamics qdot defect allowed while selecting the F blend weight "
            "when --periodic-fes-warmup-projection-mode=all_force_adaptive_blend."
        ),
    )
    parser.add_argument(
        "--periodic-fes-warmup-force-adaptive-steps",
        type=int,
        default=10,
        help="Number of candidate F blend weights tested in all_force_adaptive_blend mode.",
    )
    parser.add_argument(
        "--periodic-fes-warmup-projection-strategy",
        choices=("rollout", "sequential", "least_squares"),
        default="sequential",
        help="Project FES warmup states by direct rollout, sequential proximal rollout, or bounded least squares.",
    )
    parser.add_argument(
        "--periodic-fes-warmup-projection-substeps",
        type=int,
        default=10,
        help="RK4 substeps per shooting interval used by the periodic FES warmup projection.",
    )
    parser.add_argument(
        "--periodic-fes-warmup-projection-proximity-weight",
        type=float,
        default=1.0,
        help="Projection weight that keeps projected FES states close to the IPOPT warmup.",
    )
    parser.add_argument(
        "--periodic-fes-warmup-projection-defect-weight",
        type=float,
        default=100.0,
        help="Projection weight applied to normalized periodic FES dynamic defects.",
    )
    parser.add_argument(
        "--periodic-fes-warmup-projection-trust-radius",
        type=float,
        default=None,
        help="Optional trust radius, in normalized state units, around the warmup FES states.",
    )
    parser.add_argument(
        "--periodic-fes-warmup-projection-max-iterations",
        type=int,
        default=200,
        help="Maximum number of function evaluations for the least-squares FES projection.",
    )
    parser.add_argument(
        "--disable-historical-ipopt-initial-guess",
        action="store_true",
        help="Do not reuse the historical IPOPT initial guess file from result/initial_guess when it exists.",
    )
    parser.add_argument(
        "--use-sx",
        dest="use_sx",
        action="store_true",
        help="Build the problem with CasADi SX graphs.",
    )
    parser.add_argument(
        "--no-use-sx",
        dest="use_sx",
        action="store_false",
        help="Build the problem with CasADi MX graphs.",
    )
    parser.add_argument(
        "--enforce-start-constraints",
        dest="enforce_start_constraints",
        action="store_true",
        help="Enable the historical start-of-window posture constraints.",
    )
    parser.add_argument(
        "--disable-start-constraints",
        dest="enforce_start_constraints",
        action="store_false",
        help="Disable the start-of-window posture constraints.",
    )
    parser.set_defaults(use_sx=True, enforce_start_constraints=False)
    return parser


def build_ode_solver(args: argparse.Namespace):
    if args.ode_solver == "collocation":
        return OdeSolver.COLLOCATION(
            polynomial_degree=args.collocation_degree, method=args.collocation_method
        )
    if args.ode_solver == "irk":
        return OdeSolver.IRK(
            polynomial_degree=args.collocation_degree, method=args.collocation_method
        )
    if args.ode_solver == "rk8":
        return OdeSolver.RK8(n_integration_steps=args.rk_steps)
    return OdeSolver.RK4(n_integration_steps=args.rk_steps)


def _ode_solver_suffix(ode_solver) -> str:
    if isinstance(ode_solver, OdeSolver.IRK):
        return f"irk_{ode_solver.polynomial_degree}_{ode_solver.method}"
    if isinstance(ode_solver, OdeSolver.COLLOCATION):
        return f"collocation_{ode_solver.polynomial_degree}_{ode_solver.method}"
    if isinstance(ode_solver, OdeSolver.RK8):
        return f"rk8_{ode_solver.n_integration_steps}"
    if isinstance(ode_solver, OdeSolver.RK4):
        return f"rk4_{ode_solver.n_integration_steps}"
    raise RuntimeError("ode_solver must be COLLOCATION, IRK, RK8, or RK4")


def _historical_initial_guess_path(cycles_per_window: int, ode_solver) -> Path | None:
    filename = f"{cycles_per_window}_initial_guess_{_ode_solver_suffix(ode_solver)}.pkl"
    candidates = (
        Path.cwd() / "result" / "initial_guess" / filename,
        Path(__file__).resolve().parent / "result" / "initial_guess" / filename,
    )
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def ensure_acados_environment(acados_source_dir: str | None = None) -> Path:
    acados_dir = Path(
        acados_source_dir
        or os.environ.get(
            "ACADOS_SOURCE_DIR", str(Path.home() / "Documents/bioptim/external/acados")
        )
    ).resolve()
    os.environ["ACADOS_SOURCE_DIR"] = str(acados_dir)

    acados_lib_dir = acados_dir / "lib"
    for env_name in (
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "LD_LIBRARY_PATH",
    ):
        current = os.environ.get(env_name, "")
        paths = [p for p in current.split(":") if p]
        if str(acados_lib_dir) not in paths:
            os.environ[env_name] = (
                ":".join([str(acados_lib_dir), *paths])
                if paths
                else str(acados_lib_dir)
            )

    if sys_platform == "darwin":
        _preload_acados_libraries(acados_lib_dir)

    return acados_dir


def _shared_lib_ext() -> str:
    if sys_platform == "darwin":
        return ".dylib"
    if sys_platform.startswith("win"):
        return ".dll"
    return ".so"


def _short_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def _source_stamp(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def _cache_root() -> Path:
    path = Path(__file__).resolve().parent / "result" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _warmup_cache_signature(
    args: argparse.Namespace,
    model_path: Path,
    simulation_conditions: dict,
    cycling_info: dict,
) -> str:
    payload = {
        "kind": "warmup",
        "nmpc_builder_version": 1,
        "model_path": str(model_path.resolve()),
        "cycles_per_window": args.cycles_per_window,
        "stimulations_per_cycle": args.stimulations_per_cycle,
        "objective": args.objective,
        "objective_shape": args.objective_shape,
        "constant_crank_torque": args.constant_crank_torque,
        "torque_application": args.torque_application,
        "state_scaling": args.state_scaling,
        "pulse_width_scaling": args.pulse_width_scaling,
        "ipopt_linear_solver": args.ipopt_linear_solver,
        "simulation_conditions": simulation_conditions,
        "cycling_info_keys": sorted(cycling_info.keys()),
        "sources": [
            _source_stamp(model_path),
            _source_stamp(
                (
                    Path(__file__).resolve().parents[3]
                    / "cocofest"
                    / "models"
                    / "dynamical_model.py"
                ).resolve()
            ),
            _source_stamp(
                (
                    Path(__file__).resolve().parents[3]
                    / "cocofest"
                    / "models"
                    / "ding2007"
                    / "ding2007_with_fatigue.py"
                ).resolve()
            ),
        ],
    }
    return _short_hash(payload)


def _warmup_cache_path(
    args: argparse.Namespace,
    model_path: Path,
    simulation_conditions: dict,
    cycling_info: dict,
) -> Path:
    return (
        _cache_root()
        / f"warmup_{_warmup_cache_signature(args, model_path, simulation_conditions, cycling_info)}.npz"
    )


def _periodic_ipopt_refinement_cache_path(
    args: argparse.Namespace, model_path: Path
) -> Path:
    repository_root = Path(__file__).resolve().parents[3]
    payload = {
        "kind": "periodic_ipopt_refinement",
        "cache_version": 1,
        "model_formulation": args.model_formulation,
        "cycles_per_window": args.cycles_per_window,
        "stimulations_per_cycle": args.stimulations_per_cycle,
        "objective": args.objective,
        "objective_shape": args.objective_shape,
        "constant_crank_torque": args.constant_crank_torque,
        "torque_application": args.torque_application,
        "state_scaling": args.state_scaling,
        "pulse_width_scaling": args.pulse_width_scaling,
        "standard_warmup_transfer": args.acados_standard_warmup_transfer,
        "fatigue_warmstart_mode": args.acados_fatigue_warmstart_mode,
        "use_sx": args.periodic_ipopt_refinement_use_sx,
        "ode_solver": args.periodic_ipopt_refinement_ode_solver,
        "acados_sim_stages": args.acados_sim_stages,
        "acados_sim_steps": args.acados_sim_steps,
        "acados_collocation_type": args.acados_collocation_type,
        "ipopt_linear_solver": args.ipopt_linear_solver,
        "sources": [
            _source_stamp(model_path),
            _source_stamp(
                repository_root
                / "cocofest"
                / "models"
                / "ding2007"
                / "ding2007_with_fatigue_periodic_node.py"
            ),
            _source_stamp(
                repository_root / "cocofest" / "models" / "dynamical_model.py"
            ),
        ],
    }
    return _cache_root() / f"periodic_ipopt_{_short_hash(payload)}.npz"


def _acados_seed_cache_path(args: argparse.Namespace, model_path: Path) -> Path | None:
    if not args.acados_seed_cache_tag:
        return None
    safe_tag = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in args.acados_seed_cache_tag
    )
    payload = {
        "kind": "acados_seed",
        "cache_version": 2,
        "model_formulation": args.model_formulation,
        "cycles_per_window": args.cycles_per_window,
        "stimulations_per_cycle": args.stimulations_per_cycle,
        "objective": args.objective,
        "objective_shape": args.objective_shape,
        "constant_crank_torque": args.constant_crank_torque,
        "torque_application": args.torque_application,
        "state_scaling": args.state_scaling,
        "pulse_width_scaling": args.pulse_width_scaling,
        "integrator_type": args.acados_integrator_type,
        "sim_stages": args.acados_sim_stages,
        "sim_steps": args.acados_sim_steps,
        "newton_iter": args.acados_newton_iter,
        "model_source": _source_stamp(
            Path(__file__).resolve().parents[3]
            / "cocofest"
            / "models"
            / "ding2007"
            / "ding2007_with_fatigue_periodic_node.py"
        ),
        "model_path": _source_stamp(model_path),
    }
    return _cache_root() / f"acados_seed_{safe_tag}_{_short_hash(payload)}.npz"


def _save_warmup_cache(cache_path: Path, solution) -> None:
    states = solution.decision_states(to_merge=SolutionMerge.NODES)
    controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
    payload = {}
    for key, values in states.items():
        payload[f"states__{key}"] = np.asarray(values)
    for key, values in controls.items():
        payload[f"controls__{key}"] = np.asarray(values)
    np.savez(cache_path, **payload)


def _load_warmup_cache(cache_path: Path) -> "_WarmupSolutionAdapter":
    data = np.load(cache_path, allow_pickle=False)
    states = {
        key.split("__", 1)[1]: data[key]
        for key in data.files
        if key.startswith("states__")
    }
    controls = {
        key.split("__", 1)[1]: data[key]
        for key in data.files
        if key.startswith("controls__")
    }
    return _WarmupSolutionAdapter(states, controls)


def _continuation_cache_signature(args: argparse.Namespace) -> str:
    repository_root = Path(__file__).resolve().parents[3]
    payload = {
        "kind": "acados_one_cycle_continuation",
        "cache_version": 2,
        "nmpc_builder_version": 1,
        "model_formulation": args.model_formulation,
        "objective": args.objective,
        "objective_shape": args.objective_shape,
        "stimulations_per_cycle": args.stimulations_per_cycle,
        "constant_crank_torque": args.constant_crank_torque,
        "torque_application": args.torque_application,
        "state_scaling": args.state_scaling,
        "pulse_width_scaling": args.pulse_width_scaling,
        "control_regularization_weight": args.control_regularization_weight,
        "control_regularization_target": args.control_regularization_target,
        "wheel_qdot_regularization_weight": args.wheel_qdot_regularization_weight,
        "wheel_qdot_regularization_target": args.wheel_qdot_regularization_target,
        "integrator_type": args.acados_integrator_type,
        "collocation_type": args.acados_collocation_type,
        "sim_stages": args.acados_sim_stages,
        "sim_steps": args.acados_sim_steps,
        "newton_iter": args.acados_newton_iter,
        "hessian_approx": args.acados_hessian_approx,
        "nlp_solver_type": args.acados_nlp_solver_type,
        "qp_solver": args.acados_qp_solver,
        "max_iterations": args.acados_continuation_source_max_iterations,
        "source_convergence_tolerance": args.acados_tolerance,
        "source_stationarity_tolerance": args.acados_stationarity_tolerance,
        "standard_warmup_transfer": args.acados_standard_warmup_transfer,
        "fatigue_warmstart_mode": args.acados_fatigue_warmstart_mode,
        "periodic_projection": not args.disable_periodic_fes_warmup_projection,
        "periodic_projection_mode": args.periodic_fes_warmup_projection_mode,
        "periodic_projection_strategy": args.periodic_fes_warmup_projection_strategy,
        "periodic_projection_weight": args.periodic_fes_warmup_projection_weight,
        "periodic_ipopt_refinement": (
            args.periodic_ipopt_refinement
            and not args.disable_periodic_ipopt_refinement
        ),
        "pulse_width_trust_radius": args.acados_pulse_width_trust_radius,
        "fes_state_trust_radius": args.acados_fes_state_trust_radius,
        "sources": [
            _source_stamp(
                repository_root
                / "cocofest"
                / "models"
                / "ding2007"
                / (
                    "ding2007_with_fatigue_periodic_node.py"
                    if args.model_formulation == "periodic_node"
                    else "ding2007_with_fatigue_periodic.py"
                )
            ),
            _source_stamp(
                repository_root / "cocofest" / "models" / "dynamical_model.py"
            ),
            _source_stamp(
                repository_root
                / "examples"
                / "msk_models"
                / "Wu"
                / "Modified_Wu_Shoulder_Model_Cycling.bioMod"
            ),
        ],
    }
    return _short_hash(payload)


def _continuation_cache_path(args: argparse.Namespace) -> Path:
    return _cache_root() / (
        f"acados_one_cycle_{_continuation_cache_signature(args)}.npz"
    )


def _horizon_seed_cache_signature(args: argparse.Namespace) -> str:
    repository_root = Path(__file__).resolve().parents[3]
    payload = {
        "kind": "acados_horizon_seed",
        "cache_version": 3,
        "nmpc_builder_version": 1,
        "model_formulation": args.model_formulation,
        "cycles_per_window": args.cycles_per_window,
        "stimulations_per_cycle": args.stimulations_per_cycle,
        "objective": args.objective,
        "objective_shape": args.objective_shape,
        "constant_crank_torque": args.constant_crank_torque,
        "torque_application": args.torque_application,
        "ode_solver": args.ode_solver,
        "rk_steps": args.rk_steps,
        "enforce_start_constraints": args.enforce_start_constraints,
        "state_scaling": args.state_scaling,
        "pulse_width_scaling": args.pulse_width_scaling,
        "control_regularization_weight": args.control_regularization_weight,
        "control_regularization_target": args.control_regularization_target,
        "control_regularization_target_source": args.control_regularization_target_source,
        "wheel_qdot_regularization_weight": args.wheel_qdot_regularization_weight,
        "wheel_qdot_regularization_target": args.wheel_qdot_regularization_target,
        "pulse_width_trust_radius": args.acados_pulse_width_trust_radius,
        "fes_state_trust_radius": args.acados_fes_state_trust_radius,
        "wheel_q_slack": args.acados_wheel_q_slack,
        "wheel_qdot_slack": args.acados_wheel_qdot_slack,
        "wheel_q_path_margin": args.acados_wheel_q_path_margin,
        "project_qdot_from_q": args.acados_project_qdot_from_q,
        "sources": [
            _source_stamp(
                repository_root
                / "cocofest"
                / "models"
                / "ding2007"
                / "ding2007_with_fatigue_periodic.py"
            ),
            _source_stamp(
                repository_root / "cocofest" / "models" / "dynamical_model.py"
            ),
            _source_stamp(
                repository_root
                / "examples"
                / "msk_models"
                / "Wu"
                / "Modified_Wu_Shoulder_Model_Cycling.bioMod"
            ),
        ],
    }
    return _short_hash(payload)


def _horizon_seed_cache_path(args: argparse.Namespace) -> Path:
    return _cache_root() / (
        f"acados_{args.cycles_per_window}_cycle_seed_"
        f"{_horizon_seed_cache_signature(args)}.npz"
    )


def _codegen_signature(args: argparse.Namespace) -> str:
    repository_root = Path(__file__).resolve().parents[3]
    payload = {
        # Increment when solve_case changes the generated OCP structure in a way that is
        # not represented by the arguments or the model sources below.
        "problem_builder_version": 1,
        "nmpc_builder_version": 1,
        "solver": args.solver,
        "model_formulation": args.model_formulation,
        "torque_application": args.torque_application,
        "cycles_per_window": args.cycles_per_window,
        "stimulations_per_cycle": args.stimulations_per_cycle,
        "ode_solver": args.ode_solver,
        "rk_steps": args.rk_steps,
        "collocation_degree": args.collocation_degree,
        "collocation_method": args.collocation_method,
        "objective": args.objective,
        "objective_shape": args.objective_shape,
        "constant_crank_torque": args.constant_crank_torque,
        "use_sx": args.use_sx,
        "enforce_start_constraints": args.enforce_start_constraints,
        "control_regularization_weight": args.control_regularization_weight,
        "control_regularization_target": args.control_regularization_target,
        "control_regularization_target_source": args.control_regularization_target_source,
        "wheel_qdot_regularization_weight": args.wheel_qdot_regularization_weight,
        "wheel_qdot_regularization_target": args.wheel_qdot_regularization_target,
        "state_scaling": args.state_scaling,
        "pulse_width_scaling": args.pulse_width_scaling,
        "acados_pulse_width_trust_radius": args.acados_pulse_width_trust_radius,
        "max_acados_iterations": args.max_acados_iterations,
        "acados_tolerance": args.acados_tolerance,
        "acados_stationarity_tolerance": args.acados_stationarity_tolerance,
        "acados_first_window_tolerance": args.acados_first_window_tolerance,
        "acados_first_window_stationarity_tolerance": (
            args.acados_first_window_stationarity_tolerance
        ),
        "acados_qp_solver": args.acados_qp_solver,
        "acados_integrator_type": args.acados_integrator_type,
        "acados_collocation_type": args.acados_collocation_type,
        "acados_sim_stages": args.acados_sim_stages,
        "acados_sim_steps": args.acados_sim_steps,
        "acados_newton_iter": args.acados_newton_iter,
        "acados_newton_tol": args.acados_newton_tol,
        "acados_jac_reuse": args.acados_jac_reuse,
        "acados_hessian_approx": args.acados_hessian_approx,
        "acados_nlp_solver_type": args.acados_nlp_solver_type,
        "acados_search_direction_mode": args.acados_search_direction_mode,
        "acados_use_constraint_hessian_in_feas_qp": (
            args.acados_use_constraint_hessian_in_feas_qp
        ),
        "acados_disable_direction_mode_switch_to_nominal": (
            args.acados_disable_direction_mode_switch_to_nominal
        ),
        "acados_regularize_method": args.acados_regularize_method,
        "acados_levenberg_marquardt": args.acados_levenberg_marquardt,
        "acados_globalization": args.acados_globalization,
        "acados_fixed_step_length": args.acados_fixed_step_length,
        "acados_nlp_qp_tol_strategy": args.acados_nlp_qp_tol_strategy,
        "acados_qp_iter_max": args.acados_qp_iter_max,
        "acados_qp_warm_start_level": args.acados_qp_warm_start_level,
        "acados_warm_start_first_qp": args.acados_warm_start_first_qp,
        "acados_warm_start_first_qp_from_nlp": (
            args.acados_warm_start_first_qp_from_nlp
        ),
        "acados_qpscaling_scale_objective": args.acados_qpscaling_scale_objective,
        "acados_qpscaling_scale_constraints": args.acados_qpscaling_scale_constraints,
        "acados_ext_qp_res": args.acados_ext_qp_res,
        "acados_print_level": args.acados_print_level,
        "sources": [
            _source_stamp(
                (
                    Path(__file__).resolve().parents[3]
                    / "cocofest"
                    / "models"
                    / "ding2007"
                    / "ding2007_with_fatigue_periodic.py"
                ).resolve()
            ),
            _source_stamp(
                repository_root / "cocofest" / "models" / "dynamical_model.py"
            ),
            _source_stamp(
                repository_root
                / "examples"
                / "msk_models"
                / "Wu"
                / "Modified_Wu_Shoulder_Model_Cycling.bioMod"
            ),
        ],
    }
    return _short_hash(payload)


def _preload_acados_libraries(acados_lib_dir: Path) -> None:
    mode = ctypes.RTLD_GLOBAL
    if hasattr(os, "RTLD_NOW"):
        mode |= os.RTLD_NOW

    library_order = (
        "libqpOASES_e.dylib",
        "libhpipm.dylib",
        "libblasfeo.dylib",
        "libacados.dylib",
    )
    for library_name in library_order:
        library_path = acados_lib_dir / library_name
        if library_path.exists():
            ctypes.CDLL(str(library_path), mode=mode)


def set_acados_unsafe_option(solver: Solver.ACADOS, value, name: str) -> None:
    # Bioptim 3.4 registers unsafe options on the class without initializing later instances.
    solver.set_option_unsafe(value, name)
    setattr(solver, f"_{name}", value)


def configure_acados_solver(
    model_name: str,
    generated_code_path: str,
    max_iterations: int,
    convergence_tolerance: float | None,
    stationarity_tolerance: float | None,
    qp_solver: str,
    integrator_type: str,
    collocation_type: str,
    sim_method_num_stages: int,
    sim_method_num_steps: int,
    sim_method_newton_iter: int,
    sim_method_newton_tol: float | None,
    sim_method_jac_reuse: int,
    hessian_approx: str,
    nlp_solver_type: str,
    search_direction_mode: str,
    use_constraint_hessian_in_feas_qp: bool,
    allow_direction_mode_switch_to_nominal: bool,
    regularize_method: str,
    levenberg_marquardt: float,
    globalization: str,
    fixed_step_length: float,
    nlp_qp_tol_strategy: str,
    qp_iter_max: int,
    qp_warm_start_level: int,
    warm_start_first_qp: bool,
    warm_start_first_qp_from_nlp: bool,
    qpscaling_scale_objective: str,
    qpscaling_scale_constraints: str,
    ext_qp_res: bool,
    print_level: int = 0,
) -> Solver.ACADOS:
    solver = Solver.ACADOS()
    solver.set_acados_dir(str(ensure_acados_environment()))
    solver.set_c_generated_code_path(generated_code_path)
    solver.set_acados_model_name(model_name)
    shared_lib_path = (
        Path(generated_code_path)
        / f"libacados_ocp_solver_{model_name}{_shared_lib_ext()}"
    )
    solver.set_c_compile(not shared_lib_path.exists())
    solver.set_qp_solver(qp_solver)
    solver.set_integrator_type(integrator_type)
    solver.set_nlp_solver_type(nlp_solver_type)
    solver.set_hessian_approx(hessian_approx)
    solver.set_sim_method_num_stages(sim_method_num_stages)
    solver.set_sim_method_num_steps(sim_method_num_steps)
    solver.set_sim_method_newton_iter(sim_method_newton_iter)
    solver.set_maximum_iterations(max_iterations)
    if convergence_tolerance is not None:
        solver.set_convergence_tolerance(convergence_tolerance)
    if stationarity_tolerance is not None:
        solver.set_nlp_solver_tol_stat(stationarity_tolerance)
    solver.set_print_level(print_level)
    set_acados_unsafe_option(solver, collocation_type, "collocation_type")
    set_acados_unsafe_option(solver, sim_method_jac_reuse, "sim_method_jac_reuse")
    set_acados_unsafe_option(solver, search_direction_mode, "search_direction_mode")
    set_acados_unsafe_option(
        solver,
        use_constraint_hessian_in_feas_qp,
        "use_constraint_hessian_in_feas_qp",
    )
    set_acados_unsafe_option(
        solver,
        allow_direction_mode_switch_to_nominal,
        "allow_direction_mode_switch_to_nominal",
    )
    if sim_method_newton_tol is not None:
        set_acados_unsafe_option(
            solver, float(sim_method_newton_tol), "sim_method_newton_tol"
        )
    # Favor numerical robustness over raw speed for this periodic MHE.
    set_acados_unsafe_option(solver, globalization, "globalization")
    set_acados_unsafe_option(
        solver, fixed_step_length, "globalization_fixed_step_length"
    )
    set_acados_unsafe_option(
        solver, 1, "globalization_line_search_use_sufficient_descent"
    )
    set_acados_unsafe_option(solver, 0, "globalization_use_SOC")
    set_acados_unsafe_option(solver, "ROBUST", "hpipm_mode")
    set_acados_unsafe_option(solver, regularize_method, "regularize_method")
    set_acados_unsafe_option(solver, levenberg_marquardt, "levenberg_marquardt")
    set_acados_unsafe_option(
        solver, qpscaling_scale_objective, "qpscaling_scale_objective"
    )
    set_acados_unsafe_option(
        solver, qpscaling_scale_constraints, "qpscaling_scale_constraints"
    )
    set_acados_unsafe_option(solver, nlp_qp_tol_strategy, "nlp_qp_tol_strategy")
    set_acados_unsafe_option(solver, qp_iter_max, "qp_solver_iter_max")
    set_acados_unsafe_option(solver, 1 if ext_qp_res else 0, "nlp_solver_ext_qp_res")
    set_acados_unsafe_option(solver, qp_warm_start_level, "qp_solver_warm_start")
    set_acados_unsafe_option(solver, 0, "qp_solver_ric_alg")
    set_acados_unsafe_option(solver, 0, "qp_solver_cond_ric_alg")
    set_acados_unsafe_option(
        solver, warm_start_first_qp, "nlp_solver_warm_start_first_qp"
    )
    set_acados_unsafe_option(
        solver,
        warm_start_first_qp_from_nlp,
        "nlp_solver_warm_start_first_qp_from_nlp",
    )
    return solver


def configure_ipopt_solver(
    max_iterations: int, linear_solver: str = "ma57"
) -> Solver.IPOPT:
    solver = Solver.IPOPT(
        show_online_optim=False,
        _max_iter=max_iterations,
        show_options=dict(show_bounds=True),
    )
    solver.set_warm_start_init_point("yes")
    solver.set_mu_init(1e-2)
    solver.set_tol(1e-6)
    solver.set_dual_inf_tol(1e-6)
    solver.set_constr_viol_tol(1e-6)
    solver.set_linear_solver(linear_solver)
    return solver


def _split_receding_solution(sol) -> tuple:
    merged_solution = sol[0]
    source_window_solutions = []
    exported_cycle_solutions = []

    if len(sol) == 3:
        source_window_solutions = sol[1]
        exported_cycle_solutions = sol[2]
    elif len(sol) == 2:
        exported_cycle_solutions = sol[1]

    return merged_solution, source_window_solutions, exported_cycle_solutions


def _wheel_trace_from_exported_cycles(
    merged_solution, exported_cycle_solutions: list
) -> np.ndarray:
    if not exported_cycle_solutions:
        return merged_solution.decision_states(to_merge=SolutionMerge.NODES)["q"][2, :]

    cycle_traces = [
        cycle_solution.decision_states(to_merge=SolutionMerge.NODES)["q"][2, :]
        for cycle_solution in exported_cycle_solutions
    ]
    return np.concatenate(
        [trace[:-1] for trace in cycle_traces[:-1]] + [cycle_traces[-1]]
    )


def _control_traces_from_exported_cycles(
    merged_solution, exported_cycle_solutions: list
) -> dict[str, np.ndarray]:
    if not exported_cycle_solutions:
        controls = merged_solution.decision_controls(to_merge=SolutionMerge.NODES)
        return {key: np.asarray(values) for key, values in controls.items()}

    control_traces = {}
    reference_controls = exported_cycle_solutions[0].decision_controls(
        to_merge=SolutionMerge.NODES
    )
    for key in reference_controls.keys():
        cycle_values = []
        for cycle_solution in exported_cycle_solutions:
            controls = cycle_solution.decision_controls(to_merge=SolutionMerge.NODES)
            values = np.asarray(controls[key])
            if values.ndim == 1:
                values = values[np.newaxis, :]
            cycle_values.append(values)
        control_traces[key] = np.concatenate(cycle_values, axis=1)

    return control_traces


def _state_traces_from_exported_cycles(
    merged_solution, exported_cycle_solutions: list
) -> dict[str, np.ndarray]:
    if not exported_cycle_solutions:
        states = merged_solution.decision_states(to_merge=SolutionMerge.NODES)
        return {key: np.asarray(values) for key, values in states.items()}

    state_traces = {}
    reference_states = exported_cycle_solutions[0].decision_states(
        to_merge=SolutionMerge.NODES
    )
    for key in reference_states.keys():
        cycle_values = []
        for cycle_solution in exported_cycle_solutions:
            states = cycle_solution.decision_states(to_merge=SolutionMerge.NODES)
            values = np.asarray(states[key])
            if values.ndim == 1:
                values = values[np.newaxis, :]
            cycle_values.append(values)
        state_traces[key] = np.concatenate(
            [values[:, :-1] for values in cycle_values[:-1]] + [cycle_values[-1]],
            axis=1,
        )

    return state_traces


def _initial_guess_traces(container) -> dict[str, np.ndarray]:
    return {
        key: np.array(container[key].init, dtype=float, copy=True)
        for key in container.keys()
    }


def _control_bounds_summary(nmpc) -> dict[str, dict[str, float]]:
    summary = {}
    original_bounds = getattr(nmpc, "_cocofest_original_control_bounds", {})
    for key in nmpc.nlp[0].controls.keys():
        bounds = original_bounds.get(key, nmpc.nlp[0].u_bounds[key])
        if isinstance(bounds, tuple):
            lower, upper = bounds
        else:
            lower, upper = bounds.min, bounds.max
        summary[key] = {
            "lower": float(np.min(np.asarray(lower, dtype=float))),
            "upper": float(np.max(np.asarray(upper, dtype=float))),
        }
    return summary


def _status_is_success(status) -> bool:
    return status == 0


def _status_label(status) -> str:
    if status is None:
        return "None"
    return ACADOS_STATUS_NAMES.get(status, str(status))


def _array_finite_summary(values) -> dict:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    finite_values = array[finite]
    return {
        "shape": array.shape,
        "finite": bool(np.all(finite)),
        "nonfinite_count": int(array.size - np.count_nonzero(finite)),
        "min": float(np.min(finite_values)) if finite_values.size else float("nan"),
        "max": float(np.max(finite_values)) if finite_values.size else float("nan"),
    }


def _dict_finite_summary(values_by_key: dict) -> dict:
    summary = {}
    for key, values in values_by_key.items():
        item = _array_finite_summary(values)
        if not item["finite"]:
            summary[key] = item
    return summary


def _get_acados_template_solver(solution):
    ocp = getattr(solution, "ocp", None)
    interface = getattr(ocp, "ocp_solver", None)
    return getattr(interface, "ocp_solver", None)


def _safe_acados_stat(acados_solver, field: str):
    try:
        return acados_solver.get_stats(field)
    except Exception as exc:  # noqa: BLE001 - diagnostics should not mask a solve.
        return {"error": str(exc)}


def _safe_acados_stage_field(acados_solver, stage: int, field: str):
    try:
        return acados_solver.get(stage, field)
    except Exception as exc:  # noqa: BLE001 - diagnostics should not mask a solve.
        return {"error": str(exc)}


def _variable_row_labels(values_by_key: dict) -> list[str]:
    labels = []
    for key, values in values_by_key.items():
        array = np.asarray(values)
        row_count = 1 if array.ndim <= 1 else array.shape[0]
        labels.extend(
            [key]
            if row_count == 1
            else [f"{key}[{index}]" for index in range(row_count)]
        )
    return labels


def _acados_bound_complementarity_rows(
    acados_solver,
    n_stages: int,
    state_labels: list[str] | None = None,
    control_labels: list[str] | None = None,
    limit: int = 8,
) -> list[dict]:
    """Locate the box bounds that dominate Acados' complementarity residual."""
    rows = []
    state_labels = state_labels or []
    control_labels = control_labels or []
    for stage in range(n_stages):
        x = _safe_acados_stage_field(acados_solver, stage, "x")
        u = _safe_acados_stage_field(acados_solver, stage, "u")
        lam = _safe_acados_stage_field(acados_solver, stage, "lam")
        if any(isinstance(values, dict) for values in (x, u, lam)):
            continue

        try:
            lower_u = np.asarray(
                acados_solver.constraints_get(stage, "lbu"), dtype=float
            ).reshape(-1)
            upper_u = np.asarray(
                acados_solver.constraints_get(stage, "ubu"), dtype=float
            ).reshape(-1)
            lower_x = np.asarray(
                acados_solver.constraints_get(stage, "lbx"), dtype=float
            ).reshape(-1)
            upper_x = np.asarray(
                acados_solver.constraints_get(stage, "ubx"), dtype=float
            ).reshape(-1)
        except Exception:  # noqa: BLE001 - diagnostics should not mask a solve.
            continue

        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        lam = np.asarray(lam, dtype=float).reshape(-1)
        one_sided_count = lam.size // 2
        bound_count = lower_u.size + lower_x.size
        if lam.size % 2 or one_sided_count < bound_count:
            continue

        blocks = (
            ("u", u, lower_u, upper_u, control_labels, 0),
            ("x", x, lower_x, upper_x, state_labels, lower_u.size),
        )
        for block_name, values, lower, upper, labels, offset in blocks:
            count = min(values.size, lower.size, upper.size)
            for index in range(count):
                variable = (
                    labels[index] if index < len(labels) else f"{block_name}[{index}]"
                )
                for side, bound, distance, multiplier_index in (
                    (
                        "lower",
                        lower[index],
                        values[index] - lower[index],
                        offset + index,
                    ),
                    (
                        "upper",
                        upper[index],
                        upper[index] - values[index],
                        one_sided_count + offset + index,
                    ),
                ):
                    multiplier = lam[multiplier_index]
                    rows.append(
                        {
                            "stage": stage,
                            "variable": variable,
                            "side": side,
                            "value": float(values[index]),
                            "bound": float(bound),
                            "distance": float(distance),
                            "multiplier": float(multiplier),
                            "product": float(abs(multiplier * distance)),
                        }
                    )

    rows.sort(key=lambda row: row["product"], reverse=True)
    return rows[:limit]


def collect_acados_diagnostics(solution) -> dict:
    diagnostics = {
        "status": solution.status,
        "status_label": _status_label(solution.status),
        "state_nonfinite": {},
        "control_nonfinite": {},
        "solver_available": False,
    }

    states = {}
    try:
        states = solution.decision_states(to_merge=SolutionMerge.NODES)
        diagnostics["state_nonfinite"] = _dict_finite_summary(states)
    except Exception as exc:  # noqa: BLE001 - diagnostics should not mask a solve.
        diagnostics["state_error"] = str(exc)

    controls = {}
    try:
        controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
        diagnostics["control_nonfinite"] = _dict_finite_summary(controls)
    except Exception as exc:  # noqa: BLE001 - diagnostics should not mask a solve.
        diagnostics["control_error"] = str(exc)

    acados_solver = _get_acados_template_solver(solution)
    if acados_solver is None:
        return diagnostics

    diagnostics["solver_available"] = True
    first_stage_parameters = _safe_acados_stage_field(acados_solver, 0, "p")
    if not isinstance(first_stage_parameters, dict):
        diagnostics["first_stage_parameters"] = np.asarray(first_stage_parameters)
    for field in (
        "sqp_iter",
        "nlp_iter",
        "qp_iter",
        "qp_stat",
        "alpha",
        "residuals",
        "res_stat_all",
        "res_eq_all",
        "res_ineq_all",
        "res_comp_all",
        "qpscaling_status",
        "time_tot",
        "time_qp",
        "time_qp_solver_call",
        "time_sim",
        "time_glob",
        "time_reg",
        "time_qpscaling",
    ):
        diagnostics[field] = _safe_acados_stat(acados_solver, field)

    stats = _safe_acados_stat(acados_solver, "statistics")
    diagnostics["statistics"] = stats
    if not isinstance(stats, dict):
        stats_array = np.asarray(stats, dtype=float)
        diagnostics["statistics_finite"] = _array_finite_summary(stats_array)
        if stats_array.size and stats_array.ndim == 2:
            diagnostics["statistics_last_column"] = stats_array[:, -1]

    stage_nonfinite = []
    stage = 0
    while True:
        stage_items = {}
        any_available = False
        for field in ("x", "u", "p", "pi", "lam"):
            values = _safe_acados_stage_field(acados_solver, stage, field)
            if isinstance(values, dict):
                continue
            any_available = True
            item = _array_finite_summary(values)
            if not item["finite"]:
                stage_items[field] = item
        if stage_items:
            stage_nonfinite.append({"stage": stage, "fields": stage_items})
        if not any_available:
            break
        stage += 1
        if stage > 10000:
            diagnostics["stage_scan_error"] = "Stopped after 10000 stages."
            break

    diagnostics["stage_nonfinite"] = stage_nonfinite
    diagnostics["n_stages_scanned"] = stage
    diagnostics["bound_complementarity_top"] = _acados_bound_complementarity_rows(
        acados_solver,
        n_stages=stage,
        state_labels=_variable_row_labels(states),
        control_labels=_variable_row_labels(controls),
    )
    residuals = diagnostics.get("residuals")
    if not isinstance(residuals, dict) and residuals is not None:
        values = np.asarray(residuals, dtype=float).reshape(-1)
        if values.size >= 4:
            diagnostics["named_residuals"] = dict(
                zip(
                    ("stationarity", "dynamics", "inequality", "complementarity"),
                    values[:4],
                )
            )
    return diagnostics


def snapshot_acados_diagnostics(solution) -> dict:
    # Every receding-horizon Solution points to the same mutable Acados solver.
    return deepcopy(collect_acados_diagnostics(solution))


def acados_diagnostics_meet_tolerances(
    diagnostics: dict,
    convergence_tolerance: float | None,
    stationarity_tolerance: float | None,
) -> bool:
    residuals = diagnostics.get("residuals")
    if residuals is None:
        return False
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    if residuals.size < 4 or not np.all(np.isfinite(residuals[:4])):
        return False

    thresholds = (
        (
            stationarity_tolerance
            if stationarity_tolerance is not None
            else convergence_tolerance
        ),
        convergence_tolerance,
        convergence_tolerance,
        convergence_tolerance,
    )
    return all(
        threshold is None or abs(residual) <= threshold
        for residual, threshold in zip(residuals[:4], thresholds)
    )


def _format_array(values) -> str:
    if values is None:
        return "None"
    if isinstance(values, dict):
        return f"error={values.get('error')}"
    array = np.asarray(values, dtype=float).squeeze()
    return np.array2string(array, precision=3, suppress_small=False)


def _format_named_values(values: dict[str, float]) -> str:
    if not values:
        return "None"
    return " ".join(f"{key}={value:.6g}" for key, value in values.items())


def _format_compact_named_values(values: dict[str, float]) -> str:
    if not values:
        return "None"
    return ",".join(f"{key}:{value:.6g}" for key, value in values.items())


def print_acados_diagnostics(label: str, diagnostics: dict) -> None:
    print(
        f"{label} acados_status={diagnostics.get('status')} "
        f"({diagnostics.get('status_label')})"
    )
    print(f"{label} solver_available={diagnostics.get('solver_available')}")
    if diagnostics.get("state_nonfinite"):
        print(f"{label} state_nonfinite={diagnostics['state_nonfinite']}")
    if diagnostics.get("control_nonfinite"):
        print(f"{label} control_nonfinite={diagnostics['control_nonfinite']}")
    if diagnostics.get("stage_nonfinite"):
        print(f"{label} stage_nonfinite={diagnostics['stage_nonfinite'][:5]}")
    if "first_stage_parameters" in diagnostics:
        print(
            f"{label} first_stage_parameters="
            f"{_format_array(diagnostics['first_stage_parameters'])}"
        )
    print(f"{label} residuals={_format_array(diagnostics.get('residuals'))}")
    if diagnostics.get("named_residuals"):
        print(
            f"{label} residuals_named="
            f"{_format_compact_named_values(diagnostics['named_residuals'])}"
        )
    for row in diagnostics.get("bound_complementarity_top", []):
        print(
            f"{label} bound_complementarity "
            f"stage={row['stage']} variable={row['variable']} side={row['side']} "
            f"product={row['product']:.6g} multiplier={row['multiplier']:.6g} "
            f"distance={row['distance']:.6g} value={row['value']:.6g} "
            f"bound={row['bound']:.6g}"
        )
    residual_history = []
    for key in ("res_stat_all", "res_eq_all", "res_ineq_all", "res_comp_all"):
        values = diagnostics.get(key)
        if isinstance(values, dict) or values is None:
            residual_history = []
            break
        residual_history.append(np.asarray(values, dtype=float).reshape(-1))
    if residual_history and all(values.size for values in residual_history):
        common_size = min(values.size for values in residual_history)
        history = np.vstack([values[:common_size] for values in residual_history])
        print(
            f"{label} residual_history_initial={_format_array(history[:, 0])} "
            f"best={_format_array(np.min(np.abs(history), axis=1))} "
            f"final={_format_array(history[:, -1])}"
        )
    print(f"{label} sqp_iter={_format_array(diagnostics.get('sqp_iter'))}")
    print(f"{label} qp_iter={_format_array(diagnostics.get('qp_iter'))}")
    print(f"{label} qp_stat={_format_array(diagnostics.get('qp_stat'))}")
    print(f"{label} alpha={_format_array(diagnostics.get('alpha'))}")
    print(
        f"{label} qpscaling_status="
        f"{_format_array(diagnostics.get('qpscaling_status'))}"
    )
    if "statistics_finite" in diagnostics:
        print(f"{label} statistics_finite={diagnostics['statistics_finite']}")
    if "statistics_last_column" in diagnostics:
        print(
            f"{label} statistics_last_column="
            f"{_format_array(diagnostics['statistics_last_column'])}"
        )


def _trajectory_bounds_for_guess(bounds, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty((bounds.min.shape[0], n_nodes))
    upper = np.empty((bounds.max.shape[0], n_nodes))
    for node in range(n_nodes):
        column = 0 if node == 0 else 2 if node == n_nodes - 1 else 1
        lower[:, node] = bounds.min[:, column]
        upper[:, node] = bounds.max[:, column]
    return lower, upper


def print_initial_guess_diagnostics(nmpc) -> None:
    states = {
        key: np.asarray(nmpc.nlp[0].x_init[key].init, dtype=float)
        for key in nmpc.nlp[0].x_init.keys()
    }
    controls = {
        key: np.asarray(nmpc.nlp[0].u_init[key].init, dtype=float)
        for key in nmpc.nlp[0].u_init.keys()
    }

    if "q" in states and "qdot" in states:
        dt = nmpc.cycle_duration / nmpc.cycle_len
        q = states["q"]
        qdot = states["qdot"]
        qdot_from_q = np.diff(q, axis=1) / dt
        q_kinematic_defect = qdot_from_q - qdot[:, :-1]
        per_dof = np.max(np.abs(q_kinematic_defect), axis=1)
        print(
            "initial_guess_q_kinematic_defect_max: "
            f"{_format_array(np.max(np.abs(q_kinematic_defect)))}"
        )
        print("initial_guess_q_kinematic_defect_per_dof: " f"{_format_array(per_dof)}")

    state_violations = {}
    for key, values in states.items():
        lower, upper = _trajectory_bounds_for_guess(
            nmpc.nlp[0].x_bounds[key], values.shape[1]
        )
        violation = np.maximum(lower - values, 0.0) + np.maximum(values - upper, 0.0)
        max_violation = float(np.max(violation)) if violation.size else 0.0
        if max_violation > 1e-9:
            state_violations[key] = max_violation

    control_violations = {}
    for key, values in controls.items():
        bounds = nmpc.nlp[0].u_bounds[key]
        lower = bounds.min[:, [0]]
        upper = bounds.max[:, [0]]
        violation = np.maximum(lower - values, 0.0) + np.maximum(values - upper, 0.0)
        max_violation = float(np.max(violation)) if violation.size else 0.0
        if max_violation > 1e-12:
            control_violations[key] = max_violation

    print(
        "initial_guess_state_bound_violations: "
        f"{state_violations if state_violations else 'None'}"
    )
    print(
        "initial_guess_control_bound_violations: "
        f"{control_violations if control_violations else 'None'}"
    )

    fes_defects = _periodic_fes_rollout_defect_details(nmpc)
    if fes_defects:
        print(
            "initial_guess_periodic_fes_defect_by_state: "
            f"{_format_named_values(fes_defects['absolute_by_state'])}"
        )
        print(
            "initial_guess_periodic_fes_defect_by_muscle: "
            f"{_format_named_values(fes_defects['absolute_by_muscle'])}"
        )
        print(
            "initial_guess_periodic_fes_scaled_defect_by_state: "
            f"{_format_named_values(fes_defects['scaled_by_state'])}"
        )

    full_defects = _full_dynamics_rollout_defect_details(nmpc)
    if full_defects:
        print(
            "initial_guess_full_rk4_defect_by_block: "
            f"{_format_named_values(full_defects['absolute_by_block'])}"
        )
        print(
            "initial_guess_full_rk4_scaled_defect_by_block: "
            f"{_format_named_values(full_defects['scaled_by_block'])}"
        )
        print(
            "initial_guess_full_rk4_defect_top_keys: "
            f"{_format_named_values(full_defects['top_keys'])}"
        )
        if full_defects["q_by_dof"]:
            print(
                "initial_guess_full_rk4_defect_q_by_dof: "
                f"{_format_named_values(full_defects['q_by_dof'])}"
            )
        if full_defects["qdot_by_dof"]:
            print(
                "initial_guess_full_rk4_defect_qdot_by_dof: "
                f"{_format_named_values(full_defects['qdot_by_dof'])}"
            )
        for item in full_defects.get("worst_qdot_nodes", []):
            q_values = ""
            if "q_current" in item:
                q_values = (
                    f" q_current={item['q_current']:.6g}"
                    f" q_next={item['q_next']:.6g}"
                    f" q_predicted_next={item['q_predicted_next']:.6g}"
                )
            print(
                "initial_guess_full_rk4_worst_qdot_node: "
                f"dof={item['dof']} node={item['node']} time={item['time']:.6g} "
                f"defect={item['defect']:.6g} "
                f"qdot_current={item['state_current']:.6g} "
                f"qdot_next={item['state_next']:.6g} "
                f"qdot_predicted_next={item['predicted_next']:.6g} "
                f"qddot_rhs={item['rhs']:.6g}"
                f"{q_values} "
                f"forces={_format_compact_named_values(item['forces'])} "
                f"pulse_widths={_format_compact_named_values(item['controls'])}"
            )


def project_qdot_initial_guess_from_q(nmpc) -> None:
    if "q" not in nmpc.nlp[0].x_init.keys() or "qdot" not in nmpc.nlp[0].x_init.keys():
        return

    dt = nmpc.cycle_duration / nmpc.cycle_len
    q = np.asarray(nmpc.nlp[0].x_init["q"].init, dtype=float)
    qdot = np.empty_like(q)
    qdot[:, :-1] = np.diff(q, axis=1) / dt
    qdot[:, -1] = qdot[:, -2]

    lower, upper = _trajectory_bounds_for_guess(
        nmpc.nlp[0].x_bounds["qdot"], qdot.shape[1]
    )
    nmpc.nlp[0].x_init["qdot"].init[:, :] = np.minimum(np.maximum(qdot, lower), upper)
    nmpc._sync_acados_state_bounds()


def _window_accounting(
    source_window_solutions: list,
    exported_cycle_solutions: list,
    cycles_per_window: int,
) -> dict:
    attempted_windows = len(source_window_solutions)
    window_statuses = [
        window_solution.status for window_solution in source_window_solutions
    ]
    successful_windows = sum(_status_is_success(status) for status in window_statuses)
    failed_windows = attempted_windows - successful_windows
    exported_cycles = len(exported_cycle_solutions)
    covered_cycles = successful_windows
    if attempted_windows and successful_windows == attempted_windows:
        covered_cycles += cycles_per_window - 1

    return {
        "attempted_windows": attempted_windows,
        "successful_windows": successful_windows,
        "failed_windows": failed_windows,
        "exported_cycles": exported_cycles,
        "covered_cycles": covered_cycles,
        "window_statuses": window_statuses,
    }


def summarize_windows(sol, requested_windows: int, cycles_per_window: int) -> None:
    def _fmt(value) -> str:
        return "None" if value is None else f"{value:.6f}"

    merged_solution, source_window_solutions, exported_cycle_solutions = (
        _split_receding_solution(sol)
    )
    accounting = _window_accounting(
        source_window_solutions, exported_cycle_solutions, cycles_per_window
    )
    wheel_trace = _wheel_trace_from_exported_cycles(
        merged_solution, exported_cycle_solutions
    )
    diagnostics = diagnose_wheel_trace(wheel_trace, requested_windows=requested_windows)
    solver_success = (
        accounting["covered_cycles"] >= requested_windows
        and accounting["failed_windows"] == 0
    )
    physical_success = (
        diagnostics["is_physical"]
        and accounting["exported_cycles"] >= requested_windows
    )
    success = solver_success and physical_success

    print(f"merged_status: {merged_solution.status}")
    print(f"merged_cost: {merged_solution.cost}")
    print(f"merged_solver_time_s: {_fmt(merged_solution.solver_time_to_optimize)}")
    print(f"merged_wall_time_s: {_fmt(merged_solution.real_time_to_optimize)}")
    print(f"requested_windows: {requested_windows}")
    print(f"attempted_windows: {accounting['attempted_windows']}")
    print(f"successful_windows: {accounting['successful_windows']}")
    print(f"failed_windows: {accounting['failed_windows']}")
    print(f"exported_cycles: {accounting['exported_cycles']}")
    print(f"covered_cycles: {accounting['covered_cycles']}")
    print(f"solver_success: {solver_success}")
    print(f"physical_success: {physical_success}")
    print(f"success: {success}")
    print(f"final_wheel_angle: {diagnostics['final_angle']:.6f}")
    print(f"max_wheel_step: {diagnostics['max_step']:.6f}")
    if diagnostics["issues"]:
        print(f"diagnostic_issues: {', '.join(diagnostics['issues'])}")

    if source_window_solutions:
        print(f"source_window_count: {len(source_window_solutions)}")
        for idx, window_solution in enumerate(source_window_solutions):
            print(
                f"window[{idx}] status={window_solution.status} "
                f"solver_time_s={_fmt(window_solution.solver_time_to_optimize)} "
                f"wall_time_s={_fmt(window_solution.real_time_to_optimize)}"
            )


def build_window_summary(sol, requested_windows: int, cycles_per_window: int) -> dict:
    merged_solution, source_window_solutions, exported_cycle_solutions = (
        _split_receding_solution(sol)
    )
    accounting = _window_accounting(
        source_window_solutions, exported_cycle_solutions, cycles_per_window
    )
    wheel_trace = _wheel_trace_from_exported_cycles(
        merged_solution, exported_cycle_solutions
    )
    control_traces = _control_traces_from_exported_cycles(
        merged_solution, exported_cycle_solutions
    )
    state_traces = _state_traces_from_exported_cycles(
        merged_solution, exported_cycle_solutions
    )
    objective = (
        float(np.nansum(merged_solution.cost))
        if getattr(merged_solution, "cost", None) is not None
        else float("nan")
    )
    diagnostics = diagnose_wheel_trace(wheel_trace, requested_windows=requested_windows)
    solver_success = (
        accounting["covered_cycles"] >= requested_windows
        and accounting["failed_windows"] == 0
    )
    physical_success = (
        diagnostics["is_physical"]
        and accounting["exported_cycles"] >= requested_windows
    )
    success = solver_success and physical_success
    return {
        "mode": "rho",
        "status": merged_solution.status,
        "objective": objective,
        "solver_time_s": merged_solution.solver_time_to_optimize,
        "wall_time_s": merged_solution.real_time_to_optimize,
        "requested_windows": requested_windows,
        "achieved_windows": accounting["attempted_windows"],
        "attempted_windows": accounting["attempted_windows"],
        "successful_windows": accounting["successful_windows"],
        "failed_windows": accounting["failed_windows"],
        "exported_cycles": accounting["exported_cycles"],
        "covered_cycles": accounting["covered_cycles"],
        "window_statuses": accounting["window_statuses"],
        "solver_success": solver_success,
        "physical_success": physical_success,
        "window_count": accounting["attempted_windows"],
        "final_wheel_angle": float(wheel_trace[-1]),
        "wheel_angle_trace": wheel_trace,
        "state_traces": state_traces,
        "control_traces": control_traces,
        "solution": merged_solution,
        "window_solutions": source_window_solutions,
        "cycle_solutions": exported_cycle_solutions,
        "diagnostics": diagnostics,
        "success": success,
    }


def summarize_single_shot(sol) -> None:
    def _fmt(value) -> str:
        return "None" if value is None else f"{value:.6f}"

    print(f"single_shot_status: {sol.status}")
    print(f"single_shot_cost: {sol.cost}")
    print(f"single_shot_solver_time_s: {_fmt(sol.solver_time_to_optimize)}")
    print(f"single_shot_wall_time_s: {_fmt(sol.real_time_to_optimize)}")


def build_single_shot_summary(sol) -> dict:
    wheel_trace = sol.decision_states(to_merge=SolutionMerge.NODES)["q"][2, :]
    state_traces = {
        key: np.asarray(values)
        for key, values in sol.decision_states(to_merge=SolutionMerge.NODES).items()
    }
    control_traces = {
        key: np.asarray(values)
        for key, values in sol.decision_controls(to_merge=SolutionMerge.NODES).items()
    }
    objective = (
        float(np.nansum(sol.cost))
        if getattr(sol, "cost", None) is not None
        else float("nan")
    )
    diagnostics = diagnose_wheel_trace(wheel_trace, requested_windows=1)
    return {
        "mode": "single_shot",
        "status": sol.status,
        "objective": objective,
        "solver_time_s": sol.solver_time_to_optimize,
        "wall_time_s": sol.real_time_to_optimize,
        "final_wheel_angle": float(wheel_trace[-1]),
        "wheel_angle_trace": wheel_trace,
        "state_traces": state_traces,
        "control_traces": control_traces,
        "solution": sol,
        "window_solutions": [],
        "diagnostics": diagnostics,
        "success": diagnostics["is_physical"],
    }


def diagnose_wheel_trace(wheel_trace: np.ndarray, requested_windows: int) -> dict:
    trace = np.asarray(wheel_trace, dtype=float).squeeze()
    finite = bool(np.all(np.isfinite(trace)))
    final_angle = float(trace[-1]) if trace.size else float("nan")
    max_abs_angle = float(np.max(np.abs(trace))) if trace.size else float("nan")
    max_step = float(np.max(np.abs(np.diff(trace)))) if trace.size > 1 else 0.0
    expected_scale = max(2 * np.pi * max(requested_windows, 1), 1.0)
    angle_limit = 10.0 * expected_scale
    jump_limit = 2.0 * np.pi
    issues = []
    if not finite:
        issues.append("non_finite_wheel_trace")
    if finite and max_abs_angle > angle_limit:
        issues.append("wheel_angle_out_of_bounds")
    if finite and max_step > jump_limit:
        issues.append("wheel_angle_jump_out_of_bounds")

    return {
        "is_physical": not issues,
        "issues": issues,
        "final_angle": final_angle,
        "max_abs_angle": max_abs_angle,
        "max_step": max_step,
        "angle_limit": angle_limit,
        "jump_limit": jump_limit,
    }


def _shift_cyclical_trajectory(values: np.ndarray, nodes_per_cycle: int) -> np.ndarray:
    n_plus_one_cycles = values[nodes_per_cycle:-1]
    last_cycle = values[-nodes_per_cycle - 1 :]
    return (
        last_cycle
        if n_plus_one_cycles.size == 0
        else np.concatenate((n_plus_one_cycles, last_cycle))
    )


def estimate_periodic_cn_sum_from_cn(
    cn_values: np.ndarray, tauc: float, dt: float
) -> np.ndarray:
    cn_dot = np.gradient(cn_values, dt)
    return cn_values + tauc * cn_dot


def _to_numpy_vector(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape((-1,))


def _ding_state_keys(muscle_name: str) -> tuple[str, str, str, str, str, str]:
    return (
        f"Cn_{muscle_name}",
        f"Cn_sum_{muscle_name}",
        f"F_{muscle_name}",
        f"A_{muscle_name}",
        f"Tau1_{muscle_name}",
        f"Km_{muscle_name}",
    )


def _state_trajectory_bounds(
    periodic_nmpc, key: str, n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    bounds = periodic_nmpc.nlp[0].x_bounds[key]
    lower = np.empty(n_nodes)
    upper = np.empty(n_nodes)
    for node in range(n_nodes):
        column = 0 if node == 0 else 2 if node == n_nodes - 1 else 1
        lower[node] = bounds.min[0, column]
        upper[node] = bounds.max[0, column]
    return lower, upper


def _clip_state_trajectory_to_bounds(
    periodic_nmpc, key: str, values: np.ndarray
) -> np.ndarray:
    lower, upper = _state_trajectory_bounds(periodic_nmpc, key, values.size)
    return np.minimum(np.maximum(values, lower), upper)


def _periodic_ding_rhs(
    muscle_model,
    state: np.ndarray,
    pulse_width: float,
) -> np.ndarray:
    return _to_numpy_vector(
        muscle_model.system_dynamics(
            cn=state[0],
            cn_sum=state[1],
            f=state[2],
            a=state[3],
            tau1=state[4],
            km=state[5],
            pulse_width=pulse_width,
        )
    )


def _periodic_calcium_rhs(muscle_model, state: np.ndarray) -> np.ndarray:
    cn = state[0]
    cn_sum = state[1]
    return np.array(
        [
            float(muscle_model.cn_dot_fun(cn, cn_sum)),
            float(muscle_model.cn_sum_dot_fun(cn_sum)),
        ]
    )


def _rk4_periodic_ding_step(
    muscle_model,
    state: np.ndarray,
    pulse_width: float,
    dt: float,
    n_substeps: int = 1,
) -> np.ndarray:
    if n_substeps < 1:
        raise ValueError("--periodic-fes-warmup-projection-substeps must be >= 1.")

    sub_dt = dt / n_substeps
    next_state = np.array(state, dtype=float, copy=True)
    for _ in range(n_substeps):
        k1 = _periodic_ding_rhs(muscle_model, next_state, pulse_width)
        k2 = _periodic_ding_rhs(
            muscle_model,
            next_state + 0.5 * sub_dt * k1,
            pulse_width,
        )
        k3 = _periodic_ding_rhs(
            muscle_model,
            next_state + 0.5 * sub_dt * k2,
            pulse_width,
        )
        k4 = _periodic_ding_rhs(
            muscle_model,
            next_state + sub_dt * k3,
            pulse_width,
        )
        next_state = next_state + sub_dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return next_state


def _exact_periodic_calcium_step(
    muscle_model,
    state: np.ndarray,
    dt: float,
) -> np.ndarray:
    tau = muscle_model.tauc
    decay = np.exp(-dt / tau)
    cn = state[0]
    cn_sum = state[1]
    cn_sum_steady = tau * muscle_model.periodic_cn_sum_gain()
    next_cn_sum = cn_sum_steady + (cn_sum - cn_sum_steady) * decay
    next_cn = (
        decay * cn
        + cn_sum_steady * (1 - decay)
        + (cn_sum - cn_sum_steady) * (dt / tau) * decay
    )
    return np.array([next_cn, next_cn_sum])


def _periodic_fes_rollout_defects(
    periodic_nmpc, projection_substeps: int = 10
) -> dict[str, float]:
    dt = periodic_nmpc.cycle_duration / periodic_nmpc.cycle_len
    defects = {}
    for muscle_model in periodic_nmpc.nlp[0].model.muscles_dynamics_model:
        muscle_name = muscle_model.muscle_name
        state_keys = _ding_state_keys(muscle_name)
        control_key = f"last_pulse_width_{muscle_name}"
        if any(key not in periodic_nmpc.nlp[0].x_init.keys() for key in state_keys):
            continue
        if control_key not in periodic_nmpc.nlp[0].u_init.keys():
            continue

        states = np.vstack(
            [periodic_nmpc.nlp[0].x_init[key].init[0, :] for key in state_keys]
        )
        controls = periodic_nmpc.nlp[0].u_init[control_key].init[0, :]
        max_defect = 0.0
        for node, pulse_width in enumerate(controls):
            expected = _rk4_periodic_ding_step(
                muscle_model,
                states[:, node],
                pulse_width,
                dt,
                n_substeps=projection_substeps,
            )
            max_defect = max(
                max_defect, float(np.max(np.abs(states[:, node + 1] - expected)))
            )
        defects[muscle_name] = max_defect
    return defects


def _periodic_fes_rollout_defect_details(
    periodic_nmpc, projection_substeps: int = 10
) -> dict[str, dict[str, float]]:
    dt = periodic_nmpc.cycle_duration / periodic_nmpc.cycle_len
    state_labels = ("Cn", "Cn_sum", "F", "A", "Tau1", "Km")
    absolute_by_state = dict.fromkeys(state_labels, 0.0)
    scaled_by_state = dict.fromkeys(state_labels, 0.0)
    absolute_by_muscle = {}

    for muscle_model in periodic_nmpc.nlp[0].model.muscles_dynamics_model:
        muscle_name = muscle_model.muscle_name
        state_keys = _ding_state_keys(muscle_name)
        control_key = f"last_pulse_width_{muscle_name}"
        if any(key not in periodic_nmpc.nlp[0].x_init.keys() for key in state_keys):
            continue
        if control_key not in periodic_nmpc.nlp[0].u_init.keys():
            continue

        states = np.vstack(
            [periodic_nmpc.nlp[0].x_init[key].init[0, :] for key in state_keys]
        )
        controls = periodic_nmpc.nlp[0].u_init[control_key].init[0, :]
        state_scales = np.maximum(np.max(np.abs(states), axis=1), 1.0)
        muscle_max = 0.0
        for node, pulse_width in enumerate(controls):
            expected = _rk4_periodic_ding_step(
                muscle_model,
                states[:, node],
                pulse_width,
                dt,
                n_substeps=projection_substeps,
            )
            defects = np.abs(states[:, node + 1] - expected)
            muscle_max = max(muscle_max, float(np.max(defects)))
            for label, absolute_defect, scale in zip(
                state_labels, defects, state_scales, strict=True
            ):
                absolute_by_state[label] = max(
                    absolute_by_state[label], float(absolute_defect)
                )
                scaled_by_state[label] = max(
                    scaled_by_state[label], float(absolute_defect / scale)
                )
        absolute_by_muscle[muscle_name] = muscle_max

    if not absolute_by_muscle:
        return {}

    return {
        "absolute_by_state": absolute_by_state,
        "absolute_by_muscle": absolute_by_muscle,
        "scaled_by_state": scaled_by_state,
    }


def _stack_initial_guess_values(container, variables, n_nodes: int) -> np.ndarray:
    values = np.zeros((variables.shape, n_nodes))
    for key in variables.keys():
        values[variables[key].index, :] = np.asarray(container[key].init, dtype=float)
    return values


def _numerical_timeseries_at_node(nlp, node: int) -> np.ndarray:
    timeseries = nlp.numerical_data_timeseries
    if not timeseries:
        return np.array([], dtype=float)

    values = []
    for array in timeseries.values():
        array = np.asarray(array, dtype=float)
        for element_index in range(array.shape[1]):
            values.append(array[:, element_index, node])
    return np.concatenate(values) if values else np.array([], dtype=float)


def _full_dynamics_rhs(
    nlp,
    time: float,
    dt: float,
    state: np.ndarray,
    control: np.ndarray,
    numerical_timeseries: np.ndarray | None = None,
) -> np.ndarray:
    numerical_timeseries = (
        np.array([], dtype=float)
        if numerical_timeseries is None
        else np.asarray(numerical_timeseries, dtype=float).reshape(-1)
    )
    return _to_numpy_vector(
        nlp.dynamics_func(
            np.array([time, dt], dtype=float),
            state,
            control,
            np.array([], dtype=float),
            np.array([], dtype=float),
            numerical_timeseries,
        )
    )


def _rk4_full_dynamics_step(
    nlp,
    state: np.ndarray,
    control: np.ndarray,
    time: float,
    dt: float,
    n_substeps: int,
    numerical_timeseries: np.ndarray | None = None,
) -> np.ndarray:
    sub_dt = dt / n_substeps
    next_state = np.array(state, dtype=float, copy=True)
    for substep in range(n_substeps):
        sub_time = time + substep * sub_dt
        k1 = _full_dynamics_rhs(
            nlp,
            sub_time,
            sub_dt,
            next_state,
            control,
            numerical_timeseries,
        )
        k2 = _full_dynamics_rhs(
            nlp,
            sub_time + 0.5 * sub_dt,
            sub_dt,
            next_state + 0.5 * sub_dt * k1,
            control,
            numerical_timeseries,
        )
        k3 = _full_dynamics_rhs(
            nlp,
            sub_time + 0.5 * sub_dt,
            sub_dt,
            next_state + 0.5 * sub_dt * k2,
            control,
            numerical_timeseries,
        )
        k4 = _full_dynamics_rhs(
            nlp,
            sub_time + sub_dt,
            sub_dt,
            next_state + sub_dt * k3,
            control,
            numerical_timeseries,
        )
        next_state = next_state + sub_dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return next_state


def _full_dynamics_rollout_defect_details(
    nmpc, n_substeps: int = 5, top_key_count: int = 8
) -> dict[str, dict[str, float]]:
    nlp = nmpc.nlp[0]
    if not hasattr(nlp, "dynamics_func") or nlp.dynamics_func is None:
        return {}
    if nlp.states.shape == 0 or nlp.controls.shape == 0:
        return {}

    first_state_key = next(iter(nlp.x_init.keys()))
    first_control_key = next(iter(nlp.u_init.keys()))
    n_state_nodes = nlp.x_init[first_state_key].init.shape[1]
    n_control_nodes = nlp.u_init[first_control_key].init.shape[1]
    if n_state_nodes != n_control_nodes + 1:
        return {}

    states = _stack_initial_guess_values(nlp.x_init, nlp.states, n_state_nodes)
    controls = _stack_initial_guess_values(nlp.u_init, nlp.controls, n_control_nodes)
    dt = nmpc.cycle_duration / nmpc.cycle_len
    stage_numerical_timeseries = [
        _numerical_timeseries_at_node(nlp, node) for node in range(n_control_nodes)
    ]

    predicted = np.empty_like(states)
    predicted[:, 0] = states[:, 0]
    for node in range(n_control_nodes):
        predicted[:, node + 1] = _rk4_full_dynamics_step(
            nlp,
            states[:, node],
            controls[:, node],
            node * dt,
            dt,
            n_substeps=n_substeps,
            numerical_timeseries=stage_numerical_timeseries[node],
        )

    defects = states[:, 1:] - predicted[:, 1:]
    state_scales = np.maximum(np.max(np.abs(states), axis=1, keepdims=True), 1.0)
    scaled_defects = defects / state_scales

    absolute_by_block = {}
    scaled_by_block = {}
    key_defects = {}
    for block_name, key_names in {
        "q": ("q",),
        "qdot": ("qdot",),
        "fes": tuple(key for key in nlp.states.keys() if key not in ("q", "qdot")),
    }.items():
        indexes = []
        for key in key_names:
            if key in nlp.states.keys():
                indexes.extend(
                    np.asarray(nlp.states[key].index).reshape((-1,)).tolist()
                )
        if not indexes:
            continue
        absolute_by_block[block_name] = float(np.max(np.abs(defects[indexes, :])))
        scaled_by_block[block_name] = float(np.max(np.abs(scaled_defects[indexes, :])))

    for key in nlp.states.keys():
        indexes = np.asarray(nlp.states[key].index).reshape((-1,)).tolist()
        key_defects[key] = float(np.max(np.abs(defects[indexes, :])))

    top_keys = dict(
        sorted(key_defects.items(), key=lambda item: item[1], reverse=True)[
            :top_key_count
        ]
    )
    dof_names = tuple(getattr(nlp.model, "name_dofs", ())) or tuple(
        f"dof_{idx}" for idx in range(nlp.model.nb_q)
    )

    def defects_by_dof(key: str) -> dict[str, float]:
        if key not in nlp.states.keys():
            return {}
        indexes = np.asarray(nlp.states[key].index).reshape((-1,)).tolist()
        values = {}
        for dof_idx, state_idx in enumerate(indexes):
            name = (
                str(dof_names[dof_idx])
                if dof_idx < len(dof_names)
                else f"dof_{dof_idx}"
            )
            values[name] = float(np.max(np.abs(defects[state_idx, :])))
        return values

    def worst_qdot_nodes() -> list[dict]:
        if "qdot" not in nlp.states.keys():
            return []

        qdot_indexes = np.asarray(nlp.states["qdot"].index).reshape((-1,)).tolist()
        q_indexes = (
            np.asarray(nlp.states["q"].index).reshape((-1,)).tolist()
            if "q" in nlp.states.keys()
            else []
        )
        force_indexes = {
            key.replace("F_", ""): int(
                np.asarray(nlp.states[key].index).reshape((-1,))[0]
            )
            for key in nlp.states.keys()
            if key.startswith("F_")
        }
        control_indexes = {
            key.replace("last_pulse_width_", ""): int(
                np.asarray(nlp.controls[key].index).reshape((-1,))[0]
            )
            for key in nlp.controls.keys()
            if key.startswith("last_pulse_width_")
        }

        rows = []
        for dof_idx, state_idx in enumerate(qdot_indexes):
            node = int(np.argmax(np.abs(defects[state_idx, :])))
            rhs = _full_dynamics_rhs(
                nlp,
                node * dt,
                dt,
                states[:, node],
                controls[:, node],
                stage_numerical_timeseries[node],
            )
            dof_name = (
                str(dof_names[dof_idx])
                if dof_idx < len(dof_names)
                else f"dof_{dof_idx}"
            )
            row = {
                "dof": dof_name,
                "node": node,
                "time": float(node * dt),
                "defect": float(defects[state_idx, node]),
                "state_next": float(states[state_idx, node + 1]),
                "predicted_next": float(predicted[state_idx, node + 1]),
                "state_current": float(states[state_idx, node]),
                "rhs": float(rhs[state_idx]),
                "forces": {
                    muscle_name: float(states[index, node])
                    for muscle_name, index in force_indexes.items()
                },
                "controls": {
                    muscle_name: float(controls[index, node])
                    for muscle_name, index in control_indexes.items()
                },
            }
            if dof_idx < len(q_indexes):
                q_idx = q_indexes[dof_idx]
                row["q_current"] = float(states[q_idx, node])
                row["q_next"] = float(states[q_idx, node + 1])
                row["q_predicted_next"] = float(predicted[q_idx, node + 1])
            rows.append(row)

        return sorted(rows, key=lambda item: abs(item["defect"]), reverse=True)

    return {
        "absolute_by_block": absolute_by_block,
        "scaled_by_block": scaled_by_block,
        "top_keys": top_keys,
        "q_by_dof": defects_by_dof("q"),
        "qdot_by_dof": defects_by_dof("qdot"),
        "worst_qdot_nodes": worst_qdot_nodes(),
    }


def high_accuracy_integrator_map_diagnostics(
    nmpc, nodes: tuple[int, ...] = (0, 1, 15, 29), rk4_substeps: int = 5
) -> list[dict]:
    """Compare the shooting trajectory and RK4 map with a DOP853 reference."""

    from scipy.integrate import solve_ivp

    nlp = nmpc.nlp[0]
    first_state_key = next(iter(nlp.x_init.keys()))
    first_control_key = next(iter(nlp.u_init.keys()))
    n_state_nodes = nlp.x_init[first_state_key].init.shape[1]
    n_control_nodes = nlp.u_init[first_control_key].init.shape[1]
    states = _stack_initial_guess_values(nlp.x_init, nlp.states, n_state_nodes)
    controls = _stack_initial_guess_values(nlp.u_init, nlp.controls, n_control_nodes)
    dt = nmpc.cycle_duration / nmpc.cycle_len
    rows = []
    for node in sorted(set(nodes)):
        if node < 0 or node >= n_control_nodes:
            continue
        data = _numerical_timeseries_at_node(nlp, node)
        start_time = node * dt
        initial_state = states[:, node]
        control = controls[:, node]
        reference = solve_ivp(
            lambda time, state: _full_dynamics_rhs(nlp, time, dt, state, control, data),
            (start_time, start_time + dt),
            initial_state,
            method="DOP853",
            rtol=1e-11,
            atol=1e-13,
        )
        if not reference.success:
            raise RuntimeError(
                f"High-accuracy integration failed at node {node}: {reference.message}"
            )
        reference_next = reference.y[:, -1]
        rk4_next = _rk4_full_dynamics_step(
            nlp,
            initial_state,
            control,
            start_time,
            dt,
            n_substeps=rk4_substeps,
            numerical_timeseries=data,
        )
        trajectory_next = states[:, node + 1]
        rows.append(
            {
                "node": node,
                "trajectory_vs_reference": float(
                    np.max(np.abs(trajectory_next - reference_next))
                ),
                "rk4_vs_reference": float(np.max(np.abs(rk4_next - reference_next))),
                "trajectory_vs_rk4": float(np.max(np.abs(trajectory_next - rk4_next))),
                "reference_evaluations": int(reference.nfev),
            }
        )
    return rows


def solution_trace_comparisons(
    reference_solution, candidate_solution, *, controls: bool
) -> list[dict]:
    """Compare node-wise traces from two solutions of the same formulation."""

    accessor = "decision_controls" if controls else "decision_states"
    reference = getattr(reference_solution, accessor)(to_merge=SolutionMerge.NODES)
    candidate = getattr(candidate_solution, accessor)(to_merge=SolutionMerge.NODES)
    rows = []
    for key in sorted(set(reference).intersection(candidate)):
        reference_values = np.atleast_2d(np.asarray(reference[key], dtype=float))
        candidate_values = np.atleast_2d(np.asarray(candidate[key], dtype=float))
        for index in range(min(reference_values.shape[0], candidate_values.shape[0])):
            common_len = max(reference_values.shape[1], candidate_values.shape[1])
            reference_trace = _resample_trace(reference_values[index], common_len)
            candidate_trace = _resample_trace(candidate_values[index], common_len)
            difference = candidate_trace - reference_trace
            reference_scale = max(
                float(np.ptp(reference_trace)),
                float(np.max(np.abs(reference_trace))),
                np.finfo(float).eps,
            )
            rows.append(
                {
                    "key": key if reference_values.shape[0] == 1 else f"{key}[{index}]",
                    "common_len": common_len,
                    "rmse": float(np.sqrt(np.mean(difference**2))),
                    "normalized_rmse": float(
                        np.sqrt(np.mean(difference**2)) / reference_scale
                    ),
                    "max_abs_error": float(np.max(np.abs(difference))),
                    "final_error": float(difference[-1]),
                    "reference_mean": float(np.mean(reference_trace)),
                    "candidate_mean": float(np.mean(candidate_trace)),
                    "reference_range": (
                        float(np.min(reference_trace)),
                        float(np.max(reference_trace)),
                    ),
                    "candidate_range": (
                        float(np.min(candidate_trace)),
                        float(np.max(candidate_trace)),
                    ),
                }
            )
    return sorted(rows, key=lambda item: item["normalized_rmse"], reverse=True)


def print_solution_trace_comparison(
    label: str,
    reference_solution,
    candidate_solution,
    *,
    controls: bool,
    limit: int,
) -> list[dict]:
    rows = solution_trace_comparisons(
        reference_solution, candidate_solution, controls=controls
    )
    print(
        f"{label} | key | common_len | rmse | normalized_rmse | max_abs_error | "
        "final_error | reference_mean | candidate_mean | reference_range | candidate_range"
    )
    for row in rows[:limit]:
        reference_min, reference_max = row["reference_range"]
        candidate_min, candidate_max = row["candidate_range"]
        print(
            f"{label} | {row['key']} | {row['common_len']} | {row['rmse']:.6g} | "
            f"{row['normalized_rmse']:.6g} | {row['max_abs_error']:.6g} | "
            f"{row['final_error']:.6g} | {row['reference_mean']:.6g} | "
            f"{row['candidate_mean']:.6g} | "
            f"[{reference_min:.6g}, {reference_max:.6g}] | "
            f"[{candidate_min:.6g}, {candidate_max:.6g}]"
        )
    return rows


def _proximal_phase_one_update(
    reference: np.ndarray,
    predicted: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    proximity_weight: float,
    defect_weight: float,
) -> np.ndarray:
    if proximity_weight < 0.0 or defect_weight < 0.0:
        raise ValueError("Phase-I weights must be non-negative.")
    denominator = proximity_weight + defect_weight
    if denominator <= 0.0:
        raise ValueError("At least one phase-I weight must be strictly positive.")
    candidate = (proximity_weight * reference + defect_weight * predicted) / denominator
    return np.minimum(np.maximum(candidate, lower), upper)


def project_full_dynamics_initial_guess(
    nmpc,
    proximity_weight: float = 1.0,
    defect_weight: float = 10.0,
    n_substeps: int = 10,
) -> dict:
    """Sequential proximal phase I with controls fixed to the reference values."""

    if n_substeps < 1:
        raise ValueError("Phase-I RK4 substeps must be strictly positive.")
    before = _full_dynamics_rollout_defect_details(nmpc, n_substeps=n_substeps)
    nlp = nmpc.nlp[0]
    first_state_key = next(iter(nlp.x_init.keys()))
    first_control_key = next(iter(nlp.u_init.keys()))
    n_state_nodes = nlp.x_init[first_state_key].init.shape[1]
    n_control_nodes = nlp.u_init[first_control_key].init.shape[1]
    if n_state_nodes != n_control_nodes + 1:
        raise ValueError(
            "The complete-dynamics phase-I projection currently requires one state node "
            "per shooting endpoint; use RK4, RK8 or IRK instead of direct collocation."
        )
    states = _stack_initial_guess_values(nlp.x_init, nlp.states, n_state_nodes)
    reference = states.copy()
    controls = _stack_initial_guess_values(nlp.u_init, nlp.controls, n_control_nodes)
    lower = np.empty_like(states)
    upper = np.empty_like(states)
    for key in nlp.states.keys():
        indexes = np.asarray(nlp.states[key].index).reshape((-1,)).tolist()
        key_lower, key_upper = _trajectory_bounds_for_guess(
            nlp.x_bounds[key], n_state_nodes
        )
        lower[indexes, :] = key_lower
        upper[indexes, :] = key_upper

    dt = nmpc.cycle_duration / nmpc.cycle_len
    for node in range(n_control_nodes):
        numerical_data = _numerical_timeseries_at_node(nlp, node)
        predicted = _rk4_full_dynamics_step(
            nlp,
            states[:, node],
            controls[:, node],
            node * dt,
            dt,
            n_substeps=n_substeps,
            numerical_timeseries=numerical_data,
        )
        states[:, node + 1] = _proximal_phase_one_update(
            reference[:, node + 1],
            predicted,
            lower[:, node + 1],
            upper[:, node + 1],
            proximity_weight,
            defect_weight,
        )

    for key in nlp.states.keys():
        indexes = np.asarray(nlp.states[key].index).reshape((-1,)).tolist()
        nlp.x_init[key].init[:, :] = states[indexes, :]
    nmpc._correct_init_guess_to_fit_bounds(corrected_input="states")
    nmpc._sync_acados_state_bounds()
    after = _full_dynamics_rollout_defect_details(nmpc, n_substeps=n_substeps)

    def maximum_scaled_defect(details: dict) -> float:
        return max(details.get("scaled_by_block", {}).values(), default=0.0)

    return {
        "proximity_weight": proximity_weight,
        "defect_weight": defect_weight,
        "n_substeps": n_substeps,
        "scaled_defect_before": maximum_scaled_defect(before),
        "scaled_defect_after": maximum_scaled_defect(after),
        "absolute_by_block_before": before.get("absolute_by_block", {}),
        "absolute_by_block_after": after.get("absolute_by_block", {}),
        "max_state_change": float(np.max(np.abs(states - reference))),
    }


def wheel_angle_periodicity_diagnostics(nmpc, node: int = 0) -> dict[str, float]:
    """Check whether a full revolution is a pure coordinate change in the dynamics."""

    nlp = nmpc.nlp[0]
    first_state_key = next(iter(nlp.x_init.keys()))
    first_control_key = next(iter(nlp.u_init.keys()))
    n_state_nodes = nlp.x_init[first_state_key].init.shape[1]
    n_control_nodes = nlp.u_init[first_control_key].init.shape[1]
    states = _stack_initial_guess_values(nlp.x_init, nlp.states, n_state_nodes)
    controls = _stack_initial_guess_values(nlp.u_init, nlp.controls, n_control_nodes)
    q_indexes = np.asarray(nlp.states["q"].index).reshape((-1,))
    if q_indexes.size < 3:
        raise ValueError("The cycling model must expose the crank as q[2].")
    if node < 0 or node >= n_control_nodes:
        raise ValueError("The periodicity diagnostic node is outside the horizon.")

    dt = nmpc.cycle_duration / nmpc.cycle_len
    numerical_data = _numerical_timeseries_at_node(nlp, node)
    base_state = states[:, node].copy()
    shifted_state = base_state.copy()
    shifted_state[int(q_indexes[2])] += 2.0 * np.pi
    base_rhs = _full_dynamics_rhs(
        nlp, node * dt, dt, base_state, controls[:, node], numerical_data
    )
    shifted_rhs = _full_dynamics_rhs(
        nlp, node * dt, dt, shifted_state, controls[:, node], numerical_data
    )
    difference = shifted_rhs - base_rhs
    return {
        "max_abs_rhs_difference": float(np.max(np.abs(difference))),
        "l2_rhs_difference": float(np.linalg.norm(difference)),
    }


def rollout_transferred_cycle_full_dynamics(
    periodic_nmpc,
    n_substeps: int = 5,
    max_allowed_bound_violation: float | None = None,
) -> dict:
    if n_substeps < 1:
        raise ValueError("ACADOS transfer rollout substeps must be >= 1.")

    nlp = periodic_nmpc.nlp[0]
    first_state_key = next(iter(nlp.x_init.keys()))
    first_control_key = next(iter(nlp.u_init.keys()))
    n_state_nodes = nlp.x_init[first_state_key].init.shape[1]
    n_control_nodes = nlp.u_init[first_control_key].init.shape[1]
    if n_state_nodes != n_control_nodes + 1:
        raise ValueError(
            "The transfer rollout requires one more state node than controls."
        )

    start_node = periodic_nmpc.nodes_per_cycle - 1
    if start_node < 0 or start_node >= n_control_nodes:
        raise ValueError("The transfer rollout start node is outside the horizon.")

    states = _stack_initial_guess_values(nlp.x_init, nlp.states, n_state_nodes)
    original_states = states.copy()
    controls = _stack_initial_guess_values(nlp.u_init, nlp.controls, n_control_nodes)
    dt = periodic_nmpc.cycle_duration / periodic_nmpc.cycle_len
    stage_numerical_timeseries = [
        _numerical_timeseries_at_node(nlp, node) for node in range(n_control_nodes)
    ]

    for node in range(start_node, n_control_nodes):
        states[:, node + 1] = _rk4_full_dynamics_step(
            nlp,
            states[:, node],
            controls[:, node],
            node * dt,
            dt,
            n_substeps=n_substeps,
            numerical_timeseries=stage_numerical_timeseries[node],
        )
        if not np.all(np.isfinite(states[:, node + 1])):
            return {
                "applied": False,
                "start_node": start_node,
                "nonfinite_node": node + 1,
            }

    max_bound_violation = 0.0
    max_bound_violation_by_key = {}
    terminal_delta = {}
    max_delta_by_key = {}
    state_values = {}
    for key in nlp.states.keys():
        indexes = np.asarray(nlp.states[key].index).reshape((-1,)).tolist()
        values = states[indexes, :]
        state_values[key] = values
        original_values = original_states[indexes, :]
        lower, upper = _trajectory_bounds_for_guess(nlp.x_bounds[key], n_state_nodes)
        violations = np.maximum(lower - values, 0.0) + np.maximum(values - upper, 0.0)
        key_bound_violation = float(np.max(violations))
        max_bound_violation_by_key[key] = key_bound_violation
        max_bound_violation = max(max_bound_violation, key_bound_violation)
        terminal_delta[key] = float(
            np.max(np.abs(values[:, -1] - original_values[:, -1]))
        )
        max_delta_by_key[key] = float(
            np.max(
                np.abs(
                    values[:, start_node + 1 :] - original_values[:, start_node + 1 :]
                )
            )
        )
    applied = (
        max_allowed_bound_violation is None
        or max_bound_violation <= max_allowed_bound_violation
    )
    if applied:
        for key, values in state_values.items():
            nlp.x_init[key].init[:, :] = values

    return {
        "applied": applied,
        "start_node": start_node,
        "max_bound_violation": max_bound_violation,
        "max_bound_violation_by_key": max_bound_violation_by_key,
        "terminal_delta": terminal_delta,
        "max_delta_by_key": max_delta_by_key,
    }


def _projection_state_keys(muscle_name: str, projection_mode: str) -> tuple[str, ...]:
    state_keys = _ding_state_keys(muscle_name)
    if projection_mode == "calcium":
        return state_keys[:2]
    if projection_mode in (
        "all",
        "all_except_force",
        "all_force_blend",
        "all_force_adaptive_blend",
    ):
        return state_keys
    raise ValueError(
        "--periodic-fes-warmup-projection-mode must be 'calcium', 'all', "
        "'all_except_force', 'all_force_blend' or 'all_force_adaptive_blend'."
    )


def _projection_write_state_keys(
    muscle_name: str, projection_mode: str
) -> tuple[str, ...]:
    state_keys = _projection_state_keys(muscle_name, projection_mode)
    if projection_mode == "all_except_force":
        force_key = f"F_{muscle_name}"
        return tuple(key for key in state_keys if key != force_key)
    return state_keys


def _qdot_defect_max_from_full_dynamics(nmpc) -> float:
    defects = _full_dynamics_rollout_defect_details(nmpc)
    qdot_defects = defects.get("qdot_by_dof", {}) if defects else {}
    if not qdot_defects:
        return float("nan")
    return float(max(abs(value) for value in qdot_defects.values()))


def _apply_force_projection_candidates(
    periodic_nmpc,
    force_candidates: dict[str, tuple[np.ndarray, np.ndarray]],
    weight: float,
) -> None:
    for key, (original_values, projected_values) in force_candidates.items():
        values = weight * projected_values + (1.0 - weight) * original_values
        lower, upper = _state_trajectory_bounds(periodic_nmpc, key, values.size)
        periodic_nmpc.nlp[0].x_init[key].init[0, :] = np.minimum(
            np.maximum(values, lower), upper
        )


def _select_force_projection_weight_by_qdot_defect(
    periodic_nmpc,
    force_candidates: dict[str, tuple[np.ndarray, np.ndarray]],
    max_weight: float,
    qdot_defect_limit: float,
    n_steps: int,
) -> dict[str, float]:
    if max_weight < 0.0 or max_weight > 1.0:
        raise ValueError(
            "--periodic-fes-warmup-force-projection-weight must be between 0 and 1."
        )
    if qdot_defect_limit < 0.0:
        raise ValueError(
            "--periodic-fes-warmup-force-qdot-defect-limit must be non-negative."
        )
    if n_steps <= 0:
        raise ValueError(
            "--periodic-fes-warmup-force-adaptive-steps must be strictly positive."
        )

    best_weight = 0.0
    best_qdot_defect = float("nan")
    tested_weights = np.linspace(0.0, max_weight, n_steps + 1)
    for candidate_weight in tested_weights:
        _apply_force_projection_candidates(
            periodic_nmpc, force_candidates, float(candidate_weight)
        )
        candidate_qdot_defect = _qdot_defect_max_from_full_dynamics(periodic_nmpc)
        if np.isfinite(candidate_qdot_defect):
            if candidate_qdot_defect <= qdot_defect_limit:
                best_weight = float(candidate_weight)
                best_qdot_defect = float(candidate_qdot_defect)
            elif candidate_weight == 0.0:
                best_qdot_defect = float(candidate_qdot_defect)

    _apply_force_projection_candidates(periodic_nmpc, force_candidates, best_weight)
    return {
        "selected_weight": best_weight,
        "qdot_defect": best_qdot_defect,
        "qdot_defect_limit": qdot_defect_limit,
        "candidate_count": float(tested_weights.size),
    }


def _project_periodic_states(
    muscle_model,
    original_states: np.ndarray,
    controls: np.ndarray,
    dt: float,
    projection_mode: str,
    projection_substeps: int,
) -> np.ndarray:
    projected_states = np.empty_like(original_states)
    projected_states[:, 0] = original_states[:, 0]
    for node, pulse_width in enumerate(controls):
        if projection_mode == "calcium":
            projected_states[:, node + 1] = _exact_periodic_calcium_step(
                muscle_model,
                projected_states[:, node],
                dt,
            )
        else:
            projected_states[:, node + 1] = _rk4_periodic_ding_step(
                muscle_model,
                projected_states[:, node],
                pulse_width,
                dt,
                n_substeps=projection_substeps,
            )
    return projected_states


def _projection_state_scales(
    states: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray
) -> np.ndarray:
    scales = np.ones((states.shape[0], 1))
    for idx in range(states.shape[0]):
        values = [np.abs(states[idx, :])]
        finite_lower = lower_bounds[idx, np.isfinite(lower_bounds[idx, :])]
        finite_upper = upper_bounds[idx, np.isfinite(upper_bounds[idx, :])]
        if finite_lower.size:
            values.append(np.abs(finite_lower))
        if finite_upper.size:
            values.append(np.abs(finite_upper))
        scale = float(np.max(np.concatenate(values)))
        scales[idx, 0] = scale if np.isfinite(scale) and scale > 1.0 else 1.0
    return scales


def _strict_projection_bounds(
    lower_bounds: np.ndarray, upper_bounds: np.ndarray, scales: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lower_flat = lower_bounds.reshape((-1,)).astype(float, copy=True)
    upper_flat = upper_bounds.reshape((-1,)).astype(float, copy=True)
    scale_flat = np.broadcast_to(scales, lower_bounds.shape).reshape((-1,))
    eps = np.maximum(1e-12 * scale_flat, 1e-12)
    locked = upper_flat <= lower_flat
    centers = 0.5 * (lower_flat[locked] + upper_flat[locked])
    lower_flat[locked] = centers - eps[locked]
    upper_flat[locked] = centers + eps[locked]
    return lower_flat, upper_flat


def _projection_jacobian_sparsity(
    n_states: int,
    n_nodes: int,
    include_defects: bool,
    include_proximity: bool,
):
    from scipy.sparse import lil_matrix

    n_variables = n_states * (n_nodes - 1)
    n_defect_rows = n_states * (n_nodes - 1) if include_defects else 0
    n_proximity_rows = n_states * (n_nodes - 1) if include_proximity else 0
    sparsity = lil_matrix((n_defect_rows + n_proximity_rows, n_variables), dtype=int)

    def variable_column(state_idx: int, node_idx: int) -> int:
        return state_idx * (n_nodes - 1) + node_idx - 1

    if include_defects:
        for interval_idx in range(n_nodes - 1):
            for residual_state_idx in range(n_states):
                row = interval_idx * n_states + residual_state_idx
                if interval_idx > 0:
                    for state_idx in range(n_states):
                        sparsity[row, variable_column(state_idx, interval_idx)] = 1
                for state_idx in range(n_states):
                    sparsity[row, variable_column(state_idx, interval_idx + 1)] = 1

    if include_proximity:
        row_offset = n_defect_rows
        for state_idx in range(n_states):
            for node_idx in range(1, n_nodes):
                row = row_offset + state_idx * (n_nodes - 1) + node_idx - 1
                sparsity[row, variable_column(state_idx, node_idx)] = 1

    return sparsity.tocsr()


def _bounded_least_squares_project_periodic_states(
    muscle_model,
    original_states: np.ndarray,
    controls: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    dt: float,
    projection_mode: str,
    projection_substeps: int,
    proximity_weight: float,
    defect_weight: float,
    trust_radius: float | None,
    max_iterations: int,
) -> tuple[np.ndarray, dict[str, float]]:
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise RuntimeError(
            "The least-squares FES warmup projection requires scipy."
        ) from exc

    if proximity_weight < 0.0:
        raise ValueError(
            "--periodic-fes-warmup-projection-proximity-weight must be >= 0."
        )
    if defect_weight < 0.0:
        raise ValueError("--periodic-fes-warmup-projection-defect-weight must be >= 0.")
    if trust_radius is not None and trust_radius < 0.0:
        raise ValueError("--periodic-fes-warmup-projection-trust-radius must be >= 0.")
    if max_iterations <= 0:
        raise ValueError(
            "--periodic-fes-warmup-projection-max-iterations must be strictly positive."
        )

    clipped_original = np.minimum(
        np.maximum(original_states, lower_bounds), upper_bounds
    )
    scales = _projection_state_scales(clipped_original, lower_bounds, upper_bounds)
    variable_lower = lower_bounds[:, 1:].copy()
    variable_upper = upper_bounds[:, 1:].copy()

    if trust_radius is not None:
        trust = trust_radius * scales
        variable_center = clipped_original[:, 1:]
        variable_lower = np.maximum(variable_lower, variable_center - trust)
        variable_upper = np.minimum(variable_upper, variable_center + trust)

    lower_flat, upper_flat = _strict_projection_bounds(
        variable_lower, variable_upper, scales
    )
    x0 = np.minimum(
        np.maximum(clipped_original[:, 1:].reshape((-1,)), lower_flat), upper_flat
    )
    first_node = clipped_original[:, 0].copy()
    warmup_variable = clipped_original[:, 1:]
    sqrt_proximity_weight = float(np.sqrt(proximity_weight))
    sqrt_defect_weight = float(np.sqrt(defect_weight))
    include_defects = bool(sqrt_defect_weight)
    include_proximity = bool(sqrt_proximity_weight)
    jac_sparsity = _projection_jacobian_sparsity(
        clipped_original.shape[0],
        clipped_original.shape[1],
        include_defects=include_defects,
        include_proximity=include_proximity,
    )

    def unpack(variable_flat: np.ndarray) -> np.ndarray:
        states = np.empty_like(clipped_original)
        states[:, 0] = first_node
        states[:, 1:] = variable_flat.reshape((clipped_original.shape[0], -1))
        return states

    def step(state: np.ndarray, pulse_width: float) -> np.ndarray:
        if projection_mode == "calcium":
            return _exact_periodic_calcium_step(muscle_model, state, dt)
        return _rk4_periodic_ding_step(
            muscle_model,
            state,
            pulse_width,
            dt,
            n_substeps=projection_substeps,
        )

    def residual(variable_flat: np.ndarray) -> np.ndarray:
        states = unpack(variable_flat)
        pieces = []
        if include_defects:
            defects = []
            for node, pulse_width in enumerate(controls):
                expected = step(states[:, node], pulse_width)
                defects.append((states[:, node + 1] - expected) / scales[:, 0])
            pieces.append(sqrt_defect_weight * np.concatenate(defects))
        if include_proximity:
            proximity = (states[:, 1:] - warmup_variable) / scales
            pieces.append(sqrt_proximity_weight * proximity.reshape((-1,)))
        if not pieces:
            return np.zeros_like(variable_flat)
        return np.nan_to_num(
            np.concatenate(pieces),
            nan=1e20,
            posinf=1e20,
            neginf=-1e20,
        )

    result = least_squares(
        residual,
        x0,
        bounds=(lower_flat, upper_flat),
        max_nfev=max_iterations,
        method="trf",
        x_scale="jac",
        jac_sparsity=jac_sparsity,
    )
    projected_states = unpack(result.x)
    projected_states = np.minimum(
        np.maximum(projected_states, lower_bounds), upper_bounds
    )
    stats = {
        "success": bool(result.success),
        "status": float(result.status),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": float(result.nfev),
    }
    return projected_states, stats


def _sequential_project_periodic_states(
    muscle_model,
    original_states: np.ndarray,
    controls: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    dt: float,
    projection_mode: str,
    projection_substeps: int,
    proximity_weight: float,
    defect_weight: float,
    trust_radius: float | None,
) -> np.ndarray:
    if proximity_weight < 0.0:
        raise ValueError(
            "--periodic-fes-warmup-projection-proximity-weight must be >= 0."
        )
    if defect_weight < 0.0:
        raise ValueError("--periodic-fes-warmup-projection-defect-weight must be >= 0.")
    if trust_radius is not None and trust_radius < 0.0:
        raise ValueError("--periodic-fes-warmup-projection-trust-radius must be >= 0.")

    clipped_original = np.minimum(
        np.maximum(original_states, lower_bounds), upper_bounds
    )
    if proximity_weight == 0.0 and defect_weight == 0.0:
        return clipped_original

    projected_states = np.empty_like(clipped_original)
    projected_states[:, 0] = clipped_original[:, 0]
    scales = _projection_state_scales(clipped_original, lower_bounds, upper_bounds)
    denominator = proximity_weight + defect_weight
    for node, pulse_width in enumerate(controls):
        if projection_mode == "calcium":
            expected = _exact_periodic_calcium_step(
                muscle_model,
                projected_states[:, node],
                dt,
            )
        else:
            expected = _rk4_periodic_ding_step(
                muscle_model,
                projected_states[:, node],
                pulse_width,
                dt,
                n_substeps=projection_substeps,
            )
        next_values = (
            defect_weight * expected + proximity_weight * clipped_original[:, node + 1]
        ) / denominator
        lower = lower_bounds[:, node + 1]
        upper = upper_bounds[:, node + 1]
        if trust_radius is not None:
            center = clipped_original[:, node + 1]
            trust = trust_radius * scales[:, 0]
            lower = np.maximum(lower, center - trust)
            upper = np.minimum(upper, center + trust)
        projected_states[:, node + 1] = np.minimum(
            np.maximum(next_values, lower), upper
        )
    return projected_states


def project_periodic_fes_initial_guess(
    periodic_nmpc,
    projection_weight: float = 1.0,
    projection_mode: str = "all",
    projection_strategy: str = "sequential",
    projection_substeps: int = 10,
    projection_proximity_weight: float = 1.0,
    projection_defect_weight: float = 100.0,
    projection_trust_radius: float | None = None,
    projection_max_iterations: int = 200,
    force_projection_weight: float = 0.25,
    force_qdot_defect_limit: float = 3.0,
    force_adaptive_steps: int = 10,
) -> dict[str, float]:
    if projection_weight < 0.0 or projection_weight > 1.0:
        raise ValueError(
            "--periodic-fes-warmup-projection-weight must be between 0 and 1."
        )
    if force_projection_weight < 0.0 or force_projection_weight > 1.0:
        raise ValueError(
            "--periodic-fes-warmup-force-projection-weight must be between 0 and 1."
        )
    if projection_strategy not in ("rollout", "sequential", "least_squares"):
        raise ValueError(
            "--periodic-fes-warmup-projection-strategy must be 'rollout', 'sequential' or 'least_squares'."
        )

    dt = periodic_nmpc.cycle_duration / periodic_nmpc.cycle_len
    defects_before = _periodic_fes_rollout_defects(
        periodic_nmpc, projection_substeps=projection_substeps
    )
    projected_muscles = 0
    clipped_values = 0
    max_bound_violation = 0.0
    ls_failures = 0
    ls_total_nfev = 0
    ls_max_cost = 0.0
    ls_max_optimality = 0.0
    force_adaptive_summary = {}
    force_candidates = {}

    for muscle_model in periodic_nmpc.nlp[0].model.muscles_dynamics_model:
        muscle_name = muscle_model.muscle_name
        state_keys = _projection_state_keys(muscle_name, projection_mode)
        control_key = f"last_pulse_width_{muscle_name}"
        if any(key not in periodic_nmpc.nlp[0].x_init.keys() for key in state_keys):
            continue
        if control_key not in periodic_nmpc.nlp[0].u_init.keys():
            continue

        original_states = np.vstack(
            [periodic_nmpc.nlp[0].x_init[key].init[0, :] for key in state_keys]
        )
        controls = periodic_nmpc.nlp[0].u_init[control_key].init[0, :]
        lower_bounds = []
        upper_bounds = []
        for key in state_keys:
            lower, upper = _state_trajectory_bounds(
                periodic_nmpc, key, original_states.shape[1]
            )
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        lower_bounds = np.vstack(lower_bounds)
        upper_bounds = np.vstack(upper_bounds)
        if projection_strategy == "least_squares":
            projected_states, ls_stats = _bounded_least_squares_project_periodic_states(
                muscle_model,
                original_states,
                controls,
                lower_bounds,
                upper_bounds,
                dt,
                projection_mode,
                projection_substeps,
                projection_proximity_weight,
                projection_defect_weight,
                projection_trust_radius,
                projection_max_iterations,
            )
            ls_failures += 0 if ls_stats["success"] else 1
            ls_total_nfev += int(ls_stats["nfev"])
            ls_max_cost = max(ls_max_cost, ls_stats["cost"])
            ls_max_optimality = max(ls_max_optimality, ls_stats["optimality"])
        elif projection_strategy == "sequential":
            projected_states = _sequential_project_periodic_states(
                muscle_model,
                original_states,
                controls,
                lower_bounds,
                upper_bounds,
                dt,
                projection_mode,
                projection_substeps,
                projection_proximity_weight,
                projection_defect_weight,
                projection_trust_radius,
            )
        else:
            projected_states = _project_periodic_states(
                muscle_model,
                original_states,
                controls,
                dt,
                projection_mode,
                projection_substeps,
            )

        blended_states = (
            projection_weight * projected_states
            + (1.0 - projection_weight) * original_states
        )
        write_state_keys = _projection_write_state_keys(muscle_name, projection_mode)
        for state_idx, key in enumerate(state_keys):
            if key not in write_state_keys:
                continue
            if projection_mode == "all_force_adaptive_blend" and key.startswith("F_"):
                force_candidates[key] = (
                    original_states[state_idx, :].copy(),
                    projected_states[state_idx, :].copy(),
                )
                values = original_states[state_idx, :]
            elif projection_mode == "all_force_blend" and key.startswith("F_"):
                state_projection_weight = force_projection_weight
                values = (
                    state_projection_weight * projected_states[state_idx, :]
                    + (1.0 - state_projection_weight) * original_states[state_idx, :]
                )
            else:
                values = blended_states[state_idx, :]
            lower, upper = _state_trajectory_bounds(periodic_nmpc, key, values.size)
            lower_violation = np.maximum(lower - values, 0.0)
            upper_violation = np.maximum(values - upper, 0.0)
            violations = lower_violation + upper_violation
            clipped_values += int(np.count_nonzero(violations))
            max_bound_violation = max(max_bound_violation, float(np.max(violations)))
            periodic_nmpc.nlp[0].x_init[key].init[0, :] = np.minimum(
                np.maximum(values, lower), upper
            )
        projected_muscles += 1

    actual_force_projection_weight = force_projection_weight
    if force_candidates:
        force_adaptive_summary = _select_force_projection_weight_by_qdot_defect(
            periodic_nmpc,
            force_candidates,
            max_weight=force_projection_weight,
            qdot_defect_limit=force_qdot_defect_limit,
            n_steps=force_adaptive_steps,
        )
        actual_force_projection_weight = force_adaptive_summary["selected_weight"]

    defects_after = _periodic_fes_rollout_defects(
        periodic_nmpc, projection_substeps=projection_substeps
    )
    periodic_nmpc._correct_init_guess_to_fit_bounds(corrected_input="states")
    periodic_nmpc._sync_acados_state_bounds()

    max_before = max(defects_before.values(), default=0.0)
    max_after = max(defects_after.values(), default=0.0)
    return {
        "projected_muscles": projected_muscles,
        "projection_weight": projection_weight,
        "projection_mode": projection_mode,
        "projection_strategy": projection_strategy,
        "projection_substeps": projection_substeps,
        "force_projection_weight": actual_force_projection_weight,
        "force_projection_weight_upper": force_projection_weight,
        "force_qdot_defect_limit": force_qdot_defect_limit,
        "force_qdot_defect_after": force_adaptive_summary.get(
            "qdot_defect", float("nan")
        ),
        "force_adaptive_candidate_count": force_adaptive_summary.get(
            "candidate_count", 0.0
        ),
        "projection_proximity_weight": projection_proximity_weight,
        "projection_defect_weight": projection_defect_weight,
        "projection_trust_radius": (
            -1.0 if projection_trust_radius is None else projection_trust_radius
        ),
        "projection_max_iterations": projection_max_iterations,
        "max_defect_before": max_before,
        "max_defect_after": max_after,
        "clipped_values": clipped_values,
        "max_bound_violation": max_bound_violation,
        "least_squares_failures": ls_failures,
        "least_squares_total_nfev": ls_total_nfev,
        "least_squares_max_cost": ls_max_cost,
        "least_squares_max_optimality": ls_max_optimality,
    }


class _WarmupSolutionAdapter:
    def __init__(self, states: dict[str, np.ndarray], controls: dict[str, np.ndarray]):
        self._states = states
        self._controls = controls

    def decision_states(self, to_merge=None):
        return self._states

    def decision_controls(self, to_merge=None):
        return self._controls


def _resample_warmup_data(
    values: np.ndarray, target_len: int, has_terminal_node: bool
) -> np.ndarray:
    current_len = values.shape[1]
    if current_len == target_len:
        return values

    if has_terminal_node:
        if (
            current_len > 1
            and target_len > 1
            and (current_len - 1) % (target_len - 1) == 0
        ):
            stride = (current_len - 1) // (target_len - 1)
            return values[:, ::stride]
    else:
        if current_len % target_len == 0:
            stride = current_len // target_len
            return values[:, ::stride][:, :target_len]

    raise ValueError(
        f"Cannot resample warmup data of length {current_len} to target length {target_len} "
        f"(has_terminal_node={has_terminal_node})."
    )


def _adapt_warmup_solution_to_periodic_nodes(
    periodic_nmpc, warmup_solution
) -> _WarmupSolutionAdapter:
    warmup_states = warmup_solution.decision_states(to_merge=SolutionMerge.NODES)
    warmup_controls = warmup_solution.decision_controls(to_merge=SolutionMerge.NODES)

    first_state_key = next(iter(periodic_nmpc.nlp[0].x_init.keys()))
    first_control_key = next(iter(periodic_nmpc.nlp[0].u_init.keys()))
    target_state_len = periodic_nmpc.nlp[0].x_init[first_state_key].init.shape[1]
    target_control_len = periodic_nmpc.nlp[0].u_init[first_control_key].init.shape[1]

    adapted_states = {
        key: _resample_warmup_data(values, target_state_len, has_terminal_node=True)
        for key, values in warmup_states.items()
    }
    adapted_controls = {
        key: _resample_warmup_data(values, target_control_len, has_terminal_node=False)
        for key, values in warmup_controls.items()
    }
    return _WarmupSolutionAdapter(adapted_states, adapted_controls)


def _tile_one_cycle_state(
    key: str, values: np.ndarray, repeat_count: int
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.shape[1] < 2:
        raise ValueError(f"One-cycle state '{key}' must contain at least two nodes.")

    drift = values[:, -1:] - values[:, :1]
    pieces = [values]
    for cycle_index in range(1, repeat_count):
        repeated = values[:, 1:].copy()
        if key == "q":
            repeated[-1, :] += cycle_index * drift[-1, 0]
            repeated[:-1, :] += drift[:-1] * np.linspace(1.0, 0.0, repeated.shape[1])
        elif key.startswith(("F_", "A_", "Tau1_", "Km_")):
            repeated += cycle_index * drift
        else:
            repeated += drift * np.linspace(1.0, 0.0, repeated.shape[1])
        pieces.append(repeated)
    return np.concatenate(pieces, axis=1)


def _tile_one_cycle_control(values: np.ndarray, repeat_count: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    return np.tile(values, (1, repeat_count))


def _recenter_boundary_bounds(bounds, values: np.ndarray) -> None:
    for value_column, bound_column in ((0, 0), (-1, -1)):
        center = values[:, value_column]
        lower = np.asarray(bounds.min[:, bound_column], dtype=float)
        upper = np.asarray(bounds.max[:, bound_column], dtype=float)
        half_width = 0.5 * (upper - lower)
        finite = np.isfinite(half_width)
        bounds.min[finite, bound_column] = center[finite] - half_width[finite]
        bounds.max[finite, bound_column] = center[finite] + half_width[finite]


def _rollout_tiled_fes_states(
    periodic_nmpc, start_node: int, n_substeps: int = 10
) -> dict:
    """Propagate the non-periodic FES states after tiling a one-cycle solution."""

    nlp = periodic_nmpc.nlp[0]
    if not hasattr(nlp, "model") or not hasattr(nlp, "states"):
        return {"applied": False, "reason": "model_or_state_mapping_unavailable"}

    fes_keys = [
        key
        for key in nlp.x_init.keys()
        if key.startswith(("Cn_", "Cn_sum_", "F_", "A_", "Tau1_", "Km_"))
    ]
    if not fes_keys:
        return {"applied": False, "reason": "no_fes_states"}

    first_state_key = next(iter(nlp.x_init.keys()))
    first_control_key = next(iter(nlp.u_init.keys()))
    n_state_nodes = nlp.x_init[first_state_key].init.shape[1]
    n_control_nodes = nlp.u_init[first_control_key].init.shape[1]
    if n_state_nodes != n_control_nodes + 1:
        raise ValueError(
            "FES continuation rollout requires one more state node than controls."
        )
    if start_node < 0 or start_node >= n_control_nodes:
        raise ValueError("FES continuation rollout start node is outside the horizon.")

    states = _stack_initial_guess_values(nlp.x_init, nlp.states, n_state_nodes)
    controls = _stack_initial_guess_values(nlp.u_init, nlp.controls, n_control_nodes)
    fes_indexes = np.concatenate(
        [np.asarray(nlp.states[key].index).reshape((-1,)) for key in fes_keys]
    ).astype(int)
    tiled_fes = states[fes_indexes, :].copy()
    dt = periodic_nmpc.cycle_duration / periodic_nmpc.cycle_len

    for node in range(start_node, n_control_nodes):
        propagated = _rk4_full_dynamics_step(
            nlp,
            states[:, node],
            controls[:, node],
            node * dt,
            dt,
            n_substeps=n_substeps,
            numerical_timeseries=_numerical_timeseries_at_node(nlp, node),
        )
        states[fes_indexes, node + 1] = propagated[fes_indexes]

    max_change = float(np.max(np.abs(states[fes_indexes, :] - tiled_fes)))
    for key in fes_keys:
        indexes = np.asarray(nlp.states[key].index).reshape((-1,)).astype(int)
        nlp.x_init[key].init[:, :] = states[indexes, :]

    return {
        "applied": True,
        "state_count": len(fes_keys),
        "start_node": start_node,
        "substeps": n_substeps,
        "max_change": max_change,
    }


def tile_one_cycle_solution_to_periodic_nmpc(periodic_nmpc, one_cycle_solution) -> dict:
    states = one_cycle_solution.decision_states(to_merge=SolutionMerge.NODES)
    controls = one_cycle_solution.decision_controls(to_merge=SolutionMerge.NODES)
    nlp = periodic_nmpc.nlp[0]

    first_control_key = next(iter(nlp.u_init.keys()))
    source_control_nodes = np.asarray(controls[first_control_key]).shape[-1]
    target_control_nodes = nlp.u_init[first_control_key].init.shape[1]
    if source_control_nodes < 1 or target_control_nodes % source_control_nodes:
        raise ValueError(
            "The target control horizon must be an integer multiple of the one-cycle horizon "
            f"({target_control_nodes} versus {source_control_nodes})."
        )
    repeat_count = target_control_nodes // source_control_nodes
    if repeat_count < 2:
        raise ValueError(
            "Horizon continuation requires a target of at least two cycles."
        )

    seam_errors = {}
    for key in nlp.x_init.keys():
        if key not in states:
            continue
        source = np.asarray(states[key], dtype=float)
        if source.ndim == 1:
            source = source[np.newaxis, :]
        expected_source_nodes = source_control_nodes + 1
        if source.shape[1] != expected_source_nodes:
            raise ValueError(
                f"One-cycle state '{key}' has {source.shape[1]} nodes; "
                f"expected {expected_source_nodes}."
            )
        tiled = _tile_one_cycle_state(key, source, repeat_count)
        target = nlp.x_init[key].init
        if tiled.shape != target.shape:
            raise ValueError(
                f"Cannot tile state '{key}' with shape {tiled.shape} into {target.shape}."
            )
        target[:, :] = tiled

        source_first_step = source[:, 1] - source[:, 0]
        seam_first_step = (
            tiled[:, expected_source_nodes] - tiled[:, expected_source_nodes - 1]
        )
        seam_error = seam_first_step - source_first_step
        seam_errors[key] = float(np.max(np.abs(seam_error)))

    for key in nlp.u_init.keys():
        if key not in controls:
            continue
        tiled = _tile_one_cycle_control(controls[key], repeat_count)
        target = nlp.u_init[key].init
        if tiled.shape != target.shape:
            raise ValueError(
                f"Cannot tile control '{key}' with shape {tiled.shape} into {target.shape}."
            )
        target[:, :] = tiled

    for key in ("q", "qdot"):
        if key in nlp.x_bounds.keys() and key in nlp.x_init.keys():
            _recenter_boundary_bounds(nlp.x_bounds[key], nlp.x_init[key].init)

    fes_rollout = _rollout_tiled_fes_states(
        periodic_nmpc, start_node=source_control_nodes
    )
    fes_before_bound_correction = {
        key: np.asarray(nlp.x_init[key].init, dtype=float).copy()
        for key in nlp.x_init.keys()
        if key.startswith(("Cn_", "Cn_sum_", "F_", "A_", "Tau1_", "Km_"))
    }
    periodic_nmpc._correct_init_guess_to_fit_bounds(corrected_input="states")
    periodic_nmpc._correct_init_guess_to_fit_bounds(corrected_input="controls")
    periodic_nmpc._sync_acados_state_bounds()
    clipped_value_count = 0
    max_clip = 0.0
    for key, before in fes_before_bound_correction.items():
        difference = np.abs(np.asarray(nlp.x_init[key].init, dtype=float) - before)
        clipped_value_count += int(np.count_nonzero(difference > 1e-12))
        max_clip = max(max_clip, float(np.max(difference)))
    fes_rollout["clipped_value_count"] = clipped_value_count
    fes_rollout["max_clip"] = max_clip
    return {
        "repeat_count": repeat_count,
        "source_control_nodes": source_control_nodes,
        "target_control_nodes": target_control_nodes,
        "max_transfer_seam_error": max(seam_errors.values(), default=0.0),
        "seam_errors": seam_errors,
        "fes_rollout": fes_rollout,
    }


def _warmup_state_is_directly_comparable(key: str) -> bool:
    return key in ("q", "qdot") or key.startswith(("F_", "A_", "Tau1_", "Km_"))


def _resample_trace(values: np.ndarray, target_len: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == target_len:
        return values
    current_grid = np.linspace(0.0, 1.0, values.size)
    target_grid = np.linspace(0.0, 1.0, target_len)
    return np.interp(target_grid, current_grid, values)


def _expected_receding_state_initial_guess(
    periodic_nmpc, states: dict, key: str, row: int
) -> np.ndarray:
    state_values = np.asarray(states[key], dtype=float)
    if state_values.ndim == 1:
        state_values = state_values[np.newaxis, :]
    values = state_values[row, :]
    nodes_per_cycle = periodic_nmpc.nodes_per_cycle

    if key == "q" and row == state_values.shape[0] - 1:
        if not getattr(periodic_nmpc, "use_signed_wheel_shift", False):
            shifted_n_plus_one_cycles = (
                values[nodes_per_cycle:-1] + periodic_nmpc.pedal_turn_in_one_cycle
            )
            last_cycle = values[-nodes_per_cycle - 1 :]
            if shifted_n_plus_one_cycles.size == 0:
                return last_cycle
            return np.concatenate((shifted_n_plus_one_cycles, last_cycle))

        wheel_cycle_shift = periodic_nmpc._wheel_cycle_shift(states)
        n_plus_one_cycles = values[nodes_per_cycle:-1]
        shifted_last_cycle = values[-nodes_per_cycle - 1 :] + wheel_cycle_shift
        if n_plus_one_cycles.size == 0:
            return shifted_last_cycle
        return np.concatenate((n_plus_one_cycles, shifted_last_cycle))

    if key in ("q", "qdot") or key.startswith(("Cn_", "Cn_sum_", "F_")):
        n_plus_one_cycles = values[nodes_per_cycle:-1]
        last_cycle = values[-nodes_per_cycle - 1 :]
        if n_plus_one_cycles.size == 0:
            return last_cycle
        return np.concatenate((n_plus_one_cycles, last_cycle))

    if key.startswith(("A_", "Tau1_", "Km_")):
        n_plus_one_cycles = values[nodes_per_cycle:-1]
        last_cycle = values[-nodes_per_cycle - 1 :]
        if n_plus_one_cycles.size == 0:
            return last_cycle
        delta = n_plus_one_cycles[-1] - last_cycle[0]
        shifted_last_cycle = last_cycle + delta
        return np.concatenate((n_plus_one_cycles, shifted_last_cycle))

    return values


def _warmup_state_comparisons(reference_solution, periodic_nmpc) -> list[dict]:
    reference_states = reference_solution.decision_states(to_merge=SolutionMerge.NODES)
    comparisons = []

    for key in sorted(
        set(reference_states).intersection(periodic_nmpc.nlp[0].x_init.keys())
    ):
        if not _warmup_state_is_directly_comparable(key):
            continue

        reference_values = np.asarray(reference_states[key], dtype=float)
        candidate_values = np.asarray(
            periodic_nmpc.nlp[0].x_init[key].init, dtype=float
        )
        if reference_values.ndim == 1:
            reference_values = reference_values[np.newaxis, :]
        if candidate_values.ndim == 1:
            candidate_values = candidate_values[np.newaxis, :]

        for row in range(min(reference_values.shape[0], candidate_values.shape[0])):
            expected = _expected_receding_state_initial_guess(
                periodic_nmpc, reference_states, key, row
            )
            candidate = candidate_values[row, :]
            common_len = max(expected.size, candidate.size)
            expected_common = _resample_trace(expected, common_len)
            candidate_common = _resample_trace(candidate, common_len)
            diff = candidate_common - expected_common
            comparisons.append(
                {
                    "key": key if reference_values.shape[0] == 1 else f"{key}[{row}]",
                    "common_len": common_len,
                    "rmse": float(np.sqrt(np.mean(diff**2))),
                    "mae": float(np.mean(np.abs(diff))),
                    "max_abs_error": float(np.max(np.abs(diff))),
                    "final_error": float(diff[-1]),
                    "expected_mean": float(np.mean(expected_common)),
                    "initial_guess_mean": float(np.mean(candidate_common)),
                    "expected_range": (
                        float(np.min(expected_common)),
                        float(np.max(expected_common)),
                    ),
                    "initial_guess_range": (
                        float(np.min(candidate_common)),
                        float(np.max(candidate_common)),
                    ),
                }
            )

    return sorted(comparisons, key=lambda item: item["rmse"], reverse=True)


def print_warmup_state_comparison(
    label: str, reference_solution, periodic_nmpc, limit: int
) -> list[dict]:
    comparisons = _warmup_state_comparisons(reference_solution, periodic_nmpc)
    if limit <= 0:
        return comparisons

    print(
        f"warmup_state_comparison_note[{label}]: "
        "Cn/Cn_sum are omitted because the standard and periodic calcium states are not directly equivalent."
    )
    if not comparisons:
        print(
            f"warmup_state_comparison_warning[{label}]: no comparable state keys were found."
        )
        return comparisons

    print(
        f"warmup_state_comparison[{label}] | key | common_len | rmse | mae | max_abs_error | "
        "final_error | warmup_mean | initial_guess_mean | warmup_range | initial_guess_range"
    )
    for metric in comparisons[:limit]:
        warmup_min, warmup_max = metric["expected_range"]
        init_min, init_max = metric["initial_guess_range"]
        print(
            f"warmup_state_comparison[{label}] | "
            f"{metric['key']} | "
            f"{metric['common_len']} | "
            f"{metric['rmse']:.6g} | "
            f"{metric['mae']:.6g} | "
            f"{metric['max_abs_error']:.6g} | "
            f"{metric['final_error']:.6g} | "
            f"{metric['expected_mean']:.6g} | "
            f"{metric['initial_guess_mean']:.6g} | "
            f"[{warmup_min:.6g}, {warmup_max:.6g}] | "
            f"[{init_min:.6g}, {init_max:.6g}]"
        )

    return comparisons


def _cyclical_state_warmstart_values(
    values: np.ndarray, nodes_per_cycle: int
) -> np.ndarray:
    n_plus_one_cycles = values[nodes_per_cycle:-1]
    last_cycle = values[-nodes_per_cycle - 1 :]
    if n_plus_one_cycles.size == 0:
        return last_cycle
    return np.concatenate((n_plus_one_cycles, last_cycle))


def apply_fatigue_warmstart_mode(
    periodic_nmpc,
    adapted_solution,
    fatigue_warmstart_mode: str,
) -> dict[str, float]:
    if fatigue_warmstart_mode != "cyclical":
        return {}

    states = adapted_solution.decision_states(to_merge=SolutionMerge.NODES)
    max_delta_by_key = {}
    for key in periodic_nmpc.nlp[0].x_init.keys():
        if not key.startswith(("A_", "Tau1_", "Km_")) or key not in states:
            continue

        state_values = np.asarray(states[key], dtype=float)
        if state_values.ndim == 1:
            state_values = state_values[np.newaxis, :]
        for row in range(state_values.shape[0]):
            target = periodic_nmpc.nlp[0].x_init[key].init[row, :]
            values = _cyclical_state_warmstart_values(
                state_values[row, :], periodic_nmpc.nodes_per_cycle
            )
            if values.shape != target.shape:
                raise ValueError(
                    f"Cannot apply cyclical fatigue warmstart for {key}: "
                    f"expected {target.shape}, got {values.shape}."
                )
            max_delta_by_key[key] = max(
                max_delta_by_key.get(key, 0.0),
                float(np.max(np.abs(target - values))),
            )
            periodic_nmpc.nlp[0].x_init[key].init[row, :] = values

    if max_delta_by_key:
        periodic_nmpc._correct_init_guess_to_fit_bounds(corrected_input="states")
        periodic_nmpc._sync_acados_state_bounds()

    return max_delta_by_key


def apply_phase_shifted_warmup_initial_guess(
    periodic_nmpc,
    adapted_solution,
) -> None:
    states = adapted_solution.decision_states(to_merge=SolutionMerge.NODES)
    controls = adapted_solution.decision_controls(to_merge=SolutionMerge.NODES)
    wheel_shift = periodic_nmpc._wheel_cycle_shift(states)

    for key in periodic_nmpc.nlp[0].x_init.keys():
        if key not in states:
            continue
        values = np.asarray(states[key], dtype=float).copy()
        if key == "q":
            values[-1, :] += wheel_shift
        target = periodic_nmpc.nlp[0].x_init[key].init
        if values.shape != target.shape:
            raise ValueError(
                f"Cannot phase-shift warmup state {key}: expected {target.shape}, got {values.shape}."
            )
        target[:, :] = values

    for key in periodic_nmpc.nlp[0].u_init.keys():
        if key not in controls:
            continue
        values = np.asarray(controls[key], dtype=float)
        target = periodic_nmpc.nlp[0].u_init[key].init
        if values.shape != target.shape:
            raise ValueError(
                f"Cannot phase-shift warmup control {key}: expected {target.shape}, got {values.shape}."
            )
        target[:, :] = values

    periodic_nmpc._correct_init_guess_to_fit_bounds(corrected_input="states")
    periodic_nmpc._correct_init_guess_to_fit_bounds(corrected_input="controls")


def pulse_width_initial_guess_summary(periodic_nmpc) -> list[dict[str, float | str]]:
    summaries = []
    for key in periodic_nmpc.nlp[0].u_init.keys():
        if not key.startswith("last_pulse_width_"):
            continue
        values = np.asarray(periodic_nmpc.nlp[0].u_init[key].init, dtype=float)
        summaries.append(
            {
                "key": key,
                "minimum": float(np.min(values)),
                "mean": float(np.mean(values)),
                "maximum": float(np.max(values)),
                "span": float(np.ptp(values)),
            }
        )
    return summaries


def apply_standard_warmup_to_periodic_nmpc(
    periodic_nmpc,
    warmup_solution,
    project_fes_warmup: bool = True,
    fatigue_warmstart_mode: str = "continuous",
    projection_weight: float = 1.0,
    projection_mode: str = "all",
    projection_strategy: str = "sequential",
    projection_substeps: int = 10,
    projection_proximity_weight: float = 1.0,
    projection_defect_weight: float = 100.0,
    projection_trust_radius: float | None = None,
    projection_max_iterations: int = 200,
    force_projection_weight: float = 0.25,
    force_qdot_defect_limit: float = 3.0,
    force_adaptive_steps: int = 10,
    warmup_transfer_mode: str = "phase_shift",
    echo: bool = False,
):
    if fatigue_warmstart_mode not in ("continuous", "cyclical"):
        raise ValueError(
            "--acados-fatigue-warmstart-mode must be 'continuous' or 'cyclical'."
        )

    adapted_solution = _adapt_warmup_solution_to_periodic_nodes(
        periodic_nmpc, warmup_solution
    )
    periodic_nmpc.advance_window_bounds_states(adapted_solution)
    if warmup_transfer_mode == "phase_shift":
        apply_phase_shifted_warmup_initial_guess(periodic_nmpc, adapted_solution)
        fatigue_warmstart_summary = {}
    elif warmup_transfer_mode == "advance":
        previous_fatigue_warmstart_mode = getattr(
            periodic_nmpc, "continuous_state_initial_guess_mode", "continuous"
        )
        periodic_nmpc.continuous_state_initial_guess_mode = fatigue_warmstart_mode
        try:
            periodic_nmpc.advance_window_initial_guess_states(adapted_solution)
        finally:
            periodic_nmpc.continuous_state_initial_guess_mode = (
                previous_fatigue_warmstart_mode
            )
        fatigue_warmstart_summary = apply_fatigue_warmstart_mode(
            periodic_nmpc, adapted_solution, fatigue_warmstart_mode
        )
        periodic_nmpc.advance_window_initial_guess_controls(adapted_solution)
    else:
        raise ValueError(
            "--acados-standard-warmup-transfer must be 'advance' or 'phase_shift'."
        )
    if echo:
        for summary in pulse_width_initial_guess_summary(periodic_nmpc):
            print(
                "warmup_pulse_width: "
                f"{summary['key']} "
                f"min={summary['minimum']:.9g} "
                f"mean={summary['mean']:.9g} "
                f"max={summary['maximum']:.9g} "
                f"span={summary['span']:.9g}"
            )
    if echo and fatigue_warmstart_summary:
        print(
            "acados_fatigue_warmstart: "
            f"mode={fatigue_warmstart_mode} "
            f"max_delta={max(fatigue_warmstart_summary.values()):.6g} "
            f"keys={len(fatigue_warmstart_summary)}"
        )
    warmup_states = adapted_solution.decision_states(to_merge=SolutionMerge.NODES)
    dt = periodic_nmpc.cycle_duration / periodic_nmpc.cycle_len
    muscle_models = {
        model.muscle_name: model
        for model in periodic_nmpc.nlp[0].model.muscles_dynamics_model
    }
    for key in periodic_nmpc.nlp[0].states.keys():
        if not key.startswith("Cn_sum_"):
            continue

        source_key = key.replace("Cn_sum_", "Cn_")
        if source_key not in warmup_states:
            continue

        muscle_name = key.replace("Cn_sum_", "")
        tauc = muscle_models[muscle_name].tauc
        source_values = warmup_states[source_key][0]
        cn_sum_values = estimate_periodic_cn_sum_from_cn(
            source_values, tauc=tauc, dt=dt
        )
        shifted_values = _shift_cyclical_trajectory(
            cn_sum_values, periodic_nmpc.nodes_per_cycle
        )
        periodic_nmpc.nlp[0].x_init[key].init[0, :] = shifted_values
        if getattr(periodic_nmpc, "bound_first_node_all_states", True):
            center = cn_sum_values[periodic_nmpc.nodes_per_cycle]
            slack = (
                periodic_nmpc._state_slack_for(key, 0)
                if hasattr(periodic_nmpc, "_state_slack_for")
                else 0.0
            )
            periodic_nmpc.nlp[0].x_bounds[key].min[0, 0] = center - slack
            periodic_nmpc.nlp[0].x_bounds[key].max[0, 0] = center + slack

    if project_fes_warmup:
        projection_summary = project_periodic_fes_initial_guess(
            periodic_nmpc,
            projection_weight=projection_weight,
            projection_mode=projection_mode,
            projection_strategy=projection_strategy,
            projection_substeps=projection_substeps,
            projection_proximity_weight=projection_proximity_weight,
            projection_defect_weight=projection_defect_weight,
            projection_trust_radius=projection_trust_radius,
            projection_max_iterations=projection_max_iterations,
            force_projection_weight=force_projection_weight,
            force_qdot_defect_limit=force_qdot_defect_limit,
            force_adaptive_steps=force_adaptive_steps,
        )
        if echo:
            print(
                "periodic_fes_warmup_projection: "
                f"projected_muscles={projection_summary['projected_muscles']} "
                f"mode={projection_summary['projection_mode']} "
                f"strategy={projection_summary['projection_strategy']} "
                f"weight={projection_summary['projection_weight']:.3g} "
                f"force_weight={projection_summary['force_projection_weight']:.3g} "
                f"force_weight_upper={projection_summary['force_projection_weight_upper']:.3g} "
                f"substeps={projection_summary['projection_substeps']} "
                f"max_defect_before={projection_summary['max_defect_before']:.6g} "
                f"max_defect_after={projection_summary['max_defect_after']:.6g} "
                f"clipped_values={projection_summary['clipped_values']} "
                f"max_bound_violation={projection_summary['max_bound_violation']:.6g}"
            )
            if projection_summary["projection_mode"] == "all_force_adaptive_blend":
                print(
                    "periodic_fes_warmup_force_adaptive: "
                    f"qdot_defect={projection_summary['force_qdot_defect_after']:.6g} "
                    f"qdot_defect_limit={projection_summary['force_qdot_defect_limit']:.6g} "
                    f"candidate_count={int(projection_summary['force_adaptive_candidate_count'])}"
                )
            if projection_summary["projection_strategy"] == "least_squares":
                print(
                    "periodic_fes_warmup_projection_ls: "
                    f"failures={projection_summary['least_squares_failures']} "
                    f"total_nfev={projection_summary['least_squares_total_nfev']} "
                    f"max_cost={projection_summary['least_squares_max_cost']:.6g} "
                    f"max_optimality={projection_summary['least_squares_max_optimality']:.6g}"
                )

    return adapted_solution


def apply_solution_directly_to_periodic_nmpc_initial_guess(
    periodic_nmpc, solution, recenter_kinematic_bounds: bool = False
):
    adapted_solution = _adapt_warmup_solution_to_periodic_nodes(periodic_nmpc, solution)
    states = adapted_solution.decision_states(to_merge=SolutionMerge.NODES)
    controls = adapted_solution.decision_controls(to_merge=SolutionMerge.NODES)

    for key in periodic_nmpc.nlp[0].x_init.keys():
        if key not in states:
            continue
        values = np.asarray(states[key], dtype=float)
        target = periodic_nmpc.nlp[0].x_init[key].init
        if values.shape != target.shape:
            raise ValueError(
                f"Cannot copy refined state '{key}' with shape {values.shape} "
                f"into initial guess shape {target.shape}."
            )
        target[:, :] = values

    for key in periodic_nmpc.nlp[0].u_init.keys():
        if key not in controls:
            continue
        values = np.asarray(controls[key], dtype=float)
        target = periodic_nmpc.nlp[0].u_init[key].init
        if values.shape != target.shape:
            raise ValueError(
                f"Cannot copy refined control '{key}' with shape {values.shape} "
                f"into initial guess shape {target.shape}."
            )
        target[:, :] = values

    if recenter_kinematic_bounds:
        for key in ("q", "qdot"):
            if key in periodic_nmpc.nlp[0].x_bounds.keys():
                _recenter_boundary_bounds(
                    periodic_nmpc.nlp[0].x_bounds[key],
                    periodic_nmpc.nlp[0].x_init[key].init,
                )

    periodic_nmpc._correct_init_guess_to_fit_bounds(corrected_input="states")
    periodic_nmpc._correct_init_guess_to_fit_bounds(corrected_input="controls")
    periodic_nmpc._sync_acados_state_bounds()
    return adapted_solution


def _copy_list_values(source, target, attribute_name: str) -> None:
    source_keys = set(source.keys())
    for key in target.keys():
        if key not in source_keys:
            continue
        source_values = np.asarray(getattr(source[key], attribute_name), dtype=float)
        target_values = getattr(target[key], attribute_name)
        if source_values.shape != target_values.shape:
            raise ValueError(
                f"Cannot copy '{key}' {attribute_name} with shape {source_values.shape} "
                f"into shape {target_values.shape}."
            )
        target_values[:, :] = source_values


def _copy_initial_guesses_and_bounds(source_nmpc, target_nmpc) -> None:
    _copy_list_values(source_nmpc.nlp[0].x_init, target_nmpc.nlp[0].x_init, "init")
    _copy_list_values(source_nmpc.nlp[0].u_init, target_nmpc.nlp[0].u_init, "init")
    _copy_list_values(source_nmpc.nlp[0].x_bounds, target_nmpc.nlp[0].x_bounds, "min")
    _copy_list_values(source_nmpc.nlp[0].x_bounds, target_nmpc.nlp[0].x_bounds, "max")
    _copy_list_values(source_nmpc.nlp[0].u_bounds, target_nmpc.nlp[0].u_bounds, "min")
    _copy_list_values(source_nmpc.nlp[0].u_bounds, target_nmpc.nlp[0].u_bounds, "max")


def _copy_objective_targets(source_nmpc, target_nmpc) -> None:
    for source_penalty, target_penalty in zip(
        source_nmpc.nlp[0].J, target_nmpc.nlp[0].J
    ):
        if not source_penalty or not target_penalty:
            continue
        if getattr(source_penalty, "target", None) is not None:
            target_penalty.target = np.array(source_penalty.target, copy=True)


def _copy_periodic_runtime_settings(source_nmpc, target_nmpc) -> None:
    target_nmpc.n_cycles_simultaneous = source_nmpc.n_cycles_simultaneous
    for attribute_name in (
        "first_node_state_slack",
        "bound_first_node_all_states",
        "bound_first_node_wheel_qdot",
        "advance_wheel_q_bounds",
        "wheel_q_path_margin",
        "use_signed_wheel_shift",
    ):
        if hasattr(source_nmpc, attribute_name):
            setattr(
                target_nmpc,
                attribute_name,
                deepcopy(getattr(source_nmpc, attribute_name)),
            )
    target_nmpc.transfer_debug = False


def build_periodic_ipopt_refinement_nmpc(
    source_nmpc,
    model_path: Path,
    stim_time: list[float],
    mhe_info: dict,
    cycling_info: dict,
    simulation_conditions: dict,
    model_formulation: str,
    refinement_ode_solver=None,
):
    refinement_model = set_fes_model(
        str(model_path),
        stim_time,
        periodic_cn_sum_approximation=model_formulation == "periodic",
        periodic_node_forcing=model_formulation == "periodic_node",
    )
    refinement_mhe_info = dict(mhe_info)
    if refinement_ode_solver is not None:
        refinement_mhe_info["ode_solver"] = refinement_ode_solver
    refinement_nmpc = prepare_nmpc(
        refinement_model,
        refinement_mhe_info,
        dict(cycling_info),
        dict(simulation_conditions),
    )
    _copy_periodic_runtime_settings(source_nmpc, refinement_nmpc)
    _copy_initial_guesses_and_bounds(source_nmpc, refinement_nmpc)
    _copy_objective_targets(source_nmpc, refinement_nmpc)
    refinement_nmpc._correct_init_guess_to_fit_bounds(corrected_input="states")
    refinement_nmpc._correct_init_guess_to_fit_bounds(corrected_input="controls")
    return refinement_nmpc


def run_periodic_ipopt_refinement(
    refinement_nmpc,
    target_nmpc,
    max_iterations: int,
    linear_solver: str,
    cache_path: Path | None = None,
    echo: bool = False,
):
    solver = configure_ipopt_solver(
        max_iterations=max_iterations,
        linear_solver=linear_solver,
    )
    try:
        refinement_sol = super(RecedingHorizonOptimization, refinement_nmpc).solve(
            solver=solver,
            warm_start=None,
        )
    except Exception as exc:
        if echo:
            print("periodic_ipopt_refinement_error: " f"{type(exc).__name__}: {exc}")
            print("periodic_ipopt_refinement_applied: False")
        return None

    success = _status_is_success(refinement_sol.status)
    if echo:
        cost = (
            float(np.nansum(refinement_sol.cost))
            if getattr(refinement_sol, "cost", None) is not None
            else float("nan")
        )
        print(f"periodic_ipopt_refinement_status: {refinement_sol.status}")
        print(f"periodic_ipopt_refinement_success: {success}")
        print(f"periodic_ipopt_refinement_cost: {cost:.6g}")
        print(
            "periodic_ipopt_refinement_solver_time_s: "
            f"{refinement_sol.solver_time_to_optimize}"
        )
        print(
            "periodic_ipopt_refinement_wall_time_s: "
            f"{refinement_sol.real_time_to_optimize}"
        )

    if success:
        if cache_path is not None:
            _save_warmup_cache(cache_path, refinement_sol)
            if echo:
                print(f"periodic_ipopt_refinement_cache: saved ({cache_path.name})")
        apply_solution_directly_to_periodic_nmpc_initial_guess(
            target_nmpc, refinement_sol
        )
        if echo:
            print("periodic_ipopt_refinement_applied: True")
    elif echo:
        print("periodic_ipopt_refinement_applied: False")

    return refinement_sol


def apply_warmup_control_regularization_targets(
    periodic_nmpc, adapted_warmup_solution
) -> list[str]:
    warmup_controls = adapted_warmup_solution.decision_controls(
        to_merge=SolutionMerge.NODES
    )
    updated_keys = []
    for penalty in periodic_nmpc.nlp[0].J:
        if not penalty:
            continue

        key = getattr(penalty, "extra_parameters", {}).get("key")
        if key not in warmup_controls:
            continue

        target = np.asarray(warmup_controls[key], dtype=float)
        if target.ndim == 1:
            target = target[np.newaxis, :]

        target_len = len(penalty.node_idx)
        if target.shape[1] == target_len - 1:
            target = np.concatenate((target, target[:, -1:]), axis=1)
        elif target.shape[1] != target_len:
            target = _resample_warmup_data(target, target_len, has_terminal_node=False)

        penalty.target = target
        updated_keys.append(key)

    return updated_keys


def apply_pulse_width_control_trust_region(
    periodic_nmpc, radius: float
) -> dict[str, dict[str, float]]:
    if radius < 0:
        raise ValueError("--acados-pulse-width-trust-radius must be non-negative.")

    if not hasattr(periodic_nmpc, "_cocofest_original_control_bounds"):
        periodic_nmpc._cocofest_original_control_bounds = {
            key: (
                np.array(
                    periodic_nmpc.nlp[0].u_bounds[key].min, dtype=float, copy=True
                ),
                np.array(
                    periodic_nmpc.nlp[0].u_bounds[key].max, dtype=float, copy=True
                ),
            )
            for key in periodic_nmpc.nlp[0].u_init.keys()
            if key.startswith("last_pulse_width_")
        }

    summary = {}
    nodewise_bounds = {}
    for key in periodic_nmpc.nlp[0].u_init.keys():
        if not key.startswith("last_pulse_width_"):
            continue

        center = np.asarray(periodic_nmpc.nlp[0].u_init[key].init, dtype=float)
        bounds = periodic_nmpc.nlp[0].u_bounds[key]
        original_min, original_max = periodic_nmpc._cocofest_original_control_bounds[
            key
        ]
        original_lower = float(np.min(original_min))
        original_upper = float(np.max(original_max))
        center_min = float(np.min(center))
        center_max = float(np.max(center))
        lower = np.maximum(original_lower, center - radius)
        upper = np.minimum(original_upper, center + radius)
        if np.any(lower > upper):
            raise RuntimeError(f"Pulse-width trust region is empty for {key}.")

        bounds.min[:, :] = float(np.min(lower))
        bounds.max[:, :] = float(np.max(upper))
        periodic_nmpc.nlp[0].u_init[key].init[:, :] = np.minimum(
            np.maximum(center, lower), upper
        )
        nodewise_bounds[key] = (lower, upper)
        summary[key] = {
            "center_min": center_min,
            "center_max": center_max,
            "lower": float(np.min(lower)),
            "upper": float(np.max(upper)),
        }

    periodic_nmpc._cocofest_nodewise_control_bounds = nodewise_bounds
    return summary


def apply_fes_state_trust_region(
    periodic_nmpc, normalized_radius: float
) -> dict[str, dict[str, float]]:
    if normalized_radius < 0:
        raise ValueError("--acados-fes-state-trust-radius must be non-negative.")

    summary = {}
    for key in periodic_nmpc.nlp[0].x_init.keys():
        if not key.startswith(("Cn_", "Cn_sum_", "F_", "A_", "Tau1_", "Km_")):
            continue

        center = np.asarray(periodic_nmpc.nlp[0].x_init[key].init, dtype=float)
        bounds = periodic_nmpc.nlp[0].x_bounds[key]
        original_lower = float(np.min(np.asarray(bounds.min, dtype=float)))
        original_upper = float(np.max(np.asarray(bounds.max, dtype=float)))
        center_min = float(np.min(center))
        center_max = float(np.max(center))
        scale = max(float(np.max(np.abs(center))), 1.0)
        absolute_radius = normalized_radius * scale
        lower = max(original_lower, center_min - absolute_radius)
        upper = min(original_upper, center_max + absolute_radius)
        if lower > upper:
            raise RuntimeError(
                f"FES state trust region is empty for {key}: lower={lower}, upper={upper}."
            )

        bounds.min[:, :] = lower
        bounds.max[:, :] = upper
        periodic_nmpc.nlp[0].x_init[key].init[:, :] = np.minimum(
            np.maximum(center, lower), upper
        )
        summary[key] = {
            "center_min": center_min,
            "center_max": center_max,
            "lower": lower,
            "upper": upper,
            "scale": scale,
        }

    periodic_nmpc._sync_acados_state_bounds()
    return summary


def relax_acados_first_node_fes_bounds(periodic_nmpc) -> list[str]:
    relaxed_keys = []
    for key in periodic_nmpc.nlp[0].x_bounds.keys():
        if key in ("q", "qdot"):
            continue

        bounds = periodic_nmpc.nlp[0].x_bounds[key]
        if bounds.min.shape[1] < 2 or bounds.max.shape[1] < 2:
            continue

        bounds.min[:, 0] = bounds.min[:, 1]
        bounds.max[:, 0] = bounds.max[:, 1]
        relaxed_keys.append(key)

    periodic_nmpc._sync_acados_state_bounds()
    return relaxed_keys


def run_standard_ipopt_warmup(
    args: argparse.Namespace,
    mhe_info: dict,
    cycling_info: dict,
    simulation_conditions: dict,
    model_path: Path,
):
    cache_path = _warmup_cache_path(
        args, model_path, simulation_conditions, cycling_info
    )
    if cache_path.exists():
        print(f"warmup_cache: hit ({cache_path.name})")
        return _load_warmup_cache(cache_path)

    warmup_mhe_info = dict(mhe_info)
    warmup_mhe_info["ode_solver"] = OdeSolver.COLLOCATION(
        polynomial_degree=3, method="radau"
    )
    warmup_mhe_info["use_sx"] = False

    warmup_cycling_info = dict(cycling_info)
    warmup_cycling_info["enforce_start_constraints"] = True

    stim_time = list(
        np.linspace(
            0,
            warmup_mhe_info["cycle_duration"] * args.cycles_per_window,
            args.stimulations_per_cycle * args.cycles_per_window,
            endpoint=False,
        )
    )
    warmup_model = set_fes_model(
        str(model_path), stim_time, periodic_cn_sum_approximation=False
    )
    warmup_nmpc = prepare_nmpc(
        warmup_model, warmup_mhe_info, warmup_cycling_info, dict(simulation_conditions)
    )
    warmup_nmpc.n_cycles_simultaneous = args.cycles_per_window

    warmup_solver = configure_ipopt_solver(
        max_iterations=args.max_ipopt_iterations,
        linear_solver=args.ipopt_linear_solver,
    )
    warmup_sol = super(RecedingHorizonOptimization, warmup_nmpc).solve(
        solver=warmup_solver,
        warm_start=None,
    )
    _save_warmup_cache(cache_path, warmup_sol)
    print(f"warmup_cache: saved ({cache_path.name})")
    return warmup_sol


def build_codegen_names(args: argparse.Namespace) -> tuple[str, str]:
    objective_slug = args.objective.replace(",", "_")
    signature = _codegen_signature(args)
    suffix = args.codegen_tag or (
        f"{args.solver}_{args.model_formulation}_{objective_slug}_{args.objective_shape}_{args.n_windows}mhe_{args.cycles_per_window}cyc"
    )
    return (
        f"cycling_fes_periodic_{suffix}_{signature}",
        f"result/acados/c_generated_code_{suffix}_{signature}",
    )


def get_one_cycle_acados_continuation_source(
    args: argparse.Namespace, echo: bool
) -> _WarmupSolutionAdapter:
    cache_path = _continuation_cache_path(args)
    if cache_path.exists():
        if echo:
            print(f"acados_continuation_cache: hit ({cache_path.name})")
        return _load_warmup_cache(cache_path)

    source_args = deepcopy(args)
    source_args.cycles_per_window = 1
    source_args.n_windows = 1
    source_args.single_shot = True
    source_args.acados_horizon_continuation = False
    source_args.max_acados_iterations = args.acados_continuation_source_max_iterations
    source_args.acados_diagnostics = False
    source_args.codegen_tag = (
        f"{args.codegen_tag}_continuation_1cyc"
        if args.codegen_tag
        else "continuation_1cyc"
    )

    if echo:
        print("running_acados_one_cycle_continuation_source: True")
    source_result = solve_case(source_args, echo=echo)
    source_status = source_result.get("status")
    if not _status_is_success(source_status):
        raise RuntimeError(
            "The one-cycle ACADOS continuation source did not converge: "
            f"status={source_status} ({ACADOS_STATUS_NAMES.get(source_status, 'unknown')})."
        )

    source_solution = source_result["solution"]
    _save_warmup_cache(cache_path, source_solution)
    if echo:
        print(f"acados_continuation_cache: saved ({cache_path.name})")
    return _load_warmup_cache(cache_path)


def solve_case(args: argparse.Namespace, echo: bool = True) -> dict:
    objectives = parse_objectives(args.objective)

    if args.n_windows < 1:
        raise ValueError("--n-windows must be >= 1")
    if args.cycles_per_window < 1:
        raise ValueError("--cycles-per-window must be >= 1")
    if args.stimulations_per_cycle < 1:
        raise ValueError("--stimulations-per-cycle must be >= 1")
    if args.acados_continuation_source_max_iterations < 1:
        raise ValueError(
            "--acados-continuation-source-max-iterations must be strictly positive."
        )
    if args.acados_horizon_continuation and args.cycles_per_window < 2:
        raise ValueError(
            "--acados-horizon-continuation requires --cycles-per-window >= 2."
        )
    if args.acados_horizon_continuation and args.solver != "acados":
        raise ValueError("--acados-horizon-continuation requires --solver acados.")
    if (
        args.acados_stationarity_tolerance is not None
        and args.acados_stationarity_tolerance <= 0
    ):
        raise ValueError("--acados-stationarity-tolerance must be strictly positive.")
    for option_name in (
        "acados_first_window_tolerance",
        "acados_first_window_stationarity_tolerance",
    ):
        value = getattr(args, option_name)
        if value is not None and value <= 0:
            raise ValueError(
                f"--{option_name.replace('_', '-')} must be strictly positive."
            )
    if args.acados_warm_start_first_qp_from_nlp and not args.acados_warm_start_first_qp:
        raise ValueError(
            "--acados-warm-start-first-qp-from-nlp requires "
            "--acados-warm-start-first-qp."
        )
    if args.acados_warm_start_first_qp and args.acados_qp_warm_start_level == 0:
        raise ValueError(
            "--acados-warm-start-first-qp requires a non-zero "
            "--acados-qp-warm-start-level."
        )
    if (
        args.acados_pulse_width_trust_radius is not None
        and args.acados_pulse_width_trust_radius < 0
    ):
        raise ValueError("--acados-pulse-width-trust-radius must be non-negative.")
    if (
        args.acados_transfer_pulse_width_trust_radius is not None
        and args.acados_transfer_pulse_width_trust_radius < 0
    ):
        raise ValueError(
            "--acados-transfer-pulse-width-trust-radius must be non-negative."
        )
    if args.acados_transfer_rollout_substeps < 1:
        raise ValueError("--acados-transfer-rollout-substeps must be >= 1.")
    if args.acados_transfer_rollout_max_bound_violation < 0:
        raise ValueError(
            "--acados-transfer-rollout-max-bound-violation must be non-negative."
        )
    if args.acados_fixed_control_tolerance <= 0:
        raise ValueError("--acados-fixed-control-tolerance must be strictly positive.")

    periodic_ipopt_refinement_enabled = (
        args.periodic_ipopt_refinement and not args.disable_periodic_ipopt_refinement
    )
    periodic_ipopt_reference_solution = None

    continuation_source = None
    horizon_seed = None
    horizon_seed_cache_path = None
    if args.acados_horizon_continuation:
        horizon_seed_cache_path = _horizon_seed_cache_path(args)
        if horizon_seed_cache_path.exists():
            horizon_seed = _load_warmup_cache(horizon_seed_cache_path)
            if echo:
                print(
                    f"acados_horizon_seed_cache: hit ({horizon_seed_cache_path.name})"
                )
        else:
            continuation_source = get_one_cycle_acados_continuation_source(
                args, echo=echo
            )

    example_dir = Path(__file__).resolve().parent
    model_path = (
        example_dir / "../../msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod"
    )
    acados_seed_cache_path = (
        _acados_seed_cache_path(args, model_path) if args.solver == "acados" else None
    )
    cycle_duration = 1.0
    total_window_duration = cycle_duration * args.cycles_per_window
    total_stimulations = args.stimulations_per_cycle * args.cycles_per_window
    stim_time = list(
        np.linspace(0, total_window_duration, total_stimulations, endpoint=False)
    )
    periodic_cn_sum_approximation = args.model_formulation != "standard"
    use_external_forces = args.torque_application == "external_forces"
    ode_solver = build_ode_solver(args)
    historical_init_guess_path = None
    adapted_warmup_solution = None
    if args.solver == "ipopt" and not args.disable_historical_ipopt_initial_guess:
        historical_init_guess_path = _historical_initial_guess_path(
            args.cycles_per_window, ode_solver
        )
    model = set_fes_model(
        str(model_path),
        stim_time,
        periodic_cn_sum_approximation=args.model_formulation == "periodic",
        periodic_node_forcing=args.model_formulation == "periodic_node",
    )

    mhe_info = {
        "cycle_duration": cycle_duration,
        "n_cycles_to_advance": 1,
        "n_cycles": args.n_windows,
        "ode_solver": ode_solver,
        "use_sx": args.use_sx,
        "cycle_len": args.stimulations_per_cycle,
        "n_cycles_simultaneous": args.cycles_per_window,
    }
    cycling_info = {
        "turn_number": args.cycles_per_window,
        "pedal_config": {"x_center": 0.35, "y_center": 0.0, "radius": 0.1},
        "enforce_start_constraints": args.enforce_start_constraints,
        "periodic_cn_sum_approximation": periodic_cn_sum_approximation,
    }
    if use_external_forces:
        cycling_info["resistive_torque"] = {
            "Segment_application": "wheel",
            "torque": np.array([0.0, 0.0, args.constant_crank_torque]),
        }
    else:
        cycling_info["constant_crank_torque"] = args.constant_crank_torque
    simulation_conditions = {
        "n_cycles_simultaneous": args.cycles_per_window,
        "stimulation": total_stimulations,
        "minimize_force": "force" in objectives,
        "minimize_fatigue": "fatigue" in objectives,
        "minimize_control": "control" in objectives,
        "cost_fun_weight": build_cost_fun_weight(objectives),
        "objective_shape": args.objective_shape,
        "control_regularization_weight": args.control_regularization_weight,
        "control_regularization_target": args.control_regularization_target,
        "wheel_qdot_regularization_weight": args.wheel_qdot_regularization_weight,
        "wheel_qdot_regularization_target": args.wheel_qdot_regularization_target,
        "state_scaling": args.state_scaling,
        "pulse_width_scaling": args.pulse_width_scaling,
        "init_guess_file_path": (
            str(historical_init_guess_path)
            if historical_init_guess_path is not None
            else None
        ),
    }
    nmpc_simulation_conditions = dict(simulation_conditions)
    if (
        args.solver == "acados"
        and args.control_regularization_target_source == "warmup"
    ):
        nmpc_simulation_conditions["control_regularization_target"] = None

    nmpc = prepare_nmpc(model, mhe_info, cycling_info, nmpc_simulation_conditions)
    nmpc.n_cycles_simultaneous = args.cycles_per_window
    if args.solver == "acados":
        patch_bioptim_acados_interface()
        nmpc.first_node_state_slack = {
            "q": [0.0, 0.0, args.acados_wheel_q_slack],
            "qdot": [0.0, 0.0, args.acados_wheel_qdot_slack],
            "Cn_": 5e-3,
            "Cn_sum_": 5e-3,
            "F_": 1e-2,
            "A_": 5e-3,
            "Tau1_": 5e-3,
            "Km_": 5e-3,
        }
        # The periodic-node states have the same physical meaning as the
        # historical Ding states. Keep their initial value near the IPOPT
        # warmup; otherwise ACADOS can manufacture a low-cost but nonphysical
        # initial fatigue state that cannot be continued to another cycle.
        nmpc.bound_first_node_all_states = args.model_formulation == "periodic_node"
        nmpc.bound_first_node_wheel_qdot = False
        nmpc.advance_wheel_q_bounds = True
        nmpc.wheel_q_path_margin = args.acados_wheel_q_path_margin
        nmpc.use_signed_wheel_shift = True
        nmpc.transfer_initial_guess_mode = "anchored"
        nmpc.transfer_debug = echo
        nmpc._cocofest_dual_warm_start_mode = args.acados_dual_warm_start_mode
        nmpc._cocofest_dual_shift_stages = args.stimulations_per_cycle
        relaxed_fes_bounds = []
        if not nmpc.bound_first_node_all_states:
            relaxed_fes_bounds = relax_acados_first_node_fes_bounds(nmpc)
        if echo and relaxed_fes_bounds:
            print(
                "acados_relaxed_first_node_fes_bounds: "
                f"keys={len(relaxed_fes_bounds)}"
            )

    if echo:
        print(f"model_formulation: {args.model_formulation}")
        print(f"torque_application: {args.torque_application}")
        print(f"resistive_torque_nm: {args.constant_crank_torque}")
        print(f"single_shot: {args.single_shot}")
        print(f"ode_solver: {args.ode_solver}")
        if args.ode_solver in ("rk4", "rk8"):
            print(f"rk_steps: {args.rk_steps}")
        else:
            print(f"collocation_degree: {args.collocation_degree}")
            print(f"collocation_method: {args.collocation_method}")
        print(f"use_sx: {args.use_sx}")
        print(f"enforce_start_constraints: {args.enforce_start_constraints}")
        print(f"control_regularization_weight: {args.control_regularization_weight}")
        print(f"control_regularization_target: {args.control_regularization_target}")
        print(
            "control_regularization_target_source: "
            f"{args.control_regularization_target_source}"
        )
        print(
            f"wheel_qdot_regularization_weight: {args.wheel_qdot_regularization_weight}"
        )
        print(
            f"wheel_qdot_regularization_target: {args.wheel_qdot_regularization_target}"
        )
        print(f"state_scaling: {args.state_scaling}")
        print(f"pulse_width_scaling: {args.pulse_width_scaling}")
        if periodic_cn_sum_approximation:
            print("periodic_cn_sum_lambda: 1.0")
        if args.solver == "acados":
            print(
                "acados_pulse_width_trust_radius: "
                f"{args.acados_pulse_width_trust_radius}"
            )
            print(
                "acados_transfer_pulse_width_trust_radius: "
                f"{args.acados_transfer_pulse_width_trust_radius}"
            )
            print(
                "acados_fes_state_trust_radius: "
                f"{args.acados_fes_state_trust_radius}"
            )
            print(
                "acados_fatigue_warmstart_mode: "
                f"{args.acados_fatigue_warmstart_mode}"
            )
            print("acados_dual_warm_start_mode: " f"{args.acados_dual_warm_start_mode}")
            print(
                "acados_standard_warmup_transfer: "
                f"{args.acados_standard_warmup_transfer}"
            )
            print(f"acados_horizon_continuation: {args.acados_horizon_continuation}")
            print(
                "acados_continuation_source_max_iterations: "
                f"{args.acados_continuation_source_max_iterations}"
            )
            print(
                "periodic_fes_warmup_projection: "
                f"{not args.disable_periodic_fes_warmup_projection}"
            )
            print(
                "periodic_fes_warmup_projection_weight: "
                f"{args.periodic_fes_warmup_projection_weight}"
            )
            print(
                "periodic_fes_warmup_force_projection_weight: "
                f"{args.periodic_fes_warmup_force_projection_weight}"
            )
            print(
                "periodic_fes_warmup_force_qdot_defect_limit: "
                f"{args.periodic_fes_warmup_force_qdot_defect_limit}"
            )
            print(
                "periodic_fes_warmup_force_adaptive_steps: "
                f"{args.periodic_fes_warmup_force_adaptive_steps}"
            )
            print(
                "periodic_fes_warmup_projection_mode: "
                f"{args.periodic_fes_warmup_projection_mode}"
            )
            print(
                "periodic_fes_warmup_projection_strategy: "
                f"{args.periodic_fes_warmup_projection_strategy}"
            )
            print(
                "periodic_fes_warmup_projection_substeps: "
                f"{args.periodic_fes_warmup_projection_substeps}"
            )
            print(
                "periodic_fes_warmup_projection_proximity_weight: "
                f"{args.periodic_fes_warmup_projection_proximity_weight}"
            )
            print(
                "periodic_fes_warmup_projection_defect_weight: "
                f"{args.periodic_fes_warmup_projection_defect_weight}"
            )
            print(
                "periodic_fes_warmup_projection_trust_radius: "
                f"{args.periodic_fes_warmup_projection_trust_radius}"
            )
            print(
                "periodic_fes_warmup_projection_max_iterations: "
                f"{args.periodic_fes_warmup_projection_max_iterations}"
            )
            print("periodic_ipopt_refinement: " f"{periodic_ipopt_refinement_enabled}")
            print(
                "periodic_ipopt_refinement_iterations: "
                f"{args.periodic_ipopt_refinement_iterations}"
            )
            print(
                "periodic_ipopt_refinement_use_sx: "
                f"{args.periodic_ipopt_refinement_use_sx}"
            )
            print(f"acados_collocation_type: {args.acados_collocation_type}")
            print(f"acados_sim_stages: {args.acados_sim_stages}")
            print(
                "acados_sim_steps: "
                f"{args.acados_sim_steps if args.acados_sim_steps is not None else max(3, args.rk_steps)}"
            )
            print(f"acados_newton_iter: {args.acados_newton_iter}")
            print(f"acados_newton_tol: {args.acados_newton_tol}")
            print(f"acados_jac_reuse: {args.acados_jac_reuse}")
            print(f"acados_tolerance: {args.acados_tolerance}")
            print(
                "acados_stationarity_tolerance: "
                f"{args.acados_stationarity_tolerance}"
            )
            print(
                "acados_first_window_tolerance: "
                f"{args.acados_first_window_tolerance}"
            )
            print(
                "acados_first_window_stationarity_tolerance: "
                f"{args.acados_first_window_stationarity_tolerance}"
            )
            print(f"acados_hessian_approx: {args.acados_hessian_approx}")
            print(f"acados_nlp_solver_type: {args.acados_nlp_solver_type}")
            print(
                "acados_search_direction_mode: " f"{args.acados_search_direction_mode}"
            )
            print(
                "acados_use_constraint_hessian_in_feas_qp: "
                f"{args.acados_use_constraint_hessian_in_feas_qp}"
            )
            print(
                "acados_allow_direction_mode_switch_to_nominal: "
                f"{not args.acados_disable_direction_mode_switch_to_nominal}"
            )
            print(f"acados_regularize_method: {args.acados_regularize_method}")
            print(f"acados_levenberg_marquardt: {args.acados_levenberg_marquardt}")
            print(f"acados_globalization: {args.acados_globalization}")
            print(f"acados_fixed_step_length: {args.acados_fixed_step_length}")
            print(f"acados_nlp_qp_tol_strategy: {args.acados_nlp_qp_tol_strategy}")
            print(f"acados_qp_iter_max: {args.acados_qp_iter_max}")
            print("acados_qp_warm_start_level: " f"{args.acados_qp_warm_start_level}")
            print("acados_warm_start_first_qp: " f"{args.acados_warm_start_first_qp}")
            print(
                "acados_warm_start_first_qp_from_nlp: "
                f"{args.acados_warm_start_first_qp_from_nlp}"
            )
            print(
                "acados_qpscaling_scale_objective: "
                f"{args.acados_qpscaling_scale_objective}"
            )
            print(
                "acados_qpscaling_scale_constraints: "
                f"{args.acados_qpscaling_scale_constraints}"
            )
            print(f"acados_ext_qp_res: {args.acados_ext_qp_res}")
            print(f"acados_diagnostics: {args.acados_diagnostics}")
            print(
                "warmup_state_comparison_limit: "
                f"{args.warmup_state_comparison_limit}"
            )
            print(f"acados_print_level: {args.acados_print_level}")
            print("bioptim_acados_interface_patch: True")
            print(f"acados_qp_solver: {args.acados_qp_solver}")
            print(f"acados_integrator_type: {args.acados_integrator_type}")
            print(f"acados_wheel_q_slack: {args.acados_wheel_q_slack}")
            print(f"acados_wheel_qdot_slack: {args.acados_wheel_qdot_slack}")
            print(f"acados_wheel_q_path_margin: {args.acados_wheel_q_path_margin}")
            print(
                "acados_transfer_full_dynamics_rollout: "
                f"{args.acados_transfer_full_dynamics_rollout}"
            )
            print(f"acados_transfer_phase_one: {args.acados_transfer_phase_one}")
            print(
                "acados_transfer_rollout_substeps: "
                f"{args.acados_transfer_rollout_substeps}"
            )
            print(
                "acados_transfer_rollout_max_bound_violation: "
                f"{args.acados_transfer_rollout_max_bound_violation}"
            )
        if (
            args.solver == "ipopt"
            or (
                periodic_cn_sum_approximation and not args.disable_standard_ipopt_warmup
            )
            or (
                args.solver == "acados"
                and periodic_cn_sum_approximation
                and periodic_ipopt_refinement_enabled
            )
        ):
            print(f"ipopt_linear_solver: {args.ipopt_linear_solver}")
        if args.solver == "ipopt":
            print(
                "historical_initial_guess: "
                f"{historical_init_guess_path if historical_init_guess_path else 'None'}"
            )
        print(f"full_dynamics_phase_one: {args.full_dynamics_phase_one}")

    if periodic_cn_sum_approximation and not args.disable_standard_ipopt_warmup:
        if echo:
            print("running_standard_ipopt_warmup: True")
        warmup_simulation_conditions = dict(simulation_conditions)
        if (
            args.solver == "acados"
            and args.control_regularization_target_source == "warmup"
        ):
            warmup_simulation_conditions["control_regularization_weight"] = 0.0
            warmup_simulation_conditions["control_regularization_target"] = None
        warmup_solution = run_standard_ipopt_warmup(
            args, mhe_info, cycling_info, warmup_simulation_conditions, model_path
        )
        adapted_warmup_solution = apply_standard_warmup_to_periodic_nmpc(
            nmpc,
            warmup_solution,
            project_fes_warmup=not args.disable_periodic_fes_warmup_projection,
            fatigue_warmstart_mode=args.acados_fatigue_warmstart_mode,
            projection_weight=args.periodic_fes_warmup_projection_weight,
            projection_mode=args.periodic_fes_warmup_projection_mode,
            projection_strategy=args.periodic_fes_warmup_projection_strategy,
            projection_substeps=args.periodic_fes_warmup_projection_substeps,
            projection_proximity_weight=(
                args.periodic_fes_warmup_projection_proximity_weight
            ),
            projection_defect_weight=args.periodic_fes_warmup_projection_defect_weight,
            projection_trust_radius=args.periodic_fes_warmup_projection_trust_radius,
            projection_max_iterations=(
                args.periodic_fes_warmup_projection_max_iterations
            ),
            force_projection_weight=args.periodic_fes_warmup_force_projection_weight,
            force_qdot_defect_limit=(args.periodic_fes_warmup_force_qdot_defect_limit),
            force_adaptive_steps=args.periodic_fes_warmup_force_adaptive_steps,
            warmup_transfer_mode=args.acados_standard_warmup_transfer,
            echo=echo,
        )
        if (
            args.solver == "acados"
            and args.control_regularization_target_source == "warmup"
            and args.control_regularization_weight
        ):
            target_keys = apply_warmup_control_regularization_targets(
                nmpc, adapted_warmup_solution
            )
            if echo:
                print(
                    "warmup_control_regularization_targets: "
                    f"{', '.join(target_keys) if target_keys else 'None'}"
                )
        if args.solver == "acados" and args.acados_diagnostics:
            print_warmup_state_comparison(
                "after_standard_projection",
                adapted_warmup_solution,
                nmpc,
                args.warmup_state_comparison_limit,
            )

    if args.check_wheel_periodicity:
        periodicity = wheel_angle_periodicity_diagnostics(nmpc)
        if echo:
            print(
                "wheel_angle_periodicity: "
                f"max_abs_rhs_difference={periodicity['max_abs_rhs_difference']:.12g} "
                f"l2_rhs_difference={periodicity['l2_rhs_difference']:.12g}"
            )

    if (
        args.solver == "acados"
        and periodic_cn_sum_approximation
        and periodic_ipopt_refinement_enabled
    ):
        if echo:
            print("running_periodic_ipopt_refinement: True")
            if args.acados_diagnostics:
                print("periodic_ipopt_refinement_initial_defects:")
                print_initial_guess_diagnostics(nmpc)
        refinement_cache_path = _periodic_ipopt_refinement_cache_path(args, model_path)
        if refinement_cache_path.exists():
            refinement_solution = _load_warmup_cache(refinement_cache_path)
            periodic_ipopt_reference_solution = refinement_solution
            apply_solution_directly_to_periodic_nmpc_initial_guess(
                nmpc, refinement_solution
            )
            if echo:
                print(
                    "periodic_ipopt_refinement_cache: "
                    f"hit ({refinement_cache_path.name})"
                )
                print("periodic_ipopt_refinement_applied: True")
        else:
            refinement_nmpc = build_periodic_ipopt_refinement_nmpc(
                source_nmpc=nmpc,
                model_path=model_path,
                stim_time=stim_time,
                mhe_info={
                    **mhe_info,
                    "use_sx": args.periodic_ipopt_refinement_use_sx,
                },
                cycling_info=cycling_info,
                simulation_conditions=nmpc_simulation_conditions,
                model_formulation=args.model_formulation,
                refinement_ode_solver=(
                    OdeSolver.IRK(
                        polynomial_degree=args.acados_sim_stages,
                        method=(
                            "legendre"
                            if args.acados_collocation_type == "GAUSS_LEGENDRE"
                            else "radau"
                        ),
                    )
                    if args.periodic_ipopt_refinement_ode_solver == "irk"
                    else (
                        OdeSolver.RK4(n_integration_steps=args.rk_steps)
                        if args.periodic_ipopt_refinement_ode_solver == "rk4"
                        else None
                    )
                ),
            )
            periodic_ipopt_reference_solution = run_periodic_ipopt_refinement(
                refinement_nmpc,
                target_nmpc=nmpc,
                max_iterations=args.periodic_ipopt_refinement_iterations,
                linear_solver=args.ipopt_linear_solver,
                cache_path=refinement_cache_path,
                echo=echo,
            )
        if echo and args.acados_diagnostics and adapted_warmup_solution is not None:
            print_warmup_state_comparison(
                "after_periodic_ipopt_refinement",
                adapted_warmup_solution,
                nmpc,
                args.warmup_state_comparison_limit,
            )

    if args.solver == "acados" and args.acados_horizon_continuation:
        if horizon_seed is not None:
            apply_solution_directly_to_periodic_nmpc_initial_guess(
                nmpc, horizon_seed, recenter_kinematic_bounds=True
            )
            if echo:
                print("acados_horizon_seed_applied: True")
        else:
            continuation_summary = tile_one_cycle_solution_to_periodic_nmpc(
                nmpc, continuation_source
            )
            if echo:
                print(
                    "acados_horizon_continuation_applied: "
                    f"repeats={continuation_summary['repeat_count']} "
                    f"source_controls={continuation_summary['source_control_nodes']} "
                    f"target_controls={continuation_summary['target_control_nodes']} "
                    "max_transfer_seam_error="
                    f"{continuation_summary['max_transfer_seam_error']:.6g}"
                )
                fes_rollout = continuation_summary["fes_rollout"]
                print(
                    "acados_horizon_continuation_fes_rollout: "
                    f"applied={fes_rollout['applied']} "
                    f"state_count={fes_rollout.get('state_count', 0)} "
                    f"start_node={fes_rollout.get('start_node')} "
                    f"substeps={fes_rollout.get('substeps')} "
                    f"max_change={fes_rollout.get('max_change')} "
                    f"clipped_values={fes_rollout.get('clipped_value_count')} "
                    f"max_clip={fes_rollout.get('max_clip')}"
                )
        if echo:
            for summary in pulse_width_initial_guess_summary(nmpc):
                print(
                    "continuation_pulse_width: "
                    f"{summary['key']} "
                    f"min={summary['minimum']:.9g} "
                    f"mean={summary['mean']:.9g} "
                    f"max={summary['maximum']:.9g} "
                    f"span={summary['span']:.9g}"
                )

    if acados_seed_cache_path is not None and acados_seed_cache_path.exists():
        cached_acados_seed = _load_warmup_cache(acados_seed_cache_path)
        apply_solution_directly_to_periodic_nmpc_initial_guess(nmpc, cached_acados_seed)
        if echo:
            print(f"acados_seed_cache: hit ({acados_seed_cache_path.name})")

    if args.solver == "acados" and args.acados_project_qdot_from_q:
        project_qdot_initial_guess_from_q(nmpc)
        if echo:
            print("acados_project_qdot_from_q: True")

    if args.full_dynamics_phase_one:
        phase_one_summary = project_full_dynamics_initial_guess(
            nmpc,
            proximity_weight=args.full_dynamics_phase_one_proximity_weight,
            defect_weight=args.full_dynamics_phase_one_defect_weight,
            n_substeps=args.full_dynamics_phase_one_substeps,
        )
        if echo:
            print(
                "full_dynamics_phase_one: "
                f"proximity_weight={phase_one_summary['proximity_weight']:.6g} "
                f"defect_weight={phase_one_summary['defect_weight']:.6g} "
                f"scaled_defect_before={phase_one_summary['scaled_defect_before']:.6g} "
                f"scaled_defect_after={phase_one_summary['scaled_defect_after']:.6g} "
                f"max_state_change={phase_one_summary['max_state_change']:.6g}"
            )

    if args.solver == "acados" and args.acados_fes_state_trust_radius is not None:
        trust_summary = apply_fes_state_trust_region(
            nmpc, args.acados_fes_state_trust_radius
        )
        if echo:
            print(f"acados_fes_state_trust_region: keys={len(trust_summary)}")
            for key, item in list(trust_summary.items())[:8]:
                print(
                    "acados_fes_state_trust_region: "
                    f"{key} center=[{item['center_min']:.6g}, {item['center_max']:.6g}] "
                    f"bounds=[{item['lower']:.6g}, {item['upper']:.6g}] "
                    f"scale={item['scale']:.6g}"
                )

    if args.solver == "acados" and args.acados_pulse_width_trust_radius is not None:
        trust_summary = apply_pulse_width_control_trust_region(
            nmpc, args.acados_pulse_width_trust_radius
        )
        if echo:
            for key, item in trust_summary.items():
                print(
                    "acados_pulse_width_trust_region: "
                    f"{key} center=[{item['center_min']:.6g}, {item['center_max']:.6g}] "
                    f"bounds=[{item['lower']:.6g}, {item['upper']:.6g}]"
                )

    if args.validate_integrator_maps:
        for row in high_accuracy_integrator_map_diagnostics(nmpc):
            if echo:
                print(
                    "integrator_map_validation: "
                    f"node={row['node']} "
                    "trajectory_vs_dop853="
                    f"{row['trajectory_vs_reference']:.6g} "
                    f"rk4_vs_dop853={row['rk4_vs_reference']:.6g} "
                    f"trajectory_vs_rk4={row['trajectory_vs_rk4']:.6g} "
                    f"dop853_nfev={row['reference_evaluations']}"
                )

    if args.solver == "acados" and args.acados_diagnostics:
        print_initial_guess_diagnostics(nmpc)

    initial_guess_state_traces = None
    initial_guess_control_traces = None
    if args.solver == "acados":
        initial_guess_state_traces = _initial_guess_traces(nmpc.nlp[0].x_init)
        initial_guess_control_traces = _initial_guess_traces(nmpc.nlp[0].u_init)

    acados_window_diagnostics = []
    transfer_rollout_summaries = []

    def cache_first_successful_window(_nmpc, solution):
        diagnostics = snapshot_acados_diagnostics(solution)
        if (
            horizon_seed_cache_path is None
            or horizon_seed_cache_path.exists()
            or _nmpc.total_optimization_run != 0
            or not _status_is_success(solution.status)
        ):
            return
        meets_strict_tolerances = acados_diagnostics_meet_tolerances(
            diagnostics,
            args.acados_tolerance,
            args.acados_stationarity_tolerance,
        )
        _save_warmup_cache(horizon_seed_cache_path, solution)
        if echo:
            print(
                "acados_horizon_seed_cache: saved "
                f"quality={'strict' if meets_strict_tolerances else 'relaxed'} "
                f"({horizon_seed_cache_path.name})"
            )

    if args.solver == "acados" and horizon_seed_cache_path is not None:
        nmpc.before_window_advance = cache_first_successful_window

    def update_functions(_nmpc, cycle_idx, _sol):
        print(f"window {cycle_idx}")
        if echo and _sol is not None:
            states = _sol.decision_states(to_merge=SolutionMerge.NODES)
            print(
                f"window {cycle_idx} terminal wheel q={states['q'][2, -1]:.6f} "
                f"qdot={states['qdot'][2, -1]:.6f}"
            )
        continue_solving = cycle_idx + 1 < args.n_windows
        if (
            continue_solving
            and _sol is not None
            and args.solver == "acados"
            and args.acados_transfer_full_dynamics_rollout
        ):
            rollout_summary = rollout_transferred_cycle_full_dynamics(
                _nmpc,
                n_substeps=args.acados_transfer_rollout_substeps,
                max_allowed_bound_violation=(
                    args.acados_transfer_rollout_max_bound_violation
                ),
            )
            transfer_rollout_summaries.append(rollout_summary)
            if echo:
                print(
                    "acados_transfer_full_dynamics_rollout: "
                    f"applied={rollout_summary['applied']} "
                    f"start_node={rollout_summary['start_node']} "
                    "max_bound_violation="
                    f"{rollout_summary.get('max_bound_violation')}"
                )
                violation_by_key = rollout_summary.get("max_bound_violation_by_key", {})
                if violation_by_key:
                    worst_key = max(violation_by_key, key=violation_by_key.get)
                    print(
                        "acados_transfer_rollout_worst_bound: "
                        f"key={worst_key} "
                        f"violation={violation_by_key[worst_key]:.6g}"
                    )
                if rollout_summary["applied"]:
                    print(
                        "acados_transfer_rollout_terminal_delta: "
                        f"q={rollout_summary['terminal_delta'].get('q')} "
                        f"qdot={rollout_summary['terminal_delta'].get('qdot')}"
                    )
                else:
                    print(
                        "acados_transfer_rollout_nonfinite_node: "
                        f"{rollout_summary.get('nonfinite_node')}"
                    )
        if (
            continue_solving
            and _sol is not None
            and args.solver == "acados"
            and args.acados_transfer_phase_one
        ):
            phase_one_summary = project_full_dynamics_initial_guess(
                _nmpc,
                proximity_weight=args.full_dynamics_phase_one_proximity_weight,
                defect_weight=args.full_dynamics_phase_one_defect_weight,
                n_substeps=args.full_dynamics_phase_one_substeps,
            )
            if echo:
                print(
                    "acados_transfer_phase_one: "
                    f"scaled_defect_before={phase_one_summary['scaled_defect_before']:.6g} "
                    f"scaled_defect_after={phase_one_summary['scaled_defect_after']:.6g} "
                    f"max_state_change={phase_one_summary['max_state_change']:.6g}"
                )
        if args.solver == "acados" and _sol is not None:
            diagnostics = snapshot_acados_diagnostics(_sol)
            acados_window_diagnostics.append(diagnostics)
            if echo and args.acados_diagnostics:
                print_acados_diagnostics(f"window[{cycle_idx - 1}]", diagnostics)
                if continue_solving:
                    print(f"window[{cycle_idx}] transferred_initial_guess_diagnostics:")
                    print_initial_guess_diagnostics(_nmpc)
        if (
            continue_solving
            and _sol is not None
            and args.solver == "acados"
            and args.acados_pulse_width_trust_radius is not None
        ):
            transfer_radius = (
                args.acados_transfer_pulse_width_trust_radius
                if args.acados_transfer_pulse_width_trust_radius is not None
                else args.acados_pulse_width_trust_radius
            )
            trust_summary = apply_pulse_width_control_trust_region(
                _nmpc, transfer_radius
            )
            if echo:
                max_center = max(item["center_max"] for item in trust_summary.values())
                print(
                    "acados_pulse_width_trust_region_recentered: "
                    f"window={cycle_idx} max_center={max_center:.9g}"
                )
        return continue_solving

    solver_first_iter = None
    if args.solver == "acados":
        nmpc._cocofest_fix_controls_to_warmup = args.acados_fix_controls_to_warmup
        nmpc._cocofest_fixed_control_tolerance = args.acados_fixed_control_tolerance
        nmpc._cocofest_discrete_substeps = (
            args.acados_sim_steps
            if args.acados_sim_steps is not None
            else max(3, args.rk_steps)
        )
        if echo:
            print(
                "acados_fix_controls_to_warmup: "
                f"{args.acados_fix_controls_to_warmup}"
            )
        model_name, generated_code_path = build_codegen_names(args)
        solver = configure_acados_solver(
            model_name=model_name,
            generated_code_path=generated_code_path,
            max_iterations=args.max_acados_iterations,
            convergence_tolerance=args.acados_tolerance,
            stationarity_tolerance=args.acados_stationarity_tolerance,
            qp_solver=args.acados_qp_solver,
            integrator_type=args.acados_integrator_type,
            collocation_type=args.acados_collocation_type,
            sim_method_num_stages=args.acados_sim_stages,
            sim_method_num_steps=(
                args.acados_sim_steps
                if args.acados_sim_steps is not None
                else max(3, args.rk_steps)
            ),
            sim_method_newton_iter=args.acados_newton_iter,
            sim_method_newton_tol=args.acados_newton_tol,
            sim_method_jac_reuse=args.acados_jac_reuse,
            hessian_approx=args.acados_hessian_approx,
            nlp_solver_type=args.acados_nlp_solver_type,
            search_direction_mode=args.acados_search_direction_mode,
            use_constraint_hessian_in_feas_qp=(
                args.acados_use_constraint_hessian_in_feas_qp
            ),
            allow_direction_mode_switch_to_nominal=(
                not args.acados_disable_direction_mode_switch_to_nominal
            ),
            regularize_method=args.acados_regularize_method,
            levenberg_marquardt=args.acados_levenberg_marquardt,
            globalization=args.acados_globalization,
            fixed_step_length=args.acados_fixed_step_length,
            nlp_qp_tol_strategy=args.acados_nlp_qp_tol_strategy,
            qp_iter_max=args.acados_qp_iter_max,
            qp_warm_start_level=args.acados_qp_warm_start_level,
            warm_start_first_qp=args.acados_warm_start_first_qp,
            warm_start_first_qp_from_nlp=(args.acados_warm_start_first_qp_from_nlp),
            qpscaling_scale_objective=args.acados_qpscaling_scale_objective,
            qpscaling_scale_constraints=args.acados_qpscaling_scale_constraints,
            ext_qp_res=args.acados_ext_qp_res,
            print_level=args.acados_print_level,
        )
        if (
            args.acados_first_window_tolerance is not None
            or args.acados_first_window_stationarity_tolerance is not None
        ):
            solver_first_iter = deepcopy(solver)
            if args.acados_first_window_tolerance is not None:
                solver_first_iter.set_convergence_tolerance(
                    args.acados_first_window_tolerance
                )
            if args.acados_first_window_stationarity_tolerance is not None:
                solver_first_iter.set_nlp_solver_tol_stat(
                    args.acados_first_window_stationarity_tolerance
                )
            # The second solver has the same generated problem and only restores the
            # strict tolerances, which Acados can update at runtime.
            solver.set_only_first_options_has_changed(False)
    else:
        solver = configure_ipopt_solver(
            max_iterations=args.max_ipopt_iterations,
            linear_solver=args.ipopt_linear_solver,
        )

    if args.single_shot:
        sol = super(RecedingHorizonOptimization, nmpc).solve(
            solver=solver,
            warm_start=None,
        )
        if echo:
            summarize_single_shot(sol)
            if args.solver == "acados" and args.acados_diagnostics:
                print_acados_diagnostics("single_shot", collect_acados_diagnostics(sol))
            if (
                args.solver == "acados"
                and periodic_ipopt_reference_solution is not None
            ):
                print_solution_trace_comparison(
                    "ipopt_acados_control_comparison",
                    periodic_ipopt_reference_solution,
                    sol,
                    controls=True,
                    limit=len(nmpc.nlp[0].controls.keys()),
                )
                print_solution_trace_comparison(
                    "ipopt_acados_state_comparison",
                    periodic_ipopt_reference_solution,
                    sol,
                    controls=False,
                    limit=args.warmup_state_comparison_limit,
                )
            if args.validate_integrator_maps:
                apply_solution_directly_to_periodic_nmpc_initial_guess(nmpc, sol)
                for row in high_accuracy_integrator_map_diagnostics(nmpc):
                    print(
                        "final_integrator_map_validation: "
                        f"node={row['node']} "
                        "trajectory_vs_dop853="
                        f"{row['trajectory_vs_reference']:.6g} "
                        f"rk4_vs_dop853={row['rk4_vs_reference']:.6g} "
                        f"trajectory_vs_rk4={row['trajectory_vs_rk4']:.6g} "
                        f"dop853_nfev={row['reference_evaluations']}"
                    )
        if acados_seed_cache_path is not None and _status_is_success(sol.status):
            _save_warmup_cache(acados_seed_cache_path, sol)
            if echo:
                print(f"acados_seed_cache: saved ({acados_seed_cache_path.name})")
        summary = build_single_shot_summary(sol)
        if args.solver == "acados" and args.acados_diagnostics:
            summary["acados_diagnostics"] = collect_acados_diagnostics(sol)
        if initial_guess_state_traces is not None:
            summary["initial_guess_state_traces"] = initial_guess_state_traces
            summary["initial_guess_control_traces"] = initial_guess_control_traces
        summary["args"] = args
        summary["control_bounds"] = _control_bounds_summary(nmpc)
        return summary

    sol = nmpc.solve_fes_nmpc(
        update_functions,
        solver=solver,
        solver_first_iter=solver_first_iter,
        total_cycles=args.n_windows,
        external_force=cycling_info.get("resistive_torque"),
        cycle_solutions=MultiCyclicCycleSolutions.ALL_CYCLES,
        get_all_iterations=True,
        cyclic_options={"states": {}},
        max_consecutive_failing=args.max_consecutive_failing,
    )
    if echo:
        summarize_windows(
            sol,
            requested_windows=args.n_windows,
            cycles_per_window=args.cycles_per_window,
        )
        if args.solver == "acados" and args.acados_diagnostics:
            if not acados_window_diagnostics:
                print_acados_diagnostics("merged", collect_acados_diagnostics(sol[0]))
    summary = build_window_summary(
        sol, requested_windows=args.n_windows, cycles_per_window=args.cycles_per_window
    )
    if args.solver == "acados" and args.acados_diagnostics:
        summary["acados_diagnostics"] = acados_window_diagnostics
    if transfer_rollout_summaries:
        summary["transfer_rollout_summaries"] = transfer_rollout_summaries
    if initial_guess_state_traces is not None:
        summary["initial_guess_state_traces"] = initial_guess_state_traces
        summary["initial_guess_control_traces"] = initial_guess_control_traces
    summary["args"] = args
    summary["control_bounds"] = _control_bounds_summary(nmpc)
    return summary


def main(cli_args: list[str] | None = None):
    parser = build_argument_parser()
    args = parser.parse_args(cli_args)
    if (
        args.acados_nlp_qp_tol_strategy == "ADAPTIVE_QPSCALING"
        and args.acados_qpscaling_scale_objective == "NO_OBJECTIVE_SCALING"
        and args.acados_qpscaling_scale_constraints == "NO_CONSTRAINT_SCALING"
    ):
        parser.error(
            "--acados-nlp-qp-tol-strategy ADAPTIVE_QPSCALING requires at least one "
            "QP scaling option to be enabled."
        )
    ensure_acados_environment()
    solve_case(args, echo=True)


if __name__ == "__main__":
    main()
