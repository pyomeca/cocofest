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
from time import perf_counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bioptim import MultiCyclicCycleSolutions, Node, OdeSolver, SolutionMerge, Solver
from bioptim.optimization.receding_horizon_optimization import (
    RecedingHorizonOptimization,
)

from cocofest.optimization.receding_horizon_initial_guess import (
    audit_initial_guess,
    copy_container_values,
    project_initial_guess_to_bounds,
    snapshot_container,
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


def apply_ipopt_dual_warm_start(nmpc, solution, mode: str) -> dict:
    """Transfer IPOPT duals without replacing the shifted primal initial guess."""
    if mode not in {"off", "constraints", "bounds", "all"}:
        raise ValueError(f"Unsupported IPOPT dual warm-start mode '{mode}'.")

    summary = {
        "mode": mode,
        "applied": False,
        "lam_g_size": 0,
        "lam_x_size": 0,
        "reason": None,
    }
    if mode == "off":
        summary["reason"] = "disabled"
        return summary
    if solution is None:
        summary["reason"] = "no_previous_solution"
        return summary

    interface = getattr(nmpc, "ocp_solver", None)
    if interface is None or not hasattr(interface, "lam_g"):
        summary["reason"] = "ipopt_interface_unavailable"
        return summary

    limits = getattr(interface, "limits", {})

    def validated_multiplier(name: str, expected_key: str):
        values = getattr(solution, name, None)
        if values is None:
            return None
        values = np.asarray(values, dtype=float).reshape(-1)
        expected = limits.get(expected_key)
        if expected is not None and values.size != np.asarray(expected).size:
            return None
        if not np.all(np.isfinite(values)):
            return None
        return values.copy()

    lam_g = None
    if mode in {"constraints", "all"}:
        lam_g = validated_multiplier("lam_g", "lbg")
        if lam_g is None:
            summary["reason"] = "invalid_constraint_multipliers"
            return summary

    lam_x = None
    if mode in {"bounds", "all"}:
        lam_x = validated_multiplier("lam_x", "x0")
        if lam_x is None:
            summary["reason"] = "invalid_bound_multipliers"
            return summary

    interface.lam_g = lam_g
    interface.lam_x = lam_x
    summary.update(
        applied=True,
        lam_g_size=0 if lam_g is None else lam_g.size,
        lam_x_size=0 if lam_x is None else lam_x.size,
    )
    return summary


def parse_objectives(raw_objective: str) -> set[str]:
    values = {item.strip().lower() for item in raw_objective.split(",") if item.strip()}
    allowed = set(OBJECTIVE_TO_WEIGHT_INDEX) | {"none"}
    invalid = values - allowed
    if invalid:
        raise ValueError(f"Unsupported objectives: {', '.join(sorted(invalid))}")
    if "none" in values and len(values) > 1:
        raise ValueError("'none' cannot be combined with other objectives.")
    return values or {"force"}


def parse_control_homotopy_radii(raw_radii: str) -> tuple[float, ...]:
    radii = tuple(float(item.strip()) for item in raw_radii.split(",") if item.strip())
    if not radii:
        raise argparse.ArgumentTypeError(
            "The control homotopy requires at least one radius."
        )
    if any(not np.isfinite(radius) or radius <= 0.0 for radius in radii):
        raise argparse.ArgumentTypeError(
            "Control homotopy radii must be finite and strictly positive."
        )
    if any(next_radius <= radius for radius, next_radius in zip(radii, radii[1:])):
        raise argparse.ArgumentTypeError(
            "Control homotopy radii must be strictly increasing."
        )
    return radii


def parse_proximal_control_weights(raw_weights: str) -> tuple[float, ...]:
    weights = tuple(
        float(item.strip()) for item in raw_weights.split(",") if item.strip()
    )
    if not weights:
        raise argparse.ArgumentTypeError(
            "The proximal control continuation requires at least one weight."
        )
    if any(not np.isfinite(weight) or weight <= 0.0 for weight in weights):
        raise argparse.ArgumentTypeError(
            "Proximal control weights must be finite and strictly positive."
        )
    if any(next_weight >= weight for weight, next_weight in zip(weights, weights[1:])):
        raise argparse.ArgumentTypeError(
            "Proximal control weights must be strictly decreasing."
        )
    return weights


def parse_terminal_wheel_q_slacks(raw_slacks: str) -> tuple[float, ...]:
    slacks = tuple(
        float(item.strip()) for item in raw_slacks.split(",") if item.strip()
    )
    if not slacks:
        raise argparse.ArgumentTypeError(
            "Terminal wheel-bound continuation requires at least one slack."
        )
    if any(not np.isfinite(slack) or slack < 0.0 for slack in slacks):
        raise argparse.ArgumentTypeError(
            "Terminal wheel-bound slacks must be finite and non-negative."
        )
    if any(next_slack >= slack for slack, next_slack in zip(slacks, slacks[1:])):
        raise argparse.ArgumentTypeError(
            "Terminal wheel-bound slacks must be strictly decreasing."
        )
    return slacks


def parse_transfer_bound_homotopy_fractions(
    raw_fractions: str,
) -> tuple[float, ...]:
    fractions = tuple(
        float(item.strip()) for item in raw_fractions.split(",") if item.strip()
    )
    if not fractions:
        raise argparse.ArgumentTypeError(
            "The transfer-bound homotopy requires at least one fraction."
        )
    if any(
        not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0
        for fraction in fractions
    ):
        raise argparse.ArgumentTypeError(
            "Transfer-bound homotopy fractions must be finite and between 0 and 1."
        )
    if any(
        next_fraction <= fraction
        for fraction, next_fraction in zip(fractions, fractions[1:])
    ):
        raise argparse.ArgumentTypeError(
            "Transfer-bound homotopy fractions must be strictly increasing."
        )
    if not np.isclose(fractions[-1], 1.0):
        raise argparse.ArgumentTypeError(
            "Transfer-bound homotopy fractions must end at 1."
        )
    return fractions


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
        "--compact-rho-output",
        action="store_true",
        help=(
            "Assemble receding-horizon states and controls numerically instead of "
            "rebuilding a symbolic OCP spanning every exported cycle."
        ),
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
        choices=("constant", "warmup", "previous"),
        default="constant",
        help=(
            "Use a constant target, the standard IPOPT warmup controls, or the "
            "shifted controls from the previous MHE window."
        ),
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
        "--wheel-qdot-bound-margin",
        type=float,
        default=3.0,
        help="Symmetric wheel-speed bound around -2*pi, in rad/s.",
    )
    parser.add_argument(
        "--terminal-qdot-regularization-weight",
        type=float,
        default=0.0,
        help="Quadratic Mayer weight retaining terminal joint velocities near their shifted reference.",
    )
    parser.add_argument(
        "--terminal-qdot-regularization-target-source",
        choices=("initial", "previous", "first_node"),
        default="previous",
        help=(
            "Keep the initial terminal-velocity target, update it from the previous "
            "MHE window, or enforce cyclic velocity relative to the current first node."
        ),
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
        "--acados-qp-cond-n",
        type=int,
        default=None,
        help=(
            "Condensed HPIPM horizon. The default preserves all shooting stages; "
            "smaller values partially condense the QP."
        ),
    )
    parser.add_argument(
        "--acados-hpipm-mode",
        choices=("BALANCE", "SPEED_ABS", "SPEED", "ROBUST"),
        default="ROBUST",
        help="HPIPM interior-point tuning profile.",
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
        default="FUNNEL_L1PEN_LINESEARCH",
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
        help="First-node slack, in rad, for the ACADOS wheel/crank angle transfer bound.",
    )
    parser.add_argument(
        "--acados-terminal-wheel-q-slack",
        type=float,
        default=0.2,
        help="Terminal crank-angle slack in rad; independent from the first-node transfer slack.",
    )
    parser.add_argument(
        "--acados-terminal-wheel-q-homotopy-slacks",
        type=parse_terminal_wheel_q_slacks,
        default=None,
        help=(
            "Comma-separated decreasing terminal crank-angle slacks. ACADOS "
            "solves each bound stage around the same one-turn target."
        ),
    )
    parser.add_argument(
        "--acados-terminal-wheel-q-homotopy-each-window",
        action="store_true",
        help="Repeat terminal crank-bound tightening after every MHE transfer.",
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
        "--acados-bind-first-node-fes-states",
        action="store_true",
        help=(
            "Enforce inter-window continuity for every FES and fatigue state. "
            "Use this for physical endurance benchmarks; relaxed first-node FES "
            "bounds remain available for solver diagnostics."
        ),
    )
    parser.add_argument(
        "--acados-project-qdot-from-q",
        action="store_true",
        help="Project the ACADOS qdot initial guess from finite differences of q before solving.",
    )
    parser.add_argument(
        "--transfer-full-dynamics-rollout",
        "--acados-transfer-full-dynamics-rollout",
        dest="acados_transfer_full_dynamics_rollout",
        action="store_true",
        help=(
            "Reintegrate the appended cycle with the complete dynamics after each "
            "window transfer. This solver-independent rollout is available to IPOPT "
            "and ACADOS."
        ),
    )
    parser.add_argument(
        "--acados-transfer-irk-rollout",
        action="store_true",
        help=(
            "Reintegrate the appended cycle with the exact generated ACADOS IRK simulator "
            "after each window transfer."
        ),
    )
    parser.add_argument(
        "--transfer-phase-one",
        "--acados-transfer-phase-one",
        dest="acados_transfer_phase_one",
        action="store_true",
        help=(
            "Apply the bounded proximal complete-dynamics phase I after each MHE window "
            "transfer for either solver. Uses the --full-dynamics-phase-one-* settings."
        ),
    )
    parser.add_argument(
        "--acados-transfer-bound-homotopy",
        action="store_true",
        help=(
            "Temporarily enlarge path and terminal state bounds around the IRK "
            "transfer, then tighten them back to the physical bounds with ACADOS."
        ),
    )
    parser.add_argument(
        "--acados-transfer-bound-homotopy-fractions",
        type=parse_transfer_bound_homotopy_fractions,
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
        help=(
            "Strictly increasing interpolation fractions ending at 1 for the "
            "transfer-bound homotopy."
        ),
    )
    parser.add_argument(
        "--acados-transfer-bound-homotopy-padding",
        type=float,
        default=0.05,
        help=(
            "Relative padding around the transferred state trajectory in the "
            "fully relaxed bound stage."
        ),
    )
    parser.add_argument(
        "--acados-transfer-bound-homotopy-iterations",
        type=int,
        default=30,
        help="Maximum SQP iterations for each transfer-bound homotopy stage.",
    )
    parser.add_argument(
        "--acados-transfer-bound-homotopy-tolerance",
        type=float,
        default=1e-4,
        help="KKT residual threshold used to accept transfer-bound homotopy stages.",
    )
    parser.add_argument(
        "--acados-transfer-bound-homotopy-solver-tolerance",
        type=float,
        default=None,
        help=(
            "Optional stricter KKT tolerance requested from ACADOS during each "
            "transfer-bound homotopy stage."
        ),
    )
    parser.add_argument(
        "--acados-transfer-sqp-restarts",
        type=int,
        default=0,
        help=(
            "Maximum number of short ACADOS restarts used to repair a transferred "
            "initial guess before solving the next MHE window."
        ),
    )
    parser.add_argument(
        "--acados-transfer-sqp-restart-iterations",
        type=int,
        default=1,
        help="SQP iterations performed by each transfer-repair attempt.",
    )
    parser.add_argument(
        "--acados-transfer-sqp-restart-feasibility-tolerance",
        type=float,
        default=1e-2,
        help=(
            "Largest dynamics, inequality, or complementarity residual accepted "
            "as a restartable transferred iterate."
        ),
    )
    parser.add_argument(
        "--acados-store-iterates",
        action="store_true",
        help=(
            "Retain every SQP iterate so a restart can select an intermediate "
            "primal. This substantially increases memory use on long MHE runs."
        ),
    )
    parser.add_argument(
        "--acados-cyclical-transfer-mode",
        choices=("extrapolate", "repeat"),
        default="extrapolate",
        help=(
            "Construct cyclic states in the appended MHE cycle by preserving the "
            "last observed cycle drift or by repeating the last cycle verbatim."
        ),
    )
    parser.add_argument(
        "--transfer-rollout-substeps",
        "--acados-transfer-rollout-substeps",
        dest="acados_transfer_rollout_substeps",
        type=int,
        default=5,
        help="RK4 substeps used by the full-dynamics inter-window rollout.",
    )
    parser.add_argument(
        "--transfer-pulse-width-scale",
        "--acados-transfer-pulse-width-scale",
        dest="acados_transfer_pulse_width_scale",
        type=float,
        default=1.0,
        help=(
            "Scale pulse widths only in the newly appended cycle before its "
            "dynamics rollout; values are clipped to the physical control bounds."
        ),
    )
    parser.add_argument(
        "--transfer-ding-force-compensation",
        action="store_true",
        help=(
            "Choose each appended-cycle pulse width with a scalar Ding rollout "
            "so its next-node force follows the preceding cycle."
        ),
    )
    parser.add_argument(
        "--transfer-ding-force-compensation-substeps",
        type=int,
        default=5,
        help="RK4 substeps used by the per-muscle Ding force compensation.",
    )
    parser.add_argument(
        "--transfer-ding-force-compensation-iterations",
        type=int,
        default=20,
        help="Bisection iterations used for each compensated pulse width.",
    )
    parser.add_argument(
        "--transfer-rollout-max-bound-violation",
        "--acados-transfer-rollout-max-bound-violation",
        dest="acados_transfer_rollout_max_bound_violation",
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
        "--ipopt-dual-warm-start-mode",
        choices=("off", "constraints", "bounds", "all"),
        default="bounds",
        help=(
            "Reuse no duals, constraint multipliers, bound multipliers, or both "
            "from the previous IPOPT MHE window."
        ),
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
        "--acados-control-homotopy-radii",
        type=parse_control_homotopy_radii,
        default=None,
        help=(
            "Comma-separated, strictly increasing pulse-width radii in seconds. "
            "ACADOS first solves with controls fixed to the IPOPT seed, then solves "
            "one proximal subproblem per radius before restoring the original bounds."
        ),
    )
    parser.add_argument(
        "--acados-control-homotopy-tolerance",
        type=float,
        default=5e-4,
        help="KKT tolerance used only by the ACADOS control-homotopy subproblems.",
    )
    parser.add_argument(
        "--acados-control-homotopy-max-restarts",
        type=int,
        default=2,
        help=(
            "Maximum number of same-radius SQP restarts when a homotopy stage has "
            "small feasibility residuals but has not reached stationarity."
        ),
    )
    parser.add_argument(
        "--acados-control-homotopy-stage-iterations",
        type=int,
        default=54,
        help=(
            "Maximum SQP iterations per homotopy attempt. Short attempts preserve "
            "a usable iterate for restart before a late QP failure."
        ),
    )
    parser.add_argument(
        "--acados-control-homotopy-keep-final-radius",
        action="store_true",
        help=(
            "Keep the largest accepted homotopy radius for the first MHE window "
            "instead of immediately restoring unrestricted control bounds."
        ),
    )
    parser.add_argument(
        "--acados-control-homotopy-each-window",
        action="store_true",
        help=(
            "Repeat the fixed-control and radius continuation after each MHE "
            "window transfer before solving the next window."
        ),
    )
    parser.add_argument(
        "--acados-control-homotopy-window-growth",
        type=float,
        default=1.0,
        help=("Factor applied to the retained homotopy radius after each MHE window."),
    )
    parser.add_argument(
        "--acados-proximal-control-weights",
        type=parse_proximal_control_weights,
        default=None,
        help=(
            "Comma-separated, strictly decreasing control-proximity weights. "
            "The first weight is compiled into the Acados problem; later weights "
            "are applied to the generated solver without changing physical bounds."
        ),
    )
    parser.add_argument(
        "--acados-proximal-control-each-window",
        action="store_true",
        help=(
            "Repeat the proximal control continuation after every MHE transfer, "
            "using the shifted previous-window controls as references."
        ),
    )
    parser.add_argument(
        "--acados-proximal-control-tolerance",
        type=float,
        default=5e-4,
        help="KKT tolerance used by the proximal control continuation stages.",
    )
    parser.add_argument(
        "--acados-proximal-control-stage-iterations",
        type=int,
        default=50,
        help="Maximum SQP iterations for each proximal control continuation stage.",
    )
    parser.add_argument(
        "--acados-proximal-control-max-restarts",
        type=int,
        default=1,
        help="Maximum same-weight restarts for a nearly feasible proximal stage.",
    )
    parser.add_argument(
        "--acados-proximal-control-try-next-weight-on-failure",
        action="store_true",
        help=(
            "After resetting a failed proximal stage, try the next lower "
            "configured weight from the unchanged transferred primal."
        ),
    )
    parser.add_argument(
        "--continue-after-acados-transfer-failure",
        action="store_true",
        help=(
            "Diagnostic mode that keeps advancing after an auxiliary ACADOS "
            "transfer solve fails. Endurance runs stop at the first failed "
            "transfer by default."
        ),
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
        "--periodic-ipopt-refinement-each-window",
        action="store_true",
        help=(
            "Diagnostically rerun the periodic IPOPT refinement after each MHE "
            "transfer before calling ACADOS. Its time is warmstart overhead."
        ),
    )
    parser.add_argument(
        "--periodic-ipopt-refinement-window-cache",
        action="store_true",
        help=(
            "Cache successful diagnostic IPOPT refinements by window so repeated "
            "ACADOS transfer experiments reuse the same feasible seed."
        ),
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
        choices=("target", "collocation", "rk4", "irk"),
        default="target",
        help=(
            "Integrator used by the periodic IPOPT bridge. 'target' preserves "
            "the main Bioptim transcription; 'collocation' provides the robust "
            "periodic bridge; 'irk' can match an ACADOS IRK map."
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
        "bind_first_node_fes_states": args.acados_bind_first_node_fes_states,
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


def _periodic_ipopt_window_refinement_cache_path(
    args: argparse.Namespace, model_path: Path, window: int
) -> Path:
    if window < 0:
        raise ValueError("The IPOPT refinement cache window must be non-negative.")
    base_path = _periodic_ipopt_refinement_cache_path(args, model_path)
    return base_path.with_name(f"{base_path.stem}_window_{window:04d}.npz")


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
        "control_regularization_target_source": args.control_regularization_target_source,
        "wheel_qdot_regularization_weight": args.wheel_qdot_regularization_weight,
        "wheel_qdot_regularization_target": args.wheel_qdot_regularization_target,
        "wheel_qdot_bound_margin": args.wheel_qdot_bound_margin,
        "terminal_qdot_regularization_weight": (
            args.terminal_qdot_regularization_weight
        ),
        "terminal_qdot_regularization_target_source": (
            args.terminal_qdot_regularization_target_source
        ),
        "terminal_wheel_q_slack": args.acados_terminal_wheel_q_slack,
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
        "wheel_qdot_bound_margin": args.wheel_qdot_bound_margin,
        "terminal_qdot_regularization_weight": (
            args.terminal_qdot_regularization_weight
        ),
        "terminal_qdot_regularization_target_source": (
            args.terminal_qdot_regularization_target_source
        ),
        "pulse_width_trust_radius": args.acados_pulse_width_trust_radius,
        "fes_state_trust_radius": args.acados_fes_state_trust_radius,
        "wheel_q_slack": args.acados_wheel_q_slack,
        "terminal_wheel_q_slack": args.acados_terminal_wheel_q_slack,
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
        "wheel_qdot_bound_margin": args.wheel_qdot_bound_margin,
        "terminal_qdot_regularization_weight": (
            args.terminal_qdot_regularization_weight
        ),
        "terminal_qdot_regularization_target_source": (
            args.terminal_qdot_regularization_target_source
        ),
        "acados_terminal_wheel_q_slack": args.acados_terminal_wheel_q_slack,
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
        "acados_qp_cond_n": args.acados_qp_cond_n,
        "acados_hpipm_mode": args.acados_hpipm_mode,
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
        "acados_store_iterates": args.acados_store_iterates,
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
    qp_cond_n: int | None,
    hpipm_mode: str,
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
    store_iterates: bool,
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
    set_acados_unsafe_option(solver, hpipm_mode, "hpipm_mode")
    if qp_cond_n is not None:
        set_acados_unsafe_option(solver, qp_cond_n, "qp_solver_cond_N")
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
    set_acados_unsafe_option(solver, store_iterates, "store_iterates")
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


def acados_homotopy_stage_is_restartable(
    diagnostics: dict, convergence_tolerance: float
) -> bool:
    """Allow another SQP call only when feasibility is already under control."""

    residuals = diagnostics.get("residuals")
    if residuals is None:
        return False
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    return bool(
        residuals.size >= 4
        and np.all(np.isfinite(residuals[:4]))
        and np.max(np.abs(residuals[1:4])) <= convergence_tolerance
    )


def set_acados_runtime_max_iterations(periodic_nmpc, max_iterations: int) -> bool:
    """Update the generated capsule when Bioptim reuses an existing Acados solver."""

    acados_interface = getattr(periodic_nmpc, "ocp_solver", None)
    acados_solver = getattr(acados_interface, "ocp_solver", None)
    options_set = getattr(acados_solver, "options_set", None)
    if options_set is None:
        return False
    try:
        options_set("nlp_solver_max_iter", int(max_iterations))
    except (AttributeError, ValueError):
        return False
    return True


def reset_acados_solver_memory(periodic_nmpc) -> bool:
    """Clear a failed Acados iterate before Bioptim reloads the primal guess."""

    acados_interface = getattr(periodic_nmpc, "ocp_solver", None)
    acados_solver = getattr(acados_interface, "ocp_solver", None)
    reset = getattr(acados_solver, "reset", None)
    if reset is None:
        return False
    reset(reset_qp_solver_mem=1)
    return True


def apply_acados_capsule_primal_to_initial_guess(
    periodic_nmpc, iterate_index: int | None = None
) -> dict:
    """Copy the current scaled ACADOS primal, including failed iterates, to Bioptim."""

    interface = getattr(periodic_nmpc, "ocp_solver", None)
    acados_solver = getattr(interface, "ocp_solver", None)
    get_stage_value = getattr(acados_solver, "get", None)
    if get_stage_value is None:
        return {"applied": False, "reason": "solver_unavailable"}

    stored_iterate = None
    stored_iterate_error = None
    get_iterate = getattr(acados_solver, "get_iterate", None)
    if get_iterate is not None:
        try:
            stored_iterate = get_iterate(-1 if iterate_index is None else iterate_index)
        except (IndexError, RuntimeError, TypeError, ValueError) as exc:
            stored_iterate_error = str(exc)

    nlp = periodic_nmpc.nlp[0]
    first_state_key = next(iter(nlp.x_init.keys()))
    first_control_key = next(iter(nlp.u_init.keys()))
    n_state_nodes = nlp.x_init[first_state_key].init.shape[1]
    n_control_nodes = nlp.u_init[first_control_key].init.shape[1]
    state_scaling = _acados_variable_scaling(nlp, nlp.states, nlp.x_scaling)
    control_scaling = _acados_variable_scaling(nlp, nlp.controls, nlp.u_scaling)
    parameter_count = int(getattr(nlp.parameters, "shape", 0))
    states = np.empty((nlp.states.shape, n_state_nodes))
    controls = np.empty((nlp.controls.shape, n_control_nodes))

    for stage in range(n_state_nodes):
        stage_state = np.asarray(
            (
                stored_iterate.x_traj[stage]
                if stored_iterate is not None
                else get_stage_value(stage, "x")
            ),
            dtype=float,
        ).reshape(-1)
        if stage_state.size < parameter_count + nlp.states.shape:
            return {
                "applied": False,
                "reason": "state_dimension_mismatch",
                "stage": stage,
            }
        states[:, stage] = (
            stage_state[parameter_count : parameter_count + nlp.states.shape]
            * state_scaling
        )
    for stage in range(n_control_nodes):
        stage_control = np.asarray(
            (
                stored_iterate.u_traj[stage]
                if stored_iterate is not None
                else get_stage_value(stage, "u")
            ),
            dtype=float,
        ).reshape(-1)
        if stage_control.size < nlp.controls.shape:
            return {
                "applied": False,
                "reason": "control_dimension_mismatch",
                "stage": stage,
            }
        controls[:, stage] = stage_control[: nlp.controls.shape] * control_scaling

    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(controls)):
        return {"applied": False, "reason": "nonfinite_primal"}

    state_before = _stack_initial_guess_values(nlp.x_init, nlp.states, n_state_nodes)
    control_before = _stack_initial_guess_values(
        nlp.u_init, nlp.controls, n_control_nodes
    )
    for key in nlp.states.keys():
        nlp.x_init[key].init[:, :] = states[nlp.states[key].index, :]
    for key in nlp.controls.keys():
        nlp.u_init[key].init[:, :] = controls[nlp.controls[key].index, :]
    bound_projection = None
    if hasattr(periodic_nmpc, "_correct_init_guess_to_fit_bounds"):
        bound_projection = project_transferred_initial_guess_to_bounds(periodic_nmpc)
    final_states = _stack_initial_guess_values(nlp.x_init, nlp.states, n_state_nodes)
    final_controls = _stack_initial_guess_values(
        nlp.u_init, nlp.controls, n_control_nodes
    )
    return {
        "applied": True,
        "reason": None,
        "source": "stored_iterate" if stored_iterate is not None else "capsule",
        "iterate_index": iterate_index,
        "stored_iterate_error": stored_iterate_error,
        "bound_projection": bound_projection,
        "state_max_change": float(np.max(np.abs(final_states - state_before))),
        "control_max_change": float(np.max(np.abs(final_controls - control_before))),
    }


def run_acados_control_homotopy(
    periodic_nmpc,
    solver,
    radii: tuple[float, ...],
    convergence_tolerance: float,
    fixed_control_tolerance: float,
    max_restarts: int = 2,
    stage_iterations: int | None = 54,
    echo: bool = True,
    solve_stage=None,
) -> list[dict]:
    """Build an IRK-feasible seed before progressively releasing pulse widths."""

    stage_solver = deepcopy(solver)
    stage_solver.set_convergence_tolerance(convergence_tolerance)
    acados_interface = getattr(periodic_nmpc, "ocp_solver", None)
    if getattr(acados_interface, "ocp_solver", None) is not None:
        # The generated solver already owns this immutable maximum. The initial
        # homotopy compiles it with stage_iterations; subsequent windows must not
        # ask Bioptim to reconfigure the same Acados capsule.
        mark_options_unchanged = getattr(
            stage_solver, "set_only_first_options_has_changed", None
        )
        if mark_options_unchanged is not None:
            mark_options_unchanged(False)
    summaries = []

    if solve_stage is None:

        def solve_stage():
            return super(RecedingHorizonOptimization, periodic_nmpc).solve(
                solver=stage_solver,
                warm_start=None,
            )

    stages = (("fixed", None),) + tuple(("radius", radius) for radius in radii)
    try:
        for stage_index, (kind, radius) in enumerate(stages):
            restore_pulse_width_control_bounds(periodic_nmpc)
            periodic_nmpc._cocofest_fix_controls_to_warmup = kind == "fixed"
            periodic_nmpc._cocofest_fixed_control_tolerance = fixed_control_tolerance
            if radius is not None:
                apply_pulse_width_control_trust_region(periodic_nmpc, radius)

            stage_accepted = False
            for attempt in range(max_restarts + 1):
                if stage_iterations is not None:
                    set_acados_runtime_max_iterations(periodic_nmpc, stage_iterations)
                solution = solve_stage()
                diagnostics = snapshot_acados_diagnostics(solution)
                accepted = _status_is_success(
                    solution.status
                ) or acados_diagnostics_meet_tolerances(
                    diagnostics,
                    convergence_tolerance=convergence_tolerance,
                    stationarity_tolerance=convergence_tolerance,
                )
                restartable = (
                    not accepted
                    and solution.status == 2
                    and attempt < max_restarts
                    and acados_homotopy_stage_is_restartable(
                        diagnostics, convergence_tolerance
                    )
                )
                residuals = diagnostics.get("residuals")
                summary = {
                    "stage": stage_index,
                    "attempt": attempt,
                    "kind": kind,
                    "radius": radius,
                    "status": solution.status,
                    "accepted": accepted,
                    "restartable": restartable,
                    "residuals": (
                        None
                        if residuals is None
                        else np.asarray(residuals, dtype=float).copy()
                    ),
                    "solver_time_s": solution.solver_time_to_optimize,
                    "wall_time_s": solution.real_time_to_optimize,
                }
                summaries.append(summary)
                if echo:
                    print(
                        "acados_control_homotopy: "
                        f"stage={stage_index} attempt={attempt} kind={kind} "
                        f"radius={radius} status={solution.status} "
                        f"accepted={accepted} restartable={restartable} "
                        f"residuals={_format_array(summary['residuals'])}"
                    )
                if accepted:
                    apply_solution_directly_to_periodic_nmpc_initial_guess(
                        periodic_nmpc, solution
                    )
                    stage_accepted = True
                    break
                if not restartable:
                    summary["solver_reset"] = reset_acados_solver_memory(periodic_nmpc)
                    break
                apply_solution_directly_to_periodic_nmpc_initial_guess(
                    periodic_nmpc, solution
                )
            if not stage_accepted:
                break
    finally:
        periodic_nmpc._cocofest_fix_controls_to_warmup = False
        restore_pulse_width_control_bounds(periodic_nmpc)

    return summaries


def run_acados_proximal_control_continuation(
    periodic_nmpc,
    solver,
    weights: tuple[float, ...],
    convergence_tolerance: float,
    max_restarts: int = 1,
    stage_iterations: int | None = 50,
    try_next_weight_on_failure: bool = False,
    echo: bool = True,
    solve_stage=None,
) -> list[dict]:
    """Relax pulse-width proximity through W while retaining physical bounds."""

    stage_solver = deepcopy(solver)
    stage_solver.set_convergence_tolerance(convergence_tolerance)
    acados_interface = getattr(periodic_nmpc, "ocp_solver", None)
    if getattr(acados_interface, "ocp_solver", None) is not None:
        mark_options_unchanged = getattr(
            stage_solver, "set_only_first_options_has_changed", None
        )
        if mark_options_unchanged is not None:
            mark_options_unchanged(False)

    if solve_stage is None:

        def solve_stage():
            return super(RecedingHorizonOptimization, periodic_nmpc).solve(
                solver=stage_solver,
                warm_start=None,
            )

    summaries = []
    accepted_weight = None
    for stage_index, weight in enumerate(weights):
        stage_accepted = False
        for attempt in range(max_restarts + 1):
            runtime_update = set_acados_runtime_control_regularization_weight(
                periodic_nmpc, weight
            )
            if stage_iterations is not None:
                set_acados_runtime_max_iterations(periodic_nmpc, stage_iterations)
            solution = solve_stage()
            diagnostics = snapshot_acados_diagnostics(solution)
            residual_history = _acados_residual_history_summary(diagnostics)
            accepted = _status_is_success(
                solution.status
            ) or acados_diagnostics_meet_tolerances(
                diagnostics,
                convergence_tolerance=convergence_tolerance,
                stationarity_tolerance=convergence_tolerance,
            )
            restart_residuals = (
                residual_history.get("best")
                if residual_history
                else diagnostics.get("residuals")
            )
            restartable = (
                not accepted
                and solution.status in (2, 4)
                and attempt < max_restarts
                and acados_homotopy_stage_is_restartable(
                    {"residuals": restart_residuals}, convergence_tolerance
                )
            )
            residuals = diagnostics.get("residuals")
            summary = {
                "stage": stage_index,
                "attempt": attempt,
                "weight": weight,
                "status": solution.status,
                "accepted": accepted,
                "restartable": restartable,
                "runtime_weight_update": runtime_update,
                "residual_history": residual_history,
                "residuals": (
                    None
                    if residuals is None
                    else np.asarray(residuals, dtype=float).copy()
                ),
                "solver_time_s": solution.solver_time_to_optimize,
                "wall_time_s": solution.real_time_to_optimize,
            }
            summaries.append(summary)
            if echo:
                print(
                    "acados_proximal_control: "
                    f"stage={stage_index} attempt={attempt} weight={weight:.6g} "
                    f"status={solution.status} accepted={accepted} "
                    f"restartable={restartable} "
                    f"runtime_update={runtime_update['applied']} "
                    f"residuals={_format_array(summary['residuals'])}"
                )
            if accepted:
                apply_solution_directly_to_periodic_nmpc_initial_guess(
                    periodic_nmpc, solution
                )
                accepted_weight = weight
                stage_accepted = True
                break
            if not restartable:
                summary["solver_reset"] = reset_acados_solver_memory(periodic_nmpc)
                break
            capsule_summary = apply_acados_capsule_primal_to_initial_guess(
                periodic_nmpc,
                iterate_index=(
                    residual_history.get("best_index") if residual_history else None
                ),
            )
            summary["restart_primal"] = capsule_summary
            if not capsule_summary["applied"]:
                apply_solution_directly_to_periodic_nmpc_initial_guess(
                    periodic_nmpc, solution
                )
            summary["solver_reset"] = reset_acados_solver_memory(periodic_nmpc)
        if not stage_accepted and not try_next_weight_on_failure:
            break

    if accepted_weight is not None:
        set_acados_runtime_control_regularization_weight(periodic_nmpc, accepted_weight)
        periodic_nmpc._cocofest_dual_warm_start_mode = "preserve"
    return summaries


def run_acados_terminal_wheel_bound_continuation(
    periodic_nmpc,
    solver,
    slacks: tuple[float, ...],
    convergence_tolerance: float,
    stage_iterations: int | None = 50,
    echo: bool = True,
    solve_stage=None,
) -> list[dict]:
    """Tighten the terminal crank-angle bound around one fixed target."""

    recenter_terminal_wheel_q_bound_slack(periodic_nmpc, slacks[0])
    stage_solver = deepcopy(solver)
    stage_solver.set_convergence_tolerance(convergence_tolerance)
    if solve_stage is None:

        def solve_stage():
            return super(RecedingHorizonOptimization, periodic_nmpc).solve(
                solver=stage_solver,
                warm_start=None,
            )

    summaries = []
    accepted_slack = None
    for stage_index, slack in enumerate(slacks):
        if stage_index:
            set_terminal_wheel_q_bound_slack(periodic_nmpc, slack)
        if stage_iterations is not None:
            set_acados_runtime_max_iterations(periodic_nmpc, stage_iterations)
        solution = solve_stage()
        diagnostics = snapshot_acados_diagnostics(solution)
        accepted = _status_is_success(
            solution.status
        ) or acados_diagnostics_meet_tolerances(
            diagnostics,
            convergence_tolerance=convergence_tolerance,
            stationarity_tolerance=convergence_tolerance,
        )
        residuals = diagnostics.get("residuals")
        summary = {
            "stage": stage_index,
            "slack": slack,
            "status": solution.status,
            "accepted": accepted,
            "residuals": (
                None if residuals is None else np.asarray(residuals, dtype=float).copy()
            ),
            "solver_time_s": solution.solver_time_to_optimize,
            "wall_time_s": solution.real_time_to_optimize,
        }
        summaries.append(summary)
        if echo:
            print(
                "acados_terminal_wheel_bound: "
                f"stage={stage_index} slack={slack:.6g} status={solution.status} "
                f"accepted={accepted} residuals={_format_array(residuals)}"
            )
        if not accepted:
            break
        apply_solution_directly_to_periodic_nmpc_initial_guess(periodic_nmpc, solution)
        accepted_slack = slack

    if accepted_slack is not None:
        set_terminal_wheel_q_bound_slack(periodic_nmpc, accepted_slack)
        terminal_slack = getattr(periodic_nmpc, "terminal_state_slack", None)
        if terminal_slack is not None and "q" in terminal_slack:
            terminal_slack["q"][2] = accepted_slack
        periodic_nmpc._cocofest_dual_warm_start_mode = "preserve"
    return summaries


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


def project_qdot_initial_guess_from_q(
    nmpc,
    start_node: int = 0,
    select_by_dynamics: bool = False,
    dynamics_substeps: int = 5,
    max_backtracking_steps: int = 5,
) -> dict:
    if "q" not in nmpc.nlp[0].x_init.keys() or "qdot" not in nmpc.nlp[0].x_init.keys():
        return {"applied": False, "max_change": 0.0, "clipped_count": 0}
    if dynamics_substeps < 1:
        raise ValueError("qdot projection dynamics_substeps must be positive.")
    if max_backtracking_steps < 0:
        raise ValueError("qdot projection backtracking steps must be non-negative.")

    dt = nmpc.cycle_duration / nmpc.cycle_len
    q = np.asarray(nmpc.nlp[0].x_init["q"].init, dtype=float)
    previous_qdot = np.asarray(nmpc.nlp[0].x_init["qdot"].init, dtype=float).copy()
    if start_node < 0 or start_node >= q.shape[1]:
        raise ValueError("qdot projection start_node must index the state trajectory.")
    qdot = np.empty_like(q)
    qdot[:, :-1] = np.diff(q, axis=1) / dt
    qdot[:, -1] = qdot[:, -2]

    lower, upper = _trajectory_bounds_for_guess(
        nmpc.nlp[0].x_bounds["qdot"], qdot.shape[1]
    )
    projected_qdot = np.minimum(np.maximum(qdot, lower), upper)
    projected_slice = projected_qdot[:, start_node:]
    previous_slice = previous_qdot[:, start_node:]
    raw_slice = qdot[:, start_node:]
    accepted_step = 1.0
    scaled_defect_before = None
    scaled_defect_after = None

    if select_by_dynamics:
        before = _full_dynamics_rollout_defect_details(
            nmpc, n_substeps=dynamics_substeps
        )

        def maximum_scaled_defect(details):
            return max(details.get("scaled_by_block", {}).values(), default=np.inf)

        scaled_defect_before = maximum_scaled_defect(before)
        if np.isfinite(scaled_defect_before):
            accepted_step = 0.0
            scaled_defect_after = scaled_defect_before
            tolerance = max(1e-12, abs(scaled_defect_before) * 1e-12)
            for step_index in range(max_backtracking_steps + 1):
                step = 0.5**step_index
                candidate = previous_slice + step * (projected_slice - previous_slice)
                nmpc.nlp[0].x_init["qdot"].init[:, start_node:] = candidate
                details = _full_dynamics_rollout_defect_details(
                    nmpc, n_substeps=dynamics_substeps
                )
                scaled_defect = maximum_scaled_defect(details)
                if (
                    np.isfinite(scaled_defect)
                    and scaled_defect < scaled_defect_after - tolerance
                ):
                    accepted_step = step
                    scaled_defect_after = scaled_defect
            accepted_qdot = previous_slice + accepted_step * (
                projected_slice - previous_slice
            )
            nmpc.nlp[0].x_init["qdot"].init[:, start_node:] = accepted_qdot
        else:
            nmpc.nlp[0].x_init["qdot"].init[:, start_node:] = projected_slice
    else:
        nmpc.nlp[0].x_init["qdot"].init[:, start_node:] = projected_slice

    nmpc._sync_acados_state_bounds()
    accepted_slice = np.asarray(
        nmpc.nlp[0].x_init["qdot"].init[:, start_node:], dtype=float
    )
    return {
        "applied": bool(accepted_step > 0.0),
        "start_node": start_node,
        "accepted_step": accepted_step,
        "scaled_defect_before": scaled_defect_before,
        "scaled_defect_after": scaled_defect_after,
        "max_change": float(np.max(np.abs(accepted_slice - previous_slice))),
        "clipped_count": int(np.count_nonzero(projected_slice != raw_slice)),
    }


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

    (
        merged_solution,
        source_window_solutions,
        exported_cycle_solutions,
    ) = _split_receding_solution(sol)
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
                f"iterations={getattr(window_solution, 'iterations', None)} "
                f"solver_time_s={_fmt(window_solution.solver_time_to_optimize)} "
                f"wall_time_s={_fmt(window_solution.real_time_to_optimize)}"
            )


def build_window_summary(sol, requested_windows: int, cycles_per_window: int) -> dict:
    (
        merged_solution,
        source_window_solutions,
        exported_cycle_solutions,
    ) = _split_receding_solution(sol)
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
        "window_iterations": [
            getattr(window_solution, "iterations", None)
            for window_solution in source_window_solutions
        ],
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


def receding_horizon_window_count(requested_cycles: int, cycles_per_window: int) -> int:
    """Number of overlapping windows required to export every requested cycle."""

    if requested_cycles < 1 or cycles_per_window < 1:
        raise ValueError("Cycle and window lengths must be strictly positive.")
    if requested_cycles < cycles_per_window:
        raise ValueError("Requested cycles must be at least the cycles per window.")
    return requested_cycles - cycles_per_window + 1


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
    periodic_node = next_state.size == 5

    def rhs(values, local_time):
        if not periodic_node:
            return _periodic_ding_rhs(muscle_model, values, pulse_width)
        return _to_numpy_vector(
            muscle_model.system_dynamics(
                states=values,
                controls=np.array([pulse_width]),
                time=np.array([local_time]),
                numerical_timeseries=np.array(
                    [muscle_model.post_stimulation_amplitude(), 0.0]
                ),
            )
        )

    for substep in range(n_substeps):
        local_time = substep * sub_dt
        k1 = rhs(next_state, local_time)
        k2 = rhs(next_state + 0.5 * sub_dt * k1, local_time + 0.5 * sub_dt)
        k3 = rhs(next_state + 0.5 * sub_dt * k2, local_time + 0.5 * sub_dt)
        k4 = rhs(next_state + sub_dt * k3, local_time + sub_dt)
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


def _maximum_state_initial_guess_bound_violation(nmpc) -> float:
    maximum = 0.0
    nlp = nmpc.nlp[0]
    for key in nlp.x_init.keys():
        values = np.asarray(nlp.x_init[key].init, dtype=float)
        lower, upper = _trajectory_bounds_for_guess(nlp.x_bounds[key], values.shape[1])
        violation = np.maximum(lower - values, 0.0) + np.maximum(values - upper, 0.0)
        if violation.size:
            maximum = max(maximum, float(np.max(violation)))
    return maximum


def project_full_dynamics_initial_guess(
    nmpc,
    proximity_weight: float = 1.0,
    defect_weight: float = 10.0,
    n_substeps: int = 10,
    max_backtracking_steps: int = 6,
) -> dict:
    """Sequential proximal phase I with a monotone backtracking safeguard."""

    if n_substeps < 1:
        raise ValueError("Phase-I RK4 substeps must be strictly positive.")
    if max_backtracking_steps < 0:
        raise ValueError("Phase-I backtracking steps must be non-negative.")
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
    state_snapshot = {
        key: np.asarray(nlp.x_init[key].init, dtype=float).copy()
        for key in nlp.x_init.keys()
    }
    bound_violation_before = _maximum_state_initial_guess_bound_violation(nmpc)
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
    candidate_after = _full_dynamics_rollout_defect_details(nmpc, n_substeps=n_substeps)
    candidate_bound_violation_after = _maximum_state_initial_guess_bound_violation(nmpc)
    candidate_state_values = {
        key: np.asarray(nlp.x_init[key].init, dtype=float).copy()
        for key in nlp.x_init.keys()
    }

    def maximum_scaled_defect(details: dict) -> float:
        return max(details.get("scaled_by_block", {}).values(), default=0.0)

    scaled_defect_before = maximum_scaled_defect(before)
    candidate_scaled_defect_after = maximum_scaled_defect(candidate_after)
    defect_tolerance = max(1e-12, abs(scaled_defect_before) * 1e-12)
    bound_tolerance = max(1e-12, abs(bound_violation_before) * 1e-12)

    def admissible(scaled_defect: float, bound_violation: float) -> bool:
        return bool(
            np.isfinite(scaled_defect)
            and np.isfinite(bound_violation)
            and scaled_defect <= scaled_defect_before + defect_tolerance
            and bound_violation <= bound_violation_before + bound_tolerance
        )

    candidate_accepted = admissible(
        candidate_scaled_defect_after, candidate_bound_violation_after
    )
    selected_step = 1.0 if candidate_accepted else 0.0
    selected_details = candidate_after if candidate_accepted else before
    selected_scaled_defect = (
        candidate_scaled_defect_after if candidate_accepted else scaled_defect_before
    )
    selected_bound_violation = (
        candidate_bound_violation_after
        if candidate_accepted
        else bound_violation_before
    )
    selected_state_values = (
        candidate_state_values if candidate_accepted else state_snapshot
    )
    backtracking_evaluations = []
    if not candidate_accepted:
        for step_index in range(1, max_backtracking_steps + 1):
            step = 0.5**step_index
            for key in nlp.x_init.keys():
                nlp.x_init[key].init[:, :] = state_snapshot[key] + step * (
                    candidate_state_values[key] - state_snapshot[key]
                )
            nmpc._correct_init_guess_to_fit_bounds(corrected_input="states")
            nmpc._sync_acados_state_bounds()
            trial_details = _full_dynamics_rollout_defect_details(
                nmpc, n_substeps=n_substeps
            )
            trial_scaled_defect = maximum_scaled_defect(trial_details)
            trial_bound_violation = _maximum_state_initial_guess_bound_violation(nmpc)
            trial_admissible = admissible(trial_scaled_defect, trial_bound_violation)
            backtracking_evaluations.append(
                {
                    "step": step,
                    "scaled_defect": trial_scaled_defect,
                    "bound_violation": trial_bound_violation,
                    "admissible": trial_admissible,
                }
            )
            if (
                trial_admissible
                and trial_scaled_defect < selected_scaled_defect - defect_tolerance
            ):
                selected_step = step
                selected_details = trial_details
                selected_scaled_defect = trial_scaled_defect
                selected_bound_violation = trial_bound_violation
                selected_state_values = {
                    key: np.asarray(nlp.x_init[key].init, dtype=float).copy()
                    for key in nlp.x_init.keys()
                }

    accepted = selected_step > 0.0
    for key, values in selected_state_values.items():
        nlp.x_init[key].init[:, :] = values
    nmpc._sync_acados_state_bounds()
    candidate_max_state_change = float(np.max(np.abs(states - reference)))
    selected_max_state_change = max(
        (
            float(np.max(np.abs(selected_state_values[key] - state_snapshot[key])))
            for key in state_snapshot
        ),
        default=0.0,
    )

    return {
        "proximity_weight": proximity_weight,
        "defect_weight": defect_weight,
        "n_substeps": n_substeps,
        "max_backtracking_steps": max_backtracking_steps,
        "backtracking_evaluations": backtracking_evaluations,
        "accepted": accepted,
        "restored": not accepted,
        "accepted_step": selected_step,
        "scaled_defect_before": scaled_defect_before,
        "scaled_defect_after": selected_scaled_defect,
        "candidate_scaled_defect_after": candidate_scaled_defect_after,
        "scaled_by_block_before": before.get("scaled_by_block", {}),
        "scaled_by_block_after": selected_details.get("scaled_by_block", {}),
        "candidate_scaled_by_block_after": candidate_after.get("scaled_by_block", {}),
        "absolute_by_block_before": before.get("absolute_by_block", {}),
        "absolute_by_block_after": selected_details.get("absolute_by_block", {}),
        "candidate_absolute_by_block_after": candidate_after.get(
            "absolute_by_block", {}
        ),
        "bound_violation_before": bound_violation_before,
        "bound_violation_after": selected_bound_violation,
        "candidate_bound_violation_after": candidate_bound_violation_after,
        "max_state_change": selected_max_state_change,
        "candidate_max_state_change": candidate_max_state_change,
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


def scale_appended_pulse_width_controls(periodic_nmpc, scale: float) -> dict:
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Appended pulse-width scale must be finite and positive.")

    start_node = periodic_nmpc.nodes_per_cycle
    summary = {}
    for key in periodic_nmpc.nlp[0].u_init.keys():
        if not key.startswith("last_pulse_width_"):
            continue
        values = periodic_nmpc.nlp[0].u_init[key].init
        if start_node < 0 or start_node >= values.shape[1]:
            raise ValueError("The appended control cycle starts outside the horizon.")
        bounds = periodic_nmpc.nlp[0].u_bounds[key]
        lower = float(np.min(np.asarray(bounds.min, dtype=float)))
        upper = float(np.max(np.asarray(bounds.max, dtype=float)))
        original = np.asarray(values[:, start_node:], dtype=float).copy()
        scaled = np.minimum(np.maximum(scale * original, lower), upper)
        values[:, start_node:] = scaled
        summary[key] = {
            "before_min": float(np.min(original)),
            "before_max": float(np.max(original)),
            "after_min": float(np.min(scaled)),
            "after_max": float(np.max(scaled)),
            "clipped_count": int(
                np.count_nonzero(
                    (scale * original < lower) | (scale * original > upper)
                )
            ),
        }
    return {"scale": scale, "start_node": start_node, "controls": summary}


def compensate_appended_pulse_widths_from_ding_force(
    periodic_nmpc,
    n_substeps: int = 5,
    bisection_iterations: int = 20,
    previous_solution=None,
) -> dict:
    """Match previous-cycle nodal forces with cheap per-muscle Ding rollouts."""

    if n_substeps < 1:
        raise ValueError("Ding force-compensation substeps must be positive.")
    if bisection_iterations < 1:
        raise ValueError("Ding force-compensation iterations must be positive.")

    nlp = periodic_nmpc.nlp[0]
    cycle_len = int(periodic_nmpc.cycle_len)
    first_control_key = next(iter(nlp.u_init.keys()))
    n_control_nodes = nlp.u_init[first_control_key].init.shape[1]
    start_node = n_control_nodes - cycle_len
    previous_start = start_node - cycle_len
    previous_states = None
    previous_controls_by_key = None
    if previous_start < 0 and previous_solution is not None:
        previous_states = previous_solution.decision_states(
            to_merge=SolutionMerge.NODES
        )
        previous_controls_by_key = previous_solution.decision_controls(
            to_merge=SolutionMerge.NODES
        )
        start_node = 0
    elif previous_start < 0:
        return {
            "applied": False,
            "reason": "requires_two_control_cycles_or_previous_solution",
            "start_node": start_node,
            "muscles": {},
        }

    dt = periodic_nmpc.cycle_duration / cycle_len
    summaries = {}
    for muscle_model in nlp.model.muscles_dynamics_model:
        muscle_name = muscle_model.muscle_name
        six_state_keys = _ding_state_keys(muscle_name)
        five_state_keys = (
            f"Cn_{muscle_name}",
            f"F_{muscle_name}",
            f"A_{muscle_name}",
            f"Tau1_{muscle_name}",
            f"Km_{muscle_name}",
        )
        state_keys = (
            six_state_keys
            if all(key in nlp.x_init.keys() for key in six_state_keys)
            else five_state_keys
        )
        control_key = f"last_pulse_width_{muscle_name}"
        if any(key not in nlp.x_init.keys() for key in state_keys):
            continue
        if control_key not in nlp.u_init.keys():
            continue

        state_trajectory = np.vstack(
            [np.asarray(nlp.x_init[key].init[0, :], dtype=float) for key in state_keys]
        )
        force_index = state_keys.index(f"F_{muscle_name}")
        controls = nlp.u_init[control_key].init
        if previous_controls_by_key is None:
            previous_controls = np.asarray(
                controls[0, previous_start:start_node], dtype=float
            )
            target_forces = np.asarray(
                state_trajectory[force_index, previous_start + 1 : start_node + 1],
                dtype=float,
            )
        else:
            if control_key not in previous_controls_by_key:
                continue
            force_key = f"F_{muscle_name}"
            if force_key not in previous_states:
                continue
            previous_controls = np.asarray(
                previous_controls_by_key[control_key], dtype=float
            ).reshape(-1)[:cycle_len]
            previous_force = np.asarray(
                previous_states[force_key], dtype=float
            ).reshape(-1)
            if (
                previous_controls.size < cycle_len
                or previous_force.size < cycle_len + 1
            ):
                continue
            target_forces = previous_force[1 : cycle_len + 1]
        original_appended = np.asarray(
            controls[0, start_node : start_node + cycle_len], dtype=float
        ).copy()
        current_state = np.asarray(state_trajectory[:, start_node], dtype=float).copy()
        baseline_state = current_state.copy()
        bounds = nlp.u_bounds[control_key]
        lower = float(np.min(np.asarray(bounds.min, dtype=float)))
        upper = float(np.max(np.asarray(bounds.max, dtype=float)))
        selected_controls = np.empty(cycle_len)
        selected_forces = np.empty(cycle_len)
        baseline_forces = np.empty(cycle_len)

        for local_node, target_force in enumerate(target_forces):
            baseline_state = _rk4_periodic_ding_step(
                muscle_model,
                baseline_state,
                original_appended[local_node],
                dt,
                n_substeps=n_substeps,
            )
            baseline_forces[local_node] = baseline_state[force_index]

            low_state = _rk4_periodic_ding_step(
                muscle_model,
                current_state,
                lower,
                dt,
                n_substeps=n_substeps,
            )
            high_state = _rk4_periodic_ding_step(
                muscle_model,
                current_state,
                upper,
                dt,
                n_substeps=n_substeps,
            )
            low_force = float(low_state[force_index])
            high_force = float(high_state[force_index])
            if (target_force - low_force) * (target_force - high_force) >= 0.0:
                if abs(target_force - low_force) <= abs(target_force - high_force):
                    selected = lower
                    selected_state = low_state
                else:
                    selected = upper
                    selected_state = high_state
            else:
                left = lower
                right = upper
                increasing = high_force >= low_force
                selected = 0.5 * (left + right)
                selected_state = _rk4_periodic_ding_step(
                    muscle_model,
                    current_state,
                    selected,
                    dt,
                    n_substeps=n_substeps,
                )
                for _ in range(bisection_iterations):
                    selected = 0.5 * (left + right)
                    selected_state = _rk4_periodic_ding_step(
                        muscle_model,
                        current_state,
                        selected,
                        dt,
                        n_substeps=n_substeps,
                    )
                    force_is_low = float(selected_state[force_index]) < target_force
                    if force_is_low == increasing:
                        left = selected
                    else:
                        right = selected

            selected_controls[local_node] = selected
            selected_forces[local_node] = selected_state[force_index]
            current_state = selected_state

        controls[0, start_node : start_node + cycle_len] = selected_controls
        safe_previous = np.maximum(np.abs(previous_controls), np.finfo(float).eps)
        gain = selected_controls / safe_previous
        summaries[muscle_name] = {
            "control_key": control_key,
            "gain_min": float(np.min(gain)),
            "gain_mean": float(np.mean(gain)),
            "gain_max": float(np.max(gain)),
            "pulse_width_min": float(np.min(selected_controls)),
            "pulse_width_max": float(np.max(selected_controls)),
            "baseline_force_rmse": float(
                np.sqrt(np.mean((baseline_forces - target_forces) ** 2))
            ),
            "compensated_force_rmse": float(
                np.sqrt(np.mean((selected_forces - target_forces) ** 2))
            ),
            "saturated_count": int(
                np.count_nonzero(
                    np.isclose(selected_controls, lower)
                    | np.isclose(selected_controls, upper)
                )
            ),
        }

    return {
        "applied": bool(summaries),
        "reason": None if summaries else "periodic_ding_states_unavailable",
        "start_node": start_node,
        "previous_start_node": previous_start if previous_states is None else None,
        "previous_solution_used": previous_states is not None,
        "n_substeps": n_substeps,
        "bisection_iterations": bisection_iterations,
        "muscles": summaries,
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

    # Nodes 0..nodes_per_cycle are retained from the solved horizon. Only the
    # intervals after that shared endpoint belong to the appended cycle.
    start_node = periodic_nmpc.nodes_per_cycle
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
    worst_bound_violation = None
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
        if key_bound_violation >= max_bound_violation:
            component, node = np.unravel_index(
                int(np.argmax(violations)), violations.shape
            )
            worst_bound_violation = {
                "key": key,
                "component": int(component),
                "node": int(node),
                "value": float(values[component, node]),
                "lower": float(lower[component, node]),
                "upper": float(upper[component, node]),
                "violation": key_bound_violation,
            }
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
        "worst_bound_violation": worst_bound_violation,
        "terminal_delta": terminal_delta,
        "max_delta_by_key": max_delta_by_key,
    }


def _acados_variable_scaling(nlp, variables, scaling_container) -> np.ndarray:
    scaling = np.ones(variables.shape)
    for key in variables.keys():
        indexes = np.asarray(variables[key].index).reshape((-1,)).tolist()
        key_scaling = np.asarray(
            scaling_container[key].scaling[:, 0], dtype=float
        ).reshape(-1)
        if key_scaling.size != len(indexes):
            raise ValueError(
                f"ACADOS scaling for '{key}' has {key_scaling.size} rows, "
                f"expected {len(indexes)}."
            )
        scaling[indexes] = key_scaling
    if not np.all(np.isfinite(scaling)) or np.any(scaling <= 0.0):
        raise ValueError(
            "ACADOS transfer scaling must be finite and strictly positive."
        )
    return scaling


def _get_or_create_acados_sim_solver(periodic_nmpc):
    cached_solver = getattr(periodic_nmpc, "_cocofest_acados_sim_solver", None)
    if cached_solver is not None:
        return cached_solver, False

    interface = getattr(periodic_nmpc, "ocp_solver", None)
    acados_ocp = getattr(interface, "acados_ocp", None)
    if acados_ocp is None or getattr(interface, "ocp_solver", None) is None:
        raise RuntimeError(
            "The ACADOS IRK transfer requires an initialized ACADOS OCP solver."
        )

    from acados_template import AcadosSimSolver

    library_path = (
        Path(acados_ocp.code_export_directory)
        / f"libacados_sim_solver_{acados_ocp.model.name}{_shared_lib_ext()}"
    )
    needs_build = not library_path.exists()
    simulator = AcadosSimSolver(
        acados_ocp,
        generate=False,
        build=needs_build,
        verbose=False,
    )
    periodic_nmpc._cocofest_acados_sim_solver = simulator
    return simulator, needs_build


def rollout_transferred_cycle_acados_irk(
    periodic_nmpc,
    max_allowed_bound_violation: float | None = None,
) -> dict:
    """Roll out the appended cycle with the same generated IRK map as the OCP."""

    nlp = periodic_nmpc.nlp[0]
    first_state_key = next(iter(nlp.x_init.keys()))
    first_control_key = next(iter(nlp.u_init.keys()))
    n_state_nodes = nlp.x_init[first_state_key].init.shape[1]
    n_control_nodes = nlp.u_init[first_control_key].init.shape[1]
    if n_state_nodes != n_control_nodes + 1:
        raise ValueError(
            "The ACADOS IRK transfer requires one more state node than controls."
        )

    start_node = (
        0
        if n_control_nodes <= periodic_nmpc.nodes_per_cycle
        else periodic_nmpc.nodes_per_cycle
    )
    if start_node < 0 or start_node >= n_control_nodes:
        raise ValueError("The ACADOS IRK transfer start node is outside the horizon.")

    simulator, simulator_built = _get_or_create_acados_sim_solver(periodic_nmpc)
    states = _stack_initial_guess_values(nlp.x_init, nlp.states, n_state_nodes)
    original_states = states.copy()
    controls = _stack_initial_guess_values(nlp.u_init, nlp.controls, n_control_nodes)
    state_scaling = _acados_variable_scaling(nlp, nlp.states, nlp.x_scaling)
    control_scaling = _acados_variable_scaling(nlp, nlp.controls, nlp.u_scaling)

    expected_states = int(simulator.acados_sim.dims.nx)
    expected_controls = int(simulator.acados_sim.dims.nu)
    if states.shape[0] != expected_states or controls.shape[0] != expected_controls:
        raise ValueError(
            "ACADOS simulator dimensions do not match the Bioptim variables "
            f"(x: {states.shape[0]} != {expected_states}, "
            f"u: {controls.shape[0]} != {expected_controls})."
        )
    interval_duration = periodic_nmpc.cycle_duration / periodic_nmpc.cycle_len

    simulation_time_s = 0.0
    for node in range(start_node, n_control_nodes):
        stage_parameters = _numerical_timeseries_at_node(nlp, node)
        simulator.set("T", np.array([interval_duration]))
        simulator.set("t0", np.array([node * interval_duration]))
        try:
            next_scaled_state = simulator.simulate(
                x=states[:, node] / state_scaling,
                u=controls[:, node] / control_scaling,
                p=stage_parameters,
            )
        except RuntimeError as exc:
            return {
                "applied": False,
                "start_node": start_node,
                "failed_node": node,
                "reason": str(exc),
                "simulator_built": simulator_built,
                "simulation_time_s": simulation_time_s,
            }
        simulation_time_s += float(simulator.get("time_tot"))
        states[:, node + 1] = np.asarray(next_scaled_state) * state_scaling
        if not np.all(np.isfinite(states[:, node + 1])):
            return {
                "applied": False,
                "start_node": start_node,
                "nonfinite_node": node + 1,
                "reason": "nonfinite_state",
                "simulator_built": simulator_built,
                "simulation_time_s": simulation_time_s,
            }

    max_bound_violation = 0.0
    max_bound_violation_by_key = {}
    worst_bound_violation = None
    terminal_delta = {}
    state_values = {}
    for key in nlp.states.keys():
        indexes = np.asarray(nlp.states[key].index).reshape((-1,)).tolist()
        values = states[indexes, :]
        state_values[key] = values
        lower, upper = _trajectory_bounds_for_guess(nlp.x_bounds[key], n_state_nodes)
        violations = np.maximum(lower - values, 0.0) + np.maximum(values - upper, 0.0)
        key_bound_violation = float(np.max(violations))
        max_bound_violation_by_key[key] = key_bound_violation
        if key_bound_violation >= max_bound_violation:
            component, node = np.unravel_index(
                int(np.argmax(violations)), violations.shape
            )
            worst_bound_violation = {
                "key": key,
                "component": int(component),
                "node": int(node),
                "value": float(values[component, node]),
                "lower": float(lower[component, node]),
                "upper": float(upper[component, node]),
                "violation": key_bound_violation,
            }
        max_bound_violation = max(max_bound_violation, key_bound_violation)
        terminal_delta[key] = float(
            np.max(np.abs(values[:, -1] - original_states[indexes, -1]))
        )

    applied = (
        max_allowed_bound_violation is None
        or max_bound_violation <= max_allowed_bound_violation
    )
    if applied:
        for key, values in state_values.items():
            nlp.x_init[key].init[:, :] = values
    rk4_defects_after = (
        _full_dynamics_rollout_defect_details(periodic_nmpc) if applied else None
    )

    return {
        "applied": applied,
        "start_node": start_node,
        "max_bound_violation": max_bound_violation,
        "max_bound_violation_by_key": max_bound_violation_by_key,
        "worst_bound_violation": worst_bound_violation,
        "terminal_delta": terminal_delta,
        "simulator_built": simulator_built,
        "simulation_time_s": simulation_time_s,
        "interval_duration": interval_duration,
        "rk4_defects_after": rk4_defects_after,
    }


def _copy_state_bounds(periodic_nmpc) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        key: (
            np.asarray(periodic_nmpc.nlp[0].x_bounds[key].min, dtype=float).copy(),
            np.asarray(periodic_nmpc.nlp[0].x_bounds[key].max, dtype=float).copy(),
        )
        for key in periodic_nmpc.nlp[0].x_bounds.keys()
    }


def build_relaxed_transfer_state_bounds(
    periodic_nmpc,
    padding: float,
    relaxed_keys: tuple[str, ...] = ("q", "qdot"),
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, float]]:
    """Enclose selected transferred states without relaxing their first node."""

    if padding < 0.0:
        raise ValueError("Transfer-bound homotopy padding must be non-negative.")

    relaxed_bounds = _copy_state_bounds(periodic_nmpc)
    expansion_by_key = {}
    for key, (relaxed_min, relaxed_max) in relaxed_bounds.items():
        if key not in relaxed_keys:
            expansion_by_key[key] = 0.0
            continue
        values = np.asarray(periodic_nmpc.nlp[0].x_init[key].init, dtype=float)
        if values.shape[1] < 2 or relaxed_min.shape[1] < 3:
            raise ValueError(
                "Transfer-bound homotopy requires path and terminal bound columns."
            )

        original_min = relaxed_min.copy()
        original_max = relaxed_max.copy()
        path_values = values[:, 1:-1]
        if path_values.shape[1]:
            path_range = np.ptp(path_values, axis=1)
            path_bound_range = original_max[:, 1] - original_min[:, 1]
            path_scale = np.maximum.reduce(
                (
                    path_range,
                    np.where(np.isfinite(path_bound_range), path_bound_range, 0.0),
                    np.full(path_range.shape, np.finfo(float).eps),
                )
            )
            path_padding = padding * path_scale
            relaxed_min[:, 1] = np.minimum(
                original_min[:, 1], np.min(path_values, axis=1) - path_padding
            )
            relaxed_max[:, 1] = np.maximum(
                original_max[:, 1], np.max(path_values, axis=1) + path_padding
            )

        terminal_values = values[:, -1]
        trajectory_range = np.ptp(values, axis=1)
        terminal_bound_range = original_max[:, 2] - original_min[:, 2]
        terminal_scale = np.maximum.reduce(
            (
                trajectory_range,
                np.where(np.isfinite(terminal_bound_range), terminal_bound_range, 0.0),
                np.full(trajectory_range.shape, np.finfo(float).eps),
            )
        )
        terminal_padding = padding * terminal_scale
        relaxed_min[:, 2] = np.minimum(
            original_min[:, 2], terminal_values - terminal_padding
        )
        relaxed_max[:, 2] = np.maximum(
            original_max[:, 2], terminal_values + terminal_padding
        )

        # Inter-window continuity is never part of this homotopy.
        relaxed_min[:, 0] = original_min[:, 0]
        relaxed_max[:, 0] = original_max[:, 0]
        expansion_by_key[key] = float(
            max(
                np.max(original_min - relaxed_min),
                np.max(relaxed_max - original_max),
            )
        )

    return relaxed_bounds, expansion_by_key


def apply_transfer_state_bound_fraction(
    periodic_nmpc,
    original_bounds: dict[str, tuple[np.ndarray, np.ndarray]],
    relaxed_bounds: dict[str, tuple[np.ndarray, np.ndarray]],
    fraction: float,
) -> None:
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError("Transfer-bound homotopy fraction must be between 0 and 1.")

    for key, (original_min, original_max) in original_bounds.items():
        relaxed_min, relaxed_max = relaxed_bounds[key]
        bounds = periodic_nmpc.nlp[0].x_bounds[key]
        bounds.min[:, :] = relaxed_min + fraction * (original_min - relaxed_min)
        bounds.max[:, :] = relaxed_max + fraction * (original_max - relaxed_max)
        bounds.min[:, 0] = original_min[:, 0]
        bounds.max[:, 0] = original_max[:, 0]
    periodic_nmpc._sync_acados_state_bounds()


def _acados_residual_history_summary(diagnostics: dict) -> dict:
    rows = []
    for key in ("res_stat_all", "res_eq_all", "res_ineq_all", "res_comp_all"):
        values = diagnostics.get(key)
        if values is None or isinstance(values, dict):
            return {}
        values = np.asarray(values, dtype=float).reshape(-1)
        if not values.size:
            return {}
        rows.append(values)
    common_size = min(values.size for values in rows)
    history = np.vstack([values[:common_size] for values in rows])
    finite_columns = np.all(np.isfinite(history), axis=0)
    if not np.any(finite_columns):
        return {}
    feasibility = np.max(np.abs(history[1:]), axis=0)
    stationarity = np.abs(history[0])
    candidate_indices = np.flatnonzero(finite_columns)
    best_index = int(
        candidate_indices[
            np.lexsort(
                (
                    stationarity[candidate_indices],
                    feasibility[candidate_indices],
                )
            )[0]
        ]
    )
    return {
        "initial": history[:, 0],
        "best": history[:, best_index],
        "best_index": best_index,
        "componentwise_best": np.min(np.abs(history), axis=1),
        "final": history[:, -1],
    }


def run_acados_transfer_bound_homotopy(
    periodic_nmpc,
    solver,
    fractions: tuple[float, ...],
    padding: float,
    convergence_tolerance: float,
    stage_iterations: int,
    solver_tolerance: float | None = None,
    max_restarts: int = 1,
    echo: bool = True,
    solve_stage=None,
) -> dict:
    """Recover a feasible transfer while tightening state bounds to their target."""

    original_bounds = _copy_state_bounds(periodic_nmpc)
    relaxed_bounds, expansion_by_key = build_relaxed_transfer_state_bounds(
        periodic_nmpc, padding
    )
    original_fix_controls = getattr(
        periodic_nmpc, "_cocofest_fix_controls_to_warmup", False
    )
    original_runtime_iterations = getattr(solver, "nlp_solver_max_iter", None)
    stage_solver = deepcopy(solver)
    stage_solver.set_convergence_tolerance(
        convergence_tolerance if solver_tolerance is None else solver_tolerance
    )
    stage_solver.set_maximum_iterations(stage_iterations)
    acados_interface = getattr(periodic_nmpc, "ocp_solver", None)
    if getattr(acados_interface, "ocp_solver", None) is not None:
        mark_options_unchanged = getattr(
            stage_solver, "set_only_first_options_has_changed", None
        )
        if mark_options_unchanged is not None:
            mark_options_unchanged(False)

    if solve_stage is None:

        def solve_stage():
            return super(RecedingHorizonOptimization, periodic_nmpc).solve(
                solver=stage_solver,
                warm_start=None,
            )

    accepted_states = snapshot_container(periodic_nmpc.nlp[0].x_init)
    accepted_controls = snapshot_container(periodic_nmpc.nlp[0].u_init)
    summaries = []
    completed = False
    try:
        for stage_index, fraction in enumerate(fractions):
            apply_transfer_state_bound_fraction(
                periodic_nmpc, original_bounds, relaxed_bounds, fraction
            )
            restore_pulse_width_control_bounds(periodic_nmpc)
            periodic_nmpc._cocofest_fix_controls_to_warmup = False
            control_radius = None
            periodic_nmpc._correct_init_guess_to_fit_bounds(corrected_input="states")
            stage_accepted = False
            for attempt in range(max_restarts + 1):
                set_acados_runtime_max_iterations(periodic_nmpc, stage_iterations)
                solution = solve_stage()
                diagnostics = snapshot_acados_diagnostics(solution)
                residuals = diagnostics.get("residuals")
                residuals = (
                    None
                    if residuals is None
                    else np.asarray(residuals, dtype=float).reshape(-1)
                )
                residual_history = _acados_residual_history_summary(diagnostics)
                final_history_residuals = residual_history.get("final")
                final_history_accepted = bool(
                    final_history_residuals is not None
                    and np.all(np.isfinite(final_history_residuals))
                    and np.max(np.abs(final_history_residuals)) <= convergence_tolerance
                )
                accepted = (
                    _status_is_success(solution.status)
                    or acados_diagnostics_meet_tolerances(
                        diagnostics,
                        convergence_tolerance=convergence_tolerance,
                        stationarity_tolerance=convergence_tolerance,
                    )
                    or final_history_accepted
                )
                retryable = bool(
                    not accepted
                    and solution.status in (2, 4)
                    and attempt < max_restarts
                    and residuals is not None
                    and residuals.size >= 4
                    and np.all(np.isfinite(residuals[:4]))
                    and np.max(np.abs(residuals[1:4])) <= 100.0 * convergence_tolerance
                )
                summary = {
                    "stage": stage_index,
                    "attempt": attempt,
                    "fraction": fraction,
                    "control_radius": control_radius,
                    "status": solution.status,
                    "accepted": accepted,
                    "accepted_from_residual_history": final_history_accepted,
                    "retryable": retryable,
                    "residuals": None if residuals is None else residuals.copy(),
                    "residual_history": residual_history,
                    "solver_time_s": solution.solver_time_to_optimize,
                    "wall_time_s": solution.real_time_to_optimize,
                }
                summaries.append(summary)
                if echo:
                    print(
                        "acados_transfer_bound_homotopy: "
                        f"stage={stage_index} attempt={attempt} "
                        f"fraction={fraction:.6g} status={solution.status} "
                        f"control_radius={control_radius} "
                        f"accepted={accepted} retryable={retryable} "
                        f"residuals={_format_array(residuals)}"
                    )
                    if residual_history:
                        print(
                            "acados_transfer_bound_homotopy_residual_history: "
                            f"initial={_format_array(residual_history['initial'])} "
                            f"best={_format_array(residual_history['best'])} "
                            f"final={_format_array(residual_history['final'])}"
                        )
                if accepted:
                    apply_solution_directly_to_periodic_nmpc_initial_guess(
                        periodic_nmpc, solution
                    )
                    if solution.status != 0:
                        summary["solver_reset"] = reset_acados_solver_memory(
                            periodic_nmpc
                        )
                    accepted_states = snapshot_container(periodic_nmpc.nlp[0].x_init)
                    accepted_controls = snapshot_container(periodic_nmpc.nlp[0].u_init)
                    stage_accepted = True
                    break
                if not retryable:
                    summary["solver_reset"] = reset_acados_solver_memory(periodic_nmpc)
                    break
                apply_solution_directly_to_periodic_nmpc_initial_guess(
                    periodic_nmpc, solution
                )
                reset_acados_solver_memory(periodic_nmpc)

            if not stage_accepted:
                break
        completed = bool(
            summaries and summaries[-1]["accepted"] and np.isclose(fractions[-1], 1.0)
        )
    finally:
        for key, values in accepted_states.items():
            periodic_nmpc.nlp[0].x_init[key].init[:, :] = values
        for key, values in accepted_controls.items():
            periodic_nmpc.nlp[0].u_init[key].init[:, :] = values
        restore_pulse_width_control_bounds(periodic_nmpc)
        for key, (lower, upper) in original_bounds.items():
            periodic_nmpc.nlp[0].x_bounds[key].min[:, :] = lower
            periodic_nmpc.nlp[0].x_bounds[key].max[:, :] = upper
        periodic_nmpc._cocofest_fix_controls_to_warmup = original_fix_controls
        periodic_nmpc._sync_acados_state_bounds()
        if original_runtime_iterations is not None:
            set_acados_runtime_max_iterations(
                periodic_nmpc, int(original_runtime_iterations)
            )

    return {
        "completed": completed,
        "stages": summaries,
        "expansion_by_key": expansion_by_key,
        "max_expansion": max(expansion_by_key.values(), default=0.0),
    }


def run_acados_transfer_sqp_restarts(
    periodic_nmpc,
    solver,
    max_restarts: int,
    stage_iterations: int,
    feasibility_tolerance: float,
    echo: bool = True,
    solve_stage=None,
) -> dict:
    """Restart short SQP attempts from the best nearly feasible transfer iterate."""

    original_runtime_iterations = getattr(solver, "nlp_solver_max_iter", None)
    stage_solver = deepcopy(solver)
    stage_solver.set_maximum_iterations(stage_iterations)
    acados_interface = getattr(periodic_nmpc, "ocp_solver", None)
    if getattr(acados_interface, "ocp_solver", None) is not None:
        mark_options_unchanged = getattr(
            stage_solver, "set_only_first_options_has_changed", None
        )
        if mark_options_unchanged is not None:
            mark_options_unchanged(False)

    if solve_stage is None:

        def solve_stage():
            return super(RecedingHorizonOptimization, periodic_nmpc).solve(
                solver=stage_solver,
                warm_start=None,
            )

    summaries = []
    best_feasibility = np.inf
    best_stationarity = np.inf
    best_states = snapshot_container(periodic_nmpc.nlp[0].x_init)
    best_controls = snapshot_container(periodic_nmpc.nlp[0].u_init)
    completed = False
    try:
        for attempt in range(max_restarts):
            set_acados_runtime_max_iterations(periodic_nmpc, stage_iterations)
            solution = solve_stage()
            diagnostics = snapshot_acados_diagnostics(solution)
            reported_residuals = diagnostics.get("residuals")
            reported_residuals = (
                None
                if reported_residuals is None
                else np.asarray(reported_residuals, dtype=float).reshape(-1)
            )
            residual_history = _acados_residual_history_summary(diagnostics)
            residuals = (
                np.asarray(residual_history["best"], dtype=float).reshape(-1)
                if residual_history
                else reported_residuals
            )
            finite = bool(
                residuals is not None
                and residuals.size >= 4
                and np.all(np.isfinite(residuals[:4]))
            )
            feasibility = float(np.max(np.abs(residuals[1:4]))) if finite else np.inf
            stationarity = float(abs(residuals[0])) if finite else np.inf
            accepted_for_restart = bool(
                finite
                and feasibility <= feasibility_tolerance
                and (
                    stationarity < best_stationarity
                    or (
                        np.isclose(stationarity, best_stationarity)
                        and feasibility < best_feasibility
                    )
                )
            )
            solver_success = _status_is_success(solution.status)
            summary = {
                "attempt": attempt,
                "status": solution.status,
                "solver_success": solver_success,
                "accepted_for_restart": accepted_for_restart,
                "residuals": None if residuals is None else residuals.copy(),
                "reported_residuals": (
                    None if reported_residuals is None else reported_residuals.copy()
                ),
                "residual_history": residual_history,
                "solver_time_s": solution.solver_time_to_optimize,
                "wall_time_s": solution.real_time_to_optimize,
            }
            summaries.append(summary)
            if echo:
                print(
                    "acados_transfer_sqp_restart: "
                    f"attempt={attempt} status={solution.status} "
                    f"accepted={accepted_for_restart} "
                    f"residuals={_format_array(residuals)}"
                )

            if solver_success:
                apply_solution_directly_to_periodic_nmpc_initial_guess(
                    periodic_nmpc, solution
                )
                summary["primal_source"] = "solution"
            elif accepted_for_restart:
                capsule_summary = apply_acados_capsule_primal_to_initial_guess(
                    periodic_nmpc,
                    iterate_index=(
                        residual_history.get("best_index") if residual_history else None
                    ),
                )
                summary["capsule_primal"] = capsule_summary
                if capsule_summary["applied"]:
                    summary["primal_source"] = "acados_capsule"
                else:
                    apply_solution_directly_to_periodic_nmpc_initial_guess(
                        periodic_nmpc, solution
                    )
                    summary["primal_source"] = "solution_fallback"
            if solver_success or accepted_for_restart:
                best_states = snapshot_container(periodic_nmpc.nlp[0].x_init)
                best_controls = snapshot_container(periodic_nmpc.nlp[0].u_init)
                best_feasibility = feasibility
                best_stationarity = stationarity
            if echo and "capsule_primal" in summary:
                capsule_summary = summary["capsule_primal"]
                print(
                    "acados_transfer_sqp_restart_primal: "
                    f"source={summary['primal_source']} "
                    f"capsule_source={capsule_summary.get('source')} "
                    f"applied={capsule_summary['applied']} "
                    f"reason={capsule_summary['reason']} "
                    "state_max_change="
                    f"{capsule_summary.get('state_max_change')} "
                    "control_max_change="
                    f"{capsule_summary.get('control_max_change')} "
                    "projection="
                    f"{capsule_summary.get('bound_projection')}"
                )
            if solver_success:
                completed = True
                break
            if not accepted_for_restart:
                break
            summary["solver_reset"] = reset_acados_solver_memory(periodic_nmpc)
    finally:
        for key, values in best_states.items():
            periodic_nmpc.nlp[0].x_init[key].init[:, :] = values
        for key, values in best_controls.items():
            periodic_nmpc.nlp[0].u_init[key].init[:, :] = values
        if original_runtime_iterations is not None:
            set_acados_runtime_max_iterations(
                periodic_nmpc, int(original_runtime_iterations)
            )

    return {
        "completed": completed,
        "attempts": summaries,
        "best_feasibility": best_feasibility,
        "best_stationarity": best_stationarity,
    }


def project_transferred_initial_guess_to_bounds(periodic_nmpc) -> dict:
    """Make the shifted primal warm start admissible after bounds have moved."""

    return project_initial_guess_to_bounds(
        periodic_nmpc,
        sync_bounds=getattr(periodic_nmpc, "_sync_acados_state_bounds", None),
    )


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
    """Compatibility wrapper for the reusable initial-guess helper."""

    copy_container_values(source, target, attribute_name)


def _copy_refinement_initial_guesses(source, target, has_terminal_node: bool) -> None:
    """Copy shooting-node guesses onto a denser collocation grid when needed."""

    source_keys = set(source.keys())
    for key in target.keys():
        if key not in source_keys:
            continue
        source_values = np.asarray(source[key].init, dtype=float)
        target_values = target[key].init
        if source_values.shape == target_values.shape:
            target_values[:, :] = source_values
            continue

        source_len = source_values.shape[1]
        target_len = target_values.shape[1]
        source_intervals = source_len - int(has_terminal_node)
        target_intervals = target_len - int(has_terminal_node)
        if (
            source_values.shape[0] != target_values.shape[0]
            or source_intervals <= 0
            or target_intervals % source_intervals != 0
        ):
            raise ValueError(
                f"Cannot copy refinement initial guess '{key}' with shape "
                f"{source_values.shape} into shape {target_values.shape}."
            )

        subdivision = target_intervals // source_intervals
        source_grid = np.arange(source_len, dtype=float)
        target_grid = np.arange(target_len, dtype=float) / subdivision
        for row, values in enumerate(source_values):
            target_values[row, :] = np.interp(target_grid, source_grid, values)


def _copy_initial_guesses_and_bounds(source_nmpc, target_nmpc) -> None:
    _copy_refinement_initial_guesses(
        source_nmpc.nlp[0].x_init,
        target_nmpc.nlp[0].x_init,
        has_terminal_node=True,
    )
    _copy_refinement_initial_guesses(
        source_nmpc.nlp[0].u_init,
        target_nmpc.nlp[0].u_init,
        has_terminal_node=False,
    )
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
        "terminal_state_slack",
        "bound_first_node_all_states",
        "bound_first_node_wheel_qdot",
        "advance_wheel_q_bounds",
        "wheel_q_path_margin",
        "use_signed_wheel_shift",
        "transfer_initial_guess_mode",
        "repeat_cyclical_state_initial_guess",
        "continuous_state_initial_guess_mode",
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


def apply_control_regularization_targets(periodic_nmpc, controls) -> list[str]:
    updated_keys = []
    for penalty in periodic_nmpc.nlp[0].J:
        if not penalty:
            continue

        key = getattr(penalty, "extra_parameters", {}).get("key")
        if key not in controls:
            continue

        target = np.asarray(controls[key], dtype=float)
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


def apply_warmup_control_regularization_targets(
    periodic_nmpc, adapted_warmup_solution
) -> list[str]:
    warmup_controls = adapted_warmup_solution.decision_controls(
        to_merge=SolutionMerge.NODES
    )
    return apply_control_regularization_targets(periodic_nmpc, warmup_controls)


def apply_initial_guess_control_regularization_targets(periodic_nmpc) -> list[str]:
    controls = {
        key: np.asarray(periodic_nmpc.nlp[0].u_init[key].init, dtype=float)
        for key in periodic_nmpc.nlp[0].u_init.keys()
    }
    return apply_control_regularization_targets(periodic_nmpc, controls)


def apply_terminal_qdot_regularization_target(periodic_nmpc, target) -> bool:
    target = np.asarray(target, dtype=float).reshape((-1, 1))
    for penalty in periodic_nmpc.nlp[0].J:
        if not penalty:
            continue
        key = getattr(penalty, "extra_parameters", {}).get("key")
        if key != "qdot" or not penalty.node or penalty.node[0] != Node.END:
            continue
        penalty.target = target
        return True
    return False


def refresh_acados_cached_objective_targets(periodic_nmpc) -> None:
    """Copy updated Bioptim targets into the yref arrays cached by Acados."""
    interface = getattr(periodic_nmpc, "ocp_solver", None)
    if interface is None or not hasattr(interface, "y_ref"):
        return

    nlp = periodic_nmpc.nlp[0]
    lagrange_index = 0
    end_index = 0
    for penalty in nlp.J:
        if not penalty:
            continue
        is_terminal = bool(penalty.node) and penalty.node[0] == Node.END
        references = interface.y_ref_end if is_terminal else interface.y_ref
        reference_index = end_index if is_terminal else lagrange_index
        if is_terminal:
            end_index += 1
        else:
            lagrange_index += 1
        if reference_index >= len(references) or penalty.target is None:
            continue

        key = getattr(penalty, "extra_parameters", {}).get("key")
        if key not in nlp.states and key not in nlp.controls:
            continue

        target = np.asarray(penalty.target, dtype=float)
        if is_terminal:
            column = target[..., -1].reshape(-1)
            references[reference_index][: column.size, 0] = column
            continue

        for node, reference in enumerate(references[reference_index]):
            target_node = min(node, target.shape[-1] - 1)
            column = target[..., target_node].reshape(-1)
            reference[: column.size, 0] = column


def _penalty_is_lagrange(penalty) -> bool:
    penalty_type = getattr(penalty, "type", None)
    get_type = getattr(penalty_type, "get_type", None)
    if get_type is None:
        return False
    return getattr(get_type(), "__name__", "") == "LagrangeFunction"


def _penalty_output_dimension(penalty, terminal: bool = False) -> int:
    functions = getattr(penalty, "function", ())
    if not functions:
        return 0
    function = functions[-1] if terminal else functions[0]
    return int(function.numel_out())


def acados_objective_weight_layout(periodic_nmpc) -> dict[str, list[dict]]:
    """Mirror Bioptim's NONLINEAR_LS block ordering for runtime W updates."""

    layouts = {"W": [], "W_0": [], "W_e": []}
    offsets = {field: 0 for field in layouts}
    for penalty in periodic_nmpc.nlp[0].J:
        if not penalty:
            continue

        node = penalty.node[0] if penalty.node else None
        key = getattr(penalty, "extra_parameters", {}).get("key")
        placements = []
        if _penalty_is_lagrange(penalty):
            placements.append(("W", False))
        if node not in (Node.INTERMEDIATES, Node.PENULTIMATE, Node.END):
            placements.append(("W_0", False))
        if node in (Node.END, Node.ALL):
            placements.append(("W_e", True))

        for field, terminal in placements:
            size = _penalty_output_dimension(penalty, terminal=terminal)
            start = offsets[field]
            stop = start + size
            layouts[field].append(
                {
                    "key": key,
                    "start": start,
                    "stop": stop,
                    "penalty": penalty,
                }
            )
            offsets[field] = stop
    return layouts


def set_acados_runtime_control_regularization_weight(
    periodic_nmpc, weight: float
) -> dict:
    """Update only pulse-width proximity blocks in an existing Acados capsule."""

    if not np.isfinite(weight) or weight <= 0.0:
        raise ValueError("The runtime control regularization weight must be positive.")
    interface = getattr(periodic_nmpc, "ocp_solver", None)
    generated_solver = getattr(interface, "ocp_solver", None)
    if generated_solver is None:
        return {"applied": False, "reason": "solver_unavailable", "weight": weight}

    layout = acados_objective_weight_layout(periodic_nmpc)
    matrices = {
        field: np.array(getattr(interface, field), dtype=float, copy=True)
        for field in layout
    }
    updated_blocks = []
    for field, blocks in layout.items():
        matrix = matrices[field]
        for block in blocks:
            key = block["key"]
            if not isinstance(key, str) or not key.startswith("last_pulse_width_"):
                continue
            rows = np.arange(block["start"], block["stop"])
            if rows.size == 0:
                continue
            matrix[np.ix_(rows, rows)] = np.eye(rows.size) * weight
            updated_blocks.append(
                {
                    "field": field,
                    "key": key,
                    "start": int(rows[0]),
                    "stop": int(rows[-1] + 1),
                }
            )

    if not updated_blocks:
        return {
            "applied": False,
            "reason": "control_regularization_objective_missing",
            "weight": weight,
        }

    horizon = int(interface.acados_ocp.solver_options.N_horizon)

    def cost_set(stage, matrix):
        try:
            generated_solver.cost_set(stage, "W", matrix, api="new")
        except TypeError:
            generated_solver.cost_set(stage, "W", matrix)

    cost_set(0, matrices["W_0"])
    for stage in range(1, horizon):
        cost_set(stage, matrices["W"])
    cost_set(horizon, matrices["W_e"])
    periodic_nmpc._cocofest_proximal_control_weight = weight
    return {
        "applied": True,
        "reason": None,
        "weight": weight,
        "updated_blocks": updated_blocks,
    }


def set_terminal_wheel_q_bound_slack(periodic_nmpc, slack: float) -> None:
    if slack < 0:
        raise ValueError("Terminal wheel q slack must be non-negative.")
    bounds = periodic_nmpc.nlp[0].x_bounds["q"]
    center = getattr(periodic_nmpc, "_cocofest_terminal_wheel_q_center", None)
    if center is None:
        q_init = np.asarray(periodic_nmpc.nlp[0].x_init["q"].init, dtype=float)
        center = float(q_init[2, -1])
        periodic_nmpc._cocofest_terminal_wheel_q_center = center
    bounds.min[2, 2] = center - slack
    bounds.max[2, 2] = center + slack
    periodic_nmpc._sync_acados_state_bounds()


def recenter_terminal_wheel_q_bound_slack(periodic_nmpc, slack: float) -> dict:
    """Recenter a terminal-angle continuation after the MHE bounds shift."""

    bounds = periodic_nmpc.nlp[0].x_bounds["q"]
    center = 0.5 * float(bounds.min[2, 2] + bounds.max[2, 2])
    periodic_nmpc._cocofest_terminal_wheel_q_center = center
    set_terminal_wheel_q_bound_slack(periodic_nmpc, slack)
    return {
        "center": center,
        "slack": slack,
        "lower": center - slack,
        "upper": center + slack,
    }


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


def build_failed_solve_summary(
    nmpc,
    args: argparse.Namespace,
    error: Exception,
    initial_guess_state_traces: dict[str, np.ndarray],
    initial_guess_control_traces: dict[str, np.ndarray],
) -> dict:
    """Return a benchmark-compatible result when no first solution exists."""

    wheel_trace = np.asarray(initial_guess_state_traces.get("q", np.empty((0, 0))))
    wheel_trace = (
        wheel_trace[2].copy()
        if wheel_trace.ndim == 2 and wheel_trace.shape[0] > 2
        else np.array([], dtype=float)
    )
    return {
        "success": False,
        "solver_success": False,
        "physical_success": False,
        "status": None,
        "objective": float("nan"),
        "solver_time_s": 0.0,
        "wall_time_s": 0.0,
        "final_wheel_angle": float(wheel_trace[-1]) if wheel_trace.size else None,
        "requested_windows": args.n_windows,
        "attempted_windows": 0,
        "successful_windows": 0,
        "failed_windows": 1,
        "exported_cycles": 0,
        "covered_cycles": 0,
        "wheel_angle_trace": wheel_trace,
        "state_traces": {},
        "control_traces": {},
        "window_statuses": [],
        "window_solutions": [],
        "window_iterations": [],
        "diagnostics": {
            "is_physical": False,
            "issues": ["no_solver_solution"],
            "max_abs_angle": (
                float(np.max(np.abs(wheel_trace))) if wheel_trace.size else None
            ),
            "max_step": (
                float(np.max(np.abs(np.diff(wheel_trace))))
                if wheel_trace.size > 1
                else None
            ),
        },
        "error": f"{type(error).__name__}: {error}",
        "args": args,
        "control_bounds": _control_bounds_summary(nmpc),
        "initial_guess_state_traces": initial_guess_state_traces,
        "initial_guess_control_traces": initial_guess_control_traces,
    }


def restore_pulse_width_control_bounds(periodic_nmpc) -> None:
    original_bounds = getattr(periodic_nmpc, "_cocofest_original_control_bounds", {})
    for key, (lower, upper) in original_bounds.items():
        bounds = periodic_nmpc.nlp[0].u_bounds[key]
        bounds.min[:, :] = lower
        bounds.max[:, :] = upper
    periodic_nmpc._cocofest_nodewise_control_bounds = {}


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
    preparation_start = perf_counter()
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
    if args.acados_qp_cond_n is not None:
        if args.acados_qp_cond_n < 1:
            raise ValueError("--acados-qp-cond-n must be strictly positive.")
        if args.acados_qp_solver != "PARTIAL_CONDENSING_HPIPM":
            raise ValueError("--acados-qp-cond-n requires PARTIAL_CONDENSING_HPIPM.")
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
        raise ValueError("--transfer-rollout-substeps must be >= 1.")
    if args.acados_transfer_rollout_max_bound_violation < 0:
        raise ValueError(
            "--acados-transfer-rollout-max-bound-violation must be non-negative."
        )
    if (
        not np.isfinite(args.acados_transfer_pulse_width_scale)
        or args.acados_transfer_pulse_width_scale <= 0
    ):
        raise ValueError(
            "--acados-transfer-pulse-width-scale must be finite and positive."
        )
    if args.transfer_ding_force_compensation_substeps < 1:
        raise ValueError(
            "--transfer-ding-force-compensation-substeps must be positive."
        )
    if args.transfer_ding_force_compensation_iterations < 1:
        raise ValueError(
            "--transfer-ding-force-compensation-iterations must be positive."
        )
    if args.transfer_ding_force_compensation and args.model_formulation == "standard":
        raise ValueError(
            "Ding force compensation requires a periodic model formulation."
        )
    if args.acados_transfer_full_dynamics_rollout and args.acados_transfer_irk_rollout:
        raise ValueError(
            "Choose only one of --transfer-full-dynamics-rollout and "
            "--acados-transfer-irk-rollout."
        )
    if args.acados_transfer_bound_homotopy and not args.acados_transfer_irk_rollout:
        raise ValueError(
            "--acados-transfer-bound-homotopy requires "
            "--acados-transfer-irk-rollout."
        )
    if args.acados_transfer_bound_homotopy and args.solver != "acados":
        raise ValueError("Transfer-bound homotopy is only available with ACADOS.")
    if args.acados_transfer_bound_homotopy_padding < 0:
        raise ValueError(
            "--acados-transfer-bound-homotopy-padding must be non-negative."
        )
    if args.acados_transfer_bound_homotopy_iterations < 1:
        raise ValueError(
            "--acados-transfer-bound-homotopy-iterations must be strictly positive."
        )
    if args.acados_transfer_bound_homotopy_tolerance <= 0:
        raise ValueError(
            "--acados-transfer-bound-homotopy-tolerance must be strictly positive."
        )
    if (
        args.acados_transfer_bound_homotopy_solver_tolerance is not None
        and args.acados_transfer_bound_homotopy_solver_tolerance <= 0
    ):
        raise ValueError(
            "--acados-transfer-bound-homotopy-solver-tolerance must be strictly positive."
        )
    if args.acados_transfer_sqp_restarts < 0:
        raise ValueError("--acados-transfer-sqp-restarts must be non-negative.")
    if args.acados_transfer_sqp_restart_iterations < 1:
        raise ValueError(
            "--acados-transfer-sqp-restart-iterations must be strictly positive."
        )
    if args.acados_transfer_sqp_restart_feasibility_tolerance <= 0:
        raise ValueError(
            "--acados-transfer-sqp-restart-feasibility-tolerance must be strictly positive."
        )
    if args.acados_transfer_sqp_restarts and args.solver != "acados":
        raise ValueError("Transfer SQP restarts are only available with ACADOS.")
    if args.acados_fixed_control_tolerance <= 0:
        raise ValueError("--acados-fixed-control-tolerance must be strictly positive.")
    if args.acados_control_homotopy_tolerance <= 0:
        raise ValueError(
            "--acados-control-homotopy-tolerance must be strictly positive."
        )
    if args.acados_control_homotopy_max_restarts < 0:
        raise ValueError(
            "--acados-control-homotopy-max-restarts must be greater than or equal to zero."
        )
    if args.acados_control_homotopy_stage_iterations < 1:
        raise ValueError(
            "--acados-control-homotopy-stage-iterations must be strictly positive."
        )
    if args.acados_control_homotopy_window_growth < 1.0:
        raise ValueError(
            "--acados-control-homotopy-window-growth must be greater than or equal to one."
        )
    if (
        args.acados_control_homotopy_radii is not None
        and args.acados_pulse_width_trust_radius is not None
    ):
        raise ValueError(
            "Control homotopy cannot be combined with a persistent pulse-width trust region."
        )
    if (
        args.acados_control_homotopy_radii is not None
        and args.acados_fix_controls_to_warmup
    ):
        raise ValueError(
            "Control homotopy already includes its own fixed-control stage."
        )
    if args.acados_control_homotopy_radii is not None and args.solver != "acados":
        raise ValueError("Control homotopy is only available with ACADOS.")
    if (
        args.acados_control_homotopy_keep_final_radius
        and args.acados_control_homotopy_radii is None
    ):
        raise ValueError(
            "Keeping the final homotopy radius requires control homotopy radii."
        )
    if (
        args.acados_control_homotopy_each_window
        and args.acados_control_homotopy_radii is None
    ):
        raise ValueError("Per-window control homotopy requires control homotopy radii.")
    if args.acados_proximal_control_weights is not None and args.solver != "acados":
        raise ValueError("Proximal control continuation is only available with ACADOS.")
    if (
        args.acados_proximal_control_each_window
        and args.acados_proximal_control_weights is None
    ):
        raise ValueError(
            "Per-window proximal continuation requires proximal control weights."
        )
    if (
        args.acados_proximal_control_weights is not None
        and args.acados_control_homotopy_radii is not None
    ):
        raise ValueError("Bound and proximal control continuations cannot be combined.")
    if args.acados_proximal_control_tolerance <= 0:
        raise ValueError(
            "--acados-proximal-control-tolerance must be strictly positive."
        )
    if args.acados_proximal_control_stage_iterations < 1:
        raise ValueError(
            "--acados-proximal-control-stage-iterations must be strictly positive."
        )
    if args.acados_proximal_control_max_restarts < 0:
        raise ValueError("--acados-proximal-control-max-restarts must be non-negative.")
    if args.acados_terminal_wheel_q_slack < 0:
        raise ValueError("--acados-terminal-wheel-q-slack must be non-negative.")
    if args.acados_terminal_wheel_q_homotopy_slacks is not None:
        if args.solver != "acados":
            raise ValueError(
                "Terminal wheel-bound continuation is only available with ACADOS."
            )
        args.acados_terminal_wheel_q_slack = (
            args.acados_terminal_wheel_q_homotopy_slacks[0]
        )
    if (
        args.acados_terminal_wheel_q_homotopy_each_window
        and args.acados_terminal_wheel_q_homotopy_slacks is None
    ):
        raise ValueError(
            "Per-window terminal bound continuation requires homotopy slacks."
        )
    if args.terminal_qdot_regularization_weight < 0:
        raise ValueError("--terminal-qdot-regularization-weight must be non-negative.")
    if args.wheel_qdot_bound_margin <= 0:
        raise ValueError("--wheel-qdot-bound-margin must be strictly positive.")

    if args.acados_proximal_control_weights is not None:
        args.control_regularization_weight = args.acados_proximal_control_weights[0]
        args.control_regularization_target = None
        args.control_regularization_target_source = "previous"

    periodic_ipopt_refinement_enabled = (
        args.periodic_ipopt_refinement and not args.disable_periodic_ipopt_refinement
    )
    if (
        args.periodic_ipopt_refinement_each_window
        and not periodic_ipopt_refinement_enabled
    ):
        raise ValueError(
            "--periodic-ipopt-refinement-each-window requires "
            "--periodic-ipopt-refinement."
        )
    periodic_ipopt_reference_solution = None
    standard_warmup_cache_hit = None

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
        "wheel_qdot_bound_margin": args.wheel_qdot_bound_margin,
        "terminal_qdot_regularization_weight": (
            args.terminal_qdot_regularization_weight
        ),
        "state_scaling": args.state_scaling,
        "pulse_width_scaling": args.pulse_width_scaling,
        "init_guess_file_path": (
            str(historical_init_guess_path)
            if historical_init_guess_path is not None
            else None
        ),
    }
    nmpc_simulation_conditions = dict(simulation_conditions)
    if args.control_regularization_target_source in {"warmup", "previous"}:
        nmpc_simulation_conditions["control_regularization_target"] = None

    nmpc = prepare_nmpc(model, mhe_info, cycling_info, nmpc_simulation_conditions)
    nmpc.n_cycles_simultaneous = args.cycles_per_window
    if args.solver == "acados":
        patch_bioptim_acados_interface()

    # These settings define the physical receding-horizon transfer, not the
    # numerical backend. Apply them to periodic IPOPT diagnostics as well so a
    # backend comparison starts from the same bounds and primal trajectory.
    if args.solver == "acados" or periodic_cn_sum_approximation:
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
        nmpc.terminal_state_slack = {
            "q": [0.0, 0.0, args.acados_terminal_wheel_q_slack],
        }
        set_terminal_wheel_q_bound_slack(nmpc, args.acados_terminal_wheel_q_slack)
        # The periodic-node states have the same physical meaning as the
        # historical Ding states. Keep their initial value near the IPOPT
        # warmup; otherwise ACADOS can manufacture a low-cost but nonphysical
        # initial fatigue state that cannot be continued to another cycle.
        nmpc.bound_first_node_all_states = (
            args.acados_bind_first_node_fes_states
            or args.model_formulation == "periodic_node"
        )
        nmpc.bound_first_node_wheel_qdot = False
        nmpc.advance_wheel_q_bounds = True
        nmpc.wheel_q_path_margin = args.acados_wheel_q_path_margin
        nmpc.use_signed_wheel_shift = True
        nmpc.transfer_initial_guess_mode = "anchored"
        nmpc.repeat_cyclical_state_initial_guess = (
            args.acados_cyclical_transfer_mode == "repeat"
        )
        nmpc.transfer_debug = echo
        nmpc._cocofest_dual_warm_start_mode = args.acados_dual_warm_start_mode
        nmpc._cocofest_dual_shift_stages = args.stimulations_per_cycle
        relaxed_fes_bounds = []
        if args.solver == "acados" and not nmpc.bound_first_node_all_states:
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
        print(f"wheel_qdot_bound_margin: {args.wheel_qdot_bound_margin}")
        print(
            "terminal_qdot_regularization_weight: "
            f"{args.terminal_qdot_regularization_weight}"
        )
        print(
            "terminal_qdot_regularization_target_source: "
            f"{args.terminal_qdot_regularization_target_source}"
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
                "acados_control_homotopy_radii: "
                f"{args.acados_control_homotopy_radii}"
            )
            print(
                "acados_control_homotopy_tolerance: "
                f"{args.acados_control_homotopy_tolerance}"
            )
            print(
                "acados_control_homotopy_max_restarts: "
                f"{args.acados_control_homotopy_max_restarts}"
            )
            print(
                "acados_control_homotopy_stage_iterations: "
                f"{args.acados_control_homotopy_stage_iterations}"
            )
            print(
                "acados_control_homotopy_keep_final_radius: "
                f"{args.acados_control_homotopy_keep_final_radius}"
            )
            print(
                "acados_control_homotopy_each_window: "
                f"{args.acados_control_homotopy_each_window}"
            )
            print(
                "acados_control_homotopy_window_growth: "
                f"{args.acados_control_homotopy_window_growth}"
            )
            print(
                "acados_proximal_control_weights: "
                f"{args.acados_proximal_control_weights}"
            )
            print(
                "acados_proximal_control_each_window: "
                f"{args.acados_proximal_control_each_window}"
            )
            print(
                "acados_proximal_control_tolerance: "
                f"{args.acados_proximal_control_tolerance}"
            )
            print(
                "acados_proximal_control_stage_iterations: "
                f"{args.acados_proximal_control_stage_iterations}"
            )
            print(
                "acados_proximal_control_try_next_weight_on_failure: "
                f"{args.acados_proximal_control_try_next_weight_on_failure}"
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
                "periodic_ipopt_refinement_each_window: "
                f"{args.periodic_ipopt_refinement_each_window}"
            )
            print(
                "periodic_ipopt_refinement_window_cache: "
                f"{args.periodic_ipopt_refinement_window_cache}"
            )
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
            print(
                "acados_terminal_wheel_q_slack: "
                f"{args.acados_terminal_wheel_q_slack}"
            )
            print(f"acados_wheel_qdot_slack: {args.acados_wheel_qdot_slack}")
            print(f"acados_wheel_q_path_margin: {args.acados_wheel_q_path_margin}")
            print(
                "acados_transfer_full_dynamics_rollout: "
                f"{args.acados_transfer_full_dynamics_rollout}"
            )
            print(f"acados_transfer_irk_rollout: {args.acados_transfer_irk_rollout}")
            print(f"acados_transfer_phase_one: {args.acados_transfer_phase_one}")
            print(
                "acados_transfer_bound_homotopy: "
                f"{args.acados_transfer_bound_homotopy}"
            )
            if args.acados_transfer_bound_homotopy:
                print(
                    "acados_transfer_bound_homotopy_fractions: "
                    f"{args.acados_transfer_bound_homotopy_fractions}"
                )
                print(
                    "acados_transfer_bound_homotopy_padding: "
                    f"{args.acados_transfer_bound_homotopy_padding}"
                )
                print(
                    "acados_transfer_bound_homotopy_iterations: "
                    f"{args.acados_transfer_bound_homotopy_iterations}"
                )
                print(
                    "acados_transfer_bound_homotopy_tolerance: "
                    f"{args.acados_transfer_bound_homotopy_tolerance}"
                )
                print(
                    "acados_transfer_bound_homotopy_solver_tolerance: "
                    f"{args.acados_transfer_bound_homotopy_solver_tolerance}"
                )
            print(f"acados_transfer_sqp_restarts: {args.acados_transfer_sqp_restarts}")
            if args.acados_transfer_sqp_restarts:
                print(
                    "acados_transfer_sqp_restart_iterations: "
                    f"{args.acados_transfer_sqp_restart_iterations}"
                )
                print(
                    "acados_transfer_sqp_restart_feasibility_tolerance: "
                    f"{args.acados_transfer_sqp_restart_feasibility_tolerance}"
                )
            print(
                "acados_cyclical_transfer_mode: "
                f"{args.acados_cyclical_transfer_mode}"
            )
            print(
                "acados_transfer_rollout_substeps: "
                f"{args.acados_transfer_rollout_substeps}"
            )
            print(
                "acados_transfer_rollout_max_bound_violation: "
                f"{args.acados_transfer_rollout_max_bound_violation}"
            )
            print(
                "acados_transfer_pulse_width_scale: "
                f"{args.acados_transfer_pulse_width_scale}"
            )
            print(
                "transfer_ding_force_compensation: "
                f"{args.transfer_ding_force_compensation}"
            )
            print(
                "transfer_ding_force_compensation_substeps: "
                f"{args.transfer_ding_force_compensation_substeps}"
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
            print("ipopt_dual_warm_start_mode: " f"{args.ipopt_dual_warm_start_mode}")
            print(
                "historical_initial_guess: "
                f"{historical_init_guess_path if historical_init_guess_path else 'None'}"
            )
        print(f"full_dynamics_phase_one: {args.full_dynamics_phase_one}")

    if periodic_cn_sum_approximation and not args.disable_standard_ipopt_warmup:
        if echo:
            print("running_standard_ipopt_warmup: True")
        warmup_simulation_conditions = dict(simulation_conditions)
        if args.control_regularization_target_source in {"warmup", "previous"}:
            warmup_simulation_conditions["control_regularization_weight"] = 0.0
            warmup_simulation_conditions["control_regularization_target"] = None
        if args.terminal_qdot_regularization_target_source == "first_node":
            # This cyclic regularization belongs to the periodic target problem;
            # keep the standard IPOPT bridge identical to the historical reference.
            warmup_simulation_conditions["terminal_qdot_regularization_weight"] = 0.0
        standard_warmup_cache_hit = _warmup_cache_path(
            args, model_path, warmup_simulation_conditions, cycling_info
        ).exists()
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

    refinement_nmpc = None
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
        if (
            not refinement_cache_path.exists()
            or args.periodic_ipopt_refinement_each_window
        ):
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
                    OdeSolver.COLLOCATION(
                        polynomial_degree=args.collocation_degree,
                        method=args.collocation_method,
                    )
                    if args.periodic_ipopt_refinement_ode_solver == "collocation"
                    else (
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
                    )
                ),
            )
        if not refinement_cache_path.exists():
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

    if (
        args.control_regularization_target_source == "previous"
        and args.control_regularization_weight
    ):
        target_keys = apply_initial_guess_control_regularization_targets(nmpc)
        if echo:
            print(
                "previous_control_regularization_targets: "
                f"{', '.join(target_keys) if target_keys else 'None'}"
            )

    if args.terminal_qdot_regularization_weight:
        qdot_guess = np.asarray(nmpc.nlp[0].x_init["qdot"].init, dtype=float)
        terminal_qdot = (
            qdot_guess[:, 0]
            if args.terminal_qdot_regularization_target_source == "first_node"
            else qdot_guess[:, -1]
        )
        apply_terminal_qdot_regularization_target(nmpc, terminal_qdot)

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
                f"accepted={phase_one_summary['accepted']} "
                f"accepted_step={phase_one_summary['accepted_step']:.6g} "
                f"scaled_defect_before={phase_one_summary['scaled_defect_before']:.6g} "
                f"scaled_defect_after={phase_one_summary['scaled_defect_after']:.6g} "
                f"candidate_scaled_defect_after="
                f"{phase_one_summary['candidate_scaled_defect_after']:.6g} "
                f"max_state_change={phase_one_summary['max_state_change']:.6g} "
                f"scaled_by_block_after="
                f"{phase_one_summary['scaled_by_block_after']}"
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

    initial_guess_audit = audit_initial_guess(nmpc)
    initial_guess_snapshot = initial_guess_audit.pop("snapshot")
    initial_guess_state_traces = initial_guess_snapshot["states"]
    initial_guess_control_traces = initial_guess_snapshot["controls"]
    if echo:
        print(
            "initial_guess_audit: "
            f"signature={initial_guess_audit['signature']} "
            f"finite={initial_guess_audit['finite']}"
        )
    initial_guess_preparation_time_s = perf_counter() - preparation_start
    if echo:
        print(
            "initial_guess_preparation_time_s: "
            f"{initial_guess_preparation_time_s:.6f}"
        )

    acados_window_diagnostics = []
    ipopt_dual_warm_start_summaries = []
    transfer_rollout_summaries = []
    transfer_control_scaling_summaries = []
    transfer_qdot_projection_summaries = []
    transfer_ding_force_compensation_summaries = []
    transfer_bound_homotopy_summaries = []
    transfer_sqp_restart_summaries = []
    transfer_bound_projection_summaries = []
    inter_window_refinement_summaries = []
    inter_window_control_homotopy_summaries = []
    control_homotopy_summaries = []
    proximal_control_summaries = []
    terminal_wheel_bound_summaries = []
    inter_window_terminal_wheel_bound_summaries = []
    inter_window_proximal_control_summaries = []
    transfer_failure_window = None
    initial_guess_audits = [{"window": 0, **initial_guess_audit}]
    requested_window_solves = receding_horizon_window_count(
        args.n_windows, args.cycles_per_window
    )

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
        nonlocal transfer_failure_window
        print(f"window {cycle_idx}")
        if args.solver == "acados":
            _nmpc._cocofest_dual_warm_start_mode = args.acados_dual_warm_start_mode
        transfer_rollout_applied = None
        completed_window_diagnostics = None
        if args.solver == "acados" and _sol is not None:
            # Auxiliary refinement and homotopy solves reuse the mutable Acados
            # backend. Snapshot the completed window before they overwrite it.
            completed_window_diagnostics = snapshot_acados_diagnostics(_sol)
            acados_window_diagnostics.append(completed_window_diagnostics)
        if echo and _sol is not None:
            states = _sol.decision_states(to_merge=SolutionMerge.NODES)
            print(
                f"window {cycle_idx} terminal wheel q={states['q'][2, -1]:.6f} "
                f"qdot={states['qdot'][2, -1]:.6f}"
            )
        if args.solver == "ipopt" and _sol is not None:
            dual_summary = apply_ipopt_dual_warm_start(
                _nmpc, _sol, args.ipopt_dual_warm_start_mode
            )
            ipopt_dual_warm_start_summaries.append(dual_summary)
            if echo:
                print(
                    "ipopt_dual_warm_start: "
                    f"mode={dual_summary['mode']} applied={dual_summary['applied']} "
                    f"lam_g={dual_summary['lam_g_size']} "
                    f"lam_x={dual_summary['lam_x_size']} "
                    f"reason={dual_summary['reason']}"
                )
        continue_solving = cycle_idx < requested_window_solves
        targets_updated = False
        if (
            continue_solving
            and _sol is not None
            and args.solver == "acados"
            and args.acados_terminal_wheel_q_homotopy_each_window
        ):
            terminal_bound_expansion = recenter_terminal_wheel_q_bound_slack(
                _nmpc, args.acados_terminal_wheel_q_homotopy_slacks[0]
            )
            if echo:
                print(
                    "acados_terminal_wheel_bound_preexpanded: "
                    f"window={cycle_idx} center={terminal_bound_expansion['center']:.6f} "
                    f"slack={terminal_bound_expansion['slack']:.6g} "
                    f"bounds=[{terminal_bound_expansion['lower']:.6f}, "
                    f"{terminal_bound_expansion['upper']:.6f}]"
                )
        if continue_solving and _sol is not None:
            if (
                args.control_regularization_target_source == "previous"
                and args.control_regularization_weight
            ):
                target_keys = apply_initial_guess_control_regularization_targets(_nmpc)
                targets_updated = bool(target_keys)
                if echo:
                    print(
                        "previous_control_regularization_targets_recentered: "
                        f"window={cycle_idx} keys={len(target_keys)}"
                    )
            if args.terminal_qdot_regularization_weight and (
                args.terminal_qdot_regularization_target_source
                in ("previous", "first_node")
            ):
                if args.terminal_qdot_regularization_target_source == "previous":
                    previous_states = _sol.decision_states(to_merge=SolutionMerge.NODES)
                    terminal_qdot_target = previous_states["qdot"][:, -1]
                else:
                    terminal_qdot_target = np.asarray(
                        _nmpc.nlp[0].x_init["qdot"].init, dtype=float
                    )[:, 0]
                targets_updated = (
                    apply_terminal_qdot_regularization_target(
                        _nmpc, terminal_qdot_target
                    )
                    or targets_updated
                )
            if targets_updated and args.solver == "acados":
                refresh_acados_cached_objective_targets(_nmpc)
        if (
            continue_solving
            and _sol is not None
            and args.solver == "acados"
            and args.periodic_ipopt_refinement_each_window
        ):
            if refinement_nmpc is None:
                raise RuntimeError(
                    "Per-window IPOPT refinement requires --periodic-ipopt-refinement."
                )
            refinement_nmpc.update_stim()
            _copy_initial_guesses_and_bounds(_nmpc, refinement_nmpc)
            _copy_objective_targets(_nmpc, refinement_nmpc)
            window_refinement_cache_path = (
                _periodic_ipopt_window_refinement_cache_path(
                    args, model_path, cycle_idx
                )
                if args.periodic_ipopt_refinement_window_cache
                else None
            )
            if echo:
                print(f"running_periodic_ipopt_refinement_window: {cycle_idx}")
            refinement_cache_hit = bool(
                window_refinement_cache_path is not None
                and window_refinement_cache_path.exists()
            )
            if refinement_cache_hit:
                refinement_solution = _load_warmup_cache(window_refinement_cache_path)
                apply_solution_directly_to_periodic_nmpc_initial_guess(
                    _nmpc, refinement_solution
                )
                refinement_success = True
                refinement_status = 0
                refinement_solver_time = 0.0
                refinement_wall_time = 0.0
                if echo:
                    print(
                        "periodic_ipopt_refinement_window_cache: "
                        f"hit ({window_refinement_cache_path.name})"
                    )
            else:
                refinement_solution = run_periodic_ipopt_refinement(
                    refinement_nmpc,
                    target_nmpc=_nmpc,
                    max_iterations=args.periodic_ipopt_refinement_iterations,
                    linear_solver=args.ipopt_linear_solver,
                    cache_path=window_refinement_cache_path,
                    echo=echo,
                )
                refinement_success = bool(
                    refinement_solution is not None
                    and _status_is_success(refinement_solution.status)
                )
                refinement_status = (
                    None if refinement_solution is None else refinement_solution.status
                )
                refinement_solver_time = (
                    None
                    if refinement_solution is None
                    else refinement_solution.solver_time_to_optimize
                )
                refinement_wall_time = (
                    None
                    if refinement_solution is None
                    else refinement_solution.real_time_to_optimize
                )
            inter_window_refinement_summaries.append(
                {
                    "window": cycle_idx,
                    "success": refinement_success,
                    "status": refinement_status,
                    "solver_time_s": refinement_solver_time,
                    "wall_time_s": refinement_wall_time,
                    "cache_hit": refinement_cache_hit,
                }
            )
        if (
            continue_solving
            and _sol is not None
            and args.solver == "acados"
            and args.acados_project_qdot_from_q
        ):
            qdot_projection_summary = project_qdot_initial_guess_from_q(
                _nmpc,
                start_node=_nmpc.cycle_len,
                select_by_dynamics=True,
                dynamics_substeps=args.full_dynamics_phase_one_substeps,
            )
            qdot_projection_summary["window"] = cycle_idx
            transfer_qdot_projection_summaries.append(qdot_projection_summary)
            if echo:
                print(
                    "transfer_qdot_projection: "
                    f"applied={qdot_projection_summary['applied']} "
                    f"start_node={qdot_projection_summary['start_node']} "
                    f"accepted_step={qdot_projection_summary['accepted_step']:.6g} "
                    f"scaled_defect={qdot_projection_summary['scaled_defect_before']}->"
                    f"{qdot_projection_summary['scaled_defect_after']} "
                    f"max_change={qdot_projection_summary['max_change']:.6g} "
                    f"clipped_values={qdot_projection_summary['clipped_count']}"
                )
        if (
            continue_solving
            and _sol is not None
            and not np.isclose(args.acados_transfer_pulse_width_scale, 1.0)
            and (
                args.acados_transfer_full_dynamics_rollout
                or (args.solver == "acados" and args.acados_transfer_irk_rollout)
            )
        ):
            control_scaling_summary = scale_appended_pulse_width_controls(
                _nmpc, args.acados_transfer_pulse_width_scale
            )
            control_scaling_summary["window"] = cycle_idx
            transfer_control_scaling_summaries.append(control_scaling_summary)
            if echo:
                clipped_count = sum(
                    item["clipped_count"]
                    for item in control_scaling_summary["controls"].values()
                )
                print(
                    "transfer_pulse_width_scaling: "
                    f"scale={args.acados_transfer_pulse_width_scale:.6g} "
                    f"start_node={control_scaling_summary['start_node']} "
                    f"clipped_values={clipped_count}"
                )
        if (
            continue_solving
            and _sol is not None
            and args.transfer_ding_force_compensation
        ):
            compensation_summary = compensate_appended_pulse_widths_from_ding_force(
                _nmpc,
                n_substeps=args.transfer_ding_force_compensation_substeps,
                bisection_iterations=(args.transfer_ding_force_compensation_iterations),
                previous_solution=_sol,
            )
            compensation_summary["window"] = cycle_idx
            transfer_ding_force_compensation_summaries.append(compensation_summary)
            if (
                compensation_summary["applied"]
                and args.control_regularization_target_source == "previous"
                and args.control_regularization_weight
            ):
                apply_initial_guess_control_regularization_targets(_nmpc)
                if args.solver == "acados":
                    refresh_acados_cached_objective_targets(_nmpc)
            if echo:
                print(
                    "transfer_ding_force_compensation: "
                    f"applied={compensation_summary['applied']} "
                    f"start_node={compensation_summary['start_node']} "
                    f"reason={compensation_summary['reason']}"
                )
                for muscle_name, item in compensation_summary["muscles"].items():
                    print(
                        "transfer_ding_force_compensation_muscle: "
                        f"muscle={muscle_name} "
                        f"gain=[{item['gain_min']:.6g}, {item['gain_mean']:.6g}, "
                        f"{item['gain_max']:.6g}] "
                        f"force_rmse={item['baseline_force_rmse']:.6g}->"
                        f"{item['compensated_force_rmse']:.6g} "
                        f"saturated={item['saturated_count']}"
                    )
        if (
            continue_solving
            and _sol is not None
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
                    "transfer_full_dynamics_rollout: "
                    f"applied={rollout_summary['applied']} "
                    f"start_node={rollout_summary['start_node']} "
                    "max_bound_violation="
                    f"{rollout_summary.get('max_bound_violation')}"
                )
                violation_by_key = rollout_summary.get("max_bound_violation_by_key", {})
                if violation_by_key:
                    worst_key = max(violation_by_key, key=violation_by_key.get)
                    print(
                        "transfer_rollout_worst_bound: "
                        f"key={worst_key} "
                        f"violation={violation_by_key[worst_key]:.6g}"
                    )
                if rollout_summary["applied"]:
                    print(
                        "transfer_rollout_terminal_delta: "
                        f"q={rollout_summary['terminal_delta'].get('q')} "
                        f"qdot={rollout_summary['terminal_delta'].get('qdot')}"
                    )
                else:
                    print(
                        "transfer_rollout_nonfinite_node: "
                        f"{rollout_summary.get('nonfinite_node')}"
                    )
        if (
            continue_solving
            and _sol is not None
            and args.solver == "acados"
            and args.acados_transfer_irk_rollout
        ):
            rollout_summary = rollout_transferred_cycle_acados_irk(
                _nmpc,
                max_allowed_bound_violation=(
                    None
                    if args.acados_transfer_bound_homotopy
                    else args.acados_transfer_rollout_max_bound_violation
                ),
            )
            transfer_rollout_applied = rollout_summary["applied"]
            transfer_rollout_summaries.append(rollout_summary)
            if echo:
                print(
                    "acados_transfer_irk_rollout: "
                    f"applied={rollout_summary['applied']} "
                    f"start_node={rollout_summary['start_node']} "
                    f"simulator_built={rollout_summary['simulator_built']} "
                    f"simulation_time_s={rollout_summary['simulation_time_s']:.6g} "
                    "max_bound_violation="
                    f"{rollout_summary.get('max_bound_violation')} "
                    f"reason={rollout_summary.get('reason')}"
                )
                worst_violation = rollout_summary.get("worst_bound_violation")
                if worst_violation is not None:
                    print(
                        "acados_transfer_irk_rollout_worst_bound: "
                        f"key={worst_violation['key']} "
                        f"component={worst_violation['component']} "
                        f"node={worst_violation['node']} "
                        f"value={worst_violation['value']:.6g} "
                        f"bounds=[{worst_violation['lower']:.6g}, "
                        f"{worst_violation['upper']:.6g}] "
                        f"violation={worst_violation['violation']:.6g}"
                    )
                rk4_defects = rollout_summary.get("rk4_defects_after") or {}
                if rk4_defects:
                    print(
                        "acados_transfer_irk_rollout_rk4_defects: "
                        f"absolute={rk4_defects.get('absolute_by_block')} "
                        f"scaled={rk4_defects.get('scaled_by_block')}"
                    )
        if (
            continue_solving
            and _sol is not None
            and args.solver == "acados"
            and args.acados_transfer_bound_homotopy
            and transfer_rollout_applied
        ):
            homotopy_summary = run_acados_transfer_bound_homotopy(
                _nmpc,
                solver,
                fractions=args.acados_transfer_bound_homotopy_fractions,
                padding=args.acados_transfer_bound_homotopy_padding,
                convergence_tolerance=(args.acados_transfer_bound_homotopy_tolerance),
                stage_iterations=args.acados_transfer_bound_homotopy_iterations,
                solver_tolerance=(args.acados_transfer_bound_homotopy_solver_tolerance),
                echo=echo,
            )
            homotopy_summary["window"] = cycle_idx
            transfer_bound_homotopy_summaries.append(homotopy_summary)
            if echo:
                expansion_by_key = homotopy_summary["expansion_by_key"]
                worst_key = (
                    max(expansion_by_key, key=expansion_by_key.get)
                    if expansion_by_key
                    else None
                )
                print(
                    "acados_transfer_bound_homotopy_summary: "
                    f"completed={homotopy_summary['completed']} "
                    f"stages={len(homotopy_summary['stages'])} "
                    f"max_expansion={homotopy_summary['max_expansion']:.6g} "
                    f"worst_key={worst_key}"
                )
        if continue_solving and _sol is not None and args.acados_transfer_phase_one:
            phase_one_summary = project_full_dynamics_initial_guess(
                _nmpc,
                proximity_weight=args.full_dynamics_phase_one_proximity_weight,
                defect_weight=args.full_dynamics_phase_one_defect_weight,
                n_substeps=args.full_dynamics_phase_one_substeps,
            )
            if echo:
                print(
                    "transfer_phase_one: "
                    f"accepted={phase_one_summary['accepted']} "
                    f"accepted_step={phase_one_summary['accepted_step']:.6g} "
                    f"scaled_defect_before={phase_one_summary['scaled_defect_before']:.6g} "
                    f"scaled_defect_after={phase_one_summary['scaled_defect_after']:.6g} "
                    f"candidate_scaled_defect_after="
                    f"{phase_one_summary['candidate_scaled_defect_after']:.6g} "
                    f"max_state_change={phase_one_summary['max_state_change']:.6g} "
                    f"scaled_by_block_before="
                    f"{phase_one_summary['scaled_by_block_before']} "
                    f"scaled_by_block_after="
                    f"{phase_one_summary['scaled_by_block_after']}"
                )
        if continue_solving and _sol is not None:
            bound_projection = project_transferred_initial_guess_to_bounds(_nmpc)
            bound_projection["window"] = cycle_idx + 1
            transfer_bound_projection_summaries.append(bound_projection)
            if echo:
                print(
                    "transfer_bound_projection: "
                    f"state_max_change={bound_projection['state_max_change']:.6g} "
                    f"control_max_change={bound_projection['control_max_change']:.6g}"
                )
        if (
            continue_solving
            and _sol is not None
            and args.solver == "acados"
            and args.acados_transfer_sqp_restarts
        ):
            restart_summary = run_acados_transfer_sqp_restarts(
                _nmpc,
                solver,
                max_restarts=args.acados_transfer_sqp_restarts,
                stage_iterations=args.acados_transfer_sqp_restart_iterations,
                feasibility_tolerance=(
                    args.acados_transfer_sqp_restart_feasibility_tolerance
                ),
                echo=echo,
            )
            restart_summary["window"] = cycle_idx
            transfer_sqp_restart_summaries.append(restart_summary)
            if echo:
                print(
                    "acados_transfer_sqp_restart_summary: "
                    f"completed={restart_summary['completed']} "
                    f"attempts={len(restart_summary['attempts'])} "
                    f"best_feasibility={restart_summary['best_feasibility']:.6g} "
                    f"best_stationarity={restart_summary['best_stationarity']:.6g}"
                )
        if (
            continue_solving
            and _sol is not None
            and args.solver == "acados"
            and args.acados_control_homotopy_each_window
        ):
            if echo:
                print(f"running_acados_control_homotopy_window: {cycle_idx}")
            window_homotopy = run_acados_control_homotopy(
                _nmpc,
                solver,
                radii=args.acados_control_homotopy_radii,
                convergence_tolerance=args.acados_control_homotopy_tolerance,
                fixed_control_tolerance=args.acados_fixed_control_tolerance,
                max_restarts=args.acados_control_homotopy_max_restarts,
                stage_iterations=args.acados_control_homotopy_stage_iterations,
                echo=echo,
            )
            for item in window_homotopy:
                item["window"] = cycle_idx
            inter_window_control_homotopy_summaries.extend(window_homotopy)
            accepted_window_radii = [
                item["radius"]
                for item in window_homotopy
                if item["accepted"] and item["radius"] is not None
            ]
            if accepted_window_radii:
                _nmpc._cocofest_retained_control_homotopy_radius = (
                    accepted_window_radii[-1]
                )
            elif echo:
                print(
                    "acados_control_homotopy_window_warning: "
                    f"window={cycle_idx} no_finite_radius_accepted"
                )
        if (
            continue_solving
            and _sol is not None
            and args.solver == "acados"
            and args.acados_proximal_control_each_window
        ):
            if echo:
                print(f"running_acados_proximal_control_window: {cycle_idx}")
            window_continuation = run_acados_proximal_control_continuation(
                _nmpc,
                solver,
                weights=args.acados_proximal_control_weights,
                convergence_tolerance=args.acados_proximal_control_tolerance,
                max_restarts=args.acados_proximal_control_max_restarts,
                stage_iterations=args.acados_proximal_control_stage_iterations,
                try_next_weight_on_failure=(
                    args.acados_proximal_control_try_next_weight_on_failure
                ),
                echo=echo,
            )
            for item in window_continuation:
                item["window"] = cycle_idx
            inter_window_proximal_control_summaries.extend(window_continuation)
            if (
                window_continuation
                and not window_continuation[-1]["accepted"]
                and not args.continue_after_acados_transfer_failure
            ):
                continue_solving = False
                transfer_failure_window = cycle_idx
                if echo:
                    print(
                        "acados_transfer_failure_stop: "
                        f"window={cycle_idx} "
                        f"status={window_continuation[-1]['status']}"
                    )
        if (
            continue_solving
            and _sol is not None
            and args.solver == "acados"
            and args.acados_terminal_wheel_q_homotopy_each_window
        ):
            if echo:
                print(f"running_acados_terminal_wheel_bound_window: {cycle_idx}")
            window_terminal_continuation = run_acados_terminal_wheel_bound_continuation(
                _nmpc,
                solver,
                slacks=args.acados_terminal_wheel_q_homotopy_slacks,
                convergence_tolerance=args.acados_proximal_control_tolerance,
                stage_iterations=args.acados_proximal_control_stage_iterations,
                echo=echo,
            )
            for item in window_terminal_continuation:
                item["window"] = cycle_idx
            inter_window_terminal_wheel_bound_summaries.extend(
                window_terminal_continuation
            )
        if completed_window_diagnostics is not None:
            if echo and args.acados_diagnostics:
                print_acados_diagnostics(
                    f"window[{cycle_idx - 1}]", completed_window_diagnostics
                )
                if continue_solving:
                    print(f"window[{cycle_idx}] transferred_initial_guess_diagnostics:")
                    print_initial_guess_diagnostics(_nmpc)
        retained_homotopy_radius = getattr(
            _nmpc, "_cocofest_retained_control_homotopy_radius", None
        )
        if (
            continue_solving
            and _sol is not None
            and args.solver == "acados"
            and (
                args.acados_pulse_width_trust_radius is not None
                or retained_homotopy_radius is not None
            )
        ):
            if retained_homotopy_radius is not None:
                transfer_radius = (
                    retained_homotopy_radius
                    * args.acados_control_homotopy_window_growth
                )
                _nmpc._cocofest_retained_control_homotopy_radius = transfer_radius
            else:
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
                    f"window={cycle_idx} radius={transfer_radius:.9g} "
                    f"max_center={max_center:.9g}"
                )
        if continue_solving and _sol is not None:
            next_audit = audit_initial_guess(_nmpc)
            next_audit.pop("snapshot")
            initial_guess_audits.append({"window": cycle_idx + 1, **next_audit})
            if echo:
                print(
                    "transferred_initial_guess_audit: "
                    f"window={cycle_idx + 1} "
                    f"signature={next_audit['signature']} "
                    f"finite={next_audit['finite']}"
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
            qp_cond_n=args.acados_qp_cond_n,
            hpipm_mode=args.acados_hpipm_mode,
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
            store_iterates=args.acados_store_iterates,
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

    if args.solver == "acados" and args.acados_control_homotopy_radii is not None:
        control_homotopy_summaries = run_acados_control_homotopy(
            nmpc,
            solver,
            radii=args.acados_control_homotopy_radii,
            convergence_tolerance=args.acados_control_homotopy_tolerance,
            fixed_control_tolerance=args.acados_fixed_control_tolerance,
            max_restarts=args.acados_control_homotopy_max_restarts,
            stage_iterations=args.acados_control_homotopy_stage_iterations,
            echo=echo,
        )
        set_acados_runtime_max_iterations(nmpc, args.max_acados_iterations)
        solver.set_only_first_options_has_changed(False)
        solver.set_nlp_solver_tol_eq(solver.nlp_solver_tol_eq)
        solver.set_nlp_solver_tol_ineq(solver.nlp_solver_tol_ineq)
        solver.set_nlp_solver_tol_comp(solver.nlp_solver_tol_comp)
        solver.set_nlp_solver_tol_stat(solver.nlp_solver_tol_stat)
        if args.acados_control_homotopy_keep_final_radius:
            accepted_radii = [
                summary["radius"]
                for summary in control_homotopy_summaries
                if summary["accepted"] and summary["radius"] is not None
            ]
            if not accepted_radii:
                raise RuntimeError(
                    "Control homotopy did not accept any finite control radius."
                )
            retained_radius = accepted_radii[-1]
            apply_pulse_width_control_trust_region(nmpc, retained_radius)
            nmpc._cocofest_retained_control_homotopy_radius = retained_radius
            if echo:
                print(
                    "acados_control_homotopy_retained_radius: " f"{retained_radius:.9g}"
                )
        if echo and args.acados_diagnostics:
            print("post_control_homotopy_initial_guess_diagnostics:")
            print_initial_guess_diagnostics(nmpc)

    if args.solver == "acados" and args.acados_proximal_control_weights is not None:
        proximal_control_summaries = run_acados_proximal_control_continuation(
            nmpc,
            solver,
            weights=args.acados_proximal_control_weights,
            convergence_tolerance=args.acados_proximal_control_tolerance,
            max_restarts=args.acados_proximal_control_max_restarts,
            stage_iterations=args.acados_proximal_control_stage_iterations,
            try_next_weight_on_failure=(
                args.acados_proximal_control_try_next_weight_on_failure
            ),
            echo=echo,
        )
        set_acados_runtime_max_iterations(nmpc, args.max_acados_iterations)
        solver.set_only_first_options_has_changed(False)
        solver.set_nlp_solver_tol_eq(solver.nlp_solver_tol_eq)
        solver.set_nlp_solver_tol_ineq(solver.nlp_solver_tol_ineq)
        solver.set_nlp_solver_tol_comp(solver.nlp_solver_tol_comp)
        solver.set_nlp_solver_tol_stat(solver.nlp_solver_tol_stat)
        if echo and args.acados_diagnostics:
            print("post_proximal_control_initial_guess_diagnostics:")
            print_initial_guess_diagnostics(nmpc)

    proximal_ready = (
        not proximal_control_summaries or proximal_control_summaries[-1]["accepted"]
    )
    if (
        args.solver == "acados"
        and args.acados_terminal_wheel_q_homotopy_slacks is not None
        and proximal_ready
    ):
        terminal_wheel_bound_summaries = run_acados_terminal_wheel_bound_continuation(
            nmpc,
            solver,
            slacks=args.acados_terminal_wheel_q_homotopy_slacks,
            convergence_tolerance=args.acados_proximal_control_tolerance,
            stage_iterations=args.acados_proximal_control_stage_iterations,
            echo=echo,
        )
        set_acados_runtime_max_iterations(nmpc, args.max_acados_iterations)

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
        summary["initial_guess_audits"] = initial_guess_audits
        summary["initial_guess_preparation_time_s"] = initial_guess_preparation_time_s
        summary["standard_warmup_cache_hit"] = standard_warmup_cache_hit
        if control_homotopy_summaries:
            summary["control_homotopy_summaries"] = control_homotopy_summaries
        if terminal_wheel_bound_summaries:
            summary["terminal_wheel_bound_summaries"] = terminal_wheel_bound_summaries
        return summary

    try:
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
            compact_solution_output=args.compact_rho_output,
        )
    except RuntimeError as exc:
        if "did not produce a valid solution" not in str(exc):
            raise
        if echo:
            print(f"solve_error: {type(exc).__name__}: {exc}")
        summary = build_failed_solve_summary(
            nmpc,
            args,
            exc,
            initial_guess_state_traces,
            initial_guess_control_traces,
        )
        summary["initial_guess_audits"] = initial_guess_audits
        summary["initial_guess_preparation_time_s"] = initial_guess_preparation_time_s
        summary["standard_warmup_cache_hit"] = standard_warmup_cache_hit
        return summary
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
    if ipopt_dual_warm_start_summaries:
        summary["ipopt_dual_warm_start_summaries"] = ipopt_dual_warm_start_summaries
    if transfer_rollout_summaries:
        summary["transfer_rollout_summaries"] = transfer_rollout_summaries
    if transfer_control_scaling_summaries:
        summary["transfer_control_scaling_summaries"] = (
            transfer_control_scaling_summaries
        )
    if transfer_qdot_projection_summaries:
        summary["transfer_qdot_projection_summaries"] = (
            transfer_qdot_projection_summaries
        )
    if transfer_ding_force_compensation_summaries:
        summary["transfer_ding_force_compensation_summaries"] = (
            transfer_ding_force_compensation_summaries
        )
    if transfer_bound_homotopy_summaries:
        summary["transfer_bound_homotopy_summaries"] = transfer_bound_homotopy_summaries
    if transfer_sqp_restart_summaries:
        summary["transfer_sqp_restart_summaries"] = transfer_sqp_restart_summaries
    if transfer_bound_projection_summaries:
        summary["transfer_bound_projection_summaries"] = (
            transfer_bound_projection_summaries
        )
    if inter_window_refinement_summaries:
        summary["inter_window_refinement_summaries"] = inter_window_refinement_summaries
    if inter_window_control_homotopy_summaries:
        summary["inter_window_control_homotopy_summaries"] = (
            inter_window_control_homotopy_summaries
        )
    if control_homotopy_summaries:
        summary["control_homotopy_summaries"] = control_homotopy_summaries
    if proximal_control_summaries:
        summary["proximal_control_summaries"] = proximal_control_summaries
    if terminal_wheel_bound_summaries:
        summary["terminal_wheel_bound_summaries"] = terminal_wheel_bound_summaries
    if inter_window_proximal_control_summaries:
        summary["inter_window_proximal_control_summaries"] = (
            inter_window_proximal_control_summaries
        )
    if transfer_failure_window is not None:
        summary["transfer_failure_window"] = transfer_failure_window
    if inter_window_terminal_wheel_bound_summaries:
        summary["inter_window_terminal_wheel_bound_summaries"] = (
            inter_window_terminal_wheel_bound_summaries
        )
    if initial_guess_state_traces is not None:
        summary["initial_guess_state_traces"] = initial_guess_state_traces
        summary["initial_guess_control_traces"] = initial_guess_control_traces
    summary["args"] = args
    summary["control_bounds"] = _control_bounds_summary(nmpc)
    summary["initial_guess_audits"] = initial_guess_audits
    summary["initial_guess_preparation_time_s"] = initial_guess_preparation_time_s
    summary["standard_warmup_cache_hit"] = standard_warmup_cache_hit
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
