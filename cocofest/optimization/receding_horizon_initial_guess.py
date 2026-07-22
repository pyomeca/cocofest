"""Solver-independent helpers for receding-horizon initial guesses."""

from __future__ import annotations

import hashlib

import numpy as np


def snapshot_container(container) -> dict[str, np.ndarray]:
    """Copy every physical array from a bioptim initial-guess container."""

    return {
        key: np.asarray(container[key].init, dtype=float).copy()
        for key in container.keys()
    }


def snapshot_initial_guess(program) -> dict[str, dict[str, np.ndarray]]:
    """Copy the physical state and control initial guesses of a one-phase program."""

    nlp = program.nlp[0]
    return {
        "states": snapshot_container(nlp.x_init),
        "controls": snapshot_container(nlp.u_init),
    }


def initial_guess_signature(snapshot: dict[str, dict[str, np.ndarray]]) -> str:
    """Return a deterministic signature for a physical primal initial guess."""

    digest = hashlib.sha256()
    for category in ("states", "controls"):
        digest.update(category.encode("ascii"))
        for key in sorted(snapshot.get(category, {})):
            values = np.ascontiguousarray(snapshot[category][key], dtype=np.float64)
            digest.update(key.encode("utf-8"))
            digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            digest.update(values.tobytes())
    return digest.hexdigest()[:16]


def audit_initial_guess(program) -> dict:
    """Summarize an initial guess without depending on a solver backend."""

    snapshot = snapshot_initial_guess(program)
    arrays = [values for category in snapshot.values() for values in category.values()]
    return {
        "signature": initial_guess_signature(snapshot),
        "finite": all(np.all(np.isfinite(values)) for values in arrays),
        "state_shapes": {
            key: values.shape for key, values in snapshot["states"].items()
        },
        "control_shapes": {
            key: values.shape for key, values in snapshot["controls"].items()
        },
        "snapshot": snapshot,
    }


def project_initial_guess_to_bounds(program, sync_bounds=None) -> dict:
    """Clip a primal initial guess to the program bounds and report every change."""

    before = snapshot_initial_guess(program)
    program._correct_init_guess_to_fit_bounds(corrected_input="states")
    program._correct_init_guess_to_fit_bounds(corrected_input="controls")
    if sync_bounds is not None:
        sync_bounds()
    after = snapshot_initial_guess(program)

    state_changes = {
        key: _max_abs(after["states"][key] - values)
        for key, values in before["states"].items()
    }
    control_changes = {
        key: _max_abs(after["controls"][key] - values)
        for key, values in before["controls"].items()
    }
    return {
        "state_max_change": max(state_changes.values(), default=0.0),
        "control_max_change": max(control_changes.values(), default=0.0),
        "state_changes": state_changes,
        "control_changes": control_changes,
        "signature_before": initial_guess_signature(before),
        "signature_after": initial_guess_signature(after),
    }


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def copy_container_values(source, target, attribute_name: str) -> None:
    """Copy matching entries between bioptim initial-guess or bounds containers."""

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
