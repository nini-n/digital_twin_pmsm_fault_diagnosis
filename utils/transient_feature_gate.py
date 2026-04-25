"""
Transient residual feature gate for short-window fault evidence.

The module computes a log-energy score from r_id and r_iq in a short window
after the fault activation time. The threshold is calibrated from healthy runs
using a selected false-alarm quantile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def _energy(x: np.ndarray) -> float:
    """Return the squared energy of a signal segment."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x * x))


def compute_transient_score(
    run: Dict,
    t_on: float = 0.10,
    win_s: float = 0.02,
    eps: float = 1e-12,
) -> float:
    """
    Compute the transient log-energy score for one run.

    The score is evaluated over the window [t_on, t_on + win_s] using the
    d-axis and q-axis current residuals:

        score = log(E_id + eps) + log(E_iq + eps)

    where E_id and E_iq are the residual energies in the selected window.
    """
    t = np.asarray(run["t"])
    r_id = np.asarray(run["r_id"])
    r_iq = np.asarray(run["r_iq"])

    mask = (t >= float(t_on)) & (t <= float(t_on + win_s))
    if not np.any(mask):
        return 0.0

    energy_id = _energy(r_id[mask])
    energy_iq = _energy(r_iq[mask])

    return float(np.log(energy_id + eps) + np.log(energy_iq + eps))


@dataclass(frozen=True)
class TransientGateCalibration:
    """Calibration result for the transient feature gate."""

    T_tr: float
    far: float
    t_on: float
    win_s: float
    meta: Dict


def calibrate_transient_gate_from_healthy(
    healthy_runs: List[Dict],
    *,
    t_on: float = 0.10,
    win_s: float = 0.02,
    far: float = 0.01,
) -> TransientGateCalibration:
    """
    Calibrate the transient score threshold from healthy runs.

    The threshold is selected as the (1 - FAR) quantile of healthy transient
    scores. This keeps the transient gate consistent with the false-alarm
    control used by the rest of the framework.
    """
    scores = [
        compute_transient_score(run, t_on=t_on, win_s=win_s)
        for run in healthy_runs
    ]

    if not scores:
        raise ValueError("No transient scores computed. Provide at least one healthy run.")

    scores = np.asarray(scores, dtype=float)
    far = float(np.clip(far, 1e-6, 0.5))
    quantile = 1.0 - far
    threshold = float(np.quantile(scores, quantile))

    meta = {
        "n_runs": int(len(scores)),
        "quantile": quantile,
        "mean_healthy": float(np.mean(scores)),
        "std_healthy": float(np.std(scores)),
        "min_healthy": float(np.min(scores)),
        "max_healthy": float(np.max(scores)),
    }

    return TransientGateCalibration(
        T_tr=threshold,
        far=far,
        t_on=float(t_on),
        win_s=float(win_s),
        meta=meta,
    )


def transient_gate_decision(
    run: Dict,
    cal: TransientGateCalibration,
) -> Tuple[int, float]:
    """
    Apply the calibrated transient gate to one run.

    Returns
    -------
    tuple
        decision and score. The decision is 1 when transient residual evidence
        exceeds the calibrated threshold, otherwise 0.
    """
    score = compute_transient_score(run, t_on=cal.t_on, win_s=cal.win_s)
    decision = 1 if score > cal.T_tr else 0

    return decision, score