"""
Residual-energy gate used for false-alarm-controlled decision logic.

The gate calibrates residual energy statistics from healthy simulation runs and
then applies an EWMA-smoothed, hysteresis-based state machine to new runs. It is
used as the residual-evidence layer in the reliability-aware fault diagnosis
framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def _safe_std(x: np.ndarray, eps: float = 1e-12) -> float:
    """Return a standard deviation with a small lower bound for stability."""
    std = float(np.std(x))
    return std if std > eps else eps


def compute_channel_sigmas_from_runs(
    runs: List[Dict],
    start_t: Optional[float] = None,
    end_t: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Estimate residual channel scales from a set of healthy runs.

    The returned standard deviations are used to normalize r_id, r_iq and
    r_omega before residual energy is computed.
    """
    rid_all, riq_all, rom_all = [], [], []

    for run in runs:
        t = np.asarray(run["t"])
        r_id = np.asarray(run["r_id"])
        r_iq = np.asarray(run["r_iq"])
        r_omega = np.asarray(run["r_omega"])

        mask = np.ones_like(t, dtype=bool)
        if start_t is not None:
            mask &= t >= float(start_t)
        if end_t is not None:
            mask &= t <= float(end_t)

        rid_all.append(r_id[mask])
        riq_all.append(r_iq[mask])
        rom_all.append(r_omega[mask])

    rid_cat = np.concatenate(rid_all) if rid_all else np.array([0.0])
    riq_cat = np.concatenate(riq_all) if riq_all else np.array([0.0])
    rom_cat = np.concatenate(rom_all) if rom_all else np.array([0.0])

    return _safe_std(rid_cat), _safe_std(riq_cat), _safe_std(rom_cat)


def residual_energy(
    r_id: float,
    r_iq: float,
    r_omega: float,
    sigmas: Tuple[float, float, float],
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    Compute normalized squared residual energy.

    E = w_id * (r_id / sigma_id)^2
      + w_iq * (r_iq / sigma_iq)^2
      + w_omega * (r_omega / sigma_omega)^2
    """
    sigma_id, sigma_iq, sigma_omega = sigmas
    w_id, w_iq, w_omega = weights

    z_id = float(r_id) / sigma_id
    z_iq = float(r_iq) / sigma_iq
    z_omega = float(r_omega) / sigma_omega

    return float(
        w_id * z_id * z_id
        + w_iq * z_iq * z_iq
        + w_omega * z_omega * z_omega
    )


def ewma_filter(x: np.ndarray, alpha: float) -> np.ndarray:
    """Apply an exponentially weighted moving average filter."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    if len(x) == 0:
        return y

    for k in range(1, len(x)):
        y[k] = alpha * x[k] + (1.0 - alpha) * y[k - 1]

    return y


def estimate_dt(t: np.ndarray) -> float:
    """Estimate the simulation step size from a time vector."""
    t = np.asarray(t)
    if len(t) < 2:
        return 1e-5
    return float(t[1] - t[0])


@dataclass(frozen=True)
class GateCalibration:
    """Calibration parameters for the residual-energy gate."""

    sigmas: Tuple[float, float, float]
    T_high: float
    T_low: float
    alpha: float
    weights: Tuple[float, float, float]
    far: float
    meta: Dict


def calibrate_gate_from_healthy(
    healthy_runs: List[Dict],
    *,
    far: float = 0.01,
    alpha: float = 0.02,
    hysteresis_ratio: float = 0.6,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    sigma_start_t: Optional[float] = None,
    sigma_end_t: Optional[float] = None,
    energy_start_t: Optional[float] = None,
    energy_end_t: Optional[float] = None,
    drop_initial_seconds: float = 0.0,
) -> GateCalibration:
    """
    Calibrate residual-energy thresholds from healthy runs.

    The high threshold is selected using a healthy-run quantile determined by
    the target false alarm rate. The low threshold is derived from the high
    threshold to provide hysteresis.
    """
    if not healthy_runs:
        raise ValueError("healthy_runs is empty. Provide at least one healthy run.")

    sigmas = compute_channel_sigmas_from_runs(
        healthy_runs,
        start_t=sigma_start_t,
        end_t=sigma_end_t,
    )

    pooled_energies = []

    for run in healthy_runs:
        t = np.asarray(run["t"])
        r_id = np.asarray(run["r_id"])
        r_iq = np.asarray(run["r_iq"])
        r_omega = np.asarray(run["r_omega"])

        mask = np.ones_like(t, dtype=bool)
        if drop_initial_seconds > 0:
            mask &= t >= t[0] + float(drop_initial_seconds)
        if energy_start_t is not None:
            mask &= t >= float(energy_start_t)
        if energy_end_t is not None:
            mask &= t <= float(energy_end_t)

        energy = np.array(
            [
                residual_energy(r_id[k], r_iq[k], r_omega[k], sigmas, weights=weights)
                for k in range(len(t))
            ],
            dtype=float,
        )

        energy = energy[mask]
        if len(energy) < 5:
            continue

        pooled_energies.append(ewma_filter(energy, alpha=alpha))

    if not pooled_energies:
        raise ValueError(
            "Calibration failed because no valid healthy samples were pooled. "
            "Check the calibration windows or reduce drop_initial_seconds."
        )

    pooled = np.concatenate(pooled_energies)

    far = float(np.clip(far, 1e-6, 0.5))
    quantile = 1.0 - far
    T_high = float(np.quantile(pooled, quantile))
    T_low = float(hysteresis_ratio * T_high)

    meta = {
        "n_runs": len(healthy_runs),
        "n_samples": int(len(pooled)),
        "quantile": quantile,
        "hysteresis_ratio": hysteresis_ratio,
        "drop_initial_seconds": drop_initial_seconds,
        "sigma_window": (sigma_start_t, sigma_end_t),
        "energy_window": (energy_start_t, energy_end_t),
    }

    return GateCalibration(
        sigmas=sigmas,
        T_high=T_high,
        T_low=T_low,
        alpha=float(alpha),
        weights=weights,
        far=far,
        meta=meta,
    )


class ResidualEnergyGate:
    """
    EWMA-smoothed residual-energy gate with hysteresis.

    State 0 represents a closed gate, interpreted as no hard residual evidence.
    State 1 represents an open gate, interpreted as residual evidence supporting
    a fault decision.
    """

    def __init__(
        self,
        *,
        sigmas: Tuple[float, float, float],
        T_high: float,
        T_low: float,
        alpha: float = 0.02,
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        N_on: int = 25,
        N_off: int = 25,
        blanking_s: float = 0.05,
        domega_ref_threshold: float = 1e6,
        dt: Optional[float] = None,
    ):
        self.sigmas = sigmas
        self.T_high = float(T_high)
        self.T_low = float(T_low)
        self.alpha = float(alpha)
        self.weights = weights

        self.N_on = int(N_on)
        self.N_off = int(N_off)

        self.dt = dt
        self.blanking_s = float(blanking_s)
        self.domega_ref_threshold = float(domega_ref_threshold)

        self.state = 0
        self._counter = 0
        self._energy_ewma = 0.0
        self._blank_countdown = 0
        self._prev_omega_ref = None

    @property
    def energy_ewma(self) -> float:
        return float(self._energy_ewma)

    @property
    def blanking_active(self) -> bool:
        return self._blank_countdown > 0

    def reset(self) -> None:
        """Reset the internal gate state before evaluating a new run."""
        self.state = 0
        self._counter = 0
        self._energy_ewma = 0.0
        self._blank_countdown = 0
        self._prev_omega_ref = None

    def _update_blanking(self, omega_ref: Optional[float], dt: float) -> None:
        if omega_ref is None or not np.isfinite(self.domega_ref_threshold):
            return

        if self._prev_omega_ref is None:
            self._prev_omega_ref = float(omega_ref)
            return

        domega = (float(omega_ref) - self._prev_omega_ref) / max(dt, 1e-12)
        self._prev_omega_ref = float(omega_ref)

        if abs(domega) > self.domega_ref_threshold:
            self._blank_countdown = int(round(self.blanking_s / max(dt, 1e-12)))

        if self._blank_countdown > 0:
            self._blank_countdown -= 1

    def update(
        self,
        r_id: float,
        r_iq: float,
        r_omega: float,
        *,
        omega_ref: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> int:
        """
        Update the gate for one time step.

        Returns
        -------
        int
            0 for closed gate and 1 for open gate.
        """
        dt_eff = float(dt) if dt is not None else (
            float(self.dt) if self.dt is not None else 1e-5
        )

        self._update_blanking(omega_ref, dt_eff)

        if self.blanking_active:
            self.state = 0
            self._counter = 0
            return self.state

        energy = residual_energy(r_id, r_iq, r_omega, self.sigmas, self.weights)
        self._energy_ewma = self.alpha * energy + (1.0 - self.alpha) * self._energy_ewma

        if self.state == 0:
            if self._energy_ewma > self.T_high:
                self._counter += 1
                if self._counter >= self.N_on:
                    self.state = 1
                    self._counter = 0
            else:
                self._counter = 0
        else:
            if self._energy_ewma < self.T_low:
                self._counter += 1
                if self._counter >= self.N_off:
                    self.state = 0
                    self._counter = 0
            else:
                self._counter = 0

        return self.state


def gate_run(
    run: Dict,
    gate: ResidualEnergyGate,
    *,
    start_t: Optional[float] = None,
    end_t: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Apply a residual-energy gate to one complete simulation run.

    Returns a dictionary containing the time vector, gate state and EWMA energy.
    """
    t = np.asarray(run["t"])
    r_id = np.asarray(run["r_id"])
    r_iq = np.asarray(run["r_iq"])
    r_omega = np.asarray(run["r_omega"])
    omega_ref = np.asarray(run["omega_ref"]) if "omega_ref" in run else None

    dt = estimate_dt(t)

    mask = np.ones_like(t, dtype=bool)
    if start_t is not None:
        mask &= t >= float(start_t)
    if end_t is not None:
        mask &= t <= float(end_t)

    states = np.zeros_like(t, dtype=int)
    energy_ewma = np.zeros_like(t, dtype=float)

    gate.reset()

    for k in np.where(mask)[0]:
        state = gate.update(
            float(r_id[k]),
            float(r_iq[k]),
            float(r_omega[k]),
            omega_ref=float(omega_ref[k]) if omega_ref is not None else None,
            dt=dt,
        )
        states[k] = state
        energy_ewma[k] = gate.energy_ewma

    return {"t": t, "state": states, "E_bar": energy_ewma}


def plot_gate_overlay(
    run: Dict,
    gate_out: Dict,
    title: str = "Residual Energy Gate",
) -> None:
    """Plot the EWMA residual energy together with the gate state."""
    import matplotlib.pyplot as plt

    t = gate_out["t"]
    energy_ewma = gate_out["E_bar"]
    state = gate_out["state"]

    plt.figure()
    plt.plot(t, energy_ewma, label="EWMA residual energy")
    plt.plot(t, state * (np.max(energy_ewma) * 0.9), label="Gate state")
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()