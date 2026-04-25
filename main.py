"""
Simulation and dataset generation for a PMSM digital-twin fault diagnosis framework.

The script runs a closed-loop PMSM plant and a nominal digital twin in parallel.
Faults are injected only into the plant, while the twin remains at the healthy
nominal parameter set. The difference between the plant and the twin is saved as 
residual signals andlater converted into steady-state and transient diagnostic features.
The generated data are used by the reliability-aware evaluation scripts in this
repository. The project is fully simulation-based and does not require hardware.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from controllers.pi import PIController
from models.pmsm import PMSM


SIGNAL_COLUMNS = [
    "t",
    "omega",
    "omega_ref",
    "id",
    "iq",
    "iq_ref",
    "Te",
    "Tl",
    "vd",
    "vq",
    "vd_pi",
    "vq_pi",
    "vd_ff",
    "vq_ff",
    "id_hat",
    "iq_hat",
    "omega_hat",
    "r_id",
    "r_iq",
    "r_omega",
]

BASE_FEATURE_COLUMNS = [
    "rms_r_iq_ss",
    "maxabs_r_iq_ss",
    "mean_r_iq_ss",
    "std_r_iq_ss",
    "rms_r_id_ss",
    "maxabs_r_id_ss",
    "mean_r_id_ss",
    "std_r_id_ss",
    "rms_r_om_ss",
    "maxabs_r_om_ss",
    "rms_r_iq_tr",
    "maxabs_r_iq_tr",
    "energy_r_iq_tr",
    "fdom_r_iq_tr",
    "fratio_r_iq_tr",
    "rms_r_id_tr",
    "maxabs_r_id_tr",
    "energy_r_id_tr",
    "fdom_r_id_tr",
    "fratio_r_id_tr",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def apply_faults(motor: PMSM, t: float, fault_cfg: Optional[Dict] = None) -> Tuple[str, float]:
    """
    Apply a scheduled parameter fault to the plant model.

    The plant parameters are reset to their nominal values at each step. If the
    fault activation time has passed, the selected parameter is modified. The
    nominal digital twin is not passed here, so it remains healthy.
    """
    motor.Rs = motor.Rs_nominal
    motor.Ld = motor.Ld_nominal

    if fault_cfg is None:
        return "healthy", 0.0

    fault_type = fault_cfg.get("type", "healthy")
    t_on = float(fault_cfg.get("t_on", 1e9))
    severity = float(fault_cfg.get("severity", 0.0))

    if t < t_on:
        return "healthy", 0.0

    if fault_type == "rs_drift":
        motor.Rs = motor.Rs_nominal * (1.0 + severity)
        return "rs_drift", severity

    if fault_type == "ld_mismatch":
        motor.Ld = motor.Ld_nominal * (1.0 + severity)
        return "ld_mismatch", severity

    return "healthy", 0.0


def severity_to_tag(severity: float) -> str:
    """Convert a fault severity value into a compact filename-safe tag."""
    value = int(round(severity * 100))
    prefix = "p" if value >= 0 else "m"
    return f"{prefix}{abs(value):02d}"


def save_run_to_csv(run: Dict[str, np.ndarray], out_csv_path: str) -> None:
    """Save one simulated run as a signal-level CSV file."""
    n_samples = len(run["t"])

    with open(out_csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(SIGNAL_COLUMNS)

        for i in range(n_samples):
            writer.writerow([run[column][i] for column in SIGNAL_COLUMNS])


def _safe_window(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = x[mask]
    return values if len(values) else np.array([0.0])


def _rms(x: np.ndarray, mask: np.ndarray) -> float:
    values = _safe_window(x, mask)
    return float(np.sqrt(np.mean(values * values)))


def _max_abs(x: np.ndarray, mask: np.ndarray) -> float:
    values = _safe_window(x, mask)
    return float(np.max(np.abs(values)))


def _mean(x: np.ndarray, mask: np.ndarray) -> float:
    values = _safe_window(x, mask)
    return float(np.mean(values))


def _std(x: np.ndarray, mask: np.ndarray) -> float:
    values = _safe_window(x, mask)
    return float(np.std(values))


def _energy(x: np.ndarray, mask: np.ndarray) -> float:
    values = _safe_window(x, mask)
    return float(np.sum(values * values))


def _dominant_frequency_and_ratio(x: np.ndarray, mask: np.ndarray, dt: float) -> Tuple[float, float]:
    """Return the dominant transient frequency and its spectral dominance ratio."""
    values = _safe_window(x, mask)
    if len(values) < 16:
        return 0.0, 0.0

    centered = values - np.mean(values)
    window = np.hanning(len(centered))
    spectrum = np.fft.rfft(centered * window)
    magnitude = np.abs(spectrum)
    frequencies = np.fft.rfftfreq(len(centered), d=dt)

    if len(magnitude) < 2:
        return 0.0, 0.0

    magnitude[0] = 0.0
    peak_idx = int(np.argmax(magnitude))
    dominance_ratio = float(magnitude[peak_idx] / (np.sum(magnitude) + 1e-12))

    return float(frequencies[peak_idx]), dominance_ratio


def compute_features_from_residual(run: Dict[str, np.ndarray], t_on: float = 0.10) -> Dict[str, float]:
    """
    Extract steady-state and transient features from plant--twin residuals.

    Residuals are defined as measured plant response minus nominal twin response:
    r_id, r_iq and r_omega. A short transient window is used after the fault
    activation time, while a later window is used for steady-state mismatch.
    """
    t = run["t"]
    dt = float(t[1] - t[0]) if len(t) > 1 else 1e-5

    r_id = run["r_id"]
    r_iq = run["r_iq"]
    r_omega = run["r_omega"]

    transient_mask = (t >= t_on) & (t < t_on + 0.02)
    steady_mask = t >= t_on + 0.05

    f_iq, fr_iq = _dominant_frequency_and_ratio(r_iq, transient_mask, dt)
    f_id, fr_id = _dominant_frequency_and_ratio(r_id, transient_mask, dt)

    return {
        "rms_r_iq_ss": _rms(r_iq, steady_mask),
        "maxabs_r_iq_ss": _max_abs(r_iq, steady_mask),
        "mean_r_iq_ss": _mean(r_iq, steady_mask),
        "std_r_iq_ss": _std(r_iq, steady_mask),
        "rms_r_id_ss": _rms(r_id, steady_mask),
        "maxabs_r_id_ss": _max_abs(r_id, steady_mask),
        "mean_r_id_ss": _mean(r_id, steady_mask),
        "std_r_id_ss": _std(r_id, steady_mask),
        "rms_r_om_ss": _rms(r_omega, steady_mask),
        "maxabs_r_om_ss": _max_abs(r_omega, steady_mask),
        "rms_r_iq_tr": _rms(r_iq, transient_mask),
        "maxabs_r_iq_tr": _max_abs(r_iq, transient_mask),
        "energy_r_iq_tr": _energy(r_iq, transient_mask),
        "fdom_r_iq_tr": f_iq,
        "fratio_r_iq_tr": fr_iq,
        "rms_r_id_tr": _rms(r_id, transient_mask),
        "maxabs_r_id_tr": _max_abs(r_id, transient_mask),
        "energy_r_id_tr": _energy(r_id, transient_mask),
        "fdom_r_id_tr": f_id,
        "fratio_r_id_tr": fr_id,
    }


def default_scenario() -> Dict[str, float]:
    return {
        "omega_step": 50.0,
        "t_speed": 0.02,
        "Tl0": 0.5,
        "Tl1": 1.0,
        "t_load": 0.12,
    }


def run_simulation(
    params: Dict[str, float],
    fault_cfg: Optional[Dict] = None,
    scenario: Optional[Dict[str, float]] = None,
    seed: int = 0,
    noise_std: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Simulate a closed-loop PMSM plant and a nominal digital twin.

    The controller acts on the plant measurements. The same voltage inputs are
    applied to the nominal twin, which provides the expected healthy response.
    Residual signals are computed from the measured plant response and the twin
    output at each simulation step.
    """
    plant = PMSM(params)
    twin = PMSM(params)

    rng = np.random.default_rng(int(seed))
    scenario = default_scenario() if scenario is None else scenario

    dt = 1e-5
    t_sim = 0.2
    steps = int(t_sim / dt)

    current_bandwidth = 500.0
    speed_bandwidth = 25.0

    kp_d = params["Ld"] * current_bandwidth
    ki_d = params["Rs"] * current_bandwidth
    kp_q = params["Lq"] * current_bandwidth
    ki_q = params["Rs"] * current_bandwidth

    voltage_limit = 200.0
    pi_d = PIController(kp_d, ki_d, limit=voltage_limit)
    pi_q = PIController(kp_q, ki_q, limit=voltage_limit)

    torque_constant = 1.5 * params["p"] * params["lambda_m"]
    kp_w = (params["J"] * speed_bandwidth) / torque_constant
    ki_w = 0.5 * kp_w * speed_bandwidth

    iq_limit = 30.0
    pi_w = PIController(kp_w, ki_w, limit=iq_limit)
    id_ref = 0.0

    logs = {column: np.zeros(steps) for column in SIGNAL_COLUMNS}
    fault_label_log = np.empty(steps, dtype=object)
    fault_severity_log = np.zeros(steps)

    for k in range(steps):
        t = k * dt

        omega_ref = 0.0 if t < float(scenario["t_speed"]) else float(scenario["omega_step"])
        load_torque = float(scenario["Tl0"]) if t < float(scenario["t_load"]) else float(scenario["Tl1"])

        fault_label, fault_severity = apply_faults(plant, t, fault_cfg)

        iq_ref = pi_w.update(omega_ref - plant.omega, dt)

        vd_pi = pi_d.update(id_ref - plant.id, dt)
        vq_pi = pi_q.update(iq_ref - plant.iq, dt)

        omega_e = params["p"] * plant.omega
        vd_ff = -omega_e * params["Lq"] * plant.iq
        vq_ff = omega_e * params["Ld"] * plant.id + omega_e * params["lambda_m"]

        vd = float(np.clip(vd_pi + vd_ff, -voltage_limit, voltage_limit))
        vq = float(np.clip(vq_pi + vq_ff, -voltage_limit, voltage_limit))

        id_f, iq_f, omega_f, torque = plant.step(vd=vd, vq=vq, Tl=load_torque, dt=dt)
        id_hat, iq_hat, omega_hat, _ = twin.step(vd=vd, vq=vq, Tl=load_torque, dt=dt)

        if noise_std > 0.0:
            id_meas = id_f + rng.normal(0.0, noise_std)
            iq_meas = iq_f + rng.normal(0.0, noise_std)
            omega_meas = omega_f + rng.normal(0.0, noise_std)
        else:
            id_meas = id_f
            iq_meas = iq_f
            omega_meas = omega_f

        logs["t"][k] = t
        logs["omega"][k] = omega_meas
        logs["omega_ref"][k] = omega_ref
        logs["id"][k] = id_meas
        logs["iq"][k] = iq_meas
        logs["iq_ref"][k] = iq_ref
        logs["Te"][k] = torque
        logs["Tl"][k] = load_torque
        logs["vd"][k] = vd
        logs["vq"][k] = vq
        logs["vd_pi"][k] = vd_pi
        logs["vq_pi"][k] = vq_pi
        logs["vd_ff"][k] = vd_ff
        logs["vq_ff"][k] = vq_ff
        logs["id_hat"][k] = id_hat
        logs["iq_hat"][k] = iq_hat
        logs["omega_hat"][k] = omega_hat
        logs["r_id"][k] = id_meas - id_hat
        logs["r_iq"][k] = iq_meas - iq_hat
        logs["r_omega"][k] = omega_meas - omega_hat

        fault_label_log[k] = fault_label
        fault_severity_log[k] = fault_severity

    logs["fault_label"] = fault_label_log
    logs["fault_severity"] = fault_severity_log
    logs["meta"] = {
        "dt": dt,
        "Tsim": t_sim,
        "current_bandwidth": current_bandwidth,
        "speed_bandwidth": speed_bandwidth,
        "Kp_w": kp_w,
        "Ki_w": ki_w,
        "Kp_d": kp_d,
        "Ki_d": ki_d,
        "Kp_q": kp_q,
        "Ki_q": ki_q,
        "fault_cfg": fault_cfg,
        "model": "closed_loop_pmsm_with_nominal_twin",
    }

    return logs


def write_json(path: str, data: Dict) -> None:
    with open(path, "w") as file:
        json.dump(data, file, indent=2)


def write_rows_to_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_dataset() -> Tuple[str, str, int]:
    """Generate all simulation runs and save the run index and feature table."""
    params = {
        "Rs": 0.5,
        "Ld": 0.001,
        "Lq": 0.001,
        "lambda_m": 0.05,
        "p": 4,
        "J": 0.001,
        "B": 0.0001,
    }

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir("data/runs")

    omega_steps = [40.0, 50.0, 60.0]
    load_profiles = [
        {"Tl0": 0.5, "Tl1": 1.0, "t_load": 0.12},
        {"Tl0": 0.3, "Tl1": 0.8, "t_load": 0.10},
        {"Tl0": 0.6, "Tl1": 1.2, "t_load": 0.15},
    ]
    seeds = [0, 1, 2]
    t_on = 0.10
    noise_std = 0.01
    healthy_reps = 3

    fault_plan = [
        ("rs_drift", [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40]),
        ("ld_mismatch", [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]),
    ]

    index_rows: List[Dict] = []
    feature_rows: List[Dict] = []

    for omega_step in omega_steps:
        for load_idx, load_profile in enumerate(load_profiles):
            for seed in seeds:
                scenario = {
                    "omega_step": omega_step,
                    "t_speed": 0.02,
                    "Tl0": load_profile["Tl0"],
                    "Tl1": load_profile["Tl1"],
                    "t_load": load_profile["t_load"],
                }
                base_tag = f"{run_id}_w{int(omega_step)}_lp{load_idx}_seed{seed}"

                for rep in range(healthy_reps):
                    rep_seed = seed + 1000 * rep
                    tag = f"{base_tag}_healthy_r{rep}"
                    run = run_simulation(
                        params,
                        fault_cfg=None,
                        scenario=scenario,
                        seed=rep_seed,
                        noise_std=noise_std,
                    )
                    add_run_to_dataset(
                        tag=tag,
                        run=run,
                        fault_type="healthy",
                        severity=0.0,
                        scenario=scenario,
                        seed=seed,
                        run_seed=rep_seed,
                        noise_std=noise_std,
                        load_profile=load_idx,
                        omega_step=omega_step,
                        t_on=t_on,
                        index_rows=index_rows,
                        feature_rows=feature_rows,
                    )

                for fault_type, severities in fault_plan:
                    for severity in severities:
                        fault_cfg = {"type": fault_type, "t_on": t_on, "severity": severity}
                        tag = f"{base_tag}_{fault_type}_{severity_to_tag(severity)}"
                        run = run_simulation(
                            params,
                            fault_cfg=fault_cfg,
                            scenario=scenario,
                            seed=seed,
                            noise_std=noise_std,
                        )
                        add_run_to_dataset(
                            tag=tag,
                            run=run,
                            fault_type=fault_type,
                            severity=severity,
                            scenario=scenario,
                            seed=seed,
                            run_seed=seed,
                            noise_std=noise_std,
                            load_profile=load_idx,
                            omega_step=omega_step,
                            t_on=t_on,
                            index_rows=index_rows,
                            feature_rows=feature_rows,
                            fault_cfg=fault_cfg,
                        )

    index_path = f"data/runs_index_{run_id}.csv"
    feature_path = f"data/features_{run_id}.csv"

    write_rows_to_csv(index_path, index_rows)
    write_rows_to_csv(feature_path, feature_rows)

    return index_path, feature_path, len(index_rows)


def add_run_to_dataset(
    *,
    tag: str,
    run: Dict[str, np.ndarray],
    fault_type: str,
    severity: float,
    scenario: Dict[str, float],
    seed: int,
    run_seed: int,
    noise_std: float,
    load_profile: int,
    omega_step: float,
    t_on: float,
    index_rows: List[Dict],
    feature_rows: List[Dict],
    fault_cfg: Optional[Dict] = None,
) -> None:
    """Save one run and append its metadata and feature row."""
    run_dir = f"data/runs/{tag}"
    ensure_dir(run_dir)

    signals_csv = f"{run_dir}/signals.csv"
    meta_json = f"{run_dir}/meta.json"

    save_run_to_csv(run, signals_csv)
    write_json(
        meta_json,
        {
            "tag": tag,
            "fault_type": fault_type,
            "severity": severity,
            "fault_cfg": fault_cfg,
            "scenario": scenario,
            "seed": run_seed,
            "noise_std": noise_std,
            **run["meta"],
        },
    )

    index_rows.append(
        {
            "tag": tag,
            "signals_csv": signals_csv,
            "meta_json": meta_json,
            "fault_type": fault_type,
            "severity": severity,
            "omega_step": omega_step,
            "load_profile": load_profile,
            "seed": seed,
            "noise_std": noise_std,
        }
    )

    features = compute_features_from_residual(run, t_on=t_on)
    feature_rows.append(
        {
            "tag": tag,
            "fault_type": fault_type,
            "severity": severity,
            "omega_step": omega_step,
            "load_profile": load_profile,
            "seed": seed,
            **features,
        }
    )


def run_quick_baseline(feature_path: str) -> None:
    """Run a small scenario-aware baseline check on the generated feature table."""
    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.model_selection import GroupShuffleSplit
    except ImportError:
        print("\nOptional baseline skipped. Install pandas and scikit-learn to enable it.")
        return

    df = pd.read_csv(feature_path)
    x = df[BASE_FEATURE_COLUMNS].values
    y = df["fault_type"].values

    groups = (
        df["omega_step"].astype(str)
        + "_"
        + df["load_profile"].astype(str)
        + "_"
        + df["seed"].astype(str)
    )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, test_idx = next(splitter.split(x, y, groups=groups))

    clf = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(x[train_idx], y[train_idx])
    y_pred = clf.predict(x[test_idx])

    print("\n[Quick baseline: healthy vs rs_drift vs ld_mismatch]")
    print(classification_report(y[test_idx], y_pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y[test_idx], y_pred))


def main() -> None:
    index_path, feature_path, n_runs = build_dataset()

    print(f"[OK] Dataset generated: {n_runs} runs")
    print(f"[OK] Index: {index_path}")
    print(f"[OK] Features: {feature_path}")

    run_quick_baseline(feature_path)


if __name__ == "__main__":
    main()
