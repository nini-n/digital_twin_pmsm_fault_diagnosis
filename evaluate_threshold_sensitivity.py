"""
Threshold-sensitivity analysis for the reliability-aware decision layer.

The script evaluates how the final confidence threshold affects false alarms,
uncertain/inspect decisions, hard fault detection, accepted accuracy and safe
decision rate. It uses the same train/calibration/test split and residual-gate
logic as the main article-level framework.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

from utils.energy_gate import ResidualEnergyGate, calibrate_gate_from_healthy, gate_run
from utils.transient_feature_gate import (
    calibrate_transient_gate_from_healthy,
    transient_gate_decision,
)


FEATURES_CSV = "data/features_20260301_204447.csv"
RUNS_INDEX = "data/runs_index_20260301_204447.csv"

OUT_DIR = "paper_results"
TABLE_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")
REPORT_DIR = os.path.join(OUT_DIR, "reports")

CONFORMAL_ALPHA = 0.10
THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

BASE_FEATURE_COLS = [
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

PHYSICS_FEATURE_COLS = [
    "phys_steady_current_energy",
    "phys_steady_total_energy",
    "phys_transient_current_energy",
    "phys_current_axis_imbalance",
    "phys_speed_residual_strength",
    "phys_transient_peak_strength",
]


os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def add_physics_guided_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add residual-energy indicators used by the physics-guided model."""
    out = df.copy()
    eps = 1e-12

    out["phys_steady_current_energy"] = out["rms_r_iq_ss"] ** 2 + out["rms_r_id_ss"] ** 2
    out["phys_steady_total_energy"] = (
        out["rms_r_iq_ss"] ** 2
        + out["rms_r_id_ss"] ** 2
        + out["rms_r_om_ss"] ** 2
    )
    out["phys_transient_current_energy"] = out["energy_r_iq_tr"] + out["energy_r_id_tr"]
    out["phys_current_axis_imbalance"] = (
        np.abs(out["rms_r_iq_ss"] - out["rms_r_id_ss"])
        / (out["rms_r_iq_ss"] + out["rms_r_id_ss"] + eps)
    )
    out["phys_speed_residual_strength"] = out["rms_r_om_ss"] + out["maxabs_r_om_ss"]
    out["phys_transient_peak_strength"] = out["maxabs_r_iq_tr"] + out["maxabs_r_id_tr"]

    return out


def load_run(signals_path: str) -> Dict[str, np.ndarray]:
    """Load the residual channels needed by the gate logic."""
    df = pd.read_csv(signals_path, usecols=["t", "omega_ref", "r_id", "r_iq", "r_omega"])
    return {
        "t": df["t"].to_numpy(),
        "omega_ref": df["omega_ref"].to_numpy(),
        "r_id": df["r_id"].to_numpy(),
        "r_iq": df["r_iq"].to_numpy(),
        "r_omega": df["r_omega"].to_numpy(),
    }


def split_train_cal_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use the same group-aware split as the main framework evaluation."""
    groups = (
        df["omega_step"].astype(str)
        + "_lp"
        + df["load_profile"].astype(str)
        + "_seed"
        + df["seed"].astype(str)
    )

    first_split = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=42)
    train_idx, temp_idx = next(first_split.split(df, df["fault_type"], groups=groups))

    temp_df = df.iloc[temp_idx].copy()
    temp_groups = groups.iloc[temp_idx]

    second_split = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=7)
    cal_rel_idx, test_rel_idx = next(
        second_split.split(temp_df, temp_df["fault_type"], groups=temp_groups)
    )

    cal_idx = temp_df.iloc[cal_rel_idx].index.to_numpy()
    test_idx = temp_df.iloc[test_rel_idx].index.to_numpy()

    return df.loc[train_idx].copy(), df.loc[cal_idx].copy(), df.loc[test_idx].copy()


def conformal_calibrate(
    clf: RandomForestClassifier,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    class_labels: List[str],
    alpha: float = CONFORMAL_ALPHA,
) -> Dict[str, float]:
    """Calibrate split conformal prediction using 1 - P(true class)."""
    proba = clf.predict_proba(X_cal)
    label_to_idx = {label: i for i, label in enumerate(class_labels)}

    scores = np.array(
        [1.0 - proba[i, label_to_idx[label]] for i, label in enumerate(y_cal)],
        dtype=float,
    )

    n_cal = len(scores)
    q_level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    qhat = float(np.quantile(scores, q_level, method="higher"))

    return {
        "alpha": float(alpha),
        "qhat": qhat,
        "n_cal": int(n_cal),
        "q_level": float(q_level),
    }


def conformal_predict(
    clf: RandomForestClassifier,
    X: np.ndarray,
    class_labels: List[str],
    qhat: float,
) -> Tuple[List[List[str]], np.ndarray, np.ndarray, np.ndarray]:
    """Return conformal prediction sets and singleton/uncertain decisions."""
    proba = clf.predict_proba(X)

    prediction_sets: List[List[str]] = []
    singleton_pred = []
    set_sizes = []
    max_probs = []

    for row in proba:
        pred_set = [label for j, label in enumerate(class_labels) if 1.0 - row[j] <= qhat]

        if not pred_set:
            pred_set = [class_labels[int(np.argmax(row))]]

        prediction_sets.append(pred_set)
        set_sizes.append(len(pred_set))
        max_probs.append(float(np.max(row)))
        singleton_pred.append(pred_set[0] if len(pred_set) == 1 else "uncertain")

    return (
        prediction_sets,
        np.asarray(singleton_pred),
        np.asarray(set_sizes),
        np.asarray(max_probs),
    )


def conformal_coverage(y_true: np.ndarray, prediction_sets: List[List[str]]) -> float:
    """Compute the empirical coverage of conformal prediction sets."""
    return float(np.mean([label in pred_set for label, pred_set in zip(y_true, prediction_sets)]))


def build_gates(train_rows: pd.DataFrame, idx_df: pd.DataFrame):
    """Calibrate the slow residual-energy gate and transient feature gate."""
    tag_to_path = dict(zip(idx_df["tag"].astype(str), idx_df["signals_csv"].astype(str)))

    healthy_train = train_rows[train_rows["fault_type"] == "healthy"].copy()
    healthy_paths = [tag_to_path[str(tag)] for tag in healthy_train["tag"].astype(str).tolist()]
    healthy_runs = [load_run(path) for path in healthy_paths[:30]]

    slow_cal = calibrate_gate_from_healthy(
        healthy_runs,
        far=0.005,
        alpha=0.02,
        hysteresis_ratio=0.6,
        drop_initial_seconds=0.02,
        weights=(1.0, 1.0, 1.0),
    )

    slow_gate = ResidualEnergyGate(
        sigmas=slow_cal.sigmas,
        T_high=slow_cal.T_high,
        T_low=slow_cal.T_low,
        alpha=slow_cal.alpha,
        weights=slow_cal.weights,
        N_on=50,
        N_off=50,
        blanking_s=0.05,
        domega_ref_threshold=1e6,
    )

    transient_cal = calibrate_transient_gate_from_healthy(
        healthy_runs,
        t_on=0.10,
        win_s=0.02,
        far=0.01,
    )

    return slow_gate, transient_cal, tag_to_path, slow_cal


def gate_fault_evidence(
    row: pd.Series,
    tag_to_path: Dict[str, str],
    slow_gate: ResidualEnergyGate,
    transient_cal,
) -> Dict[str, float | bool]:
    """Evaluate residual-gate evidence for one test row."""
    run = load_run(tag_to_path[str(row["tag"])])

    slow_out = gate_run(run, slow_gate)
    slow_open = bool(np.any(slow_out["state"] == 1))

    transient_decision, transient_score = transient_gate_decision(run, transient_cal)
    transient_open = bool(transient_decision == 1)

    return {
        "fault_evidence": bool(slow_open or transient_open),
        "slow_open": slow_open,
        "transient_open": transient_open,
        "max_E_bar": float(np.max(slow_out["E_bar"])),
        "transient_score": float(transient_score),
    }


def apply_decision_with_threshold(
    test_rows: pd.DataFrame,
    cp_pred: np.ndarray,
    prediction_sets: List[List[str]],
    max_probs: np.ndarray,
    tag_to_path: Dict[str, str],
    slow_gate: ResidualEnergyGate,
    transient_cal,
    confidence_threshold: float,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Apply the full decision rule for one selected confidence threshold."""
    decisions = []
    rows = []

    for i, (_, row) in enumerate(test_rows.iterrows()):
        pred_set = prediction_sets[i]
        pred = cp_pred[i]
        max_prob = float(max_probs[i])
        evidence = gate_fault_evidence(row, tag_to_path, slow_gate, transient_cal)

        if len(pred_set) != 1:
            decision = "uncertain"
        elif pred == "healthy":
            decision = "healthy"
        elif evidence["fault_evidence"] or max_prob >= confidence_threshold:
            decision = pred
        else:
            decision = "uncertain"

        decisions.append(decision)
        rows.append(
            {
                "tag": row["tag"],
                "true_label": row["fault_type"],
                "cp_prediction": pred,
                "prediction_set": "|".join(pred_set),
                "max_probability": max_prob,
                "confidence_threshold": confidence_threshold,
                "final_decision": decision,
                "fault_evidence": int(evidence["fault_evidence"]),
                "slow_gate_open": int(evidence["slow_open"]),
                "transient_gate_open": int(evidence["transient_open"]),
                "max_E_bar": evidence["max_E_bar"],
                "transient_score": evidence["transient_score"],
            }
        )

    return np.asarray(decisions), pd.DataFrame(rows)


def reliability_metrics_with_uncertain(
    y_true: np.ndarray,
    y_decision: np.ndarray,
    labels: List[str],
) -> Dict[str, float]:
    """Compute reliability metrics while keeping uncertain as its own state."""
    y_true = np.asarray(y_true)
    y_decision = np.asarray(y_decision)

    healthy_mask = y_true == "healthy"
    fault_mask = y_true != "healthy"

    if np.any(healthy_mask):
        healthy_decisions = y_decision[healthy_mask]
        false_alarm_rate = float(
            np.mean((healthy_decisions != "healthy") & (healthy_decisions != "uncertain"))
        )
        healthy_uncertain_rate = float(np.mean(healthy_decisions == "uncertain"))
        healthy_correct_rate = float(np.mean(healthy_decisions == "healthy"))
    else:
        false_alarm_rate = np.nan
        healthy_uncertain_rate = np.nan
        healthy_correct_rate = np.nan

    if np.any(fault_mask):
        true_faults = y_true[fault_mask]
        fault_decisions = y_decision[fault_mask]

        hard_correct = fault_decisions == true_faults
        wrong_fault = (
            (fault_decisions != true_faults)
            & (fault_decisions != "healthy")
            & (fault_decisions != "uncertain")
        )
        missed = fault_decisions == "healthy"
        uncertain_fault = fault_decisions == "uncertain"

        hard_detection_rate = float(np.mean(hard_correct))
        wrong_fault_rate = float(np.mean(wrong_fault))
        missed_detection_rate = float(np.mean(missed))
        uncertain_fault_rate = float(np.mean(uncertain_fault))
    else:
        hard_detection_rate = np.nan
        wrong_fault_rate = np.nan
        missed_detection_rate = np.nan
        uncertain_fault_rate = np.nan

    accepted_mask = y_decision != "uncertain"
    overall_uncertain_rate = float(np.mean(y_decision == "uncertain"))

    if np.any(accepted_mask):
        accepted_accuracy = float(accuracy_score(y_true[accepted_mask], y_decision[accepted_mask]))
        accepted_macro_f1 = float(
            f1_score(
                y_true[accepted_mask],
                y_decision[accepted_mask],
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
    else:
        accepted_accuracy = np.nan
        accepted_macro_f1 = np.nan

    safe_decision_rate = float(np.mean((y_decision == y_true) | (y_decision == "uncertain")))

    return {
        "False Alarm Rate": false_alarm_rate,
        "Healthy Uncertain Rate": healthy_uncertain_rate,
        "Healthy Correct Rate": healthy_correct_rate,
        "Hard Fault Detection Rate": hard_detection_rate,
        "Wrong Fault-Class Rate": wrong_fault_rate,
        "Missed Detection Rate": missed_detection_rate,
        "Uncertain Fault Rate": uncertain_fault_rate,
        "Overall Uncertain Rate": overall_uncertain_rate,
        "Accepted Accuracy": accepted_accuracy,
        "Accepted Macro-F1": accepted_macro_f1,
        "Safe Decision Rate": safe_decision_rate,
    }


def save_threshold_plots(results: pd.DataFrame) -> None:
    """Save the two diagnostic plots used to inspect threshold behavior."""
    plt.figure()
    plt.plot(results["Threshold"], results["False Alarm Rate"], marker="o", label="False Alarm Rate")
    plt.plot(results["Threshold"], results["Overall Uncertain Rate"], marker="o", label="Overall Uncertain Rate")
    plt.plot(results["Threshold"], results["Hard Fault Detection Rate"], marker="o", label="Hard Fault Detection")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Rate")
    plt.title("Threshold Sensitivity of Decision Logic")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "threshold_sensitivity_tradeoff.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("[OK] Saved figure:", fig_path)

    plt.figure()
    plt.plot(results["Threshold"], results["Accepted Accuracy"], marker="o", label="Accepted Accuracy")
    plt.plot(results["Threshold"], results["Safe Decision Rate"], marker="o", label="Safe Decision Rate")
    plt.plot(results["Threshold"], results["Missed Detection Rate"], marker="o", label="Missed Detection Rate")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Rate")
    plt.title("Reliability Metrics vs Threshold")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "threshold_sensitivity_reliability.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("[OK] Saved figure:", fig_path)


def main() -> None:
    print("[INFO] Loading data...")

    df = add_physics_guided_features(pd.read_csv(FEATURES_CSV))
    idx_df = pd.read_csv(RUNS_INDEX)

    class_labels = sorted(df["fault_type"].unique().tolist())
    print("[INFO] Classes:", class_labels)

    train_df, cal_df, test_df = split_train_cal_test(df)
    print(
        f"[INFO] Split sizes: train={len(train_df)}, "
        f"cal={len(cal_df)}, test={len(test_df)}"
    )

    feature_cols = BASE_FEATURE_COLS + PHYSICS_FEATURE_COLS
    X_train = train_df[feature_cols].values
    X_cal = cal_df[feature_cols].values
    X_test = test_df[feature_cols].values

    y_train = train_df["fault_type"].values
    y_cal = cal_df["fault_type"].values
    y_test = test_df["fault_type"].values

    print("[INFO] Training physics-guided diagnostic model...")
    model = RandomForestClassifier(
        n_estimators=700,
        random_state=123,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    print("[INFO] Calibrating conformal prediction...")
    cp = conformal_calibrate(model, X_cal, y_cal, class_labels, alpha=CONFORMAL_ALPHA)
    prediction_sets, cp_pred, set_sizes, max_probs = conformal_predict(
        model,
        X_test,
        class_labels,
        cp["qhat"],
    )

    coverage = conformal_coverage(y_test, prediction_sets)
    avg_set_size = float(np.mean(set_sizes))

    print("[INFO] Building residual-evidence gates...")
    slow_gate, transient_cal, tag_to_path, slow_cal = build_gates(train_df, idx_df)

    result_rows = []
    decision_rows = []

    for threshold in THRESHOLDS:
        print(f"[INFO] Evaluating threshold={threshold:.2f}...")
        final_decision, decision_df = apply_decision_with_threshold(
            test_df,
            cp_pred,
            prediction_sets,
            max_probs,
            tag_to_path,
            slow_gate,
            transient_cal,
            confidence_threshold=threshold,
        )

        metrics = reliability_metrics_with_uncertain(y_test, final_decision, class_labels)
        result_rows.append({"Threshold": threshold, "Coverage": coverage, "Avg Set Size": avg_set_size, **metrics})
        decision_rows.append(decision_df)

    results = pd.DataFrame(result_rows)
    results_path = os.path.join(TABLE_DIR, "threshold_sensitivity_results.csv")
    results.to_csv(results_path, index=False)

    decisions = pd.concat(decision_rows, ignore_index=True)
    decisions_path = os.path.join(REPORT_DIR, "threshold_sensitivity_decisions.csv")
    decisions.to_csv(decisions_path, index=False)

    save_threshold_plots(results)

    meta = {
        "features_csv": FEATURES_CSV,
        "runs_index": RUNS_INDEX,
        "classes": class_labels,
        "split_sizes": {
            "train": int(len(train_df)),
            "calibration": int(len(cal_df)),
            "test": int(len(test_df)),
        },
        "conformal": cp,
        "coverage": coverage,
        "avg_set_size": avg_set_size,
        "thresholds": THRESHOLDS,
        "gate_calibration": {
            "slow_gate": slow_cal.meta,
            "transient_gate": transient_cal.meta,
        },
    }

    meta_path = os.path.join(REPORT_DIR, "threshold_sensitivity_meta.json")
    with open(meta_path, "w") as file:
        json.dump(meta, file, indent=2)

    print("\n=== Threshold Sensitivity Results ===")
    print(results.to_string(index=False))
    print("[OK] Saved:", results_path)
    print("[OK] Saved decisions:", decisions_path)
    print("[OK] Saved meta:", meta_path)


if __name__ == "__main__":
    main()
