"""
Main article-level evaluation for the reliability-aware PMSM fault diagnosis framework.

This script compares four diagnostic configurations:

1. ML baseline
2. Physics-guided PIML feature model
3. PIML with conformal prediction
4. Full framework with conformal uncertainty and residual-evidence gating

The evaluation uses reliability-aware metrics. Uncertain fault cases are treated
as inspection decisions, not as missed detections. This keeps the reported
metrics consistent with the intended use of the framework: avoid unreliable hard
alarms when the diagnostic evidence is weak.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit

from utils.energy_gate import (
    ResidualEnergyGate,
    calibrate_gate_from_healthy,
    gate_run,
)
from utils.transient_feature_gate import (
    calibrate_transient_gate_from_healthy,
    transient_gate_decision,
)


FEATURES_CSV = "data/features_20260301_204447.csv"
RUNS_INDEX = "data/runs_index_20260301_204447.csv"

OUT_DIR = "paper_results"
TABLE_DIR = os.path.join(OUT_DIR, "tables")
REPORT_DIR = os.path.join(OUT_DIR, "reports")

CONFIDENCE_THRESHOLD = 0.80
CONFORMAL_ALPHA = 0.10

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


def ensure_output_dirs() -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


def add_physics_guided_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add residual-based physics-guided diagnostic indicators.

    These features are not a full PINN formulation. They are practical
    physics-guided indicators derived from residual energy, current-axis
    imbalance, speed residual strength, and transient peak behavior.
    """
    out = df.copy()
    eps = 1e-12

    out["phys_steady_current_energy"] = (
        out["rms_r_iq_ss"] ** 2 + out["rms_r_id_ss"] ** 2
    )

    out["phys_steady_total_energy"] = (
        out["rms_r_iq_ss"] ** 2
        + out["rms_r_id_ss"] ** 2
        + out["rms_r_om_ss"] ** 2
    )

    out["phys_transient_current_energy"] = (
        out["energy_r_iq_tr"] + out["energy_r_id_tr"]
    )

    out["phys_current_axis_imbalance"] = (
        np.abs(out["rms_r_iq_ss"] - out["rms_r_id_ss"])
        / (out["rms_r_iq_ss"] + out["rms_r_id_ss"] + eps)
    )

    out["phys_speed_residual_strength"] = (
        out["rms_r_om_ss"] + out["maxabs_r_om_ss"]
    )

    out["phys_transient_peak_strength"] = (
        out["maxabs_r_iq_tr"] + out["maxabs_r_id_tr"]
    )

    return out


def load_run(signals_path: str) -> Dict[str, np.ndarray]:
    """Load the residual channels needed by the gate layer."""
    df = pd.read_csv(
        signals_path,
        usecols=["t", "omega_ref", "r_id", "r_iq", "r_omega"],
    )

    return {
        "t": df["t"].to_numpy(),
        "omega_ref": df["omega_ref"].to_numpy(),
        "r_id": df["r_id"].to_numpy(),
        "r_iq": df["r_iq"].to_numpy(),
        "r_omega": df["r_omega"].to_numpy(),
    }


def make_group_labels(df: pd.DataFrame) -> pd.Series:
    """Create scenario-level group labels to reduce leakage between splits."""
    return (
        df["omega_step"].astype(str)
        + "_lp"
        + df["load_profile"].astype(str)
        + "_seed"
        + df["seed"].astype(str)
    )


def split_train_cal_test(
    df: pd.DataFrame,
    groups: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train, calibration, and test groups."""
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

    train_df = df.loc[train_idx].copy()
    cal_df = df.loc[cal_idx].copy()
    test_df = df.loc[test_idx].copy()

    return train_df, cal_df, test_df


def binary_metrics_no_uncertain(y_true: Sequence[str], y_pred: Sequence[str]) -> Tuple[float, float]:
    """
    Compute binary alarm metrics for hard classifiers.

    FAR is the rate of healthy samples classified as a fault. MDR is the rate of
    fault samples classified as healthy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_fault = y_true != "healthy"
    pred_fault = y_pred != "healthy"

    healthy_mask = ~true_fault
    fault_mask = true_fault

    far = np.mean(pred_fault[healthy_mask]) if np.any(healthy_mask) else np.nan
    mdr = np.mean(~pred_fault[fault_mask]) if np.any(fault_mask) else np.nan

    return float(far), float(mdr)


def multiclass_summary(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, float]:
    """Compute standard multiclass and binary alarm metrics."""
    far, mdr = binary_metrics_no_uncertain(y_true, y_pred)

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro-F1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "Macro-Precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "Macro-Recall": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "FAR": far,
        "MDR": mdr,
    }


def reliability_metrics_with_uncertain(
    y_true: Sequence[str],
    y_decision: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, float]:
    """
    Evaluate decisions that can include an uncertain state.

    Uncertain decisions are not counted as hard false alarms or hard missed
    detections. They are reported separately as inspection/uncertain outcomes.
    """
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

        hard_correct_fault = fault_decisions == true_faults
        hard_wrong_fault = (
            (fault_decisions != true_faults)
            & (fault_decisions != "healthy")
            & (fault_decisions != "uncertain")
        )
        missed = fault_decisions == "healthy"
        uncertain_fault = fault_decisions == "uncertain"

        hard_detection_rate = float(np.mean(hard_correct_fault))
        wrong_fault_class_rate = float(np.mean(hard_wrong_fault))
        missed_detection_rate = float(np.mean(missed))
        uncertain_fault_rate = float(np.mean(uncertain_fault))
    else:
        hard_detection_rate = np.nan
        wrong_fault_class_rate = np.nan
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

    correct_hard = y_decision == y_true
    uncertain = y_decision == "uncertain"
    safe_decision_rate = float(np.mean(correct_hard | uncertain))

    return {
        "False Alarm Rate": false_alarm_rate,
        "Healthy Uncertain Rate": healthy_uncertain_rate,
        "Healthy Correct Rate": healthy_correct_rate,
        "Hard Fault Detection Rate": hard_detection_rate,
        "Wrong Fault-Class Rate": wrong_fault_class_rate,
        "Missed Detection Rate": missed_detection_rate,
        "Uncertain Fault Rate": uncertain_fault_rate,
        "Overall Uncertain Rate": overall_uncertain_rate,
        "Accepted Accuracy": accepted_accuracy,
        "Accepted Macro-F1": accepted_macro_f1,
        "Safe Decision Rate": safe_decision_rate,
    }


def conformal_calibrate(
    clf: RandomForestClassifier,
    X_cal: np.ndarray,
    y_cal: Sequence[str],
    class_labels: Sequence[str],
    alpha: float = CONFORMAL_ALPHA,
) -> Dict[str, float]:
    """
    Calibrate split conformal prediction with score = 1 - P(true class).
    """
    proba = clf.predict_proba(X_cal)
    label_to_idx = {label: i for i, label in enumerate(class_labels)}

    scores = np.asarray(
        [1.0 - proba[i, label_to_idx[label]] for i, label in enumerate(y_cal)],
        dtype=float,
    )

    n_cal = len(scores)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
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
    class_labels: Sequence[str],
    qhat: float,
) -> Tuple[List[List[str]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build conformal prediction sets and singleton/uncertain decisions.
    """
    proba = clf.predict_proba(X)

    prediction_sets = []
    singleton_pred = []
    set_sizes = []
    max_probs = []

    for row in proba:
        pred_set = [
            class_label
            for j, class_label in enumerate(class_labels)
            if 1.0 - row[j] <= qhat
        ]

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
        proba,
    )


def conformal_coverage(y_true: Sequence[str], prediction_sets: Sequence[Sequence[str]]) -> float:
    """Return empirical conformal coverage."""
    return float(np.mean([label in pred_set for label, pred_set in zip(y_true, prediction_sets)]))


def build_gates(
    train_rows: pd.DataFrame,
    idx_df: pd.DataFrame,
) -> Tuple[ResidualEnergyGate, object, Dict[str, str], object]:
    """Calibrate the slow residual-energy gate and transient feature gate."""
    tag_to_path = dict(zip(idx_df["tag"].astype(str), idx_df["signals_csv"].astype(str)))

    healthy_train = train_rows[train_rows["fault_type"] == "healthy"].copy()
    healthy_paths = [tag_to_path[str(tag)] for tag in healthy_train["tag"].astype(str).tolist()]
    healthy_runs = [load_run(path) for path in healthy_paths[:30]]

    slow_calibration = calibrate_gate_from_healthy(
        healthy_runs,
        far=0.005,
        alpha=0.02,
        hysteresis_ratio=0.6,
        drop_initial_seconds=0.02,
        weights=(1.0, 1.0, 1.0),
    )

    slow_gate = ResidualEnergyGate(
        sigmas=slow_calibration.sigmas,
        T_high=slow_calibration.T_high,
        T_low=slow_calibration.T_low,
        alpha=slow_calibration.alpha,
        weights=slow_calibration.weights,
        N_on=50,
        N_off=50,
        blanking_s=0.05,
        domega_ref_threshold=1e6,
    )

    transient_calibration = calibrate_transient_gate_from_healthy(
        healthy_runs,
        t_on=0.10,
        win_s=0.02,
        far=0.01,
    )

    return slow_gate, transient_calibration, tag_to_path, slow_calibration


def gate_fault_evidence(
    row: pd.Series,
    tag_to_path: Dict[str, str],
    slow_gate: ResidualEnergyGate,
    transient_calibration,
) -> Dict[str, float | int | bool]:
    """Evaluate whether a run has residual evidence supporting a hard fault."""
    run = load_run(tag_to_path[str(row["tag"])])

    slow_output = gate_run(run, slow_gate)
    slow_open = bool(np.any(slow_output["state"] == 1))

    transient_decision, transient_score = transient_gate_decision(run, transient_calibration)
    transient_open = bool(transient_decision == 1)

    return {
        "fault_evidence": bool(slow_open or transient_open),
        "slow_open": slow_open,
        "tr_open": transient_open,
        "max_E_bar": float(np.max(slow_output["E_bar"])),
        "tr_score": float(transient_score),
    }


def apply_full_decision_v3(
    test_rows: pd.DataFrame,
    cp_pred: np.ndarray,
    prediction_sets: Sequence[Sequence[str]],
    max_probs: np.ndarray,
    tag_to_path: Dict[str, str],
    slow_gate: ResidualEnergyGate,
    transient_calibration,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Apply the final reliability-aware decision rule.

    A non-singleton conformal set is routed to uncertain. A confident healthy
    prediction remains healthy. A confident fault prediction becomes a hard
    fault only if supported by residual evidence or by a high model confidence.
    Otherwise, it is routed to uncertain/inspect.
    """
    final_decisions = []
    evidence_rows = []

    for i, (_, row) in enumerate(test_rows.iterrows()):
        pred_set = prediction_sets[i]
        pred = cp_pred[i]
        max_prob = float(max_probs[i])

        evidence = gate_fault_evidence(row, tag_to_path, slow_gate, transient_calibration)

        if len(pred_set) != 1:
            decision = "uncertain"
        elif pred == "healthy":
            decision = "healthy"
        elif evidence["fault_evidence"] or max_prob >= confidence_threshold:
            decision = pred
        else:
            decision = "uncertain"

        final_decisions.append(decision)

        evidence_rows.append(
            {
                "tag": row["tag"],
                "true_label": row["fault_type"],
                "cp_prediction": pred,
                "prediction_set": "|".join(pred_set),
                "max_probability": max_prob,
                "final_decision": decision,
                "fault_evidence": int(evidence["fault_evidence"]),
                "slow_gate_open": int(evidence["slow_open"]),
                "transient_gate_open": int(evidence["tr_open"]),
                "max_E_bar": evidence["max_E_bar"],
                "transient_score": evidence["tr_score"],
            }
        )

    return np.asarray(final_decisions), pd.DataFrame(evidence_rows)


def make_summary_rows(
    ml_summary: Dict[str, float],
    piml_summary: Dict[str, float],
    cp_reliability: Dict[str, float],
    full_reliability: Dict[str, float],
    coverage: float,
    avg_set_size: float,
) -> pd.DataFrame:
    """Build the article summary table saved by this script."""
    rows = [
        {
            "Method": "ML baseline",
            "Accuracy": ml_summary["Accuracy"],
            "Macro-F1": ml_summary["Macro-F1"],
            "Macro-Precision": ml_summary["Macro-Precision"],
            "Macro-Recall": ml_summary["Macro-Recall"],
            "False Alarm Rate": ml_summary["FAR"],
            "Missed Detection Rate": ml_summary["MDR"],
            "Coverage": np.nan,
            "Avg Set Size": np.nan,
            "Overall Uncertain Rate": np.nan,
            "Healthy Uncertain Rate": np.nan,
            "Hard Fault Detection Rate": np.nan,
            "Uncertain Fault Rate": np.nan,
            "Accepted Accuracy": np.nan,
            "Accepted Macro-F1": np.nan,
            "Safe Decision Rate": np.nan,
        },
        {
            "Method": "Physics-guided PIML",
            "Accuracy": piml_summary["Accuracy"],
            "Macro-F1": piml_summary["Macro-F1"],
            "Macro-Precision": piml_summary["Macro-Precision"],
            "Macro-Recall": piml_summary["Macro-Recall"],
            "False Alarm Rate": piml_summary["FAR"],
            "Missed Detection Rate": piml_summary["MDR"],
            "Coverage": np.nan,
            "Avg Set Size": np.nan,
            "Overall Uncertain Rate": np.nan,
            "Healthy Uncertain Rate": np.nan,
            "Hard Fault Detection Rate": np.nan,
            "Uncertain Fault Rate": np.nan,
            "Accepted Accuracy": np.nan,
            "Accepted Macro-F1": np.nan,
            "Safe Decision Rate": np.nan,
        },
        {
            "Method": "PIML + Conformal",
            "Accuracy": np.nan,
            "Macro-F1": np.nan,
            "Macro-Precision": np.nan,
            "Macro-Recall": np.nan,
            "False Alarm Rate": cp_reliability["False Alarm Rate"],
            "Missed Detection Rate": cp_reliability["Missed Detection Rate"],
            "Coverage": coverage,
            "Avg Set Size": avg_set_size,
            "Overall Uncertain Rate": cp_reliability["Overall Uncertain Rate"],
            "Healthy Uncertain Rate": cp_reliability["Healthy Uncertain Rate"],
            "Hard Fault Detection Rate": cp_reliability["Hard Fault Detection Rate"],
            "Uncertain Fault Rate": cp_reliability["Uncertain Fault Rate"],
            "Accepted Accuracy": cp_reliability["Accepted Accuracy"],
            "Accepted Macro-F1": cp_reliability["Accepted Macro-F1"],
            "Safe Decision Rate": cp_reliability["Safe Decision Rate"],
        },
        {
            "Method": "Full framework V3",
            "Accuracy": np.nan,
            "Macro-F1": np.nan,
            "Macro-Precision": np.nan,
            "Macro-Recall": np.nan,
            "False Alarm Rate": full_reliability["False Alarm Rate"],
            "Missed Detection Rate": full_reliability["Missed Detection Rate"],
            "Coverage": coverage,
            "Avg Set Size": avg_set_size,
            "Overall Uncertain Rate": full_reliability["Overall Uncertain Rate"],
            "Healthy Uncertain Rate": full_reliability["Healthy Uncertain Rate"],
            "Hard Fault Detection Rate": full_reliability["Hard Fault Detection Rate"],
            "Uncertain Fault Rate": full_reliability["Uncertain Fault Rate"],
            "Accepted Accuracy": full_reliability["Accepted Accuracy"],
            "Accepted Macro-F1": full_reliability["Accepted Macro-F1"],
            "Safe Decision Rate": full_reliability["Safe Decision Rate"],
        },
    ]

    return pd.DataFrame(rows)


def save_outputs(
    summary_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    y_test: Sequence[str],
    final_decision: Sequence[str],
    meta: Dict,
) -> None:
    """Save summary, decisions, decision breakdown, and metadata files."""
    summary_path = os.path.join(TABLE_DIR, "article_framework_summary_v3_metrics.csv")
    evidence_path = os.path.join(REPORT_DIR, "full_framework_decisions_v3_metrics.csv")
    breakdown_path = os.path.join(TABLE_DIR, "decision_breakdown_v3_metrics.csv")
    meta_path = os.path.join(REPORT_DIR, "article_framework_meta_v3_metrics.json")

    summary_df.to_csv(summary_path, index=False)
    evidence_df.to_csv(evidence_path, index=False)

    breakdown = (
        pd.DataFrame({"true_label": y_test, "final_decision": final_decision})
        .groupby(["true_label", "final_decision"])
        .size()
        .reset_index(name="count")
    )
    breakdown.to_csv(breakdown_path, index=False)

    with open(meta_path, "w") as file:
        json.dump(meta, file, indent=2)

    print("\n[OK] Saved summary:", summary_path)
    print("[OK] Saved full decisions:", evidence_path)
    print("[OK] Saved decision breakdown:", breakdown_path)
    print("[OK] Saved meta:", meta_path)

    print("\n=== FINAL SUMMARY V3 METRICS ===")
    print(summary_df.to_string(index=False))

    print("\n=== DECISION BREAKDOWN V3 ===")
    print(breakdown.to_string(index=False))


def main() -> None:
    ensure_output_dirs()

    print("[INFO] Loading data...")
    df = pd.read_csv(FEATURES_CSV)
    idx_df = pd.read_csv(RUNS_INDEX)

    df = add_physics_guided_features(df)

    class_labels = sorted(df["fault_type"].unique().tolist())
    feature_cols = BASE_FEATURE_COLS + PHYSICS_FEATURE_COLS

    print("[INFO] Classes:", class_labels)

    train_df, cal_df, test_df = split_train_cal_test(df, make_group_labels(df))

    print(
        f"[INFO] Split sizes: train={len(train_df)}, "
        f"cal={len(cal_df)}, test={len(test_df)}"
    )

    X_train_base = train_df[BASE_FEATURE_COLS].values
    X_test_base = test_df[BASE_FEATURE_COLS].values

    X_train_piml = train_df[feature_cols].values
    X_cal_piml = cal_df[feature_cols].values
    X_test_piml = test_df[feature_cols].values

    y_train = train_df["fault_type"].values
    y_cal = cal_df["fault_type"].values
    y_test = test_df["fault_type"].values

    print("\n[1] Training ML baseline...")
    ml = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        class_weight="balanced",
    )
    ml.fit(X_train_base, y_train)
    y_ml = ml.predict(X_test_base)
    ml_summary = multiclass_summary(y_test, y_ml, class_labels)

    print("[ML baseline]")
    print(ml_summary)
    print(classification_report(y_test, y_ml, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_ml, labels=class_labels))

    print("\n[2] Training physics-guided PIML model...")
    piml = RandomForestClassifier(
        n_estimators=700,
        random_state=123,
        class_weight="balanced",
    )
    piml.fit(X_train_piml, y_train)
    y_piml = piml.predict(X_test_piml)
    piml_summary = multiclass_summary(y_test, y_piml, class_labels)

    print("[Physics-guided PIML]")
    print(piml_summary)
    print(classification_report(y_test, y_piml, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_piml, labels=class_labels))

    print("\n[3] Calibrating conformal prediction...")
    conformal = conformal_calibrate(
        piml,
        X_cal_piml,
        y_cal,
        class_labels,
        alpha=CONFORMAL_ALPHA,
    )

    prediction_sets, cp_pred, set_sizes, max_probs, _ = conformal_predict(
        piml,
        X_test_piml,
        class_labels,
        conformal["qhat"],
    )

    coverage = conformal_coverage(y_test, prediction_sets)
    avg_set_size = float(np.mean(set_sizes))
    uncertain_rate_cp = float(np.mean(cp_pred == "uncertain"))

    cp_reliability = reliability_metrics_with_uncertain(
        y_test,
        cp_pred,
        class_labels,
    )

    print("[PIML + Conformal]")
    print("qhat:", conformal["qhat"])
    print("coverage:", coverage)
    print("avg_set_size:", avg_set_size)
    print("uncertain_rate:", uncertain_rate_cp)
    print(cp_reliability)

    print("\n[4] Applying full framework...")
    slow_gate, transient_calibration, tag_to_path, slow_calibration = build_gates(
        train_df,
        idx_df,
    )

    final_decision, evidence_df = apply_full_decision_v3(
        test_df,
        cp_pred,
        prediction_sets,
        max_probs,
        tag_to_path,
        slow_gate,
        transient_calibration,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    full_reliability = reliability_metrics_with_uncertain(
        y_test,
        final_decision,
        class_labels,
    )

    print("[Full framework V3]")
    print(full_reliability)

    summary_df = make_summary_rows(
        ml_summary=ml_summary,
        piml_summary=piml_summary,
        cp_reliability=cp_reliability,
        full_reliability=full_reliability,
        coverage=coverage,
        avg_set_size=avg_set_size,
    )

    meta = {
        "features_csv": FEATURES_CSV,
        "runs_index": RUNS_INDEX,
        "classes": class_labels,
        "split_sizes": {
            "train": int(len(train_df)),
            "calibration": int(len(cal_df)),
            "test": int(len(test_df)),
        },
        "conformal": conformal,
        "coverage": coverage,
        "avg_set_size": avg_set_size,
        "uncertain_rate_cp": uncertain_rate_cp,
        "confidence_threshold_v3": CONFIDENCE_THRESHOLD,
        "gate_calibration": {
            "slow_gate": slow_calibration.meta,
            "transient_gate": transient_calibration.meta,
        },
    }

    save_outputs(
        summary_df=summary_df,
        evidence_df=evidence_df,
        y_test=y_test,
        final_decision=final_decision,
        meta=meta,
    )


if __name__ == "__main__":
    main()
