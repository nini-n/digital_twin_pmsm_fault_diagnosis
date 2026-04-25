"""
Domain generalization evaluation for the PMSM digital-twin diagnosis framework.

The experiment trains and calibrates the models on seen speed conditions
(omega_step = 40 and 50) and evaluates them on an unseen speed condition
(omega_step = 60). The goal is to check whether the reliability-aware decision
logic keeps the same false-alarm and inspect-state behavior outside the training
speed domain.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
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
FIG_DIR = os.path.join(OUT_DIR, "figures")
REPORT_DIR = os.path.join(OUT_DIR, "reports")

SEEN_SPEEDS = [40.0, 50.0]
UNSEEN_SPEED = 60.0
CONFORMAL_ALPHA = 0.10
CONFIDENCE_THRESHOLD = 0.80

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
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


def add_physics_guided_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add residual-energy features used by the physics-guided model."""
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
    """Load the residual signals needed by the gate logic."""
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


def binary_metrics_no_uncertain(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute false-alarm and missed-detection rates for hard decisions."""
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
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
) -> Dict[str, float]:
    """Summarize standard multiclass and binary alarm metrics."""
    far, mdr = binary_metrics_no_uncertain(y_true, y_pred)

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro-F1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "Macro-Precision": float(
            precision_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            )
        ),
        "Macro-Recall": float(
            recall_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            )
        ),
        "FAR": far,
        "MDR": mdr,
    }


def reliability_metrics_with_uncertain(
    y_true: np.ndarray,
    y_decision: np.ndarray,
    labels: List[str],
) -> Dict[str, float]:
    """
    Compute reliability metrics for decisions that may include an inspect state.

    Uncertain fault cases are reported as uncertain/inspect rather than being
    counted as missed detections.
    """
    y_true = np.asarray(y_true)
    y_decision = np.asarray(y_decision)

    healthy_mask = y_true == "healthy"
    fault_mask = y_true != "healthy"

    if np.any(healthy_mask):
        healthy_decisions = y_decision[healthy_mask]
        false_alarm_rate = float(
            np.mean(
                (healthy_decisions != "healthy")
                & (healthy_decisions != "uncertain")
            )
        )
        healthy_uncertain_rate = float(np.mean(healthy_decisions == "uncertain"))
        healthy_correct_rate = float(np.mean(healthy_decisions == "healthy"))
    else:
        false_alarm_rate = np.nan
        healthy_uncertain_rate = np.nan
        healthy_correct_rate = np.nan

    if np.any(fault_mask):
        true_faults = y_true[fault_mask]
        decisions = y_decision[fault_mask]

        hard_correct = decisions == true_faults
        wrong_fault = (
            (decisions != true_faults)
            & (decisions != "healthy")
            & (decisions != "uncertain")
        )
        missed = decisions == "healthy"
        uncertain_fault = decisions == "uncertain"

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
        accepted_accuracy = float(
            accuracy_score(y_true[accepted_mask], y_decision[accepted_mask])
        )
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
        "Wrong Fault-Class Rate": wrong_fault_rate,
        "Missed Detection Rate": missed_detection_rate,
        "Uncertain Fault Rate": uncertain_fault_rate,
        "Overall Uncertain Rate": overall_uncertain_rate,
        "Accepted Accuracy": accepted_accuracy,
        "Accepted Macro-F1": accepted_macro_f1,
        "Safe Decision Rate": safe_decision_rate,
    }


def conformal_calibrate(
    clf: RandomForestClassifier,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    class_labels: List[str],
    alpha: float = CONFORMAL_ALPHA,
) -> Dict[str, float]:
    """Calibrate split conformal prediction using score = 1 - P(true class)."""
    probabilities = clf.predict_proba(x_cal)
    label_to_idx = {label: idx for idx, label in enumerate(class_labels)}

    scores = np.asarray(
        [1.0 - probabilities[i, label_to_idx[label]] for i, label in enumerate(y_cal)],
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
    x: np.ndarray,
    class_labels: List[str],
    qhat: float,
) -> Tuple[List[List[str]], np.ndarray, np.ndarray, np.ndarray]:
    """Return conformal prediction sets and singleton/uncertain decisions."""
    probabilities = clf.predict_proba(x)

    prediction_sets = []
    singleton_predictions = []
    set_sizes = []
    max_probabilities = []

    for row in probabilities:
        pred_set = [
            label for idx, label in enumerate(class_labels) if (1.0 - row[idx]) <= qhat
        ]

        if not pred_set:
            pred_set = [class_labels[int(np.argmax(row))]]

        prediction_sets.append(pred_set)
        set_sizes.append(len(pred_set))
        max_probabilities.append(float(np.max(row)))

        if len(pred_set) == 1:
            singleton_predictions.append(pred_set[0])
        else:
            singleton_predictions.append("uncertain")

    return (
        prediction_sets,
        np.asarray(singleton_predictions),
        np.asarray(set_sizes),
        np.asarray(max_probabilities),
    )


def conformal_coverage(y_true: np.ndarray, prediction_sets: List[List[str]]) -> float:
    """Compute empirical coverage of the conformal prediction sets."""
    return float(np.mean([label in pred_set for label, pred_set in zip(y_true, prediction_sets)]))


def split_seen_unseen_speed(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the seen-speed train/calibration sets and the unseen-speed test set."""
    traincal_df = df[df["omega_step"].isin(SEEN_SPEEDS)].copy()
    test_df = df[df["omega_step"] == UNSEEN_SPEED].copy()

    if len(traincal_df) == 0 or len(test_df) == 0:
        raise RuntimeError("Domain split failed. Check omega_step values in the feature CSV.")

    groups = (
        traincal_df["load_profile"].astype(str)
        + "_seed"
        + traincal_df["seed"].astype(str)
    )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, cal_idx = next(
        splitter.split(traincal_df, traincal_df["fault_type"], groups=groups)
    )

    train_df = traincal_df.iloc[train_idx].copy()
    cal_df = traincal_df.iloc[cal_idx].copy()

    return train_df, cal_df, test_df


def build_gates(train_rows: pd.DataFrame, idx_df: pd.DataFrame):
    """Calibrate the slow residual-energy gate and the transient feature gate."""
    tag_to_path = dict(zip(idx_df["tag"].astype(str), idx_df["signals_csv"].astype(str)))

    healthy_train = train_rows[train_rows["fault_type"] == "healthy"].copy()
    healthy_paths = [tag_to_path[str(tag)] for tag in healthy_train["tag"].astype(str)]
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
) -> Dict[str, float]:
    """Evaluate residual evidence for one run."""
    run = load_run(tag_to_path[str(row["tag"])])

    slow_out = gate_run(run, slow_gate)
    slow_open = bool(np.any(slow_out["state"] == 1))

    transient_decision, transient_score = transient_gate_decision(run, transient_cal)
    transient_open = bool(transient_decision == 1)

    return {
        "fault_evidence": bool(slow_open or transient_open),
        "slow_open": slow_open,
        "tr_open": transient_open,
        "max_E_bar": float(np.max(slow_out["E_bar"])),
        "tr_score": float(transient_score),
    }


def apply_full_decision(
    test_rows: pd.DataFrame,
    cp_pred: np.ndarray,
    prediction_sets: List[List[str]],
    max_probs: np.ndarray,
    tag_to_path: Dict[str, str],
    slow_gate: ResidualEnergyGate,
    transient_cal,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Apply the full reliability-aware decision rule to the unseen-speed set."""
    decisions = []
    evidence_rows = []

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
        evidence_rows.append(
            {
                "tag": row["tag"],
                "true_label": row["fault_type"],
                "omega_step": row["omega_step"],
                "load_profile": row["load_profile"],
                "seed": row["seed"],
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

    return np.asarray(decisions), pd.DataFrame(evidence_rows)


def make_summary_table(
    ml_summary: Dict[str, float],
    piml_summary: Dict[str, float],
    cp_metrics: Dict[str, float],
    full_metrics: Dict[str, float],
    coverage: float,
    avg_set_size: float,
) -> pd.DataFrame:
    """Build the domain-generalization summary table."""
    rows = [
        {
            "Method": "ML baseline",
            "Accuracy": ml_summary["Accuracy"],
            "Macro-F1": ml_summary["Macro-F1"],
            "False Alarm Rate": ml_summary["FAR"],
            "Missed Detection Rate": ml_summary["MDR"],
            "Coverage": np.nan,
            "Avg Set Size": np.nan,
            "Overall Uncertain Rate": np.nan,
            "Hard Fault Detection Rate": np.nan,
            "Uncertain Fault Rate": np.nan,
            "Accepted Accuracy": np.nan,
            "Safe Decision Rate": np.nan,
        },
        {
            "Method": "Physics-guided PIML",
            "Accuracy": piml_summary["Accuracy"],
            "Macro-F1": piml_summary["Macro-F1"],
            "False Alarm Rate": piml_summary["FAR"],
            "Missed Detection Rate": piml_summary["MDR"],
            "Coverage": np.nan,
            "Avg Set Size": np.nan,
            "Overall Uncertain Rate": np.nan,
            "Hard Fault Detection Rate": np.nan,
            "Uncertain Fault Rate": np.nan,
            "Accepted Accuracy": np.nan,
            "Safe Decision Rate": np.nan,
        },
        {
            "Method": "PIML + Conformal",
            "Accuracy": np.nan,
            "Macro-F1": np.nan,
            "False Alarm Rate": cp_metrics["False Alarm Rate"],
            "Missed Detection Rate": cp_metrics["Missed Detection Rate"],
            "Coverage": coverage,
            "Avg Set Size": avg_set_size,
            "Overall Uncertain Rate": cp_metrics["Overall Uncertain Rate"],
            "Hard Fault Detection Rate": cp_metrics["Hard Fault Detection Rate"],
            "Uncertain Fault Rate": cp_metrics["Uncertain Fault Rate"],
            "Accepted Accuracy": cp_metrics["Accepted Accuracy"],
            "Safe Decision Rate": cp_metrics["Safe Decision Rate"],
        },
        {
            "Method": "Full framework",
            "Accuracy": np.nan,
            "Macro-F1": np.nan,
            "False Alarm Rate": full_metrics["False Alarm Rate"],
            "Missed Detection Rate": full_metrics["Missed Detection Rate"],
            "Coverage": coverage,
            "Avg Set Size": avg_set_size,
            "Overall Uncertain Rate": full_metrics["Overall Uncertain Rate"],
            "Hard Fault Detection Rate": full_metrics["Hard Fault Detection Rate"],
            "Uncertain Fault Rate": full_metrics["Uncertain Fault Rate"],
            "Accepted Accuracy": full_metrics["Accepted Accuracy"],
            "Safe Decision Rate": full_metrics["Safe Decision Rate"],
        },
    ]

    return pd.DataFrame(rows)


def plot_domain_generalization(summary_df: pd.DataFrame) -> str:
    """Save a compact plot of the main unseen-speed reliability metrics."""
    metrics = [
        "False Alarm Rate",
        "Missed Detection Rate",
        "Overall Uncertain Rate",
        "Hard Fault Detection Rate",
    ]

    plt.figure()
    for metric in metrics:
        plt.plot(summary_df["Method"], summary_df[metric], marker="o", label=metric)

    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Rate")
    plt.title("Domain Generalization on Unseen Speed Condition")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "domain_generalization_unseen_speed.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return fig_path


def save_outputs(
    summary_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    y_test: np.ndarray,
    final_decision: np.ndarray,
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    conformal_meta: Dict,
    coverage: float,
    avg_set_size: float,
    slow_cal,
    transient_cal,
) -> None:
    """Save summary tables, decision details, figure and metadata."""
    summary_path = os.path.join(TABLE_DIR, "domain_generalization_unseen_speed_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    evidence_path = os.path.join(REPORT_DIR, "domain_generalization_unseen_speed_decisions.csv")
    evidence_df.to_csv(evidence_path, index=False)

    breakdown = (
        pd.DataFrame({"true_label": y_test, "final_decision": final_decision})
        .groupby(["true_label", "final_decision"])
        .size()
        .reset_index(name="count")
    )

    breakdown_path = os.path.join(TABLE_DIR, "domain_generalization_unseen_speed_breakdown.csv")
    breakdown.to_csv(breakdown_path, index=False)

    fig_path = plot_domain_generalization(summary_df)

    meta = {
        "features_csv": FEATURES_CSV,
        "runs_index": RUNS_INDEX,
        "train_omega_steps": sorted([float(x) for x in train_df["omega_step"].unique()]),
        "cal_omega_steps": sorted([float(x) for x in cal_df["omega_step"].unique()]),
        "test_omega_steps": sorted([float(x) for x in test_df["omega_step"].unique()]),
        "split_sizes": {
            "train": int(len(train_df)),
            "calibration": int(len(cal_df)),
            "test_unseen_speed": int(len(test_df)),
        },
        "conformal": conformal_meta,
        "coverage": coverage,
        "avg_set_size": avg_set_size,
        "gate_calibration": {
            "slow_gate": slow_cal.meta,
            "transient_gate": transient_cal.meta,
        },
    }

    meta_path = os.path.join(REPORT_DIR, "domain_generalization_unseen_speed_meta.json")
    with open(meta_path, "w") as file:
        json.dump(meta, file, indent=2)

    print("\n=== Domain Generalization Summary: Unseen Speed ===")
    print(summary_df.to_string(index=False))

    print("\n=== Domain Generalization Decision Breakdown ===")
    print(breakdown.to_string(index=False))

    print("[OK] Saved summary:", summary_path)
    print("[OK] Saved decisions:", evidence_path)
    print("[OK] Saved breakdown:", breakdown_path)
    print("[OK] Saved figure:", fig_path)
    print("[OK] Saved meta:", meta_path)


def main() -> None:
    ensure_output_dirs()

    print("[INFO] Loading data...")
    df = pd.read_csv(FEATURES_CSV)
    idx_df = pd.read_csv(RUNS_INDEX)
    df = add_physics_guided_features(df)

    class_labels = sorted(df["fault_type"].unique().tolist())
    feature_cols = BASE_FEATURE_COLS + PHYSICS_FEATURE_COLS

    train_df, cal_df, test_df = split_seen_unseen_speed(df)

    print(
        f"[INFO] Domain split sizes: train={len(train_df)}, "
        f"cal={len(cal_df)}, test_unseen_speed={len(test_df)}"
    )
    print("[INFO] Train omega steps:", sorted(train_df["omega_step"].unique()))
    print("[INFO] Cal omega steps:", sorted(cal_df["omega_step"].unique()))
    print("[INFO] Test omega steps:", sorted(test_df["omega_step"].unique()))

    x_train_base = train_df[BASE_FEATURE_COLS].values
    x_test_base = test_df[BASE_FEATURE_COLS].values

    x_train_piml = train_df[feature_cols].values
    x_cal_piml = cal_df[feature_cols].values
    x_test_piml = test_df[feature_cols].values

    y_train = train_df["fault_type"].values
    y_cal = cal_df["fault_type"].values
    y_test = test_df["fault_type"].values

    print("\n[1] Training ML baseline on seen speed conditions...")
    ml = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        class_weight="balanced",
    )
    ml.fit(x_train_base, y_train)
    y_ml = ml.predict(x_test_base)
    ml_summary = multiclass_summary(y_test, y_ml, class_labels)

    print("[ML baseline - unseen speed]")
    print(ml_summary)
    print(classification_report(y_test, y_ml, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_ml, labels=class_labels))

    print("\n[2] Training physics-guided PIML on seen speed conditions...")
    piml = RandomForestClassifier(
        n_estimators=700,
        random_state=123,
        class_weight="balanced",
    )
    piml.fit(x_train_piml, y_train)
    y_piml = piml.predict(x_test_piml)
    piml_summary = multiclass_summary(y_test, y_piml, class_labels)

    print("[Physics-guided PIML - unseen speed]")
    print(piml_summary)
    print(classification_report(y_test, y_piml, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_piml, labels=class_labels))

    print("\n[3] Calibrating conformal prediction on seen speed conditions...")
    conformal_meta = conformal_calibrate(
        piml,
        x_cal_piml,
        y_cal,
        class_labels,
        alpha=CONFORMAL_ALPHA,
    )

    prediction_sets, cp_pred, set_sizes, max_probs = conformal_predict(
        piml,
        x_test_piml,
        class_labels,
        conformal_meta["qhat"],
    )

    coverage = conformal_coverage(y_test, prediction_sets)
    avg_set_size = float(np.mean(set_sizes))
    cp_metrics = reliability_metrics_with_uncertain(y_test, cp_pred, class_labels)

    print("[PIML + Conformal - unseen speed]")
    print("qhat:", conformal_meta["qhat"])
    print("coverage:", coverage)
    print("avg_set_size:", avg_set_size)
    print(cp_metrics)

    print("\n[4] Applying full framework on unseen speed condition...")
    slow_gate, transient_cal, tag_to_path, slow_cal = build_gates(train_df, idx_df)

    final_decision, evidence_df = apply_full_decision(
        test_df,
        cp_pred,
        prediction_sets,
        max_probs,
        tag_to_path,
        slow_gate,
        transient_cal,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    full_metrics = reliability_metrics_with_uncertain(
        y_test,
        final_decision,
        class_labels,
    )

    print("[Full framework - unseen speed]")
    print(full_metrics)

    summary_df = make_summary_table(
        ml_summary,
        piml_summary,
        cp_metrics,
        full_metrics,
        coverage,
        avg_set_size,
    )

    save_outputs(
        summary_df,
        evidence_df,
        y_test,
        final_decision,
        train_df,
        cal_df,
        test_df,
        conformal_meta,
        coverage,
        avg_set_size,
        slow_cal,
        transient_cal,
    )


if __name__ == "__main__":
    main()
