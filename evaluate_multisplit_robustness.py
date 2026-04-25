"""
Multi-split robustness evaluation for the PMSM digital-twin diagnosis framework.

The script repeats the reliability-aware evaluation over several scenario-aware
train/calibration/test splits. It is used to check whether the full framework
keeps similar false-alarm, uncertainty, coverage and accepted-accuracy behavior
when the data partition changes.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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

SPLIT_SEEDS = [0, 1, 2, 3, 4]
CONFIDENCE_THRESHOLD = 0.80
CONFORMAL_ALPHA = 0.10

BASE_FEATURE_COLS = [
    "rms_r_iq_ss", "maxabs_r_iq_ss", "mean_r_iq_ss", "std_r_iq_ss",
    "rms_r_id_ss", "maxabs_r_id_ss", "mean_r_id_ss", "std_r_id_ss",
    "rms_r_om_ss", "maxabs_r_om_ss",
    "rms_r_iq_tr", "maxabs_r_iq_tr", "energy_r_iq_tr",
    "fdom_r_iq_tr", "fratio_r_iq_tr",
    "rms_r_id_tr", "maxabs_r_id_tr", "energy_r_id_tr",
    "fdom_r_id_tr", "fratio_r_id_tr",
]

PHYSICS_FEATURE_COLS = [
    "phys_steady_current_energy",
    "phys_steady_total_energy",
    "phys_transient_current_energy",
    "phys_current_axis_imbalance",
    "phys_speed_residual_strength",
    "phys_transient_peak_strength",
]

ROBUSTNESS_METRICS = [
    "baseline_accuracy",
    "baseline_macro_f1",
    "baseline_far",
    "baseline_mdr",
    "coverage",
    "avg_set_size",
    "full_far",
    "full_mdr",
    "full_overall_uncertain",
    "full_healthy_uncertain",
    "full_hard_fault_detection",
    "full_uncertain_fault_rate",
    "full_accepted_accuracy",
    "full_safe_decision_rate",
]


def ensure_output_dirs() -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


def add_physics_guided_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add residual-energy features used by the physics-guided classifier."""
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
    """Load the signal channels required by the gate layer."""
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


def scenario_groups(df: pd.DataFrame) -> pd.Series:
    """Return scenario group labels used to avoid train/test leakage."""
    return (
        df["omega_step"].astype(str)
        + "_lp" + df["load_profile"].astype(str)
        + "_seed" + df["seed"].astype(str)
    )


def split_train_cal_test(
    df: pd.DataFrame,
    split_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a 60/20/20 scenario-aware split for one robustness seed."""
    groups = scenario_groups(df)

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.40,
        random_state=split_seed,
    )
    train_idx, temp_idx = next(first_split.split(df, df["fault_type"], groups=groups))

    temp_df = df.iloc[temp_idx].copy()
    temp_groups = groups.iloc[temp_idx]

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=split_seed + 100,
    )
    cal_rel_idx, test_rel_idx = next(
        second_split.split(temp_df, temp_df["fault_type"], groups=temp_groups)
    )

    cal_idx = temp_df.iloc[cal_rel_idx].index.to_numpy()
    test_idx = temp_df.iloc[test_rel_idx].index.to_numpy()

    return df.loc[train_idx].copy(), df.loc[cal_idx].copy(), df.loc[test_idx].copy()


def binary_metrics_no_uncertain(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute false-alarm and missed-detection rates for hard decisions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_fault = y_true != "healthy"
    pred_fault = y_pred != "healthy"

    healthy_mask = ~true_fault
    fault_mask = true_fault

    far = float(np.mean(pred_fault[healthy_mask])) if np.any(healthy_mask) else np.nan
    mdr = float(np.mean(~pred_fault[fault_mask])) if np.any(fault_mask) else np.nan

    return far, mdr


def multiclass_summary(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, float]:
    """Summarize hard-classifier performance."""
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
    y_true: np.ndarray,
    y_decision: np.ndarray,
    labels: List[str],
) -> Dict[str, float]:
    """Compute reliability metrics for decisions that may include 'uncertain'."""
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
        wrong_fault_class_rate = float(np.mean(wrong_fault))
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

    safe_decision_rate = float(np.mean((y_decision == y_true) | (y_decision == "uncertain")))

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
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    class_labels: List[str],
    alpha: float = CONFORMAL_ALPHA,
) -> Dict[str, float]:
    """Calibrate split-conformal prediction using score = 1 - P(true class)."""
    proba = clf.predict_proba(x_cal)
    label_to_idx = {label: idx for idx, label in enumerate(class_labels)}

    scores = np.asarray(
        [1.0 - proba[i, label_to_idx[label]] for i, label in enumerate(y_cal)],
        dtype=float,
    )

    n = len(scores)
    q_level = min(float(np.ceil((n + 1) * (1 - alpha)) / n), 1.0)
    qhat = float(np.quantile(scores, q_level, method="higher"))

    return {
        "alpha": float(alpha),
        "qhat": qhat,
        "n_cal": int(n),
        "q_level": q_level,
    }


def conformal_predict(
    clf: RandomForestClassifier,
    x: np.ndarray,
    class_labels: List[str],
    qhat: float,
) -> Tuple[List[List[str]], np.ndarray, np.ndarray, np.ndarray]:
    """Return conformal prediction sets and singleton/uncertain decisions."""
    proba = clf.predict_proba(x)

    prediction_sets = []
    decisions = []
    set_sizes = []
    max_probs = []

    for row in proba:
        pred_set = [
            label
            for idx, label in enumerate(class_labels)
            if 1.0 - row[idx] <= qhat
        ]

        if not pred_set:
            pred_set = [class_labels[int(np.argmax(row))]]

        prediction_sets.append(pred_set)
        set_sizes.append(len(pred_set))
        max_probs.append(float(np.max(row)))
        decisions.append(pred_set[0] if len(pred_set) == 1 else "uncertain")

    return (
        prediction_sets,
        np.asarray(decisions),
        np.asarray(set_sizes),
        np.asarray(max_probs),
    )


def conformal_coverage(y_true: np.ndarray, prediction_sets: List[List[str]]) -> float:
    """Compute empirical conformal coverage."""
    return float(np.mean([label in pred_set for label, pred_set in zip(y_true, prediction_sets)]))


def build_gates(
    train_rows: pd.DataFrame,
    idx_df: pd.DataFrame,
) -> Tuple[ResidualEnergyGate, object, Dict[str, str], object]:
    """Calibrate the slow residual-energy gate and the transient feature gate."""
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
) -> Dict[str, float]:
    """Evaluate slow and transient residual evidence for one run."""
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
    cp_decisions: np.ndarray,
    prediction_sets: List[List[str]],
    max_probs: np.ndarray,
    tag_to_path: Dict[str, str],
    slow_gate: ResidualEnergyGate,
    transient_cal,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Apply the reliability-aware full decision rule."""
    final_decisions = []
    evidence_rows = []

    for i, (_, row) in enumerate(test_rows.iterrows()):
        pred_set = prediction_sets[i]
        pred = cp_decisions[i]
        max_prob = float(max_probs[i])

        evidence = gate_fault_evidence(row, tag_to_path, slow_gate, transient_cal)
        fault_evidence = evidence["fault_evidence"]

        if len(pred_set) != 1:
            decision = "uncertain"
        elif pred == "healthy":
            decision = "healthy"
        elif fault_evidence or max_prob >= confidence_threshold:
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
                "fault_evidence": int(fault_evidence),
                "slow_gate_open": int(evidence["slow_open"]),
                "transient_gate_open": int(evidence["tr_open"]),
                "max_E_bar": evidence["max_E_bar"],
                "transient_score": evidence["tr_score"],
            }
        )

    return np.asarray(final_decisions), pd.DataFrame(evidence_rows)


def evaluate_one_split(
    df: pd.DataFrame,
    idx_df: pd.DataFrame,
    split_seed: int,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Run one train/calibration/test split and return raw robustness outputs."""
    class_labels = sorted(df["fault_type"].unique().tolist())
    feature_cols = BASE_FEATURE_COLS + PHYSICS_FEATURE_COLS

    train_df, cal_df, test_df = split_train_cal_test(df, split_seed)

    x_train = train_df[feature_cols].values
    x_cal = cal_df[feature_cols].values
    x_test = test_df[feature_cols].values

    y_train = train_df["fault_type"].values
    y_cal = cal_df["fault_type"].values
    y_test = test_df["fault_type"].values

    clf = RandomForestClassifier(
        n_estimators=700,
        random_state=123 + split_seed,
        class_weight="balanced",
    )
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    baseline = multiclass_summary(y_test, y_pred, class_labels)

    conformal = conformal_calibrate(
        clf,
        x_cal,
        y_cal,
        class_labels,
        alpha=CONFORMAL_ALPHA,
    )
    prediction_sets, cp_decisions, set_sizes, max_probs = conformal_predict(
        clf,
        x_test,
        class_labels,
        conformal["qhat"],
    )

    coverage = conformal_coverage(y_test, prediction_sets)
    avg_set_size = float(np.mean(set_sizes))

    slow_gate, transient_cal, tag_to_path, _ = build_gates(train_df, idx_df)

    final_decision, evidence_df = apply_full_decision(
        test_df,
        cp_decisions,
        prediction_sets,
        max_probs,
        tag_to_path,
        slow_gate,
        transient_cal,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    full = reliability_metrics_with_uncertain(y_test, final_decision, class_labels)

    breakdown = (
        pd.DataFrame({"true_label": y_test, "final_decision": final_decision})
        .groupby(["true_label", "final_decision"])
        .size()
        .reset_index(name="count")
    )
    breakdown["split_seed"] = split_seed

    raw_row = {
        "split_seed": split_seed,
        "n_train": int(len(train_df)),
        "n_calibration": int(len(cal_df)),
        "n_test": int(len(test_df)),
        "baseline_accuracy": baseline["Accuracy"],
        "baseline_macro_f1": baseline["Macro-F1"],
        "baseline_far": baseline["FAR"],
        "baseline_mdr": baseline["MDR"],
        "coverage": coverage,
        "avg_set_size": avg_set_size,
        "qhat": conformal["qhat"],
        "full_far": full["False Alarm Rate"],
        "full_mdr": full["Missed Detection Rate"],
        "full_overall_uncertain": full["Overall Uncertain Rate"],
        "full_healthy_uncertain": full["Healthy Uncertain Rate"],
        "full_hard_fault_detection": full["Hard Fault Detection Rate"],
        "full_uncertain_fault_rate": full["Uncertain Fault Rate"],
        "full_accepted_accuracy": full["Accepted Accuracy"],
        "full_safe_decision_rate": full["Safe Decision Rate"],
    }

    return raw_row, breakdown, evidence_df


def summarize_mean_std(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize robustness metrics across split seeds."""
    rows = []

    for metric in ROBUSTNESS_METRICS:
        rows.append(
            {
                "metric": metric,
                "mean": float(raw_df[metric].mean()),
                "std": float(raw_df[metric].std(ddof=1)),
                "min": float(raw_df[metric].min()),
                "max": float(raw_df[metric].max()),
            }
        )

    return pd.DataFrame(rows)


def save_results(
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    breakdowns: List[pd.DataFrame],
    evidences: List[pd.DataFrame],
) -> None:
    """Save raw, summary and per-split decision outputs."""
    raw_path = os.path.join(TABLE_DIR, "multisplit_robustness_raw.csv")
    summary_path = os.path.join(TABLE_DIR, "multisplit_robustness_summary.csv")
    breakdown_path = os.path.join(TABLE_DIR, "multisplit_decision_breakdown.csv")
    evidence_path = os.path.join(REPORT_DIR, "multisplit_full_decisions.csv")

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pd.concat(breakdowns, ignore_index=True).to_csv(breakdown_path, index=False)
    pd.concat(evidences, ignore_index=True).to_csv(evidence_path, index=False)

    print("[OK] Saved raw:", raw_path)
    print("[OK] Saved summary:", summary_path)
    print("[OK] Saved breakdown:", breakdown_path)
    print("[OK] Saved evidence:", evidence_path)


def plot_robustness_summary(summary_df: pd.DataFrame) -> None:
    """Plot mean ± standard deviation for the main robustness metrics."""
    wanted = [
        ("coverage", "Coverage"),
        ("full_far", "FAR"),
        ("full_mdr", "MDR"),
        ("full_overall_uncertain", "Uncertain"),
        ("full_hard_fault_detection", "Hard detect"),
        ("full_accepted_accuracy", "Accepted acc."),
        ("full_safe_decision_rate", "Safe decision"),
    ]

    rows = []
    for metric, label in wanted:
        match = summary_df[summary_df["metric"] == metric]
        if len(match) == 0:
            continue
        rows.append(
            {
                "label": label,
                "mean": float(match["mean"].iloc[0]),
                "std": float(match["std"].iloc[0]),
            }
        )

    plot_df = pd.DataFrame(rows)
    x = np.arange(len(plot_df))

    plt.figure()
    plt.bar(x, plot_df["mean"], yerr=plot_df["std"], capsize=4)
    plt.xticks(x, plot_df["label"], rotation=25, ha="right")
    plt.ylabel("Rate")
    plt.title("Multi-Split Robustness of Full Framework")
    plt.ylim(0, 1.10)
    plt.grid(True, axis="y")
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "multisplit_robustness_errorbars.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("[OK] Saved figure:", fig_path)


def save_meta() -> None:
    """Save experiment metadata."""
    meta = {
        "features_csv": FEATURES_CSV,
        "runs_index": RUNS_INDEX,
        "split_seeds": SPLIT_SEEDS,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "conformal_alpha": CONFORMAL_ALPHA,
        "description": "Five-split robustness check using different train/calibration/test partitions.",
    }

    meta_path = os.path.join(REPORT_DIR, "multisplit_robustness_meta.json")
    with open(meta_path, "w") as file:
        json.dump(meta, file, indent=2)

    print("[OK] Saved meta:", meta_path)


def main() -> None:
    ensure_output_dirs()

    print("[INFO] Loading data...")
    df = pd.read_csv(FEATURES_CSV)
    idx_df = pd.read_csv(RUNS_INDEX)

    df = add_physics_guided_features(df)

    raw_rows = []
    breakdowns = []
    evidences = []

    for seed in SPLIT_SEEDS:
        print(f"\n[INFO] Evaluating split seed {seed}...")
        raw_row, breakdown, evidence = evaluate_one_split(df, idx_df, seed)

        raw_rows.append(raw_row)
        breakdowns.append(breakdown)

        evidence = evidence.copy()
        evidence["split_seed"] = seed
        evidences.append(evidence)

        print(
            f"seed={seed} | "
            f"baseline_acc={raw_row['baseline_accuracy']:.3f}, "
            f"baseline_FAR={raw_row['baseline_far']:.3f}, "
            f"coverage={raw_row['coverage']:.3f}, "
            f"full_FAR={raw_row['full_far']:.3f}, "
            f"full_uncertain={raw_row['full_overall_uncertain']:.3f}, "
            f"accepted_acc={raw_row['full_accepted_accuracy']:.3f}"
        )

    raw_df = pd.DataFrame(raw_rows)
    summary_df = summarize_mean_std(raw_df)

    print("\n=== Multi-Split Robustness Raw Results ===")
    print(raw_df.to_string(index=False))

    print("\n=== Multi-Split Robustness Summary ===")
    print(summary_df.to_string(index=False))

    save_results(raw_df, summary_df, breakdowns, evidences)
    plot_robustness_summary(summary_df)
    save_meta()


if __name__ == "__main__":
    main()
