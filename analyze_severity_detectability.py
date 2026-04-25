"""
Severity and detectability analysis for the reliability-aware framework.

This script combines the final framework decisions with the extracted residual
features. It summarizes how each fault type is handled across severity levels
and produces tables and figures for hard detections, inspect decisions, missed
detections, and residual-evidence trends.
"""

from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DECISIONS_CSV = "paper_results/reports/full_framework_decisions_v3_metrics.csv"
FEATURES_CSV = "data/features_20260301_204447.csv"

OUT_DIR = "paper_results"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TABLE_DIR = os.path.join(OUT_DIR, "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


def classify_decision(row: pd.Series) -> str:
    """Assign a deployment-oriented decision category to one evaluated sample."""
    true_label = row["fault_type"]
    decision = row["final_decision"]

    if true_label == "healthy":
        if decision == "healthy":
            return "healthy_correct"
        if decision == "uncertain":
            return "healthy_uncertain"
        return "false_alarm"

    if decision == true_label:
        return "hard_detected"
    if decision == "uncertain":
        return "uncertain_inspect"
    if decision == "healthy":
        return "missed"

    return "wrong_fault_class"


def add_residual_energy_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add compact residual-energy indicators used in detectability analysis."""
    out = df.copy()

    out["severity_abs"] = np.abs(out["severity"].astype(float))
    out["steady_current_energy"] = (
        out["rms_r_iq_ss"] ** 2 + out["rms_r_id_ss"] ** 2
    )
    out["steady_total_energy"] = (
        out["rms_r_iq_ss"] ** 2
        + out["rms_r_id_ss"] ** 2
        + out["rms_r_om_ss"] ** 2
    )
    out["transient_current_energy"] = (
        out["energy_r_iq_tr"] + out["energy_r_id_tr"]
    )

    return out


def load_analysis_data() -> pd.DataFrame:
    """Load feature rows and final framework decisions, then merge them by tag."""
    decisions = pd.read_csv(DECISIONS_CSV)
    features = pd.read_csv(FEATURES_CSV)

    df = features.merge(decisions, on="tag", how="inner")
    if df.empty:
        raise RuntimeError(
            "Merged dataframe is empty. Check that decision and feature tags match."
        )

    df["decision_category"] = df.apply(classify_decision, axis=1)
    return df


def make_severity_breakdown(faults: pd.DataFrame) -> pd.DataFrame:
    """Build a rate table by fault type, severity, and decision category."""
    counts = (
        faults
        .groupby(["fault_type", "severity", "severity_abs", "decision_category"])
        .size()
        .reset_index(name="count")
    )

    totals = (
        faults
        .groupby(["fault_type", "severity", "severity_abs"])
        .size()
        .reset_index(name="total")
    )

    breakdown = counts.merge(
        totals,
        on=["fault_type", "severity", "severity_abs"],
        how="left",
    )
    breakdown["rate"] = breakdown["count"] / breakdown["total"]

    return breakdown


def make_pivot_table(breakdown: pd.DataFrame) -> pd.DataFrame:
    """Convert the breakdown into one compact row per fault severity."""
    pivot = breakdown.pivot_table(
        index=["fault_type", "severity", "severity_abs"],
        columns="decision_category",
        values="rate",
        fill_value=0.0,
    ).reset_index()

    for column in ["hard_detected", "uncertain_inspect", "missed"]:
        if column not in pivot.columns:
            pivot[column] = 0.0

    return pivot


def save_table(df: pd.DataFrame, filename: str, label: str) -> str:
    """Save a table under paper_results/tables and print its location."""
    path = os.path.join(TABLE_DIR, filename)
    df.to_csv(path, index=False)
    print(f"[OK] Saved {label}: {path}")
    return path


def plot_detectability_vs_severity(pivot: pd.DataFrame) -> None:
    """Plot hard-detected, inspect, and missed rates across severity values."""
    for fault_type in sorted(pivot["fault_type"].unique()):
        sub = pivot[pivot["fault_type"] == fault_type].copy()
        sub = sub.sort_values("severity_abs")

        plt.figure()
        plt.plot(
            sub["severity_abs"],
            sub["hard_detected"],
            marker="o",
            label="Hard detected",
        )
        plt.plot(
            sub["severity_abs"],
            sub["uncertain_inspect"],
            marker="o",
            label="Uncertain / inspect",
        )
        plt.plot(
            sub["severity_abs"],
            sub["missed"],
            marker="o",
            label="Missed",
        )

        plt.xlabel("Fault severity magnitude")
        plt.ylabel("Rate")
        plt.title(f"Detectability vs Severity: {fault_type}")
        plt.ylim(-0.05, 1.05)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        fig_path = os.path.join(FIG_DIR, f"detectability_vs_severity_{fault_type}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

        print("[OK] Saved figure:", fig_path)


def plot_residual_evidence_vs_severity(faults: pd.DataFrame) -> None:
    """Plot residual-evidence indicators as a function of fault severity."""
    for fault_type in sorted(faults["fault_type"].unique()):
        sub = faults[faults["fault_type"] == fault_type].copy()
        sub = sub.sort_values("severity_abs")

        agg = (
            sub
            .groupby("severity_abs")
            .agg(
                steady_current_energy=("steady_current_energy", "mean"),
                steady_total_energy=("steady_total_energy", "mean"),
                transient_current_energy=("transient_current_energy", "mean"),
            )
            .reset_index()
        )

        plt.figure()
        plt.plot(
            agg["severity_abs"],
            agg["steady_current_energy"],
            marker="o",
            label="Steady current residual energy",
        )
        plt.plot(
            agg["severity_abs"],
            agg["steady_total_energy"],
            marker="o",
            label="Steady total residual energy",
        )
        plt.plot(
            agg["severity_abs"],
            agg["transient_current_energy"],
            marker="o",
            label="Transient current energy",
        )

        plt.xlabel("Fault severity magnitude")
        plt.ylabel("Residual energy indicator")
        plt.title(f"Residual Evidence vs Severity: {fault_type}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        fig_path = os.path.join(FIG_DIR, f"residual_energy_vs_severity_{fault_type}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

        print("[OK] Saved figure:", fig_path)


def make_fault_type_interpretation(pivot: pd.DataFrame) -> pd.DataFrame:
    """Summarize the average decision behavior for each fault type."""
    rows: list[Dict[str, float | str]] = []

    for fault_type in sorted(pivot["fault_type"].unique()):
        sub = pivot[pivot["fault_type"] == fault_type].copy()

        rows.append(
            {
                "fault_type": fault_type,
                "mean_hard_detection_rate": float(sub["hard_detected"].mean()),
                "mean_uncertain_inspect_rate": float(sub["uncertain_inspect"].mean()),
                "mean_missed_rate": float(sub["missed"].mean()),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    print("[INFO] Loading final framework decisions and residual features...")

    df = load_analysis_data()
    faults = df[df["fault_type"] != "healthy"].copy()
    faults = add_residual_energy_indicators(faults)

    breakdown = make_severity_breakdown(faults)
    save_table(
        breakdown,
        "severity_detectability_breakdown.csv",
        "severity breakdown",
    )
    print("\n=== Severity Detectability Breakdown ===")
    print(breakdown.to_string(index=False))

    pivot = make_pivot_table(breakdown)
    save_table(
        pivot,
        "severity_detectability_pivot.csv",
        "severity pivot",
    )
    print("\n=== Severity Detectability Pivot ===")
    print(pivot.to_string(index=False))

    plot_detectability_vs_severity(pivot)
    plot_residual_evidence_vs_severity(faults)

    interpretation = make_fault_type_interpretation(pivot)
    save_table(
        interpretation,
        "severity_detectability_interpretation.csv",
        "fault-type interpretation table",
    )
    print("\n=== Overall Fault-Type Detectability ===")
    print(interpretation.to_string(index=False))


if __name__ == "__main__":
    main()
