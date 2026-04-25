"""
Create article-ready summary tables from the framework outputs.

The script reformats the main reliability results, fault detectability summary,
closed-loop masking summary, and final decision breakdown into compact CSV
tables used in the report and paper figures.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd


TABLE_DIR = "paper_results/tables"
os.makedirs(TABLE_DIR, exist_ok=True)


def format_value(value, digits: int = 3) -> str:
    """Format numeric table values while keeping missing values readable."""
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def format_numeric_columns(
    df: pd.DataFrame,
    skip_columns: Iterable[str],
    digits: int = 3,
) -> pd.DataFrame:
    """Format all numeric columns except selected label columns."""
    out = df.copy()
    skip = set(skip_columns)

    for column in out.columns:
        if column not in skip:
            out[column] = out[column].apply(lambda value: format_value(value, digits))

    return out


def save_table(df: pd.DataFrame, filename: str, title: str) -> None:
    """Save a table and print a compact preview."""
    out_path = os.path.join(TABLE_DIR, filename)
    df.to_csv(out_path, index=False)

    print(f"\n=== {title} ===")
    print(df.to_string(index=False))
    print("[OK] Saved:", out_path)


def make_main_reliability_table() -> None:
    """Create the main reliability comparison table."""
    src = os.path.join(TABLE_DIR, "article_framework_summary_v3_metrics.csv")
    df = pd.read_csv(src)

    columns = [
        "Method",
        "Accuracy",
        "Macro-F1",
        "False Alarm Rate",
        "Missed Detection Rate",
        "Coverage",
        "Overall Uncertain Rate",
        "Hard Fault Detection Rate",
        "Uncertain Fault Rate",
        "Accepted Accuracy",
        "Safe Decision Rate",
    ]

    out = df[columns].copy()
    out = out.rename(
        columns={
            "False Alarm Rate": "FAR",
            "Missed Detection Rate": "MDR",
            "Overall Uncertain Rate": "Uncertain Rate",
            "Hard Fault Detection Rate": "Hard Fault Detection",
            "Uncertain Fault Rate": "Fault Inspect Rate",
        }
    )

    out = format_numeric_columns(out, skip_columns=["Method"])

    save_table(
        out,
        filename="TABLE_1_main_reliability_results.csv",
        title="TABLE 1: Main Reliability Results",
    )


def make_detectability_table() -> None:
    """Create the fault-type detectability table."""
    src = os.path.join(TABLE_DIR, "severity_detectability_interpretation.csv")
    df = pd.read_csv(src)

    out = df.rename(
        columns={
            "fault_type": "Fault Type",
            "mean_hard_detection_rate": "Mean Hard Detection Rate",
            "mean_uncertain_inspect_rate": "Mean Fault Inspect Rate",
            "mean_missed_rate": "Mean Missed Rate",
        }
    )

    out = format_numeric_columns(out, skip_columns=["Fault Type"])

    save_table(
        out,
        filename="TABLE_2_fault_detectability_by_type.csv",
        title="TABLE 2: Fault Detectability by Type",
    )


def make_masking_table() -> None:
    """Create the closed-loop masking summary table."""
    src = os.path.join(TABLE_DIR, "masking_effect_interpretation.csv")
    df = pd.read_csv(src)

    out = df.rename(
        columns={
            "fault_type": "Fault Type",
            "speed_error_ratio_vs_healthy": "Speed Error Ratio",
            "residual_energy_ratio_vs_healthy": "Residual Energy Ratio",
            "control_effort_ratio_vs_healthy": "Control Effort Ratio",
            "masking_index_residual_over_speed": "Masking Index",
        }
    )

    out = format_numeric_columns(out, skip_columns=["Fault Type"])

    save_table(
        out,
        filename="TABLE_3_masking_effect_summary.csv",
        title="TABLE 3: Masking Effect Summary",
    )


def make_decision_breakdown_table() -> None:
    """Create the final decision breakdown table."""
    src = os.path.join(TABLE_DIR, "decision_breakdown_v3_metrics.csv")
    df = pd.read_csv(src)

    out = df.rename(
        columns={
            "true_label": "True Label",
            "final_decision": "Final Decision",
            "count": "Count",
        }
    )

    save_table(
        out,
        filename="TABLE_4_decision_breakdown.csv",
        title="TABLE 4: Decision Breakdown",
    )


def main() -> None:
    make_main_reliability_table()
    make_detectability_table()
    make_masking_table()
    make_decision_breakdown_table()

    print("\n[OK] Article-ready summary tables created.")


if __name__ == "__main__":
    main()
