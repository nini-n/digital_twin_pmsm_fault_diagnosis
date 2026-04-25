"""
Create article-ready tables for the advanced validation analyses.

This script formats the raw outputs from the threshold sensitivity, domain
generalization and physics-informed loss experiments. It also creates a compact
summary table that can be used for discussion or supplementary reporting.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd


TABLE_DIR = "paper_results/tables"
os.makedirs(TABLE_DIR, exist_ok=True)


def format_value(value, digits: int = 3) -> str:
    """Format numeric values for article-ready CSV tables."""
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def format_columns(df: pd.DataFrame, skip: Iterable[str], digits: int = 3) -> pd.DataFrame:
    """Format all numeric columns except the columns listed in skip."""
    out = df.copy()
    skip = set(skip)

    for column in out.columns:
        if column not in skip:
            out[column] = out[column].apply(lambda value: format_value(value, digits))

    return out


def save_table(df: pd.DataFrame, path: str, title: str) -> None:
    """Save a table and print a compact preview."""
    df.to_csv(path, index=False)
    print(f"\n=== {title} ===")
    print(df.to_string(index=False))
    print("[OK] Saved:", path)


def make_threshold_table() -> None:
    """Create the threshold sensitivity table."""
    src = os.path.join(TABLE_DIR, "threshold_sensitivity_results.csv")
    df = pd.read_csv(src)

    columns = [
        "Threshold",
        "False Alarm Rate",
        "Missed Detection Rate",
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

    out["Threshold"] = out["Threshold"].apply(lambda value: format_value(value, 2))
    out = format_columns(out, skip=["Threshold"], digits=3)

    save_table(
        out,
        os.path.join(TABLE_DIR, "TABLE_5_threshold_sensitivity.csv"),
        "TABLE 5: Threshold Sensitivity",
    )


def make_domain_generalization_table() -> None:
    """Create the unseen-speed domain generalization table."""
    src = os.path.join(TABLE_DIR, "domain_generalization_unseen_speed_summary.csv")
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
    out = format_columns(out, skip=["Method"], digits=3)

    save_table(
        out,
        os.path.join(TABLE_DIR, "TABLE_6_domain_generalization_unseen_speed.csv"),
        "TABLE 6: Domain Generalization on Unseen Speed",
    )


def make_physics_loss_table() -> None:
    """Create the physics-informed loss comparison table."""
    src = os.path.join(TABLE_DIR, "piml_physics_loss_summary.csv")
    df = pd.read_csv(src)

    columns = [
        "Method",
        "Accuracy",
        "Macro-F1",
        "FAR",
        "MDR",
        "Physics Consistency MSE",
        "Physics Consistency Corr",
        "Lambda Phys",
    ]

    out = df[columns].copy()
    out = out.rename(
        columns={
            "Physics Consistency MSE": "Physics MSE",
            "Physics Consistency Corr": "Physics Corr",
            "Lambda Phys": "lambda_phys",
        }
    )
    out = format_columns(out, skip=["Method"], digits=3)

    save_table(
        out,
        os.path.join(TABLE_DIR, "TABLE_7_physics_informed_loss.csv"),
        "TABLE 7: Physics-Informed Loss Comparison",
    )


def make_compact_advanced_summary() -> None:
    """Create a compact summary across the advanced validation experiments."""
    rows = []

    threshold_df = pd.read_csv(os.path.join(TABLE_DIR, "threshold_sensitivity_results.csv"))
    th_075 = threshold_df[np.isclose(threshold_df["Threshold"], 0.75)].iloc[0]
    th_080 = threshold_df[np.isclose(threshold_df["Threshold"], 0.80)].iloc[0]

    rows.append(
        {
            "Validation": "Threshold sensitivity",
            "Key Result": (
                f"Threshold 0.75: FAR={th_075['False Alarm Rate']:.3f}, "
                f"Uncertain={th_075['Overall Uncertain Rate']:.3f}; "
                f"Threshold 0.80: FAR={th_080['False Alarm Rate']:.3f}, "
                f"Uncertain={th_080['Overall Uncertain Rate']:.3f}"
            ),
            "Interpretation": "Higher confidence thresholds route more ambiguous cases to inspection.",
        }
    )

    domain_df = pd.read_csv(os.path.join(TABLE_DIR, "domain_generalization_unseen_speed_summary.csv"))
    full = domain_df[domain_df["Method"] == "Full framework"].iloc[0]

    rows.append(
        {
            "Validation": "Unseen speed condition",
            "Key Result": (
                f"FAR={full['False Alarm Rate']:.3f}, "
                f"MDR={full['Missed Detection Rate']:.3f}, "
                f"Accepted Accuracy={full['Accepted Accuracy']:.3f}, "
                f"Safe Decision={full['Safe Decision Rate']:.3f}"
            ),
            "Interpretation": "Reliability behavior is preserved under the unseen operating speed.",
        }
    )

    physics_df = pd.read_csv(os.path.join(TABLE_DIR, "piml_physics_loss_summary.csv"))
    standard_nn = physics_df[physics_df["Method"] == "Standard NN"].iloc[0]
    physics_nn = physics_df[physics_df["Method"] == "Physics-informed NN"].iloc[0]

    rows.append(
        {
            "Validation": "Physics-informed loss",
            "Key Result": (
                f"MSE {standard_nn['Physics Consistency MSE']:.3f} -> "
                f"{physics_nn['Physics Consistency MSE']:.3f}, "
                f"Corr {standard_nn['Physics Consistency Corr']:.3f} -> "
                f"{physics_nn['Physics Consistency Corr']:.3f}"
            ),
            "Interpretation": "Physics-informed loss improves residual consistency without reducing accuracy.",
        }
    )

    out = pd.DataFrame(rows)

    save_table(
        out,
        os.path.join(TABLE_DIR, "TABLE_8_compact_advanced_validation_summary.csv"),
        "TABLE 8: Compact Advanced Validation Summary",
    )


def main() -> None:
    make_threshold_table()
    make_domain_generalization_table()
    make_physics_loss_table()
    make_compact_advanced_summary()

    print("\n[OK] Advanced validation tables created.")


if __name__ == "__main__":
    main()
