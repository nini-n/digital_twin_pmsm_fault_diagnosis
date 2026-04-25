"""
Create journal style figures from the article-ready result tables.

The script reads CSV tables from ``paper_results/tables`` and exports the final
figures used in the PMSM digital-twin fault diagnosis report. Each figure is
saved as both PNG and PDF under ``paper_results/figures``.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TABLE_DIR = "paper_results/tables"
FIG_DIR = "paper_results/figures"

os.makedirs(FIG_DIR, exist_ok=True)


FIGURE_DPI = 300
BAR_EDGE = "#333333"

COLORS = {
    "blue": "#2F5D8C",
    "orange": "#C77C2B",
    "green": "#4F7F3A",
    "red": "#9E3D3D",
    "gray": "#6B6B6B",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    }
)


def table_path(filename: str) -> str:
    """Return the full path of a table stored in the result directory."""
    return os.path.join(TABLE_DIR, filename)


def safe_float(value) -> float:
    """Convert table values to float while preserving missing entries."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str) and value.strip().upper() == "N/A":
        return np.nan
    return float(value)


def style_axes(ax) -> None:
    """Apply a compact journal-style axis format."""
    ax.grid(True, axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="both", colors="#222222", width=0.8)


def save_figure(fig, filename: str) -> None:
    """Save a figure as PNG and PDF."""
    path_png = os.path.join(FIG_DIR, f"{filename}.png")
    path_pdf = os.path.join(FIG_DIR, f"{filename}.pdf")

    fig.tight_layout()
    fig.savefig(path_png, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {path_png}")
    print(f"[OK] Saved: {path_pdf}")


def figure_main_reliability() -> None:
    """Create the main comparison figure for reliability-aware metrics."""
    df = pd.read_csv(table_path("TABLE_1_main_reliability_results.csv"))

    method_labels = [
        "ML",
        "Physics-guided\nPIML",
        "PIML + CP",
        "Full\nframework",
    ]

    metrics = [
        ("FAR", "FAR", COLORS["red"]),
        ("Uncertain Rate", "Uncertain", COLORS["orange"]),
        ("Hard Fault Detection", "Hard detect", COLORS["blue"]),
        ("Accepted Accuracy", "Accepted acc.", COLORS["green"]),
    ]

    x = np.arange(len(method_labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(7.2, 3.7))

    for idx, (column, label, color) in enumerate(metrics):
        values = [safe_float(value) for value in df[column]]
        offset = (idx - 1.5) * width

        ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=color,
            edgecolor=BAR_EDGE,
            linewidth=0.4,
        )

    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    style_axes(ax)

    save_figure(fig, "PAPER_FINAL_FIG_1_main_reliability")


def figure_threshold_sensitivity() -> None:
    """Create the confidence-threshold trade-off figure."""
    df = pd.read_csv(table_path("TABLE_5_threshold_sensitivity.csv"))
    thresholds = df["Threshold"].astype(float)

    fig, ax = plt.subplots(figsize=(6.2, 3.7))

    curves = [
        ("FAR", "FAR", "o", COLORS["red"]),
        ("Uncertain Rate", "Uncertain rate", "s", COLORS["orange"]),
        ("Hard Fault Detection", "Hard fault detection", "^", COLORS["blue"]),
        ("Accepted Accuracy", "Accepted accuracy", "D", COLORS["green"]),
    ]

    for column, label, marker, color in curves:
        ax.plot(
            thresholds,
            df[column].astype(float),
            marker=marker,
            color=color,
            label=label,
        )

    ax.axvline(0.80, color=COLORS["gray"], linestyle="--", linewidth=1.0)
    ax.text(
        0.805,
        0.10,
        "selected threshold",
        fontsize=7,
        color=COLORS["gray"],
        va="bottom",
    )

    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.03, 1.08)
    ax.set_xticks(thresholds)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4)
    style_axes(ax)

    save_figure(fig, "PAPER_FINAL_FIG_2_threshold_tradeoff")


def figure_fault_detectability() -> None:
    """Create the fault-type detectability figure."""
    df = pd.read_csv(table_path("TABLE_2_fault_detectability_by_type.csv"))

    fault_labels = ["Ld mismatch", "Rs drift"]
    x = np.arange(len(fault_labels))
    width = 0.23

    series = [
        ("Mean Hard Detection Rate", "Hard detected", -width, COLORS["blue"]),
        ("Mean Fault Inspect Rate", "Uncertain / inspect", 0.0, COLORS["orange"]),
        ("Mean Missed Rate", "Missed", width, COLORS["red"]),
    ]

    fig, ax = plt.subplots(figsize=(5.8, 3.6))

    for column, label, offset, color in series:
        ax.bar(
            x + offset,
            df[column].astype(float).values,
            width,
            label=label,
            color=color,
            edgecolor=BAR_EDGE,
            linewidth=0.4,
        )

    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(fault_labels)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    style_axes(ax)

    save_figure(fig, "PAPER_FINAL_FIG_3_fault_type_detectability")


def figure_closed_loop_masking() -> None:
    """Create the closed-loop masking summary figure."""
    df = pd.read_csv(table_path("TABLE_3_masking_effect_summary.csv"))

    fault_labels = ["Ld mismatch", "Rs drift"]
    x = np.arange(len(fault_labels))
    width = 0.18

    series = [
        ("Speed Error Ratio", "Speed error ratio", -1.5 * width, COLORS["gray"]),
        ("Residual Energy Ratio", "Residual energy ratio", -0.5 * width, COLORS["blue"]),
        ("Control Effort Ratio", "Control effort ratio", 0.5 * width, COLORS["green"]),
        ("Masking Index", "Masking index", 1.5 * width, COLORS["orange"]),
    ]

    fig, ax = plt.subplots(figsize=(6.6, 3.8))

    for column, label, offset, color in series:
        ax.bar(
            x + offset,
            df[column].astype(float).values,
            width,
            label=label,
            color=color,
            edgecolor=BAR_EDGE,
            linewidth=0.4,
        )

    ax.set_yscale("log")
    ax.set_ylabel("Ratio relative to healthy operation")
    ax.set_xticks(x)
    ax.set_xticklabels(fault_labels)
    ax.legend(frameon=False, loc="upper left", ncol=1)
    style_axes(ax)

    save_figure(fig, "PAPER_FINAL_FIG_4_closed_loop_masking")


def figure_multisplit_robustness() -> None:
    """Create the multi-split robustness figure."""
    df = pd.read_csv(table_path("multisplit_robustness_summary.csv"))

    metrics = [
        ("coverage", "Coverage"),
        ("full_far", "FAR"),
        ("full_mdr", "MDR"),
        ("full_overall_uncertain", "Uncertain"),
        ("full_hard_fault_detection", "Hard detect"),
        ("full_accepted_accuracy", "Accepted acc."),
        ("full_safe_decision_rate", "Safe decision"),
    ]

    labels: List[str] = []
    means: List[float] = []
    stds: List[float] = []

    for metric, label in metrics:
        row = df[df["metric"] == metric]
        if row.empty:
            continue

        labels.append(label)
        means.append(float(row["mean"].iloc[0]))
        stds.append(float(row["std"].iloc[0]))

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=3,
        color=COLORS["blue"],
        edgecolor=BAR_EDGE,
        linewidth=0.4,
        error_kw={"linewidth": 0.8},
    )

    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    style_axes(ax)

    save_figure(fig, "PAPER_FINAL_FIG_5_multisplit_robustness")


def figure_physics_loss() -> None:
    """Create the physics-informed loss validation figure."""
    df = pd.read_csv(table_path("TABLE_7_physics_informed_loss.csv"))
    df = df[df["Method"].isin(["Standard NN", "Physics-informed NN"])].copy()

    methods = ["Standard NN", "Physics-informed NN"]
    x = np.arange(len(methods))
    width = 0.28

    fig, ax = plt.subplots(figsize=(5.8, 3.6))

    ax.bar(
        x - width / 2,
        df["Physics MSE"].astype(float).values,
        width,
        label="Physics MSE",
        color=COLORS["red"],
        edgecolor=BAR_EDGE,
        linewidth=0.4,
    )
    ax.bar(
        x + width / 2,
        df["Physics Corr"].astype(float).values,
        width,
        label="Physics correlation",
        color=COLORS["blue"],
        edgecolor=BAR_EDGE,
        linewidth=0.4,
    )

    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    style_axes(ax)

    save_figure(fig, "PAPER_FINAL_FIG_6_physics_loss_validation")


def main() -> None:
    figure_main_reliability()
    figure_threshold_sensitivity()
    figure_fault_detectability()
    figure_closed_loop_masking()
    figure_multisplit_robustness()
    figure_physics_loss()

    print("\n[OK] Journal-style paper figures created.")


if __name__ == "__main__":
    main()
