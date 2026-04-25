"""
Closed-loop masking analysis for the PMSM digital-twin framework.

The analysis compares output-level speed tracking error with internal residual
and control-domain indicators. It is used to show when a fault has limited
visible effect on speed tracking while still changing the residual behavior.
"""

from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS_INDEX = "data/runs_index_20260301_204447.csv"
FAULT_ON_TIME = 0.10
EPS = 1e-12

OUT_DIR = "paper_results"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TABLE_DIR = os.path.join(OUT_DIR, "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


def load_signals(path: str) -> pd.DataFrame:
    """Load one signal-level simulation run."""
    return pd.read_csv(path)


def rms(x) -> float:
    """Return the root-mean-square value of a signal."""
    values = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(values * values))) if len(values) else 0.0


def mean_abs(x) -> float:
    """Return the mean absolute value of a signal."""
    values = np.asarray(x, dtype=float)
    return float(np.mean(np.abs(values))) if len(values) else 0.0


def max_abs(x) -> float:
    """Return the maximum absolute value of a signal."""
    values = np.asarray(x, dtype=float)
    return float(np.max(np.abs(values))) if len(values) else 0.0


def control_effort_metrics(signals: pd.DataFrame, mask: np.ndarray, d_col: str, q_col: str) -> Dict[str, float]:
    """Compute RMS and mean voltage-vector effort if the requested columns exist."""
    if d_col not in signals.columns or q_col not in signals.columns:
        return {"rms": np.nan, "mean": np.nan}

    effort = np.sqrt(signals.loc[mask, d_col] ** 2 + signals.loc[mask, q_col] ** 2)
    return {"rms": rms(effort), "mean": float(np.mean(effort))}


def compute_run_masking_metrics(row: pd.Series) -> Dict[str, float]:
    """Compute output-level and residual-level indicators for one run."""
    signals = load_signals(row["signals_csv"])

    mask = signals["t"] >= FAULT_ON_TIME
    if not np.any(mask):
        mask = np.ones(len(signals), dtype=bool)

    speed_error = signals.loc[mask, "omega_ref"] - signals.loc[mask, "omega"]

    r_id = signals.loc[mask, "r_id"]
    r_iq = signals.loc[mask, "r_iq"]
    r_omega = signals.loc[mask, "r_omega"]

    residual_energy = r_id**2 + r_iq**2 + r_omega**2
    current_residual_energy = r_id**2 + r_iq**2

    control_effort = control_effort_metrics(signals, mask, "vd", "vq")
    pi_effort = control_effort_metrics(signals, mask, "vd_pi", "vq_pi")

    return {
        "tag": row["tag"],
        "fault_type": row["fault_type"],
        "severity": float(row["severity"]),
        "omega_step": float(row["omega_step"]),
        "load_profile": int(row["load_profile"]),
        "seed": int(row["seed"]),
        "speed_error_rms": rms(speed_error),
        "speed_error_mean_abs": mean_abs(speed_error),
        "speed_error_max_abs": max_abs(speed_error),
        "residual_energy_mean": float(np.mean(residual_energy)),
        "current_residual_energy_mean": float(np.mean(current_residual_energy)),
        "r_id_rms": rms(r_id),
        "r_iq_rms": rms(r_iq),
        "r_omega_rms": rms(r_omega),
        "control_effort_rms": control_effort["rms"],
        "control_effort_mean": control_effort["mean"],
        "pi_effort_rms": pi_effort["rms"],
        "pi_effort_mean": pi_effort["mean"],
    }


def summarize_by_fault_type(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create a fault-type summary from run-level masking indicators."""
    return (
        metrics.groupby("fault_type")
        .agg(
            n=("tag", "count"),
            speed_error_rms_mean=("speed_error_rms", "mean"),
            speed_error_rms_std=("speed_error_rms", "std"),
            residual_energy_mean=("residual_energy_mean", "mean"),
            residual_energy_std=("residual_energy_mean", "std"),
            current_residual_energy_mean=("current_residual_energy_mean", "mean"),
            current_residual_energy_std=("current_residual_energy_mean", "std"),
            control_effort_rms_mean=("control_effort_rms", "mean"),
            control_effort_rms_std=("control_effort_rms", "std"),
            pi_effort_rms_mean=("pi_effort_rms", "mean"),
            pi_effort_rms_std=("pi_effort_rms", "std"),
        )
        .reset_index()
    )


def normalize_against_healthy(metrics: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Normalize fault indicators relative to the healthy mean."""
    healthy = metrics[metrics["fault_type"] == "healthy"]
    if len(healthy) == 0:
        raise RuntimeError("No healthy samples found for masking normalization.")

    healthy_speed = healthy["speed_error_rms"].mean()
    healthy_residual = healthy["residual_energy_mean"].mean()
    healthy_current_residual = healthy["current_residual_energy_mean"].mean()
    healthy_control = healthy["control_effort_rms"].mean()

    out = summary.copy()
    out["speed_error_ratio_vs_healthy"] = out["speed_error_rms_mean"] / (healthy_speed + EPS)
    out["residual_energy_ratio_vs_healthy"] = out["residual_energy_mean"] / (healthy_residual + EPS)
    out["current_residual_ratio_vs_healthy"] = out["current_residual_energy_mean"] / (
        healthy_current_residual + EPS
    )
    out["control_effort_ratio_vs_healthy"] = out["control_effort_rms_mean"] / (healthy_control + EPS)

    return out


def make_interpretation_table(norm_summary: pd.DataFrame) -> pd.DataFrame:
    """Create the compact masking table used in the paper summary."""
    rows = []

    for _, row in norm_summary.iterrows():
        if row["fault_type"] == "healthy":
            continue

        speed_ratio = row["speed_error_ratio_vs_healthy"]
        residual_ratio = row["residual_energy_ratio_vs_healthy"]
        control_ratio = row["control_effort_ratio_vs_healthy"]

        rows.append(
            {
                "fault_type": row["fault_type"],
                "speed_error_ratio_vs_healthy": speed_ratio,
                "residual_energy_ratio_vs_healthy": residual_ratio,
                "control_effort_ratio_vs_healthy": control_ratio,
                "masking_index_residual_over_speed": residual_ratio / (speed_ratio + EPS),
            }
        )

    return pd.DataFrame(rows)


def plot_speed_error_vs_residual_energy(metrics: pd.DataFrame) -> None:
    """Plot output-level error against internal residual energy."""
    plt.figure()
    for fault_type in sorted(metrics["fault_type"].unique()):
        subset = metrics[metrics["fault_type"] == fault_type]
        plt.scatter(
            subset["speed_error_rms"],
            subset["residual_energy_mean"],
            label=fault_type,
            alpha=0.8,
        )

    plt.xlabel("Speed error RMS")
    plt.ylabel("Mean residual energy")
    plt.title("Closed-Loop Masking: Speed Error vs Residual Energy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "masking_speed_error_vs_residual_energy.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("[OK] Saved figure:", fig_path)


def plot_normalized_indicators(norm_summary: pd.DataFrame) -> None:
    """Plot normalized output, residual, and control indicators by fault type."""
    plot_df = norm_summary.sort_values("fault_type").copy()
    x = np.arange(len(plot_df))
    width = 0.2

    plt.figure()
    plt.bar(x - 1.5 * width, plot_df["speed_error_ratio_vs_healthy"], width, label="Speed error")
    plt.bar(x - 0.5 * width, plot_df["residual_energy_ratio_vs_healthy"], width, label="Residual energy")
    plt.bar(x + 0.5 * width, plot_df["current_residual_ratio_vs_healthy"], width, label="Current residual")
    plt.bar(x + 1.5 * width, plot_df["control_effort_ratio_vs_healthy"], width, label="Control effort")

    plt.xticks(x, plot_df["fault_type"])
    plt.ylabel("Ratio relative to healthy")
    plt.title("Output-Level vs Internal Indicators")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "masking_normalized_indicators_by_fault_type.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("[OK] Saved figure:", fig_path)


def plot_indicators_vs_severity(metrics: pd.DataFrame) -> None:
    """Plot masking indicators as a function of fault severity."""
    faults = metrics[metrics["fault_type"] != "healthy"].copy()
    faults["severity_abs"] = np.abs(faults["severity"])

    for fault_type in sorted(faults["fault_type"].unique()):
        subset = faults[faults["fault_type"] == fault_type]
        aggregated = (
            subset.groupby("severity_abs")
            .agg(
                speed_error_rms=("speed_error_rms", "mean"),
                residual_energy_mean=("residual_energy_mean", "mean"),
                control_effort_rms=("control_effort_rms", "mean"),
            )
            .reset_index()
            .sort_values("severity_abs")
        )

        plt.figure()
        plt.plot(aggregated["severity_abs"], aggregated["speed_error_rms"], marker="o", label="Speed error RMS")
        plt.plot(aggregated["severity_abs"], aggregated["residual_energy_mean"], marker="o", label="Residual energy")
        plt.plot(aggregated["severity_abs"], aggregated["control_effort_rms"], marker="o", label="Control effort RMS")

        plt.xlabel("Fault severity magnitude")
        plt.ylabel("Indicator value")
        plt.title(f"Masking Indicators vs Severity: {fault_type}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        fig_path = os.path.join(FIG_DIR, f"masking_indicators_vs_severity_{fault_type}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print("[OK] Saved figure:", fig_path)


def save_outputs(metrics: pd.DataFrame, summary: pd.DataFrame, norm_summary: pd.DataFrame, interpretation: pd.DataFrame) -> None:
    """Save the masking tables used by the paper and plotting scripts."""
    paths = {
        "run_metrics": os.path.join(TABLE_DIR, "masking_effect_run_metrics.csv"),
        "summary": os.path.join(TABLE_DIR, "masking_effect_summary_by_fault_type.csv"),
        "normalized": os.path.join(TABLE_DIR, "masking_effect_normalized_summary.csv"),
        "interpretation": os.path.join(TABLE_DIR, "masking_effect_interpretation.csv"),
    }

    metrics.to_csv(paths["run_metrics"], index=False)
    summary.to_csv(paths["summary"], index=False)
    norm_summary.to_csv(paths["normalized"], index=False)
    interpretation.to_csv(paths["interpretation"], index=False)

    for name, path in paths.items():
        print(f"[OK] Saved {name}:", path)


def main() -> None:
    print("[INFO] Loading run index...")
    index = pd.read_csv(RUNS_INDEX)

    metrics = pd.DataFrame(
        [compute_run_masking_metrics(row) for _, row in index.iterrows()]
    )

    summary = summarize_by_fault_type(metrics)
    norm_summary = normalize_against_healthy(metrics, summary)
    interpretation = make_interpretation_table(norm_summary)

    save_outputs(metrics, summary, norm_summary, interpretation)

    print("\n=== Masking Summary by Fault Type ===")
    print(summary.to_string(index=False))

    print("\n=== Normalized Masking Summary ===")
    print(
        norm_summary[
            [
                "fault_type",
                "speed_error_ratio_vs_healthy",
                "residual_energy_ratio_vs_healthy",
                "current_residual_ratio_vs_healthy",
                "control_effort_ratio_vs_healthy",
            ]
        ].to_string(index=False)
    )

    print("\n=== Masking Interpretation ===")
    print(interpretation.to_string(index=False))

    plot_speed_error_vs_residual_energy(metrics)
    plot_normalized_indicators(norm_summary)
    plot_indicators_vs_severity(metrics)


if __name__ == "__main__":
    main()
