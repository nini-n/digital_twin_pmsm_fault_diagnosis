"""
Physics-informed loss validation for the PMSM digital-twin diagnosis framework.

The experiment compares a physics-guided Random Forest baseline, a standard
neural classifier and a neural classifier trained with an additional physics
consistency loss. The physics term encourages the predicted fault probability
to follow a residual-evidence score derived from digital-twin residual features.
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
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for this experiment. Install it with: pip install torch"
    ) from exc


FEATURES_CSV = "data/features_20260301_204447.csv"

OUT_DIR = "paper_results"
TABLE_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")
REPORT_DIR = os.path.join(OUT_DIR, "reports")

CONFORMAL_SPLIT_SEED = 42
CAL_TEST_SPLIT_SEED = 7
NN_SEED = 42
N_EPOCHS = 250
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
LAMBDA_PHYS = 0.20

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


def ensure_output_dirs() -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


def add_physics_guided_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add residual-energy features used by the physics-guided models."""
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

    out["phys_residual_evidence_raw"] = (
        out["phys_steady_total_energy"]
        + out["phys_transient_current_energy"]
        + 0.25 * out["phys_speed_residual_strength"]
    )

    return out


def make_physics_evidence_score(
    train_raw: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Map raw residual evidence to [0, 1] using robust train-set quantiles.

    The same quantile limits from the training set are applied to test values.
    """
    q_low = float(np.quantile(train_raw, 0.10))
    q_high = float(np.quantile(train_raw, 0.90))

    if q_high <= q_low:
        q_high = q_low + 1e-6

    score = (values - q_low) / (q_high - q_low)
    score = np.clip(score, 0.0, 1.0)

    return score.astype(np.float32), {"q_low": q_low, "q_high": q_high}


def binary_alarm_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute false alarm rate and missed detection rate."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_fault = y_true != "healthy"
    pred_fault = y_pred != "healthy"

    healthy_mask = ~true_fault
    fault_mask = true_fault

    far = float(np.mean(pred_fault[healthy_mask])) if np.any(healthy_mask) else np.nan
    mdr = float(np.mean(~pred_fault[fault_mask])) if np.any(fault_mask) else np.nan

    return far, mdr


def classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
) -> Dict[str, float]:
    """Return multiclass and alarm-level metrics."""
    far, mdr = binary_alarm_metrics(y_true, y_pred)

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


class MLPClassifier(nn.Module):
    """Small feedforward classifier used for the physics-loss comparison."""

    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    phys_score_train: np.ndarray,
    input_dim: int,
    n_classes: int,
    lambda_phys: float = 0.0,
    epochs: int = N_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    seed: int = NN_SEED,
) -> Tuple[MLPClassifier, pd.DataFrame]:
    """
    Train a neural classifier with optional physics-consistency regularization.

    The physics term compares predicted fault probability with the residual
    evidence score. A zero lambda reduces the method to a standard classifier.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPClassifier(input_dim=input_dim, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(phys_score_train, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_ce = 0.0
        total_phys = 0.0
        n_seen = 0

        for xb, yb, pb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pb = pb.to(device)

            logits = model(xb)
            ce_loss = F.cross_entropy(logits, yb)

            probs = F.softmax(logits, dim=1)
            healthy_prob = probs[:, 0]
            fault_prob = 1.0 - healthy_prob
            physics_loss = F.mse_loss(fault_prob, pb)

            loss = ce_loss + lambda_phys * physics_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_actual = xb.shape[0]
            total_loss += float(loss.item()) * batch_size_actual
            total_ce += float(ce_loss.item()) * batch_size_actual
            total_phys += float(physics_loss.item()) * batch_size_actual
            n_seen += batch_size_actual

        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / n_seen,
                "ce_loss": total_ce / n_seen,
                "physics_loss": total_phys / n_seen,
            }
        )

    return model, pd.DataFrame(history)


def predict_nn(
    model: MLPClassifier,
    X: np.ndarray,
    label_encoder: LabelEncoder,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return predicted labels and class probabilities for a trained network."""
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(x_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        pred_idx = np.argmax(probs, axis=1)

    return label_encoder.inverse_transform(pred_idx), probs


def physics_consistency_score(
    probs: np.ndarray,
    phys_score: np.ndarray,
    healthy_index: int = 0,
) -> Tuple[float, float]:
    """
    Compare model fault probability with residual evidence.

    Lower MSE and higher correlation indicate better consistency with the
    residual-evidence score.
    """
    fault_prob = 1.0 - probs[:, healthy_index]
    mse = float(np.mean((fault_prob - phys_score) ** 2))
    corr = (
        float(np.corrcoef(fault_prob, phys_score)[0, 1])
        if len(phys_score) > 2
        else np.nan
    )

    return mse, corr


def make_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use the same group-aware train/calibration/test split as the main framework."""
    groups = (
        df["omega_step"].astype(str)
        + "_lp" + df["load_profile"].astype(str)
        + "_seed" + df["seed"].astype(str)
    )

    splitter_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.40,
        random_state=CONFORMAL_SPLIT_SEED,
    )
    train_idx, temp_idx = next(splitter_1.split(df, df["fault_type"], groups=groups))

    temp_df = df.iloc[temp_idx].copy()
    temp_groups = groups.iloc[temp_idx]

    splitter_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=CAL_TEST_SPLIT_SEED,
    )
    _, test_rel_idx = next(
        splitter_2.split(temp_df, temp_df["fault_type"], groups=temp_groups)
    )

    test_idx = temp_df.iloc[test_rel_idx].index.to_numpy()
    cal_idx = temp_df.drop(index=test_idx).index.to_numpy()

    return df.loc[train_idx].copy(), df.loc[cal_idx].copy(), df.loc[test_idx].copy()


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Train and evaluate the Random Forest physics-guided baseline."""
    model = RandomForestClassifier(
        n_estimators=700,
        random_state=123,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def plot_training_losses(hist_std: pd.DataFrame, hist_piml: pd.DataFrame) -> str:
    """Save the training-loss comparison figure."""
    plt.figure()
    plt.plot(hist_std["epoch"], hist_std["loss"], label="Standard NN total loss")
    plt.plot(hist_piml["epoch"], hist_piml["loss"], label="Physics-informed NN total loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Training Loss Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "piml_physics_loss_training_curve.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return fig_path


def plot_physics_consistency(
    phys_test: np.ndarray,
    probs_std: np.ndarray,
    probs_piml: np.ndarray,
) -> str:
    """Save the residual-evidence versus fault-probability figure."""
    plt.figure()
    plt.scatter(phys_test, 1.0 - probs_std[:, 0], alpha=0.8, label="Standard NN")
    plt.scatter(phys_test, 1.0 - probs_piml[:, 0], alpha=0.8, label="Physics-informed NN")
    plt.xlabel("Residual evidence score")
    plt.ylabel("Predicted fault probability")
    plt.title("Physics Consistency: Residual Evidence vs Fault Probability")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "piml_physics_consistency.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return fig_path


def save_results(
    summary: pd.DataFrame,
    hist_std: pd.DataFrame,
    hist_piml: pd.DataFrame,
    feature_cols: List[str],
    class_labels: List[str],
    split_sizes: Dict[str, int],
    phys_meta: Dict[str, float],
) -> None:
    """Save summary tables, training history, figures and metadata."""
    summary_path = os.path.join(TABLE_DIR, "piml_physics_loss_summary.csv")
    summary.to_csv(summary_path, index=False)

    hist_std = hist_std.copy()
    hist_piml = hist_piml.copy()
    hist_std["model"] = "Standard NN"
    hist_piml["model"] = "Physics-informed NN"

    history = pd.concat([hist_std, hist_piml], ignore_index=True)
    history_path = os.path.join(REPORT_DIR, "piml_physics_loss_training_history.csv")
    history.to_csv(history_path, index=False)

    train_curve_path = plot_training_losses(hist_std, hist_piml)

    meta = {
        "features_csv": FEATURES_CSV,
        "feature_columns": feature_cols,
        "classes": class_labels,
        "split_sizes": split_sizes,
        "physics_loss": {
            "definition": (
                "CrossEntropy + lambda_phys * "
                "MSE(fault_probability, residual_evidence_score)"
            ),
            "lambda_phys": LAMBDA_PHYS,
            "residual_evidence_score_meta": phys_meta,
        },
        "outputs": {
            "summary": summary_path,
            "training_history": history_path,
            "training_curve": train_curve_path,
        },
    }

    meta_path = os.path.join(REPORT_DIR, "piml_physics_loss_meta.json")
    with open(meta_path, "w") as file:
        json.dump(meta, file, indent=2)

    print("[OK] Saved summary:", summary_path)
    print("[OK] Saved training history:", history_path)
    print("[OK] Saved training curve:", train_curve_path)
    print("[OK] Saved meta:", meta_path)


def main() -> None:
    ensure_output_dirs()

    print("[INFO] Loading feature dataset...")
    df = pd.read_csv(FEATURES_CSV)
    df = add_physics_guided_features(df)

    feature_cols = BASE_FEATURE_COLS + PHYSICS_FEATURE_COLS
    class_labels = sorted(df["fault_type"].unique().tolist())

    if class_labels[0] != "healthy":
        raise RuntimeError("Expected 'healthy' to be label index 0 after sorting.")

    train_df, cal_df, test_df = make_split(df)
    print(
        f"[INFO] Split sizes: train={len(train_df)}, "
        f"cal={len(cal_df)}, test={len(test_df)}"
    )

    X_train_raw = train_df[feature_cols].values
    X_test_raw = test_df[feature_cols].values

    y_train_str = train_df["fault_type"].values
    y_test_str = test_df["fault_type"].values

    label_encoder = LabelEncoder()
    label_encoder.fit(class_labels)

    y_train = label_encoder.transform(y_train_str)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    train_phys_raw = train_df["phys_residual_evidence_raw"].values.astype(float)
    test_phys_raw = test_df["phys_residual_evidence_raw"].values.astype(float)

    phys_train, phys_meta = make_physics_evidence_score(train_phys_raw, train_phys_raw)
    phys_test, _ = make_physics_evidence_score(train_phys_raw, test_phys_raw)

    print("\n[1] Training Random Forest physics-guided baseline...")
    y_rf = train_random_forest(X_train_raw, y_train_str, X_test_raw)
    rf_summary = classification_summary(y_test_str, y_rf, class_labels)

    print("\n[2] Training standard neural classifier...")
    nn_std, hist_std = train_nn(
        X_train=X_train,
        y_train=y_train,
        phys_score_train=phys_train,
        input_dim=X_train.shape[1],
        n_classes=len(class_labels),
        lambda_phys=0.0,
        seed=NN_SEED,
    )
    y_std, probs_std = predict_nn(nn_std, X_test, label_encoder)
    std_summary = classification_summary(y_test_str, y_std, class_labels)
    std_phys_mse, std_phys_corr = physics_consistency_score(probs_std, phys_test)

    print("\n[3] Training physics-informed neural classifier...")
    nn_piml, hist_piml = train_nn(
        X_train=X_train,
        y_train=y_train,
        phys_score_train=phys_train,
        input_dim=X_train.shape[1],
        n_classes=len(class_labels),
        lambda_phys=LAMBDA_PHYS,
        seed=NN_SEED,
    )
    y_piml, probs_piml = predict_nn(nn_piml, X_test, label_encoder)
    piml_summary = classification_summary(y_test_str, y_piml, class_labels)
    piml_phys_mse, piml_phys_corr = physics_consistency_score(probs_piml, phys_test)

    consistency_fig = plot_physics_consistency(phys_test, probs_std, probs_piml)

    summary = pd.DataFrame(
        [
            {
                "Method": "RF physics-guided",
                **rf_summary,
                "Physics Consistency MSE": np.nan,
                "Physics Consistency Corr": np.nan,
                "Lambda Phys": np.nan,
            },
            {
                "Method": "Standard NN",
                **std_summary,
                "Physics Consistency MSE": std_phys_mse,
                "Physics Consistency Corr": std_phys_corr,
                "Lambda Phys": 0.0,
            },
            {
                "Method": "Physics-informed NN",
                **piml_summary,
                "Physics Consistency MSE": piml_phys_mse,
                "Physics Consistency Corr": piml_phys_corr,
                "Lambda Phys": LAMBDA_PHYS,
            },
        ]
    )

    split_sizes = {
        "train": int(len(train_df)),
        "calibration": int(len(cal_df)),
        "test": int(len(test_df)),
    }

    save_results(
        summary=summary,
        hist_std=hist_std,
        hist_piml=hist_piml,
        feature_cols=feature_cols,
        class_labels=class_labels,
        split_sizes=split_sizes,
        phys_meta=phys_meta,
    )

    meta_path = os.path.join(REPORT_DIR, "piml_physics_loss_meta.json")
    with open(meta_path, "r") as file:
        meta = json.load(file)
    meta["outputs"]["physics_consistency"] = consistency_fig
    with open(meta_path, "w") as file:
        json.dump(meta, file, indent=2)

    print("\n=== PIML Physics-Loss Summary ===")
    print(summary.to_string(index=False))
    print("[OK] Saved physics consistency figure:", consistency_fig)


if __name__ == "__main__":
    main()
