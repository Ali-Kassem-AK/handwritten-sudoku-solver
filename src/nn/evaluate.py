"""Evaluation script for the untouched MNIST test set (digits 1-9).

Implements Section 4.A requirements:
- Evaluates the trained `DigitMLP` and `LogisticRegressionBaseline` on the untouched
  official MNIST test set (9,020 images for digits 1..9).
- Computes and saves:
  * Overall test accuracy and per-digit precision/recall/F1.
  * 9x9 confusion matrix (for digits 1..9).
  * Expected Calibration Error (ECE), Maximum Calibration Error (MCE), NLL, and Brier score
    before and after validation-fitted temperature scaling.
  * Publication-quality diagnostic plots (`results/plots/nn_evaluation.png`) and CSV/JSON tables
    (`results/tables/nn_test_metrics.json`, `results/tables/nn_confusion_matrix.csv`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.nn.calibration import compute_calibration_metrics
from src.nn.model import DigitMLP, LogisticRegressionBaseline


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 9) -> np.ndarray:
    """Compute C x C confusion matrix where row=true class (0..8), col=predicted class (0..8)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def compute_per_class_metrics(cm: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute per-digit precision, recall, F1-score, and support for digits 1..9."""
    per_digit: Dict[str, Dict[str, float]] = {}
    for c in range(cm.shape[0]):
        tp = float(cm[c, c])
        fp = float(np.sum(cm[:, c]) - tp)
        fn = float(np.sum(cm[c, :]) - tp)
        support = int(np.sum(cm[c, :]))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_digit[str(c + 1)] = {
            "precision": round(prec, 5),
            "recall": round(rec, 5),
            "f1": round(f1, 5),
            "support": support,
        }
    return per_digit


def run_evaluation(project_root: str | Path) -> Dict[str, object]:
    """Evaluate trained DigitMLP and LogisticRegressionBaseline on untouched test set."""
    root = Path(project_root)
    npz_path = root / "data" / "processed" / "mnist_sudoku_1_9.npz"
    mlp_path = root / "models" / "digit_mlp.npz"
    lr_path = root / "models" / "logreg_baseline.npz"
    cfg_path = root / "models" / "model_config.json"

    if not mlp_path.exists() or not lr_path.exists():
        from src.nn.train import run_training_pipeline

        run_training_pipeline(root)

    data = np.load(npz_path)
    X_test, y_test = data["X_test"], data["y_test"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_train, y_train = data["X_train"], data["y_train"]

    mlp = DigitMLP.load(mlp_path)
    logreg = LogisticRegressionBaseline.load(lr_path)

    # Evaluate MLP on Train, Val, and Untouched Test
    mlp_train_probs = mlp.predict_proba(X_train, use_temperature=True)
    mlp_val_probs = mlp.predict_proba(X_val, use_temperature=True)
    mlp_test_probs_uncal = mlp.predict_proba(X_test, use_temperature=False)
    mlp_test_probs_cal = mlp.predict_proba(X_test, use_temperature=True)

    mlp_train_acc = float(np.mean(np.argmax(mlp_train_probs, axis=1) == y_train))
    mlp_val_acc = float(np.mean(np.argmax(mlp_val_probs, axis=1) == y_val))
    mlp_test_preds = np.argmax(mlp_test_probs_cal, axis=1)
    mlp_test_acc = float(np.mean(mlp_test_preds == y_test))

    mlp_cal_uncal = compute_calibration_metrics(mlp_test_probs_uncal, y_test)
    mlp_cal_cal = compute_calibration_metrics(mlp_test_probs_cal, y_test)
    mlp_cm = compute_confusion_matrix(y_test, mlp_test_preds)
    mlp_per_digit = compute_per_class_metrics(mlp_cm)

    # Evaluate Logistic Regression Baseline on Train, Val, and Untouched Test
    lr_train_probs = logreg.predict_proba(X_train, use_temperature=True)
    lr_val_probs = logreg.predict_proba(X_val, use_temperature=True)
    lr_test_probs_uncal = logreg.predict_proba(X_test, use_temperature=False)
    lr_test_probs_cal = logreg.predict_proba(X_test, use_temperature=True)

    lr_train_acc = float(np.mean(np.argmax(lr_train_probs, axis=1) == y_train))
    lr_val_acc = float(np.mean(np.argmax(lr_val_probs, axis=1) == y_val))
    lr_test_preds = np.argmax(lr_test_probs_cal, axis=1)
    lr_test_acc = float(np.mean(lr_test_preds == y_test))

    lr_cal_uncal = compute_calibration_metrics(lr_test_probs_uncal, y_test)
    lr_cal_cal = compute_calibration_metrics(lr_test_probs_cal, y_test)
    lr_cm = compute_confusion_matrix(y_test, lr_test_preds)
    lr_per_digit = compute_per_class_metrics(lr_cm)

    tables_dir = root / "results" / "tables"
    plots_dir = root / "results" / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save Confusion Matrix CSV
    digits_labels = [f"Digit_{d}" for d in range(1, 10)]
    cm_df = pd.DataFrame(mlp_cm, index=digits_labels, columns=digits_labels)
    cm_df.to_csv(tables_dir / "nn_confusion_matrix.csv")

    lr_cm_df = pd.DataFrame(lr_cm, index=digits_labels, columns=digits_labels)
    lr_cm_df.to_csv(tables_dir / "logreg_confusion_matrix.csv")

    eval_summary: Dict[str, object] = {
        "test_samples": int(len(y_test)),
        "mlp": {
            "train_accuracy": round(mlp_train_acc, 6),
            "val_accuracy": round(mlp_val_acc, 6),
            "test_accuracy": round(mlp_test_acc, 6),
            "temperature": round(float(mlp.temperature), 6),
            "uncalibrated_test": mlp_cal_uncal.to_dict(),
            "calibrated_test": mlp_cal_cal.to_dict(),
            "per_digit_metrics": mlp_per_digit,
            "confusion_matrix": mlp_cm.tolist(),
        },
        "logistic_regression_baseline": {
            "train_accuracy": round(lr_train_acc, 6),
            "val_accuracy": round(lr_val_acc, 6),
            "test_accuracy": round(lr_test_acc, 6),
            "temperature": round(float(logreg.temperature), 6),
            "uncalibrated_test": lr_cal_uncal.to_dict(),
            "calibrated_test": lr_cal_cal.to_dict(),
            "per_digit_metrics": lr_per_digit,
            "confusion_matrix": lr_cm.tolist(),
        },
    }

    with open(tables_dir / "nn_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    # Plot 3-panel evaluation figure: (a) Confusion Matrix, (b) Reliability Diagram, (c) Training Curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=180)

    # Panel A: Confusion Matrix
    ax0 = axes[0]
    im = ax0.imshow(mlp_cm, cmap="Blues")
    ax0.set_title(f"MLP Test Confusion Matrix (Acc: {mlp_test_acc*100:.2f}%)", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Predicted Sudoku Digit (1–9)", fontsize=10)
    ax0.set_ylabel("True Sudoku Digit (1–9)", fontsize=10)
    ax0.set_xticks(range(9))
    ax0.set_yticks(range(9))
    ax0.set_xticklabels(range(1, 10))
    ax0.set_yticklabels(range(1, 10))
    thresh = mlp_cm.max() / 2.0
    for i in range(9):
        for j in range(9):
            val = mlp_cm[i, j]
            ax0.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                color="white" if val > thresh else "#0f172a",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    # Panel B: Reliability Diagram (Calibration)
    ax1 = axes[1]
    ax1.plot([0, 1], [0, 1], "--", color="#64748b", label="Perfect Calibration")
    non_empty_u = [i for i, c in enumerate(mlp_cal_uncal.bin_counts) if c > 0]
    ax1.plot(
        [mlp_cal_uncal.bin_confidences[i] for i in non_empty_u],
        [mlp_cal_uncal.bin_accuracies[i] for i in non_empty_u],
        "o-",
        color="#dc2626",
        label=f"MLP Uncal (ECE={mlp_cal_uncal.ece:.4f})",
    )
    non_empty_c = [i for i, c in enumerate(mlp_cal_cal.bin_counts) if c > 0]
    ax1.plot(
        [mlp_cal_cal.bin_confidences[i] for i in non_empty_c],
        [mlp_cal_cal.bin_accuracies[i] for i in non_empty_c],
        "s-",
        color="#2563eb",
        label=f"MLP Temp-Scaled (ECE={mlp_cal_cal.ece:.4f})",
    )
    non_empty_lr = [i for i, c in enumerate(lr_cal_cal.bin_counts) if c > 0]
    ax1.plot(
        [lr_cal_cal.bin_confidences[i] for i in non_empty_lr],
        [lr_cal_cal.bin_accuracies[i] for i in non_empty_lr],
        "^-",
        color="#059669",
        alpha=0.8,
        label=f"LogReg Baseline (ECE={lr_cal_cal.ece:.4f})",
    )
    ax1.set_title("Reliability Diagram on Untouched Test Set", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Mean Predicted Confidence", fontsize=10)
    ax1.set_ylabel("Empirical Accuracy", fontsize=10)
    ax1.set_xlim(0.0, 1.02)
    ax1.set_ylim(0.0, 1.02)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=8.5)

    # Panel C: Model Comparison & Learning Curve
    ax2 = axes[2]
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        hist = cfg.get("mlp_history", {})
        if "val_acc" in hist:
            ep = range(1, len(hist["val_acc"]) + 1)
            ax2.plot(ep, [x * 100 for x in hist["train_acc"]], "o-", color="#2563eb", markersize=3.5, label="MLP Train Acc")
            ax2.plot(ep, [x * 100 for x in hist["val_acc"]], "s-", color="#0d9488", markersize=3.5, label="MLP Val Acc")
    ax2.axhline(mlp_test_acc * 100, color="#1d4ed8", linestyle="--", label=f"MLP Test ({mlp_test_acc*100:.2f}%)")
    ax2.axhline(lr_test_acc * 100, color="#d97706", linestyle="-.", label=f"LogReg Test ({lr_test_acc*100:.2f}%)")
    ax2.set_title("Training Convergence & Baseline Comparison", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Training Epoch", fontsize=10)
    ax2.set_ylabel("Classification Accuracy (%)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(plots_dir / "nn_evaluation.png", bbox_inches="tight")
    plt.close(fig)

    return eval_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DigitMLP and LogisticRegressionBaseline on MNIST 1-9 test set.")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Project root path.",
    )
    args = parser.parse_args()

    res = run_evaluation(args.project_root)
    mlp_res = res["mlp"]  # type: ignore[index]
    lr_res = res["logistic_regression_baseline"]  # type: ignore[index]
    print("=== Untouched MNIST Test Set Evaluation (Digits 1–9, N=9,020) ===")
    print(f"MLP Test Accuracy:            {mlp_res['test_accuracy'] * 100:.2f}%")  # type: ignore[index]
    print(f"MLP Test ECE (Uncalibrated):  {mlp_res['uncalibrated_test']['ece']:.4f}")  # type: ignore[index]
    print(f"MLP Test ECE (Calibrated):    {mlp_res['calibrated_test']['ece']:.4f} (T={mlp_res['temperature']:.4f})")  # type: ignore[index]
    print(f"LogReg Baseline Test Acc:     {lr_res['test_accuracy'] * 100:.2f}%")  # type: ignore[index]
    print(f"LogReg Baseline Test ECE:     {lr_res['calibrated_test']['ece']:.4f}")  # type: ignore[index]


if __name__ == "__main__":
    main()
