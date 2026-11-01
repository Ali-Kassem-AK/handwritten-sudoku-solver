"""Training script for the Handwritten Digit MLP (784 -> 128 -> 64 -> 9) and Logistic Regression baseline.

Implements Section 4.A requirements:
- Uses the 50,000 / 10,000 training-validation split (with digit 0 excluded -> 45,073 train, 9,004 val).
- Keeps the official test set (9,020 samples for digits 1..9) completely untouched during training
  and hyperparameter/calibration fitting.
- Trains `DigitMLP` using mini-batch Adam with weight decay and cosine/step learning rate scheduling,
  and implements mild 1-pixel shift augmentation option for handwriting robustness (Extension 4).
- Trains `LogisticRegressionBaseline` on the exact same training split.
- Fits post-hoc Temperature Scaling on the validation set ONLY.
- Saves trained models (`models/digit_mlp.npz`, `models/logreg_baseline.npz`), training history,
  and configuration (`models/model_config.json`).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.nn.calibration import compute_calibration_metrics, fit_temperature_scaling
from src.nn.model import (
    DigitMLP,
    LogisticRegressionBaseline,
    cross_entropy_loss,
    softmax,
)


def _augment_shift_batch(X_batch: np.ndarray, rng: np.random.Generator, prob: float = 0.25) -> np.ndarray:
    """Apply mild +/-1 pixel random translation to a fraction of training images for robustness."""
    if prob <= 0.0:
        return X_batch
    out = X_batch.copy()
    n = len(out)
    mask = rng.random(n) < prob
    shifts_r = rng.integers(-1, 2, size=n)
    shifts_c = rng.integers(-1, 2, size=n)
    for i in np.where(mask)[0]:
        dr, dc = int(shifts_r[i]), int(shifts_c[i])
        img = out[i]
        shifted = np.zeros_like(img)
        r_src_start = max(0, -dr)
        r_src_end = min(28, 28 - dr)
        c_src_start = max(0, -dc)
        c_src_end = min(28, 28 - dc)
        shifted[r_src_start + dr : r_src_end + dr, c_src_start + dc : c_src_end + dc] = img[
            r_src_start:r_src_end, c_src_start:c_src_end
        ]
        out[i] = shifted
    return out


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 0.0015,
    weight_decay: float = 1e-4,
    seed: int = 42,
    augment_shifts: bool = True,
) -> Tuple[DigitMLP, Dict[str, List[float]]]:
    """Train DigitMLP (784 -> 128 -> 64 -> 9) using mini-batch Adam optimizer."""
    rng = np.random.default_rng(seed)
    model = DigitMLP.initialize(seed=seed)

    # Adam first and second moment accumulators
    params = ["W1", "b1", "W2", "b2", "W3", "b3"]
    m_adam = {k: np.zeros_like(getattr(model, k)) for k in params}
    v_adam = {k: np.zeros_like(getattr(model, k)) for k in params}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0

    best_val_acc = -1.0
    best_weights = {k: getattr(model, k).copy() for k in params}

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time_sec": [],
    }

    n_train = len(X_train)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        perm = rng.permutation(n_train)
        # Cosine annealing learning rate schedule
        cur_lr = lr * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * (epoch - 1) / epochs)))

        for start in range(0, n_train, batch_size):
            step += 1
            batch_idx = perm[start : start + batch_size]
            xb = X_train[batch_idx]
            if augment_shifts:
                xb = _augment_shift_batch(xb, rng, prob=0.25)
            yb = y_train[batch_idx]
            bs = len(yb)

            x_flat, z1, a1, z2, a2, z3 = model.forward_cache(xb)
            probs = softmax(z3, temperature=1.0)

            # Backpropagation of cross-entropy + softmax
            dz3 = probs.copy()
            dz3[np.arange(bs), yb] -= 1.0
            dz3 /= bs

            dW3 = a2.T @ dz3 + weight_decay * model.W3
            db3 = np.sum(dz3, axis=0)

            da2 = dz3 @ model.W3.T
            dz2 = da2 * (z2 > 0)
            dW2 = a1.T @ dz2 + weight_decay * model.W2
            db2 = np.sum(dz2, axis=0)

            da1 = dz2 @ model.W2.T
            dz1 = da1 * (z1 > 0)
            dW1 = x_flat.T @ dz1 + weight_decay * model.W1
            db1 = np.sum(dz1, axis=0)

            grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

            bc1 = 1.0 - beta1**step
            bc2 = 1.0 - beta2**step
            for k in params:
                g = grads[k].astype(np.float32)
                m_adam[k] = beta1 * m_adam[k] + (1.0 - beta1) * g
                v_adam[k] = beta2 * v_adam[k] + (1.0 - beta2) * (g * g)
                m_hat = m_adam[k] / bc1
                v_hat = v_adam[k] / bc2
                updated = getattr(model, k) - cur_lr * m_hat / (np.sqrt(v_hat) + eps)
                setattr(model, k, updated.astype(np.float32))

        # Evaluate epoch metrics on clean train and validation splits
        train_probs = model.predict_proba(X_train, use_temperature=False)
        val_probs = model.predict_proba(X_val, use_temperature=False)

        tr_loss = cross_entropy_loss(train_probs, y_train)
        tr_acc = float(np.mean(np.argmax(train_probs, axis=1) == y_train))
        v_loss = cross_entropy_loss(val_probs, y_val)
        v_acc = float(np.mean(np.argmax(val_probs, axis=1) == y_val))
        dt = time.perf_counter() - t0

        history["train_loss"].append(round(tr_loss, 6))
        history["train_acc"].append(round(tr_acc, 6))
        history["val_loss"].append(round(v_loss, 6))
        history["val_acc"].append(round(v_acc, 6))
        history["epoch_time_sec"].append(round(dt, 4))

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_weights = {k: getattr(model, k).copy() for k in params}

    # Restore best validation weights
    for k in params:
        setattr(model, k, best_weights[k])

    # Fit post-hoc temperature scaling on validation set ONLY
    val_logits = model.forward_logits(X_val)
    optimal_temp = fit_temperature_scaling(val_logits, y_val)
    model.temperature = optimal_temp

    return model, history


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 0.05,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> Tuple[LogisticRegressionBaseline, Dict[str, List[float]]]:
    """Train Multinomial Logistic Regression baseline using mini-batch SGD with momentum."""
    rng = np.random.default_rng(seed)
    model = LogisticRegressionBaseline.initialize(seed=seed)

    vW = np.zeros_like(model.W)
    vb = np.zeros_like(model.b)
    momentum = 0.9

    n_train = len(X_train)
    x_train_flat = X_train.reshape(n_train, -1).astype(np.float32)

    best_val_acc = -1.0
    best_W, best_b = model.W.copy(), model.b.copy()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n_train)
        cur_lr = lr * (0.5 ** (epoch // 8))
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            xb = x_train_flat[idx]
            yb = y_train[idx]
            bs = len(yb)

            logits = xb @ model.W + model.b
            probs = softmax(logits, temperature=1.0)
            dz = probs.copy()
            dz[np.arange(bs), yb] -= 1.0
            dz /= bs

            dW = xb.T @ dz + weight_decay * model.W
            db = np.sum(dz, axis=0)

            vW = momentum * vW - cur_lr * dW
            vb = momentum * vb - cur_lr * db
            model.W = (model.W + vW).astype(np.float32)
            model.b = (model.b + vb).astype(np.float32)

        tr_probs = model.predict_proba(X_train, use_temperature=False)
        val_probs = model.predict_proba(X_val, use_temperature=False)
        tr_loss = cross_entropy_loss(tr_probs, y_train)
        tr_acc = float(np.mean(np.argmax(tr_probs, axis=1) == y_train))
        v_loss = cross_entropy_loss(val_probs, y_val)
        v_acc = float(np.mean(np.argmax(val_probs, axis=1) == y_val))

        history["train_loss"].append(round(tr_loss, 6))
        history["train_acc"].append(round(tr_acc, 6))
        history["val_loss"].append(round(v_loss, 6))
        history["val_acc"].append(round(v_acc, 6))

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_W, best_b = model.W.copy(), model.b.copy()

    model.W, model.b = best_W, best_b
    val_logits = model.forward_logits(X_val)
    model.temperature = fit_temperature_scaling(val_logits, y_val)

    return model, history


def run_training_pipeline(
    project_root: str | Path,
    epochs: int = 25,
    seed: int = 42,
) -> Dict[str, object]:
    """Load processed MNIST 1-9 splits, train MLP and Logistic Regression baseline, and save artifacts."""
    root = Path(project_root)
    npz_path = root / "data" / "processed" / "mnist_sudoku_1_9.npz"
    if not npz_path.exists():
        from src.data.make_data import run_make_data

        run_make_data(root, seed=seed)

    data = np.load(npz_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    t0 = time.perf_counter()
    mlp, mlp_hist = train_mlp(
        X_train, y_train, X_val, y_val, epochs=epochs, seed=seed, augment_shifts=True
    )
    mlp_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    logreg, lr_hist = train_logistic_regression(
        X_train, y_train, X_val, y_val, epochs=20, seed=seed
    )
    lr_time = time.perf_counter() - t1

    # Validation metrics before and after temperature scaling
    val_probs_uncal = mlp.predict_proba(X_val, use_temperature=False)
    val_probs_cal = mlp.predict_proba(X_val, use_temperature=True)
    val_cal_before = compute_calibration_metrics(val_probs_uncal, y_val)
    val_cal_after = compute_calibration_metrics(val_probs_cal, y_val)

    lr_val_probs = logreg.predict_proba(X_val, use_temperature=True)
    lr_val_acc = float(np.mean(np.argmax(lr_val_probs, axis=1) == y_val))

    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    mlp_path = models_dir / "digit_mlp.npz"
    lr_path = models_dir / "logreg_baseline.npz"
    cfg_path = models_dir / "model_config.json"

    summary_meta: Dict[str, object] = {
        "epochs": epochs,
        "seed": seed,
        "mlp_training_time_sec": round(mlp_time, 3),
        "logreg_training_time_sec": round(lr_time, 3),
        "mlp_train_accuracy": float(mlp_hist["train_acc"][-1]),
        "mlp_val_accuracy": float(max(mlp_hist["val_acc"])),
        "mlp_val_ece_uncalibrated": round(val_cal_before.ece, 6),
        "mlp_val_ece_calibrated": round(val_cal_after.ece, 6),
        "mlp_temperature": round(float(mlp.temperature), 6),
        "logreg_val_accuracy": round(lr_val_acc, 6),
        "logreg_temperature": round(float(logreg.temperature), 6),
        "mlp_history": mlp_hist,
        "logreg_history": lr_hist,
    }

    mlp.save(mlp_path, config_path=cfg_path, extra_meta=summary_meta)
    logreg.save(lr_path)

    return summary_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Sudoku Digit MLP and Logistic Regression baseline.")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Project root path.",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs for MLP (default: 25).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args()

    res = run_training_pipeline(args.project_root, epochs=args.epochs, seed=args.seed)
    print("=== Model Training & Validation Summary ===")
    print(f"MLP Train Accuracy:           {res['mlp_train_accuracy'] * 100:.2f}%")  # type: ignore[operator]
    print(f"MLP Validation Accuracy:      {res['mlp_val_accuracy'] * 100:.2f}%")  # type: ignore[operator]
    print(f"MLP Validation ECE (Uncal):   {res['mlp_val_ece_uncalibrated']:.4f}")
    print(f"MLP Optimal Temperature (T):  {res['mlp_temperature']:.4f}")
    print(f"MLP Validation ECE (Calib):   {res['mlp_val_ece_calibrated']:.4f}")
    print(f"LogReg Baseline Val Accuracy: {res['logreg_val_accuracy'] * 100:.2f}%")  # type: ignore[operator]


if __name__ == "__main__":
    main()
