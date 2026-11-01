"""Probability calibration metrics (Expected Calibration Error, MCE, Brier score) and Temperature Scaling.

Follows Section 4.A and Section 14 (Optional Extension 3):
- Evaluates probability calibration using Expected Calibration Error (ECE) across M=15 equal-width
  confidence bins on [0, 1]:
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
- Fits post-hoc Temperature Scaling (T > 0) on the validation set ONLY by minimizing
  Negative Log-Likelihood (cross-entropy), then evaluates on the untouched test set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.optimize import minimize_scalar

from src.nn.model import cross_entropy_loss, softmax


@dataclass
class CalibrationReport:
    """Calibration metrics and bin-wise reliability statistics."""

    ece: float
    mce: float
    nll: float
    brier_score: float
    num_bins: int
    bin_Boundaries: List[float]
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]

    def to_dict(self) -> Dict[str, object]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "ece": round(float(self.ece), 6),
            "mce": round(float(self.mce), 6),
            "nll": round(float(self.nll), 6),
            "brier_score": round(float(self.brier_score), 6),
            "num_bins": int(self.num_bins),
            "bin_boundaries": [round(float(x), 4) for x in self.bin_Boundaries],
            "bin_accuracies": [round(float(x), 6) for x in self.bin_accuracies],
            "bin_confidences": [round(float(x), 6) for x in self.bin_confidences],
            "bin_counts": [int(x) for x in self.bin_counts],
        }


def compute_calibration_metrics(
    probs: np.ndarray, y_true: np.ndarray, num_bins: int = 15
) -> CalibrationReport:
    """Compute Expected Calibration Error (ECE), Maximum Calibration Error (MCE), NLL, and Brier score."""
    n = len(y_true)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == y_true).astype(np.float64)

    boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_accs: List[float] = []
    bin_confs: List[float] = []
    bin_cnts: List[int] = []

    for i in range(num_bins):
        low, high = boundaries[i], boundaries[i + 1]
        if i == 0:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences > low) & (confidences <= high)
        cnt = int(np.sum(in_bin))
        bin_cnts.append(cnt)
        if cnt > 0:
            acc = float(np.mean(correct[in_bin]))
            conf = float(np.mean(confidences[in_bin]))
            gap = abs(acc - conf)
            ece += (cnt / n) * gap
            mce = max(mce, gap)
            bin_accs.append(acc)
            bin_confs.append(conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(float((low + high) / 2.0))

    nll = cross_entropy_loss(probs, y_true)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), y_true] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    return CalibrationReport(
        ece=float(ece),
        mce=float(mce),
        nll=float(nll),
        brier_score=brier,
        num_bins=num_bins,
        bin_Boundaries=boundaries.tolist(),
        bin_accuracies=bin_accs,
        bin_confidences=bin_confs,
        bin_counts=bin_cnts,
    )


def fit_temperature_scaling(val_logits: np.ndarray, y_val: np.ndarray) -> float:
    """Fit post-hoc temperature T > 0 on validation set logits by minimizing NLL."""

    def objective(temp: float) -> float:
        p = softmax(val_logits, temperature=float(temp))
        return cross_entropy_loss(p, y_val)

    res = minimize_scalar(objective, bounds=(0.1, 10.0), method="bounded")
    return float(res.x)
