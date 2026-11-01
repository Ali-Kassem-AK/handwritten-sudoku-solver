"""Prediction interface for single cells and 9x9 Sudoku image mosaics.

Follows Section 4.A requirements:
- Accepts 28x28 grayscale images normalized to [0, 1].
- Excludes digit 0 and classifies digits 1..9.
- Returns probabilities for all 9 digits, the predicted digit (1..9), and its confidence.
- Provides a mock/uniform predictor when NN is switched off (`use_nn=False`) for demonstration
  and ablation comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.data.mnist_loader import class_to_digit
from src.data.mosaic_generator import detect_blank_cells_grid
from src.nn.model import DigitMLP, LogisticRegressionBaseline


@dataclass
class CellPrediction:
    """Prediction details for a single Sudoku cell."""

    row: int
    col: int
    is_blank: bool
    predicted_digit: int  # 0 if blank, else 1..9
    confidence: float  # 1.0 if blank, else max probability in [1/9, 1.0]
    probabilities: List[float]  # length 9, corresponding to digits 1..9
    top_candidates: List[Tuple[int, float]]  # sorted (digit, prob) descending


@dataclass
class GridRecognitionResult:
    """Complete recognition output for a 9x9 Sudoku mosaic."""

    predicted_grid: np.ndarray  # shape (9, 9), int64 (0 for blank, 1..9 for recognized givens)
    probability_grid: np.ndarray  # shape (9, 9, 9), float64 (probabilities for digits 1..9)
    confidence_grid: np.ndarray  # shape (9, 9), float64
    blank_mask: np.ndarray  # shape (9, 9), bool
    cell_predictions: List[List[CellPrediction]]
    nn_enabled: bool


def predict_single_image(
    model: DigitMLP | LogisticRegressionBaseline,
    image_28x28: np.ndarray,
    use_calibrated: bool = True,
) -> Tuple[int, float, np.ndarray]:
    """Predict digit 1..9, confidence, and 9-class probability vector for one 28x28 image."""
    if image_28x28.shape != (28, 28):
        raise ValueError(f"Expected 28x28 image, got {image_28x28.shape}")
    batch = image_28x28.reshape(1, 28, 28).astype(np.float32)
    probs = model.predict_proba(batch, use_temperature=use_calibrated)[0].astype(np.float64)
    cls_idx = int(np.argmax(probs))
    pred_digit = int(class_to_digit(cls_idx))
    conf = float(probs[cls_idx])
    return pred_digit, conf, probs


def recognize_sudoku_mosaic(
    cell_images: np.ndarray,
    model: DigitMLP | LogisticRegressionBaseline,
    brightness_threshold: float = 0.02,
    use_calibrated: bool = True,
    use_nn: bool = True,
    fallback_grid: np.ndarray | None = None,
    seed: int = 42,
) -> GridRecognitionResult:
    """Recognize all nonblank cells of a 9x9 Sudoku mosaic (shape (9, 9, 28, 28)).

    When `use_nn=True` (default):
        Uses the trained neural network to predict probabilities over digits 1..9
        for every nonblank cell detected by the brightness threshold.
    When `use_nn=False` (demonstration / ablation mode):
        Bypasses neural network probability guidance. If `fallback_grid` is provided,
        uses those initial digits with a flat/uninformative distribution over alternatives;
        otherwise assigns a random digit 1..9 with uniform probability 1/9 across all 9 digits.
    """
    if cell_images.shape != (9, 9, 28, 28):
        raise ValueError(f"Expected cell_images of shape (9, 9, 28, 28), got {cell_images.shape}")

    blank_mask = detect_blank_cells_grid(cell_images, brightness_threshold=brightness_threshold)
    pred_grid = np.zeros((9, 9), dtype=np.int64)
    prob_grid = np.full((9, 9, 9), 1.0 / 9.0, dtype=np.float64)
    conf_grid = np.ones((9, 9), dtype=np.float64)

    rng = np.random.default_rng(seed)

    # Batch predict all nonblank cells for speed
    nonblank_coords = [
        (r, c) for r in range(9) for c in range(9) if not blank_mask[r, c]
    ]

    if nonblank_coords and use_nn:
        batch_imgs = np.stack([cell_images[r, c] for r, c in nonblank_coords], axis=0)
        batch_probs = model.predict_proba(batch_imgs, use_temperature=use_calibrated).astype(np.float64)
        for idx, (r, c) in enumerate(nonblank_coords):
            p = batch_probs[idx]
            cls_idx = int(np.argmax(p))
            d = int(class_to_digit(cls_idx))
            pred_grid[r, c] = d
            prob_grid[r, c] = p
            conf_grid[r, c] = float(p[cls_idx])
    elif nonblank_coords and not use_nn:
        for r, c in nonblank_coords:
            if fallback_grid is not None and int(fallback_grid[r, c]) != 0:
                d = int(fallback_grid[r, c])
            else:
                d = int(rng.integers(1, 10))
            # Uninformative uniform probabilities over alternatives
            p = np.full(9, (1.0 - 0.12) / 8.0, dtype=np.float64)
            p[d - 1] = 0.12
            pred_grid[r, c] = d
            prob_grid[r, c] = p
            conf_grid[r, c] = float(p[d - 1])

    cell_predictions: List[List[CellPrediction]] = []
    for r in range(9):
        row_preds: List[CellPrediction] = []
        for c in range(9):
            if blank_mask[r, c]:
                row_preds.append(
                    CellPrediction(
                        row=r,
                        col=c,
                        is_blank=True,
                        predicted_digit=0,
                        confidence=1.0,
                        probabilities=[round(1.0 / 9.0, 6)] * 9,
                        top_candidates=[],
                    )
                )
            else:
                p = prob_grid[r, c]
                d = int(pred_grid[r, c])
                conf = float(p[d - 1])
                ranked = sorted(
                    [(int(digit), float(p[digit - 1])) for digit in range(1, 10)],
                    key=lambda item: item[1],
                    reverse=True,
                )
                row_preds.append(
                    CellPrediction(
                        row=r,
                        col=c,
                        is_blank=False,
                        predicted_digit=d,
                        confidence=conf,
                        probabilities=[round(float(x), 6) for x in p],
                        top_candidates=ranked,
                    )
                )
        cell_predictions.append(row_preds)

    return GridRecognitionResult(
        predicted_grid=pred_grid,
        probability_grid=prob_grid,
        confidence_grid=conf_grid,
        blank_mask=blank_mask,
        cell_predictions=cell_predictions,
        nn_enabled=use_nn,
    )
