"""Sudoku 9x9 image mosaic generator and controlled hard-digit error injection.

Implements Section 5 requirements:
- Selects handwritten digit images from MNIST corresponding to the puzzle's nonblank digits.
- Arranges them into a 9x9 grid (252x252 grayscale mosaic image, made of 9x9 tiles of 28x28).
- Detects blank cells using a simple documented brightness threshold (`mean_intensity < threshold`,
  default `0.02`).
- Preserves the relationship between each image cell, its MNIST source index, the generated
  puzzle, and its ground-truth digit.
- Implements a documented hard-digit injection mechanism that deliberately creates
  low-confidence or incorrect predictions for selected nonblank cells without altering
  the reported neural network accuracy on genuine MNIST test evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class InjectedErrorRecord:
    """Metadata record for a single deliberately injected recognition error."""

    row: int
    col: int
    ground_truth_digit: int
    original_predicted_digit: int
    injected_predicted_digit: int
    original_probabilities: List[float]  # length 9, for digits 1..9
    modified_probabilities: List[float]  # length 9, for digits 1..9
    injection_seed: int
    total_injected_errors: int


@dataclass
class SudokuMosaic:
    """Represents a 9x9 handwritten Sudoku mosaic and its cell-level metadata."""

    mosaic_image: np.ndarray  # shape (252, 252), float32 in [0.0, 1.0]
    cell_images: np.ndarray  # shape (9, 9, 28, 28), float32 in [0.0, 1.0]
    puzzle_grid: np.ndarray  # shape (9, 9), int64 in {0..9}
    solution_grid: np.ndarray  # shape (9, 9), int64 in {1..9}
    mnist_indices: np.ndarray  # shape (9, 9), int64 (-1 for blank cells)
    blank_mask: np.ndarray  # shape (9, 9), bool (True if blank)
    brightness_threshold: float
    seed: int
    injected_errors: List[InjectedErrorRecord] = field(default_factory=list)

    def to_metadata_dict(self) -> Dict[str, object]:
        """Serialize non-image metadata to a JSON-compatible dictionary."""
        return {
            "puzzle_grid": self.puzzle_grid.tolist(),
            "solution_grid": self.solution_grid.tolist(),
            "mnist_indices": self.mnist_indices.tolist(),
            "blank_mask": self.blank_mask.tolist(),
            "brightness_threshold": float(self.brightness_threshold),
            "seed": int(self.seed),
            "num_givens": int(np.sum(self.puzzle_grid != 0)),
            "injected_errors": [asdict(err) for err in self.injected_errors],
        }


def is_blank_cell(cell_img: np.ndarray, brightness_threshold: float = 0.02) -> bool:
    """Detect whether a 28x28 cell image is blank using a mean-brightness rule.

    Since normalized MNIST digit images have foreground strokes with mean intensity
    typically in [0.06, 0.35] and blank cells are zero-filled (mean 0.0), any cell
    with `mean(cell_img) < brightness_threshold` is classified as blank without
    invoking the neural network.
    """
    return float(np.mean(cell_img)) < brightness_threshold


def detect_blank_cells_grid(
    cell_images: np.ndarray, brightness_threshold: float = 0.02
) -> np.ndarray:
    """Return a 9x9 boolean mask where True indicates a blank cell."""
    means = np.mean(cell_images, axis=(2, 3))
    return means < brightness_threshold


def create_sudoku_mosaic(
    puzzle_grid: np.ndarray,
    solution_grid: np.ndarray,
    mnist_images: np.ndarray,
    mnist_digits: np.ndarray,
    seed: int = 42,
    brightness_threshold: float = 0.02,
) -> SudokuMosaic:
    """Construct a 9x9 Sudoku mosaic (252x252) from real MNIST digit images.

    Args:
        puzzle_grid: 9x9 integer array (0 for blank, 1..9 for givens).
        solution_grid: 9x9 integer array (1..9 ground-truth solution).
        mnist_images: Pool of normalized MNIST images (N, 28, 28) in [0, 1].
        mnist_digits: Corresponding digit labels (N,) in {1..9}.
        seed: Random seed for selecting MNIST handwriting samples.
        brightness_threshold: Mean pixel intensity threshold for blank detection.
    """
    rng = np.random.default_rng(seed)
    cell_images = np.zeros((9, 9, 28, 28), dtype=np.float32)
    mosaic_image = np.zeros((9 * 28, 9 * 28), dtype=np.float32)
    mnist_indices = np.full((9, 9), -1, dtype=np.int64)

    # Pre-index MNIST pool by digit 1..9
    digit_pools: Dict[int, np.ndarray] = {}
    for d in range(1, 10):
        pool = np.where(mnist_digits == d)[0]
        if len(pool) == 0:
            raise ValueError(f"No MNIST images found for digit {d}")
        digit_pools[d] = pool

    for r in range(9):
        for c in range(9):
            d = int(puzzle_grid[r, c])
            if d != 0:
                pool = digit_pools[d]
                chosen_idx = int(pool[rng.integers(0, len(pool))])
                img = mnist_images[chosen_idx].astype(np.float32)
                # Ensure nonblank image exceeds brightness threshold
                if float(np.mean(img)) < brightness_threshold:
                    img = np.clip(img * 2.0 + brightness_threshold, 0.0, 1.0)
                cell_images[r, c] = img
                mosaic_image[r * 28 : (r + 1) * 28, c * 28 : (c + 1) * 28] = img
                mnist_indices[r, c] = chosen_idx

    blank_mask = detect_blank_cells_grid(cell_images, brightness_threshold)

    return SudokuMosaic(
        mosaic_image=mosaic_image,
        cell_images=cell_images,
        puzzle_grid=puzzle_grid.copy(),
        solution_grid=solution_grid.copy(),
        mnist_indices=mnist_indices,
        blank_mask=blank_mask,
        brightness_threshold=brightness_threshold,
        seed=seed,
        injected_errors=[],
    )


def inject_hard_digit_errors(
    pred_grid: np.ndarray,
    prob_grid: np.ndarray,
    puzzle_grid: np.ndarray,
    solution_grid: np.ndarray,
    num_errors: int,
    seed: int = 42,
    primary_prob_range: Tuple[float, float] = (0.48, 0.56),
    true_prob_range: Tuple[float, float] = (0.36, 0.44),
) -> Tuple[np.ndarray, np.ndarray, List[InjectedErrorRecord]]:
    """Deliberately inject `num_errors` realistic low-confidence misreads into nonblank cells.

    Documented Procedure (Section 3 & Section 5):
    - Selects `num_errors` distinct nonblank cells `(r, c)` using the seeded RNG.
    - Preferentially picks cells that are currently correctly recognized so that
      exact ground-truth error counts are controlled across experimental conditions
      (0, 1, 2, 3, 4 misreads).
    - For each selected cell `(r, c)` with ground-truth digit `d_true`:
      1. Chooses an incorrect digit `d_wrong != d_true`. To make the conflict
         realistic (like the Section 3 example where a 7 is misread as a 1 with
         52% vs 45% probability, creating a duplicate digit in the same row/col/box),
         prefers a digit `d_wrong` that already appears as a given in the same row,
         column, or 3x3 box, or the network's second-most likely digit.
      2. Constructs a calibrated ambiguous probability distribution where:
         - `P(d_wrong) = p_wrong` sampled in `primary_prob_range` (e.g. ~0.52),
         - `P(d_true)  = p_true`  sampled in `true_prob_range` (e.g. ~0.42),
         - Remaining probability mass `1 - p_wrong - p_true` is distributed
           across the other 7 digits proportional to their original NN probabilities.
    - Records full metadata (`InjectedErrorRecord`) for every modified cell.
    """
    mod_pred = pred_grid.copy().astype(np.int64)
    mod_prob = prob_grid.copy().astype(np.float64)
    records: List[InjectedErrorRecord] = []

    nonblank_coords = [
        (int(r), int(c))
        for r in range(9)
        for c in range(9)
        if int(puzzle_grid[r, c]) != 0
    ]

    if num_errors <= 0 or len(nonblank_coords) == 0:
        # Ensure any unintended natural NN errors are snapped to ground truth when num_errors == 0
        # in controlled misread-level experiments if needed, or leave as is.
        # For controlled experiments where exact k misreads are requested:
        for r, c in nonblank_coords:
            true_d = int(puzzle_grid[r, c])
            if int(mod_pred[r, c]) != true_d:
                # Restore ground truth as top prediction so 0-misread condition has 0 misreads
                p = mod_prob[r, c].copy()
                top_idx = int(np.argmax(p))
                true_idx = true_d - 1
                p[top_idx], p[true_idx] = p[true_idx], max(p[top_idx], 0.85)
                p = p / np.sum(p)
                mod_prob[r, c] = p
                mod_pred[r, c] = true_d
        return mod_pred, mod_prob, records

    rng = np.random.default_rng(seed)

    # First ensure non-selected cells start from clean ground-truth top prediction
    # so that `num_errors` controls the exact number of misreads in the grid.
    for r, c in nonblank_coords:
        true_d = int(puzzle_grid[r, c])
        if int(mod_pred[r, c]) != true_d:
            p = mod_prob[r, c].copy()
            top_idx = int(np.argmax(p))
            true_idx = true_d - 1
            p[top_idx], p[true_idx] = p[true_idx], max(p[top_idx], 0.85)
            p = p / np.sum(p)
            mod_prob[r, c] = p
            mod_pred[r, c] = true_d

    actual_k = min(num_errors, len(nonblank_coords))
    chosen_indices = rng.choice(len(nonblank_coords), size=actual_k, replace=False)
    chosen_cells = [nonblank_coords[int(i)] for i in chosen_indices]

    for r, c in chosen_cells:
        true_d = int(puzzle_grid[r, c])
        orig_pred_d = int(mod_pred[r, c])
        orig_probs = mod_prob[r, c].copy()

        # Find digits already present in the same row, column, or box to create a direct conflict
        peer_digits = set()
        for j in range(9):
            if j != c and int(puzzle_grid[r, j]) != 0:
                peer_digits.add(int(puzzle_grid[r, j]))
        for i in range(9):
            if i != r and int(puzzle_grid[i, c]) != 0:
                peer_digits.add(int(puzzle_grid[i, c]))
        br, bc = (r // 3) * 3, (c // 3) * 3
        for i in range(br, br + 3):
            for j in range(bc, bc + 3):
                if (i != r or j != c) and int(puzzle_grid[i, j]) != 0:
                    peer_digits.add(int(puzzle_grid[i, j]))

        peer_digits.discard(true_d)
        if peer_digits:
            # Pick the peer digit that had the highest original NN probability (most plausible confusion)
            peer_list = sorted(peer_digits)
            peer_probs = [orig_probs[d - 1] for d in peer_list]
            # Mix deterministic plausibility with seeded randomness
            if rng.random() < 0.75:
                wrong_d = int(peer_list[int(np.argmax(peer_probs))])
            else:
                wrong_d = int(rng.choice(peer_list))
        else:
            other_digits = [d for d in range(1, 10) if d != true_d]
            wrong_d = int(rng.choice(other_digits))

        p_wrong = float(rng.uniform(primary_prob_range[0], primary_prob_range[1]))
        p_true = float(rng.uniform(true_prob_range[0], min(true_prob_range[1], p_wrong - 0.03)))
        rem_mass = max(1e-4, 1.0 - p_wrong - p_true)

        new_probs = orig_probs.copy()
        new_probs[wrong_d - 1] = 0.0
        new_probs[true_d - 1] = 0.0
        s_other = float(np.sum(new_probs))
        if s_other > 1e-12:
            new_probs = (new_probs / s_other) * rem_mass
        else:
            for d in range(1, 10):
                if d not in (wrong_d, true_d):
                    new_probs[d - 1] = rem_mass / 7.0

        new_probs[wrong_d - 1] = p_wrong
        new_probs[true_d - 1] = p_true
        new_probs = new_probs / np.sum(new_probs)

        mod_prob[r, c] = new_probs
        mod_pred[r, c] = wrong_d

        records.append(
            InjectedErrorRecord(
                row=r,
                col=c,
                ground_truth_digit=true_d,
                original_predicted_digit=orig_pred_d,
                injected_predicted_digit=wrong_d,
                original_probabilities=[round(float(x), 6) for x in orig_probs],
                modified_probabilities=[round(float(x), 6) for x in new_probs],
                injection_seed=seed,
                total_injected_errors=actual_k,
            )
        )

    return mod_pred, mod_prob, records
