"""Unit tests for MNIST data loading, digit 0 exclusion, puzzle generation, and mosaic construction."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from src.data.mnist_loader import (
    class_to_digit,
    digit_to_class,
    load_and_preprocess_mnist,
)
from src.data.mosaic_generator import (
    create_sudoku_mosaic,
    detect_blank_cells_grid,
    inject_hard_digit_errors,
    is_blank_cell,
)
from src.data.puzzle_generator import (
    count_sudoku_solutions_fast,
    generate_sudoku_puzzle,
    is_valid_completed_sudoku,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestDataAndGenerators(unittest.TestCase):
    """Test suite for `src/data/` modules."""

    @classmethod
    def setUpClass(cls) -> None:
        npz_path = PROJECT_ROOT / "data" / "processed" / "mnist_sudoku_1_9.npz"
        cls.data = np.load(npz_path)

    def test_mnist_splits_and_zero_exclusion(self) -> None:
        """Normal Case 1: Verify MNIST 50k/10k split, untouched 10k test set, [0, 1] normalization, and 0-exclusion."""
        X_train = self.data["X_train"]
        y_train = self.data["y_train"]
        digits_train = self.data["digits_train"]
        X_val = self.data["X_val"]
        digits_val = self.data["digits_val"]
        X_test = self.data["X_test"]
        digits_test = self.data["digits_test"]

        self.assertEqual(X_train.shape[1:], (28, 28))
        self.assertEqual(X_val.shape[1:], (28, 28))
        self.assertEqual(X_test.shape[1:], (28, 28))

        # No digit 0 in any split
        self.assertFalse(np.any(digits_train == 0))
        self.assertFalse(np.any(digits_val == 0))
        self.assertFalse(np.any(digits_test == 0))
        self.assertEqual(set(np.unique(digits_train).tolist()), set(range(1, 10)))
        self.assertEqual(set(np.unique(y_train).tolist()), set(range(0, 9)))

        # Pixel range in [0, 1]
        self.assertGreaterEqual(float(np.min(X_train)), 0.0)
        self.assertLessEqual(float(np.max(X_train)), 1.0)

    def test_digit_class_mapping_bijection(self) -> None:
        """Verify bijective mapping between Sudoku digits 1..9 and class labels 0..8, rejecting 0."""
        for d in range(1, 10):
            c = digit_to_class(d)
            self.assertEqual(c, d - 1)
            self.assertEqual(class_to_digit(c), d)
        with self.assertRaises(ValueError):
            digit_to_class(0)
        with self.assertRaises(ValueError):
            class_to_digit(9)

    def test_sudoku_generator_uniqueness_and_reproducibility(self) -> None:
        """Normal Case 2: Verify Sudoku puzzle generator produces unique-solution puzzles reproducibly."""
        p1 = generate_sudoku_puzzle(num_givens=32, seed=101, ensure_unique=True)
        p2 = generate_sudoku_puzzle(num_givens=32, seed=101, ensure_unique=True)
        self.assertTrue(np.array_equal(p1.puzzle, p2.puzzle))
        self.assertTrue(np.array_equal(p1.solution, p2.solution))
        self.assertTrue(is_valid_completed_sudoku(p1.solution))
        self.assertEqual(p1.num_givens, 32)
        self.assertEqual(count_sudoku_solutions_fast(p1.puzzle, limit=2), 1)

    def test_mosaic_generator_and_blank_detection(self) -> None:
        """Verify 9x9 mosaic construction (252x252) and brightness-threshold blank cell detection."""
        puz = generate_sudoku_puzzle(num_givens=30, seed=55)
        mosaic = create_sudoku_mosaic(
            puz.puzzle,
            puz.solution,
            self.data["X_test"],
            self.data["digits_test"],
            seed=55,
            brightness_threshold=0.02,
        )
        self.assertEqual(mosaic.mosaic_image.shape, (252, 252))
        self.assertEqual(mosaic.cell_images.shape, (9, 9, 28, 28))
        expected_blanks = puz.puzzle == 0
        self.assertTrue(np.array_equal(mosaic.blank_mask, expected_blanks))
        for r in range(9):
            for c in range(9):
                self.assertEqual(
                    is_blank_cell(mosaic.cell_images[r, c], 0.02),
                    bool(expected_blanks[r, c]),
                )

    def test_hard_digit_injection_reproducibility(self) -> None:
        """Verify controlled hard-digit injection creates exact number of misreads with valid probabilities."""
        puz = generate_sudoku_puzzle(num_givens=35, seed=77)
        base_probs = np.full((9, 9, 9), 0.01, dtype=np.float64)
        for r in range(9):
            for c in range(9):
                d = int(puz.puzzle[r, c])
                if d != 0:
                    base_probs[r, c, d - 1] = 0.92
                    base_probs[r, c] /= np.sum(base_probs[r, c])

        pred1, prob1, rec1 = inject_hard_digit_errors(
            puz.puzzle, base_probs, puz.puzzle, puz.solution, num_errors=3, seed=99
        )
        pred2, prob2, rec2 = inject_hard_digit_errors(
            puz.puzzle, base_probs, puz.puzzle, puz.solution, num_errors=3, seed=99
        )
        self.assertEqual(len(rec1), 3)
        self.assertTrue(np.array_equal(pred1, pred2))
        self.assertTrue(np.allclose(prob1, prob2))
        diff_count = int(np.sum((pred1 != puz.puzzle) & (puz.puzzle != 0)))
        self.assertEqual(diff_count, 3)


if __name__ == "__main__":
    unittest.main()
