"""Unit tests for Neural Network digit classifier, Logistic Regression baseline, and calibration."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from src.nn.calibration import compute_calibration_metrics, fit_temperature_scaling
from src.nn.model import DigitMLP, LogisticRegressionBaseline
from src.nn.predict import predict_single_image, recognize_sudoku_mosaic

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestNeuralNetwork(unittest.TestCase):
    """Test suite for `src/nn/` modules."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mlp = DigitMLP.load(PROJECT_ROOT / "models" / "digit_mlp.npz")
        cls.logreg = LogisticRegressionBaseline.load(PROJECT_ROOT / "models" / "logreg_baseline.npz")
        cls.data = np.load(PROJECT_ROOT / "data" / "processed" / "mnist_sudoku_1_9.npz")

    def test_mlp_and_baseline_test_accuracy(self) -> None:
        """Normal Case 3: Verify trained MLP achieves >97% test accuracy and outperforms Logistic Regression."""
        X_test = self.data["X_test"]
        y_test = self.data["y_test"]

        mlp_probs = self.mlp.predict_proba(X_test, use_temperature=True)
        lr_probs = self.logreg.predict_proba(X_test, use_temperature=True)

        # Verify probabilities sum to 1 and have shape (N, 9)
        self.assertEqual(mlp_probs.shape, (len(y_test), 9))
        self.assertTrue(np.allclose(np.sum(mlp_probs, axis=1), 1.0, atol=1e-5))
        self.assertTrue(np.all(mlp_probs >= 0.0))

        mlp_acc = float(np.mean(np.argmax(mlp_probs, axis=1) == y_test))
        lr_acc = float(np.mean(np.argmax(lr_probs, axis=1) == y_test))

        self.assertGreater(mlp_acc, 0.97)
        self.assertGreater(mlp_acc, lr_acc)

    def test_temperature_scaling_calibration_improvement(self) -> None:
        """Verify temperature scaling fitted on validation data maintains or reduces ECE on test data."""
        X_test = self.data["X_test"]
        y_test = self.data["y_test"]

        probs_uncal = self.mlp.predict_proba(X_test, use_temperature=False)
        probs_cal = self.mlp.predict_proba(X_test, use_temperature=True)

        cal_before = compute_calibration_metrics(probs_uncal, y_test)
        cal_after = compute_calibration_metrics(probs_cal, y_test)

        self.assertLessEqual(cal_after.ece, cal_before.ece + 1e-4)
        self.assertLess(cal_after.ece, 0.01)

    def test_single_image_prediction_consistency(self) -> None:
        """Verify single-image prediction returns valid digit in 1..9 and matching confidence."""
        img = self.data["X_test"][0]
        pred_digit, conf, probs = predict_single_image(self.mlp, img, use_calibrated=True)
        self.assertIn(pred_digit, range(1, 10))
        self.assertEqual(len(probs), 9)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
        self.assertAlmostEqual(conf, float(probs[pred_digit - 1]), places=6)
        self.assertEqual(pred_digit, int(np.argmax(probs)) + 1)


if __name__ == "__main__":
    unittest.main()
