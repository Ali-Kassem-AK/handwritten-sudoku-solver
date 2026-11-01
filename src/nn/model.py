"""Multi-Layer Perceptron (MLP) and Multinomial Logistic Regression baseline for Sudoku digits 1-9.

Follows the architecture and mathematical formulation in `Neural Networks Lecture.pdf`
(Cyrill Stachniss / Michael Nielsen, slides 40-72):
- Input layer: 784 features (flattened 28x28 grayscale pixel intensities in [0, 1]).
- Hidden Layer 1: 128 neurons with ReLU activation: a^(1) = ReLU(W^(1) a^(0) + b^(1)).
- Hidden Layer 2: 64 neurons with ReLU activation:  a^(2) = ReLU(W^(2) a^(1) + b^(2)).
- Output Layer: 9 logits (corresponding to Sudoku digits 1..9, mapped as class c = d - 1)
  followed by Softmax activation: P(y = c | x) = exp(z_c / T) / sum_k exp(z_k / T).
- Baseline Classifier: Multinomial Logistic Regression (linear layer 784 -> 9 with Softmax
  and L2 regularization).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit: ReLU(z) = max(0, z)."""
    return np.maximum(0.0, z)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable Softmax over last axis with optional temperature scaling T > 0."""
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    scaled = logits / float(temperature)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, targets: np.ndarray, eps: float = 1e-12) -> float:
    """Mean categorical cross-entropy loss over batch."""
    n = len(targets)
    clipped = np.clip(probs[np.arange(n), targets], eps, 1.0)
    return float(-np.mean(np.log(clipped)))


@dataclass
class DigitMLP:
    """3-Layer Feedforward Neural Network (784 -> 128 -> 64 -> 9) with ReLU and Softmax."""

    W1: np.ndarray  # shape (784, 128)
    b1: np.ndarray  # shape (128,)
    W2: np.ndarray  # shape (128, 64)
    b2: np.ndarray  # shape (64,)
    W3: np.ndarray  # shape (64, 9)
    b3: np.ndarray  # shape (9,)
    temperature: float = 1.0
    seed: int = 42

    @classmethod
    def initialize(
        cls,
        input_dim: int = 784,
        hidden1: int = 128,
        hidden2: int = 64,
        num_classes: int = 9,
        seed: int = 42,
    ) -> "DigitMLP":
        """Initialize weights using He (Kaiming) normal initialization for ReLU layers."""
        rng = np.random.default_rng(seed)
        W1 = rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden1)).astype(np.float32)
        b1 = np.zeros(hidden1, dtype=np.float32)
        W2 = rng.normal(0.0, np.sqrt(2.0 / hidden1), size=(hidden1, hidden2)).astype(np.float32)
        b2 = np.zeros(hidden2, dtype=np.float32)
        W3 = rng.normal(0.0, np.sqrt(2.0 / hidden2), size=(hidden2, num_classes)).astype(np.float32)
        b3 = np.zeros(num_classes, dtype=np.float32)
        return cls(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, temperature=1.0, seed=seed)

    def forward_logits(self, X: np.ndarray) -> np.ndarray:
        """Compute raw unscaled logits z^(3) for input X of shape (N, 28, 28) or (N, 784)."""
        x_flat = X.reshape(X.shape[0], -1).astype(np.float32)
        z1 = x_flat @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        z3 = a2 @ self.W3 + self.b3
        return z3

    def forward_cache(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass returning intermediate activations for backpropagation."""
        x_flat = X.reshape(X.shape[0], -1).astype(np.float32)
        z1 = x_flat @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        z3 = a2 @ self.W3 + self.b3
        return x_flat, z1, a1, z2, a2, z3

    def predict_proba(self, X: np.ndarray, use_temperature: bool = True) -> np.ndarray:
        """Return probability distribution over digits 1..9 (shape (N, 9))."""
        logits = self.forward_logits(X)
        temp = self.temperature if use_temperature else 1.0
        return softmax(logits, temperature=temp)

    def save(self, model_path: str | Path, config_path: Optional[str | Path] = None, extra_meta: Optional[Dict[str, object]] = None) -> None:
        """Save trained model parameters (.npz) and configuration (.json)."""
        m_path = Path(model_path)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            m_path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            W3=self.W3,
            b3=self.b3,
            temperature=np.array([self.temperature], dtype=np.float32),
            seed=np.array([self.seed], dtype=np.int64),
        )
        if config_path is not None:
            c_path = Path(config_path)
            cfg: Dict[str, object] = {
                "model_type": "DigitMLP",
                "architecture": [
                    int(self.W1.shape[0]),
                    int(self.W1.shape[1]),
                    int(self.W2.shape[1]),
                    int(self.W3.shape[1]),
                ],
                "activations": ["ReLU", "ReLU", "Softmax"],
                "temperature": float(self.temperature),
                "seed": int(self.seed),
                "num_parameters": int(
                    self.W1.size + self.b1.size + self.W2.size + self.b2.size + self.W3.size + self.b3.size
                ),
            }
            if extra_meta:
                cfg.update(extra_meta)
            with open(c_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

    @classmethod
    def load(cls, model_path: str | Path) -> "DigitMLP":
        """Load trained DigitMLP from .npz file."""
        data = np.load(Path(model_path))
        return cls(
            W1=data["W1"].astype(np.float32),
            b1=data["b1"].astype(np.float32),
            W2=data["W2"].astype(np.float32),
            b2=data["b2"].astype(np.float32),
            W3=data["W3"].astype(np.float32),
            b3=data["b3"].astype(np.float32),
            temperature=float(data["temperature"][0]),
            seed=int(data["seed"][0]),
        )


@dataclass
class LogisticRegressionBaseline:
    """Multinomial Logistic Regression baseline (784 -> 9 linear model with Softmax)."""

    W: np.ndarray  # shape (784, 9)
    b: np.ndarray  # shape (9,)
    temperature: float = 1.0
    seed: int = 42

    @classmethod
    def initialize(
        cls, input_dim: int = 784, num_classes: int = 9, seed: int = 42
    ) -> "LogisticRegressionBaseline":
        """Initialize weights near zero with fixed seed."""
        rng = np.random.default_rng(seed)
        W = rng.normal(0.0, 0.01, size=(input_dim, num_classes)).astype(np.float32)
        b = np.zeros(num_classes, dtype=np.float32)
        return cls(W=W, b=b, temperature=1.0, seed=seed)

    def forward_logits(self, X: np.ndarray) -> np.ndarray:
        """Compute raw logits X @ W + b."""
        x_flat = X.reshape(X.shape[0], -1).astype(np.float32)
        return x_flat @ self.W + self.b

    def predict_proba(self, X: np.ndarray, use_temperature: bool = True) -> np.ndarray:
        """Return probability distribution over digits 1..9 (shape (N, 9))."""
        logits = self.forward_logits(X)
        temp = self.temperature if use_temperature else 1.0
        return softmax(logits, temperature=temp)

    def save(self, model_path: str | Path) -> None:
        """Save logistic regression parameters to .npz."""
        m_path = Path(model_path)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            m_path,
            W=self.W,
            b=self.b,
            temperature=np.array([self.temperature], dtype=np.float32),
            seed=np.array([self.seed], dtype=np.int64),
        )

    @classmethod
    def load(cls, model_path: str | Path) -> "LogisticRegressionBaseline":
        """Load logistic regression baseline from .npz."""
        data = np.load(Path(model_path))
        return cls(
            W=data["W"].astype(np.float32),
            b=data["b"].astype(np.float32),
            temperature=float(data["temperature"][0]),
            seed=int(data["seed"][0]),
        )
