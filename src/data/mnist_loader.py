"""MNIST IDX file loader, verifier, and 1-9 digit preprocessor.

Follows the specification in Section 4.A and Section 5:
- Reads standard IDX3/IDX1 gzip files:
  * train-images-idx3-ubyte.gz (60,000 28x28 grayscale images)
  * train-labels-idx1-ubyte.gz (60,000 labels)
  * t10k-images-idx3-ubyte.gz  (10,000 28x28 grayscale images)
  * t10k-labels-idx1-ubyte.gz  (10,000 labels)
- Creates the 50,000 / 10,000 training-validation split from the 60,000 training set
  using a fixed random seed, and preserves the untouched 10,000 official test set.
- Excludes digit 0 for Sudoku digit recognition (retaining digits 1..9) and maps
  Sudoku digits d in {1..9} to 0-indexed class labels c = d - 1 in {0..8}.
- Normalizes pixel intensities to float32 in [0.0, 1.0].
"""

from __future__ import annotations

import gzip
import hashlib
import os
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

MNIST_FILES: Dict[str, Dict[str, object]] = {
    "train_images": {
        "filename": "train-images-idx3-ubyte.gz",
        "magic": 2051,
        "count": 60000,
        "urls": [
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        ],
    },
    "train_labels": {
        "filename": "train-labels-idx1-ubyte.gz",
        "magic": 2049,
        "count": 60000,
        "urls": [
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        ],
    },
    "test_images": {
        "filename": "t10k-images-idx3-ubyte.gz",
        "magic": 2051,
        "count": 10000,
        "urls": [
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        ],
    },
    "test_labels": {
        "filename": "t10k-labels-idx1-ubyte.gz",
        "magic": 2049,
        "count": 10000,
        "urls": [
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
        ],
    },
}


@dataclass
class MNISTSplit:
    """Container for preprocessed MNIST splits (digits 1-9)."""

    X_train: np.ndarray  # shape (N_train, 28, 28), float32 in [0, 1]
    y_train: np.ndarray  # shape (N_train,), int64 in {0..8} (digit - 1)
    digits_train: np.ndarray  # shape (N_train,), int64 in {1..9}
    X_val: np.ndarray  # shape (N_val, 28, 28), float32 in [0, 1]
    y_val: np.ndarray  # shape (N_val,), int64 in {0..8}
    digits_val: np.ndarray  # shape (N_val,), int64 in {1..9}
    X_test: np.ndarray  # shape (N_test, 28, 28), float32 in [0, 1]
    y_test: np.ndarray  # shape (N_test,), int64 in {0..8}
    digits_test: np.ndarray  # shape (N_test,), int64 in {1..9}
    metadata: Dict[str, object]


def digit_to_class(digit: int | np.ndarray) -> int | np.ndarray:
    """Map Sudoku digit d in {1..9} to class index c = d - 1 in {0..8}."""
    if isinstance(digit, np.ndarray):
        if np.any((digit < 1) | (digit > 9)):
            raise ValueError("Sudoku digits must be in 1..9 (digit 0 is excluded).")
        return (digit - 1).astype(np.int64)
    if not (1 <= int(digit) <= 9):
        raise ValueError(f"Sudoku digit must be in 1..9, got {digit}.")
    return int(digit) - 1


def class_to_digit(cls_idx: int | np.ndarray) -> int | np.ndarray:
    """Map class index c in {0..8} back to Sudoku digit d = c + 1 in {1..9}."""
    if isinstance(cls_idx, np.ndarray):
        if np.any((cls_idx < 0) | (cls_idx > 8)):
            raise ValueError("Class indices must be in 0..8.")
        return (cls_idx + 1).astype(np.int64)
    if not (0 <= int(cls_idx) <= 8):
        raise ValueError(f"Class index must be in 0..8, got {cls_idx}.")
    return int(cls_idx) + 1


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def ensure_mnist_raw(raw_dir: str | Path, fallback_dirs: Tuple[str | Path, ...] = ()) -> Dict[str, Path]:
    """Ensure the four official MNIST .gz files exist in raw_dir and verify headers.

    If a file is not present in raw_dir, checks fallback_dirs first, then attempts
    to download from official mirrors if network access is available.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    resolved_paths: Dict[str, Path] = {}

    for key, spec in MNIST_FILES.items():
        fname = str(spec["filename"])
        target = raw_path / fname
        if not target.exists():
            # Check fallback directories
            for fb in fallback_dirs:
                cand = Path(fb) / fname
                if cand.exists():
                    target.write_bytes(cand.read_bytes())
                    break
        if not target.exists():
            # Attempt download if network is available
            last_err = None
            for url in spec["urls"]:  # type: ignore[attr-defined]
                try:
                    urllib.request.urlretrieve(url, target)
                    break
                except Exception as exc:  # pragma: no cover
                    last_err = exc
            if not target.exists():
                raise FileNotFoundError(
                    f"MNIST file {fname} not found in {raw_path} and download failed ({last_err})."
                )
        resolved_paths[key] = target

    return resolved_paths


def read_idx_images(filepath: str | Path) -> np.ndarray:
    """Read an IDX3 gzip file and return uint8 array of shape (N, 28, 28)."""
    path = Path(filepath)
    with gzip.open(path, "rb") as f:
        header = f.read(16)
        if len(header) != 16:
            raise ValueError(f"Invalid IDX3 image header in {path}")
        magic, num_images, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Expected magic 2051 in {path}, got {magic}")
        buf = f.read()
        expected_bytes = num_images * rows * cols
        if len(buf) != expected_bytes:
            raise ValueError(
                f"Byte count mismatch in {path}: expected {expected_bytes}, got {len(buf)}"
            )
        images = np.frombuffer(buf, dtype=np.uint8).reshape(num_images, rows, cols)
    return images


def read_idx_labels(filepath: str | Path) -> np.ndarray:
    """Read an IDX1 gzip file and return uint8 array of shape (N,)."""
    path = Path(filepath)
    with gzip.open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"Invalid IDX1 label header in {path}")
        magic, num_labels = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Expected magic 2049 in {path}, got {magic}")
        buf = f.read()
        if len(buf) != num_labels:
            raise ValueError(
                f"Label count mismatch in {path}: expected {num_labels}, got {len(buf)}"
            )
        labels = np.frombuffer(buf, dtype=np.uint8).copy()
    return labels


def load_and_preprocess_mnist(
    raw_dir: str | Path,
    seed: int = 42,
    train_split_size: int = 50000,
    val_split_size: int = 10000,
    fallback_dirs: Tuple[str | Path, ...] = (),
) -> MNISTSplit:
    """Load raw MNIST files, split 60,000 into 50,000/10,000, exclude digit 0, and normalize to [0, 1].

    Procedure:
    1. Load raw 60,000 training images/labels and 10,000 untouched test images/labels.
    2. Permute the 60,000 training indices using a seeded NumPy Generator (`seed`)
       and partition into `train_split_size` (50,000) training samples and
       `val_split_size` (10,000) validation samples.
    3. Filter out digit 0 (`label != 0`) across train, validation, and test splits.
    4. Normalize pixel values from uint8 [0, 255] to float32 [0.0, 1.0].
    5. Map remaining digits d in {1..9} to class labels c = d - 1 in {0..8}.
    """
    paths = ensure_mnist_raw(raw_dir, fallback_dirs=fallback_dirs)

    raw_train_img = read_idx_images(paths["train_images"])
    raw_train_lbl = read_idx_labels(paths["train_labels"])
    raw_test_img = read_idx_images(paths["test_images"])
    raw_test_lbl = read_idx_labels(paths["test_labels"])

    if len(raw_train_img) != train_split_size + val_split_size:
        raise ValueError(
            f"Expected {train_split_size + val_split_size} raw train samples, got {len(raw_train_img)}"
        )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(raw_train_img))
    train_idx = indices[:train_split_size]
    val_idx = indices[train_split_size : train_split_size + val_split_size]

    train_img_50k = raw_train_img[train_idx]
    train_lbl_50k = raw_train_lbl[train_idx]
    val_img_10k = raw_train_img[val_idx]
    val_lbl_10k = raw_train_lbl[val_idx]

    # Count zeros before filtering for metadata transparency
    zeros_train = int(np.sum(train_lbl_50k == 0))
    zeros_val = int(np.sum(val_lbl_10k == 0))
    zeros_test = int(np.sum(raw_test_lbl == 0))

    # Filter out digit 0
    train_mask = train_lbl_50k != 0
    val_mask = val_lbl_10k != 0
    test_mask = raw_test_lbl != 0

    X_train = (train_img_50k[train_mask].astype(np.float32)) / 255.0
    digits_train = train_lbl_50k[train_mask].astype(np.int64)
    y_train = digit_to_class(digits_train)

    X_val = (val_img_10k[val_mask].astype(np.float32)) / 255.0
    digits_val = val_lbl_10k[val_mask].astype(np.int64)
    y_val = digit_to_class(digits_val)

    X_test = (raw_test_img[test_mask].astype(np.float32)) / 255.0
    digits_test = raw_test_lbl[test_mask].astype(np.int64)
    y_test = digit_to_class(digits_test)

    metadata: Dict[str, object] = {
        "seed": seed,
        "raw_train_total": int(len(raw_train_img)),
        "raw_test_total": int(len(raw_test_img)),
        "raw_train_split": int(train_split_size),
        "raw_val_split": int(val_split_size),
        "zeros_excluded_train": zeros_train,
        "zeros_excluded_val": zeros_val,
        "zeros_excluded_test": zeros_test,
        "filtered_train_count": int(len(X_train)),
        "filtered_val_count": int(len(X_val)),
        "filtered_test_count": int(len(X_test)),
        "image_shape": [28, 28],
        "pixel_range": [0.0, 1.0],
        "class_mapping": {str(c): int(c + 1) for c in range(9)},
        "digit_to_class_formula": "class_idx = digit - 1 for digit in 1..9",
        "sha256": {k: compute_sha256(p) for k, p in paths.items()},
    }

    return MNISTSplit(
        X_train=X_train,
        y_train=y_train,  # type: ignore[arg-type]
        digits_train=digits_train,
        X_val=X_val,
        y_val=y_val,  # type: ignore[arg-type]
        digits_val=digits_val,
        X_test=X_test,
        y_test=y_test,  # type: ignore[arg-type]
        digits_test=digits_test,
        metadata=metadata,
    )
