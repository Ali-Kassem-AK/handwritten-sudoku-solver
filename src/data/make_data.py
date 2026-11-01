"""CLI and library entry point to prepare MNIST splits and sample generated Sudoku datasets.

Executes Section 5 requirements:
1. Verifies or downloads raw MNIST IDX gzip files in `data/raw/`.
2. Extracts and preprocesses images (normalizing to [0, 1], excluding digit 0,
   splitting 50,000/10,000 train/val with fixed seed, keeping 10,000 test untouched).
3. Saves processed splits (`data/processed/mnist_sudoku_1_9.npz` and `preprocessing_config.json`).
4. Generates a reproducible sample set of Sudoku puzzles and 9x9 MNIST mosaics in `data/generated/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from src.data.mnist_loader import load_and_preprocess_mnist
from src.data.mosaic_generator import create_sudoku_mosaic
from src.data.puzzle_generator import generate_sudoku_puzzle


def run_make_data(
    project_root: str | Path,
    seed: int = 42,
    sample_puzzles: int = 5,
    givens: int = 35,
) -> Dict[str, object]:
    """Run full data preparation and save processed/generated artifacts."""
    root = Path(project_root)
    raw_dir = root / "data" / "raw"
    proc_dir = root / "data" / "processed"
    gen_dir = root / "data" / "generated"

    proc_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Check common fallback locations for uploaded MNIST files
    fallback_dirs = (
        root.parent / "user_uploaded_files" / "ttl=63d" / "user_uploaded",
    )

    mnist = load_and_preprocess_mnist(
        raw_dir=raw_dir,
        seed=seed,
        train_split_size=50000,
        val_split_size=10000,
        fallback_dirs=fallback_dirs,
    )

    npz_path = proc_dir / "mnist_sudoku_1_9.npz"
    np.savez_compressed(
        npz_path,
        X_train=mnist.X_train,
        y_train=mnist.y_train,
        digits_train=mnist.digits_train,
        X_val=mnist.X_val,
        y_val=mnist.y_val,
        digits_val=mnist.digits_val,
        X_test=mnist.X_test,
        y_test=mnist.y_test,
        digits_test=mnist.digits_test,
    )

    config_path = proc_dir / "preprocessing_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(mnist.metadata, f, indent=2)

    # Generate sample puzzles and mosaics using untouched test set digits
    sample_records: List[Dict[str, object]] = []
    for i in range(sample_puzzles):
        p_seed = seed + i * 100
        gen_puz = generate_sudoku_puzzle(num_givens=givens, seed=p_seed, ensure_unique=True)
        mosaic = create_sudoku_mosaic(
            puzzle_grid=gen_puz.puzzle,
            solution_grid=gen_puz.solution,
            mnist_images=mnist.X_test,
            mnist_digits=mnist.digits_test,
            seed=p_seed,
        )
        img_uint8 = np.clip(mosaic.mosaic_image * 255.0, 0, 255).astype(np.uint8)
        img_path = gen_dir / f"sample_mosaic_seed_{p_seed}.png"
        Image.fromarray(img_uint8, mode="L").save(img_path)

        meta = mosaic.to_metadata_dict()
        meta["mosaic_image_file"] = img_path.name
        meta["unique_solution"] = gen_puz.unique_solution
        sample_records.append(meta)

    manifest_path = gen_dir / "sample_mosaics_metadata.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(sample_records, f, indent=2)

    return {
        "processed_npz": str(npz_path),
        "config_json": str(config_path),
        "generated_manifest": str(manifest_path),
        "metadata": mnist.metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download/verify MNIST, preprocess 1-9 splits, and generate sample Sudoku mosaics."
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Path to project root directory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--sample-puzzles",
        type=int,
        default=5,
        help="Number of sample Sudoku mosaics to generate (default: 5).",
    )
    parser.add_argument(
        "--givens",
        type=int,
        default=35,
        help="Number of givens per sample puzzle (default: 35).",
    )
    args = parser.parse_args()

    result = run_make_data(
        project_root=args.project_root,
        seed=args.seed,
        sample_puzzles=args.sample_puzzles,
        givens=args.givens,
    )
    meta = result["metadata"]
    print("=== MNIST Preprocessing & Sudoku Data Generation Complete ===")
    print(f"Seed:                   {meta['seed']}")  # type: ignore[index]
    print(f"Raw Train / Val / Test: {meta['raw_train_split']} / {meta['raw_val_split']} / {meta['raw_test_total']}")  # type: ignore[index]
    print(
        f"Zeros Excluded:         Train={meta['zeros_excluded_train']}, "  # type: ignore[index]
        f"Val={meta['zeros_excluded_val']}, Test={meta['zeros_excluded_test']}"  # type: ignore[index]
    )
    print(
        f"Digits 1-9 Retained:    Train={meta['filtered_train_count']}, "  # type: ignore[index]
        f"Val={meta['filtered_val_count']}, Test={meta['filtered_test_count']}"  # type: ignore[index]
    )
    print(f"Saved Processed Data:   {result['processed_npz']}")
    print(f"Saved Config JSON:      {result['config_json']}")
    print(f"Saved Sample Mosaics:   {result['generated_manifest']}")


if __name__ == "__main__":
    main()
