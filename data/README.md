# Data Directory Overview (`data/`)

This directory stores the raw MNIST IDX archives, the preprocessed digit `1–9` training/validation/test splits, and sample generated Sudoku image mosaics.

## Subdirectories

- **`data/raw/`**: Contains the four official MNIST gzip archives (Yann LeCun, Corinna Cortes, Christopher J.C. Burges):
  - `train-images-idx3-ubyte.gz` (60,000 grayscale `28×28` handwritten digit images)
  - `train-labels-idx1-ubyte.gz` (60,000 integer labels `0–9`)
  - `t10k-images-idx3-ubyte.gz` (10,000 untouched test grayscale `28×28` images)
  - `t10k-labels-idx1-ubyte.gz` (10,000 untouched test integer labels `0–9`)
- **`data/processed/`**: Created reproducibly by `python -m src.data.make_data --seed 42`:
  - `mnist_sudoku_1_9.npz`: Compressed NumPy archive containing normalized (`[0.0, 1.0]`, `float32`) images and mapped class labels (`0–8` corresponding to digits `1–9`, with digit `0` excluded) partitioned into `50,000` training (`45,073` non-zero), `10,000` validation (`9,004` non-zero), and `10,000` untouched test (`9,020` non-zero) splits.
  - `preprocessing_config.json`: Records the exact random seed (`42`), split sizes, zero-exclusion counts, class mapping, and SHA-256 checksums of the raw archives.
- **`data/generated/`**: Contains generated Sudoku puzzles, `9×9` MNIST handwriting mosaics (`252×252` PNG images), and cell-level provenance metadata (`sample_mosaics_metadata.json`).

> **Note on Provenance**: While the individual `28×28` handwritten digit tiles are real human handwriting from the MNIST dataset, the `9×9` Sudoku puzzles, `252×252` image mosaics, and injected recognition errors are synthetically generated for controlled experimental evaluation. They are not photographs of physical paper Sudoku grids.
