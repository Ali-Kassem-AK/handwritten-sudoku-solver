# Handwritten Sudoku Reader and Solver

An end-to-end intelligent system that reads `9×9` Sudoku puzzles composed of handwritten MNIST digits (`1–9`), detects duplicate-digit conflicts and domain contradictions using Arc Consistency (`AC-3`), searches for the minimum-cost probability-guided digit corrections (`UCS`, `Greedy Best-First Search`, and `A* Search`), and solves the resulting puzzle using a switchable Constraint Satisfaction Problem (`CSP`) solver (`Backtracking`, `Backtracking + Forward Checking`, and `Backtracking + Forward Checking + AC-3`).

## 1. Project Overview

Optical and handwritten Sudoku readers frequently fail when a single handwritten digit is misclassified (for example, when a handwritten `7` is read as a `1` with `52%` vs. `45%` confidence). A single misread given either creates an immediate duplicate-digit violation in a row, column, or `3×3` box, or renders the puzzle unsolvable. Rather than discarding the puzzle or blindly guessing replacements, this project couples three core paradigms from Artificial Intelligence:

1. **Calibrated Neural Network (`DigitMLP`)**: Classifies `28×28` grayscale handwritten digit images across digits `1–9` (excluding `0`) and outputs temperature-scaled probability distributions $P(d \mid I_{r,c})$.
2. **Heuristic Graph Search (`UCS`, `Greedy`, `A*`)**: Formulates misread recovery as a minimum-cost state-space search over sets of cell corrections, where the step cost $c(r,c, d_{\text{orig}} \to d_{\text{new}}) = -\ln\!\big(\frac{P(d_{\text{new}}) + \epsilon}{P(d_{\text{orig}}) + \epsilon}\big) \ge 0$ corresponds to the negative log-likelihood ratio of the replacement, and an admissible conflict-matching heuristic $h(s)$ guides `A*` to the optimal Maximum A Posteriori (MAP) valid puzzle reading in minimal node expansions.
3. **Constraint Satisfaction Problem (`CSP`) Solver**: Represents Sudoku as an 81-variable binary CSP with 1,620 directed inequality arcs and provides three independently switchable solver configurations (`BT`, `BT+FC`, and `BT+FC+AC3`) with full instrumentation (`assignments`, `backtracks`, `constraint_checks`, `domain_reductions`, `execution_time`).

## 2. Objectives and System Architecture

```
+------------------------+     +-------------------------------+     +-------------------------------+
|  MNIST 9x9 Mosaic      | --> | Brightness Blank Detector     | --> | Calibrated DigitMLP (1-9)     |
|  (252x252 Grayscale)   |     | (mean_intensity < 0.02)       |     | 784 -> 128 -> 64 -> 9 (T=1.14)|
+------------------------+     +-------------------------------+     +-------------------------------+
                                                                                     |
                                                                                     v
+------------------------+     +-------------------------------+     +-------------------------------+
|  Final Solved Sudoku   | <-- | Switchable CSP Solver         | <-- | Graph Search Correction       |
|  + Correction Log      |     | (BT / BT+FC / BT+FC+AC3)      |     | (UCS / Greedy / A* + AC-3)    |
+------------------------+     +-------------------------------+     +-------------------------------+
```

- **Academic Reference Alignment**:
  - **Search (`Searching Lecture.pdf`, `Searching Lab.pdf`)**: Implements Graph Search with a `closed` set (`Searching Lecture.pdf`, slides 107 & 119), goal testing on dequeue (`slide 73`), Uniform Cost Search (`slides 43–46`), Greedy Best-First Search (`slides 65–69`), and `A*` Search (`slides 70–87`) with a dominance-improved admissible heuristic (`slides 77, 103`).
  - **CSP (`CSP Lecture.pdf`, `CSP Lab.pdf`)**: Formulates Sudoku with 81 variables and pairwise binary inequality constraints (`CSP Lecture.pdf`, slide 12), Backtracking Search (`slide 26`), Forward Checking (`slides 29–32`), Arc Consistency `AC-3` (`slide 39`, `CSP Lab.pdf` slides 15–21), Minimum Remaining Values (`MRV`, `slide 47`, `CSP Lab.pdf` slides 22 & 45), and Least Constraining Value (`LCV`, `slide 48`).
  - **Neural Networks (`Neural Networks Lecture.pdf`)**: Implements the exact `784 -> 128 (ReLU) -> 64 (ReLU) -> Softmax` Multi-Layer Perceptron architecture taught in `Neural Networks Lecture.pdf` (slides 60–72), adapted to 9 outputs for Sudoku digits `1–9`.

## 3. Module Descriptions

- **`src/data/mnist_loader.py`**: Verifies SHA-256 checksums and IDX magic headers of the 4 official MNIST `.gz` files, creates the seeded `50,000 / 10,000` train/validation split (`seed=42`), preserves the untouched `10,000` test set, excludes digit `0`, maps digits `1..9` to classes `0..8`, and normalizes pixel values to `[0.0, 1.0]`.
- **`src/data/puzzle_generator.py`**: Generates complete valid `9×9` Sudoku boards via randomized backtracking and removes cells while verifying solution uniqueness (`count_sudoku_solutions_fast(limit=2) == 1`).
- **`src/data/mosaic_generator.py`**: Tiling engine that constructs `252×252` (`9×9` tiles of `28×28`) handwritten Sudoku mosaics from `X_test`, detects blank cells via mean pixel intensity (`< 0.02`), and implements controlled hard-digit error injection (`inject_hard_digit_errors`).
- **`src/data/make_data.py`**: Reproducible CLI script that executes the full data preparation and saves `data/processed/mnist_sudoku_1_9.npz`, `preprocessing_config.json`, and sample mosaics in `data/generated/`.
- **`src/nn/model.py`**: Implements `DigitMLP` (`784 -> 128 -> 64 -> 9`) and `LogisticRegressionBaseline` (`784 -> 9`).
- **`src/nn/calibration.py`**: Computes Expected Calibration Error (`ECE`), Maximum Calibration Error (`MCE`), Negative Log-Likelihood (`NLL`), Brier score, and fits post-hoc Temperature Scaling (`T = 1.1406`) on the validation split only.
- **`src/nn/train.py`**: Trains `DigitMLP` and `LogisticRegressionBaseline` with fixed seeds and saves weights/metadata to `models/`.
- **`src/nn/predict.py`**: Provides single-cell and `9×9` mosaic recognition with an explicit `use_nn=True/False` switch.
- **`src/nn/evaluate.py`**: Evaluates both models on the untouched MNIST `1–9` test set (`9,020` samples) and saves confusion matrices, reliability diagrams, and metrics to `results/`.
- **`src/csp/sudoku.py`, `backtracking.py`, `forward_checking.py`, `ac3.py`, `metrics.py`**: Implements `SudokuCSPSolver` with three separate, switchable modes (`BT`, `BT+FC`, `BT+FC+AC3`), standalone `AC-3` conflict detection, and solution counting.
- **`src/search/problem.py`, `ucs.py`, `greedy.py`, `astar.py`, `metrics.py`**: Implements `SudokuCorrectionProblem`, `uniform_cost_search`, `greedy_best_first_search`, and `astar_search` with full instrumentation and known-answer optimality verification.
- **`src/integration/pipeline.py`, `correction.py`, `visualization.py`**: Connects all three components into the 12-step pipeline, generates correction logs and `Unreadable` diagnostic reports, and plots component cooperation.
- **`src/eval/experiment_search.py`, `experiment_csp.py`, `experiment_correction.py`, `results.py`**: Executes the three multi-seed experiments (`5` seeds each, `100` puzzles per misread level) and generates CSV/JSON tables and PNG plots.

## 4. Dataset Information and Download Instructions

The project uses the official **MNIST Database of Handwritten Digits**:
- `train-images-idx3-ubyte.gz` (60,000 training images, `9.91 MB`, SHA-256: `440fcabf...23609`)
- `train-labels-idx1-ubyte.gz` (60,000 training labels, `28.9 KB`, SHA-256: `3552534a...0255c`)
- `t10k-images-idx3-ubyte.gz` (10,000 test images, `1.65 MB`, SHA-256: `8d422c7b...bc4e6`)
- `t10k-labels-idx1-ubyte.gz` (10,000 test labels, `4.54 KB`, SHA-256: `f7ae60f9...aec6`)

Place these four files in `data/raw/` (already present in this repository). If missing, `python -m src.data.make_data` automatically checks local fallback directories or downloads them from the official mirror (`https://storage.googleapis.com/cvdf-datasets/mnist/`).

## 5. Data Preprocessing Procedure

Running `python -m src.data.make_data --seed 42`:
1. Verifies the IDX3 (`2051`) and IDX1 (`2049`) magic numbers and byte lengths of all four gzip archives.
2. Uses a seeded NumPy random generator (`seed = 42`) to permute the `60,000` training samples into `50,000` training samples and `10,000` validation samples, leaving the `10,000` official test set untouched.
3. Filters out digit `0` (`label != 0`), removing `4,927` training zeros, `996` validation zeros, and `980` test zeros, leaving:
   - **Training set**: `45,073` images across digits `1–9`
   - **Validation set**: `9,004` images across digits `1–9`
   - **Untouched Test set**: `9,020` images across digits `1–9`
4. Maps Sudoku digits $d \in \{1,\dots,9\}$ to 0-indexed classes $c = d - 1 \in \{0,\dots,8\}$ and normalizes pixel intensities from `[0, 255]` (`uint8`) to `[0.0, 1.0]` (`float32`).

## 6. Environment Setup, Python Version & Pinned Dependencies

- **Python Version**: Python `3.11.2`
- **Hardware Used for Benchmarks**: 2-core x86_64 Linux VM (`5.0 GiB` RAM, CPU-only execution).
- **Pinned Dependencies (`requirements.txt`)**:
  ```text
  numpy==2.1.3
  scipy==1.15.2
  matplotlib==3.10.1
  seaborn==0.13.2
  pandas==2.2.3
  pillow==11.1.0
  tabulate==0.9.0
  python-pptx==1.0.2
  pypdf==6.21.1
  pypdfium2==5.14.0
  ```

### Installation Command
```bash
pip install -r requirements.txt
```

## 7. Exact Execution Commands

Run all commands from the `handwritten-sudoku-solver/` project root directory:

1. **Data Preprocessing & Sample Mosaic Generation**:
   ```bash
   python -m src.data.make_data --seed 42 --sample-puzzles 5 --givens 35
   ```
2. **Model Training (`DigitMLP` + `LogisticRegressionBaseline` + Validation Temperature Scaling)**:
   ```bash
   python -m src.nn.train --epochs 25 --seed 42
   ```
3. **Model Evaluation on Untouched Test Set**:
   ```bash
   python -m src.nn.evaluate
   ```
4. **Search Algorithm Standalone Execution (`UCS`, `Greedy`, or `A*`)**:
   ```bash
   python -m src.search.astar --algorithm "A*" --misreads 2 --budget 3 --seed 42
   python -m src.search.astar --algorithm UCS --misreads 2 --budget 3 --seed 42
   python -m src.search.astar --algorithm Greedy --misreads 2 --budget 3 --seed 42
   ```
5. **CSP Solver Standalone Execution (`BT`, `BT+FC`, or `BT+FC+AC3`)**:
   ```bash
   python -m src.csp.sudoku --mode BT --givens 30 --seed 42
   python -m src.csp.sudoku --mode BT+FC --givens 30 --seed 42
   python -m src.csp.sudoku --mode BT+FC+AC3 --givens 30 --seed 42
   ```
6. **Sudoku Puzzle Generation (via Python CLI)**:
   ```bash
   python -c "from src.data.puzzle_generator import generate_sudoku_puzzle; p=generate_sudoku_puzzle(35, 42); print(p.puzzle)"
   ```
7. **Execute All Three Primary Experiments & Generate Tables/Plots**:
   ```bash
   python -m src.eval.experiment_search
   python -m src.eval.experiment_csp
   python -m src.eval.experiment_correction
   python -m src.eval.results
   ```
8. **Run Main Integrated Demo (`run_demo.py`)**:
   ```bash
   python run_demo.py
   # Or customize demo options:
   python run_demo.py --givens 35 --misreads 2 --budget 3 --search "A*" --csp "BT+FC+AC3" --nn-mode both --seed 42
   ```
9. **Run Automated Test Suite (24 Unit, Integration, Edge & Impossible Tests)**:
   ```bash
   pytest
   # Or equivalently via unittest module runner:
   python -m tests
   ```

## 8. Output and Results Locations

- **Trained Models & Config**:
  - `models/digit_mlp.npz` (Trained `784->128->64->9` MLP weights and temperature `T=1.1406`)
  - `models/logreg_baseline.npz` (Trained Multinomial Logistic Regression baseline weights)
  - `models/model_config.json` (Architecture metadata, parameters count `109,385`, training/validation history)
- **Experimental Summary Tables (`results/tables/`)**:
  - `nn_test_metrics.json`, `nn_confusion_matrix.csv`, `logreg_confusion_matrix.csv`
  - `experiment1_search_summary.csv` & `.json`
  - `experiment2_csp_summary.csv` & `.json`
  - `experiment3_correction_summary.csv` & `.json`
- **Raw Multi-Seed Experimental Logs (`results/logs/`)**:
  - `experiment1_search_raw.csv` (1,600 search runs across 5 seeds and 0–4 misreads)
  - `experiment2_csp_raw.csv` (450 CSP runs across 5 seeds and 25, 30, 35 givens)
  - `experiment3_correction_raw.csv` (1,200 strategy runs across 5 seeds and 0–3 misreads)
  - `demo_run_log.json` (Complete trace of `python run_demo.py`)
- **Plots (`results/plots/`)**:
  - `nn_evaluation.png`, `demo_cooperation.png`, `demo_mosaic.png`, `experiment1_search.png`, `experiment2_csp.png`, `experiment3_correction.png`
- **Submission Deliverables (`reports/`)**:
  - `reports/final_report.pdf` (Academic report, $\le 12$ pages)
  - `reports/presentation.pptx` (10-minute presentation + 5-minute live demo deck)

## 9. Troubleshooting and Known Limitations

- **Synthetic Mosaic Provenance**: The `9×9` Sudoku mosaics are constructed by tiling real MNIST handwritten digits into a clean grid without camera perspective distortion, uneven lighting, or printed grid-line intersection noise. Extending the pipeline to real camera photographs would require OpenCV perspective warping (`cv2.getPerspectiveTransform`) and border-line removal before feeding `28×28` crops to `DigitMLP`.
- **Correction Budget vs. High Misread Density**: When the number of misread cells exceeds `max_corrections` (e.g., 4 misreads under the default budget of 3), the system cannot recover the puzzle and explicitly reports `Unreadable` along with the lowest-confidence cells and detected conflicts. Increasing `--budget 4` recovers 4-misread puzzles at the cost of a larger search frontier.
- **Reproducibility & Sources of Nondeterminism**: All random operations in data splitting, weight initialization, mini-batch shuffling, puzzle generation, mosaic sampling, and error injection are controlled via explicit `np.random.default_rng(seed)` instances. The only minor source of variation across different machines is floating-point reduction order in multi-threaded BLAS (`OpenBLAS`) matrix multiplication (`< 1e-6` difference) and OS wall-clock timing (`execution_time_ms`).
