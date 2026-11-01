"""Experiment 3: Correction Strategy Comparison (NN-Guided vs No Correction vs Random Correction).

Implements Section 9 Experiment 3 requirements:
- Compares:
  1. `nn_guided` (NN-guided A* search correction)
  2. `no_correction` (No correction — direct CSP solving on raw predictions)
  3. `random_correction` (Random/uninformed candidate correction without NN probability rankings)
- Uses equivalent puzzle sets, controlled misread levels (`0, 1, 2, 3`), and 5 matching random seeds
  (`SEEDS = [101, 202, 303, 404, 505]`), with 20 puzzles per seed = 100 puzzles per misread level.
- Explicitly distinguishes between:
  * Solvability rate (% of puzzles where a valid completed Sudoku solution was found)
  * Exact ground-truth solution recovery rate (% of puzzles where the recovered solution matches
    the original ground-truth Sudoku solution)
- Also measures number of corrections, correction cost, and execution time (ms).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.data.mosaic_generator import create_sudoku_mosaic, inject_hard_digit_errors
from src.data.puzzle_generator import generate_sudoku_puzzle
from src.integration.correction import run_correction_strategy
from src.nn.model import DigitMLP
from src.nn.predict import recognize_sudoku_mosaic

DEFAULT_SEEDS = [101, 202, 303, 404, 505]
MISREAD_LEVELS = [0, 1, 2, 3]
STRATEGIES = ["nn_guided", "no_correction", "random_correction"]


def run_experiment_correction(
    project_root: str | Path,
    seeds: List[int] = DEFAULT_SEEDS,
    puzzles_per_seed: int = 20,
    givens: int = 35,
) -> Dict[str, object]:
    """Execute Experiment 3 (NN-guided vs No correction vs Random correction) across 0..3 misreads."""
    root = Path(project_root)
    npz_path = root / "data" / "processed" / "mnist_sudoku_1_9.npz"
    mlp_path = root / "models" / "digit_mlp.npz"
    data = np.load(npz_path)
    X_test, digits_test = data["X_test"], data["digits_test"]
    model = DigitMLP.load(mlp_path)

    raw_rows: List[Dict[str, object]] = []
    t_start = time.perf_counter()

    for misreads in MISREAD_LEVELS:
        for seed in seeds:
            for p_idx in range(puzzles_per_seed):
                instance_seed = seed * 1000 + misreads * 100 + p_idx
                puz = generate_sudoku_puzzle(num_givens=givens, seed=instance_seed, ensure_unique=True)
                mosaic = create_sudoku_mosaic(
                    puz.puzzle, puz.solution, X_test, digits_test, seed=instance_seed
                )
                rec = recognize_sudoku_mosaic(mosaic.cell_images, model, use_calibrated=True, use_nn=True)
                pred_grid, prob_grid, _ = inject_hard_digit_errors(
                    rec.predicted_grid,
                    rec.probability_grid,
                    puz.puzzle,
                    puz.solution,
                    num_errors=misreads,
                    seed=instance_seed,
                )

                for strat in STRATEGIES:
                    res = run_correction_strategy(
                        strategy=strat,
                        initial_grid=pred_grid,
                        prob_grid=prob_grid,
                        search_algorithm="A*",
                        csp_mode="BT+FC",
                        max_corrections=3,
                        top_k_alternatives=3,
                        require_unique=False,
                        seed=instance_seed,
                        max_expansions=1500,
                    )
                    solvable = bool(res.solution_found and res.solved_grid is not None)
                    exact_sol = bool(
                        solvable
                        and res.solved_grid is not None
                        and np.array_equal(res.solved_grid, puz.solution)
                    )
                    raw_rows.append(
                        {
                            "seed": seed,
                            "puzzle_idx": p_idx,
                            "instance_seed": instance_seed,
                            "misreads": misreads,
                            "strategy": strat,
                            "solvable": int(solvable),
                            "exact_solution_recovery": int(exact_sol),
                            "num_corrections": res.metrics.num_corrections,
                            "correction_cost": res.metrics.final_correction_cost,
                            "nodes_expanded": res.metrics.nodes_expanded,
                            "execution_time_ms": res.metrics.execution_time * 1000.0,
                        }
                    )

    df_raw = pd.DataFrame(raw_rows)
    tables_dir = root / "results" / "tables"
    logs_dir = root / "results" / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    df_raw.to_csv(logs_dir / "experiment3_correction_raw.csv", index=False)

    seed_means = (
        df_raw.groupby(["misreads", "strategy", "seed"], as_index=False)
        .agg(
            solvable_rate=("solvable", lambda x: np.mean(x) * 100.0),
            exact_recovery_rate=("exact_solution_recovery", lambda x: np.mean(x) * 100.0),
            num_corrections=("num_corrections", "mean"),
            correction_cost=("correction_cost", "mean"),
            execution_time_ms=("execution_time_ms", "mean"),
        )
    )

    summary_rows: List[Dict[str, object]] = []
    for (misreads, strat), grp in seed_means.groupby(["misreads", "strategy"]):
        summary_rows.append(
            {
                "misreads": int(misreads),
                "strategy": str(strat),
                "n_seeds": int(len(grp)),
                "total_puzzles": int(len(grp) * puzzles_per_seed),
                "exact_recovery_rate_mean": round(float(grp["exact_recovery_rate"].mean()), 2),
                "exact_recovery_rate_std": round(float(grp["exact_recovery_rate"].std(ddof=1)), 2),
                "solvable_rate_mean": round(float(grp["solvable_rate"].mean()), 2),
                "solvable_rate_std": round(float(grp["solvable_rate"].std(ddof=1)), 2),
                "num_corrections_mean": round(float(grp["num_corrections"].mean()), 2),
                "num_corrections_std": round(float(grp["num_corrections"].std(ddof=1)), 2),
                "correction_cost_mean": round(float(grp["correction_cost"].mean()), 4),
                "correction_cost_std": round(float(grp["correction_cost"].std(ddof=1)), 4),
                "execution_time_ms_mean": round(float(grp["execution_time_ms"].mean()), 2),
                "execution_time_ms_std": round(float(grp["execution_time_ms"].std(ddof=1)), 2),
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(tables_dir / "experiment3_correction_summary.csv", index=False)

    out_payload = {
        "experiment": "Experiment 3: Correction Strategy Comparison (NN-Guided vs No Correction vs Random Correction)",
        "seeds": seeds,
        "puzzles_per_seed": puzzles_per_seed,
        "total_puzzles_per_misread_level": len(seeds) * puzzles_per_seed,
        "total_elapsed_sec": round(time.perf_counter() - t_start, 2),
        "summary": summary_rows,
    }
    with open(tables_dir / "experiment3_correction_summary.json", "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    return out_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 3: Correction Strategy Comparison.")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Project root path.",
    )
    parser.add_argument(
        "--puzzles-per-seed",
        type=int,
        default=20,
        help="Number of puzzles per seed per misread level (default: 20 -> 100 puzzles/level).",
    )
    args = parser.parse_args()

    res = run_experiment_correction(args.project_root, puzzles_per_seed=args.puzzles_per_seed)
    print(f"=== Experiment 3 Complete ({res['total_elapsed_sec']}s) ===")
    rows = res["summary"]  # type: ignore[assignment]
    disp = [
        [
            r["misreads"],
            r["strategy"],
            f"{r['exact_recovery_rate_mean']:.1f}% ± {r['exact_recovery_rate_std']:.1f}%",
            f"{r['solvable_rate_mean']:.1f}% ± {r['solvable_rate_std']:.1f}%",
            f"{r['num_corrections_mean']:.2f} ± {r['num_corrections_std']:.2f}",
            f"{r['correction_cost_mean']:.4f} ± {r['correction_cost_std']:.4f}",
            f"{r['execution_time_ms_mean']:.2f} ± {r['execution_time_ms_std']:.2f} ms",
        ]
        for r in rows  # type: ignore[union-attr]
    ]
    print(
        tabulate(
            disp,
            headers=[
                "Misreads",
                "Strategy",
                "Exact Recovery",
                "Solvability Rate",
                "Corrections",
                "Cost",
                "Time (ms)",
            ],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()
