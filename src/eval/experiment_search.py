"""Experiment 1: Search Algorithm Comparison (UCS vs Greedy Best-First Search vs A* Search).

Implements Section 9 Experiment 1 requirements:
- Compares Uniform Cost Search (`UCS`), Greedy Best-First Search (`Greedy`), and A* Search (`A*`).
- Evaluates puzzles across 0, 1, 2, 3, and 4 injected misreads.
- Uses 5 independent random seeds (`SEEDS = [101, 202, 303, 404, 505]`), with 20 puzzles per seed
  per misread level = 100 puzzles per misread level (500 puzzles total across the 5 misread levels).
- Reconciles the default maximum of 3 corrections with the 4-misread condition:
  * Evaluates `max_corrections = 4` for the 4-misread condition when measuring full recovery, AND
  * Explicitly records the default `max_corrections = 3` budget behavior on the 4-misread condition
    (which yields 0% full 4-cell recovery as expected when 4 errors occur under a strict 3-edit cap).
- Measures and reports mean and standard deviation across the 5 seeds for:
  * Nodes generated
  * Nodes expanded
  * Solvability recovery rate (%)
  * Exact ground-truth solution recovery rate (%)
  * Final correction cost
  * Execution time (ms)
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

from src.csp.sudoku import SudokuCSPSolver
from src.data.mosaic_generator import create_sudoku_mosaic, inject_hard_digit_errors
from src.data.puzzle_generator import generate_sudoku_puzzle
from src.nn.model import DigitMLP
from src.nn.predict import recognize_sudoku_mosaic
from src.search.astar import astar_search
from src.search.greedy import greedy_best_first_search
from src.search.problem import SudokuCorrectionProblem
from src.search.ucs import uniform_cost_search

DEFAULT_SEEDS = [101, 202, 303, 404, 505]
MISREAD_LEVELS = [0, 1, 2, 3, 4]
ALGORITHMS = ["UCS", "Greedy", "A*"]


def run_experiment_search(
    project_root: str | Path,
    seeds: List[int] = DEFAULT_SEEDS,
    puzzles_per_seed: int = 20,
    givens: int = 35,
) -> Dict[str, object]:
    """Execute Experiment 1 (UCS vs Greedy vs A*) across 0..4 misreads and 5 seeds (100 puzzles/level)."""
    root = Path(project_root)
    npz_path = root / "data" / "processed" / "mnist_sudoku_1_9.npz"
    mlp_path = root / "models" / "digit_mlp.npz"
    data = np.load(npz_path)
    X_test, digits_test = data["X_test"], data["digits_test"]
    model = DigitMLP.load(mlp_path)
    csp_solver = SudokuCSPSolver(mode="BT+FC")

    raw_rows: List[Dict[str, object]] = []
    t_exp_start = time.perf_counter()

    for misreads in MISREAD_LEVELS:
        # Reconcile budget: default 3 for 0..3 misreads, configurable to 4 for 4-misread full recovery
        budget_full = max(3, misreads)

        for seed_idx, seed in enumerate(seeds):
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

                # Evaluate UCS, Greedy, and A* on the exact same corrupted puzzle & probabilities
                for alg in ALGORITHMS:
                    problem = SudokuCorrectionProblem(
                        initial_grid=pred_grid,
                        prob_grid=prob_grid,
                        csp_solver=csp_solver,
                        max_corrections=budget_full,
                        top_k_alternatives=3,
                        uncertainty_threshold=0.90,
                    )
                    if alg == "UCS":
                        res = uniform_cost_search(problem, max_expansions=1500, record_trace=False)
                    elif alg == "Greedy":
                        res = greedy_best_first_search(problem, max_expansions=1500, record_trace=False)
                    else:
                        res = astar_search(problem, max_expansions=1500, record_trace=False)

                    solvable = bool(res.solution_found and res.solved_grid is not None)
                    exact_sol = bool(
                        solvable
                        and res.solved_grid is not None
                        and np.array_equal(res.solved_grid, puz.solution)
                    )
                    exact_puz = bool(
                        solvable
                        and res.corrected_grid is not None
                        and np.array_equal(res.corrected_grid, puz.puzzle)
                    )

                    raw_rows.append(
                        {
                            "seed": seed,
                            "puzzle_idx": p_idx,
                            "instance_seed": instance_seed,
                            "misreads": misreads,
                            "budget": budget_full,
                            "algorithm": alg,
                            "nodes_generated": res.metrics.nodes_generated,
                            "nodes_expanded": res.metrics.nodes_expanded,
                            "goal_checks": res.metrics.goal_checks,
                            "num_corrections": res.metrics.num_corrections,
                            "correction_cost": res.metrics.final_correction_cost,
                            "solvable": int(solvable),
                            "exact_puzzle_recovery": int(exact_puz),
                            "exact_solution_recovery": int(exact_sol),
                            "execution_time_ms": res.metrics.execution_time * 1000.0,
                        }
                    )

                # Also record the default budget=3 behavior on the 4-misread condition for A*
                if misreads == 4:
                    prob_b3 = SudokuCorrectionProblem(
                        initial_grid=pred_grid,
                        prob_grid=prob_grid,
                        csp_solver=csp_solver,
                        max_corrections=3,
                        top_k_alternatives=3,
                        uncertainty_threshold=0.90,
                    )
                    res_b3 = astar_search(prob_b3, max_expansions=800, record_trace=False)
                    solvable_b3 = bool(res_b3.solution_found and res_b3.solved_grid is not None)
                    exact_sol_b3 = bool(
                        solvable_b3
                        and res_b3.solved_grid is not None
                        and np.array_equal(res_b3.solved_grid, puz.solution)
                    )
                    raw_rows.append(
                        {
                            "seed": seed,
                            "puzzle_idx": p_idx,
                            "instance_seed": instance_seed,
                            "misreads": 4,
                            "budget": 3,
                            "algorithm": "A* (Budget=3)",
                            "nodes_generated": res_b3.metrics.nodes_generated,
                            "nodes_expanded": res_b3.metrics.nodes_expanded,
                            "goal_checks": res_b3.metrics.goal_checks,
                            "num_corrections": res_b3.metrics.num_corrections,
                            "correction_cost": res_b3.metrics.final_correction_cost,
                            "solvable": int(solvable_b3),
                            "exact_puzzle_recovery": 0,
                            "exact_solution_recovery": int(exact_sol_b3),
                            "execution_time_ms": res_b3.metrics.execution_time * 1000.0,
                        }
                    )

    df_raw = pd.DataFrame(raw_rows)
    tables_dir = root / "results" / "tables"
    logs_dir = root / "results" / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    df_raw.to_csv(logs_dir / "experiment1_search_raw.csv", index=False)

    # First aggregate per-seed means, then compute mean and std across the 5 independent seeds
    seed_means = (
        df_raw.groupby(["misreads", "budget", "algorithm", "seed"], as_index=False)
        .agg(
            nodes_generated=("nodes_generated", "mean"),
            nodes_expanded=("nodes_expanded", "mean"),
            goal_checks=("goal_checks", "mean"),
            solvable_rate=("solvable", lambda x: np.mean(x) * 100.0),
            exact_recovery_rate=("exact_solution_recovery", lambda x: np.mean(x) * 100.0),
            correction_cost=("correction_cost", "mean"),
            execution_time_ms=("execution_time_ms", "mean"),
        )
    )

    summary_rows: List[Dict[str, object]] = []
    for (misreads, budget, alg), grp in seed_means.groupby(["misreads", "budget", "algorithm"]):
        summary_rows.append(
            {
                "misreads": int(misreads),
                "budget": int(budget),
                "algorithm": str(alg),
                "n_seeds": int(len(grp)),
                "total_puzzles": int(len(grp) * puzzles_per_seed),
                "nodes_generated_mean": round(float(grp["nodes_generated"].mean()), 2),
                "nodes_generated_std": round(float(grp["nodes_generated"].std(ddof=1)), 2),
                "nodes_expanded_mean": round(float(grp["nodes_expanded"].mean()), 2),
                "nodes_expanded_std": round(float(grp["nodes_expanded"].std(ddof=1)), 2),
                "solvable_rate_mean": round(float(grp["solvable_rate"].mean()), 2),
                "solvable_rate_std": round(float(grp["solvable_rate"].std(ddof=1)), 2),
                "exact_recovery_rate_mean": round(float(grp["exact_recovery_rate"].mean()), 2),
                "exact_recovery_rate_std": round(float(grp["exact_recovery_rate"].std(ddof=1)), 2),
                "correction_cost_mean": round(float(grp["correction_cost"].mean()), 4),
                "correction_cost_std": round(float(grp["correction_cost"].std(ddof=1)), 4),
                "execution_time_ms_mean": round(float(grp["execution_time_ms"].mean()), 2),
                "execution_time_ms_std": round(float(grp["execution_time_ms"].std(ddof=1)), 2),
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(tables_dir / "experiment1_search_summary.csv", index=False)

    out_payload = {
        "experiment": "Experiment 1: Search Algorithm Comparison (UCS vs Greedy vs A*)",
        "seeds": seeds,
        "puzzles_per_seed": puzzles_per_seed,
        "total_puzzles_per_misread_level": len(seeds) * puzzles_per_seed,
        "total_elapsed_sec": round(time.perf_counter() - t_exp_start, 2),
        "summary": summary_rows,
    }
    with open(tables_dir / "experiment1_search_summary.json", "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    return out_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 1: Search Algorithm Comparison.")
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
        help="Number of puzzles per seed per misread level (default: 20 -> 100 puzzles/level across 5 seeds).",
    )
    args = parser.parse_args()

    res = run_experiment_search(args.project_root, puzzles_per_seed=args.puzzles_per_seed)
    print(f"=== Experiment 1 Complete ({res['total_elapsed_sec']}s) ===")
    rows = res["summary"]  # type: ignore[assignment]
    disp = [
        [
            r["misreads"],
            r["budget"],
            r["algorithm"],
            f"{r['nodes_generated_mean']:.1f} ± {r['nodes_generated_std']:.1f}",
            f"{r['nodes_expanded_mean']:.2f} ± {r['nodes_expanded_std']:.2f}",
            f"{r['exact_recovery_rate_mean']:.1f}% ± {r['exact_recovery_rate_std']:.1f}%",
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
                "Budget",
                "Algorithm",
                "Nodes Gen",
                "Nodes Exp",
                "Exact Recovery",
                "Cost",
                "Time (ms)",
            ],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()
