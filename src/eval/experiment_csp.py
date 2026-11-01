"""Experiment 2: CSP Performance Comparison (BT vs BT+FC vs BT+FC+AC3).

Implements Section 9 Experiment 2 requirements:
- Compares:
  1. `BT` (Backtracking only)
  2. `BT+FC` (Backtracking + Forward Checking)
  3. `BT+FC+AC3` (Backtracking + Forward Checking + Arc Consistency using AC-3)
- Evaluates generated Sudoku puzzles with:
  * 25 givens
  * 30 givens
  * 35 givens
- Uses 5 independent random seeds (`SEEDS = [101, 202, 303, 404, 505]`), with 10 puzzles per seed
  per givens level = 50 puzzles per givens level (150 puzzles x 3 solvers = 450 solver runs).
- Measures and reports mean and standard deviation across the 5 seeds for:
  * Assignments
  * Backtracks
  * Constraint checks
  * Domain reductions
  * Execution time (ms)
  * Success rate (%)
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

from src.csp.sudoku import VALID_MODES, SudokuCSPSolver
from src.data.puzzle_generator import generate_sudoku_puzzle

DEFAULT_SEEDS = [101, 202, 303, 404, 505]
GIVENS_LEVELS = [25, 30, 35]


def run_experiment_csp(
    project_root: str | Path,
    seeds: List[int] = DEFAULT_SEEDS,
    puzzles_per_seed: int = 10,
) -> Dict[str, object]:
    """Execute Experiment 2 (BT vs BT+FC vs BT+FC+AC3) across 25, 30, and 35 givens."""
    root = Path(project_root)
    raw_rows: List[Dict[str, object]] = []
    t_start = time.perf_counter()

    for givens in GIVENS_LEVELS:
        for seed in seeds:
            for p_idx in range(puzzles_per_seed):
                instance_seed = seed * 1000 + givens * 10 + p_idx
                puz = generate_sudoku_puzzle(num_givens=givens, seed=instance_seed, ensure_unique=True)

                for mode in VALID_MODES:
                    solver = SudokuCSPSolver(mode=mode, max_bt_backtracks=100000)
                    res = solver.solve(puz.puzzle)
                    exact_match = bool(
                        res.solved
                        and res.solution is not None
                        and np.array_equal(res.solution, puz.solution)
                    )
                    raw_rows.append(
                        {
                            "seed": seed,
                            "puzzle_idx": p_idx,
                            "instance_seed": instance_seed,
                            "givens": givens,
                            "mode": mode,
                            "solved": int(res.solved),
                            "exact_match": int(exact_match),
                            "assignments": res.metrics.assignments,
                            "backtracks": res.metrics.backtracks,
                            "constraint_checks": res.metrics.constraint_checks,
                            "domain_reductions": res.metrics.domain_reductions,
                            "execution_time_ms": res.metrics.execution_time * 1000.0,
                        }
                    )

    df_raw = pd.DataFrame(raw_rows)
    tables_dir = root / "results" / "tables"
    logs_dir = root / "results" / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    df_raw.to_csv(logs_dir / "experiment2_csp_raw.csv", index=False)

    seed_means = (
        df_raw.groupby(["givens", "mode", "seed"], as_index=False)
        .agg(
            assignments=("assignments", "mean"),
            backtracks=("backtracks", "mean"),
            constraint_checks=("constraint_checks", "mean"),
            domain_reductions=("domain_reductions", "mean"),
            execution_time_ms=("execution_time_ms", "mean"),
            success_rate=("solved", lambda x: np.mean(x) * 100.0),
        )
    )

    summary_rows: List[Dict[str, object]] = []
    for (givens, mode), grp in seed_means.groupby(["givens", "mode"]):
        summary_rows.append(
            {
                "givens": int(givens),
                "mode": str(mode),
                "n_seeds": int(len(grp)),
                "total_puzzles": int(len(grp) * puzzles_per_seed),
                "assignments_mean": round(float(grp["assignments"].mean()), 2),
                "assignments_std": round(float(grp["assignments"].std(ddof=1)), 2),
                "backtracks_mean": round(float(grp["backtracks"].mean()), 2),
                "backtracks_std": round(float(grp["backtracks"].std(ddof=1)), 2),
                "constraint_checks_mean": round(float(grp["constraint_checks"].mean()), 2),
                "constraint_checks_std": round(float(grp["constraint_checks"].std(ddof=1)), 2),
                "domain_reductions_mean": round(float(grp["domain_reductions"].mean()), 2),
                "domain_reductions_std": round(float(grp["domain_reductions"].std(ddof=1)), 2),
                "execution_time_ms_mean": round(float(grp["execution_time_ms"].mean()), 2),
                "execution_time_ms_std": round(float(grp["execution_time_ms"].std(ddof=1)), 2),
                "success_rate_mean": round(float(grp["success_rate"].mean()), 2),
                "success_rate_std": round(float(grp["success_rate"].std(ddof=1)), 2),
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(tables_dir / "experiment2_csp_summary.csv", index=False)

    out_payload = {
        "experiment": "Experiment 2: CSP Performance Comparison (BT vs BT+FC vs BT+FC+AC3)",
        "seeds": seeds,
        "puzzles_per_seed": puzzles_per_seed,
        "total_puzzles_per_givens_level": len(seeds) * puzzles_per_seed,
        "total_elapsed_sec": round(time.perf_counter() - t_start, 2),
        "summary": summary_rows,
    }
    with open(tables_dir / "experiment2_csp_summary.json", "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    return out_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 2: CSP Performance Comparison.")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Project root path.",
    )
    parser.add_argument(
        "--puzzles-per-seed",
        type=int,
        default=10,
        help="Number of puzzles per seed per givens level (default: 10 -> 50 puzzles/level).",
    )
    args = parser.parse_args()

    res = run_experiment_csp(args.project_root, puzzles_per_seed=args.puzzles_per_seed)
    print(f"=== Experiment 2 Complete ({res['total_elapsed_sec']}s) ===")
    rows = res["summary"]  # type: ignore[assignment]
    disp = [
        [
            r["givens"],
            r["mode"],
            f"{r['assignments_mean']:.1f} ± {r['assignments_std']:.1f}",
            f"{r['backtracks_mean']:.1f} ± {r['backtracks_std']:.1f}",
            f"{r['constraint_checks_mean']:.1f} ± {r['constraint_checks_std']:.1f}",
            f"{r['domain_reductions_mean']:.1f} ± {r['domain_reductions_std']:.1f}",
            f"{r['execution_time_ms_mean']:.2f} ± {r['execution_time_ms_std']:.2f} ms",
            f"{r['success_rate_mean']:.1f}%",
        ]
        for r in rows  # type: ignore[union-attr]
    ]
    print(
        tabulate(
            disp,
            headers=[
                "Givens",
                "CSP Mode",
                "Assignments",
                "Backtracks",
                "Constraint Checks",
                "Domain Reductions",
                "Time (ms)",
                "Success",
            ],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()
