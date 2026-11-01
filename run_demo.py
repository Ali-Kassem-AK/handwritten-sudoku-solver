#!/usr/bin/env python3
"""Main integration and live demonstration script (`python run_demo.py`).

Satisfies Section 6 and Section 12 Live Demo requirements:
1. Loads or generates a sample Sudoku puzzle and creates its 9x9 MNIST handwriting mosaic.
2. Displays the original image mosaic and saves visual artifacts (`results/plots/demo_cooperation.png`,
   `results/plots/demo_mosaic.png`).
3. Displays the original Neural Network predictions and relevant probability information.
4. Shows at least one deliberately misrecognized digit (default 2 misreads, configurable via `--misreads`).
5. Shows detected duplicate-digit conflicts and AC-3 domain pruning effects on the puzzle.
6. Displays the search trace showing explored correction states (`g(n)`, `h(n)`, `f(n)`).
7. Displays the CSP pruning and performance comparison with and without Forward Checking and AC-3
   (`BT` vs `BT+FC` vs `BT+FC+AC3`).
8. Demonstrates the system operating with NN enabled (`--nn-mode on`) versus NN disabled (`--nn-mode off`),
   or runs both side-by-side (`--nn-mode both`, default) for the 5-minute live demonstration.
9. Displays the final correction log table and the solved 9x9 Sudoku grid, and also demonstrates
   graceful `Unreadable` failure handling when errors exceed the correction budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
from tabulate import tabulate

from src.csp.sudoku import VALID_MODES, SudokuCSPSolver
from src.data.puzzle_generator import generate_sudoku_puzzle
from src.integration.pipeline import HandwrittenSudokuPipeline, PipelineResult
from src.integration.visualization import plot_pipeline_cooperation


def format_sudoku_ascii(grid: np.ndarray, highlight_coords: set[tuple[int, int]] | None = None) -> str:
    """Format a 9x9 Sudoku grid as a clean ASCII board, optionally marking highlighted cells with `*`."""
    highlight = highlight_coords or set()
    lines = ["+---------+---------+---------+"]
    for r in range(9):
        row_cells = []
        for c in range(9):
            val = int(grid[r, c])
            ch = "." if val == 0 else str(val)
            if (r, c) in highlight:
                row_cells.append(f"*{ch}")
            else:
                row_cells.append(f" {ch}")
            if c in (2, 5):
                row_cells.append(" |")
        lines.append("|" + "".join(row_cells) + " |")
        if r in (2, 5, 8):
            lines.append("+---------+---------+---------+")
    return "\n".join(lines)


def print_pipeline_walkthrough(res: PipelineResult, title: str) -> None:
    """Print a detailed, structured walkthrough of a `PipelineResult` to stdout."""
    print("\n" + "=" * 82)
    print(f"  {title}")
    print("=" * 82)
    print(
        f"Configuration: NN={'ENABLED' if res.nn_enabled else 'DISABLED'} | "
        f"Search={res.search_algorithm} | CSP={res.csp_mode} | "
        f"Givens={res.num_givens} | Injected Misreads={res.num_injected_errors} | "
        f"Budget={res.max_corrections_budget} | Seed={res.puzzle_seed}"
    )

    # 1. Ground-Truth Puzzle vs Initial NN Reading
    inj_set = {(e.row, e.col) for e in res.injected_errors}
    print("\n--- [Step 1 & 2] Ground-Truth Sudoku Givens (`. ` = blank) ---")
    print(format_sudoku_ascii(res.ground_truth_puzzle))

    print("\n--- [Step 3 & 4] Initial NN Predictions (`*d` = misread digit) ---")
    print(format_sudoku_ascii(res.initial_predicted_grid, highlight_coords=inj_set))
    print(f"Initial Nonblank Cell Recognition Accuracy: {res.initial_cell_accuracy * 100:.2f}%")

    if res.injected_errors:
        print("\nDeliberately Injected Low-Confidence Misreads & NN Probability Info:")
        inj_rows = []
        for err in res.injected_errors:
            p_mod = err.modified_probabilities
            top3 = sorted([(d, p_mod[d - 1]) for d in range(1, 10)], key=lambda x: x[1], reverse=True)[:3]
            top3_str = ", ".join(f"{d}: {p*100:.1f}%" for d, p in top3)
            inj_rows.append(
                [
                    f"({err.row}, {err.col})",
                    err.ground_truth_digit,
                    err.injected_predicted_digit,
                    f"{p_mod[err.injected_predicted_digit - 1]*100:.1f}%",
                    f"{p_mod[err.ground_truth_digit - 1]*100:.1f}%",
                    top3_str,
                ]
            )
        print(
            tabulate(
                inj_rows,
                headers=["Cell", "True Digit", "Read Digit", "P(Read)", "P(True)", "Top-3 NN Candidates"],
                tablefmt="github",
            )
        )

    # 2. Conflict Detection & AC-3 Pruning on Initial Grid
    print("\n--- [Step 5 & 6] Conflict Detection & AC-3 Domain Pruning on Initial Reading ---")
    if res.initial_conflicts:
        print(f"Detected {len(res.initial_conflicts)} Duplicate-Digit Conflict(s):")
        for ca, cb, dig in res.initial_conflicts:
            print(f"  * Conflict on digit {dig} between cell {ca} and peer cell {cb}")
    else:
        print("Detected 0 direct duplicate-digit conflicts among initial givens.")

    print(
        f"AC-3 Preprocessing Result: consistent={res.ac3_initial_consistent} | "
        f"domain_reductions={res.ac3_initial_metrics.domain_reductions} | "
        f"constraint_checks={res.ac3_initial_metrics.constraint_checks} | "
        f"singleton_domains={res.ac3_singleton_domains_count} | "
        f"emptied_domains={res.ac3_emptied_domains_count}"
    )

    # 3. Search Trace
    print(f"\n--- [Step 7–10] Search Trace ({res.search_algorithm}) ---")
    sm = res.search_result.metrics
    print(
        f"Nodes Generated: {sm.nodes_generated} | Nodes Expanded: {sm.nodes_expanded} | "
        f"Goal Checks: {sm.goal_checks} | Execution Time: {sm.execution_time*1000:.2f} ms"
    )
    if res.search_result.trace:
        trace_rows = []
        for t in res.search_result.trace[:12]:
            if not t.corrections:
                corr_desc = "<Initial Reading (0 edits)>"
            else:
                corr_desc = "; ".join(
                    f"({c['row']},{c['col']}): {c['old_digit']}->{c['new_digit']}"
                    for c in t.corrections
                )
            trace_rows.append(
                [
                    t.step_index,
                    corr_desc,
                    f"{t.g_cost:.4f}",
                    f"{t.h_cost:.4f}",
                    f"{t.f_cost:.4f}",
                    t.num_conflicts,
                    "YES (GOAL)" if t.goal_passed else "No",
                ]
            )
        print(
            tabulate(
                trace_rows,
                headers=["Step #", "Corrections Applied", "g(n)", "h(n)", "f(n)", "Conflicts", "Goal Test"],
                tablefmt="github",
            )
        )

    # 4. Correction Log or Unreadable Failure Diagnostics
    if res.status == "SOLVED" and res.final_solved_grid is not None:
        print("\n--- [Step 11 & 12] Final Correction Log ---")
        if res.correction_log:
            log_dicts = [entry.to_table_dict() for entry in res.correction_log]
            print(tabulate(log_dicts, headers="keys", tablefmt="github"))
        else:
            print("No digit corrections were required (initial reading was already solvable).")

        corr_coords = {(entry.row, entry.col) for entry in res.correction_log}
        print("\n--- Final Solved Sudoku Grid (`*d` = corrected cell) ---")
        print(format_sudoku_ascii(res.final_solved_grid, highlight_coords=corr_coords))
        print(
            f"Verification: Valid Sudoku = True | "
            f"Exact Ground-Truth Puzzle Recovered = {res.exact_puzzle_recovered} | "
            f"Exact Ground-Truth Solution Recovered = {res.exact_solution_recovered}"
        )
    else:
        print("\n--- [Failure Handling] Puzzle Reported as UNREADABLE ---")
        if res.unreadable_report is not None:
            print(f"Reason: {res.unreadable_report.explanation}")
            print("\nLeast Confident Nonblank Cells:")
            lc_rows = [
                [
                    c["cell"],
                    c["predicted_digit"],
                    f"{float(c['confidence'])*100:.1f}%",
                    ", ".join(f"{d} ({p*100:.1f}%)" for d, p in c["top_alternatives"]),  # type: ignore[index]
                ]
                for c in res.unreadable_report.least_confident_cells
            ]
            print(
                tabulate(
                    lc_rows,
                    headers=["Cell", "Predicted Digit", "Confidence", "Top-3 Candidates"],
                    tablefmt="github",
                )
            )


def run_demo(
    project_root: str | Path,
    givens: int = 35,
    misreads: int = 2,
    budget: int = 3,
    search_algorithm: str = "A*",
    csp_mode: str = "BT+FC+AC3",
    nn_mode: str = "both",
    seed: int = 42,
) -> Dict[str, object]:
    """Run the complete demonstration workflow and save visual and JSON outputs."""
    root = Path(project_root)
    pipeline = HandwrittenSudokuPipeline.from_project_root(root)
    pre_puz = generate_sudoku_puzzle(num_givens=givens, seed=seed, ensure_unique=True)

    results_summary: Dict[str, object] = {}
    plots_dir = root / "results" / "plots"
    logs_dir = root / "results" / "logs"
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run with NN Enabled (if nn_mode in {"on", "both"})
    res_on: PipelineResult | None = None
    if nn_mode in ("on", "both"):
        res_on = pipeline.run(
            num_givens=givens,
            num_injected_errors=misreads,
            max_corrections=budget,
            search_algorithm=search_algorithm,
            csp_mode=csp_mode,
            use_nn=True,
            require_unique=False,
            seed=seed,
            pregenerated_puzzle=pre_puz,
        )
        print_pipeline_walkthrough(
            res_on,
            title=f"LIVE DEMO PART A: System Operating with Neural Network ENABLED ({search_algorithm} + {csp_mode})",
        )
        results_summary["nn_enabled"] = res_on.to_summary_dict()

    # 2. Compare CSP Solver Configurations (BT vs BT+FC vs BT+FC+AC3) on the Corrected Puzzle
    target_puzzle = (
        res_on.corrected_puzzle_grid
        if (res_on and res_on.corrected_puzzle_grid is not None)
        else pre_puz.puzzle
    )
    csp_comp: Dict[str, Dict[str, float | int]] = {}
    csp_rows = []
    for m in VALID_MODES:
        solver = SudokuCSPSolver(mode=m)
        c_res = solver.solve(target_puzzle)
        csp_comp[m] = c_res.metrics.to_dict()
        csp_rows.append(
            [
                m,
                c_res.status,
                c_res.metrics.assignments,
                c_res.metrics.backtracks,
                c_res.metrics.constraint_checks,
                c_res.metrics.domain_reductions,
                f"{c_res.metrics.execution_time * 1000:.2f} ms",
            ]
        )

    print("\n" + "=" * 82)
    print("  LIVE DEMO PART B: CSP Pruning & Performance Comparison (BT vs BT+FC vs BT+FC+AC3)")
    print("=" * 82)
    print(
        tabulate(
            csp_rows,
            headers=[
                "CSP Mode",
                "Status",
                "Assignments",
                "Backtracks",
                "Constraint Checks",
                "Domain Reductions",
                "Execution Time",
            ],
            tablefmt="github",
        )
    )
    results_summary["csp_comparison"] = csp_comp

    # Save cooperation plot and standalone mosaic image from res_on
    if res_on is not None:
        coop_plot_path = plots_dir / "demo_cooperation.png"
        plot_pipeline_cooperation(res_on, coop_plot_path, csp_comparison=csp_comp)
        if res_on.mosaic is not None:
            mosaic_uint8 = np.clip((1.0 - res_on.mosaic.mosaic_image) * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(mosaic_uint8, mode="L").save(plots_dir / "demo_mosaic.png")
        print(f"\n[Saved Visual Cooperation Figure]: {coop_plot_path}")

    # 3. Run with NN Disabled (if nn_mode in {"off", "both"})
    if nn_mode in ("off", "both"):
        res_off = pipeline.run(
            num_givens=givens,
            num_injected_errors=misreads,
            max_corrections=budget,
            search_algorithm=search_algorithm,
            csp_mode=csp_mode,
            use_nn=False,
            require_unique=False,
            seed=seed,
            pregenerated_puzzle=pre_puz,
        )
        print_pipeline_walkthrough(
            res_off,
            title="LIVE DEMO PART C: System Operating with Neural Network DISABLED (Uninformed Correction)",
        )
        results_summary["nn_disabled"] = res_off.to_summary_dict()
        if res_on is None:
            coop_plot_path = plots_dir / "demo_cooperation.png"
            plot_pipeline_cooperation(res_off, coop_plot_path, csp_comparison=csp_comp)

    # 4. Demonstrate Unreadable Failure Handling when Misreads Exceed Budget (e.g. 4 misreads vs budget 2)
    res_unreadable = pipeline.run(
        num_givens=givens,
        num_injected_errors=4,
        max_corrections=2,
        search_algorithm=search_algorithm,
        csp_mode=csp_mode,
        use_nn=True,
        require_unique=False,
        seed=seed + 7,
    )
    print_pipeline_walkthrough(
        res_unreadable,
        title="LIVE DEMO PART D: Graceful 'Unreadable' Failure Handling (4 Misreads with Budget=2)",
    )
    results_summary["unreadable_demo"] = res_unreadable.to_summary_dict()

    log_path = logs_dir / "demo_run_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n[Saved Complete Demo Log JSON]: {log_path}")
    return results_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Handwritten Sudoku Reader and Solver — Main Integration & Live Demo."
    )
    parser.add_argument("--givens", type=int, default=35, help="Number of givens in demo puzzle (default: 35).")
    parser.add_argument("--misreads", type=int, default=2, help="Number of injected misreads (default: 2).")
    parser.add_argument("--budget", type=int, default=3, help="Maximum correction budget (default: 3).")
    parser.add_argument(
        "--search",
        type=str,
        default="A*",
        choices=["UCS", "Greedy", "A*"],
        help="Search algorithm (UCS, Greedy, or A*).",
    )
    parser.add_argument(
        "--csp",
        type=str,
        default="BT+FC+AC3",
        choices=list(VALID_MODES),
        help="CSP solver mode (BT, BT+FC, or BT+FC+AC3).",
    )
    parser.add_argument(
        "--nn-mode",
        type=str,
        default="both",
        choices=["on", "off", "both"],
        help="Switch Neural Network on, off, or run both for comparison (default: both).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args()

    run_demo(
        project_root=Path(__file__).resolve().parent,
        givens=args.givens,
        misreads=args.misreads,
        budget=args.budget,
        search_algorithm=args.search,
        csp_mode=args.csp,
        nn_mode=args.nn_mode,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
