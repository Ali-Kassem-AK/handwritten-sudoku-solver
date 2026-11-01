"""Visualization utilities for the Handwritten Sudoku Reader and Solver pipeline.

Generates high-resolution figures showing how the three intelligent components
(Neural Network, Search Algorithms, and CSP Solver) cooperate on a Sudoku puzzle:
1. Original 9x9 MNIST Handwritten Mosaic (`252x252` pixels) with 3x3 box boundaries.
2. Initial NN Reading & Confidence Map, highlighting deliberately misrecognized cells
   and duplicate-digit row/column/box conflicts detected by AC-3.
3. Search Trace & Candidate Probabilities for uncertain/corrected cells, showing top
   candidate digits and cumulative correction costs `g(n)` and heuristic `h(n)`.
4. Final Corrected & CSP-Solved Sudoku Grid, distinguishing original givens, search-corrected
   givens, and CSP-inferred solution digits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from src.integration.pipeline import PipelineResult


def _draw_sudoku_grid_lines(ax: plt.Axes, color_thin: str = "#94a3b8", color_thick: str = "#0f172a") -> None:
    """Draw 9x9 cell lines and 3x3 box boundaries on `ax` (coordinate range 0..9)."""
    for i in range(10):
        lw = 2.2 if i % 3 == 0 else 0.6
        col = color_thick if i % 3 == 0 else color_thin
        ax.plot([0, 9], [i, i], color=col, linewidth=lw)
        ax.plot([i, i], [0, 9], color=col, linewidth=lw)
    ax.set_xlim(0, 9)
    ax.set_ylim(9, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def plot_pipeline_cooperation(
    result: PipelineResult,
    save_path: str | Path,
    csp_comparison: Optional[Dict[str, Dict[str, float | int]]] = None,
) -> Path:
    """Generate a comprehensive 4-panel figure illustrating NN + Search + CSP cooperation."""
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16.5, 8.8), dpi=180)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 1.25], height_ratios=[1.0, 1.0], hspace=0.28, wspace=0.22)

    # Panel 1 (Top-Left): Original 9x9 MNIST Handwritten Mosaic
    ax_mosaic = fig.add_subplot(gs[0, 0])
    if result.mosaic is not None:
        # Invert grayscale for clean dark-ink on light-paper appearance
        ax_mosaic.imshow(1.0 - result.mosaic.mosaic_image, cmap="gray", vmin=0.0, vmax=1.0, extent=[0, 9, 9, 0])
    _draw_sudoku_grid_lines(ax_mosaic, color_thin="#cbd5e1", color_thick="#1e293b")
    ax_mosaic.set_title(
        f"1. Input 9×9 MNIST Mosaic ({result.num_givens} Givens)",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )

    # Panel 2 (Top-Middle): Initial NN Predictions, Confidence & AC-3 Conflicts
    ax_init = fig.add_subplot(gs[0, 1])
    _draw_sudoku_grid_lines(ax_init)
    conflict_cells = set()
    for ca, cb, _ in result.initial_conflicts:
        conflict_cells.add(ca)
        conflict_cells.add(cb)
    injected_coords = {(e.row, e.col) for e in result.injected_errors}

    for r in range(9):
        for c in range(9):
            d = int(result.initial_predicted_grid[r, c])
            if d != 0:
                conf = float(result.initial_confidence_grid[r, c])
                is_inj = (r, c) in injected_coords
                is_conf = (r, c) in conflict_cells
                if is_inj:
                    bg_color = "#fee2e2"
                    txt_color = "#b91c1c"
                elif is_conf:
                    bg_color = "#fef3c7"
                    txt_color = "#b45309"
                else:
                    bg_color = "#f1f5f9"
                    txt_color = "#0f172a"
                rect = patches.Rectangle((c, r), 1, 1, facecolor=bg_color, edgecolor="none", zorder=0)
                ax_init.add_patch(rect)
                ax_init.text(
                    c + 0.5,
                    r + 0.42,
                    str(d),
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color=txt_color,
                )
                ax_init.text(
                    c + 0.5,
                    r + 0.80,
                    f"{conf*100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="#475569",
                )

    ax_init.set_title(
        f"2. Initial NN Reading & Conflicts (Red=Misread, Amber=Conflict)",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )

    # Panel 3 (Bottom-Left): Final Corrected & CSP-Solved Sudoku Grid
    ax_sol = fig.add_subplot(gs[1, 0])
    _draw_sudoku_grid_lines(ax_sol)
    corrected_coords = {(entry.row, entry.col) for entry in result.correction_log}

    if result.final_solved_grid is not None:
        for r in range(9):
            for c in range(9):
                val = int(result.final_solved_grid[r, c])
                was_given = int(result.ground_truth_puzzle[r, c]) != 0
                was_corr = (r, c) in corrected_coords
                if was_corr:
                    bg_color = "#dcfce7"
                    txt_color = "#15803d"
                    fw = "bold"
                elif was_given:
                    bg_color = "#f8fafc"
                    txt_color = "#0f172a"
                    fw = "bold"
                else:
                    bg_color = "#ffffff"
                    txt_color = "#2563eb"
                    fw = "normal"
                rect = patches.Rectangle((c, r), 1, 1, facecolor=bg_color, edgecolor="none", zorder=0)
                ax_sol.add_patch(rect)
                ax_sol.text(
                    c + 0.5,
                    r + 0.52,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=11.5,
                    fontweight=fw,
                    color=txt_color,
                )
        ax_sol.set_title(
            f"4. Solved Sudoku (Green=Search Corrected, Blue=CSP Filled)",
            fontsize=10.5,
            fontweight="bold",
            pad=8,
        )
    else:
        ax_sol.text(
            4.5,
            4.5,
            "UNREADABLE PUZZLE\n(No Valid Solution Within Budget)",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#b91c1c",
        )
        ax_sol.set_title("4. Final Status: Unreadable", fontsize=11, fontweight="bold", pad=8)

    # Panel 4 (Bottom-Middle): NN Probability Distributions for Corrected / Uncertain Cells
    ax_probs = fig.add_subplot(gs[1, 1])
    target_cells: List[Tuple[int, int]] = []
    for entry in result.correction_log:
        target_cells.append((entry.row, entry.col))
    if not target_cells and result.injected_errors:
        for err in result.injected_errors:
            target_cells.append((err.row, err.col))
    if not target_cells:
        # Pick 2 lowest-confidence nonblank cells
        nonblank = [
            (float(result.initial_confidence_grid[r, c]), r, c)
            for r in range(9)
            for c in range(9)
            if int(result.initial_predicted_grid[r, c]) != 0
        ]
        nonblank.sort()
        target_cells = [(r, c) for _, r, c in nonblank[:2]]

    digits = np.arange(1, 10)
    width = 0.38 if len(target_cells) > 1 else 0.55
    colors = ["#2563eb", "#0d9488", "#d97706"]
    for idx, (r, c) in enumerate(target_cells[:3]):
        probs = result.initial_probability_grid[r, c] * 100.0
        true_d = int(result.ground_truth_puzzle[r, c])
        init_d = int(result.initial_predicted_grid[r, c])
        offset = (idx - (min(len(target_cells), 3) - 1) / 2.0) * width
        ax_probs.bar(
            digits + offset,
            probs,
            width=width * 0.9,
            color=colors[idx % len(colors)],
            alpha=0.88,
            label=f"Cell ({r},{c}): Read {init_d} → True {true_d}",
        )
    ax_probs.set_xticks(digits)
    ax_probs.set_xlabel("Candidate Sudoku Digit (1–9)", fontsize=9.5)
    ax_probs.set_ylabel("NN Probability (%)", fontsize=9.5)
    ax_probs.set_title("3a. NN Candidate Probabilities on Uncertain Cells", fontsize=10.5, fontweight="bold", pad=8)
    ax_probs.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax_probs.legend(loc="upper right", fontsize=8)

    # Panel 5 (Top-Right): Search Trace Progression (g(n), h(n), f(n))
    ax_trace = fig.add_subplot(gs[0, 2])
    trace = result.search_result.trace
    if trace:
        steps_x = [t.step_index for t in trace]
        g_vals = [t.g_cost for t in trace]
        h_vals = [t.h_cost for t in trace]
        f_vals = [t.f_cost for t in trace]
        ax_trace.plot(steps_x, g_vals, "o-", color="#2563eb", label="Backward Cost g(n)")
        ax_trace.plot(steps_x, h_vals, "s--", color="#d97706", label="Heuristic h(n)")
        ax_trace.plot(steps_x, f_vals, "^-", color="#059669", linewidth=2.0, label="Total Priority f(n)")
        for t in trace:
            if t.goal_passed:
                ax_trace.axvline(t.step_index, color="#16a34a", linestyle=":", linewidth=1.5, label="Goal Reached")
                break
        ax_trace.set_xlabel("Expanded State Index", fontsize=9.5)
        ax_trace.set_ylabel("Correction Penalty Cost", fontsize=9.5)
        ax_trace.set_title(
            f"3b. {result.search_algorithm} Search Trace ({len(trace)} States Expanded)",
            fontsize=10.5,
            fontweight="bold",
            pad=8,
        )
        ax_trace.grid(True, linestyle=":", alpha=0.5)
        ax_trace.legend(loc="best", fontsize=8)

    # Panel 6 (Bottom-Right): CSP Mode Comparison (BT vs BT+FC vs BT+FC+AC3) on the Corrected Puzzle
    ax_csp = fig.add_subplot(gs[1, 2])
    if csp_comparison:
        modes = list(csp_comparison.keys())
        assigns = [float(csp_comparison[m]["assignments"]) for m in modes]
        checks = [float(csp_comparison[m]["constraint_checks"]) for m in modes]
        reducts = [float(csp_comparison[m]["domain_reductions"]) for m in modes]

        x = np.arange(len(modes))
        w = 0.25
        ax_csp.bar(x - w, assigns, width=w, color="#2563eb", label="Assignments")
        ax_csp.bar(x, checks, width=w, color="#d97706", label="Constraint Checks")
        ax_csp.bar(x + w, [max(1.0, r) for r in reducts], width=w, color="#0d9488", label="Domain Reductions")
        ax_csp.set_yscale("log")
        ax_csp.set_xticks(x)
        ax_csp.set_xticklabels(modes, fontsize=9.5, fontweight="bold")
        ax_csp.set_ylabel("Count (Log Scale)", fontsize=9.5)
        ax_csp.set_title("5. CSP Pruning & Solver Comparison on Puzzle", fontsize=10.5, fontweight="bold", pad=8)
        ax_csp.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax_csp.legend(loc="upper right", fontsize=8)
    else:
        ax_csp.axis("off")

    fig.suptitle(
        f"Handwritten Sudoku Reader & Solver — End-to-End Cooperation "
        f"(NN={'ON' if result.nn_enabled else 'OFF'}, Search={result.search_algorithm}, CSP={result.csp_mode})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
