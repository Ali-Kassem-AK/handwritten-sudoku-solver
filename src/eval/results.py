"""Master experiment runner and publication-quality plot generator (`src/eval/results.py`).

Executes all three required experiments (if not already run) and generates
high-resolution figures (`results/plots/`) and summary Markdown/CSV/JSON tables (`results/tables/`):
1. `results/plots/experiment1_search.png`: UCS vs Greedy vs A* across 0–4 misreads (Nodes Expanded,
   Nodes Generated, Exact Recovery Rate, and Execution Time with error bars across 5 seeds).
2. `results/plots/experiment2_csp.png`: BT vs BT+FC vs BT+FC+AC3 across 25, 30, 35 givens
   (Assignments, Backtracks, Constraint Checks, Domain Reductions, and Execution Time).
3. `results/plots/experiment3_correction.png`: NN-Guided vs No Correction vs Random Correction
   across 0–3 misreads (Exact Ground-Truth Recovery Rate vs Solvability Rate, Correction Cost, and Time).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.experiment_correction import run_experiment_correction
from src.eval.experiment_csp import run_experiment_csp
from src.eval.experiment_search import run_experiment_search


def plot_experiment1_search(summary_df: pd.DataFrame, save_path: Path) -> None:
    """Plot 3-panel comparison for Experiment 1 (UCS vs Greedy vs A* across 0-4 misreads)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=180)
    core_df = summary_df[summary_df["algorithm"].isin(["UCS", "Greedy", "A*"])].copy()

    colors = {"UCS": "#dc2626", "Greedy": "#d97706", "A*": "#2563eb"}
    markers = {"UCS": "o", "Greedy": "s", "A*": "^"}

    # Panel 1: Nodes Expanded vs Misread Level
    ax0 = axes[0]
    for alg in ["UCS", "Greedy", "A*"]:
        sub = core_df[core_df["algorithm"] == alg].sort_values("misreads")
        ax0.errorbar(
            sub["misreads"],
            sub["nodes_expanded_mean"],
            yerr=sub["nodes_expanded_std"],
            marker=markers[alg],
            color=colors[alg],
            linewidth=2.0,
            capsize=4,
            label=alg,
        )
    ax0.set_title("Exp 1A: Nodes Expanded vs. Injected Misreads", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Number of Injected Digit Misreads (0–4)", fontsize=10)
    ax0.set_ylabel("Mean Nodes Expanded (±1 SD)", fontsize=10)
    ax0.set_xticks([0, 1, 2, 3, 4])
    ax0.grid(True, linestyle=":", alpha=0.5)
    ax0.legend(loc="upper left", fontsize=9)

    # Panel 2: Nodes Generated vs Misread Level
    ax1 = axes[1]
    for alg in ["UCS", "Greedy", "A*"]:
        sub = core_df[core_df["algorithm"] == alg].sort_values("misreads")
        ax1.errorbar(
            sub["misreads"],
            sub["nodes_generated_mean"],
            yerr=sub["nodes_generated_std"],
            marker=markers[alg],
            color=colors[alg],
            linewidth=2.0,
            capsize=4,
            label=alg,
        )
    ax1.set_title("Exp 1B: Nodes Generated vs. Injected Misreads", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Number of Injected Digit Misreads (0–4)", fontsize=10)
    ax1.set_ylabel("Mean Nodes Generated (±1 SD)", fontsize=10)
    ax1.set_xticks([0, 1, 2, 3, 4])
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=9)

    # Panel 3: Exact Recovery Rate & Execution Time
    ax2 = axes[2]
    for alg in ["UCS", "Greedy", "A*"]:
        sub = core_df[core_df["algorithm"] == alg].sort_values("misreads")
        ax2.errorbar(
            sub["misreads"],
            sub["exact_recovery_rate_mean"],
            yerr=sub["exact_recovery_rate_std"],
            marker=markers[alg],
            color=colors[alg],
            linewidth=2.0,
            capsize=4,
            label=f"{alg} (Budget=max(3,k))",
        )
    # Also show the 4-misread Budget=3 point
    b3_row = summary_df[summary_df["algorithm"] == "A* (Budget=3)"]
    if not b3_row.empty:
        ax2.scatter(
            [4],
            b3_row["exact_recovery_rate_mean"],
            color="#7c3aed",
            marker="X",
            s=90,
            zorder=5,
            label="A* at 4 Misreads (Budget=3 Cap)",
        )
    ax2.set_title("Exp 1C: Exact Ground-Truth Recovery Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Number of Injected Digit Misreads (0–4)", fontsize=10)
    ax2.set_ylabel("Exact Solution Recovery Rate (%)", fontsize=10)
    ax2.set_xticks([0, 1, 2, 3, 4])
    ax2.set_ylim(-5, 108)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_experiment2_csp(summary_df: pd.DataFrame, save_path: Path) -> None:
    """Plot 3-panel comparison for Experiment 2 (BT vs BT+FC vs BT+FC+AC3 across 25, 30, 35 givens)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=180)
    givens_list = [25, 30, 35]
    modes = ["BT", "BT+FC", "BT+FC+AC3"]
    colors = {"BT": "#dc2626", "BT+FC": "#2563eb", "BT+FC+AC3": "#0d9488"}

    x = np.arange(len(givens_list))
    w = 0.25

    # Panel 1: Backtracks (Log Scale)
    ax0 = axes[0]
    for idx, m in enumerate(modes):
        sub = summary_df[summary_df["mode"] == m].sort_values("givens")
        vals = np.maximum(sub["backtracks_mean"].to_numpy(), 0.5)
        ax0.bar(x + (idx - 1) * w, vals, width=w, color=colors[m], label=m)
    ax0.set_yscale("log")
    ax0.set_xticks(x)
    ax0.set_xticklabels([f"{g} Givens" for g in givens_list], fontsize=10)
    ax0.set_ylabel("Mean Backtracks (Log Scale)", fontsize=10)
    ax0.set_title("Exp 2A: Backtracks by Solver & Givens Count", fontsize=11, fontweight="bold")
    ax0.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax0.legend(loc="upper right", fontsize=9)

    # Panel 2: Constraint Checks & Domain Reductions
    ax1 = axes[1]
    for idx, m in enumerate(modes):
        sub = summary_df[summary_df["mode"] == m].sort_values("givens")
        ax1.bar(
            x + (idx - 1) * w,
            sub["constraint_checks_mean"],
            width=w,
            color=colors[m],
            label=f"{m} Checks",
        )
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{g} Givens" for g in givens_list], fontsize=10)
    ax1.set_ylabel("Mean Constraint Checks (Log Scale)", fontsize=10)
    ax1.set_title("Exp 2B: Constraint Checks Across Givens", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 3: Execution Time (ms, Log Scale)
    ax2 = axes[2]
    for idx, m in enumerate(modes):
        sub = summary_df[summary_df["mode"] == m].sort_values("givens")
        ax2.bar(
            x + (idx - 1) * w,
            sub["execution_time_ms_mean"],
            width=w,
            color=colors[m],
            label=m,
        )
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{g} Givens" for g in givens_list], fontsize=10)
    ax2.set_ylabel("Mean Execution Time in ms (Log Scale)", fontsize=10)
    ax2.set_title("Exp 2C: Solver Runtime vs. Filtering Overhead", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_experiment3_correction(summary_df: pd.DataFrame, save_path: Path) -> None:
    """Plot 3-panel comparison for Experiment 3 (NN-Guided vs No Correction vs Random Correction)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=180)
    strategies = ["nn_guided", "random_correction", "no_correction"]
    labels = {
        "nn_guided": "NN-Guided Correction (A*)",
        "random_correction": "Random Correction (Uninformed)",
        "no_correction": "No Correction (Direct CSP)",
    }
    colors = {
        "nn_guided": "#2563eb",
        "random_correction": "#d97706",
        "no_correction": "#dc2626",
    }
    markers = {"nn_guided": "o", "random_correction": "s", "no_correction": "^"}

    # Panel 1: Exact Ground-Truth Recovery Rate (%)
    ax0 = axes[0]
    for s in strategies:
        sub = summary_df[summary_df["strategy"] == s].sort_values("misreads")
        ax0.errorbar(
            sub["misreads"],
            sub["exact_recovery_rate_mean"],
            yerr=sub["exact_recovery_rate_std"],
            marker=markers[s],
            color=colors[s],
            linewidth=2.2,
            capsize=4,
            label=labels[s],
        )
    ax0.set_title("Exp 3A: Exact Ground-Truth Solution Recovery (%)", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Number of Injected Digit Misreads (0–3)", fontsize=10)
    ax0.set_ylabel("Exact Solution Recovery Rate (%)", fontsize=10)
    ax0.set_xticks([0, 1, 2, 3])
    ax0.set_ylim(-5, 108)
    ax0.grid(True, linestyle=":", alpha=0.5)
    ax0.legend(loc="lower left", fontsize=8.5)

    # Panel 2: Solvability Rate vs. Exact Recovery Rate (Highlighting Spurious Solvable Grids in Random)
    ax1 = axes[1]
    for s in strategies:
        sub = summary_df[summary_df["strategy"] == s].sort_values("misreads")
        ax1.errorbar(
            sub["misreads"],
            sub["solvable_rate_mean"],
            yerr=sub["solvable_rate_std"],
            marker=markers[s],
            color=colors[s],
            linewidth=2.2,
            linestyle="--" if s == "random_correction" else "-",
            capsize=4,
            label=f"{labels[s]} (Solvable)",
        )
    ax1.set_title("Exp 3B: Solvability Rate (% Merely Solvable Grids)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Number of Injected Digit Misreads (0–3)", fontsize=10)
    ax1.set_ylabel("Puzzle Solvability Rate (%)", fontsize=10)
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_ylim(-5, 108)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=8.5)

    # Panel 3: Execution Time (ms)
    ax2 = axes[2]
    for s in strategies:
        sub = summary_df[summary_df["strategy"] == s].sort_values("misreads")
        ax2.errorbar(
            sub["misreads"],
            sub["execution_time_ms_mean"],
            yerr=sub["execution_time_ms_std"],
            marker=markers[s],
            color=colors[s],
            linewidth=2.2,
            capsize=4,
            label=labels[s],
        )
    ax2.set_title("Exp 3C: Mean Execution Time per Puzzle (ms)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Number of Injected Digit Misreads (0–3)", fontsize=10)
    ax2.set_ylabel("Execution Time in ms (±1 SD)", fontsize=10)
    ax2.set_xticks([0, 1, 2, 3])
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def run_all_experiments_and_plots(
    project_root: str | Path,
    search_puzzles_per_seed: int = 20,
    csp_puzzles_per_seed: int = 10,
    correction_puzzles_per_seed: int = 20,
) -> Dict[str, object]:
    """Run all three primary experiments and generate all comparison tables and plots."""
    root = Path(project_root)
    plots_dir = root / "results" / "plots"
    tables_dir = root / "results" / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    exp1 = run_experiment_search(root, puzzles_per_seed=search_puzzles_per_seed)
    exp2 = run_experiment_csp(root, puzzles_per_seed=csp_puzzles_per_seed)
    exp3 = run_experiment_correction(root, puzzles_per_seed=correction_puzzles_per_seed)

    df1 = pd.DataFrame(exp1["summary"])  # type: ignore[arg-type]
    df2 = pd.DataFrame(exp2["summary"])  # type: ignore[arg-type]
    df3 = pd.DataFrame(exp3["summary"])  # type: ignore[arg-type]

    plot_experiment1_search(df1, plots_dir / "experiment1_search.png")
    plot_experiment2_csp(df2, plots_dir / "experiment2_csp.png")
    plot_experiment3_correction(df3, plots_dir / "experiment3_correction.png")

    combined = {
        "experiment_1_search": exp1,
        "experiment_2_csp": exp2,
        "experiment_3_correction": exp3,
    }
    with open(tables_dir / "all_experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all three required experiments and generate plots.")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Project root path.",
    )
    args = parser.parse_args()
    run_all_experiments_and_plots(args.project_root)
    print("=== All Three Experiments & Plots Successfully Generated ===")


if __name__ == "__main__":
    main()
