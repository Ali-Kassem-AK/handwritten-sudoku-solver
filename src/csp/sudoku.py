"""Sudoku CSP formulation and unified interface for the three switchable solver configurations:
1. "BT": Backtracking only (`src/csp/backtracking.py`)
2. "BT+FC": Backtracking + Forward Checking (`src/csp/forward_checking.py`)
3. "BT+FC+AC3": Backtracking + Forward Checking + Arc Consistency AC-3 (`src/csp/ac3.py` + MAC)

Follows Section 4.C requirements:
- 81 variables representing the 81 Sudoku cells `(r, c)` for `r, c in 0..8`.
- Domains containing digits `{1..9}` (or `{given}` for fixed initial givens).
- Binary inequality constraints `X_i != X_j` between every pair of cells sharing a row,
  column, or 3x3 box (20 peers per cell, 810 undirected binary constraints = 1,620 directed arcs).
- Switchable solver configurations (`mode in {"BT", "BT+FC", "BT+FC+AC3"}`) that can be
  independently enabled and evaluated.
- Explicit validation of initial input and conflict detection via both direct duplicate check
  and AC-3 empty-domain detection.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.csp.ac3 import run_ac3
from src.csp.backtracking import solve_backtracking_only
from src.csp.forward_checking import (
    forward_check,
    order_domain_values_lcv,
    select_unassigned_variable_mrv,
    solve_backtracking_fc,
)
from src.csp.metrics import CSPMetrics, CSPResult
from src.data.puzzle_generator import is_valid_completed_sudoku

Cell = Tuple[int, int]
Domains = Dict[Cell, Set[int]]

VALID_MODES = ("BT", "BT+FC", "BT+FC+AC3")


def build_sudoku_topology() -> Tuple[List[Cell], Dict[Cell, List[Cell]], List[List[Cell]]]:
    """Precompute the 81 Sudoku variables, the 20 neighbors per variable, and the 27 alldiff units."""
    variables: List[Cell] = [(r, c) for r in range(9) for c in range(9)]
    units: List[List[Cell]] = []

    # 9 rows
    for r in range(9):
        units.append([(r, c) for c in range(9)])
    # 9 columns
    for c in range(9):
        units.append([(r, c) for r in range(9)])
    # 9 3x3 boxes
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            units.append([(r, c) for r in range(br, br + 3) for c in range(bc, bc + 3)])

    neighbors: Dict[Cell, List[Cell]] = {}
    for r, c in variables:
        peers: Set[Cell] = set()
        for j in range(9):
            if j != c:
                peers.add((r, j))
        for i in range(9):
            if i != r:
                peers.add((i, c))
        br, bc = (r // 3) * 3, (c // 3) * 3
        for i in range(br, br + 3):
            for j in range(bc, bc + 3):
                if (i, j) != (r, c):
                    peers.add((i, j))
        neighbors[(r, c)] = sorted(peers)

    return variables, neighbors, units


VARIABLES, NEIGHBORS, UNITS = build_sudoku_topology()


def validate_sudoku_input(grid: np.ndarray) -> Optional[str]:
    """Validate that `grid` is a 9x9 integer-compatible array with values in 0..9.

    Returns None if valid, or a descriptive validation error message if invalid.
    """
    if not isinstance(grid, np.ndarray):
        try:
            grid = np.asarray(grid)
        except Exception as exc:
            return f"Invalid input type: could not convert to array ({exc})."
    if grid.shape != (9, 9):
        return f"Invalid grid shape: expected (9, 9), got {grid.shape}."
    if not np.issubdtype(grid.dtype, np.integer) and not np.issubdtype(grid.dtype, np.floating):
        return f"Invalid grid dtype: {grid.dtype}."
    if np.any(np.isnan(grid)):
        return "Invalid grid values: NaN detected."
    int_grid = grid.astype(np.int64)
    if not np.all(grid == int_grid):
        return "Invalid grid values: non-integer entries detected."
    if np.any((int_grid < 0) | (int_grid > 9)):
        bad = int_grid[(int_grid < 0) | (int_grid > 9)][0]
        return f"Invalid digit {int(bad)} in grid: allowed values are 0 (blank) or 1..9."
    return None


def find_duplicate_conflicts(
    grid: np.ndarray, metrics: Optional[CSPMetrics] = None
) -> List[Tuple[Cell, Cell, int]]:
    """Find all pairs of fixed cells `(cell_a, cell_b, digit)` with equal digits in the same row, col, or box."""
    conflicts: List[Tuple[Cell, Cell, int]] = []
    seen_pairs: Set[Tuple[Cell, Cell]] = set()
    for cell_a in VARIABLES:
        val_a = int(grid[cell_a[0], cell_a[1]])
        if val_a == 0:
            continue
        for cell_b in NEIGHBORS[cell_a]:
            if cell_a < cell_b:
                if metrics is not None:
                    metrics.constraint_checks += 1
                val_b = int(grid[cell_b[0], cell_b[1]])
                if val_a == val_b:
                    pair = (cell_a, cell_b)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        conflicts.append((cell_a, cell_b, val_a))
    return conflicts


def initialize_domains(grid: np.ndarray) -> Tuple[Domains, Dict[Cell, int]]:
    """Initialize CSP domains `{1..9}` for open cells and `{given}` for fixed cells."""
    domains: Domains = {}
    assignment: Dict[Cell, int] = {}
    for r, c in VARIABLES:
        val = int(grid[r, c])
        if val == 0:
            domains[(r, c)] = set(range(1, 10))
        else:
            domains[(r, c)] = {val}
            assignment[(r, c)] = val
    return domains, assignment


def copy_domains(domains: Domains) -> Domains:
    """Deep-copy a domain dictionary."""
    return {cell: dom.copy() for cell, dom in domains.items()}


def _solve_bt_fc_ac3(
    assignment: Dict[Cell, int],
    domains: Domains,
    metrics: CSPMetrics,
    count_limit: int = 1,
) -> List[Dict[Cell, int]]:
    """Solve Sudoku CSP using Backtracking + Forward Checking + AC-3 (BT + FC + AC-3).

    Runs full AC-3 over all 1,620 arcs initially (`CSP Lecture.pdf`, slide 38: "Can be run as a
    preprocessor or after each assignment"), and at each search step runs Forward Checking
    followed by AC-3 propagation across affected arcs, selecting the next unassigned variable via MRV.
    """
    solutions: List[Dict[Cell, int]] = []

    # 1. Run initial AC-3 over the entire CSP constraint graph
    consistent, _ = run_ac3(domains, NEIGHBORS, metrics, initial_arcs=None)
    if not consistent:
        return solutions

    def recursive_search(cur_assign: Dict[Cell, int], cur_domains: Domains) -> None:
        if len(solutions) >= count_limit:
            return
        if len(cur_assign) == len(VARIABLES):
            solutions.append(cur_assign.copy())
            return

        # MRV naturally picks any variable whose domain was reduced to size 1 by AC-3 first
        var = select_unassigned_variable_mrv(VARIABLES, cur_assign, cur_domains, NEIGHBORS)
        if len(cur_domains[var]) == 0:
            return

        for val in order_domain_values_lcv(var, cur_assign, cur_domains, NEIGHBORS):
            consistent_step = True
            for nb in NEIGHBORS[var]:
                if nb in cur_assign:
                    metrics.constraint_checks += 1
                    if cur_assign[nb] == val:
                        consistent_step = False
                        break
            if not consistent_step:
                continue

            next_domains = copy_domains(cur_domains)
            next_domains[var] = {val}
            cur_assign[var] = val
            metrics.assignments += 1

            # Step 1: Forward Checking from `var = val`
            fc_ok, _ = forward_check(var, val, cur_assign, next_domains, NEIGHBORS, metrics)
            if fc_ok:
                # Step 2: Maintain Arc Consistency (AC-3) seeded from neighbors of `var`
                seed_arcs = [
                    (xk, xj)
                    for xj in NEIGHBORS[var]
                    if xj not in cur_assign
                    for xk in NEIGHBORS[xj]
                    if xk != xj and xk not in cur_assign
                ]
                ac_ok, _ = run_ac3(next_domains, NEIGHBORS, metrics, initial_arcs=seed_arcs)
                if ac_ok:
                    recursive_search(cur_assign, next_domains)
                    if len(solutions) >= count_limit:
                        return

            del cur_assign[var]
            metrics.backtracks += 1

    recursive_search(assignment, domains)
    return solutions


class SudokuCSPSolver:
    """Unified interface for solving and validating Sudoku puzzles across BT, BT+FC, and BT+FC+AC3."""

    def __init__(self, mode: str = "BT+FC+AC3", max_bt_backtracks: int = 250000) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported CSP mode '{mode}'. Choose from {VALID_MODES}.")
        self.mode = mode
        self.max_bt_backtracks = max_bt_backtracks

    def run_ac3_conflict_check(self, grid: np.ndarray) -> Tuple[bool, Domains, CSPMetrics, List[Tuple[Cell, Cell, int]]]:
        """Run standalone AC-3 preprocessing on `grid` to detect conflicts and shrink domains.

        If two equal givens exist in the same row, column, or 3x3 box, `remove_inconsistent_values`
        empties the domain of the first cell (`DOMAIN[X_i] == set()`) and AC-3 immediately
        flags the contradiction.
        """
        metrics = CSPMetrics()
        t0 = time.perf_counter()
        err = validate_sudoku_input(grid)
        if err is not None:
            metrics.execution_time = time.perf_counter() - t0
            return False, {}, metrics, []

        domains, _ = initialize_domains(grid)
        conflicts = find_duplicate_conflicts(grid, metrics=metrics)
        ac_ok, _ = run_ac3(domains, NEIGHBORS, metrics, initial_arcs=None)
        metrics.execution_time = time.perf_counter() - t0
        return (ac_ok and len(conflicts) == 0), domains, metrics, conflicts

    def solve(self, grid: np.ndarray, count_solutions: bool = False, count_limit: int = 2) -> CSPResult:
        """Solve the given 9x9 Sudoku puzzle using the configured solver mode (`BT`, `BT+FC`, or `BT+FC+AC3`).

        Args:
            grid: 9x9 integer array (0 for blank, 1..9 for givens).
            count_solutions: If True, searches up to `count_limit` solutions to test uniqueness.
            count_limit: Maximum number of solutions to count when `count_solutions=True` (default 2).
        """
        metrics = CSPMetrics()
        t0 = time.perf_counter()

        # 1. Validate input grid structure and values
        val_err = validate_sudoku_input(grid)
        if val_err is not None:
            metrics.execution_time = time.perf_counter() - t0
            return CSPResult(
                status="INVALID_INPUT",
                solved=False,
                message=f"Validation error: {val_err}",
                solution=None,
                mode=self.mode,
                metrics=metrics,
                solutions_count=0,
                conflicts=[],
            )

        clean_grid = grid.astype(np.int64)
        domains, assignment = initialize_domains(clean_grid)

        # 2. Check for contradictory fixed givens
        conflicts = find_duplicate_conflicts(clean_grid, metrics=metrics)
        if self.mode == "BT+FC+AC3":
            # Let AC-3 also detect empty domains directly from equal givens
            ac_ok, _ = run_ac3(copy_domains(domains), NEIGHBORS, metrics, initial_arcs=None)
            if not ac_ok or conflicts:
                metrics.execution_time = time.perf_counter() - t0
                return CSPResult(
                    status="IMPOSSIBLE",
                    solved=False,
                    message="No solution: AC-3 detected contradictory givens or empty domain.",
                    solution=None,
                    mode=self.mode,
                    metrics=metrics,
                    solutions_count=0,
                    conflicts=conflicts,
                )
        elif conflicts:
            metrics.execution_time = time.perf_counter() - t0
            return CSPResult(
                status="IMPOSSIBLE",
                solved=False,
                message="No solution: initial givens violate Sudoku row/column/box uniqueness.",
                solution=None,
                mode=self.mode,
                metrics=metrics,
                solutions_count=0,
                conflicts=conflicts,
            )

        # Run the selected solver configuration
        solve_metrics = CSPMetrics()
        t_solve = time.perf_counter()
        domains_run, assignment_run = initialize_domains(clean_grid)
        limit = count_limit if count_solutions else 1

        if self.mode == "BT":
            sols = solve_backtracking_only(
                VARIABLES,
                assignment_run,
                domains_run,
                NEIGHBORS,
                solve_metrics,
                count_limit=limit,
                max_backtracks=self.max_bt_backtracks,
            )
        elif self.mode == "BT+FC":
            sols = solve_backtracking_fc(
                VARIABLES,
                assignment_run,
                domains_run,
                NEIGHBORS,
                solve_metrics,
                count_limit=limit,
            )
        else:  # "BT+FC+AC3"
            sols = _solve_bt_fc_ac3(
                assignment_run,
                domains_run,
                solve_metrics,
                count_limit=limit,
            )

        solve_metrics.execution_time = time.perf_counter() - t_solve

        if not sols:
            return CSPResult(
                status="IMPOSSIBLE",
                solved=False,
                message="No solution",
                solution=None,
                mode=self.mode,
                metrics=solve_metrics,
                solutions_count=0,
                conflicts=[],
            )

        first_sol = sols[0]
        sol_grid = np.zeros((9, 9), dtype=np.int64)
        for (r, c), val in first_sol.items():
            sol_grid[r, c] = val

        # Final verification against all Sudoku constraints
        if not is_valid_completed_sudoku(sol_grid):
            return CSPResult(
                status="IMPOSSIBLE",
                solved=False,
                message="No solution",
                solution=None,
                mode=self.mode,
                metrics=solve_metrics,
                solutions_count=0,
                conflicts=[],
            )

        return CSPResult(
            status="SOLVED",
            solved=True,
            message="Solved",
            solution=sol_grid,
            mode=self.mode,
            metrics=solve_metrics,
            solutions_count=len(sols),
            conflicts=[],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sudoku CSP solver in BT, BT+FC, or BT+FC+AC3 mode.")
    parser.add_argument(
        "--mode",
        type=str,
        default="BT+FC+AC3",
        choices=list(VALID_MODES),
        help="CSP solver mode (BT, BT+FC, or BT+FC+AC3).",
    )
    parser.add_argument("--givens", type=int, default=30, help="Number of givens for test puzzle.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    from src.data.puzzle_generator import generate_sudoku_puzzle

    puz = generate_sudoku_puzzle(num_givens=args.givens, seed=args.seed)
    solver = SudokuCSPSolver(mode=args.mode)
    res = solver.solve(puz.puzzle, count_solutions=True)
    print(f"=== CSP Solver ({args.mode}) on {args.givens}-Given Puzzle (Seed={args.seed}) ===")
    print(f"Status:            {res.status} ({res.message})")
    print(f"Solutions Count:   {res.solutions_count}")
    print(f"Assignments:       {res.metrics.assignments}")
    print(f"Backtracks:        {res.metrics.backtracks}")
    print(f"Constraint Checks: {res.metrics.constraint_checks}")
    print(f"Domain Reductions: {res.metrics.domain_reductions}")
    print(f"Execution Time:    {res.metrics.execution_time*1000:.2f} ms")


if __name__ == "__main__":
    main()
