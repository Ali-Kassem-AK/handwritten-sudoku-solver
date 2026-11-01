"""Search Problem formulation and Admissible Heuristics for Sudoku Digit Correction.

Follows `Searching Lecture.pdf` (Dan Klein & Pieter Abbeel, UC Berkeley CS188, slides 11, 72-84, 107-119)
and Section 4.B of the project specification:

1. Search Formulation:
   - State `s`: A canonical `frozenset` of cell corrections `((r, c), new_digit)` made to the
     neural network's initial Sudoku reading `G_0`. Since changing cell A then cell B produces
     the exact same grid as changing cell B then cell A, representing states as canonical sets
     of corrections (and using a `closed` set in Graph Search, slide 107 & 119) prevents duplicate
     states from being re-expanded.
   - Initial State `s_0`: `frozenset()` (zero corrections, corresponding to the initial reading `G_0`).
   - Action `a = ((r, c), d_new)`: Change one uncertain nonblank cell `(r, c)` that has not yet been
     corrected in `s` from its initial predicted digit `d_orig = G_0[r, c]` to one of its top-K
     alternative candidate digits `d_new != d_orig`.
   - Step Cost `c(r, c, d_orig -> d_new)`:
       c(r, c, d_orig -> d_new) = -log( (P(r, c, d_new) + eps) / (P(r, c, d_orig) + eps) )
     where `eps = 1e-12`. Because `d_orig = argmax_d P(r, c, d)` is the neural network's top prediction,
     `P(r, c, d_orig) >= P(r, c, d_new)` for every alternative `d_new`, guaranteeing:
       c(r, c, d_orig -> d_new) >= 0  (strictly nonnegative, and monotonically decreasing with P(d_new)).
     Under conditional independence of cell predictions across the mosaic, the cumulative path cost
       g(s) = sum_{((r, c), d_new) in s} c(r, c, d_orig -> d_new)
     equals the negative log-likelihood ratio `-log( P(G_s) / P(G_0) )`. Therefore, minimizing `g(s)`
     is mathematically equivalent to finding the Maximum A Posteriori (MAP) / most likely valid
     Sudoku grid reading!
   - Goal Test `IS-GOAL(s)`:
     Applies the corrections in `s` to `G_0` to obtain `G_s`. First checks that `G_s` has zero
     duplicate-digit conflicts in any row, column, or 3x3 box, then invokes the CSP solver
     (`SudokuCSPSolver`). In standard mode (`require_unique=False`), returns True iff the CSP
     solver finds a valid completed Sudoku solution (`status == "SOLVED"`). In Optional Extension 1
     mode (`require_unique=True`), returns True iff the CSP solver finds *exactly one* valid solution
     (`solutions_count == 1`).
   - Correction Budget `B`:
     Supports a configurable maximum number of corrections `max_corrections` (default `B = 3`,
     configurable to `B = 4` for the 4-misread full-recovery experiment).

2. Admissible Heuristic `h(s)` (and Consistency Proof):
   - Project Specification Principle: "Use the project specification's idea that at least one cell
     involved in each duplicate-digit conflict must change."
   - For any cell `(r, c)` not yet corrected in `s`, let `min_cost(r, c)` be the cheapest available
     action cost among its top-K alternative candidate digits:
       min_cost(r, c) = min_{d_new in Alternatives(r, c)} c(r, c, d_orig -> d_new) >= 0.
     (If a cell has no allowed alternatives, `min_cost(r, c) = +inf`.)
   - For any duplicate-digit conflict pair `e = (u, v)` in `G_s` (where cells `u` and `v` share a row,
     column, or 3x3 box and currently hold the same digit `G_s[u] == G_s[v]`), any valid goal grid
      MUST change at least one of `u` or `v`. Thus resolving conflict `e` alone requires a cost of at least:
       w(e) = min( min_cost(u), min_cost(v) ).
   - Standard Admissible Heuristic (`heuristic_mode="single_conflict"`):
       h_1(s) = max_{e in Conflicts(G_s)} w(e)   (and 0 if Conflicts(G_s) is empty).
     Admissibility: Since every goal state must resolve every conflict `e in Conflicts(G_s)`, the true
     remaining cost `h*(s)` must be at least `w(e)` for each conflict `e`, hence `h*(s) >= max_e w(e) = h_1(s)`.
   - Improved Admissible Heuristic (`heuristic_mode="disjoint_conflicts"`, Optional Extension 2, enabled by default):
     Constructs a maximal collection `M` of *vertex-disjoint* conflict pairs `{(u_1, v_1), ..., (u_m, v_m)}`
     from `Conflicts(G_s)` (sorted in descending order of `w(e)`). Because the pairs in `M` share NO cells
     in common, no single cell modification can resolve more than one conflict pair in `M`. Therefore, any
     valid correction set must modify at least one distinct cell from each pair `(u_k, v_k) in M`, and the
     sum of their minimum costs is a strict lower bound on `h*(s)`:
       h_2(s) = sum_{(u_k, v_k) in M} min( min_cost(u_k), min_cost(v_k) ) <= h*(s).
     Furthermore, `h_2(s)` dominates `h_1(s)` (`0 <= h_1(s) <= h_2(s) <= h*(s)`), forming a semi-lattice
     dominance improvement as taught in `Searching Lecture.pdf` (slide 103).
"""

from __future__ import annotations

import math
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from src.csp.sudoku import SudokuCSPSolver, find_duplicate_conflicts
from src.search.metrics import CorrectionStep, SearchMetrics

Cell = Tuple[int, int]
State = FrozenSet[Tuple[Cell, int]]


def compute_step_cost(orig_prob: float, new_prob: float, eps: float = 1e-12) -> float:
    """Compute nonnegative probability-based penalty for changing a digit with `orig_prob` to `new_prob`.

    Formula:
        cost = max(0.0, -log((new_prob + eps) / (orig_prob + eps)))
    Since `orig_prob` is the probability of the initial argmax prediction (`orig_prob >= new_prob`),
    `-log((new_prob + eps) / (orig_prob + eps))` is naturally `>= 0`. The `max(0.0, ...)` guard
    ensures nonnegativity even if a custom test case supplies an alternative with higher probability.
    """
    ratio = (float(new_prob) + eps) / (float(orig_prob) + eps)
    if ratio >= 1.0:
        return 0.0
    return float(-math.log(ratio))


class SudokuCorrectionProblem:
    """Formal search problem for correcting misrecognized Sudoku digits."""

    def __init__(
        self,
        initial_grid: np.ndarray,
        prob_grid: np.ndarray,
        csp_solver: Optional[SudokuCSPSolver] = None,
        max_corrections: int = 3,
        top_k_alternatives: int = 3,
        uncertainty_threshold: float = 0.999,
        require_unique: bool = False,
        heuristic_mode: str = "disjoint_conflicts",
    ) -> None:
        """Initialize the Sudoku digit-correction search problem.

        Args:
            initial_grid: 9x9 array of initial recognized digits (0 for blank, 1..9 for nonblank).
            prob_grid: 9x9x9 array of probabilities over digits 1..9 for each cell.
            csp_solver: Instance of `SudokuCSPSolver` used in the goal test (default `BT+FC`).
            max_corrections: Maximum number of cell modifications allowed (default 3).
            top_k_alternatives: Number of top alternative digits considered per uncertain cell (default 3).
            uncertainty_threshold: Cells with top confidence below this threshold OR involved in
                a duplicate conflict are eligible for candidate corrections.
            require_unique: Optional Extension 1 — if True, goal test requires a unique Sudoku solution.
            heuristic_mode: `"disjoint_conflicts"` (Extension 2 dominant admissible heuristic) or
                `"single_conflict"` (base admissible heuristic).
        """
        self.initial_grid = initial_grid.copy().astype(np.int64)
        self.prob_grid = prob_grid.copy().astype(np.float64)
        self.csp_solver = csp_solver if csp_solver is not None else SudokuCSPSolver(mode="BT+FC")
        self.max_corrections = int(max_corrections)
        self.top_k_alternatives = int(top_k_alternatives)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.require_unique = bool(require_unique)
        self.heuristic_mode = heuristic_mode

        # Precompute candidate actions and minimum replacement cost per nonblank cell
        self.candidate_actions: Dict[Cell, List[CorrectionStep]] = {}
        self.min_cell_cost: Dict[Cell, float] = {}
        self._build_candidate_actions()

    def _build_candidate_actions(self) -> None:
        """Precompute the top-K alternative digits and their step costs for each eligible cell."""
        initial_conflicts = find_duplicate_conflicts(self.initial_grid)
        conflicted_cells: Set[Cell] = set()
        for ca, cb, _ in initial_conflicts:
            conflicted_cells.add(ca)
            conflicted_cells.add(cb)

        nonblank_cells: List[Cell] = [
            (r, c)
            for r in range(9)
            for c in range(9)
            if int(self.initial_grid[r, c]) != 0
        ]

        # Determine which cells are eligible to be modified
        for r, c in nonblank_cells:
            orig_d = int(self.initial_grid[r, c])
            probs = self.prob_grid[r, c]
            orig_p = float(probs[orig_d - 1])

            if (r, c) not in conflicted_cells and orig_p >= self.uncertainty_threshold:
                continue

            # Rank alternative digits d != orig_d by descending probability
            alternatives = [
                (d, float(probs[d - 1]))
                for d in range(1, 10)
                if d != orig_d
            ]
            alternatives.sort(key=lambda item: (-item[1], item[0]))
            top_alts = alternatives[: self.top_k_alternatives]

            steps: List[CorrectionStep] = []
            for new_d, new_p in top_alts:
                cost = compute_step_cost(orig_p, new_p)
                steps.append(
                    CorrectionStep(
                        row=r,
                        col=c,
                        old_digit=orig_d,
                        new_digit=new_d,
                        old_prob=orig_p,
                        new_prob=new_p,
                        step_cost=cost,
                    )
                )
            if steps:
                self.candidate_actions[(r, c)] = steps
                self.min_cell_cost[(r, c)] = min(s.step_cost for s in steps)

    def get_initial_state(self) -> State:
        """Return the empty correction set `frozenset()`."""
        return frozenset()

    def apply_state(self, state: State) -> np.ndarray:
        """Return the 9x9 Sudoku grid resulting from applying all corrections in `state`."""
        grid = self.initial_grid.copy()
        for (r, c), new_d in state:
            grid[r, c] = int(new_d)
        return grid

    def get_successors(self, state: State) -> List[Tuple[State, CorrectionStep, float]]:
        """Generate valid successor states `(next_state, action_step, step_cost)`.

        If `len(state) >= self.max_corrections`, returns an empty list (enforcing budget).
        To focus branching when duplicate conflicts exist in the current grid `G_s`,
        prioritizes cells involved in current duplicate conflicts plus the most uncertain cells.
        """
        if len(state) >= self.max_corrections:
            return []

        already_corrected: Set[Cell] = {cell for cell, _ in state}
        cur_grid = self.apply_state(state)
        conflicts = find_duplicate_conflicts(cur_grid)

        if conflicts:
            # If duplicate conflicts exist, any path to a goal must resolve them;
            # consider all cells in candidate_actions (ordered with conflicted cells first).
            conflict_cells: Set[Cell] = set()
            for ca, cb, _ in conflicts:
                conflict_cells.add(ca)
                conflict_cells.add(cb)
            ordered_cells = sorted(
                self.candidate_actions.keys(),
                key=lambda cell: (0 if cell in conflict_cells else 1, self.min_cell_cost[cell], cell),
            )
        else:
            ordered_cells = sorted(
                self.candidate_actions.keys(),
                key=lambda cell: (self.min_cell_cost[cell], cell),
            )

        # Control branching factor to top uncertain/conflicted cells (up to 18 cells)
        max_active_cells = 18
        active_cells = [c for c in ordered_cells if c not in already_corrected][:max_active_cells]

        successors: List[Tuple[State, CorrectionStep, float]] = []
        for cell in active_cells:
            for step in self.candidate_actions[cell]:
                # Skip actions that immediately create a direct duplicate with another fixed cell
                # ONLY if that peer cell cannot itself be changed within the remaining budget
                next_state: State = state | {((step.row, step.col), step.new_digit)}
                successors.append((next_state, step, step.step_cost))

        return successors

    def heuristic(self, state: State) -> float:
        """Compute admissible lower bound `h(state)` on remaining correction cost to a valid Sudoku.

        Mathematical Proof of Admissibility:
        Let `Conflicts(G_s)` be the set of duplicate-digit conflict pairs `(u, v)` in the current
        grid `G_s = apply_state(s)`. In any valid Sudoku grid, no two peer cells `u, v` can hold
        the same digit. Because our state representation never modifies the same cell twice along
        a path, any cell `x` already in `state` has remaining cost `+inf` to change again, while
        any uncorrected cell `x` costs at least `min_cell_cost[x]` to change to any alternative digit.
        Thus, resolving conflict `e = (u, v)` requires spending at least:
            w(u, v) = min( rem_cost(u), rem_cost(v) ).
        When we select a subset of *vertex-disjoint* conflict pairs `M = {(u_1, v_1), ..., (u_m, v_m)}`
        (where all `2m` cells are mutually distinct), no single cell correction can resolve more than
        one pair in `M`. Since action costs are additive and nonnegative, the true minimum remaining
        cost `h*(s)` satisfies:
            h*(s) >= sum_{(u_k, v_k) in M} w(u_k, v_k) = h(s).
        Hence `0 <= h(s) <= h*(s)` for all states `s`, and `h(goal) = 0` because a goal state has
        zero conflicts.
        """
        cur_grid = self.apply_state(state)
        conflicts = find_duplicate_conflicts(cur_grid)
        if not conflicts:
            return 0.0

        already_corrected: Set[Cell] = {cell for cell, _ in state}

        def rem_min_cost(cell: Cell) -> float:
            if cell in already_corrected:
                return float("inf")
            return self.min_cell_cost.get(cell, float("inf"))

        weighted_edges: List[Tuple[float, Cell, Cell]] = []
        for ca, cb, _ in conflicts:
            w = min(rem_min_cost(ca), rem_min_cost(cb))
            weighted_edges.append((w, ca, cb))

        if self.heuristic_mode == "single_conflict":
            return max(w for w, _, _ in weighted_edges)

        # Greedy maximum-weight vertex-disjoint conflict matching (`disjoint_conflicts`)
        weighted_edges.sort(key=lambda item: item[0], reverse=True)
        used_cells: Set[Cell] = set()
        total_h = 0.0
        for w, ca, cb in weighted_edges:
            if ca not in used_cells and cb not in used_cells:
                if math.isinf(w):
                    return float("inf")
                total_h += w
                used_cells.add(ca)
                used_cells.add(cb)

        return float(total_h)

    def is_goal(self, state: State, metrics: SearchMetrics) -> Tuple[bool, Optional[np.ndarray]]:
        """Test whether `state` produces a conflict-free, CSP-solvable (or uniquely solvable) Sudoku."""
        metrics.goal_checks += 1
        cur_grid = self.apply_state(state)

        # Fast conflict check before calling full CSP backtracking
        if find_duplicate_conflicts(cur_grid):
            return False, None

        csp_res = self.csp_solver.solve(
            cur_grid,
            count_solutions=self.require_unique,
            count_limit=2 if self.require_unique else 1,
        )
        if not csp_res.solved or csp_res.solution is None:
            return False, None
        if self.require_unique and csp_res.solutions_count != 1:
            return False, None
        return True, csp_res.solution
