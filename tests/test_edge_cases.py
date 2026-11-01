"""Unit tests for Edge Cases (>= 3) and Impossible Cases (>= 2).

Follows Section 8 Testing Requirements:
- At least 3 edge cases (e.g., already solved 81-given grid, zero-misread puzzle requiring 0 edits,
  invalid input grid dimensions / out-of-range digit values).
- At least 2 impossible cases (e.g., direct duplicate givens in same row/col/box, and an
  unsolvable Sudoku where no immediate duplicate exists among givens but a cell has zero legal
  values left because digits 1..9 all appear in its peers).
- All impossible cases MUST produce a clear `"No solution"` (`status == "IMPOSSIBLE"`) result
  instead of crashing, looping indefinitely, or silently returning an invalid grid.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.csp.sudoku import VALID_MODES, SudokuCSPSolver
from src.data.puzzle_generator import generate_sudoku_puzzle
from src.search.astar import astar_search
from src.search.problem import SudokuCorrectionProblem
from src.search.ucs import uniform_cost_search


class TestEdgeAndImpossibleCases(unittest.TestCase):
    """Test suite for edge cases and impossible Sudoku puzzles."""

    # --- EDGE CASES (4 tests) ---

    def test_edge_case_1_already_completed_81_givens_grid(self) -> None:
        """Edge Case 1: Puzzle with all 81 cells already validly filled requires 0 assignments/corrections."""
        puz = generate_sudoku_puzzle(num_givens=81, seed=11)
        for mode in VALID_MODES:
            solver = SudokuCSPSolver(mode=mode)
            res = solver.solve(puz.solution)
            self.assertEqual(res.status, "SOLVED")
            self.assertTrue(res.solved)
            self.assertEqual(res.metrics.assignments, 0)
            self.assertEqual(res.metrics.backtracks, 0)
            self.assertTrue(np.array_equal(res.solution, puz.solution))

    def test_edge_case_2_zero_misreads_initial_state_is_goal(self) -> None:
        """Edge Case 2: Search problem with 0 misreads terminates at step 1 with 0 corrections and 0.0 cost."""
        puz = generate_sudoku_puzzle(num_givens=35, seed=22)
        probs = np.full((9, 9, 9), 0.01, dtype=np.float64)
        for r in range(9):
            for c in range(9):
                d = int(puz.puzzle[r, c])
                if d != 0:
                    probs[r, c, d - 1] = 0.92
                    probs[r, c] /= np.sum(probs[r, c])
        prob = SudokuCorrectionProblem(puz.puzzle, probs, max_corrections=3)
        res = astar_search(prob)
        self.assertTrue(res.solution_found)
        self.assertEqual(res.metrics.num_corrections, 0)
        self.assertEqual(res.metrics.final_correction_cost, 0.0)
        self.assertEqual(res.metrics.nodes_expanded, 1)

    def test_edge_case_3_invalid_grid_dimensions_and_out_of_range_digits(self) -> None:
        """Edge Case 3: Malformed grid shapes or invalid digit values (e.g. -1 or 10) return INVALID_INPUT."""
        solver = SudokuCSPSolver(mode="BT+FC+AC3")

        bad_shape = np.zeros((8, 9), dtype=np.int64)
        res_shape = solver.solve(bad_shape)
        self.assertEqual(res_shape.status, "INVALID_INPUT")
        self.assertFalse(res_shape.solved)
        self.assertIsNone(res_shape.solution)

        bad_val = np.zeros((9, 9), dtype=np.int64)
        bad_val[0, 0] = 10
        res_val = solver.solve(bad_val)
        self.assertEqual(res_val.status, "INVALID_INPUT")
        self.assertFalse(res_val.solved)

    def test_edge_case_4_zero_correction_budget_on_corrupted_grid(self) -> None:
        """Edge Case 4: Setting max_corrections=0 on a corrupted puzzle immediately fails after 1 goal check."""
        puz = generate_sudoku_puzzle(num_givens=35, seed=33)
        corrupted = puz.puzzle.copy()
        # Introduce a duplicate in Row 0
        corrupted[0, 0] = 9
        corrupted[0, 1] = 9
        probs = np.full((9, 9, 9), 1.0 / 9.0, dtype=np.float64)
        prob = SudokuCorrectionProblem(corrupted, probs, max_corrections=0)
        res = uniform_cost_search(prob)
        self.assertFalse(res.solution_found)
        self.assertEqual(res.metrics.nodes_expanded, 1)
        self.assertIsNone(res.solved_grid)

    # --- IMPOSSIBLE CASES (3 tests) ---

    def test_impossible_case_1_contradictory_duplicate_givens(self) -> None:
        """Impossible Case 1: Grid with duplicate fixed givens in a row, column, or box returns 'No solution'."""
        puz = generate_sudoku_puzzle(num_givens=30, seed=44)
        impossible_grid = puz.puzzle.copy()
        impossible_grid[2, 2] = 7
        impossible_grid[2, 8] = 7  # Contradictory 7s in Row 2

        for mode in VALID_MODES:
            solver = SudokuCSPSolver(mode=mode)
            res = solver.solve(impossible_grid)
            self.assertEqual(res.status, "IMPOSSIBLE")
            self.assertFalse(res.solved)
            self.assertIsNone(res.solution)
            self.assertIn("No solution", res.message)

    def test_impossible_case_2_locally_consistent_but_globally_unsolvable_cell(self) -> None:
        """Impossible Case 2: No duplicate givens exist anywhere, but cell (0, 0) has all digits 1..9 blocked by peers.

        Constructs a grid where:
        - Row 0 contains givens 1, 2, 3, 4 in columns 3..6
        - Col 0 contains givens 5, 6, 7, 8 in rows 3..6
        - Box 0 peer of (0, 0) contains given 9 at (1, 1)
        No two givens share a row, column, or box with the same value (so `find_duplicate_conflicts` is empty!),
        yet open cell `(0, 0)` has an empty legal domain `{}`. All three CSP modes must return `IMPOSSIBLE`
        with `"No solution"`.
        """
        grid = np.zeros((9, 9), dtype=np.int64)
        # Row 0 peers of (0, 0)
        grid[0, 3] = 1
        grid[0, 4] = 2
        grid[0, 5] = 3
        grid[0, 6] = 4
        # Col 0 peers of (0, 0)
        grid[3, 0] = 5
        grid[4, 0] = 6
        grid[5, 0] = 7
        grid[6, 0] = 8
        # Box 0 peer of (0, 0)
        grid[1, 1] = 9

        for mode in VALID_MODES:
            solver = SudokuCSPSolver(mode=mode)
            res = solver.solve(grid)
            self.assertEqual(res.status, "IMPOSSIBLE")
            self.assertFalse(res.solved)
            self.assertIsNone(res.solution)
            self.assertIn("No solution", res.message)

    def test_impossible_case_3_unrecoverable_corrupted_puzzle_in_search(self) -> None:
        """Impossible Case 3: Search on a puzzle with 3 independent duplicate conflicts under budget=1 returns no solution."""
        puz = generate_sudoku_puzzle(num_givens=35, seed=55)
        grid = puz.puzzle.copy()
        # Introduce 3 disjoint duplicate-digit conflicts in rows 0, 4, and 8
        grid[0, 0] = 9
        grid[0, 1] = 9
        grid[4, 2] = 8
        grid[4, 3] = 8
        grid[8, 6] = 7
        grid[8, 7] = 7

        probs = np.full((9, 9, 9), 1.0 / 9.0, dtype=np.float64)
        problem = SudokuCorrectionProblem(grid, probs, max_corrections=1, top_k_alternatives=3)
        for search_fn in (uniform_cost_search, astar_search):
            s_res = search_fn(problem)
            self.assertFalse(s_res.solution_found)
            self.assertIsNone(s_res.solved_grid)
            self.assertIn("exhausted", s_res.failure_reason)


if __name__ == "__main__":
    unittest.main()
