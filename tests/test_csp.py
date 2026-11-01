"""Unit tests for CSP Sudoku solvers (BT, BT+FC, BT+FC+AC3), Forward Checking pruning, and AC-3."""

from __future__ import annotations

import unittest

import numpy as np

from src.csp.ac3 import run_ac3
from src.csp.metrics import CSPMetrics
from src.csp.sudoku import (
    NEIGHBORS,
    SudokuCSPSolver,
    initialize_domains,
    validate_sudoku_input,
)
from src.data.puzzle_generator import (
    generate_sudoku_puzzle,
    is_valid_completed_sudoku,
)


class TestCSPSolver(unittest.TestCase):
    """Test suite for `src/csp/` modules."""

    def test_all_three_csp_modes_solve_valid_puzzle(self) -> None:
        """Normal Case 5: Verify BT, BT+FC, and BT+FC+AC3 all solve a valid Sudoku to the exact ground truth."""
        puz = generate_sudoku_puzzle(num_givens=30, seed=42, ensure_unique=True)

        results = {}
        for mode in ("BT", "BT+FC", "BT+FC+AC3"):
            solver = SudokuCSPSolver(mode=mode)
            res = solver.solve(puz.puzzle)
            self.assertEqual(res.status, "SOLVED")
            self.assertTrue(res.solved)
            self.assertIsNotNone(res.solution)
            self.assertTrue(is_valid_completed_sudoku(res.solution))  # type: ignore[arg-type]
            self.assertTrue(np.array_equal(res.solution, puz.solution))
            results[mode] = res

        # Verify Forward Checking and AC-3 record positive domain reductions while pure BT records 0
        self.assertEqual(results["BT"].metrics.domain_reductions, 0)
        self.assertGreater(results["BT+FC"].metrics.domain_reductions, 0)
        self.assertGreater(results["BT+FC+AC3"].metrics.domain_reductions, 0)
        # Verify filtering reduces assignments and backtracks compared to pure BT
        self.assertLess(results["BT+FC"].metrics.backtracks, results["BT"].metrics.backtracks)
        self.assertLessEqual(results["BT+FC+AC3"].metrics.backtracks, results["BT+FC"].metrics.backtracks)

    def test_ac3_detects_equal_fixed_givens_by_emptying_domain(self) -> None:
        """Verify AC-3 directly detects two equal fixed values in the same row, column, or box via an empty domain."""
        grid = np.zeros((9, 9), dtype=np.int64)
        grid[0, 0] = 5
        grid[0, 7] = 5  # Duplicate 5 in Row 0
        domains, _ = initialize_domains(grid)
        metrics = CSPMetrics()
        consistent, emptied_cell = run_ac3(domains, NEIGHBORS, metrics)
        self.assertFalse(consistent)
        self.assertIn(emptied_cell, [(0, 0), (0, 7)])
        self.assertEqual(len(domains[emptied_cell]), 0)  # type: ignore[index]

    def test_solution_counting_for_uniqueness(self) -> None:
        """Verify CSP solver counts multiple solutions on an under-constrained grid and 1 on a unique puzzle."""
        puz = generate_sudoku_puzzle(num_givens=35, seed=88, ensure_unique=True)
        solver = SudokuCSPSolver(mode="BT+FC")
        res_unique = solver.solve(puz.puzzle, count_solutions=True, count_limit=2)
        self.assertEqual(res_unique.solutions_count, 1)

        # An almost empty grid with only 2 givens has >1 solutions
        sparse = np.zeros((9, 9), dtype=np.int64)
        sparse[0, 0] = 1
        sparse[1, 3] = 2
        res_multi = solver.solve(sparse, count_solutions=True, count_limit=2)
        self.assertEqual(res_multi.solutions_count, 2)


if __name__ == "__main__":
    unittest.main()
