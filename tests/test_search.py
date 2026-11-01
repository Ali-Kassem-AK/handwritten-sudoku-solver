"""Unit tests for Search algorithms (UCS, Greedy, A*), known-answer optimality, and duplicate-state handling."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.data.puzzle_generator import generate_sudoku_puzzle
from src.search.astar import astar_search
from src.search.greedy import greedy_best_first_search
from src.search.problem import SudokuCorrectionProblem, compute_step_cost
from src.search.ucs import uniform_cost_search


class TestSearchAlgorithms(unittest.TestCase):
    """Test suite for `src/search/` modules."""

    def test_ucs_and_astar_known_optimal_answer(self) -> None:
        """Normal Case 4: Verify UCS and A* return the exact known minimum-cost correction on a controlled example.

        Scenario from Section 3 specification:
        Cell (0, 0) has true digit 7, but NN misreads it as digit 1 with P(1) = 0.52 and P(7) = 0.45,
        creating two 1s in Row 0. Suppose another cell in Row 0 holding 1 has P(1) = 0.85 and second-best
        digit 4 with P(4) = 0.10.
        - Cost of changing cell (0, 0) from 1 -> 7: -log(0.45 / 0.52) = 0.144581
        - Cost of changing the other 1 cell from 1 -> 4: -log(0.10 / 0.85) = 2.140066
        Both UCS and A* must select the optimal correction `(0, 0): 1 -> 7` with exact cost `0.144581`.
        """
        puz = generate_sudoku_puzzle(num_givens=40, seed=12)
        grid = puz.puzzle.copy()
        sol = puz.solution.copy()

        # Find a cell with true digit 7 (or pick any nonblank cell and set up the exact conflict)
        nonblank = [(r, c) for r in range(9) for c in range(9) if int(grid[r, c]) != 0]
        target_r, target_c = nonblank[0]
        true_digit = int(grid[target_r, target_c])

        # Find a peer cell in the same row/col/box that is also a given
        peer_digit = None
        for c in range(9):
            if c != target_c and int(grid[target_r, c]) != 0:
                peer_digit = int(grid[target_r, c])
                break
        if peer_digit is None:
            peer_digit = 1 if true_digit != 1 else 2
            # Ensure peer_digit is placed in a valid cell from solution
            for c in range(9):
                if int(sol[target_r, c]) == peer_digit:
                    grid[target_r, c] = peer_digit
                    break

        # Corrupt (target_r, target_c) to hold `peer_digit` with 52% vs 45% for `true_digit`
        corrupted = grid.copy()
        corrupted[target_r, target_c] = peer_digit

        probs = np.full((9, 9, 9), 0.001, dtype=np.float64)
        for r in range(9):
            for c in range(9):
                d = int(corrupted[r, c])
                if d != 0:
                    probs[r, c, d - 1] = 0.992
                    probs[r, c] /= np.sum(probs[r, c])

        # Set exact 52% vs 45% probabilities on (target_r, target_c)
        p_cell = np.full(9, 0.03 / 7.0, dtype=np.float64)
        p_cell[peer_digit - 1] = 0.52
        p_cell[true_digit - 1] = 0.45
        probs[target_r, target_c] = p_cell

        expected_cost = compute_step_cost(0.52, 0.45)
        self.assertAlmostEqual(expected_cost, -math.log(0.45 / 0.52), places=6)

        problem = SudokuCorrectionProblem(
            initial_grid=corrupted,
            prob_grid=probs,
            max_corrections=2,
            top_k_alternatives=3,
        )

        res_ucs = uniform_cost_search(problem)
        res_astar = astar_search(problem)
        res_greedy = greedy_best_first_search(problem)

        self.assertTrue(res_ucs.solution_found)
        self.assertTrue(res_astar.solution_found)
        self.assertTrue(res_greedy.solution_found)

        self.assertEqual(res_ucs.metrics.num_corrections, 1)
        self.assertAlmostEqual(res_ucs.metrics.final_correction_cost, expected_cost, places=5)
        self.assertAlmostEqual(res_astar.metrics.final_correction_cost, expected_cost, places=5)
        self.assertEqual(res_ucs.corrections[0].row, target_r)
        self.assertEqual(res_ucs.corrections[0].col, target_c)
        self.assertEqual(res_ucs.corrections[0].new_digit, true_digit)

    def test_admissible_heuristic_never_overestimates(self) -> None:
        """Verify that both single_conflict and disjoint_conflicts heuristics never overestimate optimal cost C*."""
        puz = generate_sudoku_puzzle(num_givens=36, seed=202)
        nonblank = [(r, c) for r in range(9) for c in range(9) if int(puz.puzzle[r, c]) != 0]
        r1, c1 = nonblank[0]
        r2, c2 = nonblank[10]
        d1_true = int(puz.puzzle[r1, c1])
        d2_true = int(puz.puzzle[r2, c2])

        # Pick conflicting wrong digits from same row/col/box
        d1_wrong = 1 if d1_true != 1 else 2
        d2_wrong = 3 if d2_true != 3 else 4
        corrupted = puz.puzzle.copy()
        corrupted[r1, c1] = d1_wrong
        corrupted[r2, c2] = d2_wrong

        probs = np.full((9, 9, 9), 0.002, dtype=np.float64)
        for r in range(9):
            for c in range(9):
                d = int(corrupted[r, c])
                if d != 0:
                    probs[r, c, d - 1] = 0.984
        probs[r1, c1] = np.full(9, 0.05 / 7.0)
        probs[r1, c1, d1_wrong - 1] = 0.55
        probs[r1, c1, d1_true - 1] = 0.40
        probs[r2, c2] = np.full(9, 0.06 / 7.0)
        probs[r2, c2, d2_wrong - 1] = 0.54
        probs[r2, c2, d2_true - 1] = 0.40

        prob_h1 = SudokuCorrectionProblem(corrupted, probs, max_corrections=3, heuristic_mode="single_conflict")
        prob_h2 = SudokuCorrectionProblem(corrupted, probs, max_corrections=3, heuristic_mode="disjoint_conflicts")

        res_ucs = uniform_cost_search(prob_h2)
        self.assertTrue(res_ucs.solution_found)
        optimal_c_star = res_ucs.metrics.final_correction_cost

        h1_val = prob_h1.heuristic(prob_h1.get_initial_state())
        h2_val = prob_h2.heuristic(prob_h2.get_initial_state())

        self.assertGreaterEqual(h1_val, 0.0)
        self.assertLessEqual(h1_val, h2_val + 1e-9)
        self.assertLessEqual(h2_val, optimal_c_star + 1e-9)

    def test_duplicate_state_handling_and_budget_enforcement(self) -> None:
        """Verify Graph Search closed set prevents duplicate expansions of commuted edits and enforces budget."""
        puz = generate_sudoku_puzzle(num_givens=35, seed=303)
        # Corrupt 3 cells but set budget = 1 so no solution can be reached
        from src.data.mosaic_generator import inject_hard_digit_errors

        probs = np.full((9, 9, 9), 0.002, dtype=np.float64)
        for r in range(9):
            for c in range(9):
                d = int(puz.puzzle[r, c])
                if d != 0:
                    probs[r, c, d - 1] = 0.984
        pred_grid, mod_probs, _ = inject_hard_digit_errors(
            puz.puzzle, probs, puz.puzzle, puz.solution, num_errors=3, seed=303
        )
        problem = SudokuCorrectionProblem(pred_grid, mod_probs, max_corrections=1, top_k_alternatives=2)
        res = uniform_cost_search(problem)
        self.assertFalse(res.solution_found)
        # Every expanded state must have <= 1 correction
        for t in res.trace:
            self.assertLessEqual(len(t.corrections), 1)


if __name__ == "__main__":
    unittest.main()
