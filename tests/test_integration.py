"""Unit tests for the end-to-end integration pipeline, NN on/off switching, and correction logging."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from src.integration.pipeline import HandwrittenSudokuPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestIntegrationPipeline(unittest.TestCase):
    """Test suite for `src/integration/` modules."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = HandwrittenSudokuPipeline.from_project_root(PROJECT_ROOT)

    def test_end_to_end_pipeline_recovery_and_logging(self) -> None:
        """Normal Case 6: Verify end-to-end pipeline recovers 2 injected misreads and logs all required fields."""
        res = self.pipeline.run(
            num_givens=35,
            num_injected_errors=2,
            max_corrections=3,
            search_algorithm="A*",
            csp_mode="BT+FC+AC3",
            use_nn=True,
            seed=42,
        )
        self.assertEqual(res.status, "SOLVED")
        self.assertTrue(res.puzzle_solvable)
        self.assertTrue(res.exact_puzzle_recovered)
        self.assertTrue(res.exact_solution_recovered)
        self.assertEqual(len(res.correction_log), 2)

        required_keys = {
            "Cell",
            "Read digit",
            "Corrected digit",
            "Original probability",
            "New probability",
            "Correction cost",
            "Search algorithm",
        }
        for entry in res.correction_log:
            table_row = entry.to_table_dict()
            self.assertEqual(set(table_row.keys()), required_keys)
            self.assertGreaterEqual(float(table_row["Correction cost"]), 0.0)  # type: ignore[arg-type]

    def test_nn_disabled_mode_operation(self) -> None:
        """Verify pipeline operates cleanly when Neural Network guidance is switched off (`use_nn=False`)."""
        res_off = self.pipeline.run(
            num_givens=35,
            num_injected_errors=2,
            max_corrections=3,
            search_algorithm="UCS",
            csp_mode="BT+FC",
            use_nn=False,
            seed=42,
        )
        self.assertFalse(res_off.nn_enabled)
        self.assertIn(res_off.status, ("SOLVED", "UNREADABLE"))

    def test_unreadable_puzzle_handling_when_budget_exceeded(self) -> None:
        """Verify pipeline reports 'Unreadable' with diagnostic details when misreads exceed budget."""
        res = self.pipeline.run(
            num_givens=35,
            num_injected_errors=4,
            max_corrections=1,
            search_algorithm="A*",
            csp_mode="BT+FC+AC3",
            use_nn=True,
            seed=49,
        )
        self.assertEqual(res.status, "UNREADABLE")
        self.assertFalse(res.puzzle_solvable)
        self.assertIsNone(res.final_solved_grid)
        self.assertIsNotNone(res.unreadable_report)
        assert res.unreadable_report is not None
        self.assertEqual(res.unreadable_report.status, "Unreadable")
        self.assertGreater(len(res.unreadable_report.least_confident_cells), 0)
        self.assertIn("Unreadable", res.unreadable_report.explanation)


if __name__ == "__main__":
    unittest.main()
