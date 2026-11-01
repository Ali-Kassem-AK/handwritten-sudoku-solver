"""Automated test suite runner for Handwritten Sudoku Reader and Solver."""

from __future__ import annotations

import sys
import unittest

from tests.test_csp import TestCSPSolver
from tests.test_data import TestDataAndGenerators
from tests.test_edge_cases import TestEdgeAndImpossibleCases
from tests.test_integration import TestIntegrationPipeline
from tests.test_nn import TestNeuralNetwork
from tests.test_search import TestSearchAlgorithms


def run_all_tests() -> unittest.TestResult:
    """Load and execute all 6 test modules with verbosity=2."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_cls in (
        TestDataAndGenerators,
        TestNeuralNetwork,
        TestSearchAlgorithms,
        TestCSPSolver,
        TestIntegrationPipeline,
        TestEdgeAndImpossibleCases,
    ):
        suite.addTests(loader.loadTestsFromTestCase(test_cls))
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    res = run_all_tests()
    sys.exit(0 if res.wasSuccessful() else 1)
