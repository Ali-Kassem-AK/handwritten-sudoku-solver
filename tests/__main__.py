"""Automated test suite runner for Handwritten Sudoku Reader and Solver (`python -m tests.__main__` or `pytest`)."""

from __future__ import annotations

import sys

from tests import run_all_tests

if __name__ == "__main__":
    res = run_all_tests()
    sys.exit(0 if res.wasSuccessful() else 1)
