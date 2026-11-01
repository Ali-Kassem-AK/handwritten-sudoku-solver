"""Metrics and result data structures for CSP Sudoku solvers.

Follows Section 4.C requirements:
- Records at least:
  * assignments
  * backtracks
  * constraint_checks
  * domain_reductions
  * execution_time
- Returns an explicit result status:
  * "SOLVED": returns the completed 9x9 Sudoku grid.
  * "IMPOSSIBLE": returns an explicit "No solution" result when no consistent assignment exists.
  * "INVALID_INPUT": provides a meaningful validation error for malformed grids or invalid values.
- Supports solution counting (`solutions_count`) so the optional uniqueness extension can be
  evaluated without changing the core solver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CSPMetrics:
    """Instrumentation counters for a CSP solver run."""

    assignments: int = 0
    backtracks: int = 0
    constraint_checks: int = 0
    domain_reductions: int = 0
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, int | float]:
        """Return dictionary representation of counters."""
        return {
            "assignments": int(self.assignments),
            "backtracks": int(self.backtracks),
            "constraint_checks": int(self.constraint_checks),
            "domain_reductions": int(self.domain_reductions),
            "execution_time": round(float(self.execution_time), 6),
        }


@dataclass
class CSPResult:
    """Standardized return object for all CSP solver configurations."""

    status: str  # "SOLVED", "IMPOSSIBLE", or "INVALID_INPUT"
    solved: bool
    message: str  # e.g., "Solved", "No solution", or validation error details
    solution: Optional[np.ndarray]  # 9x9 numpy array if solved, else None
    mode: str  # "BT", "BT+FC", or "BT+FC+AC3"
    metrics: CSPMetrics
    solutions_count: int = 0
    conflicts: List[Tuple[Tuple[int, int], Tuple[int, int], int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        """Serialize CSP result to a JSON-compatible dictionary."""
        return {
            "status": self.status,
            "solved": self.solved,
            "message": self.message,
            "solution": self.solution.tolist() if self.solution is not None else None,
            "mode": self.mode,
            "metrics": self.metrics.to_dict(),
            "solutions_count": int(self.solutions_count),
            "conflicts": [
                {"cell_a": list(a), "cell_b": list(b), "digit": int(d)}
                for a, b, d in self.conflicts
            ],
        }
