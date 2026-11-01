"""Metrics, trace step records, and result structures for Search-based digit correction.

Follows Section 4.B requirements:
Each search algorithm reports:
- nodes_generated
- nodes_expanded
- goal_checks
- num_corrections
- final_correction_cost
- solution_found (boolean)
- execution_time
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CorrectionStep:
    """Single action changing one recognized cell `(row, col)` from `old_digit` to `new_digit`."""

    row: int
    col: int
    old_digit: int
    new_digit: int
    old_prob: float
    new_prob: float
    step_cost: float


@dataclass
class SearchTraceEntry:
    """Record of a state expanded during the search trace."""

    step_index: int
    corrections: List[Dict[str, object]]
    g_cost: float
    h_cost: float
    f_cost: float
    num_conflicts: int
    goal_passed: bool


@dataclass
class SearchMetrics:
    """Instrumentation counters for a search algorithm run."""

    nodes_generated: int = 0
    nodes_expanded: int = 0
    goal_checks: int = 0
    num_corrections: int = 0
    final_correction_cost: float = 0.0
    solution_found: bool = False
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        """Convert search counters to dictionary."""
        return {
            "nodes_generated": int(self.nodes_generated),
            "nodes_expanded": int(self.nodes_expanded),
            "goal_checks": int(self.goal_checks),
            "num_corrections": int(self.num_corrections),
            "final_correction_cost": round(float(self.final_correction_cost), 6),
            "solution_found": bool(self.solution_found),
            "execution_time": round(float(self.execution_time), 6),
        }


@dataclass
class SearchResult:
    """Complete result returned by UCS, Greedy Best-First Search, or A* Search."""

    algorithm: str  # "UCS", "Greedy", or "A*"
    solution_found: bool
    corrected_grid: Optional[np.ndarray]  # 9x9 corrected puzzle givens if found
    solved_grid: Optional[np.ndarray]  # 9x9 completed Sudoku solution from CSP solver if found
    corrections: List[CorrectionStep]
    metrics: SearchMetrics
    trace: List[SearchTraceEntry] = field(default_factory=list)
    failure_reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        """Serialize search result to a JSON-compatible dictionary."""
        return {
            "algorithm": self.algorithm,
            "solution_found": self.solution_found,
            "corrected_grid": self.corrected_grid.tolist() if self.corrected_grid is not None else None,
            "solved_grid": self.solved_grid.tolist() if self.solved_grid is not None else None,
            "corrections": [asdict(c) for c in self.corrections],
            "metrics": self.metrics.to_dict(),
            "trace": [asdict(t) for t in self.trace],
            "failure_reason": self.failure_reason,
        }
