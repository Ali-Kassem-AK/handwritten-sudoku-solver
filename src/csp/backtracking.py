"""Pure Backtracking Search (BT) implementation for Sudoku CSP.

Follows `CSP Lecture.pdf` (Dan Klein & Pieter Abbeel, UC Berkeley CS188, slide 26):

```
function BACKTRACKING-SEARCH(csp) returns solution/failure
  return RECURSIVE-BACKTRACKING({}, csp)

function RECURSIVE-BACKTRACKING(assignment, csp) returns soln/failure
  if assignment is complete then return assignment
  var <- SELECT-UNASSIGNED-VARIABLE(VARIABLES[csp], assignment, csp)
  for each value in ORDER-DOMAIN-VALUES(var, assignment, csp) do
    if value is consistent with assignment given CONSTRAINTS[csp] then
      add {var = value} to assignment
      result <- RECURSIVE-BACKTRACKING(assignment, csp)
      if result != failure then return result
      remove {var = value} from assignment
  return failure
```

In pure BT mode (Filtering = None):
- Does NOT perform Forward Checking or AC-3 domain pruning on unassigned variables.
- Checks constraints incrementally as each value is tried against already-assigned neighbors.
- Selects unassigned variables in static most-constrained peer order (or static row-major order)
  without looking ahead into filtered future domains, so the experimental comparison between
  `BT`, `BT+FC`, and `BT+FC+AC3` genuinely reflects the impact of domain filtering and propagation.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from src.csp.metrics import CSPMetrics

Cell = Tuple[int, int]
Domains = Dict[Cell, Set[int]]


def is_consistent_with_assignment(
    var: Cell,
    val: int,
    assignment: Dict[Cell, int],
    neighbors: Dict[Cell, List[Cell]],
    metrics: CSPMetrics,
) -> bool:
    """Check whether `var = val` conflicts with any currently assigned neighbor in `assignment`."""
    for nb in neighbors[var]:
        if nb in assignment:
            metrics.constraint_checks += 1
            if assignment[nb] == val:
                return False
    return True


def select_unassigned_variable_bt(
    variables: List[Cell],
    assignment: Dict[Cell, int],
    neighbors: Dict[Cell, List[Cell]],
) -> Cell:
    """Select the next unassigned variable for pure BT.

    Orders unassigned variables by the number of already-assigned neighbors (degree to assigned set),
    breaking ties by row-major coordinate `(r, c)`. Unlike Forward Checking, pure BT does not maintain
    pruned future domains.
    """
    best_var = variables[0]
    best_assigned_peers = -1
    for var in variables:
        if var not in assignment:
            assigned_peers = sum(1 for nb in neighbors[var] if nb in assignment)
            if assigned_peers > best_assigned_peers:
                best_assigned_peers = assigned_peers
                best_var = var
    return best_var


def solve_backtracking_only(
    variables: List[Cell],
    assignment: Dict[Cell, int],
    domains: Domains,
    neighbors: Dict[Cell, List[Cell]],
    metrics: CSPMetrics,
    count_limit: int = 1,
    max_backtracks: int = 250000,
) -> List[Dict[Cell, int]]:
    """Solve Sudoku CSP using pure Backtracking (BT) without Forward Checking or AC-3."""
    solutions: List[Dict[Cell, int]] = []

    def recursive_bt() -> None:
        if len(solutions) >= count_limit or metrics.backtracks >= max_backtracks:
            return
        if len(assignment) == len(variables):
            solutions.append(assignment.copy())
            return

        var = select_unassigned_variable_bt(variables, assignment, neighbors)
        for val in sorted(domains[var]):
            if is_consistent_with_assignment(var, val, assignment, neighbors, metrics):
                assignment[var] = val
                metrics.assignments += 1
                recursive_bt()
                if len(solutions) >= count_limit or metrics.backtracks >= max_backtracks:
                    return
                del assignment[var]
                metrics.backtracks += 1

    recursive_bt()
    return solutions
