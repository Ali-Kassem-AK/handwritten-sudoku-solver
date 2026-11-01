"""Arc Consistency (AC-3) implementation for Sudoku CSP.

Follows `CSP Lecture.pdf` (Dan Klein & Pieter Abbeel, UC Berkeley CS188, slide 39)
and `CSP Lab.pdf` (slides 15-21):

```
function AC-3(csp) returns the CSP, possibly with reduced domains
  local variables: queue, a queue of arcs, initially all the arcs in csp
  while queue is not empty do
    (X_i, X_j) <- REMOVE-FIRST(queue)
    if REMOVE-INCONSISTENT-VALUES(X_i, X_j) then
      if DOMAIN[X_i] is empty then return failure
      for each X_k in NEIGHBORS[X_i] - {X_j} do
        add (X_k, X_i) to queue
```

Also satisfies Section 4.C requirement:
"The AC-3 implementation must correctly identify conflicts between equal fixed values
in the same row, column, or box by detecting empty domains or equivalent contradictions."
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Dict, Iterable, List, Optional, Set, Tuple

from src.csp.metrics import CSPMetrics

Cell = Tuple[int, int]
Domains = Dict[Cell, Set[int]]


def remove_inconsistent_values(
    xi: Cell,
    xj: Cell,
    domains: Domains,
    metrics: CSPMetrics,
) -> bool:
    """Enforce consistency of directed arc `xi -> xj` (`X_i != X_j`).

    In `CSP Lecture.pdf` (slide 32 & 39):
    "An arc X -> Y is consistent iff for every x in the tail (X) there is some y in the head (Y)
    which could be assigned without violating a constraint."
    For binary inequality `X_i != X_j`:
    For each value `x` in `DOMAIN[X_i]`, we check whether there exists at least one `y` in `DOMAIN[X_j]`
    such that `x != y`. If no such `y` exists (i.e. `DOMAIN[X_j] == {x}` or `DOMAIN[X_j]` is empty),
    `x` is deleted from `DOMAIN[X_i]`, `metrics.domain_reductions` is incremented, and True is returned.
    """
    removed = False
    dom_i = domains[xi]
    dom_j = domains[xj]

    to_delete: List[int] = []
    for x in dom_i:
        has_support = False
        for y in dom_j:
            metrics.constraint_checks += 1
            if x != y:
                has_support = True
                break
        if not has_support:
            to_delete.append(x)

    if to_delete:
        for x in to_delete:
            dom_i.remove(x)
            metrics.domain_reductions += 1
        removed = True

    return removed


def run_ac3(
    domains: Domains,
    neighbors: Dict[Cell, List[Cell]],
    metrics: CSPMetrics,
    initial_arcs: Optional[Iterable[Tuple[Cell, Cell]]] = None,
) -> Tuple[bool, Optional[Cell]]:
    """Run the AC-3 algorithm on `domains` in-place.

    Args:
        domains: Mapping from each cell `(r, c)` to its current set of allowed digits `{1..9}`.
        neighbors: Precomputed map of binary constraint peers for each cell (20 peers per cell).
        metrics: Counter object updated in-place (`constraint_checks`, `domain_reductions`).
        initial_arcs: Optional subset of directed arcs `(X_k, X_i)` to seed the queue (e.g. during MAC).
            If None, initializes the queue with all 1,620 directed arcs in the Sudoku constraint graph.

    Returns:
        `(is_consistent, emptied_cell)`:
        - `(True, None)` if all arcs are made consistent and no domain becomes empty.
        - `(False, cell)` as soon as any cell's domain becomes empty (detecting contradiction,
          including two equal fixed givens in the same row, column, or 3x3 box).
    """
    # Check if any domain is already empty
    for cell, dom in domains.items():
        if len(dom) == 0:
            return False, cell

    if initial_arcs is None:
        queue: Deque[Tuple[Cell, Cell]] = deque(
            (xi, xj) for xi, peers in neighbors.items() for xj in peers
        )
    else:
        queue = deque(initial_arcs)

    in_queue: Set[Tuple[Cell, Cell]] = set(queue)

    while queue:
        xi, xj = queue.popleft()
        in_queue.discard((xi, xj))

        if remove_inconsistent_values(xi, xj, domains, metrics):
            if len(domains[xi]) == 0:
                return False, xi
            for xk in neighbors[xi]:
                if xk != xj and (xk, xi) not in in_queue:
                    queue.append((xk, xi))
                    in_queue.add((xk, xi))

    return True, None
