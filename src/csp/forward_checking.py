"""Forward Checking (FC) implementation for Sudoku CSP.

Follows `CSP Lecture.pdf` (slides 29-32) and `CSP Lab.pdf` (slides 10-14, 33-47):
- "Filtering: Keep track of domains for unassigned variables and cross off bad options."
- "Forward checking: Cross off values that violate a constraint when added to the existing assignment."
- "Enforcing consistency of arcs pointing to each new assignment."
- Uses Minimum Remaining Values (MRV, slide 47) variable ordering to select the unassigned variable
  with the fewest legal values remaining in its domain (breaking ties by row-major order), and
  Least Constraining Value (LCV, slide 48) value ordering.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.csp.metrics import CSPMetrics

Cell = Tuple[int, int]
Domains = Dict[Cell, Set[int]]


def forward_check(
    var: Cell,
    val: int,
    assignment: Dict[Cell, int],
    domains: Domains,
    neighbors: Dict[Cell, List[Cell]],
    metrics: CSPMetrics,
) -> Tuple[bool, List[Tuple[Cell, int]]]:
    """Perform Forward Checking after assigning `var = val`.

    For every unassigned neighbor `nb` of `var`, checks every candidate value `d` in `domains[nb]`
    against `val` (`d != val`). If `val` is present in `domains[nb]`, removes `val` from `domains[nb]`,
    increments `metrics.domain_reductions`, and records `(nb, val)` so it can be restored upon backtrack.
    If any neighbor's domain becomes empty, returns `(False, pruned_list)` immediately.
    """
    pruned: List[Tuple[Cell, int]] = []
    for nb in neighbors[var]:
        if nb in assignment:
            continue
        dom_nb = domains[nb]
        metrics.constraint_checks += 1
        if val in dom_nb:
            dom_nb.remove(val)
            metrics.domain_reductions += 1
            pruned.append((nb, val))
            if len(dom_nb) == 0:
                return False, pruned
    return True, pruned


def restore_pruned(domains: Domains, pruned: List[Tuple[Cell, int]]) -> None:
    """Restore values pruned from domains during a failed branch."""
    for cell, val in pruned:
        domains[cell].add(val)


def select_unassigned_variable_mrv(
    variables: List[Cell],
    assignment: Dict[Cell, int],
    domains: Domains,
    neighbors: Dict[Cell, List[Cell]],
) -> Cell:
    """Select the next unassigned variable using Minimum Remaining Values (MRV) heuristic.

    As taught in `CSP Lecture.pdf` (slide 47) and `CSP Lab.pdf` (slide 22 & 45):
    Chooses the unassigned variable with the smallest domain size `|DOMAIN[X_i]|`,
    breaking ties deterministically by row-major coordinate `(r, c)`.
    """
    best_var: Optional[Cell] = None
    best_size = 10
    for var in variables:
        if var not in assignment:
            sz = len(domains[var])
            if sz < best_size:
                best_size = sz
                best_var = var
                if sz <= 1:
                    break
    assert best_var is not None
    return best_var


def order_domain_values_lcv(
    var: Cell,
    assignment: Dict[Cell, int],
    domains: Domains,
    neighbors: Dict[Cell, List[Cell]],
) -> List[int]:
    """Order domain values using Least Constraining Value (LCV) heuristic (`CSP Lecture.pdf`, slide 48).

    Orders values `v` in `domains[var]` by how few times `v` appears in the domains of
    unassigned neighbors of `var` (ties broken by ascending numerical value 1..9).
    """
    dom = list(domains[var])
    if len(dom) <= 1:
        return dom

    def conflict_score(val: int) -> Tuple[int, int]:
        ruled_out = 0
        for nb in neighbors[var]:
            if nb not in assignment and val in domains[nb]:
                ruled_out += 1
        return (ruled_out, val)

    return sorted(dom, key=conflict_score)


def solve_backtracking_fc(
    variables: List[Cell],
    assignment: Dict[Cell, int],
    domains: Domains,
    neighbors: Dict[Cell, List[Cell]],
    metrics: CSPMetrics,
    count_limit: int = 1,
) -> List[Dict[Cell, int]]:
    """Solve Sudoku CSP using Backtracking + Forward Checking (BT + FC) without AC-3."""
    solutions: List[Dict[Cell, int]] = []

    # Initial unary/given forward checking propagation from fixed givens
    for fixed_var, fixed_val in list(assignment.items()):
        ok, _ = forward_check(fixed_var, fixed_val, assignment, domains, neighbors, metrics)
        if not ok:
            return solutions

    def recursive_bt_fc() -> None:
        if len(solutions) >= count_limit:
            return
        if len(assignment) == len(variables):
            solutions.append(assignment.copy())
            return

        var = select_unassigned_variable_mrv(variables, assignment, domains, neighbors)
        if len(domains[var]) == 0:
            return

        for val in order_domain_values_lcv(var, assignment, domains, neighbors):
            # Verify consistency with already assigned neighbors
            consistent = True
            for nb in neighbors[var]:
                if nb in assignment:
                    metrics.constraint_checks += 1
                    if assignment[nb] == val:
                        consistent = False
                        break
            if not consistent:
                continue

            assignment[var] = val
            metrics.assignments += 1
            saved_dom = domains[var].copy()
            domains[var] = {val}

            ok, pruned = forward_check(var, val, assignment, domains, neighbors, metrics)
            if ok:
                recursive_bt_fc()
                if len(solutions) >= count_limit:
                    return

            restore_pruned(domains, pruned)
            domains[var] = saved_dom
            del assignment[var]
            metrics.backtracks += 1

    recursive_bt_fc()
    return solutions
