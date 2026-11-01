"""Uniform Cost Search (UCS) implementation for Sudoku digit correction.

Follows `Searching Lecture.pdf` (Dan Klein & Pieter Abbeel, UC Berkeley CS188, slides 43-46, 72-73, 107, 119)
and `Searching Lab.pdf` (slides 75-89):
- Expands the node on the priority queue fringe with the lowest cumulative path cost `g(n)`.
- Performs goal test on dequeue (expansion), NOT on enqueue (`Searching Lecture.pdf`, slide 73),
  guaranteeing optimality (`C*`).
- Uses a `closed` set of expanded states (`Graph Search`, slide 107 & 119) so repeated states
  reached via different correction orderings are never expanded twice.
"""

from __future__ import annotations

from dataclasses import asdict
import heapq
import math
import time
from typing import Dict, List, Set, Tuple

from src.csp.sudoku import find_duplicate_conflicts
from src.search.metrics import (
    CorrectionStep,
    SearchMetrics,
    SearchResult,
    SearchTraceEntry,
)
from src.search.problem import State, SudokuCorrectionProblem


def uniform_cost_search(
    problem: SudokuCorrectionProblem,
    max_expansions: int = 5000,
    record_trace: bool = True,
) -> SearchResult:
    """Run Uniform Cost Search (UCS) on `problem` to find the minimum-cost valid correction set."""
    metrics = SearchMetrics()
    t0 = time.perf_counter()

    init_state: State = problem.get_initial_state()
    metrics.nodes_generated = 1

    # Priority queue entries: (g_cost, num_corrections, tie_counter, state, steps_list)
    counter = 0
    fringe: List[Tuple[float, int, int, State, List[CorrectionStep]]] = [
        (0.0, 0, counter, init_state, [])
    ]
    closed: Set[State] = set()
    best_g: Dict[State, float] = {init_state: 0.0}
    trace: List[SearchTraceEntry] = []

    while fringe and metrics.nodes_expanded < max_expansions:
        g_cost, num_corr, _, state, steps = heapq.heappop(fringe)

        # Graph search check: never expand a state twice
        if state in closed:
            continue
        closed.add(state)
        metrics.nodes_expanded += 1

        # Goal test upon dequeue (expansion)
        is_goal, solved_grid = problem.is_goal(state, metrics)

        if record_trace and len(trace) < 50:
            cur_grid = problem.apply_state(state)
            n_conf = len(find_duplicate_conflicts(cur_grid))
            trace.append(
                SearchTraceEntry(
                    step_index=metrics.nodes_expanded,
                    corrections=[asdict(s) for s in steps],
                    g_cost=round(g_cost, 6),
                    h_cost=0.0,
                    f_cost=round(g_cost, 6),
                    num_conflicts=n_conf,
                    goal_passed=is_goal,
                )
            )

        if is_goal and solved_grid is not None:
            metrics.num_corrections = len(steps)
            metrics.final_correction_cost = g_cost
            metrics.solution_found = True
            metrics.execution_time = time.perf_counter() - t0
            return SearchResult(
                algorithm="UCS",
                solution_found=True,
                corrected_grid=problem.apply_state(state),
                solved_grid=solved_grid,
                corrections=steps,
                metrics=metrics,
                trace=trace,
            )

        # Expand node and enqueue successors
        for next_state, step, step_cost in problem.get_successors(state):
            if next_state in closed:
                continue
            next_g = g_cost + step_cost
            if math.isinf(next_g):
                continue
            if next_state not in best_g or next_g < best_g[next_state] - 1e-12:
                best_g[next_state] = next_g
                counter += 1
                metrics.nodes_generated += 1
                heapq.heappush(
                    fringe,
                    (next_g, len(steps) + 1, counter, next_state, steps + [step]),
                )

    metrics.execution_time = time.perf_counter() - t0
    reason = (
        f"Correction budget ({problem.max_corrections}) or search space exhausted "
        f"after expanding {metrics.nodes_expanded} nodes without finding a solvable puzzle."
    )
    return SearchResult(
        algorithm="UCS",
        solution_found=False,
        corrected_grid=None,
        solved_grid=None,
        corrections=[],
        metrics=metrics,
        trace=trace,
        failure_reason=reason,
    )
