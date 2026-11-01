"""Greedy Best-First Search implementation for Sudoku digit correction.

Follows `Searching Lecture.pdf` (Dan Klein & Pieter Abbeel, UC Berkeley CS188, slides 65-69, 72)
and `Searching Lab.pdf` (slides 92-99):
- Orders nodes on the priority queue fringe by the heuristic estimate `h(n)` (forward cost),
  breaking ties by fewer remaining conflicts and lower step cost.
- Expands the node that appears closest to a conflict-free goal state without considering
  backward path cost `g(n)` in its primary priority ordering.
- Uses a `closed` set (`Graph Search`) to avoid re-expanding repeated states.
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


def greedy_best_first_search(
    problem: SudokuCorrectionProblem,
    max_expansions: int = 5000,
    record_trace: bool = True,
) -> SearchResult:
    """Run Greedy Best-First Search on `problem` ordered by heuristic `h(n)`."""
    metrics = SearchMetrics()
    t0 = time.perf_counter()

    init_state: State = problem.get_initial_state()
    init_h = problem.heuristic(init_state)
    init_n_conf = len(find_duplicate_conflicts(problem.initial_grid))
    metrics.nodes_generated = 1

    # Priority queue entries: (h_cost, num_conflicts, tie_counter, g_cost, state, steps_list)
    counter = 0
    fringe: List[Tuple[float, int, int, float, State, List[CorrectionStep]]] = [
        (init_h, init_n_conf, counter, 0.0, init_state, [])
    ]
    closed: Set[State] = set()
    seen_states: Set[State] = {init_state}
    trace: List[SearchTraceEntry] = []

    while fringe and metrics.nodes_expanded < max_expansions:
        h_cost, n_conf, _, g_cost, state, steps = heapq.heappop(fringe)

        if state in closed:
            continue
        closed.add(state)
        metrics.nodes_expanded += 1

        is_goal, solved_grid = problem.is_goal(state, metrics)

        if record_trace and len(trace) < 50:
            trace.append(
                SearchTraceEntry(
                    step_index=metrics.nodes_expanded,
                    corrections=[asdict(s) for s in steps],
                    g_cost=round(g_cost, 6),
                    h_cost=round(h_cost, 6),
                    f_cost=round(h_cost, 6),
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
                algorithm="Greedy",
                solution_found=True,
                corrected_grid=problem.apply_state(state),
                solved_grid=solved_grid,
                corrections=steps,
                metrics=metrics,
                trace=trace,
            )

        for next_state, step, step_cost in problem.get_successors(state):
            if next_state in closed or next_state in seen_states:
                continue
            next_h = problem.heuristic(next_state)
            if math.isinf(next_h):
                continue
            next_g = g_cost + step_cost
            next_grid = problem.apply_state(next_state)
            next_conf = len(find_duplicate_conflicts(next_grid))
            seen_states.add(next_state)
            counter += 1
            metrics.nodes_generated += 1
            heapq.heappush(
                fringe,
                (next_h, next_conf, counter, next_g, next_state, steps + [step]),
            )

    metrics.execution_time = time.perf_counter() - t0
    reason = (
        f"Correction budget ({problem.max_corrections}) or search space exhausted "
        f"after expanding {metrics.nodes_expanded} nodes without finding a solvable puzzle."
    )
    return SearchResult(
        algorithm="Greedy",
        solution_found=False,
        corrected_grid=None,
        solved_grid=None,
        corrections=[],
        metrics=metrics,
        trace=trace,
        failure_reason=reason,
    )
