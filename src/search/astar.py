"""A* Search implementation for Sudoku digit correction.

Follows `Searching Lecture.pdf` (Dan Klein & Pieter Abbeel, UC Berkeley CS188, slides 70-87, 107-113, 119)
and `Searching Lab.pdf` (slides 100-113):
- Orders nodes on the priority queue fringe by `f(n) = g(n) + h(n)`, combining backward path cost
  `g(n)` (cumulative probability penalty) and admissible forward cost estimate `h(n)`.
- Terminates only when a goal state is dequeued from the fringe (`Searching Lecture.pdf`, slide 73).
- Maintains a `closed` set (`Graph Search`, slide 107 & 119) so repeated states are never expanded twice.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import heapq
import math
import time
from typing import Dict, List, Set, Tuple

import numpy as np

from src.csp.sudoku import find_duplicate_conflicts
from src.search.metrics import (
    CorrectionStep,
    SearchMetrics,
    SearchResult,
    SearchTraceEntry,
)
from src.search.problem import State, SudokuCorrectionProblem


def astar_search(
    problem: SudokuCorrectionProblem,
    max_expansions: int = 5000,
    record_trace: bool = True,
) -> SearchResult:
    """Run A* Search (`f(n) = g(n) + h(n)`) on `problem` to find the optimal valid correction set."""
    metrics = SearchMetrics()
    t0 = time.perf_counter()

    init_state: State = problem.get_initial_state()
    init_h = problem.heuristic(init_state)
    metrics.nodes_generated = 1

    # Priority queue entries: (f_cost, h_cost, num_corrections, tie_counter, g_cost, state, steps_list)
    counter = 0
    fringe: List[Tuple[float, float, int, int, float, State, List[CorrectionStep]]] = [
        (init_h, init_h, 0, counter, 0.0, init_state, [])
    ]
    closed: Set[State] = set()
    best_g: Dict[State, float] = {init_state: 0.0}
    trace: List[SearchTraceEntry] = []

    while fringe and metrics.nodes_expanded < max_expansions:
        f_cost, h_cost, num_corr, _, g_cost, state, steps = heapq.heappop(fringe)

        if state in closed:
            continue
        closed.add(state)
        metrics.nodes_expanded += 1

        is_goal, solved_grid = problem.is_goal(state, metrics)

        if record_trace and len(trace) < 50:
            cur_grid = problem.apply_state(state)
            n_conf = len(find_duplicate_conflicts(cur_grid))
            trace.append(
                SearchTraceEntry(
                    step_index=metrics.nodes_expanded,
                    corrections=[asdict(s) for s in steps],
                    g_cost=round(g_cost, 6),
                    h_cost=round(h_cost, 6),
                    f_cost=round(f_cost, 6),
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
                algorithm="A*",
                solution_found=True,
                corrected_grid=problem.apply_state(state),
                solved_grid=solved_grid,
                corrections=steps,
                metrics=metrics,
                trace=trace,
            )

        for next_state, step, step_cost in problem.get_successors(state):
            if next_state in closed:
                continue
            next_g = g_cost + step_cost
            if math.isinf(next_g):
                continue
            if next_state not in best_g or next_g < best_g[next_state] - 1e-12:
                next_h = problem.heuristic(next_state)
                if math.isinf(next_h):
                    continue
                best_g[next_state] = next_g
                next_f = next_g + next_h
                counter += 1
                metrics.nodes_generated += 1
                heapq.heappush(
                    fringe,
                    (
                        next_f,
                        next_h,
                        len(steps) + 1,
                        counter,
                        next_g,
                        next_state,
                        steps + [step],
                    ),
                )

    metrics.execution_time = time.perf_counter() - t0
    reason = (
        f"Correction budget ({problem.max_corrections}) or search space exhausted "
        f"after expanding {metrics.nodes_expanded} nodes without finding a solvable puzzle."
    )
    return SearchResult(
        algorithm="A*",
        solution_found=False,
        corrected_grid=None,
        solved_grid=None,
        corrections=[],
        metrics=metrics,
        trace=trace,
        failure_reason=reason,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UCS, Greedy, or A* search on a sample misread Sudoku puzzle.")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="A*",
        choices=["UCS", "Greedy", "A*"],
        help="Search algorithm to execute.",
    )
    parser.add_argument("--misreads", type=int, default=2, help="Number of injected misreads.")
    parser.add_argument("--budget", type=int, default=3, help="Maximum correction budget.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    from src.data.mosaic_generator import inject_hard_digit_errors
    from src.data.puzzle_generator import generate_sudoku_puzzle
    from src.search.greedy import greedy_best_first_search
    from src.search.ucs import uniform_cost_search

    puz = generate_sudoku_puzzle(num_givens=35, seed=args.seed)
    # Create high-confidence baseline probabilities for true givens, then inject errors
    prob_grid = np.full((9, 9, 9), 0.0025, dtype=np.float64)
    for r in range(9):
        for c in range(9):
            d = int(puz.puzzle[r, c])
            if d != 0:
                prob_grid[r, c, d - 1] = 0.98
                prob_grid[r, c] /= np.sum(prob_grid[r, c])

    pred_grid, mod_probs, injected = inject_hard_digit_errors(
        puz.puzzle, prob_grid, puz.puzzle, puz.solution, num_errors=args.misreads, seed=args.seed
    )

    problem = SudokuCorrectionProblem(
        initial_grid=pred_grid,
        prob_grid=mod_probs,
        max_corrections=args.budget,
    )

    if args.algorithm == "UCS":
        res = uniform_cost_search(problem)
    elif args.algorithm == "Greedy":
        res = greedy_best_first_search(problem)
    else:
        res = astar_search(problem)

    print(f"=== Search Algorithm ({args.algorithm}) on {args.misreads}-Misread Puzzle (Budget={args.budget}) ===")
    print(f"Solution Found:   {res.solution_found}")
    print(f"Nodes Generated:  {res.metrics.nodes_generated}")
    print(f"Nodes Expanded:   {res.metrics.nodes_expanded}")
    print(f"Goal Checks:      {res.metrics.goal_checks}")
    print(f"Corrections Made: {res.metrics.num_corrections}")
    print(f"Correction Cost:  {res.metrics.final_correction_cost:.4f}")
    print(f"Execution Time:   {res.metrics.execution_time*1000:.2f} ms")
    for c in res.corrections:
        print(
            f"  -> Cell ({c.row},{c.col}): {c.old_digit} (p={c.old_prob:.3f}) "
            f"=> {c.new_digit} (p={c.new_prob:.3f}), cost={c.step_cost:.4f}"
        )


if __name__ == "__main__":
    main()
