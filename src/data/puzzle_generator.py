"""Sudoku puzzle generator with randomized backtracking and uniqueness checking.

Implements Section 5 requirements:
1. Generates a complete valid 9x9 Sudoku solution using randomized backtracking.
2. Removes cells while preserving a unique solution for normal generated puzzles.
3. Supports configurable numbers of givens (e.g., 25, 30, 35, or custom).
4. Accepts a reproducible random seed.
5. Retains the original completed solution as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class GeneratedPuzzle:
    """Represents a generated Sudoku puzzle and its ground-truth completed solution."""

    puzzle: np.ndarray  # shape (9, 9), int64, 0 for blank cells, 1..9 for givens
    solution: np.ndarray  # shape (9, 9), int64, 1..9 complete valid Sudoku grid
    num_givens: int
    seed: int
    unique_solution: bool


def _get_peer_mask(r: int, c: int, grid: np.ndarray) -> int:
    """Return a bitmask of digits (bits 1..9) already used in row r, col c, or 3x3 box."""
    mask = 0
    for j in range(9):
        val = int(grid[r, j])
        if val != 0:
            mask |= 1 << val
    for i in range(9):
        val = int(grid[i, c])
        if val != 0:
            mask |= 1 << val
    br, bc = (r // 3) * 3, (c // 3) * 3
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            val = int(grid[i, j])
            if val != 0:
                mask |= 1 << val
    return mask


def is_valid_completed_sudoku(grid: np.ndarray) -> bool:
    """Verify that a 9x9 grid is completely filled with 1..9 and satisfies all Sudoku rules."""
    if grid.shape != (9, 9):
        return False
    expected = set(range(1, 10))
    for i in range(9):
        if set(int(x) for x in grid[i, :]) != expected:
            return False
        if set(int(x) for x in grid[:, i]) != expected:
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = grid[br : br + 3, bc : bc + 3].flatten()
            if set(int(x) for x in box) != expected:
                return False
    return True


def count_sudoku_solutions_fast(grid: np.ndarray, limit: int = 2) -> int:
    """Fast bitwise MRV backtracking solver to count solutions up to `limit`.

    Used during puzzle generation to verify uniqueness (`count == 1`) when digging holes.
    The main CSP module (`src/csp/`) implements full instrumented BT, BT+FC, and BT+FC+AC3
    for experimental evaluation and pipeline solving.
    """
    work = grid.copy().astype(np.int64)
    row_mask = [0] * 9
    col_mask = [0] * 9
    box_mask = [0] * 9

    empty_cells: List[Tuple[int, int]] = []
    for r in range(9):
        for c in range(9):
            val = int(work[r, c])
            if val != 0:
                if not (1 <= val <= 9):
                    return 0
                bit = 1 << val
                b_idx = (r // 3) * 3 + (c // 3)
                if (row_mask[r] & bit) or (col_mask[c] & bit) or (box_mask[b_idx] & bit):
                    return 0
                row_mask[r] |= bit
                col_mask[c] |= bit
                box_mask[b_idx] |= bit
            else:
                empty_cells.append((r, c))

    n_empty = len(empty_cells)
    assigned = [False] * n_empty
    solutions = 0

    def dfs(remaining: int) -> None:
        nonlocal solutions
        if solutions >= limit:
            return
        if remaining == 0:
            solutions += 1
            return

        # Select unassigned cell with Minimum Remaining Values (MRV)
        best_idx = -1
        best_count = 10
        best_avail = 0

        for idx in range(n_empty):
            if assigned[idx]:
                continue
            r, c = empty_cells[idx]
            b_idx = (r // 3) * 3 + (c // 3)
            used = row_mask[r] | col_mask[c] | box_mask[b_idx]
            avail = (~used) & 0x3FE  # bits 1..9
            if avail == 0:
                return
            cnt = avail.bit_count()
            if cnt < best_count:
                best_count = cnt
                best_idx = idx
                best_avail = avail
                if cnt == 1:
                    break

        assigned[best_idx] = True
        r, c = empty_cells[best_idx]
        b_idx = (r // 3) * 3 + (c // 3)

        avail = best_avail
        while avail and solutions < limit:
            bit = avail & -avail
            avail ^= bit
            row_mask[r] |= bit
            col_mask[c] |= bit
            box_mask[b_idx] |= bit
            dfs(remaining - 1)
            row_mask[r] ^= bit
            col_mask[c] ^= bit
            box_mask[b_idx] ^= bit

        assigned[best_idx] = False

    dfs(n_empty)
    return solutions


def generate_complete_grid(rng: np.random.Generator) -> np.ndarray:
    """Generate a valid completed 9x9 Sudoku grid using randomized backtracking."""
    grid = np.zeros((9, 9), dtype=np.int64)

    def fill_cell(pos: int) -> bool:
        if pos == 81:
            return True
        r, c = divmod(pos, 9)
        used = _get_peer_mask(r, c, grid)
        candidates = [d for d in range(1, 10) if not (used & (1 << d))]
        if not candidates:
            return False
        rng.shuffle(candidates)
        for d in candidates:
            grid[r, c] = d
            if fill_cell(pos + 1):
                return True
            grid[r, c] = 0
        return False

    fill_cell(0)
    return grid


def generate_sudoku_puzzle(
    num_givens: int = 35,
    seed: int = 42,
    ensure_unique: bool = True,
    max_restarts: int = 10,
) -> GeneratedPuzzle:
    """Generate a Sudoku puzzle with `num_givens` non-blank cells and a unique solution.

    Args:
        num_givens: Target number of givens (must be between 17 and 81).
        seed: Reproducible random seed.
        ensure_unique: If True, only removes a cell if the puzzle retains a unique solution.
        max_restarts: Number of random restarts if a low-given target (e.g. 25) gets stuck.
    """
    if not (17 <= num_givens <= 81):
        raise ValueError(f"num_givens must be between 17 and 81, got {num_givens}")

    master_rng = np.random.default_rng(seed)

    best_puzzle: Optional[np.ndarray] = None
    best_solution: Optional[np.ndarray] = None
    best_diff = 82

    for attempt in range(max_restarts):
        sub_seed = int(master_rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng(sub_seed)
        solution = generate_complete_grid(rng)
        puzzle = solution.copy()

        positions = [(r, c) for r in range(9) for c in range(9)]
        rng.shuffle(positions)

        current_givens = 81
        for r, c in positions:
            if current_givens <= num_givens:
                break
            backup = int(puzzle[r, c])
            puzzle[r, c] = 0
            if ensure_unique:
                if count_sudoku_solutions_fast(puzzle, limit=2) == 1:
                    current_givens -= 1
                else:
                    puzzle[r, c] = backup
            else:
                current_givens -= 1

        diff = abs(current_givens - num_givens)
        if diff < best_diff:
            best_diff = diff
            best_puzzle = puzzle.copy()
            best_solution = solution.copy()
        if current_givens == num_givens:
            break

    assert best_puzzle is not None and best_solution is not None
    actual_givens = int(np.sum(best_puzzle != 0))
    unique = count_sudoku_solutions_fast(best_puzzle, limit=2) == 1

    return GeneratedPuzzle(
        puzzle=best_puzzle,
        solution=best_solution,
        num_givens=actual_givens,
        seed=seed,
        unique_solution=unique,
    )
