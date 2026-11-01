"""End-to-end integration pipeline:
MNIST -> Neural Network -> Sudoku Reading -> AC-3 Conflict Detection -> Search-Based Correction -> CSP Validation -> Final Sudoku Solution.

Follows Section 6 required workflow:
1. Generate or load a Sudoku puzzle with its ground-truth solution.
2. Create a 9x9 image mosaic using MNIST handwriting.
3. Use the neural network to recognize the nonblank cell images (or bypass when `use_nn=False`).
4. Store the predicted digits and probabilities.
5. Detect duplicate-digit conflicts and other Sudoku inconsistencies.
6. Run AC-3 to help identify contradictions and restrict domains.
7. Generate candidate corrections for uncertain cells.
8. Run the selected search algorithm (`UCS`, `Greedy`, or `A*`).
9. For each candidate correction set, call the CSP solver (`BT`, `BT+FC`, or `BT+FC+AC3`).
10. Accept the first correction set that satisfies the defined goal.
11. Return the corrected puzzle and its completed Sudoku solution.
12. Produce a structured correction log (or an `Unreadable` diagnostic report on failure).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.csp.metrics import CSPMetrics, CSPResult
from src.csp.sudoku import SudokuCSPSolver
from src.data.mosaic_generator import (
    InjectedErrorRecord,
    SudokuMosaic,
    create_sudoku_mosaic,
    inject_hard_digit_errors,
)
from src.data.puzzle_generator import GeneratedPuzzle, generate_sudoku_puzzle
from src.integration.correction import (
    CorrectionLogEntry,
    UnreadableReport,
    build_correction_log,
    build_unreadable_report,
    run_correction_strategy,
)
from src.nn.model import DigitMLP
from src.nn.predict import GridRecognitionResult, recognize_sudoku_mosaic
from src.search.metrics import SearchResult


@dataclass
class PipelineResult:
    """Comprehensive output of the end-to-end Handwritten Sudoku Reader and Solver pipeline."""

    status: str  # "SOLVED" or "UNREADABLE"
    nn_enabled: bool
    search_algorithm: str
    csp_mode: str
    num_givens: int
    num_injected_errors: int
    max_corrections_budget: int
    puzzle_seed: int
    # Grids
    ground_truth_puzzle: np.ndarray
    ground_truth_solution: np.ndarray
    initial_predicted_grid: np.ndarray
    initial_probability_grid: np.ndarray
    initial_confidence_grid: np.ndarray
    corrected_puzzle_grid: Optional[np.ndarray]
    final_solved_grid: Optional[np.ndarray]
    # Accuracy & recovery checks
    initial_cell_accuracy: float
    exact_puzzle_recovered: bool
    exact_solution_recovered: bool
    puzzle_solvable: bool
    # Conflict & AC-3 diagnostics on initial reading
    initial_conflicts: List[Tuple[Tuple[int, int], Tuple[int, int], int]]
    ac3_initial_consistent: bool
    ac3_initial_metrics: CSPMetrics
    ac3_singleton_domains_count: int
    ac3_emptied_domains_count: int
    # Search & CSP results
    injected_errors: List[InjectedErrorRecord]
    search_result: SearchResult
    final_csp_result: Optional[CSPResult]
    correction_log: List[CorrectionLogEntry]
    unreadable_report: Optional[UnreadableReport] = None
    total_pipeline_time_sec: float = 0.0
    mosaic: Optional[SudokuMosaic] = field(default=None, repr=False)

    def to_summary_dict(self) -> Dict[str, object]:
        """Serialize pipeline result to a clean JSON-compatible dictionary."""
        return {
            "status": self.status,
            "nn_enabled": self.nn_enabled,
            "search_algorithm": self.search_algorithm,
            "csp_mode": self.csp_mode,
            "num_givens": self.num_givens,
            "num_injected_errors": self.num_injected_errors,
            "max_corrections_budget": self.max_corrections_budget,
            "puzzle_seed": self.puzzle_seed,
            "initial_cell_accuracy": round(float(self.initial_cell_accuracy), 6),
            "exact_puzzle_recovered": bool(self.exact_puzzle_recovered),
            "exact_solution_recovered": bool(self.exact_solution_recovered),
            "puzzle_solvable": bool(self.puzzle_solvable),
            "initial_conflicts": [
                {"cell_a": list(ca), "cell_b": list(cb), "digit": int(d)}
                for ca, cb, d in self.initial_conflicts
            ],
            "ac3_initial_consistent": bool(self.ac3_initial_consistent),
            "ac3_initial_metrics": self.ac3_initial_metrics.to_dict(),
            "ac3_singleton_domains_count": int(self.ac3_singleton_domains_count),
            "ac3_emptied_domains_count": int(self.ac3_emptied_domains_count),
            "injected_errors": [asdict(e) for e in self.injected_errors],
            "search_metrics": self.search_result.metrics.to_dict(),
            "final_csp_metrics": self.final_csp_result.metrics.to_dict() if self.final_csp_result else None,
            "correction_log": [entry.to_table_dict() for entry in self.correction_log],
            "unreadable_report": asdict(self.unreadable_report) if self.unreadable_report else None,
            "total_pipeline_time_sec": round(float(self.total_pipeline_time_sec), 6),
        }


class HandwrittenSudokuPipeline:
    """End-to-end orchestrator connecting MNIST mosaic generation, DigitMLP, AC-3, Search, and CSP."""

    def __init__(
        self,
        model: DigitMLP,
        mnist_test_images: np.ndarray,
        mnist_test_digits: np.ndarray,
        brightness_threshold: float = 0.02,
    ) -> None:
        self.model = model
        self.mnist_test_images = mnist_test_images
        self.mnist_test_digits = mnist_test_digits
        self.brightness_threshold = brightness_threshold

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "HandwrittenSudokuPipeline":
        """Load trained DigitMLP and untouched MNIST test set from `project_root`."""
        root = Path(project_root)
        npz_path = root / "data" / "processed" / "mnist_sudoku_1_9.npz"
        mlp_path = root / "models" / "digit_mlp.npz"
        if not npz_path.exists():
            from src.data.make_data import run_make_data

            run_make_data(root)
        if not mlp_path.exists():
            from src.nn.train import run_training_pipeline

            run_training_pipeline(root)

        data = np.load(npz_path)
        model = DigitMLP.load(mlp_path)
        return cls(
            model=model,
            mnist_test_images=data["X_test"],
            mnist_test_digits=data["digits_test"],
        )

    def run(
        self,
        num_givens: int = 35,
        num_injected_errors: Optional[int] = 2,
        max_corrections: int = 3,
        top_k_alternatives: int = 3,
        search_algorithm: str = "A*",
        csp_mode: str = "BT+FC+AC3",
        use_nn: bool = True,
        require_unique: bool = False,
        seed: int = 42,
        pregenerated_puzzle: Optional[GeneratedPuzzle] = None,
    ) -> PipelineResult:
        """Execute the complete 12-step Handwritten Sudoku Reader and Solver workflow.

        Args:
            num_givens: Number of initial givens in the generated Sudoku puzzle.
            num_injected_errors: If an integer >= 0, uses controlled hard-digit injection
                to create exactly `num_injected_errors` misreads. If None, uses raw natural
                neural network predictions without synthetic injection.
            max_corrections: Correction budget `B` (default 3).
            top_k_alternatives: Number of top alternative digits per uncertain cell (default 3).
            search_algorithm: `"UCS"`, `"Greedy"`, or `"A*"`.
            csp_mode: `"BT"`, `"BT+FC"`, or `"BT+FC+AC3"`.
            use_nn: Whether to enable neural network recognition and probability guidance.
            require_unique: Optional Extension 1 — require unique Sudoku solution in goal test.
            seed: Reproducible random seed.
            pregenerated_puzzle: Optional pre-built `GeneratedPuzzle` to reuse across comparisons.
        """
        t_start = time.perf_counter()

        # Step 1: Generate or load Sudoku puzzle with ground-truth solution
        if pregenerated_puzzle is not None:
            gen_puz = pregenerated_puzzle
        else:
            gen_puz = generate_sudoku_puzzle(num_givens=num_givens, seed=seed, ensure_unique=True)

        # Step 2: Create 9x9 image mosaic using MNIST handwriting
        mosaic = create_sudoku_mosaic(
            puzzle_grid=gen_puz.puzzle,
            solution_grid=gen_puz.solution,
            mnist_images=self.mnist_test_images,
            mnist_digits=self.mnist_test_digits,
            seed=seed,
            brightness_threshold=self.brightness_threshold,
        )

        # Step 3 & 4: Use Neural Network to recognize nonblank cell images and store probabilities
        rec_res: GridRecognitionResult = recognize_sudoku_mosaic(
            cell_images=mosaic.cell_images,
            model=self.model,
            brightness_threshold=self.brightness_threshold,
            use_calibrated=True,
            use_nn=True,  # First get true NN predictions so controlled injection can be applied if requested
            seed=seed,
        )

        pred_grid = rec_res.predicted_grid.copy()
        prob_grid = rec_res.probability_grid.copy()
        injected_records: List[InjectedErrorRecord] = []

        if num_injected_errors is not None:
            pred_grid, prob_grid, injected_records = inject_hard_digit_errors(
                pred_grid=pred_grid,
                prob_grid=prob_grid,
                puzzle_grid=gen_puz.puzzle,
                solution_grid=gen_puz.solution,
                num_errors=num_injected_errors,
                seed=seed,
            )
            mosaic.injected_errors = injected_records

        if not use_nn:
            # When NN is switched off for demonstration/ablation, strip the NN's probability guidance
            # (replacing it with uninformative uniform probabilities over alternatives)
            rec_off = recognize_sudoku_mosaic(
                cell_images=mosaic.cell_images,
                model=self.model,
                brightness_threshold=self.brightness_threshold,
                use_calibrated=True,
                use_nn=False,
                fallback_grid=pred_grid,
                seed=seed,
            )
            pred_grid = rec_off.predicted_grid.copy()
            prob_grid = rec_off.probability_grid.copy()

        conf_grid = np.max(prob_grid, axis=2)
        nonblank_mask = gen_puz.puzzle != 0
        cell_acc = float(np.mean(pred_grid[nonblank_mask] == gen_puz.puzzle[nonblank_mask]))

        # Step 5 & 6: Detect duplicate-digit conflicts and run AC-3 to identify contradictions & restrict domains
        csp_solver = SudokuCSPSolver(mode=csp_mode)
        ac3_ok, ac3_domains, ac3_metrics, initial_conflicts = csp_solver.run_ac3_conflict_check(pred_grid)
        singleton_count = sum(1 for d in ac3_domains.values() if len(d) == 1)
        emptied_count = sum(1 for d in ac3_domains.values() if len(d) == 0)

        # Step 7, 8, 9, 10: Generate candidate corrections, run Search, and validate each candidate with CSP
        strategy = "nn_guided" if use_nn else "random_correction"
        search_res = run_correction_strategy(
            strategy=strategy,
            initial_grid=pred_grid,
            prob_grid=prob_grid,
            search_algorithm=search_algorithm,
            csp_mode="BT+FC",  # Fast exact CSP goal check inside search loop
            max_corrections=max_corrections,
            top_k_alternatives=top_k_alternatives,
            require_unique=require_unique,
            seed=seed,
        )

        # Step 11 & 12: Final CSP verification with the requested `csp_mode` and Correction Log / Failure Report
        if search_res.solution_found and search_res.corrected_grid is not None:
            final_csp_res = csp_solver.solve(
                search_res.corrected_grid,
                count_solutions=require_unique,
                count_limit=2 if require_unique else 1,
            )
            corr_log = build_correction_log(search_res.corrections, search_res.algorithm)
            exact_puz = bool(np.array_equal(search_res.corrected_grid, gen_puz.puzzle))
            exact_sol = bool(
                final_csp_res.solution is not None
                and np.array_equal(final_csp_res.solution, gen_puz.solution)
            )
            total_dt = time.perf_counter() - t_start
            return PipelineResult(
                status="SOLVED",
                nn_enabled=use_nn,
                search_algorithm=search_res.algorithm,
                csp_mode=csp_mode,
                num_givens=gen_puz.num_givens,
                num_injected_errors=len(injected_records),
                max_corrections_budget=max_corrections,
                puzzle_seed=seed,
                ground_truth_puzzle=gen_puz.puzzle,
                ground_truth_solution=gen_puz.solution,
                initial_predicted_grid=pred_grid,
                initial_probability_grid=prob_grid,
                initial_confidence_grid=conf_grid,
                corrected_puzzle_grid=search_res.corrected_grid,
                final_solved_grid=final_csp_res.solution,
                initial_cell_accuracy=cell_acc,
                exact_puzzle_recovered=exact_puz,
                exact_solution_recovered=exact_sol,
                puzzle_solvable=True,
                initial_conflicts=initial_conflicts,
                ac3_initial_consistent=ac3_ok,
                ac3_initial_metrics=ac3_metrics,
                ac3_singleton_domains_count=singleton_count,
                ac3_emptied_domains_count=emptied_count,
                injected_errors=injected_records,
                search_result=search_res,
                final_csp_result=final_csp_res,
                correction_log=corr_log,
                unreadable_report=None,
                total_pipeline_time_sec=total_dt,
                mosaic=mosaic,
            )

        # Failure Handling: Report "Unreadable" without inventing a solution
        unreadable_rep = build_unreadable_report(
            initial_grid=pred_grid,
            prob_grid=prob_grid,
            max_corrections=max_corrections,
            top_k_alternatives=top_k_alternatives,
            nodes_expanded=search_res.metrics.nodes_expanded,
            ac3_domains=ac3_domains,
        )
        total_dt = time.perf_counter() - t_start
        return PipelineResult(
            status="UNREADABLE",
            nn_enabled=use_nn,
            search_algorithm=search_res.algorithm,
            csp_mode=csp_mode,
            num_givens=gen_puz.num_givens,
            num_injected_errors=len(injected_records),
            max_corrections_budget=max_corrections,
            puzzle_seed=seed,
            ground_truth_puzzle=gen_puz.puzzle,
            ground_truth_solution=gen_puz.solution,
            initial_predicted_grid=pred_grid,
            initial_probability_grid=prob_grid,
            initial_confidence_grid=conf_grid,
            corrected_puzzle_grid=None,
            final_solved_grid=None,
            initial_cell_accuracy=cell_acc,
            exact_puzzle_recovered=False,
            exact_solution_recovered=False,
            puzzle_solvable=False,
            initial_conflicts=initial_conflicts,
            ac3_initial_consistent=ac3_ok,
            ac3_initial_metrics=ac3_metrics,
            ac3_singleton_domains_count=singleton_count,
            ac3_emptied_domains_count=emptied_count,
            injected_errors=injected_records,
            search_result=search_res,
            final_csp_result=None,
            correction_log=[],
            unreadable_report=unreadable_rep,
            total_pipeline_time_sec=total_dt,
            mosaic=mosaic,
        )
