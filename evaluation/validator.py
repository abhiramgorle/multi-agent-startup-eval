"""
Validation runner: executes the debate engine against ground-truth cases
and collects CaseResult objects for metric computation.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from evaluation.ground_truths import GroundTruth, GROUND_TRUTH_DATASET
from evaluation.metrics import CaseResult, EvaluationMetrics, MetricsCalculator
from src.core.debate_engine import DebateEngine, EvaluationResult
from src.agents.base_agent import AgentResponse


def _extract_round_0_scores(
    debate_history: list[dict[str, AgentResponse]]
) -> dict[str, float]:
    if not debate_history:
        return {}
    return {name: resp.score for name, resp in debate_history[0].items()}


def _extract_final_scores(
    debate_history: list[dict[str, AgentResponse]]
) -> dict[str, float]:
    if not debate_history:
        return {}
    return {name: resp.score for name, resp in debate_history[-1].items()}


class ValidationRunner:
    """
    Runs the DebateEngine against ground-truth cases and measures performance.

    Args:
        engine_factory: callable returning a DebateEngine (or compatible mock).
            Defaults to creating a real DebateEngine.
        max_workers: number of parallel evaluation threads.
        progress_callback: optional callable(case_id, status) for progress reporting.
    """

    def __init__(
        self,
        engine_factory: Callable[[], DebateEngine] | None = None,
        max_workers: int = 3,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self._engine_factory = engine_factory or DebateEngine
        self._max_workers = max_workers
        self._progress = progress_callback or (lambda cid, status: None)
        self._calculator = MetricsCalculator()

    def _run_single(
        self,
        engine: DebateEngine,
        gt: GroundTruth,
    ) -> CaseResult:
        self._progress(gt.id, "RUNNING")
        try:
            result: EvaluationResult = engine.evaluate(gt.description)
            round_0 = _extract_round_0_scores(result.debate_history)
            final = _extract_final_scores(result.debate_history)

            case = CaseResult(
                ground_truth=gt,
                final_score=result.verdict.final_score,
                recommendation=result.verdict.recommendation,
                agent_scores=final,
                round_0_scores=round_0,
                debate_rounds=result.verdict.debate_rounds,
                early_stopped=result.early_stopped,
                elapsed_seconds=result.elapsed_seconds,
            )
            self._progress(gt.id, "DONE")
            return case
        except Exception as exc:
            self._progress(gt.id, f"ERROR: {exc}")
            raise

    def run(
        self,
        dataset: list[GroundTruth] | None = None,
        parallel: bool = True,
    ) -> tuple[list[CaseResult], EvaluationMetrics]:
        """
        Evaluate all ground-truth cases (or a provided subset).

        Returns:
            (case_results, aggregate_metrics)
        """
        cases = dataset if dataset is not None else GROUND_TRUTH_DATASET
        results: list[CaseResult] = []

        if parallel and self._max_workers > 1:
            # Each thread gets its own engine instance for thread safety
            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                futures = {
                    pool.submit(self._run_single, self._engine_factory(), gt): gt
                    for gt in cases
                }
                for future in as_completed(futures):
                    gt = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        print(f"[WARN] Case {gt.id} failed: {exc}")
        else:
            engine = self._engine_factory()
            for gt in cases:
                try:
                    results.append(self._run_single(engine, gt))
                except Exception as exc:
                    print(f"[WARN] Case {gt.id} failed: {exc}")

        metrics = self._calculator.compute(results)
        return results, metrics

    def run_consistency_check(
        self,
        dataset: list[GroundTruth] | None = None,
        n_reruns: int = 3,
    ) -> tuple[float, float]:
        """
        Run each case n_reruns times to measure output consistency.

        Returns:
            (rerun_score_std, rerun_recommendation_consistency)
        """
        cases = dataset if dataset is not None else GROUND_TRUTH_DATASET
        results_by_case: dict[str, list[CaseResult]] = {}

        for gt in cases:
            run_results = []
            for _ in range(n_reruns):
                engine = self._engine_factory()
                try:
                    run_results.append(self._run_single(engine, gt))
                except Exception:
                    pass
            if run_results:
                results_by_case[gt.id] = run_results

        return self._calculator.compute_consistency(results_by_case)

    def run_tier_analysis(
        self,
        results: list[CaseResult],
    ) -> dict[int, dict[str, object]]:
        """
        Returns per-tier breakdown of predictions vs ground truth.
        Useful for identifying which startup archetypes the model struggles with.
        """
        analysis: dict[int, dict[str, object]] = {}
        for tier in range(1, 6):
            tier_results = [r for r in results if r.ground_truth.expected_tier == tier]
            if not tier_results:
                continue

            analysis[tier] = {
                "n_cases": len(tier_results),
                "exact_match": sum(r.tier_error == 0 for r in tier_results),
                "off_by_1": sum(r.tier_error == 1 for r in tier_results),
                "off_by_2_plus": sum(r.tier_error >= 2 for r in tier_results),
                "avg_predicted_score": round(
                    sum(r.final_score for r in tier_results) / len(tier_results), 2
                ),
                "avg_expected_midpoint": round(
                    sum(r.expected_midpoint for r in tier_results) / len(tier_results), 2
                ),
                "avg_latency_s": round(
                    sum(r.elapsed_seconds for r in tier_results) / len(tier_results), 2
                ),
                "cases": [
                    {
                        "id": r.ground_truth.id,
                        "name": r.ground_truth.name,
                        "expected": r.ground_truth.expected_recommendation,
                        "predicted": r.recommendation,
                        "expected_score_range": (
                            r.ground_truth.expected_score_min,
                            r.ground_truth.expected_score_max,
                        ),
                        "actual_score": round(r.final_score, 2),
                        "within_range": r.within_expected_range,
                        "tier_error": r.tier_error,
                    }
                    for r in tier_results
                ],
            }
        return analysis
