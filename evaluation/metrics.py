"""
AI performance metrics for the multi-agent startup evaluation system.

Metrics computed:
  Accuracy:
    - score_mae            Mean absolute error vs expected score midpoint
    - score_rmse           Root mean squared error vs expected score midpoint
    - within_range_rate    % of cases where final score falls in expected range
    - recommendation_accuracy  % of cases with correct recommendation label
    - tier_accuracy_strict     % of cases in exact expected tier
    - tier_accuracy_lenient    % of cases within ±1 tier of expected

  Agent behavior:
    - avg_inter_agent_std  Average std-dev of agent scores per evaluation (lower = more consensus)
    - debate_improvement   Average absolute score change from round 0 → final round
    - early_stop_rate      % of evaluations that triggered dynamic stopping

  Efficiency:
    - avg_latency_s        Average wall-clock seconds per evaluation
    - avg_debate_rounds    Average number of debate rounds completed
    - p95_latency_s        95th-percentile latency

  Consistency (requires multiple runs on the same input):
    - rerun_score_std      Std-dev of scores across repeated runs of identical inputs
    - rerun_recommendation_consistency  % of reruns returning the same recommendation
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from evaluation.ground_truths import GroundTruth

if TYPE_CHECKING:
    pass


def _score_to_recommendation(score: float) -> str:
    if score >= 8.0:
        return "Strong Invest"
    elif score >= 6.5:
        return "Invest"
    elif score >= 5.0:
        return "Conditional Pass"
    elif score >= 3.5:
        return "Pass"
    else:
        return "Strong Pass"


def _recommendation_to_tier(rec: str) -> int:
    return {
        "Strong Invest": 1,
        "Invest": 2,
        "Conditional Pass": 3,
        "Pass": 4,
        "Strong Pass": 5,
    }.get(rec, 3)


@dataclass
class CaseResult:
    """Result for a single ground-truth evaluation case."""
    ground_truth: GroundTruth
    final_score: float
    recommendation: str
    agent_scores: dict[str, float]         # per-agent final scores
    round_0_scores: dict[str, float]       # per-agent initial scores
    debate_rounds: int
    early_stopped: bool
    elapsed_seconds: float

    # Derived
    @property
    def expected_midpoint(self) -> float:
        return (self.ground_truth.expected_score_min + self.ground_truth.expected_score_max) / 2

    @property
    def score_error(self) -> float:
        return abs(self.final_score - self.expected_midpoint)

    @property
    def within_expected_range(self) -> bool:
        return (
            self.ground_truth.expected_score_min
            <= self.final_score
            <= self.ground_truth.expected_score_max
        )

    @property
    def recommendation_correct(self) -> bool:
        return self.recommendation == self.ground_truth.expected_recommendation

    @property
    def predicted_tier(self) -> int:
        return _recommendation_to_tier(self.recommendation)

    @property
    def tier_error(self) -> int:
        return abs(self.predicted_tier - self.ground_truth.expected_tier)

    @property
    def inter_agent_std(self) -> float:
        scores = list(self.agent_scores.values())
        if len(scores) < 2:
            return 0.0
        avg = sum(scores) / len(scores)
        return math.sqrt(sum((s - avg) ** 2 for s in scores) / len(scores))

    @property
    def debate_score_delta(self) -> float:
        """Average absolute score change from round 0 to final round."""
        deltas = []
        for agent, final in self.agent_scores.items():
            r0 = self.round_0_scores.get(agent)
            if r0 is not None:
                deltas.append(abs(final - r0))
        return sum(deltas) / len(deltas) if deltas else 0.0


@dataclass
class EvaluationMetrics:
    """Aggregate performance metrics across all evaluated cases."""

    # Accuracy
    score_mae: float = 0.0
    score_rmse: float = 0.0
    within_range_rate: float = 0.0
    recommendation_accuracy: float = 0.0
    tier_accuracy_strict: float = 0.0
    tier_accuracy_lenient: float = 0.0        # within ±1 tier

    # Agent behavior
    avg_inter_agent_std: float = 0.0          # avg disagreement across agents
    avg_debate_improvement: float = 0.0       # avg score delta through debate
    early_stop_rate: float = 0.0

    # Efficiency
    avg_latency_s: float = 0.0
    p95_latency_s: float = 0.0
    avg_debate_rounds: float = 0.0

    # Consistency (populated only when reruns > 1)
    rerun_score_std: float | None = None
    rerun_recommendation_consistency: float | None = None

    # Per-tier breakdown
    tier_accuracy_by_tier: dict[int, float] = field(default_factory=dict)
    recommendation_accuracy_by_tier: dict[int, float] = field(default_factory=dict)

    # Per-agent scores
    mean_score_by_agent: dict[str, float] = field(default_factory=dict)
    std_score_by_agent: dict[str, float] = field(default_factory=dict)

    # Dataset size
    n_cases: int = 0
    n_tiers: int = 5


class MetricsCalculator:
    """Computes all performance metrics from a list of CaseResult objects."""

    def compute(self, results: list[CaseResult]) -> EvaluationMetrics:
        if not results:
            return EvaluationMetrics()

        m = EvaluationMetrics(n_cases=len(results))

        # ── Accuracy ────────────────────────────────────────────────────────
        errors = [r.score_error for r in results]
        m.score_mae = sum(errors) / len(errors)
        m.score_rmse = math.sqrt(sum(e ** 2 for e in errors) / len(errors))
        m.within_range_rate = sum(r.within_expected_range for r in results) / len(results)
        m.recommendation_accuracy = sum(r.recommendation_correct for r in results) / len(results)
        m.tier_accuracy_strict = sum(r.tier_error == 0 for r in results) / len(results)
        m.tier_accuracy_lenient = sum(r.tier_error <= 1 for r in results) / len(results)

        # ── Agent behavior ───────────────────────────────────────────────────
        m.avg_inter_agent_std = sum(r.inter_agent_std for r in results) / len(results)
        m.avg_debate_improvement = sum(r.debate_score_delta for r in results) / len(results)
        m.early_stop_rate = sum(r.early_stopped for r in results) / len(results)

        # ── Efficiency ───────────────────────────────────────────────────────
        latencies = sorted(r.elapsed_seconds for r in results)
        m.avg_latency_s = sum(latencies) / len(latencies)
        m.p95_latency_s = latencies[max(0, int(len(latencies) * 0.95) - 1)]
        m.avg_debate_rounds = sum(r.debate_rounds for r in results) / len(results)

        # ── Per-tier breakdown ───────────────────────────────────────────────
        for tier in range(1, 6):
            tier_cases = [r for r in results if r.ground_truth.expected_tier == tier]
            if tier_cases:
                m.tier_accuracy_by_tier[tier] = (
                    sum(r.tier_error == 0 for r in tier_cases) / len(tier_cases)
                )
                m.recommendation_accuracy_by_tier[tier] = (
                    sum(r.recommendation_correct for r in tier_cases) / len(tier_cases)
                )

        # ── Per-agent scoring ────────────────────────────────────────────────
        all_agent_names: set[str] = set()
        for r in results:
            all_agent_names.update(r.agent_scores.keys())

        for agent in all_agent_names:
            scores = [r.agent_scores[agent] for r in results if agent in r.agent_scores]
            if scores:
                avg = sum(scores) / len(scores)
                m.mean_score_by_agent[agent] = round(avg, 3)
                std = math.sqrt(sum((s - avg) ** 2 for s in scores) / len(scores))
                m.std_score_by_agent[agent] = round(std, 3)

        return m

    def compute_consistency(
        self,
        results_by_case: dict[str, list[CaseResult]],
    ) -> tuple[float, float]:
        """
        Compute rerun consistency metrics.

        Args:
            results_by_case: mapping of case_id → list of CaseResult (multiple runs)

        Returns:
            (rerun_score_std, rerun_recommendation_consistency)
        """
        score_stds = []
        rec_consistencies = []

        for case_id, runs in results_by_case.items():
            if len(runs) < 2:
                continue
            scores = [r.final_score for r in runs]
            avg = sum(scores) / len(scores)
            std = math.sqrt(sum((s - avg) ** 2 for s in scores) / len(scores))
            score_stds.append(std)

            recs = [r.recommendation for r in runs]
            mode_count = max(recs.count(r) for r in set(recs))
            rec_consistencies.append(mode_count / len(recs))

        avg_std = sum(score_stds) / len(score_stds) if score_stds else 0.0
        avg_cons = sum(rec_consistencies) / len(rec_consistencies) if rec_consistencies else 1.0
        return avg_std, avg_cons

    @staticmethod
    def score_card(metrics: EvaluationMetrics) -> dict[str, str]:
        """
        Returns a letter-graded scorecard (A/B/C/D/F) for each metric category.
        Useful for quick reporting and CI pass/fail thresholds.
        """
        def grade(value: float, thresholds: list[tuple[float, str]]) -> str:
            for threshold, letter in thresholds:
                if value >= threshold:
                    return letter
            return "F"

        return {
            "within_range_rate": grade(
                metrics.within_range_rate,
                [(0.80, "A"), (0.67, "B"), (0.53, "C"), (0.40, "D")],
            ),
            "recommendation_accuracy": grade(
                metrics.recommendation_accuracy,
                [(0.73, "A"), (0.60, "B"), (0.47, "C"), (0.33, "D")],
            ),
            "tier_accuracy_lenient": grade(
                metrics.tier_accuracy_lenient,
                [(0.87, "A"), (0.73, "B"), (0.60, "C"), (0.47, "D")],
            ),
            "early_stop_efficiency": grade(
                metrics.early_stop_rate,
                [(0.60, "A"), (0.40, "B"), (0.20, "C"), (0.10, "D")],
            ),
            "agent_consensus": grade(
                max(0.0, 1.0 - metrics.avg_inter_agent_std / 3.0),
                [(0.80, "A"), (0.60, "B"), (0.40, "C"), (0.20, "D")],
            ),
        }

    @staticmethod
    def passes_ci_threshold(metrics: EvaluationMetrics) -> tuple[bool, list[str]]:
        """
        Returns (passed, list_of_failures) for CI gate.
        Thresholds represent minimum acceptable performance.
        """
        failures = []

        if metrics.within_range_rate < 0.50:
            failures.append(
                f"within_range_rate {metrics.within_range_rate:.0%} < 50% minimum"
            )
        if metrics.recommendation_accuracy < 0.40:
            failures.append(
                f"recommendation_accuracy {metrics.recommendation_accuracy:.0%} < 40% minimum"
            )
        if metrics.tier_accuracy_lenient < 0.60:
            failures.append(
                f"tier_accuracy_lenient {metrics.tier_accuracy_lenient:.0%} < 60% minimum"
            )
        if metrics.score_mae > 2.5:
            failures.append(
                f"score_mae {metrics.score_mae:.2f} > 2.5 maximum"
            )

        return len(failures) == 0, failures
