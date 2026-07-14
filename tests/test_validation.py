"""
Tests for the validation and metrics framework.
All tests use mocked debate engines — no real LLM calls.
"""
from __future__ import annotations

import math
import pytest
from unittest.mock import MagicMock

from evaluation.ground_truths import (
    GROUND_TRUTH_DATASET,
    GroundTruth,
    _T1_DEVOPS_AI,
    _T1_HEALTH_RECORDS,
    _T2_MARKETPLACE,
    _T3_SOCIAL,
    _T4_GENERIC_AI,
    _T5_PERPETUAL_MOTION,
)
from evaluation.metrics import (
    CaseResult,
    EvaluationMetrics,
    MetricsCalculator,
    _score_to_recommendation,
    _recommendation_to_tier,
)
from evaluation.validator import ValidationRunner
from evaluation.report import ReportGenerator
from src.agents.base_agent import AgentResponse
from src.agents.synthesis_judge import FinalVerdict
from src.core.debate_engine import EvaluationResult


# ── Helpers ─────────────────────────────────────────────────────────────────

def make_agent_response(score: float) -> AgentResponse:
    return AgentResponse(
        reasoning="Test reasoning.",
        score=score,
        key_points=["Point A"],
        concerns=["Concern B"],
        agrees_with=[],
        disagrees_with=[],
    )


def make_eval_result(
    description: str,
    final_score: float,
    round_0_scores: dict[str, float] | None = None,
    final_agent_scores: dict[str, float] | None = None,
    debate_rounds: int = 1,
    early_stopped: bool = True,
    elapsed: float = 10.0,
) -> EvaluationResult:
    r0 = round_0_scores or {"A": 7.0, "B": 6.5, "C": 7.5, "D": 6.0}
    final = final_agent_scores or {"A": 7.0, "B": 6.5, "C": 7.5, "D": 6.0}

    verdict = FinalVerdict(
        executive_summary="Test summary.",
        final_score=final_score,
        recommendation=_score_to_recommendation(final_score),
        strengths=["S1"],
        weaknesses=["W1"],
        key_risks=["R1"],
        next_steps=["N1"],
        agent_scores=final,
        consensus_level="High",
        debate_rounds=debate_rounds,
    )

    history = [
        {name: make_agent_response(score) for name, score in r0.items()},
    ]
    if debate_rounds > 1:
        history.append({name: make_agent_response(score) for name, score in final.items()})

    return EvaluationResult(
        startup_description=description,
        verdict=verdict,
        debate_history=history,
        elapsed_seconds=elapsed,
        early_stopped=early_stopped,
    )


def make_mock_engine(score_map: dict[str, float]) -> MagicMock:
    """Creates a mock DebateEngine that returns preset scores per startup description."""
    engine = MagicMock()

    def fake_evaluate(description: str) -> EvaluationResult:
        score = score_map.get(description, 5.0)
        return make_eval_result(description, final_score=score)

    engine.evaluate.side_effect = fake_evaluate
    return engine


def make_perfect_engine() -> MagicMock:
    """Engine that always scores at the expected midpoint."""
    engine = MagicMock()

    def fake_evaluate(description: str) -> EvaluationResult:
        gt = next((g for g in GROUND_TRUTH_DATASET if g.description == description), None)
        if gt:
            midpoint = (gt.expected_score_min + gt.expected_score_max) / 2
        else:
            midpoint = 5.0
        return make_eval_result(description, final_score=midpoint)

    engine.evaluate.side_effect = fake_evaluate
    return engine


def make_case_result(
    gt: GroundTruth,
    final_score: float,
    round_0_scores: dict[str, float] | None = None,
    final_scores: dict[str, float] | None = None,
    elapsed: float = 10.0,
    early_stopped: bool = True,
    debate_rounds: int = 1,
) -> CaseResult:
    r0 = round_0_scores or {"A": final_score, "B": final_score}
    fs = final_scores or {"A": final_score, "B": final_score}
    return CaseResult(
        ground_truth=gt,
        final_score=final_score,
        recommendation=_score_to_recommendation(final_score),
        agent_scores=fs,
        round_0_scores=r0,
        debate_rounds=debate_rounds,
        early_stopped=early_stopped,
        elapsed_seconds=elapsed,
    )


# ── GroundTruth Dataset Tests ────────────────────────────────────────────────

class TestGroundTruthDataset:
    def test_dataset_has_15_cases(self):
        assert len(GROUND_TRUTH_DATASET) == 15

    def test_exactly_3_per_tier(self):
        for tier in range(1, 6):
            cases = [gt for gt in GROUND_TRUTH_DATASET if gt.expected_tier == tier]
            assert len(cases) == 3, f"Tier {tier} should have 3 cases, got {len(cases)}"

    def test_all_ids_unique(self):
        ids = [gt.id for gt in GROUND_TRUTH_DATASET]
        assert len(set(ids)) == len(ids)

    def test_all_descriptions_nonempty(self):
        for gt in GROUND_TRUTH_DATASET:
            assert len(gt.description) >= 50, f"{gt.id} description too short"

    def test_tier1_score_ranges_above_7(self):
        for gt in GROUND_TRUTH_DATASET:
            if gt.expected_tier == 1:
                assert gt.expected_score_min >= 7.0, f"{gt.id} tier 1 min too low"

    def test_tier5_score_ranges_below_4(self):
        for gt in GROUND_TRUTH_DATASET:
            if gt.expected_tier == 5:
                assert gt.expected_score_max <= 4.0, f"{gt.id} tier 5 max too high"

    def test_expected_ranges_are_valid(self):
        for gt in GROUND_TRUTH_DATASET:
            assert gt.expected_score_min < gt.expected_score_max, (
                f"{gt.id} min {gt.expected_score_min} >= max {gt.expected_score_max}"
            )
            assert 1.0 <= gt.expected_score_min
            assert gt.expected_score_max <= 10.0

    def test_recommendation_matches_expected_tier(self):
        tier_to_rec = {
            1: "Strong Invest",
            2: "Invest",
            3: "Conditional Pass",
            4: "Pass",
            5: "Strong Pass",
        }
        for gt in GROUND_TRUTH_DATASET:
            assert gt.expected_recommendation == tier_to_rec[gt.expected_tier], (
                f"{gt.id}: tier {gt.expected_tier} should map to "
                f"'{tier_to_rec[gt.expected_tier]}' not '{gt.expected_recommendation}'"
            )


# ── CaseResult Tests ─────────────────────────────────────────────────────────

class TestCaseResult:
    def test_within_range_true(self):
        gt = _T1_DEVOPS_AI
        mid = (gt.expected_score_min + gt.expected_score_max) / 2
        case = make_case_result(gt, final_score=mid)
        assert case.within_expected_range

    def test_within_range_false_too_low(self):
        case = make_case_result(_T1_DEVOPS_AI, final_score=3.0)
        assert not case.within_expected_range

    def test_within_range_false_too_high(self):
        case = make_case_result(_T5_PERPETUAL_MOTION, final_score=9.5)
        assert not case.within_expected_range

    def test_recommendation_correct(self):
        mid = (_T1_DEVOPS_AI.expected_score_min + _T1_DEVOPS_AI.expected_score_max) / 2
        case = make_case_result(_T1_DEVOPS_AI, final_score=mid)
        assert case.recommendation_correct

    def test_recommendation_incorrect(self):
        case = make_case_result(_T1_DEVOPS_AI, final_score=2.0)
        assert not case.recommendation_correct

    def test_tier_error_zero_on_correct(self):
        mid = (_T2_MARKETPLACE.expected_score_min + _T2_MARKETPLACE.expected_score_max) / 2
        case = make_case_result(_T2_MARKETPLACE, final_score=mid)
        assert case.predicted_tier == 2
        assert case.tier_error == 0

    def test_tier_error_calculated_correctly(self):
        # T5 case scored as T1
        case = make_case_result(_T5_PERPETUAL_MOTION, final_score=9.0)
        assert case.tier_error == 4

    def test_inter_agent_std_zero_when_all_same(self):
        case = make_case_result(
            _T1_DEVOPS_AI, final_score=7.5,
            final_scores={"A": 7.5, "B": 7.5, "C": 7.5, "D": 7.5}
        )
        assert case.inter_agent_std == pytest.approx(0.0)

    def test_inter_agent_std_nonzero_when_different(self):
        case = make_case_result(
            _T1_DEVOPS_AI, final_score=7.0,
            final_scores={"A": 5.0, "B": 9.0, "C": 6.0, "D": 8.0}
        )
        assert case.inter_agent_std > 0

    def test_debate_score_delta(self):
        case = make_case_result(
            _T1_DEVOPS_AI, final_score=8.0,
            round_0_scores={"A": 6.0, "B": 6.0},
            final_scores={"A": 8.0, "B": 8.0},
        )
        assert case.debate_score_delta == pytest.approx(2.0)

    def test_expected_midpoint(self):
        gt = _T1_DEVOPS_AI
        case = make_case_result(gt, final_score=8.5)
        expected_mid = (gt.expected_score_min + gt.expected_score_max) / 2
        assert case.expected_midpoint == pytest.approx(expected_mid)

    def test_score_error(self):
        gt = _T3_SOCIAL
        mid = (gt.expected_score_min + gt.expected_score_max) / 2
        case = make_case_result(gt, final_score=mid + 1.0)
        assert case.score_error == pytest.approx(1.0)


# ── MetricsCalculator Tests ──────────────────────────────────────────────────

class TestMetricsCalculator:
    def setup_method(self):
        self.calc = MetricsCalculator()

    def _perfect_results(self) -> list[CaseResult]:
        results = []
        for gt in GROUND_TRUTH_DATASET:
            mid = (gt.expected_score_min + gt.expected_score_max) / 2
            results.append(make_case_result(gt, final_score=mid))
        return results

    def _worst_results(self) -> list[CaseResult]:
        """All predictions maximally wrong."""
        results = []
        for gt in GROUND_TRUTH_DATASET:
            wrong_score = 10.0 if gt.expected_tier == 5 else 1.0
            results.append(make_case_result(gt, final_score=wrong_score))
        return results

    def test_empty_returns_default_metrics(self):
        m = self.calc.compute([])
        assert m.n_cases == 0
        assert m.score_mae == 0.0

    def test_perfect_predictions_zero_mae(self):
        m = self.calc.compute(self._perfect_results())
        assert m.score_mae == pytest.approx(0.0, abs=1e-6)

    def test_perfect_predictions_100_pct_in_range(self):
        m = self.calc.compute(self._perfect_results())
        assert m.within_range_rate == pytest.approx(1.0)

    def test_perfect_predictions_100_pct_recommendation(self):
        m = self.calc.compute(self._perfect_results())
        assert m.recommendation_accuracy == pytest.approx(1.0)

    def test_worst_predictions_high_mae(self):
        m = self.calc.compute(self._worst_results())
        assert m.score_mae > 3.0

    def test_worst_predictions_zero_accuracy(self):
        m = self.calc.compute(self._worst_results())
        assert m.within_range_rate == pytest.approx(0.0)

    def test_mae_matches_manual_calculation(self):
        gt = _T1_DEVOPS_AI
        mid = (gt.expected_score_min + gt.expected_score_max) / 2
        cases = [make_case_result(gt, final_score=mid + 1.0)]
        m = self.calc.compute(cases)
        assert m.score_mae == pytest.approx(1.0)

    def test_rmse_always_gte_mae(self):
        m = self.calc.compute(self._worst_results())
        assert m.score_rmse >= m.score_mae

    def test_per_tier_breakdown_has_all_tiers(self):
        m = self.calc.compute(self._perfect_results())
        for tier in range(1, 6):
            assert tier in m.tier_accuracy_by_tier

    def test_early_stop_rate(self):
        cases = [
            make_case_result(_T1_DEVOPS_AI, 8.5, early_stopped=True),
            make_case_result(_T2_MARKETPLACE, 7.0, early_stopped=False),
            make_case_result(_T3_SOCIAL, 5.5, early_stopped=True),
            make_case_result(_T4_GENERIC_AI, 3.5, early_stopped=False),
        ]
        m = self.calc.compute(cases)
        assert m.early_stop_rate == pytest.approx(0.5)

    def test_avg_latency(self):
        cases = [
            make_case_result(_T1_DEVOPS_AI, 8.5, elapsed=10.0),
            make_case_result(_T2_MARKETPLACE, 7.0, elapsed=30.0),
        ]
        m = self.calc.compute(cases)
        assert m.avg_latency_s == pytest.approx(20.0)

    def test_p95_latency_with_small_set(self):
        cases = [make_case_result(_T1_DEVOPS_AI, 8.0, elapsed=float(i)) for i in range(1, 5)]
        m = self.calc.compute(cases)
        assert m.p95_latency_s >= 3.0

    def test_per_agent_scoring_computed(self):
        cases = [
            make_case_result(_T1_DEVOPS_AI, 8.0, final_scores={"MarketAnalyst": 8.0, "TechEval": 7.5}),
            make_case_result(_T2_MARKETPLACE, 7.0, final_scores={"MarketAnalyst": 7.0, "TechEval": 7.0}),
        ]
        m = self.calc.compute(cases)
        assert "MarketAnalyst" in m.mean_score_by_agent
        assert m.mean_score_by_agent["MarketAnalyst"] == pytest.approx(7.5)

    def test_tier_accuracy_lenient_allows_off_by_one(self):
        # Score a Tier 1 case as Tier 2 (off by 1 — should count in lenient)
        case = make_case_result(_T1_DEVOPS_AI, final_score=7.0)  # "Invest" = Tier 2
        assert case.ground_truth.expected_tier == 1
        assert case.predicted_tier == 2
        m = self.calc.compute([case])
        assert m.tier_accuracy_lenient == pytest.approx(1.0)
        assert m.tier_accuracy_strict == pytest.approx(0.0)

    def test_consistency_with_reruns(self):
        results_by_case = {
            "case1": [
                make_case_result(_T1_DEVOPS_AI, 8.0),
                make_case_result(_T1_DEVOPS_AI, 8.5),
                make_case_result(_T1_DEVOPS_AI, 7.8),
            ]
        }
        std, cons = self.calc.compute_consistency(results_by_case)
        assert 0.0 < std < 1.0
        assert 0.0 <= cons <= 1.0

    def test_consistency_with_identical_runs(self):
        results_by_case = {
            "case1": [
                make_case_result(_T1_DEVOPS_AI, 8.5),
                make_case_result(_T1_DEVOPS_AI, 8.5),
                make_case_result(_T1_DEVOPS_AI, 8.5),
            ]
        }
        std, cons = self.calc.compute_consistency(results_by_case)
        assert std == pytest.approx(0.0)
        assert cons == pytest.approx(1.0)

    def test_passes_ci_threshold_with_perfect_results(self):
        m = self.calc.compute(self._perfect_results())
        passed, failures = MetricsCalculator.passes_ci_threshold(m)
        assert passed
        assert failures == []

    def test_fails_ci_threshold_with_worst_results(self):
        m = self.calc.compute(self._worst_results())
        passed, failures = MetricsCalculator.passes_ci_threshold(m)
        assert not passed
        assert len(failures) > 0

    def test_scorecard_returns_all_keys(self):
        m = self.calc.compute(self._perfect_results())
        card = MetricsCalculator.score_card(m)
        assert "within_range_rate" in card
        assert "recommendation_accuracy" in card
        assert "tier_accuracy_lenient" in card
        assert "early_stop_efficiency" in card
        assert "agent_consensus" in card

    def test_scorecard_all_A_on_perfect(self):
        m = self.calc.compute(self._perfect_results())
        card = MetricsCalculator.score_card(m)
        for metric, grade in card.items():
            assert grade in ("A", "B"), f"{metric} got {grade} on perfect results"


# ── ValidationRunner Tests ───────────────────────────────────────────────────

class TestValidationRunner:
    def _perfect_factory(self):
        return make_perfect_engine()

    def test_run_returns_15_results(self):
        runner = ValidationRunner(
            engine_factory=self._perfect_factory,
            max_workers=1,
        )
        results, metrics = runner.run(parallel=False)
        assert len(results) == 15

    def test_run_perfect_engine_high_accuracy(self):
        runner = ValidationRunner(
            engine_factory=self._perfect_factory,
            max_workers=1,
        )
        results, metrics = runner.run(parallel=False)
        assert metrics.within_range_rate == pytest.approx(1.0)
        assert metrics.recommendation_accuracy == pytest.approx(1.0)

    def test_run_subset_by_tier(self):
        tier_1_cases = [gt for gt in GROUND_TRUTH_DATASET if gt.expected_tier == 1]
        runner = ValidationRunner(
            engine_factory=self._perfect_factory,
            max_workers=1,
        )
        results, metrics = runner.run(dataset=tier_1_cases, parallel=False)
        assert len(results) == 3
        assert metrics.n_cases == 3

    def test_progress_callback_called(self):
        calls = []
        runner = ValidationRunner(
            engine_factory=self._perfect_factory,
            max_workers=1,
            progress_callback=lambda cid, status: calls.append((cid, status)),
        )
        runner.run(dataset=[_T1_DEVOPS_AI], parallel=False)
        assert any(s == "DONE" for _, s in calls)

    def test_run_tier_analysis_structure(self):
        runner = ValidationRunner(engine_factory=self._perfect_factory, max_workers=1)
        results, _ = runner.run(parallel=False)
        analysis = runner.run_tier_analysis(results)
        assert set(analysis.keys()) == {1, 2, 3, 4, 5}
        for tier, data in analysis.items():
            assert "n_cases" in data
            assert "exact_match" in data
            assert "cases" in data
            assert len(data["cases"]) == 3

    def test_run_parallel(self):
        runner = ValidationRunner(engine_factory=self._perfect_factory, max_workers=3)
        results, metrics = runner.run(parallel=True)
        assert len(results) == 15

    def test_consistency_check(self):
        runner = ValidationRunner(engine_factory=self._perfect_factory, max_workers=1)
        std, cons = runner.run_consistency_check(
            dataset=[_T1_DEVOPS_AI, _T2_MARKETPLACE],
            n_reruns=2,
        )
        assert isinstance(std, float)
        assert isinstance(cons, float)
        assert 0.0 <= cons <= 1.0


# ── Utility Function Tests ────────────────────────────────────────────────────

class TestUtilityFunctions:
    def test_score_to_recommendation_boundaries(self):
        assert _score_to_recommendation(10.0) == "Strong Invest"
        assert _score_to_recommendation(8.0) == "Strong Invest"
        assert _score_to_recommendation(7.9) == "Invest"
        assert _score_to_recommendation(6.5) == "Invest"
        assert _score_to_recommendation(6.4) == "Conditional Pass"
        assert _score_to_recommendation(5.0) == "Conditional Pass"
        assert _score_to_recommendation(4.9) == "Pass"
        assert _score_to_recommendation(3.5) == "Pass"
        assert _score_to_recommendation(3.4) == "Strong Pass"
        assert _score_to_recommendation(1.0) == "Strong Pass"

    def test_recommendation_to_tier_mapping(self):
        assert _recommendation_to_tier("Strong Invest") == 1
        assert _recommendation_to_tier("Invest") == 2
        assert _recommendation_to_tier("Conditional Pass") == 3
        assert _recommendation_to_tier("Pass") == 4
        assert _recommendation_to_tier("Strong Pass") == 5

    def test_recommendation_to_tier_unknown_defaults_to_3(self):
        assert _recommendation_to_tier("Unknown") == 3


# ── ReportGenerator Tests ─────────────────────────────────────────────────────

class TestReportGenerator:
    def setup_method(self):
        self.reporter = ReportGenerator()
        self.cases = [
            make_case_result(
                _T1_DEVOPS_AI, 8.9,
                final_scores={"MarketAnalyst": 9.0, "TechEval": 8.8},
                round_0_scores={"MarketAnalyst": 8.0, "TechEval": 7.5},
                elapsed=12.0,
            ),
            make_case_result(
                _T5_PERPETUAL_MOTION, 2.0,
                final_scores={"MarketAnalyst": 2.0, "TechEval": 2.5},
                round_0_scores={"MarketAnalyst": 2.5, "TechEval": 3.0},
                elapsed=8.0,
            ),
        ]
        calc = MetricsCalculator()
        self.metrics = calc.compute(self.cases)

    def test_to_json_valid_json(self):
        import json
        output = self.reporter.to_json(self.cases, self.metrics)
        data = json.loads(output)
        assert "aggregate_metrics" in data
        assert "case_results" in data

    def test_to_json_has_ci_result(self):
        import json
        data = json.loads(self.reporter.to_json(self.cases, self.metrics))
        assert "ci_passed" in data
        assert "ci_failures" in data

    def test_to_json_case_count(self):
        import json
        data = json.loads(self.reporter.to_json(self.cases, self.metrics))
        assert data["n_cases"] == 2
        assert len(data["case_results"]) == 2

    def test_to_markdown_contains_header(self):
        md = self.reporter.to_markdown(self.cases, self.metrics)
        assert "# Multi-Agent Startup Evaluator" in md

    def test_to_markdown_contains_scorecard(self):
        md = self.reporter.to_markdown(self.cases, self.metrics)
        assert "Scorecard" in md

    def test_to_markdown_contains_case_names(self):
        md = self.reporter.to_markdown(self.cases, self.metrics)
        assert "DevOps AI Copilot"[:15] in md

    def test_print_summary_runs_without_error(self, capsys):
        import io
        buf = io.StringIO()
        self.reporter.print_summary(self.cases, self.metrics, out=buf)
        output = buf.getvalue()
        assert "Validation Report" in output
        assert "ACCURACY" in output

    def test_to_json_includes_tier_analysis_when_provided(self):
        import json
        tier_analysis = {1: {"n_cases": 1, "exact_match": 1, "cases": []}}
        data = json.loads(
            self.reporter.to_json(self.cases, self.metrics, tier_analysis=tier_analysis)
        )
        assert "tier_analysis" in data
        assert "1" in data["tier_analysis"]
