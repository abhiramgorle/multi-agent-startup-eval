"""Unit tests for the DebateEngine and ContextManager."""
import json
import pytest
from unittest.mock import MagicMock, patch

from src.agents.base_agent import AgentResponse
from src.core.context_manager import DebateContext, AgentContext
from src.core.debate_engine import DebateEngine, EvaluationResult


SAMPLE_RESPONSE = AgentResponse(
    reasoning="Strong market opportunity with clear path to revenue.",
    score=7.0,
    key_points=["Large TAM", "Strong team"],
    concerns=["Regulatory risk"],
    agrees_with=[],
    disagrees_with=[],
)

STARTUP_DESC = "AI-driven supply chain optimization for SMB retailers."


def make_mock_agent(name: str, score: float = 7.0) -> MagicMock:
    agent = MagicMock()
    agent.name = name
    resp = AgentResponse(
        reasoning=f"{name} analysis complete.",
        score=score,
        key_points=["Point A", "Point B"],
        concerns=["Concern X"],
        agrees_with=[],
        disagrees_with=[],
    )
    agent.initial_assessment.return_value = resp
    agent.debate_response.return_value = resp
    agent.reset.return_value = None
    return agent


# ── ContextManager Tests ─────────────────────────────────────────────────────

class TestDebateContext:
    def test_update_adds_response(self):
        ctx = DebateContext(startup_description=STARTUP_DESC)
        ctx.update("MarketAnalyst", SAMPLE_RESPONSE, round_num=0)
        assert "MarketAnalyst" in ctx.agent_contexts

    def test_get_latest_responses(self):
        ctx = DebateContext(startup_description=STARTUP_DESC)
        ctx.update("MarketAnalyst", SAMPLE_RESPONSE, round_num=0)
        latest = ctx.get_latest_responses()
        assert "MarketAnalyst" in latest
        assert latest["MarketAnalyst"].score == SAMPLE_RESPONSE.score

    def test_score_variance_single_agent(self):
        ctx = DebateContext(startup_description=STARTUP_DESC)
        ctx.update("MarketAnalyst", SAMPLE_RESPONSE, round_num=0)
        assert ctx.score_variance() == 0.0

    def test_score_variance_multiple_agents(self):
        ctx = DebateContext(startup_description=STARTUP_DESC)
        resp_high = AgentResponse(
            reasoning="High score.", score=9.0, key_points=[], concerns=[],
            agrees_with=[], disagrees_with=[]
        )
        resp_low = AgentResponse(
            reasoning="Low score.", score=3.0, key_points=[], concerns=[],
            agrees_with=[], disagrees_with=[]
        )
        ctx.update("Agent1", resp_high, round_num=0)
        ctx.update("Agent2", resp_low, round_num=0)
        assert ctx.score_variance() > 0

    def test_consensus_reached_low_variance(self):
        ctx = DebateContext(startup_description=STARTUP_DESC)
        for i, name in enumerate(["A", "B", "C", "D"]):
            resp = AgentResponse(
                reasoning="OK", score=7.0 + i * 0.1,
                key_points=[], concerns=[], agrees_with=[], disagrees_with=[]
            )
            ctx.update(name, resp, round_num=0)
        assert ctx.consensus_reached(threshold=1.0)

    def test_consensus_not_reached_high_variance(self):
        ctx = DebateContext(startup_description=STARTUP_DESC)
        scores = [2.0, 9.0, 3.0, 8.0]
        for name, score in zip(["A", "B", "C", "D"], scores):
            resp = AgentResponse(
                reasoning="OK", score=score,
                key_points=[], concerns=[], agrees_with=[], disagrees_with=[]
            )
            ctx.update(name, resp, round_num=0)
        assert not ctx.consensus_reached(threshold=0.5)

    def test_needs_compression(self):
        ctx = DebateContext(startup_description=STARTUP_DESC, max_tokens=1)
        big_resp = AgentResponse(
            reasoning="X" * 5000, score=5.0, key_points=["A" * 100],
            concerns=["B" * 100], agrees_with=[], disagrees_with=[]
        )
        ctx.update("Agent1", big_resp, round_num=0)
        assert ctx.needs_compression()

    def test_get_scores_by_round(self):
        ctx = DebateContext(startup_description=STARTUP_DESC)
        ctx.update("A", SAMPLE_RESPONSE, round_num=0)
        scores = ctx.get_scores_by_round()
        assert 0 in scores
        assert scores[0]["A"] == SAMPLE_RESPONSE.score


class TestAgentContext:
    def test_score_trajectory(self):
        ctx = AgentContext(agent_name="Test")
        for score in [5.0, 6.0, 7.0]:
            resp = AgentResponse(
                reasoning="ok", score=score, key_points=[], concerns=[],
                agrees_with=[], disagrees_with=[]
            )
            ctx.add_response(resp, round_num=0)
        assert ctx.score_trajectory() == [5.0, 6.0, 7.0]

    def test_latest_response_none_when_empty(self):
        ctx = AgentContext(agent_name="Test")
        assert ctx.latest_response() is None


# ── DebateEngine Tests ───────────────────────────────────────────────────────

class TestDebateEngine:
    @pytest.fixture
    def engine_with_mocks(self, monkeypatch):
        monkeypatch.setenv("navigator_api", "test-key")
        engine = DebateEngine.__new__(DebateEngine)
        from concurrent.futures import ThreadPoolExecutor
        engine._executor = ThreadPoolExecutor(max_workers=5)

        # Mock all 4 domain agents with agreeing scores (triggers early stop)
        engine.agents = [
            make_mock_agent("MarketAnalyst", 7.0),
            make_mock_agent("TechnicalEvaluator", 7.2),
            make_mock_agent("BusinessModelCritic", 6.8),
            make_mock_agent("RiskAssessor", 7.1),
        ]

        # Mock synthesis judge
        from src.agents.synthesis_judge import FinalVerdict
        mock_judge = MagicMock()
        mock_judge.synthesize.return_value = FinalVerdict(
            executive_summary="Strong startup concept with good market fit.",
            final_score=7.0,
            recommendation="Invest",
            strengths=["Large market", "Strong team"],
            weaknesses=["Competitive market"],
            key_risks=["Regulatory risk"],
            next_steps=["Conduct customer discovery", "Build MVP"],
            agent_scores={"MarketAnalyst": 7.0, "TechnicalEvaluator": 7.2,
                          "BusinessModelCritic": 6.8, "RiskAssessor": 7.1},
            consensus_level="High",
            debate_rounds=1,
        )
        engine.judge = mock_judge
        return engine

    def test_evaluate_returns_result(self, engine_with_mocks):
        result = engine_with_mocks.evaluate(STARTUP_DESC)
        assert isinstance(result, EvaluationResult)

    def test_evaluate_has_verdict(self, engine_with_mocks):
        result = engine_with_mocks.evaluate(STARTUP_DESC)
        assert result.verdict is not None
        assert result.verdict.final_score > 0

    def test_evaluate_has_debate_history(self, engine_with_mocks):
        result = engine_with_mocks.evaluate(STARTUP_DESC)
        assert len(result.debate_history) >= 1

    def test_evaluate_early_stop_on_consensus(self, engine_with_mocks):
        # With very similar scores, should stop early
        result = engine_with_mocks.evaluate(STARTUP_DESC)
        assert result.early_stopped is True

    def test_evaluate_timing(self, engine_with_mocks):
        result = engine_with_mocks.evaluate(STARTUP_DESC)
        assert result.elapsed_seconds >= 0

    def test_all_agents_called_in_round_0(self, engine_with_mocks):
        engine_with_mocks.evaluate(STARTUP_DESC)
        for agent in engine_with_mocks.agents:
            agent.initial_assessment.assert_called_once()

    def test_agents_reset_before_evaluation(self, engine_with_mocks):
        engine_with_mocks.evaluate(STARTUP_DESC)
        for agent in engine_with_mocks.agents:
            agent.reset.assert_called_once()

    def test_judge_synthesize_called(self, engine_with_mocks):
        engine_with_mocks.evaluate(STARTUP_DESC)
        engine_with_mocks.judge.synthesize.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_async(self, engine_with_mocks):
        result = await engine_with_mocks.evaluate_async(STARTUP_DESC)
        assert isinstance(result, EvaluationResult)
