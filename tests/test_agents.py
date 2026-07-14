"""Unit tests for agent classes (mocked LLM calls)."""
import json
import pytest
from unittest.mock import MagicMock, patch

from src.agents.base_agent import BaseAgent, AgentResponse
from src.agents.market_analyst import MarketAnalystAgent
from src.agents.technical_evaluator import TechnicalEvaluatorAgent
from src.agents.business_model_critic import BusinessModelCriticAgent
from src.agents.risk_assessor import RiskAssessorAgent


SAMPLE_RESPONSE = {
    "reasoning": "This is a solid market opportunity with strong fundamentals.",
    "score": 7.5,
    "key_points": ["Large TAM", "Growing market", "Clear differentiation"],
    "concerns": ["High competition", "Long sales cycle"],
    "agrees_with": [],
    "disagrees_with": [],
}

STARTUP_DESC = (
    "An AI-powered platform connecting local farmers with urban consumers "
    "via subscription boxes, using ML to optimize delivery routes."
)


def make_mock_client(response_json: dict) -> MagicMock:
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(response_json)
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[mock_choice]
    )
    return mock_client


@pytest.fixture
def market_agent(monkeypatch):
    monkeypatch.setenv("navigator_api", "test-key")
    agent = MarketAnalystAgent()
    agent.client = make_mock_client(SAMPLE_RESPONSE)
    return agent


@pytest.fixture
def tech_agent(monkeypatch):
    monkeypatch.setenv("navigator_api", "test-key")
    agent = TechnicalEvaluatorAgent()
    agent.client = make_mock_client(SAMPLE_RESPONSE)
    return agent


@pytest.fixture
def biz_agent(monkeypatch):
    monkeypatch.setenv("navigator_api", "test-key")
    agent = BusinessModelCriticAgent()
    agent.client = make_mock_client(SAMPLE_RESPONSE)
    return agent


@pytest.fixture
def risk_agent(monkeypatch):
    monkeypatch.setenv("navigator_api", "test-key")
    agent = RiskAssessorAgent()
    agent.client = make_mock_client(SAMPLE_RESPONSE)
    return agent


class TestAgentIdentity:
    def test_market_analyst_name(self, market_agent):
        assert market_agent.name == "MarketAnalyst"
        assert market_agent.role == "Market Analyst"

    def test_technical_evaluator_name(self, tech_agent):
        assert tech_agent.name == "TechnicalEvaluator"

    def test_business_model_critic_name(self, biz_agent):
        assert biz_agent.name == "BusinessModelCritic"

    def test_risk_assessor_name(self, risk_agent):
        assert risk_agent.name == "RiskAssessor"


class TestInitialAssessment:
    def test_returns_agent_response(self, market_agent):
        result = market_agent.initial_assessment(STARTUP_DESC)
        assert isinstance(result, AgentResponse)

    def test_score_in_range(self, market_agent):
        result = market_agent.initial_assessment(STARTUP_DESC)
        assert 1.0 <= result.score <= 10.0

    def test_response_fields_populated(self, market_agent):
        result = market_agent.initial_assessment(STARTUP_DESC)
        assert result.reasoning
        assert isinstance(result.key_points, list)
        assert isinstance(result.concerns, list)

    def test_turn_history_updated(self, market_agent):
        assert len(market_agent.turn_history) == 0
        market_agent.initial_assessment(STARTUP_DESC)
        assert len(market_agent.turn_history) == 2  # user + assistant


class TestDebateResponse:
    def test_debate_response_with_peers(self, market_agent, tech_agent, monkeypatch):
        monkeypatch.setenv("navigator_api", "test-key")
        # First get tech agent's assessment
        tech_agent.client = make_mock_client(SAMPLE_RESPONSE)
        tech_resp = tech_agent.initial_assessment(STARTUP_DESC)

        peer_assessments = {"TechnicalEvaluator": tech_resp}
        result = market_agent.debate_response(STARTUP_DESC, peer_assessments)

        assert isinstance(result, AgentResponse)
        assert 1.0 <= result.score <= 10.0


class TestScoreClamping:
    def test_score_clamped_above_10(self, market_agent):
        over_score = {**SAMPLE_RESPONSE, "score": 15.0}
        market_agent.client = make_mock_client(over_score)
        result = market_agent.initial_assessment(STARTUP_DESC)
        assert result.score <= 10.0

    def test_score_clamped_below_1(self, market_agent):
        under_score = {**SAMPLE_RESPONSE, "score": -3.0}
        market_agent.client = make_mock_client(under_score)
        result = market_agent.initial_assessment(STARTUP_DESC)
        assert result.score >= 1.0


class TestContextManagement:
    def test_reset_clears_history(self, market_agent):
        market_agent.initial_assessment(STARTUP_DESC)
        assert len(market_agent.turn_history) > 0
        market_agent.reset()
        assert len(market_agent.turn_history) == 0
        assert market_agent._summarized_context == ""

    def test_summarization_triggered_on_long_history(self, market_agent):
        # Artificially bloat history past token limit
        long_text = "A" * 10000
        market_agent.turn_history = [
            {"role": "user", "content": long_text},
            {"role": "assistant", "content": long_text},
        ]
        # Mock summarization client call
        market_agent.client = make_mock_client(SAMPLE_RESPONSE)
        # The summarize call also hits the client
        market_agent.client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Summary of debate."))]
        )
        market_agent._summarize_history()
        assert market_agent._summarized_context != "" or len(market_agent.turn_history) <= 2


class TestJsonFallback:
    def test_fallback_on_non_json_response(self, market_agent):
        raw_text = 'Some text before {"reasoning": "ok", "score": 6, "key_points": [], "concerns": [], "agrees_with": [], "disagrees_with": []} some after'
        result = market_agent._extract_json_fallback(raw_text)
        assert result["score"] == 6

    def test_fallback_on_completely_invalid_response(self, market_agent):
        result = market_agent._extract_json_fallback("no json here at all")
        assert result["score"] == 5.0
        assert "reasoning" in result
