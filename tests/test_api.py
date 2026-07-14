"""Integration tests for the FastAPI application."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

from src.api.app import app
from src.core.debate_engine import EvaluationResult
from src.agents.base_agent import AgentResponse
from src.agents.synthesis_judge import FinalVerdict


STARTUP_DESC = "An AI marketplace for freelance robotics engineers, matching them with manufacturing SMBs."

MOCK_AGENT_RESPONSE = AgentResponse(
    reasoning="Strong market with growing demand for specialized robotics talent.",
    score=7.5,
    key_points=["Large underserved market", "Recurring revenue potential"],
    concerns=["Long enterprise sales cycles"],
    agrees_with=[],
    disagrees_with=[],
)

MOCK_VERDICT = FinalVerdict(
    executive_summary="Promising marketplace with strong fundamentals and manageable risks.",
    final_score=7.3,
    recommendation="Invest",
    strengths=["Large TAM", "Clear problem", "Technical feasibility"],
    weaknesses=["B2B sales complexity", "Chicken-and-egg marketplace problem"],
    key_risks=["Incumbent platforms pivoting", "Low initial liquidity"],
    next_steps=["Build waitlist", "Partner with 3 anchor manufacturers", "Hire BD lead"],
    agent_scores={
        "MarketAnalyst": 7.5,
        "TechnicalEvaluator": 7.0,
        "BusinessModelCritic": 7.2,
        "RiskAssessor": 7.3,
    },
    consensus_level="High",
    debate_rounds=1,
)

MOCK_RESULT = EvaluationResult(
    startup_description=STARTUP_DESC,
    verdict=MOCK_VERDICT,
    debate_history=[
        {
            "MarketAnalyst": MOCK_AGENT_RESPONSE,
            "TechnicalEvaluator": MOCK_AGENT_RESPONSE,
            "BusinessModelCritic": MOCK_AGENT_RESPONSE,
            "RiskAssessor": MOCK_AGENT_RESPONSE,
        }
    ],
    elapsed_seconds=12.4,
    early_stopped=True,
)


@pytest.fixture
async def client():
    """Async test client with mocked debate engine."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def mock_engine():
    """Replace the global engine with a mock."""
    import src.api.app as api_module
    mock = MagicMock()
    mock.evaluate_async = AsyncMock(return_value=MOCK_RESULT)
    original = api_module.engine
    api_module.engine = mock
    yield mock
    api_module.engine = original


@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_root_endpoint(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "evaluate" in data


@pytest.mark.asyncio
async def test_evaluate_success(client):
    resp = await client.post("/evaluate", json={"startup_description": STARTUP_DESC})
    assert resp.status_code == 200
    data = resp.json()
    assert "verdict" in data
    assert data["verdict"]["final_score"] == 7.3
    assert data["verdict"]["recommendation"] == "Invest"


@pytest.mark.asyncio
async def test_evaluate_response_structure(client):
    resp = await client.post("/evaluate", json={"startup_description": STARTUP_DESC})
    data = resp.json()
    assert "startup_description" in data
    assert "verdict" in data
    assert "debate_rounds" in data
    assert "elapsed_seconds" in data
    assert "early_stopped" in data
    assert "system_info" in data


@pytest.mark.asyncio
async def test_evaluate_verdict_fields(client):
    resp = await client.post("/evaluate", json={"startup_description": STARTUP_DESC})
    verdict = resp.json()["verdict"]
    assert "executive_summary" in verdict
    assert "final_score" in verdict
    assert "recommendation" in verdict
    assert "strengths" in verdict
    assert "weaknesses" in verdict
    assert "key_risks" in verdict
    assert "next_steps" in verdict
    assert "agent_scores" in verdict
    assert "consensus_level" in verdict


@pytest.mark.asyncio
async def test_evaluate_debate_rounds_structure(client):
    resp = await client.post("/evaluate", json={"startup_description": STARTUP_DESC})
    rounds = resp.json()["debate_rounds"]
    assert len(rounds) == 1
    assert rounds[0]["round_number"] == 0
    assert len(rounds[0]["agents"]) == 4


@pytest.mark.asyncio
async def test_evaluate_too_short_description(client):
    resp = await client.post("/evaluate", json={"startup_description": "short"})
    assert resp.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_evaluate_empty_description(client):
    resp = await client.post("/evaluate", json={"startup_description": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_evaluate_no_body(client):
    resp = await client.post("/evaluate")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_evaluate_engine_none(client):
    import src.api.app as api_module
    api_module.engine = None
    resp = await client.post("/evaluate", json={"startup_description": STARTUP_DESC})
    assert resp.status_code == 503
    api_module.engine = MagicMock()
    api_module.engine.evaluate_async = AsyncMock(return_value=MOCK_RESULT)


@pytest.mark.asyncio
async def test_evaluate_engine_error(client, mock_engine):
    mock_engine.evaluate_async = AsyncMock(side_effect=RuntimeError("LLM timeout"))
    resp = await client.post("/evaluate", json={"startup_description": STARTUP_DESC})
    assert resp.status_code == 500
