"""FastAPI application for the Multi-Agent Startup Evaluation System."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.core.debate_engine import DebateEngine, EvaluationResult
from src.agents.base_agent import AgentResponse
from src.agents.synthesis_judge import FinalVerdict


# ── Lifespan ────────────────────────────────────────────────────────────────

engine: DebateEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global engine
    engine = DebateEngine()
    yield
    if engine:
        engine.shutdown()


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Agent Startup Evaluator",
    description=(
        "A 5-agent debate-style AI system that evaluates startup concepts using parallel "
        "LLM calls, dynamic stopping criteria, and context-aware reasoning."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    startup_description: str = Field(
        ...,
        min_length=20,
        max_length=2000,
        description="Description of the startup concept to evaluate.",
        examples=["An AI-powered platform that connects local farmers with urban consumers via subscription boxes, using ML to optimize delivery routes and reduce food waste."],
    )


class AgentRoundSummary(BaseModel):
    agent_name: str
    score: float
    reasoning_snippet: str
    key_points: list[str]
    concerns: list[str]


class DebateRoundSummary(BaseModel):
    round_number: int
    agents: list[AgentRoundSummary]


class EvaluateResponse(BaseModel):
    startup_description: str
    verdict: FinalVerdict
    debate_rounds: list[DebateRoundSummary]
    elapsed_seconds: float
    early_stopped: bool
    system_info: dict[str, str]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "Multi-Agent Startup Evaluator",
        "version": "1.0.0",
    }


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_startup(request: EvaluateRequest) -> EvaluateResponse:
    """
    Evaluate a startup concept using a 5-agent debate.

    Agents: MarketAnalyst, TechnicalEvaluator, BusinessModelCritic, RiskAssessor,
    and a SynthesisJudge that renders the final verdict.

    LLM calls are parallelized per round. Dynamic stopping criteria halt the debate
    early when agents reach consensus (score variance < threshold).
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")

    try:
        result: EvaluationResult = await engine.evaluate_async(request.startup_description)
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(exc)}",
        ) from exc

    # Format debate history
    debate_rounds = [
        DebateRoundSummary(
            round_number=i,
            agents=[
                AgentRoundSummary(
                    agent_name=name,
                    score=resp.score,
                    reasoning_snippet=resp.reasoning[:200],
                    key_points=resp.key_points,
                    concerns=resp.concerns,
                )
                for name, resp in round_data.items()
            ],
        )
        for i, round_data in enumerate(result.debate_history)
    ]

    return EvaluateResponse(
        startup_description=result.startup_description,
        verdict=result.verdict,
        debate_rounds=debate_rounds,
        elapsed_seconds=result.elapsed_seconds,
        early_stopped=result.early_stopped,
        system_info={
            "model": "gpt-4o",
            "api_base": "https://api.ai.it.ufl.edu",
            "agents": "MarketAnalyst, TechnicalEvaluator, BusinessModelCritic, RiskAssessor, SynthesisJudge",
        },
    )


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "Multi-Agent Startup Evaluator",
        "docs": "/docs",
        "health": "/health",
        "evaluate": "POST /evaluate",
    }
