"""Debate engine: orchestrates 5-agent parallel debate with dynamic stopping criteria."""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from src.agents.base_agent import AgentResponse
from src.agents.market_analyst import MarketAnalystAgent
from src.agents.technical_evaluator import TechnicalEvaluatorAgent
from src.agents.business_model_critic import BusinessModelCriticAgent
from src.agents.risk_assessor import RiskAssessorAgent
from src.agents.synthesis_judge import SynthesisJudgeAgent, FinalVerdict
from src.core.context_manager import DebateContext

MAX_DEBATE_ROUNDS = 3
CONSENSUS_THRESHOLD = 0.6  # Variance below this = early stop
MAX_WORKERS = 5  # Parallel LLM calls


@dataclass
class EvaluationResult:
    startup_description: str
    verdict: FinalVerdict
    debate_history: list[dict[str, AgentResponse]]
    elapsed_seconds: float
    early_stopped: bool


class DebateEngine:
    """
    Orchestrates a multi-agent debate for startup evaluation.

    Flow:
      1. Round 0: All 4 domain agents assess independently (parallel LLM calls).
      2. Rounds 1-N: Each agent sees peer assessments and may revise (parallel).
      3. Dynamic stop: If score variance < threshold or max rounds reached, stop.
      4. SynthesisJudge renders final verdict.
    """

    def __init__(self) -> None:
        self.agents = [
            MarketAnalystAgent(),
            TechnicalEvaluatorAgent(),
            BusinessModelCriticAgent(),
            RiskAssessorAgent(),
        ]
        self.judge = SynthesisJudgeAgent()
        self._executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    def _run_initial_assessments(
        self, startup: str
    ) -> dict[str, AgentResponse]:
        """Run all initial agent assessments in parallel."""
        futures = {
            agent.name: self._executor.submit(agent.initial_assessment, startup)
            for agent in self.agents
        }
        return {name: future.result() for name, future in futures.items()}

    def _run_debate_round(
        self,
        startup: str,
        current_assessments: dict[str, AgentResponse],
    ) -> dict[str, AgentResponse]:
        """Run one debate round: all agents respond to peers in parallel."""
        futures = {
            agent.name: self._executor.submit(
                agent.debate_response, startup, current_assessments
            )
            for agent in self.agents
        }
        return {name: future.result() for name, future in futures.items()}

    async def evaluate_async(self, startup_description: str) -> EvaluationResult:
        """Async wrapper for evaluate(), runs blocking IO in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self.evaluate, startup_description
        )

    def evaluate(self, startup_description: str) -> EvaluationResult:
        """
        Run full debate evaluation pipeline.

        Returns EvaluationResult with verdict, debate history, and timing.
        """
        start_time = time.time()
        context = DebateContext(startup_description=startup_description)
        debate_history: list[dict[str, AgentResponse]] = []
        early_stopped = False

        # Reset all agents
        for agent in self.agents:
            agent.reset()

        # Round 0: Independent initial assessments (parallel)
        round_0 = self._run_initial_assessments(startup_description)
        debate_history.append(round_0)
        for agent_name, resp in round_0.items():
            context.update(agent_name, resp, round_num=0)

        current = round_0

        # Debate rounds with dynamic stopping
        for round_num in range(1, MAX_DEBATE_ROUNDS + 1):
            if context.consensus_reached(CONSENSUS_THRESHOLD):
                early_stopped = True
                break

            current = self._run_debate_round(startup_description, current)
            debate_history.append(current)
            for agent_name, resp in current.items():
                context.update(agent_name, resp, round_num=round_num)

        # Extract final scores from last round
        final_scores = {
            name: resp.score for name, resp in current.items()
        }

        # Synthesis judge renders final verdict
        verdict = self.judge.synthesize(
            startup_description=startup_description,
            debate_history=debate_history,
            final_scores=final_scores,
            rounds_completed=len(debate_history),
        )

        elapsed = time.time() - start_time

        return EvaluationResult(
            startup_description=startup_description,
            verdict=verdict,
            debate_history=debate_history,
            elapsed_seconds=round(elapsed, 2),
            early_stopped=early_stopped,
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
