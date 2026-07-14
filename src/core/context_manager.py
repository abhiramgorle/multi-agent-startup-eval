"""Context manager for maintaining agent reasoning coherence within token limits."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.base_agent import AgentResponse

CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOKENS = 3000


@dataclass
class AgentContext:
    """Tracks the full context for a single agent across debate rounds."""
    agent_name: str
    responses: list["AgentResponse"] = field(default_factory=list)
    round_indices: list[int] = field(default_factory=list)

    def add_response(self, response: "AgentResponse", round_num: int) -> None:
        self.responses.append(response)
        self.round_indices.append(round_num)

    def latest_response(self) -> "AgentResponse | None":
        return self.responses[-1] if self.responses else None

    def score_trajectory(self) -> list[float]:
        return [r.score for r in self.responses]

    def token_estimate(self) -> int:
        total = sum(
            len(r.reasoning) + len(" ".join(r.key_points)) + len(" ".join(r.concerns))
            for r in self.responses
        )
        return total // CHARS_PER_TOKEN


@dataclass
class DebateContext:
    """Aggregates context across all agents and rounds."""
    startup_description: str
    agent_contexts: dict[str, AgentContext] = field(default_factory=dict)
    max_tokens: int = DEFAULT_MAX_TOKENS

    def update(self, agent_name: str, response: "AgentResponse", round_num: int) -> None:
        if agent_name not in self.agent_contexts:
            self.agent_contexts[agent_name] = AgentContext(agent_name=agent_name)
        self.agent_contexts[agent_name].add_response(response, round_num)

    def get_latest_responses(self) -> dict[str, "AgentResponse"]:
        return {
            name: ctx.latest_response()
            for name, ctx in self.agent_contexts.items()
            if ctx.latest_response() is not None
        }

    def get_scores_by_round(self) -> dict[int, dict[str, float]]:
        round_scores: dict[int, dict[str, float]] = {}
        for name, ctx in self.agent_contexts.items():
            for resp, rnd in zip(ctx.responses, ctx.round_indices):
                round_scores.setdefault(rnd, {})[name] = resp.score
        return round_scores

    def total_token_estimate(self) -> int:
        return sum(ctx.token_estimate() for ctx in self.agent_contexts.values())

    def needs_compression(self) -> bool:
        return self.total_token_estimate() > self.max_tokens

    def score_variance(self) -> float:
        """Compute variance of latest scores across agents."""
        latest = self.get_latest_responses()
        if len(latest) < 2:
            return 0.0
        scores = [r.score for r in latest.values()]
        avg = sum(scores) / len(scores)
        return sum((s - avg) ** 2 for s in scores) / len(scores)

    def consensus_reached(self, threshold: float = 0.5) -> bool:
        """Return True if score variance is below threshold (agents converging)."""
        return self.score_variance() < threshold
