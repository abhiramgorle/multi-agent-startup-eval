"""Synthesis Judge Agent — aggregates debate, resolves disagreements, and delivers final verdict."""
import json
import os
from openai import OpenAI
from pydantic import BaseModel
from .base_agent import AgentResponse


class FinalVerdict(BaseModel):
    executive_summary: str
    final_score: float  # Weighted composite 1-10
    recommendation: str  # "Strong Invest", "Invest", "Pass", "Strong Pass"
    strengths: list[str]
    weaknesses: list[str]
    key_risks: list[str]
    next_steps: list[str]
    agent_scores: dict[str, float]
    consensus_level: str  # "High", "Medium", "Low"
    debate_rounds: int


class SynthesisJudgeAgent:
    """Final synthesis agent that aggregates all debate rounds and renders a verdict."""

    name = "SynthesisJudge"
    role = "Synthesis Judge"

    # Weights for each agent's domain score in the final composite
    AGENT_WEIGHTS: dict[str, float] = {
        "MarketAnalyst": 0.25,
        "TechnicalEvaluator": 0.20,
        "BusinessModelCritic": 0.25,
        "RiskAssessor": 0.30,
    }

    def __init__(self) -> None:
        api_key = os.getenv("navigator_api")
        if not api_key:
            raise EnvironmentError("navigator_api environment variable not set.")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.ai.it.ufl.edu",
        )
        self.model = "gpt-4o"

    def synthesize(
        self,
        startup_description: str,
        debate_history: list[dict[str, AgentResponse]],
        final_scores: dict[str, float],
        rounds_completed: int,
    ) -> FinalVerdict:
        """Synthesize all debate rounds into a final verdict."""
        # Build debate summary for the judge
        debate_summary = self._format_debate_history(debate_history)

        # Compute weighted final score
        weighted_score = sum(
            final_scores.get(agent, 5.0) * weight
            for agent, weight in self.AGENT_WEIGHTS.items()
        )
        weighted_score = max(1.0, min(10.0, weighted_score))

        # Determine consensus level from score variance
        scores = list(final_scores.values())
        if len(scores) > 1:
            avg = sum(scores) / len(scores)
            variance = sum((s - avg) ** 2 for s in scores) / len(scores)
            std_dev = variance ** 0.5
            if std_dev < 0.8:
                consensus = "High"
            elif std_dev < 1.5:
                consensus = "Medium"
            else:
                consensus = "Low"
        else:
            consensus = "High"

        # Determine recommendation
        recommendation = self._score_to_recommendation(weighted_score)

        synthesis_prompt = (
            f"STARTUP CONCEPT:\n{startup_description}\n\n"
            f"MULTI-AGENT DEBATE SUMMARY ({rounds_completed} rounds):\n{debate_summary}\n\n"
            f"FINAL DOMAIN SCORES: {json.dumps(final_scores, indent=2)}\n"
            f"WEIGHTED COMPOSITE SCORE: {weighted_score:.1f}/10\n"
            f"CONSENSUS LEVEL: {consensus}\n\n"
            "As the Synthesis Judge, provide a comprehensive final verdict. "
            "Identify the most compelling arguments from the debate, resolve key disagreements, "
            "and provide actionable next steps. "
            "Return JSON with: executive_summary, strengths (list), weaknesses (list), "
            "key_risks (list), next_steps (list)."
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an experienced venture capital Investment Committee Chair. "
                            "You synthesize multi-agent debate outcomes into clear, actionable investment verdicts. "
                            "Be decisive, balanced, and specific. Always respond with valid JSON."
                        ),
                    },
                    {"role": "user", "content": synthesis_prompt},
                ],
                temperature=0.5,
                max_tokens=1000,
            )
            raw = resp.choices[0].message.content or "{}"
            start = raw.find("{")
            end = raw.rfind("}") + 1
            data = json.loads(raw[start:end]) if start >= 0 else {}
        except Exception:
            data = {}

        return FinalVerdict(
            executive_summary=data.get(
                "executive_summary",
                f"Composite evaluation of startup yielded a score of {weighted_score:.1f}/10.",
            ),
            final_score=round(weighted_score, 2),
            recommendation=recommendation,
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            key_risks=data.get("key_risks", []),
            next_steps=data.get("next_steps", []),
            agent_scores=final_scores,
            consensus_level=consensus,
            debate_rounds=rounds_completed,
        )

    def _format_debate_history(
        self, debate_history: list[dict[str, AgentResponse]]
    ) -> str:
        lines = []
        for i, round_data in enumerate(debate_history):
            lines.append(f"\n--- Round {i + 1} ---")
            for agent_name, resp in round_data.items():
                lines.append(
                    f"{agent_name} (score={resp.score}/10): {resp.reasoning[:200]}"
                )
        return "\n".join(lines)

    def _score_to_recommendation(self, score: float) -> str:
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
