"""Base agent class for the multi-agent startup evaluation system."""
import os
import json
import time
from typing import Any
from openai import OpenAI
from pydantic import BaseModel

# Rough token estimate: 1 token ≈ 4 characters
CHARS_PER_TOKEN = 4
MAX_HISTORY_TOKENS = 2000  # Summarize when history exceeds this


class AgentResponse(BaseModel):
    reasoning: str
    score: float  # 1-10 domain-specific score
    key_points: list[str]
    concerns: list[str]
    agrees_with: list[str]  # Agent names this agent agrees with
    disagrees_with: list[str]  # Agent names this agent disagrees with


class BaseAgent:
    """Base class for all evaluation agents."""

    name: str = "BaseAgent"
    role: str = "Evaluator"
    system_prompt: str = "You are an expert evaluator."

    def __init__(self) -> None:
        api_key = os.getenv("navigator_api")
        if not api_key:
            raise EnvironmentError("navigator_api environment variable not set.")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.ai.it.ufl.edu",
        )
        self.model = "gpt-4o"
        self.turn_history: list[dict[str, str]] = []
        self._summarized_context: str = ""

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // CHARS_PER_TOKEN

    def _history_token_count(self) -> int:
        total = sum(
            self._estimate_tokens(m["content"]) for m in self.turn_history
        )
        if self._summarized_context:
            total += self._estimate_tokens(self._summarized_context)
        return total

    def _summarize_history(self) -> None:
        """Summarize turn history to stay within token limits while maintaining coherence."""
        if not self.turn_history:
            return

        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self.turn_history
        )
        summary_prompt = (
            f"Summarize the following debate history for a {self.role} agent named {self.name}. "
            "Preserve key arguments, scores, agreements, and disagreements. Be concise.\n\n"
            f"{history_text}"
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a concise summarizer for AI agent reasoning history."},
                    {"role": "user", "content": summary_prompt},
                ],
                max_tokens=400,
                temperature=0.3,
            )
            summary = resp.choices[0].message.content or ""
            self._summarized_context = (
                f"[PRIOR CONTEXT SUMMARY]:\n{summary}\n[END SUMMARY]"
            )
            self.turn_history = []  # Clear history after summarizing
        except Exception:
            # If summarization fails, keep last 2 turns
            self.turn_history = self.turn_history[-2:]

    def _build_messages(self, user_content: str) -> list[dict[str, str]]:
        """Build the message list, injecting summarized context if present."""
        system = self.system_prompt
        if self._summarized_context:
            system += f"\n\n{self._summarized_context}"

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend(self.turn_history)
        messages.append({"role": "user", "content": user_content})
        return messages

    def _call_llm(self, user_content: str) -> AgentResponse:
        """Call the university LLM API and parse structured response."""
        if self._history_token_count() > MAX_HISTORY_TOKENS:
            self._summarize_history()

        messages = self._build_messages(user_content)

        response_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "agent_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": "string"},
                        "score": {"type": "number"},
                        "key_points": {"type": "array", "items": {"type": "string"}},
                        "concerns": {"type": "array", "items": {"type": "string"}},
                        "agrees_with": {"type": "array", "items": {"type": "string"}},
                        "disagrees_with": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["reasoning", "score", "key_points", "concerns", "agrees_with", "disagrees_with"],
                    "additionalProperties": False,
                },
            },
        }

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.7,
                max_tokens=800,
                response_format=response_schema,  # type: ignore[arg-type]
            )
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)
        except Exception:
            # Fallback: plain completion, extract JSON manually
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.7,
                max_tokens=800,
            )
            raw = resp.choices[0].message.content or ""
            data = self._extract_json_fallback(raw)

        agent_resp = AgentResponse(**data)
        # Clamp score to [1, 10]
        agent_resp.score = max(1.0, min(10.0, agent_resp.score))

        # Update turn history
        self.turn_history.append({"role": "user", "content": user_content})
        self.turn_history.append({"role": "assistant", "content": json.dumps(data)})

        return agent_resp

    def _extract_json_fallback(self, raw: str) -> dict[str, Any]:
        """Extract JSON from raw LLM text as fallback."""
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
        return {
            "reasoning": raw[:500] if raw else "No response.",
            "score": 5.0,
            "key_points": [],
            "concerns": [],
            "agrees_with": [],
            "disagrees_with": [],
        }

    def initial_assessment(self, startup_description: str) -> AgentResponse:
        """Provide initial independent assessment of the startup."""
        prompt = (
            f"You are evaluating the following startup concept as a {self.role}.\n\n"
            f"STARTUP CONCEPT:\n{startup_description}\n\n"
            f"Provide your initial assessment from the perspective of a {self.role}. "
            "Score the startup from 1-10 within your domain. "
            "Return a JSON response with: reasoning, score (1-10), key_points (list), "
            "concerns (list), agrees_with (empty list for initial), disagrees_with (empty list for initial)."
        )
        return self._call_llm(prompt)

    def debate_response(
        self,
        startup_description: str,
        other_assessments: dict[str, AgentResponse],
    ) -> AgentResponse:
        """Respond to other agents' assessments in debate round."""
        assessments_text = "\n\n".join(
            f"[{agent_name}] Score: {resp.score}/10\n"
            f"Key Points: {', '.join(resp.key_points)}\n"
            f"Concerns: {', '.join(resp.concerns)}\n"
            f"Reasoning: {resp.reasoning[:300]}"
            for agent_name, resp in other_assessments.items()
            if agent_name != self.name
        )

        prompt = (
            f"STARTUP CONCEPT:\n{startup_description}\n\n"
            f"OTHER AGENTS' ASSESSMENTS:\n{assessments_text}\n\n"
            f"As {self.name} ({self.role}), review the other agents' assessments. "
            "Update your score if new arguments are compelling. "
            "Explicitly agree or disagree with specific agents by name. "
            "Return JSON with: reasoning, score (1-10), key_points, concerns, "
            "agrees_with (list of agent names you agree with), disagrees_with (list of agent names you disagree with)."
        )
        return self._call_llm(prompt)

    def reset(self) -> None:
        """Reset agent state for a new evaluation."""
        self.turn_history = []
        self._summarized_context = ""
