# Multi-Agent Collaborative Reasoning System for Startup Evaluation

A production-grade **5-agent debate-style AI workflow** that evaluates startup concepts using parallel LLM calls, dynamic stopping criteria, and context-aware reasoning — all exposed via a REST API.

**Live API:** https://api-production-c3ce.up.railway.app
**Docs:** https://api-production-c3ce.up.railway.app/docs

---

## Architecture

```
Startup Description
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                   Debate Engine                      │
│                                                     │
│  Round 0 (parallel)     Rounds 1-N (parallel)       │
│  ┌─────────────────┐    ┌─────────────────────────┐ │
│  │ MarketAnalyst   │    │ Each agent sees peers'  │ │
│  │ TechEvaluator   │───▶│ scores & can update     │ │
│  │ BizModelCritic  │    │ their own assessment    │ │
│  │ RiskAssessor    │    └──────────┬──────────────┘ │
│  └─────────────────┘               │                │
│                         Dynamic Stop:               │
│                         variance < 0.6 → early stop │
└───────────────────────────┬─────────────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ SynthesisJudge  │
                   │ Final Verdict   │
                   └─────────────────┘
```

### The 5 Agents

| Agent | Role | Scoring Criteria |
|-------|------|-----------------|
| **MarketAnalyst** | Market size, TAM/SAM, competition | Market size (30%) · Differentiation (30%) · Timing (20%) · Accessibility (20%) |
| **TechnicalEvaluator** | Tech feasibility, innovation, scalability | Novelty (25%) · Feasibility (30%) · Scalability (25%) · IP (20%) |
| **BusinessModelCritic** | Revenue model, unit economics, GTM | Revenue clarity (30%) · Unit economics (30%) · GTM (20%) · Scalability (20%) |
| **RiskAssessor** | Market, technical, execution, competitive risk | Risk-adjusted composite score |
| **SynthesisJudge** | Final aggregation and verdict | Weighted composite (Risk: 30% · Market: 25% · Biz: 25% · Tech: 20%) |

### Key Engineering Features

- **Parallel LLM Calls** — `ThreadPoolExecutor` with 5 workers runs all agent calls concurrently per round, reducing latency from O(N×rounds) to O(rounds).
- **Dynamic Stopping Criteria** — Debate terminates early when score variance across agents falls below threshold (default: 0.6), saving unnecessary LLM calls.
- **Context Management** — Each agent tracks its own `turn_history`. When token estimate exceeds 2000 tokens, history is summarized via LLM and replaced with a compact context block, maintaining reasoning coherence.
- **Structured Outputs** — Agents return typed `AgentResponse` (Pydantic) with `score`, `reasoning`, `key_points`, `concerns`, `agrees_with`, `disagrees_with` — enabling programmatic debate analysis.

---

## API Reference

### `POST /evaluate`

Evaluate a startup concept through the 5-agent debate.

**Request:**
```json
{
  "startup_description": "An AI-powered platform connecting local farmers with urban consumers via subscription boxes, using ML to optimize delivery routes and reduce food waste."
}
```

**Response:**
```json
{
  "startup_description": "...",
  "verdict": {
    "executive_summary": "...",
    "final_score": 7.3,
    "recommendation": "Invest",
    "strengths": ["Large underserved market", "..."],
    "weaknesses": ["Long enterprise sales cycles", "..."],
    "key_risks": ["Regulatory uncertainty", "..."],
    "next_steps": ["Build MVP", "..."],
    "agent_scores": {
      "MarketAnalyst": 7.5,
      "TechnicalEvaluator": 7.0,
      "BusinessModelCritic": 7.2,
      "RiskAssessor": 7.3
    },
    "consensus_level": "High",
    "debate_rounds": 1
  },
  "debate_rounds": [
    {
      "round_number": 0,
      "agents": [
        {
          "agent_name": "MarketAnalyst",
          "score": 7.5,
          "reasoning_snippet": "...",
          "key_points": ["..."],
          "concerns": ["..."]
        }
      ]
    }
  ],
  "elapsed_seconds": 12.4,
  "early_stopped": true,
  "system_info": { "model": "gpt-4o", "agents": "..." }
}
```

**Recommendation scale:**
- `Strong Invest` — Score ≥ 8.0
- `Invest` — Score ≥ 6.5
- `Conditional Pass` — Score ≥ 5.0
- `Pass` — Score ≥ 3.5
- `Strong Pass` — Score < 3.5

### `GET /health`
Returns service health status.

### `GET /docs`
Interactive Swagger UI for exploring the API.

---

## Quick Start

### Prerequisites
- Python 3.11+
- University LitLLM API key (`navigator_api`)

### Setup

```bash
git clone https://github.com/AbhiramReddy59/multi-agent-startup-eval.git
cd multi-agent-startup-eval

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and set navigator_api=your_key_here

python main.py
```

API available at `http://localhost:8000` · Docs at `http://localhost:8000/docs`

### Docker

```bash
docker build -t startup-eval .
docker run -e navigator_api=your_key -p 8000:8000 startup-eval
```

---

## Running Tests

```bash
pytest -v
```

Tests use mocked LLM calls — no API key needed.

---

## Project Structure

```
├── src/
│   ├── agents/
│   │   ├── base_agent.py          # BaseAgent with context management & LLM calls
│   │   ├── market_analyst.py      # Market opportunity evaluator
│   │   ├── technical_evaluator.py # Technical feasibility assessor
│   │   ├── business_model_critic.py # Revenue & GTM critic
│   │   ├── risk_assessor.py       # Risk-adjusted scoring
│   │   └── synthesis_judge.py     # Final verdict aggregator
│   ├── core/
│   │   ├── debate_engine.py       # Parallel debate orchestration
│   │   └── context_manager.py     # Token-aware context tracking
│   └── api/
│       └── app.py                 # FastAPI REST interface
├── tests/
│   ├── test_agents.py             # Agent unit tests (mocked)
│   ├── test_debate_engine.py      # Engine & context manager tests
│   └── test_api.py                # API integration tests
├── main.py                        # Entry point
├── Dockerfile
└── requirements.txt
```

---

## Resume Context

This project demonstrates:
- **Multi-agent orchestration** — 5 specialized agents with distinct evaluation lenses debate startup viability
- **Parallel LLM execution** — concurrent API calls reduce evaluation latency to under 45s for complex startups
- **Dynamic stopping** — variance-based consensus detection eliminates redundant debate rounds
- **Context-aware reasoning** — summarization strategy keeps agents within LLM token limits while maintaining coherence across multi-turn debates
- **Production-grade REST API** — typed Pydantic schemas, proper error handling, async endpoints
