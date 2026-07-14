"""
Report generator — formats validation results as JSON, Markdown, or terminal output.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TextIO

from evaluation.metrics import EvaluationMetrics, CaseResult, MetricsCalculator


_TIER_LABELS = {
    1: "Tier 1 — Strong Invest",
    2: "Tier 2 — Invest",
    3: "Tier 3 — Conditional Pass",
    4: "Tier 4 — Pass",
    5: "Tier 5 — Strong Pass",
}

_GRADE_EMOJI = {"A": "✅", "B": "🟡", "C": "🟠", "D": "🔴", "F": "❌"}


class ReportGenerator:
    """Generates human-readable and machine-readable validation reports."""

    def __init__(self) -> None:
        self._calculator = MetricsCalculator()

    def to_json(
        self,
        results: list[CaseResult],
        metrics: EvaluationMetrics,
        tier_analysis: dict[int, dict] | None = None,
    ) -> str:
        scorecard = MetricsCalculator.score_card(metrics)
        passed, failures = MetricsCalculator.passes_ci_threshold(metrics)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_cases": metrics.n_cases,
            "ci_passed": passed,
            "ci_failures": failures,
            "scorecard": scorecard,
            "aggregate_metrics": {
                "accuracy": {
                    "score_mae": round(metrics.score_mae, 4),
                    "score_rmse": round(metrics.score_rmse, 4),
                    "within_range_rate": round(metrics.within_range_rate, 4),
                    "recommendation_accuracy": round(metrics.recommendation_accuracy, 4),
                    "tier_accuracy_strict": round(metrics.tier_accuracy_strict, 4),
                    "tier_accuracy_lenient": round(metrics.tier_accuracy_lenient, 4),
                },
                "agent_behavior": {
                    "avg_inter_agent_std": round(metrics.avg_inter_agent_std, 4),
                    "avg_debate_improvement": round(metrics.avg_debate_improvement, 4),
                    "early_stop_rate": round(metrics.early_stop_rate, 4),
                    "avg_debate_rounds": round(metrics.avg_debate_rounds, 2),
                },
                "efficiency": {
                    "avg_latency_s": round(metrics.avg_latency_s, 2),
                    "p95_latency_s": round(metrics.p95_latency_s, 2),
                },
                "per_agent_scoring": {
                    "mean_by_agent": metrics.mean_score_by_agent,
                    "std_by_agent": metrics.std_score_by_agent,
                },
                "consistency": {
                    "rerun_score_std": metrics.rerun_score_std,
                    "rerun_recommendation_consistency": metrics.rerun_recommendation_consistency,
                },
                "per_tier": {
                    str(tier): {
                        "tier_accuracy": round(acc, 4),
                        "recommendation_accuracy": round(
                            metrics.recommendation_accuracy_by_tier.get(tier, 0.0), 4
                        ),
                    }
                    for tier, acc in metrics.tier_accuracy_by_tier.items()
                },
            },
            "case_results": [
                {
                    "id": r.ground_truth.id,
                    "name": r.ground_truth.name,
                    "expected_tier": r.ground_truth.expected_tier,
                    "expected_recommendation": r.ground_truth.expected_recommendation,
                    "expected_score_range": [
                        r.ground_truth.expected_score_min,
                        r.ground_truth.expected_score_max,
                    ],
                    "actual_score": round(r.final_score, 2),
                    "actual_recommendation": r.recommendation,
                    "predicted_tier": r.predicted_tier,
                    "within_range": r.within_expected_range,
                    "recommendation_correct": r.recommendation_correct,
                    "tier_error": r.tier_error,
                    "score_error": round(r.score_error, 4),
                    "inter_agent_std": round(r.inter_agent_std, 4),
                    "debate_score_delta": round(r.debate_score_delta, 4),
                    "agent_scores": {k: round(v, 2) for k, v in r.agent_scores.items()},
                    "round_0_scores": {k: round(v, 2) for k, v in r.round_0_scores.items()},
                    "debate_rounds": r.debate_rounds,
                    "early_stopped": r.early_stopped,
                    "elapsed_s": round(r.elapsed_seconds, 2),
                }
                for r in results
            ],
            "tier_analysis": tier_analysis or {},
        }
        return json.dumps(report, indent=2)

    def to_markdown(
        self,
        results: list[CaseResult],
        metrics: EvaluationMetrics,
    ) -> str:
        scorecard = MetricsCalculator.score_card(metrics)
        passed, failures = MetricsCalculator.passes_ci_threshold(metrics)
        ci_badge = "✅ PASS" if passed else "❌ FAIL"

        lines = [
            "# Multi-Agent Startup Evaluator — Validation Report",
            f"\n**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Cases evaluated:** {metrics.n_cases}",
            f"**CI Gate:** {ci_badge}",
        ]

        if failures:
            lines.append("\n**CI Failures:**")
            for f in failures:
                lines.append(f"- {f}")

        # Scorecard
        lines += [
            "\n## Scorecard",
            "| Metric | Grade |",
            "|--------|-------|",
        ]
        for metric, grade in scorecard.items():
            emoji = _GRADE_EMOJI.get(grade, "")
            label = metric.replace("_", " ").title()
            lines.append(f"| {label} | {emoji} {grade} |")

        # Aggregate accuracy
        lines += [
            "\n## Accuracy Metrics",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Score MAE | {metrics.score_mae:.3f} |",
            f"| Score RMSE | {metrics.score_rmse:.3f} |",
            f"| Within Expected Range | {metrics.within_range_rate:.0%} |",
            f"| Recommendation Accuracy | {metrics.recommendation_accuracy:.0%} |",
            f"| Tier Accuracy (strict) | {metrics.tier_accuracy_strict:.0%} |",
            f"| Tier Accuracy (±1 tier) | {metrics.tier_accuracy_lenient:.0%} |",
        ]

        # Agent behavior
        lines += [
            "\n## Agent Behavior",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Avg Inter-Agent Std Dev | {metrics.avg_inter_agent_std:.3f} |",
            f"| Avg Score Change (debate) | {metrics.avg_debate_improvement:.3f} |",
            f"| Early Stop Rate | {metrics.early_stop_rate:.0%} |",
            f"| Avg Debate Rounds | {metrics.avg_debate_rounds:.1f} |",
        ]

        # Efficiency
        lines += [
            "\n## Efficiency",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Avg Latency | {metrics.avg_latency_s:.1f}s |",
            f"| P95 Latency | {metrics.p95_latency_s:.1f}s |",
        ]

        # Per-agent scores
        if metrics.mean_score_by_agent:
            lines += [
                "\n## Per-Agent Scoring (avg ± std across all cases)",
                "| Agent | Mean Score | Std Dev |",
                "|-------|-----------|---------|",
            ]
            for agent, mean in sorted(metrics.mean_score_by_agent.items()):
                std = metrics.std_score_by_agent.get(agent, 0.0)
                lines.append(f"| {agent} | {mean:.2f} | ±{std:.2f} |")

        # Per-tier accuracy
        if metrics.tier_accuracy_by_tier:
            lines += [
                "\n## Per-Tier Accuracy",
                "| Tier | Tier Acc | Rec Acc |",
                "|------|---------|---------|",
            ]
            for tier in range(1, 6):
                if tier in metrics.tier_accuracy_by_tier:
                    t_acc = metrics.tier_accuracy_by_tier[tier]
                    r_acc = metrics.recommendation_accuracy_by_tier.get(tier, 0.0)
                    label = _TIER_LABELS[tier]
                    lines.append(f"| {label} | {t_acc:.0%} | {r_acc:.0%} |")

        # Case-by-case results
        lines += [
            "\n## Case-by-Case Results",
            "| ID | Name | Expected | Predicted | Score | In Range | Tier Err |",
            "|----|------|---------|-----------|-------|----------|---------|",
        ]
        for r in sorted(results, key=lambda x: x.ground_truth.expected_tier):
            ok = "✅" if r.within_expected_range else "❌"
            lines.append(
                f"| {r.ground_truth.id} | {r.ground_truth.name[:30]} "
                f"| {r.ground_truth.expected_recommendation} "
                f"| {r.recommendation} "
                f"| {r.final_score:.1f} "
                f"| {ok} "
                f"| {r.tier_error} |"
            )

        return "\n".join(lines)

    def print_summary(
        self,
        results: list[CaseResult],
        metrics: EvaluationMetrics,
        out: TextIO | None = None,
    ) -> None:
        import sys
        out = out or sys.stdout
        passed, failures = MetricsCalculator.passes_ci_threshold(metrics)
        scorecard = MetricsCalculator.score_card(metrics)

        banner = "✅ CI PASS" if passed else "❌ CI FAIL"
        print(f"\n{'='*60}", file=out)
        print(f"  Validation Report  |  {metrics.n_cases} cases  |  {banner}", file=out)
        print(f"{'='*60}", file=out)

        print(f"\n  ACCURACY", file=out)
        print(f"    Score MAE:               {metrics.score_mae:.3f}", file=out)
        print(f"    Score RMSE:              {metrics.score_rmse:.3f}", file=out)
        print(f"    Within expected range:   {metrics.within_range_rate:.0%}", file=out)
        print(f"    Recommendation accuracy: {metrics.recommendation_accuracy:.0%}", file=out)
        print(f"    Tier accuracy (strict):  {metrics.tier_accuracy_strict:.0%}", file=out)
        print(f"    Tier accuracy (±1 tier): {metrics.tier_accuracy_lenient:.0%}", file=out)

        print(f"\n  AGENT BEHAVIOR", file=out)
        print(f"    Avg inter-agent std dev: {metrics.avg_inter_agent_std:.3f}", file=out)
        print(f"    Avg debate improvement:  {metrics.avg_debate_improvement:.3f}", file=out)
        print(f"    Early stop rate:         {metrics.early_stop_rate:.0%}", file=out)
        print(f"    Avg debate rounds:       {metrics.avg_debate_rounds:.1f}", file=out)

        print(f"\n  EFFICIENCY", file=out)
        print(f"    Avg latency:             {metrics.avg_latency_s:.1f}s", file=out)
        print(f"    P95 latency:             {metrics.p95_latency_s:.1f}s", file=out)

        print(f"\n  SCORECARD", file=out)
        for metric, grade in scorecard.items():
            emoji = _GRADE_EMOJI.get(grade, "")
            label = metric.replace("_", " ").title()
            print(f"    {label:<35} {emoji} {grade}", file=out)

        if failures:
            print(f"\n  CI FAILURES", file=out)
            for f in failures:
                print(f"    ✗ {f}", file=out)

        print(f"\n{'='*60}\n", file=out)
