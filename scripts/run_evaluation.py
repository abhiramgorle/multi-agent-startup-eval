"""
CLI script to run the full validation suite against the ground-truth dataset.

Usage:
    python scripts/run_evaluation.py [OPTIONS]

Options:
    --output-json PATH      Write JSON report to file (default: evaluation_report.json)
    --output-md PATH        Write Markdown report to file (default: evaluation_report.md)
    --subset TIER           Only run cases for a specific tier (1-5)
    --no-parallel           Disable parallel evaluation (useful for debugging)
    --reruns N              Run each case N times for consistency testing (default: 1)
    --quiet                 Suppress per-case progress output
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from evaluation.ground_truths import GROUND_TRUTH_DATASET
from evaluation.validator import ValidationRunner
from evaluation.report import ReportGenerator
from src.core.debate_engine import DebateEngine


def main() -> int:
    parser = argparse.ArgumentParser(description="Run startup evaluator validation suite")
    parser.add_argument("--output-json", default="evaluation_report.json")
    parser.add_argument("--output-md", default="evaluation_report.md")
    parser.add_argument("--subset", type=int, choices=[1, 2, 3, 4, 5], default=None)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--reruns", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    dataset = GROUND_TRUTH_DATASET
    if args.subset is not None:
        dataset = [gt for gt in dataset if gt.expected_tier == args.subset]
        print(f"[INFO] Running {len(dataset)} cases for Tier {args.subset}")
    else:
        print(f"[INFO] Running full dataset: {len(dataset)} cases")

    def progress(case_id: str, status: str) -> None:
        if not args.quiet:
            print(f"  [{case_id}] {status}")

    runner = ValidationRunner(
        engine_factory=DebateEngine,
        max_workers=3,
        progress_callback=progress,
    )

    print("[INFO] Starting evaluation...\n")
    results, metrics = runner.run(
        dataset=dataset,
        parallel=not args.no_parallel,
    )

    # Consistency check if reruns requested
    if args.reruns > 1:
        print(f"\n[INFO] Running consistency check ({args.reruns} reruns per case)...")
        score_std, rec_consistency = runner.run_consistency_check(
            dataset=dataset, n_reruns=args.reruns
        )
        metrics.rerun_score_std = score_std
        metrics.rerun_recommendation_consistency = rec_consistency
        print(f"  Score std across reruns:       {score_std:.3f}")
        print(f"  Recommendation consistency:    {rec_consistency:.0%}")

    tier_analysis = runner.run_tier_analysis(results)

    reporter = ReportGenerator()
    reporter.print_summary(results, metrics)

    # Write JSON
    json_out = reporter.to_json(results, metrics, tier_analysis)
    with open(args.output_json, "w") as f:
        f.write(json_out)
    print(f"[INFO] JSON report written to {args.output_json}")

    # Write Markdown
    md_out = reporter.to_markdown(results, metrics)
    with open(args.output_md, "w") as f:
        f.write(md_out)
    print(f"[INFO] Markdown report written to {args.output_md}")

    # CI gate
    from evaluation.metrics import MetricsCalculator
    passed, failures = MetricsCalculator.passes_ci_threshold(metrics)
    if not passed:
        print(f"\n[FAIL] CI gate failed:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"\n[PASS] All CI thresholds met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
