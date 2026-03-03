#!/usr/bin/env python3
"""Run all EthicAgent experiments.

Executes the full experimental suite using the **real** pipeline:
1. Scenario evaluation across all 4 domains (real orchestrator)
2. Baseline comparisons (real evaluators)
3. Ablation study (real component-disable experiments)
4. Statistical significance testing (real math on real data)
5. Report generation (LaTeX, Markdown, HTML, JSON)

Usage:
    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --use-llm --output-dir results/
    python scripts/run_all_experiments.py --domains healthcare finance
    python scripts/run_all_experiments.py --skip-ablation --skip-baselines
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ethicagent.evaluation import ReportGenerator, StatisticalAnalyzer  # noqa: E402
from ethicagent.evaluation.ablation import ABLATION_VARIANTS, AblationStudy  # noqa: E402
from ethicagent.evaluation.baselines import get_all_baselines  # noqa: E402
from ethicagent.evaluation.metrics import verdict_accuracy  # noqa: E402
from ethicagent.scenarios import (  # noqa: E402
    DisasterResponseScenario,
    HealthcareTriageScenario,
    HiringDecisionScenario,
    LoanApprovalScenario,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCENARIO_CLASSES = {
    "healthcare": HealthcareTriageScenario,
    "finance": LoanApprovalScenario,
    "hiring": HiringDecisionScenario,
    "disaster": DisasterResponseScenario,
}

# ── Shared orchestrator ──────────────────────────────────────────
_ORCHESTRATOR: Any = None


def _get_orchestrator(*, use_llm: bool = False) -> Any:
    """Lazy-create a shared EthicAgentOrchestrator."""
    global _ORCHESTRATOR  # noqa: PLW0603
    if _ORCHESTRATOR is None:
        from ethicagent.orchestrator import EthicAgentOrchestrator

        _ORCHESTRATOR = EthicAgentOrchestrator(use_llm=use_llm)
    return _ORCHESTRATOR


def _normalize_verdict(verdict: str) -> str:
    """Normalize ``auto_approve`` -> ``approve`` for scenario matching."""
    return verdict.lower().replace("auto_", "")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete EthicAgent experimental suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        default=False,
        help="Use LLM-backed pipeline (requires OPENAI_API_KEY). Default: False",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output files. Default: results/",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=list(SCENARIO_CLASSES.keys()),
        default=list(SCENARIO_CLASSES.keys()),
        help="Domains to evaluate. Default: all",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip baseline comparisons",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip ablation study",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip statistical analysis",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["json", "latex", "markdown", "html"],
        default=["json", "latex", "markdown"],
        help="Report formats to generate. Default: json latex markdown",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# Phase 1 — Scenario evaluation (real orchestrator)
# ═══════════════════════════════════════════════════════════════


def run_scenario_evaluation(domains: list[str]) -> dict:
    """Evaluate all scenarios using the real orchestrator pipeline."""
    logger.info("=" * 60)
    logger.info("PHASE 1: SCENARIO EVALUATION (real pipeline)")
    logger.info("=" * 60)

    orch = _get_orchestrator()
    all_results: dict[str, Any] = {}

    for domain in domains:
        logger.info("  Evaluating %s...", domain)
        scenario_cls = SCENARIO_CLASSES[domain]
        scenario = scenario_cls()
        cases = scenario.get_cases()

        domain_results: list[dict[str, Any]] = []
        for case in cases:
            result = orch.run(task=case.task, domain=domain)
            d = result.to_dict() if hasattr(result, "to_dict") else dict(result)

            verdict = _normalize_verdict(d.get("verdict", "unknown"))
            eds = float(d.get("eds_score", 0.0))

            domain_results.append(
                {
                    "case_id": case.case_id,
                    "task": case.task,
                    "expected_verdict": case.expected_verdict,
                    "predicted_verdict": verdict,
                    "actual_verdict": verdict,
                    "expected_eds_range": case.expected_eds_range,
                    "predicted_eds": eds,
                    "actual_eds": eds,
                    "match": verdict == case.expected_verdict,
                    "verdict_match": verdict == case.expected_verdict,
                    "scores": d.get("philosophy_scores", {}),
                }
            )

        n = len(domain_results)
        correct = sum(1 for r in domain_results if r["match"])
        accuracy = correct / n if n else 0.0
        all_results[domain] = {
            "cases": domain_results,
            "accuracy": accuracy,
            "total_cases": n,
            "correct": correct,
        }
        logger.info("    %s: %.1f%% accuracy (%d/%d)", domain, accuracy * 100, correct, n)

    return all_results


# ═══════════════════════════════════════════════════════════════
# Phase 2 — Baseline comparisons (real evaluators)
# ═══════════════════════════════════════════════════════════════


def run_baseline_comparison(scenario_results: dict) -> dict:
    """Compare EthicAgent against all baselines using real evaluators."""
    logger.info("=" * 60)
    logger.info("PHASE 2: BASELINE COMPARISONS (real evaluators)")
    logger.info("=" * 60)

    baselines = get_all_baselines()
    baseline_results: dict[str, Any] = {}

    # EthicAgent accuracy from Phase 1
    all_ea: list[dict[str, Any]] = []
    for domain_data in scenario_results.values():
        for case in domain_data["cases"]:
            all_ea.append(
                {
                    "actual_verdict": case["predicted_verdict"],
                    "expected_verdict": case["expected_verdict"],
                }
            )
    ea_acc = verdict_accuracy(all_ea).get("overall_accuracy", 0.0)
    baseline_results["EthicAgent"] = {"accuracy": ea_acc}
    logger.info("  EthicAgent: %.1f%%", ea_acc * 100)

    # Re-load scenario case objects for baselines
    from ethicagent.scenarios import get_all_cases

    cases = get_all_cases()

    for bl_name, bl in baselines.items():
        bl_verdicts: list[dict[str, Any]] = []
        for case in cases:
            d = bl.evaluate(case)
            verdict = _normalize_verdict(d.get("verdict", "unknown"))
            bl_verdicts.append(
                {
                    "actual_verdict": verdict,
                    "expected_verdict": case.expected_verdict,
                }
            )
        acc = verdict_accuracy(bl_verdicts).get("overall_accuracy", 0.0)
        baseline_results[bl_name] = {"accuracy": acc}
        logger.info("  %s: %.1f%%", bl_name, acc * 100)

    return baseline_results


# ═══════════════════════════════════════════════════════════════
# Phase 3 — Ablation study (real component-disable experiments)
# ═══════════════════════════════════════════════════════════════


def run_ablation_study(scenario_results: dict) -> dict:
    """Run real ablation study using AblationStudy module."""
    logger.info("=" * 60)
    logger.info("PHASE 3: ABLATION STUDY (real pipeline)")
    logger.info("=" * 60)

    study = AblationStudy()

    # Build test-case dicts from scenario results
    test_dicts: list[dict[str, Any]] = []
    for domain_data in scenario_results.values():
        for case in domain_data["cases"]:
            test_dicts.append(
                {
                    "task": case["task"],
                    "domain": case.get("domain", "general"),
                    "expected_verdict": case["expected_verdict"],
                    "expected_eds_range": case.get("expected_eds_range"),
                }
            )

    ablation = study.run(test_dicts)

    # Extract results per variant
    ablation_results: dict[str, Any] = {}
    full_accuracy = 0.0
    for name, data in ablation.get("variants", {}).items():
        if "error" in data:
            logger.warning("  Variant '%s' errored: %s", name, data["error"])
            continue
        metrics = data.get("metrics", {})
        acc = metrics.get("verdict_accuracy", {}).get("overall_accuracy", 0.0)
        if name == "full_system":
            full_accuracy = acc
        ablation_results[name] = {"accuracy": acc, "delta": 0.0}

    # Backfill deltas
    for name, info in ablation_results.items():
        if name != "full_system":
            info["delta"] = round(info["accuracy"] - full_accuracy, 4)

    for variant in ABLATION_VARIANTS:
        if variant in ablation_results:
            info = ablation_results[variant]
            delta_str = f"{info['delta']:+.1%}" if variant != "full_system" else "  --"
            logger.info("  %-28s: %.1f%% (%s)", variant, info["accuracy"] * 100, delta_str)

    return ablation_results


# ═══════════════════════════════════════════════════════════════
# Phase 4 — Statistical analysis (real math on real data)
# ═══════════════════════════════════════════════════════════════


def run_statistical_analysis(
    scenario_results: dict,
    baseline_results: dict,
) -> dict:
    """Run statistical tests on real EDS scores."""
    logger.info("=" * 60)
    logger.info("PHASE 4: STATISTICAL ANALYSIS (real data)")
    logger.info("=" * 60)

    analyzer = StatisticalAnalyzer()

    # Collect real EDS scores from Phase 1
    ea_scores = [
        case["actual_eds"]
        for domain_data in scenario_results.values()
        for case in domain_data["cases"]
    ]

    # We need real baseline EDS scores too — re-run baselines
    from ethicagent.scenarios import get_all_cases

    cases = get_all_cases()
    baselines = get_all_baselines()

    comparisons: dict[str, Any] = {}
    for bl_name, bl in baselines.items():
        bl_scores = [float(bl.evaluate(c).get("eds_score", 0.0)) for c in cases]
        result = analyzer.compare_systems(ea_scores, bl_scores)
        comparisons[bl_name] = result

        sig = "Y" if result.get("significant", False) else "N"
        p_val = result.get("paired_t_test", {}).get("p_value", 0)
        d_val = result.get("cohens_d", {}).get("d", 0)
        logger.info("  vs %-15s: p=%.6f d=%.3f sig=%s", bl_name, p_val, d_val, sig)

    # Summary from last comparison
    last_comparison = {}
    for comp in comparisons.values():
        last_comparison = comp
    summary = analyzer.generate_summary(last_comparison) if last_comparison else ""

    return {
        "comparisons": comparisons,
        "summary": summary,
    }


# ═══════════════════════════════════════════════════════════════
# Phase 5 — Report generation
# ═══════════════════════════════════════════════════════════════


def generate_reports(
    scenario_results: dict,
    baseline_results: dict,
    ablation_results: dict,
    stats_results: dict,
    output_dir: Path,
    formats: list[str],
) -> None:
    """Generate reports in specified formats."""
    logger.info("=" * 60)
    logger.info("PHASE 5: REPORT GENERATION")
    logger.info("=" * 60)

    combined_results = {
        "main_results": baseline_results,
        "scenario_results": {
            domain: {"accuracy": data["accuracy"], "total_cases": data["total_cases"]}
            for domain, data in scenario_results.items()
        },
        "ablation": ablation_results,
        "statistics": stats_results,
    }
    generator = ReportGenerator(results=combined_results)

    for fmt in formats:
        filename = f"ethicagent_results.{fmt if fmt != 'latex' else 'tex'}"
        filepath = output_dir / filename

        try:
            if fmt == "json":
                content = json.dumps(combined_results, indent=2, default=str)
            elif fmt == "latex":
                content = generator.generate_latex()
            elif fmt == "markdown":
                content = generator.generate_markdown()
            elif fmt == "html":
                content = generator.generate_html()
            else:
                continue

            filepath.write_text(content)
            logger.info("  Generated: %s", filepath)
        except Exception as e:
            logger.error("  Failed to generate %s: %s", fmt, e)

    # Also save raw results as JSON
    raw_path = output_dir / "raw_results.json"
    raw_data = {
        "timestamp": datetime.now().isoformat(),
        "scenario_results": {
            domain: {
                "accuracy": data["accuracy"],
                "total_cases": data["total_cases"],
                "correct": data["correct"],
            }
            for domain, data in scenario_results.items()
        },
        "baseline_results": baseline_results,
        "ablation_results": ablation_results,
    }
    raw_path.write_text(json.dumps(raw_data, indent=2, default=str))
    logger.info("  Raw results: %s", raw_path)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("EthicAgent — Full Experimental Suite")
    logger.info("Started: %s", datetime.now().isoformat())
    logger.info("Domains: %s", ", ".join(args.domains))
    logger.info("LLM mode: %s", "ON" if args.use_llm else "OFF (heuristic)")
    logger.info("=" * 60)

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # Pre-warm orchestrator
    _get_orchestrator(use_llm=args.use_llm)

    # Phase 1: Scenario Evaluation (real pipeline)
    scenario_results = run_scenario_evaluation(args.domains)

    # Phase 2: Baseline Comparisons (real evaluators)
    baseline_results: dict[str, Any] = {}
    if not args.skip_baselines:
        baseline_results = run_baseline_comparison(scenario_results)

    # Phase 3: Ablation Study (real component-disable experiments)
    ablation_results: dict[str, Any] = {}
    if not args.skip_ablation:
        ablation_results = run_ablation_study(scenario_results)

    # Phase 4: Statistical Analysis (real math on real data)
    stats_results: dict[str, Any] = {}
    if not args.skip_stats and baseline_results:
        stats_results = run_statistical_analysis(scenario_results, baseline_results)

    # Phase 5: Report Generation
    generate_reports(
        scenario_results,
        baseline_results,
        ablation_results,
        stats_results,
        output_dir,
        args.formats,
    )

    # Summary
    elapsed = time.time() - start_time
    total_cases = sum(d["total_cases"] for d in scenario_results.values())
    total_correct = sum(d["correct"] for d in scenario_results.values())
    overall_accuracy = total_correct / total_cases if total_cases > 0 else 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    logger.info("  Domains evaluated:     %d", len(args.domains))
    logger.info("  Total test cases:      %d", total_cases)
    logger.info("  Overall accuracy:      %.1f%%", overall_accuracy * 100)
    logger.info("  Baselines compared:    %d", len(baseline_results))
    logger.info("  Ablation variants:     %d", len(ablation_results))
    logger.info("  Reports generated:     %d", len(args.formats))
    logger.info("  Output directory:      %s", output_dir)
    logger.info("  Elapsed time:          %.1fs", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
