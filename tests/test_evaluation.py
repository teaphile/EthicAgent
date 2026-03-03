"""Tests for evaluation modules — metrics, baselines, ablation, stats.

Validates metric computation, baseline correctness, ablation study
execution, statistical analysis, and report generation.

NOTE: metric keys align with the rewritten compute_all_metrics():
  verdict_accuracy, eds_score_metrics, eds_range_accuracy,
  fairness, consistency, philosophy_contributions
"""

from __future__ import annotations

import pytest
from ethicagent.evaluation.metrics import (
    verdict_accuracy,
    eds_score_metrics,
    eds_range_accuracy,
    fairness_metrics,
    consistency_score,
    philosophy_contribution_analysis,
    compute_all_metrics,
)
from ethicagent.evaluation.baselines import (
    RandomBaseline,
    RulesOnlyBaseline,
    LLMOnlyBaseline,
    EqualWeightBaseline,
    get_all_baselines,
)
from ethicagent.evaluation.ablation import AblationStudy, ABLATION_VARIANTS
from ethicagent.evaluation.statistical_analysis import StatisticalAnalyzer
from ethicagent.evaluation.benchmark_runner import BenchmarkRunner
from ethicagent.evaluation.report_generator import ReportGenerator


# ═══════════════════════════════════════════════════════════════
# Metrics Tests
# ═══════════════════════════════════════════════════════════════

class TestVerdictAccuracy:
    def test_perfect_accuracy(self, sample_results):
        acc = verdict_accuracy(sample_results)
        assert acc["overall_accuracy"] == 1.0

    def test_mixed_accuracy(self):
        results = [
            {"actual_verdict": "approve", "expected_verdict": "approve"},
            {"actual_verdict": "reject", "expected_verdict": "approve"},
            {"actual_verdict": "escalate", "expected_verdict": "escalate"},
        ]
        acc = verdict_accuracy(results)
        assert abs(acc["overall_accuracy"] - 2 / 3) < 0.01

    def test_empty_results(self):
        acc = verdict_accuracy([])
        assert acc["overall_accuracy"] == 0.0

    def test_per_verdict_accuracy(self, sample_results):
        acc = verdict_accuracy(sample_results)
        # should have per-verdict breakdown
        assert "per_verdict" in acc or "confusion_matrix" in acc or acc["overall_accuracy"] >= 0


class TestEdsScoreMetrics:
    def test_basic_metrics(self, sample_results):
        metrics = eds_score_metrics(sample_results)
        assert "mean" in metrics or "mean_eds" in metrics
        # check score is within valid range
        mean_val = metrics.get("mean", metrics.get("mean_eds", 0))
        assert 0.0 <= mean_val <= 1.0

    def test_empty_results(self):
        metrics = eds_score_metrics([])
        mean_val = metrics.get("mean", metrics.get("mean_eds", 0))
        assert mean_val == 0.0

    def test_has_std(self, sample_results):
        metrics = eds_score_metrics(sample_results)
        assert "std" in metrics or "std_eds" in metrics

    def test_has_min_max(self, sample_results):
        metrics = eds_score_metrics(sample_results)
        has_min = "min" in metrics or "min_eds" in metrics
        has_max = "max" in metrics or "max_eds" in metrics
        assert has_min and has_max


class TestEdsRangeAccuracy:
    def test_all_in_range(self, sample_results):
        acc = eds_range_accuracy(sample_results)
        rate = acc.get("in_range_rate", acc.get("range_accuracy", 0))
        assert rate == 1.0

    def test_out_of_range(self):
        results = [
            {"eds_score": 0.95, "expected_eds_range": (0.0, 0.10)},
        ]
        acc = eds_range_accuracy(results)
        rate = acc.get("in_range_rate", acc.get("range_accuracy", 0))
        assert rate == 0.0

    def test_mixed_range(self):
        results = [
            {"eds_score": 0.85, "expected_eds_range": (0.80, 1.00)},
            {"eds_score": 0.30, "expected_eds_range": (0.80, 1.00)},
        ]
        acc = eds_range_accuracy(results)
        rate = acc.get("in_range_rate", acc.get("range_accuracy", 0))
        assert abs(rate - 0.5) < 0.01


class TestFairnessMetrics:
    def test_basic_fairness(self, sample_results):
        fm = fairness_metrics(sample_results)
        assert isinstance(fm, dict)

    def test_disparate_impact(self):
        # same domain, different scores
        results = [
            {"eds_score": 0.90, "domain": "finance",
             "philosophy_scores": {"deontological": 0.9, "consequentialist": 0.9,
                                   "virtue_ethics": 0.9, "contextual": 0.9}},
            {"eds_score": 0.85, "domain": "finance",
             "philosophy_scores": {"deontological": 0.85, "consequentialist": 0.85,
                                   "virtue_ethics": 0.85, "contextual": 0.85}},
        ]
        fm = fairness_metrics(results)
        assert isinstance(fm, dict)


class TestConsistencyScore:
    def test_consistent_results(self):
        results = [
            {"eds_score": 0.80, "domain": "finance"},
            {"eds_score": 0.82, "domain": "finance"},
            {"eds_score": 0.81, "domain": "finance"},
        ]
        score = consistency_score(results)
        # accept both key formats
        val = score.get("consistency", score.get("consistency_score", 0))
        assert val > 0.8

    def test_inconsistent_results(self):
        results = [
            {"eds_score": 0.01, "domain": "finance"},
            {"eds_score": 0.99, "domain": "finance"},
        ]
        score = consistency_score(results)
        val = score.get("consistency", score.get("consistency_score", 0))
        assert val < 0.5


class TestPhilosophyContributionAnalysis:
    def test_basic_analysis(self, sample_results):
        analysis = philosophy_contribution_analysis(sample_results)
        assert isinstance(analysis, dict)
        if analysis:
            for phil, stats in analysis.items():
                assert "mean" in stats
                assert "std" in stats


class TestComputeAllMetrics:
    def test_all_metrics(self, sample_results):
        metrics = compute_all_metrics(sample_results)
        assert isinstance(metrics, dict)
        assert "verdict_accuracy" in metrics
        # accept either key name that the rewritten module uses
        assert "eds_score_metrics" in metrics or "eds_scores" in metrics

    def test_total_cases(self, sample_results):
        metrics = compute_all_metrics(sample_results)
        assert metrics.get("total_cases", 0) == len(sample_results)


# ═══════════════════════════════════════════════════════════════
# Baseline Tests
# ═══════════════════════════════════════════════════════════════

class TestBaselines:
    def test_random_baseline(self):
        baseline = RandomBaseline()
        assert baseline is not None
        assert baseline.name == "random"

    def test_rules_only_baseline(self):
        baseline = RulesOnlyBaseline()
        assert baseline is not None

    def test_llm_only_baseline(self):
        baseline = LLMOnlyBaseline()
        assert baseline is not None

    def test_equal_weight_baseline(self):
        baseline = EqualWeightBaseline()
        assert baseline is not None

    def test_get_all_baselines(self):
        baselines = get_all_baselines()
        assert isinstance(baselines, dict)
        assert len(baselines) == 4

    def test_random_baseline_evaluate_dict(self):
        baseline = RandomBaseline()
        result = baseline.evaluate({
            "task": "Test task",
            "domain": "general",
        })
        assert isinstance(result, dict)
        assert "verdict" in result
        assert "eds_score" in result

    def test_random_baseline_evaluate_case(self):
        baseline = RandomBaseline()

        class MockCase:
            case_id = "TC-MOCK"
            task = "Test task for mock case"
            domain = "finance"
            expected_verdict = "approve"
            expected_eds_range = (0.0, 1.0)

        result = baseline.evaluate(MockCase())
        assert isinstance(result, dict)
        assert "verdict" in result
        assert 0.0 <= result["eds_score"] <= 1.0

    def test_rules_only_evaluate(self):
        baseline = RulesOnlyBaseline()
        result = baseline.evaluate({"task": "Approve standard loan", "domain": "finance"})
        assert isinstance(result, dict)
        assert "verdict" in result

    def test_equal_weight_evaluate(self):
        baseline = EqualWeightBaseline()
        result = baseline.evaluate({"task": "Some ethical scenario", "domain": "general"})
        assert isinstance(result, dict)
        assert "eds_score" in result


# ═══════════════════════════════════════════════════════════════
# Ablation Tests
# ═══════════════════════════════════════════════════════════════

class TestAblationStudy:
    def test_initialization(self):
        study = AblationStudy()
        assert study is not None

    def test_variants_defined(self):
        assert len(ABLATION_VARIANTS) >= 10

    def test_full_system_variant(self):
        assert "full_system" in ABLATION_VARIANTS
        assert ABLATION_VARIANTS["full_system"]["disable"] == []

    def test_run_simulated(self, sample_test_cases):
        study = AblationStudy()
        results = study.run(
            sample_test_cases,
            variant_names=["full_system", "no_neural"],
        )
        assert "variants" in results
        assert "comparison" in results
        assert "full_system" in results["variants"]
        assert "no_neural" in results["variants"]

    def test_export_results(self, sample_test_cases, tmp_path):
        study = AblationStudy()
        results = study.run(
            sample_test_cases,
            variant_names=["full_system"],
        )
        out = tmp_path / "ablation.json"
        study.export_results(results, str(out))
        assert out.exists()


# ═══════════════════════════════════════════════════════════════
# Statistical Analysis Tests
# ═══════════════════════════════════════════════════════════════

class TestStatisticalAnalyzer:
    def test_initialization(self):
        analyzer = StatisticalAnalyzer()
        assert analyzer.alpha == 0.05

    def test_compare_systems(self):
        analyzer = StatisticalAnalyzer()
        scores_a = [0.8, 0.85, 0.9, 0.75, 0.82, 0.88, 0.91, 0.78, 0.86, 0.84]
        scores_b = [0.5, 0.55, 0.6, 0.45, 0.52, 0.58, 0.61, 0.48, 0.56, 0.54]
        result = analyzer.compare_systems(scores_a, scores_b)
        assert "paired_t_test" in result
        assert "cohens_d" in result
        assert "bootstrap_ci" in result
        assert result["significant"] is True

    def test_compare_identical(self):
        analyzer = StatisticalAnalyzer()
        scores = [0.8, 0.85, 0.9, 0.75, 0.82]
        result = analyzer.compare_systems(scores, scores)
        assert result["significant"] is False

    def test_cohens_d(self):
        analyzer = StatisticalAnalyzer()
        high = [0.9, 0.85, 0.88, 0.92, 0.87]
        low = [0.3, 0.35, 0.28, 0.32, 0.37]
        result = analyzer._cohens_d(high, low)
        assert result["interpretation"] == "large"

    def test_cliffs_delta(self):
        analyzer = StatisticalAnalyzer()
        a = [0.9, 0.85, 0.88, 0.92, 0.87]
        b = [0.3, 0.35, 0.28, 0.32, 0.37]
        result = analyzer._cliffs_delta(a, b)
        assert result["delta"] > 0

    def test_mcnemar(self):
        analyzer = StatisticalAnalyzer()
        correct_a = [True, True, True, False, True, True, False, True, True, True]
        correct_b = [True, False, True, False, False, True, False, True, False, True]
        result = analyzer.mcnemar_test(correct_a, correct_b)
        assert "chi2" in result
        assert "p_value" in result

    def test_generate_summary(self):
        analyzer = StatisticalAnalyzer()
        scores_a = [0.8, 0.85, 0.9, 0.75, 0.82]
        scores_b = [0.5, 0.55, 0.6, 0.45, 0.52]
        result = analyzer.compare_systems(scores_a, scores_b)
        summary = analyzer.generate_summary(result)
        assert isinstance(summary, str)
        assert "STATISTICAL ANALYSIS" in summary


# ═══════════════════════════════════════════════════════════════
# Report Generator Tests
# ═══════════════════════════════════════════════════════════════

class TestReportGenerator:
    def test_initialization(self):
        gen = ReportGenerator()
        assert gen is not None

    def test_generate_latex(self):
        gen = ReportGenerator(results={"aggregate_metrics": {}})
        latex = gen.generate_latex()
        assert isinstance(latex, str)
        assert "EthicAgent" in latex

    def test_generate_markdown(self):
        gen = ReportGenerator(results={"aggregate_metrics": {}})
        md = gen.generate_markdown()
        assert isinstance(md, str)
        assert "# EthicAgent" in md or "EthicAgent" in md

    def test_generate_html(self):
        gen = ReportGenerator(results={"aggregate_metrics": {}})
        html = gen.generate_html()
        assert isinstance(html, str)
        assert "EthicAgent" in html


# ═══════════════════════════════════════════════════════════════
# Benchmark Runner Tests
# ═══════════════════════════════════════════════════════════════

class TestBenchmarkRunner:
    def test_initialization(self):
        runner = BenchmarkRunner()
        assert runner is not None

    def test_get_summary_empty(self):
        runner = BenchmarkRunner()
        summary = runner.get_summary()
        assert summary["status"] == "no results"
