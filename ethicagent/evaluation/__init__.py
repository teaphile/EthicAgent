"""Evaluation Package — Experimental Evaluation Framework.

High-level API:

>>> runner = BenchmarkRunner(orchestrator=my_orch)
>>> results = runner.run_full_benchmark([scenario])
>>> report = ReportGenerator(results)
>>> report.generate_full_report()
"""

from ethicagent.evaluation.benchmark_runner import BenchmarkRunner
from ethicagent.evaluation.statistical_analysis import StatisticalAnalyzer
from ethicagent.evaluation.report_generator import ReportGenerator
from ethicagent.evaluation.metrics import compute_all_metrics

__all__ = [
    "BenchmarkRunner",
    "StatisticalAnalyzer",
    "ReportGenerator",
    "compute_all_metrics",
]
