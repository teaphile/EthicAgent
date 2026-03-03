"""Benchmarks — Standardised evaluation suites.

Provides reusable benchmark suites for cross-domain evaluation,
performance profiling, and reproducibility checks.

>>> from ethicagent.benchmarks import BenchmarkSuite
>>> suite = BenchmarkSuite()
>>> report = suite.run_all()
"""

from __future__ import annotations

from ethicagent.benchmarks.benchmark_suite import BenchmarkSuite
from ethicagent.benchmarks.cross_domain import CrossDomainBenchmark
from ethicagent.benchmarks.performance import PerformanceBenchmark
from ethicagent.benchmarks.reproducibility import ReproducibilityBenchmark

__all__ = [
    "BenchmarkSuite",
    "CrossDomainBenchmark",
    "PerformanceBenchmark",
    "ReproducibilityBenchmark",
]
