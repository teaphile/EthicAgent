"""Performance Benchmark — Latency and throughput profiling.

Measures per-case latency, throughput under load, and memory
consumption to ensure the pipeline meets real-time requirements.

TODO: Add memory profiling once we settle on a profiler.
"""

from __future__ import annotations

import logging
import statistics
import time
from typing import Any

from ethicagent.scenarios import SCENARIO_REGISTRY
from ethicagent.utils.helpers import now_iso

logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Latency and throughput benchmark.

    Runs a representative subset of cases and records per-case
    timing statistics.
    """

    def __init__(
        self,
        orchestrator: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.config = config or {}
        self.max_cases = config.get("max_cases", 50) if config else 50
        self._fallback_orch: Any | None = None

    def _get_orchestrator(self) -> Any:
        """Return user-supplied orchestrator or a real offline fallback."""
        if self.orchestrator:
            return self.orchestrator
        if self._fallback_orch is None:
            from ethicagent.orchestrator import EthicAgentOrchestrator

            self._fallback_orch = EthicAgentOrchestrator(use_llm=False)
        return self._fallback_orch

    def run(self) -> dict[str, Any]:
        """Run the performance benchmark.

        Returns:
            Dict with latency stats and throughput.
        """
        t0 = time.perf_counter()
        latencies: list[float] = []
        per_domain: dict[str, list[float]] = {}

        # Collect cases (sample across domains)
        all_cases: list[Any] = []
        for _domain_name, cls in SCENARIO_REGISTRY.items():
            scenario = cls()
            cases = scenario.get_cases()
            all_cases.extend(cases)

        # Limit to max_cases for speed
        import random

        rng = random.Random(42)
        if len(all_cases) > self.max_cases:
            all_cases = rng.sample(all_cases, self.max_cases)

        logger.info("Performance benchmark: %d cases", len(all_cases))

        for case in all_cases:
            case_t0 = time.perf_counter()
            self._run_case(case)
            elapsed = time.perf_counter() - case_t0
            latencies.append(elapsed)

            domain = getattr(case, "domain", "general")
            per_domain.setdefault(domain, []).append(elapsed)

        total_time = time.perf_counter() - t0

        # Stats
        result: dict[str, Any] = {
            "total_cases": len(all_cases),
            "total_seconds": round(total_time, 3),
            "throughput_cases_per_sec": round(len(all_cases) / total_time, 2)
            if total_time > 0
            else 0,
            "latency": self._latency_stats(latencies),
            "per_domain_latency": {d: self._latency_stats(lats) for d, lats in per_domain.items()},
            "timestamp": now_iso(),
        }

        # Summary for suite integration
        result["summary"] = {
            "mean_latency_ms": result["latency"]["mean_ms"],
            "p99_latency_ms": result["latency"]["p99_ms"],
            "throughput": result["throughput_cases_per_sec"],
        }

        return result

    def _run_case(self, case: Any) -> None:
        """Run a single case through the orchestrator."""
        orch = self._get_orchestrator()
        orch.run(
            task=case.task,
            domain=getattr(case, "domain", None),
        )

    @staticmethod
    def _latency_stats(latencies: list[float]) -> dict[str, float]:
        """Compute latency statistics in milliseconds."""
        if not latencies:
            return {}
        ms = [t * 1000 for t in latencies]
        return {
            "mean_ms": round(statistics.mean(ms), 2),
            "median_ms": round(statistics.median(ms), 2),
            "std_ms": round(statistics.stdev(ms), 2) if len(ms) > 1 else 0.0,
            "min_ms": round(min(ms), 2),
            "max_ms": round(max(ms), 2),
            "p95_ms": round(sorted(ms)[int(len(ms) * 0.95)], 2),
            "p99_ms": round(sorted(ms)[min(int(len(ms) * 0.99), len(ms) - 1)], 2),
            "count": len(ms),
        }
