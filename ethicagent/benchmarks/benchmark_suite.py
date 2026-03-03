"""Benchmark Suite — Orchestrates all benchmark types.

Combines cross-domain, performance, and reproducibility benchmarks
into a single runnable suite with unified reporting.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ethicagent.utils.helpers import now_iso

logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """Unified benchmark suite runner.

    Runs all registered benchmarks and produces a combined report.

    Usage::

        suite = BenchmarkSuite(orchestrator=my_orch)
        report = suite.run_all()
        suite.export("benchmarks_report.json")
    """

    def __init__(
        self,
        orchestrator: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.config = config or {}
        self.results: dict[str, Any] = {}

    def run_all(
        self,
        benchmarks: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run all (or selected) benchmarks.

        Args:
            benchmarks: Subset to run, e.g. ``["cross_domain", "performance"]``.
                Default: all three.

        Returns:
            Combined results dict.
        """
        from ethicagent.benchmarks.cross_domain import CrossDomainBenchmark
        from ethicagent.benchmarks.performance import PerformanceBenchmark
        from ethicagent.benchmarks.reproducibility import ReproducibilityBenchmark

        available = {
            "cross_domain": CrossDomainBenchmark,
            "performance": PerformanceBenchmark,
            "reproducibility": ReproducibilityBenchmark,
            # External benchmark adapters — load on demand to avoid
            # heavy imports when only running internal benchmarks.
            "external_ethics": None,
            "external_bbq": None,
        }

        to_run = benchmarks or list(available.keys())
        t0 = time.perf_counter()

        logger.info("Starting benchmark suite: %s", ", ".join(to_run))

        suite_results: dict[str, Any] = {
            "suite_id": f"suite_{int(time.time())}",
            "start_time": now_iso(),
            "benchmarks": {},
        }

        for name in to_run:
            if name not in available:
                logger.warning("Unknown benchmark '%s' — skipping", name)
                continue

            # External benchmarks use a different runner path — they
            # load an adapter, convert to ScenarioCase, and evaluate.
            if name == "external_ethics":
                try:
                    result = self._run_external_ethics()
                    suite_results["benchmarks"][name] = result
                    logger.info("Benchmark '%s' complete", name)
                except Exception as exc:
                    logger.error("Benchmark '%s' failed: %s", name, exc)
                    suite_results["benchmarks"][name] = {"error": str(exc)}
                continue

            if name == "external_bbq":
                try:
                    result = self._run_external_bbq()
                    suite_results["benchmarks"][name] = result
                    logger.info("Benchmark '%s' complete", name)
                except Exception as exc:
                    logger.error("Benchmark '%s' failed: %s", name, exc)
                    suite_results["benchmarks"][name] = {"error": str(exc)}
                continue

            logger.info("Running benchmark: %s", name)
            bench = available[name](
                orchestrator=self.orchestrator,
                config=self.config,
            )
            try:
                result = bench.run()
                suite_results["benchmarks"][name] = result
                logger.info("Benchmark '%s' complete", name)
            except Exception as exc:
                logger.error("Benchmark '%s' failed: %s", name, exc)
                suite_results["benchmarks"][name] = {"error": str(exc)}

        suite_results["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
        suite_results["end_time"] = now_iso()
        self.results = suite_results

        logger.info(
            "Benchmark suite done in %.1fs",
            suite_results["elapsed_seconds"],
        )
        return suite_results

    def export(self, filepath: str) -> None:
        """Export results to JSON."""
        import json
        from pathlib import Path

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info("Suite results exported → %s", filepath)

    def summary(self) -> dict[str, Any]:
        """Get a high-level summary of suite results."""
        if not self.results:
            return {"status": "no results"}

        out: dict[str, Any] = {
            "suite_id": self.results.get("suite_id", ""),
            "elapsed_seconds": self.results.get("elapsed_seconds", 0),
            "benchmarks_run": list(self.results.get("benchmarks", {}).keys()),
        }
        for name, data in self.results.get("benchmarks", {}).items():
            if "error" in data:
                out[name] = {"status": "failed", "error": data["error"]}
            else:
                out[name] = {"status": "passed", "summary": data.get("summary", {})}
        return out

    # ── External benchmark helpers ─────────────────────────────

    def _run_external_ethics(self) -> dict[str, Any]:
        """Run the ETHICS dataset adapter benchmark."""
        from ethicagent.benchmarks.external.ethics_adapter import EthicsDatasetAdapter

        adapter = EthicsDatasetAdapter()
        adapter.load(use_builtin=True)
        cases = adapter.to_scenario_cases()

        return {
            "adapter": "ETHICS (Hendrycks et al., 2021)",
            "total_cases": len(cases),
            "summary": adapter.summary(),
            "status": "loaded",
        }

    def _run_external_bbq(self) -> dict[str, Any]:
        """Run the BBQ bias benchmark adapter."""
        from ethicagent.benchmarks.external.bbq_adapter import BBQAdapter

        adapter = BBQAdapter()
        adapter.load(use_builtin=True)
        cases = adapter.to_scenario_cases()

        return {
            "adapter": "BBQ (Parrish et al., 2022)",
            "total_cases": len(cases),
            "summary": adapter.summary(),
            "status": "loaded",
        }
