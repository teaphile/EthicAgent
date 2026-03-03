"""Benchmark Runner — Orchestrates full experimental evaluation.

Runs scenarios through EthicAgent and baselines, collecting
results for comparative analysis and statistical testing.

Usage::

    from ethicagent.evaluation import BenchmarkRunner
    from ethicagent.scenarios import HealthcareTriageScenario

    runner = BenchmarkRunner(orchestrator=my_orch)
    scenario = HealthcareTriageScenario()
    scenario.get_cases()
    results = runner.run_full_benchmark([scenario])
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ethicagent.evaluation.baselines import get_all_baselines
from ethicagent.evaluation.metrics import compute_all_metrics
from ethicagent.utils.helpers import now_iso

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run benchmark experiments comparing EthicAgent against baselines.

    Supports running full scenario suites, individual cases,
    and ablation configurations with progress tracking.
    """

    def __init__(
        self,
        orchestrator: Any | None = None,
        baselines: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.baselines = baselines or get_all_baselines()
        self.config = config or {}
        self.results: dict[str, Any] = {}
        self._progress_cb: Callable | None = None
        self._fallback_orch: Any | None = None

    def set_progress_callback(
        self,
        callback: Callable[[str, int, int, str], None],
    ) -> None:
        """Set a callback for progress updates.

        ``callback(phase, current, total, message)``
        """
        self._progress_cb = callback

    def _progress(
        self,
        phase: str,
        current: int,
        total: int,
        message: str = "",
    ) -> None:
        if self._progress_cb:
            self._progress_cb(phase, current, total, message)
        logger.info("[%s] %d/%d — %s", phase, current, total, message)

    # ─── Run Full Benchmark ──────────────────────────────────

    def run_full_benchmark(
        self,
        scenarios: list[Any],
        include_baselines: bool = True,
        include_ablation: bool = False,
        ablation_config: dict | None = None,
    ) -> dict[str, Any]:
        """Run a complete benchmark evaluation.

        Args:
            scenarios: Scenario instances whose ``get_cases()`` has
                already been called (cases are in ``scenario.cases``).
            include_baselines: Whether to run baseline comparisons.
            include_ablation: Whether to run ablation studies.
            ablation_config: Dict with ``factory`` key for ablation.

        Returns:
            Complete benchmark results dict.
        """
        t0 = time.perf_counter()
        benchmark_id = f"benchmark_{int(time.time())}"

        logger.info(
            "Starting benchmark '%s' — %d scenarios",
            benchmark_id,
            len(scenarios),
        )

        result: dict[str, Any] = {
            "benchmark_id": benchmark_id,
            "start_time": now_iso(),
            "scenario_results": {},
            "baseline_results": {},
            "ablation_results": {},
        }

        # Ensure cases are loaded
        for s in scenarios:
            if not getattr(s, "cases", None):
                s.get_cases()

        # Phase 1: EthicAgent
        total_cases = sum(len(s.cases) for s in scenarios)
        self._progress("ethicagent", 0, total_cases, "Starting EthicAgent eval")

        done = 0
        for scenario in scenarios:
            name = scenario.__class__.__name__
            logger.info("Running scenario: %s (%d cases)", name, len(scenario.cases))
            result["scenario_results"][name] = self._run_scenario(scenario)
            done += len(scenario.cases)
            self._progress("ethicagent", done, total_cases, name)

        # Phase 2: Baselines
        if include_baselines:
            n_bl = len(self.baselines)
            self._progress("baselines", 0, n_bl, "Starting baselines")
            for i, (bl_name, baseline) in enumerate(self.baselines.items()):
                logger.info("Running baseline: %s", bl_name)
                result["baseline_results"][bl_name] = self._run_baseline(baseline, scenarios)
                self._progress("baselines", i + 1, n_bl, bl_name)

        # Phase 3: Ablation (optional)
        if include_ablation:
            from ethicagent.evaluation.ablation import AblationStudy

            all_cases = []
            for s in scenarios:
                all_cases.extend(self._scenario_to_dicts(s))

            study = AblationStudy(
                orchestrator_factory=(ablation_config or {}).get("factory"),
            )
            result["ablation_results"] = study.run(all_cases)

        # Phase 4: Aggregate metrics
        result["aggregate_metrics"] = self._compute_aggregates(result)
        result["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
        result["end_time"] = now_iso()

        self.results = result
        logger.info("Benchmark done in %.1fs", result["elapsed_seconds"])
        return result

    # ─── Single Scenario ─────────────────────────────────────

    def _run_scenario(self, scenario: Any) -> dict[str, Any]:
        case_results: list[dict[str, Any]] = []
        t0 = time.perf_counter()

        for case in scenario.cases:
            try:
                r = self._evaluate_case(case)
                case_results.append(r)
            except Exception as exc:
                logger.error("Case %s failed: %s", case.case_id, exc)
                case_results.append(
                    {
                        "case_id": case.case_id,
                        "error": str(exc),
                        "expected_verdict": getattr(case, "expected_verdict", ""),
                        "actual_verdict": "error",
                    }
                )

        return {
            "scenario": scenario.__class__.__name__,
            "total_cases": len(scenario.cases),
            "successful_cases": sum(1 for r in case_results if "error" not in r),
            "results": case_results,
            "metrics": compute_all_metrics(case_results),
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }

    def _get_orchestrator(self) -> Any:
        """Return the user-supplied orchestrator, or create a real offline one."""
        if self.orchestrator:
            return self.orchestrator
        if self._fallback_orch is None:
            from ethicagent.orchestrator import EthicAgentOrchestrator

            self._fallback_orch = EthicAgentOrchestrator(use_llm=False)
        return self._fallback_orch

    @staticmethod
    def _to_dict(result: Any) -> dict[str, Any]:
        """Convert PipelineResult (or dict) to a plain dict."""
        if hasattr(result, "to_dict"):
            return result.to_dict()
        return result if isinstance(result, dict) else dict(result)

    def _evaluate_case(self, case: Any) -> dict[str, Any]:
        orch = self._get_orchestrator()
        result = orch.run(
            task=case.task,
            domain=getattr(case, "domain", None),
            metadata=getattr(case, "metadata", None),
        )
        r = self._to_dict(result)

        r["case_id"] = case.case_id
        r["expected_verdict"] = getattr(case, "expected_verdict", "")
        r["expected_eds_range"] = getattr(case, "expected_eds_range", None)
        r["difficulty"] = getattr(case, "difficulty", "unknown")
        r["actual_verdict"] = r.get("verdict", "unknown")
        return r

    # ─── Baseline Run ────────────────────────────────────────

    def _run_baseline(self, baseline: Any, scenarios: list[Any]) -> dict[str, Any]:
        all_results: list[dict[str, Any]] = []
        t0 = time.perf_counter()

        for scenario in scenarios:
            for case in scenario.cases:
                try:
                    r = baseline.evaluate(case)
                    r["case_id"] = case.case_id
                    r["expected_verdict"] = getattr(case, "expected_verdict", "")
                    r["actual_verdict"] = r.get("verdict", "unknown")
                    r["expected_eds_range"] = getattr(case, "expected_eds_range", None)
                    r["difficulty"] = getattr(case, "difficulty", "unknown")
                    all_results.append(r)
                except Exception as exc:
                    logger.error(
                        "Baseline '%s' failed on %s: %s",
                        getattr(baseline, "name", "?"),
                        case.case_id,
                        exc,
                    )
                    all_results.append(
                        {
                            "case_id": case.case_id,
                            "error": str(exc),
                            "expected_verdict": getattr(case, "expected_verdict", ""),
                            "actual_verdict": "error",
                        }
                    )

        return {
            "baseline": getattr(baseline, "name", baseline.__class__.__name__),
            "total_cases": len(all_results),
            "results": all_results,
            "metrics": compute_all_metrics(all_results),
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }

    # ─── Aggregation ─────────────────────────────────────────

    def _compute_aggregates(self, results: dict[str, Any]) -> dict[str, Any]:
        aggregate: dict[str, Any] = {}

        # EthicAgent aggregate
        ea_results: list[dict] = []
        for sd in results.get("scenario_results", {}).values():
            ea_results.extend(sd.get("results", []))
        if ea_results:
            aggregate["ethicagent"] = compute_all_metrics(ea_results)

        # Per-baseline aggregate
        for bl_name, bl_data in results.get("baseline_results", {}).items():
            aggregate[bl_name] = bl_data.get("metrics", {})

        # Improvement %
        if "ethicagent" in aggregate:
            ea_acc = (
                aggregate["ethicagent"].get("verdict_accuracy", {}).get("overall_accuracy", 0.0)
            )
            improvements: dict[str, float] = {}
            for name, m in aggregate.items():
                if name == "ethicagent":
                    continue
                bl_acc = m.get("verdict_accuracy", {}).get("overall_accuracy", 0.0)
                if bl_acc > 0:
                    improvements[name] = round((ea_acc - bl_acc) / bl_acc * 100, 2)
                else:
                    improvements[name] = float("inf") if ea_acc > 0 else 0.0
            aggregate["improvement_over_baselines"] = improvements

        return aggregate

    # ─── Helpers ──────────────────────────────────────────────

    def _scenario_to_dicts(self, scenario: Any) -> list[dict[str, Any]]:
        return [
            {
                "case_id": c.case_id,
                "task": c.task,
                "domain": getattr(c, "domain", "general"),
                "expected_verdict": getattr(c, "expected_verdict", ""),
                "expected_eds_range": getattr(c, "expected_eds_range", None),
                "metadata": getattr(c, "metadata", {}),
            }
            for c in scenario.cases
        ]

    def export_results(self, filepath: str) -> None:
        """Export benchmark results to JSON."""

        def _sanitize(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            if isinstance(obj, float):
                if obj != obj:  # NaN
                    return None
                if obj in (float("inf"), float("-inf")):
                    return str(obj)
            return obj

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(_sanitize(self.results), f, indent=2, default=str)
        logger.info("Results exported → %s", filepath)

    def get_summary(self) -> dict[str, Any]:
        """Get a brief summary of benchmark results."""
        if not self.results:
            return {"status": "no results"}

        summary: dict[str, Any] = {
            "benchmark_id": self.results.get("benchmark_id", ""),
            "total_scenarios": len(self.results.get("scenario_results", {})),
            "total_baselines": len(self.results.get("baseline_results", {})),
            "elapsed_seconds": self.results.get("elapsed_seconds", 0),
        }

        agg = self.results.get("aggregate_metrics", {})
        if "ethicagent" in agg:
            summary["ethicagent_accuracy"] = (
                agg["ethicagent"].get("verdict_accuracy", {}).get("overall_accuracy", 0.0)
            )

        if "improvement_over_baselines" in agg:
            summary["improvements"] = agg["improvement_over_baselines"]

        return summary
