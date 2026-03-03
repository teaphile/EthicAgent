"""Base Scenario — Abstract Scenario Framework.

Defines the base class for all ethical evaluation scenarios.
Every domain-specific scenario (healthcare, hiring, etc.) inherits
from this and just needs to implement get_cases() and get_domain().

The base class handles:
  - Loading cases from JSON
  - Running cases against the pipeline orchestrator
  - Collecting per-case results
  - Aggregating statistics
  - Exporting results for reports

Author: EthicAgent Team
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Data containers ─────────────────────────────────────────────


@dataclass
class ScenarioCase:
    """A single test case within a scenario.

    Each case describes a decision the pipeline must evaluate.
    """

    case_id: str
    domain: str
    task: str
    difficulty: str = "medium"  # easy | medium | hard
    expected_verdict: str = ""  # approve / reject / escalate / hard_block
    expected_eds_range: tuple[float, float] = (0.0, 1.0)
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)  # e.g. ["discrimination", "safety"]


@dataclass
class ScenarioResult:
    """What happened when we ran a single case."""

    case_id: str
    domain: str
    actual_verdict: str = ""
    expected_verdict: str = ""
    actual_eds: float = 0.0
    expected_eds_range: tuple[float, float] = (0.0, 1.0)
    verdict_match: bool = False
    eds_in_range: bool = False
    elapsed_seconds: float = 0.0
    pipeline_result: Any = field(default_factory=dict)
    error: str = ""


# ── Base class ──────────────────────────────────────────────────


class BaseScenario(ABC):
    """Abstract base class for ethical evaluation scenarios.

    Subclasses must implement:
      - get_cases()  → list of ScenarioCase
      - get_domain() → str

    Usage::

        scenario = HealthcareTriageScenario()
        results  = scenario.run(my_orchestrator)
        stats    = scenario.get_statistics()
        scenario.export_results("results/healthcare.json")
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self.cases: list[ScenarioCase] = []
        self.results: list[ScenarioResult] = []

    # ── Abstract interface ──────────────────────────────────────

    @abstractmethod
    def get_cases(self) -> list[ScenarioCase]:
        """Return all test cases for this scenario."""
        ...

    @abstractmethod
    def get_domain(self) -> str:
        """Return the domain name (e.g. 'healthcare')."""
        ...

    # ── JSON loading ────────────────────────────────────────────

    def load_cases_from_json(self, filepath: str | Path) -> list[ScenarioCase]:
        """Load cases from a JSON file.

        Expected format::

            { "cases": [ { "case_id": "...", "task": "..." }, ... ] }

        or just a top-level list.
        """
        filepath = Path(filepath)
        try:
            raw = json.loads(filepath.read_text())
            items = raw.get("cases", raw) if isinstance(raw, dict) else raw

            cases: list[ScenarioCase] = []
            for item in items:
                case = ScenarioCase(
                    case_id=item.get("case_id", f"CASE-{len(cases) + 1:03d}"),
                    domain=item.get("domain", self.get_domain()),
                    task=item.get("task", ""),
                    difficulty=item.get("difficulty", "medium"),
                    expected_verdict=item.get("expected_verdict", ""),
                    expected_eds_range=tuple(item.get("expected_eds_range", [0.0, 1.0])),
                    context=item.get("context", {}),
                    metadata=item.get("metadata", {}),
                    tags=item.get("tags", []),
                )
                cases.append(case)

            self.cases = cases
            logger.info(
                "Loaded %d cases from %s for scenario '%s'", len(cases), filepath, self.name
            )
            return cases

        except Exception as exc:
            logger.error("Failed to load cases from %s: %s", filepath, exc)
            return []

    # ── Execution ───────────────────────────────────────────────

    def run(self, orchestrator, *, stop_on_error: bool = False) -> list[ScenarioResult]:
        """Run all scenario cases against the orchestrator.

        Args:
            orchestrator: EthicAgentOrchestrator instance (must have .run()).
            stop_on_error: If True, stop on the first error instead of
                           continuing to the next case.

        Returns:
            List of ScenarioResult objects.
        """
        if not self.cases:
            self.cases = self.get_cases()

        self.results = []
        total = len(self.cases)
        logger.info("Running scenario '%s' with %d cases", self.name, total)

        for i, case in enumerate(self.cases, 1):
            logger.info("[%d/%d] Case %s", i, total, case.case_id)
            result = self._run_case(case, orchestrator)
            self.results.append(result)

            if result.error and stop_on_error:
                logger.warning("Stopping early due to error in %s", case.case_id)
                break

        stats = self.get_statistics()
        logger.info(
            "Scenario '%s' complete: %.1f%% verdict accuracy, avg EDS=%.3f",
            self.name,
            stats.get("verdict_accuracy", 0) * 100,
            stats.get("average_eds", 0),
        )
        return self.results

    def _run_case(self, case: ScenarioCase, orchestrator) -> ScenarioResult:
        """Run a single case. Never raises — errors are captured."""
        t0 = time.perf_counter()
        try:
            pipeline_result = orchestrator.run(
                task=case.task,
                domain=case.domain,
                metadata=case.metadata,
            )
            elapsed = time.perf_counter() - t0

            actual_verdict = str(getattr(pipeline_result, "verdict", "unknown")).lower()
            actual_eds = float(getattr(pipeline_result, "eds_score", 0.0))

            verdict_match = (
                actual_verdict == case.expected_verdict.lower() if case.expected_verdict else True
            )
            eds_in_range = case.expected_eds_range[0] <= actual_eds <= case.expected_eds_range[1]

            return ScenarioResult(
                case_id=case.case_id,
                domain=case.domain,
                actual_verdict=actual_verdict,
                expected_verdict=case.expected_verdict,
                actual_eds=actual_eds,
                expected_eds_range=case.expected_eds_range,
                verdict_match=verdict_match,
                eds_in_range=eds_in_range,
                elapsed_seconds=elapsed,
                pipeline_result=pipeline_result,
            )

        except Exception as exc:
            logger.error("Case %s failed: %s", case.case_id, exc)
            return ScenarioResult(
                case_id=case.case_id,
                domain=case.domain,
                expected_verdict=case.expected_verdict,
                expected_eds_range=case.expected_eds_range,
                elapsed_seconds=time.perf_counter() - t0,
                error=str(exc),
            )

    # ── Statistics ──────────────────────────────────────────────

    def get_statistics(self) -> dict[str, Any]:
        """Compute scenario-level statistics."""
        if not self.results:
            return {"total_cases": 0}

        total = len(self.results)
        errors = sum(1 for r in self.results if r.error)
        executed = total - errors

        verdict_matches = sum(1 for r in self.results if r.verdict_match)
        eds_ok = sum(1 for r in self.results if r.eds_in_range)

        eds_scores = [r.actual_eds for r in self.results if not r.error]
        avg_eds = sum(eds_scores) / len(eds_scores) if eds_scores else 0.0

        # Per-difficulty breakdown
        difficulty_stats: dict[str, dict[str, Any]] = {}
        for diff in ("easy", "medium", "hard"):
            diff_results = [
                r
                for r, c in zip(self.results, self.cases, strict=False)
                if c.difficulty == diff and not r.error
            ]
            if diff_results:
                difficulty_stats[diff] = {
                    "count": len(diff_results),
                    "verdict_accuracy": (
                        sum(1 for r in diff_results if r.verdict_match) / len(diff_results)
                    ),
                    "avg_eds": (sum(r.actual_eds for r in diff_results) / len(diff_results)),
                }

        # Per-tag breakdown
        tag_stats: dict[str, int] = {}
        for case in self.cases:
            for tag in case.tags:
                tag_stats[tag] = tag_stats.get(tag, 0) + 1

        times = [r.elapsed_seconds for r in self.results if not r.error]

        return {
            "scenario": self.name,
            "domain": self.get_domain(),
            "total_cases": total,
            "executed": executed,
            "errors": errors,
            "verdict_accuracy": verdict_matches / total if total else 0.0,
            "eds_in_range_rate": eds_ok / total if total else 0.0,
            "average_eds": avg_eds,
            "min_eds": min(eds_scores) if eds_scores else 0.0,
            "max_eds": max(eds_scores) if eds_scores else 0.0,
            "difficulty_breakdown": difficulty_stats,
            "tag_counts": tag_stats,
            "avg_elapsed_seconds": sum(times) / len(times) if times else 0.0,
        }

    # ── Export ──────────────────────────────────────────────────

    def export_results(self, filepath: str | Path) -> None:
        """Export results to JSON for downstream reporting."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "scenario": self.name,
            "domain": self.get_domain(),
            "description": self.description,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "statistics": self.get_statistics(),
            "results": [
                {
                    "case_id": r.case_id,
                    "domain": r.domain,
                    "actual_verdict": r.actual_verdict,
                    "expected_verdict": r.expected_verdict,
                    "actual_eds": r.actual_eds,
                    "expected_eds_range": list(r.expected_eds_range),
                    "verdict_match": r.verdict_match,
                    "eds_in_range": r.eds_in_range,
                    "elapsed_seconds": round(r.elapsed_seconds, 4),
                    "error": r.error,
                }
                for r in self.results
            ],
        }

        filepath.write_text(json.dumps(export_data, indent=2, default=str))
        logger.info("Exported %d results to %s", len(self.results), filepath)

    # ── Filtering helpers ───────────────────────────────────────

    def filter_cases(
        self, *, difficulty: str | None = None, tag: str | None = None
    ) -> list[ScenarioCase]:
        """Return a subset of cases matching the given filters."""
        result = self.cases
        if difficulty:
            result = [c for c in result if c.difficulty == difficulty]
        if tag:
            result = [c for c in result if tag in c.tags]
        return result

    def failed_cases(self) -> list[ScenarioResult]:
        """Return results where the verdict didn't match expectations."""
        return [r for r in self.results if not r.verdict_match]
