"""Reproducibility Benchmark — Determinism and consistency checks.

Runs identical cases multiple times under the same config and
measures how stable the outputs are.  High reproducibility is
critical for auditable AI systems.
"""

from __future__ import annotations

import logging
import random
import statistics
import time
from typing import Any

from ethicagent.scenarios import SCENARIO_REGISTRY
from ethicagent.utils.helpers import now_iso

logger = logging.getLogger(__name__)


class ReproducibilityBenchmark:
    """Determinism and output-stability benchmark.

    Runs the same case *n_repeats* times and checks whether
    verdicts and EDS scores stay identical (or within tolerance).
    """

    def __init__(
        self,
        orchestrator: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.config = config or {}
        self.n_repeats: int = self.config.get("n_repeats", 3)
        self.max_cases: int = self.config.get("max_cases", 30)
        self.eds_tolerance: float = self.config.get("eds_tolerance", 0.01)
        self._fallback_orch: Any | None = None

    def _get_orchestrator(self) -> Any:
        """Return user-supplied orchestrator or a real offline fallback."""
        if self.orchestrator:
            return self.orchestrator
        if self._fallback_orch is None:
            from ethicagent.orchestrator import EthicAgentOrchestrator

            self._fallback_orch = EthicAgentOrchestrator(use_llm=False)
        return self._fallback_orch

    # ── public API ──────────────────────────────────────────────
    def run(self) -> dict[str, Any]:
        """Execute the reproducibility benchmark.

        Returns dict with per-case consistency, aggregate stats,
        and a summary suitable for the unified suite.
        """
        t0 = time.perf_counter()

        # gather a sample of cases
        cases = self._gather_cases()
        logger.info(
            "Reproducibility benchmark: %d cases × %d repeats",
            len(cases),
            self.n_repeats,
        )

        case_results: list[dict[str, Any]] = []

        for case in cases:
            verdicts: list[str] = []
            eds_scores: list[float] = []

            for _ in range(self.n_repeats):
                result = self._run_case(case)
                verdicts.append(result.get("verdict", "UNKNOWN"))
                eds_scores.append(result.get("eds_score", 0.0))

            # check determinism
            verdict_consistent = len(set(verdicts)) == 1
            eds_range = max(eds_scores) - min(eds_scores) if eds_scores else 0.0
            eds_consistent = eds_range <= self.eds_tolerance

            case_results.append(
                {
                    "case_id": getattr(case, "case_id", str(id(case))),
                    "domain": getattr(case, "domain", "general"),
                    "verdicts": verdicts,
                    "eds_scores": [round(s, 4) for s in eds_scores],
                    "verdict_consistent": verdict_consistent,
                    "eds_consistent": eds_consistent,
                    "eds_range": round(eds_range, 4),
                }
            )

        total_time = time.perf_counter() - t0

        # aggregate
        n_verdict_ok = sum(1 for r in case_results if r["verdict_consistent"])
        n_eds_ok = sum(1 for r in case_results if r["eds_consistent"])
        n_total = len(case_results)

        agg: dict[str, Any] = {
            "total_cases": n_total,
            "n_repeats": self.n_repeats,
            "verdict_consistency_rate": round(n_verdict_ok / n_total, 4) if n_total else 0,
            "eds_consistency_rate": round(n_eds_ok / n_total, 4) if n_total else 0,
            "eds_tolerance": self.eds_tolerance,
            "mean_eds_range": round(statistics.mean(r["eds_range"] for r in case_results), 4)
            if case_results
            else 0,
            "seconds": round(total_time, 3),
            "timestamp": now_iso(),
        }

        # show problematic cases (if any)
        flagged = [
            r for r in case_results if not r["verdict_consistent"] or not r["eds_consistent"]
        ]
        if flagged:
            agg["flagged_cases"] = flagged

        # suite-compatible summary
        agg["summary"] = {
            "verdict_consistency": agg["verdict_consistency_rate"],
            "eds_consistency": agg["eds_consistency_rate"],
        }

        return agg

    # ── internals ───────────────────────────────────────────────
    def _gather_cases(self) -> list[Any]:
        all_cases: list[Any] = []
        for _domain_name, cls in SCENARIO_REGISTRY.items():
            scenario = cls()
            all_cases.extend(scenario.get_cases())
        rng = random.Random(1337)
        if len(all_cases) > self.max_cases:
            all_cases = rng.sample(all_cases, self.max_cases)
        return all_cases

    def _run_case(self, case: Any) -> dict[str, Any]:
        """Run one case through the orchestrator."""
        orch = self._get_orchestrator()
        result = orch.run(
            task=case.task,
            domain=getattr(case, "domain", None),
        )
        if hasattr(result, "to_dict"):
            return result.to_dict()
        return result if isinstance(result, dict) else dict(result)
