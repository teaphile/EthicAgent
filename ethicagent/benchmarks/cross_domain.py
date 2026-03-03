"""Cross-Domain Benchmark — Transfer and generalisation tests.

Measures how well the EthicAgent pipeline generalises across
domains. A system trained/tuned for healthcare should still
produce reasonable results on finance cases, etc.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ethicagent.evaluation.metrics import compute_all_metrics
from ethicagent.scenarios import SCENARIO_REGISTRY
from ethicagent.utils.helpers import now_iso

logger = logging.getLogger(__name__)


class CrossDomainBenchmark:
    """Cross-domain transfer benchmark.

    For each domain D, evaluates:
    1. **In-domain** accuracy (cases from D, weights for D).
    2. **Out-of-domain** accuracy (cases from D, weights for "general").
    3. **Transfer gap** = in-domain - out-of-domain accuracy.

    A small gap means the system generalises well.
    """

    def __init__(
        self,
        orchestrator: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.config = config or {}
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
        """Run the cross-domain benchmark.

        Returns:
            Results with in-domain / out-of-domain metrics per domain.
        """
        t0 = time.perf_counter()
        results: dict[str, Any] = {"domains": {}}

        for domain_name, scenario_cls in SCENARIO_REGISTRY.items():
            logger.info("Cross-domain: evaluating %s", domain_name)
            scenario = scenario_cls()
            cases = scenario.get_cases()

            # In-domain evaluation
            in_domain = self._evaluate_cases(cases, domain_name)

            # Out-of-domain evaluation (general weights)
            ood = self._evaluate_cases(cases, "general")

            in_acc = in_domain.get("verdict_accuracy", {}).get("overall_accuracy", 0.0)
            ood_acc = ood.get("verdict_accuracy", {}).get("overall_accuracy", 0.0)

            results["domains"][domain_name] = {
                "in_domain": in_domain,
                "out_of_domain": ood,
                "transfer_gap": round(in_acc - ood_acc, 4),
                "total_cases": len(cases),
            }

        # Aggregate
        gaps = [d["transfer_gap"] for d in results["domains"].values()]
        results["summary"] = {
            "mean_transfer_gap": round(sum(gaps) / len(gaps), 4) if gaps else 0.0,
            "max_transfer_gap": max(gaps) if gaps else 0.0,
            "min_transfer_gap": min(gaps) if gaps else 0.0,
            "domains_tested": list(results["domains"].keys()),
        }
        results["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
        results["timestamp"] = now_iso()

        return results

    def _evaluate_cases(
        self,
        cases: list[Any],
        domain_override: str,
    ) -> dict[str, Any]:
        """Evaluate cases, optionally overriding the domain parameter."""
        orch = self._get_orchestrator()
        case_results: list[dict[str, Any]] = []

        for case in cases:
            result = orch.run(
                task=case.task,
                domain=domain_override,
                metadata=getattr(case, "metadata", None),
            )
            r = result.to_dict() if hasattr(result, "to_dict") else result

            r["expected_verdict"] = getattr(case, "expected_verdict", "")
            r["actual_verdict"] = r.get("verdict", "unknown")
            r["expected_eds_range"] = getattr(case, "expected_eds_range", None)
            case_results.append(r)

        return compute_all_metrics(case_results)
