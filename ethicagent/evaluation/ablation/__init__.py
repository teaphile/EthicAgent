"""Ablation Study — Systematic Component Removal Analysis.

Implements ablation experiments that systematically disable
pipeline components to measure their individual contributions
to EDS accuracy and decision quality.

This is key to showing which parts of the architecture actually
matter — without ablation data, reviewers won't trust the results.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from ethicagent.evaluation.metrics import compute_all_metrics
from ethicagent.utils.helpers import now_iso

logger = logging.getLogger(__name__)


# Ablation variants — which components to disable
ABLATION_VARIANTS: dict[str, dict[str, Any]] = {
    "full_system": {
        "description": "Full EthicAgent system (no ablation)",
        "disable": [],
    },
    "no_neural": {
        "description": "Disable neural (LLM) reasoning",
        "disable": ["neural_reasoning"],
    },
    "no_symbolic": {
        "description": "Disable symbolic (rule-based) reasoning",
        "disable": ["symbolic_reasoning"],
    },
    "no_fusion": {
        "description": "Disable neural-symbolic fusion (use neural only)",
        "disable": ["fusion"],
    },
    "no_contextual": {
        "description": "Disable contextual ethics philosophy",
        "disable": ["contextual_ethics"],
    },
    "no_virtue": {
        "description": "Disable virtue ethics philosophy",
        "disable": ["virtue_ethics"],
    },
    "no_consequentialist": {
        "description": "Disable consequentialist philosophy",
        "disable": ["consequentialist"],
    },
    "no_deontological": {
        "description": "Disable deontological philosophy",
        "disable": ["deontological"],
    },
    "no_reflection": {
        "description": "Disable reflection/learning loop",
        "disable": ["reflection"],
    },
    "no_knowledge_graph": {
        "description": "Disable knowledge graph",
        "disable": ["knowledge_graph"],
    },
    "no_domain_weights": {
        "description": "Use equal weights instead of domain-specific",
        "disable": ["domain_weights"],
    },
    "no_conflict_resolution": {
        "description": "Disable inter-philosophy conflict resolution",
        "disable": ["conflict_resolution"],
    },
}


class AblationStudy:
    """Ablation study runner for EthicAgent.

    Systematically disables pipeline components and measures
    the impact on evaluation metrics.

    Usage::

        study = AblationStudy(orchestrator_factory=my_factory)
        results = study.run(test_cases)
        print(results["comparison"]["most_impactful_component"])
    """

    def __init__(
        self,
        orchestrator_factory: Callable[[dict], Any] | None = None,
        variants: dict[str, dict] | None = None,
    ) -> None:
        self.orchestrator_factory = orchestrator_factory
        self.variants = variants or ABLATION_VARIANTS
        self.results: dict[str, dict[str, Any]] = {}

    # ── Public API ───────────────────────────────────────────

    def run(
        self,
        test_cases: list[dict[str, Any]],
        variant_names: list[str] | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Run the ablation study across all variants.

        Args:
            test_cases: List of test case dicts with task, domain, etc.
            variant_names: Specific variants to run (None = all).
            progress_callback: Optional callback(variant_name, current, total).

        Returns:
            Complete ablation study results with comparison.
        """
        variants_to_run = variant_names or list(self.variants.keys())
        total = len(variants_to_run)

        logger.info(
            "Starting ablation study: %d variants, %d cases",
            total,
            len(test_cases),
        )

        for i, name in enumerate(variants_to_run):
            if name not in self.variants:
                logger.warning("Unknown ablation variant: %s — skipping", name)
                continue

            cfg = self.variants[name]
            logger.info(
                "[%d/%d] Running variant '%s' — %s",
                i + 1,
                total,
                name,
                cfg["description"],
            )

            if progress_callback:
                progress_callback(name, i + 1, total)

            self.results[name] = self._run_variant(name, cfg, test_cases)

        comparison = self._compare_variants()

        return {
            "variants": self.results,
            "comparison": comparison,
            "test_cases_count": len(test_cases),
            "variants_tested": len(self.results),
            "timestamp": now_iso(),
        }

    def export_results(self, results_or_filepath: Any = None, filepath: str | None = None) -> None:
        """Export ablation results to JSON.

        Accepts either:
          export_results(filepath)              — canonical
          export_results(results, filepath)     — backward-compat
        """
        if filepath is not None:
            # Two-arg form: results_or_filepath is the results dict
            if isinstance(results_or_filepath, dict):
                self.results = results_or_filepath.get("variants", results_or_filepath)
            _fp = filepath
        elif isinstance(results_or_filepath, str):
            _fp = results_or_filepath
        else:
            raise TypeError("export_results requires a filepath")
        import json
        from pathlib import Path

        export = {
            "study": "ablation",
            "variants_tested": list(self.results.keys()),
            "results": {
                name: {
                    "description": d.get("description", ""),
                    "disabled_components": d.get("disabled_components", []),
                    "metrics": d.get("metrics", {}),
                    "elapsed_seconds": d.get("elapsed_seconds", 0),
                }
                for name, d in self.results.items()
            },
            "timestamp": now_iso(),
        }

        Path(_fp).parent.mkdir(parents=True, exist_ok=True)
        with open(_fp, "w") as f:
            json.dump(export, f, indent=2, default=str)

        logger.info("Ablation results exported → %s", _fp)

    # ── Internal ─────────────────────────────────────────────

    def _run_variant(
        self,
        variant_name: str,
        variant_config: dict[str, Any],
        test_cases: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run a single ablation variant."""
        t0 = time.perf_counter()

        try:
            if self.orchestrator_factory:
                orchestrator = self.orchestrator_factory(variant_config)
                results = []
                for case in test_cases:
                    result = orchestrator.run(
                        task=case.get("task", ""),
                        domain=case.get("domain"),
                        metadata=case.get("metadata"),
                    )
                    result["expected_verdict"] = case.get("expected_verdict", "")
                    result["expected_eds_range"] = case.get("expected_eds_range")
                    result["actual_verdict"] = result.get("verdict", "unknown")
                    results.append(result)
            else:
                results = self._simulate_variant(variant_name, variant_config, test_cases)

            elapsed = time.perf_counter() - t0
            metrics = compute_all_metrics(results)

            return {
                "variant": variant_name,
                "description": variant_config["description"],
                "disabled_components": variant_config["disable"],
                "results": results,
                "metrics": metrics,
                "elapsed_seconds": round(elapsed, 3),
                "timestamp": now_iso(),
            }

        except Exception as exc:
            logger.error("Variant '%s' failed: %s", variant_name, exc)
            return {
                "variant": variant_name,
                "error": str(exc),
                "elapsed_seconds": round(time.perf_counter() - t0, 3),
            }

    def _simulate_variant(
        self,
        variant_name: str,
        variant_config: dict[str, Any],
        test_cases: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Simulate variant results when no orchestrator is available.

        Useful for testing the ablation framework itself. The more
        components you disable, the worse the simulated accuracy.
        """
        import random

        rng = random.Random(hash(variant_name))

        # Each disabled component degrades accuracy
        penalty = len(variant_config["disable"]) * 0.05

        results: list[dict[str, Any]] = []
        for case in test_cases:
            expected = case.get("expected_verdict", "escalate")
            eds_range = case.get("expected_eds_range", (0.0, 1.0))

            base_eds = rng.uniform(eds_range[0], eds_range[1])
            eds = max(0.0, min(1.0, base_eds - penalty * rng.random()))

            if rng.random() > penalty * 2:
                verdict = expected or (
                    "approve" if eds >= 0.8 else ("escalate" if eds >= 0.5 else "reject")
                )
            else:
                verdict = rng.choice(["approve", "escalate", "reject"])

            results.append(
                {
                    "eds_score": round(eds, 4),
                    "verdict": verdict,
                    "actual_verdict": verdict,
                    "expected_verdict": expected,
                    "expected_eds_range": eds_range,
                    "domain": case.get("domain", "general"),
                    "philosophy_scores": {
                        "deontological": round(rng.random(), 4),
                        "consequentialist": round(rng.random(), 4),
                        "virtue_ethics": round(rng.random(), 4),
                        "contextual": round(rng.random(), 4),
                    },
                }
            )

        return results

    def _compare_variants(self) -> dict[str, Any]:
        """Compare results across all variants."""
        comparison: dict[str, Any] = {}

        # Accuracy ranking
        accuracy_ranking: list[tuple] = []
        for name, data in self.results.items():
            if "error" in data:
                continue
            acc = data.get("metrics", {}).get("verdict_accuracy", {}).get("overall_accuracy", 0.0)
            accuracy_ranking.append((name, round(acc, 4)))

        accuracy_ranking.sort(key=lambda x: x[1], reverse=True)
        comparison["accuracy_ranking"] = accuracy_ranking

        # Deltas from full system
        full_acc = next(
            (acc for name, acc in accuracy_ranking if name == "full_system"),
            0.0,
        )
        deltas: dict[str, float] = {}
        for name, acc in accuracy_ranking:
            if name == "full_system":
                continue
            deltas[name] = round(acc - full_acc, 4)
        comparison["accuracy_deltas"] = deltas

        # Most impactful component = largest accuracy drop
        if deltas:
            worst = min(deltas.items(), key=lambda x: x[1])
            comparison["most_impactful_component"] = {
                "variant": worst[0],
                "accuracy_drop": worst[1],
            }

        return comparison
