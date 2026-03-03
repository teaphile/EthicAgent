"""Baselines — Comparative evaluation methods.

Implements baseline ethical decision methods for comparing
against the full EthicAgent framework:

1. **RandomBaseline** — assigns random verdicts (lower bound)
2. **RulesOnlyBaseline** — symbolic reasoning only (no LLM)
3. **LLMOnlyBaseline** — neural reasoning only (no rules)
4. **EqualWeightBaseline** — all philosophies weighted equally

Each baseline exposes an ``evaluate(case)`` method that accepts
a :class:`ScenarioCase` (or plain dict with ``task``/``domain``)
and returns a result dict compatible with the metrics module.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from ethicagent.utils.helpers import clamp, now_iso

logger = logging.getLogger(__name__)


# ── Random Baseline ──────────────────────────────────────────


class RandomBaseline:
    """Random baseline — assigns random verdicts and EDS scores.

    Serves as the lower bound on every metric.  If EthicAgent can't
    beat random, something is deeply wrong.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.name = "random"

    def evaluate(self, case: Any, **kwargs: Any) -> dict[str, Any]:
        """Evaluate a single case randomly.

        Args:
            case: A ScenarioCase or dict with ``task`` and ``domain``.
        """
        task, domain = _extract_task_domain(case)
        eds = self.rng.random()
        verdict = _eds_to_verdict(eds)

        return {
            "action_id": f"RND-{self.rng.randint(1, 99999):05d}",
            "eds_score": round(eds, 4),
            "verdict": verdict,
            "domain": domain,
            "confidence": round(self.rng.uniform(0.3, 0.9), 4),
            "philosophy_scores": {
                "deontological": round(self.rng.random(), 4),
                "consequentialist": round(self.rng.random(), 4),
                "virtue_ethics": round(self.rng.random(), 4),
                "contextual": round(self.rng.random(), 4),
            },
            "reasoning": "Random baseline — no real analysis performed.",
            "baseline": self.name,
            "timestamp": now_iso(),
        }


# ── Rules-Only Baseline ─────────────────────────────────────


class RulesOnlyBaseline:
    """Rules-only baseline — pure symbolic reasoning.

    No neural/LLM component.  Demonstrates the value of neural
    reasoning by showing what rules alone can (and can't) do.
    """

    def __init__(self, rules: dict | None = None) -> None:
        self.rules = rules or {}
        self.name = "rules_only"

    def evaluate(self, case: Any, **kwargs: Any) -> dict[str, Any]:
        task, domain = _extract_task_domain(case)

        try:
            from ethicagent.agents.symbolic_reasoner import SymbolicReasoner
            from ethicagent.knowledge.knowledge_graph import KnowledgeGraph

            kg = KnowledgeGraph()
            reasoner = SymbolicReasoner(rules=self.rules, knowledge_graph=kg)

            context = {"task": task, "domain": domain, **kwargs}
            result = reasoner.reason(context, domain)

            blocked = result.get("blocked", False)
            penalty = result.get("total_penalty", 0.0)

            if blocked:
                eds, verdict = 0.0, "hard_block"
            else:
                eds = clamp(1.0 - penalty, 0.0, 1.0)
                verdict = _eds_to_verdict(eds)

            return {
                "action_id": f"SYM-{abs(hash(task)) % 99999:05d}",
                "eds_score": round(eds, 4),
                "verdict": verdict,
                "domain": domain,
                "confidence": result.get("confidence", 0.5),
                "philosophy_scores": {
                    "deontological": round(eds, 4),
                    "consequentialist": 0.5,
                    "virtue_ethics": 0.5,
                    "contextual": 0.5,
                },
                "rules_triggered": [r.get("rule_id", "") for r in result.get("matched_rules", [])],
                "reasoning": f"Rules-only: penalty={penalty:.2f}",
                "baseline": self.name,
                "timestamp": now_iso(),
            }
        except Exception as exc:
            logger.warning("Rules-only baseline failed: %s", exc)
            return _fallback_result(task, domain, self.name, str(exc))


# ── LLM-Only Baseline ───────────────────────────────────────


class LLMOnlyBaseline:
    """LLM-only baseline — pure neural reasoning.

    No rule-based constraints.  Demonstrates the value of
    symbolic guardrails by showing LLM-alone failure modes.
    """

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.name = "llm_only"

    def evaluate(self, case: Any, **kwargs: Any) -> dict[str, Any]:
        task, domain = _extract_task_domain(case)

        try:
            from ethicagent.agents.neural_reasoner import NeuralReasoner

            reasoner = NeuralReasoner(config=self.config, use_llm=True)
            context = {"task": task, "domain": domain, **kwargs}
            result = reasoner.reason(context, domain)

            rec = result.get("recommendation", "escalate")
            confidence = result.get("confidence", 0.5)

            if rec == "approve":
                eds = clamp(0.7 + confidence * 0.3, 0.0, 1.0)
            elif rec in ("reject", "hard_block"):
                eds = clamp(0.3 - confidence * 0.2, 0.0, 1.0)
            else:
                eds = 0.6
            verdict = _eds_to_verdict(eds)

            return {
                "action_id": f"LLM-{abs(hash(task)) % 99999:05d}",
                "eds_score": round(eds, 4),
                "verdict": verdict,
                "domain": domain,
                "confidence": round(confidence, 4),
                "philosophy_scores": {
                    "deontological": 0.5,
                    "consequentialist": 0.5,
                    "virtue_ethics": 0.5,
                    "contextual": 0.5,
                },
                "reasoning": result.get("reasoning", "LLM-only evaluation."),
                "baseline": self.name,
                "timestamp": now_iso(),
            }
        except Exception as exc:
            logger.warning("LLM-only baseline failed: %s", exc)
            return _fallback_result(task, domain, self.name, str(exc))


# ── Equal-Weight Baseline ────────────────────────────────────


class EqualWeightBaseline:
    """Equal-weight baseline — all philosophies weighted 0.25.

    No domain-specific tuning.  Demonstrates the value of domain
    weights in the EDS formula.
    """

    def __init__(self, rules: dict | None = None) -> None:
        self.rules = rules or {}
        self.name = "equal_weight"

    def evaluate(self, case: Any, **kwargs: Any) -> dict[str, Any]:
        task, domain = _extract_task_domain(case)

        try:
            from ethicagent.agents.ethical_reasoner import EthicalReasonerAgent

            equal_weights = {
                d: {
                    "deontological": 0.25,
                    "consequentialist": 0.25,
                    "virtue_ethics": 0.25,
                    "contextual": 0.25,
                }
                for d in ["healthcare", "finance", "hiring", "disaster", "general"]
            }

            reasoner = EthicalReasonerAgent(
                rules=self.rules,
                domain_weights=equal_weights,
            )

            action_id = f"EQ-{abs(hash(task)) % 99999:05d}"
            context = {"task": task, "domain": domain, "action_id": action_id, **kwargs}
            fusion_result = {"recommendation": "escalate", "confidence": 0.5}

            decision = reasoner.evaluate(context, fusion_result, domain)

            return {
                "action_id": decision.action_id,
                "eds_score": round(decision.eds_score, 4),
                "verdict": decision.verdict.value
                if hasattr(decision.verdict, "value")
                else str(decision.verdict),
                "domain": domain,
                "confidence": round(decision.confidence, 4),
                "philosophy_scores": {
                    pr.name: round(pr.score, 4) for pr in decision.philosophy_results
                },
                "reasoning": decision.reasoning,
                "rules_triggered": decision.rules_triggered,
                "baseline": self.name,
                "timestamp": now_iso(),
            }
        except Exception as exc:
            logger.warning("Equal-weight baseline failed: %s", exc)
            return _fallback_result(task, domain, self.name, str(exc))


# ── Helpers ──────────────────────────────────────────────────


def _extract_task_domain(case: Any) -> tuple[str, str]:
    """Pull task and domain from a ScenarioCase or dict."""
    if isinstance(case, dict):
        return case.get("task", ""), case.get("domain", "general")
    return getattr(case, "task", ""), getattr(case, "domain", "general")


def _eds_to_verdict(eds: float) -> str:
    """Map an EDS score to a verdict string."""
    if eds >= 0.8:
        return "approve"
    if eds >= 0.5:
        return "escalate"
    return "reject"


def _fallback_result(
    task: str,
    domain: str,
    baseline_name: str,
    error: str,
) -> dict[str, Any]:
    """Return a safe result dict when a baseline raises."""
    return {
        "action_id": f"ERR-{abs(hash(task)) % 99999:05d}",
        "eds_score": 0.5,
        "verdict": "escalate",
        "domain": domain,
        "confidence": 0.0,
        "philosophy_scores": {},
        "reasoning": f"Baseline error: {error}",
        "baseline": baseline_name,
        "timestamp": now_iso(),
    }


def get_all_baselines(
    rules: dict | None = None,
    llm_config: dict | None = None,
) -> dict[str, Any]:
    """Get all baseline instances.

    Args:
        rules: Ethical rules configuration.
        llm_config: LLM configuration.

    Returns:
        Dictionary mapping baseline name to instance.
    """
    return {
        "random": RandomBaseline(),
        "rules_only": RulesOnlyBaseline(rules),
        "llm_only": LLMOnlyBaseline(llm_config),
        "equal_weight": EqualWeightBaseline(rules),
    }
