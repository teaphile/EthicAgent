"""Reflection Agent — post-decision learning loop (Stage 8).

After the decision gate, this agent:
  • Records the decision as a precedent
  • Checks consistency against past rulings
  • Detects emerging bias patterns
  • Generates recommendations for system improvement

The reflection loop is what makes EthicAgent *learn* from its own
decisions (in a limited, non-gradient sense).  It's closest to
case-based reasoning in the AI lit.

# NOTE: this doesn't update model weights — it adjusts memory and
#       precedent stores that influence future evaluations.
# FIXME: bias detection is simplistic (just checks verdict distribution
#        per domain).  Should add demographic slicing.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ReflectionAgent:
    """Post-decision reflection and learning.

    Maintains a rolling history of past decisions and uses it to
    detect inconsistencies, bias patterns, and improvement opportunities.
    """

    def __init__(
        self,
        precedent_store: Any = None,
        memory_store: Any = None,
        history_limit: int = 500,
    ) -> None:
        self._precedent_store = precedent_store
        self._memory_store = memory_store
        self._history: List[Dict[str, Any]] = []
        self._history_limit = history_limit
        self._domain_verdicts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    # ------------------------------------------------------------------

    def reflect(
        self,
        decision: Any,
        context: Dict[str, Any],
        execution_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the full reflection loop.

        Returns a report dict with consistency analysis, bias checks,
        recommendations, and memory updates.
        """
        domain = context.get("domain", "general")
        verdict = (
            decision.verdict.value
            if hasattr(decision, "verdict") else str(decision)
        )
        eds = getattr(decision, "eds_score", 0.0)

        record = {
            "domain": domain,
            "verdict": verdict,
            "eds_score": eds,
            "action": context.get("action", "")[:200],
            "urgency": context.get("urgency", "normal"),
        }

        # -- 1. Record as precedent -------------------------------------------
        self._record_precedent(record)

        # -- 2. Consistency analysis ------------------------------------------
        consistency = self._analyse_consistency(record)

        # -- 3. Bias detection ------------------------------------------------
        bias = self._detect_bias_patterns(domain)

        # -- 4. Recommendations -----------------------------------------------
        recs = self._generate_recommendations(consistency, bias, record)

        # -- 5. Memory update -------------------------------------------------
        self._update_memory(record, context)

        # -- 6. Track in history ----------------------------------------------
        self._add_to_history(record)

        report = {
            "consistency_analysis": consistency,
            "bias_detection": bias,
            "recommendations": recs,
            "precedents_recorded": True,
            "memory_updated": self._memory_store is not None,
            "history_size": len(self._history),
        }

        logger.info(
            f"Reflection complete: consistent={consistency.get('is_consistent')}, "
            f"bias_flags={len(bias.get('flags', []))}, "
            f"recs={len(recs)}"
        )
        return report

    def get_learning_summary(self) -> Dict[str, Any]:
        """Aggregate stats from all reflections."""
        if not self._history:
            return {"total_decisions": 0, "domains": {}}

        verdict_dist: Dict[str, int] = defaultdict(int)
        for h in self._history:
            verdict_dist[h.get("verdict", "unknown")] += 1

        return {
            "total_decisions": len(self._history),
            "verdict_distribution": dict(verdict_dist),
            "domain_stats": {
                d: dict(vd) for d, vd in self._domain_verdicts.items()
            },
        }

    # ---------- Backward-compat public wrappers (used by tests) ----------

    @property
    def decision_history(self) -> List[Dict[str, Any]]:
        """Public access to history (backward-compat)."""
        return self._history

    def record_decision(
        self,
        task: str = "",
        domain: str = "general",
        verdict: str = "escalate",
        eds_score: float = 0.0,
        philosophy_scores: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """Record a decision directly (backward-compat).

        Simpler than the full reflect() pipeline — just records to history.
        """
        record = {
            "domain": domain,
            "verdict": verdict,
            "eds_score": eds_score,
            "action": task[:200],
            "urgency": kwargs.get("urgency", "normal"),
            "philosophy_scores": philosophy_scores or {},
        }
        self._add_to_history(record)

    def analyze_consistency(self) -> Dict[str, Any]:
        """Run consistency analysis over all recorded history."""
        if not self._history:
            return {"is_consistent": True, "similar_cases": 0, "note": "no history"}
        # Analyze the most recent record against all prior ones
        return self._analyse_consistency(self._history[-1])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_precedent(self, record: Dict[str, Any]) -> None:
        if self._precedent_store:
            try:
                self._precedent_store.add_precedent(
                    case_id=record.get("case_id", ""),
                    task=record.get("action", ""),
                    domain=record.get("domain", "general"),
                    verdict=record.get("verdict", ""),
                    eds_score=record.get("eds_score", 0.0),
                    reasoning=record.get("reasoning", ""),
                )
            except Exception as exc:
                logger.debug(f"Precedent store write failed: {exc}")

    def _analyse_consistency(
        self, record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if this decision is consistent with past similar cases."""
        domain = record["domain"]
        verdict = record["verdict"]

        # find similar past decisions in same domain
        similar = [
            h for h in self._history
            if h.get("domain") == domain
        ]
        if not similar:
            return {"is_consistent": True, "similar_cases": 0, "note": "first case in domain"}

        # check verdict distribution for this domain
        same_verdict = sum(1 for h in similar if h.get("verdict") == verdict)
        ratio = same_verdict / len(similar)

        is_consistent = ratio > 0.3  # at least 30% precedent
        return {
            "is_consistent": is_consistent,
            "similar_cases": len(similar),
            "same_verdict_ratio": round(ratio, 3),
            "note": (
                "Consistent with prior decisions"
                if is_consistent
                else f"Unusual verdict — only {ratio:.0%} of similar cases got '{verdict}'"
            ),
        }

    def _detect_bias_patterns(
        self, domain: str
    ) -> Dict[str, Any]:
        """Look for skewed verdict distributions that might indicate bias."""
        flags: List[str] = []
        dist = dict(self._domain_verdicts.get(domain, {}))
        total = sum(dist.values())

        if total < 10:
            return {"flags": [], "note": "Not enough data for bias detection"}

        for verdict, count in dist.items():
            fraction = count / total
            if fraction > 0.8:
                flags.append(
                    f"Domain '{domain}' overwhelmingly produces '{verdict}' "
                    f"({fraction:.0%}).  Possible systematic bias."
                )
            elif fraction < 0.05 and verdict in ("escalate",):
                flags.append(
                    f"Very few escalations in '{domain}' — "
                    f"system may be overconfident."
                )

        return {"flags": flags, "distribution": dist}

    def _generate_recommendations(
        self,
        consistency: Dict[str, Any],
        bias: Dict[str, Any],
        record: Dict[str, Any],
    ) -> List[str]:
        recs: List[str] = []
        if not consistency.get("is_consistent"):
            recs.append(
                "Review this decision — it's inconsistent with prior "
                "rulings in the same domain."
            )
        for flag in bias.get("flags", []):
            recs.append(f"Bias alert: {flag}")

        if record.get("eds_score", 1.0) < 0.6:
            recs.append(
                "Low EDS score — consider additional human review."
            )
        return recs

    def _update_memory(
        self, record: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        if self._memory_store:
            try:
                self._memory_store.store_from_dict({
                    "task": context.get("action", ""),
                    "domain": record.get("domain", "general"),
                    "verdict": record.get("verdict", ""),
                    "eds_score": record.get("eds_score", 0.0),
                })
            except Exception as exc:
                logger.debug(f"Memory store write failed: {exc}")

    def _add_to_history(self, record: Dict[str, Any]) -> None:
        self._history.append(record)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

        domain = record.get("domain", "general")
        verdict = record.get("verdict", "unknown")
        self._domain_verdicts[domain][verdict] += 1
