"""Fusion Agent — merges neural + symbolic reasoning (Stage 5).

The key insight that makes this more than a simple average: the
safety-first principle.  If the symbolic reasoner flags a hard rule
violation we *always* respect it, even if the LLM says "looks fine".

The LLM once approved denying insulin to a diabetic patient because
the prompt was ambiguous.  The symbolic rule caught it.  That's when
we made safety-first non-negotiable.

Conflict severity classification:
  • minor    — slight disagreement on confidence (Δ < 0.2)
  • moderate — different recommendations but same direction
  • severe   — one says approve, the other reject
  • critical — symbolic hard-block vs neural approve

# TODO: add a "negotiation" mode where the neural reasoner is re-prompted
#       with the symbolic constraints and asked to revise its assessment.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class FusionAgent:
    """Fuses neural and symbolic reasoning into a single recommendation.

    The fusion is asymmetric by design — symbolic rules have veto power
    on safety-critical issues, while the neural reasoner's confidence
    weights non-safety trade-offs.
    """

    def fuse(
        self,
        neural: dict[str, Any],
        symbolic: dict[str, Any],
        domain: str = "general",
    ) -> dict[str, Any]:
        """Merge neural + symbolic outputs.

        Returns a dict with: recommendation, confidence, agreement,
        conflict_severity, combined_reasoning, fused_scores, and
        safety_override flag.
        """
        n_rec = neural.get("recommendation", "escalate")
        n_conf = float(neural.get("confidence", neural.get("score", 0.5)))
        n_scores = neural.get("scores", {})
        n_reasoning = neural.get("reasoning", "")

        s_status = symbolic.get("status", "compliant")
        s_blocked = symbolic.get("blocked", symbolic.get("hard_block", False))
        s_score = float(symbolic.get("symbolic_score", symbolic.get("score", 1.0)))
        s_matched = symbolic.get("matched_rules", [])
        s_explanations = symbolic.get("explanations", [])

        # -- map symbolic status → a recommendation --------------------------
        s_rec = self._symbolic_to_recommendation(s_status, s_blocked)

        # -- detect agreement / conflict -------------------------------------
        agreement = n_rec == s_rec
        conflict_severity = self._compute_conflict_severity(
            n_rec, s_rec, n_conf, s_score, s_blocked
        )

        # -- safety-first override -------------------------------------------
        # Symbolic ALWAYS wins on safety-critical rules
        safety_override = False
        if s_blocked:
            final_rec = "reject"
            final_conf = 0.95
            safety_override = True
            logger.warning(
                "Safety-first override: symbolic hard-block takes priority "
                "over neural recommendation"
            )
        elif agreement:
            final_rec = n_rec
            # agreement boosts confidence
            final_conf = min((n_conf + s_score) / 2 + 0.1, 1.0)
        else:
            # disagreement — use weighted fusion
            final_rec, final_conf = self._weighted_fusion(n_rec, n_conf, s_rec, s_score, domain)

        # -- fuse scores (average neural + symbolic where available) ----------
        fused_scores = dict(n_scores)  # start with neural scores
        if fused_scores:
            # dampen neural scores by symbolic compliance
            for k in fused_scores:
                fused_scores[k] = round(fused_scores[k] * s_score, 3)

        # -- combined reasoning -----------------------------------------------
        combined_reasoning = self._combine_reasoning(
            n_reasoning, s_explanations, agreement, safety_override
        )

        result: dict[str, Any] = {
            "recommendation": final_rec,
            "confidence": round(final_conf, 3),
            "score": round(final_conf, 3),  # backward-compat alias
            "fused_score": round(final_conf, 3),  # backward-compat alias
            "agreement": agreement,
            "conflict_severity": conflict_severity,
            "safety_override": safety_override,
            "fused_scores": fused_scores,
            "neural_recommendation": n_rec,
            "symbolic_status": s_status,
            "rules_matched": len(s_matched),
            "combined_reasoning": combined_reasoning,
        }

        # If safety override (hard block), score should reflect rejection
        if safety_override:
            result["score"] = 0.0
            result["fused_score"] = 0.0

        logger.info(
            f"Fusion: neural={n_rec}({n_conf:.2f}) + "
            f"symbolic={s_rec}({s_score:.2f}) → "
            f"{final_rec}({final_conf:.2f}), "
            f"agreement={agreement}, severity={conflict_severity}"
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _symbolic_to_recommendation(status: str, blocked: bool) -> str:
        if blocked:
            return "reject"
        if status == "compliant":
            return "approve"
        if status == "caution":
            return "escalate"
        return "escalate"

    @staticmethod
    def _compute_conflict_severity(
        n_rec: str, s_rec: str, n_conf: float, s_score: float, s_blocked: bool
    ) -> str:
        """Classify how bad the disagreement is."""
        if n_rec == s_rec:
            return "none"

        if s_blocked and n_rec == "approve":
            return "critical"  # most dangerous type

        opposite = (n_rec == "approve" and s_rec == "reject") or (
            n_rec == "reject" and s_rec == "approve"
        )
        if opposite:
            return "severe"

        # one is escalate, the other approve/reject
        if abs(n_conf - s_score) > 0.3:
            return "moderate"
        return "minor"

    def _weighted_fusion(
        self,
        n_rec: str,
        n_conf: float,
        s_rec: str,
        s_score: float,
        domain: str,
    ) -> tuple:
        """Weighted fusion for non-safety disagreements.

        In safety-critical domains (healthcare, disaster) we lean
        more toward the symbolic reasoner.
        """
        # domain-dependent weights for neural vs symbolic
        sym_weight = {
            "healthcare": 0.6,
            "disaster": 0.6,
            "finance": 0.5,
            "hiring": 0.5,
            "general": 0.5,
        }.get(domain, 0.5)
        n_weight = 1.0 - sym_weight

        # map recommendations to numeric: approve=1, escalate=0.5, reject=0
        rec_map = {"approve": 1.0, "escalate": 0.5, "reject": 0.0}
        n_val = rec_map.get(n_rec, 0.5) * n_conf
        s_val = rec_map.get(s_rec, 0.5) * s_score

        fused_val = n_weight * n_val + sym_weight * s_val

        # map back to recommendation
        if fused_val >= 0.65:
            rec = "approve"
        elif fused_val >= 0.35:
            rec = "escalate"
        else:
            rec = "reject"

        conf = abs(fused_val - 0.5) * 2  # higher when more decisive
        conf = max(0.3, min(conf, 0.9))
        return rec, round(conf, 3)

    @staticmethod
    def _combine_reasoning(
        neural_reasoning: str,
        symbolic_explanations: list[str],
        agreement: bool,
        safety_override: bool,
    ) -> str:
        parts = []
        if safety_override:
            parts.append(
                "⚠️ SAFETY OVERRIDE: Symbolic rules flagged a hard "
                "violation that takes priority over neural assessment."
            )
        if neural_reasoning:
            parts.append(f"Neural analysis: {neural_reasoning[:300]}")
        if symbolic_explanations:
            parts.append("Symbolic rules: " + "; ".join(symbolic_explanations[:5]))
        if agreement:
            parts.append("Both reasoners agree on this assessment.")
        else:
            parts.append(
                "Neural and symbolic reasoners disagreed — "
                "fusion arbitrated based on confidence weights."
            )
        return " | ".join(parts)
