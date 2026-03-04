"""Action Executor — decision gate routing (Stage 7).

Routes the ethical verdict to one of four handlers:
  approve    → log + execute
  escalate   → queue for human review
  reject     → log + block + suggest alternatives
  hard_block → log + block + alert

# NOTE: the "execute" part is intentionally a no-op in this research
#       framework — in production you'd wire it to the actual agent
#       action system.
"""

from __future__ import annotations

import logging
from typing import Any

from ethicagent.core.logger import AuditLogger
from ethicagent.ethics.ethical_score import APPROVAL_THRESHOLD

logger = logging.getLogger(__name__)


class ActionExecutor:
    """Routes ethical verdicts to appropriate handlers."""

    def __init__(
        self,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self._audit = audit_logger

    def execute(
        self,
        decision: Any = None,
        context: dict[str, Any] | None = None,
        *,
        verdict: str | None = None,
        eds_score: float | None = None,
        task: str | None = None,
        reasoning: str | None = None,
    ) -> dict[str, Any]:
        """Execute the decision based on the verdict.

        Accepts either:
          execute(decision_obj, context_dict)        — canonical form
          execute(verdict=..., eds_score=..., ...)   — keyword form (backward-compat)

        Returns a result dict with status, actions taken, and
        any warnings or alternatives.
        """
        # Keyword-arg form: build a simple namespace
        if verdict is not None or (decision is None and context is None):
            from types import SimpleNamespace

            v = verdict or "escalate"
            # Map simple verdict names to enum-style
            verdict_map = {
                "approve": "auto_approve",
                "reject": "reject",
                "escalate": "escalate",
                "hard_block": "hard_block",
            }
            mapped = verdict_map.get(v, v)
            decision = SimpleNamespace(
                verdict=SimpleNamespace(value=mapped),
                eds_score=eds_score or 0.0,
                reasoning=reasoning or "",
                philosophy_results=[],
                conflict_analysis={},
                rules_triggered=[],
            )
            context = {
                "action": task or "",
                "domain": "general",
            }

        verdict_val = decision.verdict.value if hasattr(decision, "verdict") else "unknown"
        eds = getattr(decision, "eds_score", 0.0)

        handlers = {
            "auto_approve": self._handle_approve,
            "escalate": self._handle_escalate,
            "reject": self._handle_reject,
            "hard_block": self._handle_hard_block,
        }

        handler = handlers.get(verdict_val, self._handle_unknown)
        result = handler(decision, context)

        # Add verdict/action to result for backward compat
        result.setdefault("verdict", verdict_val)
        result.setdefault("action", verdict_val.replace("auto_", ""))

        # audit log
        if self._audit:
            self._audit.log_decision(
                run_id=context.get("action_id", "unknown"),
                action=context.get("action", "")[:200],
                domain=context.get("domain", "general"),
                ethical_scores={
                    pr.name: pr.score for pr in getattr(decision, "philosophy_results", [])
                },
                verdict=verdict_val,
                reasoning=getattr(decision, "reasoning", ""),
                eds_score=eds,
                rules_triggered=getattr(decision, "rules_triggered", []),
            )

        return result

    # -- verdict handlers ------------------------------------------------------

    def _handle_approve(self, decision: Any, ctx: dict[str, Any]) -> dict[str, Any]:
        warnings = self._extract_warnings(decision, ctx)
        logger.info(f"Action APPROVED (EDS={getattr(decision, 'eds_score', 0):.3f})")
        return {
            "status": "approved",
            "requires_human_review": False,
            "warnings": warnings,
            "message": "Action approved — all ethical checks passed.",
        }

    def _handle_escalate(self, decision: Any, ctx: dict[str, Any]) -> dict[str, Any]:
        priority = self._compute_review_priority(decision)
        logger.info(f"Action ESCALATED for human review (priority={priority})")
        return {
            "status": "escalated",
            "requires_human_review": True,
            "review_priority": priority,
            "message": (
                "Action requires human review — ethical scores are in the ambiguous range."
            ),
        }

    def _handle_reject(self, decision: Any, ctx: dict[str, Any]) -> dict[str, Any]:
        alternatives = self._suggest_alternatives(decision, ctx)
        logger.warning(f"Action REJECTED (EDS={getattr(decision, 'eds_score', 0):.3f})")
        return {
            "status": "rejected",
            "requires_human_review": False,
            "alternatives": alternatives,
            "message": "Action rejected — ethical evaluation below threshold.",
        }

    def _handle_hard_block(self, decision: Any, ctx: dict[str, Any]) -> dict[str, Any]:
        rules = getattr(decision, "rules_triggered", [])
        logger.error(f"Action HARD-BLOCKED — rule violations: {rules}")
        return {
            "status": "hard_blocked",
            "requires_human_review": False,
            "rules_violated": rules,
            "message": (
                f"Hard rule violation detected: {', '.join(rules[:3]) or 'undisclosed'}. "
                "This action is blocked regardless of other scores."
            ),
        }

    def _handle_unknown(self, decision: Any, ctx: dict[str, Any]) -> dict[str, Any]:
        logger.warning("Unknown verdict — defaulting to escalate")
        return {
            "status": "escalated",
            "requires_human_review": True,
            "message": "Unknown verdict — escalated for safety.",
        }

    # -- helpers ---------------------------------------------------------------

    def _extract_warnings(self, decision: Any, ctx: dict[str, Any]) -> list[str]:
        warnings = []
        eds = getattr(decision, "eds_score", 1.0)
        if APPROVAL_THRESHOLD <= eds < APPROVAL_THRESHOLD + 0.05:
            warnings.append("EDS is just above the approval threshold.")
        conflict = getattr(decision, "conflict_analysis", {})
        if conflict.get("severity") in ("moderate", "severe"):
            warnings.append(f"Philosophical disagreement: {conflict.get('severity')}.")
        if ctx.get("reversibility") == "irreversible":
            warnings.append("This action is irreversible — proceed with caution.")
        return warnings

    def _compute_review_priority(self, decision: Any) -> str:
        eds = getattr(decision, "eds_score", 0.5)
        if eds < 0.55:
            return "high"
        elif eds < 0.70:
            return "medium"
        return "low"

    @staticmethod
    def _suggest_alternatives(decision: Any, ctx: dict[str, Any]) -> list[str]:
        alts = []
        domain = ctx.get("domain", "general")
        if domain == "healthcare":
            alts.append("Consider seeking a second medical opinion.")
            alts.append("Consult the hospital ethics committee.")
        elif domain == "finance":
            alts.append("Offer the applicant an alternative product.")
            alts.append("Request additional documentation for review.")
        elif domain == "hiring":
            alts.append("Re-evaluate the candidate against objective criteria.")
            alts.append("Consider a skills-based assessment.")
        elif domain == "disaster":
            alts.append("Re-prioritize using updated triage criteria.")
            alts.append("Escalate to the incident commander.")
        else:
            alts.append("Seek additional human review before proceeding.")
        return alts
