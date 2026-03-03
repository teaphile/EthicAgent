"""Multi-philosophy conflict resolution module.

When different ethical lenses disagree — and they *will* disagree —
this module decides how to reconcile them.  This is arguably the most
interesting part of the system, because it's where the hard tradeoffs
happen.

Resolution strategies:
  1. **Weighted voting** — each philosophy gets a vote weighted by domain
  2. **Priority override** — deontological hard-blocks always win
  3. **Deliberative** — look for the majority position and explain dissent
  4. **Moral uncertainty** — compute a formal uncertainty score

The conflict resolver doesn't just pick a winner; it produces a
detailed record of the disagreement and its resolution, so that
human reviewers can audit the reasoning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ConflictSeverity(Enum):
    NONE = "none"  # All lenses agree
    MINOR = "minor"  # Small score differences, same verdict direction
    MODERATE = "moderate"  # Different recommendations but no hard-block
    SEVERE = "severe"  # One lens blocks, others permit
    CRITICAL = "critical"  # Hard-block from deontological


class ResolutionStrategy(Enum):
    WEIGHTED_VOTE = "weighted_vote"
    PRIORITY_OVERRIDE = "priority_override"
    DELIBERATIVE = "deliberative"
    MORAL_UNCERTAINTY = "moral_uncertainty"


@dataclass
class PhilosophyPosition:
    """One philosophy's stance on an action."""

    philosophy: str  # "deontological", "consequentialist", "virtue", "contextual"
    score: float  # 0-1
    verdict: str  # "approve", "escalate", "reject", "hard-block"
    confidence: float  # 0-1
    key_argument: str  # one-sentence summary
    weight: float = 0.25  # domain-adjusted weight


@dataclass
class ConflictRecord:
    """Record of a conflict between philosophies and how it was resolved."""

    severity: ConflictSeverity
    strategy_used: ResolutionStrategy
    positions: list[PhilosophyPosition]
    resolved_verdict: str
    resolved_score: float
    dissenting_views: list[str]
    moral_uncertainty: float  # 0-1, higher = more uncertain
    explanation: str = ""


# ---------------------------------------------------------------------------
# Domain weight presets
# ---------------------------------------------------------------------------

_DOMAIN_WEIGHTS: dict[str, dict[str, float]] = {
    "healthcare": {
        "deontological": 0.35,
        "consequentialist": 0.25,
        "virtue": 0.20,
        "contextual": 0.20,
    },
    "finance": {
        "deontological": 0.20,
        "consequentialist": 0.25,
        "virtue": 0.35,
        "contextual": 0.20,
    },
    "hiring": {
        "deontological": 0.15,
        "consequentialist": 0.20,
        "virtue": 0.40,
        "contextual": 0.25,
    },
    "disaster": {
        "deontological": 0.20,
        "consequentialist": 0.35,
        "virtue": 0.15,
        "contextual": 0.30,
    },
    "general": {
        "deontological": 0.25,
        "consequentialist": 0.25,
        "virtue": 0.25,
        "contextual": 0.25,
    },
}


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class ConflictResolver:
    """Resolves inter-philosophy conflicts to produce a unified verdict.

    The resolver selects a strategy based on conflict severity:
    - NONE/MINOR → weighted_vote (simple, all agree)
    - MODERATE → deliberative (explain the disagreement)
    - SEVERE → moral_uncertainty (formally acknowledge the tension)
    - CRITICAL → priority_override (deontological hard-block wins)
    """

    def __init__(
        self,
        domain_weights: dict[str, dict[str, float]] | None = None,
        conflict_history_limit: int = 50,
    ) -> None:
        self.domain_weights = domain_weights or dict(_DOMAIN_WEIGHTS)
        self._history: list[ConflictRecord] = []
        self._history_limit = conflict_history_limit

    # ------------------------------------------------------------------
    # Dict-based convenience API (used by tests)
    # ------------------------------------------------------------------

    def detect_conflicts(self, scores: dict[str, float]) -> list[dict[str, Any]]:
        """Detect pairwise conflicts in a plain scores dict.

        Returns a list of conflict dicts, each with 'philosophies',
        'difference', and 'severity' keys.
        """
        conflicts: list[dict[str, Any]] = []
        philosophies = list(scores.keys())
        for i, p1 in enumerate(philosophies):
            for p2 in philosophies[i + 1 :]:
                diff = abs(scores[p1] - scores[p2])
                if diff < 0.15:
                    continue  # no meaningful conflict
                severity = "minor"
                if diff >= 0.5:
                    severity = "severe"
                elif diff >= 0.3:
                    severity = "moderate"
                conflicts.append(
                    {
                        "philosophies": (p1, p2),
                        "difference": round(diff, 4),
                        "severity": severity,
                    }
                )
        return conflicts

    def resolve(
        self,
        positions_or_scores: list[PhilosophyPosition] | dict[str, float],
        domain: str = "general",
        force_strategy: ResolutionStrategy | None = None,
    ) -> ConflictRecord | dict[str, Any]:
        """Resolve conflict between philosophy positions.

        Accepts either:
          resolve(positions: list[PhilosophyPosition])  — canonical
          resolve(scores: dict[str, float])              — simplified
        """
        # Simplified dict form
        if isinstance(positions_or_scores, dict):
            return self._resolve_from_dict(positions_or_scores, domain)

        positions = positions_or_scores
        return self._resolve_canonical(positions, domain, force_strategy)

    def _resolve_from_dict(
        self, scores: dict[str, float], domain: str = "general"
    ) -> dict[str, Any]:
        """Resolve from a plain scores dict — return a plain dict."""
        positions = [
            PhilosophyPosition(
                philosophy=name,
                score=score,
                verdict="approve" if score >= 0.7 else ("escalate" if score >= 0.4 else "reject"),
                confidence=0.7,
                key_argument="",
            )
            for name, score in scores.items()
        ]
        record = self._resolve_canonical(positions, domain)
        conflicts = self.detect_conflicts(scores)
        return {
            "resolved_scores": {p.philosophy: p.score for p in record.positions},
            "resolution": record.resolved_verdict,
            "conflicts": conflicts,
            "severity": record.severity.value,
            "strategy": record.strategy_used.value,
            "moral_uncertainty": record.moral_uncertainty,
            "explanation": record.explanation,
        }

    def _resolve_canonical(
        self,
        positions: list[PhilosophyPosition],
        domain: str = "general",
        force_strategy: ResolutionStrategy | None = None,
    ) -> ConflictRecord:
        """Internal canonical resolve using PhilosophyPosition list."""
        # Apply domain weights
        weights = self.domain_weights.get(domain, self.domain_weights["general"])
        for pos in positions:
            if pos.philosophy in weights:
                pos.weight = weights[pos.philosophy]

        # Classify conflict severity
        severity = self._classify_severity(positions)

        # Select resolution strategy
        strategy = force_strategy or self._select_strategy(severity, positions)

        # Execute strategy
        if strategy == ResolutionStrategy.PRIORITY_OVERRIDE:
            verdict, score = self._priority_override(positions)
        elif strategy == ResolutionStrategy.DELIBERATIVE:
            verdict, score = self._deliberative(positions)
        elif strategy == ResolutionStrategy.MORAL_UNCERTAINTY:
            verdict, score = self._moral_uncertainty_resolve(positions)
        else:
            verdict, score = self._weighted_vote(positions)

        # Compute moral uncertainty
        uncertainty = self._compute_moral_uncertainty(positions)

        # Identify dissenting views
        dissent = self._identify_dissent(positions, verdict)

        # Build explanation
        explanation = self._build_explanation(
            severity,
            strategy,
            positions,
            verdict,
            score,
            uncertainty,
            dissent,
        )

        record = ConflictRecord(
            severity=severity,
            strategy_used=strategy,
            positions=positions,
            resolved_verdict=verdict,
            resolved_score=round(score, 4),
            dissenting_views=dissent,
            moral_uncertainty=round(uncertainty, 4),
            explanation=explanation,
        )

        # Track history
        self._history.append(record)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit :]

        return record

    # ---------- Severity classification ----------

    def _classify_severity(self, positions: list[PhilosophyPosition]) -> ConflictSeverity:
        """Classify how bad the disagreement is."""
        if not positions:
            return ConflictSeverity.NONE

        # Check for hard-block
        has_hard_block = any(p.verdict == "hard-block" for p in positions)
        if has_hard_block:
            return ConflictSeverity.CRITICAL

        verdicts = [p.verdict for p in positions]
        unique_verdicts = set(verdicts)

        if len(unique_verdicts) == 1:
            # All agree on verdict
            scores = [p.score for p in positions]
            score_range = max(scores) - min(scores)
            if score_range < 0.15:
                return ConflictSeverity.NONE
            return ConflictSeverity.MINOR

        # Check if disagreement crosses verdict boundaries
        has_approve = "approve" in unique_verdicts
        has_reject = "reject" in unique_verdicts

        if has_approve and has_reject:
            return ConflictSeverity.SEVERE

        return ConflictSeverity.MODERATE

    # ---------- Strategy selection ----------

    def _select_strategy(
        self,
        severity: ConflictSeverity,
        positions: list[PhilosophyPosition],
    ) -> ResolutionStrategy:
        if severity == ConflictSeverity.CRITICAL:
            return ResolutionStrategy.PRIORITY_OVERRIDE
        if severity == ConflictSeverity.SEVERE:
            return ResolutionStrategy.MORAL_UNCERTAINTY
        if severity == ConflictSeverity.MODERATE:
            return ResolutionStrategy.DELIBERATIVE
        return ResolutionStrategy.WEIGHTED_VOTE

    # ---------- Resolution strategies ----------

    def _weighted_vote(
        self,
        positions: list[PhilosophyPosition],
    ) -> tuple[str, float]:
        """Simple weighted average."""
        if not positions:
            return "escalate", 0.5

        total_weight = sum(p.weight for p in positions)
        if total_weight == 0:
            total_weight = 1.0

        weighted_score = sum(p.score * p.weight for p in positions) / total_weight

        verdict = self._score_to_verdict(weighted_score)
        return verdict, weighted_score

    def _priority_override(
        self,
        positions: list[PhilosophyPosition],
    ) -> tuple[str, float]:
        """Deontological hard-block overrides everything."""
        deontological = [p for p in positions if p.philosophy == "deontological"]
        if deontological and deontological[0].verdict == "hard-block":
            return "hard-block", 0.0
        # Shouldn't get here, but fall back to weighted vote
        return self._weighted_vote(positions)

    def _deliberative(
        self,
        positions: list[PhilosophyPosition],
    ) -> tuple[str, float]:
        """Find the majority position, weighted by confidence."""
        if not positions:
            return "escalate", 0.5

        # Count weighted votes for each verdict
        verdict_weights: dict[str, float] = {}
        for p in positions:
            v = p.verdict
            w = p.weight * p.confidence
            verdict_weights[v] = verdict_weights.get(v, 0.0) + w

        # Pick the verdict with highest weighted support
        winner = max(verdict_weights, key=verdict_weights.get)  # type: ignore

        # Score is the weighted average of positions sharing the winner verdict
        winning_positions = [p for p in positions if p.verdict == winner]
        score = sum(p.score * p.weight for p in winning_positions) / max(
            sum(p.weight for p in winning_positions), 0.01
        )

        return winner, score

    def _moral_uncertainty_resolve(
        self,
        positions: list[PhilosophyPosition],
    ) -> tuple[str, float]:
        """When we're genuinely uncertain, escalate to human review
        unless there's a strong safety signal."""
        has_reject = any(p.verdict == "reject" for p in positions)
        has_approve = any(p.verdict == "approve" for p in positions)

        if has_reject and has_approve:
            # Genuine disagreement — escalate
            score = sum(p.score * p.weight for p in positions) / max(
                sum(p.weight for p in positions), 0.01
            )
            # Lean towards caution
            return "escalate", min(score, 0.65)

        return self._weighted_vote(positions)

    # ---------- Moral uncertainty ----------

    def _compute_moral_uncertainty(self, positions: list[PhilosophyPosition]) -> float:
        """Compute formal moral uncertainty score.

        Based on:
        1. Score variance across philosophies
        2. Verdict disagreement
        3. Average confidence level

        Range: 0.0 (perfectly certain) to 1.0 (maximum uncertainty)
        """
        if not positions or len(positions) < 2:
            return 0.3  # some baseline uncertainty

        scores = [p.score for p in positions]
        mean_score = sum(scores) / len(scores)

        # Variance
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        # Normalize: max variance for [0,1] scores is 0.25
        normalized_variance = min(variance / 0.25, 1.0)

        # Verdict disagreement
        verdicts = set(p.verdict for p in positions)
        disagreement = (len(verdicts) - 1) / max(len(positions) - 1, 1)

        # Confidence: low average confidence → high uncertainty
        avg_confidence = sum(p.confidence for p in positions) / len(positions)
        confidence_uncertainty = 1.0 - avg_confidence

        # Combine
        uncertainty = (
            0.4 * normalized_variance + 0.35 * disagreement + 0.25 * confidence_uncertainty
        )

        return max(0.0, min(1.0, uncertainty))

    # ---------- Dissent identification ----------

    def _identify_dissent(
        self,
        positions: list[PhilosophyPosition],
        resolved_verdict: str,
    ) -> list[str]:
        """Find philosophies that disagree with the resolved verdict."""
        dissent = []
        for p in positions:
            if p.verdict != resolved_verdict:
                dissent.append(
                    f"{p.philosophy}: recommended '{p.verdict}' "
                    f"(score={p.score:.2f}) — {p.key_argument}"
                )
        return dissent

    # ---------- Utility ----------

    @staticmethod
    def _score_to_verdict(score: float) -> str:
        if score >= 0.80:
            return "approve"
        if score >= 0.50:
            return "escalate"
        return "reject"

    def get_conflict_history(self) -> list[ConflictRecord]:
        """Return history of resolved conflicts (for analysis/learning)."""
        return list(self._history)

    def get_conflict_stats(self) -> dict[str, Any]:
        """Summary statistics of conflict patterns."""
        if not self._history:
            return {"total_conflicts": 0}

        severities = [r.severity.value for r in self._history]
        strategies = [r.strategy_used.value for r in self._history]
        uncertainties = [r.moral_uncertainty for r in self._history]

        return {
            "total_conflicts": len(self._history),
            "severity_distribution": {s: severities.count(s) for s in set(severities)},
            "strategy_distribution": {s: strategies.count(s) for s in set(strategies)},
            "avg_moral_uncertainty": round(sum(uncertainties) / len(uncertainties), 4),
            "max_moral_uncertainty": round(max(uncertainties), 4),
        }

    # ---------- Explanation ----------

    def _build_explanation(
        self,
        severity: ConflictSeverity,
        strategy: ResolutionStrategy,
        positions: list[PhilosophyPosition],
        verdict: str,
        score: float,
        uncertainty: float,
        dissent: list[str],
    ) -> str:
        lines = [
            f"Conflict resolution: severity={severity.value}, strategy={strategy.value}",
            f"Resolved verdict: {verdict} (score={score:.3f})",
            f"Moral uncertainty: {uncertainty:.3f}",
        ]

        lines.append("Philosophy positions:")
        for p in positions:
            lines.append(
                f"  • {p.philosophy}: score={p.score:.2f}, "
                f"verdict={p.verdict}, "
                f"confidence={p.confidence:.2f}, "
                f"weight={p.weight:.2f}"
            )

        if dissent:
            lines.append("Dissenting views:")
            for d in dissent:
                lines.append(f"  ⚠ {d}")

        if uncertainty > 0.6:
            lines.append("⚠ High moral uncertainty — human review strongly recommended")

        return "\n".join(lines)
