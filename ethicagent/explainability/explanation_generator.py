"""Explanation generator — turning pipeline outputs into human-readable text.

This is where we make the system auditable and trustworthy.
Nobody trusts a black box that just says "REJECTED"; they need to
understand *why*.  This module generates explanations at multiple
levels of detail:

  - Brief — one sentence, for notification banners
  - Standard — one paragraph, for dashboard displays
  - Detailed — full breakdown with per-component scores
  - Technical — all the numbers, for auditors and developers

The generator also supports counterfactual explanations:
"If the deontological score had been 0.85 instead of 0.45,
the verdict would have been APPROVE."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DetailLevel(str, Enum):
    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"
    TECHNICAL = "technical"


@dataclass
class Explanation:
    """A structured explanation of an ethical decision."""
    level: DetailLevel
    summary: str
    full_text: str
    factors: list[dict[str, Any]] = field(default_factory=list)
    counterfactuals: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ExplanationGenerator:
    """Generates human-readable explanations of ethical decisions.

    Takes the pipeline's internal representation and produces
    explanations that non-technical stakeholders can understand.
    """

    def __init__(self) -> None:
        self._templates: dict[str, str] = {}

    def generate(
        self,
        decision: dict[str, Any],
        level: DetailLevel | str = DetailLevel.STANDARD,
        audience: str = "general",
    ) -> str:
        """Generate an explanation for an ethical decision.

        Parameters
        ----------
        decision : dict
            The decision data (eds_score, verdict, philosophy_scores, etc.)
        level : DetailLevel or str
            How much detail to include.
        audience : str
            Who the explanation is for ("general", "technical", "legal").

        Returns
        -------
        str
            The human-readable explanation text.
        """
        # Accept string level values
        if isinstance(level, str):
            _level_map = {
                "brief": DetailLevel.BRIEF,
                "summary": DetailLevel.BRIEF,  # alias
                "standard": DetailLevel.STANDARD,
                "detailed": DetailLevel.DETAILED,
                "technical": DetailLevel.TECHNICAL,
            }
            level = _level_map.get(level.lower(), DetailLevel.STANDARD)
        eds = decision.get("eds_score", 0.0)
        verdict = decision.get("verdict", "UNKNOWN")
        phil_scores = decision.get("philosophy_scores", {})
        domain = decision.get("domain", "general")
        reasoning = decision.get("reasoning", "")

        summary = self._generate_summary(eds, verdict, domain)
        factors = self._extract_factors(phil_scores, decision)
        counterfactuals = self._generate_counterfactuals(phil_scores, domain, eds, verdict)
        recommendations = self._generate_recommendations(verdict, phil_scores, decision)

        if level == DetailLevel.BRIEF:
            full_text = summary
        elif level == DetailLevel.STANDARD:
            full_text = self._standard_explanation(eds, verdict, domain, phil_scores, reasoning)
        elif level == DetailLevel.DETAILED:
            full_text = self._detailed_explanation(
                eds, verdict, domain, phil_scores, reasoning,
                factors, counterfactuals, recommendations,
            )
        else:
            full_text = self._technical_explanation(decision)

        return full_text

    def extract_key_factors(self, decision: dict[str, Any]) -> list[dict[str, Any]]:
        """Public wrapper for extracting key decision factors."""
        phil_scores = decision.get("philosophy_scores", {})
        return self._extract_factors(phil_scores, decision)

    def _generate_summary(self, eds: float, verdict: str, domain: str) -> str:
        """One-sentence summary."""
        verdict_desc = {
            "AUTO_APPROVE": "approved automatically",
            "ESCALATE": "flagged for human review",
            "REJECT": "rejected as ethically problematic",
            "HARD_BLOCK": "blocked due to a fundamental ethical violation",
        }
        desc = verdict_desc.get(verdict, "evaluated")
        return (
            f"The proposed action in the {domain} domain was {desc} "
            f"with an ethical score of {eds:.2f}."
        )

    def _extract_factors(
        self,
        phil_scores: dict[str, float],
        decision: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Extract key decision factors."""
        factors = []
        labels = {
            "deontological": "Duty-based rules",
            "consequentialist": "Expected outcomes",
            "virtue": "Fairness and character",
            "contextual": "Situational context",
        }
        for phil, score in phil_scores.items():
            factors.append({
                "name": labels.get(phil, phil),
                "philosophy": phil,
                "score": score,
                "rating": "strong" if score >= 0.8 else "adequate" if score >= 0.5 else "weak",
            })
        return factors

    def _generate_counterfactuals(
        self,
        phil_scores: dict[str, float],
        domain: str,
        current_eds: float,
        current_verdict: str,
    ) -> list[str]:
        """Generate counterfactual explanations — 'what if' scenarios."""
        counterfactuals = []

        # For each philosophy, what if the score were different?
        from ethicagent.ethics.ethical_score import compute_eds, determine_verdict

        for phil, score in phil_scores.items():
            if score < 0.8:
                # What if this score were higher?
                alt_scores = dict(phil_scores)
                alt_scores[phil] = 0.85
                alt_eds = compute_eds(alt_scores, domain=domain)
                alt_verdict = determine_verdict(alt_eds).value

                if alt_verdict != current_verdict:
                    counterfactuals.append(
                        f"If the {phil} score were 0.85 (instead of {score:.2f}), "
                        f"the verdict would change from {current_verdict} to {alt_verdict}."
                    )

            if score > 0.5:
                # What if this score were lower?
                alt_scores = dict(phil_scores)
                alt_scores[phil] = 0.30
                alt_eds = compute_eds(alt_scores, domain=domain)
                alt_verdict = determine_verdict(alt_eds).value

                if alt_verdict != current_verdict:
                    counterfactuals.append(
                        f"If the {phil} score dropped to 0.30, "
                        f"the verdict would change from {current_verdict} to {alt_verdict}."
                    )

        return counterfactuals[:4]  # cap at 4 to not overwhelm

    def _generate_recommendations(
        self,
        verdict: str,
        phil_scores: dict[str, float],
        decision: dict[str, Any],
    ) -> list[str]:
        recs = []
        if verdict in ("REJECT", "HARD_BLOCK"):
            recs.append("Consider alternative approaches that address the identified concerns.")
        if verdict == "ESCALATE":
            recs.append("Human review is recommended before proceeding.")

        # Find weakest philosophy
        if phil_scores:
            weakest = min(phil_scores, key=phil_scores.get)  # type: ignore
            weakest_score = phil_scores[weakest]
            if weakest_score < 0.5:
                advice = {
                    "deontological": "Ensure the action complies with relevant rules and duties.",
                    "consequentialist": "Evaluate potential harms and consider mitigations.",
                    "virtue": "Review fairness implications and bias potential.",
                    "contextual": "Consider legal requirements and cultural context.",
                }
                recs.append(advice.get(weakest, f"Address concerns in {weakest} evaluation."))

        warnings = decision.get("warnings", [])
        if warnings:
            recs.append(f"Address {len(warnings)} warning(s) identified in the evaluation.")

        return recs

    def _standard_explanation(
        self,
        eds: float, verdict: str, domain: str,
        phil_scores: dict[str, float], reasoning: str,
    ) -> str:
        lines = [self._generate_summary(eds, verdict, domain)]
        lines.append("")

        if phil_scores:
            lines.append("The evaluation considered four ethical perspectives:")
            for phil, score in phil_scores.items():
                rating = "strong" if score >= 0.8 else "adequate" if score >= 0.5 else "weak"
                lines.append(f"  • {phil.title()}: {score:.2f} ({rating})")

        if reasoning:
            lines.append("")
            lines.append(f"Key reasoning: {reasoning[:300]}")

        return "\n".join(lines)

    def _detailed_explanation(
        self,
        eds: float, verdict: str, domain: str,
        phil_scores: dict[str, float], reasoning: str,
        factors: list, counterfactuals: list, recommendations: list,
    ) -> str:
        lines = [
            "=" * 60,
            "ETHICAL DECISION REPORT",
            "=" * 60,
            "",
            self._generate_summary(eds, verdict, domain),
            "",
            f"EDS Score: {eds:.4f}",
            f"Verdict:   {verdict}",
            f"Domain:    {domain}",
            "",
        ]

        lines.append("PHILOSOPHY SCORES:")
        for f in factors:
            lines.append(f"  {f['name']:30s} {f['score']:.3f}  ({f['rating']})")

        if counterfactuals:
            lines.append("")
            lines.append("COUNTERFACTUAL ANALYSIS:")
            for cf in counterfactuals:
                lines.append(f"  → {cf}")

        if recommendations:
            lines.append("")
            lines.append("RECOMMENDATIONS:")
            for r in recommendations:
                lines.append(f"  • {r}")

        if reasoning:
            lines.append("")
            lines.append("REASONING:")
            lines.append(f"  {reasoning}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def _technical_explanation(self, decision: dict[str, Any]) -> str:
        """Full technical dump for auditors."""
        import json
        return json.dumps(decision, indent=2, default=str)
