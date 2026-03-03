"""Consequentialist (outcome-based) ethical evaluation.

"The right action is the one that produces the best overall consequences."
— broadly, utilitarianism and its variants.

This module scores C(a) by estimating:
  1. Multi-stakeholder impact  — who benefits, who is harmed, and by how much
  2. Reversibility scoring     — can we undo this if it goes wrong?
  3. Temporal analysis         — short-term vs. long-term tradeoffs
  4. Scale factor              — how many people are affected?
  5. Distributional analysis   — are the harms concentrated on vulnerable groups?
  6. Probability weighting     — expected value under uncertainty

Design rationale:
  We want C(a) ∈ [0, 1] where 1 = clearly net-positive and 0 = clearly
  net-negative.  The score is a weighted combination of the sub-scores
  above.  We intentionally penalize irreversible harms because "oops we
  can't undo it" is an underweighted risk in most naive utility calcs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ethicagent.ethics.ethical_score import PhilosophyResult as _PhilosophyResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ImpactType(Enum):
    BENEFIT = "benefit"
    HARM = "harm"
    NEUTRAL = "neutral"


class TimeHorizon(Enum):
    IMMEDIATE = "immediate"  # < 1 day
    SHORT_TERM = "short_term"  # 1 day – 1 month
    MEDIUM_TERM = "medium_term"  # 1 month – 1 year
    LONG_TERM = "long_term"  # 1 year+


@dataclass
class StakeholderImpact:
    """Impact assessment for a single stakeholder group."""

    group: str
    impact_type: ImpactType
    magnitude: float  # 0.0 – 1.0  (how severe/beneficial)
    probability: float  # 0.0 – 1.0  (how likely)
    population_size: int  # approximate number of people affected
    vulnerable: bool = False  # is this a vulnerable population?
    reversible: bool = True  # can the impact be undone?
    time_horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM
    description: str = ""

    @property
    def expected_impact(self) -> float:
        """Signed expected value: positive for benefits, negative for harms."""
        sign = 1.0 if self.impact_type == ImpactType.BENEFIT else -1.0
        if self.impact_type == ImpactType.NEUTRAL:
            return 0.0
        return sign * self.magnitude * self.probability


@dataclass
class ConsequentialistResult:
    """Output of the consequentialist evaluator."""

    score: float  # C(a), 0.0 – 1.0
    net_utility: float  # raw utility before normalization
    stakeholder_impacts: list[StakeholderImpact] = field(default_factory=list)
    reversibility_score: float = 0.5
    temporal_score: float = 0.5
    scale_factor: float = 1.0
    distributional_score: float = 0.5
    explanation: str = ""
    breakdown: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Keyword signals (used when we don't have structured stakeholder data)
# ---------------------------------------------------------------------------

# Words that suggest positive outcomes
_BENEFIT_SIGNALS = {
    "save",
    "protect",
    "help",
    "improve",
    "benefit",
    "cure",
    "heal",
    "rescue",
    "prevent",
    "reduce harm",
    "safe",
    "support",
    "empower",
    "enable",
    "opportunity",
    "thrive",
    "well-being",
    "welfare",
    "accessible",
    "inclusive",
    "equitable",
    "fair",
}

# Words that suggest negative outcomes
_HARM_SIGNALS = {
    "harm",
    "damage",
    "hurt",
    "injure",
    "kill",
    "destroy",
    "loss",
    "suffer",
    "pain",
    "deny",
    "exclude",
    "discriminate",
    "violate",
    "exploit",
    "manipulate",
    "deceive",
    "coerce",
    "endanger",
    "risk",
    "bankrupt",
    "homeless",
    "starve",
    "toxic",
    "poison",
    "contaminate",
    "pollute",
    "dump",
    "chemical",
    "spill",
    "leak",
    "release toxic",
    "hazardous",
}

# Words suggesting irreversibility
_IRREVERSIBLE_SIGNALS = {
    "permanent",
    "irreversible",
    "cannot undo",
    "fatal",
    "death",
    "destroy",
    "delete permanently",
    "terminate",
    "irrecoverable",
}

# Words suggesting large scale
_LARGE_SCALE_SIGNALS = {
    "everyone",
    "all patients",
    "population",
    "community-wide",
    "nationwide",
    "global",
    "thousands",
    "millions",
    "mass",
    "widespread",
    "systemic",
    "institutional",
}

_VULNERABLE_SIGNALS = {
    "child",
    "elderly",
    "disabled",
    "pregnant",
    "minority",
    "refugee",
    "homeless",
    "impoverished",
    "low-income",
    "undocumented",
    "indigenous",
    "mental health",
    "veteran",
}


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class ConsequentialistEvaluator:
    """Evaluates actions based on their expected outcomes.

    This is basically a structured expected-utility calculator with
    extra safeguards for distributional justice and irreversibility.

    The sub-component weights are configurable but default to a
    balanced mix that penalizes irreversible harms and rewards
    broad benefits.
    """

    # Sub-score weights (sum to 1.0)
    DEFAULT_WEIGHTS = {
        "net_utility": 0.35,
        "reversibility": 0.20,
        "temporal": 0.15,
        "distributional": 0.20,
        "scale": 0.10,
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)
        # Normalize just in case
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def evaluate(
        self,
        action_text: str,
        context: dict[str, Any] | None = None,
        stakeholder_impacts: list[StakeholderImpact] | None = None,
    ) -> ConsequentialistResult:
        """Run the full consequentialist evaluation.

        If structured stakeholder_impacts are provided, we use those.
        Otherwise, we fall back to heuristic text analysis.
        """
        context = context or {}
        text_lower = action_text.lower()

        # Build stakeholder impacts from structured data or heuristics
        if stakeholder_impacts:
            impacts = stakeholder_impacts
        else:
            impacts = self._infer_impacts(text_lower, context)

        # Compute sub-scores
        net_utility = self._compute_net_utility(impacts)
        reversibility = self._compute_reversibility(impacts, text_lower)
        temporal = self._compute_temporal_score(impacts, text_lower, context)
        distributional = self._compute_distributional_score(impacts, text_lower)
        scale = self._compute_scale_factor(impacts, text_lower)

        # Weighted combination → C(a)
        raw = (
            self.weights["net_utility"] * self._normalize_utility(net_utility)
            + self.weights["reversibility"] * reversibility
            + self.weights["temporal"] * temporal
            + self.weights["distributional"] * distributional
            + self.weights["scale"] * scale
        )

        # Clamp to [0, 1]
        c_score = max(0.0, min(1.0, raw))

        breakdown = {
            "net_utility_normalized": round(self._normalize_utility(net_utility), 4),
            "reversibility": round(reversibility, 4),
            "temporal": round(temporal, 4),
            "distributional": round(distributional, 4),
            "scale": round(scale, 4),
        }

        _cr = ConsequentialistResult(
            score=round(c_score, 4),
            net_utility=round(net_utility, 4),
            stakeholder_impacts=impacts,
            reversibility_score=round(reversibility, 4),
            temporal_score=round(temporal, 4),
            scale_factor=round(scale, 4),
            distributional_score=round(distributional, 4),
            explanation=self._build_explanation(c_score, net_utility, breakdown, impacts),
            breakdown=breakdown,
        )
        return self._wrap(_cr)

    @staticmethod
    def _wrap(cr: ConsequentialistResult) -> _PhilosophyResult:
        return _PhilosophyResult(
            philosophy="consequentialist",
            score=cr.score,
            reasoning=cr.explanation,
            key_argument=cr.explanation,
            details={"breakdown": cr.breakdown, "net_utility": cr.net_utility},
        )

    # ---------- Sub-score computations ----------

    def _compute_net_utility(self, impacts: list[StakeholderImpact]) -> float:
        """Sum of (signed magnitude × probability × log2(population)).

        We use log-scaled population to avoid letting a single huge group
        dominate the calculation — a common criticism of naive utilitarianism.
        """
        if not impacts:
            return 0.0

        total = 0.0
        for imp in impacts:
            pop_weight = math.log2(max(imp.population_size, 2))
            total += imp.expected_impact * pop_weight
        return total

    def _normalize_utility(self, raw: float) -> float:
        """Map raw utility to [0, 1] using a sigmoid-like transform.

        We want:  large positive → ~1,  zero → 0.5,  large negative → ~0
        """
        # Tanh gives us (-1, 1), rescale to (0, 1)
        return 0.5 + 0.5 * math.tanh(raw / 5.0)

    def _compute_reversibility(
        self,
        impacts: list[StakeholderImpact],
        text_lower: str,
    ) -> float:
        """Higher score = more reversible (good).

        Irreversible harms get a strong penalty.
        """
        if not impacts:
            # Fall back to text signals
            has_irreversible = any(sig in text_lower for sig in _IRREVERSIBLE_SIGNALS)
            return 0.3 if has_irreversible else 0.75

        # Fraction of harmful impacts that are reversible
        harmful = [i for i in impacts if i.impact_type == ImpactType.HARM]
        if not harmful:
            return 0.95  # no harms → very reversible (trivially)

        reversible_count = sum(1 for i in harmful if i.reversible)
        frac = reversible_count / len(harmful)

        # Weight by magnitude — irreversible HIGH-magnitude harms matter more
        weighted_reversible = 0.0
        total_magnitude = 0.0
        for h in harmful:
            total_magnitude += h.magnitude
            if h.reversible:
                weighted_reversible += h.magnitude

        if total_magnitude > 0:
            weighted_frac = weighted_reversible / total_magnitude
        else:
            weighted_frac = 1.0

        return 0.5 * frac + 0.5 * weighted_frac

    def _compute_temporal_score(
        self,
        impacts: list[StakeholderImpact],
        text_lower: str,
        context: dict[str, Any],
    ) -> float:
        """Score how well the action balances short-term and long-term.

        Penalizes short-term-only thinking and rewards actions with
        lasting positive effects.
        """
        if not impacts:
            # Heuristic: emergency contexts favor short-term action
            is_emergency = context.get("urgency", "") == "critical"
            if is_emergency:
                return 0.7  # short-term focus is appropriate
            return 0.5  # neutral

        # Compute time-horizon distribution
        horizon_values = {
            TimeHorizon.IMMEDIATE: 0,
            TimeHorizon.SHORT_TERM: 1,
            TimeHorizon.MEDIUM_TERM: 2,
            TimeHorizon.LONG_TERM: 3,
        }

        benefit_horizons = []
        harm_horizons = []
        for imp in impacts:
            h_val = horizon_values.get(imp.time_horizon, 1)
            if imp.impact_type == ImpactType.BENEFIT:
                benefit_horizons.append(h_val)
            elif imp.impact_type == ImpactType.HARM:
                harm_horizons.append(h_val)

        score = 0.5  # baseline

        # Reward long-term benefits
        if benefit_horizons:
            avg_benefit_horizon = sum(benefit_horizons) / len(benefit_horizons)
            score += 0.15 * (avg_benefit_horizon / 3.0)  # up to +0.15

        # Penalize long-term harms
        if harm_horizons:
            avg_harm_horizon = sum(harm_horizons) / len(harm_horizons)
            score -= 0.2 * (avg_harm_horizon / 3.0)  # up to -0.2

        return max(0.0, min(1.0, score))

    def _compute_distributional_score(
        self,
        impacts: list[StakeholderImpact],
        text_lower: str,
    ) -> float:
        """Score distributional justice.

        Higher = harms are not concentrated on vulnerable groups.
        Lower = vulnerable groups bear disproportionate burden.
        """
        if not impacts:
            # Heuristic: check for vulnerable population mentions + harm
            has_vulnerable = any(sig in text_lower for sig in _VULNERABLE_SIGNALS)
            has_harm = any(sig in text_lower for sig in _HARM_SIGNALS)
            if has_vulnerable and has_harm:
                return 0.3
            if has_vulnerable:
                return 0.5
            return 0.65

        vulnerable_harm = 0.0
        total_harm = 0.0
        vulnerable_benefit = 0.0
        total_benefit = 0.0

        for imp in impacts:
            if imp.impact_type == ImpactType.HARM:
                total_harm += imp.magnitude * imp.probability
                if imp.vulnerable:
                    vulnerable_harm += imp.magnitude * imp.probability
            elif imp.impact_type == ImpactType.BENEFIT:
                total_benefit += imp.magnitude * imp.probability
                if imp.vulnerable:
                    vulnerable_benefit += imp.magnitude * imp.probability

        score = 0.5

        # Penalize if vulnerable groups bear disproportionate harm
        if total_harm > 0:
            vulnerable_harm_ratio = vulnerable_harm / total_harm
            score -= 0.3 * vulnerable_harm_ratio

        # Reward if vulnerable groups receive benefits
        if total_benefit > 0:
            vulnerable_benefit_ratio = vulnerable_benefit / total_benefit
            score += 0.2 * vulnerable_benefit_ratio

        return max(0.0, min(1.0, score))

    def _compute_scale_factor(
        self,
        impacts: list[StakeholderImpact],
        text_lower: str,
    ) -> float:
        """Adjust for scale of impact.

        Large-scale benefits → higher score.
        Large-scale harms → lower score.
        """
        if not impacts:
            has_large = any(sig in text_lower for sig in _LARGE_SCALE_SIGNALS)
            has_benefit = any(sig in text_lower for sig in _BENEFIT_SIGNALS)
            has_harm = any(sig in text_lower for sig in _HARM_SIGNALS)
            if has_large and has_benefit:
                return 0.8
            if has_large and has_harm:
                return 0.2
            return 0.5

        total_pop = sum(i.population_size for i in impacts)
        benefit_pop = sum(i.population_size for i in impacts if i.impact_type == ImpactType.BENEFIT)
        harm_pop = sum(i.population_size for i in impacts if i.impact_type == ImpactType.HARM)

        if total_pop == 0:
            return 0.5

        benefit_ratio = benefit_pop / total_pop
        return 0.3 + 0.7 * benefit_ratio  # maps to [0.3, 1.0]

    # ---------- Heuristic inference ----------

    def _infer_impacts(
        self,
        text_lower: str,
        context: dict[str, Any],
    ) -> list[StakeholderImpact]:
        """When we don't have structured stakeholder data, infer from text."""
        impacts: list[StakeholderImpact] = []
        stakeholders = context.get("stakeholders", [])

        # Count benefit and harm signals
        benefit_count = sum(1 for sig in _BENEFIT_SIGNALS if sig in text_lower)
        harm_count = sum(1 for sig in _HARM_SIGNALS if sig in text_lower)
        has_vulnerable = any(sig in text_lower for sig in _VULNERABLE_SIGNALS)
        has_irreversible = any(sig in text_lower for sig in _IRREVERSIBLE_SIGNALS)

        # Build generic stakeholder impacts
        if benefit_count > 0:
            mag = min(0.9, 0.3 + 0.1 * benefit_count)
            impacts.append(
                StakeholderImpact(
                    group="primary_beneficiaries",
                    impact_type=ImpactType.BENEFIT,
                    magnitude=mag,
                    probability=0.7,
                    population_size=100,
                    description="Inferred benefit from text signals",
                )
            )

        if harm_count > 0:
            mag = min(0.9, 0.3 + 0.1 * harm_count)
            impacts.append(
                StakeholderImpact(
                    group="affected_parties",
                    impact_type=ImpactType.HARM,
                    magnitude=mag,
                    probability=0.6,
                    population_size=50,
                    reversible=not has_irreversible,
                    description="Inferred harm from text signals",
                )
            )

        if has_vulnerable:
            impacts.append(
                StakeholderImpact(
                    group="vulnerable_population",
                    impact_type=ImpactType.HARM
                    if harm_count > benefit_count
                    else ImpactType.BENEFIT,
                    magnitude=0.5,
                    probability=0.6,
                    population_size=30,
                    vulnerable=True,
                    description="Vulnerable population identified",
                )
            )

        # If context has stakeholders, create entries for them
        for sh in stakeholders:
            if isinstance(sh, dict):
                name = sh.get("name", "unknown")
                role = sh.get("role", "bystander")
                is_vulnerable = sh.get("vulnerable", False)
                impact_type = (
                    ImpactType.BENEFIT
                    if role in ("beneficiary", "decision-maker")
                    else ImpactType.HARM
                    if role == "victim"
                    else ImpactType.NEUTRAL
                )
                impacts.append(
                    StakeholderImpact(
                        group=name,
                        impact_type=impact_type,
                        magnitude=0.5,
                        probability=0.6,
                        population_size=1,
                        vulnerable=is_vulnerable,
                        description=f"Stakeholder: {name} (role: {role})",
                    )
                )

        # Fallback — at least one neutral impact
        if not impacts:
            impacts.append(
                StakeholderImpact(
                    group="general_public",
                    impact_type=ImpactType.NEUTRAL,
                    magnitude=0.1,
                    probability=0.5,
                    population_size=100,
                    description="No specific impacts identified",
                )
            )

        return impacts

    # ---------- Explanation ----------

    def _build_explanation(
        self,
        score: float,
        net_utility: float,
        breakdown: dict[str, float],
        impacts: list[StakeholderImpact],
    ) -> str:
        lines = [f"Consequentialist score C(a) = {score:.3f}"]

        if score >= 0.75:
            lines.append("Assessment: Net positive outcomes expected.")
        elif score >= 0.5:
            lines.append("Assessment: Mixed outcomes — benefits and harms roughly balanced.")
        elif score >= 0.25:
            lines.append("Assessment: Net negative skew — potential harms outweigh benefits.")
        else:
            lines.append("Assessment: Significant net harm expected.")

        lines.append(f"Raw net utility: {net_utility:+.3f}")
        lines.append("Sub-scores:")
        for key, val in breakdown.items():
            lines.append(f"  • {key}: {val:.3f}")

        # Summarize stakeholders
        benefits = [i for i in impacts if i.impact_type == ImpactType.BENEFIT]
        harms = [i for i in impacts if i.impact_type == ImpactType.HARM]
        if benefits:
            groups = ", ".join(i.group for i in benefits[:3])
            lines.append(f"Beneficiaries: {groups}")
        if harms:
            groups = ", ".join(i.group for i in harms[:3])
            lines.append(f"Potentially harmed: {groups}")

        vulnerable = [i for i in impacts if i.vulnerable]
        if vulnerable:
            lines.append(f"⚠ Vulnerable populations affected: {len(vulnerable)} group(s)")

        return "\n".join(lines)
