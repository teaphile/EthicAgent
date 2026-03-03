"""Contextual ethics evaluator — Ctx(a) in the EDS formula.

This module handles the "it depends" part of ethical reasoning.
Something acceptable in healthcare triage may be unacceptable in routine
hiring; cultural norms vary; legal frameworks differ by jurisdiction.

Key components:
  1. Legal framework checking — EU AI Act risk levels, HIPAA, GDPR, etc.
  2. Cultural norm awareness — not to impose, but to *inform*
  3. Temporal context        — urgency, time pressure, evolving norms
  4. Precedent matching      — how were similar situations handled before?
  5. Domain-specific norms   — professional codes of conduct

The score Ctx(a) rewards actions that are well-adapted to their context
and penalizes ones that ignore relevant contextual factors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ethicagent.ethics.ethical_score import PhilosophyResult as _PhilosophyResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------


class EUAIActRisk(Enum):
    """EU AI Act risk classifications."""

    UNACCEPTABLE = "unacceptable"  # Banned
    HIGH = "high"  # Strict requirements
    LIMITED = "limited"  # Transparency obligations
    MINIMAL = "minimal"  # No specific requirements


class Jurisdiction(Enum):
    EU = "eu"
    US = "us"
    UK = "uk"
    GLOBAL = "global"
    UNKNOWN = "unknown"


@dataclass
class LegalAssessment:
    """Assessment of legal compliance for the context."""

    jurisdiction: Jurisdiction
    applicable_laws: list[str]
    risk_level: EUAIActRisk
    compliance_gaps: list[str]
    score: float  # 0-1, higher = more compliant


@dataclass
class CulturalContext:
    """Cultural factors that may affect ethical assessment."""

    region: str
    relevant_norms: list[str]
    sensitivity_level: str  # "low", "medium", "high"
    considerations: list[str]


@dataclass
class PrecedentMatch:
    """A previously handled similar situation."""

    case_id: str
    similarity: float  # 0-1
    outcome: str
    was_appropriate: bool
    lessons: str


@dataclass
class ContextualResult:
    """Output of the contextual ethics evaluator."""

    score: float  # Ctx(a), 0.0 – 1.0
    legal_assessment: LegalAssessment | None = None
    cultural_context: CulturalContext | None = None
    precedents: list[PrecedentMatch] = field(default_factory=list)
    temporal_factors: dict[str, Any] = field(default_factory=dict)
    domain_norms: list[str] = field(default_factory=list)
    explanation: str = ""


# ---------------------------------------------------------------------------
# Legal framework database
# ---------------------------------------------------------------------------

# EU AI Act domain → risk mapping
# These are simplified; real mapping needs detailed system analysis
_EU_AI_ACT_RISK_MAP: dict[str, EUAIActRisk] = {
    "healthcare": EUAIActRisk.HIGH,  # Medical device / diagnostic
    "finance": EUAIActRisk.HIGH,  # Credit scoring
    "hiring": EUAIActRisk.HIGH,  # Employment decisions
    "disaster": EUAIActRisk.LIMITED,  # Emergency management
    "general": EUAIActRisk.MINIMAL,
}

_DOMAIN_LAWS: dict[str, list[str]] = {
    "healthcare": [
        "HIPAA (US)",
        "GDPR Art. 9 (EU)",
        "Medical Device Regulation (EU)",
        "Health Insurance Portability Act",
        "Mental Health Parity Act",
    ],
    "finance": [
        "ECOA (US)",
        "Fair Lending Act (US)",
        "GDPR (EU)",
        "PSD2 (EU)",
        "Consumer Credit Directive (EU)",
        "Dodd-Frank Act (US)",
        "Basel III",
    ],
    "hiring": [
        "Title VII Civil Rights Act (US)",
        "ADA (US)",
        "ADEA (US)",
        "GDPR (EU)",
        "EU AI Act Annex III",
        "Equal Pay Act (US)",
    ],
    "disaster": [
        "Stafford Act (US)",
        "EMTALA (US)",
        "EU Civil Protection Mechanism",
    ],
    "general": ["GDPR (EU)", "CCPA (US)", "AI Act (EU)"],
}

# Professional codes of conduct by domain
_DOMAIN_NORMS: dict[str, list[str]] = {
    "healthcare": [
        "Do no harm (primum non nocere)",
        "Patient autonomy and informed consent",
        "Confidentiality of medical information",
        "Clinical evidence-based practice",
        "Equitable resource allocation",
        "Duty of care to all patients regardless of status",
    ],
    "finance": [
        "Fair lending and equal access to credit",
        "Transparent pricing and terms",
        "Fiduciary duty where applicable",
        "Anti-money laundering compliance",
        "Consumer protection and suitability",
        "Adverse action notice requirements",
    ],
    "hiring": [
        "Merit-based selection criteria",
        "Equal opportunity for all applicants",
        "Bona fide occupational qualification exceptions only",
        "Reasonable accommodation for disabilities",
        "Consistent interview processes",
        "Transparent rejection criteria",
    ],
    "disaster": [
        "Greatest good for greatest number under urgency",
        "Non-discrimination in aid distribution",
        "Proportional response to threat level",
        "Coordination with official agencies",
        "Protection of most vulnerable first",
        "Rapid response takes priority",
    ],
    "general": [
        "Transparency in AI decision-making",
        "Respect for user privacy",
        "Accountability for outcomes",
        "Regular bias auditing",
    ],
}

# Cultural norm database — very simplified
# In production, this would be much richer and regularly updated
_CULTURAL_NORMS: dict[str, list[str]] = {
    "western": [
        "Individual autonomy highly valued",
        "Direct communication preferred",
        "Merit-based evaluation expected",
        "Strong privacy expectations",
    ],
    "east_asian": [
        "Collective harmony important",
        "Hierarchical respect norms",
        "Group consensus valued",
        "Face-saving considerations",
    ],
    "south_asian": [
        "Family consultation in major decisions",
        "Community impact considerations",
        "Respect for elders emphasized",
    ],
    "middle_eastern": [
        "Family honor considerations",
        "Religious observance accommodations",
        "Hospitality norms",
    ],
    "african": [
        "Ubuntu philosophy — communal well-being",
        "Elder council consultation traditions",
        "Community-centered decisions",
    ],
    "global": [
        "Universal human rights baseline",
        "Cultural sensitivity without cultural relativism",
        "Minimum dignity standards apply everywhere",
    ],
}


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class ContextualEthicsEvaluator:
    """Evaluates Ctx(a) — how well the action fits its context.

    This is the "situational awareness" component of the EDS formula.
    An action that's perfectly fine in one context might be terrible
    in another, and this evaluator captures those nuances.
    """

    def __init__(
        self,
        precedent_store: Any = None,  # Optional PrecedentStore instance
        jurisdiction: str = "global",
    ) -> None:
        self.precedent_store = precedent_store
        self.default_jurisdiction = (
            Jurisdiction(jurisdiction)
            if jurisdiction in [j.value for j in Jurisdiction]
            else Jurisdiction.GLOBAL
        )

    def evaluate(
        self,
        action_text: str,
        context: dict[str, Any] | None = None,
    ) -> ContextualResult:
        """Run the full contextual evaluation."""
        context = context or {}
        domain = context.get("domain", "general")
        text_lower = action_text.lower()

        # 1. Legal assessment
        legal = self._assess_legal(domain, text_lower, context)

        # 2. Cultural context
        cultural = self._assess_cultural(context)

        # 3. Precedent matching
        precedents = self._find_precedents(action_text, context)

        # 4. Temporal factors
        temporal = self._assess_temporal(context)

        # 5. Domain norms
        norms = _DOMAIN_NORMS.get(domain, _DOMAIN_NORMS["general"])

        # 6. Compute composite score
        scores = {
            "legal": legal.score,
            "cultural_sensitivity": self._cultural_score(cultural, text_lower),
            "precedent_alignment": self._precedent_score(precedents),
            "temporal_appropriateness": self._temporal_score(temporal, text_lower),
            "domain_norm_compliance": self._norm_compliance_score(norms, text_lower, context),
        }

        weights = {
            "legal": 0.30,
            "cultural_sensitivity": 0.15,
            "precedent_alignment": 0.15,
            "temporal_appropriateness": 0.15,
            "domain_norm_compliance": 0.25,
        }

        ctx_score = sum(scores[k] * weights[k] for k in scores)
        ctx_score = max(0.0, min(1.0, ctx_score))

        _cr = ContextualResult(
            score=round(ctx_score, 4),
            legal_assessment=legal,
            cultural_context=cultural,
            precedents=precedents,
            temporal_factors=temporal,
            domain_norms=norms,
            explanation=self._build_explanation(ctx_score, scores, legal, cultural, precedents),
        )
        return self._wrap(_cr)

    @staticmethod
    def _wrap(cr: ContextualResult) -> _PhilosophyResult:
        return _PhilosophyResult(
            philosophy="contextual",
            score=cr.score,
            reasoning=cr.explanation,
            key_argument=cr.explanation,
            details={"domain_norms": cr.domain_norms},
        )

    # ---------- Legal assessment ----------

    def _assess_legal(
        self,
        domain: str,
        text_lower: str,
        context: dict[str, Any],
    ) -> LegalAssessment:
        jurisdiction_str = context.get("jurisdiction", self.default_jurisdiction.value)
        try:
            jurisdiction = Jurisdiction(jurisdiction_str)
        except ValueError:
            jurisdiction = Jurisdiction.UNKNOWN

        risk_level = _EU_AI_ACT_RISK_MAP.get(domain, EUAIActRisk.MINIMAL)
        laws = _DOMAIN_LAWS.get(domain, _DOMAIN_LAWS["general"])

        # Check for compliance gaps (heuristic)
        gaps: list[str] = []

        if risk_level in (EUAIActRisk.HIGH, EUAIActRisk.UNACCEPTABLE):
            if not context.get("risk_assessment_done"):
                gaps.append("Missing required risk assessment (EU AI Act)")
            if not context.get("human_oversight"):
                gaps.append("No human oversight mechanism (EU AI Act Art. 14)")

        if domain == "healthcare":
            if not context.get("metadata", {}).get("patient_consent"):
                gaps.append("Patient consent not confirmed (HIPAA)")
        elif domain == "finance":
            if not context.get("metadata", {}).get("adverse_notice"):
                if any(w in text_lower for w in ["deny", "reject", "decline"]):
                    gaps.append("Adverse action notice not provided (ECOA)")

        # Score based on gaps
        if not gaps:
            score = 0.9
        elif len(gaps) == 1:
            score = 0.6
        elif len(gaps) == 2:
            score = 0.4
        else:
            score = 0.2

        # Unacceptable risk → zero
        if risk_level == EUAIActRisk.UNACCEPTABLE:
            score = 0.0
            gaps.append("Action falls under EU AI Act unacceptable risk category")

        return LegalAssessment(
            jurisdiction=jurisdiction,
            applicable_laws=laws,
            risk_level=risk_level,
            compliance_gaps=gaps,
            score=score,
        )

    # ---------- Cultural assessment ----------

    def _assess_cultural(self, context: dict[str, Any]) -> CulturalContext:
        region = context.get("cultural_region", "global")
        norms = _CULTURAL_NORMS.get(region, _CULTURAL_NORMS["global"])

        # Sensitivity level based on domain + region
        domain = context.get("domain", "general")
        if domain in ("healthcare", "hiring") or region not in ("western", "global"):
            sensitivity = "high"
        elif domain == "finance":
            sensitivity = "medium"
        else:
            sensitivity = "low"

        considerations = []
        if context.get("cross_cultural"):
            considerations.append("Cross-cultural context — extra sensitivity needed")
        if context.get("religious_context"):
            considerations.append("Religious considerations present")

        return CulturalContext(
            region=region,
            relevant_norms=norms,
            sensitivity_level=sensitivity,
            considerations=considerations,
        )

    def _cultural_score(self, cultural: CulturalContext, text_lower: str) -> float:
        """Score how well the action respects cultural context."""
        score = 0.6  # baseline

        # Bonus for culturally aware language
        if any(w in text_lower for w in ["respect", "accommodat", "sensitiv", "inclusive"]):
            score += 0.2
        if any(w in text_lower for w in ["cultural", "community", "local context"]):
            score += 0.1

        # Penalty for culturally insensitive signals
        if any(w in text_lower for w in ["one-size-fits-all", "universal standard", "ignore"]):
            score -= 0.2

        if cultural.sensitivity_level == "high" and score < 0.5:
            score *= 0.8  # additional penalty in high-sensitivity contexts

        return max(0.0, min(1.0, score))

    # ---------- Precedent matching ----------

    def _find_precedents(
        self,
        action_text: str,
        context: dict[str, Any],
    ) -> list[PrecedentMatch]:
        """Look up similar past decisions."""
        if self.precedent_store is None:
            return []

        try:
            domain = context.get("domain", "general")
            results = self.precedent_store.query(
                action_text,
                domain=domain,
                top_k=3,
            )
            precedents = []
            for r in results:
                precedents.append(
                    PrecedentMatch(
                        case_id=r.get("id", "unknown"),
                        similarity=r.get("similarity", 0.0),
                        outcome=r.get("outcome", "unknown"),
                        was_appropriate=r.get("was_appropriate", True),
                        lessons=r.get("lessons", ""),
                    )
                )
            return precedents
        except Exception as e:
            logger.warning("Precedent lookup failed: %s", e)
            return []

    def _precedent_score(self, precedents: list[PrecedentMatch]) -> float:
        """Score based on precedent alignment."""
        if not precedents:
            return 0.5  # no data → neutral

        # Weight by similarity
        weighted_sum = 0.0
        weight_total = 0.0
        for p in precedents:
            w = p.similarity
            s = 0.8 if p.was_appropriate else 0.2
            weighted_sum += w * s
            weight_total += w

        if weight_total > 0:
            return weighted_sum / weight_total
        return 0.5

    # ---------- Temporal assessment ----------

    def _assess_temporal(self, context: dict[str, Any]) -> dict[str, Any]:
        urgency = context.get("urgency", "normal")
        temporal: dict[str, Any] = {
            "urgency": urgency,
            "time_pressure": urgency in ("high", "critical"),
            "evolving_situation": context.get("evolving", False),
        }

        if urgency == "critical":
            temporal["note"] = "Critical urgency — expedited decision process appropriate"
        elif urgency == "high":
            temporal["note"] = "High urgency — reduced deliberation acceptable"
        else:
            temporal["note"] = "Normal conditions — full deliberation expected"

        return temporal

    def _temporal_score(self, temporal: dict[str, Any], text_lower: str) -> float:
        """Score temporal appropriateness."""
        urgency = temporal.get("urgency", "normal")

        if urgency == "critical":
            # In emergencies, decisive action is appropriate
            if any(w in text_lower for w in ["immediate", "urgent", "emergency", "rapid"]):
                return 0.85  # good — responding to urgency
            return 0.5  # not adapting to urgency

        # Normal conditions: measured response is appropriate
        if any(w in text_lower for w in ["hasty", "rush", "skip", "shortcut"]):
            return 0.3  # bad — cutting corners without urgency
        return 0.7

    # ---------- Norm compliance ----------

    def _norm_compliance_score(
        self,
        norms: list[str],
        text_lower: str,
        context: dict[str, Any],
    ) -> float:
        """Score compliance with domain professional norms."""
        # This is inherently approximate — we look for alignment signals
        score = 0.55  # baseline

        domain = context.get("domain", "general")

        if domain == "healthcare":
            if "patient" in text_lower and any(w in text_lower for w in ["consent", "inform"]):
                score += 0.15
            if any(w in text_lower for w in ["evidence", "clinical", "protocol"]):
                score += 0.1
            if "confidentiali" in text_lower or "privacy" in text_lower:
                score += 0.1

        elif domain == "finance":
            if any(w in text_lower for w in ["fair", "transparent", "equal access"]):
                score += 0.15
            if "adverse" in text_lower and "notice" in text_lower:
                score += 0.1

        elif domain == "hiring":
            if any(w in text_lower for w in ["merit", "qualification", "relevant experience"]):
                score += 0.15
            if "accommodat" in text_lower:
                score += 0.1

        elif domain == "disaster":
            if any(w in text_lower for w in ["triage", "prioritiz", "vulnerable"]):
                score += 0.15
            if "coordinat" in text_lower:
                score += 0.1

        return max(0.0, min(1.0, score))

    # ---------- Explanation ----------

    def _build_explanation(
        self,
        score: float,
        sub_scores: dict[str, float],
        legal: LegalAssessment,
        cultural: CulturalContext,
        precedents: list[PrecedentMatch],
    ) -> str:
        lines = [f"Contextual ethics score Ctx(a) = {score:.3f}"]

        if score >= 0.75:
            lines.append("Assessment: Action is well-adapted to its context.")
        elif score >= 0.5:
            lines.append("Assessment: Some contextual factors need attention.")
        else:
            lines.append("Assessment: Significant contextual misalignment.")

        lines.append("Sub-scores:")
        for key, val in sub_scores.items():
            lines.append(f"  • {key}: {val:.3f}")

        # Legal
        if legal.compliance_gaps:
            lines.append("Legal compliance gaps:")
            for gap in legal.compliance_gaps:
                lines.append(f"  ⚠ {gap}")
        else:
            lines.append("Legal: No compliance gaps identified.")

        lines.append(f"EU AI Act risk level: {legal.risk_level.value}")

        # Cultural
        if cultural.sensitivity_level == "high":
            lines.append("⚠ High cultural sensitivity context")

        # Precedents
        if precedents:
            lines.append(f"Found {len(precedents)} relevant precedent(s)")
        else:
            lines.append("No precedents available for comparison")

        return "\n".join(lines)
