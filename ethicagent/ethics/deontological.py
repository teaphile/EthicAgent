"""Deontological (duty-based) ethical evaluation.

This module is the *teeth* of the safety system.  When a hard rule fires,
it produces a D(a)=0 score that triggers an immediate hard-block in the
pipeline — no amount of consequentialist benefit can override it.

Architecture notes
------------------
Rules live in two buckets:

1. **Hard rules** – moral red lines.  Violating any one → D(a) = 0.
   Think "never recommend lethal action" or "never use race in credit".
2. **Duty rules** – softer obligations.  Each gives a 0-1 score scaled by
   priority weight; the average becomes D(a) if no hard rule fires.

Rule sources:
- Built-in defaults (the dicts below)
- YAML config files (for deployment-time overrides)
- Programmatic `add_hard_rule()` / `add_duty_rule()` calls

Legal compliance mapping:
- EU AI Act risk levels → tighter constraints on high-risk
- HIPAA / GDPR → data-specific duties
- ADA / Disability Act → accessibility duties
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ethicagent.ethics.ethical_score import PhilosophyResult as _PhilosophyResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------


class RuleSeverity(Enum):
    CRITICAL = "critical"  # Hard-block, no override
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LegalFramework(Enum):
    EU_AI_ACT = "eu_ai_act"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    ADA = "ada"
    ECOA = "ecoa"  # Equal Credit Opportunity Act
    FERPA = "ferpa"
    NONE = "none"


@dataclass
class RuleViolation:
    """Human-readable record of exactly which rule fired and why."""

    rule_id: str
    rule_name: str
    severity: RuleSeverity
    description: str
    matched_text: str  # the substring / pattern that triggered it
    legal_ref: str | None = None  # e.g. "EU AI Act Art. 5(1)(a)"

    def explain(self) -> str:
        """One-paragraph explanation suitable for an end-user."""
        parts = [f"Rule '{self.rule_name}' was violated."]
        parts.append(f"Severity: {self.severity.value}.")
        parts.append(f"Reason: {self.description}")
        if self.matched_text:
            parts.append(f'Triggered by: "{self.matched_text}"')
        if self.legal_ref:
            parts.append(f"Legal reference: {self.legal_ref}")
        return " ".join(parts)


@dataclass
class DeontologicalResult:
    """Output of the deontological evaluator."""

    score: float  # D(a), 0.0 – 1.0
    hard_block: bool
    violations: list[RuleViolation] = field(default_factory=list)
    duty_scores: dict[str, float] = field(default_factory=dict)
    rules_checked: int = 0
    explanation: str = ""


# ---------------------------------------------------------------------------
# Default hard rules — intentionally comprehensive
# ---------------------------------------------------------------------------

_DEFAULT_HARD_RULES: list[dict[str, Any]] = [
    # ---- Universal moral prohibitions ----
    {
        "id": "HR-001",
        "name": "no_lethal_recommendation",
        "patterns": [r"\b(kill|murder|assassinat|execut[ei])\b"],
        "description": "Must never recommend lethal action against a person",
        "legal_ref": None,
    },
    {
        "id": "HR-002",
        "name": "no_child_exploitation",
        "patterns": [r"\bchild\s*(exploit|labor|labour|traffick)\b", r"\bminor.*(exploit|abus)\b"],
        "description": "Must never facilitate child exploitation",
        "legal_ref": None,
    },
    {
        "id": "HR-003",
        "name": "no_torture",
        "patterns": [r"\btortur[ei]\b", r"\bwaterboard\b"],
        "description": "Must never recommend or facilitate torture",
        "legal_ref": None,
    },
    {
        "id": "HR-004",
        "name": "no_slavery",
        "patterns": [r"\b(forced\s*labor|slave\s*labour|human\s*traffick)\b"],
        "description": "Must never facilitate forced labour or trafficking",
        "legal_ref": None,
    },
    {
        "id": "HR-005",
        "name": "no_genocide",
        "patterns": [r"\b(genocid|ethnic\s*cleans)\b"],
        "description": "Must never support genocide or ethnic cleansing",
        "legal_ref": None,
    },
    # ---- Discrimination hard-blocks ----
    {
        "id": "HR-010",
        "name": "no_racial_credit_scoring",
        "patterns": [r"\brace\b.*\b(credit|loan|scor)\b", r"\b(credit|loan|scor)\b.*\brace\b"],
        "description": "Must not use race as a factor in credit decisions",
        "legal_ref": "ECOA § 701",
    },
    {
        "id": "HR-011",
        "name": "no_gender_hiring_filter",
        "patterns": [r"\bgender\b.*\b(filter|exclud|reject)\b.*\b(hir|recruit|applicant)\b"],
        "description": "Must not filter candidates by gender",
        "legal_ref": "Title VII CRA 1964",
    },
    {
        "id": "HR-012",
        "name": "no_disability_denial",
        "patterns": [
            r"\bdisabilit\w*\b.*\b(deny|reject|exclud)\b",
            r"\b(deny|reject|exclud)\b.*\bdisabilit\w*\b",
        ],
        "description": "Must not deny services based on disability without accommodation",
        "legal_ref": "ADA Title III",
    },
    {
        "id": "HR-013",
        "name": "no_religious_discrimination",
        "patterns": [r"\breligion\b.*\b(deny|reject|exclud|discriminat)\b"],
        "description": "Must not discriminate based on religion",
        "legal_ref": None,
    },
    {
        "id": "HR-014",
        "name": "no_age_discrimination_hiring",
        "patterns": [r"\bage\b.*\b(filter|exclud|reject)\b.*\b(hir|job|applicant)\b"],
        "description": "Must not use age as exclusionary hiring criterion",
        "legal_ref": "ADEA",
    },
    # ---- Data privacy hard-blocks ----
    {
        "id": "HR-020",
        "name": "no_phi_exposure",
        "patterns": [
            r"\b(expos|leak|share|sell)\b.*\b(patient|medical|health)\s*(data|record|info)\b"
        ],
        "description": "Must not expose protected health information",
        "legal_ref": "HIPAA § 164.502",
    },
    {
        "id": "HR-021",
        "name": "no_child_data_collection",
        "patterns": [r"\bcollect\b.*\bchild\w*\b.*\bdata\b"],
        "description": "Must not collect data from children without consent",
        "legal_ref": "COPPA",
    },
    {
        "id": "HR-022",
        "name": "no_biometric_without_consent",
        "patterns": [r"\bbiometric\b.*\bwithout\s*(consent|permiss)\b"],
        "description": "Must not process biometric data without explicit consent",
        "legal_ref": "GDPR Art. 9",
    },
    # ---- Safety hard-blocks ----
    {
        "id": "HR-030",
        "name": "no_deny_emergency_care",
        "patterns": [
            r"\bdeny\b.*\bemergency\b.*\b(care|treatment|medical)\b",
            r"\brefuse\b.*\bemergency\b.*\bpatient\b",
        ],
        "description": "Must never deny emergency medical care",
        "legal_ref": "EMTALA",
    },
    {
        "id": "HR-031",
        "name": "no_weapons_instructions",
        "patterns": [r"\b(build|make|assembl)\b.*\b(bomb|weapon|explosiv)\b"],
        "description": "Must not provide instructions for building weapons",
        "legal_ref": None,
    },
]


# ---------------------------------------------------------------------------
# Default duty rules — these produce fractional scores
# ---------------------------------------------------------------------------

_DEFAULT_DUTY_RULES: list[dict[str, Any]] = [
    # ---- Transparency duties ----
    {
        "id": "DT-001",
        "name": "explain_decision",
        "description": "Decisions should be explainable to affected parties",
        "check": "has_explanation",
        "weight": 0.8,
    },
    {
        "id": "DT-002",
        "name": "disclose_ai_involvement",
        "description": "Must disclose that AI was involved in the decision",
        "check": "ai_disclosed",
        "weight": 0.7,
    },
    {
        "id": "DT-003",
        "name": "data_source_transparency",
        "description": "Data sources used should be documented",
        "check": "data_sources_listed",
        "weight": 0.5,
    },
    # ---- Fairness duties ----
    {
        "id": "DT-010",
        "name": "equal_treatment",
        "description": "Similarly situated individuals should receive similar treatment",
        "check": "no_disparate_treatment",
        "weight": 0.9,
    },
    {
        "id": "DT-011",
        "name": "reasonable_accommodation",
        "description": "Provide reasonable accommodation for disabilities",
        "check": "accommodation_considered",
        "weight": 0.7,
    },
    {
        "id": "DT-012",
        "name": "appeal_mechanism",
        "description": "Affected parties should have a way to appeal",
        "check": "appeal_available",
        "weight": 0.6,
    },
    # ---- Privacy duties ----
    {
        "id": "DT-020",
        "name": "data_minimization",
        "description": "Collect only the minimum data necessary",
        "check": "minimal_data",
        "weight": 0.6,
    },
    {
        "id": "DT-021",
        "name": "consent_obtained",
        "description": "User consent should be obtained for data processing",
        "check": "consent_present",
        "weight": 0.8,
    },
    {
        "id": "DT-022",
        "name": "right_to_erasure",
        "description": "Users should be able to request data deletion",
        "check": "erasure_possible",
        "weight": 0.5,
    },
    # ---- Beneficence duties ----
    {
        "id": "DT-030",
        "name": "do_no_harm",
        "description": "Action should not cause foreseeable harm",
        "check": "no_foreseeable_harm",
        "weight": 1.0,
    },
    {
        "id": "DT-031",
        "name": "proportionality",
        "description": "Response should be proportionate to the situation",
        "check": "proportionate_response",
        "weight": 0.7,
    },
    {
        "id": "DT-032",
        "name": "reversibility_preference",
        "description": "Prefer reversible actions over irreversible ones",
        "check": "reversible_action",
        "weight": 0.6,
    },
    # ---- Professional duties ----
    {
        "id": "DT-040",
        "name": "competence_boundary",
        "description": "Operate within established competence boundaries",
        "check": "within_competence",
        "weight": 0.7,
    },
    {
        "id": "DT-041",
        "name": "human_oversight",
        "description": "High-stakes decisions should have human oversight",
        "check": "human_oversight_available",
        "weight": 0.8,
    },
    {
        "id": "DT-042",
        "name": "documentation",
        "description": "Decisions and rationale should be documented",
        "check": "documented",
        "weight": 0.5,
    },
    # ---- Domain-specific duties ----
    # Healthcare
    {
        "id": "DT-050",
        "name": "informed_consent_medical",
        "description": "Medical decisions require informed patient consent",
        "check": "medical_consent",
        "weight": 0.9,
        "domain": "healthcare",
    },
    {
        "id": "DT-051",
        "name": "triage_protocol",
        "description": "Triage must follow established medical protocols",
        "check": "triage_protocol_followed",
        "weight": 0.8,
        "domain": "healthcare",
    },
    # Finance
    {
        "id": "DT-060",
        "name": "adverse_action_notice",
        "description": "Credit denials must include adverse action notice",
        "check": "adverse_notice_provided",
        "weight": 0.9,
        "domain": "finance",
    },
    {
        "id": "DT-061",
        "name": "fair_lending",
        "description": "Lending criteria must be applied uniformly",
        "check": "uniform_criteria",
        "weight": 0.8,
        "domain": "finance",
    },
    # Hiring
    {
        "id": "DT-070",
        "name": "job_relevance",
        "description": "Selection criteria must be job-relevant",
        "check": "criteria_job_relevant",
        "weight": 0.8,
        "domain": "hiring",
    },
    {
        "id": "DT-071",
        "name": "interview_consistency",
        "description": "Interview process should be consistent across candidates",
        "check": "consistent_process",
        "weight": 0.7,
        "domain": "hiring",
    },
    # ---- Cultural sensitivity duties ----
    {
        "id": "DT-080",
        "name": "cultural_awareness",
        "description": "Decisions should account for cultural context",
        "check": "cultural_context_considered",
        "weight": 0.5,
    },
    {
        "id": "DT-081",
        "name": "language_accessibility",
        "description": "Information should be accessible in relevant languages",
        "check": "language_accessible",
        "weight": 0.4,
    },
    # ---- EU AI Act specific duties ----
    {
        "id": "DT-090",
        "name": "risk_assessment",
        "description": "High-risk AI systems must undergo risk assessment",
        "check": "risk_assessed",
        "weight": 0.7,
        "domain_condition": "eu_ai_act_high_risk",
    },
    {
        "id": "DT-091",
        "name": "bias_monitoring",
        "description": "Ongoing bias monitoring for AI decision systems",
        "check": "bias_monitored",
        "weight": 0.6,
    },
]


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------


class DeontologicalEvaluator:
    """Evaluates an action through the lens of duty-based ethics.

    The big idea: certain things are just *wrong*, regardless of outcomes.
    We encode those as hard rules.  Softer duties contribute to the
    overall deontological score proportionally.

    Usage::

        evaluator = DeontologicalEvaluator()
        result = evaluator.evaluate(action_text, context)
        if result.hard_block:
            print("BLOCKED:", result.violations[0].explain())
    """

    def __init__(
        self,
        hard_rules: list[dict[str, Any]] | None = None,
        duty_rules: list[dict[str, Any]] | None = None,
    ) -> None:
        self.hard_rules = list(hard_rules or _DEFAULT_HARD_RULES)
        self.duty_rules = list(duty_rules or _DEFAULT_DUTY_RULES)

        # Pre-compile regex patterns for speed
        self._compiled_hard: list[tuple[dict, list[re.Pattern]]] = []
        for rule in self.hard_rules:
            patterns = [re.compile(p, re.IGNORECASE) for p in rule.get("patterns", [])]
            self._compiled_hard.append((rule, patterns))

        logger.debug(
            "DeontologicalEvaluator initialized: %d hard rules, %d duty rules",
            len(self.hard_rules),
            len(self.duty_rules),
        )

    # ---------- Public API ----------

    def evaluate(
        self,
        action_text: str,
        context: dict[str, Any] | None = None,
    ) -> DeontologicalResult:
        """Run the full deontological evaluation.

        Parameters
        ----------
        action_text : str
            The proposed action or recommendation to evaluate.
        context : dict, optional
            Pipeline context (domain, metadata, stakeholders, etc.)

        Returns
        -------
        DeontologicalResult
            Contains score D(a), hard_block flag, violations, duty scores.
        """
        context = context or {}
        violations: list[RuleViolation] = []
        domain = context.get("domain", "general")

        # --- Phase 1: check hard rules ---
        for rule, patterns in self._compiled_hard:
            for pat in patterns:
                match = pat.search(action_text)
                if match:
                    violation = RuleViolation(
                        rule_id=rule["id"],
                        rule_name=rule["name"],
                        severity=RuleSeverity.CRITICAL,
                        description=rule["description"],
                        matched_text=match.group(0),
                        legal_ref=rule.get("legal_ref"),
                    )
                    violations.append(violation)
                    logger.warning(
                        "Hard rule %s (%s) triggered: '%s'",
                        rule["id"],
                        rule["name"],
                        match.group(0),
                    )
                    break  # one match per rule is enough

        if violations:
            _dr = DeontologicalResult(
                score=0.0,
                hard_block=True,
                violations=violations,
                duty_scores={},
                rules_checked=len(self._compiled_hard),
                explanation=self._build_block_explanation(violations),
            )
            return self._wrap(_dr)

        # --- Phase 2: evaluate duty rules ---
        duty_scores: dict[str, float] = {}
        total_weight = 0.0
        weighted_sum = 0.0

        for duty in self.duty_rules:
            # Skip domain-specific duties that don't apply
            rule_domain = duty.get("domain")
            if rule_domain and rule_domain != domain:
                continue

            # Skip EU AI Act conditions if not applicable
            if duty.get("domain_condition") == "eu_ai_act_high_risk":
                if not context.get("eu_ai_act_high_risk", False):
                    continue

            check_name = duty["check"]
            weight = duty["weight"]
            score = self._check_duty(check_name, action_text, context)
            duty_scores[duty["id"]] = score

            weighted_sum += score * weight
            total_weight += weight

        # Compute weighted average
        if total_weight > 0:
            d_score = weighted_sum / total_weight
        else:
            # No applicable duties → assume compliant
            d_score = 0.85

        # Mild penalties can still produce soft violations
        soft_violations = self._check_soft_violations(action_text, context)
        violations.extend(soft_violations)

        rules_checked = len(self._compiled_hard) + len(duty_scores)

        _dr = DeontologicalResult(
            score=round(d_score, 4),
            hard_block=False,
            violations=violations,
            duty_scores=duty_scores,
            rules_checked=rules_checked,
            explanation=self._build_duty_explanation(d_score, duty_scores, violations),
        )
        return self._wrap(_dr)

    @staticmethod
    def _wrap(dr: DeontologicalResult) -> _PhilosophyResult:
        """Convert domain-specific result to PhilosophyResult."""
        return _PhilosophyResult(
            philosophy="deontological",
            score=dr.score,
            hard_block=dr.hard_block,
            reasoning=dr.explanation,
            key_argument=dr.explanation,
            details={
                "violations": [
                    v.__dict__ if hasattr(v, "__dict__") else str(v) for v in dr.violations
                ],
                "duty_scores": dr.duty_scores,
                "rules_checked": dr.rules_checked,
            },
        )

    def add_hard_rule(self, rule: dict[str, Any]) -> None:
        """Add a hard rule at runtime (e.g. from YAML config)."""
        patterns = [re.compile(p, re.IGNORECASE) for p in rule.get("patterns", [])]
        self.hard_rules.append(rule)
        self._compiled_hard.append((rule, patterns))
        logger.info("Added hard rule: %s", rule.get("id", "unknown"))

    def add_duty_rule(self, rule: dict[str, Any]) -> None:
        """Add a duty rule at runtime."""
        self.duty_rules.append(rule)
        logger.info("Added duty rule: %s", rule.get("id", "unknown"))

    def explain_violation(self, violation: RuleViolation) -> str:
        """Return a detailed human-readable explanation of a violation."""
        return violation.explain()

    def get_legal_obligations(self, domain: str) -> list[str]:
        """Return legal frameworks relevant to a domain."""
        mapping = {
            "healthcare": [LegalFramework.HIPAA, LegalFramework.ADA],
            "finance": [LegalFramework.ECOA, LegalFramework.GDPR],
            "hiring": [LegalFramework.ADA, LegalFramework.ECOA],
            "disaster": [],
            "general": [LegalFramework.GDPR],
        }
        frameworks = mapping.get(domain, [])
        return [f.value for f in frameworks]

    # ---------- Internal helpers ----------

    def _check_duty(
        self,
        check_name: str,
        action_text: str,
        context: dict[str, Any],
    ) -> float:
        """Evaluate a single duty check.  Returns 0.0 – 1.0.

        This is intentionally heuristic-based.  In a production system
        you'd wire these to actual policy checks; here we use text
        signals as proxies.
        """
        # HACK: These are text-based heuristics.  Good enough for research,
        #       but a real deployment would integrate with actual compliance
        #       APIs and policy engines.
        text_lower = action_text.lower()
        meta = context.get("metadata", {})

        checks: dict[str, float] = {
            # Transparency
            "has_explanation": (
                0.9
                if any(w in text_lower for w in ["because", "reason", "explain", "due to"])
                else 0.4
            ),
            "ai_disclosed": (
                1.0
                if meta.get("ai_disclosed", False)
                else 0.6
                if "ai" in text_lower or "algorithm" in text_lower
                else 0.3
            ),
            "data_sources_listed": (0.8 if meta.get("data_sources") else 0.5),
            # Fairness
            "no_disparate_treatment": (
                0.3 if any(w in text_lower for w in ["only for", "except", "excluding"]) else 0.85
            ),
            "accommodation_considered": (
                0.9 if "accommodat" in text_lower or meta.get("accommodation_offered") else 0.6
            ),
            "appeal_available": (
                1.0
                if meta.get("appeal_mechanism", False)
                else 0.7
                if "appeal" in text_lower
                else 0.5
            ),
            # Privacy
            "minimal_data": (
                0.4
                if any(w in text_lower for w in ["all data", "everything", "comprehensive data"])
                else 0.8
            ),
            "consent_present": (1.0 if meta.get("consent_obtained", False) else 0.5),
            "erasure_possible": (1.0 if meta.get("erasure_supported", False) else 0.6),
            # Beneficence
            "no_foreseeable_harm": (
                0.3 if any(w in text_lower for w in ["harm", "damage", "injur", "suffer"]) else 0.85
            ),
            "proportionate_response": 0.75,  # hard to detect from text alone
            "reversible_action": (
                0.4
                if any(w in text_lower for w in ["permanent", "irreversible", "cannot undo"])
                else 0.8
            ),
            # Professional
            "within_competence": 0.8,
            "human_oversight_available": (1.0 if meta.get("human_oversight", False) else 0.6),
            "documented": (0.9 if meta.get("documented", False) else 0.5),
            # Domain-specific
            "medical_consent": (1.0 if meta.get("patient_consent", False) else 0.4),
            "triage_protocol_followed": (
                0.9 if "triage" in text_lower and "protocol" in text_lower else 0.6
            ),
            "adverse_notice_provided": (1.0 if meta.get("adverse_notice", False) else 0.4),
            "uniform_criteria": (
                0.3
                if any(w in text_lower for w in ["special case", "exception", "override"])
                else 0.8
            ),
            "criteria_job_relevant": 0.75,  # default assumption
            "consistent_process": 0.75,
            # Cultural
            "cultural_context_considered": (0.8 if meta.get("cultural_context") else 0.6),
            "language_accessible": (1.0 if meta.get("multilingual", False) else 0.5),
            # Compliance
            "risk_assessed": (1.0 if meta.get("risk_assessment_done", False) else 0.4),
            "bias_monitored": (1.0 if meta.get("bias_monitoring", False) else 0.5),
        }

        return checks.get(check_name, 0.5)

    def _check_soft_violations(
        self,
        action_text: str,
        context: dict[str, Any],
    ) -> list[RuleViolation]:
        """Check for soft violations — things that are concerning but
        not hard-block level.  These get reported but don't zero D(a)."""
        violations = []
        text_lower = action_text.lower()

        # Warn if action mentions sensitive attributes without justification
        sensitive_attrs = ["race", "gender", "religion", "ethnicity", "sexuality"]
        for attr in sensitive_attrs:
            if attr in text_lower:
                violations.append(
                    RuleViolation(
                        rule_id=f"SOFT-{attr.upper()}",
                        rule_name=f"sensitive_attribute_{attr}",
                        severity=RuleSeverity.MEDIUM,
                        description=f"Action references '{attr}' — ensure it's justified",
                        matched_text=attr,
                    )
                )

        # Warn about irreversible actions
        if any(w in text_lower for w in ["permanent", "irreversible", "delete all"]):
            violations.append(
                RuleViolation(
                    rule_id="SOFT-IRREVERSIBLE",
                    rule_name="irreversible_action_warning",
                    severity=RuleSeverity.HIGH,
                    description="Action appears irreversible — extra caution advised",
                    matched_text="irreversible action detected",
                )
            )

        return violations

    def _build_block_explanation(self, violations: list[RuleViolation]) -> str:
        """Build explanation text for a hard-block."""
        lines = ["HARD BLOCK — deontological constraint violated:"]
        for v in violations:
            lines.append(f"  • [{v.rule_id}] {v.rule_name}: {v.description}")
            if v.legal_ref:
                lines.append(f"    Legal ref: {v.legal_ref}")
        return "\n".join(lines)

    def _build_duty_explanation(
        self,
        score: float,
        duty_scores: dict[str, float],
        violations: list[RuleViolation],
    ) -> str:
        """Build explanation text summarizing duty evaluation."""
        lines = [f"Deontological score D(a) = {score:.3f}"]

        if score >= 0.8:
            lines.append("Assessment: Duties substantially met.")
        elif score >= 0.5:
            lines.append("Assessment: Some duties partially unmet — review recommended.")
        else:
            lines.append("Assessment: Significant duty concerns identified.")

        # Show lowest-scoring duties
        if duty_scores:
            sorted_duties = sorted(duty_scores.items(), key=lambda x: x[1])
            weakest = sorted_duties[:3]
            if weakest:
                lines.append("Weakest areas:")
                for did, ds in weakest:
                    lines.append(f"  • {did}: {ds:.2f}")

        if violations:
            lines.append(f"Soft warnings: {len(violations)}")

        return "\n".join(lines)
