"""Canonical EDS formula and ethical verdict definitions.

This is the "math" module — it owns the EDS formula and the
EthicalVerdict / EthicalDecision types that flow through the pipeline.

EDS(a) = w₁·D(a) + w₂·C(a) + w₃·V(a) + w₄·Ctx(a)

where:
  D(a)   = Deontological score  (duty-based)
  C(a)   = Consequentialist score (outcome-based)
  V(a)   = Virtue/fairness score
  Ctx(a) = Contextual score

Decision thresholds:
  EDS ≥ 0.80           → AUTO_APPROVE
  0.50 ≤ EDS < 0.80    → ESCALATE (needs human review)
  EDS < 0.50            → REJECT
  D(a) == 0             → HARD_BLOCK (overrides everything)

The domain weights are the most controversial part of this system.
See the DOMAIN_WEIGHTS dict for the current calibration — we spent
a *lot* of time tuning these based on ethicist feedback and scenario
testing.  They're still not perfect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verdict enum — these are the pipeline's final outputs
# ---------------------------------------------------------------------------


class EthicalVerdict(str, Enum):
    """The four possible outcomes of the ethics pipeline."""

    AUTO_APPROVE = "auto_approve"
    ESCALATE = "escalate"
    REJECT = "reject"
    HARD_BLOCK = "hard_block"

    def __str__(self) -> str:
        return self.value

    @property
    def is_safe(self) -> bool:
        return self == EthicalVerdict.AUTO_APPROVE

    @property
    def needs_review(self) -> bool:
        return self == EthicalVerdict.ESCALATE

    @property
    def is_blocked(self) -> bool:
        return self in (EthicalVerdict.REJECT, EthicalVerdict.HARD_BLOCK)


# ---------------------------------------------------------------------------
# Philosophy result — standardized output from each evaluator
# ---------------------------------------------------------------------------


@dataclass
class PhilosophyResult:
    """Standardized result from a philosophy evaluator."""

    philosophy: str  # "deontological", "consequentialist", "virtue_ethics", "contextual"
    score: float  # 0.0 – 1.0
    hard_block: bool = False
    confidence: float = 0.7
    key_argument: str = ""
    reasoning: str = ""  # backward-compat alias for key_argument
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Sync reasoning ↔ key_argument
        if self.reasoning and not self.key_argument:
            self.key_argument = self.reasoning
        elif self.key_argument and not self.reasoning:
            self.reasoning = self.key_argument


# ---------------------------------------------------------------------------
# Ethical decision — the final pipeline output
# ---------------------------------------------------------------------------


@dataclass
class EthicalDecision:
    """Complete ethical assessment produced by the pipeline.

    This is the object that gets logged, displayed to users, and
    used for downstream decisions.  It's designed to be both
    machine-readable and human-explainable.
    """

    eds_score: float
    verdict: EthicalVerdict
    philosophy_scores: dict[str, float] = field(default_factory=dict)
    domain: str = "general"
    weights_used: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    confidence_interval: tuple[float, float] = (0.0, 1.0)
    sensitivity: dict[str, float] = field(default_factory=dict)
    philosophy_results: list[PhilosophyResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary (backward-compat)."""
        return str(self)

    def __repr__(self) -> str:
        return (
            f"EthicalDecision(eds={self.eds_score:.3f}, "
            f"verdict={self.verdict.value}, domain={self.domain})"
        )

    def __str__(self) -> str:
        lines = [
            f"EDS Score: {self.eds_score:.4f}",
            f"Verdict:   {self.verdict.value}",
            f"Domain:    {self.domain}",
            f"CI:        [{self.confidence_interval[0]:.3f}, {self.confidence_interval[1]:.3f}]",
            "",
            "Philosophy scores:",
        ]
        for phil, score in self.philosophy_scores.items():
            weight = self.weights_used.get(phil, 0.0)
            lines.append(f"  {phil:20s}: {score:.3f} (weight={weight:.2f})")
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        lines.append("")
        lines.append(f"Reasoning: {self.reasoning}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON/logging."""
        return {
            "eds_score": self.eds_score,
            "verdict": self.verdict.value,
            "philosophy_scores": self.philosophy_scores,
            "domain": self.domain,
            "weights_used": self.weights_used,
            "reasoning": self.reasoning,
            "confidence_interval": list(self.confidence_interval),
            "sensitivity": self.sensitivity,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Domain weight definitions
# ---------------------------------------------------------------------------

DOMAIN_WEIGHTS: dict[str, dict[str, float]] = {
    "healthcare": {
        "deontological": 0.35,
        "consequentialist": 0.25,
        "virtue_ethics": 0.20,
        "contextual": 0.20,
    },
    "finance": {
        "deontological": 0.20,
        "consequentialist": 0.25,
        "virtue_ethics": 0.35,
        "contextual": 0.20,
    },
    "hiring": {
        "deontological": 0.15,
        "consequentialist": 0.20,
        "virtue_ethics": 0.40,
        "contextual": 0.25,
    },
    "disaster": {
        "deontological": 0.20,
        "consequentialist": 0.35,
        "virtue_ethics": 0.15,
        "contextual": 0.30,
    },
    "general": {
        "deontological": 0.25,
        "consequentialist": 0.25,
        "virtue_ethics": 0.25,
        "contextual": 0.25,
    },
}

# Decision thresholds — kept here so they're in one place
APPROVAL_THRESHOLD = 0.80
ESCALATION_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# EDS formula functions
# ---------------------------------------------------------------------------


def compute_eds(
    philosophy_scores: dict[str, float],
    domain: str = "general",
    custom_weights: dict[str, float] | None = None,
) -> float:
    """Compute the Ethical Decision Score.

    EDS(a) = w₁·D(a) + w₂·C(a) + w₃·V(a) + w₄·Ctx(a)

    Parameters
    ----------
    philosophy_scores : dict
        Keys: "deontological", "consequentialist", "virtue_ethics", "contextual"
        Values: 0.0 – 1.0
    domain : str
        Domain for weight selection.
    custom_weights : dict, optional
        Override default domain weights.

    Returns
    -------
    float
        EDS score in [0, 1].
    """
    weights = custom_weights or DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS["general"])

    eds = 0.0
    total_weight = 0.0

    for philosophy, weight in weights.items():
        score = philosophy_scores.get(philosophy, 0.5)  # default to neutral
        eds += weight * score
        total_weight += weight

    # Normalize if weights don't sum to 1
    if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
        eds /= total_weight

    return round(max(0.0, min(1.0, eds)), 4)


def determine_verdict(
    eds_score: float,
    deontological_score: float = 1.0,
    hard_block: bool = False,
) -> EthicalVerdict:
    """Map EDS score to verdict.

    Parameters
    ----------
    eds_score : float
        The computed EDS score.
    deontological_score : float
        D(a) — if zero, triggers hard-block.
    hard_block : bool
        Explicit hard-block flag from deontological evaluator.
    """
    # Hard-block overrides everything
    if hard_block or deontological_score == 0.0:
        return EthicalVerdict.HARD_BLOCK

    if eds_score >= APPROVAL_THRESHOLD:
        return EthicalVerdict.AUTO_APPROVE
    if eds_score >= ESCALATION_THRESHOLD:
        return EthicalVerdict.ESCALATE
    return EthicalVerdict.REJECT


def compute_confidence_interval(
    philosophy_scores: dict[str, float],
    weights: dict[str, float],
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Approximate confidence interval for the EDS score.

    We treat each philosophy score as having inherent uncertainty
    (±0.05-0.15 depending on the method) and propagate that through
    the weighted sum.
    """
    # Estimation of per-philosophy uncertainty
    uncertainties = {
        "deontological": 0.05,  # rules are fairly crisp
        "consequentialist": 0.12,  # outcome estimation is uncertain
        "virtue_ethics": 0.10,  # heuristic-based
        "contextual": 0.08,  # context-dependent
    }

    # Propagate uncertainty through weighted sum
    variance = 0.0
    for phil, weight in weights.items():
        sigma = uncertainties.get(phil, 0.10)
        variance += (weight * sigma) ** 2

    import math

    std = math.sqrt(variance)

    # z-score for confidence level
    z = 1.96 if confidence_level >= 0.95 else 1.645

    eds = compute_eds(philosophy_scores, custom_weights=weights)
    lower = max(0.0, eds - z * std)
    upper = min(1.0, eds + z * std)

    return (round(lower, 4), round(upper, 4))


def sensitivity_analysis(
    philosophy_scores: dict[str, float],
    domain: str = "general",
    perturbation: float = 0.05,
) -> dict[str, float]:
    """Check how sensitive EDS is to small changes in each philosophy score.

    For each philosophy, we perturb its weight by ±perturbation and
    measure the EDS change.  Larger changes = more sensitive.
    """
    base_eds = compute_eds(philosophy_scores, domain=domain)
    weights = DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS["general"])
    sensitivities: dict[str, float] = {}

    for phil in weights:
        # Perturb weight up
        w_up = dict(weights)
        w_up[phil] = min(1.0, w_up[phil] + perturbation)
        eds_up = compute_eds(philosophy_scores, custom_weights=w_up)

        # Perturb weight down
        w_down = dict(weights)
        w_down[phil] = max(0.0, w_down[phil] - perturbation)
        eds_down = compute_eds(philosophy_scores, custom_weights=w_down)

        sensitivities[phil] = round(abs(eds_up - eds_down), 4)

    return sensitivities
