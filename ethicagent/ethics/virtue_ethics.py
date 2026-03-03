"""Virtue ethics and fairness metrics evaluator.

This module handles V(a) — the virtue-ethics component of EDS.  Despite
the name, it does double duty:

  1. **Classical virtue analysis** — does the action embody virtues like
     justice, compassion, prudence, honesty, courage?
  2. **Quantitative fairness metrics** — SPD, DIR, EOD, predictive parity,
     calibration, and vulnerable population detection.

The rationale: fairness IS a virtue, and it's one we can actually measure.
So we blend the qualitative virtue assessment with hard fairness numbers.

The fairness metrics follow standard ML fairness literature:
- Statistical Parity Difference (SPD)
- Disparate Impact Ratio (DIR)
- Equal Opportunity Difference (EOD)
- Predictive Parity Difference
- Calibration Difference

When we don't have actual group-level data, we fall back to heuristic
text analysis — looking for signals of fairness/unfairness.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ethicagent.ethics.ethical_score import PhilosophyResult as _PhilosophyResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Virtue(Enum):
    """The core virtues we evaluate.  Based on Aristotelian ethics with
    modern additions for AI contexts."""
    JUSTICE = "justice"
    COMPASSION = "compassion"
    PRUDENCE = "prudence"
    HONESTY = "honesty"
    COURAGE = "courage"
    TEMPERANCE = "temperance"
    RESPECT = "respect"
    RESPONSIBILITY = "responsibility"


class BiasType(Enum):
    """Types of bias we can detect."""
    RACIAL = "racial"
    GENDER = "gender"
    AGE = "age"
    DISABILITY = "disability"
    SOCIOECONOMIC = "socioeconomic"
    RELIGIOUS = "religious"
    NONE = "none"


@dataclass
class FairnessMetrics:
    """Standard fairness metrics for group comparisons."""
    spd: Optional[float] = None   # Statistical Parity Difference
    dir_: Optional[float] = None  # Disparate Impact Ratio (dir is a builtin, hence dir_)
    eod: Optional[float] = None   # Equal Opportunity Difference
    ppd: Optional[float] = None   # Predictive Parity Difference
    calibration_diff: Optional[float] = None
    groups_compared: tuple[str, str] = ("", "")

    @property
    def overall_fairness(self) -> float:
        """Quick aggregate fairness score, 0 = unfair, 1 = perfectly fair."""
        scores = []
        if self.spd is not None:
            scores.append(1.0 - min(abs(self.spd), 1.0))
        if self.dir_ is not None:
            # DIR ∈ (0, ∞), 1.0 is fair.  Penalize deviation from 1.
            scores.append(1.0 - min(abs(self.dir_ - 1.0), 1.0))
        if self.eod is not None:
            scores.append(1.0 - min(abs(self.eod), 1.0))
        if self.ppd is not None:
            scores.append(1.0 - min(abs(self.ppd), 1.0))
        if self.calibration_diff is not None:
            scores.append(1.0 - min(abs(self.calibration_diff), 1.0))
        return sum(scores) / len(scores) if scores else 0.5


@dataclass
class VulnerablePopulation:
    """Detected vulnerable group and potential impact."""
    group_name: str
    detected_from: str  # what text/signal triggered detection
    risk_level: str     # "low", "medium", "high"
    recommendation: str


@dataclass
class VirtueEthicsResult:
    """Output of the virtue ethics evaluator."""
    score: float  # V(a), 0.0 – 1.0
    virtue_scores: dict[str, float] = field(default_factory=dict)
    fairness_metrics: Optional[FairnessMetrics] = None
    bias_detected: list[BiasType] = field(default_factory=list)
    vulnerable_populations: list[VulnerablePopulation] = field(default_factory=list)
    explanation: str = ""


# ---------------------------------------------------------------------------
# Keyword banks for heuristic analysis
# ---------------------------------------------------------------------------

_VIRTUE_SIGNALS: dict[Virtue, dict[str, list[str]]] = {
    Virtue.JUSTICE: {
        "positive": ["fair", "equitable", "impartial", "equal", "justice", "unbiased",
                      "merit-based", "transparent", "accountable"],
        "negative": ["unfair", "biased", "discriminat", "prejudic", "partial",
                      "favoritism", "nepotism", "rigged"],
    },
    Virtue.COMPASSION: {
        "positive": ["compassion", "empathy", "care", "sympathy", "kindness",
                      "humanitarian", "mercy", "understanding", "support"],
        "negative": ["cruel", "callous", "heartless", "indifferent", "neglect",
                      "abandon", "cold", "apathetic"],
    },
    Virtue.PRUDENCE: {
        "positive": ["careful", "cautious", "prudent", "measured", "thoughtful",
                      "considered", "wise", "deliberate", "risk-aware"],
        "negative": ["reckless", "hasty", "impulsive", "careless", "negligent",
                      "rash", "irresponsible"],
    },
    Virtue.HONESTY: {
        "positive": ["honest", "truthful", "transparent", "candid", "forthright",
                      "sincere", "authentic", "disclose"],
        "negative": ["dishonest", "deceiv", "mislead", "manipulat", "lie",
                      "conceal", "cover-up", "fabricat"],
    },
    Virtue.COURAGE: {
        "positive": ["courageous", "brave", "bold", "principled", "stand up",
                      "speak out", "challenge", "confront"],
        "negative": ["coward", "capitulat", "cave in", "abandon principle"],
    },
    Virtue.TEMPERANCE: {
        "positive": ["moderate", "balanced", "restrained", "proportionate",
                      "measured", "reasonable"],
        "negative": ["extreme", "excessive", "disproportionate", "overreact",
                      "heavy-handed"],
    },
    Virtue.RESPECT: {
        "positive": ["respect", "dignit", "autonom", "consent", "privacy",
                      "rights", "agency"],
        "negative": ["disrespect", "dehumaniz", "objectif", "degrad",
                      "violate privacy", "strip agency"],
    },
    Virtue.RESPONSIBILITY: {
        "positive": ["responsible", "accountable", "duty", "obligat",
                      "stewardship", "oversight", "monitor"],
        "negative": ["irresponsible", "unaccountable", "negligent", "evade",
                      "pass the buck", "deflect blame"],
    },
}

_BIAS_SIGNALS: dict[BiasType, list[str]] = {
    BiasType.RACIAL: ["race", "racial", "ethnicity", "ethnic", "skin color",
                       "african american", "hispanic", "caucasian", "asian"],
    BiasType.GENDER: ["gender", "sex", "male", "female", "man", "woman",
                       "masculine", "feminine", "non-binary"],
    BiasType.AGE: ["age", "elderly", "senior", "young", "millennial",
                    "boomer", "generation"],
    BiasType.DISABILITY: ["disability", "disabled", "handicap", "impairment",
                           "wheelchair", "blind", "deaf", "mental health"],
    BiasType.SOCIOECONOMIC: ["income", "poverty", "wealthy", "poor", "zip code",
                              "neighborhood", "socioeconomic", "class"],
    BiasType.RELIGIOUS: ["religion", "religious", "muslim", "christian",
                          "jewish", "hindu", "buddhist", "atheist"],
}

_VULNERABLE_GROUPS = {
    "children": ["child", "minor", "kid", "infant", "toddler", "teen"],
    "elderly": ["elderly", "senior citizen", "aged", "geriatric"],
    "disabled": ["disabled", "disability", "handicap", "wheelchair",
                  "visually impaired", "hearing impaired"],
    "pregnant": ["pregnant", "expectant mother", "prenatal"],
    "refugees": ["refugee", "asylum seeker", "displaced"],
    "homeless": ["homeless", "houseless", "unhoused"],
    "low_income": ["low-income", "poverty", "impoverished", "food insecure"],
    "indigenous": ["indigenous", "native", "aboriginal", "tribal"],
    "undocumented": ["undocumented", "illegal immigrant", "unauthorized"],
    "mental_health": ["mental illness", "psychiatric", "suicidal", "depression"],
}


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class VirtueEthicsEvaluator:
    """Evaluates V(a) by combining virtue assessment with fairness metrics.

    The final score is: V(a) = 0.5 * virtue_avg + 0.3 * fairness + 0.2 * vulnerability_adjustment
    """

    def __init__(
        self,
        virtue_weight: float = 0.5,
        fairness_weight: float = 0.3,
        vulnerability_weight: float = 0.2,
    ) -> None:
        self.virtue_weight = virtue_weight
        self.fairness_weight = fairness_weight
        self.vulnerability_weight = vulnerability_weight

    def evaluate(
        self,
        action_text: str,
        context: dict[str, Any] | None = None,
        group_data: dict[str, Any] | None = None,
    ) -> VirtueEthicsResult:
        """Run the full virtue ethics + fairness evaluation.

        Parameters
        ----------
        action_text : str
            The action to evaluate.
        context : dict
            Pipeline context.
        group_data : dict, optional
            Structured fairness data with group outcomes:
            {
                "group_a": {"name": "...", "positive_rate": 0.8, ...},
                "group_b": {"name": "...", "positive_rate": 0.6, ...},
            }
        """
        context = context or {}
        text_lower = action_text.lower()

        # --- Virtue analysis ---
        virtue_scores = self._evaluate_virtues(text_lower, context)
        virtue_avg = sum(virtue_scores.values()) / len(virtue_scores) if virtue_scores else 0.5

        # --- Fairness metrics ---
        if group_data:
            fairness = self._compute_fairness_metrics(group_data)
            fairness_score = fairness.overall_fairness
        else:
            fairness = self._infer_fairness(text_lower, context)
            fairness_score = fairness.overall_fairness if fairness else 0.5

        # --- Bias detection ---
        bias_detected = self._detect_bias_signals(text_lower, context)

        # --- Vulnerable population detection ---
        vulnerable = self._detect_vulnerable_populations(text_lower, context)

        # Vulnerability adjustment: penalize if vulnerable groups at risk
        vulnerability_score = 1.0
        high_risk = [v for v in vulnerable if v.risk_level == "high"]
        medium_risk = [v for v in vulnerable if v.risk_level == "medium"]
        if high_risk:
            vulnerability_score -= 0.3 * len(high_risk)
        if medium_risk:
            vulnerability_score -= 0.1 * len(medium_risk)
        vulnerability_score = max(0.0, vulnerability_score)

        # Additional penalty for detected biases
        if bias_detected:
            # Not all bias mentions are bad — context matters.
            # But in a hiring/finance context, mentioning protected categories
            # as decision factors is a problem.
            domain = context.get("domain", "general")
            if domain in ("hiring", "finance"):
                fairness_score *= max(0.5, 1.0 - 0.15 * len(bias_detected))

        # --- Combine ---
        v_score = (
            self.virtue_weight * virtue_avg
            + self.fairness_weight * fairness_score
            + self.vulnerability_weight * vulnerability_score
        )
        v_score = max(0.0, min(1.0, v_score))

        _vr = VirtueEthicsResult(
            score=round(v_score, 4),
            virtue_scores={v.value: round(s, 4) for v, s in virtue_scores.items()},
            fairness_metrics=fairness,
            bias_detected=bias_detected,
            vulnerable_populations=vulnerable,
            explanation=self._build_explanation(
                v_score, virtue_scores, fairness_score,
                bias_detected, vulnerable,
            ),
        )
        return self._wrap(_vr)

    @staticmethod
    def _wrap(vr: VirtueEthicsResult) -> _PhilosophyResult:
        return _PhilosophyResult(
            philosophy="virtue_ethics",
            score=vr.score,
            reasoning=vr.explanation,
            key_argument=vr.explanation,
            details={"virtue_scores": vr.virtue_scores},
        )

    # ---------- Virtue scoring ----------

    def _evaluate_virtues(
        self,
        text_lower: str,
        context: dict[str, Any],
    ) -> dict[Virtue, float]:
        """Score each virtue based on text signals."""
        scores: dict[Virtue, float] = {}

        for virtue, signals in _VIRTUE_SIGNALS.items():
            pos_count = sum(1 for w in signals["positive"] if w in text_lower)
            neg_count = sum(1 for w in signals["negative"] if w in text_lower)

            # Start at neutral
            base = 0.5

            # Positive signals push towards 1
            if pos_count > 0:
                base += min(0.4, 0.1 * pos_count)

            # Negative signals push towards 0
            if neg_count > 0:
                base -= min(0.4, 0.15 * neg_count)  # negatives are weighted more

            scores[virtue] = max(0.0, min(1.0, base))

        # Domain-specific virtue adjustments
        domain = context.get("domain", "general")
        if domain == "healthcare":
            # Compassion matters more in healthcare
            if Virtue.COMPASSION in scores:
                scores[Virtue.COMPASSION] = scores[Virtue.COMPASSION] * 1.15
                scores[Virtue.COMPASSION] = min(1.0, scores[Virtue.COMPASSION])
        elif domain == "finance":
            # Justice/fairness matters more in finance
            if Virtue.JUSTICE in scores:
                scores[Virtue.JUSTICE] = scores[Virtue.JUSTICE] * 1.15
                scores[Virtue.JUSTICE] = min(1.0, scores[Virtue.JUSTICE])

        return scores

    # ---------- Fairness metrics ----------

    def _compute_fairness_metrics(
        self,
        group_data: dict[str, Any],
    ) -> FairnessMetrics:
        """Compute standard fairness metrics from structured group data.

        Expected format::
            {
                "group_a": {"name": "Male", "positive_rate": 0.8, "tpr": 0.85, "ppv": 0.9},
                "group_b": {"name": "Female", "positive_rate": 0.6, "tpr": 0.7, "ppv": 0.85},
            }
        """
        ga = group_data.get("group_a", {})
        gb = group_data.get("group_b", {})

        pr_a = ga.get("positive_rate", 0.5)
        pr_b = gb.get("positive_rate", 0.5)
        tpr_a = ga.get("tpr")  # true positive rate
        tpr_b = gb.get("tpr")
        ppv_a = ga.get("ppv")  # positive predictive value
        ppv_b = gb.get("ppv")
        cal_a = ga.get("calibration")
        cal_b = gb.get("calibration")

        # Statistical Parity Difference: |P(Ŷ=1|A=a) - P(Ŷ=1|A=b)|
        spd = pr_a - pr_b

        # Disparate Impact Ratio: P(Ŷ=1|A=b) / P(Ŷ=1|A=a)
        dir_ = (pr_b / pr_a) if pr_a > 0 else None

        # Equal Opportunity Difference
        eod = (tpr_a - tpr_b) if (tpr_a is not None and tpr_b is not None) else None

        # Predictive Parity Difference
        ppd = (ppv_a - ppv_b) if (ppv_a is not None and ppv_b is not None) else None

        # Calibration difference
        cal_diff = None
        if cal_a is not None and cal_b is not None:
            cal_diff = cal_a - cal_b

        return FairnessMetrics(
            spd=round(spd, 4),
            dir_=round(dir_, 4) if dir_ is not None else None,
            eod=round(eod, 4) if eod is not None else None,
            ppd=round(ppd, 4) if ppd is not None else None,
            calibration_diff=round(cal_diff, 4) if cal_diff is not None else None,
            groups_compared=(ga.get("name", "A"), gb.get("name", "B")),
        )

    def _infer_fairness(
        self,
        text_lower: str,
        context: dict[str, Any],
    ) -> FairnessMetrics:
        """Heuristic fairness estimate when no structured data available."""
        # We can't compute real metrics, but we can flag concerns
        fairness_positive = any(w in text_lower for w in [
            "fair", "equitable", "equal", "impartial", "unbiased",
            "merit-based", "objective criteria",
        ])
        fairness_negative = any(w in text_lower for w in [
            "discriminat", "biased", "unfair", "prejudic", "exclud",
            "disparate", "disproportionate",
        ])

        # Estimate SPD from text
        if fairness_negative:
            est_spd = 0.3  # concerning
        elif fairness_positive:
            est_spd = 0.05  # looks fair
        else:
            est_spd = 0.15  # unknown

        return FairnessMetrics(
            spd=est_spd,
            dir_=max(0.1, 1.0 - est_spd),
            groups_compared=("estimated", "estimated"),
        )

    # ---------- Bias detection ----------

    def _detect_bias_signals(
        self,
        text_lower: str,
        context: dict[str, Any],
    ) -> list[BiasType]:
        """Detect which types of bias might be present."""
        detected = []
        for bias_type, signals in _BIAS_SIGNALS.items():
            if any(sig in text_lower for sig in signals):
                # Check if it's in a discriminatory context
                disc_context = any(w in text_lower for w in [
                    "deny", "reject", "exclude", "filter", "discriminat",
                    "score lower", "penalize", "disadvantage",
                ])
                if disc_context:
                    detected.append(bias_type)
                    logger.info("Bias signal detected: %s", bias_type.value)

        return detected

    # ---------- Vulnerable population detection ----------

    def _detect_vulnerable_populations(
        self,
        text_lower: str,
        context: dict[str, Any],
    ) -> list[VulnerablePopulation]:
        """Find mentions of vulnerable populations and assess risk."""
        found: list[VulnerablePopulation] = []

        harm_context = any(w in text_lower for w in [
            "deny", "reject", "exclude", "harm", "risk", "danger",
            "reduce", "cut", "eliminate", "deprioritize",
        ])
        benefit_context = any(w in text_lower for w in [
            "help", "protect", "support", "assist", "accommodate",
            "prioritize", "serve",
        ])

        for group_name, keywords in _VULNERABLE_GROUPS.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if not matches:
                continue

            # Determine risk level based on co-occurring signals
            if harm_context and not benefit_context:
                risk = "high"
                rec = f"Careful review needed — action may harm {group_name}"
            elif harm_context and benefit_context:
                risk = "medium"
                rec = f"Mixed signals for {group_name} — verify net impact"
            else:
                risk = "low"
                rec = f"{group_name.replace('_', ' ').title()} mentioned — no immediate concern"

            found.append(VulnerablePopulation(
                group_name=group_name,
                detected_from=matches[0],
                risk_level=risk,
                recommendation=rec,
            ))

        return found

    # ---------- Explanation ----------

    def _build_explanation(
        self,
        score: float,
        virtue_scores: dict[Virtue, float],
        fairness_score: float,
        bias_detected: list[BiasType],
        vulnerable: list[VulnerablePopulation],
    ) -> str:
        lines = [f"Virtue ethics score V(a) = {score:.3f}"]

        if score >= 0.75:
            lines.append("Assessment: Action aligns well with ethical virtues and fairness.")
        elif score >= 0.5:
            lines.append("Assessment: Some virtue/fairness concerns warrant attention.")
        else:
            lines.append("Assessment: Significant fairness or virtue concerns identified.")

        # Top and bottom virtues
        sorted_virtues = sorted(virtue_scores.items(), key=lambda x: x[1])
        if sorted_virtues:
            weakest = sorted_virtues[0]
            strongest = sorted_virtues[-1]
            lines.append(f"Strongest virtue: {strongest[0].value} ({strongest[1]:.2f})")
            lines.append(f"Weakest virtue: {weakest[0].value} ({weakest[1]:.2f})")

        lines.append(f"Fairness score: {fairness_score:.3f}")

        if bias_detected:
            bias_names = ", ".join(b.value for b in bias_detected)
            lines.append(f"⚠ Potential bias detected: {bias_names}")

        if vulnerable:
            high_risk = [v for v in vulnerable if v.risk_level == "high"]
            if high_risk:
                groups = ", ".join(v.group_name for v in high_risk)
                lines.append(f"⚠ High-risk vulnerable groups: {groups}")

        return "\n".join(lines)

    # ---------- Standalone fairness metric helpers ----------
    # These are exposed for use by other modules (benchmarks, etc.)

    @staticmethod
    def compute_spd(rate_a: float, rate_b: float) -> float:
        """Statistical Parity Difference.  Fair when |SPD| < 0.1."""
        return rate_a - rate_b

    @staticmethod
    def compute_dir(rate_a: float, rate_b: float) -> float:
        """Disparate Impact Ratio.  Fair when 0.8 ≤ DIR ≤ 1.25."""
        if rate_a == 0:
            return 0.0
        return rate_b / rate_a

    @staticmethod
    def compute_eod(tpr_a: float, tpr_b: float) -> float:
        """Equal Opportunity Difference.  Fair when |EOD| < 0.1."""
        return tpr_a - tpr_b
