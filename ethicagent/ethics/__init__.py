"""Ethics evaluation modules.

Each module corresponds to one of the four philosophy lenses in the
EDS formula, plus the conflict resolver and the canonical score/verdict
definitions.
"""

from ethicagent.ethics.deontological import (
    DeontologicalEvaluator,
    DeontologicalResult,
    RuleViolation,
    RuleSeverity,
)
from ethicagent.ethics.consequentialist import (
    ConsequentialistEvaluator,
    ConsequentialistResult,
    StakeholderImpact,
    ImpactType,
)
from ethicagent.ethics.virtue_ethics import (
    VirtueEthicsEvaluator,
    VirtueEthicsResult,
    FairnessMetrics,
)
from ethicagent.ethics.contextual_ethics import (
    ContextualEthicsEvaluator,
    ContextualResult,
)
from ethicagent.ethics.conflict_resolver import (
    ConflictResolver,
    ConflictRecord,
    ConflictSeverity,
    PhilosophyPosition,
)
from ethicagent.ethics.ethical_score import (
    EthicalDecision,
    EthicalVerdict,
    PhilosophyResult,
    DOMAIN_WEIGHTS,
    compute_eds,
    determine_verdict,
    compute_confidence_interval,
    sensitivity_analysis,
)

__all__ = [
    "DeontologicalEvaluator", "DeontologicalResult", "RuleViolation", "RuleSeverity",
    "ConsequentialistEvaluator", "ConsequentialistResult", "StakeholderImpact", "ImpactType",
    "VirtueEthicsEvaluator", "VirtueEthicsResult", "FairnessMetrics",
    "ContextualEthicsEvaluator", "ContextualResult",
    "ConflictResolver", "ConflictRecord", "ConflictSeverity", "PhilosophyPosition",
    "EthicalDecision", "EthicalVerdict", "PhilosophyResult", "DOMAIN_WEIGHTS",
    "compute_eds", "determine_verdict", "compute_confidence_interval", "sensitivity_analysis",
]
