"""Ethics evaluation modules.

Each module corresponds to one of the four philosophy lenses in the
EDS formula, plus the conflict resolver and the canonical score/verdict
definitions.
"""

from ethicagent.ethics.conflict_resolver import (
    ConflictRecord,
    ConflictResolver,
    ConflictSeverity,
    PhilosophyPosition,
)
from ethicagent.ethics.consequentialist import (
    ConsequentialistEvaluator,
    ConsequentialistResult,
    ImpactType,
    StakeholderImpact,
)
from ethicagent.ethics.contextual_ethics import (
    ContextualEthicsEvaluator,
    ContextualResult,
)
from ethicagent.ethics.deontological import (
    DeontologicalEvaluator,
    DeontologicalResult,
    RuleSeverity,
    RuleViolation,
)
from ethicagent.ethics.ethical_score import (
    DOMAIN_WEIGHTS,
    EthicalDecision,
    EthicalVerdict,
    PhilosophyResult,
    compute_confidence_interval,
    compute_eds,
    determine_verdict,
    sensitivity_analysis,
)
from ethicagent.ethics.virtue_ethics import (
    FairnessMetrics,
    VirtueEthicsEvaluator,
    VirtueEthicsResult,
)

__all__ = [
    "DeontologicalEvaluator",
    "DeontologicalResult",
    "RuleViolation",
    "RuleSeverity",
    "ConsequentialistEvaluator",
    "ConsequentialistResult",
    "StakeholderImpact",
    "ImpactType",
    "VirtueEthicsEvaluator",
    "VirtueEthicsResult",
    "FairnessMetrics",
    "ContextualEthicsEvaluator",
    "ContextualResult",
    "ConflictResolver",
    "ConflictRecord",
    "ConflictSeverity",
    "PhilosophyPosition",
    "EthicalDecision",
    "EthicalVerdict",
    "PhilosophyResult",
    "DOMAIN_WEIGHTS",
    "compute_eds",
    "determine_verdict",
    "compute_confidence_interval",
    "sensitivity_analysis",
]
