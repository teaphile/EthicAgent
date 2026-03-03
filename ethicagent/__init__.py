"""EthicAgent — Context-Aware Neuro-Symbolic Framework for Ethical
Autonomous Decision-Making in Agentic AI Systems.

Started this framework during our lab's ethics-in-AI hackathon back in 2024.
What began as a quick prototype for scoring ethical risk in healthcare triage
grew into a multi-domain, multi-philosophy reasoning pipeline.  The core
insight — fuse an LLM's contextual understanding with hard symbolic rules,
then overlay four philosophical lenses — still drives the architecture today.

Author : EthicAgent Research Team
Date   : 2024-06 (initial), 2025-03 (v1.0 release)
License: MIT
"""

# NOTE: we keep __doc__ short for tools that inspect it programmatically.
__doc__ = "EthicAgent: neuro-symbolic ethical reasoning for autonomous agents."

__version__ = "1.0.0"
__author__ = "EthicAgent Research Team"

# -- Public API ---------------------------------------------------------------
# Import path is ethicagent.orchestrator (top-level), NOT ethicagent.core.*
from ethicagent.core.logger import AuditLogger
from ethicagent.core.state import PipelineStage, PipelineState, StageResult
from ethicagent.ethics.ethical_score import EthicalDecision, EthicalVerdict
from ethicagent.orchestrator import EthicAgentOrchestrator

__all__ = [
    "EthicAgentOrchestrator",
    "EthicalDecision",
    "EthicalVerdict",
    "PipelineState",
    "PipelineStage",
    "StageResult",
    "AuditLogger",
    "__version__",
]
