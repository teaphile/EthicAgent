# ethicagent.core — shared infrastructure (state, logging, types)
# Kept deliberately small; heavy logic lives in agents/ and ethics/.

from ethicagent.core.state import PipelineState, PipelineStage, StageResult, StateManager
from ethicagent.core.logger import AuditLogger, AuditEntry

__all__ = [
    "PipelineState",
    "PipelineStage",
    "StageResult",
    "StateManager",
    "AuditLogger",
    "AuditEntry",
]
