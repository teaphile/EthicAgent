"""Pipeline state management — thread-safe shared state container.

Every pipeline run gets a ``PipelineState`` that flows through all 8
stages, accumulating context, reasoning outputs, and audit data.  The
``StateManager`` keeps a registry of active / completed states so that
concurrent evaluations don't stomp on each other.

Design choice: we use dataclasses + a simple threading lock rather than
something heavier like SQLAlchemy — the pipeline is latency-sensitive
and we rarely need persistence beyond the current process.

# NOTE: if we ever move to async (FastAPI, etc.) swap threading.Lock
#       for asyncio.Lock.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class PipelineStage(Enum):
    """Named stages that a pipeline run moves through."""

    INITIALIZED = "initialized"
    CONTEXT_EXTRACTION = "context_extraction"
    CONTEXT_ANALYSIS = "context_analysis"  # backward-compat alias
    KNOWLEDGE_QUERY = "knowledge_query"  # also referenced as "knowledge_retrieval" in logs
    NEURAL_REASONING = "neural_reasoning"
    SYMBOLIC_REASONING = "symbolic_reasoning"
    FUSION = "fusion"
    ETHICAL_EVALUATION = "ethical_evaluation"
    DECISION_GATE = "decision_gate"
    ACTION_EXECUTION = "action_execution"
    HUMAN_REVIEW = "human_review"
    REFLECTION = "reflection"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class StageResult:
    """Snapshot of one completed (or failed) pipeline stage.

    We store both a freeform *result* dict and structured timing /
    success flags so downstream consumers can iterate over the list
    without parsing heterogeneous dicts.

    Accepts ``data=`` as an alias for ``result=``.
    """

    stage: PipelineStage
    result: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Sync data ↔ result so either kwarg works
        if self.data and not self.result:
            self.result = self.data
        elif self.result and not self.data:
            self.data = self.result
        elif not self.result and not self.data:
            self.result = {}
            self.data = self.result


@dataclass
class PipelineState:
    """Shared mutable state for a single pipeline execution.

    All stage methods write into this object; the orchestrator reads
    it back at the end to build the final ``PipelineResult``.

    Attributes
    ----------
    run_id : str
        UUID generated at creation time.
    task / domain : str
        Original inputs.
    context … ethical_decision : dict / Any
        Intermediate outputs — populated lazily as stages complete.
    stage_results : dict[PipelineStage, StageResult]
        Dict keyed by stage for O(1) lookup; used for audit & explainability.
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task: str = ""
    domain: str | None = None
    current_stage: PipelineStage = PipelineStage.INITIALIZED

    # Stage outputs — populated progressively
    context: dict[str, Any] = field(default_factory=dict)
    knowledge: dict[str, Any] = field(default_factory=dict)
    neural_result: dict[str, Any] = field(default_factory=dict)
    symbolic_result: dict[str, Any] = field(default_factory=dict)
    fusion_result: dict[str, Any] = field(default_factory=dict)
    ethical_decision: Any | None = None

    # Trace & audit — dict keyed by PipelineStage for O(1) lookup
    stage_results: dict[PipelineStage, StageResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def update_stage(
        self,
        stage: PipelineStage,
        result: dict[str, Any],
        duration_ms: float = 0.0,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Record that *stage* just finished (or failed).

        The orchestrator calls this after every stage completes.
        Results are appended — never overwritten — so the full trace
        is always available.
        """
        self.current_stage = stage
        self.updated_at = datetime.now(timezone.utc).isoformat()
        sr = StageResult(
            stage=stage,
            result=result,
            duration_ms=duration_ms,
            success=success,
            error=error,
        )
        self.stage_results[stage] = sr

    def get_decision_trace(self) -> list[dict[str, Any]]:
        """Flatten stage_results into plain dicts (JSON-friendly)."""
        return [
            {
                "stage": sr.stage.value,
                "result": sr.result,
                "timestamp": sr.timestamp,
                "duration_ms": sr.duration_ms,
                "success": sr.success,
                "error": sr.error,
            }
            for sr in self.stage_results.values()
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire state to a JSON-safe dict."""
        return {
            "run_id": self.run_id,
            "task": self.task,
            "domain": self.domain,
            "current_stage": self.current_stage.value,
            "context": self.context,
            "knowledge": self.knowledge,
            "neural_result": self.neural_result,
            "symbolic_result": self.symbolic_result,
            "fusion_result": self.fusion_result,
            "stage_results": self.get_decision_trace(),
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class StateManager:
    """Thread-safe registry of pipeline states.

    Keeps active runs separate from completed ones so callers can
    query either set without locking contention.
    """

    def __init__(self, *, max_history: int = 5000) -> None:
        self._states: dict[str, PipelineState] = {}
        self._lock = threading.Lock()
        self._history: list[PipelineState] = []
        self._max_history = max_history

    def create_state(self, task: str = "", domain: str | None = None) -> PipelineState:
        """Spin up a fresh PipelineState and register it."""
        state = PipelineState(task=task, domain=domain)
        with self._lock:
            self._states[state.run_id] = state
        return state

    def get_state(self, run_id: str) -> PipelineState | None:
        with self._lock:
            return self._states.get(run_id)

    def complete_state(self, run_id: str) -> None:
        """Move state from active → history."""
        with self._lock:
            st = self._states.pop(run_id, None)
            if st:
                self._history.append(st)
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history :]

    # ------------------------------------------------------------------
    # Backward-compat helpers used by tests
    # ------------------------------------------------------------------

    def update_stage(self, state: PipelineState, result: StageResult) -> None:
        """Convenience wrapper: record *result* on *state*."""
        state.stage_results[result.stage] = result
        state.current_stage = result.stage
        state.updated_at = datetime.now(timezone.utc).isoformat()

    def get_stage_result(self, state: PipelineState, stage: PipelineStage) -> StageResult | None:
        """Look up a completed stage result (or *None*)."""
        return state.stage_results.get(stage)

    def get_history(self) -> list[PipelineState]:
        with self._lock:
            return list(self._history)

    def get_active_runs(self) -> list[str]:
        with self._lock:
            return list(self._states.keys())
