"""Decision trace — full audit trail for ethical decisions.

Every step the pipeline takes gets recorded in a trace.  This is
critical for accountability: if someone asks "why did the system
block my loan application?", we need to show exactly what happened
at each stage.

The trace is structured as a list of TraceEntry objects, each with:
  - Stage name
  - Input summary
  - Output summary
  - Duration
  - Any warnings or flags

The trace can be serialized to JSON for storage in audit logs.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TraceEntry:
    """One step in the decision trace."""

    stage: str
    timestamp: float
    duration_ms: float
    input_summary: str
    output_summary: str
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class DecisionTrace:
    """Records the full audit trail of a pipeline run.

    Usage::

        trace = DecisionTrace(task="Evaluate loan application")
        trace.start_stage("context_extraction")
        # ... do work ...
        trace.end_stage(output_summary="Domain: finance, Urgency: normal")

        # At the end:
        print(trace.to_json())
    """

    def __init__(self, task: str = "", run_id: str = "") -> None:
        self.task = task
        self.run_id = run_id or f"trace-{int(time.time())}"
        self.entries: list[TraceEntry] = []
        self._current_stage: str | None = None
        self._stage_start: float = 0.0
        self._stage_input: str = ""
        self.start_time = time.time()
        self.end_time: float | None = None
        self.metadata: dict[str, Any] = {}

    def start_stage(self, stage: str, input_summary: str = "") -> None:
        """Mark the start of a pipeline stage."""
        self._current_stage = stage
        self._stage_start = time.time()
        self._stage_input = input_summary

    def end_stage(
        self,
        output_summary: str = "",
        details: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        """Mark the end of a pipeline stage."""
        if not self._current_stage:
            logger.warning("end_stage called without start_stage")
            return

        duration_ms = (time.time() - self._stage_start) * 1000.0

        entry = TraceEntry(
            stage=self._current_stage,
            timestamp=self._stage_start,
            duration_ms=round(duration_ms, 2),
            input_summary=self._stage_input,
            output_summary=output_summary,
            details=details or {},
            warnings=warnings or [],
        )
        self.entries.append(entry)
        self._current_stage = None

    def add_entry(
        self,
        stage: str,
        duration_ms: float,
        input_summary: str = "",
        output_summary: str = "",
        details: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        """Add a pre-built trace entry (for stages tracked externally)."""
        self.entries.append(
            TraceEntry(
                stage=stage,
                timestamp=time.time(),
                duration_ms=duration_ms,
                input_summary=input_summary,
                output_summary=output_summary,
                details=details or {},
                warnings=warnings or [],
            )
        )

    def finalize(self) -> None:
        """Mark the trace as complete."""
        self.end_time = time.time()

    @property
    def total_duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000.0
        return sum(e.duration_ms for e in self.entries)

    @property
    def stage_count(self) -> int:
        return len(self.entries)

    @property
    def all_warnings(self) -> list[str]:
        warnings = []
        for e in self.entries:
            for w in e.warnings:
                warnings.append(f"[{e.stage}] {w}")
        return warnings

    def get_stage(self, stage_name: str) -> TraceEntry | None:
        """Get a specific stage's trace entry."""
        for e in self.entries:
            if e.stage == stage_name:
                return e
        return None

    def summary(self) -> str:
        """Human-readable summary of the trace."""
        lines = [
            f"Decision Trace: {self.run_id}",
            f"Task: {self.task}",
            f"Stages: {self.stage_count}",
            f"Total time: {self.total_duration_ms:.1f}ms",
            "",
        ]
        for e in self.entries:
            status = "⚠" if e.warnings else "✓"
            lines.append(
                f"  {status} {e.stage:30s} {e.duration_ms:8.1f}ms  {e.output_summary[:50]}"
            )

        if self.all_warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.all_warnings:
                lines.append(f"  ⚠ {w}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_ms": self.total_duration_ms,
            "stage_count": self.stage_count,
            "entries": [asdict(e) for e in self.entries],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Backward-compat aliases used by tests
# ---------------------------------------------------------------------------


@dataclass
class TraceStep:
    """One step in a managed trace (backward-compat alias for TraceEntry)."""

    stage: str
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class _ManagedTrace:
    """Internal wrapper holding a trace + metadata for DecisionTracer."""

    def __init__(self, task: str, trace_id: str) -> None:
        self.task = task
        self.trace_id = trace_id
        self.steps: list[TraceStep] = []
        self.verdict: str | None = None
        self.eds_score: float | None = None
        self.finalized: bool = False
        self.start_time: float = time.time()
        self.end_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task": self.task,
            "steps": [asdict(s) for s in self.steps],
            "verdict": self.verdict,
            "eds_score": self.eds_score,
            "finalized": self.finalized,
        }


class DecisionTracer:
    """Manager for multiple decision traces (backward-compat).

    Higher-level than :class:`DecisionTrace`: creates, stores, and
    queries many traces simultaneously.
    """

    def __init__(self) -> None:
        self._traces: dict[str, _ManagedTrace] = {}
        self._counter: int = 0

    def start_trace(self, task: str) -> str:
        """Create a new trace and return its ID."""
        self._counter += 1
        trace_id = f"trace-{self._counter}"
        self._traces[trace_id] = _ManagedTrace(task=task, trace_id=trace_id)
        return trace_id

    def record_step(
        self,
        trace_id: str,
        stage: str,
        data: dict[str, Any] | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Append a step to an existing trace."""
        trace = self._traces.get(trace_id)
        if trace is None:
            logger.warning("record_step: unknown trace_id %s", trace_id)
            return
        trace.steps.append(
            TraceStep(
                stage=stage,
                data=data or {},
                duration_ms=duration_ms,
            )
        )

    def finalize_trace(
        self,
        trace_id: str,
        verdict: str = "",
        eds_score: float = 0.0,
    ) -> None:
        """Mark a trace as finalized with a verdict and score."""
        trace = self._traces.get(trace_id)
        if trace is None:
            return
        trace.verdict = verdict
        trace.eds_score = eds_score
        trace.finalized = True
        trace.end_time = time.time()

    def get_trace(self, trace_id: str) -> _ManagedTrace | None:
        return self._traces.get(trace_id)

    def export_trace(self, trace_id: str) -> dict[str, Any]:
        trace = self._traces.get(trace_id)
        if trace is None:
            return {}
        return trace.to_dict()

    def search_by_verdict(self, verdict: str) -> list[_ManagedTrace]:
        return [t for t in self._traces.values() if t.verdict == verdict]

    def get_statistics(self) -> dict[str, Any]:
        verdicts: dict[str, int] = {}
        scores: list[float] = []
        for t in self._traces.values():
            if t.verdict:
                verdicts[t.verdict] = verdicts.get(t.verdict, 0) + 1
            if t.eds_score is not None:
                scores.append(t.eds_score)
        return {
            "total_traces": len(self._traces),
            "verdict_distribution": verdicts,
            "avg_eds_score": sum(scores) / len(scores) if scores else 0.0,
        }
