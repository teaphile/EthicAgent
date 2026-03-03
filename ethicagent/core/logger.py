"""Audit trail logger — every ethical decision the system makes gets
recorded here for governance, compliance, and reproducibility.

We went through a few iterations on this:
 - v0.1 just dumped JSON to a file
 - v0.2 added thread-safety after concurrent pipeline runs garbled the log
 - v0.3 (current) structured AuditEntry dataclass + CSV/JSON export

The audit log is append-only by design — nobody should be able to
retroactively edit a decision record.  If we ever move to a database
backend, this is the place to add it.

# TODO: add log rotation for long-running processes
# FIXME: export_to_csv doesn't handle Unicode names well on Windows
"""

from __future__ import annotations

import csv
import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class AuditEntry:
    """One record in the audit trail.

    Could be a decision, a conflict, a stage transition, or an error.
    Stores scores and verdict separately from *details* so they're
    easy to aggregate without parsing JSON blobs.
    """

    entry_id: str = ""
    run_id: str = ""
    timestamp: str = ""
    event_type: str = ""  # "decision" | "conflict" | "error" | "stage" …
    stage: str = ""
    action: str = ""
    domain: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    ethical_scores: dict[str, float] | None = None
    verdict: str | None = None
    user: str = "system"
    # Backward-compat fields
    message: str = ""
    data: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.entry_id:
            self.entry_id = f"AE-{id(self):06d}"
        # Sync message/action and data/details
        if self.message and not self.action:
            self.action = self.message
        if self.data is not None and not self.details:
            self.details = self.data

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditLogger:
    """Thread-safe audit logger for the EthicAgent pipeline.

    All decision events, conflicts, and errors flow through here.
    Entries are stored in memory and can be exported to JSON or CSV
    for offline analysis.

    Usage::

        al = AuditLogger(log_dir="logs")
        al.log_decision(run_id="abc", action="deny loan", domain="finance",
                        ethical_scores={"deontological": 0.3}, verdict="reject",
                        explanation="...", decision_trace=[])
        al.export_to_json()
    """

    def __init__(
        self,
        log_dir: str = "logs",
        log_level: int = logging.INFO,
    ) -> None:
        self._entries: list[AuditEntry] = []
        self._lock = threading.Lock()
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0

        # -- Python logger setup --------------------------------------------------
        self._logger = logging.getLogger("ethicagent.audit")
        self._logger.setLevel(log_level)

        if not self._logger.handlers:
            fmt = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            # console
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
            ch.setFormatter(fmt)
            self._logger.addHandler(ch)
            # file — append, so nothing is lost between restarts
            fh = logging.FileHandler(str(self._log_dir / "ethicagent_audit.log"))
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            self._logger.addHandler(fh)

    # -- id generation --------------------------------------------------------

    def _next_id(self) -> str:
        self._counter += 1
        return f"AE-{self._counter:06d}"

    @property
    def entries(self) -> list[AuditEntry]:
        """Public access to entries list (backward-compat)."""
        return self._entries

    # -- core logging methods -------------------------------------------------

    def log_event(
        self,
        run_id_or_event_type: str = "",
        event_type_or_message: str = "",
        stage_or_data: Any = None,
        action: str = "",
        domain: str = "",
        details: dict[str, Any] | None = None,
        ethical_scores: dict[str, float] | None = None,
        verdict: str | None = None,
        user: str = "system",
        *,
        # Named kwargs for canonical form
        run_id: str = "",
        event_type: str = "",
        stage: str = "",
    ) -> AuditEntry:
        """Record a generic audit event.

        Accepts either:
          log_event(run_id, event_type, stage, action, domain, details)  — canonical
          log_event(event_type, message, data)                            — simplified
          log_event(event_type, message)                                  — minimal
        """
        # Detect simplified form: if stage_or_data is a dict or missing
        if isinstance(stage_or_data, dict) or (
            stage_or_data is None and not action and not details
        ):
            # Simplified form: log_event(event_type, message, optional_data)
            _event_type = event_type or run_id_or_event_type
            _message = event_type_or_message
            _data = stage_or_data if isinstance(stage_or_data, dict) else {}
            _run_id = run_id or "default"
            _stage = stage or "general"
            _action = _message
            _domain = domain or "general"
            _details = _data
        else:
            # Canonical form
            _run_id = run_id or run_id_or_event_type
            _event_type = event_type or event_type_or_message
            _stage = stage or (stage_or_data if isinstance(stage_or_data, str) else "")
            _action = action
            _domain = domain
            _details = details or {}

        entry = AuditEntry(
            entry_id=self._next_id(),
            run_id=_run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=_event_type,
            stage=_stage,
            action=_action,
            domain=_domain,
            details=_details,
            ethical_scores=ethical_scores,
            verdict=verdict,
            user=user,
        )

        with self._lock:
            self._entries.append(entry)

        # route to the right log level
        msg = f"[{_event_type}] run={_run_id} stage={_stage} domain={_domain} verdict={verdict}"
        if _event_type == "error":
            self._logger.error(msg)
        elif _event_type in ("conflict", "human_review"):
            self._logger.warning(msg)
        else:
            self._logger.info(msg)

        return entry

    def log_decision(
        self,
        run_id: str = "",
        action: str = "",
        domain: str = "",
        ethical_scores: dict[str, float] | None = None,
        verdict: str = "",
        explanation: str = "",
        decision_trace: list[dict[str, Any]] | None = None,
        *,
        # the ActionExecutor sometimes passes these keyword args —
        # accept them gracefully even if we don't store them all
        eds_score: float | None = None,
        rules_triggered: list[str] | None = None,
        reasoning: str | None = None,
        metadata: dict[str, Any] | None = None,
        # backward-compat kwargs from test
        task: str = "",
        data: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Log a final ethical decision.

        Flexible signature — accepts both the 'canonical' form
        (explanation + decision_trace) and the shorthand that the
        ActionExecutor prefers (eds_score + rules_triggered + reasoning).
        Also accepts test-style kwargs: task, verdict, eds_score, data.
        """
        _action = action or task
        _domain = domain or (data.get("domain", "general") if data else "general")
        _verdict = verdict
        _scores = ethical_scores or {}

        details: dict[str, Any] = {
            "explanation": explanation or reasoning or "",
            "decision_trace": decision_trace or [],
        }
        if eds_score is not None:
            details["eds_score"] = eds_score
        if rules_triggered:
            details["rules_triggered"] = rules_triggered
        if metadata:
            details["metadata"] = metadata
        if data:
            details.update(data)

        return self.log_event(
            run_id=run_id or "default",
            event_type="decision",
            stage="decision_gate",
            action=_action,
            domain=_domain,
            details=details,
            ethical_scores=_scores or None,
            verdict=_verdict,
        )

    def log_conflict(
        self,
        run_id: str = "",
        action: str = "",
        domain: str = "",
        scores: dict[str, float] | None = None,
        conflict_description: str = "",
        resolution: str = "",
        *,
        conflict_type: str = "",
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Record an ethical conflict between philosophies."""
        _details = {
            "conflict_description": conflict_description or conflict_type,
            "resolution": resolution,
        }
        if details:
            _details.update(details)

        return self.log_event(
            run_id=run_id or "default",
            event_type="conflict",
            stage="ethical_evaluation",
            action=action,
            domain=domain,
            details=_details,
            ethical_scores=scores,
        )

    def log_error(
        self,
        run_id_or_message: str = "",
        stage_or_error: Any = None,
        error_message: str = "",
        action: str = "",
        domain: str = "",
        *,
        run_id: str = "",
        error: Any = None,
    ) -> AuditEntry:
        """Record an error event.

        Accepts either:
          log_error(run_id, stage, error_message)  — canonical
          log_error(message, error=exc)             — simplified
        """
        if error is not None or (isinstance(stage_or_error, Exception)):
            # Simplified form
            _msg = run_id_or_message
            _err = error or stage_or_error
            _err_str = f"{_msg}: {_err}" if _err else _msg
            return self.log_event(
                run_id=run_id or "default",
                event_type="error",
                stage="general",
                action=_msg,
                domain=domain or "general",
                details={"error": str(_err_str)},
            )
        # Canonical form
        return self.log_event(
            run_id=run_id or run_id_or_message,
            event_type="error",
            stage=stage_or_error if isinstance(stage_or_error, str) else "",
            action=action,
            domain=domain,
            details={"error": error_message},
        )

    # -- querying -------------------------------------------------------------

    def get_entries(
        self,
        run_id: str | None = None,
        event_type: str | None = None,
        domain: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Retrieve entries with optional filters."""
        with self._lock:
            entries = list(self._entries)

        if run_id:
            entries = [e for e in entries if e.run_id == run_id]
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        if domain:
            entries = [e for e in entries if e.domain == domain]

        return entries[-limit:]

    # -- export ---------------------------------------------------------------

    def export_to_json(self, filepath: str | None = None) -> str:
        if filepath is None:
            filepath = str(self._log_dir / "audit_export.json")

        with self._lock:
            data = [asdict(e) for e in self._entries]

        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)

        self._logger.info(f"Exported {len(data)} audit entries → {filepath}")
        return filepath

    def export_to_csv(self, filepath: str | None = None) -> str:
        if filepath is None:
            filepath = str(self._log_dir / "audit_export.csv")

        with self._lock:
            entries = list(self._entries)

        cols = [
            "entry_id",
            "run_id",
            "timestamp",
            "event_type",
            "stage",
            "action",
            "domain",
            "verdict",
            "user",
            "details",
        ]
        with open(filepath, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for e in entries:
                writer.writerow(
                    {
                        "entry_id": e.entry_id,
                        "run_id": e.run_id,
                        "timestamp": e.timestamp,
                        "event_type": e.event_type,
                        "stage": e.stage,
                        "action": e.action,
                        "domain": e.domain,
                        "verdict": e.verdict or "",
                        "user": e.user,
                        "details": json.dumps(e.details),
                    }
                )

        self._logger.info(f"Exported {len(entries)} entries → CSV: {filepath}")
        return filepath

    def clear(self) -> None:
        """Wipe all stored entries (useful in test teardown)."""
        with self._lock:
            self._entries.clear()
        self._logger.info("Audit log cleared")
