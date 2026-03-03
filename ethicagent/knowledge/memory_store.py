"""Memory store for past decisions and learning.

This is the system's "institutional memory" — it remembers past
decisions so the ReflectionAgent can detect drift and the pipeline
can learn from experience.

Storage backends:
  - In-memory (default, for testing/research)
  - JSON file (persistent, for demos)
  - ChromaDB (vector search, for production — optional)

The store supports:
  1. Storing decisions with full context
  2. Querying by domain, verdict, EDS range
  3. Similarity search (if ChromaDB is available)
  4. Statistical summaries
  5. Windowed queries (last N decisions)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """A single stored decision."""
    decision_id: str
    timestamp: float
    task: str
    domain: str
    eds_score: float
    verdict: str       # "AUTO_APPROVE", "ESCALATE", "REJECT", "HARD_BLOCK"
    philosophy_scores: dict[str, float] = field(default_factory=dict)
    context_summary: str = ""
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    """Persistent memory of past ethical decisions.

    This is deliberately simple — a list of records with index dicts
    for fast lookups.  For a real system you'd want a database, but
    this is fine for research with < 10k decisions.
    """

    def __init__(
        self,
        persist_path: str | Path | None = None,
        max_records: int = 10000,
    ) -> None:
        self._records: list[DecisionRecord] = []
        self._by_domain: dict[str, list[int]] = {}  # domain → list of indices
        self._by_verdict: dict[str, list[int]] = {}
        self._persist_path = Path(persist_path) if persist_path else None
        self._max_records = max_records
        self._counter = 0

        # Load from disk if available
        if self._persist_path and self._persist_path.exists():
            self._load()

        logger.debug("MemoryStore initialized with %d records", len(self._records))

    # ---------- Store/retrieve ----------

    def store(self, record: DecisionRecord) -> str:
        """Store a decision record and return its ID."""
        if not record.decision_id:
            self._counter += 1
            record.decision_id = f"DEC-{self._counter:06d}"

        if not record.timestamp:
            record.timestamp = time.time()

        idx = len(self._records)
        self._records.append(record)

        # Update indices
        self._by_domain.setdefault(record.domain, []).append(idx)
        self._by_verdict.setdefault(record.verdict, []).append(idx)

        # Enforce max records (drop oldest)
        if len(self._records) > self._max_records:
            self._compact()

        # Auto-persist
        if self._persist_path:
            self._save()

        return record.decision_id

    def store_from_dict(self, data: dict[str, Any]) -> str:
        """Store a decision from a dict (convenience method)."""
        self._counter += 1
        record = DecisionRecord(
            decision_id=data.get("decision_id", f"DEC-{self._counter:06d}"),
            timestamp=data.get("timestamp", time.time()),
            task=data.get("task", ""),
            domain=data.get("domain", "general"),
            eds_score=data.get("eds_score", 0.0),
            verdict=data.get("verdict", "ESCALATE"),
            philosophy_scores=data.get("philosophy_scores", {}),
            context_summary=data.get("context_summary", ""),
            reasoning=data.get("reasoning", ""),
            metadata=data.get("metadata", {}),
        )
        return self.store(record)

    def get(self, decision_id: str) -> DecisionRecord | None:
        """Retrieve a specific decision by ID."""
        for r in self._records:
            if r.decision_id == decision_id:
                return r
        return None

    def get_recent(self, n: int = 10) -> list[DecisionRecord]:
        """Get the N most recent decisions."""
        return list(reversed(self._records[-n:]))

    def query(
        self,
        domain: str | None = None,
        verdict: str | None = None,
        eds_min: float | None = None,
        eds_max: float | None = None,
        limit: int = 50,
    ) -> list[DecisionRecord]:
        """Query decisions with optional filters."""
        results = self._records

        if domain:
            indices = set(self._by_domain.get(domain, []))
            results = [r for i, r in enumerate(results) if i in indices]

        if verdict:
            indices = set(self._by_verdict.get(verdict, []))
            results = [r for i, r in enumerate(self._records) if i in indices]
            if domain:
                results = [r for r in results if r.domain == domain]

        if eds_min is not None:
            results = [r for r in results if r.eds_score >= eds_min]
        if eds_max is not None:
            results = [r for r in results if r.eds_score <= eds_max]

        return results[-limit:]

    # ---------- Statistics ----------

    @property
    def count(self) -> int:
        return len(self._records)

    def get_stats(self, domain: str | None = None) -> dict[str, Any]:
        """Get summary statistics for decisions."""
        records = self._records
        if domain:
            records = [r for r in records if r.domain == domain]

        if not records:
            return {"count": 0}

        scores = [r.eds_score for r in records]
        verdicts = [r.verdict for r in records]

        return {
            "count": len(records),
            "avg_eds": round(sum(scores) / len(scores), 4),
            "min_eds": round(min(scores), 4),
            "max_eds": round(max(scores), 4),
            "verdict_distribution": {
                v: verdicts.count(v) for v in set(verdicts)
            },
            "domains": list(set(r.domain for r in records)),
        }

    def get_verdict_trend(self, window: int = 20) -> list[dict[str, Any]]:
        """Get verdict trend over last N decisions."""
        recent = self._records[-window:]
        return [
            {
                "decision_id": r.decision_id,
                "eds_score": r.eds_score,
                "verdict": r.verdict,
                "domain": r.domain,
            }
            for r in recent
        ]

    # ---------- Similarity search ----------

    def find_similar(
        self,
        task: str,
        domain: str | None = None,
        top_k: int = 5,
    ) -> list[DecisionRecord]:
        """Find similar past decisions (basic keyword overlap).

        For real similarity search, use ChromaDB integration.
        This is a fallback that uses simple word overlap.
        """
        task_words = set(task.lower().split())

        candidates = self._records
        if domain:
            candidates = [r for r in candidates if r.domain == domain]

        scored = []
        for r in candidates:
            record_words = set(r.task.lower().split())
            overlap = len(task_words & record_words)
            if overlap > 0:
                similarity = overlap / max(len(task_words | record_words), 1)
                scored.append((similarity, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    # ---------- Persistence ----------

    def _save(self) -> None:
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = [asdict(r) for r in self._records]
            with open(self._persist_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save memory store: %s", e)

    def _load(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            with open(self._persist_path) as f:
                data = json.load(f)
            for i, d in enumerate(data):
                record = DecisionRecord(**d)
                self._records.append(record)
                self._by_domain.setdefault(record.domain, []).append(i)
                self._by_verdict.setdefault(record.verdict, []).append(i)
            self._counter = len(self._records)
            logger.info("Loaded %d records from %s", len(data), self._persist_path)
        except Exception as e:
            logger.warning("Failed to load memory store: %s", e)

    def _compact(self) -> None:
        """Remove oldest records when we exceed max_records."""
        if len(self._records) <= self._max_records:
            return
        excess = len(self._records) - self._max_records
        self._records = self._records[excess:]
        # Rebuild indices
        self._by_domain.clear()
        self._by_verdict.clear()
        for i, r in enumerate(self._records):
            self._by_domain.setdefault(r.domain, []).append(i)
            self._by_verdict.setdefault(r.verdict, []).append(i)

    def clear(self) -> None:
        """Clear all records (for testing)."""
        self._records.clear()
        self._by_domain.clear()
        self._by_verdict.clear()
        self._counter = 0
