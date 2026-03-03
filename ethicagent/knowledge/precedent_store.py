"""Precedent store — case-law style reasoning for AI ethics.

Like legal precedent (stare decisis), this module stores notable past
ethical decisions and matches incoming cases against them.  The idea is
that consistency matters: similar cases should get similar treatment.

Data sources:
  1. Built-in seed precedents (classic dilemmas + domain-specific)
  2. JSON files in data/precedents/
  3. Runtime additions from pipeline outcomes

Each precedent has:
  - A case description
  - Domain and keywords
  - The decision that was made
  - Whether it was later deemed appropriate
  - Lessons learned
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Precedent:
    """A stored ethical precedent (case)."""

    id: str
    domain: str
    description: str
    keywords: list[str]
    decision: str
    outcome: str
    was_appropriate: bool
    lessons: str
    eds_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Seed precedents — classic and domain-specific examples
# ---------------------------------------------------------------------------

_SEED_PRECEDENTS: list[dict[str, Any]] = [
    # Healthcare
    {
        "id": "PREC-HC-001",
        "domain": "healthcare",
        "description": "Emergency triage with limited ventilators during pandemic",
        "keywords": ["triage", "ventilator", "pandemic", "resource", "scarce"],
        "decision": "Allocate based on medical urgency and likelihood of survival",
        "outcome": "Prioritized medical criteria over age/socioeconomic factors",
        "was_appropriate": True,
        "lessons": "Medical triage should use clinical evidence, not social value judgments",
    },
    {
        "id": "PREC-HC-002",
        "domain": "healthcare",
        "description": "AI diagnostic system flagging potential cancer in minority patient",
        "keywords": ["diagnostic", "cancer", "minority", "bias", "screening"],
        "decision": "Flag for human review with explicit note about training data limitations",
        "outcome": "Correctly escalated to radiologist who confirmed the finding",
        "was_appropriate": True,
        "lessons": "AI diagnostic tools should always include confidence levels and demographic performance notes",
    },
    {
        "id": "PREC-HC-003",
        "domain": "healthcare",
        "description": "Denying treatment based on patient's insurance status in emergency",
        "keywords": ["deny", "treatment", "insurance", "emergency", "uninsured"],
        "decision": "BLOCKED — emergency care cannot be denied regardless of insurance",
        "outcome": "Patient received treatment as required by EMTALA",
        "was_appropriate": True,
        "lessons": "Emergency care is a legal and ethical obligation regardless of payment ability",
    },
    # Finance
    {
        "id": "PREC-FI-001",
        "domain": "finance",
        "description": "Loan application using zip code as proxy for race",
        "keywords": ["loan", "zip code", "proxy", "discrimination", "redlining"],
        "decision": "REJECTED — zip code is a known proxy for racial discrimination in lending",
        "outcome": "Removed zip code from model features",
        "was_appropriate": True,
        "lessons": "Proxy variables for protected attributes must be identified and removed",
    },
    {
        "id": "PREC-FI-002",
        "domain": "finance",
        "description": "Credit scoring model with disparate impact on minority applicants",
        "keywords": ["credit", "disparate impact", "minority", "fair lending"],
        "decision": "Escalated for bias audit and model recalibration",
        "outcome": "Model was retrained with fairness constraints",
        "was_appropriate": True,
        "lessons": "Regular disparate impact testing is essential for credit models",
    },
    # Hiring
    {
        "id": "PREC-HR-001",
        "domain": "hiring",
        "description": "Resume screening tool penalizing women's college names",
        "keywords": ["resume", "screening", "gender", "bias", "college"],
        "decision": "REJECTED — model exhibited clear gender bias",
        "outcome": "Tool was decommissioned and replaced with blind screening",
        "was_appropriate": True,
        "lessons": "Resume screening AI must be tested for demographic bias before deployment",
    },
    {
        "id": "PREC-HR-002",
        "domain": "hiring",
        "description": "AI interview system with lower scores for non-native speakers",
        "keywords": ["interview", "accent", "language", "non-native", "bias"],
        "decision": "Escalated — potential disability/national origin discrimination",
        "outcome": "System was modified to evaluate content rather than delivery style",
        "was_appropriate": True,
        "lessons": "Communication style should not be penalized when job doesn't require it",
    },
    # Disaster
    {
        "id": "PREC-DR-001",
        "domain": "disaster",
        "description": "Resource allocation during natural disaster favoring wealthy neighborhoods",
        "keywords": ["disaster", "resource", "allocation", "neighborhood", "equity"],
        "decision": "REJECTED — aid must be distributed based on need, not property values",
        "outcome": "Allocation algorithm was corrected to use vulnerability indices",
        "was_appropriate": True,
        "lessons": "Disaster response must prioritize the most vulnerable, not the most valuable",
    },
]


class PrecedentStore:
    """Store and query ethical precedents.

    Think of this as a mini case-law database.  When the pipeline encounters
    a new case, it can look up similar past decisions to inform the current
    reasoning and promote consistency.
    """

    def __init__(
        self, data_dir: str | Path | None = None, filepath: str | Path | None = None
    ) -> None:
        self._precedents: list[Precedent] = []
        self._by_domain: dict[str, list[int]] = {}
        self._filepath = filepath

        # Load seeds
        for p in _SEED_PRECEDENTS:
            self.add(Precedent(**p))

        # Load from a single file (backward-compat)
        if filepath and Path(filepath).exists():
            self._load_single_file(Path(filepath))

        # Load from data directory
        if data_dir:
            self._load_from_directory(Path(data_dir))

        logger.debug("PrecedentStore initialized: %d precedents", len(self._precedents))

    def add(self, precedent: Precedent) -> None:
        """Add a precedent to the store."""
        idx = len(self._precedents)
        self._precedents.append(precedent)
        self._by_domain.setdefault(precedent.domain, []).append(idx)

    def add_from_dict(self, data: dict[str, Any]) -> None:
        """Add a precedent from a dict."""
        self.add(
            Precedent(
                id=data.get("id", f"PREC-CUSTOM-{len(self._precedents) + 1:03d}"),
                domain=data.get("domain", "general"),
                description=data.get("description", ""),
                keywords=data.get("keywords", []),
                decision=data.get("decision", ""),
                outcome=data.get("outcome", ""),
                was_appropriate=data.get("was_appropriate", True),
                lessons=data.get("lessons", ""),
                eds_score=data.get("eds_score", 0.0),
                metadata=data.get("metadata", {}),
            )
        )

    def query(
        self,
        text: str,
        domain: str | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Find precedents similar to the given text.

        Uses keyword overlap for matching.  A production system would
        use embeddings/vector search.
        """
        text_words = set(text.lower().split())

        candidates = self._precedents
        if domain:
            indices = self._by_domain.get(domain, [])
            candidates = [self._precedents[i] for i in indices]

        scored = []
        for prec in candidates:
            # Match against keywords and description
            prec_words = set(prec.description.lower().split())
            prec_words.update(w.lower() for w in prec.keywords)

            overlap = len(text_words & prec_words)
            if overlap > 0:
                similarity = overlap / max(len(text_words | prec_words), 1)
                scored.append((similarity, prec))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, prec in scored[:top_k]:
            results.append(
                {
                    "id": prec.id,
                    "similarity": round(sim, 4),
                    "description": prec.description,
                    "decision": prec.decision,
                    "outcome": prec.outcome,
                    "was_appropriate": prec.was_appropriate,
                    "lessons": prec.lessons,
                    "domain": prec.domain,
                }
            )

        return results

    def get_by_domain(self, domain: str) -> list[Precedent]:
        """Get all precedents for a domain."""
        indices = self._by_domain.get(domain, [])
        return [self._precedents[i] for i in indices]

    @property
    def count(self) -> int:
        return len(self._precedents)

    # ---------- Backward-compat aliases (used by tests & orchestrator) -----

    @property
    def precedents(self) -> list[dict[str, Any]]:
        """Return all precedents as dicts (backward-compat)."""
        return [
            {
                "id": p.id,
                "domain": p.domain,
                "description": p.description,
                "keywords": p.keywords,
                "decision": p.decision,
                "outcome": p.outcome,
                "was_appropriate": p.was_appropriate,
                "lessons": p.lessons,
                "eds_score": p.eds_score,
                "verdict": p.decision,  # alias used by filter_by_verdict
                **p.metadata,
            }
            for p in self._precedents
        ]

    def add_precedent(
        self,
        case_id: str,
        task: str,
        domain: str,
        verdict: str,
        eds_score: float = 0.0,
        reasoning: str = "",
        **kwargs: Any,
    ) -> None:
        """Add a precedent using keyword arguments (backward-compat)."""
        self.add(
            Precedent(
                id=case_id,
                domain=domain,
                description=task,
                keywords=kwargs.get("keywords", task.lower().split()),
                decision=verdict,
                outcome=kwargs.get("outcome", reasoning),
                was_appropriate=kwargs.get("was_appropriate", True),
                lessons=reasoning,
                eds_score=eds_score,
                metadata={"task": task, "reasoning": reasoning, "verdict": verdict},
            )
        )

    def search_similar(
        self,
        text: str,
        domain: str | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Alias for :meth:`query` (backward-compat)."""
        return self.query(text, domain=domain, top_k=top_k)

    def filter_by_verdict(self, verdict: str) -> list[dict[str, Any]]:
        """Return precedents whose decision matches *verdict*."""
        results = []
        for p in self._precedents:
            v = p.metadata.get("verdict", p.decision)
            if v == verdict:
                results.append(
                    {
                        "id": p.id,
                        "domain": p.domain,
                        "description": p.description,
                        "decision": p.decision,
                        "verdict": verdict,
                        "eds_score": p.eds_score,
                    }
                )
        return results

    def get_statistics(self) -> dict[str, Any]:
        """Return summary statistics about stored precedents."""
        domains: dict[str, int] = {}
        for p in self._precedents:
            domains[p.domain] = domains.get(p.domain, 0) + 1
        return {
            "total_precedents": len(self._precedents),
            "domains": list(domains.keys()),
            "precedents_per_domain": domains,
        }

    def save(self, filepath: str | Path | None = None) -> None:
        """Persist precedents to a JSON file."""
        fp = Path(filepath) if filepath else (Path(self._filepath) if self._filepath else None)
        if fp is None:
            logger.warning("save() called without a filepath — skipping")
            return
        fp.parent.mkdir(parents=True, exist_ok=True)
        with open(fp, "w") as f:
            json.dump(self.precedents, f, indent=2, default=str)
        logger.debug("Saved %d precedents to %s", len(self._precedents), fp)

    def load(self, filepath: str | Path | None = None) -> None:
        """Load precedents from a JSON file (merges with existing)."""
        fp = Path(filepath) if filepath else (Path(self._filepath) if self._filepath else None)
        if fp is None or not fp.exists():
            return
        self._load_single_file(fp)

    def _load_single_file(self, path: Path) -> None:
        """Load precedents from a single JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("precedents", [])
            for p in items:
                self.add_from_dict(p)
            logger.debug("Loaded precedents from %s", path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)

    def _load_from_directory(self, data_dir: Path) -> None:
        if not data_dir.exists():
            return
        for json_file in sorted(data_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else data.get("precedents", [])
                for p in items:
                    self.add_from_dict(p)
                logger.debug("Loaded precedents from %s", json_file.name)
            except Exception as e:
                logger.warning("Failed to load %s: %s", json_file, e)

    # ---------- Convenience loaders ----------

    _DOMAIN_FILE_MAP: dict[str, str] = {
        "healthcare": "healthcare_precedents.json",
        "finance": "finance_precedents.json",
        "hiring": "hiring_precedents.json",
        "disaster": "disaster_precedents.json",
    }

    def load_precedents(self, domain: str, data_dir: str | Path | None = None) -> int:
        """Load precedents for a specific domain from its JSON file.

        Parameters
        ----------
        domain : str
            One of 'healthcare', 'finance', 'hiring', 'disaster'.
        data_dir : str or Path, optional
            Directory containing the JSON files.
            Defaults to ``<project_root>/data/precedents/``.

        Returns
        -------
        int
            Number of precedents loaded.

        Raises
        ------
        ValueError
            If the domain is not recognized.
        """
        filename = self._DOMAIN_FILE_MAP.get(domain.lower())
        if filename is None:
            raise ValueError(
                f"Unknown domain '{domain}'. Choose from: {', '.join(self._DOMAIN_FILE_MAP)}"
            )

        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "precedents"
        else:
            data_dir = Path(data_dir)

        filepath = data_dir / filename
        if not filepath.exists():
            logger.warning("Precedent file not found: %s", filepath)
            return 0

        before = self.count
        with open(filepath) as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get("precedents", [])
        for p in items:
            self.add_from_dict(p)

        loaded = self.count - before
        logger.info(
            "Loaded %d precedents for domain '%s' from %s",
            loaded,
            domain,
            filepath.name,
        )
        return loaded

    @classmethod
    def auto_load(cls, data_dir: str | Path | None = None) -> PrecedentStore:
        """Create a PrecedentStore pre-populated with all domain precedents.

        This is the recommended factory method for production use.
        It loads every domain precedent JSON in ``data/precedents/``.

        Parameters
        ----------
        data_dir : str or Path, optional
            Override the default ``data/precedents/`` directory.

        Returns
        -------
        PrecedentStore
            Fully loaded instance.

        Example
        -------
        >>> store = PrecedentStore.auto_load()
        >>> store.count >= 128
        True
        """
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "precedents"

        store = cls(data_dir=data_dir)
        logger.info("auto_load complete — %d precedents", store.count)
        return store
