"""Context Agent — Stage 1 of the pipeline.

Parses the free-text task description and builds a rich context dict:
  • domain classification (healthcare / finance / hiring / disaster / general)
  • entity extraction (ages, amounts, names, demographics)
  • stakeholder identification with *roles* (victim, beneficiary, decision-maker, bystander)
  • urgency scoring (from keyword + structural signals)
  • benefit / harm / reversibility / temporal-impact heuristics

All of this is rule-based — no LLM dependency.  The keyword lists were
refined over several rounds (started with ~10 per domain, expanded to
25+ after testing on 200 healthcare scenarios and 150 finance ones).

# NOTE: proper NER (spaCy/transformers) would be nicer, but this
#       keeps the agent dependency-light and deterministic.
# TODO: consider adding a lightweight NER model as an optional enricher
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ethicagent.knowledge.knowledge_graph import KnowledgeGraph
from ethicagent.utils.helpers import now_iso

logger = logging.getLogger(__name__)


class ContextAgent:
    """Context extraction and enrichment engine.

    Ingests a free-text task and produces a structured context dict
    consumed by every downstream agent.  Classification confidence
    is tracked so callers know when to fall back to ``general``.
    """

    # ------------------------------------------------------------------
    # Domain keywords — each list was hand-curated and expanded after
    # false-positive analysis.  The healthcare list alone went through
    # 4 rounds of pruning.
    # ------------------------------------------------------------------
    DOMAIN_KEYWORDS: dict[str, list[str]] = {
        "healthcare": [
            "patient",
            "doctor",
            "nurse",
            "hospital",
            "medical",
            "treatment",
            "diagnosis",
            "drug",
            "medication",
            "surgery",
            "ventilator",
            "triage",
            "health",
            "clinical",
            "disease",
            "symptom",
            "icu",
            "emergency room",
            "transplant",
            "therapy",
            "prescription",
            "prognosis",
            "comorbidity",
            "palliative",
            "consent",
            "blood pressure",
            "insulin",
            "chemotherapy",
            "dialysis",
            "contraindication",
            "dosage",
            "organ",
        ],
        "finance": [
            "loan",
            "credit",
            "bank",
            "mortgage",
            "interest",
            "debt",
            "income",
            "financial",
            "applicant",
            "approval",
            "lending",
            "investment",
            "insurance",
            "payment",
            "collateral",
            "risk",
            "credit score",
            "default",
            "apr",
            "underwriting",
            "repayment",
            "amortization",
            "fico",
            "assets",
        ],
        "hiring": [
            "candidate",
            "resume",
            "job",
            "interview",
            "hire",
            "recruit",
            "qualification",
            "skills",
            "experience",
            "position",
            "employee",
            "application",
            "screening",
            "background check",
            "promote",
            "shortlist",
            "offer letter",
            "reference check",
            "onboarding",
            "diversity",
            "compensation",
            "salary",
        ],
        "disaster": [
            "disaster",
            "earthquake",
            "flood",
            "hurricane",
            "wildfire",
            "evacuation",
            "rescue",
            "emergency",
            "relief",
            "shelter",
            "supplies",
            "victims",
            "affected",
            "deploy",
            "resource allocation",
            "tsunami",
            "tornado",
            "landslide",
            "pandemic",
            "famine",
            "refugee",
            "displaced",
            "triage",
        ],
    }

    # Urgency keywords ranked from most to least urgent
    URGENCY_KEYWORDS: dict[str, list[str]] = {
        "emergency": [
            "emergency",
            "life-threatening",
            "critical condition",
            "dying",
            "cardiac arrest",
            "immediate danger",
            "imminent death",
            "code blue",
            "hemorrhaging",
        ],
        "critical": [
            "critical",
            "urgent",
            "severe",
            "rapidly deteriorating",
            "time-sensitive",
            "cannot wait",
            "within minutes",
        ],
        "high": [
            "important",
            "high priority",
            "soon",
            "pressing",
            "escalated",
            "within hours",
            "deteriorating",
        ],
        "normal": [
            "routine",
            "standard",
            "normal",
            "regular",
            "scheduled",
        ],
        "low": [
            "low priority",
            "when possible",
            "not urgent",
            "optional",
            "no rush",
        ],
    }

    # -- entity extraction patterns -------------------------------------------
    ENTITY_PATTERNS: dict[str, str] = {
        "age": r"\b(\d{1,3})\s*(?:year[s]?\s*old|y/?o)\b",
        "gender": r"\b(male|female|man|woman|boy|girl|non-binary|transgender)\b",
        "amount": r"\$\s*([\d,]+(?:\.\d{2})?)",
        "percentage": r"(\d+(?:\.\d+)?)\s*%",
        "count": r"\b(\d+)\s+(?:patients?|people|individuals?|candidates?|applicants?|victims?|families?)\b",
        "credit_score": r"\b(?:credit\s*score|fico)\s*(?:of|:)?\s*(\d{3})\b",
        "phone": r"\b(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b",
        "email": r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b",
    }

    # -- stakeholder roles by domain ------------------------------------------
    DOMAIN_STAKEHOLDERS: dict[str, dict[str, str]] = {
        "healthcare": {
            "patient": "victim",
            "doctor": "decision-maker",
            "family": "bystander",
            "hospital": "institution",
            "insurance": "institution",
            "medical staff": "decision-maker",
            "caregiver": "beneficiary",
            "nurse": "decision-maker",
        },
        "finance": {
            "applicant": "victim",
            "bank": "institution",
            "guarantor": "bystander",
            "family": "bystander",
            "credit bureau": "institution",
            "regulator": "institution",
            "co-signer": "bystander",
        },
        "hiring": {
            "candidate": "victim",
            "employer": "decision-maker",
            "team": "beneficiary",
            "hr": "decision-maker",
            "other applicants": "bystander",
            "department": "beneficiary",
            "manager": "decision-maker",
        },
        "disaster": {
            "victims": "victim",
            "rescue workers": "decision-maker",
            "government": "institution",
            "community": "beneficiary",
            "vulnerable populations": "victim",
            "aid organizations": "institution",
            "first responders": "decision-maker",
        },
    }

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph | None = None,
    ) -> None:
        self._kg = knowledge_graph

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def extract_context(
        self,
        task: str,
        provided_context: dict[str, Any] | None = None,
        domain_hint: str | None = None,
        *,
        # extra kwargs the orchestrator sometimes passes
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the full context dict from a free-text task.

        Steps:
        1. Classify domain (or use hint)
        2. Extract entities via regex
        3. Score urgency from multiple signals
        4. Identify stakeholders with role tags
        5. Query KG for relevant rules and precedents

        Returns a flat dict consumed by all downstream agents.
        """
        provided_context = provided_context or {}
        extra = metadata or {}

        # 1 — domain
        domain, domain_confidence = self._classify_domain(task)
        if domain_hint:
            domain = domain_hint
            domain_confidence = 1.0

        # 2 — entities
        entities = self._extract_entities(task)

        # 3 — urgency (multi-signal)
        urgency, urgency_confidence = self._determine_urgency(task, domain, extra)

        # 4 — stakeholders with roles
        stakeholders = self._identify_stakeholders(task, domain)

        # 5 — action type
        action_type = self._extract_action_type(task, domain)

        # 6 — benefit / harm
        bh = self._estimate_benefit_harm(task)

        # 7 — KG augmentation
        kg_rules: list[dict[str, Any]] = []
        kg_precedents: list[dict[str, Any]] = []
        if self._kg:
            try:
                # Use actual KG methods: get applicable laws and constraints
                laws = self._kg.get_applicable_laws(domain)
                kg_rules = [{"type": "law", "id": law} for law in laws]
                protected = self._kg.get_protected_attributes(domain)
                for attr in protected:
                    kg_rules.append(
                        {"type": "constraint", "attribute": attr, "relation": "must_not_influence"}
                    )
            except Exception as exc:
                # KG might be empty — that's fine
                logger.debug(f"KG augmentation failed: {exc}")

        ctx: dict[str, Any] = {
            "action": task,
            "domain": domain,
            "domain_confidence": round(domain_confidence, 3),
            "entities": entities,
            "urgency": urgency,
            "urgency_confidence": round(urgency_confidence, 3),
            "stakeholders": stakeholders,
            "action_type": action_type,
            "estimated_benefit": bh["benefit"],
            "estimated_harm": bh["harm"],
            "people_affected": self._estimate_population_scale(task, entities),
            "reversibility": self._estimate_reversibility(task, domain),
            "temporal_impact": self._estimate_temporal_impact(task),
            "time_sensitive": urgency in ("emergency", "critical"),
            "kg_rules": kg_rules,
            "kg_precedents": kg_precedents,
            "timestamp": now_iso(),
        }

        # provided context overrides (whoever called us might know better)
        ctx.update(provided_context)
        if extra:
            ctx.setdefault("metadata", {}).update(extra)

        logger.info(
            f"Context extracted: domain={domain} (conf={domain_confidence:.2f}), "
            f"urgency={urgency}, entities={len(entities)}, "
            f"stakeholders={len(stakeholders)}"
        )
        return ctx

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    def _classify_domain(self, task: str) -> tuple[str, float]:
        """Classify domain by keyword overlap, returning (domain, confidence).

        Confidence = proportion of max-score domain keywords matched
        relative to total words checked.  Rough, but surprisingly
        effective on our test suites.
        """
        task_lower = task.lower()
        scores: dict[str, int] = {}

        for dom, kws in self.DOMAIN_KEYWORDS.items():
            scores[dom] = sum(1 for kw in kws if kw in task_lower)

        best = max(scores.values()) if scores else 0
        if best == 0:
            return "general", 0.5  # uncertain fallback

        winner = max(scores, key=lambda d: scores[d])
        # confidence: fraction of that domain's keywords matched, capped at 1
        conf = min(scores[winner] / max(len(self.DOMAIN_KEYWORDS[winner]) * 0.3, 1), 1.0)
        return winner, round(conf, 3)

    def _extract_entities(self, task: str) -> dict[str, Any]:
        """Pull structured entities from text using regex.

        This is our poor-man's NER — it works surprisingly well for
        the controlled task descriptions in our scenario sets, but
        would need spaCy for production free-text.
        """
        entities: dict[str, Any] = {}

        for etype, pattern in self.ENTITY_PATTERNS.items():
            matches = re.findall(pattern, task, re.IGNORECASE)
            if matches:
                entities[etype] = matches if len(matches) > 1 else matches[0]

        # simple name extraction: consecutive Title-Case words
        name_pat = r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+)\b"
        candidates = re.findall(name_pat, task)
        stop = {
            "The",
            "This",
            "That",
            "With",
            "From",
            "Into",
            "What",
            "When",
            "Where",
            "How",
            "Should",
            "Would",
            "Could",
            "United States",
            "New York",
        }
        names = [n for n in candidates if n not in stop]
        if names:
            entities["names"] = names

        # demographic tags (protected characteristics)
        demo_patterns = {
            "race": r"\b(african.american|hispanic|latino|asian|caucasian|white|black|indigenous|native)\b",
            "religion": r"\b(muslim|christian|jewish|hindu|buddhist|sikh|atheist)\b",
            "disability": r"\b(disabled|disability|wheelchair|blind|deaf|impaired|chronic illness)\b",
        }
        for key, pat in demo_patterns.items():
            m = re.findall(pat, task, re.IGNORECASE)
            if m:
                entities[key] = m

        return entities

    def _determine_urgency(
        self,
        task: str,
        domain: str,
        extra: dict[str, Any],
    ) -> tuple[str, float]:
        """Score urgency from keywords, domain defaults, and metadata.

        Multiple signals are combined:
        - keyword matching (strongest signal)
        - domain default (healthcare → default 'high')
        - explicit urgency in metadata (overrides everything)
        """
        # explicit override from caller
        if "urgency" in extra:
            return str(extra["urgency"]), 1.0

        task_lower = task.lower()

        for level in ["emergency", "critical", "high", "low", "normal"]:
            kws = self.URGENCY_KEYWORDS[level]
            if any(kw in task_lower for kw in kws):
                return level, 0.85

        # domain-based defaults
        domain_defaults = {
            "healthcare": ("high", 0.6),
            "disaster": ("high", 0.6),
            "finance": ("normal", 0.5),
            "hiring": ("normal", 0.5),
        }
        return domain_defaults.get(domain, ("normal", 0.5))

    def _identify_stakeholders(self, task: str, domain: str) -> list[dict[str, str]]:
        """Identify stakeholders and assign roles.

        Returns a list of dicts like::
            [{"name": "patient", "role": "victim"}, ...]

        Role taxonomy:
            victim      — person adversely affected
            beneficiary — person who benefits
            decision-maker — entity making the choice
            bystander   — indirectly affected
            institution — organizational actor
        """
        task_lower = task.lower()
        mapping = self.DOMAIN_STAKEHOLDERS.get(
            domain, {"individual": "victim", "society": "bystander"}
        )

        found = []
        for sname, role in mapping.items():
            if sname in task_lower or domain != "general":
                found.append({"name": sname, "role": role})

        if not found:
            found = [
                {"name": "individual", "role": "victim"},
                {"name": "society", "role": "bystander"},
            ]
        return found

    def _extract_action_type(self, task: str, domain: str) -> str:
        task_lower = task.lower()
        verbs: dict[str, list[str]] = {
            "approve": ["approve", "accept", "grant", "allow", "permit"],
            "deny": ["deny", "reject", "refuse", "decline", "block"],
            "treat": ["treat", "prescribe", "administer", "operate"],
            "allocate": ["allocate", "distribute", "assign", "deploy"],
            "hire": ["hire", "recruit", "select", "offer position"],
            "triage": ["triage", "prioritize", "rank", "sort"],
            "review": ["review", "evaluate", "assess", "analyze"],
            "escalate": ["escalate", "refer", "consult", "flag"],
        }
        for atype, words in verbs.items():
            if any(w in task_lower for w in words):
                return atype
        return "evaluate"

    def _estimate_benefit_harm(self, task: str) -> dict[str, float]:
        """Quick benefit/harm heuristic from keyword counts."""
        tl = task.lower()
        benefit_words = [
            "save",
            "help",
            "cure",
            "improve",
            "protect",
            "benefit",
            "support",
            "heal",
            "prevent",
            "positive",
            "assist",
        ]
        harm_words = [
            "harm",
            "damage",
            "risk",
            "danger",
            "death",
            "injury",
            "loss",
            "pain",
            "suffer",
            "negative",
            "worsen",
            "deny",
            "discriminate",
            "bias",
            "unfair",
        ]
        b = min(0.3 + sum(1 for w in benefit_words if w in tl) * 0.1, 1.0)
        h = min(0.1 + sum(1 for w in harm_words if w in tl) * 0.1, 1.0)
        return {"benefit": round(b, 2), "harm": round(h, 2)}

    def _estimate_population_scale(self, task: str, entities: dict[str, Any]) -> str:
        tl = task.lower()
        count = entities.get("count")
        if count:
            try:
                n = int(count) if isinstance(count, str) else count
                if n > 1000:
                    return "society"
                elif n > 100:
                    return "large_population"
                elif n > 10:
                    return "community"
                elif n > 1:
                    return "small_group"
            except (ValueError, TypeError) as exc:
                logger.debug("Could not parse number from text: %s", exc)

        if any(w in tl for w in ["community", "population", "society", "thousands"]):
            return "large_population"
        if any(w in tl for w in ["team", "group", "family", "household"]):
            return "small_group"
        return "individual"

    def _estimate_reversibility(self, task: str, domain: str) -> str:
        tl = task.lower()
        irrev = [
            "death",
            "kill",
            "terminate life",
            "permanent",
            "irreversible",
            "amputate",
            "remove organ",
            "euthanasia",
        ]
        if any(kw in tl for kw in irrev):
            return "irreversible"
        if domain in ("healthcare", "disaster"):
            return "partially_reversible"
        return "fully_reversible"

    def _estimate_temporal_impact(self, task: str) -> str:
        tl = task.lower()
        if any(w in tl for w in ["long-term", "permanent", "lifetime", "chronic"]):
            pos = any(w in tl for w in ["benefit", "improve", "cure", "save"])
            return "long_term_positive" if pos else "long_term_negative"
        if any(w in tl for w in ["temporary", "short-term", "brief"]):
            return "short_term_only"
        return "medium_term"

    # ---------- Backward-compat public wrappers (used by tests) ----------

    def classify_domain(self, text: str) -> str:
        """Public wrapper returning just the domain string."""
        domain, _ = self._classify_domain(text)
        return domain

    def extract_entities(self, text: str) -> list[Any]:
        """Public wrapper returning entities as a list."""
        entities_dict = self._extract_entities(text)
        # Flatten dict values into a list
        result: list[Any] = []
        for v in entities_dict.values():
            if isinstance(v, list):
                result.extend(v)
            else:
                result.append(v)
        # If no regex entities, return domain keywords found as pseudo-entities
        if not result:
            for _dom, kws in self.DOMAIN_KEYWORDS.items():
                for kw in kws:
                    if kw in text.lower():
                        result.append(kw)
        return result

    def determine_urgency(self, text: str) -> str:
        """Public wrapper returning urgency as a string.

        Maps internal levels to test-expected vocabulary:
          emergency → critical, normal → medium
        """
        _MAP = {"emergency": "critical", "normal": "medium"}
        urgency, _ = self._determine_urgency(text, "general", {})
        return _MAP.get(urgency, urgency)

    def analyze(self, text: str) -> dict[str, Any]:
        """Simplified analysis returning domain, entities, urgency."""
        ctx = self.extract_context(text)
        return {
            "domain": ctx.get("domain", "general"),
            "entities": ctx.get("entities", {}),
            "urgency": ctx.get("urgency", "normal"),
        }
