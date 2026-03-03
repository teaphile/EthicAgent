"""Ontology loader — domain concept hierarchies.

This module loads and manages domain-specific ontologies (concept
hierarchies) that help the pipeline understand the relationships
between domain concepts.

Example hierarchy (healthcare):
  medical_intervention
  ├── surgical
  │   ├── emergency_surgery
  │   └── elective_surgery
  ├── pharmaceutical
  │   ├── prescription
  │   └── over_the_counter
  └── diagnostic
      ├── imaging
      └── lab_test

The ontology supports:
  - Checking if concept A is a subtype of concept B
  - Finding the nearest common ancestor
  - Getting domain-specific vocabulary
  - Mapping user terms to canonical concepts

Data can come from YAML/JSON files or be built programmatically.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class OntologyNode:
    """A node in the concept hierarchy."""

    def __init__(self, name: str, parent: Optional[OntologyNode] = None, **attrs: Any):
        self.name = name
        self.parent = parent
        self.children: list[OntologyNode] = []
        self.attrs = attrs
        self.synonyms: list[str] = attrs.get("synonyms", [])

    def add_child(self, child: OntologyNode) -> OntologyNode:
        child.parent = self
        self.children.append(child)
        return child

    @property
    def depth(self) -> int:
        d = 0
        node = self
        while node.parent:
            d += 1
            node = node.parent
        return d

    @property
    def path(self) -> list[str]:
        """Path from root to this node."""
        parts = []
        node: Optional[OntologyNode] = self
        while node:
            parts.append(node.name)
            node = node.parent
        return list(reversed(parts))

    def is_descendant_of(self, ancestor_name: str) -> bool:
        """Check if this node is a descendant of the named ancestor."""
        node: Optional[OntologyNode] = self.parent
        while node:
            if node.name == ancestor_name:
                return True
            node = node.parent
        return False

    def __repr__(self) -> str:
        return f"OntologyNode({self.name}, depth={self.depth})"


class DomainOntology:
    """A complete domain ontology (concept hierarchy tree)."""

    def __init__(self, domain: str, root: OntologyNode | None = None):
        self.domain = domain
        self.root = root or OntologyNode(domain)
        self._index: dict[str, OntologyNode] = {self.root.name: self.root}
        self._synonym_map: dict[str, str] = {}

    def add_concept(
        self,
        name: str,
        parent_name: str | None = None,
        synonyms: list[str] | None = None,
        **attrs: Any,
    ) -> OntologyNode:
        """Add a concept to the ontology."""
        parent = self._index.get(parent_name or self.root.name, self.root)
        node = OntologyNode(name, **attrs, synonyms=synonyms or [])
        parent.add_child(node)
        self._index[name] = node

        # Index synonyms
        for syn in (synonyms or []):
            self._synonym_map[syn.lower()] = name

        return node

    def get_node(self, name: str) -> OntologyNode | None:
        """Look up a concept by name or synonym."""
        if name in self._index:
            return self._index[name]
        # Try synonym lookup
        canonical = self._synonym_map.get(name.lower())
        if canonical:
            return self._index.get(canonical)
        return None

    def is_subtype(self, child_name: str, parent_name: str) -> bool:
        """Check if child_name is a subtype of parent_name."""
        child = self.get_node(child_name)
        if not child:
            return False
        if child.name == parent_name:
            return True
        return child.is_descendant_of(parent_name)

    def common_ancestor(self, name_a: str, name_b: str) -> str | None:
        """Find the nearest common ancestor of two concepts."""
        node_a = self.get_node(name_a)
        node_b = self.get_node(name_b)
        if not node_a or not node_b:
            return None

        path_a = set(n for n in self._ancestors(node_a))
        node: Optional[OntologyNode] = node_b
        while node:
            if node.name in path_a:
                return node.name
            node = node.parent
        return None

    def _ancestors(self, node: OntologyNode) -> list[str]:
        result = [node.name]
        while node.parent:
            node = node.parent
            result.append(node.name)
        return result

    def get_vocabulary(self) -> list[str]:
        """Get all concept names in the ontology."""
        return list(self._index.keys())

    def resolve_term(self, term: str) -> str | None:
        """Map a user term to its canonical concept name."""
        if term in self._index:
            return term
        return self._synonym_map.get(term.lower())

    @property
    def concept_count(self) -> int:
        return len(self._index)


# ---------------------------------------------------------------------------
# Built-in domain ontologies
# ---------------------------------------------------------------------------

def _build_healthcare_ontology() -> DomainOntology:
    o = DomainOntology("healthcare")
    
    # Interventions
    o.add_concept("intervention", "healthcare")
    o.add_concept("surgical", "intervention", synonyms=["surgery", "operation"])
    o.add_concept("emergency_surgery", "surgical")
    o.add_concept("elective_surgery", "surgical")
    o.add_concept("pharmaceutical", "intervention", synonyms=["medication", "drug"])
    o.add_concept("prescription", "pharmaceutical")
    o.add_concept("over_the_counter", "pharmaceutical", synonyms=["OTC"])
    o.add_concept("diagnostic", "intervention", synonyms=["diagnosis", "test"])
    o.add_concept("imaging", "diagnostic", synonyms=["scan", "x-ray", "MRI"])
    o.add_concept("lab_test", "diagnostic", synonyms=["blood test", "biopsy"])
    o.add_concept("therapy", "intervention", synonyms=["treatment"])
    o.add_concept("physical_therapy", "therapy")
    o.add_concept("psychotherapy", "therapy", synonyms=["counseling"])
    
    # Conditions
    o.add_concept("condition", "healthcare")
    o.add_concept("acute", "condition")
    o.add_concept("chronic", "condition")
    o.add_concept("terminal", "condition")
    o.add_concept("mental_health", "condition")
    
    # Stakeholders
    o.add_concept("healthcare_role", "healthcare")
    o.add_concept("patient", "healthcare_role")
    o.add_concept("physician", "healthcare_role", synonyms=["doctor"])
    o.add_concept("nurse", "healthcare_role")
    o.add_concept("caregiver", "healthcare_role")
    
    return o


def _build_finance_ontology() -> DomainOntology:
    o = DomainOntology("finance")
    
    o.add_concept("financial_product", "finance")
    o.add_concept("loan", "financial_product", synonyms=["credit", "lending"])
    o.add_concept("mortgage", "loan")
    o.add_concept("personal_loan", "loan")
    o.add_concept("auto_loan", "loan")
    o.add_concept("insurance", "financial_product")
    o.add_concept("investment", "financial_product")
    o.add_concept("credit_card", "financial_product")
    
    o.add_concept("risk_factor", "finance")
    o.add_concept("credit_score", "risk_factor", synonyms=["FICO"])
    o.add_concept("income", "risk_factor", synonyms=["salary", "earnings"])
    o.add_concept("debt_ratio", "risk_factor", synonyms=["DTI"])
    o.add_concept("employment", "risk_factor")
    o.add_concept("collateral", "risk_factor")
    
    o.add_concept("finance_role", "finance")
    o.add_concept("borrower", "finance_role", synonyms=["applicant"])
    o.add_concept("lender", "finance_role")
    o.add_concept("underwriter", "finance_role")
    
    return o


def _build_hiring_ontology() -> DomainOntology:
    o = DomainOntology("hiring")
    
    o.add_concept("selection_stage", "hiring")
    o.add_concept("resume_screening", "selection_stage")
    o.add_concept("interview", "selection_stage")
    o.add_concept("assessment", "selection_stage")
    o.add_concept("background_check", "selection_stage")
    o.add_concept("offer", "selection_stage")
    
    o.add_concept("qualification", "hiring")
    o.add_concept("education", "qualification")
    o.add_concept("experience", "qualification")
    o.add_concept("skill", "qualification")
    o.add_concept("certification", "qualification")
    
    o.add_concept("hiring_role", "hiring")
    o.add_concept("candidate", "hiring_role", synonyms=["applicant"])
    o.add_concept("recruiter", "hiring_role")
    o.add_concept("hiring_manager", "hiring_role")
    
    return o


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_BUILTIN_ONTOLOGIES: dict[str, DomainOntology] = {}


def get_ontology(domain: str) -> DomainOntology:
    """Get (or lazily build) the ontology for a domain."""
    if domain not in _BUILTIN_ONTOLOGIES:
        builders = {
            "healthcare": _build_healthcare_ontology,
            "finance": _build_finance_ontology,
            "hiring": _build_hiring_ontology,
        }
        builder = builders.get(domain)
        if builder:
            _BUILTIN_ONTOLOGIES[domain] = builder()
        else:
            _BUILTIN_ONTOLOGIES[domain] = DomainOntology(domain)
    return _BUILTIN_ONTOLOGIES[domain]


def load_ontology_from_json(filepath: str | Path) -> DomainOntology:
    """Load an ontology from a JSON file.

    Expected format::
        {
            "domain": "healthcare",
            "concepts": [
                {"name": "intervention", "parent": "healthcare", "synonyms": ["treatment"]},
                ...
            ]
        }
    """
    path = Path(filepath)
    with open(path) as f:
        data = json.load(f)

    domain = data.get("domain", path.stem)
    ontology = DomainOntology(domain)

    for concept in data.get("concepts", []):
        ontology.add_concept(
            name=concept["name"],
            parent_name=concept.get("parent"),
            synonyms=concept.get("synonyms", []),
        )

    _BUILTIN_ONTOLOGIES[domain] = ontology
    return ontology
