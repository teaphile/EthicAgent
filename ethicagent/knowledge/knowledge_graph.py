"""Knowledge graph for ethical relationships and constraints.

This is the structured-knowledge backbone of the system.  It encodes
relationships like:

  - "race" --[protected_attribute_of]--> "credit_decision"
  - "HIPAA" --[requires]--> "patient_consent"
  - "emergency" --[overrides]--> "standard_consent"

We use NetworkX for the graph and provide query methods that the
SymbolicReasoner and ContextualEthicsEvaluator can call.

Data sources:
  1. Built-in seed graph (below)
  2. JSON files in data/knowledge/ (loaded at startup)
  3. Runtime additions via add_node/add_edge

The graph is intentionally modest in size for now — a real production
system would use a proper triplestore, but NetworkX is fine for
research and demo purposes.

TODO: Consider RDFLib or Neo4j for production scale.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore

logger = logging.getLogger(__name__)


class _FallbackGraphView:
    """Minimal graph-like view when NetworkX is unavailable.

    This preserves backward compatibility for callers that expect
    ``kg.graph.has_node(...)``, ``kg.graph.has_edge(...)``, and
    ``len(kg.graph.nodes)`` to work even in fallback mode.
    """

    def __init__(
        self,
        nodes: dict[str, dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> None:
        self._nodes = nodes
        self._edges = edges

    @property
    def nodes(self) -> list[str]:
        return list(self._nodes.keys())

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def has_edge(self, source: str, target: str) -> bool:
        return any(e.get("source") == source and e.get("target") == target for e in self._edges)


# ---------------------------------------------------------------------------
# Seed graph data
# ---------------------------------------------------------------------------

_SEED_NODES: list[dict[str, Any]] = [
    # Protected attributes
    {"id": "race", "type": "protected_attribute", "label": "Race/Ethnicity"},
    {"id": "gender", "type": "protected_attribute", "label": "Gender"},
    {"id": "age", "type": "protected_attribute", "label": "Age"},
    {"id": "disability", "type": "protected_attribute", "label": "Disability"},
    {"id": "religion", "type": "protected_attribute", "label": "Religion"},
    {"id": "sexual_orientation", "type": "protected_attribute", "label": "Sexual Orientation"},
    {"id": "national_origin", "type": "protected_attribute", "label": "National Origin"},
    # Decision domains
    {"id": "credit_decision", "type": "decision_domain", "label": "Credit/Lending Decision"},
    {"id": "hiring_decision", "type": "decision_domain", "label": "Employment Decision"},
    {"id": "medical_decision", "type": "decision_domain", "label": "Medical Decision"},
    {"id": "triage_decision", "type": "decision_domain", "label": "Emergency Triage"},
    {"id": "insurance_decision", "type": "decision_domain", "label": "Insurance Decision"},
    # Legal frameworks
    {"id": "hipaa", "type": "legal_framework", "label": "HIPAA"},
    {"id": "gdpr", "type": "legal_framework", "label": "GDPR"},
    {"id": "ecoa", "type": "legal_framework", "label": "Equal Credit Opportunity Act"},
    {"id": "ada", "type": "legal_framework", "label": "Americans with Disabilities Act"},
    {"id": "title_vii", "type": "legal_framework", "label": "Title VII Civil Rights Act"},
    {"id": "eu_ai_act", "type": "legal_framework", "label": "EU AI Act"},
    # Ethical principles
    {"id": "patient_consent", "type": "ethical_principle", "label": "Patient Consent"},
    {"id": "informed_consent", "type": "ethical_principle", "label": "Informed Consent"},
    {"id": "data_minimization", "type": "ethical_principle", "label": "Data Minimization"},
    {"id": "transparency", "type": "ethical_principle", "label": "Transparency"},
    {"id": "accountability", "type": "ethical_principle", "label": "Accountability"},
    {"id": "fairness", "type": "ethical_principle", "label": "Fairness"},
    {"id": "do_no_harm", "type": "ethical_principle", "label": "Do No Harm"},
    # Stakeholder types
    {"id": "patient", "type": "stakeholder", "label": "Patient"},
    {"id": "applicant", "type": "stakeholder", "label": "Job Applicant"},
    {"id": "borrower", "type": "stakeholder", "label": "Loan Borrower"},
    {"id": "clinician", "type": "stakeholder", "label": "Clinician"},
    {"id": "employer", "type": "stakeholder", "label": "Employer"},
]

_SEED_EDGES: list[dict[str, Any]] = [
    # Protected attributes → decision domains (constraints)
    {"source": "race", "target": "credit_decision", "relation": "must_not_influence"},
    {"source": "race", "target": "hiring_decision", "relation": "must_not_influence"},
    {"source": "gender", "target": "hiring_decision", "relation": "must_not_influence"},
    {"source": "gender", "target": "credit_decision", "relation": "must_not_influence"},
    {"source": "age", "target": "hiring_decision", "relation": "must_not_influence"},
    {"source": "disability", "target": "hiring_decision", "relation": "requires_accommodation"},
    {"source": "religion", "target": "hiring_decision", "relation": "must_not_influence"},
    # Legal frameworks → requirements
    {"source": "hipaa", "target": "patient_consent", "relation": "requires"},
    {"source": "hipaa", "target": "data_minimization", "relation": "requires"},
    {"source": "gdpr", "target": "informed_consent", "relation": "requires"},
    {"source": "gdpr", "target": "data_minimization", "relation": "requires"},
    {"source": "gdpr", "target": "transparency", "relation": "requires"},
    {"source": "ecoa", "target": "fairness", "relation": "requires"},
    {"source": "ada", "target": "fairness", "relation": "requires"},
    {"source": "title_vii", "target": "fairness", "relation": "requires"},
    {"source": "eu_ai_act", "target": "transparency", "relation": "requires"},
    {"source": "eu_ai_act", "target": "accountability", "relation": "requires"},
    # Legal frameworks → decision domains (applicability)
    {"source": "hipaa", "target": "medical_decision", "relation": "applies_to"},
    {"source": "ecoa", "target": "credit_decision", "relation": "applies_to"},
    {"source": "title_vii", "target": "hiring_decision", "relation": "applies_to"},
    {"source": "ada", "target": "hiring_decision", "relation": "applies_to"},
    # Decision domains → principles
    {"source": "medical_decision", "target": "do_no_harm", "relation": "requires"},
    {"source": "medical_decision", "target": "patient_consent", "relation": "requires"},
    {"source": "triage_decision", "target": "do_no_harm", "relation": "requires"},
    {"source": "credit_decision", "target": "transparency", "relation": "requires"},
    {"source": "hiring_decision", "target": "fairness", "relation": "requires"},
    # Stakeholders → decision domains
    {"source": "patient", "target": "medical_decision", "relation": "affected_by"},
    {"source": "patient", "target": "triage_decision", "relation": "affected_by"},
    {"source": "applicant", "target": "hiring_decision", "relation": "affected_by"},
    {"source": "borrower", "target": "credit_decision", "relation": "affected_by"},
    {"source": "clinician", "target": "medical_decision", "relation": "decides_in"},
    {"source": "employer", "target": "hiring_decision", "relation": "decides_in"},
]


# ---------------------------------------------------------------------------
# Knowledge graph class
# ---------------------------------------------------------------------------


class KnowledgeGraph:
    """Graph-based ethical knowledge store.

    Nodes are entities (protected attributes, decision domains, legal
    frameworks, etc.) and edges are relationships between them.

    Usage::

        kg = KnowledgeGraph()
        constraints = kg.get_constraints("hiring_decision")
        # → [{"source": "race", "relation": "must_not_influence"}, ...]
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        if nx is None:
            logger.warning("NetworkX not installed — KnowledgeGraph will use fallback dict mode")
            self._graph = None
            self._nodes: dict[str, dict] = {}
            self._edges: list[dict] = []
            self._fallback_graph = _FallbackGraphView(self._nodes, self._edges)
        else:
            self._graph = nx.DiGraph()
            self._nodes = {}
            self._edges = []
            self._fallback_graph = None

        # Load seed data
        self._load_seed_data()

        # Load additional data from JSON files
        if data_dir:
            self._load_from_directory(Path(data_dir))

        logger.info(
            "KnowledgeGraph initialized: %d nodes, %d edges",
            self.node_count,
            self.edge_count,
        )

    # ---------- Properties ----------

    @property
    def graph(self):
        """Public access to the underlying NetworkX graph (backward-compat)."""
        if self._graph is not None:
            return self._graph
        return self._fallback_graph

    @property
    def node_count(self) -> int:
        if self._graph is not None:
            return self._graph.number_of_nodes()
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        if self._graph is not None:
            return self._graph.number_of_edges()
        return len(self._edges)

    # ---------- Data loading ----------

    def _load_seed_data(self) -> None:
        for node in _SEED_NODES:
            self.add_node(node["id"], node.get("type", ""), node.get("label", ""))
        for edge in _SEED_EDGES:
            self.add_edge(edge["source"], edge["target"], edge["relation"])

    def _load_from_directory(self, data_dir: Path) -> None:
        """Load knowledge graph data from JSON files in a directory."""
        if not data_dir.exists():
            logger.debug("Knowledge data directory not found: %s", data_dir)
            return

        for json_file in sorted(data_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                nodes = data.get("nodes", [])
                edges = data.get("edges", [])

                for n in nodes:
                    self.add_node(n["id"], n.get("type", ""), n.get("label", ""))
                for e in edges:
                    self.add_edge(e["source"], e["target"], e["relation"])

                logger.debug(
                    "Loaded %d nodes, %d edges from %s",
                    len(nodes),
                    len(edges),
                    json_file.name,
                )
            except Exception as e:
                logger.warning("Failed to load %s: %s", json_file, e)

    def load_from_json(self, filepath: str | Path) -> None:
        """Load additional knowledge from a single JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        for n in data.get("nodes", []):
            self.add_node(n["id"], n.get("type", ""), n.get("label", ""))
        for e in data.get("edges", []):
            self.add_edge(e["source"], e["target"], e["relation"])

    # ---------- Graph operations ----------

    def add_node(
        self,
        node_id: str,
        node_type: str = "",
        label: str = "",
        **attrs: Any,
    ) -> None:
        meta = {"type": node_type, "label": label or node_id, **attrs}
        if self._graph is not None:
            self._graph.add_node(node_id, **meta)
        self._nodes[node_id] = meta

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str = "",
        *,
        relationship: str = "",
        **attrs: Any,
    ) -> None:
        # Accept 'relationship' as alias for 'relation' (backward-compat)
        rel = relation or relationship
        meta = {"relation": rel, **attrs}
        if self._graph is not None:
            # Ensure nodes exist
            if source not in self._graph:
                self._graph.add_node(source)
            if target not in self._graph:
                self._graph.add_node(target)
            self._graph.add_edge(source, target, **meta)
        self._edges.append({"source": source, "target": target, **meta})

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        return self._nodes.get(node_id)

    def get_nodes_by_type(self, node_type: str) -> list[str]:
        return [nid for nid, meta in self._nodes.items() if meta.get("type") == node_type]

    # ---------- Backward-compat aliases (used by tests) ----------

    def get_neighbors(self, node_id: str) -> list[str]:
        """Return successor node IDs for *node_id*."""
        if self._graph is not None and node_id in self._graph:
            return list(self._graph.successors(node_id))
        # Fallback: scan edge list
        return [e["target"] for e in self._edges if e.get("source") == node_id]

    def initialize_domain(self, domain: str, data_dir: str | Path | None = None) -> int:
        """Alias for :meth:`load_domain_data` (backward-compat).

        If the domain file is not found, the seed data still covers it,
        so we silently return 0 instead of raising.
        """
        try:
            return self.load_domain_data(domain, data_dir=data_dir)
        except ValueError:
            logger.debug("initialize_domain: unrecognized domain '%s'", domain)
            return 0

    def query_ethical_context(
        self,
        query: str,
        domain: str | None = None,
    ) -> dict[str, Any]:
        """Return ethical context relevant to *query* in *domain*.

        A lightweight semantic lookup: matches query words against node
        IDs / labels and returns related constraints, laws, and
        stakeholders.
        """
        query_words = set(query.lower().split())
        matched_nodes: list[str] = []
        for nid, meta in self._nodes.items():
            node_words = set(nid.lower().replace("_", " ").split())
            label_words = set(meta.get("label", "").lower().split())
            if query_words & (node_words | label_words):
                matched_nodes.append(nid)

        constraints: list[dict] = []
        laws: list[str] = []
        stakeholders: list[str] = []

        if domain:
            constraints = self.get_constraints(
                {
                    "healthcare": "medical_decision",
                    "finance": "credit_decision",
                    "hiring": "hiring_decision",
                    "disaster": "triage_decision",
                }.get(domain, domain)
            )
            laws = self.get_applicable_laws(domain)
            stakeholders = self.get_stakeholders(domain)

        return {
            "matched_nodes": matched_nodes,
            "constraints": constraints,
            "applicable_laws": laws,
            "stakeholders": stakeholders,
        }

    # ---------- Query methods ----------

    def get_constraints(self, target_id: str) -> list[dict[str, Any]]:
        """Get all constraint edges pointing TO a node.

        Returns edges where the relation implies a constraint:
        must_not_influence, requires, requires_accommodation, etc.
        """
        constraint_relations = {
            "must_not_influence",
            "requires",
            "requires_accommodation",
            "prohibits",
            "restricts",
        }
        results = []

        if self._graph is not None:
            for source, _, data in self._graph.in_edges(target_id, data=True):
                if data.get("relation") in constraint_relations:
                    results.append(
                        {
                            "source": source,
                            "relation": data["relation"],
                            **{k: v for k, v in data.items() if k != "relation"},
                        }
                    )
        else:
            for edge in self._edges:
                if edge.get("target") == target_id and edge.get("relation") in constraint_relations:
                    results.append(edge)

        return results

    def get_applicable_laws(self, domain: str) -> list[str]:
        """Get legal frameworks that apply to a domain."""
        # Map domain names to graph node IDs
        domain_map = {
            "healthcare": "medical_decision",
            "finance": "credit_decision",
            "hiring": "hiring_decision",
            "disaster": "triage_decision",
        }
        target = domain_map.get(domain, domain)

        laws = []
        if self._graph is not None:
            for source, _, data in self._graph.in_edges(target, data=True):
                if data.get("relation") == "applies_to":
                    laws.append(source)
        else:
            for edge in self._edges:
                if edge.get("target") == target and edge.get("relation") == "applies_to":
                    laws.append(edge["source"])

        return laws

    def get_protected_attributes(self, domain: str) -> list[str]:
        """Get protected attributes that must not influence a domain."""
        domain_map = {
            "healthcare": "medical_decision",
            "finance": "credit_decision",
            "hiring": "hiring_decision",
        }
        target = domain_map.get(domain, domain)

        attrs = []
        if self._graph is not None:
            for source, _, data in self._graph.in_edges(target, data=True):
                if data.get("relation") == "must_not_influence":
                    attrs.append(source)
        else:
            for edge in self._edges:
                if edge.get("target") == target and edge.get("relation") == "must_not_influence":
                    attrs.append(edge["source"])

        return attrs

    def get_requirements(self, entity_id: str) -> list[str]:
        """Get what an entity requires (outgoing 'requires' edges)."""
        reqs = []
        if self._graph is not None:
            for _, target, data in self._graph.out_edges(entity_id, data=True):
                if data.get("relation") == "requires":
                    reqs.append(target)
        else:
            for edge in self._edges:
                if edge.get("source") == entity_id and edge.get("relation") == "requires":
                    reqs.append(edge["target"])
        return reqs

    def get_stakeholders(self, domain: str) -> list[str]:
        """Get stakeholders affected by a domain."""
        domain_map = {
            "healthcare": "medical_decision",
            "finance": "credit_decision",
            "hiring": "hiring_decision",
            "disaster": "triage_decision",
        }
        target = domain_map.get(domain, domain)

        stakeholders = []
        if self._graph is not None:
            for source, _, data in self._graph.in_edges(target, data=True):
                if data.get("relation") == "affected_by":
                    stakeholders.append(source)
        else:
            for edge in self._edges:
                if edge.get("target") == target and edge.get("relation") == "affected_by":
                    stakeholders.append(edge["source"])
        return stakeholders

    def check_constraint(self, attribute: str, domain: str) -> bool:
        """Check if using an attribute in a domain violates a constraint.

        Returns True if constraint is violated (the attribute MUST NOT
        influence the domain).
        """
        domain_map = {
            "healthcare": "medical_decision",
            "finance": "credit_decision",
            "hiring": "hiring_decision",
        }
        target = domain_map.get(domain, domain)

        if self._graph is not None:
            if self._graph.has_edge(attribute, target):
                data = self._graph.get_edge_data(attribute, target)
                return data.get("relation") == "must_not_influence"
        else:
            for edge in self._edges:
                if (
                    edge.get("source") == attribute
                    and edge.get("target") == target
                    and edge.get("relation") == "must_not_influence"
                ):
                    return True
        return False

    def shortest_path(self, source: str, target: str) -> list[str] | None:
        """Find shortest path between two nodes (if NetworkX available)."""
        if self._graph is not None:
            try:
                return nx.shortest_path(self._graph, source, target)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph for JSON export."""
        nodes = [{"id": nid, **meta} for nid, meta in self._nodes.items()]
        return {"nodes": nodes, "edges": self._edges}

    def summary(self) -> str:
        """Quick summary for debugging."""
        types = {}
        for meta in self._nodes.values():
            t = meta.get("type", "unknown")
            types[t] = types.get(t, 0) + 1

        lines = [f"KnowledgeGraph: {self.node_count} nodes, {self.edge_count} edges"]
        for t, count in sorted(types.items()):
            lines.append(f"  {t}: {count}")
        return "\n".join(lines)

    # ---------- Convenience loaders ----------

    _DOMAIN_FILE_MAP: dict[str, str] = {
        "healthcare": "healthcare_kg.json",
        "finance": "finance_kg.json",
        "hiring": "hiring_kg.json",
        "disaster": "disaster_kg.json",
        "ethical": "ethical_ontology.json",
    }

    def load_domain_data(self, domain: str, data_dir: str | Path | None = None) -> int:
        """Load knowledge for a specific domain from its JSON file.

        Parameters
        ----------
        domain : str
            One of 'healthcare', 'finance', 'hiring', 'disaster', 'ethical'.
        data_dir : str or Path, optional
            Directory containing the JSON files.
            Defaults to ``<project_root>/data/knowledge/``.

        Returns
        -------
        int
            Number of nodes loaded.

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
            data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge"
        else:
            data_dir = Path(data_dir)

        filepath = data_dir / filename
        if not filepath.exists():
            logger.warning("Domain file not found: %s", filepath)
            return 0

        before = self.node_count
        self.load_from_json(filepath)
        loaded = self.node_count - before
        logger.info(
            "Loaded %d new nodes for domain '%s' from %s",
            loaded,
            domain,
            filepath.name,
        )
        return loaded

    @classmethod
    def auto_load(cls, data_dir: str | Path | None = None) -> KnowledgeGraph:
        """Create a KnowledgeGraph pre-populated with all domain data.

        This is the recommended factory method for production use.
        It loads seed data *and* every domain JSON in
        ``data/knowledge/``, giving you the richest possible graph
        out of the box.

        Parameters
        ----------
        data_dir : str or Path, optional
            Override the default ``data/knowledge/`` directory.

        Returns
        -------
        KnowledgeGraph
            Fully loaded instance.

        Example
        -------
        >>> kg = KnowledgeGraph.auto_load()
        >>> kg.node_count > 50
        True
        """
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge"

        # Constructor already loads seed data + directory
        kg = cls(data_dir=data_dir)
        logger.info("auto_load complete — %d nodes, %d edges", kg.node_count, kg.edge_count)
        return kg
