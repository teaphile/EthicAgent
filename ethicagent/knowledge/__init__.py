"""Knowledge layer — graphs, memory, precedents, ontologies."""

from ethicagent.knowledge.knowledge_graph import KnowledgeGraph
from ethicagent.knowledge.memory_store import DecisionRecord, MemoryStore
from ethicagent.knowledge.ontology_loader import (
    DomainOntology,
    OntologyNode,
    get_ontology,
    load_ontology_from_json,
)
from ethicagent.knowledge.precedent_store import Precedent, PrecedentStore

__all__ = [
    "KnowledgeGraph",
    "MemoryStore",
    "DecisionRecord",
    "PrecedentStore",
    "Precedent",
    "DomainOntology",
    "OntologyNode",
    "get_ontology",
    "load_ontology_from_json",
]
