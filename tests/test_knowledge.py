"""Tests for knowledge modules — knowledge graph, memory, precedents."""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from ethicagent.knowledge.knowledge_graph import KnowledgeGraph
from ethicagent.knowledge.precedent_store import PrecedentStore


class TestKnowledgeGraph:
    def test_initialization(self):
        kg = KnowledgeGraph()
        assert kg is not None

    def test_add_node(self):
        kg = KnowledgeGraph()
        kg.add_node("test_concept", node_type="concept", label="Test")
        assert kg.graph.has_node("test_concept")

    def test_add_edge(self):
        kg = KnowledgeGraph()
        kg.add_node("a", node_type="concept")
        kg.add_node("b", node_type="concept")
        kg.add_edge("a", "b", relationship="related_to")
        assert kg.graph.has_edge("a", "b")

    def test_query_neighbors(self):
        kg = KnowledgeGraph()
        kg.add_node("center", node_type="concept")
        kg.add_node("neighbor1", node_type="concept")
        kg.add_node("neighbor2", node_type="concept")
        kg.add_edge("center", "neighbor1", relationship="related_to")
        kg.add_edge("center", "neighbor2", relationship="related_to")
        neighbors = kg.get_neighbors("center")
        assert len(neighbors) >= 2

    def test_domain_initialization(self):
        kg = KnowledgeGraph()
        kg.initialize_domain("healthcare")
        # Should have nodes after initialization
        assert len(kg.graph.nodes) > 0

    def test_query_by_domain(self):
        kg = KnowledgeGraph()
        kg.initialize_domain("finance")
        results = kg.query_ethical_context("loan", domain="finance")
        assert isinstance(results, (dict, list))


class TestPrecedentStore:
    def test_initialization(self):
        store = PrecedentStore()
        assert store is not None

    def test_add_precedent(self):
        store = PrecedentStore()
        store.add_precedent(
            case_id="P001",
            task="Approve standard loan",
            domain="finance",
            verdict="approve",
            eds_score=0.85,
            reasoning="Standard approval criteria met",
        )
        assert len(store.precedents) >= 1

    def test_search_similar(self):
        store = PrecedentStore()
        store.add_precedent(
            case_id="P001",
            task="Approve mortgage for qualified buyer",
            domain="finance",
            verdict="approve",
            eds_score=0.85,
            reasoning="Good credit, stable income",
        )
        results = store.search_similar("mortgage loan approval", domain="finance")
        assert isinstance(results, list)

    def test_filter_by_verdict(self):
        store = PrecedentStore()
        store.add_precedent(
            case_id="P001", task="Approve loan", domain="finance",
            verdict="approve", eds_score=0.85, reasoning="ok",
        )
        store.add_precedent(
            case_id="P002", task="Deny loan", domain="finance",
            verdict="reject", eds_score=0.30, reasoning="violation",
        )
        approved = store.filter_by_verdict("approve")
        assert all(p.get("verdict") == "approve" for p in approved)

    def test_persistence(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            filepath = f.name

        try:
            store = PrecedentStore(filepath=filepath)
            store.add_precedent(
                case_id="PERSIST001",
                task="Test persistence",
                domain="general",
                verdict="approve",
                eds_score=0.90,
                reasoning="Test",
            )
            store.save()

            # Reload
            store2 = PrecedentStore(filepath=filepath)
            store2.load()
            assert len(store2.precedents) >= 1
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
