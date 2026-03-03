"""Tests for agent modules — context, reasoning, fusion, ethics, execution."""

from __future__ import annotations

import pytest
from ethicagent.agents.context_agent import ContextAgent
from ethicagent.agents.fusion_agent import FusionAgent
from ethicagent.agents.ethical_reasoner import EthicalReasonerAgent
from ethicagent.agents.action_executor import ActionExecutor
from ethicagent.agents.reflection_agent import ReflectionAgent


class TestContextAgent:
    def test_initialization(self):
        agent = ContextAgent()
        assert agent is not None

    def test_classify_domain_healthcare(self):
        agent = ContextAgent()
        domain = agent.classify_domain(
            "Patient needs emergency surgery for cardiac arrest"
        )
        assert domain == "healthcare"

    def test_classify_domain_finance(self):
        agent = ContextAgent()
        domain = agent.classify_domain(
            "Approve a mortgage loan for a first-time homebuyer"
        )
        assert domain == "finance"

    def test_classify_domain_hiring(self):
        agent = ContextAgent()
        domain = agent.classify_domain(
            "Screen candidates for a software engineering position"
        )
        assert domain == "hiring"

    def test_classify_domain_disaster(self):
        agent = ContextAgent()
        domain = agent.classify_domain(
            "Evacuate residents before the hurricane makes landfall"
        )
        assert domain == "disaster"

    def test_extract_entities(self):
        agent = ContextAgent()
        entities = agent.extract_entities(
            "The hospital needs to allocate ventilators to ICU patients"
        )
        assert isinstance(entities, list)
        assert len(entities) > 0

    def test_determine_urgency(self):
        agent = ContextAgent()
        urgency = agent.determine_urgency(
            "Emergency evacuation needed immediately"
        )
        assert urgency in ["low", "medium", "high", "critical"]

    def test_full_analysis(self):
        agent = ContextAgent()
        result = agent.analyze(
            "Approve a loan for an applicant with good credit"
        )
        assert isinstance(result, dict)
        assert "domain" in result
        assert "entities" in result
        assert "urgency" in result


class TestFusionAgent:
    def test_initialization(self):
        agent = FusionAgent()
        assert agent is not None

    def test_fuse_agreement(self):
        agent = FusionAgent()
        neural = {"score": 0.85, "reasoning": "LLM says approve"}
        symbolic = {"score": 0.90, "reasoning": "Rules satisfied", "hard_block": False}
        result = agent.fuse(neural, symbolic)
        assert isinstance(result, dict)
        assert "score" in result or "fused_score" in result

    def test_fuse_disagreement(self):
        agent = FusionAgent()
        neural = {"score": 0.80, "reasoning": "LLM says OK"}
        symbolic = {"score": 0.20, "reasoning": "Rule violation", "hard_block": True}
        result = agent.fuse(neural, symbolic)
        # Symbolic should dominate when hard_block
        assert isinstance(result, dict)

    def test_safety_first(self):
        agent = FusionAgent()
        neural = {"score": 0.90, "reasoning": "Looks fine"}
        symbolic = {"score": 0.0, "reasoning": "Hard rule violated", "hard_block": True}
        result = agent.fuse(neural, symbolic)
        # Should respect the hard block
        fused = result.get("score", result.get("fused_score", 1.0))
        assert fused < 0.5


class TestEthicalReasonerAgent:
    def test_initialization(self):
        agent = EthicalReasonerAgent()
        assert agent is not None

    def test_compute_eds(self, sample_philosophy_scores, sample_weights):
        agent = EthicalReasonerAgent()
        eds = agent.compute_eds(sample_philosophy_scores, sample_weights)
        assert 0.0 <= eds <= 1.0

    def test_eds_formula_correctness(self):
        agent = EthicalReasonerAgent()
        scores = {
            "deontological": 1.0,
            "consequentialist": 1.0,
            "virtue_ethics": 1.0,
            "contextual": 1.0,
        }
        weights = {
            "deontological": 0.25,
            "consequentialist": 0.25,
            "virtue_ethics": 0.25,
            "contextual": 0.25,
        }
        eds = agent.compute_eds(scores, weights)
        assert abs(eds - 1.0) < 0.001

    def test_eds_zero_deontological_hard_block(self):
        agent = EthicalReasonerAgent()
        scores = {
            "deontological": 0.0,
            "consequentialist": 0.80,
            "virtue_ethics": 0.70,
            "contextual": 0.90,
        }
        weights = {
            "deontological": 0.25,
            "consequentialist": 0.25,
            "virtue_ethics": 0.25,
            "contextual": 0.25,
        }
        verdict = agent.determine_verdict(scores, weights)
        assert verdict == "hard_block"

    def test_verdict_approve(self):
        agent = EthicalReasonerAgent()
        scores = {
            "deontological": 0.90,
            "consequentialist": 0.85,
            "virtue_ethics": 0.80,
            "contextual": 0.85,
        }
        weights = {
            "deontological": 0.25,
            "consequentialist": 0.25,
            "virtue_ethics": 0.25,
            "contextual": 0.25,
        }
        verdict = agent.determine_verdict(scores, weights)
        assert verdict in ["approve", "auto_approve"]

    def test_verdict_escalate(self):
        agent = EthicalReasonerAgent()
        scores = {
            "deontological": 0.70,
            "consequentialist": 0.60,
            "virtue_ethics": 0.55,
            "contextual": 0.65,
        }
        weights = {
            "deontological": 0.25,
            "consequentialist": 0.25,
            "virtue_ethics": 0.25,
            "contextual": 0.25,
        }
        verdict = agent.determine_verdict(scores, weights)
        assert verdict == "escalate"

    def test_verdict_reject(self):
        agent = EthicalReasonerAgent()
        scores = {
            "deontological": 0.30,
            "consequentialist": 0.20,
            "virtue_ethics": 0.25,
            "contextual": 0.35,
        }
        weights = {
            "deontological": 0.25,
            "consequentialist": 0.25,
            "virtue_ethics": 0.25,
            "contextual": 0.25,
        }
        verdict = agent.determine_verdict(scores, weights)
        assert verdict == "reject"


class TestActionExecutor:
    def test_initialization(self):
        executor = ActionExecutor()
        assert executor is not None

    def test_execute_approve(self):
        executor = ActionExecutor()
        result = executor.execute(
            verdict="approve",
            eds_score=0.85,
            task="Approve standard loan",
            reasoning="All criteria met",
        )
        assert isinstance(result, dict)
        assert result.get("action") == "approve" or result.get("verdict") == "approve"

    def test_execute_reject(self):
        executor = ActionExecutor()
        result = executor.execute(
            verdict="reject",
            eds_score=0.30,
            task="Deny discriminatory practice",
            reasoning="Ethical violation",
        )
        assert isinstance(result, dict)

    def test_execute_escalate(self):
        executor = ActionExecutor()
        result = executor.execute(
            verdict="escalate",
            eds_score=0.60,
            task="Complex case",
            reasoning="Needs human review",
        )
        assert isinstance(result, dict)

    def test_execute_hard_block(self):
        executor = ActionExecutor()
        result = executor.execute(
            verdict="hard_block",
            eds_score=0.0,
            task="Illegal discrimination",
            reasoning="Hard rule violation",
        )
        assert isinstance(result, dict)


class TestReflectionAgent:
    def test_initialization(self):
        agent = ReflectionAgent()
        assert agent is not None

    def test_record_decision(self):
        agent = ReflectionAgent()
        agent.record_decision(
            task="Approve loan",
            domain="finance",
            verdict="approve",
            eds_score=0.85,
            philosophy_scores={
                "deontological": 0.90,
                "consequentialist": 0.80,
                "virtue_ethics": 0.85,
                "contextual": 0.82,
            },
        )
        assert len(agent.decision_history) >= 1

    def test_consistency_analysis(self):
        agent = ReflectionAgent()
        # Add similar decisions
        for i in range(5):
            agent.record_decision(
                task=f"Standard loan {i}",
                domain="finance",
                verdict="approve",
                eds_score=0.85 + i * 0.01,
                philosophy_scores={
                    "deontological": 0.90,
                    "consequentialist": 0.80,
                    "virtue_ethics": 0.85,
                    "contextual": 0.82,
                },
            )
        analysis = agent.analyze_consistency()
        assert isinstance(analysis, dict)

    def test_learning_summary(self):
        agent = ReflectionAgent()
        agent.record_decision(
            task="Test", domain="general", verdict="approve",
            eds_score=0.80,
            philosophy_scores={"deontological": 0.9, "consequentialist": 0.7,
                              "virtue_ethics": 0.8, "contextual": 0.8},
        )
        summary = agent.get_learning_summary()
        assert isinstance(summary, (str, dict))
