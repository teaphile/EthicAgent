"""Tests for ethics modules — EDS formula, philosophies, conflict resolution."""

from __future__ import annotations

from ethicagent.ethics.conflict_resolver import ConflictResolver
from ethicagent.ethics.consequentialist import ConsequentialistEvaluator
from ethicagent.ethics.contextual_ethics import ContextualEthicsEvaluator
from ethicagent.ethics.deontological import DeontologicalEvaluator
from ethicagent.ethics.ethical_score import (
    EthicalDecision,
    EthicalVerdict,
    PhilosophyResult,
)
from ethicagent.ethics.virtue_ethics import VirtueEthicsEvaluator

# ═══════════════════════════════════════════════════════════════
# Ethical Score & Verdict
# ═══════════════════════════════════════════════════════════════


class TestEthicalVerdict:
    def test_verdict_values(self):
        assert EthicalVerdict.AUTO_APPROVE.value == "auto_approve"
        assert EthicalVerdict.ESCALATE.value == "escalate"
        assert EthicalVerdict.REJECT.value == "reject"
        assert EthicalVerdict.HARD_BLOCK.value == "hard_block"


class TestPhilosophyResult:
    def test_creation(self):
        result = PhilosophyResult(
            philosophy="deontological",
            score=0.85,
            reasoning="All rules satisfied",
            details={"rules_checked": 10, "rules_passed": 10},
        )
        assert result.score == 0.85
        assert result.philosophy == "deontological"


class TestEthicalDecision:
    def test_approve_verdict(self):
        decision = EthicalDecision(
            eds_score=0.85,
            verdict=EthicalVerdict.AUTO_APPROVE,
            philosophy_results=[],
            reasoning="High ethical score",
        )
        assert decision.verdict == EthicalVerdict.AUTO_APPROVE

    def test_hard_block_verdict(self):
        decision = EthicalDecision(
            eds_score=0.0,
            verdict=EthicalVerdict.HARD_BLOCK,
            philosophy_results=[],
            reasoning="Deontological violation",
        )
        assert decision.verdict == EthicalVerdict.HARD_BLOCK

    def test_to_dict(self):
        decision = EthicalDecision(
            eds_score=0.62,
            verdict=EthicalVerdict.ESCALATE,
            philosophy_results=[],
            reasoning="Moderate score",
        )
        d = decision.to_dict()
        assert isinstance(d, dict)
        assert d["eds_score"] == 0.62

    def test_summary(self):
        decision = EthicalDecision(
            eds_score=0.85,
            verdict=EthicalVerdict.AUTO_APPROVE,
            philosophy_results=[],
            reasoning="Test",
        )
        summary = decision.summary()
        assert isinstance(summary, str)
        assert "0.85" in summary


# ═══════════════════════════════════════════════════════════════
# Deontological Evaluator
# ═══════════════════════════════════════════════════════════════


class TestDeontologicalEvaluator:
    def test_initialization(self):
        evaluator = DeontologicalEvaluator()
        assert evaluator is not None

    def test_evaluate_safe_action(self, sample_context):
        evaluator = DeontologicalEvaluator()
        result = evaluator.evaluate(
            "Approve standard medical checkup for a healthy adult",
            sample_context,
        )
        assert isinstance(result, PhilosophyResult)
        assert result.philosophy == "deontological"
        assert 0.0 <= result.score <= 1.0

    def test_evaluate_harmful_action(self, sample_context):
        evaluator = DeontologicalEvaluator()
        result = evaluator.evaluate(
            "Deny emergency treatment to a dying patient based on their inability to pay",
            sample_context,
        )
        assert isinstance(result, PhilosophyResult)
        # Should detect rule violation
        assert result.score < 0.8

    def test_hard_rule_violation(self, sample_context):
        evaluator = DeontologicalEvaluator()
        result = evaluator.evaluate(
            "Discriminate against the applicant based on their race and deny the loan",
            sample_context,
        )
        # Score should be very low or zero for discrimination
        assert result.score <= 0.3


# ═══════════════════════════════════════════════════════════════
# Consequentialist Evaluator
# ═══════════════════════════════════════════════════════════════


class TestConsequentialistEvaluator:
    def test_initialization(self):
        evaluator = ConsequentialistEvaluator()
        assert evaluator is not None

    def test_evaluate_beneficial_action(self, sample_context):
        evaluator = ConsequentialistEvaluator()
        result = evaluator.evaluate(
            "Provide free vaccinations to all children in the community",
            sample_context,
        )
        assert isinstance(result, PhilosophyResult)
        assert result.philosophy == "consequentialist"
        assert result.score > 0.5

    def test_evaluate_harmful_action(self, sample_context):
        evaluator = ConsequentialistEvaluator()
        result = evaluator.evaluate(
            "Release toxic chemicals into the river affecting thousands of residents downstream",
            sample_context,
        )
        assert result.score < 0.5

    def test_score_range(self, sample_context):
        evaluator = ConsequentialistEvaluator()
        result = evaluator.evaluate("Neutral action", sample_context)
        assert 0.0 <= result.score <= 1.0


# ═══════════════════════════════════════════════════════════════
# Virtue Ethics Evaluator
# ═══════════════════════════════════════════════════════════════


class TestVirtueEthicsEvaluator:
    def test_initialization(self):
        evaluator = VirtueEthicsEvaluator()
        assert evaluator is not None

    def test_evaluate_fair_action(self, sample_context):
        evaluator = VirtueEthicsEvaluator()
        result = evaluator.evaluate(
            "Treat all applicants equally regardless of background",
            sample_context,
        )
        assert isinstance(result, PhilosophyResult)
        assert result.philosophy == "virtue_ethics"
        assert result.score >= 0.5

    def test_evaluate_unfair_action(self, sample_context):
        evaluator = VirtueEthicsEvaluator()
        result = evaluator.evaluate(
            "Give preferential treatment to wealthy applicants and reject poor ones",
            sample_context,
        )
        assert result.score < 0.7


# ═══════════════════════════════════════════════════════════════
# Contextual Ethics Evaluator
# ═══════════════════════════════════════════════════════════════


class TestContextualEthicsEvaluator:
    def test_initialization(self):
        evaluator = ContextualEthicsEvaluator()
        assert evaluator is not None

    def test_evaluate_appropriate_action(self, sample_context):
        evaluator = ContextualEthicsEvaluator()
        result = evaluator.evaluate(
            "Follow standard lending procedures for a routine loan application",
            sample_context,
        )
        assert isinstance(result, PhilosophyResult)
        assert result.philosophy == "contextual"
        assert result.score >= 0.5

    def test_score_range(self, sample_context):
        evaluator = ContextualEthicsEvaluator()
        result = evaluator.evaluate("Some action", sample_context)
        assert 0.0 <= result.score <= 1.0


# ═══════════════════════════════════════════════════════════════
# Conflict Resolver
# ═══════════════════════════════════════════════════════════════


class TestConflictResolver:
    def test_initialization(self):
        resolver = ConflictResolver()
        assert resolver is not None

    def test_no_conflict(self):
        resolver = ConflictResolver()
        scores = {
            "deontological": 0.80,
            "consequentialist": 0.78,
            "virtue_ethics": 0.82,
            "contextual": 0.79,
        }
        conflicts = resolver.detect_conflicts(scores)
        # Scores are close together — no severe conflict
        severe = [c for c in conflicts if c.get("severity") == "severe"]
        assert len(severe) == 0

    def test_severe_conflict(self):
        resolver = ConflictResolver()
        scores = {
            "deontological": 0.95,
            "consequentialist": 0.20,
            "virtue_ethics": 0.80,
            "contextual": 0.70,
        }
        conflicts = resolver.detect_conflicts(scores)
        # Large gap between deontological and consequentialist
        assert len(conflicts) > 0

    def test_resolve(self):
        resolver = ConflictResolver()
        scores = {
            "deontological": 0.90,
            "consequentialist": 0.30,
            "virtue_ethics": 0.70,
            "contextual": 0.60,
        }
        result = resolver.resolve(scores)
        assert isinstance(result, dict)
        # Should contain resolution information
        assert "resolved_scores" in result or "resolution" in result or "conflicts" in result
