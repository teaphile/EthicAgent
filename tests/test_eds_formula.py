"""Tests for the EDS formula — core novelty of EthicAgent.

Thoroughly validates the Ethical Decision Score computation:
  EDS(a) = w₁·D(a) + w₂·C(a) + w₃·V(a) + w₄·Ctx(a)

Tests cover:
- Correct weighted sum computation
- Verdict thresholds (≥0.8 approve, 0.5-0.8 escalate, <0.5 reject)
- Hard block when D(a)=0
- Domain weight configurations
- Edge cases and boundary conditions
"""

from __future__ import annotations

from ethicagent.agents.ethical_reasoner import EthicalReasonerAgent


class TestEDSFormula:
    """Core EDS formula tests."""

    def setup_method(self):
        self.agent = EthicalReasonerAgent()

    def test_perfect_scores(self):
        """All perfect scores should produce EDS = 1.0."""
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
        eds = self.agent.compute_eds(scores, weights)
        assert abs(eds - 1.0) < 0.001

    def test_zero_scores(self):
        """All zero scores should produce EDS = 0.0."""
        scores = {
            "deontological": 0.0,
            "consequentialist": 0.0,
            "virtue_ethics": 0.0,
            "contextual": 0.0,
        }
        weights = {
            "deontological": 0.25,
            "consequentialist": 0.25,
            "virtue_ethics": 0.25,
            "contextual": 0.25,
        }
        eds = self.agent.compute_eds(scores, weights)
        assert abs(eds) < 0.001

    def test_weighted_sum_correctness(self):
        """Verify EDS = w1*D + w2*C + w3*V + w4*Ctx."""
        scores = {
            "deontological": 0.80,
            "consequentialist": 0.60,
            "virtue_ethics": 0.70,
            "contextual": 0.90,
        }
        weights = {
            "deontological": 0.35,
            "consequentialist": 0.25,
            "virtue_ethics": 0.20,
            "contextual": 0.20,
        }
        expected = 0.35 * 0.80 + 0.25 * 0.60 + 0.20 * 0.70 + 0.20 * 0.90
        eds = self.agent.compute_eds(scores, weights)
        assert abs(eds - expected) < 0.001

    def test_healthcare_weights(self):
        """Healthcare domain: w=(0.35, 0.25, 0.20, 0.20)."""
        scores = {
            "deontological": 0.90,
            "consequentialist": 0.70,
            "virtue_ethics": 0.80,
            "contextual": 0.75,
        }
        weights = {
            "deontological": 0.35,
            "consequentialist": 0.25,
            "virtue_ethics": 0.20,
            "contextual": 0.20,
        }
        expected = 0.35 * 0.90 + 0.25 * 0.70 + 0.20 * 0.80 + 0.20 * 0.75
        eds = self.agent.compute_eds(scores, weights)
        assert abs(eds - expected) < 0.001

    def test_finance_weights(self):
        """Finance domain: w=(0.20, 0.25, 0.35, 0.20)."""
        scores = {
            "deontological": 0.85,
            "consequentialist": 0.75,
            "virtue_ethics": 0.90,
            "contextual": 0.80,
        }
        weights = {
            "deontological": 0.20,
            "consequentialist": 0.25,
            "virtue_ethics": 0.35,
            "contextual": 0.20,
        }
        expected = 0.20 * 0.85 + 0.25 * 0.75 + 0.35 * 0.90 + 0.20 * 0.80
        eds = self.agent.compute_eds(scores, weights)
        assert abs(eds - expected) < 0.001

    def test_hiring_weights(self):
        """Hiring domain: w=(0.15, 0.20, 0.40, 0.25)."""
        scores = {
            "deontological": 0.80,
            "consequentialist": 0.65,
            "virtue_ethics": 0.85,
            "contextual": 0.70,
        }
        weights = {
            "deontological": 0.15,
            "consequentialist": 0.20,
            "virtue_ethics": 0.40,
            "contextual": 0.25,
        }
        expected = 0.15 * 0.80 + 0.20 * 0.65 + 0.40 * 0.85 + 0.25 * 0.70
        eds = self.agent.compute_eds(scores, weights)
        assert abs(eds - expected) < 0.001

    def test_disaster_weights(self):
        """Disaster domain: w=(0.20, 0.35, 0.15, 0.30)."""
        scores = {
            "deontological": 0.75,
            "consequentialist": 0.90,
            "virtue_ethics": 0.60,
            "contextual": 0.85,
        }
        weights = {
            "deontological": 0.20,
            "consequentialist": 0.35,
            "virtue_ethics": 0.15,
            "contextual": 0.30,
        }
        expected = 0.20 * 0.75 + 0.35 * 0.90 + 0.15 * 0.60 + 0.30 * 0.85
        eds = self.agent.compute_eds(scores, weights)
        assert abs(eds - expected) < 0.001


class TestVerdictThresholds:
    """Test verdict determination based on EDS thresholds."""

    def setup_method(self):
        self.agent = EthicalReasonerAgent()
        self.equal_weights = {
            "deontological": 0.25,
            "consequentialist": 0.25,
            "virtue_ethics": 0.25,
            "contextual": 0.25,
        }

    def test_approve_threshold(self):
        """EDS >= 0.80 → AUTO_APPROVE."""
        scores = {
            "deontological": 0.85,
            "consequentialist": 0.82,
            "virtue_ethics": 0.80,
            "contextual": 0.83,
        }
        verdict = self.agent.determine_verdict(scores, self.equal_weights)
        assert verdict in ["approve", "auto_approve"]

    def test_escalate_threshold(self):
        """0.50 <= EDS < 0.80 → ESCALATE."""
        scores = {
            "deontological": 0.65,
            "consequentialist": 0.60,
            "virtue_ethics": 0.55,
            "contextual": 0.70,
        }
        verdict = self.agent.determine_verdict(scores, self.equal_weights)
        assert verdict == "escalate"

    def test_reject_threshold(self):
        """EDS < 0.50 → REJECT."""
        scores = {
            "deontological": 0.30,
            "consequentialist": 0.25,
            "virtue_ethics": 0.40,
            "contextual": 0.35,
        }
        verdict = self.agent.determine_verdict(scores, self.equal_weights)
        assert verdict == "reject"

    def test_hard_block(self):
        """D(a) == 0 → HARD_BLOCK (regardless of other scores)."""
        scores = {
            "deontological": 0.0,
            "consequentialist": 0.90,
            "virtue_ethics": 0.85,
            "contextual": 0.95,
        }
        verdict = self.agent.determine_verdict(scores, self.equal_weights)
        assert verdict == "hard_block"

    def test_boundary_080(self):
        """EDS exactly at 0.80 should be AUTO_APPROVE."""
        scores = {
            "deontological": 0.80,
            "consequentialist": 0.80,
            "virtue_ethics": 0.80,
            "contextual": 0.80,
        }
        verdict = self.agent.determine_verdict(scores, self.equal_weights)
        assert verdict in ["approve", "auto_approve"]

    def test_boundary_050(self):
        """EDS exactly at 0.50 should be ESCALATE."""
        scores = {
            "deontological": 0.50,
            "consequentialist": 0.50,
            "virtue_ethics": 0.50,
            "contextual": 0.50,
        }
        verdict = self.agent.determine_verdict(scores, self.equal_weights)
        assert verdict == "escalate"

    def test_just_below_080(self):
        """EDS just below 0.80 should be ESCALATE."""
        # 0.25*0.79 + 0.25*0.79 + 0.25*0.79 + 0.25*0.79 = 0.79
        scores = {
            "deontological": 0.79,
            "consequentialist": 0.79,
            "virtue_ethics": 0.79,
            "contextual": 0.79,
        }
        verdict = self.agent.determine_verdict(scores, self.equal_weights)
        assert verdict == "escalate"

    def test_just_below_050(self):
        """EDS just below 0.50 should be REJECT."""
        scores = {
            "deontological": 0.49,
            "consequentialist": 0.49,
            "virtue_ethics": 0.49,
            "contextual": 0.49,
        }
        verdict = self.agent.determine_verdict(scores, self.equal_weights)
        assert verdict == "reject"


class TestEDSEdgeCases:
    """Edge cases for EDS formula."""

    def setup_method(self):
        self.agent = EthicalReasonerAgent()

    def test_single_high_score(self):
        """One philosophy dominates with high weight."""
        scores = {
            "deontological": 1.0,
            "consequentialist": 0.0,
            "virtue_ethics": 0.0,
            "contextual": 0.0,
        }
        weights = {
            "deontological": 0.70,
            "consequentialist": 0.10,
            "virtue_ethics": 0.10,
            "contextual": 0.10,
        }
        eds = self.agent.compute_eds(scores, weights)
        assert abs(eds - 0.70) < 0.001

    def test_weights_must_sum_to_one(self):
        """Weights should sum to 1.0 for correct EDS."""
        weights = {
            "deontological": 0.25,
            "consequentialist": 0.25,
            "virtue_ethics": 0.25,
            "contextual": 0.25,
        }
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001

    def test_all_domain_weights_sum_to_one(self):
        """All domain weight configs should sum to 1.0."""
        domain_configs = {
            "healthcare": (0.35, 0.25, 0.20, 0.20),
            "finance": (0.20, 0.25, 0.35, 0.20),
            "hiring": (0.15, 0.20, 0.40, 0.25),
            "disaster": (0.20, 0.35, 0.15, 0.30),
            "general": (0.25, 0.25, 0.25, 0.25),
        }
        for domain, weights in domain_configs.items():
            total = sum(weights)
            assert abs(total - 1.0) < 0.001, f"{domain} weights sum to {total}"
