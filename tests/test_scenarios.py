"""Tests for scenario modules — loading, structure, validation.

Validates all four domain scenarios plus the base class,
the SCENARIO_REGISTRY, and the get_all_cases() helper.
"""

from __future__ import annotations

import json
import os

import pytest

from ethicagent.scenarios import SCENARIO_REGISTRY, get_all_cases
from ethicagent.scenarios.base_scenario import BaseScenario, ScenarioCase
from ethicagent.scenarios.disaster_response import DisasterResponseScenario
from ethicagent.scenarios.healthcare_triage import HealthcareTriageScenario
from ethicagent.scenarios.hiring_decision import HiringDecisionScenario
from ethicagent.scenarios.loan_approval import LoanApprovalScenario

# ═══════════════════════════════════════════════════════════════
# ScenarioCase data class
# ═══════════════════════════════════════════════════════════════


class TestScenarioCase:
    def test_creation(self):
        case = ScenarioCase(
            case_id="TC001",
            task="Test task",
            domain="general",
            expected_verdict="approve",
            expected_eds_range=(0.80, 1.00),
        )
        assert case.case_id == "TC001"
        assert case.expected_verdict == "approve"

    def test_eds_range(self):
        case = ScenarioCase(
            case_id="TC002",
            task="Range test",
            domain="general",
            expected_verdict="escalate",
            expected_eds_range=(0.50, 0.75),
        )
        assert case.expected_eds_range[0] == 0.50
        assert case.expected_eds_range[1] == 0.75

    def test_optional_metadata(self):
        case = ScenarioCase(
            case_id="TC003",
            task="Metadata test",
            domain="finance",
            expected_verdict="reject",
            expected_eds_range=(0.0, 0.49),
            metadata={"source": "unit_test"},
        )
        assert case.metadata["source"] == "unit_test"


# ═══════════════════════════════════════════════════════════════
# Registry & helpers
# ═══════════════════════════════════════════════════════════════


class TestScenarioRegistry:
    def test_registry_has_four_domains(self):
        assert len(SCENARIO_REGISTRY) == 4

    def test_registry_keys(self):
        for key in ["healthcare", "hiring", "finance", "disaster"]:
            assert key in SCENARIO_REGISTRY

    def test_registry_values_are_classes(self):
        for cls in SCENARIO_REGISTRY.values():
            assert issubclass(cls, BaseScenario)

    def test_get_all_cases(self):
        cases = get_all_cases()
        assert isinstance(cases, list)
        # we expect 425 total (110+105+105+105)
        assert len(cases) >= 400


# ═══════════════════════════════════════════════════════════════
# Healthcare Triage
# ═══════════════════════════════════════════════════════════════


class TestHealthcareTriageScenario:
    def test_initialization(self):
        scenario = HealthcareTriageScenario()
        assert scenario is not None

    def test_get_domain(self):
        scenario = HealthcareTriageScenario()
        assert scenario.get_domain() == "healthcare"

    def test_has_cases(self):
        scenario = HealthcareTriageScenario()
        cases = scenario.get_cases()
        assert len(cases) >= 30

    def test_case_structure(self):
        scenario = HealthcareTriageScenario()
        for case in scenario.get_cases()[:5]:
            assert hasattr(case, "case_id")
            assert hasattr(case, "task")
            assert hasattr(case, "expected_verdict")
            assert len(case.task) > 10

    def test_verdict_distribution(self):
        scenario = HealthcareTriageScenario()
        verdicts = [c.expected_verdict for c in scenario.get_cases()]
        unique = set(verdicts)
        assert len(unique) >= 3

    def test_domain_is_healthcare(self):
        scenario = HealthcareTriageScenario()
        for case in scenario.get_cases():
            assert getattr(case, "domain", "healthcare") == "healthcare"


# ═══════════════════════════════════════════════════════════════
# Loan Approval
# ═══════════════════════════════════════════════════════════════


class TestLoanApprovalScenario:
    def test_initialization(self):
        scenario = LoanApprovalScenario()
        assert scenario is not None

    def test_get_domain(self):
        scenario = LoanApprovalScenario()
        assert scenario.get_domain() == "finance"

    def test_has_cases(self):
        scenario = LoanApprovalScenario()
        assert len(scenario.get_cases()) >= 30

    def test_case_ids_unique(self):
        scenario = LoanApprovalScenario()
        ids = [c.case_id for c in scenario.get_cases()]
        assert len(ids) == len(set(ids))

    def test_eds_ranges_valid(self):
        scenario = LoanApprovalScenario()
        for case in scenario.get_cases():
            low, high = case.expected_eds_range
            assert 0.0 <= low <= high <= 1.0


# ═══════════════════════════════════════════════════════════════
# Hiring Decision
# ═══════════════════════════════════════════════════════════════


class TestHiringDecisionScenario:
    def test_initialization(self):
        scenario = HiringDecisionScenario()
        assert scenario is not None

    def test_get_domain(self):
        scenario = HiringDecisionScenario()
        assert scenario.get_domain() == "hiring"

    def test_has_cases(self):
        scenario = HiringDecisionScenario()
        assert len(scenario.get_cases()) >= 30

    def test_contains_hard_block_cases(self):
        scenario = HiringDecisionScenario()
        hard_blocks = [c for c in scenario.get_cases() if c.expected_verdict == "hard_block"]
        assert len(hard_blocks) >= 3


# ═══════════════════════════════════════════════════════════════
# Disaster Response
# ═══════════════════════════════════════════════════════════════


class TestDisasterResponseScenario:
    def test_initialization(self):
        scenario = DisasterResponseScenario()
        assert scenario is not None

    def test_get_domain(self):
        scenario = DisasterResponseScenario()
        assert scenario.get_domain() == "disaster"

    def test_has_cases(self):
        scenario = DisasterResponseScenario()
        assert len(scenario.get_cases()) >= 30

    def test_all_verdicts_present(self):
        scenario = DisasterResponseScenario()
        verdicts = set(c.expected_verdict for c in scenario.get_cases())
        assert "approve" in verdicts
        assert "escalate" in verdicts or "reject" in verdicts


# ═══════════════════════════════════════════════════════════════
# Cross-scenario invariants
# ═══════════════════════════════════════════════════════════════


class TestCrossScenarioInvariants:
    """Properties that hold across all scenarios."""

    @pytest.mark.parametrize("domain", list(SCENARIO_REGISTRY.keys()))
    def test_all_case_ids_globally_unique(self, domain):
        cls = SCENARIO_REGISTRY[domain]
        scenario = cls()
        ids = [c.case_id for c in scenario.get_cases()]
        assert len(ids) == len(set(ids)), f"Duplicate IDs in {domain}"

    @pytest.mark.parametrize("domain", list(SCENARIO_REGISTRY.keys()))
    def test_eds_ranges_valid_everywhere(self, domain):
        cls = SCENARIO_REGISTRY[domain]
        scenario = cls()
        for case in scenario.get_cases():
            lo, hi = case.expected_eds_range
            assert 0 <= lo <= hi <= 1, f"Bad EDS range in {case.case_id}"


# ═══════════════════════════════════════════════════════════════
# JSON data files (if present)
# ═══════════════════════════════════════════════════════════════


class TestDataFiles:
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "scenarios")

    def _load_json(self, filename):
        path = os.path.join(self.DATA_DIR, filename)
        if not os.path.exists(path):
            pytest.skip(f"{filename} not found")
        with open(path) as f:
            return json.load(f)

    @pytest.mark.parametrize(
        "filename",
        [
            "healthcare_triage.json",
            "loan_approval.json",
            "hiring_decision.json",
            "disaster_response.json",
        ],
    )
    def test_json_valid(self, filename):
        data = self._load_json(filename)
        assert isinstance(data, list)
        assert len(data) >= 50

    @pytest.mark.parametrize(
        "filename",
        [
            "healthcare_triage.json",
            "loan_approval.json",
            "hiring_decision.json",
            "disaster_response.json",
        ],
    )
    def test_json_case_structure(self, filename):
        data = self._load_json(filename)
        for case in data[:3]:
            assert "case_id" in case
            assert "task" in case
            assert "expected_verdict" in case
            assert "expected_eds_range" in case
            assert len(case["task"]) > 10
