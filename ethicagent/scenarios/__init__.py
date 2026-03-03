"""Scenarios Package — Ethical Scenario Test Suites.

Provides 425 hand-crafted scenario cases across four high-stakes
domains: healthcare triage, hiring & employment, loan & insurance,
and disaster response.  Each scenario class inherits from
:class:`BaseScenario` and exposes its cases via ``get_cases()``.

Quick-start
-----------
>>> from ethicagent.scenarios import HealthcareTriageScenario
>>> scenario = HealthcareTriageScenario()
>>> cases = scenario.get_cases()
>>> len(cases)
110
"""

from __future__ import annotations

from typing import Dict, List, Type

from ethicagent.scenarios.base_scenario import BaseScenario, ScenarioCase, ScenarioResult
from ethicagent.scenarios.disaster_response import DisasterResponseScenario
from ethicagent.scenarios.healthcare_triage import HealthcareTriageScenario
from ethicagent.scenarios.hiring_decision import HiringDecisionScenario
from ethicagent.scenarios.loan_approval import LoanApprovalScenario

__all__ = [
    "BaseScenario",
    "ScenarioCase",
    "ScenarioResult",
    "HealthcareTriageScenario",
    "LoanApprovalScenario",
    "HiringDecisionScenario",
    "DisasterResponseScenario",
    "SCENARIO_REGISTRY",
    "get_all_cases",
]

# Registry mapping domain names to scenario classes — handy for
# CLI tools, benchmarks, or anything that needs to iterate domains.
SCENARIO_REGISTRY: dict[str, type[BaseScenario]] = {
    "healthcare": HealthcareTriageScenario,
    "hiring": HiringDecisionScenario,
    "finance": LoanApprovalScenario,
    "disaster": DisasterResponseScenario,
}


def get_all_cases() -> list[ScenarioCase]:
    """Load and return every scenario case across all domains."""
    cases: list[ScenarioCase] = []
    for cls in SCENARIO_REGISTRY.values():
        cases.extend(cls().get_cases())
    return cases
