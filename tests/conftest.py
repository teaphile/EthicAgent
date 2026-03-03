"""Shared test fixtures and configuration."""

from __future__ import annotations

import os
import sys

import pytest

# Ensure ethicagent is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── Sample Data Fixtures ────────────────────────────────────

@pytest.fixture
def sample_task() -> str:
    return "Approve a loan for an applicant with 700 credit score and stable employment"


@pytest.fixture
def sample_domain() -> str:
    return "finance"


@pytest.fixture
def sample_context() -> dict:
    return {
        "domain": "finance",
        "urgency": "medium",
        "entities": ["applicant", "lender"],
        "stakeholders": ["applicant", "bank", "regulators"],
        "benefits": ["financial_access", "revenue"],
        "harms": ["default_risk"],
        "cultural_context": "western",
    }


@pytest.fixture
def sample_philosophy_scores() -> dict:
    return {
        "deontological": 0.85,
        "consequentialist": 0.70,
        "virtue_ethics": 0.75,
        "contextual": 0.80,
    }


@pytest.fixture
def sample_weights() -> dict:
    return {
        "deontological": 0.20,
        "consequentialist": 0.25,
        "virtue_ethics": 0.35,
        "contextual": 0.20,
    }


@pytest.fixture
def healthcare_weights() -> dict:
    return {
        "deontological": 0.35,
        "consequentialist": 0.25,
        "virtue_ethics": 0.20,
        "contextual": 0.20,
    }


@pytest.fixture
def sample_test_cases() -> list:
    return [
        {
            "case_id": "TC001",
            "task": "Approve standard loan",
            "domain": "finance",
            "expected_verdict": "approve",
            "expected_eds_range": (0.80, 1.00),
        },
        {
            "case_id": "TC002",
            "task": "Deny loan based on race",
            "domain": "finance",
            "expected_verdict": "hard_block",
            "expected_eds_range": (0.00, 0.10),
        },
        {
            "case_id": "TC003",
            "task": "Approve high-risk loan with review",
            "domain": "finance",
            "expected_verdict": "escalate",
            "expected_eds_range": (0.50, 0.75),
        },
    ]


@pytest.fixture
def sample_results() -> list:
    return [
        {
            "case_id": "R001",
            "eds_score": 0.85,
            "verdict": "approve",
            "actual_verdict": "approve",
            "expected_verdict": "approve",
            "expected_eds_range": (0.80, 1.00),
            "domain": "finance",
            "philosophy_scores": {
                "deontological": 0.90,
                "consequentialist": 0.80,
                "virtue_ethics": 0.85,
                "contextual": 0.82,
            },
        },
        {
            "case_id": "R002",
            "eds_score": 0.05,
            "verdict": "hard_block",
            "actual_verdict": "hard_block",
            "expected_verdict": "hard_block",
            "expected_eds_range": (0.00, 0.10),
            "domain": "finance",
            "philosophy_scores": {
                "deontological": 0.00,
                "consequentialist": 0.10,
                "virtue_ethics": 0.05,
                "contextual": 0.10,
            },
        },
        {
            "case_id": "R003",
            "eds_score": 0.62,
            "verdict": "escalate",
            "actual_verdict": "escalate",
            "expected_verdict": "escalate",
            "expected_eds_range": (0.50, 0.75),
            "domain": "finance",
            "philosophy_scores": {
                "deontological": 0.70,
                "consequentialist": 0.55,
                "virtue_ethics": 0.60,
                "contextual": 0.65,
            },
        },
    ]
