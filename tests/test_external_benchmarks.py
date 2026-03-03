"""Tests for external benchmark adapters.

These tests use built-in sample data (no internet required).
Mark with @pytest.mark.external for CI that wants to skip them.
"""

from __future__ import annotations

import pytest

from ethicagent.benchmarks.external.ethics_adapter import EthicsDatasetAdapter
from ethicagent.benchmarks.external.bbq_adapter import BBQAdapter
from ethicagent.scenarios.base_scenario import ScenarioCase


# ═══════════════════════════════════════════════════════════════
# ETHICS Dataset Adapter
# ═══════════════════════════════════════════════════════════════

class TestEthicsAdapter:
    """Tests for the ETHICS dataset adapter (Hendrycks et al., 2021)."""

    def test_load_builtin(self):
        adapter = EthicsDatasetAdapter()
        adapter.load(use_builtin=True)
        assert adapter.case_count == 50

    def test_to_scenario_cases_returns_valid(self):
        adapter = EthicsDatasetAdapter()
        adapter.load(use_builtin=True)
        cases = adapter.to_scenario_cases()
        assert len(cases) == 50
        for case in cases:
            assert isinstance(case, ScenarioCase)

    def test_case_ids_are_unique(self):
        adapter = EthicsDatasetAdapter()
        adapter.load(use_builtin=True)
        cases = adapter.to_scenario_cases()
        ids = [c.case_id for c in cases]
        assert len(ids) == len(set(ids)), "Duplicate case IDs found"

    def test_expected_verdicts_valid(self):
        valid_verdicts = {"approve", "escalate", "reject", "hard_block"}
        adapter = EthicsDatasetAdapter()
        adapter.load(use_builtin=True)
        cases = adapter.to_scenario_cases()
        for case in cases:
            assert case.expected_verdict in valid_verdicts, (
                f"Invalid verdict '{case.expected_verdict}' in {case.case_id}"
            )

    def test_eds_ranges_valid(self):
        adapter = EthicsDatasetAdapter()
        adapter.load(use_builtin=True)
        cases = adapter.to_scenario_cases()
        for case in cases:
            lo, hi = case.expected_eds_range
            assert 0 <= lo <= hi <= 1, f"Bad EDS range in {case.case_id}"

    def test_subsets_covered(self):
        adapter = EthicsDatasetAdapter()
        adapter.load(use_builtin=True)
        assert set(adapter.subsets) == {
            "commonsense", "justice", "deontology",
            "utilitarianism", "virtue",
        }

    def test_summary_has_expected_keys(self):
        adapter = EthicsDatasetAdapter()
        adapter.load(use_builtin=True)
        summary = adapter.summary()
        assert summary["loaded"] is True
        assert summary["total_cases"] == 50
        assert "by_subset" in summary

    def test_load_not_called_raises(self):
        adapter = EthicsDatasetAdapter()
        with pytest.raises(RuntimeError, match="Call load"):
            adapter.to_scenario_cases()

    def test_domain_is_general(self):
        """ETHICS cases don't map to a specific domain — they should be 'general'."""
        adapter = EthicsDatasetAdapter()
        adapter.load(use_builtin=True)
        cases = adapter.to_scenario_cases()
        for case in cases:
            assert case.domain == "general"

    def test_tags_include_external(self):
        adapter = EthicsDatasetAdapter()
        adapter.load(use_builtin=True)
        cases = adapter.to_scenario_cases()
        for case in cases:
            assert "external" in case.tags
            assert "ethics_dataset" in case.tags


# ═══════════════════════════════════════════════════════════════
# BBQ Adapter
# ═══════════════════════════════════════════════════════════════

class TestBBQAdapter:
    """Tests for the BBQ bias benchmark adapter (Parrish et al., 2022)."""

    def test_load_builtin(self):
        adapter = BBQAdapter()
        adapter.load(use_builtin=True)
        assert adapter.case_count == 40

    def test_to_scenario_cases_returns_valid(self):
        adapter = BBQAdapter()
        adapter.load(use_builtin=True)
        cases = adapter.to_scenario_cases()
        assert len(cases) == 40
        for case in cases:
            assert isinstance(case, ScenarioCase)

    def test_case_ids_are_unique(self):
        adapter = BBQAdapter()
        adapter.load(use_builtin=True)
        cases = adapter.to_scenario_cases()
        ids = [c.case_id for c in cases]
        assert len(ids) == len(set(ids)), "Duplicate case IDs found"

    def test_expected_verdicts_valid(self):
        valid_verdicts = {"approve", "escalate", "reject", "hard_block"}
        adapter = BBQAdapter()
        adapter.load(use_builtin=True)
        cases = adapter.to_scenario_cases()
        for case in cases:
            assert case.expected_verdict in valid_verdicts, (
                f"Invalid verdict '{case.expected_verdict}' in {case.case_id}"
            )

    def test_categories_covered(self):
        adapter = BBQAdapter()
        adapter.load(use_builtin=True)
        assert set(adapter.categories) == {"age", "gender", "race", "religion"}

    def test_summary(self):
        adapter = BBQAdapter()
        adapter.load(use_builtin=True)
        summary = adapter.summary()
        assert summary["loaded"] is True
        assert summary["total_cases"] == 40
        assert "by_category" in summary

    def test_load_not_called_raises(self):
        adapter = BBQAdapter()
        with pytest.raises(RuntimeError, match="Call load"):
            adapter.to_scenario_cases()

    def test_tags_include_external_and_bbq(self):
        adapter = BBQAdapter()
        adapter.load(use_builtin=True)
        cases = adapter.to_scenario_cases()
        for case in cases:
            assert "external" in case.tags
            assert "bbq" in case.tags

    def test_task_description_contains_context(self):
        """Each BBQ task should include the original context and question."""
        adapter = BBQAdapter()
        adapter.load(use_builtin=True)
        cases = adapter.to_scenario_cases()
        for case in cases:
            assert len(case.task) > 50, f"Task too short in {case.case_id}"
