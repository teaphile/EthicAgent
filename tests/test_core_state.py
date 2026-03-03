"""Tests for core state management module."""

from __future__ import annotations

import pytest
from ethicagent.core.state import PipelineState, PipelineStage, StageResult, StateManager


class TestPipelineStage:
    """Tests for PipelineStage enum."""

    def test_all_stages_defined(self):
        stages = list(PipelineStage)
        assert len(stages) >= 8

    def test_stage_values(self):
        assert PipelineStage.CONTEXT_ANALYSIS.value == "context_analysis"
        assert PipelineStage.NEURAL_REASONING.value == "neural_reasoning"
        assert PipelineStage.SYMBOLIC_REASONING.value == "symbolic_reasoning"


class TestPipelineState:
    """Tests for PipelineState dataclass."""

    def test_creation_defaults(self):
        state = PipelineState(task="test task")
        assert state.task == "test task"
        assert state.domain is None
        assert state.stage_results == {}

    def test_creation_with_domain(self):
        state = PipelineState(task="test", domain="healthcare")
        assert state.domain == "healthcare"

    def test_state_is_dataclass(self):
        state = PipelineState(task="test")
        assert hasattr(state, "__dataclass_fields__")


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_creation(self):
        result = StageResult(
            stage=PipelineStage.CONTEXT_ANALYSIS,
            data={"domain": "finance"},
            success=True,
        )
        assert result.stage == PipelineStage.CONTEXT_ANALYSIS
        assert result.data["domain"] == "finance"
        assert result.success is True

    def test_creation_with_error(self):
        result = StageResult(
            stage=PipelineStage.NEURAL_REASONING,
            data={},
            success=False,
            error="LLM unavailable",
        )
        assert result.success is False
        assert result.error == "LLM unavailable"


class TestStateManager:
    """Tests for StateManager."""

    def test_create_state(self):
        manager = StateManager()
        state = manager.create_state("test task", domain="finance")
        assert state.task == "test task"
        assert state.domain == "finance"

    def test_update_stage(self):
        manager = StateManager()
        state = manager.create_state("test")
        result = StageResult(
            stage=PipelineStage.CONTEXT_ANALYSIS,
            data={"entities": ["loan"]},
            success=True,
        )
        manager.update_stage(state, result)
        assert PipelineStage.CONTEXT_ANALYSIS in state.stage_results

    def test_get_stage_result(self):
        manager = StateManager()
        state = manager.create_state("test")
        data = {"domain": "healthcare"}
        result = StageResult(
            stage=PipelineStage.CONTEXT_ANALYSIS,
            data=data,
            success=True,
        )
        manager.update_stage(state, result)
        retrieved = manager.get_stage_result(state, PipelineStage.CONTEXT_ANALYSIS)
        assert retrieved is not None
        assert retrieved.data["domain"] == "healthcare"

    def test_get_missing_stage_result(self):
        manager = StateManager()
        state = manager.create_state("test")
        retrieved = manager.get_stage_result(state, PipelineStage.NEURAL_REASONING)
        assert retrieved is None
