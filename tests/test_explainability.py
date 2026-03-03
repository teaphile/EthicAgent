"""Tests for explainability modules — explanation, traces, visualization."""

from __future__ import annotations

from ethicagent.explainability.decision_trace import DecisionTracer
from ethicagent.explainability.explanation_generator import ExplanationGenerator


class TestExplanationGenerator:
    def test_initialization(self):
        gen = ExplanationGenerator()
        assert gen is not None

    def test_generate_summary(self):
        gen = ExplanationGenerator()
        decision = {
            "eds_score": 0.85,
            "verdict": "approve",
            "philosophy_scores": {
                "deontological": 0.90,
                "consequentialist": 0.80,
                "virtue_ethics": 0.85,
                "contextual": 0.82,
            },
            "reasoning": "All ethics criteria satisfied",
        }
        explanation = gen.generate(decision, level="summary")
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_generate_standard(self):
        gen = ExplanationGenerator()
        decision = {
            "eds_score": 0.62,
            "verdict": "escalate",
            "philosophy_scores": {
                "deontological": 0.70,
                "consequentialist": 0.55,
                "virtue_ethics": 0.60,
                "contextual": 0.65,
            },
            "reasoning": "Moderate score requires review",
        }
        explanation = gen.generate(decision, level="standard")
        assert isinstance(explanation, str)
        assert len(explanation) > 50

    def test_generate_detailed(self):
        gen = ExplanationGenerator()
        decision = {
            "eds_score": 0.30,
            "verdict": "reject",
            "philosophy_scores": {
                "deontological": 0.40,
                "consequentialist": 0.20,
                "virtue_ethics": 0.30,
                "contextual": 0.25,
            },
            "reasoning": "Multiple ethical concerns",
            "conflicts": [{"type": "philosophy_disagreement"}],
        }
        explanation = gen.generate(decision, level="detailed")
        assert isinstance(explanation, str)
        assert len(explanation) > 100

    def test_key_factors(self):
        gen = ExplanationGenerator()
        decision = {
            "eds_score": 0.85,
            "verdict": "approve",
            "philosophy_scores": {
                "deontological": 0.90,
                "consequentialist": 0.80,
                "virtue_ethics": 0.85,
                "contextual": 0.82,
            },
        }
        factors = gen.extract_key_factors(decision)
        assert isinstance(factors, list)


class TestDecisionTracer:
    def test_initialization(self):
        tracer = DecisionTracer()
        assert tracer is not None

    def test_record_step(self):
        tracer = DecisionTracer()
        trace_id = tracer.start_trace("test_task")
        tracer.record_step(
            trace_id,
            stage="context_analysis",
            data={"domain": "finance"},
            duration_ms=15,
        )
        trace = tracer.get_trace(trace_id)
        assert trace is not None
        assert len(trace.steps) >= 1

    def test_finalize_trace(self):
        tracer = DecisionTracer()
        trace_id = tracer.start_trace("test_task")
        tracer.record_step(
            trace_id,
            stage="analysis",
            data={},
            duration_ms=10,
        )
        tracer.finalize_trace(trace_id, verdict="approve", eds_score=0.85)
        trace = tracer.get_trace(trace_id)
        assert trace.verdict == "approve"
        assert trace.eds_score == 0.85

    def test_export_trace(self):
        tracer = DecisionTracer()
        trace_id = tracer.start_trace("export_test")
        tracer.record_step(
            trace_id,
            stage="test",
            data={"key": "value"},
            duration_ms=5,
        )
        tracer.finalize_trace(trace_id, verdict="escalate", eds_score=0.60)
        exported = tracer.export_trace(trace_id)
        assert isinstance(exported, dict)
        assert "steps" in exported

    def test_search_by_verdict(self):
        tracer = DecisionTracer()

        t1 = tracer.start_trace("approved_task")
        tracer.finalize_trace(t1, verdict="approve", eds_score=0.85)

        t2 = tracer.start_trace("rejected_task")
        tracer.finalize_trace(t2, verdict="reject", eds_score=0.30)

        approved = tracer.search_by_verdict("approve")
        assert len(approved) >= 1

    def test_statistics(self):
        tracer = DecisionTracer()
        for i in range(5):
            tid = tracer.start_trace(f"task_{i}")
            tracer.finalize_trace(tid, verdict="approve", eds_score=0.80 + i * 0.02)

        stats = tracer.get_statistics()
        assert isinstance(stats, dict)
        assert stats.get("total_traces", 0) >= 5
