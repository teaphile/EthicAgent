"""Explainability modules — making decisions understandable."""
from ethicagent.explainability.explanation_generator import ExplanationGenerator, Explanation, DetailLevel
from ethicagent.explainability.visualization import (
    spider_chart, eds_trend, verdict_distribution,
    scenario_heatmap, sensitivity_tornado, confidence_interval_chart,
)
from ethicagent.explainability.decision_trace import DecisionTrace, TraceEntry

__all__ = [
    "ExplanationGenerator", "Explanation", "DetailLevel",
    "spider_chart", "eds_trend", "verdict_distribution",
    "scenario_heatmap", "sensitivity_tornado", "confidence_interval_chart",
    "DecisionTrace", "TraceEntry",
]
