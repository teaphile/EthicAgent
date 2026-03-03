"""Explainability modules — making decisions understandable."""

from ethicagent.explainability.decision_trace import DecisionTrace, TraceEntry
from ethicagent.explainability.explanation_generator import (
    DetailLevel,
    Explanation,
    ExplanationGenerator,
)
from ethicagent.explainability.visualization import (
    confidence_interval_chart,
    eds_trend,
    scenario_heatmap,
    sensitivity_tornado,
    spider_chart,
    verdict_distribution,
)

__all__ = [
    "ExplanationGenerator",
    "Explanation",
    "DetailLevel",
    "spider_chart",
    "eds_trend",
    "verdict_distribution",
    "scenario_heatmap",
    "sensitivity_tornado",
    "confidence_interval_chart",
    "DecisionTrace",
    "TraceEntry",
]
