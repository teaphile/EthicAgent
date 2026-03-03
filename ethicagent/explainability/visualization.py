"""Visualization utilities for EthicAgent results.

Generates charts and diagrams for the dashboard and reports.
All functions return Plotly figures — they work in both Streamlit
and Jupyter notebooks.

Chart types:
  - Spider/radar chart for philosophy scores
  - EDS trend line chart
  - Verdict distribution pie chart
  - Heatmap comparing scenarios
  - Sensitivity tornado chart
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# We try to import plotly, but fall back gracefully if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None  # type: ignore
    px = None  # type: ignore


def _check_plotly() -> None:
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required for visualization. Install with: pip install plotly")


def spider_chart(
    philosophy_scores: dict[str, float],
    title: str = "Philosophy Scores",
) -> Any:
    """Radar chart showing the four philosophy scores."""
    _check_plotly()

    categories = list(philosophy_scores.keys())
    values = list(philosophy_scores.values())

    # Close the polygon
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values,
            theta=[c.title() for c in categories],
            fill="toself",
            name="Scores",
            line=dict(color="#2ecc71"),
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
        ),
        showlegend=False,
        title=title,
        template="plotly_white",
    )
    return fig


def eds_trend(
    scores: list[float],
    labels: list[str] | None = None,
    title: str = "EDS Score Trend",
) -> Any:
    """Line chart showing EDS scores across decisions."""
    _check_plotly()

    x = labels or list(range(1, len(scores) + 1))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=scores,
            mode="lines+markers",
            name="EDS",
            line=dict(color="#3498db", width=2),
        )
    )

    # Add threshold lines
    fig.add_hline(y=0.80, line_dash="dash", line_color="green", annotation_text="Approve (0.80)")
    fig.add_hline(y=0.50, line_dash="dash", line_color="orange", annotation_text="Escalate (0.50)")

    fig.update_layout(
        title=title,
        xaxis_title="Decision",
        yaxis_title="EDS Score",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
    )
    return fig


def verdict_distribution(
    verdicts: list[str],
    title: str = "Verdict Distribution",
) -> Any:
    """Pie chart of verdict distribution."""
    _check_plotly()

    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1

    colors = {
        "AUTO_APPROVE": "#2ecc71",
        "ESCALATE": "#f39c12",
        "REJECT": "#e74c3c",
        "HARD_BLOCK": "#c0392b",
    }

    fig = go.Figure(
        data=go.Pie(
            labels=list(counts.keys()),
            values=list(counts.values()),
            marker=dict(colors=[colors.get(v, "#95a5a6") for v in counts]),
            textinfo="label+percent",
        )
    )

    fig.update_layout(title=title, template="plotly_white")
    return fig


def scenario_heatmap(
    scenarios: list[str],
    philosophies: list[str],
    scores: list[list[float]],
    title: str = "Scenario × Philosophy Heatmap",
) -> Any:
    """Heatmap comparing scenarios across philosophy scores."""
    _check_plotly()

    fig = go.Figure(
        data=go.Heatmap(
            z=scores,
            x=[p.title() for p in philosophies],
            y=scenarios,
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in scores],
            texttemplate="%{text}",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Philosophy",
        yaxis_title="Scenario",
        template="plotly_white",
    )
    return fig


def sensitivity_tornado(
    sensitivities: dict[str, float],
    title: str = "EDS Sensitivity Analysis",
) -> Any:
    """Tornado chart showing how sensitive EDS is to each factor."""
    _check_plotly()

    sorted_items = sorted(sensitivities.items(), key=lambda x: abs(x[1]))
    names = [item[0].title() for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig = go.Figure(
        data=go.Bar(
            y=names,
            x=values,
            orientation="h",
            marker=dict(color=["#e74c3c" if v > 0.05 else "#3498db" for v in values]),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Sensitivity (EDS change)",
        template="plotly_white",
    )
    return fig


def confidence_interval_chart(
    decisions: list[dict[str, Any]],
    title: str = "EDS Scores with Confidence Intervals",
) -> Any:
    """Chart showing EDS scores with CI error bars."""
    _check_plotly()

    labels = [d.get("label", f"Case {i + 1}") for i, d in enumerate(decisions)]
    eds_scores = [d["eds_score"] for d in decisions]
    ci_lower = [d.get("ci_lower", d["eds_score"] - 0.05) for d in decisions]
    ci_upper = [d.get("ci_upper", d["eds_score"] + 0.05) for d in decisions]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=eds_scores,
            mode="markers",
            marker=dict(size=10, color="#3498db"),
            error_y=dict(
                type="data",
                symmetric=False,
                array=[u - e for u, e in zip(ci_upper, eds_scores)],
                arrayminus=[e - l for e, l in zip(eds_scores, ci_lower)],
            ),
            name="EDS ± CI",
        )
    )

    fig.add_hline(y=0.80, line_dash="dash", line_color="green")
    fig.add_hline(y=0.50, line_dash="dash", line_color="orange")

    fig.update_layout(
        title=title,
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
    )
    return fig
