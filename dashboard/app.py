"""EthicAgent Dashboard — Streamlit Application.

Interactive dashboard for exploring ethical decisions,
comparing philosophies, running scenarios, and auditing traces.

Launch with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import streamlit as st

# Add project root so ethicagent is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

DOMAINS = ["General", "Healthcare", "Finance", "Hiring", "Disaster"]
DOMAIN_WEIGHTS: dict[str, tuple] = {
    "Healthcare": (0.35, 0.25, 0.20, 0.20),
    "Finance": (0.20, 0.25, 0.35, 0.20),
    "Hiring": (0.15, 0.20, 0.40, 0.25),
    "Disaster": (0.20, 0.35, 0.15, 0.30),
    "General": (0.25, 0.25, 0.25, 0.25),
}

VERDICT_COLORS = {
    "approve": "🟢",
    "auto_approve": "🟢",
    "escalate": "🟡",
    "reject": "🔴",
    "hard_block": "⛔",
}

# ═══════════════════════════════════════════════════════════════
# Authentication
# ═══════════════════════════════════════════════════════════════

# Set ETHICAGENT_DASHBOARD_PASSWORD env var to enable auth.
# When unset the dashboard runs without a login gate (dev mode).
_DASHBOARD_PASSWORD = os.environ.get("ETHICAGENT_DASHBOARD_PASSWORD", "")


def _check_auth() -> bool:
    """Simple password gate for the dashboard.

    Guards sensitive ethical-decision data behind a login form
    when ``ETHICAGENT_DASHBOARD_PASSWORD`` is configured.  In
    production, prefer a reverse-proxy with OAuth2/OIDC instead.

    Returns True if the user is authenticated (or auth is disabled).
    """
    if not _DASHBOARD_PASSWORD:
        # Auth not configured — allow access (dev / local mode)
        return True

    if st.session_state.get("authenticated"):
        return True

    st.title("🔒 EthicAgent Dashboard — Login")
    st.markdown("Access to the dashboard requires authentication.")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login", type="primary"):
        if password == _DASHBOARD_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid password. Please try again.")
    return False


def main() -> None:
    """Main dashboard entry point."""
    st.set_page_config(
        page_title="EthicAgent Dashboard",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── authentication gate ─────────────────────────────────
    if not _check_auth():
        return

    # Load custom CSS
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # ── sidebar ─────────────────────────────────────────────
    st.sidebar.title("⚖️ EthicAgent")
    st.sidebar.markdown("*Neuro-Symbolic Ethical Reasoning*")

    page = st.sidebar.radio(
        "Navigate",
        [
            "🏠 Overview",
            "🔬 Scenario Analysis",
            "📊 Philosophy Comparison",
            "📋 Audit Trail",
            "🧪 Benchmark Results",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Settings")
    domain = st.sidebar.selectbox("Domain", DOMAINS)
    st.sidebar.markdown(f"**Active domain:** {domain}")

    w = DOMAIN_WEIGHTS[domain]
    st.sidebar.caption(f"D={w[0]} C={w[1]} V={w[2]} Ctx={w[3]}")

    # ── page router ─────────────────────────────────────────
    if page.startswith("🏠"):
        _page_overview(domain)
    elif page.startswith("🔬"):
        _page_scenario_analysis(domain)
    elif page.startswith("📊"):
        _page_philosophy_comparison(domain)
    elif page.startswith("📋"):
        _page_audit_trail()
    elif page.startswith("🧪"):
        _page_benchmark_results()


# ═══════════════════════════════════════════════════════════════
# Page: Overview
# ═══════════════════════════════════════════════════════════════


def _page_overview(domain: str) -> None:
    st.title("EthicAgent Dashboard")
    st.markdown("A Context-Aware Neuro-Symbolic Framework for Ethical Autonomous Decision-Making")

    # Quick Decision Input
    st.header("Quick Decision Analysis")

    col1, col2 = st.columns([3, 1])
    with col1:
        task = st.text_area(
            "Describe the action to evaluate:",
            placeholder=(
                "e.g., Approve a loan for an applicant with low "
                "credit score but stable employment..."
            ),
            height=100,
        )
    with col2:
        st.markdown("**Domain weights:**")
        _show_domain_weights(domain)

    if st.button("⚡ Analyze", type="primary"):
        if task.strip():
            _run_quick_analysis(task, domain)
        else:
            st.warning("Please enter a task description.")

    st.markdown("---")

    # Statistics
    st.header("System Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Decisions", "0", help="Total decisions processed")
    with c2:
        st.metric("Avg EDS Score", "0.000", help="Average Ethical Decision Score")
    with c3:
        st.metric("Auto-Approved", "0%", help="Decisions automatically approved")
    with c4:
        st.metric("Conflicts Resolved", "0", help="Inter-philosophy conflicts resolved")

    # EDS Formula
    st.header("EDS Formula")
    st.latex(r"EDS(a) = w_1 \cdot D(a) + w_2 \cdot C(a) + w_3 \cdot V(a) + w_4 \cdot Ctx(a)")
    st.markdown("""
    Where:
    - **D(a)**: Deontological score (rule compliance)
    - **C(a)**: Consequentialist score (outcome analysis)
    - **V(a)**: Virtue ethics score (fairness & character)
    - **Ctx(a)**: Contextual ethics score (situational factors)
    """)

    # Decision Thresholds
    st.header("Decision Thresholds")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.success("**AUTO_APPROVE**\nEDS ≥ 0.80")
    with c2:
        st.warning("**ESCALATE**\n0.50 ≤ EDS < 0.80")
    with c3:
        st.error("**REJECT**\nEDS < 0.50")
    with c4:
        st.error("**HARD_BLOCK**\nD(a) = 0")


# ═══════════════════════════════════════════════════════════════
# Page: Scenario Analysis
# ═══════════════════════════════════════════════════════════════


def _page_scenario_analysis(domain: str) -> None:
    st.title("🔬 Scenario Analysis")

    scenario_type = st.selectbox(
        "Select Scenario Type",
        ["Healthcare Triage", "Loan Approval", "Hiring Decision", "Disaster Response"],
    )

    st.markdown(f"### {scenario_type} Scenarios")

    try:
        cases = _load_scenario_cases(scenario_type)
        if not cases:
            st.warning("No scenarios loaded. Check the scenario modules.")
            return

        st.info(f"Loaded {len(cases)} test cases")

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            difficulty = st.multiselect(
                "Difficulty",
                ["easy", "medium", "hard", "dilemma"],
                default=["easy", "medium", "hard"],
            )
        with col2:
            expected = st.multiselect(
                "Expected Verdict",
                ["approve", "escalate", "reject", "hard_block"],
                default=["approve", "escalate", "reject", "hard_block"],
            )

        # Filter & display
        filtered = [
            c
            for c in cases
            if c.get("expected_verdict") in expected and c.get("difficulty", "medium") in difficulty
        ]

        st.markdown(f"### Test Cases ({len(filtered)} shown)")
        for i, case in enumerate(filtered[:30]):
            ev = case.get("expected_verdict", "unknown")
            icon = VERDICT_COLORS.get(ev, "⚪")
            with st.expander(f"{icon} {case.get('case_id', f'Case-{i + 1}')} — Expected: {ev}"):
                st.markdown(f"**Task:** {case.get('task', '')}")
                st.markdown(f"**Domain:** {case.get('domain', domain)}")
                st.markdown(f"**Difficulty:** {case.get('difficulty', 'medium')}")
                eds_r = case.get("expected_eds_range", (0, 1))
                st.markdown(f"**EDS Range:** [{eds_r[0]:.2f}, {eds_r[1]:.2f}]")

                if st.button("Run Analysis", key=f"run_{i}"):
                    _run_quick_analysis(case.get("task", ""), domain)

    except Exception as e:
        st.error(f"Error loading scenarios: {e}")

    # Batch run
    st.markdown("---")
    st.header("Batch Evaluation")
    if st.button("🚀 Run All Scenarios", type="primary"):
        st.info("Batch evaluation would run here with the orchestrator.")
        _show_batch_placeholder()


# ═══════════════════════════════════════════════════════════════
# Page: Philosophy Comparison
# ═══════════════════════════════════════════════════════════════


def _page_philosophy_comparison(domain: str) -> None:
    st.title("📊 Philosophy Comparison")
    st.markdown("Compare contributions from each ethical philosophy across scenarios and domains.")

    w = DOMAIN_WEIGHTS[domain]
    labels = [
        "Deontological (w₁)",
        "Consequentialist (w₂)",
        "Virtue Ethics (w₃)",
        "Contextual (w₄)",
    ]

    st.header("Domain Weight Configuration")
    cols = st.columns(4)
    for col, label, val in zip(cols, labels, w, strict=False):
        with col:
            st.metric(label, f"{val:.2f}")

    # Radar chart
    st.header("Philosophy Scores Radar")
    try:
        from ethicagent.explainability.visualization import philosophy_radar_chart

        fig = philosophy_radar_chart(
            {
                "deontological": w[0],
                "consequentialist": w[1],
                "virtue_ethics": w[2],
                "contextual": w[3],
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Run an analysis to see the philosophy radar chart.")

    # Cross-domain weight comparison
    st.header("Cross-Domain Weight Comparison")
    try:
        from ethicagent.explainability.visualization import weight_comparison_chart

        fig = weight_comparison_chart(
            {
                d: dict(
                    zip(
                        ["deontological", "consequentialist", "virtue_ethics", "contextual"],
                        DOMAIN_WEIGHTS[d],
                        strict=False,
                    )
                )
                for d in DOMAINS
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Weight comparison chart unavailable: {e}")

    # Philosophy descriptions
    st.header("Philosophy Descriptions")
    _philosophy_descriptions = {
        "🔒 Deontological Ethics": (
            "**Focus:** Rule-based moral duties and obligations.\n\n"
            "Evaluates actions against hard rules (absolute prohibitions), "
            "duty rules (obligations), and domain-specific regulations. "
            "A score of 0 triggers an automatic HARD_BLOCK."
        ),
        "📈 Consequentialist Ethics": (
            "**Focus:** Outcomes and consequences of actions.\n\n"
            "Analyzes benefits vs harms across all affected stakeholders, "
            "considering reversibility, temporal effects, and scale of impact."
        ),
        "⚖️ Virtue Ethics": (
            "**Focus:** Fairness, character, and moral virtue.\n\n"
            "Measures statistical parity, disparate impact, equal opportunity, "
            "vulnerable population protections, and representation fairness."
        ),
        "🌍 Contextual Ethics": (
            "**Focus:** Situational and environmental factors.\n\n"
            "Considers domain appropriateness, urgency level, cultural "
            "sensitivity, legal compliance, temporal context, and "
            "stakeholder impact distribution."
        ),
    }
    for title, desc in _philosophy_descriptions.items():
        with st.expander(title):
            st.markdown(desc)


# ═══════════════════════════════════════════════════════════════
# Page: Audit Trail
# ═══════════════════════════════════════════════════════════════


def _page_audit_trail() -> None:
    st.title("📋 Audit Trail")
    st.markdown("Full decision trace and audit log for transparency and regulatory compliance.")

    col1, col2, col3 = st.columns(3)
    with col1:
        search = st.text_input("Search decisions", placeholder="keyword...")
    with col2:
        verdict_filter = st.selectbox(
            "Filter by verdict",
            ["All", "approve", "escalate", "reject", "hard_block"],
        )
    with col3:
        st.date_input("Date range", [])

    st.header("Decision Log")

    log_path = os.path.join(os.path.dirname(__file__), "..", "data", "audit_log.json")
    if not os.path.exists(log_path):
        st.info(
            "No audit log found. Run some decisions through "
            "the orchestrator to generate an audit trail."
        )
        return

    try:
        with open(log_path) as f:
            entries: list[dict] = json.load(f)

        if search:
            entries = [e for e in entries if search.lower() in json.dumps(e).lower()]
        if verdict_filter != "All":
            entries = [e for e in entries if e.get("verdict") == verdict_filter]

        st.info(f"Showing {len(entries)} entries")

        for entry in entries[:50]:
            ts = entry.get("timestamp", "unknown")
            verdict = entry.get("verdict", "unknown")
            eds = entry.get("eds_score", 0)
            icon = VERDICT_COLORS.get(verdict, "⚪")

            with st.expander(f"{icon} [{ts}] {verdict.upper()} — EDS: {eds:.3f}"):
                st.json(entry)

    except Exception as e:
        st.error(f"Failed to load audit log: {e}")

    # Export
    st.markdown("---")
    st.header("Export")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📄 Export JSON"):
            st.info("JSON export would be triggered here.")
    with c2:
        if st.button("📊 Export CSV"):
            st.info("CSV export would be triggered here.")
    with c3:
        if st.button("📑 Export PDF"):
            st.info("PDF export would be triggered here.")


# ═══════════════════════════════════════════════════════════════
# Page: Benchmark Results
# ═══════════════════════════════════════════════════════════════


def _page_benchmark_results() -> None:
    """Show pre-generated benchmark results if available."""
    st.title("🧪 Benchmark Results")

    results_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "results",
    )

    if not os.path.isdir(results_path):
        st.info(
            "No results directory found. Run benchmarks first:\n\n"
            "```python\nfrom ethicagent.benchmarks import BenchmarkSuite\n"
            "suite = BenchmarkSuite()\nresults = suite.run_all()\n"
            "suite.export('results/')\n```"
        )
        return

    # List available result files
    files = sorted(f for f in os.listdir(results_path) if f.endswith(".json"))
    if not files:
        st.warning("No JSON result files found in results/")
        return

    selected = st.selectbox("Result file", files)
    fpath = os.path.join(results_path, selected)

    try:
        with open(fpath) as f:
            data = json.load(f)
        st.json(data)
    except Exception as e:
        st.error(f"Failed to load {selected}: {e}")


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════


def _show_domain_weights(domain: str) -> None:
    w = DOMAIN_WEIGHTS.get(domain, (0.25, 0.25, 0.25, 0.25))
    st.caption(f"D: {w[0]:.2f} | C: {w[1]:.2f}")
    st.caption(f"V: {w[2]:.2f} | Ctx: {w[3]:.2f}")


def _run_quick_analysis(task: str, domain: str) -> None:
    """Run a quick ethical analysis on a task."""
    with st.spinner("Analyzing..."):
        try:
            from ethicagent.orchestrator import EthicAgentOrchestrator

            orch = EthicAgentOrchestrator()
            result = orch.run(task=task, domain=domain.lower())

            verdict = getattr(result, "verdict", "unknown")
            eds = getattr(result, "eds_score", 0)

            color_fn = {
                "approve": st.success,
                "auto_approve": st.success,
                "escalate": st.warning,
                "reject": st.error,
                "hard_block": st.error,
            }.get(verdict, st.info)
            color_fn(f"**Verdict: {verdict.upper()}** | EDS: {eds:.3f}")

            # Philosophy breakdown
            phil = getattr(result, "philosophy_scores", {})
            if phil:
                cols = st.columns(min(len(phil), 4))
                for i, (name, score) in enumerate(phil.items()):
                    with cols[i % len(cols)]:
                        st.metric(name.replace("_", " ").title(), f"{score:.3f}")

            explanation = getattr(result, "reasoning", "")
            if explanation:
                with st.expander("Detailed Explanation"):
                    st.markdown(explanation)

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.info(
                "Make sure OpenAI API key is configured or "
                "Ollama is running for LLM-based reasoning."
            )


def _load_scenario_cases(scenario_type: str) -> list[dict[str, Any]]:
    """Load scenario cases from Python modules or JSON data files."""
    type_map = {
        "Healthcare Triage": "healthcare",
        "Loan Approval": "finance",
        "Hiring Decision": "hiring",
        "Disaster Response": "disaster",
    }
    key = type_map.get(scenario_type, "healthcare")
    file_key = {
        "Healthcare Triage": "healthcare_triage",
        "Loan Approval": "loan_approval",
        "Hiring Decision": "hiring_decision",
        "Disaster Response": "disaster_response",
    }[scenario_type]

    # Try JSON data file first
    data_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "scenarios",
        f"{file_key}.json",
    )
    if os.path.exists(data_path):
        with open(data_path) as f:
            return json.load(f)

    # Fall back to scenario module
    try:
        from ethicagent.scenarios import SCENARIO_REGISTRY

        cls = SCENARIO_REGISTRY.get(key)
        if cls is None:
            return []
        scenario = cls()
        cases = scenario.get_cases()  # NOTE: use get_cases(), not .cases
        return [
            {
                "case_id": c.case_id,
                "task": c.task,
                "domain": getattr(c, "domain", key),
                "difficulty": getattr(c, "difficulty", "medium"),
                "expected_verdict": getattr(c, "expected_verdict", ""),
                "expected_eds_range": getattr(c, "expected_eds_range", (0, 1)),
            }
            for c in cases
        ]
    except Exception:
        return []


def _show_batch_placeholder() -> None:
    """Run a real batch evaluation on all scenarios and show results."""
    from ethicagent.scenarios import get_all_cases

    cases = get_all_cases()
    progress = st.progress(0)
    total = len(cases)
    results: list[dict[str, Any]] = []

    try:
        from ethicagent.orchestrator import EthicAgentOrchestrator

        orch = EthicAgentOrchestrator(use_llm=False)
        for i, case in enumerate(cases):
            r = orch.run(task=case.task, domain=case.domain)
            d = r.to_dict() if hasattr(r, "to_dict") else dict(r)
            verdict = d.get("verdict", "unknown").lower().replace("auto_", "")
            results.append(
                {
                    "verdict_match": verdict == case.expected_verdict,
                    "eds_score": float(d.get("eds_score", 0.0)),
                }
            )
            progress.progress((i + 1) / total)

        n = len(results)
        accuracy = sum(1 for r in results if r["verdict_match"]) / n if n else 0
        eds_values = [r["eds_score"] for r in results]
        mean_eds = sum(eds_values) / n if n else 0
        eds_std = (sum((e - mean_eds) ** 2 for e in eds_values) / n) ** 0.5 if n else 0
        consistency = max(1.0 - (eds_std / mean_eds), 0.0) if mean_eds > 0 else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Accuracy", f"{accuracy:.1%}")
        with c2:
            st.metric("Mean EDS", f"{mean_eds:.3f}")
        with c3:
            st.metric("Consistency", f"{consistency:.3f}")
    except Exception as e:
        st.error(f"Batch evaluation failed: {e}")


if __name__ == "__main__":
    main()
