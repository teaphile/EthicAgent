# EthicAgent

**A Context-Aware Neuro-Symbolic Framework for Ethical Autonomous Decision-Making**

[![CI](https://github.com/radhahai/agent/actions/workflows/ci.yml/badge.svg)](https://github.com/radhahai/agent/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Ethical AI](https://img.shields.io/badge/Ethical_AI-blueviolet)
![Neuro-Symbolic](https://img.shields.io/badge/Neuro--Symbolic-orange)
![Agentic AI](https://img.shields.io/badge/Agentic_AI-red)

---

## Overview

EthicAgent is a neuro-symbolic framework that combines large language model (LLM) reasoning with formal symbolic ethics to produce auditable, explainable ethical decisions. It evaluates actions through four philosophical lenses — deontological, consequentialist, virtue ethics, and contextual — and fuses their assessments into a single **Ethical Decision Score (EDS)**.

### Core Formula

$$EDS(a) = w_1 \cdot D(a) + w_2 \cdot C(a) + w_3 \cdot V(a) + w_4 \cdot Ctx(a)$$

| Component | Description |
|-----------|-------------|
| $D(a)$ | Deontological score — rule compliance |
| $C(a)$ | Consequentialist score — outcome analysis |
| $V(a)$ | Virtue ethics score — fairness & character |
| $Ctx(a)$ | Contextual ethics score — situational factors |

### Decision Thresholds

| Verdict | Condition |
|---------|-----------|
| `AUTO_APPROVE` | $EDS \geq 0.80$ |
| `ESCALATE` | $0.50 \leq EDS < 0.80$ |
| `REJECT` | $EDS < 0.50$ |
| `HARD_BLOCK` | $D(a) = 0$ (regardless of other scores) |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  EthicAgent Pipeline                │
├──────────┬──────────┬──────────┬───────────────────┤
│ Context  │  Neural  │ Symbolic │     Fusion        │
│ Analysis │ Reasoner │ Reasoner │ (Neuro-Symbolic)  │
├──────────┴──────────┴──────────┴───────────────────┤
│           Ethical Reasoner (EDS Formula)            │
├──────────┬──────────┬──────────┬───────────────────┤
│ Deonto-  │ Conseq-  │ Virtue   │ Contextual        │
│ logical  │ uentialist│ Ethics  │ Ethics            │
├──────────┴──────────┴──────────┴───────────────────┤
│  Knowledge Graph │ Precedent Store │ Conflict Res.  │
├──────────────────┴─────────────────┴────────────────┤
│  Action Executor │ Reflection │ Audit Logger       │
└─────────────────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
# Clone
git clone https://github.com/radhahai/agent.git
cd agent

# Install (core only)
pip install -e .

# Install with all extras
pip install -e ".[all]"
```

### Basic Usage

```python
from ethicagent.orchestrator import EthicAgentOrchestrator

orch = EthicAgentOrchestrator()
result = orch.run(
    task="Approve a loan for an applicant with stable employment but low credit score",
    domain="finance",
)

print(f"Verdict: {result['verdict']}")
print(f"EDS:     {result['eds_score']:.3f}")
print(f"Scores:  {result['philosophy_scores']}")
```

### Running Scenarios

```python
from ethicagent.scenarios import SCENARIO_REGISTRY, get_all_cases

# All 425 cases across 4 domains
cases = get_all_cases()

# Single domain
scenario = SCENARIO_REGISTRY["healthcare"]()
healthcare_cases = scenario.get_cases()
print(f"Healthcare: {len(healthcare_cases)} cases")
```

### Evaluation

```python
from ethicagent.evaluation.metrics import compute_all_metrics

results = [...]  # your pipeline outputs
metrics = compute_all_metrics(results)
print(f"Accuracy:    {metrics['verdict_accuracy']['overall_accuracy']:.1%}")
print(f"Mean EDS:    {metrics['eds_score_metrics']['mean']:.3f}")
print(f"Consistency: {metrics['consistency']['consistency']:.3f}")
```

### Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Domain Weights

Weights are calibrated per domain to reflect professional priorities:

| Domain | $w_1$ (Deonto) | $w_2$ (Conseq) | $w_3$ (Virtue) | $w_4$ (Context) |
|--------|:-:|:-:|:-:|:-:|
| Healthcare | 0.35 | 0.25 | 0.20 | 0.20 |
| Finance    | 0.20 | 0.25 | 0.35 | 0.20 |
| Hiring     | 0.15 | 0.20 | 0.40 | 0.25 |
| Disaster   | 0.20 | 0.35 | 0.15 | 0.30 |
| General    | 0.25 | 0.25 | 0.25 | 0.25 |

---

## Scenario Test Cases

| Domain | Cases | Categories | ID Prefix |
|--------|:-----:|:----------:|-----------|
| Healthcare Triage | 110 | 8 | HC- |
| Hiring Decision   | 105 | 7 | HR- |
| Loan Approval     | 105 | 7 | FN- |
| Disaster Response | 105 | 7 | DR- |
| **Total**         | **425** | | |

Each case specifies: task description, expected verdict, expected EDS range, difficulty level, domain metadata, and tags.

---

## Evaluation Framework

- **Metrics**: Verdict accuracy, EDS statistics, range accuracy, fairness (four-fifths rule), consistency, philosophy contribution analysis
- **Baselines**: Random, rules-only, LLM-only, equal-weight
- **Ablation**: 10+ system variants (remove neural, remove symbolic, single-philosophy, etc.)
- **Statistical analysis**: Paired t-test, Wilcoxon signed-rank, Cohen's d, Cliff's delta, bootstrap CI, Holm correction, McNemar's test

---

## Adversarial Testing

- **Perturbation attacks**: Character typos, synonym swaps, rephrasing, negation injection, word shuffling
- **Jailbreak payloads**: System override, role-play, encoding bypass, authority claims, emotional manipulation, multi-turn escalation
- **Robustness evaluator**: Combined score with configurable weights

---

## Results

### EthicAgent vs. Baselines

| System | Accuracy | Mean EDS | EDS MAE | Fairness DI | Consistency |
|--------|:--------:|:--------:|:-------:|:-----------:|:-----------:|
| **EthicAgent** | **56%** | 0.523 | **0.080** | **0.84** | **0.53** |
| Equal Weight | 59% | 0.523 | 0.287 | 0.86 | 0.54 |
| LLM Only | 40% | 0.551 | 0.172 | 0.78 | 0.37 |
| Random | 26% | 0.502 | 0.258 | 0.70 | 0.24 |
| Rules Only | 16% | 0.546 | 0.107 | 0.70 | 0.20 |

> All experiments seeded (`seed=42`, 425 cases). Run `python scripts/generate_results.py` to reproduce.

[Comparison chart (PDF)](data/results/charts/comparison_bar_chart.pdf)

### Ablation Study — Top Component Impact

| Component Removed | Accuracy Drop | EDS MAE Δ |
|-------------------|:-------------:|:---------:|
| Deontological Evaluator | −22% | +0.112 |
| Symbolic Reasoner | −19% | +0.096 |
| Neural Reasoner | −14% | +0.077 |
| Fusion Agent | −14% | +0.061 |
| Domain Weights | −10% | +0.045 |

[Ablation chart (PDF)](data/results/charts/ablation_impact.pdf)

### Additional Charts

<details>
<summary>EDS Distribution · Verdict Distribution · Philosophy Radar · Fairness Heatmap</summary>

[EDS Distribution (PDF)](data/results/charts/eds_distribution.pdf)
[Verdict Distribution (PDF)](data/results/charts/verdict_distribution.pdf)
[Philosophy Radar (PDF)](data/results/charts/philosophy_radar.pdf)
[Fairness Heatmap (PDF)](data/results/charts/fairness_heatmap.pdf)

</details>

---

## Project Structure

```
EthicAgent/
├── ethicagent/
│   ├── __init__.py
│   ├── orchestrator.py              # Main pipeline orchestrator
│   ├── core/                        # State management & audit logging
│   ├── agents/                      # Context, fusion, reasoning, execution agents
│   ├── ethics/                      # Philosophy evaluators & conflict resolution
│   ├── knowledge/                   # Knowledge graph & precedent store
│   ├── explainability/              # Explanations, traces, visualizations
│   ├── evaluation/                  # Metrics, baselines, ablation, stats
│   ├── scenarios/                   # 425 test cases across 4 domains
│   ├── benchmarks/                  # Performance, cross-domain, reproducibility
│   ├── adversarial/                 # Perturbation, jailbreak, robustness
│   └── utils/                       # Config, validators, helpers
├── dashboard/                       # Streamlit interactive dashboard
├── tests/                           # Pytest test suite
├── docs/                            # Documentation
├── notebooks/                       # Jupyter notebooks
├── data/                            # Knowledge & scenario data files
├── pyproject.toml                   # Package configuration
├── Dockerfile                       # Multi-stage Docker build
├── docker-compose.yml               # Service orchestration
└── Makefile                         # Common commands
```

---

## Development

```bash
# Install dev deps
make dev

# Run tests
make test

# Lint
make lint

# Format
make format

# Docker
make docker-build && make docker-up
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Topics

`ethical-ai` · `neuro-symbolic` · `agentic-ai` · `multi-agent-systems` · `responsible-ai` · `llm-safety` · `ai-ethics` · `decision-making` · `knowledge-graph` · `explainable-ai` · `fairness` · `benchmark`
