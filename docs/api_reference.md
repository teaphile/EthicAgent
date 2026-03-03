# API Reference

## Core

### `ethicagent.orchestrator.EthicAgentOrchestrator`

Main pipeline entry point.

```python
orch = EthicAgentOrchestrator(config=None)
result = orch.run(task="...", domain="finance")
```

**Returns:** `dict` with keys:
- `verdict` — `"auto_approve"`, `"escalate"`, `"reject"`, or `"hard_block"`
- `eds_score` — float in [0, 1]
- `philosophy_scores` — `{"deontological": float, ...}`
- `explanation` — natural-language rationale

### `ethicagent.core.state.StateManager`

```python
manager = StateManager()
state = manager.create_state(task, domain=None)
manager.update_stage(state, stage_result)
result = manager.get_stage_result(state, PipelineStage.CONTEXT_ANALYSIS)
```

### `ethicagent.core.logger.AuditLogger`

```python
logger = AuditLogger()
logger.log_event(event_type, message, data=None)
logger.log_decision(task=..., verdict=..., eds_score=..., **kwargs)
logger.export_to_json(path)
logger.export_to_csv(path)
```

---

## Agents

### `ContextAgent`
- `analyze(task) → dict` — domain, entities, urgency
- `classify_domain(task) → str`
- `extract_entities(task) → list`
- `determine_urgency(task) → str`

### `EthicalReasonerAgent`
- `compute_eds(scores, weights) → float`
- `determine_verdict(scores, weights) → str`

### `FusionAgent`
- `fuse(neural, symbolic) → dict`

### `ActionExecutor`
- `execute(verdict, eds_score, task, reasoning) → dict`

### `ReflectionAgent`
- `record_decision(task, domain, verdict, eds_score, philosophy_scores)`
- `analyze_consistency() → dict`
- `get_learning_summary() → str | dict`

---

## Ethics

### Evaluators

All evaluators expose: `evaluate(task, context) → PhilosophyResult`

- `DeontologicalEvaluator`
- `ConsequentialistEvaluator`
- `VirtueEthicsEvaluator`
- `ContextualEthicsEvaluator`

### Data Classes

```python
@dataclass
class PhilosophyResult:
    philosophy: str
    score: float
    reasoning: str
    details: dict

class EthicalVerdict(Enum):
    AUTO_APPROVE = "auto_approve"
    ESCALATE = "escalate"
    REJECT = "reject"
    HARD_BLOCK = "hard_block"

@dataclass
class EthicalDecision:
    eds_score: float
    verdict: EthicalVerdict
    philosophy_results: list[PhilosophyResult]
    reasoning: str
```

### `ConflictResolver`
- `detect_conflicts(scores) → list[dict]`
- `resolve(scores) → dict`

---

## Scenarios

### Registry

```python
from ethicagent.scenarios import SCENARIO_REGISTRY, get_all_cases

SCENARIO_REGISTRY  # {"healthcare": cls, "hiring": cls, "finance": cls, "disaster": cls}
get_all_cases()    # → list[ScenarioCase], 425 total
```

### `ScenarioCase` (dataclass)

```python
@dataclass
class ScenarioCase:
    case_id: str
    task: str
    domain: str
    expected_verdict: str
    expected_eds_range: tuple[float, float]
    difficulty: str = "medium"
    context: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
```

### Scenario Classes

Each extends `BaseScenario` and implements `get_cases() → list[ScenarioCase]` and `get_domain() → str`.

- `HealthcareTriageScenario` — 110 cases, prefix `HC-`
- `HiringDecisionScenario` — 105 cases, prefix `HR-`
- `LoanApprovalScenario` — 105 cases, prefix `FN-`
- `DisasterResponseScenario` — 105 cases, prefix `DR-`

---

## Evaluation

### Metrics

```python
from ethicagent.evaluation.metrics import compute_all_metrics

metrics = compute_all_metrics(results)
# Keys: verdict_accuracy, eds_score_metrics, eds_range_accuracy,
#        fairness, consistency, philosophy_contributions, total_cases
```

Individual functions:
- `verdict_accuracy(results) → dict`
- `eds_score_metrics(results) → dict`
- `eds_range_accuracy(results) → dict`
- `fairness_metrics(results) → dict`
- `consistency_score(results) → dict`
- `philosophy_contribution_analysis(results) → dict`

### Baselines

```python
from ethicagent.evaluation.baselines import get_all_baselines

baselines = get_all_baselines()  # dict of name → baseline
result = baselines["random"].evaluate(case)
```

- `RandomBaseline`, `RulesOnlyBaseline`, `LLMOnlyBaseline`, `EqualWeightBaseline`

### Statistical Analysis

```python
analyzer = StatisticalAnalyzer(alpha=0.05)
result = analyzer.compare_systems(scores_a, scores_b)
summary = analyzer.generate_summary(result)
mcnemar = analyzer.mcnemar_test(correct_a, correct_b)
```

---

## Benchmarks

```python
from ethicagent.benchmarks import BenchmarkSuite

suite = BenchmarkSuite()
results = suite.run_all()
suite.export("results/")
print(suite.summary(results))
```

- `CrossDomainBenchmark` — transfer gap analysis
- `PerformanceBenchmark` — latency and throughput
- `ReproducibilityBenchmark` — determinism checks

---

## Adversarial

```python
from ethicagent.adversarial import PerturbationAttack, JailbreakAttack, RobustnessEvaluator

# Perturbation
attack = PerturbationAttack()
results = attack.run(cases)
print(attack.summary(results))

# Jailbreak
jb = JailbreakAttack()
results = jb.run()
print(jb.summary(results))

# Combined
evaluator = RobustnessEvaluator()
report = evaluator.run()
print(report["overall_robustness_score"])
```

---

## Explainability

### `ExplanationGenerator`
- `generate(decision, level="standard") → str`
- `extract_key_factors(decision) → list`

### `DecisionTracer`
- `start_trace(task) → str` (trace ID)
- `record_step(trace_id, stage, data, duration_ms)`
- `finalize_trace(trace_id, verdict, eds_score)`
- `export_trace(trace_id) → dict`
- `search_by_verdict(verdict) → list`
- `get_statistics() → dict`

---

## Knowledge

### `KnowledgeGraph`
- `add_node(node_id, node_type, **attrs)`
- `add_edge(source, target, relationship, **attrs)`
- `get_neighbors(node_id) → list`
- `initialize_domain(domain)`
- `query_ethical_context(query, domain) → dict | list`

### `PrecedentStore`
- `add_precedent(case_id, task, domain, verdict, eds_score, reasoning)`
- `search_similar(query, domain) → list`
- `filter_by_verdict(verdict) → list`
- `save()` / `load()`

---

## Utilities

### `ethicagent.utils.helpers`
- `clamp(value, lo, hi) → float`
- `hash_dict(d) → str`
- `merge_dicts(*dicts) → dict`
- `truncate_text(text, max_len) → str`
- `safe_divide(a, b, default=0.0) → float`
- `flatten_dict(d, sep=".") → dict`
- `format_score_breakdown(scores) → str`
- `now_iso() → str`
- `timer` — decorator

### `ethicagent.utils.validators`
- `validate_task_input(task) → bool`
- `validate_domain(domain) → bool`
- `validate_scores(scores) → bool`
- `validate_weights(weights) → bool`
