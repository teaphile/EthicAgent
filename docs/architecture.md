# Architecture

## Pipeline Overview

EthicAgent processes ethical decisions through a multi-stage pipeline:

```
Input Task  ──►  Context Analysis  ──►  Neural Reasoning
                                          │
                                   Symbolic Reasoning
                                          │
                                    Fusion Agent
                                          │
                                   Ethical Reasoner
                                     (EDS Formula)
                                          │
                               ┌────┬────┬────┐
                               │ D  │ C  │ V  │ Ctx │
                               └────┴────┴────┘
                                          │
                                   Conflict Resolution
                                          │
                                   Action Executor
                                          │
                                   Reflection + Audit
                                          │
                                     Output Result
```

## Components

### Core Layer
- **StateManager**: immutable pipeline state, stage results
- **AuditLogger**: structured event logging with JSON/CSV export

### Agent Layer
- **ContextAgent**: domain classification, entity extraction, urgency detection
- **NeuralReasonerAgent**: LLM-based ethical analysis
- **SymbolicReasonerAgent**: rule-based reasoning using knowledge graph
- **FusionAgent**: neuro-symbolic integration with safety override
- **EthicalReasonerAgent**: EDS computation and verdict determination
- **ActionExecutor**: verdict execution with audit entry
- **ReflectionAgent**: consistency tracking and learning

### Ethics Layer
- **DeontologicalEvaluator**: hard rules, duty rules, domain rules
- **ConsequentialistEvaluator**: benefit-harm analysis
- **VirtueEthicsEvaluator**: fairness metrics, disparate impact
- **ContextualEthicsEvaluator**: situational appropriateness
- **ConflictResolver**: philosophy disagreement resolution

### Knowledge Layer
- **KnowledgeGraph**: NetworkX-based ethical ontology per domain
- **PrecedentStore**: case-based reasoning with similarity search
- **EthicalMemory**: short-term and long-term decision memory

### Explainability Layer
- **ExplanationGenerator**: multi-level explanations (summary, standard, detailed)
- **DecisionTracer**: full pipeline trace for audit
- **Visualization**: Plotly charts (radar, comparison, timeline)

## Data Flow

1. User provides task + optional domain
2. ContextAgent classifies domain, extracts entities, determines urgency
3. Neural and symbolic reasoners analyze in parallel
4. FusionAgent combines outputs (symbolic safety override)
5. EthicalReasonerAgent computes EDS, determines verdict
6. ActionExecutor logs and returns result
7. ReflectionAgent updates consistency memory
