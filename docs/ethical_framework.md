# Ethical Framework

## The Ethical Decision Score (EDS)

EthicAgent's core novelty is the **Ethical Decision Score** — a weighted
aggregation of four philosophical perspectives:

$$EDS(a) = w_1 \cdot D(a) + w_2 \cdot C(a) + w_3 \cdot V(a) + w_4 \cdot Ctx(a)$$

where $\sum w_i = 1$.

## Philosophical Lenses

### Deontological Ethics (D)

Evaluates rule compliance across three tiers:

1. **Hard rules** — absolute prohibitions (score = 0 triggers HARD_BLOCK)
   - No discrimination based on protected characteristics
   - No fraud, deception, or privacy violations
   - No actions causing irreversible harm to vulnerable populations

2. **Duty rules** — domain-specific obligations
   - Healthcare: duty of care, informed consent
   - Finance: fiduciary duty, fair lending
   - Hiring: equal opportunity, merit-based evaluation

3. **Domain rules** — contextual regulatory requirements

### Consequentialist Ethics (C)

Analyzes expected outcomes:

- **Benefit magnitude** — scale and reversibility of positive outcomes
- **Harm magnitude** — scale and reversibility of negative outcomes  
- **Stakeholder scope** — number and vulnerability of affected parties
- **Temporal horizon** — short-term vs long-term consequences
- **Probability** — likelihood of outcomes materializing

### Virtue Ethics (V)

Measures fairness and moral character:

- **Statistical parity** — equal treatment rates across groups
- **Disparate impact** — four-fifths rule compliance
- **Equal opportunity** — comparable outcomes for comparable inputs
- **Vulnerable population protection** — extra safeguards
- **Representation** — balanced consideration of interests

### Contextual Ethics (Ctx)

Situational assessment:

- Domain appropriateness of the action
- Urgency level and time pressure
- Cultural sensitivity and local norms
- Legal and regulatory compliance
- Stakeholder impact distribution

## Domain Weight Rationale

| Domain | D | C | V | Ctx | Rationale |
|--------|:-:|:-:|:-:|:---:|-----------|
| Healthcare | 0.35 | 0.25 | 0.20 | 0.20 | "Do no harm" — deontological duty dominates |
| Finance | 0.20 | 0.25 | 0.35 | 0.20 | Fair lending — virtue ethics (fairness) dominates |
| Hiring | 0.15 | 0.20 | 0.40 | 0.25 | Anti-discrimination — virtue ethics strongest |
| Disaster | 0.20 | 0.35 | 0.15 | 0.30 | Outcomes matter most under urgency |
| General | 0.25 | 0.25 | 0.25 | 0.25 | Balanced — no domain-specific bias |

## Conflict Resolution

When philosophies disagree (e.g., deontological score is high but
consequentialist score is low), the conflict resolver:

1. Detects disagreement via score spread analysis
2. Identifies the dominant and minority positions
3. Applies domain-weighted resolution
4. Logs the conflict for audit

Safety takes priority: a deontological score of 0 always triggers
HARD_BLOCK, regardless of other scores.

## Verdict Semantics

| Verdict | Meaning | System Action |
|---------|---------|---------------|
| AUTO_APPROVE | Ethically sound | Proceed automatically |
| ESCALATE | Uncertain | Route to human review |
| REJECT | Ethically problematic | Block with explanation |
| HARD_BLOCK | Absolute violation | Block, log, alert |
