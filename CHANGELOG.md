# Changelog

All notable changes to EthicAgent are documented in this file.

## [1.0.0] — 2025-03-02

### Changed
- Updated Python version requirement from 3.11+ to 3.10+
- Fixed Documentation URLs in pyproject.toml and setup.py (EthicAgent/docs → docs)
- Added Python 3.10 classifier to setup.py
- Synchronized pytest markers across pytest.ini and pyproject.toml (unit, integration, slow, llm, external)
- Updated SECURITY.md supported version from 0.1.x to 1.0.x

### Added
- Import aliases in utils for backward compatibility (ConfigLoader, validate_task_input, validate_scores, hash_dict, merge_dicts, truncate_text, format_score_breakdown)
- Exported all aliases from ethicagent.utils.__init__
- Adversarial testing module (perturbation, jailbreak, robustness)
- Cross-domain benchmark suite
- Statistical analysis and reproducibility framework
- Neural reasoner agent for neuro-symbolic fusion
- Reflection agent for self-evaluation

### Removed
- Unused spacy dependency from requirements.txt

### Fixed
- Import mismatches between orchestrator.py, test_utils.py and utility modules
- Marker definitions synced between pytest.ini and pyproject.toml

## [0.1.0] — 2024-03-02

### Added
- Core EDS formula: `EDS(a) = w₁·D(a) + w₂·C(a) + w₃·V(a) + w₄·Ctx(a)`
- Four ethical philosophy evaluators (deontological, consequentialist, virtue ethics, contextual)
- Domain-aware weight configuration (healthcare, finance, hiring, disaster, general)
- Neuro-symbolic fusion agent combining LLM reasoning with symbolic rules
- Knowledge graph with domain-specific ethical ontologies
- Precedent store for case-based reasoning
- Decision tracing and audit logging
- Explainability module (natural-language explanations, visualizations)
- Conflict resolution between philosophies
- 425 scenario test cases across 4 domains
- Evaluation framework with metrics, baselines, ablation studies, and statistical analysis
- Benchmark suite (cross-domain, performance, reproducibility)
- Adversarial testing module (perturbation, jailbreak, robustness)
- Streamlit dashboard for interactive exploration
- Docker support with multi-stage builds
- GitHub Actions CI pipeline
