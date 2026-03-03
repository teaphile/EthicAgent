# Contributing to EthicAgent

Thanks for your interest in contributing!

## Getting Started

1. Fork and clone the repository
2. Install dev dependencies: `make dev`
3. Create a feature branch: `git checkout -b feature/my-feature`
4. Run tests: `make test`
5. Submit a pull request

## Code Style

- Follow PEP 8 / ruff defaults
- Add docstrings to public functions
- Type-hint all function signatures
- Keep functions focused and short

## Testing

- All new features must include tests
- Target 85%+ coverage
- Use `pytest` with fixtures from `conftest.py`

## Domain-Specific Rules

When adding scenarios or modifying ethics evaluators:

- **Domain weights must sum to 1.0**
- **EDS formula: `EDS(a) = w₁·D(a) + w₂·C(a) + w₃·V(a) + w₄·Ctx(a)`**
- **Thresholds:** ≥0.80 AUTO_APPROVE, [0.50, 0.80) ESCALATE, <0.50 REJECT, D(a)=0 HARD_BLOCK
- Scenario case IDs must be globally unique

## Commit Messages

Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `ci:`
