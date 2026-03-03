.PHONY: help install dev test lint format clean dashboard docker-build docker-up bench

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

dev:  ## Install with dev dependencies
	pip install -e ".[dev]"

test:  ## Run tests with coverage
	pytest tests/ --tb=short --cov=ethicagent --cov-report=term-missing -q

lint:  ## Run linter
	ruff check ethicagent/ tests/

format:  ## Auto-format code
	ruff format ethicagent/ tests/

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

dashboard:  ## Launch Streamlit dashboard
	streamlit run dashboard/app.py

docker-build:  ## Build Docker image
	docker compose build

docker-up:  ## Start all services
	docker compose up -d

bench:  ## Run benchmark suite
	python -c "from ethicagent.benchmarks import BenchmarkSuite; s=BenchmarkSuite(); print(s.run_all())"
