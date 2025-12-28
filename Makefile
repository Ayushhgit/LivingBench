# LivingBench Makefile
# Common commands for development and operations

.PHONY: help install install-dev install-all lint format test coverage \
        docker-build docker-run docker-up docker-down clean \
        run-experiment run-groq docs

# Default target
help:
	@echo "LivingBench Development Commands"
	@echo "================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install base dependencies"
	@echo "  make install-dev    Install with dev dependencies"
	@echo "  make install-all    Install all dependencies (dev + providers)"
	@echo "  make setup-hooks    Install pre-commit hooks"
	@echo ""
	@echo "Development:"
	@echo "  make lint           Run linter (ruff)"
	@echo "  make format         Format code (ruff)"
	@echo "  make test           Run tests"
	@echo "  make coverage       Run tests with coverage report"
	@echo "  make typecheck      Run type checker (mypy)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-run     Run container interactively"
	@echo "  make docker-up      Start all services (docker-compose)"
	@echo "  make docker-down    Stop all services"
	@echo ""
	@echo "Experiments:"
	@echo "  make run-experiment Run default experiment"
	@echo "  make run-groq       Run Groq experiment"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          Clean build artifacts"
	@echo "  make docs           Generate documentation"

# =============================================================================
# Setup
# =============================================================================

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

setup-hooks:
	pip install pre-commit
	pre-commit install
	pre-commit install --hook-type commit-msg

# =============================================================================
# Development
# =============================================================================

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest tests/ -v

coverage:
	pytest tests/ -v --cov=livingbench --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

typecheck:
	mypy livingbench/ --ignore-missing-imports

check: lint typecheck test
	@echo "All checks passed!"

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker build -t livingbench:latest .

docker-run:
	docker run -it --rm \
		-v $(PWD)/outputs:/app/outputs \
		-v $(PWD)/.env:/app/.env:ro \
		livingbench:latest bash

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# =============================================================================
# Experiments
# =============================================================================

run-experiment:
	python -m experiments.run_experiment --n-tasks 50

run-groq:
	python scripts/run_groq_experiment.py --n-tasks 50 --models llama-3.3-70b-versatile

run-groq-compare:
	python scripts/run_groq_experiment.py --n-tasks 100 \
		--models llama-3.3-70b-versatile llama-3.1-8b-instant

# =============================================================================
# MLflow
# =============================================================================

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

mlflow-server:
	mlflow server \
		--host 0.0.0.0 \
		--port 5000 \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlflow-artifacts

# =============================================================================
# Utilities
# =============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-outputs:
	rm -rf outputs/*

clean-cache:
	rm -rf .cache/

clean-all: clean clean-outputs clean-cache

# Generate documentation
docs:
	@echo "Documentation generation not yet configured"
	@echo "Consider adding: mkdocs, sphinx, or pdoc"

# Show project stats
stats:
	@echo "Project Statistics"
	@echo "=================="
	@echo "Python files: $$(find livingbench -name '*.py' | wc -l)"
	@echo "Test files: $$(find tests -name '*.py' | wc -l)"
	@echo "Lines of code: $$(find livingbench -name '*.py' -exec cat {} + | wc -l)"
	@echo "Lines of tests: $$(find tests -name '*.py' -exec cat {} + | wc -l)"
