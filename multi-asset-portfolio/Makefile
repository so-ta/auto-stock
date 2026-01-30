# Multi-Asset Portfolio Makefile
# ==============================
# Common development commands

.PHONY: help lint lint-fix type-check format test test-cov clean install all static-analysis

# Default target
help:
	@echo "Multi-Asset Portfolio Development Commands"
	@echo "==========================================="
	@echo ""
	@echo "Static Analysis:"
	@echo "  make lint           - Run all linters (mypy, ruff, vulture)"
	@echo "  make lint-fix       - Run ruff with auto-fix"
	@echo "  make type-check     - Run mypy type checker only"
	@echo "  make static-analysis - Run full static analysis with report"
	@echo ""
	@echo "Formatting:"
	@echo "  make format         - Format code with black and isort"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all tests"
	@echo "  make test-cov       - Run tests with coverage"
	@echo "  make test-unit      - Run unit tests only"
	@echo ""
	@echo "Other:"
	@echo "  make install        - Install package in development mode"
	@echo "  make clean          - Remove build artifacts"
	@echo "  make all            - Run format, lint, and test"

# =============================================================================
# Static Analysis
# =============================================================================

lint:
	@echo "Running static analysis..."
	python -m mypy src/
	python -m ruff check src/
	python -m vulture src/ --min-confidence 80

lint-fix:
	@echo "Running ruff with auto-fix..."
	python -m ruff check --fix src/

type-check:
	@echo "Running mypy..."
	python -m mypy src/ --show-error-codes

static-analysis:
	@echo "Running full static analysis..."
	python scripts/run_static_analysis.py --report

# =============================================================================
# Formatting
# =============================================================================

format:
	@echo "Formatting code..."
	python -m black src/ tests/ scripts/
	python -m isort src/ tests/ scripts/

# =============================================================================
# Testing
# =============================================================================

test:
	@echo "Running tests..."
	python -m pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

test-unit:
	@echo "Running unit tests..."
	python -m pytest tests/unit/ -v

# =============================================================================
# Installation & Cleanup
# =============================================================================

install:
	@echo "Installing in development mode..."
	pip install -e ".[dev]"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# =============================================================================
# Combined Commands
# =============================================================================

all: format lint test
	@echo "All checks passed!"
