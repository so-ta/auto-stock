# Multi-Asset Portfolio Makefile
# ==============================
# Common development commands

.PHONY: help lint lint-fix type-check format test test-cov clean install all
.PHONY: setup viewer backtest backtest-test backtest-daily backtest-weekly backtest-monthly results stats

# Python executable (use venv)
PYTHON := .venv/bin/python
PIP := $(PYTHON) -m pip

# Default port and host for web viewer
PORT ?= 8080
HOST ?= 0.0.0.0

# Default target
help:
	@echo "Multi-Asset Portfolio Development Commands"
	@echo "==========================================="
	@echo ""
	@echo "Quick Start:"
	@echo "  make setup          - Setup venv and install all dependencies"
	@echo "  make viewer         - Start Web Viewer (http://localhost:8080)"
	@echo "  make backtest-test  - Run quick test backtest"
	@echo ""
	@echo "Backtest:"
	@echo "  make backtest       - Run monthly backtest (default)"
	@echo "  make backtest-test  - Quick test (5 symbols, 1 year)"
	@echo "  make backtest-daily - Daily rebalance backtest"
	@echo "  make backtest-weekly - Weekly rebalance backtest"
	@echo "  make backtest-monthly - Monthly rebalance backtest"
	@echo ""
	@echo "Results:"
	@echo "  make viewer         - Start Web Viewer (PORT=8080)"
	@echo "  make results        - List saved archives"
	@echo "  make stats          - Show storage statistics"
	@echo ""
	@echo "Static Analysis:"
	@echo "  make lint           - Run all linters (mypy, ruff, vulture)"
	@echo "  make lint-fix       - Run ruff with auto-fix"
	@echo "  make type-check     - Run mypy type checker only"
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

# =============================================================================
# Setup
# =============================================================================

setup:
	@echo "Setting up environment..."
	@if [ ! -d ".venv" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv .venv; \
	fi
	$(PIP) install -e ".[dev]"
	$(PIP) install fastapi uvicorn
	@echo ""
	@echo "Setup complete! Run 'make viewer' to start the web viewer."

# =============================================================================
# Web Viewer
# =============================================================================

viewer:
	@echo "Starting Web Viewer on http://$(HOST):$(PORT)"
	@echo "API Docs: http://$(HOST):$(PORT)/docs"
	@echo "Press Ctrl+C to stop"
	@echo ""
	$(PYTHON) -m scripts.result_viewer.app --host $(HOST) --port $(PORT)

# =============================================================================
# Backtest
# =============================================================================

backtest: backtest-monthly

backtest-test:
	@echo "Running test backtest (5 symbols, 1 year)..."
	$(PYTHON) scripts/run_backtest.py -t

backtest-daily:
	@echo "Running daily backtest..."
	$(PYTHON) scripts/run_backtest.py -f daily

backtest-weekly:
	@echo "Running weekly backtest..."
	$(PYTHON) scripts/run_backtest.py -f weekly

backtest-monthly:
	@echo "Running monthly backtest..."
	$(PYTHON) scripts/run_backtest.py -f monthly

# =============================================================================
# Results
# =============================================================================

results:
	@$(PYTHON) scripts/backtest_results.py list

stats:
	@$(PYTHON) scripts/backtest_results.py stats
