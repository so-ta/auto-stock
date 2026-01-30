# Multi-Asset Portfolio

> Automated multi-asset portfolio management system with dynamic allocation, walk-forward validation, and multi-strategy ensemble.

[![CI](https://github.com/YOUR_ORG/multi-asset-portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_ORG/multi-asset-portfolio/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Multi-Asset Portfolio is a quantitative portfolio management system designed for:

- **Multi-asset allocation** across stocks, ETFs, forex, and commodities
- **Walk-forward validation** to prevent overfitting
- **Dynamic rebalancing** based on market regimes
- **Risk management** with drawdown protection and fallback modes

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **HRP Allocation** | Hierarchical Risk Parity for robust diversification |
| **Risk Parity** | Equal risk contribution across assets |
| **Walk-Forward** | Rolling train/test validation |
| **Multi-Strategy** | Ensemble of momentum, mean-reversion, macro signals |
| **Regime Detection** | Adaptive parameters based on market conditions |

### Advanced Features

- **Numba JIT acceleration** for 5-10x faster calculations
- **Multiple backtest engines** (Fast, Streaming, VectorBT-style)
- **Transaction cost optimization** with turnover constraints
- **Drawdown protection** with staged risk reduction
- **Fallback modes** for system resilience

## Installation

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/YOUR_ORG/multi-asset-portfolio.git
cd multi-asset-portfolio

# Install with pip
pip install -e ".[dev]"

# Or with uv (faster)
uv pip install -e ".[dev]"
```

### Verify Installation

```bash
# Run tests
make test

# Check lint
make lint
```

## Quick Start

### 1. Basic Backtest

```bash
# Run backtest with default universe
python -m src.main --backtest --start 2020-01-01 --end 2024-12-31

# Run with specific assets
python -m src.main --backtest --universe SPY,QQQ,TLT,GLD,BTC-USD
```

### 2. Using Custom Configuration

```bash
# Use default config
python -m src.main --config config/default.yaml

# Use local overrides
cp config/default.yaml config/local.yaml
# Edit config/local.yaml
python -m src.main --config config/local.yaml
```

### 3. Python API

```python
from src.backtest.fast_engine import FastBacktestEngine, FastBacktestConfig
from datetime import datetime

# Configure
config = FastBacktestConfig(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 12, 31),
    initial_capital=100000.0,
    rebalance_frequency="monthly",
)

# Run backtest
engine = FastBacktestEngine(config)
result = engine.run(prices_df)

# View results
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
print(f"Total Return: {result.total_return:.2%}")
```

## CLI Usage

```bash
# Show help
python -m src.main --help

# Backtest mode
python -m src.main --backtest [OPTIONS]

# Live mode (paper trading)
python -m src.main --live [OPTIONS]

# Options:
#   --config PATH       Path to YAML config file
#   --universe ASSETS   Comma-separated asset list
#   --start DATE        Start date (YYYY-MM-DD)
#   --end DATE          End date (YYYY-MM-DD)
#   --capital AMOUNT    Initial capital
#   --output PATH       Output directory for results
```

## Configuration

Configuration is managed via YAML files in `config/`:

```
config/
├── default.yaml      # Default settings (source of truth)
├── local.yaml        # Local overrides (gitignored)
├── universe.yaml     # Asset universe definition
└── universe_full.yaml # Full universe (800+ assets)
```

### Key Configuration Sections

```yaml
# Rebalancing
rebalance:
  frequency: "monthly"        # weekly | monthly | quarterly
  min_trade_threshold: 0.02   # Skip trades < 2%

# Walk-Forward Validation
walk_forward:
  train_period_days: 504      # ~2 years training
  test_period_days: 126       # ~6 months testing
  purge_gap_days: 5           # Gap to prevent leakage

# Risk Management
hard_gates:
  min_sharpe_ratio: 0.5
  max_drawdown_pct: 25.0
  min_win_rate_pct: 45.0

# Asset Allocation
asset_allocation:
  method: "HRP"               # HRP | risk_parity | mean_variance
  w_asset_max: 0.2            # Max 20% per asset
```

See `config/default.yaml` for full configuration options.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline Orchestrator                     │
│                   (src/orchestrator/pipeline.py)             │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│     Data      │    │    Signals    │    │  Allocation   │
│  (src/data/)  │    │ (src/signals/)│    │(src/allocation│
│               │    │               │    │               │
│ - Fetchers    │    │ - Momentum    │    │ - HRP         │
│ - Quality     │    │ - Reversion   │    │ - Risk Parity │
│ - Cache       │    │ - Regime      │    │ - CVaR        │
└───────────────┘    └───────────────┘    └───────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            ┌───────────────┐    ┌───────────────┐
            │   Backtest    │    │     Risk      │
            │(src/backtest/)│    │  (src/risk/)  │
            │               │    │               │
            │ - Fast Engine │    │ - Drawdown    │
            │ - Streaming   │    │ - VaR/CVaR    │
            │ - VectorBT    │    │ - Stress Test │
            └───────────────┘    └───────────────┘
```

### Module Overview

| Module | Purpose |
|--------|---------|
| `src/data/` | Data fetching, caching, quality checks |
| `src/signals/` | Signal generation (momentum, reversion, macro) |
| `src/allocation/` | Portfolio allocation algorithms |
| `src/backtest/` | Backtest engines |
| `src/orchestrator/` | Pipeline coordination |
| `src/risk/` | Risk metrics and management |
| `src/strategy/` | Strategy evaluation and gates |
| `src/config/` | Configuration management |

## Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/unit/test_fast_engine.py -v

# Run integration tests
pytest tests/integration/ -v
```

### Test Structure

```
tests/
├── unit/              # Unit tests for individual modules
├── integration/       # Integration tests for engine compatibility
└── conftest.py        # Shared fixtures
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Ensure package is installed in development mode
pip install -e ".[dev]"
```

#### 2. Numba Threading Error

```
ValueError: No threading layer could be loaded
```

Solution:
```bash
pip install tbb  # or intel-openmp
```

#### 3. Data Fetching Fails

- Check internet connection
- Verify API rate limits
- Check `config/universe.yaml` for valid tickers

#### 4. Memory Issues with Large Universe

```yaml
# In config/default.yaml, reduce:
universe:
  max_assets: 100  # Reduce from 500
```

### Debug Mode

```bash
# Enable debug logging
export PORTFOLIO_LOG_LEVEL=DEBUG
python -m src.main --backtest
```

## Performance Tips

1. **Use Numba** - Enable `use_numba: true` in config for 5-10x speedup
2. **Cache Data** - Enable data caching to avoid repeated API calls
3. **Reduce Universe** - Limit assets for faster iteration during development
4. **Use FastBacktestEngine** - Fastest engine for most use cases

## Dedicated Server Setup

For running on a dedicated server with full resource utilization:

### 1. Dynamic Resource Configuration

The system automatically detects and utilizes available hardware resources:

```python
from src.config import get_current_resource_config, print_resource_summary

# View detected resources and calculated settings
print_resource_summary()

# Access configuration programmatically
config = get_current_resource_config()
print(f"CPU Workers: {config.max_workers}")
print(f"Cache Memory: {config.cache_max_memory_mb} MB")
print(f"GPU Available: {config.use_gpu}")
```

### 2. Resource Configuration Options

| Setting | Default (Shared) | Dedicated Server |
|---------|------------------|------------------|
| `max_workers` | CPU cores - 1 | All CPU cores |
| `cache_max_memory_mb` | 25% of available | 70% of available |
| `cache_max_entries` | 5,000 | Memory-based (auto) |
| `disable_chunking` | False | True (if RAM >= 16GB) |
| `cache_max_disk_mb` | 10% of free | Unlimited |

### 3. S3 Cache Setup (Cloud Environments)

For cloud deployments with shared cache across instances:

```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key

# Configure S3 backend in config/settings.yaml
```

```yaml
# config/settings.yaml
storage:
  backend: "s3"  # "local" or "s3"
  s3_bucket: "your-bucket-name"
  s3_prefix: ".cache"
  local_cache_enabled: true
  local_cache_path: "/tmp/.backtest_cache"
  local_cache_ttl_hours: 24
```

### 4. Migrate Existing Cache to S3

```bash
# Dry run (preview files to upload)
python scripts/migrate_cache_to_s3.py --dry-run

# Execute migration
python scripts/migrate_cache_to_s3.py --bucket your-bucket-name
```

### 5. GPU Acceleration (Optional)

For NVIDIA GPU support:

```bash
# Install CuPy (CUDA 11.x)
pip install cupy-cuda11x

# Or for CUDA 12.x
pip install cupy-cuda12x

# Verify installation
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

The system will automatically detect and use GPU when available.

### 6. Ray Distributed Processing (Optional)

For distributed processing across multiple cores/machines:

```bash
# Install Ray
pip install ray

# Run with Ray backend
python -m src.main --backtest --engine ray
```

### 7. Recommended Server Specs

| Workload | CPU | RAM | Storage | GPU |
|----------|-----|-----|---------|-----|
| Development | 4+ cores | 8GB | SSD 50GB | - |
| Production (Monthly) | 8+ cores | 16GB | SSD 100GB | Optional |
| Production (Daily) | 16+ cores | 32GB+ | SSD 200GB | Recommended |
| High-Performance | 32+ cores | 64GB+ | NVMe 500GB | Required |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`make test`)
4. Run linters (`make lint`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- Python 3.11+ type hints required
- Ruff for linting
- MyPy for type checking
- Black for formatting (via ruff)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- VectorBT for architecture inspiration
- PyPortfolioOpt for allocation algorithms
- Numba team for JIT compilation support

---

**Version**: 1.2.0 | **Last Updated**: 2026-01-30
