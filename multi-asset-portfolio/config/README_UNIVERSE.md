# Universe Configuration Guide

## Overview

This directory contains universe configuration files that define the investable asset universe for the multi-asset portfolio system.

## File Structure

```
config/
├── universe.yaml           # System default configuration
├── universe_standard.yaml  # Standard backtest universe (cmd_029)
└── universe_full.yaml      # Complete ticker list for data fetching
```

> **Note**: `universe_optimized.yaml` was removed in cmd_029 cleanup (task_029_10).
> Use `universe_standard.yaml` for all standardized backtests.

## File Descriptions

### 1. universe.yaml (Default Configuration)

**Purpose**: System-wide default universe configuration

**Used by**:
- `config/default.yaml` - Main configuration file
- `src/config/settings.py` - Default universe path
- `src/orchestrator/pipeline.py` - Pipeline execution

**Format**: Structured configuration with enabled/disabled flags per category

```yaml
universe:
  us_stocks:
    enabled: true
    source: "sp500"
    default_tickers: [...]
  japan_stocks:
    enabled: true
    ...
```

**When to use**: Default system runs, development, testing

---

### 2. universe_standard.yaml (Standard Backtest Universe)

**Purpose**: Quality-filtered universe for standardized backtests (cmd_029 specification)

**Used by**:
- `scripts/run_standard_backtest.py` - Standard backtest runner
- `scripts/build_standard_cache.py` - Cache builder

**Statistics**:
- Total symbols: 828
- US stocks: 471
- Japan stocks: 219
- ETFs: 103
- Forex: 35

**Format**: Flat symbol list organized by category

```yaml
version: '1.0'
name: Standard Universe
total_symbols: 828
us_stocks:
  symbols: [AAL, AAPL, ...]
```

**When to use**: Standardized performance comparison, production backtests

---

### 3. universe_full.yaml (Complete Ticker List)

**Purpose**: Complete universe including all available tickers before filtering

**Used by**:
- `scripts/fetch_universe_lists.py` - Generates this file
- `scripts/fetch_full_universe.py` - Data fetcher
- `scripts/run_daily_backtest_15y.py` - 15-year daily backtest

**Statistics**:
- US stocks: 501 (S&P 500)
- Japan stocks: 233 (Nikkei 225)
- ETFs: 119
- Forex: 37
- Crypto: 15

**Format**: Structured with metadata

```yaml
universe:
  us_stocks:
    enabled: true
    count: 501
    tickers: [...]
```

**When to use**: Data fetching, cache building, full universe analysis

---

### 4. universe_optimized.yaml (Optimization Template)

**Purpose**: Configuration template for optimized universe selection

**Used by**:
- `src/data/universe_loader.py` - Universe loading

**Features**:
- Configurable filters (volume, market cap, price range)
- Sector diversification constraints
- Regional allocation targets
- Correlation-based exclusion

**Format**: Configuration parameters

```yaml
universe:
  max_tickers: 150
  filters:
    min_daily_volume: 10000000
    min_market_cap: 1000000000
    max_sector_weight: 0.15
```

**When to use**: Experimental backtests, parameter optimization

---

## Selection Guide

| Use Case | Recommended File |
|----------|------------------|
| Standard backtest | `universe_standard.yaml` |
| Development/Testing | `universe.yaml` |
| Data fetching | `universe_full.yaml` |
| Optimization experiments | `universe_optimized.yaml` |

## Quality Filters Applied to Standard Universe

The standard universe (universe_standard.yaml) has been filtered with:

- Maximum missing data rate: 5%
- Maximum OHLC error rate: 1%
- Minimum average volume: 100,000
- Minimum trading days: 500

## Maintenance Notes

1. **universe_full.yaml**: Regenerate periodically with `scripts/fetch_universe_lists.py`
2. **universe_standard.yaml**: Update when index constituents change
3. **universe.yaml**: Edit for system-wide default changes
4. **universe_optimized.yaml**: Modify for experimental settings

## Deprecated Files (Removed)

- `universe_filtered.yaml` - Intermediate output, replaced by `universe_standard.yaml`

---

*Last updated: 2026-01-30*
*Task: task_032_15 (cmd_032)*
