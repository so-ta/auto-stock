# Universe Configuration Guide (v3.0)

## Overview

The multi-asset portfolio system uses a **single unified configuration file** (`asset_master.yaml`) for all asset universe definitions.

## File Structure

```
config/
├── asset_master.yaml      # Single source of truth for all assets
├── default.yaml           # System configuration (references asset_master.yaml)
└── README_UNIVERSE.md     # This file
```

## asset_master.yaml Structure

```yaml
version: '3.0'
name: "Asset Master"
description: "Single source of truth for all tradeable assets"

# Taxonomy definitions (extensible via YAML only)
taxonomy:
  market:
    us: "US Markets"
    japan: "Japanese Markets"
    ...
  asset_class:
    equity: "Individual Stocks"
    etf: "Exchange Traded Funds"
    ...
  sector:
    technology: "Technology"
    healthcare: "Healthcare"
    ...

# Symbol list with taxonomy and tags
symbols:
  - ticker: AAPL
    taxonomy:
      market: us
      asset_class: equity
      sector: technology
    tags: [sp500, sbi, quality]

  - ticker: 7203.T
    taxonomy:
      market: japan
      asset_class: equity
    tags: [nikkei225, sbi, quality]

# Named subsets for filtering
subsets:
  standard:
    name: "Standard Universe"
    description: "Quality-filtered subset for backtesting"
    filters:
      tags: [quality]

  japan:
    name: "Japan Universe"
    filters:
      taxonomy:
        market: [japan]
```

## Available Subsets

| Subset | Description | Filters |
|--------|-------------|---------|
| `standard` | Quality-filtered backtesting universe | `tags: [quality]` |
| `sbi` | SBI Securities tradeable | `tags: [sbi]` |
| `japan` | Japanese market stocks | `taxonomy.market: [japan]` |
| `us_equity` | US individual stocks | `taxonomy: {market: [us], asset_class: [equity]}` |
| `etf_only` | All ETFs | `taxonomy.asset_class: [etf]` |

## Usage in Code

### Basic Usage

```python
from src.data.universe_loader import UniverseLoader

loader = UniverseLoader()

# Get all symbols
all_symbols = loader.get_all_symbols()

# Get a named subset
standard = loader.get_subset("standard")
japan = loader.get_subset("japan")

# Get tickers only
tickers = loader.get_subset_tickers("standard")
```

### Filtering by Criteria

```python
# Filter by taxonomy
us_tech = loader.filter_symbols(
    taxonomy={"market": ["us"], "sector": ["technology"]}
)

# Filter by tags
sbi_tradeable = loader.filter_symbols(tags=["sbi"])

# Exclude certain tags
active = loader.filter_symbols(exclude_tags=["delisted", "illiquid"])
```

### Report Generation

```python
# Group symbols by taxonomy key
by_sector = loader.group_by_taxonomy(symbols, "sector")
for sector, syms in by_sector.items():
    label = loader.get_taxonomy_label("sector", sector)
    print(f"{label}: {len(syms)} symbols")
```

### Backward Compatibility

```python
# Legacy format (still supported)
legacy = loader.load_standard_universe()
# Returns: {"us_stocks": [...], "japan_stocks": [...], "etfs": [...]}
```

## Configuration in default.yaml

```yaml
universe:
  master_file: "config/asset_master.yaml"
  default_subset: "standard"
  max_assets: 500
```

## Extending Taxonomy

To add a new taxonomy key (e.g., dividend yield):

1. Add to `taxonomy` section:
```yaml
taxonomy:
  dividend_yield:
    high: "High Yield (>4%)"
    medium: "Medium Yield (2-4%)"
    low: "Low Yield (<2%)"
```

2. Add to symbols:
```yaml
symbols:
  - ticker: AAPL
    taxonomy:
      dividend_yield: low
```

3. Use in code (no code changes needed):
```python
by_yield = loader.group_by_taxonomy(symbols, "dividend_yield")
```

## Migration from v1.0/v2.0

The old universe files have been consolidated:

| Old File | New Location |
|----------|--------------|
| `universe.yaml` | `asset_master.yaml` |
| `universe_standard.yaml` | `asset_master.yaml` (subset: `standard`) |
| `universe_sbi.yaml` | `asset_master.yaml` (subset: `sbi`) |
| `universe_japan_all.yaml` | `asset_master.yaml` (subset: `japan`) |

---

*Last updated: 2026-02-01*
*Version: 3.0*
