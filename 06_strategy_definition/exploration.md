# Chapter 5: Strategy Definition - Codebase Analysis

**Generated**: 2026-01-25
**Analysis Type**: Deep Architecture & Pattern Review

---

## Executive Summary

Chapter 5 is a **mature, well-architected** educational codebase demonstrating strategy pre-specification. The code quality is high with consistent patterns, strong documentation, and proper separation of concerns. The architecture follows a **hypothesis-driven exploration** pattern rather than a production trading system pattern.

**Overall Quality**: ★★★★☆ (4/5) - Publication-ready with minor improvements possible

---

## Architecture Overview

```
06_strategy_definition/
├── chapter/                    # Book manuscript (section-based workflow)
│   ├── draft.md               # Auto-generated from sections
│   ├── summary.md             # Chapter scope definition
│   ├── bibliography.md        # Academic references
│   └── sections/              # 10 section files (ground truth)
│
├── code/                      # 5 Jupytext notebooks (~4,500 LOC)
│   ├── 01_etf_momentum.py     # ETF correlation, regime, rotation (858 lines)
│   ├── 02_crypto_premium.py   # Funding rate mean reversion (792 lines)
│   ├── 03_algoseek_intraday.py # Microstructure exploration (817 lines)
│   ├── 04_aqr_factor_performance.py # Century of factor evidence (1,395 lines)
│   └── 05_strategy_term_sheet_template.py # Python dataclasses (694 lines)
│
├── term_sheets/               # 8 completed strategy specifications
├── catalog/                   # 5 notebook metadata JSON files
├── figures/                   # Book figures (AI-generated + notebook)
└── reviews/                   # Code review artifacts
```

### Pattern: Hypothesis-Driven Exploration

The codebase follows a clear pedagogical pattern:

```
1. Data Contract (documented assumptions)
      ↓
2. Configuration Dataclass (frozen, versioned)
      ↓
3. Exploratory Analysis (correlation, distribution, regime)
      ↓
4. Event Study (with cooldown, forward returns)
      ↓
5. Key Priors Summary (for Term Sheet)
```

---

## Code Quality Assessment

### Strengths

| Aspect | Score | Evidence |
|--------|-------|----------|
| **Documentation** | ★★★★★ | Every notebook has purpose, data contract, assumptions, limitations |
| **Type Safety** | ★★★★☆ | Frozen dataclasses, type hints in key functions |
| **Polars Usage** | ★★★★★ | Proper lazy API, `.over()` windows, no pandas except boundaries |
| **Time Handling** | ★★★★★ | Time-based joins (not shifts), timezone-aware, tolerance parameters |
| **Point-in-Time** | ★★★★☆ | Rolling z-scores, macro lag handling, clear look-ahead warnings |

### Patterns Observed

#### 1. Configuration Dataclasses (Excellent)

```python
@dataclass(frozen=True)
class EtfExplorationConfig:
    """Configuration for ETF momentum exploration (Chapter 2)."""
    start_date_str: str = "2007-01-01"
    trading_days_per_year: int = 252
    momentum_formation_days: int = 126  # 6 months
    skip_month_days: int = 21  # Skip most recent month
    # ... etc
```

**Benefit**: Immutable, documented, single source of truth for notebook parameters.

#### 2. Time-Based Forward Returns (Production-Quality)

```python
def _compute_forward_returns_hourly(
    df: pl.DataFrame,
    *,
    price_col: str,
    horizons_hours: tuple[int, ...],
    group_col: str,
    ts_col: str,
    tolerance_hours: int = 2,
) -> pl.DataFrame:
```

**Pattern**: Uses `join_asof` with tolerance instead of `.shift()` to handle gaps correctly.

#### 3. Event Study with Proper Cooldown

```python
def event_study_with_cooldown(
    df: pl.DataFrame,
    regime_col: str = "regime",
    cooldown_hours: int = 24,
) -> pl.DataFrame:
```

**Pattern**: Stateful thinning that compares to last KEPT event, not previous event.

#### 4. Strategy Term Sheet as Python Dataclass

```python
@dataclass
class StrategyTermSheet:
    name: str
    version: str
    status: Literal["Draft", "Review", "Approved"]
    classification: Literal["Price-Based", "Fundamental", "Structural", "ML-Enhanced"]
    hypothesis: FalsifiableHypothesis
    blueprint: ImplementationBlueprint
    feasibility: FeasibilityGate
    validation: ValidationPlan
```

**Benefit**: Enforces structure, enables JSON/YAML export, supports version control.

---

## Component Inventory

### Notebooks (5)

| Notebook | Purpose | Data Sources | Lines | Quality |
|----------|---------|--------------|-------|---------|
| `01_etf_momentum` | Correlation, regime, rotation | ETF Universe, FRED | 858 | ★★★★★ |
| `02_crypto_premium` | Funding rate mean reversion | Binance Premium/OHLCV | 792 | ★★★★★ |
| `03_algoseek_intraday` | Microstructure exploration | AlgoSeek NASDAQ100 | 817 | ★★★★☆ |
| `04_aqr_factor_performance` | Century of factor evidence | AQR, Fama-French | 1,395 | ★★★★★ |
| `05_strategy_term_sheet_template` | Interactive Term Sheet | None | 694 | ★★★★★ |

### Term Sheets (8)

| Term Sheet | Classification | Status |
|------------|----------------|--------|
| `etf_momentum__rotational_momentum.md` | Price-Based | Complete |
| `crypto_premium__premium_funding_reversal.md` | Structural | Complete |
| `nasdaq100_reversal__intraday_orderflow_reversal.md` | Structural | Complete |
| `us_factors__cross_sectional_factor_momentum.md` | Price-Based | Complete |
| `futures_carry__term_structure_momentum.md` | Structural | Complete |
| `fx_momentum__cross_sectional_momentum.md` | Price-Based | Complete |
| `algoseek_sp500__options_volatility_alpha.md` | Structural | Complete |
| `etf_momentum.md` | Price-Based | Legacy (use etf_momentum version) |

### Chapter Sections (10)

```
00_preamble.md
01_the_strategy_specification_problem.md
02_a_map_of_strategies_and_edges.md
03_the_strategy_term_sheet.md
04_filling_the_term_sheet.md
05_minimum_specification_feasibility.md
06_validation_plan_and_failure_conditions.md
07_where_ideas_come_from.md
08_evidence_discipline_and_the_factor_zoo.md
09_key_takeaways.md
```

---

## Data Flow Architecture

```
                    ┌─────────────────┐
                    │   Data Layer    │
                    │ (utils, DATA_DIR)│
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  ETF Universe │  │ Crypto Premium│  │   AlgoSeek    │
│   + FRED      │  │   + OHLCV     │  │  Minute Bars  │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────────────────────────────────────────┐
│              Exploration Notebooks                │
│  (correlation, regime, event study, statistics)   │
└───────────────────────┬───────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────┐
│               Strategy Term Sheets                │
│    (hypothesis, blueprint, feasibility, validation)│
└───────────────────────────────────────────────────┘
                        │
                        ▼
               Downstream Chapters
           (Ch7 features, Ch17 backtest)
```

---

## Statistical Methods Used

### Factor Analysis (04_aqr_factor_performance.py)

| Method | Implementation | Quality |
|--------|----------------|---------|
| Newey-West HAC t-stat | `calculate_mean_return_tstat()` | ★★★★★ |
| Lo (2002) Sharpe SE | `calculate_sharpe_stats()` | ★★★★★ |
| Max Drawdown | `calculate_max_drawdown()` | ★★★★☆ |
| Rolling Correlation | `pl.rolling_corr()` native | ★★★★★ |

**Notable**: Proper distinction between Harvey et al. (2016) mean-return t-stat (for factor discovery) and Lo (2002) Sharpe ratio significance (for performance).

### Momentum Analysis (01_etf_momentum.py)

| Method | Implementation | Quality |
|--------|----------------|---------|
| Cross-sectional quintile ranking | `.rank("ordinal").over("timestamp")` | ★★★★★ |
| 6-1 Momentum (skip-month) | Vectorized with `.shift()` | ★★★★★ |
| Yield curve regime filter | FRED data with 2-bday lag | ★★★★★ |
| Rotation simulation | Month-by-month loop (pedagogical) | ★★★★☆ |

### Event Studies (02_crypto_premium.py, 03_algoseek_intraday.py)

| Method | Implementation | Quality |
|--------|----------------|---------|
| Rolling z-score | Point-in-time (168h window) | ★★★★★ |
| Event cooldown | Stateful per-group thinning | ★★★★★ |
| Forward returns | Time-based join with tolerance | ★★★★★ |
| Winsorization | Global quantiles (exploration only) | ★★★★☆ |

---

## Improvement Recommendations

### Priority 1: Minor Code Improvements

1. **Add `__all__` exports to notebooks** for potential module reuse:
   ```python
   __all__ = ["EtfExplorationConfig", "calculate_momentum_score"]
   ```

2. **Extract common utilities** to shared module:
   - `_compute_forward_returns_*` functions appear in multiple notebooks
   - `_event_cooldown_filter` is duplicated
   - Could live in `utils/exploration.py`

3. **Add catalog entry for `etf_momentum.md` legacy term sheet** or remove:
   - Currently `etf_momentum.md` exists alongside `etf_momentum__rotational_momentum.md`
   - Confusing which is canonical

### Priority 2: Documentation Enhancements

1. **Add notebook execution order** to README.md:
   ```markdown
   ## Execution Order
   1. 01_etf_momentum.py (ETF data exploration)
   2. 02_crypto_premium.py (crypto case study)
   ...
   ```

2. **Document term sheet naming convention**:
   - Pattern: `{dataset}__{strategy_type}.md`
   - Examples: `etf_momentum__rotational_momentum.md`, `crypto_premium__premium_funding_reversal.md`

### Priority 3: Testing Coverage

1. **Add unit tests for statistical functions**:
   - `calculate_mean_return_tstat()` - verify against known values
   - `calculate_sharpe_stats()` - test Lo (2002) SE calculation
   - `_event_cooldown_filter()` - edge cases

2. **Add data contract validation tests**:
   - Verify parquet schemas match documented contracts
   - Check for expected columns and dtypes

---

## Risk Areas

### Low Risk (Acceptable)

| Area | Observation | Mitigation |
|------|-------------|------------|
| Pandas boundary | Matplotlib/scipy require pandas | Minimal, conversion at boundary only |
| Global winsorization | Uses full-sample quantiles | Documented as exploration-only |
| Loop in rotation sim | Month-by-month for clarity | Acceptable for pedagogy |

### Medium Risk (Monitor)

| Area | Observation | Mitigation |
|------|-------------|------------|
| Duplicate term sheets | `etf_momentum.md` vs `etf_momentum__*.md` | Clarify canonical version |
| Hardcoded magic numbers | Some thresholds in notebook body | Move to config dataclass |
| Missing type hints | Helper functions lack full typing | Add for production reuse |

---

## Dependency Analysis

### External Dependencies

```python
# Core
polars          # DataFrames (primary)
numpy           # Numerical operations
pandas          # Boundary conversions only
plotly          # Visualization

# Domain
scipy.stats     # Statistical tests
IPython.display # Jupyter rendering

# ML4T
utils           # DATA_DIR, paths
ml4t.data.providers  # AQR, Fama-French
```

### Internal Dependencies

```
05_strategy_term_sheet_template.py
    ↓ exports
StrategyTermSheet (dataclass)
    ↓ used by
term_sheets/*.md (via to_markdown())
```

---

## Alignment with Book Standards

| Standard | Compliance | Notes |
|----------|------------|-------|
| Polars-first | ✅ Full | No pandas except visualization boundaries |
| Frozen config dataclass | ✅ Full | All notebooks use this pattern |
| Time-based joins | ✅ Full | No `.shift()` for forward returns |
| Data contracts documented | ✅ Full | Every notebook has contract table |
| Output to `figures/` only | ✅ Full | No ad-hoc file saves |
| No code in chapter text | ✅ Full | References notebooks by name |
| TEST mode support | ⚠️ Partial | Not all notebooks have explicit TEST guards |

---

## Summary

Chapter 5 demonstrates **exemplary educational code architecture**:

1. **Clear separation**: Data contracts → Config → Analysis → Summary
2. **Proper statistics**: Newey-West HAC, Lo Sharpe SE, time-based forward returns
3. **Point-in-time discipline**: Rolling z-scores, macro lag handling
4. **Versioned artifacts**: Term sheets as structured documents

**Recommended actions**:
- Extract common utilities to reduce duplication
- Clarify canonical term sheet naming
- Add unit tests for statistical functions

**Overall**: Ready for publication with minor cleanup.

---

*Analysis performed by Claude Code /development:analyze*
