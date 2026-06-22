"""Chapter and data path management for ML4T notebooks.

This module provides:
1. CHAPTERS - Canonical registry of chapter numbers to directory names
2. get_chapter_dir() - Type-safe chapter path access
3. get_case_study_dir() - Centralized case study artifact store
4. get_output_dir() - Chapter-local outputs (non-case-study data)
5. CH01-CH27 - Direct chapter path constants

Naming Convention:
- Dataset IDs: etf, crypto, futures, fx, equities_us, equities_nasdaq100, options_sp500
- Strategy IDs: etfs, crypto_perps_funding, cme_futures, fx_pairs, us_equities_panel,
                nasdaq100_microstructure, sp500_equity_option_analytics, sp500_options,
                us_firm_characteristics

Usage:
    from utils.paths import get_case_study_dir

    # Write features in Ch7
    CASE_DIR = get_case_study_dir("etfs")
    features.write_parquet(CASE_DIR / "features" / "features.parquet")

    # Read features in Ch9
    CASE_DIR = get_case_study_dir("etfs")
    features = pl.read_parquet(CASE_DIR / "features" / "features.parquet")
    predictions.write_parquet(CASE_DIR / "models" / "linear" / "predictions.parquet")

    # Chapter-local outputs (benchmarks, demonstrations - NOT case studies)
    from utils.paths import get_output_dir
    OUTPUT_DIR = get_output_dir(7, "benchmark_results")
"""

from pathlib import Path

# =============================================================================
# Core Paths
# =============================================================================

REPO_ROOT = Path(__file__).parent.parent.resolve()

# =============================================================================
# Chapter Registry (Single Source of Truth)
# =============================================================================

CHAPTERS: dict[int, str] = {
    1: "01_process_is_edge",
    2: "02_financial_data_universe",
    3: "03_market_microstructure",
    4: "04_fundamental_alternative_data",
    5: "05_synthetic_data",
    6: "06_strategy_definition",
    7: "07_defining_the_learning_task",
    8: "08_financial_features",
    9: "09_model_based_features",
    10: "10_text_feature_engineering",
    11: "11_ml_pipeline",
    12: "12_gradient_boosting",
    13: "13_dl_time_series",
    14: "14_latent_factors",
    15: "15_causal_estimation",
    16: "16_strategy_simulation",
    17: "17_portfolio_construction",
    18: "18_transaction_costs",
    19: "19_risk_management",
    20: "20_strategy_synthesis",
    21: "21_rl_execution_hedging",
    22: "22_rag_financial_research",
    23: "23_knowledge_graphs",
    24: "24_autonomous_agents",
    25: "25_live_trading",
    26: "26_mlops_governance",
    27: "27_systematic_edge",
}

# Sub-chapters (not in main sequence)
SUB_CHAPTERS: dict[str, str] = {}

# Strategy IDs (Tier 3 naming)
# Updated 2026-02-12 to match case_studies/ directory structure
STRATEGY_IDS = frozenset(
    {
        "etfs",
        "crypto_perps_funding",
        "nasdaq100_microstructure",
        "sp500_equity_option_analytics",
        "us_firm_characteristics",
        "fx_pairs",
        "cme_futures",
        "sp500_options",
        "us_equities_panel",
    }
)


# Stage mapping: chapter number → subdirectory within case_studies/{strategy}/
# Used as documentation reference; notebooks use string literals directly.
# Updated 2026-02-25 for 27-chapter scheme (old Ch16+21 merged → Ch20)
CHAPTER_STAGES: dict[int, str] = {
    6: "exploration",
    7: "labels",
    8: "features",
    9: "features",  # Temporal features
    10: "features/text",
    11: "models/linear",
    12: "models/gbm",
    13: "models/deep_learning",
    14: "models/latent_factors",
    15: "models/causal",
    16: "backtest/simulation",
    17: "backtest/portfolio",
    18: "backtest/costs",
    19: "backtest/risk",
    20: "synthesis",
}


# =============================================================================
# Chapter Path Functions
# =============================================================================


def display_path(path: Path | str) -> str:
    """Return `path` relative to REPO_ROOT when possible, else as-is.

    Use in `print(...)` statements inside notebooks so the committed cell
    output never bakes in machine-specific absolute paths (e.g. `/home/<user>/...`).
    """
    p = Path(path)
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def get_chapter_dir(chapter: int | str) -> Path:
    """Get the directory path for a chapter.

    Args:
        chapter: Chapter number (1-27)

    Returns:
        Absolute path to chapter directory

    Raises:
        ValueError: If chapter number is invalid

    Examples:
        >>> get_chapter_dir(7)
        ML4T_PATH / "code/07_feature_engineering"
    """
    if isinstance(chapter, int):
        if chapter not in CHAPTERS:
            raise ValueError(f"Invalid chapter number: {chapter}. Valid: 1-27")
        return REPO_ROOT / CHAPTERS[chapter]

    # String: check sub-chapters
    if chapter in SUB_CHAPTERS:
        return REPO_ROOT / SUB_CHAPTERS[chapter]

    raise ValueError(
        f"Invalid chapter: {chapter!r}. "
        f"Use integers 1-27 or sub-chapter IDs: {list(SUB_CHAPTERS.keys())}"
    )


def get_output_dir(
    chapter: int | str,
    strategy_id: str,
    *,
    create: bool = True,
) -> Path:
    """Get output directory for cross-chapter data flow.

    This is the primary function for case study data that flows between chapters:
    - Ch7 creates features.parquet -> Ch9 reads for signal evaluation
    - Ch9 creates signals.parquet -> Ch12 reads for ML pipeline
    - Ch12 creates predictions.parquet -> Ch18 reads for backtest

    When ML4T_OUTPUT_DIR is set (e.g., by pytest), outputs are redirected to
    a temporary directory to prevent tests from overwriting production data.

    Args:
        chapter: Chapter number (1-27) or sub-chapter ID
        strategy_id: Strategy identifier (e.g., "etfs", "crypto_perps_funding")
        create: If True, create directory if it doesn't exist

    Returns:
        Path to output directory: {chapter_dir}/output/{strategy_id}/

    Examples:
        >>> get_output_dir(7, "etfs")
        PosixPath('.../07_defining_the_learning_task/output/etfs')

        >>> # Read from previous chapter
        >>> features = pl.read_parquet(get_output_dir(7, "etfs") / "features.parquet")

        >>> # In test mode with ML4T_OUTPUT_DIR=/tmp/test
        >>> get_output_dir(7, "etfs")
        PosixPath('/tmp/test/ch07_etfs')
    """
    import os

    # Test mode: allow chapter outputs to redirect independently from case studies.
    test_output = os.environ.get("ML4T_CHAPTER_OUTPUT_DIR") or os.environ.get("ML4T_OUTPUT_DIR")
    if test_output:
        # Get chapter number for directory naming
        if isinstance(chapter, int):
            ch_num = chapter
        else:
            # Sub-chapter: use string as-is
            ch_num = chapter
        output_dir = (
            Path(test_output) / f"ch{ch_num:02d}_{strategy_id}"
            if isinstance(ch_num, int)
            else Path(test_output) / f"{ch_num}_{strategy_id}"
        )
        if create:
            output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    # Production mode: normal chapter output
    chapter_dir = get_chapter_dir(chapter)
    output_dir = chapter_dir / "output" / strategy_id

    if create:
        output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def get_case_study_source_dir(strategy_id: str) -> Path:
    """Get the source-controlled case study directory, preferring sibling dev assets."""
    dev_case_dir = REPO_ROOT.parent / "dev" / "case_studies" / strategy_id
    if dev_case_dir.exists():
        return dev_case_dir
    return REPO_ROOT / "case_studies" / strategy_id


def get_case_study_dir(strategy_id: str, *, create: bool = True) -> Path:
    """Get the case study directory for a strategy.

    Case studies are centralized under CASE_STUDIES_DIR (default: repo_root/case_studies/).
    Each strategy has its own directory with stage-based subdirectories:

        case_studies/{strategy_id}/
        ├── exploration/           # Ch6 priors & EDA
        ├── labels/                # Ch7 labels & evaluation
        ├── features/              # Ch8-10 feature engineering
        ├── models/                # Ch11-15 ML models
        │   ├── linear/            # Ch11
        │   ├── gbm/               # Ch12
        │   ├── deep_learning/     # Ch13
        │   ├── latent_factors/    # Ch14
        │   └── causal/            # Ch15
        ├── backtest/              # Ch16-19 strategy implementation
        │   ├── simulation/        # Ch16
        │   ├── portfolio/         # Ch17
        │   ├── costs/             # Ch18
        │   └── risk/              # Ch19
        └── synthesis/             # Ch20 strategy development synthesis

    When ML4T_OUTPUT_DIR is set (e.g., by pytest), outputs are redirected to
    a temporary directory to prevent tests from overwriting production data.

    Args:
        strategy_id: Strategy identifier (e.g., "etfs", "crypto_perps_funding")
        create: If True, create directory if it doesn't exist

    Returns:
        Path to case study directory: case_studies/{strategy_id}/

    Examples:
        >>> get_case_study_dir("etfs")
        PosixPath('.../case_studies/etfs')

        >>> # Typical usage with stage subdirectory
        >>> CASE_DIR = get_case_study_dir("etfs")
        >>> features.write_parquet(CASE_DIR / "features" / "features.parquet")

        >>> # In test mode with ML4T_OUTPUT_DIR=/tmp/test
        >>> get_case_study_dir("etfs")
        PosixPath('/tmp/test/etfs')
    """
    import os

    # Test mode: redirect to temp directory (prevents overwriting production data)
    test_output = os.environ.get("ML4T_OUTPUT_DIR")
    if test_output:
        output_dir = Path(test_output) / strategy_id
    else:
        from utils import CASE_STUDIES_DIR

        output_dir = CASE_STUDIES_DIR / strategy_id

    if create:
        output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


# =============================================================================
# Dataset IDs (Tier 2 naming)
# =============================================================================

DATASET_IDS = frozenset(
    {
        "etf",
        "crypto",
        "futures",
        "fx",
        "equities_us",
        "equities_nasdaq100",
        "equities_sp500",
        "options_sp500",
    }
)


# =============================================================================
# Chapter Path Constants (for direct imports)
# =============================================================================

CH01 = REPO_ROOT / CHAPTERS[1]
CH02 = REPO_ROOT / CHAPTERS[2]
CH03 = REPO_ROOT / CHAPTERS[3]
CH04 = REPO_ROOT / CHAPTERS[4]
CH05 = REPO_ROOT / CHAPTERS[5]
CH06 = REPO_ROOT / CHAPTERS[6]
CH07 = REPO_ROOT / CHAPTERS[7]
CH08 = REPO_ROOT / CHAPTERS[8]
CH09 = REPO_ROOT / CHAPTERS[9]
CH10 = REPO_ROOT / CHAPTERS[10]
CH11 = REPO_ROOT / CHAPTERS[11]
CH12 = REPO_ROOT / CHAPTERS[12]
CH13 = REPO_ROOT / CHAPTERS[13]
CH14 = REPO_ROOT / CHAPTERS[14]
CH15 = REPO_ROOT / CHAPTERS[15]
CH16 = REPO_ROOT / CHAPTERS[16]
CH17 = REPO_ROOT / CHAPTERS[17]
CH18 = REPO_ROOT / CHAPTERS[18]
CH19 = REPO_ROOT / CHAPTERS[19]
CH20 = REPO_ROOT / CHAPTERS[20]
CH21 = REPO_ROOT / CHAPTERS[21]
CH22 = REPO_ROOT / CHAPTERS[22]
CH23 = REPO_ROOT / CHAPTERS[23]
CH24 = REPO_ROOT / CHAPTERS[24]
CH25 = REPO_ROOT / CHAPTERS[25]
CH26 = REPO_ROOT / CHAPTERS[26]
CH27 = REPO_ROOT / CHAPTERS[27]

__all__ = [
    # Registry
    "CHAPTERS",
    "SUB_CHAPTERS",
    "STRATEGY_IDS",
    "CHAPTER_STAGES",
    "CASE_STUDIES",  # Legacy alias
    "DATASET_IDS",
    "REPO_ROOT",
    # Functions
    "get_chapter_dir",
    "get_output_dir",
    "get_case_study_dir",
    # Constants
    "CH01",
    "CH02",
    "CH03",
    "CH04",
    "CH05",
    "CH06",
    "CH07",
    "CH08",
    "CH09",
    "CH10",
    "CH11",
    "CH12",
    "CH13",
    "CH14",
    "CH15",
    "CH16",
    "CH17",
    "CH18",
    "CH19",
    "CH20",
    "CH21",
    "CH22",
    "CH23",
    "CH24",
    "CH25",
    "CH26",
    "CH27",
]
