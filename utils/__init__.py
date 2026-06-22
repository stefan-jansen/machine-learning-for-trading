"""ML4T utilities - configuration, paths, styling, and data quality.

This package provides:
- Configuration management (utils.config)
- Path utilities (utils.paths)
- Visualization styling (utils.style)
- Data quality checks (utils.data_quality)

Usage:
    >>> from utils import ML4T_PATH, ML4T_DATA_PATH, DATA_DIR
    >>> from utils.paths import CHAPTERS, get_output_dir
    >>> from utils.style import COLORS
    >>> from utils.data_quality import describe_coverage, check_ohlc_invariants
"""

from utils.config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    CASE_STUDIES_DIR,
    DATABENTO_API_KEY,
    ML4T_DATA_PATH,
    ML4T_PATH,
    OANDA_API_KEY,
    REPO_ROOT,
)

# Backward compatibility alias
DATA_DIR = ML4T_DATA_PATH

# Plotly: include PNG in cell output so GitHub can render .ipynb figures.
# Override with PLOTLY_RENDERER env var (e.g. "json" for headless CI).
try:
    import os as _os

    import plotly.io as _pio

    if not _os.environ.get("PLOTLY_RENDERER"):
        _pio.renderers.default = "plotly_mimetype+png"
except ImportError:
    pass

__all__ = [
    # Core configuration
    "ML4T_PATH",
    "ML4T_DATA_PATH",
    "DATA_DIR",  # Backward compatibility
    "CASE_STUDIES_DIR",
    "REPO_ROOT",
    # API keys
    "DATABENTO_API_KEY",
    "OANDA_API_KEY",
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
]
