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
#
# The "png" half goes through kaleido, and kaleido v1 drives a Chrome it does not
# bundle. Where no Chrome is installed - the Docker images, a slim server - every
# `fig.show()` died on ChromeNotFoundError (#432), which made the figure-producing
# notebooks unrunnable rather than merely PNG-less. So ask for PNG only when a
# Chrome is actually resolvable and otherwise fall back to "plotly_mimetype",
# which still renders interactively in Jupyter and is inert in a plain script.
# A probe that raises is treated as "no Chrome": this must never be what breaks
# the import.
def _default_plotly_renderer() -> str:
    try:
        from choreographer.browsers.chromium import Chromium

        if Chromium.find_browser(skip_local=False):
            return "plotly_mimetype+png"
    except Exception:  # noqa: BLE001
        pass
    return "plotly_mimetype"


try:
    import os as _os

    import plotly.io as _pio

    if not _os.environ.get("PLOTLY_RENDERER"):
        _pio.renderers.default = _default_plotly_renderer()
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
