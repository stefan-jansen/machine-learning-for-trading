"""Point-in-time S&P 500 returns that respect security identity boundaries.

The lineage rule is shared with `sp500_equity_option_analytics`, which reads the
same bars into a backtest price panel; it lives in
`case_studies/utils/sp500_price_lineage.py` so the boundary is defined once.
"""

from __future__ import annotations

from case_studies.utils.sp500_price_lineage import (
    reconcile_underlying_log_returns,
    validate_reconciled_returns,
)

__all__ = ["reconcile_underlying_log_returns", "validate_reconciled_returns"]
