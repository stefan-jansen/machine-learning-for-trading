"""Cross-platform MetaTrader 5 client.

Windows uses the official MetaTrader5 package and talks to a running terminal.
Linux uses mt5linux and talks to the Wine-side RPyC bridge.
"""

from __future__ import annotations

import sys
from typing import Any


def get_mt5(host: str = "127.0.0.1", port: int = 18812) -> Any:
    if sys.platform == "win32":
        import MetaTrader5 as mt5

        return mt5

    from mt5linux import MetaTrader5

    return MetaTrader5(host=host, port=port)
