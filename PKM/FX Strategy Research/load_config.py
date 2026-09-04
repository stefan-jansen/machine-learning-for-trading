"""Load config/setup.yaml from this project root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
SETUP_PATH = ROOT / "config" / "setup.yaml"


def load_setup() -> dict[str, Any]:
    with SETUP_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def mt5_connect():
    """Return (mt5_module, setup) after initialize, or raise RuntimeError."""
    from mt5_client import get_mt5

    setup = load_setup()
    cfg = setup.get("mt5", {})
    mt5 = get_mt5(host=cfg.get("host", "127.0.0.1"), port=int(cfg.get("port", 18812)))
    if not mt5.initialize():
        raise RuntimeError(f"initialize failed — {mt5.last_error()}")
    return mt5, setup
