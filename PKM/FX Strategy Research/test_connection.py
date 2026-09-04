"""Smoke-test MT5 bridge connectivity (no price data)."""

from mt5_client import get_mt5


def main() -> None:
    mt5 = get_mt5()
    if not mt5.initialize():
        print(f"FAIL: initialize failed — {mt5.last_error()}")
        return

    try:
        print("OK: connected")
        print("version:", mt5.version())
        print("account_info:", mt5.account_info())
    except Exception as exc:
        print(f"FAIL: connected but query failed — {exc}")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}")
