#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "https://fxmacrodata.com/api/v1/"


def request_json(path: str, params: dict[str, Any], base_url: str, api_key: str) -> dict[str, Any]:
    query = {key: value for key, value in params.items() if value is not None}
    if api_key:
        query["api_key"] = api_key
    url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    if query:
        url = url + "?" + urlencode(query)

    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"FXMacroData HTTP {exc.code}: {body}") from exc


def rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("data", "rows", "results", "items"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row if isinstance(row, dict) else {"value": row} for row in rows]
        return [payload]
    if isinstance(payload, list):
        return [row if isinstance(row, dict) else {"value": row} for row in payload]
    return [{"value": payload}]


def write_rows(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        import polars as pl

        pl.DataFrame(rows).write_parquet(output)
        return

    with output.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, sort_keys=True, default=str)
        file.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download optional FXMacroData macro, FX, COT, and calendar data."
    )
    parser.add_argument("--currency", default="usd", help="Currency code such as usd or eur")
    parser.add_argument("--quote", default="usd", help="Quote currency for FX pairs")
    parser.add_argument("--indicator", default="policy_rate", help="Macro indicator slug")
    parser.add_argument("--endpoint", default="announcements", choices=[
        "announcements",
        "calendar",
        "catalogue",
        "cot",
        "forex",
        "commodity",
        "commodities_latest",
        "market_sessions",
        "risk_sentiment",
    ])
    parser.add_argument("--commodity", default="gold", help="Commodity indicator slug")
    parser.add_argument("--start-date", help="Optional start date, YYYY-MM-DD")
    parser.add_argument("--end-date", help="Optional end date, YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=500, help="Maximum rows to request")
    parser.add_argument("--output", type=Path, default=Path("fxmacrodata_macro.json"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    return parser


def endpoint_path(args: argparse.Namespace) -> str:
    currency = args.currency.lower()
    quote = args.quote.lower()
    if args.endpoint == "announcements":
        return f"announcements/{currency}/{args.indicator}"
    if args.endpoint == "calendar":
        return f"calendar/{currency}"
    if args.endpoint == "catalogue":
        return f"data_catalogue/{currency}"
    if args.endpoint == "cot":
        return f"cot/{currency}"
    if args.endpoint == "forex":
        return f"forex/{currency}/{quote}"
    if args.endpoint == "commodity":
        return f"commodities/{args.commodity}"
    if args.endpoint == "commodities_latest":
        return "commodities/latest"
    if args.endpoint == "market_sessions":
        return "market_sessions"
    if args.endpoint == "risk_sentiment":
        return "risk_sentiment"
    raise ValueError(f"unsupported endpoint: {args.endpoint}")


def main() -> None:
    args = build_parser().parse_args()
    api_key = os.getenv("FXMACRODATA_API_KEY") or os.getenv("FXMD_API_KEY") or ""
    params = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "limit": args.limit,
    }
    payload = request_json(endpoint_path(args), params, args.base_url, api_key)
    rows = rows_from_payload(payload)
    write_rows(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()