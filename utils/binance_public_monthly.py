"""Patch BinancePublicProvider to prefer monthly archives in async OHLCV fetches.

ml4t-data's async OHLCV path downloads one daily ZIP per day. For multi-year
hourly histories that is tens of thousands of requests and almost no progress
logs. Sync already uses monthly ZIPs when the range is >60 days; this patch
aligns async with that strategy and logs each completed month.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
import structlog

logger = structlog.get_logger()

_PATCH_ATTR = "_ml4t_monthly_async_patched"


def _iter_months(start_dt: datetime, end_dt: datetime) -> list[tuple[int, int]]:
    current = start_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    months: list[tuple[int, int]] = []
    while current <= end_dt:
        months.append((current.year, current.month))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return months


def _month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    month_start = datetime(year, month, 1, tzinfo=UTC)
    if month == 12:
        month_end = datetime(year + 1, 1, 1, tzinfo=UTC) - timedelta(days=1)
    else:
        month_end = datetime(year, month + 1, 1, tzinfo=UTC) - timedelta(days=1)
    month_end = month_end.replace(hour=23, minute=59, second=59)
    return month_start, month_end


def apply_binance_monthly_async_patch(provider_cls: type[Any]) -> type[Any]:
    """Monkeypatch monthly async OHLCV + per-month progress onto a provider class."""
    if getattr(provider_cls, _PATCH_ATTR, False):
        return provider_cls

    original_fetch_and_transform_async = provider_cls._fetch_and_transform_data_async
    original_fetch_monthly = provider_cls._fetch_monthly_data
    original_premium_async = provider_cls.fetch_premium_index_async

    async def _fetch_monthly_data_async(
        self,
        symbol: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> list[pl.DataFrame]:
        months = _iter_months(start_dt, end_dt)
        total = len(months)
        if total == 0:
            return []

        logger.info(
            "Downloading monthly archives (async)",
            symbol=symbol,
            interval=interval,
            months=total,
            start=f"{months[0][0]}-{months[0][1]:02d}",
            end=f"{months[-1][0]}-{months[-1][1]:02d}",
        )

        semaphore = asyncio.Semaphore(10)
        progress_lock = asyncio.Lock()
        completed = 0

        async def fetch_one(year: int, month: int) -> pl.DataFrame | None:
            nonlocal completed
            month_label = f"{year}-{month:02d}"
            async with semaphore:
                url = self._build_monthly_url(symbol, interval, year, month)
                try:
                    df = await self._download_and_parse_zip_async(url)
                    source = "monthly"
                    if df is None or df.is_empty():
                        month_start, month_end = _month_bounds(year, month)
                        fetch_start = max(month_start, start_dt)
                        fetch_end = min(month_end, end_dt)
                        daily = await self._fetch_daily_data_async(
                            symbol, interval, fetch_start, fetch_end
                        )
                        df = pl.concat(daily) if daily else None
                        source = "daily_fallback"
                except Exception as exc:
                    logger.warning(
                        "Month download failed",
                        symbol=symbol,
                        month=month_label,
                        error=str(exc),
                    )
                    df = None
                    source = "error"

            rows = 0 if df is None or df.is_empty() else len(df)
            async with progress_lock:
                completed += 1
                logger.info(
                    "Month download progress",
                    symbol=symbol,
                    month=month_label,
                    done=completed,
                    total=total,
                    rows=rows,
                    source=source,
                )
            return df

        results = await asyncio.gather(*(fetch_one(y, m) for y, m in months))
        return [df for df in results if df is not None and not df.is_empty()]

    async def _fetch_and_transform_data_async(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pl.DataFrame:
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)

        freq_lower = frequency.lower()
        if freq_lower not in ["daily", "1day", "weekly", "1week", "monthly", "1month"]:
            end_dt = end_dt.replace(hour=23, minute=59, second=59)

        symbol = self._normalize_symbol(symbol)

        if freq_lower not in self.INTERVAL_MAP:
            raise ValueError(f"Unsupported frequency: {frequency}")
        interval = self.INTERVAL_MAP[freq_lower]

        logger.info(
            f"Fetching {symbol} data async from Binance Public Data ({self.market})",
            frequency=frequency,
            interval=interval,
            start=start,
            end=end,
        )

        days_requested = (end_dt - start_dt).days + 1
        if days_requested > 60:
            all_data = await self._fetch_monthly_data_async(symbol, interval, start_dt, end_dt)
        else:
            all_data = await self._fetch_daily_data_async(symbol, interval, start_dt, end_dt)

        if not all_data:
            logger.info(f"No data found for {symbol} in requested range")
            return self._create_empty_dataframe()

        df = pl.concat(all_data)
        df = df.filter((pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt))
        df = df.sort("timestamp").unique(subset=["timestamp"], maintain_order=True)

        logger.info(
            f"Fetched {len(df)} rows for {symbol} (async)",
            start_date=df["timestamp"].min() if not df.is_empty() else None,
            end_date=df["timestamp"].max() if not df.is_empty() else None,
        )
        return df

    def _fetch_monthly_data(
        self, symbol: str, interval: str, start_dt: datetime, end_dt: datetime
    ) -> list[pl.DataFrame]:
        months = _iter_months(start_dt, end_dt)
        total = len(months)
        logger.info(
            "Downloading monthly archives",
            symbol=symbol,
            interval=interval,
            months=total,
        )
        all_data: list[pl.DataFrame] = []
        for index, (year, month) in enumerate(months, start=1):
            month_label = f"{year}-{month:02d}"
            url = self._build_monthly_url(symbol, interval, year, month)
            source = "monthly"
            rows = 0
            try:
                df = self._download_and_parse_zip(url)
                if df is not None and not df.is_empty():
                    all_data.append(df)
                    rows = len(df)
                else:
                    source = "daily_fallback"
                    month_start, month_end = _month_bounds(year, month)
                    fetch_start = max(month_start, start_dt)
                    fetch_end = min(month_end, end_dt)
                    daily_data = self._fetch_daily_data(symbol, interval, fetch_start, fetch_end)
                    all_data.extend(daily_data)
                    rows = sum(len(part) for part in daily_data)
            except Exception as exc:
                source = "error"
                logger.warning(f"Failed to download {month_label}: {exc}")

            logger.info(
                "Month download progress",
                symbol=symbol,
                month=month_label,
                done=index,
                total=total,
                rows=rows,
                source=source,
            )
            self._acquire_rate_limit()
        return all_data

    async def fetch_premium_index_async(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "8h",
    ) -> pl.DataFrame:
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=UTC
        )
        symbol = self._normalize_symbol(symbol)
        months = _iter_months(start_dt, end_dt)
        total = len(months)
        logger.info(
            "Downloading premium index monthly archives (async)",
            symbol=symbol,
            interval=interval,
            months=total,
        )

        semaphore = asyncio.Semaphore(10)
        progress_lock = asyncio.Lock()
        completed = 0

        async def fetch_month(year: int, month: int) -> pl.DataFrame | None:
            nonlocal completed
            month_label = f"{year}-{month:02d}"
            async with semaphore:
                url = self._build_premium_index_monthly_url(symbol, interval, year, month)
                try:
                    df = await self._download_and_parse_premium_index_zip_async(url, symbol)
                    source = "monthly" if df is not None and not df.is_empty() else "missing"
                except Exception as exc:
                    logger.debug(f"Monthly premium fetch failed for {month_label}: {exc}")
                    df = None
                    source = "error"

            rows = 0 if df is None or df.is_empty() else len(df)
            async with progress_lock:
                completed += 1
                logger.info(
                    "Month download progress",
                    symbol=symbol,
                    month=month_label,
                    done=completed,
                    total=total,
                    rows=rows,
                    source=source,
                    dataset="premium_index",
                )
            return df

        results = await asyncio.gather(*(fetch_month(y, m) for y, m in months))
        all_data = [df for df in results if df is not None and not df.is_empty()]
        if not all_data:
            return self._create_empty_premium_index_dataframe()

        df = pl.concat(all_data)
        df = df.filter((pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt))
        df = df.sort("timestamp").unique(subset=["timestamp"], maintain_order=True)
        return df

    provider_cls._fetch_monthly_data_async = _fetch_monthly_data_async
    provider_cls._fetch_and_transform_data_async = _fetch_and_transform_data_async
    provider_cls._fetch_monthly_data = _fetch_monthly_data
    provider_cls.fetch_premium_index_async = fetch_premium_index_async
    # Keep originals available for tests / debugging.
    provider_cls._fetch_and_transform_data_async_unpatched = original_fetch_and_transform_async
    provider_cls._fetch_monthly_data_unpatched = original_fetch_monthly
    provider_cls.fetch_premium_index_async_unpatched = original_premium_async
    setattr(provider_cls, _PATCH_ATTR, True)
    return provider_cls
