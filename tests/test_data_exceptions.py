"""Contract tests for data/exceptions.py.

Every loader in data/ raises DataNotFoundError with a specific combination
of keyword arguments (download_script vs instructions vs download_url, plus
readme). The tests below pin the message shape so a reformat of _build_message
does not silently drop the reader-facing download instructions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data.exceptions import DataNotFoundError, DownloadError, MissingDependencyError


def test_data_not_found_with_download_script_shows_command() -> None:
    err = DataNotFoundError(
        dataset_name="ETF Universe",
        path=Path("/data/etfs/market/etf_universe.parquet"),
        download_script="data/etfs/market/download.py",
    )
    msg = str(err)

    assert "DATA NOT FOUND: ETF Universe" in msg
    assert "/data/etfs/market/etf_universe.parquet" in msg
    assert "uv run python data/etfs/market/download.py" in msg
    assert "data/README.md" in msg  # Default readme fallback


def test_data_not_found_with_custom_readme_overrides_default() -> None:
    err = DataNotFoundError(
        dataset_name="SEC Filings",
        path=Path("/data/equities/fundamentals/"),
        download_script="data/equities/fundamentals/filings_download.py",
        readme="data/equities/fundamentals/README.md",
    )
    msg = str(err)

    assert "data/equities/fundamentals/README.md" in msg
    # Default readme should not appear when custom is set
    assert msg.count("README.md") == 1


def test_data_not_found_with_download_url_prefers_url_over_script() -> None:
    err = DataNotFoundError(
        dataset_name="AlgoSeek Options",
        path=Path("/data/options/sp500_options.parquet"),
        download_url="https://example.com/algoseek.zip",
        download_script="data/ignored/download.py",  # Should be ignored
        requires_api_key="ALGOSEEK_KEY",
    )
    msg = str(err)

    assert "https://example.com/algoseek.zip" in msg
    # download_script is shadowed by download_url in the elif chain
    assert "uv run python data/ignored/download.py" not in msg
    assert "Extract to:" in msg
    assert "ALGOSEEK_KEY" in msg


def test_data_not_found_with_instructions_overrides_other_branches() -> None:
    err = DataNotFoundError(
        dataset_name="Derived Daily Bars",
        path=Path("/data/futures/market/continuous/daily/continuous_daily.parquet"),
        instructions="Run: python 02/05_futures_session_aggregation.py",
        download_script="should_not_appear.py",
    )
    msg = str(err)

    assert "02/05_futures_session_aggregation.py" in msg
    assert "should_not_appear.py" not in msg


def test_data_not_found_with_derivation_notebook_adds_pointer() -> None:
    err = DataNotFoundError(
        dataset_name="Derived Label",
        path=Path("/data/labels/fwd_ret_5d.parquet"),
        derivation_notebook="07_defining_the_learning_task/03_label_methods.py",
    )
    msg = str(err)

    assert "How this dataset is built" in msg
    assert "07_defining_the_learning_task/03_label_methods.py" in msg


def test_data_not_found_is_filenotfounderror() -> None:
    """Callers catch FileNotFoundError generically; guard the inheritance."""
    err = DataNotFoundError(
        dataset_name="x",
        path=Path("/tmp/x"),
        download_script="x.py",
    )

    assert isinstance(err, FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        raise err


def test_data_not_found_accepts_str_path() -> None:
    """Some callers pass str paths; the class must normalize to Path."""
    err = DataNotFoundError(
        dataset_name="x",
        path="/tmp/string_path.parquet",
        download_script="x.py",
    )

    assert isinstance(err.path, Path)
    assert "/tmp/string_path.parquet" in str(err)


def test_download_error_includes_reason_and_suggestion() -> None:
    err = DownloadError(
        dataset_name="Crypto Perps",
        reason="API returned 429",
        suggestion="Retry after backoff or check rate limits",
    )
    msg = str(err)

    assert "DOWNLOAD FAILED: Crypto Perps" in msg
    assert "API returned 429" in msg
    assert "Retry after backoff" in msg


def test_download_error_without_suggestion_still_valid() -> None:
    err = DownloadError(dataset_name="x", reason="unknown")
    msg = str(err)

    assert "DOWNLOAD FAILED: x" in msg
    assert "Suggestion:" not in msg


def test_download_error_is_runtime_error() -> None:
    err = DownloadError(dataset_name="x", reason="y")
    assert isinstance(err, RuntimeError)


def test_missing_dependency_error_defaults_install_command() -> None:
    err = MissingDependencyError(package="edgartools")
    msg = str(err)

    assert "edgartools" in msg
    assert "pip install edgartools" in msg


def test_missing_dependency_error_custom_install_and_purpose() -> None:
    err = MissingDependencyError(
        package="torch",
        install_command="uv sync --extra gpu",
        purpose="LSTM training in Ch13",
    )
    msg = str(err)

    assert "torch" in msg
    assert "uv sync --extra gpu" in msg
    assert "LSTM training in Ch13" in msg


def test_missing_dependency_is_import_error() -> None:
    """Callers catch ImportError to gate optional backends."""
    err = MissingDependencyError(package="x")
    assert isinstance(err, ImportError)
