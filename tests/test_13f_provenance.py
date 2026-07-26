import importlib
import xml.etree.ElementTree as ET
from datetime import date

import numpy as np
import polars as pl

from data.equities import loader

downloader = importlib.import_module("data.equities.positioning.13f_download")


class _Response:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {
            "name": "Example Manager",
            "filings": {
                "recent": {
                    "form": ["13F-HR"],
                    "accessionNumber": ["0001234567-24-000001"],
                    "reportDate": ["2024-09-30"],
                    "filingDate": ["2024-11-14"],
                }
            },
        }


def test_recent_filings_preserve_sec_report_date(monkeypatch) -> None:
    monkeypatch.setattr(downloader.requests, "get", lambda *args, **kwargs: _Response())

    filings = downloader.get_recent_13f_filings("0001234567", num_filings=1)

    assert filings[0]["report_date"] == "2024-09-30"
    assert filings[0]["filing_date"] == "2024-11-14"


def test_bulk_normalization_preserves_period_of_report() -> None:
    infotable = pl.DataFrame(
        {
            "ACCESSION_NUMBER": ["0001234567-24-000001"],
            "NAMEOFISSUER": ["Example Issuer"],
            "CUSIP": ["123456789"],
            "VALUE": [1000],
            "SSHPRNAMT": [50],
            "PUTCALL": ["PUT"],
        }
    )
    coverpage = pl.DataFrame(
        {
            "ACCESSION_NUMBER": ["0001234567-24-000001"],
            "FILINGMANAGER_NAME": ["Example Manager"],
        }
    )
    submission = pl.DataFrame(
        {
            "ACCESSION_NUMBER": ["0001234567-24-000001"],
            "SUBMISSIONTYPE": ["13F-HR"],
            "CIK": ["1234567"],
            "PERIODOFREPORT": ["30-SEP-2024"],
            "FILING_DATE": ["14-NOV-2024"],
        }
    )

    result = downloader._normalize_bulk_to_canonical(infotable, coverpage, submission)

    assert result["report_date"].to_list() == [date(2024, 9, 30)]
    assert result["filing_date"].to_list() == [date(2024, 11, 14)]
    assert result["put_call"].to_list() == ["PUT"]


def test_per_cik_parser_preserves_option_marker(monkeypatch) -> None:
    root = ET.fromstring(
        """
        <informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
          <infoTable>
            <nameOfIssuer>Example Inc</nameOfIssuer>
            <cusip>123456789</cusip>
            <value>100</value>
            <shrsOrPrnAmt><sshPrnamt>10</sshPrnamt></shrsOrPrnAmt>
          </infoTable>
          <infoTable>
            <nameOfIssuer>Example Inc</nameOfIssuer>
            <cusip>123456789</cusip>
            <value>900</value>
            <shrsOrPrnAmt><sshPrnamt>90</sshPrnamt></shrsOrPrnAmt>
            <putCall>PUT</putCall>
          </infoTable>
        </informationTable>
        """
    )
    monkeypatch.setattr(downloader, "fetch_13f_xml_root", lambda *args: root)

    holdings = downloader.parse_13f_holdings("0001234567", "0001234567-24-000001")

    assert [row["put_call"] for row in holdings] == [None, "PUT"]


def test_derived_graph_is_latest_quarter_positive_equity_only() -> None:
    holdings = pl.DataFrame(
        {
            "cik": ["1", "1", "1", "1", "2", "3", "4"],
            "accession_no": ["a", "a", "a", "a", "b", "c", "d"],
            "issuer": [
                "EXAMPLE",
                "EXAMPLE INC",
                "EXAMPLE",
                "EXAMPLE",
                "EXAMPLE",
                "STALE",
                "ZERO",
            ],
            "cusip": ["123", "123", "123", "123", "123", "999", "000"],
            "value_thousands": [100, 50, 0, 900, 200, 999, 0],
            "shares": [10, 5, 1, 90, 20, 99, 0],
            "put_call": [None, None, None, "PUT", None, None, None],
            "report_date": [
                date(2024, 9, 30),
                date(2024, 9, 30),
                date(2024, 9, 30),
                date(2024, 9, 30),
                date(2024, 9, 30),
                date(2024, 6, 30),
                date(2024, 9, 30),
            ],
            "filing_date": [
                date(2024, 11, 10),
                date(2024, 11, 10),
                date(2024, 11, 10),
                date(2024, 11, 10),
                date(2024, 11, 14),
                date(2024, 8, 14),
                date(2024, 11, 12),
            ],
            "company_name": [
                "Manager 1",
                "Manager 1",
                "Manager 1",
                "Manager 1",
                "Manager 2",
                "Old",
                "Zero",
            ],
        }
    )

    features, edges, matrix, stocks = downloader.build_features_and_matrix(holdings)

    assert edges.sort("institution_id")["weight_value"].to_list() == [150, 200]
    assert edges.sort("institution_id")["weight_shares"].to_list() == [16, 20]
    assert edges["stock_id"].unique().to_list() == ["123"]
    assert features["total_inst_value_usd"].to_list() == [350]
    assert features["n_inst_holders"].to_list() == [2]
    assert features["timestamp"].to_list() == [date(2024, 11, 14)]
    assert stocks == ["123"]
    np.testing.assert_allclose(matrix, np.ones((1, 1), dtype=np.float32), atol=1e-6)


def test_holdings_loader_parses_both_sec_dates(tmp_path, monkeypatch) -> None:
    target = tmp_path / "equities" / "positioning" / "13f" / "institutional_holdings.parquet"
    target.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "cik": ["0001234567"],
            "accession_no": ["0001234567-24-000001"],
            "issuer": ["Example Issuer"],
            "cusip": ["123456789"],
            "value_thousands": [1000],
            "shares": [50],
            "put_call": [None],
            "report_date": ["2024-09-30"],
            "filing_date": ["2024-11-14"],
            "company_name": ["Example Manager"],
        }
    ).write_parquet(target)
    monkeypatch.setattr(loader, "ML4T_DATA_PATH", tmp_path)

    result = loader.load_institutional_holdings_13f()

    assert result.schema["report_date"] == pl.Date
    assert result.schema["filing_date"] == pl.Date
    assert result["put_call"].to_list() == [None]
