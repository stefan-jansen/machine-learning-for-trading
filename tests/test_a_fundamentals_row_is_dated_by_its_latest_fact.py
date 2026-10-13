"""The XBRL fundamentals panel must not date a value before it was filed.

`xbrl_download.py` builds one row per (cik, fiscal quarter) and fills it from
several SEC Frames calls. Frames returns each fact as it appears in the *latest*
filing reporting that period, so a quarter's facts arrive from more than one
filing: the balance sheet from the timely original, the income statement from a
later comparative that re-reports it.

The bug these tests exist to prevent (reported by a reader, 2026-10-10) is the
row being dated by the *first* accession seen rather than the last. That
publishes a restated number under the original filing's date - lookahead, in a
notebook whose subject is point-in-time correctness. Measured over the default
20 CIKs: 109 of 118 paired facts came from a different filing, all of them filed
later, by up to twelve months.

The second bug is `filings.recent` being read alone. Submissions pages a large
filer's older filings into `filings.files`, so 53 of the 312 accessions the
default 2022-2024 grid needs are invisible to a reader of `recent` - which
reaches the panel as a null `announcement_date` and a row the as-of query drops
without saying so.
"""

from __future__ import annotations

import importlib
import json

import polars as pl

downloader = importlib.import_module("data.equities.fundamentals.xbrl_download")


def _panel(accessions: list[str]) -> pl.DataFrame:
    """One row of the shape `_build_fundamentals` hands to `_add_filing_dates`."""
    return pl.DataFrame(
        {
            "cik": [320193],
            "entity_name": ["Apple Inc."],
            "period_end": ["2024-09-28"],
            "accessions": [accessions],
            "assets": [364980000000.0],
        }
    )


def test_a_row_is_dated_by_its_latest_fact(tmp_path):
    """The balance sheet filed first must not date a revenue restated later."""
    original = "0000320193-24-000123"
    comparative = "0000320193-25-000456"
    downloader._save_cached_filing_dates(
        tmp_path,
        320193,
        {original: "2024-11-01", comparative: "2025-10-31"},
    )

    dated = downloader._add_filing_dates(_panel([original, comparative]), tmp_path)

    assert dated.height == 1, "the row must survive dating"
    assert dated["filing_date"][0] == "2025-10-31", (
        "dated by the earlier filing: the row would claim a 2025 restatement was public in 2024"
    )
    assert dated["accession"][0] == comparative, (
        "`accession` must name the filing that `announcement_date` came from"
    )


def test_the_latest_fact_wins_whatever_order_the_frames_arrive_in(tmp_path):
    """Dating must not depend on which concept the Frames loop reached first."""
    early, late = "0000320193-24-000001", "0000320193-25-000002"
    downloader._save_cached_filing_dates(
        tmp_path, 320193, {early: "2024-11-01", late: "2025-10-31"}
    )

    forwards = downloader._add_filing_dates(_panel([early, late]), tmp_path)
    backwards = downloader._add_filing_dates(_panel([late, early]), tmp_path)

    assert forwards["filing_date"][0] == backwards["filing_date"][0] == "2025-10-31"


def test_an_unresolvable_accession_keeps_provenance_and_nulls_the_date(tmp_path):
    """A row whose filing dates do not resolve says so, rather than guessing."""
    downloader._save_cached_filing_dates(tmp_path, 320193, {"0000000000-00-000000": "2024-01-01"})

    dated = downloader._add_filing_dates(_panel(["0000320193-24-999999"]), tmp_path)

    assert dated["filing_date"][0] is None, "an unresolved accession must not invent a date"
    assert dated["accession"][0] == "0000320193-24-999999", (
        "the row must still record which filing it came from"
    )


def test_filing_dates_read_the_paged_submissions_documents(monkeypatch):
    """`filings.recent` alone hides an older filer's accessions."""
    recent = {
        "filings": {
            "recent": {
                "accessionNumber": ["0000320193-25-000456"],
                "filingDate": ["2025-10-31"],
            },
            "files": [{"name": "CIK0000320193-submissions-001.json"}],
        }
    }
    paged = {
        "accessionNumber": ["0000320193-24-000123"],
        "filingDate": ["2024-11-01"],
    }

    class _Response:
        def __init__(self, payload):
            self.status_code = 200
            self._payload = payload

        def json(self):
            return json.loads(json.dumps(self._payload))

    def fake_get(url, **_kwargs):
        return _Response(paged if "submissions-001" in url else recent)

    monkeypatch.setattr(downloader.requests, "get", fake_get)
    monkeypatch.setattr(downloader.time, "sleep", lambda _s: None)

    mapping = downloader._fetch_filing_dates(320193)

    assert mapping["0000320193-25-000456"] == "2025-10-31", "the recent block must still be read"
    assert mapping["0000320193-24-000123"] == "2024-11-01", (
        "the paged document was skipped: older accessions resolve to no filing date, "
        "and the panel drops those rows silently"
    )


def test_cik_732717_is_labelled_as_the_filer_that_files():
    """CIK 732717 is AT&T Inc.; it shipped labelled COST."""
    assert downloader.DEFAULT_CIK_TO_TICKER[732717] == "T"


def test_a_partly_resolved_row_is_not_dated_by_the_filing_that_did_resolve(tmp_path):
    """The gap RoboRev found: dating by the latest *resolved* fact is lookahead.

    If the comparative that supplied the restated value is the accession whose
    filing date is missing, taking the latest of what remains hands the row the
    original's date while it still carries the restated number - the defect this
    module exists to prevent, arrived at from the other side.
    """
    original = "0000320193-24-000123"
    comparative = "0000320193-25-000456"
    # Only the original resolves.
    downloader._save_cached_filing_dates(tmp_path, 320193, {original: "2024-11-01"})

    dated = downloader._add_filing_dates(_panel([original, comparative]), tmp_path)

    assert dated["filing_date"][0] is None, (
        "dated 2024-11-01 while carrying a value from a 2025 filing: a row whose "
        "facts do not all resolve has no knowable knowledge time"
    )
    assert dated["accession"][0] in {original, comparative}, (
        "the row must still record provenance even when it cannot be dated"
    )
