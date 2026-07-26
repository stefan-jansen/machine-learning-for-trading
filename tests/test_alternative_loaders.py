from datetime import datetime

import polars as pl

from data.alternative import loader


def test_bloomberg_loader_normalizes_provider_date_to_timestamp(tmp_path, monkeypatch) -> None:
    target = tmp_path / "alternative" / "news" / "bloomberg" / "bloomberg_news.parquet"
    target.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "headline": ["Older", "Newer"],
            "journalists": [["A"], ["B"]],
            "date": [
                datetime(2011, 1, 1, 9, 30),
                datetime(2012, 1, 1, 9, 30),
            ],
            "link": ["https://example.com/older", "https://example.com/newer"],
            "article": ["Older article body", "Newer article body"],
        }
    ).write_parquet(target)
    monkeypatch.setattr(loader, "ML4T_DATA_PATH", tmp_path)

    result = loader.load_bloomberg_news(start_date="2012-01-01")

    assert "timestamp" in result.columns
    assert "date" not in result.columns
    assert result["headline"].to_list() == ["Newer"]


def test_bloomberg_end_date_keeps_intraday_articles(tmp_path, monkeypatch) -> None:
    target = tmp_path / "alternative" / "news" / "bloomberg" / "bloomberg_news.parquet"
    target.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "headline": ["Midnight", "Intraday", "Next day"],
            "timestamp": [
                datetime(2013, 12, 31, 0, 0),
                datetime(2013, 12, 31, 16, 45),
                datetime(2014, 1, 1, 9, 30),
            ],
        }
    ).write_parquet(target)
    monkeypatch.setattr(loader, "ML4T_DATA_PATH", tmp_path)

    result = loader.load_bloomberg_news(end_date="2013-12-31")

    assert result["headline"].to_list() == ["Midnight", "Intraday"]
