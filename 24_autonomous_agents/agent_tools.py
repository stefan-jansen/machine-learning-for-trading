"""Search provider abstraction and tool execution framework.

Providers: Tavily (real web search), Mock (deterministic CI fallback).
No disk cache — teaching code stays transparent.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

from agent_schemas import SearchResult


def eligible_results(results: list[SearchResult], cutoff: date | None = None) -> list[SearchResult]:
    usable = []
    for result in results:
        url = urlparse(result.url or "")
        if url.scheme not in {"http", "https"} or not url.hostname or not result.snippet:
            continue
        published = _parse_published_date(result.published or "")
        if cutoff is not None and (published is None or published >= cutoff):
            continue
        usable.append(result)
    return usable


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SearchClient(Protocol):
    """Protocol for search providers -- agents search via this interface."""

    def search(
        self,
        query: str,
        max_results: int = 5,
        cutoff_date: date | None = None,
    ) -> list[SearchResult]: ...


# ---------------------------------------------------------------------------
# Tavily (real web search)
# ---------------------------------------------------------------------------


class TavilySearchClient:
    """Tavily search client with point-in-time filtering.

    Supports backtesting by filtering results to only those published
    before a specified cutoff date, preventing lookahead bias.
    """

    def __init__(
        self,
        api_key: str,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> None:
        try:
            import httpx  # noqa: F401
        except ImportError as e:
            raise ImportError("pip install httpx  (or: uv add httpx)") from e
        self._api_key = api_key
        self._allowed = {d.lower() for d in (allowed_domains or [])}
        self._blocked = {d.lower() for d in (blocked_domains or [])}

    def search(
        self,
        query: str,
        max_results: int = 5,
        cutoff_date: date | None = None,
    ) -> list[SearchResult]:
        import httpx

        payload: dict[str, Any] = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max_results * 2 if cutoff_date else max_results,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post("https://api.tavily.com/search", json=payload)
            resp.raise_for_status()
            data = resp.json()

        results = self._parse_results(data.get("results", []))

        if cutoff_date:
            results = self._filter_by_date(results, cutoff_date)

        return results[:max_results]

    def _parse_results(self, items: list[dict[str, Any]]) -> list[SearchResult]:
        results: list[SearchResult] = []
        for item in items:
            url = item.get("url")
            if url and not self._domain_allowed(url):
                continue
            results.append(
                SearchResult(
                    title=str(item.get("title") or ""),
                    url=url,
                    snippet=item.get("content"),
                    published=item.get("published_date"),
                    score=item.get("score"),
                )
            )
        return results

    def _filter_by_date(self, results: list[SearchResult], cutoff: date) -> list[SearchResult]:
        """Keep only results dated before cutoff (prevents lookahead).

        An undated result may have been written after the cutoff, so it is dropped.
        """
        filtered: list[SearchResult] = []
        for r in results:
            pub = _parse_published_date(r.published) if r.published else None
            if pub is not None and pub < cutoff:
                filtered.append(r)
        return filtered

    def _domain_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if not host:
            return True
        for blocked in self._blocked:
            if host == blocked or host.endswith("." + blocked):
                return False
        if not self._allowed:
            return True
        return any(host == a or host.endswith("." + a) for a in self._allowed)


# ---------------------------------------------------------------------------
# Mock search (explicit synthetic fixtures)
# ---------------------------------------------------------------------------


# Canned results keyed by keyword for deterministic testing
_MOCK_RESULTS: dict[str, list[SearchResult]] = {
    "default": [
        SearchResult(
            title="Analysis: Key factors for this event",
            url="https://reuters.com/analysis/key-factors",
            snippet="Multiple indicators suggest moderate probability. Historical base rates and recent developments point to a balanced outlook.",
            published="2025-02-15",
            score=0.95,
        ),
        SearchResult(
            title="Expert consensus on upcoming resolution",
            url="https://wsj.com/expert-consensus",
            snippet="Analysts remain divided. Bull case centers on momentum; bear case on structural headwinds and valuation concerns.",
            published="2025-02-18",
            score=0.88,
        ),
    ],
    "nvidia": [
        SearchResult(
            title="NVIDIA suppliers signal continued AI server demand",
            url="https://wsj.com/nvidia-supply-chain",
            snippet="Supply-chain checks point to resilient GPU demand ahead of earnings report.",
            published="2025-02-17",
            score=0.94,
        ),
        SearchResult(
            title="Analysts raise NVIDIA targets ahead of earnings",
            url="https://nasdaq.com/nvidia-targets",
            snippet="Street revisions remain constructive ahead of the Feb 26 report.",
            published="2025-02-18",
            score=0.91,
        ),
        SearchResult(
            title="Investors watch valuation risk as AI rally broadens",
            url="https://wsj.com/ai-rally-valuation",
            snippet="Valuation remains the main pushback in the bullish narrative.",
            published="2025-02-20",
            score=0.85,
        ),
    ],
    "federal reserve": [
        SearchResult(
            title="Fed officials signal patience on rate cuts",
            url="https://federalreserve.gov/press-releases",
            snippet="Multiple FOMC members indicate no urgency to cut rates at upcoming meeting.",
            published="2025-03-05",
            score=0.96,
        ),
        SearchResult(
            title="CME FedWatch: Markets price near-zero chance of March cut",
            url="https://cmegroup.com/fedwatch",
            snippet="Fed funds futures imply 4% probability of a rate cut at the March 18-19 meeting.",
            published="2025-03-08",
            score=0.93,
        ),
    ],
    "bitcoin": [
        SearchResult(
            title="Bitcoin consolidates below $100K resistance",
            url="https://coindesk.com/bitcoin-resistance",
            snippet="BTC has tested the $100K level multiple times but failed to sustain a breakout.",
            published="2025-02-15",
            score=0.90,
        ),
    ],
    "ipo": [
        SearchResult(
            title="IPO market remains quiet in Q1 2025",
            url="https://nasdaq.com/ipo-pipeline",
            snippet="No major tech IPO has filed for listing. Pipeline remains thin despite market recovery.",
            published="2025-03-10",
            score=0.88,
        ),
    ],
}


# These records are synthetic; reserved domains prevent mistaking fixture URLs for sources.
for _scenario, _records in _MOCK_RESULTS.items():
    for _index, _result in enumerate(_records):
        _result.title = f"Synthetic fixture: {_result.title}"
        _result.url = f"https://example.org/{_scenario}-{_index}"


class MockSearchClient:
    """Deterministic search client returning canned results for CI testing."""

    def search(
        self,
        query: str,
        max_results: int = 5,
        cutoff_date: date | None = None,
    ) -> list[SearchResult]:
        query_lower = query.lower()

        # Match by keyword
        results = _MOCK_RESULTS["default"]
        for keyword, keyword_results in _MOCK_RESULTS.items():
            if keyword in query_lower:
                results = keyword_results
                break

        return eligible_results(results, cutoff_date)[:max_results]


# ---------------------------------------------------------------------------
# Tool execution logging
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolExecution:
    """Record of a single tool call for audit trail."""

    tool_name: str
    args: dict[str, Any]
    status: str  # success, error, blocked
    duration_ms: float
    result_preview: str = ""
    provenance: dict[str, str] = field(default_factory=dict)


class ToolExecutor:
    """Wraps a SearchClient with execution logging and domain policy.

    Provides an audit trail of every search call, independent of the
    agent's own trace -- captures what *actually* executed.
    """

    def __init__(
        self, search: SearchClient | None = None, allowed_domains: set[str] | None = None
    ) -> None:
        self.search = search
        self.allowed_domains = allowed_domains or set()
        self.execution_log: list[ToolExecution] = []

    def execute_search(
        self, query: str, max_results: int = 5, cutoff_date: date | None = None
    ) -> list[SearchResult]:
        """Execute a search call with logging."""
        if not isinstance(query, str) or not query.strip() or max_results < 1:
            self.execution_log.append(
                ToolExecution(
                    tool_name="search",
                    args={"query": query},
                    status="blocked",
                    duration_ms=0.0,
                    result_preview="Invalid search arguments",
                )
            )
            return []
        if self.search is None:
            self.execution_log.append(
                ToolExecution(
                    tool_name="search", args={"query": query}, status="disabled", duration_ms=0.0
                )
            )
            return []

        start = time.perf_counter()
        try:
            results = eligible_results(
                self.search.search(query, max_results, cutoff_date), cutoff_date
            )
            if self.allowed_domains:
                results = [
                    r
                    for r in results
                    if any(
                        urlparse(r.url).hostname == domain
                        or urlparse(r.url).hostname.endswith("." + domain)
                        for domain in self.allowed_domains
                    )
                ]
            duration = (time.perf_counter() - start) * 1000
            self.execution_log.append(
                ToolExecution(
                    tool_name="search",
                    args={"query": query, "max_results": max_results},
                    status="success",
                    duration_ms=duration,
                    result_preview=f"{len(results)} results",
                    provenance={"source": type(self.search).__name__},
                )
            )
            return results
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            self.execution_log.append(
                ToolExecution(
                    tool_name="search",
                    args={"query": query},
                    status="error",
                    duration_ms=duration,
                    result_preview=str(e)[:200],
                )
            )
            return []

    def get_log_summary(self) -> list[dict[str, Any]]:
        """Summarize execution log for display."""
        return [
            {
                "tool": e.tool_name,
                "status": e.status,
                "duration_ms": e.duration_ms,
                "preview": e.result_preview,
            }
            for e in self.execution_log
        ]


# ---------------------------------------------------------------------------
# Tool schema (for teaching Anthropic/OpenAI format translation)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolDefinition:
    """Typed tool schema that renders to both Anthropic and OpenAI formats."""

    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    required: list[str] = field(default_factory=list)

    def to_anthropic_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            },
        }

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }


# The search tool definition (the only tool in the AIA Forecaster)
SEARCH_TOOL = ToolDefinition(
    name="search",
    description="Search the web for recent news, analysis, and data relevant to a forecasting question.",
    parameters={
        "query": {"type": "string", "description": "Search query string"},
    },
    required=["query"],
)

# Default domain allowlist for financial research
DEFAULT_ALLOWED_DOMAINS: set[str] = {
    "reuters.com",
    "wsj.com",
    "ft.com",
    "bloomberg.com",
    "cnbc.com",
    "nasdaq.com",
    "sec.gov",
    "federalreserve.gov",
    "bls.gov",
    "coindesk.com",
    "cmegroup.com",
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_search_client(provider: str = "") -> SearchClient | None:
    """Create a search client. Returns None if no provider is available.

    Set TAVILY_API_KEY in .env for real search. Missing requested integrations raise an error; mocks require explicit selection.
    """
    provider = (provider or os.environ.get("SEARCH_PROVIDER", "")).strip().lower()
    if provider not in {"", "none", "mock", "tavily"}:
        raise ValueError(f"Unknown search provider: {provider!r}")

    if provider == "none":
        return None

    if provider == "mock":
        return MockSearchClient()

    api_key = os.environ.get("TAVILY_API_KEY", "")
    if api_key:
        return TavilySearchClient(api_key=api_key)

    raise RuntimeError(
        "Search is unavailable. Configure TAVILY_API_KEY, or explicitly select mock/none for a teaching fixture or baseline."
    )


# ---------------------------------------------------------------------------
# Utility: date parsing and result formatting
# ---------------------------------------------------------------------------


def _parse_published_date(published: str) -> date | None:
    """Parse various date formats from published field.

    Returns None for unparseable dates.
    """
    if not published:
        return None

    published = published.strip()

    # ISO 8601
    try:
        return datetime.fromisoformat(published.replace("Z", "+00:00")).date()
    except ValueError:
        pass

    # RFC 2822
    try:
        return parsedate_to_datetime(published).date()
    except (ValueError, TypeError):
        pass

    # Common formats
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(published, fmt).date()
        except ValueError:
            continue

    return None


def format_search_results(results: list[SearchResult]) -> str:
    """Format search results for agent context (as tool message content)."""
    if not results:
        return "No search results found."
    lines: list[str] = ["SEARCH RESULTS:"]
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. {r.title}")
        if r.url:
            lines.append(f"   URL: {r.url}")
        if r.snippet:
            lines.append(f"   {r.snippet}")
        if r.published:
            lines.append(f"   Published: {r.published}")
        lines.append("")
    return "\n".join(lines)
