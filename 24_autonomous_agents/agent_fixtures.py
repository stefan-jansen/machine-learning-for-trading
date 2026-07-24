"""Prediction market question fixtures for the forecasting agent workshop.

Two sources:
- **Live questions** (default): fetched from Polymarket's Gamma API —
  genuinely unresolved, so the LLM cannot answer from training data.
- **Evaluation panel** (static): 10 resolved questions with known outcomes
  for scoring methodology demonstrations (NB09).

Live question functions fall back to static demos when the API is
unavailable (offline, CI, rate-limited).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime

from agent_schemas import ForecastQuestion

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evaluation panel: 10 resolved binary questions
# ---------------------------------------------------------------------------

RESOLVED_QUESTIONS: list[ForecastQuestion] = [
    # --- US Macro ---
    ForecastQuestion(
        question="Will the Federal Reserve cut rates at the March 2025 FOMC meeting?",
        description="Cut = any reduction in the federal funds target rate at the March 18-19, 2025 meeting.",
        resolution_date="2025-03-19",
        cutoff_date="2025-03-10",
        current_market_price=0.04,
        resolved_outcome=0.0,  # Fed held rates
    ),
    ForecastQuestion(
        question="Will US CPI year-over-year for January 2025 come in at or above 3.0%?",
        description="Based on BLS CPI release for January 2025. YoY headline CPI >= 3.0% resolves YES.",
        resolution_date="2025-02-12",
        cutoff_date="2025-02-05",
        current_market_price=0.55,
        resolved_outcome=1.0,  # CPI was 3.0% — resolves YES at >=3.0%
    ),
    # --- Tech Earnings ---
    ForecastQuestion(
        question="Will NVIDIA beat Q4 FY2025 earnings expectations?",
        description="NVIDIA reports Q4 FY2025 on Feb 26, 2025. Beat = revenue AND EPS above consensus estimates.",
        resolution_date="2025-02-26",
        cutoff_date="2025-02-20",
        current_market_price=0.68,
        resolved_outcome=1.0,  # NVIDIA beat on both revenue and EPS
    ),
    ForecastQuestion(
        question="Will TSLA report Q4 2024 deliveries above 500,000 vehicles?",
        description="Tesla Q4 2024 delivery numbers announced Jan 2, 2025. Resolves YES if total deliveries > 500K.",
        resolution_date="2025-01-02",
        cutoff_date="2024-12-28",
        current_market_price=0.62,
        resolved_outcome=0.0,  # Tesla delivered 495K — below the 500K threshold
    ),
    # --- Markets ---
    ForecastQuestion(
        question="Will the S&P 500 close above 6,000 on January 31, 2025?",
        description="Based on the S&P 500 index closing value on Jan 31, 2025.",
        resolution_date="2025-01-31",
        cutoff_date="2025-01-25",
        current_market_price=0.45,
        resolved_outcome=0.0,  # S&P closed at ~5,994 on Jan 31
    ),
    ForecastQuestion(
        question="Will a major tech IPO (>$1B valuation) price in Q1 2025?",
        description="A technology company must complete its IPO on a US exchange with initial valuation above $1B by March 31, 2025.",
        resolution_date="2025-03-31",
        cutoff_date="2025-03-15",
        current_market_price=0.30,
        resolved_outcome=0.0,  # No major tech IPO priced in Q1 2025
    ),
    # --- Crypto ---
    ForecastQuestion(
        question="Will Bitcoin exceed $100,000 before March 1, 2025?",
        description="BTC/USD must trade above $100,000 on any major exchange before March 1, 2025 UTC.",
        resolution_date="2025-03-01",
        cutoff_date="2025-02-20",
        current_market_price=0.42,
        resolved_outcome=0.0,  # Bitcoin did not sustain above $100K before March 1
    ),
    # --- Geopolitical ---
    ForecastQuestion(
        question="Will the US impose new tariffs on Chinese goods before April 2025?",
        description="New tariff actions (executive order, proclamation, or trade measure) targeting Chinese imports announced before April 1, 2025.",
        resolution_date="2025-04-01",
        cutoff_date="2025-03-20",
        current_market_price=0.75,
        resolved_outcome=1.0,  # Trump imposed tariffs in February 2025
    ),
    # --- Technology ---
    ForecastQuestion(
        question="Will OpenAI release GPT-5 before April 2025?",
        description="Official release (not preview/beta) of a model explicitly named GPT-5 by OpenAI before April 1, 2025.",
        resolution_date="2025-04-01",
        cutoff_date="2025-03-20",
        current_market_price=0.15,
        resolved_outcome=0.0,  # GPT-5 not released by April 2025
    ),
    # --- Corporate ---
    ForecastQuestion(
        question="Will Apple announce a foldable device at its Spring 2025 event?",
        description="Apple officially announces (not rumor) a foldable iPhone, iPad, or Mac at any event before July 2025.",
        resolution_date="2025-06-30",
        cutoff_date="2025-03-15",
        current_market_price=0.08,
        resolved_outcome=0.0,  # No foldable device announced
    ),
]


def get_evaluation_panel() -> list[ForecastQuestion]:
    """Return the full evaluation panel of resolved questions."""
    return list(RESOLVED_QUESTIONS)


def get_demo_question() -> ForecastQuestion:
    """Return a single demo question for NB01-NB04 teaching notebooks.

    Uses the NVIDIA earnings question — well-known event with clear resolution.
    """
    return RESOLVED_QUESTIONS[2]  # NVIDIA Q4 FY2025 earnings


def get_demo_questions(n: int = 3) -> list[ForecastQuestion]:
    """Return n demo questions for pipeline demonstrations."""
    return RESOLVED_QUESTIONS[:n]


# ---------------------------------------------------------------------------
# Pinned chapter questions (coordinated across the forecasting arc)
# ---------------------------------------------------------------------------
#
# The forecasting notebooks forecast one of two *pinned* questions so the
# chapter tells a single coherent story rather than whatever market happened to
# be trending the last time each notebook ran. Both are real, currently
# unresolved prediction markets; the prices below are snapshots captured on
# 2026-06-09. Prediction-market forecasts are point-in-time: re-running pulls
# fresh web evidence and will produce different numbers, and once a market
# resolves you should swap in a current question of the same kind.
#
#   CHAPTER_CLEAR_QUESTION      — a macro question whose public evidence points
#       one way, with broad, consistent coverage in the financial press. Used in
#       NB06 to show that independent agents reach similar forecasts here (and
#       run again through the full pipeline in NB08): when the evidence points
#       one direction, different search paths still land close together.
#   CHAPTER_CONTESTED_QUESTION  — a question where credible evidence points both
#       ways. Used in NB04, NB07, NB08, and NB10 to show the wider spread that
#       gives adversarial debate and supervisor reconciliation something to work
#       on.
#
# The contrast is the lesson: forecast spread is a property of how contested the
# question's evidence is, not of the temperature setting. NB09 deliberately uses
# a separate panel of *resolved* questions instead, because scoring calibration
# needs known outcomes.

CHAPTER_CLEAR_QUESTION = ForecastQuestion(
    question="Will the US enter a recession by the end of 2026?",
    description=(
        "Resolves YES if US real GDP (seasonally adjusted annualized) contracts "
        "for two consecutive quarters between Q2 2025 and Q4 2026, or the NBER "
        "declares a recession dated within that window."
    ),
    resolution_date="2027-01-31",
    cutoff_date="",
    current_market_price=0.175,
    resolved_outcome=None,
)

CHAPTER_CONTESTED_QUESTION = ForecastQuestion(
    question="Will the Federal Reserve hike rates in 2026?",
    description=(
        "Resolves YES if the upper bound of the federal funds target rate is "
        "increased at any point between January 1, 2026 and the Fed's December "
        "2026 meeting (scheduled December 8-9, 2026)."
    ),
    resolution_date="2026-12-09",
    cutoff_date="",
    current_market_price=0.545,
    resolved_outcome=None,
)


def get_chapter_clear_question() -> ForecastQuestion:
    """Return the pinned one-directional question (NB06 agreement demo).

    The public evidence points one way, so independent research agents reach
    similar forecasts: the "spread comes from the question, not the temperature"
    result. See the module note on pinned chapter questions.
    """
    return CHAPTER_CLEAR_QUESTION


def get_chapter_contested_question() -> ForecastQuestion:
    """Return the pinned contested question (NB04, NB07, NB08, NB10).

    Credible evidence points both ways, so the agents spread out: the
    disagreement that debate and supervision are designed to reconcile. See the
    module note on pinned chapter questions.
    """
    return CHAPTER_CONTESTED_QUESTION


# ---------------------------------------------------------------------------
# Live questions from Polymarket (Gamma API)
# ---------------------------------------------------------------------------

_GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"

# Financial / economic tags — keeps questions book-relevant and avoids
# politically polarizing or meme content.
_FINANCIAL_TAGS = ("finance", "economy", "stocks", "crypto", "fed")


def _parse_market_price(market: dict) -> float | None:
    """Extract p(YES) from a market dict, handling JSON-string fields."""
    outcomes = market.get("outcomes", "[]")
    prices = market.get("outcomePrices", "[]")
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except (json.JSONDecodeError, TypeError):
            return None

    if not isinstance(outcomes, list) or len(outcomes) != 2:
        return None
    if not isinstance(prices, list) or len(prices) != 2:
        return None

    names = [str(o).strip().lower() for o in outcomes]
    if "yes" not in names:
        return None

    try:
        return float(prices[names.index("yes")])
    except (ValueError, TypeError, IndexError):
        return None


def _fetch_polymarket_markets(n: int = 10) -> list[tuple[dict, float]]:
    """Fetch open binary financial markets from Polymarket's Gamma API.

    Uses the ``/events`` endpoint filtered by financial tag slugs
    (finance, economy, stocks, crypto, fed) so results are book-relevant.

    Returns list of (market_dict, p_yes) tuples sorted by volume,
    or an empty list if the API is unavailable.
    """
    try:
        import httpx
    except ImportError:
        log.debug("httpx not installed — skipping Polymarket fetch")
        return []

    # Collect markets across all financial tags, dedup by market id
    seen: set[str] = set()
    candidates: list[tuple[dict, float]] = []

    for tag in _FINANCIAL_TAGS:
        try:
            resp = httpx.get(
                _GAMMA_EVENTS_URL,
                params={"closed": "false", "tag_slug": tag, "limit": 15},
                timeout=10,
            )
            resp.raise_for_status()
            events = resp.json()
        except Exception as exc:
            log.debug("Polymarket API error (tag=%s): %s", tag, exc)
            continue

        for event in events:
            for m in event.get("markets", []):
                if m.get("closed"):
                    continue
                mid = str(m.get("id", ""))
                if mid in seen:
                    continue
                seen.add(mid)

                p_yes = _parse_market_price(m)
                if p_yes is None:
                    continue
                # Skip near-resolved markets (keep 8-92% range)
                if p_yes > 0.92 or p_yes < 0.08:
                    continue

                candidates.append((m, p_yes))

    candidates.sort(key=lambda x: float(x[0].get("volume", 0) or 0), reverse=True)
    return candidates[:n]


def _sanitize_for_prompt(text: str, max_len: int) -> str:
    """Strip control characters and bound length for safe LLM prompt interpolation."""
    cleaned = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 0x20)
    return cleaned[:max_len]


def _market_to_question(market: dict, p_yes: float) -> ForecastQuestion:
    """Convert a Polymarket market dict to a ForecastQuestion.

    Title and description are sanitized (control chars stripped, bounded length)
    before interpolation into LLM prompts to limit prompt-injection risk via
    attacker-controlled market metadata.
    """
    raw_title = str(market.get("question") or market.get("title") or "")
    raw_description = market.get("description") or ""
    title = _sanitize_for_prompt(raw_title, max_len=300)
    description = _sanitize_for_prompt(raw_description, max_len=500)
    end_date = str(market.get("endDate") or "")
    if len(end_date) >= 10:
        end_date = end_date[:10]  # YYYY-MM-DD
    else:
        end_date = ""

    return ForecastQuestion(
        question=title,
        description=description,
        resolution_date=end_date,
        cutoff_date="",
        current_market_price=round(p_yes, 3),
        resolved_outcome=None,
    )


def get_live_question(source: str = "polymarket") -> ForecastQuestion:
    """Fetch a single live prediction market question.

    Falls back to get_demo_question() if the API is unavailable.
    """
    if source != "polymarket":
        raise ValueError(f"Unsupported source: {source!r}; only 'polymarket' is implemented")
    markets = _fetch_polymarket_markets(n=1)
    if markets:
        return _market_to_question(*markets[0])
    log.info("Polymarket unavailable — falling back to static demo question")
    return get_demo_question()


def get_live_questions(n: int = 3, source: str = "polymarket") -> list[ForecastQuestion]:
    """Fetch n live prediction market questions.

    Falls back to get_demo_questions(n) if the API is unavailable.
    """
    if source != "polymarket":
        raise ValueError(f"Unsupported source: {source!r}; only 'polymarket' is implemented")
    markets = _fetch_polymarket_markets(n=n)
    if markets:
        return [_market_to_question(m, p) for m, p in markets]
    log.info("Polymarket unavailable — falling back to %d static demo questions", n)
    return get_demo_questions(n)


# ---------------------------------------------------------------------------
# Live questions from Kalshi (Trade API v2)
# ---------------------------------------------------------------------------

_KALSHI_MARKET_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"


def _kalshi_field_float(market: dict, *names: str) -> float | None:
    """Parse a Kalshi *_dollars or numeric field across schema generations."""
    for name in names:
        v = market.get(name)
        if v in (None, ""):
            continue
        try:
            return float(v)
        except (ValueError, TypeError):
            continue
    return None


def _kalshi_market_to_question(market: dict) -> ForecastQuestion:
    """Convert a Kalshi market dict to a ForecastQuestion.

    Probability is taken from ``last_price_dollars`` (last trade) or the
    mid of ``yes_bid_dollars`` / ``yes_ask_dollars`` when no last trade
    is available.
    """
    last = _kalshi_field_float(market, "last_price_dollars")
    bid = _kalshi_field_float(market, "yes_bid_dollars")
    ask = _kalshi_field_float(market, "yes_ask_dollars")
    if last is not None and last > 0:
        p_yes = last
    elif bid is not None and ask is not None:
        p_yes = (bid + ask) / 2
    elif bid is not None:
        p_yes = bid
    else:
        raise RuntimeError(f"Kalshi market {market.get('ticker')!r} has no price data")

    raw_title = str(market.get("title") or market.get("ticker") or "")
    raw_subtitle = str(market.get("yes_sub_title") or market.get("subtitle") or "")
    raw_rules = str(market.get("rules_primary") or "")
    title = _sanitize_for_prompt(raw_title, max_len=300)
    description = _sanitize_for_prompt(f"{raw_subtitle}. {raw_rules}".strip(". "), max_len=500)
    close = str(market.get("close_time") or "")
    resolution_date = close[:10] if len(close) >= 10 else ""

    return ForecastQuestion(
        question=title,
        description=description,
        resolution_date=resolution_date,
        cutoff_date="",
        current_market_price=round(p_yes, 3),
        resolved_outcome=None,
    )


# Static Fed-policy stand-in. Used as the last-resort fallback when both the
# pinned ticker AND the event-series search fail (Kalshi outage, schema
# change, all FEDHIKE markets simultaneously resolved). Mirrors the
# chapter-narrative anchor so §24.7's contested-question story still
# has a coherent question to attach to.
_FED_FALLBACK_QUESTION = ForecastQuestion(
    question="Will the Federal Reserve hike rates in the next 12 months?",
    description=(
        "Contested macro forecast on near-term Fed policy. Static stand-in "
        "used when no live Kalshi FEDHIKE market resolves; the agent runs "
        "against this question with no live market price."
    ),
    resolution_date="",
    cutoff_date="",
    current_market_price=None,
    resolved_outcome=None,
)


def _find_active_event_market(
    event_ticker: str,
    *,
    price_min: float = 0.20,
    price_max: float = 0.80,
    min_days_to_close: int = 180,
) -> dict | None:
    """Return the most-traded active market in a Kalshi event series.

    Filters: status=active, last-trade price within ``[price_min, price_max]``
    (contested), close_time at least ``min_days_to_close`` days out. Picks
    the highest-volume match. Returns ``None`` if no qualifying market
    exists or the API is unavailable.
    """
    try:
        import httpx
    except ImportError:
        return None

    try:
        resp = httpx.get(
            _KALSHI_MARKET_URL,
            params={"event_ticker": event_ticker, "status": "open", "limit": 50},
            timeout=10,
        )
        resp.raise_for_status()
        markets = resp.json().get("markets", []) or []
    except Exception as exc:
        log.debug("Kalshi event-series search failed for %s: %s", event_ticker, exc)
        return None

    today = datetime.now(UTC).date()
    candidates: list[tuple[float, dict]] = []
    for m in markets:
        if m.get("status") != "active":
            continue
        price = _kalshi_field_float(m, "last_price_dollars")
        if price is None or not (price_min <= price <= price_max):
            continue
        close = str(m.get("close_time") or "")[:10]
        try:
            close_date = date.fromisoformat(close)
        except ValueError:
            continue
        if (close_date - today).days < min_days_to_close:
            continue
        volume = float(m.get("volume", 0) or 0)
        candidates.append((volume, m))

    if not candidates:
        return None
    candidates.sort(key=lambda v: v[0], reverse=True)
    return candidates[0][1]


def get_kalshi_question(ticker: str) -> ForecastQuestion:
    """Fetch a Kalshi market by ticker, rolling forward when the pinned market resolves.

    Used by NB06 to pin the multi-agent demo to a high-uncertainty macro
    question. Fallback chain:

    1. Try the pinned ticker. If it resolves and the captured probability
       is still in the contested band, return it.
    2. If the pinned ticker is unavailable (404, market settled), search
       the same event series (parsed as the prefix before the first ``-``)
       for the highest-volume active market in the contested band with a
       sufficiently far resolution date. This rolls forward across cycles
       — when ``FEDHIKE-26DEC31`` resolves, the search finds
       ``FEDHIKE-27DEC31`` and the chapter narrative still attaches.
    3. As a last resort, return a static Fed-policy stand-in so the
       notebook completes even when Kalshi is unreachable.
    """
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx is required for Kalshi fetch") from exc

    # Step 1: try the pinned ticker
    try:
        resp = httpx.get(f"{_KALSHI_MARKET_URL}/{ticker}", timeout=10)
        if resp.status_code == 200:
            market = resp.json().get("market")
            if market and market.get("status") == "active":
                return _kalshi_market_to_question(market)
            log.info("Kalshi market %s is no longer active — searching event series", ticker)
        else:
            log.info(
                "Kalshi ticker %s returned %d — searching event series", ticker, resp.status_code
            )
    except Exception as exc:
        log.info("Kalshi unavailable for %s (%s) — searching event series", ticker, exc)

    # Step 2: roll forward within the same event series
    event_ticker = ticker.split("-", 1)[0]
    fallback = _find_active_event_market(event_ticker)
    if fallback is not None:
        log.info(
            "Kalshi event-series roll-forward: %s → %s (close=%s)",
            ticker,
            fallback.get("ticker"),
            str(fallback.get("close_time", ""))[:10],
        )
        return _kalshi_market_to_question(fallback)

    # Step 3: static Fed-policy stand-in
    log.info(
        "Kalshi event series %s has no qualifying active market — using static Fed stand-in",
        event_ticker,
    )
    return _FED_FALLBACK_QUESTION
