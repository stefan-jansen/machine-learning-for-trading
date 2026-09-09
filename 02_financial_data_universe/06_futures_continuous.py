# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Constructing Continuous Futures Contracts
#
# **Docker image**: `ml4t`
#
# **Purpose**: Walk through the construction of a continuous futures price
# series from individual expiring contracts: detect rolls, compare
# adjustment methods (raw, Panama / additive back-adjustment, ratio /
# multiplicative back-adjustment), and validate against the vendor-built
# continuous series.
#
# **Learning objectives**:
#
# - Detect roll dates using volume-based front-month identification (with a
#   no-rollback constraint to avoid spurious switches).
# - Apply Panama (additive) back-adjustment to preserve dollar P&L across
#   rolls.
# - Apply ratio (multiplicative) back-adjustment to preserve percentage
#   returns across rolls.
# - Cross-check constructed continuous prices against Databento's pre-built
#   continuous series and quantify the disagreement.
#
# **Book reference**: §2.2, "The asset-class market data landscape" - the futures part of
# it. The methodology comparison here is what underpins that section's engineering
# decision to store raw contract histories alongside one or more continuous variants.
#
# **Prerequisites**: `data` package on `PYTHONPATH`; individual ES contract
# parquet at `ML4T_DATA_PATH/futures/market/individual/ES/data.parquet` and
# the contract-definitions parquet at
# `ML4T_DATA_PATH/futures/market/contract_definitions.parquet`.

# %%
"""Continuous Futures Construction."""

import re
from datetime import UTC, datetime, timedelta

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from data import load_cme_futures
from utils import ML4T_DATA_PATH
from utils.style import COLORS, show_plotly_with_alt

# %% [markdown]
# ### Two declared parameters
#
# `MIN_OUTRIGHT_PRICE` separates outright contracts from calendar spreads. CME lists both, and
# a spread trades at the inter-month price difference rather than at the index level, so a
# price floor tells them apart without needing a separate instrument-type field. Raise it for
# a higher-priced index, lower it for a cheaper one.
#
# `CALENDAR_ROLL_SESSIONS_BEFORE` is how far ahead of a contract's last trading day the
# calendar method switches to the next one, counted in **trading sessions** rather than
# calendar days. Five sessions is the conservative end of the range used for equity index
# futures, and the unit matters here: every ES expiry is a Friday, so five calendar days before
# one lands on the preceding Sunday and pushes the switch out to the Monday.
#
# Both are declared here so Papermill can override them for CI, and so that nothing further
# down repeats the value as a default.

# %% tags=["parameters"]
MIN_OUTRIGHT_PRICE = 500.0
CALENDAR_ROLL_SESSIONS_BEFORE = 5

# %% [markdown]
# ## 1. Understanding the Data
#
# ### Loading the individual contracts

# %%
es_individual = load_cme_futures(products=["ES"], frequency="hourly", continuous=False)

print(f"Individual contracts: {es_individual.shape}")
print(f"Unique contracts (by instrument_id): {es_individual['instrument_id'].n_unique()}")
print(f"Date range: {es_individual['timestamp'].min()} to {es_individual['timestamp'].max()}")
print("Sample:")
es_individual.head()

# %% [markdown]
# ### The bar length is in the data, not in the argument name
#
# `frequency="hourly"` selects the raw per-contract capture rather than the session-aggregated
# daily file that `05_futures_session_aggregation` writes. It does not promise hourly bars, and
# for individual contracts it does not deliver them: Databento captured this feed at daily
# resolution. The capture records its own bar length in `rtype` - 34 is a one-hour bar, 35 a
# one-day bar - so the frame can be asked rather than assumed, and the modal gap between
# consecutive timestamps confirms the answer independently.
#
# This matters for the rest of the notebook. The validation section compares what we build against the
# vendor's continuous series, and that series *is* hourly. Joining the two on a bare timestamp
# without noticing would silently pair a whole day against one hour of it.

# %%
_RTYPE_BAR = {32: "1 second", 33: "1 minute", 34: "1 hour", 35: "1 day"}


def describe_bars(frame: pl.DataFrame, label: str) -> None:
    """Report a frame's declared bar length and the spacing actually observed."""
    codes = frame["rtype"].unique().to_list()
    declared = ", ".join(_RTYPE_BAR.get(c, f"rtype {c}") for c in sorted(codes))
    stamps = frame.select("timestamp").unique().sort("timestamp")["timestamp"]
    modal_gap = stamps.diff().drop_nulls().value_counts().sort("count", descending=True)[0, 0]
    print(f"{label}: declared {declared}; most common gap between timestamps {modal_gap}")


describe_bars(es_individual, "individual ES contracts")

# %%
contract_stats = (
    es_individual.group_by("instrument_id")
    .agg(
        pl.col("timestamp").min().alias("first_trade"),
        pl.col("timestamp").max().alias("last_trade"),
        pl.col("volume").sum().alias("total_volume"),
        pl.len().alias("trading_days"),
    )
    .sort("first_trade")
)

print(f"Contracts: {len(contract_stats)} (sorted by first trade)")
contract_stats.head(10)

# %% [markdown]
# ### Reading a contract symbol
#
# A CME contract symbol is the product code, then a month code, then a year code:
#
# - H = March, M = June, U = September, Z = December (the standard quarterly cycle)
# - F = January, G = February, J = April, K = May, N = July, Q = August, V = October, X = November
#
# The month code is unambiguous. **The year code is not**, and the definitions file this
# notebook reads writes it with a single digit. `ESM1` is the June contract of a year ending in
# 1, which over a long enough history could be 2001, 2011, 2021 or 2031. Nothing in the symbol
# says which. A parser that assumes a two-digit year and pads it reads `ESM1` as June 2001.
#
# The next cell measures what that assumption costs on this file, rather than asserting that it
# is a problem.

# %%
_MONTH_CODES = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}
_SYMBOL_RE = re.compile(r"^([A-Z]+)([FGHJKMNQUVXZ])(\d+)$")


def parse_contract_symbol(symbol: str) -> dict:
    """Split a contract symbol into the parts the symbol actually determines.

    The product code and the delivery month are fully determined by the symbol. The year is
    not, so it is returned as the raw digits and a decade has to come from somewhere else.
    """
    match = _SYMBOL_RE.match(symbol)
    if not match:
        raise ValueError(f"Cannot parse symbol: {symbol}")
    product, month_code, year_code = match.groups()
    return {
        "product": product,
        "month_code": month_code,
        "month": _MONTH_CODES[month_code],
        "year_code": year_code,
    }


def naive_year(year_code: str) -> int:
    """The two-digit-year assumption, kept only so its cost can be measured below."""
    year = int(year_code)
    return year + 2000 if year < 50 else year + 1900


# %%
defn_path = ML4T_DATA_PATH / "futures" / "market" / "contract_definitions.parquet"
contract_defs = pl.read_parquet(defn_path).filter(pl.col("product") == "ES")
contract_df = (
    pl.DataFrame(
        [
            {**parse_contract_symbol(r["symbol"]), "symbol": r["symbol"]}
            for r in contract_defs.iter_rows(named=True)
        ]
    )
    .join(contract_defs.select("symbol", "expiration"), on="symbol")
    .with_columns(
        pl.col("expiration").dt.year().alias("year"),
        pl.col("year_code").map_elements(naive_year, return_dtype=pl.Int64).alias("naive_year"),
    )
    .sort("expiration")
)

_year_code_widths = sorted({len(c) for c in contract_df["year_code"]})
_wrong = contract_df.filter(pl.col("naive_year") != pl.col("year"))
print(f"ES contract definitions: {contract_df.height} contracts")
print(f"Year codes in this file are {_year_code_widths} digit(s) wide")
print(
    f"Contracts the two-digit-year assumption dates to the wrong decade: "
    f"{_wrong.height} of {contract_df.height}"
)
print("The delivery month it reads off the symbol is right for every one of them.")
contract_df.select("symbol", "month_code", "month", "naive_year", "year", "expiration").head(10)

# %% [markdown]
# The assumption is wrong on every contract in the file, and the printed table shows why it is
# the kind of error a reader is unlikely to catch: `naive_year` and `expiration` sit on the
# same row and contradict each other, and nothing about the frame looks empty or missing.
#
# The fix is not a cleverer parser. A one-digit year code does not carry a decade, so no rule
# over the symbol alone can recover one. The `expiration` column is the authority, and `year`
# above is read from it. Where a definitions file is not available, Recovering expiration from the price history recovers the
# same information from the price history itself.

# %% [markdown]
# ### Recovering expiration from the price history
#
# The definitions file gives expirations by symbol, and the price bars are keyed by
# `instrument_id`. Neither file carries the other's key, so the two cannot be joined, and the
# calendar-roll method below needs an expiry per *contract in the bars*.
#
# It can be recovered without them. A futures contract stops printing when it stops existing,
# so the last date on which a contract trades is its last trading day. That is an observation
# rather than a rule, which makes it worth checking against what the rule would say: ES expires
# on the third Friday of the delivery month, so every last trading day should be a Friday
# falling between the 15th and the 21st.

# %%
contract_life = (
    es_individual.filter(pl.col("close") >= MIN_OUTRIGHT_PRICE)
    .with_columns(pl.col("timestamp").dt.date().alias("date"))
    .group_by("instrument_id")
    .agg(
        pl.col("date").min().alias("first_trade"),
        pl.col("date").max().alias("last_trade"),
        pl.col("volume").sum().alias("total_volume"),
        pl.len().alias("sessions"),
    )
    .sort("last_trade")
)

_liquid = contract_life.filter(pl.col("total_volume") >= pl.col("total_volume").median())
_third_friday = _liquid.filter(
    (pl.col("last_trade").dt.weekday() == 5) & (pl.col("last_trade").dt.day().is_between(15, 21))
)
# ES expires on the third Friday of the delivery month. Contracts whose last print falls there
# are the ones whose observed end really is an expiry; the rest simply stopped trading.
scheduled_contracts = contract_life.filter(
    (pl.col("last_trade").dt.weekday() == 5) & (pl.col("last_trade").dt.day().is_between(15, 21))
)
_all_third_friday = scheduled_contracts
print(f"Outright contracts with a price history: {contract_life.height}")
print(
    f"Ending on the third Friday of their month: {_all_third_friday.height} of "
    f"{contract_life.height} overall, {_third_friday.height} of {_liquid.height} among those "
    f"with above-median volume"
)
contract_life.filter(pl.col("total_volume") >= pl.col("total_volume").quantile(0.9)).sort(
    "last_trade"
).head(10)

# %% [markdown]
# The rule holds without exception for the contracts carrying real volume, which is the
# population the roll logic cares about. It does not hold for every contract, and that is what
# should be expected: the ones that miss are thin deferred listings whose last print is the day
# the last interested party lost interest, which is not an expiration date at all.
#
# The claim is therefore reported as two counts over two populations rather than as a rule.
# Stating it as "last trade is expiry" would be false for the file as a whole and true for the
# part of it the notebook uses, and the difference between those is the reason to count.

# %% [markdown]
# ## 2. Roll Detection
#
# The "roll" is when we switch from the near-month contract to the next contract.
# There are several approaches:
#
# 1. **Volume-based**: Roll when the next contract has higher daily volume
# 2. **Open Interest-based**: Roll when next contract has higher open interest
# 3. **Fixed Schedule**: Roll N days before expiration (e.g., first Thursday of expiry month)
#
# We implement volume-based rolling first, then calendar rolling, and compare
# the two.
#
# The identifier throughout is `instrument_id`, not the `ESH24`-style symbol above.
# The price file is keyed by instrument id and nothing joins it to the symbol table, so the
# columns below are named for what they hold.


# %%
def identify_front_month(
    individual_df: pl.DataFrame, min_outright_price: float = MIN_OUTRIGHT_PRICE
) -> pl.DataFrame:
    """Pick each day's front contract by traded volume, and never return to a retired one."""
    # Calendar spreads trade at the inter-month difference rather than the index level, so a
    # price floor separates outright contracts from them.
    outrights = individual_df.filter(pl.col("close") >= min_outright_price)

    daily_volume = (
        outrights.with_columns(pl.col("timestamp").dt.date().alias("date"))
        .group_by(["date", "instrument_id"])
        .agg(pl.col("volume").sum().alias("daily_volume"))
    )

    daily_leader = (
        daily_volume.group_by("date")
        .agg(pl.col("instrument_id").sort_by("daily_volume").last().alias("volume_leader"))
        .sort("date")
    )

    # No-rollback constraint: adopt a new leader, but never switch back to one already retired.
    leader_ids = daily_leader["volume_leader"].to_list()
    dates = daily_leader["date"].to_list()
    retired = {leader_ids[0]}
    current_front = leader_ids[0]
    front = [current_front]

    for i in range(1, len(leader_ids)):
        if leader_ids[i] != current_front and leader_ids[i] not in retired:
            current_front = leader_ids[i]
            retired.add(current_front)
        front.append(current_front)

    daily_front = pl.DataFrame(
        {"date": dates, "volume_leader": leader_ids, "front_instrument_id": front}
    )

    bars = individual_df.select("timestamp").unique().sort("timestamp")
    bars = bars.with_columns(pl.col("timestamp").dt.date().alias("date"))
    front_month = bars.join(daily_front, on="date", how="left").drop("date")

    front_month = front_month.with_columns(
        pl.col("front_instrument_id").shift(1).alias("prev_front"),
    ).with_columns(
        (pl.col("front_instrument_id") != pl.col("prev_front")).alias("is_roll"),
    )
    return front_month


# %%
front_months = identify_front_month(es_individual, min_outright_price=MIN_OUTRIGHT_PRICE)
print("Front month identification (2024 sample):")
front_months.filter(pl.col("timestamp") >= datetime(2024, 1, 1, tzinfo=UTC)).head(20)

# %%
roll_dates = front_months.filter(pl.col("is_roll"))
print(f"Roll events detected: {len(roll_dates)}")
print("Most recent 10 rolls:")
roll_dates.tail(10).select("timestamp", "prev_front", "front_instrument_id")

# %% [markdown]
# ### What the no-rollback constraint is worth
#
# The loop above exists to stop the series flickering between two contracts when their daily
# volumes are close. That is a plausible failure and a cheap guard, and it is also a claim
# about this data that can be checked: the constraint changes the answer on exactly those days
# where the raw volume leader differs from the contract the constraint holds us to.
#
# It is worth checking rather than assuming, because the guard is not free of risk in the other
# direction. It is irreversible: one spurious leader is adopted permanently, and the series can
# never return to the contract that was really the front month.

# %%
_disagree = front_months.filter(pl.col("volume_leader") != pl.col("front_instrument_id")).select(
    "timestamp", "volume_leader", "front_instrument_id"
)
print(
    f"Days where the raw volume leader differs from the no-rollback front: "
    f"{_disagree.height} of {front_months.height}"
)
if _disagree.height:
    print(_disagree.head(10))
else:
    print("On this history the constraint never changes the contract selected.")

# %% [markdown]
# On ES over this sample the constraint never fires: the raw volume leader is already
# monotone, so the guarded series and the unguarded one are the same series. The guard should
# stay - it costs nothing and the flicker it prevents is real on thinner products - but the
# notebook should not tell the reader it is what makes the roll dates come out right here,
# because on this data it makes no difference at all. What produces the roll dates reported
# above is the volume rule itself.

# %% [markdown]
# ### Rolling on the calendar instead
#
# The alternative is to ignore volume and roll a fixed number of days before the contract stops
# trading. It is predictable and reproducible - the roll dates can be published a year ahead -
# and it does not depend on a volume field that a vendor may revise.
#
# Each contract's last trading day was recovered from its own price history, so this
# runs on the same frame as the volume method and the two can be compared on their output
# rather than on their descriptions.


# %% [markdown]
# Counting sessions needs one more step, and it is the same point the notebook has already made
# twice. These daily bars are keyed by UTC calendar date, and a CME session spans two of them:
# the session that ends on Monday afternoon opens on Sunday evening, so the file carries a
# Sunday-dated bar holding those few opening hours.
#
# Treating each date as a session would therefore count Sunday as a trading day, and five
# "sessions" back from a Friday expiry would land on the preceding Sunday rather than the
# preceding Friday - the exact error that counting calendar days makes, arrived at by a
# different route. The Sunday bar is folded onto the Monday session it belongs to, and the cell
# below shows the volume evidence that this is what it is.


# %%
def session_grid(individual_df: pl.DataFrame) -> pl.DataFrame:
    """Number the trading sessions in the file and map each UTC date onto its session.

    A Sunday-dated bar holds the opening hours of Monday's session, so it carries Monday's
    session number rather than one of its own.
    """
    dates = individual_df.select(pl.col("timestamp").dt.date().alias("date")).unique().sort("date")
    dates = dates.with_columns(
        pl.when(pl.col("date").dt.weekday() == 7)
        .then(pl.col("date").dt.offset_by("1d"))
        .otherwise(pl.col("date"))
        .alias("session_date")
    )
    numbering = (
        dates.select("session_date")
        .unique()
        .sort("session_date")
        .with_columns(pl.int_range(pl.len()).alias("session_no"))
    )
    return dates.join(numbering, on="session_date", how="left").select(
        "date", "session_date", "session_no"
    )


# %%
_grid = session_grid(es_individual)
_by_weekday = (
    es_individual.with_columns(pl.col("timestamp").dt.date().alias("date"))
    .group_by("date")
    .agg(pl.col("volume").sum().alias("volume"))
    .with_columns(pl.col("date").dt.weekday().alias("weekday"))
    .group_by("weekday")
    .agg(pl.col("volume").median().alias("median_volume"), pl.len().alias("dates"))
    .sort("weekday")
)
print(f"UTC dates in the file: {_grid.height}")
print(f"Trading sessions once Sunday folds onto Monday: {_grid['session_no'].n_unique()}")
print("Median volume per UTC date, by weekday (1 = Monday):")
_by_weekday

# %% [markdown]
# A Sunday-dated bar carries a small fraction of a weekday's volume, which is what a few
# evening hours look like beside a full session. It is not a quiet trading day; it is part of
# the next one.


# %%
def identify_front_month_calendar(
    individual_df: pl.DataFrame,
    scheduled_df: pl.DataFrame,
    roll_sessions_before: int = CALENDAR_ROLL_SESSIONS_BEFORE,
    min_outright_price: float = MIN_OUTRIGHT_PRICE,
) -> pl.DataFrame:
    """Pick each day's front contract as the nearest one still more than N sessions from expiry.

    ``scheduled_df`` must hold only contracts whose last trading day is a real expiry. A thin
    listing that merely stopped printing would otherwise be treated as an expiring front month
    and pull the series onto it for a few days.

    The offset counts trading sessions taken from the data rather than calendar days, because
    an ES expiry is always a Friday and counting calendar days back from one lands on a
    weekend.
    """
    sessions = session_grid(individual_df)

    last_day = (
        scheduled_df.select("instrument_id", "last_trade")
        .join(sessions, left_on="last_trade", right_on="date", how="inner")
        .with_columns((pl.col("session_no") - roll_sessions_before).alias("roll_out_session"))
        .select("instrument_id", "last_trade", "roll_out_session")
    )

    candidates = (
        individual_df.filter(pl.col("close") >= min_outright_price)
        .with_columns(pl.col("timestamp").dt.date().alias("date"))
        .join(sessions, on="date", how="inner")
        .join(last_day, on="instrument_id", how="inner")
        .filter(pl.col("session_no") < pl.col("roll_out_session"))
    )

    front = (
        candidates.sort(["date", "last_trade", "instrument_id"])
        .group_by("date")
        .first()
        .select("date", pl.col("instrument_id").alias("front_instrument_id"))
        .sort("date")
    )

    return front.with_columns(
        pl.col("front_instrument_id").shift(1).alias("prev_front")
    ).with_columns((pl.col("front_instrument_id") != pl.col("prev_front")).alias("is_roll"))


# %%
calendar_front = identify_front_month_calendar(es_individual, scheduled_contracts)
calendar_rolls = calendar_front.filter(pl.col("is_roll"))

volume_front_daily = (
    front_months.with_columns(pl.col("timestamp").dt.date().alias("date"))
    .group_by("date")
    .agg(pl.col("front_instrument_id").last())
    .sort("date")
)

both = volume_front_daily.join(
    calendar_front.select("date", pl.col("front_instrument_id").alias("calendar_front")),
    on="date",
    how="inner",
)
agree = both.filter(pl.col("front_instrument_id") == pl.col("calendar_front"))

_uncovered = volume_front_daily.join(calendar_front.select("date"), on="date", how="anti")

print(f"Volume-based rolls:   {len(roll_dates)}")
print(f"Calendar-based rolls: {calendar_rolls.height}")
print(
    f"Days the two methods hold the same contract: {agree.height} of {both.height} "
    f"({100 * agree.height / both.height:.1f}%)"
)
if _uncovered.height:
    print(
        f"Days the calendar method cannot cover: {_uncovered.height} "
        f"({_uncovered['date'].min()} to {_uncovered['date'].max()})"
    )

# %% [markdown]
# ### Volume against calendar
#
# The two methods select the same contract on almost every day and differ only around the
# rolls. How long each disagreement lasts is the number the choice between them turns on, so
# the episodes are measured rather than described.
#
# The comparison also stops short of the end of the sample, and the count above says by how
# much. The nearest contract when the download ends has not expired, so its last observed print
# is the day the data stops rather than an expiry, and it fails the third-Friday test the
# scheduled set is built from. A production pipeline reads expirations from the exchange
# calendar and has no such gap. A notebook working from price history alone cannot know the
# expiry of a contract that is still trading, and reporting the excluded days is the honest
# alternative to letting an inner join swallow them.

# %% [markdown]
# Episodes are numbered on the complete ordered comparison frame rather than on the disagreeing
# rows alone. Numbering them by the gaps between disagreement dates would split any episode
# spanning a weekend or a holiday into two, and report the roll disagreement as lasting half as
# long as it really does.

# %%
_flagged = both.sort("date").with_columns(
    (pl.col("front_instrument_id") != pl.col("calendar_front")).alias("differs")
)
_flagged = _flagged.with_columns(
    (pl.col("differs") & ~pl.col("differs").shift(1).fill_null(False)).cum_sum().alias("episode")
)
_disagreements = _flagged.filter(pl.col("differs"))
if _disagreements.height:
    _episode_lengths = (
        _disagreements.join(_grid.select("date", "session_date"), on="date", how="left")
        .group_by("episode")
        .agg(
            pl.col("date").min().alias("from"),
            pl.col("date").max().alias("to"),
            pl.col("session_date").n_unique().alias("sessions"),
        )
        .sort("from")
    )
    print(f"Disagreement episodes: {_episode_lengths.height}")
    print(
        f"Length in trading sessions: median {_episode_lengths['sessions'].median():.0f}, "
        f"max {_episode_lengths['sessions'].max()}"
    )
    _episode_lengths.tail(10)
else:
    print("The two methods never disagree on this history.")

# %% [markdown]
# **What the choice comes down to.** On ES the two methods land within a session or two of each
# other, so the cost of picking one over the other is a day or two of holding the other
# contract, a few times a year. That is a statement about a deeply liquid index future whose
# volume crossover is sharp, not a general result: on a product where liquidity migrates
# gradually, the volume rule would drift away from any fixed schedule for much longer.
#
# The tradeoff is otherwise the familiar one. Volume rolling follows the liquidity, so the
# series is always on the contract most people are actually trading, but its roll date is
# known only after the fact and moves with the market. Calendar rolling fixes the date in
# advance and anyone with an expiry schedule can reproduce it, at the cost of holding a
# contract through a stretch where the next one is already the more liquid of the two.
#
# The rest of the notebook uses volume-based detection, because the validation section holds our series against a
# vendor series built the same way and comparing like with like is the point of that section.

# %% [markdown]
# ## 3. Adjustment Methods
#
# When we roll from contract A to contract B, there's usually a price gap.
# If we don't adjust, our time series will have artificial jumps.
#
# ### No adjustment (raw)
#
# Simply use prices as-is. Returns calculated on roll dates are invalid.


# %%
def create_continuous_raw(individual_df: pl.DataFrame, front_months: pl.DataFrame) -> pl.DataFrame:
    """Create continuous series with no adjustment (raw prices)."""
    # Join individual prices with front month info
    continuous = (
        individual_df.join(
            front_months.select(["timestamp", "front_instrument_id"]), on="timestamp", how="inner"
        )
        .filter(pl.col("instrument_id") == pl.col("front_instrument_id"))
        .select(["timestamp", "open", "high", "low", "close", "volume", "instrument_id"])
        .sort("timestamp")
    )

    return continuous


# %%
es_continuous_raw = create_continuous_raw(es_individual, front_months)
print(
    f"Raw continuous series: {len(es_continuous_raw)} daily bars, one per session on the front contract"
)
es_continuous_raw.head(10)

# %% [markdown]
# ### Panama, or additive back-adjustment
#
# Add the price gap to all historical prices. This preserves dollar P&L
# but distorts percentage returns for old data.
#
# Gap = Close_new_contract - Close_old_contract
# Adjusted_price = Price + cumulative_gap
#
# Note: We add (not subtract) because we're bringing old prices UP to the
# current contract's level, eliminating the discontinuity at roll dates.


# %%
def _compute_roll_gaps(individual_df: pl.DataFrame, front_months: pl.DataFrame) -> pl.DataFrame:
    """Compute price gaps at each roll date (new - old contract close)."""
    roll_info = front_months.filter(pl.col("is_roll"))
    prices_lookup = individual_df.select(["timestamp", "instrument_id", "close"])

    old_prices = (
        roll_info.select(["timestamp", pl.col("prev_front").alias("instrument_id")])
        .join(prices_lookup, on=["timestamp", "instrument_id"], how="left")
        .rename({"close": "old_close"})
    )

    new_prices = (
        roll_info.select(["timestamp", pl.col("front_instrument_id").alias("instrument_id")])
        .join(prices_lookup, on=["timestamp", "instrument_id"], how="left")
        .rename({"close": "new_close"})
    )

    return (
        old_prices.select(["timestamp", "old_close"])
        .join(new_prices.select(["timestamp", "new_close"]), on="timestamp", how="inner")
        .with_columns((pl.col("new_close") - pl.col("old_close")).alias("gap"))
        .select(["timestamp", "gap"])
        .drop_nulls()
    )


# %% [markdown]
# ### Panama Adjustment
#
# Apply the computed gaps cumulatively backwards through the raw series.


# %%
def create_continuous_panama(
    individual_df: pl.DataFrame, front_months: pl.DataFrame
) -> pl.DataFrame:
    """Create continuous series with Panama (back) adjustment.

    Uses vectorized Polars joins instead of row-by-row iteration for O(n) complexity.
    """
    raw = create_continuous_raw(individual_df, front_months)
    roll_info = front_months.filter(pl.col("is_roll"))

    if len(roll_info) == 0:
        return raw.with_columns(pl.lit(0.0).alias("cumulative_adjustment"))

    gaps_df = _compute_roll_gaps(individual_df, front_months)

    if len(gaps_df) == 0:
        return raw.with_columns(pl.lit(0.0).alias("cumulative_adjustment"))

    # Adjustment applies to dates STRICTLY BEFORE each roll date
    raw_with_gaps = raw.join(gaps_df, on="timestamp", how="left").with_columns(
        pl.col("gap").fill_null(0.0)
    )

    # Cumulative sum in reverse, shift by 1 to exclude roll date from adjustment
    raw_with_gaps = raw_with_gaps.with_columns(
        pl.col("gap")
        .reverse()
        .cum_sum()
        .shift(1)
        .fill_null(0.0)
        .reverse()
        .alias("cumulative_adjustment")
    )

    adjusted = raw_with_gaps.with_columns(
        [
            (pl.col("open") + pl.col("cumulative_adjustment")).alias("adj_open"),
            (pl.col("high") + pl.col("cumulative_adjustment")).alias("adj_high"),
            (pl.col("low") + pl.col("cumulative_adjustment")).alias("adj_low"),
            (pl.col("close") + pl.col("cumulative_adjustment")).alias("adj_close"),
        ]
    )

    return adjusted


# %%
es_continuous_panama = create_continuous_panama(es_individual, front_months)
panama_first = es_continuous_panama["cumulative_adjustment"][0]
panama_first_close = es_continuous_panama["close"][0]
print(
    f"Panama: the adjustment carried back to the start of the series is {panama_first:+.2f} "
    f"index points"
)
print(
    f"  applied to the earliest close of {panama_first_close:,.2f}, that is a shift of "
    f"{100 * panama_first / panama_first_close:+.1f}%"
)
print("  every historical price moves by the same number of points, so dollar P&L is preserved")
es_continuous_panama.select(
    "timestamp", "close", "adj_close", "cumulative_adjustment", "instrument_id"
).head(10)

# %% [markdown]
# ### Ratio, or multiplicative back-adjustment
#
# Multiply historical prices by the ratio of new/old contract prices.
# This preserves percentage returns but distorts dollar amounts.
#
# Ratio = Close_new_contract / Close_old_contract
# Adjusted_price = Price * cumulative_ratio


# %%
def _compute_roll_ratios(individual_df: pl.DataFrame, front_months: pl.DataFrame) -> pl.DataFrame:
    """Compute price ratios (new/old) at each roll date."""
    roll_info = front_months.filter(pl.col("is_roll"))
    prices_lookup = individual_df.select(["timestamp", "instrument_id", "close"])

    old_prices = (
        roll_info.select(["timestamp", pl.col("prev_front").alias("instrument_id")])
        .join(prices_lookup, on=["timestamp", "instrument_id"], how="left")
        .rename({"close": "old_close"})
    )

    new_prices = (
        roll_info.select(["timestamp", pl.col("front_instrument_id").alias("instrument_id")])
        .join(prices_lookup, on=["timestamp", "instrument_id"], how="left")
        .rename({"close": "new_close"})
    )

    return (
        old_prices.select(["timestamp", "old_close"])
        .join(new_prices.select(["timestamp", "new_close"]), on="timestamp", how="inner")
        .filter(pl.col("old_close") != 0)
        .with_columns((pl.col("new_close") / pl.col("old_close")).alias("ratio"))
        .select(["timestamp", "ratio"])
        .drop_nulls()
    )


# %% [markdown]
# ### Ratio Adjustment
#
# Apply the computed ratios cumulatively backwards through the raw series.


# %%
def create_continuous_ratio(
    individual_df: pl.DataFrame, front_months: pl.DataFrame
) -> pl.DataFrame:
    """Create continuous series with ratio adjustment.

    Uses vectorized Polars joins instead of row-by-row iteration for O(n) complexity.
    """
    raw = create_continuous_raw(individual_df, front_months)
    roll_info = front_months.filter(pl.col("is_roll"))

    if len(roll_info) == 0:
        return raw.with_columns(pl.lit(1.0).alias("cumulative_ratio"))

    ratios_df = _compute_roll_ratios(individual_df, front_months)

    if len(ratios_df) == 0:
        return raw.with_columns(pl.lit(1.0).alias("cumulative_ratio"))

    # Adjustment applies to dates STRICTLY BEFORE each roll date
    raw_with_ratios = raw.join(ratios_df, on="timestamp", how="left").with_columns(
        pl.col("ratio").fill_null(1.0)
    )

    # Cumulative product in reverse, shift by 1 to exclude roll date
    raw_with_ratios = raw_with_ratios.with_columns(
        pl.col("ratio")
        .reverse()
        .cum_prod()
        .shift(1)
        .fill_null(1.0)
        .reverse()
        .alias("cumulative_ratio")
    )

    adjusted = raw_with_ratios.with_columns(
        [
            (pl.col("open") * pl.col("cumulative_ratio")).alias("adj_open"),
            (pl.col("high") * pl.col("cumulative_ratio")).alias("adj_high"),
            (pl.col("low") * pl.col("cumulative_ratio")).alias("adj_low"),
            (pl.col("close") * pl.col("cumulative_ratio")).alias("adj_close"),
        ]
    )

    return adjusted


# %%
es_continuous_ratio = create_continuous_ratio(es_individual, front_months)
ratio_first = es_continuous_ratio["cumulative_ratio"][0]
print(f"Ratio: the factor carried back to the start of the series is {ratio_first:.4f}")
print(f"  that is a shift of {100 * (ratio_first - 1):+.1f}% at the earliest close")
print("  every historical price moves by the same proportion, so percentage returns are preserved")

# %% [markdown]
# The two methods disagree about the earliest prices, and the size of the disagreement is the
# reason the choice matters. Panama moves the start of the history by a fixed number of index
# points; that was a large fraction of the index in 2016 and would be a small one today. Ratio
# moves it by a fixed proportion, which is the same fraction whenever it is applied. Neither is
# a correction of the other - they preserve different things, and the table at the end of the
# notebook says which to reach for.

# %%
es_continuous_ratio.select(
    "timestamp", "close", "adj_close", "cumulative_ratio", "instrument_id"
).head(10)

# %% [markdown]
# ## 4. Validation
#
# The test of a construction is whether it reproduces one built independently. Databento ships
# a pre-built continuous series for ES, also volume-rolled, so ours can be held against it.
#
# **The two frames do not have the same bar length.** The loading section measured it: the
# individual contracts arrive as daily bars stamped at midnight UTC, the vendor's continuous
# series as hourly bars. Joining them on `timestamp` matches our whole day against the vendor's
# midnight hour alone. The comparison would run, print a plausible-looking difference, and be
# measuring the wrong thing.
#
# So the vendor's hours are collapsed to a daily grid first. **The grid is the UTC calendar
# day**, which is what the individual daily bars already use, and it is worth being exact about
# that because it is not the CME session.
#
# A CME equity-index session opens at 17:00 Chicago time and closes the following afternoon, so
# a UTC calendar day holds the tail of one session and the start of the next. Chicago is six
# hours behind UTC in winter and five in summer, so that 17:00 open is 23:00 UTC under central
# standard time and 22:00 UTC under daylight saving. The last hourly bar inside a UTC day is
# the 23:00 bar, which is the opening hour of the next session in winter and its second hour in
# summer.
#
# In both halves of the year that bar sits inside the session starting that evening rather than
# the session that closed that afternoon, so the accurate name for it is a UTC day close. The
# cell below measures which hour it lands on rather than taking that on trust.
#
# Matching the vendor's daily convention is exactly what makes the comparison valid: both sides
# are then the same object, and any difference is about roll logic. It is also the convention
# `05_futures_session_aggregation` exists to replace, because a UTC calendar day splits a
# trading session in the middle and is the wrong grid for anything downstream.

# %%
es_databento = load_cme_futures(products=["ES"], tenors=[0], frequency="hourly", continuous=True)
print(f"Databento continuous: {es_databento.shape}")
describe_bars(es_databento, "vendor continuous ES")
es_databento.head()

# %%
_day_edges = (
    es_databento.with_columns(pl.col("timestamp").dt.date().alias("date"))
    .sort("timestamp")
    .group_by("date")
    .agg(pl.col("timestamp").last().alias("last_bar"))
)
_modal_last_hour = (
    _day_edges.select(pl.col("last_bar").dt.time().alias("hour"))
    .group_by("hour")
    .agg(pl.len().alias("days"))
    .sort("days", descending=True)
)
print("Last hourly bar inside a UTC calendar day, by hour of day:")
print(_modal_last_hour.head(4))
print(
    "The 23:00 bar belongs to the session starting that evening, not to the one that closed \n"
    "that afternoon, so it is a UTC day close rather than a session close."
)

# %%
databento_daily = (
    es_databento.with_columns(pl.col("timestamp").dt.date().alias("date"))
    .group_by("date")
    .agg(
        pl.col("close").sort_by("timestamp").first().alias("first_hour_close"),
        pl.col("close").sort_by("timestamp").last().alias("utc_day_close"),
        pl.len().alias("hours"),
    )
    .sort("date")
)

ours_daily = es_continuous_raw.select(
    pl.col("timestamp").dt.date().alias("date"),
    pl.col("close").alias("our_close"),
    "instrument_id",
)

comparison = ours_daily.join(databento_daily, on="date", how="inner").with_columns(
    (pl.col("our_close") - pl.col("utc_day_close")).alias("diff"),
    (pl.col("our_close") - pl.col("first_hour_close")).alias("diff_vs_first_hour"),
)

print(f"Days compared: {len(comparison):,}")
for label, col in [
    ("against the vendor's first hour", "diff_vs_first_hour"),
    ("against the vendor's UTC day close", "diff"),
]:
    series = comparison[col]
    print(
        f"  {label:36s} mean abs ${series.abs().mean():7.2f}   "
        f"median ${series.median():+6.2f}   max abs ${series.abs().max():7.2f}"
    )

# %% [markdown]
# The two rows differ by more than an order of magnitude, and only the second one is about the
# construction. Aligning the bars is not a presentational detail here: it is the difference
# between a validation that says our roll logic disagrees with the vendor's everywhere and one
# that says it agrees almost everywhere.
#
# ### Where the remaining disagreement lives
#
# With the bars aligned, the residual can be attributed. Our volume rule leaves the expiring
# contract as soon as the next one out-trades it; the vendor keeps its own timing. Between our
# roll date and the day the old contract stops trading, the two series can be quoting different
# contracts, and the calendar spread between those is what a reader would see.
#
# That interval is the prediction, and it is derived rather than chosen: it runs from each roll
# to the last trading day of the contract just left. Everywhere outside it, the two series
# should hold the same contract and agree. The window is not a tuned constant, so the check can
# fail.

# %%
_rolls = (
    es_continuous_raw.select(pl.col("timestamp").dt.date().alias("date"), "instrument_id")
    .sort("date")
    .with_columns(pl.col("instrument_id").shift(1).alias("left_behind"))
    .filter(pl.col("instrument_id") != pl.col("left_behind"))
    .join(
        contract_life.select("instrument_id", pl.col("last_trade").alias("old_last_trade")),
        left_on="left_behind",
        right_on="instrument_id",
        how="inner",
    )
)

_handover = set()
for _roll_date, _old_last in zip(
    _rolls["date"].to_list(), _rolls["old_last_trade"].to_list(), strict=True
):
    _day = _roll_date
    while _day <= _old_last:
        _handover.add(_day)
        _day += timedelta(days=1)

inside = comparison.filter(pl.col("date").is_in(list(_handover)))
outside = comparison.filter(~pl.col("date").is_in(list(_handover)))
_total_abs = comparison["diff"].abs().sum()
_span_days = [
    (o - r).days
    for r, o in zip(_rolls["date"].to_list(), _rolls["old_last_trade"].to_list(), strict=True)
]

print(f"Rolls in the constructed series: {_rolls.height}")
print(
    f"Handover spans roll to old contract's last trade: median "
    f"{sorted(_span_days)[len(_span_days) // 2]} calendar days, max {max(_span_days)}"
)
print(
    f"  inside the handover: {inside.height:5,} days, mean abs ${inside['diff'].abs().mean():7.2f}, "
    f"max ${inside['diff'].abs().max():7.2f}"
)
print(
    f"  outside it:          {outside.height:5,} days, mean abs ${outside['diff'].abs().mean():7.2f}, "
    f"max ${outside['diff'].abs().max():7.2f}"
)
print(
    f"  share of all absolute difference inside the handover: "
    f"{100 * inside['diff'].abs().sum() / _total_abs:.1f}%"
)

# %% [markdown]
# The prediction holds exactly. Outside the handover the two series are not merely close, they
# are identical - every day, to the cent - and the whole of the disagreement falls inside a
# window that was derived from the mechanism rather than fitted to the residual.
#
# That is a stronger result than the one the notebook set out to report, and it is only visible
# because the bars were aligned first. Against the unaligned comparison every day looked
# different, so no window could have separated the days that disagree from the days that do
# not: the signal was there, buried under an artifact an order of magnitude larger.
#
# What remains is a single explained effect. Our volume rule leaves the expiring contract as
# soon as the next one out-trades it; the vendor holds on for roughly another week, through the
# old contract's last trading day. In between, the two series quote different contracts, and
# what separates them is the calendar spread between two expiries - small most of the time, and
# large when the term structure moves, which is why the biggest gaps sit on the March and June
# 2020 expirations.

# %%
comparison.with_columns(pl.col("diff").abs().alias("abs_diff")).sort(
    "abs_diff", descending=True
).head(10).select("date", "instrument_id", "our_close", "utc_day_close", "diff")

# %% [markdown]
# ### The size of the handover gap is not constant
#
# What separates the two series during a handover is the spread between two expiries, and that
# spread is carry: roughly the index level times the difference between the financing rate and
# the dividend yield, over the time between the two deliveries. None of those three terms is
# fixed, so the gap should be small when the financing rate sits near the dividend yield and
# wider when it does not.
#
# Expressing it in basis points of the index removes the effect of the index level itself, so
# what is left is the part that is about rates.

# %%
gap_by_year = (
    comparison.filter(pl.col("diff") != 0)
    .with_columns(pl.col("date").dt.year().alias("year"))
    .group_by("year")
    .agg(
        pl.len().alias("days_apart"),
        pl.col("diff").abs().mean().alias("mean_gap_points"),
        pl.col("diff").abs().max().alias("max_gap_points"),
        pl.col("utc_day_close").mean().alias("index_level"),
    )
    .with_columns(
        (10_000 * pl.col("mean_gap_points") / pl.col("index_level")).alias("mean_gap_bps")
    )
    .sort("year")
    .select("year", "days_apart", "mean_gap_points", "max_gap_points", "mean_gap_bps")
)
print("Handover gap by year (days on which the two series differ at all):")
gap_by_year

# %% [markdown]
# The gap widens by roughly an order of magnitude between the first four years of the sample
# and the last three, and it widens in basis points, not just in points - so it is not the
# index having got bigger. The zero-rate years leave almost nothing between one expiry and the
# next; once financing costs exceed the dividend yield by several percent, a quarter of carry
# is worth tens of index points and a week of disagreement about which contract to hold becomes
# visible.
#
# The count of days apart moves too, for the same reason. The handover window has the same
# length throughout, but in the early years the spread is small enough that the two contracts
# frequently print the same close and the difference is exactly zero.
#
# For a reader this is the practical case for the adjustment methods in the previous section
# rather than a curiosity: the roll convention mattered least in precisely the period whose data
# is most often used to prototype, and it matters most now.

# %% [markdown]
# ### Seeing the difference

# %%
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.62, 0.38],
    vertical_spacing=0.08,
    subplot_titles=("ES continuous close", "Difference, ours minus vendor"),
)

comp_pd = comparison.sort("date").to_pandas()

fig.add_trace(
    go.Scatter(
        x=comp_pd["date"],
        y=comp_pd["our_close"],
        name="Our construction",
        line=dict(color=COLORS["blue"], width=1.2),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=comp_pd["date"],
        y=comp_pd["utc_day_close"],
        name="Databento",
        line=dict(color=COLORS["amber"], width=1.2, dash="dot"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=comp_pd["date"],
        y=comp_pd["diff"],
        name="Difference",
        line=dict(color=COLORS["slate"], width=0.9),
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.update_yaxes(title_text="ES index points", row=1, col=1)
fig.update_yaxes(title_text="Index points", row=2, col=1)
fig.update_layout(
    height=620,
    title="ES continuous: two volume-rolled constructions and their difference",
    legend=dict(orientation="h", yanchor="bottom", y=-0.14, x=0),
)
show_plotly_with_alt(
    fig,
    "Two panels sharing a date axis from 2016 to 2025. The upper panel draws our constructed "
    "ES close and Databento's over each other; at this scale they read as a single rising "
    "line, from roughly two thousand index points to close to seven thousand. The lower panel "
    "draws the difference between them, which lies exactly on zero for long stretches and "
    "breaks into narrow isolated spikes in both directions. The spikes are barely visible "
    "before 2020, appear as a cluster of downward spikes in 2020 reaching below minus one "
    "hundred and fifty, and from 2022 onward become a regular quarterly comb of upward spikes "
    "that grows steadily taller, the largest reaching about one hundred and fifty index "
    "points.",
)

# %% [markdown]
# ## 5. Construct + Validate Helper
#
# The construction logic is generic across products, so it is worth wrapping. What the wrapper
# must carry with it is the daily alignment the validation section established: a helper that repeated the raw
# timestamp join would reintroduce the same error on every product it was pointed at.
#
# It is applied to ES alone because ES is the only product for which individual contract data
# is on disk. The Databento subscription bundled with the book delivers the other 29 products
# exclusively as pre-built continuous series, so there is nothing to reconstruct them from.


# %%
def construct_and_validate(product: str, min_outright_price: float = MIN_OUTRIGHT_PRICE) -> dict:
    """Build a raw continuous series for one product and score it against the vendor's."""
    individual = load_cme_futures(products=[product], frequency="hourly", continuous=False)
    fronts = identify_front_month(individual, min_outright_price=min_outright_price)
    continuous_raw = create_continuous_raw(individual, fronts)

    vendor = load_cme_futures(products=[product], tenors=[0], frequency="hourly", continuous=True)
    vendor_daily = (
        vendor.with_columns(pl.col("timestamp").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.col("close").sort_by("timestamp").last().alias("vendor_close"))
    )
    ours_daily = continuous_raw.select(
        pl.col("timestamp").dt.date().alias("date"), pl.col("close").alias("our_close")
    )

    paired = ours_daily.join(vendor_daily, on="date", how="inner")
    diff = (paired["our_close"] - paired["vendor_close"]).abs()
    return {
        "product": product,
        "rows": len(continuous_raw),
        "contracts_used": continuous_raw["instrument_id"].n_unique(),
        "validation_days": len(paired),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff_bps": float((diff / paired["vendor_close"]).mean() * 10_000),
    }


# %%
validation_summary = pl.DataFrame([construct_and_validate("ES", MIN_OUTRIGHT_PRICE)])
print("Construction-vs-vendor validation (ES):")
validation_summary

# %% [markdown]
# ## 6. Production Pipeline
#
# The teaching examples above demonstrate roll detection and adjustment methods on a single product.
# For production use, the pipeline is:
#
# 1. **Download**: Databento provides pre-rolled continuous contracts (hourly OHLCV) for
#    front, second, and third month tenors -> `data/futures/market/continuous/hourly/`
# 2. **Session aggregation**: [`05_futures_session_aggregation`](05_futures_session_aggregation.ipynb) assigns CME session dates
#    and aggregates hourly bars into daily OHLCV -> `data/futures/market/continuous/daily/continuous_daily.parquet`
# 3. **Loading**: `load_cme_futures()` reads the daily parquet for downstream analysis

# %% [markdown]
# ---
#
# ## Key Takeaways
#
# 1. **A symbol is not a date.** The contract-definitions file writes the year with one digit,
#    so `ESM1` is the June contract of a year ending in 1 and nothing in the symbol says which
#    decade. A parser that pads it to a two-digit year dates every contract in this file to the
#    wrong one, and prints the wrong year on the same row as the right expiration. The delivery
#    month is recoverable from the symbol; the year has to come from the expiration column, or
#    from the price history, as this notebook does.
#
# 2. **An argument named for a bar length does not guarantee one.** `frequency="hourly"`
#    selects the raw per-contract capture; for individual contracts that capture is daily. The
#    frame declares its own bar length in `rtype`, and the spacing between timestamps confirms
#    it. The notebook asks rather than assumes, because the validation depends on the answer.
#
# 3. **Aligning the bars is the validation.** Joining our daily series to the vendor's hourly
#    one on a bare timestamp pairs a whole trading day against the vendor's midnight hour, and
#    the resulting difference is an order of magnitude larger than the one being studied. It is
#    also completely plausible on the page: a mean absolute difference of tens of points on an
#    index in the thousands reads like an ordinary construction disagreement. Aggregating the
#    vendor's hours to session closes first is what makes the comparison a comparison.
#
# 4. **Once aligned, the two constructions are identical except during the handover.** Outside
#    the interval between our roll and the old contract's last trading day, they agree to the
#    cent on every day of the sample; inside it, they quote different contracts and differ by
#    the calendar spread. All of the disagreement lives there. The window is derived from the
#    mechanism rather than fitted, so the check could have failed - and against the unaligned
#    comparison it would have, because there roll days were no worse than ordinary ones and
#    every day disagreed.
#
# 5. **The no-rollback constraint never fires on this history.** The notebook counts the days
#    where it changes the contract selected, and the count is zero: the raw volume leader is
#    already monotone across the sample. The guard is cheap and the flicker it prevents is real
#    on thinner products, so it stays - but it is not what makes the ES roll dates come out
#    clean, and the notebook no longer says it is.
#
# 6. **Calendar spreads contaminate raw individual data.** CME lists them alongside outright
#    contracts, and they trade at the inter-month difference rather than the index level. The
#    `MIN_OUTRIGHT_PRICE` floor is what stops a high-volume spread being selected as the front
#    month; the sample includes spread rows quoted in single-digit negative numbers.
#
# 7. **Panama preserves dollars, ratio preserves percentages, and on a decade of ES the two
#    disagree substantially about the earliest prices.** Both figures are printed by the adjustment section
#    rather than described here, because both move every time the history is extended.
#
# ### Adjustment Method Selection
# | Use Case | Recommended Method | Reason |
# |----------|-------------------|--------|
# | Backtesting P&L | Panama (additive) | Preserves dollar gains/losses across rolls |
# | Statistical analysis | Ratio | Preserves percentage returns accurately |
# | Live trading | Raw + position management | Handle rolls in execution layer |
#
# ### Next Steps
#
# - **Chapter 8**: Carry and momentum features built on continuous series.
# - **Chapter 16**: Backtesting with adjusted P&L.
