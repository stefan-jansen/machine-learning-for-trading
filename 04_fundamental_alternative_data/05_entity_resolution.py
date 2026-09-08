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
# # Entity Resolution: Matching Company Names to Identifiers
#
# **Chapter 4: Fundamental and Alternative Data**
# **Docker image**: `ml4t`
# **Section Reference**: Section 4.2 (Entity Resolution and Mapping)
#
# ## Purpose
#
# Every dataset that is not a price feed arrives keyed on a company name someone typed. A job
# postings vendor writes "Microsoft Corporation - Redmond"; a news feed writes "Microsoft
# Corp."; the SEC writes "MICROSOFT CORP". Joining any of them to a price series means deciding
# that these are one company, and deciding wrongly in either direction is expensive: a missed
# match drops the row and biases the sample toward companies with tidy names, while a wrong
# match attaches one company's data to another's returns and produces a signal that is pure
# noise wearing a name.
#
# This notebook builds the three stages that a production mapping uses, and scores each of them
# on the same labelled set so that "each stage handles what the previous one cannot" is a
# measurement rather than a slogan.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
#
# - Name the standard identifiers a security master links, and say which population each covers.
# - Join two sources on an identifier when one is present, and fall back in a stated order when
#   it is not.
# - Normalize a company name so that legal suffixes and punctuation stop dominating a string
#   comparison.
# - Score a name against a reference list with a fuzzy string metric, and choose the acceptance
#   threshold from a measured precision and recall rather than from a rule of thumb.
# - Score the same names with a sentence embedding, and show which failures it recovers and
#   which it does not.
# - Explain why a curated alias table is the only thing that resolves a rename or a subsidiary,
#   and build one.
#
# ## Cross-References
#
# - **Related**: [`02_sec_filing_explorer`](02_sec_filing_explorer.ipynb) (CIK lookup on live filings)
# - **Downstream**: `08_financial_features/04_fundamentals_macro_calendar.py` (joins fundamentals onto a price panel)
#
# ## Key Concepts
#
# - **Deterministic matching**: joining on an identifier that is equal or not, with no score.
# - **Probabilistic matching**: scoring the similarity of two names and accepting above a
#   threshold.
# - **Security master**: the table that holds one canonical record per entity, every identifier
#   that points at it, and every name it has been known by.
# - **Precision and recall**: of the matches made, the share that are right; and of the matches
#   that exist, the share that were made. Raising a threshold trades the second for the first.

# %%
"""Entity Resolution - match company names to identifiers using deterministic, fuzzy and embedding stages."""

import os
import warnings

# `multiprocess`, pulled in by the sentence-transformers stack, has a `return` inside a
# `finally` block that Python 3.14 warns about while compiling the module. It is one warning,
# raised once at import, about code this notebook does not call.
warnings.filterwarnings("ignore", category=SyntaxWarning, module="multiprocess")

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from IPython.display import display
from rapidfuzz import fuzz as rfuzz
from rapidfuzz import process as rprocess

# Importing utils.style registers and activates the ML4T Plotly template
# (palette, gridlines, fonts) repo-wide; the figures below inherit its colorway.
from utils.style import COLORS

# %% [markdown]
# The acceptance threshold is the one setting in this notebook that decides an outcome. Part 5
# measures what it costs at every value, so the number below is where that measurement lands
# rather than a default carried in from somewhere else. The sweep grid is the set of values
# that measurement is taken at.

# %% tags=["parameters"]
FUZZY_THRESHOLD = 70  # a fuzzy score at or above this is accepted as a match
THRESHOLD_GRID = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # values the sweep is taken at
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small enough to run on a CPU in a second

# %% [markdown]
# ## 1. Why names do not join
#
# The same company reaches five different tables under five different strings. None of the
# differences below is an error in the source: each one is what that source's own convention
# produces.

# %%
messy_company_names = pl.DataFrame(
    {
        "source": ["SEC filing", "News feed", "Alt data", "Price feed", "Research note"],
        "company_name": [
            "MICROSOFT CORPORATION",
            "Microsoft Corp.",
            "microsoft corp",
            "MSFT",
            "Microsoft (NASDAQ: MSFT)",
        ],
        "ticker_if_available": [None, None, None, "MSFT", "MSFT"],
    }
)
messy_company_names

# %% [markdown]
# Four kinds of difficulty sit behind those strings, and they are not equally hard. Casing,
# punctuation and legal suffixes are cosmetic and a normalizer removes them. A ticker used as a
# name shares no characters with the company's name. A rename means today's name and yesterday's
# name are both correct for the same entity. And a subsidiary's name is correct for a legal
# entity that is not the one whose stock trades.
#
# | Source A | Source B | Same traded entity? | Why it is hard |
# |----------|----------|---------------------|----------------|
# | "Apple Inc." | "APPLE INC" | Yes | Casing and punctuation |
# | "Microsoft Corporation" | "MSFT" | Yes | A ticker is not a name |
# | "Meta Platforms Inc." | "Facebook Inc." | Yes | The company was renamed |
# | "Alphabet Inc." | "Google LLC" | Yes | A subsidiary of the listed parent |
# | "Zoom Video Communications" | "Zoom Technologies Inc" | No | Two companies, similar names |
#
# The last row is the one that costs money. In 2020 the ticker ZOOM belonged to Zoom
# Technologies, a small unrelated company, while the video conferencing firm traded as ZM.
# Retail flow chasing the wrong ticker moved it by a multiple of its own value before trading
# was suspended. A name matcher that scores on spelling ranks those two names as near
# identical.

# %% [markdown]
# ## 2. The identifiers a security master links
#
# Where an identifier is present, matching stops being a similarity problem and becomes a join.
# The identifiers differ in what they name and in who is covered, which is why a master holds
# several rather than picking one.
#
# | Identifier | Names | Covers | Example |
# |------------|-------|--------|---------|
# | **CIK** | The filer, as an entity | Companies that file with the SEC | 0000789019 |
# | **LEI** | The legal entity | Global, any party to a financial transaction | INR2EJN1ERAN0W5ZP974 |
# | **FIGI** | The instrument on an exchange | Global securities, open licence | BBG000BPH459 |
# | **CUSIP** | The security | US and Canada, licensed | 594918104 |
# | **ISIN** | The security | Global, built from a national number | US5949181045 |
# | **Ticker** | The listing | One exchange at a time, and reused over time | MSFT |
#
# The ordering matters. A CIK and an LEI name a company; a CUSIP, an ISIN and a FIGI name a
# security, and one company can issue several. A ticker names neither durably: it is unique on
# an exchange on a date and is reassigned after a delisting, which is what the ZOOM case above
# turned on.

# %%
reference_securities = pl.DataFrame(
    {
        "company_name": [
            "Microsoft Corporation",
            "Apple Inc.",
            "Alphabet Inc.",
            "Amazon.com Inc.",
            "Meta Platforms Inc.",
            "NVIDIA Corporation",
            "Tesla Inc.",
            "Berkshire Hathaway Inc.",
            "JPMorgan Chase & Co.",
            "Johnson & Johnson",
        ],
        "ticker": ["MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK.B", "JPM", "JNJ"],
        "cik": [
            "0000789019",
            "0000320193",
            "0001652044",
            "0001018724",
            "0001326801",
            "0001045810",
            "0001318605",
            "0001067983",
            "0000019617",
            "0000200406",
        ],
        "exchange": ["NASDAQ"] * 7 + ["NYSE"] * 3,
    }
)
reference_securities

# %% [markdown]
# ## 3. Stage 1: join on an identifier
#
# The first stage tries each identifier in turn and stops at the first one that resolves. The
# order is a statement about which identifier is trusted more, and it is written into the call
# rather than left to a reader to infer: a CIK is assigned once and never reused, so it is tried
# before a ticker, which is.
#
# Two things have to be true of the reference side for this to be a join rather than a row
# multiplier. The key has to be unique in the reference, and a row that fails to match has to
# survive as an unmatched row rather than disappear. Both are asserted below.


# %%
def deterministic_match(
    source: pl.DataFrame, reference: pl.DataFrame, identifiers: list[str]
) -> pl.DataFrame:
    """Attach `ticker` and the identifier that found it, trying `identifiers` in order."""
    result = source.with_columns(
        matched_ticker=pl.lit(None, dtype=pl.Utf8),
        match_method=pl.lit(None, dtype=pl.Utf8),
    )
    for identifier in identifiers:
        if identifier not in source.columns or identifier not in reference.columns:
            continue
        lookup = reference.select(identifier, pl.col("ticker").alias("_found"))
        if lookup[identifier].n_unique() != len(lookup):
            raise ValueError(f"{identifier!r} is not unique in the reference table")

        before = len(result)
        result = result.join(lookup, on=identifier, how="left")
        assert len(result) == before, f"the join on {identifier!r} changed the row count"

        result = result.with_columns(
            # coalesce keeps whatever an earlier, more trusted identifier already found.
            matched_ticker=pl.coalesce("matched_ticker", "_found"),
            match_method=pl.coalesce(
                "match_method",
                pl.when(pl.col("_found").is_not_null()).then(pl.lit(identifier)),
            ),
        ).drop("_found")
    return result


# %% [markdown]
# Three filings, two of which carry a CIK the reference knows and one of which carries a CIK
# that does not exist. The third row is the case the assertion above protects: it must come back
# unmatched, not missing.

# %%
filings = pl.DataFrame(
    {
        "filing_id": [1, 2, 3],
        "company_name": ["MSFT INC", "APPLE COMPUTER", "UNKNOWN CORP"],
        "cik": ["0000789019", "0000320193", "9999999999"],
    }
)
deterministic_match(filings, reference_securities, identifiers=["cik", "ticker"])

# %% [markdown]
# ## 4. Stage 2: score the names
#
# Most alternative data carries no identifier at all, and then the name is what there is. Two
# steps make a name comparable: normalization, which removes the differences that carry no
# information, and a similarity score over what is left.
#
# ### Normalizing
#
# A legal suffix is not part of a company's identity for matching purposes: "Inc.", "Corp." and
# "plc" say how the entity is incorporated, and every entry in a reference list has one. Left in
# place they inflate every score toward every candidate. The normalizer upper-cases, strips
# suffixes until none remains, and removes punctuation.


# %%
LEGAL_SUFFIXES = (
    " INCORPORATED",
    " INC.",
    " INC",
    " CORPORATION",
    " CORP.",
    " CORP",
    " COMPANY",
    " CO.",
    " CO",
    " LLC",
    " LLP",
    " LP",
    " LIMITED",
    " LTD",
    " PLC",
    " SA",
    " AG",
    " NV",
    " SE",
    " GROUP",
    " HOLDINGS",
)


def normalize_company_name(name: str) -> str:
    """Upper-case a company name and strip legal suffixes and punctuation from it."""
    if not name:
        return ""
    normalized = name.upper().strip()
    # Repeat until nothing is stripped: "ACME HOLDINGS LTD" carries two suffixes, and a
    # single pass over the tuple removes only whichever it reaches last.
    stripped = True
    while stripped:
        stripped = False
        for suffix in LEGAL_SUFFIXES:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                stripped = True
    normalized = normalized.replace(",", "").replace(".", "").replace("&", "AND")
    return " ".join(normalized.split())


# %% [markdown]
# Running the normalizer over the reference list shows what each stage of matching is actually
# comparing, which is worth seeing before any score is interpreted.

# %%
reference_names = reference_securities["company_name"].to_list()
reference_norm = [normalize_company_name(name) for name in reference_names]
pl.DataFrame({"company_name": reference_names, "normalized": reference_norm})

# %% [markdown]
# ### Scoring
#
# `rapidfuzz` implements several string metrics, and they differ in what they forgive:
#
# | Scorer | Compares | Forgives |
# |--------|----------|----------|
# | `ratio` | The two strings as sequences of characters | Small typos |
# | `partial_ratio` | The shorter string against its closest-scoring window of the longer | One name being contained in the other |
# | `token_sort_ratio` | The words, alphabetically ordered | Word order |
# | `token_set_ratio` | The set of words, ignoring extras on either side | Word order and extra words |
#
# `token_set_ratio` is the default here because the differences that survive normalization are
# usually extra words: a location appended by the vendor, a share class, a division. It scores
# out of one hundred.


# %%
def fuzzy_match(
    query: str,
    candidates: list[str],
    candidates_norm: list[str],
    scorer=rfuzz.token_set_ratio,
) -> tuple[str | None, float]:
    """Return the best-scoring candidate for `query` and its score, without any cutoff."""
    result = rprocess.extractOne(normalize_company_name(query), candidates_norm, scorer=scorer)
    if result is None:
        return None, 0.0
    return candidates[result[2]], float(result[1])


# %% [markdown]
# ### The labelled set everything is scored on
#
# The three stages are compared on one set of names with the right answer written down beside
# each. Twelve rows, ten of which have a match in the reference list and two of which do not.
# Without the two that have no match, precision cannot be measured at all: every match would be
# either right or missing, and a matcher that says yes to everything would look perfect.
#
# The `job_postings` column stands in for whatever the alternative data actually carries. It is
# the reason the join is being attempted.

# %%
labelled = pl.DataFrame(
    {
        "raw_company_name": [
            "Microsoft Corporation - Redmond",
            "APPLE INC",
            "GOOGLE",
            "amazon.com",
            "Facebook, Inc.",
            "NVIDIA Corp",
            "Tesla Motors Inc",
            "Berkshire Hathaway",
            "JP Morgan Chase",
            "J&J",
            "Zoom Technologies Inc",
            "Palantir Technologies",
        ],
        "true_company_name": [
            "Microsoft Corporation",
            "Apple Inc.",
            "Alphabet Inc.",
            "Amazon.com Inc.",
            "Meta Platforms Inc.",
            "NVIDIA Corporation",
            "Tesla Inc.",
            "Berkshire Hathaway Inc.",
            "JPMorgan Chase & Co.",
            "Johnson & Johnson",
            None,
            None,
        ],
        "difficulty": [
            "extra words",
            "casing",
            "brand name of a subsidiary",
            "punctuation",
            "former name",
            "suffix",
            "former name",
            "suffix",
            "spacing",
            "abbreviation",
            "not in the reference list",
            "not in the reference list",
        ],
        "job_postings": [150, 200, 180, 300, 120, 80, 90, 50, 160, 70, 40, 110],
    }
)
labelled

# %%
fuzzy_scored = labelled.with_columns(
    pl.Series(
        "fuzzy_name",
        [fuzzy_match(q, reference_names, reference_norm)[0] for q in labelled["raw_company_name"]],
    ),
    pl.Series(
        "fuzzy_score",
        [fuzzy_match(q, reference_names, reference_norm)[1] for q in labelled["raw_company_name"]],
    ),
).with_columns(
    accepted=pl.col("fuzzy_score") >= FUZZY_THRESHOLD,
    correct=pl.col("fuzzy_name") == pl.col("true_company_name"),
)
fuzzy_scored.select(
    "raw_company_name", "difficulty", "true_company_name", "fuzzy_name", "fuzzy_score", "accepted"
)

# %% [markdown]
# ## 5. Choosing the threshold by measuring it
#
# The threshold decides which scores become matches, and it trades two errors against each
# other. **Precision** is the share of accepted matches that are right; lowering the threshold
# admits more wrong ones and precision falls. **Recall** is the share of the true matches that
# were made; raising the threshold rejects borderline right ones and recall falls. **F1** is
# their harmonic mean, which is one number to compare settings by when both errors matter.
#
# A name with no true match counts against precision whenever it is accepted, and never against
# recall. That is what the two unmatchable rows in the labelled set are for.


# %%
def score_at_threshold(scored: pl.DataFrame, name_col: str, score_col: str, threshold: float):
    """Precision, recall and F1 for accepting `score_col >= threshold` as a match."""
    accepted = scored.filter(pl.col(score_col) >= threshold)
    true_positives = int((accepted[name_col] == accepted["true_company_name"]).sum())
    false_positives = len(accepted) - true_positives
    resolvable = int(scored["true_company_name"].is_not_null().sum())

    precision = true_positives / (true_positives + false_positives) if len(accepted) else 1.0
    recall = true_positives / resolvable
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "accepted": len(accepted),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# %%
sweep = pl.DataFrame(
    [
        score_at_threshold(fuzzy_scored, "fuzzy_name", "fuzzy_score", threshold)
        for threshold in THRESHOLD_GRID
    ]
)
sweep

# %%
fig = go.Figure()
for metric, color in [
    ("precision", COLORS["blue"]),
    ("recall", COLORS["copper"]),
    ("f1", COLORS["amber"]),
]:
    fig.add_trace(
        go.Scatter(
            x=sweep["threshold"],
            y=sweep[metric],
            mode="lines+markers",
            name=metric,
            line=dict(color=color),
        )
    )
fig.add_vline(
    x=FUZZY_THRESHOLD,
    line_dash="dash",
    line_color=COLORS["neutral"],
    annotation_text="threshold in use",
)
fig.update_layout(
    title="Lowering the cutoff buys recall only until wrong matches arrive",
    xaxis_title="Acceptance threshold (fuzzy score)",
    yaxis_title="Share",
    yaxis_range=[0, 1.05],
    height=420,
)
fig.show()

# %% [markdown]
# Two features of that curve are worth naming, because they are what a threshold can and cannot
# do. Over most of its range precision is flat and recall is a staircase: each step is one name
# whose right answer scores just below where the cutoff was. Those are the names a lower
# threshold recovers. Below the point where precision breaks, the accepted set starts filling
# with names whose nearest candidate is merely the closest of ten, and no cutoff separates them
# from a genuine weak match.
#
# The names that never appear in the accepted set at any threshold are the interesting ones,
# because they are not a threshold problem at all.

# %%
fuzzy_scored.filter(~pl.col("correct") & pl.col("true_company_name").is_not_null()).select(
    "raw_company_name", "difficulty", "true_company_name", "fuzzy_name", "fuzzy_score"
)

# %% [markdown]
# ## 6. Stage 3: score the meaning instead of the spelling
#
# A fuzzy scorer compares surface forms. It has no way to know that a ticker stands for a
# company, that a brand is a division of a listed parent, or that a company changed its name,
# because none of those facts is visible in the characters.
#
# A sentence embedding maps a string to a vector positioned by what the model saw the words used
# for during training, so two strings that name the same thing land near each other even when
# they share no characters. Cosine similarity between the vectors is the score. The model used
# here is small enough to run on a CPU in about a second on this data; it loads from the local
# Hugging Face cache, and `HF_HUB_OFFLINE` keeps it from reaching the network. Where it is not
# cached, this stage is skipped and the two stages above stand on their own.

# %%
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

embedding_model = None
try:
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(EMBED_MODEL)
    print(f"Loaded {EMBED_MODEL}")
except Exception as exc:  # noqa: BLE001 - offline and uncached, or the optional dep is absent
    print(f"Stage 3 skipped: {EMBED_MODEL} is unavailable ({type(exc).__name__}).")

# %% [markdown]
# The names are normalized before they are embedded, for the same reason they were before being
# scored: the suffixes are shared by every reference name and add nothing to distinguish them.
# The embeddings are unit length, so their dot product is the cosine of the angle between them.

# %%
if embedding_model is not None:
    reference_embeddings = embedding_model.encode(reference_norm, normalize_embeddings=True)
    query_embeddings = embedding_model.encode(
        [normalize_company_name(q) for q in labelled["raw_company_name"]],
        normalize_embeddings=True,
    )
    similarity = query_embeddings @ reference_embeddings.T
    nearest = similarity.argmax(axis=1)

    embedding_scored = fuzzy_scored.with_columns(
        embedding_name=pl.Series([reference_names[i] for i in nearest]),
        embedding_score=pl.Series(similarity[np.arange(len(nearest)), nearest].astype(float)),
    ).with_columns(
        embedding_correct=pl.col("embedding_name") == pl.col("true_company_name"),
    )
    display_cols = [
        "raw_company_name",
        "difficulty",
        "true_company_name",
        "fuzzy_name",
        "embedding_name",
        "embedding_score",
    ]

# %%
if embedding_model is not None:
    display(embedding_scored.select(display_cols))

# %% [markdown]
# ### Where the embedding helps, and why its score cannot be thresholded
#
# The comparison above splits into three groups. On the names fuzzy already matched, the
# embedding agrees, which is the least interesting result and the one that has to hold before
# anything else is worth reading. On abbreviations and paraphrases it recovers matches the
# fuzzy scorer had no route to, because the relationship is in the meaning rather than the
# spelling. And on renames and subsidiaries it produces answers of both kinds.
#
# That last group is where the method has to stop being trusted, and the reason is visible in
# the scores rather than in an argument about them. The figure plots every query's cosine
# similarity against whether the match it produced was right.

# %%
if embedding_model is not None:
    outcome = (
        pl.when(pl.col("true_company_name").is_null())
        .then(pl.lit("No true match exists"))
        .when(pl.col("embedding_correct"))
        .then(pl.lit("Matched correctly"))
        .otherwise(pl.lit("Matched the wrong company"))
    )
    fig = px.strip(
        embedding_scored.with_columns(outcome=outcome).to_pandas(),
        x="embedding_score",
        y="outcome",
        hover_name="raw_company_name",
        color="outcome",
        color_discrete_map={
            "Matched correctly": COLORS["blue"],
            "Matched the wrong company": COLORS["negative"],
            "No true match exists": COLORS["neutral"],
        },
        stripmode="overlay",
    )
    fig.update_traces(marker=dict(size=11, opacity=0.85))
    fig.update_layout(
        title="Right and wrong embedding matches occupy the same range of scores",
        xaxis_title="Cosine similarity to the nearest reference name",
        yaxis_title="",
        showlegend=False,
        height=340,
        margin=dict(l=200),
    )
    fig.show()

# %% [markdown]
# The correct matches and the wrong ones overlap on the horizontal axis. Any cutoff that admits
# the correct low-scoring matches also admits at least one wrong one, and any cutoff that
# excludes the wrong ones also excludes correct matches the stage exists to recover. A wrong
# match is not a missing row: it attaches one company's alternative data to another company's
# returns, and every statistic computed afterwards is contaminated in a way that nothing
# downstream will flag.
#
# So the embedding extends the probabilistic stage and does not close it. What resolves a rename
# or a subsidiary is a fact about corporate history, which no property of the strings can supply
# and no model can infer. It has to be written down.

# %% [markdown]
# ## 7. The security master, and the alias table
#
# The table that holds those written-down facts is the security master. It carries one canonical
# record per entity, the identifiers that point at it, and every name the entity has been known
# by, each with the dates over which that name was current. Matching then becomes: try the
# identifiers, then try the alias list, and only then fall back to scoring.
#
# The dates matter as much as the names. "Facebook Inc." was the right name until October 2021
# and the wrong one after, so an alias without a validity window will resolve a 2023 document
# to a name that no longer existed, and will silently accept a document that predates a rename
# as if it postdated it.


# %%
class SecurityMaster:
    """One canonical record per entity, plus every name it has been filed under."""

    def __init__(self) -> None:
        self._entities: list[dict] = []
        self._aliases: list[dict] = []

    def add(
        self,
        canonical_name: str,
        ticker: str,
        cik: str,
        aliases: list[str] | None = None,
        parent_ticker: str | None = None,
    ) -> None:
        """Register an entity and the names it is also known by."""
        self._entities.append(
            {
                "canonical_name": canonical_name,
                "ticker": ticker,
                "cik": cik,
                "parent_ticker": parent_ticker,
            }
        )
        for alias in [canonical_name, *(aliases or [])]:
            self._aliases.append({"ticker": ticker, "alias": alias})

    @property
    def entities(self) -> pl.DataFrame:
        return pl.DataFrame(self._entities)

    @property
    def aliases(self) -> pl.DataFrame:
        return pl.DataFrame(self._aliases)

    def resolve(self, query: str, cik: str | None = None) -> dict:
        """Resolve a name or CIK to a ticker, reporting which stage answered."""
        if cik is not None:
            hit = self.entities.filter(pl.col("cik") == cik)
            if len(hit):
                return {"ticker": hit["ticker"][0], "stage": "identifier", "score": 100.0}

        normalized = normalize_company_name(query)
        alias_hit = self.aliases.filter(
            pl.col("alias").map_elements(normalize_company_name, return_dtype=pl.Utf8) == normalized
        )
        if len(alias_hit):
            return {"ticker": alias_hit["ticker"][0], "stage": "alias", "score": 100.0}

        alias_names = self.aliases["alias"].to_list()
        alias_norm = [normalize_company_name(a) for a in alias_names]
        match, score = fuzzy_match(query, alias_names, alias_norm)
        if match is not None and score >= FUZZY_THRESHOLD:
            ticker = self.aliases.filter(pl.col("alias") == match)["ticker"][0]
            return {"ticker": ticker, "stage": "fuzzy", "score": score}

        return {"ticker": None, "stage": "unresolved", "score": 0.0}


# %% [markdown]
# The four entities below carry exactly the aliases that the two previous stages could not
# reach: the former names, the subsidiary brands, and the tickers. Each one is a decision
# somebody made and recorded, which is what makes it auditable and what makes it correct.

# %%
master = SecurityMaster()
master.add("Microsoft Corporation", "MSFT", "0000789019", aliases=["Microsoft Corp", "MSFT"])
master.add("Apple Inc.", "AAPL", "0000320193", aliases=["Apple Computer", "AAPL"])
master.add(
    "Meta Platforms Inc.", "META", "0001326801", aliases=["Facebook Inc", "Facebook", "META"]
)
master.add(
    "Alphabet Inc.",
    "GOOGL",
    "0001652044",
    aliases=["Google Inc", "Google LLC", "Google", "GOOGL", "GOOG"],
)

master.entities

# %%
queries = [
    ("Microsoft Corporation", "0000789019"),
    ("Facebook", None),
    ("Google LLC", None),
    ("MSFT", None),
    ("Palantir Technologies", None),
]
pl.DataFrame(
    [{"query": query, "cik": cik, **master.resolve(query, cik=cik)} for query, cik in queries]
)

# %% [markdown]
# The two names that no amount of string or embedding similarity resolved - a former name and a
# subsidiary brand - are answered by the alias stage at full confidence, and the name that no
# entity in the master accounts for comes back unresolved rather than attached to its nearest
# candidate. That is the whole argument for the alias table: it turns two of the three failure
# modes from a scoring problem into a lookup, and it makes the third one visible.

# %% [markdown]
# ## Key Takeaways
#
# 1. Match on an identifier wherever one exists, and choose the order deliberately. A CIK or an
#    LEI names an entity permanently; a ticker names a listing at a point in time and is
#    reassigned, so a pipeline that joins on tickers will eventually join two companies together.
# 2. Normalize before scoring. Legal suffixes and punctuation are shared by every name in a
#    reference list, so leaving them in raises every score toward every candidate and compresses
#    the difference the score is supposed to measure.
# 3. Pick the acceptance threshold from a measurement on labelled examples, including examples
#    with no correct answer. Without those, precision is unmeasurable and a matcher that accepts
#    everything scores perfectly.
# 4. A fuzzy score compares spelling and an embedding compares meaning, so the embedding recovers
#    abbreviations and paraphrases that the fuzzy scorer cannot reach. It does not recover
#    renames or subsidiaries, and its scores on those cases sit among the scores of its outright
#    errors, so no threshold separates them.
# 5. A rename and a parent-subsidiary link are facts about corporate history rather than
#    properties of a string. They are recorded in an alias table with validity dates, or they are
#    wrong. Every stage above exists to reduce how much has to be recorded, not to replace the
#    record.
# 6. A wrong match is worse than no match. An unmatched row is a visible gap; a wrongly matched
#    row joins one company's data to another's returns and every statistic computed from it is
#    contaminated with nothing to indicate it.
