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
# **Section Reference**: See Section 4.2 for entity resolution concepts
#
# ## Purpose
#
# Entity resolution is the keystone problem in multi-source data integration. Before any data
# can be combined, we must correctly link disparate real-world names like "IBM Corp" and
# "International Business Machines" to the same unique security identifier. This notebook
# demonstrates hierarchical matching approaches from deterministic to ML-based.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
# - Understand the entity resolution problem and why it's critical
# - Build a hierarchical matching approach: deterministic → probabilistic → ML-based
# - Work with standard identifiers (LEI, CIK, FIGI, CUSIP, ISIN)
# - Apply fuzzy string matching for inconsistent names
# - Evaluate matching quality and handle edge cases
#
# ## Cross-References
#
# - **Upstream**: Alternative data vendors, government contracts, web scraped data
# - **Downstream**: Chapter 8 `fundamental_factors.py`, any multi-source analysis
# - **Related**: `02_sec_filing_explorer.py` (CIK mapping)
#
# ## Key Concepts
#
# - **Deterministic Matching**: Exact matches on strong identifiers (LEI, CIK, FIGI)
# - **Probabilistic Matching**: Fuzzy string algorithms (Levenshtein, Jaro-Winkler)
# - **ML-Based Matching**: Embeddings recover paraphrases and abbreviations; renames and subsidiaries still need a curated alias table
# - **Master Security Database**: Central repository linking all identifiers

# %%
"""Entity Resolution — match company names to identifiers using hierarchical deterministic and fuzzy matching."""

import os
import warnings

warnings.filterwarnings("ignore")


import numpy as np
import plotly.express as px
import polars as pl
from rapidfuzz import fuzz as rfuzz
from rapidfuzz import process as rprocess

# Importing utils.style registers and activates the ML4T Plotly template
# (palette, gridlines, fonts) repo-wide; the px figure below inherits its colorway.
from utils.style import COLORS

# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI

# %% [markdown]
# ## 1. The Entity Resolution Problem
#
# ### Why This Matters
#
# Consider these real-world scenarios that break naive matching:
#
# | Source A | Source B | Same Company? |
# |----------|----------|---------------|
# | "Apple Inc." | "APPLE INC" | Yes |
# | "Microsoft Corporation" | "MSFT" | Yes |
# | "ZOOM Video Communications" | "Zoom Technologies Inc" | **NO!** (ZOOM vs ZM) |
# | "Alphabet Inc." | "Google LLC" | Yes (subsidiary) |
# | "Meta Platforms Inc." | "Facebook Inc." | Yes (renamed) |
#
# Getting this wrong can be catastrophic for your strategy.

# %%
# Create sample messy data to demonstrate the problem
messy_company_names = pl.DataFrame(
    {
        "source": ["SEC Filing", "News Feed", "Alt Data", "Price Feed", "Research"],
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
# ## 2. Standard Identifiers: The First Line of Defense
#
# When available, standard identifiers provide deterministic matching:
#
# | Identifier | Description | Coverage | Example |
# |------------|-------------|----------|---------|
# | **CIK** | SEC Central Index Key | US SEC filers | 0000789019 (MSFT) |
# | **LEI** | Legal Entity Identifier | Global, 2.5M+ entities | INR2EJN1ERAN0W5ZP974 (MSFT) |
# | **FIGI** | Financial Instrument Global Identifier | Global securities | BBG000BPH459 (MSFT) |
# | **CUSIP** | US/Canada securities | US/Canada | 594918104 (MSFT) |
# | **ISIN** | International Securities ID | Global | US5949181045 (MSFT) |
# | **Ticker** | Exchange symbol | Exchange-specific | MSFT |

# %%
# Create a reference database with standard identifiers
COMPANY_NAMES = [
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
]
TICKERS = ["MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK.B", "JPM", "JNJ"]
CIKS = [
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
]
EXCHANGES = ["NASDAQ"] * 7 + ["NYSE"] * 3

# %%
reference_securities = pl.DataFrame(
    {
        "company_name": COMPANY_NAMES,
        "ticker": TICKERS,
        "cik": CIKS,
        "exchange": EXCHANGES,
    }
)
reference_securities

# %% [markdown]
# ## 3. Stage 1: Deterministic Matching
#
# The first stage uses exact matches on strong identifiers.


# %%
def deterministic_match(
    source_df: pl.DataFrame, reference_df: pl.DataFrame, match_columns: list[str]
) -> pl.DataFrame:
    """Stage 1: Deterministic matching on exact identifier values."""
    result = source_df.clone()
    result = result.with_columns(pl.lit(None).alias("matched_ticker"))
    result = result.with_columns(pl.lit(None).alias("match_method"))

    for col in match_columns:
        if col not in source_df.columns or col not in reference_df.columns:
            continue

        # Find unmatched rows
        unmatched_mask = result["matched_ticker"].is_null()

        if unmatched_mask.sum() == 0:
            break

        # Try to match on this column
        # Dedupe reference to avoid row expansion on duplicate keys
        ref_dedup = reference_df.select([col, "ticker"]).unique(subset=[col], keep="first")
        matches = source_df.join(ref_dedup, on=col, how="left")

        # Update matched rows
        result = result.with_columns(
            [
                pl.when(unmatched_mask & matches["ticker"].is_not_null())
                .then(matches["ticker"])
                .otherwise(result["matched_ticker"])
                .alias("matched_ticker"),
                pl.when(unmatched_mask & matches["ticker"].is_not_null())
                .then(pl.lit(f"deterministic:{col}"))
                .otherwise(result["match_method"])
                .alias("match_method"),
            ]
        )

    return result


# Example: matching on CIK
source_with_cik = pl.DataFrame(
    {
        "filing_id": [1, 2, 3],
        "company_name": ["MSFT INC", "APPLE COMPUTER", "UNKNOWN CORP"],
        "cik": ["0000789019", "0000320193", "9999999999"],
    }
)

matched = deterministic_match(
    source_with_cik, reference_securities, match_columns=["cik", "ticker"]
)
matched

# %% [markdown]
# ## 4. Stage 2: Probabilistic Matching with Fuzzy Strings
#
# When identifiers aren't available, we use fuzzy string matching algorithms:
#
# - **Levenshtein Distance**: Edit distance (insertions, deletions, substitutions)
# - **Jaro-Winkler**: Emphasizes prefix matches (good for company names)
# - **Token Set Ratio**: Handles word reordering ("Apple Inc" vs "Inc Apple")
# - **Partial Ratio**: Handles substrings ("Microsoft" in "Microsoft Corporation")


# %%
def normalize_company_name(name: str) -> str:
    """
    Normalize company name for matching.

    Removes common suffixes, punctuation, and standardizes case.
    """
    if not name:
        return ""

    name = name.upper().strip()

    # Remove common suffixes
    suffixes = [
        " INC.",
        " INC",
        " INCORPORATED",
        " CORP.",
        " CORP",
        " CORPORATION",
        " LLC",
        " LLP",
        " LP",
        " LTD",
        " LIMITED",
        " CO.",
        " CO",
        " COMPANY",
        " PLC",
        " SA",
        " AG",
        " NV",
        " SE",
        " GROUP",
        " HOLDINGS",
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    # Remove punctuation
    name = name.replace(",", "").replace(".", "").replace("&", "AND")

    # Remove extra whitespace
    name = " ".join(name.split())

    return name


# %%
def prepare_candidates(candidates: list[str]) -> tuple[list[str], list[str]]:
    """
    Pre-normalize candidate names for fuzzy matching.

    Call this ONCE before batch matching to avoid O(N×M) normalization cost.

    Returns
    -------
    tuple[list[str], list[str]]
        (normalized_candidates, original_candidates)
    """
    return [normalize_company_name(c) for c in candidates], candidates


# %% [markdown]
# ### Fuzzy Match Company
# Find the best fuzzy match for a company name against a list of candidates.


# %%
def fuzzy_match_company(
    query: str,
    candidates: list[str],
    threshold: int = 80,
    method: str = "token_set_ratio",
    candidates_norm: list[str] | None = None,
) -> tuple[str | None, int]:
    """Find best fuzzy match for a company name. Returns (match, score) or (None, 0)."""
    query_norm = normalize_company_name(query)

    # Use pre-normalized candidates if provided, otherwise normalize (slower for batch)
    if candidates_norm is None:
        candidates_norm = [normalize_company_name(c) for c in candidates]

    # Build index lookup for O(1) access (first occurrence wins for duplicates)
    norm_to_first_idx: dict[str, int] = {}
    for i, n in enumerate(candidates_norm):
        if n not in norm_to_first_idx:
            norm_to_first_idx[n] = i

    scorer = {
        "ratio": rfuzz.ratio,
        "partial_ratio": rfuzz.partial_ratio,
        "token_sort_ratio": rfuzz.token_sort_ratio,
        "token_set_ratio": rfuzz.token_set_ratio,
    }.get(method, rfuzz.token_set_ratio)

    result = rprocess.extractOne(query_norm, candidates_norm, scorer=scorer, score_cutoff=threshold)

    if result:
        idx = norm_to_first_idx[result[0]]
        return candidates[idx], int(result[1])

    return None, 0


# Test fuzzy matching
test_names = [
    "MICROSOFT CORP",
    "Microsoft Corporation Inc",
    "msft",
    "Apple Computer",
    "ALPHABET INC CLASS A",
    "Google LLC",  # Subsidiary - harder to match
    "Amazon Web Services",  # Subsidiary
    "Facebook Inc",  # Old name
    "NVIDIA CORP",
    "Tesla Motors",  # Old name
]

reference_names = reference_securities["company_name"].to_list()

print("Fuzzy Matching Results:")
for name in test_names:
    match, score = fuzzy_match_company(name, reference_names, threshold=70)
    status = "[OK]  " if match else "[FAIL]"
    print(f"{status} {name!r:42}  -> {str(match)!r:28}  (score: {score})")

# %% [markdown]
# ## 5. Stage 3: ML-Based Matching with Embeddings
#
# Fuzzy scoring compares *surface forms*: it rewards shared tokens and characters.
# That leaves two gaps. A ticker used as a name ("msft") shares no whole token with
# "Microsoft Corporation", and a paraphrase like "Amazon Web Services" shares no
# token with "Amazon.com" once suffixes are stripped — both fall through fuzzy
# matching entirely. A sentence-embedding model maps each name to a vector whose
# neighbors are *semantically* related, so cosine similarity can recover matches
# that share meaning rather than spelling.
#
# We embed with `all-MiniLM-L6-v2`, a small model that runs locally in well under
# a second on this data, and match each query to its nearest reference name. The
# model loads from the local Hugging Face cache; `HF_HUB_OFFLINE=1` keeps it off
# the network. Where the model is not cached — a minimal CI image, for instance —
# the cell skips Stage 3 and the deterministic and fuzzy stages stand on their own.


# %%
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = None
try:
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(EMBED_MODEL)
except Exception as exc:  # offline + uncached, or optional dependency missing
    print(
        f"[skip] embedding model '{EMBED_MODEL}' unavailable ({type(exc).__name__}); "
        "showing deterministic + fuzzy stages only. Pre-cache the model to run Stage 3."
    )


def embedding_match(
    query: str,
    candidates: list[str],
    model,
    candidate_embeddings: np.ndarray | None = None,
    candidates_norm: list[str] | None = None,
) -> tuple[str | None, float]:
    """Stage 3: match a name to its nearest reference by embedding cosine similarity.

    Names are normalized first so suffix noise ("Inc.", "Corp.") does not dominate
    the vector. Embeddings are L2-normalized, so their dot product is the cosine.
    """
    if model is None:
        return None, 0.0

    if candidates_norm is None:
        candidates_norm = [normalize_company_name(c) for c in candidates]
    if candidate_embeddings is None:
        candidate_embeddings = model.encode(candidates_norm, normalize_embeddings=True)

    query_emb = model.encode([normalize_company_name(query)], normalize_embeddings=True)[0]
    sims = candidate_embeddings @ query_emb
    best = int(np.argmax(sims))
    return candidates[best], float(sims[best])


# %%
# Compare fuzzy and embedding matches on the same queries, side by side.
if embedding_model is not None:
    ref_norm = [normalize_company_name(c) for c in reference_names]
    ref_embeddings = embedding_model.encode(ref_norm, normalize_embeddings=True)

    print(f"{'Query':<27}{'Fuzzy':<22}{'Embedding':<22}{'cos':>5}")
    print("-" * 76)
    for name in test_names:
        fz_match, _ = fuzzy_match_company(name, reference_names, threshold=70)
        em_match, em_score = embedding_match(
            name,
            reference_names,
            embedding_model,
            candidate_embeddings=ref_embeddings,
            candidates_norm=ref_norm,
        )
        print(f"{name:<27}{str(fz_match):<22}{str(em_match):<22}{em_score:>5.2f}")

# %% [markdown]
# ### What embeddings add — and where they mislead
#
# Two queries that fuzzy matching drops entirely come back with the embedding:
#
# - **`Amazon Web Services` → Amazon.com** (cos ≈ 0.65): no shared token survives
#   normalization, so fuzzy returns nothing, but the names are semantically close.
# - **`msft` → Microsoft** (cos ≈ 0.53): a ticker is a compressed label the fuzzy
#   scorer rejects; the embedding still places it nearest the right entity.
#
# The other recovered names — `Apple Computer`, `Tesla Motors`, `Alphabet Inc Class
# A` — fuzzy already matches, because the canonical name is a token subset. The
# embedding agrees there; it earns its place on the two cases above.
#
# The renames and subsidiaries are where the method breaks, and the scores show
# exactly why:
#
# - **`Google LLC` → Microsoft** (cos ≈ 0.38) is simply **wrong**. "Google" and
#   "Alphabet" share no linguistic similarity; the parent–subsidiary link is a
#   corporate fact, not a property of the strings.
# - **`Facebook Inc` → Meta Platforms** (cos ≈ 0.35) is *correct* — yet it scores
#   **below the wrong Google match**. The signal that should flag a rename is
#   weaker than the signal behind an outright error, so no single threshold
#   separates the good low-confidence match from the bad one.
#
# That non-separability is the lesson. A confident wrong match silently corrupts
# every downstream join, and the model is *least* reliable on exactly the cases —
# renames, subsidiaries, ticker reassignments — that a master database exists to
# track. Embeddings extend the probabilistic stage for paraphrases and
# abbreviations; they do not replace a curated alias layer keyed to the security
# master, where each such mapping is recorded deliberately rather than inferred.

# %% [markdown]
# ## 6. Building a Master Security Database
#
# A production-grade entity resolution system maintains a master database that:
# 1. Links all identifier types
# 2. Tracks name changes over time
# 3. Maps subsidiaries to parents
# 4. Stores confidence scores for probabilistic matches


# %%
class MasterSecurityDatabase:
    """
    Master Security Database for entity resolution.

    Maintains canonical mappings between company names and identifiers,
    with support for name changes, subsidiaries, and confidence tracking.

    Note: Uses list accumulation internally to avoid O(N²) memory copying
    from iterative pl.concat(). DataFrames are materialized lazily.
    """

    def __init__(self):
        # Use list accumulation to avoid O(N²) concat anti-pattern
        self._entity_storage: list[dict] = []
        self._variant_storage: list[dict] = []
        self._entities_df: pl.DataFrame | None = None
        self._variants_df: pl.DataFrame | None = None
        self._next_id = 1

    @property
    def entities(self) -> pl.DataFrame:
        """Lazily materialize entities DataFrame."""
        if self._entities_df is None or len(self._entity_storage) > 0:
            if self._entity_storage:
                new_df = pl.DataFrame(self._entity_storage)
                if self._entities_df is not None:
                    self._entities_df = pl.concat([self._entities_df, new_df])
                else:
                    self._entities_df = new_df
                self._entity_storage = []
            elif self._entities_df is None:
                # Return empty DataFrame with correct schema
                self._entities_df = pl.DataFrame(
                    schema={
                        "entity_id": pl.Int64,
                        "canonical_name": pl.Utf8,
                        "ticker": pl.Utf8,
                        "cik": pl.Utf8,
                        "lei": pl.Utf8,
                        "figi": pl.Utf8,
                        "parent_entity_id": pl.Int64,
                        "is_active": pl.Boolean,
                    }
                )
        return self._entities_df

    @property
    def name_variants(self) -> pl.DataFrame:
        """Lazily materialize name_variants DataFrame."""
        if self._variants_df is None or len(self._variant_storage) > 0:
            if self._variant_storage:
                new_df = pl.DataFrame(self._variant_storage)
                if self._variants_df is not None:
                    self._variants_df = pl.concat([self._variants_df, new_df])
                else:
                    self._variants_df = new_df
                self._variant_storage = []
            elif self._variants_df is None:
                # Return empty DataFrame with correct schema
                self._variants_df = pl.DataFrame(
                    schema={
                        "entity_id": pl.Int64,
                        "name_variant": pl.Utf8,
                        "valid_from": pl.Date,
                        "valid_to": pl.Date,
                        "is_primary": pl.Boolean,
                    }
                )
        return self._variants_df

    def add_entity(
        self,
        canonical_name: str,
        ticker: str,
        cik: str = None,
        lei: str = None,
        figi: str = None,
        name_variants: list[str] = None,
        parent_entity_id: int = None,
    ) -> int:
        """Add a new entity to the database."""
        entity_id = self._next_id
        self._next_id += 1

        # Accumulate to list (O(1)) instead of concat (O(N))
        self._entity_storage.append(
            {
                "entity_id": entity_id,
                "canonical_name": canonical_name,
                "ticker": ticker,
                "cik": cik,
                "lei": lei,
                "figi": figi,
                "parent_entity_id": parent_entity_id,
                "is_active": True,
            }
        )

        # Add name variants
        all_names = [canonical_name] + (name_variants or [])
        for i, name in enumerate(all_names):
            self._variant_storage.append(
                {
                    "entity_id": entity_id,
                    "name_variant": name,
                    "valid_from": None,
                    "valid_to": None,
                    "is_primary": i == 0,
                }
            )

        return entity_id

    def resolve(self, query: str, identifier_type: str = None, threshold: int = 80) -> dict | None:
        """
        Resolve a company name or identifier to canonical entity.

        Returns entity info or None if no match found.
        """
        # Stage 1: Try deterministic match on identifier
        if identifier_type and identifier_type in self.entities.columns:
            matches = self.entities.filter(pl.col(identifier_type) == query)
            if len(matches) > 0:
                return {
                    "entity_id": matches["entity_id"][0],
                    "canonical_name": matches["canonical_name"][0],
                    "ticker": matches["ticker"][0],
                    "match_method": f"deterministic:{identifier_type}",
                    "confidence": 100,
                }

        # Stage 2: Try fuzzy match on name variants
        all_variants = self.name_variants["name_variant"].to_list()
        match, score = fuzzy_match_company(query, all_variants, threshold=threshold)

        if match:
            # Find entity for this variant
            variant_row = self.name_variants.filter(pl.col("name_variant") == match)
            entity_id = variant_row["entity_id"][0]
            entity = self.entities.filter(pl.col("entity_id") == entity_id)

            return {
                "entity_id": entity_id,
                "canonical_name": entity["canonical_name"][0],
                "ticker": entity["ticker"][0],
                "match_method": "fuzzy:name_variant",
                "confidence": score,
                "matched_variant": match,
            }

        return None


# Build sample master database
msdb = MasterSecurityDatabase()

# Add entities with name variants
msdb.add_entity(
    "Microsoft Corporation",
    "MSFT",
    cik="0000789019",
    name_variants=["Microsoft Corp", "Microsoft Inc", "MSFT"],
)

msdb.add_entity(
    "Apple Inc.",
    "AAPL",
    cik="0000320193",
    name_variants=["Apple Computer", "Apple Computer Inc", "AAPL"],
)

msdb.add_entity(
    "Meta Platforms Inc.",
    "META",
    cik="0001326801",
    name_variants=["Facebook Inc", "Facebook", "Meta", "META"],
)

msdb.add_entity(
    "Alphabet Inc.",
    "GOOGL",
    cik="0001652044",
    name_variants=["Google Inc", "Google LLC", "GOOGL", "GOOG"],
)

# Test resolution
test_queries = [
    ("0000789019", "cik"),  # Deterministic
    ("Facebook", None),  # Fuzzy - old name
    ("Google LLC", None),  # Fuzzy - subsidiary name
    ("MSFT", None),  # Fuzzy - ticker as name
    ("Unknown Corp", None),  # No match
]

print("Master Database Resolution Results:")
for query, id_type in test_queries:
    result = msdb.resolve(query, identifier_type=id_type)
    if result:
        print(
            f"[OK]   {query!r:18}  -> {result['canonical_name']:24}  "
            f"({result['match_method']}, conf: {result['confidence']})"
        )
    else:
        print(f"[FAIL] {query!r:18}  -> No match")

# %% [markdown]
# ## 7. Practical Application: Matching Alternative Data
#
# Real-world entity resolution must handle name changes (Facebook → Meta),
# subsidiaries (Google LLC → Alphabet), ticker confusion (ZOOM vs ZM),
# and special characters (AT&T, S&P Global).
#
# Let's apply entity resolution to match messy alternative data (e.g., job postings)
# to our reference securities.

# %%
# Simulate messy alternative data
alt_data_companies = pl.DataFrame(
    {
        "source_id": range(1, 11),
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
        ],
        "job_postings": [150, 200, 180, 300, 120, 80, 90, 50, 160, 70],
    }
)


def batch_resolve(
    df: pl.DataFrame, name_column: str, reference_names: list[str], threshold: int = 75
) -> pl.DataFrame:
    """
    Batch resolve company names in a dataframe.

    Pre-normalizes candidates once to avoid O(N×M) string operations.

    NOTE: This uses iter_rows() for educational clarity - it shows exactly what
    happens per-row during entity resolution. For production systems with >10K rows,
    consider using map_elements() or batch processing with vectorized operations.
    The key optimization here is pre-normalizing candidates once outside the loop,
    reducing complexity from O(N×M) to O(N+M).
    """
    results = []

    # Pre-normalize candidates ONCE (not per-row) to avoid O(N×M) cost
    candidates_norm, candidates_orig = prepare_candidates(reference_names)

    for row in df.iter_rows(named=True):
        name = row[name_column]
        match, score = fuzzy_match_company(
            name, candidates_orig, threshold=threshold, candidates_norm=candidates_norm
        )

        results.append(
            {
                **row,
                "matched_name": match,
                "match_score": score,
                "match_status": "matched" if match else "unmatched",
            }
        )

    return pl.DataFrame(results)


# Resolve alternative data. Keep the applied cutoff in one constant so the
# threshold line on the score histogram below always reflects the value used here.
MATCH_THRESHOLD = 70
alt_data_to_resolve = alt_data_companies
resolved_alt_data = batch_resolve(
    alt_data_to_resolve,
    "raw_company_name",
    reference_securities["company_name"].to_list(),
    threshold=MATCH_THRESHOLD,
)

resolved_alt_data.select(
    ["source_id", "raw_company_name", "matched_name", "match_score", "job_postings"]
)

# %% [markdown]
# ## 8. Measuring Match Quality
#
# Always evaluate your entity resolution quality:
# - **Precision**: What fraction of matches are correct?
# - **Recall**: What fraction of true matches did we find?
# - **F1 Score**: Harmonic mean of precision and recall

# %%
match_stats = resolved_alt_data.group_by("match_status").agg(
    [pl.len().alias("count"), pl.mean("match_score").alias("avg_score")]
)
match_stats

# %%
# Score distribution
fig = px.histogram(
    resolved_alt_data.to_pandas(),
    x="match_score",
    nbins=20,
    title="Fuzzy scores are bimodal: confident matches near 100, non-matches at zero",
    labels={"match_score": "Fuzzy match score (0–100)", "count": "Number of names"},
    color_discrete_sequence=[COLORS["blue"]],
)
fig.update_yaxes(title_text="Number of names")
fig.add_vline(
    x=MATCH_THRESHOLD,
    line_dash="dash",
    line_color=COLORS["negative"],
    annotation_text=f"Threshold ({MATCH_THRESHOLD})",
)
fig.show()

# %% [markdown]
# ## Key Takeaways
#
# **Hierarchical matching**: Deterministic (CIK, LEI, FIGI) → Probabilistic (fuzzy strings) → embeddings, each stage handling what the previous one cannot
#
# **Algorithm selection**: `token_set_ratio` for word reordering, `partial_ratio` for substrings
#
# **Embeddings extend, not replace**: cosine similarity recovers paraphrases and tickers fuzzy misses (Amazon Web Services, msft), but renames and subsidiaries score low and non-separably — a curated alias table keyed to the security master is the real fix
#
# **Thresholds**: 85+ for precision, 70-80 for coverage (requires validation)
#
# **Libraries**: `rapidfuzz` (C-backed scorer used here), `sentence-transformers` (`all-MiniLM-L6-v2` embeddings), `polars` (data manipulation)
