# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Text Data Extraction: Structuring Unstructured Financial Documents
#
# **Chapter 4: Fundamental and Alternative Data**
# **Docker image**: `ml4t`
# **Section Reference**: See Section 4.5 for text dataset engineering concepts
#
# ## Purpose
#
# Corporate filings contain valuable information locked in unstructured text. This notebook
# demonstrates how to extract and structure high-value text blocks (MD&A, Risk Factors) from
# 10-K and 10-Q filings, creating clean datasets ready for NLP analysis in later chapters.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
# - Navigate SEC EDGAR filing structure
# - Extract specific sections from 10-K/10-Q filings (MD&A, Risk Factors)
# - Clean and normalize extracted text
# - Create point-in-time correct text datasets
# - Prepare data for downstream NLP analysis
#
# ## Cross-References
#
# - **Upstream**: SEC EDGAR filings via `02_sec_filing_explorer.py`
# - **Downstream**: Chapter 10 `08_text_feature_evaluation.py` (NLP feature engineering)
# - **Related**: `02_sec_filing_explorer.py` (filing access)
#
# ## Key Concepts
#
# - **10-K**: Annual report with comprehensive company information
# - **10-Q**: Quarterly report with interim financial information
# - **MD&A**: Management's Discussion and Analysis of Financial Condition
# - **Risk Factors**: Required disclosure of material risks
# - **Item Numbers**: Standard sections in SEC filings (Item 7 = MD&A)

# %%
"""Text Data Extraction — extract and structure high-value text blocks from SEC filings for NLP analysis."""

import warnings

warnings.filterwarnings("ignore")

import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from bs4 import BeautifulSoup
from edgar import Company, set_identity

# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI
EDGAR_TICKER = "AAPL"
EDGAR_FORM = "10-K"

# %% [markdown]
# ## 1. SEC Filing Structure
#
# 10-K filings follow a standard structure:
#
# | Item | Section | Content |
# |------|---------|---------|
# | Item 1 | Business | Company description and operations |
# | Item 1A | Risk Factors | Material risks to the business |
# | Item 7 | MD&A | Management's analysis of financial condition |
# | Item 7A | Quantitative Disclosures | Market risk exposures |
# | Item 8 | Financial Statements | Audited financials |
#
# The most valuable sections for NLP analysis:
# - **Item 1A (Risk Factors)**: Sentiment, risk themes, changes over time
# - **Item 7 (MD&A)**: Management's perspective on performance

# %%
# Define section patterns for extraction
# Note: 10-K and 10-Q have different item numbers for the same content
# 10-K: MD&A = Item 7, Risk Factors = Item 1A
# 10-Q: MD&A = Item 2 (Part I), Risk Factors = Item 1A (Part II, if updated)

SECTION_PATTERNS_10K = {
    "risk_factors": {
        # Line-anchored patterns to avoid Table of Contents matches
        "start_patterns": [
            r"^\s*ITEM\s*1A\.?\s*[-–—]?\s*RISK\s*FACTORS",
            r"^\s*Item\s*1A\.?\s*[-–—]?\s*Risk\s*Factors",
        ],
        "end_patterns": [
            r"^\s*ITEM\s*1B\b",
            r"^\s*ITEM\s*2\b",
            r"^\s*Item\s*1B\b",
            r"^\s*Item\s*2\b",
        ],
        "item_number": "1A",
    },
    "mda": {
        # Handle both straight and curly apostrophes: ' (U+0027) and ' (U+2019)
        "start_patterns": [
            r"^\s*ITEM\s*7\.?\s*[-–—]?\s*MANAGEMENT(?:'|\u2019)?S?\s*DISCUSSION",
            r"^\s*Item\s*7\.?\s*[-–—]?\s*Management(?:'|\u2019)?s?\s*Discussion",
        ],
        "end_patterns": [
            r"^\s*ITEM\s*7A\b",
            r"^\s*ITEM\s*8\b",
            r"^\s*Item\s*7A\b",
            r"^\s*Item\s*8\b",
        ],
        "item_number": "7",
    },
    "business": {
        "start_patterns": [
            r"^\s*ITEM\s*1\.?\s*[-–—]?\s*BUSINESS",
            r"^\s*Item\s*1\.?\s*[-–—]?\s*Business",
        ],
        "end_patterns": [
            r"^\s*ITEM\s*1A\b",
            r"^\s*ITEM\s*2\b",
            r"^\s*Item\s*1A\b",
            r"^\s*Item\s*2\b",
        ],
        "item_number": "1",
    },
}

# 10-Q patterns differ from 10-K
SECTION_PATTERNS_10Q = {
    "risk_factors": {
        # In 10-Q, risk factor updates appear in Part II, Item 1A
        # Line-anchored patterns with MULTILINE support (handled in extract_section)
        "start_patterns": [
            r"^\s*ITEM\s*1A\.?\s*[-–—]?\s*RISK\s*FACTORS",
            r"^\s*Item\s*1A\.?\s*[-–—]?\s*Risk\s*Factors",
        ],
        "end_patterns": [
            r"^\s*ITEM\s*1B\b",
            r"^\s*ITEM\s*2\b",
            r"^\s*Item\s*2\b",
        ],
        "item_number": "1A",
        # For 10-Q risk factors, prefer matches after PART II
        "require_after": r"PART\s*II",
    },
    "mda": {
        # In 10-Q, MD&A is Part I, Item 2 (NOT Item 7)
        # Handle both straight and curly apostrophes
        "start_patterns": [
            r"^\s*ITEM\s*2\.?\s*[-–—]?\s*MANAGEMENT(?:'|\u2019)?S?\s*DISCUSSION",
            r"^\s*Item\s*2\.?\s*[-–—]?\s*Management(?:'|\u2019)?s?\s*Discussion",
        ],
        "end_patterns": [
            r"^\s*ITEM\s*3\b",
            r"^\s*Item\s*3\b",
        ],
        "item_number": "2",
    },
}

# Default to 10-K patterns (backward compatible)
SECTION_PATTERNS = SECTION_PATTERNS_10K


def get_patterns_for_form(form_type: str) -> dict:
    """Get appropriate section patterns based on filing type."""
    form_type_upper = form_type.upper() if form_type else "10-K"
    if "10-Q" in form_type_upper:
        return SECTION_PATTERNS_10Q
    return SECTION_PATTERNS_10K


# %% [markdown]
# ## 2. Text Cleaning Functions
#
# Raw SEC filings contain HTML, special characters, and formatting that must be cleaned.


# %%
def clean_html(html_text: str, drop_tables: bool = False) -> str:
    """Remove HTML tags and convert to plain text while preserving paragraph boundaries."""
    if not html_text:
        return ""

    soup = BeautifulSoup(html_text, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Optionally remove tables (embedded financial tables in narrative sections)
    if drop_tables:
        for table in soup.find_all("table"):
            table.decompose()

    # Preserve paragraph boundaries by using newline separator
    text = soup.get_text(separator="\n")

    return text


# %%
def clean_text(text: str) -> str:
    """
    Clean extracted text while preserving paragraph breaks.

    Removes common filing artifacts and normalizes whitespace without
    flattening everything into a single line (preserves paragraph structure
    for downstream chunking and change detection).
    """
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove common filing artifacts (line-anchored for safety)
    text = re.sub(r"(?im)^\s*table of contents\s*$", "", text)
    text = re.sub(r"(?im)^\s*page\s+\d+\s*$", "", text)
    text = re.sub(r"(?im)^\s*\d+\s*of\s*\d+\s*$", "", text)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove excessive special characters but keep newlines
    text = re.sub(r"[_=\-]{3,}", " ", text)
    text = re.sub(r"[•●◦▪]", " ", text)

    # Normalize spaces within each line, but preserve line breaks
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n"))

    # Collapse excessive blank lines to paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text


# %%
def normalize_for_analysis(text: str) -> str:
    """
    Normalize text for NLP analysis.

    Converts to lowercase, removes numbers (optional),
    and standardizes formatting.
    """
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Standardize common abbreviations
    text = re.sub(r"\bu\.s\.?\b", "united states", text)
    text = re.sub(r"\be\.g\.?\b", "for example", text)
    text = re.sub(r"\bi\.e\.?\b", "that is", text)

    # Remove possessive 's
    text = re.sub(r"'s\b", "", text)

    # Remove numbers (optional - depends on use case)
    # text = re.sub(r"\d+", "", text)

    return text


# Test cleaning
sample_html = """
<p>ITEM 1A. RISK FACTORS</p>
<p>The following risks could <b>materially</b> affect our business:</p>
<p>• Competition may increase</p>
<p>• Economic conditions may deteriorate</p>
<p>Page 15 of 120</p>
<p>Table of Contents</p>
"""

cleaned = clean_text(clean_html(sample_html))
print("Sample cleaning:")
print(f"Original HTML length: {len(sample_html)}")
print(f"Cleaned text: {cleaned}")

# %% [markdown]
# ## 3. Section Extraction


# %%
def extract_section(text: str, section_name: str, patterns: dict = None) -> str | None:
    """
    Extract a specific section from a 10-K/10-Q filing using anchored item boundaries.

    Robust to TOC duplicates by finding all start_pattern matches and
    picking the candidate (start, end) span with the most content
    between it and the next end_pattern. Inline references inside
    narrative text are filtered out by the same span-length test —
    they don't have a matching end-pattern Item N+1 immediately after.

    Parameters
    ----------
    text : str
        Full filing text
    section_name : str
        Name of section to extract (e.g., 'risk_factors', 'mda')
    patterns : dict
        Custom patterns, or use SECTION_PATTERNS

    Returns
    -------
    str or None
        Extracted section text, or None if no candidate has substantive content.
    """
    if not text:
        return None

    if patterns is None:
        patterns = SECTION_PATTERNS

    if section_name not in patterns:
        raise ValueError(f"Unknown section: {section_name}")

    config = patterns[section_name]
    flags = re.IGNORECASE | re.MULTILINE | re.DOTALL

    # Some patterns may require matching after a specific marker
    # (e.g., PART II for 10-Q risk factors).
    offset = 0
    search_text = text
    require_after = config.get("require_after")
    if require_after:
        after_match = re.search(require_after, search_text, flags)
        if after_match:
            offset = after_match.start()
            search_text = search_text[after_match.start() :]

    # Collect every candidate start position from every start pattern.
    starts: list[int] = []
    for pattern in config["start_patterns"]:
        starts.extend(offset + m.end() for m in re.finditer(pattern, search_text, flags))
    if not starts:
        return None
    starts = sorted(set(starts))

    # For each candidate start, find the next end_pattern match and
    # measure the span. The real section beats every TOC entry on
    # span length because TOC entries are followed by the next TOC
    # entry within a few dozen characters.
    best_span: tuple[int, int] | None = None
    for start_pos in starts:
        end_pos = len(text)
        for end_pattern in config["end_patterns"]:
            end_match = re.search(end_pattern, text[start_pos:], flags)
            if end_match:
                end_pos = start_pos + end_match.start()
                break
        span = end_pos - start_pos
        if best_span is None or span > (best_span[1] - best_span[0]):
            best_span = (start_pos, end_pos)

    if best_span is None or (best_span[1] - best_span[0]) < 200:
        return None

    return clean_text(text[best_span[0] : best_span[1]])


# Create sample 10-K text for demonstration
sample_10k = """
FORM 10-K
ANNUAL REPORT

Table of Contents

PART I

ITEM 1. BUSINESS

Our company is a leading provider of technology solutions for enterprise
customers worldwide. We design, manufacture, and support a portfolio of
hardware, software, and cloud services that help organizations modernize
their data infrastructure.

We operate across three reportable segments — Cloud Platform, Enterprise
Software, and Professional Services — and sell directly through a global
sales force as well as a network of channel partners. Our customers span
financial services, healthcare, public sector, and manufacturing.

Recent strategic initiatives include expanding our hyperscale data center
footprint, deepening integrations with leading public cloud providers,
and broadening our generative-AI product portfolio.

ITEM 1A. RISK FACTORS

An investment in our securities involves a high degree of risk.
You should carefully consider the following risk factors:

COMPETITION RISKS
We face significant competition from larger companies with more resources.
New entrants may disrupt our market position.

ECONOMIC RISKS
Economic downturns could reduce customer spending on our products.
Currency fluctuations may impact our international revenue.

REGULATORY RISKS
Changes in regulations could increase our compliance costs.
Data privacy laws continue to evolve across jurisdictions.

ITEM 1B. UNRESOLVED STAFF COMMENTS

None.

ITEM 2. PROPERTIES

We lease our headquarters in San Francisco, California.

PART II

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION

Overview
The past year has been transformative for our company.
Revenue increased 25% driven by strong product adoption.

Results of Operations
Our gross margin improved to 72% from 68% in the prior year.
Operating expenses remained well controlled.

Liquidity and Capital Resources
We ended the year with $500 million in cash.
We believe current resources are sufficient for operations.

ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK

We are exposed to interest rate and foreign currency risks.

ITEM 8. FINANCIAL STATEMENTS
"""

for section in ["business", "risk_factors", "mda"]:
    extracted = extract_section(sample_10k, section)
    if extracted:
        preview = extracted[:200] + "..." if len(extracted) > 200 else extracted
        print(f"{section.upper()} ({len(extracted)} chars):")
        print(preview)
        print()

# %% [markdown]
# ## 4. Building a Text Dataset


# %%
_DEFAULT_SECTIONS = ("risk_factors", "mda")


def create_text_dataset(
    filings: list[dict], sections: list[str] = _DEFAULT_SECTIONS
) -> pl.DataFrame:
    """
    Create a structured dataset from multiple filings.

    Parameters
    ----------
    filings : list of dict
        Each dict should contain:
        - cik: Company CIK
        - company_name: Company name
        - accession_no: EDGAR accession number (stable filing identifier)
        - filing_date: Filing date
        - accepted_at: SEC acceptance timestamp (for intraday PIT)
        - period_end: Reporting period end date
        - filing_type: 10-K, 10-Q, etc.
        - text: Raw filing text
    sections : list of str
        Sections to extract

    Returns
    -------
    pl.DataFrame
        Structured text dataset with PIT-correct schema

    Note
    ----
    Automatically uses correct section patterns based on filing_type (10-K vs 10-Q).
    In 10-Q, MD&A is Item 2 (not Item 7 as in 10-K).
    """
    records = []

    for filing in filings:
        filing_type = filing.get("filing_type", "10-K")
        base_record = {
            "cik": filing.get("cik"),
            "company_name": filing.get("company_name"),
            # Stable filing identifier (prefer over cik + filing_date)
            "accession_no": filing.get("accession_no"),
            "filing_type": filing_type,
            # PIT timestamps
            "filing_date": filing.get("filing_date"),
            "accepted_at": filing.get("accepted_at"),  # Intraday PIT
            "period_end": filing.get("period_end"),
        }

        text = filing.get("text", "")

        # Use form-type-aware patterns
        patterns = get_patterns_for_form(filing_type)

        for section in sections:
            # Skip sections not defined for this form type
            if section not in patterns:
                continue

            extracted = extract_section(text, section, patterns)

            record = {
                **base_record,
                "section": section,
                "text": extracted or "",
                "text_length": len(extracted) if extracted else 0,
                "word_count": len(extracted.split()) if extracted else 0,
            }
            records.append(record)

    return pl.DataFrame(records)


# Create sample filings with PIT-correct schema
sample_filings = [
    {
        "cik": "0000320193",
        "company_name": "Apple Inc.",
        "accession_no": "0000320193-24-000081",
        "filing_date": datetime(2024, 10, 31),
        "accepted_at": datetime(2024, 10, 31, 16, 35, 12),  # SEC acceptance timestamp
        "period_end": datetime(2024, 9, 28),
        "filing_type": "10-K",
        "text": sample_10k,
    },
    {
        "cik": "0000789019",
        "company_name": "Microsoft Corporation",
        "accession_no": "0000789019-24-000042",
        "filing_date": datetime(2024, 7, 30),
        "accepted_at": datetime(2024, 7, 30, 16, 5, 45),
        "period_end": datetime(2024, 6, 30),
        "filing_type": "10-K",
        "text": sample_10k,  # Using same sample for demo
    },
]

text_dataset = create_text_dataset(sample_filings)
text_dataset

# %% [markdown]
# ## 5. Text Statistics and Quality Checks


# %%
def calculate_text_statistics(text: str) -> dict:
    """Calculate statistics about extracted text."""
    if not text:
        return {
            "word_count": 0,
            "sentence_count": 0,
            "avg_word_length": 0,
            "avg_sentence_length": 0,
            "unique_words": 0,
            "lexical_diversity": 0,
        }

    words = text.lower().split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    word_count = len(words)
    unique_words = len(set(words))

    return {
        "word_count": word_count,
        "sentence_count": len(sentences),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        "avg_sentence_length": word_count / len(sentences) if sentences else 0,
        "unique_words": unique_words,
        "lexical_diversity": unique_words / word_count if word_count > 0 else 0,
    }


# %% [markdown]
# ### Quality Check Extraction
# Flag potential issues: missing sections, truncated extractions, or boundary contamination.


# %%
def quality_check_extraction(df: pl.DataFrame, min_words: int = 100) -> pl.DataFrame:
    """
    Check quality of extracted text.

    Flags potential issues:
    - Missing (extraction returned nothing)
    - Too short (extraction may have failed partially)
    - Too long (may have captured extra sections)
    - Boundary leak (captured content from next section)
    """
    # Add quality flags - NOTE: order matters! Check missing first since 0 < min_words
    result = df.with_columns(
        [
            pl.when(pl.col("word_count") == 0)
            .then(pl.lit("missing"))
            .when(pl.col("word_count") < min_words)
            .then(pl.lit("too_short"))
            .when(pl.col("word_count") > 50000)
            .then(pl.lit("suspiciously_long"))
            .otherwise(pl.lit("ok"))
            .alias("quality_flag")
        ]
    )

    # Check for boundary contamination (e.g., "Item 8" appearing in MD&A)
    # This indicates the end boundary wasn't properly detected
    result = result.with_columns(
        pl.when((pl.col("section") == "mda") & pl.col("text").str.contains(r"(?i)\bITEM\s*8\b"))
        .then(pl.lit(True))
        .when(
            (pl.col("section") == "risk_factors")
            & pl.col("text").str.contains(r"(?i)\bITEM\s*2\b.*\bPROPERTIES\b")
        )
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("possible_boundary_leak")
    )

    return result


text_dataset_checked = quality_check_extraction(text_dataset)
text_dataset_checked.select(["company_name", "section", "word_count", "quality_flag"])

# %%
sample_text = text_dataset.filter(
    (pl.col("section") == "risk_factors") & (pl.col("word_count") > 0)
)["text"][0]

stats = calculate_text_statistics(sample_text)
print("Text Statistics (Risk Factors section):")
for k, v in stats.items():
    print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

# %% [markdown]
# ## 6. Change Analysis Over Time
#
# Track how MD&A and Risk Factors change between filings.


# %%
def calculate_text_similarity(text1: str, text2: str) -> dict:
    """
    Calculate similarity between two text documents.

    Uses simple word overlap metrics (Jaccard similarity).
    For production, consider using TF-IDF or embeddings.
    """
    if not text1 or not text2:
        return {"jaccard": 0, "word_overlap_pct": 0}

    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    intersection = words1 & words2
    union = words1 | words2

    jaccard = len(intersection) / len(union) if union else 0
    overlap_pct = len(intersection) / len(words1) if words1 else 0

    return {
        "jaccard_similarity": jaccard,
        "word_overlap_pct": overlap_pct,
        "new_words": len(words2 - words1),
        "removed_words": len(words1 - words2),
    }


# %%
def extract_new_risk_factors(old_text: str, new_text: str) -> list[str]:
    """
    Identify potentially new risk factors by finding new paragraphs.

    Simple heuristic: paragraphs in new filing not present in old.
    """
    if not old_text or not new_text:
        return []

    # Split into paragraphs
    old_paragraphs = set(p.strip().lower() for p in old_text.split("\n\n") if len(p.strip()) > 50)
    new_paragraphs = [p.strip() for p in new_text.split("\n\n") if len(p.strip()) > 50]

    new_risks = []
    for para in new_paragraphs:
        # Check if paragraph is substantially new (low similarity to all old paragraphs)
        para_lower = para.lower()
        if para_lower not in old_paragraphs:
            # Simple check - could be improved with fuzzy matching
            new_risks.append(para[:200] + "..." if len(para) > 200 else para)

    return new_risks[:5]  # Return top 5 new items


# %%
# Simulate two versions of risk factors
old_risks = """
COMPETITION RISKS
We face significant competition from larger companies.

ECONOMIC RISKS
Economic downturns could reduce customer spending.
"""

new_risks = """
COMPETITION RISKS
We face significant competition from larger companies with more resources.

ECONOMIC RISKS
Economic downturns could reduce customer spending.

AI TECHNOLOGY RISKS
Rapid advances in artificial intelligence may disrupt our business model.
We may need to make significant investments to remain competitive.

CYBERSECURITY RISKS
We face increasing threats from sophisticated cyber attacks.
"""

similarity = calculate_text_similarity(old_risks, new_risks)
print("Similarity Metrics:")
for k, v in similarity.items():
    print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

new_items = extract_new_risk_factors(old_risks, new_risks)
print(f"\nNew Risk Factors Identified: {len(new_items)}")
for i, item in enumerate(new_items, 1):
    print(f"  {i}. {item[:100]}...")

# %% [markdown]
# ## 7. Saving the Dataset


# %%
def save_text_dataset(df: pl.DataFrame, output_path: str, format: str = "parquet") -> None:
    """
    Save text dataset to file.

    Parameters
    ----------
    df : pl.DataFrame
        Text dataset
    output_path : str
        Output file path
    format : str
        'parquet' or 'csv'
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        df.write_parquet(path)
    elif format == "csv":
        df.write_csv(path)
    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Saved {len(df)} records to {path}")


# %%
# Example save (commented out to avoid file creation in demo)
# save_text_dataset(text_dataset, "/tmp/sec_text_data.parquet")
print("\nTo save dataset:")
print("  save_text_dataset(text_dataset, 'data/sec_text_data.parquet')")

# %% [markdown]
# ## 8. End-to-End Run Against a Live EDGAR Filing
#
# The previous sections developed the extraction pipeline against an
# inline sample 10-K. The same helpers run unchanged against a real
# filing: `edgartools` fetches the latest AAPL 10-K from EDGAR, our
# `extract_section` + `clean_text` + `calculate_text_statistics`
# helpers process it, and the resulting per-section summary mirrors
# what the synthetic example produced earlier in the notebook.
#
# The SEC requires a real User-Agent identity for every request and
# blocks placeholder addresses. Set `EDGAR_IDENTITY` in your environment
# to `"Your Name your.email@domain.com"` before running this section:
#
# ```bash
# export EDGAR_IDENTITY="Jane Doe jane@example.org"
# ```
#
# The fetch is read-only, rate-limited by edgartools, and typically
# completes in 1–3 seconds.

# %%
edgar_identity = os.environ.get("EDGAR_IDENTITY")
if not edgar_identity:
    raise RuntimeError(
        "EDGAR_IDENTITY environment variable is not set. The SEC requires a "
        "real User-Agent (name + email) for every EDGAR request and blocks "
        "placeholder addresses. Set it before running this section, e.g. "
        '`export EDGAR_IDENTITY="Jane Doe jane@example.org"`.'
    )
set_identity(edgar_identity)

company = Company(EDGAR_TICKER)
latest_filing = company.get_filings(form=EDGAR_FORM).latest()
print(
    f"{EDGAR_TICKER} {EDGAR_FORM}: accession {latest_filing.accession_no}, filed {latest_filing.filing_date}"
)

# %% [markdown]
# Fetch the filing text and run the section extractor with the
# form-aware patterns. The two highest-value sections (Risk Factors
# and MD&A) come back as plain strings — ready for `clean_text`,
# tokenisation, or downstream NLP feature engineering.

# %%
real_text = latest_filing.text()
form_patterns = get_patterns_for_form(EDGAR_FORM)

real_risk = extract_section(real_text, "risk_factors", form_patterns)
real_mda = extract_section(real_text, "mda", form_patterns)
real_business = extract_section(real_text, "business", form_patterns)

real_stats = pl.DataFrame(
    [
        {
            "section": name,
            "extracted": text is not None,
            "char_count": len(text) if text else 0,
            "word_count": calculate_text_statistics(text)["word_count"] if text else 0,
        }
        for name, text in [
            ("business", real_business),
            ("risk_factors", real_risk),
            ("mda", real_mda),
        ]
    ]
)
real_stats

# %% [markdown]
# A successful run shows non-zero word counts on each row — the same
# pipeline that produced the synthetic-text statistics in Section 5
# works against a current SEC filing without code changes. From here,
# the cleaned strings feed directly into the NLP feature pipeline in
# Chapter 10.

# %% [markdown]
# ## 9. Key Takeaways
#
# ### Text Extraction Best Practices
#
# 1. **Use line-anchored regex patterns for section boundaries**
#    - Item numbers are standardized (Item 1A, Item 7)
#    - But formatting varies - need multiple patterns with `^` anchoring
#    - Skip Table of Contents matches by searching after TOC marker
#
# 2. **Clean thoroughly but preserve paragraph structure**
#    - Remove HTML tags and filing artifacts
#    - Normalize whitespace *within* lines, but keep paragraph breaks
#    - Paragraph structure enables change detection over time
#
# 3. **Validate extraction quality**
#    - Check text length (too short = extraction failed)
#    - Check for boundary leaks (e.g., "Item 8" in MD&A)
#    - Track quality metrics for iterative rule refinement
#
# 4. **Track point-in-time with stable identifiers**
#    - Use **accession number** as primary key (not cik + filing_date)
#    - Use **accepted_at** for intraday PIT correctness
#    - Keep both filing_date and period_end for different query patterns
#
# 5. **Store efficiently with audit trail**
#    - Parquet format for large datasets
#    - Include metadata (CIK, accession, timestamps, section)
#    - Consider storing both raw and cleaned text for reproducibility
#
# ### Most Valuable Sections
#
# | Section | Alpha Use Case |
# |---------|----------------|
# | **Risk Factors (1A)** | Sentiment analysis, risk themes, change detection |
# | **MD&A (Item 7)** | Management sentiment, outlook changes |
# | **Business (Item 1)** | Competitive position, strategy changes |
#
# ### Forward Reference
#
# This structured text dataset is the input for NLP techniques covered in later chapters:
# - Sentiment scoring (positive/negative language)
# - Topic modeling (identify themes)
# - Named entity recognition (extract companies, products)
# - Text embeddings (semantic similarity)
