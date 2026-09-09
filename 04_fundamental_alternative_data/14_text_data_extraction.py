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
# # Text Data Extraction: Structuring Unstructured Financial Documents
#
# **Chapter 4: Fundamental and Alternative Data**
# **Docker image**: `ml4t`
# **Section Reference**: Section 4.5 (Using Text Data for NLP Features)
#
# ## Purpose
#
# An annual report is a quarter of a million characters of which a model wants perhaps a
# twentieth. The Securities and Exchange Commission prescribes what a 10-K must contain and
# numbers the parts, so the document has a structure; what it does not have is markup that a
# parser can rely on, because the numbering appears in the table of contents, in cross-references
# inside the narrative, and again as the actual section heading, in whatever typography the filer
# chose.
#
# Getting from a filing to a section, reliably, over thousands of documents, is the whole job of
# this notebook. Everything downstream in the book - sentiment scoring, topic models, embeddings -
# assumes that job was done properly, and the failure mode is silent: an extractor that returns
# the table of contents entry instead of the section returns a plausible-looking short string.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
#
# - Name the items of a 10-K a research pipeline usually wants, and say where the same content
#   sits in a 10-Q.
# - Strip HTML from a filing without destroying the paragraph boundaries a later stage needs.
# - Locate a section's boundaries when its item number appears many times in the document, and
#   explain why the first match is the wrong one.
# - Check an extraction for the two failures it can have - returning nothing, and running past
#   the section's end - and report the rate rather than assuming success.
# - Measure how much of a section changed between two consecutive filings, which is the input to
#   the change-detection signals in Chapter 10.
#
# ## Prerequisites
#
# This notebook reads live filings from EDGAR, which requires an identifying `User-Agent`:
#
# ```bash
# export EDGAR_IDENTITY="Jane Doe jane@example.org"
# ```
#
# ## Cross-References
#
# - **Related**: [`02_sec_filing_explorer`](02_sec_filing_explorer.ipynb) (reaching filings through the library)
# - **Downstream**: `10_text_feature_engineering/09_filing_text_signals.py` (signals built on filing text)
#
# ## Key Concepts
#
# - **10-K**: the annual report, filed once a year and the most complete narrative disclosure.
# - **10-Q**: the quarterly report, shorter, and with the same content under different item
#   numbers.
# - **Item 1A, Risk Factors**: the risks management is required to disclose. The largest narrative
#   block in a modern filing and the most mined.
# - **Item 7, Management's Discussion and Analysis**: management's account of the period's
#   results, in its own words.
# - **Accession number**: the identifier the SEC assigns to a filing. Unique, permanent, and the
#   right primary key for a text dataset.

# %%
"""Text Data Extraction - extract and structure high-value text blocks from SEC filings for NLP analysis."""

import os
import re
import warnings

# EdgarTools re-exports three legacy HTML helpers and each one raises a DeprecationWarning as it
# is imported. They are the same three every time and say nothing about the filings.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="edgar")

import plotly.express as px
import polars as pl
from bs4 import BeautifulSoup
from edgar import Company, set_identity

from utils.paths import get_output_dir
from utils.style import COLORS, show_plotly_with_alt

# %% [markdown]
# The company and form choose which filings are read. Two consecutive filings of the same form
# are fetched, because the second is what makes the change measurement in Part 6 possible; one
# filing alone can be extracted from but not compared.

# %% tags=["parameters"]
TICKER = "AAPL"  # any filer with a long 10-K history works here
FORM = "10-K"  # switch to "10-Q" to exercise the quarterly item numbering
N_FILINGS = 2  # consecutive filings to read, most recent first
MIN_SECTION_WORDS = 100  # an extraction shorter than this is treated as a failure
MAX_SECTION_WORDS = 50_000  # and one longer than this as having run past its boundary

# %%
OUTPUT_DIR = get_output_dir(4, "sec_text")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. What a filing is made of
#
# A 10-K is a numbered sequence of items whose meaning is fixed by regulation. Three of them carry
# almost all the narrative a research pipeline wants.
#
# | Item | Section | What it holds |
# |------|---------|---------------|
# | 1 | Business | What the company does, its segments and its markets |
# | 1A | Risk Factors | The risks management is required to disclose |
# | 7 | Management's Discussion and Analysis | Management's account of the period's results |
# | 7A | Quantitative and Qualitative Disclosures | Market risk exposures |
# | 8 | Financial Statements | The audited financials |
#
# The numbering is **form-specific**, and this is the single most common source of a silently
# empty extraction. In a 10-Q, management's discussion is Item 2 of Part I, not Item 7, and the
# risk factors, if updated at all, are Item 1A of Part II. An extractor that hardcodes the 10-K
# numbers returns nothing on a 10-Q and says nothing about why.

# %%
SECTIONS = {
    "10-K": {
        "business": {
            "starts": [r"^\s*ITEM\s*1\.?\s*[-–—]?\s*BUSINESS"],
            "ends": [r"^\s*ITEM\s*1A\b", r"^\s*ITEM\s*2\b"],
        },
        "risk_factors": {
            "starts": [r"^\s*ITEM\s*1A\.?\s*[-–—]?\s*RISK\s*FACTORS"],
            "ends": [r"^\s*ITEM\s*1B\b", r"^\s*ITEM\s*2\b"],
        },
        "mda": {
            # Filers use both the straight and the curly apostrophe in "Management's".
            "starts": [r"^\s*ITEM\s*7\.?\s*[-–—]?\s*MANAGEMENT(?:'|’)?S?\s*DISCUSSION"],
            "ends": [r"^\s*ITEM\s*7A\b", r"^\s*ITEM\s*8\b"],
        },
    },
    "10-Q": {
        "risk_factors": {
            "starts": [r"^\s*ITEM\s*1A\.?\s*[-–—]?\s*RISK\s*FACTORS"],
            "ends": [r"^\s*ITEM\s*1B\b", r"^\s*ITEM\s*2\b"],
            "after": r"PART\s*II",
        },
        "mda": {
            "starts": [r"^\s*ITEM\s*2\.?\s*[-–—]?\s*MANAGEMENT(?:'|’)?S?\s*DISCUSSION"],
            "ends": [r"^\s*ITEM\s*3\b"],
        },
    },
}
FLAGS = re.IGNORECASE | re.MULTILINE | re.DOTALL

# %% [markdown]
# ## 2. The filings
#
# EDGAR requires every request to identify its sender, and rejects placeholder addresses. The
# fetch is read-only and rate-limited by the library.

# %%
identity = os.environ.get("EDGAR_IDENTITY")
if not identity:
    raise RuntimeError(
        "EDGAR_IDENTITY environment variable is not set. The SEC requires a "
        "real User-Agent (name + email) for every EDGAR request and blocks "
        "placeholder addresses. Set it before running this notebook, e.g. "
        '`export EDGAR_IDENTITY="Jane Doe jane@example.org"`.'
    )
set_identity(identity)

# `amendments=False` keeps 10-K/A out of the list. An amendment covers a period the original
# already covered and often omits the narrative items entirely, so a pair drawn without this
# filter can be two filings of the same period presented as consecutive ones.
filings = Company(TICKER).get_filings(form=FORM, amendments=False)[:N_FILINGS]
documents = [
    {
        "cik": filing.cik,
        "company_name": filing.company,
        "accession_no": filing.accession_no,
        "form": filing.form,
        "filing_date": filing.filing_date,
        "accepted_at": filing.acceptance_datetime,
        "period_end": filing.period_of_report,
        "text": filing.text(),
    }
    for filing in filings
]

pl.DataFrame(
    [
        {
            "accession_no": d["accession_no"],
            "filing_date": d["filing_date"],
            "period_end": d["period_end"],
            "characters": len(d["text"]),
        }
        for d in documents
    ]
)

# %% [markdown]
# The acceptance timestamp is worth carrying alongside the filing date. A filing accepted after
# the close is public that evening and tradeable the next morning, and a strategy that acts on the
# filing date alone is a few hours early on every after-hours submission.

# %% [markdown]
# ## 3. Cleaning
#
# `filing.text()` already returns text rather than markup, but a pipeline that fetches documents
# directly gets HTML, so the conversion belongs here. The one thing to preserve is the paragraph
# boundary: a converter that joins everything with spaces destroys the unit that the change
# detection in Part 6 compares.


# %%
def html_to_text(html: str, drop_tables: bool = False) -> str:
    """Convert filing HTML to text, keeping one newline per block element."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
    if drop_tables:
        # Financial tables inside a narrative section are numbers, not prose; dropping them is
        # right for sentiment work and wrong for anything reading the figures.
        for table in soup.find_all("table"):
            table.decompose()
    return soup.get_text(separator="\n")


def clean_text(text: str) -> str:
    """Remove filing furniture and normalize whitespace without flattening paragraphs."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Page furniture, anchored to whole lines so a mention inside a sentence survives.
    text = re.sub(r"(?im)^\s*table of contents\s*$", "", text)
    text = re.sub(r"(?im)^\s*page\s+\d+\s*$", "", text)
    text = re.sub(r"(?im)^\s*\d+\s*of\s*\d+\s*$", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[_=\-]{3,}", " ", text)
    text = re.sub(r"[•●◦▪]", " ", text)
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# %% [markdown]
# ## 4. Finding a section
#
# The naive approach is to search for the item heading and take everything to the next one. It
# fails on the first line of every filing, because the item heading appears in the table of
# contents before it appears as a section, and it appears again wherever the narrative
# cross-references another item.
#
# Counting the matches in a real document is the quickest way to see the size of the problem.

# %%
first = documents[0]
for section, config in SECTIONS[FORM].items():
    matches = [
        m.start()
        for pattern in config["starts"]
        for m in re.finditer(pattern, first["text"], FLAGS)
    ]
    print(f"{section}: {len(matches)} places match the heading pattern")

# %% [markdown]
# Every section matches in more than one place, and only one of those places is the section. What
# separates them is **span**: a table of contents entry is followed by the next contents entry a
# line or two later, while the real heading is followed by the whole section before the next item
# begins. Taking the candidate with the longest span to its own end pattern picks the section
# without needing to know where the table of contents ends, and it keeps working on a filer whose
# contents page is formatted differently.


# %%
def extract_section(text: str, section: str, form: str) -> str | None:
    """The longest span between this section's heading and the item that follows it."""
    config = SECTIONS[form][section]

    # A 10-Q's risk factors live in Part II, and Part I has an Item 1A of its own meaning
    # something else. Where a section declares a marker, the search starts after it.
    offset, haystack = 0, text
    if marker := config.get("after"):
        if found := re.search(marker, text, FLAGS):
            offset, haystack = found.start(), text[found.start() :]

    starts = sorted(
        {
            offset + m.end()
            for pattern in config["starts"]
            for m in re.finditer(pattern, haystack, FLAGS)
        }
    )
    if not starts:
        return None

    best = None
    for start in starts:
        end = len(text)
        for pattern in config["ends"]:
            if found := re.search(pattern, text[start:], FLAGS):
                end = start + found.start()
                break
        if best is None or (end - start) > (best[1] - best[0]):
            best = (start, end)
    return clean_text(text[best[0] : best[1]])


# %% [markdown]
# ## 5. The dataset, and whether the extraction worked
#
# One row per filing and section, keyed on the accession number. The word count is what the
# quality check reads: an extraction can fail by returning nothing, by returning a fragment - the
# table of contents entry, if the span rule went wrong - or by running past its own end and
# swallowing the next item.

# %%
records = [
    {
        "cik": document["cik"],
        "company_name": document["company_name"],
        "accession_no": document["accession_no"],
        "form": document["form"],
        "filing_date": document["filing_date"],
        "accepted_at": document["accepted_at"],
        "period_end": document["period_end"],
        "section": section,
        "text": extracted or "",
        "word_count": len(extracted.split()) if extracted else 0,
    }
    for document in documents
    for section in SECTIONS[FORM]
    for extracted in [extract_section(document["text"], section, FORM)]
]
dataset = pl.DataFrame(records).with_columns(
    quality=pl.when(pl.col("word_count") == 0)
    .then(pl.lit("nothing extracted"))
    .when(pl.col("word_count") < MIN_SECTION_WORDS)
    .then(pl.lit("too short"))
    .when(pl.col("word_count") > MAX_SECTION_WORDS)
    .then(pl.lit("ran past the boundary"))
    .otherwise(pl.lit("ok")),
    # A section that swallowed the next item carries that item's heading inside it.
    boundary_leak=pl.when(pl.col("section") == "mda")
    .then(pl.col("text").str.contains(r"(?im)^\s*ITEM\s*8\b"))
    .when(pl.col("section") == "risk_factors")
    .then(pl.col("text").str.contains(r"(?im)^\s*ITEM\s*1B\b"))
    .otherwise(False),
)

print(f"Extractions attempted: {len(dataset)}")
print(f"Extractions the quality check passed: {(dataset['quality'] == 'ok').sum()}")
print(f"Extractions carrying the next item's heading: {int(dataset['boundary_leak'].sum())}")
dataset.select("accession_no", "filing_date", "section", "word_count", "quality", "boundary_leak")

# %% [markdown]
# Reporting the rate rather than eyeballing one extraction is the point. Over two filings it is a
# formality; over a corpus of thousands it is the only way to know that a formatting change at one
# filer has not silently emptied a column, and the same three columns scale unchanged.

# %%
latest_sections = dataset.filter(pl.col("accession_no") == documents[0]["accession_no"]).sort(
    "word_count", descending=True
)
labels = {"business": "Business", "risk_factors": "Risk Factors", "mda": "MD&A"}
fig = px.bar(
    latest_sections.with_columns(label=pl.col("section").replace(labels)).to_pandas(),
    x="word_count",
    y="label",
    orientation="h",
    title="Risk factors are several times longer than any other narrative section",
    labels={"word_count": "Words extracted", "label": ""},
    color_discrete_sequence=[COLORS["blue"]],
    text="word_count",
)
fig.update_traces(textposition="outside", cliponaxis=False)
fig.update_layout(
    height=320,
    yaxis=dict(autorange="reversed"),
    xaxis_range=[0, float(latest_sections["word_count"].max()) * 1.18],
    margin=dict(l=120, r=60),
)
show_plotly_with_alt(
    fig,
    "Horizontal bar chart of the words extracted from each narrative section of the most recent "
    "filing. The risk factors bar is several times longer than the other two, which are of "
    "similar length to each other.",
)

# %% [markdown]
# The proportions are why risk factors are the most mined section in the corpus. They are also why
# the section is hard to use: a block that long changes a little every year for reasons that have
# nothing to do with the business, so the signal is in what changed rather than in what it says.

# %% [markdown]
# ## 6. What changed between the two filings
#
# The same section from two consecutive filings can be compared directly. Two measurements are
# worth having and they answer different questions. **Jaccard similarity** over the vocabulary
# says how much of the language is shared, which is a measure of drift. The **paragraphs present
# in the new filing and absent from the old** are the additions, and those are the ones that name
# a risk management decided to start disclosing.


# %%
def vocabulary_overlap(old: str, new: str) -> dict:
    """Shared vocabulary between two documents, and what each has that the other does not."""
    old_words, new_words = set(old.lower().split()), set(new.lower().split())
    union = old_words | new_words
    return {
        "jaccard_similarity": len(old_words & new_words) / len(union) if union else 0.0,
        "words_only_in_new": len(new_words - old_words),
        "words_only_in_old": len(old_words - new_words),
    }


def added_paragraphs(old: str, new: str, min_characters: int = 200) -> list[str]:
    """Paragraphs of the new document that do not appear verbatim in the old one."""
    old_paragraphs = {p.strip().lower() for p in old.split("\n\n")}
    return [
        paragraph.strip()
        for paragraph in new.split("\n\n")
        if len(paragraph.strip()) >= min_characters
        and paragraph.strip().lower() not in old_paragraphs
    ]


# %%
CHANGE_SECTION = "risk_factors"
newer, older = documents[0]["accession_no"], documents[1]["accession_no"]
comparable = dataset.filter(
    (pl.col("section") == CHANGE_SECTION)
    & (pl.col("quality") == "ok")
    & ~pl.col("boundary_leak")
    & pl.col("accession_no").is_in([older, newer])
)

# An extraction the quality check rejected must not enter the comparison. An empty older section
# scores zero similarity and marks every paragraph of the newer one as an addition, which reads
# as a company rewriting its risk factors rather than as a failed extraction.
if comparable.height < 2:
    additions = []
    print(f"Both filings' {CHANGE_SECTION} did not pass the quality check; no comparison made.")
else:
    texts = {row["accession_no"]: row["text"] for row in comparable.iter_rows(named=True)}
    overlap = vocabulary_overlap(texts[older], texts[newer])
    additions = added_paragraphs(texts[older], texts[newer])
    print(f"Comparing {CHANGE_SECTION} between {older} and {newer}")
    for name, value in overlap.items():
        print(f"  {name}: {value:.3f}" if isinstance(value, float) else f"  {name}: {value}")
    print(f"  paragraphs in the newer filing that are not verbatim in the older: {len(additions)}")

# %% [markdown]
# A high vocabulary overlap with a substantial number of changed paragraphs is the normal result,
# and it is why the vocabulary measure alone is a poor change detector: a filer can rewrite a
# paragraph entirely without introducing a single new word.
#
# The paragraph comparison is exact rather than fuzzy, so a single reworded clause makes a
# paragraph count as new. That over-reports for a change-detection signal and under-reports
# nothing, which is the safer direction for a screen a human reads; a production version scores
# similarity between paragraphs rather than testing them for equality.

# %%
for i, paragraph in enumerate(additions[:3], 1):
    print(f"--- changed paragraph {i} ---")
    print(paragraph[:400] + ("..." if len(paragraph) > 400 else ""))
    print()

# %% [markdown]
# ## 7. Saving the dataset
#
# Parquet, keyed on the accession number, with the timestamps and the section name beside the
# text. The raw filing text is deliberately not stored alongside it: it is reproducible from the
# accession number, and it is twenty times the size.

# %%
output_file = OUTPUT_DIR / "sec_filing_sections.parquet"
dataset.write_parquet(output_file)
print(f"Wrote {len(dataset)} rows to {output_file}")
pl.read_parquet(output_file).select("accession_no", "section", "word_count", "quality")

# %% [markdown]
# ## Key Takeaways
#
# 1. The item numbers are form-specific. Management's discussion is Item 7 of a 10-K and Item 2 of
#    a 10-Q's Part I, so an extractor built for one silently returns nothing on the other. Select
#    the pattern set from the form type rather than defaulting to it.
# 2. An item heading appears several times in a filing - in the table of contents, in
#    cross-references, and once as the section. The first match is almost never the right one, and
#    the span to the next item is what distinguishes them: a contents entry runs a line, a section
#    runs thousands.
# 3. Preserve paragraph boundaries through the cleaning. They are the unit a change detector
#    compares, and a converter that joins blocks with spaces destroys them irrecoverably.
# 4. Report the extraction rate, not an example. The failure is silent - a plausible short string
#    where a section should be - and the only thing that catches it at corpus scale is counting
#    the empty, the short, and the ones carrying the next item's heading.
# 5. Key the dataset on the accession number and carry the acceptance timestamp. A company and a
#    date do not identify a filing when an amendment exists, and a filing accepted after the close
#    is not tradeable until the next session.
# 6. Vocabulary overlap and changed paragraphs measure different things. A filer can rewrite a
#    paragraph without adding a word, so a signal built on vocabulary alone misses exactly the
#    rewrites that matter.
