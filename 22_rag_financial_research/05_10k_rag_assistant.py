# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 10-K Due Diligence Assistant: RAG with Verifiable Citations
#
# **Docker image**: `ml4t-gpu`
#
# **Chapter 22: RAG for Financial Research**
#
# This notebook builds a 10-K due-diligence RAG (Retrieval-Augmented Generation)
# assistant on top of LlamaIndex. Key features:
#
# 1. **Sentence-window chunking** - `SentenceSplitter` over the canonical 10-K
#    parquet text (LlamaParse-style structure-aware parsing is a production
#    upgrade discussed in §22.3, not implemented here)
# 2. **Dense vector retrieval** - local BGE-small embeddings indexed in ChromaDB
# 3. **Constraint-Based Prompting** - enforcing grounded, citable answers
# 4. **Numeric Retrieve→Extract→Compute→Narrate** - code-side arithmetic on
#    extracted figures so the LLM never does the math itself
# 5. **Harness Diagnostics** - lightweight retrieval/citation/abstention checks
#
# ## Learning Objectives
# - Build a citation-constrained 10-K assistant with abstention checks.
# - Run the retrieve-extract-compute-narrate pattern for numeric questions.
# - Analyze RAG pipeline error sources and select/justify components.
#
# **Book Reference**: Chapter 22, Section 22.8 (Applications)
#
# ## Prerequisites
# - SP100 10-K filings (`data/equities/fundamentals/10k/sp100/`)
# - OpenAI API key for LLM generation (optional - retrieval works without it)
# - Embeddings use local HuggingFace model (no API key needed)

# %% [markdown]
# ## 1. Setup and Imports
#
# This setup fixes the document budget, retrieval depth, and optional
# dependencies that determine whether the notebook runs a full or fallback RAG path.

# %%
"""10-K Due Diligence Assistant - RAG pipeline with verifiable citations for SEC filings."""

import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")

import chromadb
import plotly.graph_objects as go
import torch

# LlamaIndex core + ChromaDB + OpenAI LLM backend
from llama_index.core import (
    Document,
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI as OpenAILLM
from llama_index.vector_stores.chroma import ChromaVectorStore

# ML4T configuration
from data import DataNotFoundError, iter_sec_filings
from utils.paths import get_output_dir
from utils.style import COLORS

VECTOR_STORE_DIR = get_output_dir(22, "10k_rag_assistant") / "vector_store"
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# %% tags=["parameters"]
MAX_DOCS = 20
TOP_K = 5
RUN_LIVE_LLM = False
REQUIRE_GPU = True

# %%
if REQUIRE_GPU and not torch.cuda.is_available():
    raise RuntimeError("This embedding workload requires the ml4t-gpu service with CUDA.")

EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_CACHE_DIR = os.environ.get("HF_HUB_CACHE")
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
OPENAI_AVAILABLE = RUN_LIVE_LLM and bool(os.getenv("OPENAI_API_KEY"))
if RUN_LIVE_LLM and not OPENAI_AVAILABLE:
    raise RuntimeError("RUN_LIVE_LLM=True requires OPENAI_API_KEY.")

print("LlamaIndex + ChromaDB available")
print(f"Embedding device: {EMBEDDING_DEVICE}")
print(f"Generation mode: {'live OpenAI' if OPENAI_AVAILABLE else 'retrieval only'}")
print()
print(f"Max documents: {MAX_DOCS}")
print(f"Similarity top_k: {TOP_K}")

# Data source - prefer the canonical SP100 10-K parquet corpus.
# A flat directory of PDF/TXT/MD files remains a fallback for ad hoc
# experiments without the staged dataset.
HAS_CANONICAL_CORPUS = True
try:
    _peek = next(iter_sec_filings(form_type="10-K", universe="sp100"), None)
    if _peek is None:
        HAS_CANONICAL_CORPUS = False
except DataNotFoundError:
    HAS_CANONICAL_CORPUS = False

SEC_FILINGS_DIR = get_output_dir(22, "sec_filing_pipeline") / "sec_filings_10k"
SEC_FILINGS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# **Interpretation**: The parameters shrink the corpus and retrieval breadth for
# quick tests, but the notebook still exercises the same RAG pipeline that the
# production chapter discussion describes.

# %% [markdown]
# ## 2. Document Ingestion
#
# The implemented path uses sentence-window chunking over canonical filing text.
# A production parser can additionally preserve sections and tables, but those
# properties are not measured by this notebook.


# %% [markdown]
# ### Load filings from disk
#
# Prefer the staged SP100 parquet corpus produced upstream. Flat PDF, TXT, and
# Markdown files remain supported for ad hoc experiments, but the notebook no
# longer fabricates a sample filing when the real corpus is missing.


# %%
def load_canonical_documents(max_docs: int = 10):
    """Load a stable first-N slice of canonical SP100 10-K filings."""
    documents = []
    records = [
        record
        for record in iter_sec_filings(form_type="10-K", universe="sp100")
        if record.get("text")
    ]
    records.sort(
        key=lambda record: (
            str(record["symbol"]),
            str(record["filing_date"]),
            str(record["accession_no"]),
        )
    )
    for record in records[:max_docs]:
        accession_no = record["accession_no"]
        metadata = {
            "symbol": record["symbol"],
            "cik": record.get("cik"),
            "year": record.get("year"),
            "filing_date": str(record["filing_date"]),
            "company_name": record.get("company_name"),
            "accession_no": accession_no,
        }
        documents.append(Document(text=record["text"], doc_id=accession_no, metadata=metadata))
    return documents


# %% [markdown]
# ### Load and route filing formats
#
# When the canonical SP100 corpus is available, stream 10-K records from it.
# Otherwise fall back to flat PDF/TXT/MD files from the notebook's scratch dir.


# %%
def load_documents(filings_dir: Path, max_docs: int = 10):
    """Load 10-K filings: canonical parquet corpus first, flat-file fallback otherwise."""
    if HAS_CANONICAL_CORPUS:
        print(f"Loading up to {max_docs} 10-K filings from canonical SP100 corpus")
        documents = load_canonical_documents(max_docs)
        print(f"Loaded {len(documents)} parquet-backed documents")
        return documents

    pdf_files = sorted(filings_dir.rglob("*.pdf"))
    txt_files = sorted(filings_dir.rglob("*.txt"))
    md_files = sorted(filings_dir.rglob("*.md"))
    all_files = (pdf_files + txt_files + md_files)[:max_docs]

    if not all_files:
        raise FileNotFoundError(
            "No staged 10-K filings found. Run 01_sec_filing_pipeline.py or point "
            f"SEC_FILINGS_DIR at a directory of PDF/TXT/MD filings under {filings_dir}."
        )

    print(f"Loading {len(all_files)} documents from {filings_dir}")
    reader = SimpleDirectoryReader(
        input_files=[str(f) for f in all_files],
        filename_as_id=True,
    )
    documents = reader.load_data()
    print(f"Loaded {len(documents)} document chunks")
    return documents


# %% [markdown]
# **Interpretation**: `load_documents` is the first grounding checkpoint. The
# notebook either streams the canonical SP100 parquet corpus or reads flat
# PDF/TXT/MD files from `SEC_FILINGS_DIR`. If neither path is populated, the
# loader raises `FileNotFoundError` rather than silently substituting synthetic
# text - there is no synthetic fallback in the current code path.
#
# %%
# Load documents
documents = load_documents(SEC_FILINGS_DIR, MAX_DOCS)

if documents:
    print("\nSample document preview (first 500 chars):")
    print("-" * 50)
    print(documents[0].text[:500])
    print("-" * 50)

# %% [markdown]
# **Interpretation**: The loaded filings preserve the key challenge of
# filing retrieval: long, heterogeneous documents need to retain enough structure
# for later citations and multi-hop questions to remain grounded.

# %% [markdown]
# ## 3. Chunking and Embedding
#
# We split documents into sentence windows and create embeddings using a local
# general-purpose model. Specialized financial candidates require a judged
# comparison before making a quality claim.
#
# Here we use the local `BAAI/bge-small-en-v1.5` model so the notebook runs
# without any API key; OpenAI `text-embedding-3-small` or voyage-finance-2 are
# drop-in alternatives when those API keys are available.


# %% [markdown]
# ### Configure LlamaIndex settings
#
# Sets the LLM, embedding model, and chunk strategy. ChromaDB provides
# persistent storage so the index survives notebook restarts.


# %% [markdown]
# ### Build the Retrieval Index
#
# This function configures chunking, embeddings, and optional ChromaDB
# persistence, then materializes the index used by the query engine.


# %%
def create_index(documents, persist_dir: Path = None):
    """Create vector index with SentenceSplitter chunking and optional ChromaDB persistence."""
    if not documents:
        print("Cannot create index - no documents")
        return None

    # Local embeddings (no API key required)
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        device=EMBEDDING_DEVICE,
        cache_folder=EMBEDDING_CACHE_DIR,
        revision=BGE_REVISION,
    )
    parameter_device = next(embed_model._model.parameters()).device
    if REQUIRE_GPU and parameter_device.type != "cuda":
        raise RuntimeError(f"Embedding parameters are on {parameter_device}, not CUDA.")
    print(f"Embedding parameters: {parameter_device}")
    Settings.embed_model = embed_model

    # LLM: OpenAI if available, otherwise retrieval-only
    if OPENAI_AVAILABLE:
        Settings.llm = OpenAILLM(model="gpt-4o-mini", temperature=0.1)
    else:
        Settings.llm = None

    Settings.node_parser = SentenceSplitter(
        chunk_size=512, chunk_overlap=50, paragraph_separator="\n\n"
    )
    return _build_vector_index(documents, persist_dir)


# %%
def _build_vector_index(documents, persist_dir: Path | None):
    """Build the configured in-memory or persistent vector index."""
    if persist_dir:
        print(f"Creating persistent ChromaDB index at {persist_dir}")
        chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        collection_name = "10k_filings"
        if collection_name in [item.name for item in chroma_client.list_collections()]:
            chroma_client.delete_collection(collection_name)
        chroma_collection = chroma_client.create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    else:
        print("Creating in-memory index")
        index = VectorStoreIndex.from_documents(documents)

    print(f"Index created with {len(documents)} documents")
    return index


# %% [markdown]
# **Interpretation**: Index creation is where chunking and embeddings become a
# concrete retrieval asset. If this step fails, every later answer should be
# read as an environment issue rather than a model-quality signal.
#
# %%
# Create or load index
index = create_index(documents, VECTOR_STORE_DIR)

# %% [markdown]
# ## 4. Query Engine with Citation Prompting
#
# The query engine is configured with constraint-based prompting to ensure:
# 1. Answers are grounded only in retrieved context
# 2. Citations reference specific pages/sections
# 3. Uncertainty is expressed when information is insufficient

# %%
# Constraint-based system prompt for citation generation
CITATION_PROMPT = """You are an expert financial analyst conducting due diligence on SEC 10-K filings.

CRITICAL INSTRUCTIONS:
1. Answer the question based ONLY on the information provided in the context below.
2. Do NOT use any outside knowledge or make assumptions beyond what is explicitly stated.
3. For each factual claim, include an inline citation referencing the source.
4. If the context does not contain sufficient information, state clearly: "The provided documents do not contain sufficient information to answer this question."
5. Structure your answer clearly with key points highlighted.

Context:
{context_str}

Question: {query_str}

Answer (with citations):"""


# %% [markdown]
# ### Create the Query Interface
#
# The query engine wraps the index with a constrained QA prompt so that
# retrieval and generation use the same citation-focused contract.


# %%
def create_query_engine(index, top_k: int = 5):
    """
    Create query engine with citation prompting.

    When no LLM is available, returns the retriever directly for
    retrieval-only mode (demonstrates indexing + search without generation).
    """
    if index is None:
        print("No index available - cannot create query engine")
        return None

    if not OPENAI_AVAILABLE:
        print(f"Retrieval-only mode (no LLM) - returning retriever with top_k={top_k}")
        return index.as_retriever(similarity_top_k=top_k)

    qa_prompt = PromptTemplate(CITATION_PROMPT)

    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        text_qa_template=qa_prompt,
        response_mode="compact",
    )

    print(f"Query engine created with top_k={top_k}")
    return query_engine


# %% [markdown]
# **Interpretation**: The query-engine contract is deliberately constrained. The
# result should be an answer interface that treats citations and abstention as
# part of the API, not as optional output formatting.
#
# %%
# Create query engine
query_engine = create_query_engine(index, TOP_K)

# %% [markdown]
# ## 5. Demonstration Queries
#
# We test the system with progressively complex queries:
# 1. **Simple extraction**: Single-fact lookup
# 2. **Multi-hop reasoning**: Synthesizing across sections
# 3. **Comparative analysis**: Connecting related concepts

# %%
# Test queries
TEST_QUERIES = [
    # Simple extraction
    "What was the revenue growth percentage in fiscal year 2023?",
    # Multi-section synthesis
    "Based on the MD&A and Risk Factors sections, what were the primary drivers of revenue growth, and what risks could threaten this growth?",
    # Specific detail extraction
    "What percentage of revenue comes from the top 10 customers?",
]


# %% [markdown]
# ### Execute constrained due-diligence queries
#
# Run one analyst question through the engine and return both the answer and the
# retrieved source snippets so we can inspect grounding quality.
#
# %% [markdown]
# #### Normalize retrieved nodes
#
# Keep source extraction identical for retrieval-only and generated responses.


# %%
def source_records(nodes) -> list[dict]:
    return [
        {
            "text_preview": node.text[:200] + "..." if len(node.text) > 200 else node.text,
            "score": getattr(node, "score", None),
            "metadata": node.metadata if hasattr(node, "metadata") else {},
        }
        for node in nodes
    ]


# %%
def run_query(engine, query: str) -> dict:
    """Execute a query without swallowing retrieval or generation failures."""
    if engine is None:
        return {
            "query": query,
            "answer": "Query engine not available - see setup instructions",
            "sources": [],
            "status": "no_engine",
        }

    # Retrieval-only mode (retriever object)
    if hasattr(engine, "retrieve") and not hasattr(engine, "query"):
        nodes = engine.retrieve(query)
        sources = source_records(nodes)
        answer = (
            f"[Retrieval-only] {len(nodes)} chunks retrieved. Top chunk: {nodes[0].text[:300]}..."
            if nodes
            else "[Retrieval-only] No matching chunks."
        )
        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "source_count": len(sources),
            "status": "success",
        }

    # Full query engine (with LLM)
    response = engine.query(query)
    sources = source_records(response.source_nodes) if hasattr(response, "source_nodes") else []

    return {
        "query": query,
        "answer": str(response),
        "sources": sources,
        "source_count": len(sources),
        "status": "success",
    }


# %%
# Run test queries
print("=== Running Test Queries ===\n")

query_results = []
for i, query in enumerate(TEST_QUERIES, 1):
    print(f"Query {i}: {query}")
    print("-" * 60)

    result = run_query(query_engine, query)
    query_results.append(result)

    print(f"Answer: {result['answer'][:500]}...")
    print(f"Sources: {result.get('source_count', 0)} chunks retrieved")
    print(f"Status: {result['status']}")
    print("\n")

# %% [markdown]
# **Interpretation**: Multi-hop queries (query 2) are the hardest for RAG
# systems because relevant evidence spans multiple filing sections.
# This run verifies retrieval mechanics. Citation and abstention behavior are
# scored only when `RUN_LIVE_LLM=True`; retrieval-only output is not an answer.

# %% [markdown]
# ## 6. Lightweight Harness Metrics
#
# Section 22.7 uses a broader evaluation harness. Here we compute lightweight
# operational checks directly from query outputs:
#
# - retrieval coverage (did we retrieve any source chunks),
# - citation presence in answers,
# - abstention behavior on unanswerable queries.


# %%
CITATION_PATTERN = re.compile(r"\[(?:c\d+|\d+)\]", re.IGNORECASE)


# %% [markdown]
# ### Score generated-answer behavior
#
# These checks apply only when a live LLM is explicitly enabled.


# %%
def answer_diagnostics(answered: list[dict]) -> tuple[float, float]:
    markers = ["do not contain sufficient information", "insufficient information", "cannot answer"]
    cited = sum(bool(CITATION_PATTERN.search(result.get("answer", ""))) for result in answered)
    abstentions = sum(
        any(marker in result.get("answer", "").lower() for marker in markers) for result in answered
    )
    denominator = max(len(answered), 1)
    return cited / denominator, abstentions / denominator


# %%
def evaluate_query_outputs(query_results: list[dict], llm_active: bool) -> dict:
    """Compute lightweight diagnostics from query outputs.

    Citation presence and abstention rate require an LLM-backed query
    engine (the retrieval-only mode does not generate answers). When
    `llm_active` is False those rates are reported as None to make the
    retrieval-only context explicit rather than silently scoring the
    `[Retrieval-only]` prefix as a citation."""
    if not query_results:
        return {
            "retrieval_coverage": 0.0,
            "citation_presence_rate": None,
            "abstention_rate": None,
            "n_queries": 0,
            "mode": "retrieval_only" if not llm_active else "full",
        }

    answered = [r for r in query_results if r.get("status") == "success"]
    retrieval_coverage = sum(1 for r in answered if r.get("source_count", 0) > 0) / max(
        len(answered), 1
    )

    if not llm_active:
        return {
            "retrieval_coverage": retrieval_coverage,
            "citation_presence_rate": None,
            "abstention_rate": None,
            "n_queries": len(query_results),
            "mode": "retrieval_only",
        }

    citation_rate, abstention_rate = answer_diagnostics(answered)
    return _full_mode_metrics(
        retrieval_coverage, citation_rate, abstention_rate, len(query_results)
    )


# %%
def _full_mode_metrics(
    retrieval_coverage: float, citation_rate: float, abstention_rate: float, n_queries: int
) -> dict:
    """Package diagnostics for LLM-backed query execution."""
    return {
        "retrieval_coverage": retrieval_coverage,
        "citation_presence_rate": citation_rate,
        "abstention_rate": abstention_rate,
        "n_queries": n_queries,
        "mode": "full",
    }


# %%
# Run lightweight diagnostics
print("=== Lightweight Query Diagnostics ===\n")

ragas_metrics = evaluate_query_outputs(query_results, llm_active=OPENAI_AVAILABLE)

print("Evaluation Metrics:")
for metric, value in ragas_metrics.items():
    if value is None:
        print(f"  {metric}: n/a (LLM disabled)")
    elif isinstance(value, float):
        print(f"  {metric}: {value:.2%}")
    else:
        print(f"  {metric}: {value}")

# %%
source_counts = [result.get("source_count", 0) for result in query_results]
fig = go.Figure(
    go.Bar(
        x=[f"Query {index}" for index in range(1, len(source_counts) + 1)],
        y=source_counts,
        marker_color=COLORS["blue"],
        text=source_counts,
        textposition="outside",
    )
)
fig.update_layout(
    title=f"All {len(source_counts)} due-diligence queries retrieve source chunks",
    xaxis_title="Analyst query",
    yaxis_title="Retrieved chunks (count)",
    height=400,
    showlegend=False,
)
fig.update_yaxes(range=[0, max(source_counts + [1]) * 1.2])
fig.show()

# %% [markdown]
# ## 7. Numeric Workflow: Retrieve → Extract → Compute → Narrate
#
# Numeric questions are where RAG most often fails silently: a fluent model
# will happily *state* a computed figure it never actually calculated. The
# safeguard is to keep arithmetic out of the language model entirely -
# retrieve the evidence, **extract** the reported figures into a typed schema,
# **compute** in Python, and **narrate** the result with a citation trace back
# to the source chunks. The block below runs deterministically (no LLM), so the
# extracted numbers and the computed aggregate are reproducible and auditable.

# %% [markdown]
# ### Typed schema and figure extractor
#
# `ExtractedFigure` is the typed schema each retrieved dollar amount is parsed
# into; the regex normalizes `$X billion/million/thousand` into a USD float so
# the computation step never re-parses free text.


# %%
@dataclass
class ExtractedFigure:
    raw: str
    value_usd: float
    source: str


_MONEY_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s?(billion|million|thousand)?", re.IGNORECASE)
_SCALE = {"billion": 1e9, "million": 1e6, "thousand": 1e3, "": 1.0}


# %% [markdown]
# ### Extraction step
#
# Walk `(text, source)` pairs - retrieved chunks or financial-statement
# excerpts alike - pull every `$`-denominated figure into the typed schema, and
# carry the source id so each value remains citable.


# %%
def extract_dollar_figures(items) -> list[ExtractedFigure]:
    """Pull $-denominated figures from (text, source) pairs into a typed schema."""
    figures: list[ExtractedFigure] = []
    for text, source in items:
        for match in _MONEY_RE.finditer(text):
            amount = float(match.group(1).replace(",", ""))
            scale = _SCALE[(match.group(2) or "").lower()]
            figures.append(
                ExtractedFigure(raw=match.group(0), value_usd=amount * scale, source=source)
            )
    return figures


# %% [markdown]
# ### What the narrative corpus contains
#
# The canonical SP100 10-K corpus loaded above stores the *narrative* sections
# (business description, risk factors), which carry almost no dollar figures -
# the financial-statement tables live in a separate exhibit a production system
# would ingest for numeric questions. We confirm that directly: retrieving a
# numeric question against the narrative index returns chunks with no
# `$`-denominated figures, so there is nothing to compute from them.

# %%
NUMERIC_QUESTION = (
    "What are the largest reported dollar figures in the filing's financial discussion?"
)

numeric_retriever = index.as_retriever(similarity_top_k=TOP_K) if index is not None else None
if numeric_retriever is not None:
    narrative_nodes = numeric_retriever.retrieve(NUMERIC_QUESTION)
    narrative_figures = extract_dollar_figures(
        (n.text, n.metadata.get("accession_no", n.metadata.get("symbol", "?")))
        for n in narrative_nodes
    )
    print(f"Question: {NUMERIC_QUESTION}")
    print(
        f"Retrieved {len(narrative_nodes)} narrative chunks; "
        f"extracted {len(narrative_figures)} dollar figures."
    )
    if not narrative_figures:
        print("Narrative sections carry no financial figures -> route numeric questions")
        print("to the financial-statement exhibit (illustrated below).")
    else:
        print(
            f"Found {len(narrative_figures)} figure(s) in narrative chunks; the "
            "financial-statement exhibit (below) remains the authoritative source."
        )
else:
    print("No index available - cannot run numeric workflow.")

# %% [markdown]
# ### Retrieve → Extract → Compute → Narrate on financial-statement evidence
#
# The controlled excerpts below stand in for the financial-statement section of a 10-K
# (the part the narrative corpus omits). They carry the real reported figures
# an analyst would query. The mechanism is the lesson: each retrieved excerpt
# is parsed into the typed `ExtractedFigure` schema, the year-over-year change
# is computed **in Python**, and the result is narrated with a citation to the
# source excerpt. They test arithmetic provenance, not retrieval accuracy.

# %%
# Controlled financial-statement fixtures for the arithmetic oracle.
FINANCIAL_EXCERPTS = [
    {"id": "fs_total_2023", "text": "Total net sales were $383.3 billion in fiscal 2023."},
    {"id": "fs_total_2022", "text": "Total net sales were $394.3 billion in fiscal 2022."},
    {"id": "fs_products_2023", "text": "Products net sales were $298.1 billion in fiscal 2023."},
    {"id": "fs_services_2023", "text": "Services net sales were $85.2 billion in fiscal 2023."},
]


figures = {
    f.source: f for f in extract_dollar_figures((e["text"], e["id"]) for e in FINANCIAL_EXCERPTS)
}

# Compute year-over-year total-net-sales growth in code.
total_2023 = figures["fs_total_2023"].value_usd
total_2022 = figures["fs_total_2022"].value_usd
yoy_growth = (total_2023 - total_2022) / total_2022

# Consistency check: products + services should reconcile to total net sales.
products_2023 = figures["fs_products_2023"].value_usd
services_2023 = figures["fs_services_2023"].value_usd
segment_sum = products_2023 + services_2023

print("Extract -> Compute trace (arithmetic done in Python, not by an LLM):")
for fid, fig in figures.items():
    print(f"  {fig.raw:>18}  = ${fig.value_usd:,.0f}   [source: {fid}]")
print(f"\n  YoY total net sales growth: {yoy_growth:.1%} (sources fs_total_2023, fs_total_2022)")
print(
    f"  Products + Services FY2023: ${segment_sum:,.0f} "
    f"vs reported total ${total_2023:,.0f} "
    f"(reconciles: {abs(segment_sum - total_2023) < 1e8})"
)
assert abs(segment_sum - total_2023) < 1e8
assert abs(yoy_growth - ((383.3 - 394.3) / 394.3)) < 1e-12

# %% [markdown]
# **Interpretation**: The arithmetic above is done by Python on values parsed
# into `ExtractedFigure`, and every figure carries its source id. A production
# assistant retrieves the financial-statement exhibit, extracts figures the same
# way, computes in code, and feeds the cited result back to the LLM only for
# narration - the model formats the answer but never performs the math, which is
# what keeps numeric RAG answers auditable.

# %% [markdown]
# ## 8. Summary
#
# | Component | Implementation |
# |-----------|---------------|
# | **Document Loading** | Canonical parquet records; sorted flat-file fallback |
# | **Document Parsing** | SentenceSplitter sentence windows |
# | **Chunking** | SentenceSplitter (512 tokens, 50 overlap) |
# | **Embeddings** | BGE-small-en (local); voyage-finance-2 or text-embedding-3 for production |
# | **Vector Store** | ChromaDB (isolated rebuild for each run) |
# | **Retrieval** | Top-k similarity search |
# | **Generation** | Retrieval-only default; GPT-4o-mini only when `RUN_LIVE_LLM=True` |
# | **Evaluation** | Lightweight harness diagnostics |
#
# **Interpretation**: The lightweight diagnostics summarize whether the assistant
# is retrieving evidence, citing it, and abstaining when needed. That is the
# minimum operating bar before moving to the broader Chapter 22 harness.

# %%
print("=== 10-K RAG Assistant Summary ===\n")

print(f"Documents processed: {len(documents) if documents else 0}")
print(f"Queries executed: {len(query_results)}")
print(f"Successful queries: {sum(1 for r in query_results if r['status'] == 'success')}")
print(f"Retrieval top_k: {TOP_K}")

if ragas_metrics:
    print("\nHarness Metrics:")
    for metric, value in ragas_metrics.items():
        if value is None:
            print(f"  {metric}: n/a (LLM disabled)")
        elif isinstance(value, float):
            print(f"  {metric}: {value:.2%}")
        else:
            print(f"  {metric}: {value}")

# %% [markdown]
# ## Key Takeaways
#
# **Interpretation**: The closing summary shows that this assistant is only as
# reliable as the ingestion, chunking, and citation contract underneath it. The
# result is a verifiable due-diligence workflow, not just a fluent QA demo.
#
# 1. **The implemented parser is deliberately bounded**: Sentence-window
#    chunking exercises retrieval mechanics but does not claim section or table
#    preservation.
#
# 2. **Constraint prompting defines a testable contract**: The prompt requires
#    citations and abstention. A separate evaluation run must verify compliance;
#    the prompt alone cannot guarantee it.
#
# 3. **Isolated indexes protect comparisons**: Rebuilding the Chroma collection
#    prevents stale or duplicated nodes from changing a candidate run.
#
# 4. **Lightweight diagnostics complement full evaluation**: The harness
#    metrics here (retrieval coverage, citation presence, abstention rate)
#    provide quick operational checks between full evaluation runs.
#
# **Next**: `06_esg_rag_vs_finetune` compares this RAG approach against
# fine-tuned classification for ESG analysis.
#
# **Book reference**: Section 22.8 discusses the 10-K assistant as a
# flagship RAG application and Section 22.6 covers constraint prompting.
