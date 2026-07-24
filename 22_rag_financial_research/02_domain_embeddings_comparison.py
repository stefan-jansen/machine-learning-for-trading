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
# # Domain-Specific Embeddings for Financial RAG
#
# **Docker image**: `ml4t-gpu`
#
# **Chapter 22: RAG for Financial Research** (Section 22.4)
#
# This notebook compares local embedding models for financial document retrieval:
#
# 1. **General open-weight embeddings** - BGE-large and MiniLM
# 2. **Optional managed candidates** - OpenAI and Voyage AI when explicitly enabled
#
# **Learning Objectives**:
# - Compare open-weight embedding models on the same financial corpus.
# - Measure how query type changes agreement with a lexical relevance proxy.
# - Use local evaluation rather than published benchmarks as the deployment decision rule.
#
# **Prerequisites**:
# - SP100 10-K filings (`data/equities/fundamentals/10k/sp100/`)
# - `sentence-transformers` for open-source models
# - OpenAI API key (`OPENAI_API_KEY`) for text-embedding-3 models (optional)
# - Voyage AI API key (`VOYAGE_API_KEY`) for voyage-finance-2 (optional)
#
# ## Evaluation Framing
# - Scores measure agreement with lexical proxy labels, not human relevance.
# - Rankings should be treated as corpus-specific, not universal.
# - FinE5 is discussed as a commercial candidate, not misrepresented as downloadable weights.
# - Query-type slices (technical/risk/general) often show different leading models.
# - Deployment decisions should combine retrieval quality, latency, and cost.

# %% [markdown]
# ## Setup and Imports
#
# The setup fixes query and document budgets so later model rankings reflect the
# embedding choice rather than changes in the evaluation workload.

# %%
"""Domain-Specific Embeddings - Compare embedding models for financial retrieval."""

import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
import torch

# Visualization
from IPython.display import Markdown, display
from plotly.subplots import make_subplots

# ML4T configuration
from data import load_sec_filings
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, ml4t_palette

# %% tags=["parameters"]
MAX_QUERIES = 0
MAX_DOCUMENTS = 0
USE_LOCAL_MODELS_ONLY = True
REQUIRE_GPU = True
SEED = 42
MODEL_REVISIONS = {
    "BAAI/bge-large-en-v1.5": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
}
MODEL_QUERY_PREFIXES = {
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "sentence-transformers/all-MiniLM-L6-v2": "",
}
EXPECTED_INPUT_SHA256 = "d0d8021167f79c49e0a5a4420294c9ff2f3534e4c9eb10f580c3e3901fb4945b"

# %%
set_global_seeds(SEED)

if REQUIRE_GPU and not torch.cuda.is_available():
    raise RuntimeError("This production benchmark requires a CUDA-capable GPU.")
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Embedding device: {EMBEDDING_DEVICE}")

# %%
# MAX_QUERIES is the cap; the actual N_QUERIES is set below after the
# FINANCIAL_QUERIES list is constructed so the print reflects what the
# evaluation actually runs.
N_DOCUMENTS = MAX_DOCUMENTS if MAX_DOCUMENTS > 0 else 200

print(f"MAX_QUERIES cap: {MAX_QUERIES if MAX_QUERIES > 0 else 'no cap (use all queries)'}")
print(f"Number of documents: {N_DOCUMENTS}")

# %% [markdown]
# ## 1. Financial Document Corpus
#
# We load real 10-K filing text from the SP100 corpus and chunk it into
# retrieval-ready passages. This gives the embedding comparison genuine
# financial language density rather than synthetic snippets.
#
# **Interpretation**: The initial query and document counts define the workload.
# That result matters because model differences may concentrate in the most
# technical slices of the corpus.


# %%
def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# %%
input_path = (
    Path(os.getenv("ML4T_DATA_PATH", "data"))
    / "equities/fundamentals/10k/sp100/reference/all_10k_filings.parquet"
)
INPUT_SHA256 = sha256_file(input_path)
if INPUT_SHA256 != EXPECTED_INPUT_SHA256:
    raise RuntimeError(f"Unexpected 10-K corpus identity: {INPUT_SHA256}")
print(f"Input SHA-256: {INPUT_SHA256}")

# %%
# Load real 10-K filing text from SP100 corpus via the canonical loader.
SAMPLE_SYMBOLS = ["AAPL", "MSFT", "AMZN", "JPM", "JNJ", "XOM", "BA", "ADBE"]
filings_df = (
    load_sec_filings(form_type="10-K", universe="sp100", symbols=SAMPLE_SYMBOLS)
    .sort(["filing_date", "symbol"], descending=[True, False])
    .unique(subset=["symbol"], keep="first")
    .sort("symbol")
)

chunks = []
for row in filings_df.iter_rows(named=True):
    symbol = row["symbol"]
    text = row["text"]
    if not text or len(text) < 200:
        continue
    # Chunk into ~300-char passages on sentence boundaries
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if len(s.strip()) > 40]
    for j in range(0, len(sentences) - 1, 2):
        chunk = ". ".join(sentences[j : j + 2]) + "."
        if 80 < len(chunk) < 600:
            chunks.append(f"[{symbol}] {chunk}")

print(f"Chunked {len(chunks)} passages from {len(SAMPLE_SYMBOLS)} symbols")

# Sample a manageable corpus
N_DOCS = min(N_DOCUMENTS, len(chunks))
rng = np.random.default_rng(SEED)
indices = rng.choice(len(chunks), size=N_DOCS, replace=False)
FINANCIAL_DOCUMENTS = [chunks[i] for i in sorted(indices)]
FINANCIAL_DOCUMENT_IDS = [
    hashlib.sha256(document.encode()).hexdigest() for document in FINANCIAL_DOCUMENTS
]
CORPUS_SHA256 = hashlib.sha256("\n".join(FINANCIAL_DOCUMENT_IDS).encode()).hexdigest()

print(f"Document corpus size: {len(FINANCIAL_DOCUMENTS)} documents")
print(f"Selected corpus SHA-256: {CORPUS_SHA256}")
assert FINANCIAL_DOCUMENTS, "The filing slice did not produce any retrieval documents."
assert len(set(FINANCIAL_DOCUMENT_IDS)) == len(FINANCIAL_DOCUMENT_IDS)

# %% [markdown]
# **Interpretation**: The corpus-size print confirms that every embedding model
# is being judged on the same financial document base before any metrics are
# compared.
#
# %%
# Queries grounded in real 10-K filing topics across our SP100 symbols
FINANCIAL_QUERIES = [
    # Technical queries (domain embeddings should excel)
    ("What competitive threats does the company face from low-cost competitors?", "technical"),
    ("How does the company protect its intellectual property?", "technical"),
    ("What are the company's key product development strategies?", "technical"),
    ("Describe the company's approach to hardware and software integration", "technical"),
    ("What regulatory compliance requirements affect operations?", "technical"),
    # Risk-focused queries
    ("What are the main risks to the company's supply chain?", "risk"),
    ("How does the company manage concentration risk in its customer base?", "risk"),
    ("What cybersecurity risks does the company disclose?", "risk"),
    ("How do currency fluctuations affect international operations?", "risk"),
    ("What legal proceedings or litigation risks exist?", "risk"),
    # General business queries (generic embeddings may suffice)
    ("How large is the company's workforce?", "general"),
    ("What markets does the company operate in?", "general"),
    ("How does the company distribute its products?", "general"),
    ("What is the company's competitive position?", "general"),
    ("What recent acquisitions or divestitures has the company made?", "general"),
]

if MAX_QUERIES > 0:
    FINANCIAL_QUERIES = FINANCIAL_QUERIES[:MAX_QUERIES]

N_QUERIES = len(FINANCIAL_QUERIES)

queries_df = pl.DataFrame(
    {"query": [q[0] for q in FINANCIAL_QUERIES], "query_type": [q[1] for q in FINANCIAL_QUERIES]}
)
QUERY_SHA256 = hashlib.sha256(json.dumps(FINANCIAL_QUERIES, sort_keys=True).encode()).hexdigest()

print(f"\nQuery set: {N_QUERIES} queries")
print(f"Query-set SHA-256: {QUERY_SHA256}")
queries_df.group_by("query_type").len()

# %% [markdown]
# **Interpretation**: The query mix spans technical, risk, and general language.
# That prevents a single average score from hiding where finance-specific
# embeddings actually change retrieval quality.

# %% [markdown]
# ## 2. Embedding Model Implementations
#
# We compare multiple embedding approaches:
#
# | Model Family | Example Candidate | Notes |
# |--------------|-------------------|-------|
# | General API  | text-embedding-*  | strong baseline, managed service |
# | Finance API  | finance-adapted API model | domain specialization, managed service |
# | Open-weight  | BGE / MiniLM / E5-style | local control and cost flexibility |

# %% [markdown]
# ### OpenAI Embeddings
#
# Managed API service providing high-quality general-purpose embeddings.
# Requires `OPENAI_API_KEY` environment variable and provides the generic
# baseline for interpreting whether finance specialization adds value.


# %%
def get_openai_embeddings(texts: list, model: str = "text-embedding-3-small") -> np.ndarray | None:
    """
    Get embeddings from OpenAI API.

    Returns None if API unavailable.
    """
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"  OpenAI API key not set - skipping {model}")
        return None

    try:
        from openai import OpenAI

        client = OpenAI()

        response = client.embeddings.create(input=texts, model=model)

        embeddings = np.array([item.embedding for item in response.data])
        print(f"  {model}: {embeddings.shape}")
        return embeddings

    except Exception as e:
        print(f"  {model} error: {e}")
        return None


# %% [markdown]
# ### Voyage Finance Embeddings
#
# Domain-specific embedding candidate for technical queries involving WACC,
# EBITDA, basis points, and other financial terminology.
#
# **Interpretation**: This is the notebook's main domain-adapted candidate. The
# result tells us whether specialized financial vocabulary is worth paying for.


# %%
def get_voyage_embeddings(texts: list, model: str = "voyage-finance-2") -> np.ndarray | None:
    """
    Get embeddings from Voyage AI API.

    voyage-finance-2 is specifically trained for financial text.
    """
    import os

    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print(f"  Voyage API key not set - skipping {model}")
        return None

    try:
        import voyageai

        client = voyageai.Client()
        result = client.embed(texts, model=model)

        embeddings = np.array(result.embeddings)
        print(f"  {model}: {embeddings.shape}")
        return embeddings

    except Exception as e:
        print(f"  {model} error: {e}")
        return None


# %% [markdown]
# ### Local Open-Weight Embeddings
#
# BGE-large and MiniLM are public, pinned open-weight controls. FinE5 is not a
# local candidate because its official repository publishes a model card but no
# runtime files. It is available through a separate managed service, so readers
# should not expect `from_pretrained()` to download it from Hugging Face.


# %%
def get_sentence_transformer_embeddings(
    texts: list[str], model_name: str, input_type: str
) -> np.ndarray:
    """
    Get embeddings from sentence-transformers (local, free).

    Local candidates are mandatory, so any model-load or encoding failure raises.
    """
    warnings.filterwarnings(
        "ignore",
        message="'return' in a 'finally' block",
        category=SyntaxWarning,
    )
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_name,
        revision=MODEL_REVISIONS[model_name],
        device=EMBEDDING_DEVICE,
        local_files_only=True,
    )
    model_device = next(model.parameters()).device
    if REQUIRE_GPU and model_device.type != "cuda":
        raise RuntimeError(f"{model_name} parameters are on {model_device}, not CUDA.")

    prefix = MODEL_QUERY_PREFIXES[model_name] if input_type == "query" else ""
    model_inputs = [f"{prefix}{text}" for text in texts]
    embeddings = model.encode(model_inputs, show_progress_bar=False, normalize_embeddings=True)

    print(f"  {model_name} ({input_type}): {embeddings.shape}; parameters={model_device}")
    return embeddings


# %% [markdown]
# ## 3. Generate Embeddings for Comparison
#
# We embed both documents and queries with each model to compute
# lexical-proxy agreement metrics.
#
# **Interpretation**: This stage turns dependency availability into a practical
# model menu. The result should be read as a deployment comparison, not only a
# pure algorithm benchmark.

# %%
print("=== Generating Document Embeddings ===\n")

document_embeddings = {}
query_embeddings = {}

# Default to two locally-runnable open-source models so the comparison
# always exercises a real retrieval workload. API-backed candidates are
# added only when the corresponding API keys are present.
MODELS: list = [
    (
        "bge-large",
        lambda texts, input_type: get_sentence_transformer_embeddings(
            texts, "BAAI/bge-large-en-v1.5", input_type
        ),
    ),
    (
        "minilm",
        lambda texts, input_type: get_sentence_transformer_embeddings(
            texts, "sentence-transformers/all-MiniLM-L6-v2", input_type
        ),
    ),
]

# %%
if not USE_LOCAL_MODELS_ONLY:
    if os.getenv("OPENAI_API_KEY"):
        MODELS.append(
            (
                "openai-small",
                lambda texts, _input_type: get_openai_embeddings(texts, "text-embedding-3-small"),
            )
        )
        MODELS.append(
            (
                "openai-large",
                lambda texts, _input_type: get_openai_embeddings(texts, "text-embedding-3-large"),
            )
        )
    if os.getenv("VOYAGE_API_KEY"):
        MODELS.append(
            (
                "voyage-finance",
                lambda texts, _input_type: get_voyage_embeddings(texts, "voyage-finance-2"),
            )
        )

# %%
print("Models in run:", [m[0] for m in MODELS])

# %% [markdown]
# ### Embed the corpus and query set
#
# Every candidate model sees the same documents and queries so the comparison
# isolates the embedding choice rather than corpus drift.
#
# **Interpretation**: The model list print shows which candidates survived the
# environment checks. That keeps later rankings honest about what was actually run.
#

# %%
for model_name, embed_fn in MODELS:
    print(f"Embedding with {model_name}...")

    # Embed documents
    doc_emb = embed_fn(FINANCIAL_DOCUMENTS, "document")
    if doc_emb is not None:
        document_embeddings[model_name] = doc_emb

    # Embed queries
    query_texts = [q[0] for q in FINANCIAL_QUERIES]
    query_emb = embed_fn(query_texts, "query")
    if query_emb is not None:
        query_embeddings[model_name] = query_emb

print(f"\nModels with embeddings: {list(document_embeddings.keys())}")
required_local_models = {"bge-large", "minilm"}
missing_local_models = required_local_models - document_embeddings.keys()
if missing_local_models:
    raise RuntimeError(f"Missing required local embedding results: {missing_local_models}")

# %% [markdown]
# **Interpretation**: Missing API keys only narrow the candidate set. The
# evaluation remains valid because every surviving model is still measured on the
# same corpus and query mix.

# %% [markdown]
# ## 4. Retrieval Quality Evaluation
#
# We measure retrieval quality using:
# - **Cosine similarity** between query and document embeddings
# - **Precision@k** for the top-k retrieved documents
# - **Mean Reciprocal Rank (MRR)** for ranking quality
#
# Since we do not have human relevance labels, we use a lexical proxy:
# documents containing query key terms are considered relevant.
# These metrics test agreement with that proxy and cannot establish deployment
# quality or a universal model ranking.

# %% [markdown]
# ### Cosine Similarity
#
# Standard similarity metric for comparing embedding vectors. We compute
# the full query-document similarity matrix for batch evaluation.


# %%
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between query and document embeddings."""
    # Normalize
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)

    # Compute similarity
    return np.dot(a_norm, b_norm.T)


# %% [markdown]
# ### Lexical-Proxy Labels
#
# Since we lack human-annotated relevance judgments, we generate proxy labels
# via term overlap. The three documents with the highest positive overlap are
# proxy-relevant. This deliberately favors literal term coverage, so the result
# is a diagnostic fixture rather than an independent relevance set.


# %%
def get_relevance_labels(query: str, documents: list[str], document_ids: list[str]) -> list[bool]:
    """
    Generate pseudo-relevance labels based on term overlap.

    A document is "relevant" if it contains key terms from the query.
    This is a proxy for true relevance in the absence of human labels.
    """
    # Extract key terms (simple approach - remove stopwords)
    stopwords = {
        "what",
        "is",
        "the",
        "how",
        "are",
        "does",
        "explain",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "has",
        "have",
        "been",
    }
    query_terms = [w.lower().strip("?.,") for w in query.split() if w.lower() not in stopwords]

    if len(documents) != len(document_ids):
        raise ValueError("documents and document_ids must have the same length")
    overlap = [sum(1 for term in query_terms if term in doc.lower()) for doc in documents]
    positive = [index for index, score in enumerate(overlap) if score > 0]
    ranked = sorted(positive, key=lambda index: (-overlap[index], document_ids[index]))[:3]
    return [index in ranked for index in range(len(documents))]


# %%
for query, _query_type in FINANCIAL_QUERIES:
    if not any(get_relevance_labels(query, FINANCIAL_DOCUMENTS, FINANCIAL_DOCUMENT_IDS)):
        raise ValueError(f"No lexical-proxy positive document for query: {query}")


# %% [markdown]
# ### Precision@k
#
# Measures the fraction of the top-k retrieved documents that match the lexical
# proxy. It is interpretable as fixture agreement, not human-rated relevance.


# %%
def stable_rank_indices(similarities: np.ndarray, document_ids: list[str]) -> list[int]:
    """Rank scores descending with immutable document identity as the tie-break."""
    return sorted(
        range(len(similarities)),
        key=lambda index: (-float(similarities[index]), document_ids[index]),
    )


# %% [markdown]
# Apply the stable ranking rule to Precision@k.


# %%
def compute_precision_at_k(
    similarities: np.ndarray, relevance: list, document_ids: list[str], k: int = 3
) -> float:
    """Compute Precision@k for a single query."""
    top_k_indices = stable_rank_indices(similarities, document_ids)[:k]
    top_k_relevant = sum(relevance[i] for i in top_k_indices)
    return top_k_relevant / k


# %% [markdown]
# ### Mean Reciprocal Rank (MRR)
#
# Measures how early the first proxy-positive document appears. MRR = 1 means
# the first result matches the lexical proxy for every query.


# %%
def compute_mrr(similarities: np.ndarray, relevance: list, document_ids: list[str]) -> float:
    """Compute Mean Reciprocal Rank for a single query."""
    ranked_indices = stable_rank_indices(similarities, document_ids)
    for rank, idx in enumerate(ranked_indices, 1):
        if relevance[idx]:
            return 1.0 / rank
    return 0.0


# %%
print("=== Computing Retrieval Metrics ===\n")

results = []

for model_name in document_embeddings:
    if model_name not in query_embeddings:
        continue

    doc_emb = document_embeddings[model_name]
    q_emb = query_embeddings[model_name]

    # Compute similarity matrix
    sim_matrix = cosine_similarity(q_emb, doc_emb)

    # Evaluate each query
    for i, (query, query_type) in enumerate(FINANCIAL_QUERIES):
        relevance = get_relevance_labels(query, FINANCIAL_DOCUMENTS, FINANCIAL_DOCUMENT_IDS)

        p_at_1 = compute_precision_at_k(sim_matrix[i], relevance, FINANCIAL_DOCUMENT_IDS, k=1)
        p_at_3 = compute_precision_at_k(sim_matrix[i], relevance, FINANCIAL_DOCUMENT_IDS, k=3)
        p_at_5 = compute_precision_at_k(sim_matrix[i], relevance, FINANCIAL_DOCUMENT_IDS, k=5)
        mrr = compute_mrr(sim_matrix[i], relevance, FINANCIAL_DOCUMENT_IDS)

        results.append(
            {
                "model": model_name,
                "query": query[:50] + "..." if len(query) > 50 else query,
                "query_type": query_type,
                "precision_at_1": p_at_1,
                "precision_at_3": p_at_3,
                "precision_at_5": p_at_5,
                "mrr": mrr,
                "n_relevant": sum(relevance),
            }
        )

results_df = pl.DataFrame(results)
print(f"Evaluated {len(results)} query-model combinations")

# %% [markdown]
# **Interpretation**: The retrieval loop applies Precision@k and MRR to a
# controlled lexical-proxy workload. Human judgments remain necessary before
# reading the scores as deployment quality.

# %% [markdown]
# ## 5. Results Analysis
#
# Compare models across query types to understand where their rankings agree
# with the lexical proxy. If no model produced embeddings the
# notebook fails loudly here rather than silently substituting fake metrics.

# %%
if len(results) == 0:
    raise RuntimeError(
        "No embedding model produced results. Install sentence-transformers "
        "or enable a configured managed embedding service and re-run."
    )

# %%
# Aggregate by model
model_summary = (
    results_df.group_by("model")
    .agg(
        pl.col("precision_at_1").mean().alias("avg_p@1"),
        pl.col("precision_at_3").mean().alias("avg_p@3"),
        pl.col("precision_at_5").mean().alias("avg_p@5"),
        pl.col("mrr").mean().alias("avg_mrr"),
    )
    .sort(["avg_mrr", "model"], descending=[True, False])
)

print("=== Model Comparison (Overall) ===\n")
model_summary

# %% [markdown]
# **Interpretation**: The overall ranking provides a starting point, but the
# aggregate hides important query-type differences. The by-type breakdown below
# reveals how much the diagnostic changes by query type.

# %%
# Aggregate by model and query type
type_summary = (
    results_df.group_by(["model", "query_type"])
    .agg(
        pl.col("precision_at_3").mean().alias("avg_p@3"),
        pl.col("mrr").mean().alias("avg_mrr"),
    )
    .sort(["query_type", "avg_mrr", "model"], descending=[False, True, False])
)

print("\n=== Model Comparison by Query Type ===\n")
type_summary

# %% [markdown]
# **Interpretation**: The query-type breakdown shows whether one aggregate hides
# different lexical-proxy behavior across the analyst workload.
#
# %% [markdown]
# ## 6. Visualization: Model Performance Comparison
#
# The chart below converts aggregate metrics into a workload view, showing when
# query slices change lexical-proxy agreement across candidate models.

# %%
# Create comparison visualization
if len(model_summary) > 0:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Precision@3 by Model", "MRR by Query Type"),
        horizontal_spacing=0.15,
    )

    # Bar chart: Overall P@3
    models = model_summary["model"].to_list()
    p_at_3 = model_summary["avg_p@3"].to_list()

    fig.add_trace(
        go.Bar(
            x=models,
            y=p_at_3,
            marker_color=COLORS["blue"],
            text=[f"{v:.1%}" for v in p_at_3],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

# %% [markdown]
# ### Add the query-type bars
#
# The grouped bars show where technical or risk-heavy queries change the
# relative ordering of candidate embedding models.
#

# %%
if len(model_summary) > 0:
    query_types = type_summary["query_type"].unique().to_list()
    query_colors = dict(zip(query_types, ml4t_palette(len(query_types)), strict=True))
    for qtype in query_types:
        subset = type_summary.filter(pl.col("query_type") == qtype)
        fig.add_trace(
            go.Bar(
                name=qtype,
                x=subset["model"].to_list(),
                y=subset["avg_mrr"].to_list(),
                text=[f"{v:.2f}" for v in subset["avg_mrr"].to_list()],
                textposition="outside",
                marker_color=query_colors[qtype],
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title=(
            f"{model_summary['model'][0]} leads on lexical-proxy MRR; query slices still disagree"
        ),
        height=440,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
        margin=dict(t=110, b=85),
    )

    fig.update_yaxes(title_text="Lexical-proxy Precision@3", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Lexical-proxy MRR", range=[0, 1], row=1, col=2)

    fig.show()

# %% [markdown]
# **Interpretation**: Read the chart as a workload comparison, not a global
# quality benchmark. A model that agrees most on technical queries may still be unnecessary
# if the production workload is mostly general-language news retrieval.

# %% [markdown]
# ## 7. Key Insights
#
# ### What changes between models
#
# A model's *aggregate* MRR can hide a meaningful split across query types.
# The query-type breakdown above is the part that drives a deployment
# decision: a model that leads overall but trails on the analyst's most
# common workload is the wrong default.
#
# ### When domain specialization is likely to matter most
#
# - Technical terminology such as WACC, basis points, and ASC 606 can expose
#   differences between generic and domain-adapted candidates.
# - Risk-factor language with consistent patterns (concentration risk,
#   counterparty exposure, CFTC/SEC compliance verbiage).
#
# ### When a generic open-source model is enough
#
# - General business queries with broad vocabulary.
# - Workloads dominated by short news-style passages.
# - Cost-sensitive deployments where a paid embedding API does not deliver a
#   material gain on human-rated relevance.

# %%
# Summary statistics
print("=== Domain Embedding Comparison Summary ===\n")

if len(model_summary) >= 2:
    best_model = model_summary.row(0, named=True)
    baseline_model = model_summary.row(-1, named=True)

    print(f"Highest lexical-proxy agreement: {best_model['model']}")
    print(f"  - Precision@3: {best_model['avg_p@3']:.1%}")
    print(f"  - MRR: {best_model['avg_mrr']:.3f}")

    print(f"\nComparison model: {baseline_model['model']}")
    print(f"  - Precision@3: {baseline_model['avg_p@3']:.1%}")
    print(f"  - MRR: {baseline_model['avg_mrr']:.3f}")

    relative_p3_improvement_pct = (
        (best_model["avg_p@3"] - baseline_model["avg_p@3"])
        / max(baseline_model["avg_p@3"], 0.01)
        * 100
    )
    relative_mrr_improvement_pct = (
        (best_model["avg_mrr"] - baseline_model["avg_mrr"])
        / max(baseline_model["avg_mrr"], 0.01)
        * 100
    )
    print(f"\nRelative Precision@3 improvement: {relative_p3_improvement_pct:.1f}%")
    print(f"Relative MRR improvement:         {relative_mrr_improvement_pct:.1f}%")
else:
    print("Insufficient models for comparison")
    print(f"Models available: {list(document_embeddings.keys())}")

# %% [markdown]
# **Interpretation**: The summary statistics compress the diagnostic comparison.
# They show lexical-proxy agreement, not the value of a more expensive stack.
#
# %%
# External benchmark context (qualitative)
print("\n=== External Benchmark Context ===")
print(
    """
External finance benchmarks are useful for candidate selection, but local
benchmarking on your own filing corpus and query set should determine the
final model choice.
"""
)

# %% [markdown]
# ## 8. Practical Recommendations
#
# ### Model Selection Guide
#
# Use a repeatable ranking process rather than fixed recommendations:
# 1. Evaluate all candidates on the same corpus snapshot.
# 2. Report Recall@k, MRR, latency, and cost.
# 3. Choose defaults by deployment constraints, not by single benchmark rank.
#
# ### Quantization Considerations
#
# Many embedding APIs support reduced dimensions; quantify the recall/latency
# trade-off on this workload before adopting compression settings.
#
# **Interpretation**: The final summary should drive a deployment rule, not a
# universal best model. The result suggests choosing the cheapest model that still
# clears the query types analysts actually submit.

# %% [markdown]
# ## Key Takeaways
#
# The exact findings below are generated from this run so they cannot drift
# from the executed output.

# %%
slice_winners = (
    type_summary.sort(["query_type", "avg_mrr", "model"], descending=[False, True, False])
    .group_by("query_type", maintain_order=True)
    .first()
)
winner_text = ", ".join(
    f"{row['query_type']}: {row['model']} ({row['avg_mrr']:.2f})"
    for row in slice_winners.iter_rows(named=True)
)
display(
    Markdown(
        f"""
1. **Aggregate lexical-proxy agreement differs by model.** `{best_model["model"]}`
   leads this run with Precision@3 of {best_model["avg_p@3"]:.1%} and MRR of
   {best_model["avg_mrr"]:.3f}, versus {baseline_model["avg_p@3"]:.1%} and
   {baseline_model["avg_mrr"]:.3f} for `{baseline_model["model"]}`.
2. **Query slices do not share one winner.** The highest lexical-proxy MRR by
   slice is {winner_text}. Workload composition therefore matters.
3. **The local comparison is reproducible.** Pinned BGE-large and MiniLM
   revisions run on the same corpus and GPU without paid API calls.
4. **Domain-adapted APIs are optional candidates.** Voyage and OpenAI models
   activate only when local-only mode is disabled and credentials are present.
5. **Human relevance judgments remain the deployment gate.** These scores use
   lexical proxy labels and exercise the evaluation pipeline; they do not
   certify which model is best for analysts.
"""
    )
)

# %%
completion_record = {
    "input_sha256": INPUT_SHA256,
    "selected_corpus_sha256": CORPUS_SHA256,
    "query_sha256": QUERY_SHA256,
    "documents": len(FINANCIAL_DOCUMENTS),
    "queries": len(FINANCIAL_QUERIES),
    "models": sorted(document_embeddings),
    "model_revisions": MODEL_REVISIONS,
    "device": EMBEDDING_DEVICE,
    "results": model_summary.to_dicts(),
}
print(f"COMPLETION_RECORD={json.dumps(completion_record, sort_keys=True)}")

# %% [markdown]
#
# **Next**: See [`03_hybrid_retrieval`](03_hybrid_retrieval.ipynb) for combining embeddings with BM25
# keyword search via Reciprocal Rank Fusion and evaluating the combined ranking.
#
# **Book Reference**: Section 22.4 discusses domain embeddings, the FinMTEB
# benchmark, and Matryoshka embedding compression.
