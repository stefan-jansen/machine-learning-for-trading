# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
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
# # Tool Contracts and Provenance
#
# **Docker image**: `ml4t`
#
# This notebook teaches the **SearchClient protocol**, real web search integration
# via Tavily, provenance enrichment, and domain policy enforcement. In the AIA
# Forecaster, search is the agent's only tool, so its contract determines what
# evidence the agent can access and how trustworthy its outputs are.
#
# **Learning Objectives**:
# - Understand the `SearchClient` protocol that abstracts search providers
# - Execute web searches and inspect `SearchResult` objects
# - Filter dated search results at a cutoff and identify undated evidence
# - Configure domain allowlists and blocklists for source control
# - Translate tool schemas to Anthropic and OpenAI formats
# - Inspect the execution audit trail for debugging and compliance
#
# **Book Reference**: Chapter 24, Section 24.4 (Tool Integration: Contracts, Controls,
# and Context Engineering)
#
# **Prerequisites**: [`01_react_reasoning`](01_react_reasoning.ipynb) (LLM providers
# and the ReAct pattern).

# %%
"""Tool Contracts and Provenance: search protocol, audit trails, and domain policy."""

import json
import warnings
from datetime import date
from urllib.parse import urlparse

warnings.filterwarnings("ignore")

import polars as pl
from agent_fixtures import get_demo_question
from agent_schemas import SearchResult
from agent_tools import (
    DEFAULT_ALLOWED_DOMAINS,
    SEARCH_TOOL,
    MockSearchClient,
    ToolExecutor,
    create_search_client,
    format_search_results,
)

# %% tags=["parameters"]
RUN_LIVE = False
SEARCH_PROVIDER = ""  # live path only; empty selects an available provider
MAX_RESULTS = 5

# %% [markdown]
# ## The SearchClient Protocol
#
# The AIA Forecaster uses a single tool: **web search**. The `SearchClient` protocol
# abstracts the provider, whether Tavily (real web search), a mock (deterministic
# for CI), or a future provider like Brave or SerpAPI.
#
# ```python
# class SearchClient(Protocol):
#     def search(
#         self, query: str, max_results: int = 5, cutoff_date: date | None = None
#     ) -> list[SearchResult]: ...
# ```
#
# The key design: a single method that returns structured `SearchResult` objects,
# with optional point-in-time filtering via `cutoff_date`.

# %% [markdown]
# ### SearchResult structure
#
# Every search result carries provenance metadata: title, URL, snippet, published
# date, and relevance score. This enables downstream quality checks and audit trails.

# %%
example = SearchResult(
    title="Fed officials signal patience on rate cuts",
    url="https://federalreserve.gov/press-releases",
    snippet="Multiple FOMC members indicate no urgency to cut rates.",
    published="2025-03-05",
    score=0.96,
)
print(f"Title:     {example.title}")
print(f"URL:       {example.url}")
print(f"Published: {example.published}")
print(f"Score:     {example.score}")

# %% [markdown]
# ## Search Execution
#
# The default path forces the deterministic mock client. Set `RUN_LIVE=True` to
# opt into provider auto-detection and possible Tavily calls.

# %%
search = create_search_client(SEARCH_PROVIDER) if RUN_LIVE else MockSearchClient()
if not RUN_LIVE:
    assert isinstance(search, MockSearchClient)
print(f"Provider: {type(search).__name__}\n")

question = get_demo_question()
cutoff = date.fromisoformat(question.cutoff_date)
print(f"Question: {question.question}")
print(f"Cutoff:   {cutoff}\n")

results = search.search(
    "NVIDIA Q4 earnings expectations", max_results=MAX_RESULTS, cutoff_date=cutoff
)
print(f"Results: {len(results)}")
for r in results:
    print(f"  [{r.published or '?'}] {r.title}")
    if r.snippet:
        print(f"    {r.snippet[:100]}")

# %% [markdown]
# ## Point-in-Time Filtering
#
# When `cutoff_date` is provided, the search client filters out results published
# on or after that date. This prevents **lookahead bias** when publication metadata
# is available: the agent only sees dated information from before the cutoff.
#
# Without this, a backtesting agent could "cheat" by reading post-resolution news.

# %%
# Search WITHOUT cutoff: all results returned
all_results = search.search("NVIDIA Q4 earnings", max_results=MAX_RESULTS)
print(f"Without cutoff: {len(all_results)} results")
for r in all_results:
    print(f"  [{r.published or '?'}] {r.title}")

print()

# Use an earlier synthetic cutoff so the fixture visibly exercises the filter.
filter_demo_cutoff = date(2025, 2, 19)
filtered_results = search.search(
    "NVIDIA Q4 earnings",
    max_results=MAX_RESULTS,
    cutoff_date=filter_demo_cutoff,
)
print(f"With demonstration cutoff ({filter_demo_cutoff}): {len(filtered_results)} results")
for r in filtered_results:
    print(f"  [{r.published or '?'}] {r.title}")

# %% [markdown]
# **Finding**: The cutoff filter removes results with known publication dates on or
# after the cutoff. Undated evidence needs a separate validation policy before a
# historical evaluation can call it point-in-time safe.

# %% [markdown]
# ### Undated evidence remains unverified
#
# Search APIs do not always return a publication date. The tool contract retains such
# results because it cannot prove they are post-cutoff. A strict historical evaluation
# should partition them explicitly rather than count them as verified evidence.

# %%
undated = SearchResult(
    title="Undated market commentary",
    url="https://www.reuters.com/markets/undated-commentary",
    snippet="A result without provider publication metadata.",
    published=None,
    score=0.75,
)
cutoff_candidates = [*filtered_results, undated]
verified_pre_cutoff = [
    result
    for result in cutoff_candidates
    if result.published and date.fromisoformat(result.published) < filter_demo_cutoff
]
unverified_dates = [result for result in cutoff_candidates if not result.published]

print(f"Verified pre-cutoff: {len(verified_pre_cutoff)}")
print(f"Unverified dates:    {len(unverified_dates)}")
assert len(unverified_dates) == 1

# %% [markdown]
# ## Formatting Results for the Agent
#
# Search results are formatted as structured text before being fed back to the LLM
# as a tool message. This format includes title, URL, snippet, and published date.

# %%
formatted = format_search_results(filtered_results)
print(formatted[:500])

# %% [markdown]
# ## Execution Audit Trail
#
# The `ToolExecutor` wraps the search client with logging. Every call is recorded
# independently of the agent's reasoning trace, capturing what *actually* executed,
# with timing and provenance.

# %%
executor = ToolExecutor(search=search)

r1 = executor.execute_search("NVIDIA earnings Q4 2025", max_results=3, cutoff_date=cutoff)
r2 = executor.execute_search("semiconductor demand AI servers", max_results=3, cutoff_date=cutoff)
r3 = executor.execute_search("NVIDIA valuation risk", max_results=3, cutoff_date=cutoff)

print(f"Searches executed: {len(executor.execution_log)}\n")

pl.DataFrame(
    {
        "query": [entry.args.get("query", "?")[:38] for entry in executor.execution_log],
        "status": [entry.status for entry in executor.execution_log],
        "duration_ms": [round(entry.duration_ms, 1) for entry in executor.execution_log],
        "result_preview": [entry.result_preview for entry in executor.execution_log],
    }
)

# %% [markdown]
# **Observation**: The audit records status and elapsed time for every call. Live
# latency depends on the provider and network, so the trace supports measurement
# without embedding a machine-specific timing claim.

# %% [markdown]
# ## Domain Policy Enforcement
#
# The Tavily search client supports domain allowlists and blocklists. This controls
# which sources the agent can access, which supports evidence-quality controls.

# %%
assert len(DEFAULT_ALLOWED_DOMAINS) == 11, "DEFAULT_ALLOWED_DOMAINS count drifted"
pl.DataFrame({"allowed_domain": sorted(DEFAULT_ALLOWED_DOMAINS)})

# %% [markdown]
# ### Domain-policy function
#
# A small policy function applies host-domain allowlists and blocklists, including
# subdomains.


# %%
def apply_domain_policy(
    items: list[SearchResult],
    *,
    allowed: set[str] | None = None,
    blocked: set[str] | None = None,
) -> list[SearchResult]:
    """Drop results whose host is outside `allowed`, or within `blocked`."""

    def matches(host: str, domain: str) -> bool:
        domain = domain.lower().lstrip(".")
        return host == domain or host.endswith(f".{domain}")

    keep: list[SearchResult] = []
    for result in items:
        host = (urlparse(result.url).hostname or "").lower().removeprefix("www.")
        if blocked and any(matches(host, domain) for domain in blocked):
            continue
        if allowed and not any(matches(host, domain) for domain in allowed):
            continue
        keep.append(result)
    return keep


# %% [markdown]
# ### Observable enforcement
#
# Policy enforcement here is **post-retrieval**: the search provider returns
# whatever matches the query, and the agent's tool layer filters results
# whose host is not on the allowlist or belongs to a blocked domain before
# handing them back. The fixture below seeds a mock
# search whose results contain a `reddit.com` URL alongside a `wsj.com`
# URL; instantiate two policies and watch the blocked one disappear.

# %%
mixed_results = [
    SearchResult(
        title="Fed reaction on rates",
        url="https://www.reuters.com/markets/fed-reaction",
        snippet="...",
        published="2025-03-04",
        score=0.91,
    ),
    SearchResult(
        title="WSJ analysis: NVIDIA earnings setup",
        url="https://www.wsj.com/articles/nvidia",
        snippet="...",
        published="2025-03-05",
        score=0.88,
    ),
    SearchResult(
        title="r/wallstreetbets: NVDA hot take",
        url="https://www.reddit.com/r/wallstreetbets/x",
        snippet="...",
        published="2025-03-05",
        score=0.40,
    ),
]


financial_only = apply_domain_policy(
    mixed_results, allowed={"reuters.com", "wsj.com", "ft.com", "bloomberg.com"}
)
no_social = apply_domain_policy(mixed_results, blocked={"reddit.com", "twitter.com"})

pl.DataFrame(
    {
        "policy": ["raw retrieval", "allowlist (financial)", "blocklist (social)"],
        "n_results": [len(mixed_results), len(financial_only), len(no_social)],
        "hosts_kept": [
            ", ".join(urlparse(r.url).netloc for r in mixed_results),
            ", ".join(urlparse(r.url).netloc for r in financial_only),
            ", ".join(urlparse(r.url).netloc for r in no_social),
        ],
    }
)

# %% [markdown]
# ## Tool Schema Translation
#
# The `ToolDefinition` class renders the search tool schema to both Anthropic and
# OpenAI formats. This is how the LLM knows what tools are available and how to
# invoke them. Same definition, different provider formats.

# %%
print(f"Tool: {SEARCH_TOOL.name}")
print(f"Description: {SEARCH_TOOL.description}")
print(f"Parameters: {json.dumps(SEARCH_TOOL.parameters, indent=2)}")
print(f"Required: {SEARCH_TOOL.required}")

# %% [markdown]
# ### Anthropic format
#
# Anthropic uses `input_schema` at the top level.

# %%
print(json.dumps(SEARCH_TOOL.to_anthropic_schema(), indent=2))

# %% [markdown]
# ### OpenAI format
#
# OpenAI wraps the schema inside a `function` object with a `type: "function"` envelope.

# %%
print(json.dumps(SEARCH_TOOL.to_openai_schema(), indent=2))

# %% [markdown]
# **Observation**: The structural difference is minor (Anthropic uses `input_schema`,
# OpenAI wraps in a `function` object). Provider-agnostic tool definitions prevent
# provider-specific schema mismatches.

# %% [markdown]
# ## Disabled Search (Graceful Degradation)
#
# When search is disabled (e.g., in an air-gapped environment), the `ToolExecutor`
# logs the attempt with status "disabled" and returns an empty result list.

# %%
disabled_executor = ToolExecutor(search=None)
empty = disabled_executor.execute_search("test query")
print(f"Results when disabled: {len(empty)}")
print(f"Log entry: {disabled_executor.execution_log[0].status}")

# %% [markdown]
# ## Key Takeaways
#
# 1. **SearchClient protocol** abstracts the provider, so the same agent code works with
#    Tavily, mock, or any future search API
# 2. **Point-in-time filtering** removes results with known dates on or after the
#    cutoff; undated results remain unverified and need an explicit policy
# 3. **Domain policies** restrict which sources the agent can access, supporting
#    evidence-quality and compliance controls
# 4. **Schema translation** renders tool definitions to both Anthropic and OpenAI
#    formats from a single source of truth
# 5. **Audit trails** log every search call with timing, status, and provenance
#
# **Next**: [`03_state_and_memory`](03_state_and_memory.ipynb), which covers explicit
# agent state, quality gates, and checkpoint/replay for reproducibility.
#
# **Book**: Section 24.4 covers tool contract design, MCP (Model Context Protocol),
# and sandboxing patterns.
