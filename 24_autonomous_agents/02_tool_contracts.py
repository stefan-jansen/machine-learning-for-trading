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
# An agent is only as good as what its tools return. In the forecasting system this chapter
# builds, search is the single tool, so its contract decides what evidence can reach the model,
# whether that evidence could have been read on the day being forecast, and whether a claim in
# the final rationale can be traced back to a document. This notebook writes that contract and
# the controls that sit on top of it.
#
# **Learning Objectives**:
# - Define a search tool behind a protocol so the provider can be replaced without touching
#   the agent
# - Filter results at a cutoff date so an agent scored on a resolved question cannot read the
#   answer, and separate the undated results the filter cannot rule on
# - Restrict which publishers an agent may read, and record in the execution log how much
#   evidence the policy removed
# - Render one tool definition into the Anthropic and OpenAI schema formats
# - Read an execution log to see what a run actually called, as distinct from what the model
#   said it was doing
#
# **Book Reference**: Chapter 24, Section 24.4 (Tool Integration: Contracts, Controls,
# and Context Engineering)
#
# **Prerequisites**: [`01_react_reasoning`](01_react_reasoning.ipynb) (LLM providers
# and the ReAct pattern).

# %%
import sys

from utils.paths import get_chapter_dir

sys.path.insert(0, str(get_chapter_dir(24)))

"""Tool Contracts and Provenance - search protocol, audit trails, and domain policy."""

import json
from datetime import date
from urllib.parse import urlparse

import polars as pl
from agent_fixtures import get_demo_question
from agent_schemas import SearchResult
from agent_tools import (
    DEFAULT_ALLOWED_DOMAINS,
    SEARCH_TOOL,
    ToolExecutor,
    create_search_client,
    format_search_results,
)

# %% tags=["parameters"]
SEARCH_PROVIDER = "mock"  # explicitly synthetic contract fixture; "tavily" for live retrieval
MAX_RESULTS = 5

# %% [markdown]
# ## The SearchClient Protocol
#
# The AIA Forecaster uses a single tool: **web search**. The `SearchClient` protocol
# abstracts the provider - whether Tavily (real web search), a mock (deterministic
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
# Every search result carries provenance metadata - title, URL, snippet, published
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
# This notebook selects `mock` explicitly for a synthetic tool-contract fixture.
# Live retrieval requires a configured provider; missing credentials raise an error.

# %%
search = create_search_client(SEARCH_PROVIDER)
print("Mode: SYNTHETIC contract fixture" if SEARCH_PROVIDER == "mock" else "Mode: LIVE retrieval")
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
# In this synthetic example, the client keeps only results with a parseable
# publication date before `cutoff_date`. This removes explicitly later or undated
# items. Publication dates do not establish historical page content or erase
# model foreknowledge, so the filter alone cannot make a backtest point-in-time accurate.

# %%
# Search WITHOUT cutoff - all results returned
all_results = search.search("NVIDIA Q4 earnings", max_results=MAX_RESULTS)
print(f"Without cutoff: {len(all_results)} results")
for r in all_results:
    print(f"  [{r.published or '?'}] {r.title}")

print()

# Search WITH cutoff - future results filtered out
filtered_results = search.search("NVIDIA Q4 earnings", max_results=MAX_RESULTS, cutoff_date=cutoff)
print(f"With cutoff ({cutoff}): {len(filtered_results)} results")
for r in filtered_results:
    print(f"  [{r.published or '?'}] {r.title}")

# %% [markdown]
# **Finding**: The synthetic filter excludes records outside its date bound.
# A historical simulation also needs evidence available at its forecast cutoff
# and an assessment of the model's prior knowledge. These fixture results test
# the filter, not historical forecasting skill.

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
# independently of the agent's reasoning trace - capturing what *actually* executed,
# with timing and provenance.

# %%
executor = ToolExecutor(search=search)

# Execute several searches
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
# **Observation**: These synthetic calls demonstrate the audit record, not
# provider latency. Inspect measured durations from a genuine live capture to
# identify slow queries; a fixture's timing cannot predict a live service.

# %% [markdown]
# ## Domain policy
#
# The tool executor filters returned URLs by hostname before providing them to the
# agent. This is an access constraint, not evidence that a retained source is true.
# The next local example is a labeled synthetic domain-filter fixture.

# %%
assert len(DEFAULT_ALLOWED_DOMAINS) == 11, "DEFAULT_ALLOWED_DOMAINS count drifted"
pl.DataFrame({"allowed_domain": sorted(DEFAULT_ALLOWED_DOMAINS)})

# %% [markdown]
# ### Custom domain policy - observable enforcement
#
# Policy enforcement here is **post-retrieval**: the search provider returns
# whatever matches the query, and the agent's tool layer filters results
# whose registered domain is not on the allowlist (or appears on a
# blocklist) before handing them back. The fixture below seeds a mock
# search whose results contain a `reddit.com` URL alongside a `wsj.com`
# URL; instantiate two policies and watch the blocked one disappear.

# %%
mixed_results = [
    SearchResult(
        title="Synthetic Fed reaction on rates",
        url="https://www.reuters.com/markets/fed-reaction",
        snippet="...",
        published="2025-03-04",
        score=0.91,
    ),
    SearchResult(
        title="Synthetic WSJ analysis: NVIDIA earnings setup",
        url="https://www.wsj.com/articles/nvidia",
        snippet="...",
        published="2025-03-05",
        score=0.88,
    ),
    SearchResult(
        title="Synthetic r/wallstreetbets: NVDA hot take",
        url="https://www.reddit.com/r/wallstreetbets/x",
        snippet="...",
        published="2025-03-05",
        score=0.40,
    ),
]


def apply_domain_policy(
    items: list[SearchResult],
    *,
    allowed: set[str] | None = None,
    blocked: set[str] | None = None,
) -> list[SearchResult]:
    """Drop results whose host is not on `allowed`, or is on `blocked`."""
    keep: list[SearchResult] = []
    for r in items:
        host = urlparse(r.url).netloc.removeprefix("www.")
        if blocked and host in blocked:
            continue
        if allowed and host not in allowed:
            continue
        keep.append(r)
    return keep


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
# OpenAI wraps in a `function` object), but getting it wrong causes silent failures.
# Provider-agnostic tool definitions prevent this class of bug.

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
# 1. **SearchClient protocol** abstracts the provider - same agent code works with
#    Tavily, mock, or any future search API
# 2. **Date filtering** excludes ineligible publication dates; historical content
#    and model foreknowledge remain separate limitations
# 3. **Domain policies** filter returned URL hostnames before results reach the
#    agent; they do not certify evidence quality or compliance
# 4. **Schema translation** renders tool definitions to both Anthropic and OpenAI
#    formats from a single source of truth
# 5. **Audit trails** log every search call with timing, status, and provenance
#
# **Next**: [`state_and_memory`](03_state_and_memory.ipynb) - explicit agent state, quality gates, and
# checkpoint/replay for reproducibility.
#
# **Book**: Section 24.4 covers tool contract design, MCP (Model Context Protocol),
# and sandboxing patterns.
