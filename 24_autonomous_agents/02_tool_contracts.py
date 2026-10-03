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
"""Tool Contracts and Provenance - search protocol, audit trails, and domain policy."""

import sys

from utils.paths import get_chapter_dir

sys.path.insert(0, str(get_chapter_dir(24)))

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
# The agent built in this chapter has one tool: web search. Every provider that can serve it,
# whether Tavily, a deterministic mock, or something added later, is reached through one method
# signature:
#
# ```python
# class SearchClient(Protocol):
#     def search(
#         self, query: str, max_results: int = 5, cutoff_date: date | None = None
#     ) -> list[SearchResult]: ...
# ```
#
# Two things in that signature are worth more than the abstraction itself. It returns typed
# `SearchResult` objects rather than a provider's raw JSON, so nothing downstream depends on a
# vendor's field names. And `cutoff_date` is in the contract rather than bolted on afterwards,
# which is what makes point-in-time filtering the tool's job instead of every caller's.

# %% [markdown]
# ### What a result carries
#
# A `SearchResult` records where the evidence came from as well as what it says. **Provenance**
# is that origin record: the URL identifies the publisher and lets a claim be traced back to
# the page it came from, and `published` is the date the filtering below depends on. `score` is
# the provider's own relevance ranking, which orders results and says nothing about whether
# they are true.

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
# ## Running a Search
#
# `SEARCH_PROVIDER` decides where results come from, and it is named rather than inferred.
# Left at `mock` it builds `MockSearchClient`, whose results are a fixed dictionary in
# `agent_tools.py`, so this notebook produces the same output on every machine and costs
# nothing. Name a real provider, `tavily` for this chapter, and it needs that provider's key;
# a missing one raises instead of quietly falling back to the fixture, which is what keeps a
# synthetic result from being read as a retrieved one.
#
# The demonstration question asks whether NVIDIA beat its Q4 FY2025 earnings expectations, which
# the company answered when it reported on 2025-02-26. The question carries a **cutoff date** of
# 2025-02-20: the boundary the search tool enforces, dropping anything published on or after it,
# so an agent answering the question reads only what existed before that day.

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
# The question above asks about an event that has already resolved, which is the only kind of
# question a forecasting agent can be scored on. It is also the kind that is trivially easy to
# get right by accident: search the open web today for "did NVIDIA beat Q4 FY2025 earnings" and
# the first result answers it. An agent evaluated that way is measuring the search index, not
# forecasting.
#
# `cutoff_date` is the guard. The client drops any result whose publication date is on or after
# the cutoff, so what reaches the agent is dated strictly earlier than that day. The question's
# own `cutoff_date` is 2025-02-20 and NVIDIA reported on 2025-02-26, so the agent works from
# what was published up to and including 2025-02-19 and the answer arrives a week later.
#
# The filter keeps a result only when it carries a publication date that parses and falls
# before the cutoff, so an undated item is dropped rather than trusted. What it cannot do is
# establish what a page said on the date it claims, or remove what the model already knows from
# training. A cutoff makes the retrieval point-in-time; it does not make the forecast one.

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
# ## Formatting Results for the Agent
#
# Results reach the model as text in a tool message, so the formatting is part of the contract
# too. `format_search_results` numbers each result and puts the title, URL, snippet and
# publication date on their own lines. The URL and the date are there so the model can weigh a
# primary source against a blog and recent reporting against stale reporting, and so that a
# quote in the final rationale can be traced back to the document it came from.

# %% [markdown]
# **Finding**: The synthetic filter excludes records outside its date bound.
# A historical simulation also needs evidence available at its forecast cutoff
# and an assessment of the model's prior knowledge. These fixture results test
# the filter, not historical forecasting skill.

# %%
formatted = format_search_results(filtered_results)
print(formatted[:500])

# %% [markdown]
# ## Execution Audit Trail
#
# `ToolExecutor` wraps the search client and records every call: the query, whether it
# succeeded, how long it took, and which client served it. That record is kept apart from the
# agent's reasoning trace on purpose. The reasoning trace is what the model said it was doing;
# the execution log is what happened. Debugging a bad forecast usually starts with the
# difference between them.

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
# The elapsed times round to zero because the mock client returns from a dictionary in memory.
# Against a live provider the same column carries the network round trip, which is where an
# agent spends most of its wall-clock time and where a per-call timeout has to be set.

# %% [markdown]
# ## Source Policy
#
# Which publishers an agent may read is a research decision, not a prompt detail. An allowlist
# names the domains whose reporting counts as evidence; a blocklist names the ones that do not.
# `apply_domain_policy` in `agent_tools.py` implements both, matching on the registered domain
# and its subdomains after stripping a leading `www.`, so `reuters.com` also covers
# `uk.reuters.com`. The blocklist is applied first, then the allowlist.
#
# `DEFAULT_ALLOWED_DOMAINS` is the chapter's starting set: wire services, financial newspapers,
# and the primary sources whose numbers the others report on.

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
# OpenAI wraps in a `function` object). Provider-agnostic tool definitions prevent
# provider-specific schema mismatches.

# %% [markdown]
# ## When the Tool Is Unavailable
#
# An agent deployed inside a network with no outbound access, or run while a provider is down,
# still has to do something. `ToolExecutor` treats an absent client as a logged outcome rather
# than an exception: the call is recorded with status `disabled` and returns no results, so the
# agent's loop continues and the record shows why the evidence is missing. An exception here
# would abort the run and leave nothing to audit.

# %%
disabled_executor = ToolExecutor(search=None)
empty = disabled_executor.execute_search("test query")
print(f"Results when disabled: {len(empty)}")
print(f"Log entry: {disabled_executor.execution_log[0].status}")

# %% [markdown]
# ## Key Takeaways
#
# 1. **The tool contract decides what the agent can know.** Everything downstream, the forecast
#    included, is a function of what search returned, so the contract is worth as much design
#    attention as the prompt.
# 2. **A publication-date filter is what separates a backtest from a demonstration.** Without
#    it an agent evaluated on a resolved question reads the answer.
# 3. **A result with no date is not a result before the cutoff.** Keep the two apart in the
#    record and decide the policy explicitly; silently counting undated evidence as
#    point-in-time safe is how a hindsight-free claim goes wrong.
# 4. **Enforce source policy in the tool layer rather than the prompt.** A model asked not to read
#    social media sometimes reads it anyway; a host that never reaches the model cannot.
# 5. **The execution log is the record of what ran**, separate from the model's account of what
#    it was doing. When those two disagree, the log is the one to trust.
# 6. **One tool definition, rendered per provider.** The schema differences between Anthropic
#    and OpenAI are mechanical, and a single `ToolDefinition` keeps them from becoming two
#    sources of truth that drift.
#
# **Known limitations of what is built here.** Source policy runs after retrieval, so a blocked
# document was still fetched and paid for; a provider-side domain filter is cheaper where the
# API offers one. The allowlist is a list of publishers, which is a crude proxy for evidence
# quality: a syndicated wire story on an allowed domain and the same story on a blocked one are
# the same evidence. Nothing here deduplicates results that repeat one underlying source, so an
# agent can mistake five copies of one story for five independent confirmations.
#
# **Next**: [`03_state_and_memory`](03_state_and_memory.ipynb) makes the agent's state explicit
# so a run can be checkpointed, replayed, and gated on evidence quality.
#
# **Book**: Section 24.4 covers tool contract design, the Model Context Protocol, and
# sandboxing patterns.
