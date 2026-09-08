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
"""Tool Contracts and Provenance: search protocol, audit trails, and domain policy."""

import json
from datetime import date
from urllib.parse import urlparse

import polars as pl
from agent_fixtures import get_demo_question
from agent_schemas import SearchResult
from agent_tools import (
    DEFAULT_ALLOWED_DOMAINS,
    SEARCH_TOOL,
    MockSearchClient,
    ToolExecutor,
    apply_domain_policy,
    create_search_client,
    format_search_results,
)
from IPython.display import display

# %% [markdown]
# ## Settings
#
# `RUN_LIVE` left at `False` uses the deterministic mock search client, so the notebook runs
# offline and prints the same results everywhere. Set it to `True`, with `TAVILY_API_KEY` set,
# to issue real queries.
#
# `SEARCH_PROVIDER` is empty so the factory picks whichever provider has a key; it is read only
# on the live path.
#
# `MAX_RESULTS` caps how many documents one search returns. Five is the working default for the
# agents in this chapter: enough for a claim to appear in more than one place, few enough that
# a handful of searches still fits in a context window.

# %% tags=["parameters"]
RUN_LIVE = False
SEARCH_PROVIDER = ""
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
# The default path uses `MockSearchClient`, whose results are a fixed dictionary in
# `agent_tools.py`, so this notebook produces the same output on every machine and costs
# nothing. `RUN_LIVE = True` swaps in whichever provider has a key set, which for this chapter
# means Tavily.
#
# The demonstration question asks whether NVIDIA beat its Q4 FY2025 earnings expectations, which
# the company answered when it reported on 2025-02-26. The question carries a **cutoff date** of
# 2025-02-20: the boundary the search tool enforces, dropping anything published on or after it,
# so an agent answering the question reads only what existed before that day.

# %%
search = create_search_client(SEARCH_PROVIDER) if RUN_LIVE else MockSearchClient()
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

# %%
all_results = search.search("NVIDIA Q4 earnings", max_results=MAX_RESULTS)
print(f"Retrieved with no cutoff: {len(all_results)}")
for r in all_results:
    print(f"  [{r.published or '?'}] {r.title}")

print()

pit_results = search.search("NVIDIA Q4 earnings", max_results=MAX_RESULTS, cutoff_date=cutoff)
print(f"Retrieved with cutoff {cutoff}: {len(pit_results)}")
for r in pit_results:
    print(f"  [{r.published or '?'}] {r.title}")

# %% [markdown]
# ### A result with no date is not a result before the cutoff
#
# Search APIs return a publication date when the page carries one, and often it does not. The
# filter above can only exclude what it can date, so an undated result passes the cutoff by
# default: the client keeps it because it cannot prove the page is too recent, not because it
# has established that the page is old enough.
#
# That default is the right one for a live run, where excluding every undated page would throw
# away most of the web. It is the wrong one for a scored historical evaluation, where an
# undated page is an unbounded hindsight risk. So the two have to be counted separately, and
# the evaluation decides what to do with the second group rather than inheriting the tool's
# choice. The cell below adds one undated result to the filtered set and partitions it.

# %%
undated = SearchResult(
    title="Undated market commentary",
    url="https://www.reuters.com/markets/undated-commentary",
    snippet="A result without provider publication metadata.",
    published=None,
    score=0.75,
)
cutoff_candidates = [*pit_results, undated]
verified_pre_cutoff = [
    result
    for result in cutoff_candidates
    if result.published and date.fromisoformat(result.published) < cutoff
]
unverified_dates = [result for result in cutoff_candidates if not result.published]

print(f"Dated and verified before the cutoff: {len(verified_pre_cutoff)}")
print(f"Undated, provenance unverified:       {len(unverified_dates)}")

# %% [markdown]
# ## Formatting Results for the Agent
#
# Results reach the model as text in a tool message, so the formatting is part of the contract
# too. `format_search_results` numbers each result and puts the title, URL, snippet and
# publication date on their own lines. The URL and the date are there so the model can weigh a
# primary source against a blog and recent reporting against stale reporting, and so that a
# quote in the final rationale can be traced back to the document it came from.

# %%
formatted = format_search_results(pit_results)
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

# %%
pl.DataFrame({"allowed_domain": sorted(DEFAULT_ALLOWED_DOMAINS)})


# %% [markdown]
# ### Enforcement is post-retrieval
#
# The filter runs on results, not on queries: the provider returns whatever matches, and the
# tool layer drops what the policy excludes before the agent sees it. That ordering has a cost,
# since a blocked document was still retrieved, and one advantage that matters more, which is
# that the policy holds whatever the model asks for. A model instructed in its prompt to avoid
# social media occasionally reads it anyway; a host filtered here never reaches the model at
# all.
#
# The three results below span a wire service, a paywalled newspaper, and a message board.
# Running the same set through an allowlist of financial publishers and through a blocklist of
# social sites shows what each policy keeps.

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

financial_only = {"reuters.com", "wsj.com", "ft.com", "bloomberg.com"}
social = {"reddit.com", "twitter.com", "x.com"}

policies = {
    "raw retrieval": mixed_results,
    "allowlist (financial)": apply_domain_policy(mixed_results, allowed=financial_only),
    "blocklist (social)": apply_domain_policy(mixed_results, blocked=social),
}

with pl.Config(fmt_str_lengths=80, tbl_rows=20):
    display(
        pl.DataFrame(
            {
                "policy": [name for name, kept in policies.items() for _ in mixed_results],
                "host": [urlparse(r.url).hostname for _ in policies for r in mixed_results],
                "kept": [r in kept for kept in policies.values() for r in mixed_results],
            }
        )
    )

# %% [markdown]
# ### The same policy inside the executor
#
# Passing the sets to `ToolExecutor` applies them to every search the agent makes, and the
# execution log records how many results the policy removed. That count is what an auditor
# reads: it says the agent was denied evidence, and how much, rather than leaving a short
# result list looking like a thin day for the query.

# %%
policed = ToolExecutor(search=search, blocked_domains={"nasdaq.com"})
policed_results = policed.execute_search("NVIDIA Q4 earnings", max_results=MAX_RESULTS)

print(f"Returned to the agent: {len(policed_results)}")
print(f"Log preview:           {policed.execution_log[0].result_preview}")
for r in policed_results:
    print(f"  {urlparse(r.url).hostname}  {r.title}")

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
print(f"Log entry status:      {disabled_executor.execution_log[0].status}")

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
