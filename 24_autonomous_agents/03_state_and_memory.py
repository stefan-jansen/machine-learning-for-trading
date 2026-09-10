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
# # Agent State, Memory, and Quality Gates
#
# **Docker image**: `ml4t`
#
# The agent in [`01_react_reasoning`](01_react_reasoning.ipynb) kept everything it knew in its
# message history. That is enough to produce an answer and not enough to defend one: when the
# conversation ends there is no record of what evidence was considered, no way to re-run the
# analysis from where it went wrong, and no way to compare two runs except by reading two
# transcripts. This notebook replaces the transcript with a typed record the agent writes to,
# adds checks that refuse to let a thin run produce a confident answer, and shows what that
# record makes possible once it exists.
#
# **Learning Objectives**:
# - Write the state a research run depends on into a typed record, separate from what the model
#   can see
# - Refuse to answer when the evidence is too thin, too old, or dated after the question's
#   cutoff, and record which check refused
# - Save a run to JSON mid-flight and restore it, so a run can be resumed rather than restarted
# - Re-run a saved state with one class of evidence removed, to find out whether that evidence
#   was doing any work
#
# **Book Reference**: Chapter 24, Section 24.3 (Agent Memory: State, Persistence,
# and Replay)
#
# **Prerequisites**: [`01_react_reasoning`](01_react_reasoning.ipynb) (providers),
# [`02_tool_contracts`](02_tool_contracts.ipynb) (tools).

# %%
"""Agent State, Memory, and Quality Gates: explicit, inspectable run state."""

import json
from datetime import date, datetime, timedelta

from agent_fixtures import get_demo_question
from agent_schemas import (
    AgentState,
    check_consistency_gate,
    check_coverage_gate,
    check_freshness_gate,
    run_quality_gates,
)

# %% [markdown]
# ## Settings
#
# `AS_OF_ISO` is the moment the run is pretending to happen. Everything time-dependent is
# measured against it rather than against the clock, which is what lets this notebook produce
# the same output today and next year. It is set to the day before the demonstration
# question's 2025-02-20 cutoff, so the run sits where a forecaster answering that question
# would have sat.
#
# `RUN_ID` names the run. A checkpoint is only useful if it can be told apart from the next
# one, and a fixed value here keeps the printed output stable; a real deployment generates one
# per run, which is what `AgentState` does when none is given.

# %% tags=["parameters"]
AS_OF_ISO = "2025-02-19T12:00:00"
RUN_ID = "ch24-state-demo"

# %%
as_of = datetime.fromisoformat(AS_OF_ISO)

# %% [markdown]
# ## Why a Trace Is Not State
#
# [`01_react_reasoning`](01_react_reasoning.ipynb) already writes a durable record: `RunTrace`
# saves the question, every prompt and reply, and every search result into a JSON file, and
# that file is what the replay path reads back. So the run is reproducible, and a reader can
# see exactly what happened.
#
# What that record cannot do is participate in the run. It is written at the end, from the
# outside, and holds the model's conversation rather than the agent's own conclusions. Four
# things a research process needs are all still missing:
#
# - **A place for derived knowledge.** The evidence the agent judged relevant, the questions it
#   has not resolved, what it has decided so far. None of that is a prompt or a reply.
# - **Something to check before answering.** A gate has to read a structured account of the
#   evidence, not a transcript, and it has to run while the agent can still act on it.
# - **A resumption point.** A conversation log replays a run; it does not let one continue from
#   the middle with a changed setting.
# - **A comparable object.** Two runs are two conversations, and no diff over prose is a
#   measurement. Two `AgentState` records diff field by field.
#
# So the trace and the state are different artifacts with different jobs, and this chapter
# keeps both: `RunTrace` is what the run looked like from outside, `AgentState` is what the
# agent knew from inside.

# %% [markdown]
# ## The AgentState Schema
#
# `AgentState` in `agent_schemas.py` holds eight fields, in four groups:
#
# - **Identity**: `run_id`, `question`, `cutoff_date` - which run this is, what it is answering,
#   and the date past which its evidence may not be published
# - **Evidence**: `evidence`, the documents gathered, and `open_questions`, what the agent has
#   not resolved yet
# - **Provenance**: `tool_trace`, every tool call the run made, including the ones that returned
#   nothing
# - **Checks**: `quality_gates`, the checks defined below and their outcomes, and
#   `synthesis_status`, which moves from `pending` through `in_progress` to `complete`, or to
#   `abstained` when the gates refuse the run
#
# It is a plain dataclass with `to_json` and `from_json`, which is all a checkpoint needs to be.

# %%
question = get_demo_question()
state = AgentState(
    question=question.question,
    cutoff_date=question.cutoff_date,
    run_id=RUN_ID,
)
print(f"Run ID:         {state.run_id}")
print(f"Question:       {state.question}")
print(f"Cutoff:         {state.cutoff_date}")
print(f"Status:         {state.synthesis_status}")
print(f"Evidence items: {len(state.evidence)}")
print(f"Quality gates:  {len(state.quality_gates)}")
# %% [markdown]
# ## Collecting Evidence
#
# Every search the agent runs appends two records: an evidence item, holding what came back,
# and a tool-trace entry, holding that the call happened at all. Keeping both matters, because
# a search that returned nothing leaves an evidence list unchanged and a trace one entry longer,
# and only the trace can tell "the agent did not look" apart from "the agent looked and found
# nothing".
#
# An evidence item carries a `type`, the `source` that produced it, the `timestamp` at which it
# was retrieved, the `query` that produced it, and the `content` itself. The three items below
# stand in for a short research session and are written out one at a time so each shape is
# visible on its own; in a real run they come from the search client of
# [`02_tool_contracts`](02_tool_contracts.ipynb).

# %% [markdown]
# The first item is a search on the question itself: what is being said now about the event
# being forecast.

# %%
search_evidence: list[dict] = []

search_evidence.append(
    {
        "type": "search_results",
        "source": "web_search",
        "timestamp": as_of.isoformat(),
        "query": "NVIDIA Q4 FY2025 earnings expectations",
        "content": {
            "results": [
                {
                    "title": "NVIDIA suppliers signal continued AI server demand",
                    "url": "https://wsj.com/nvidia-supply-chain",
                    "snippet": "Supply-chain checks point to resilient GPU demand ahead of earnings report.",
                    "published": "2025-02-17",
                },
                {
                    "title": "Analysts raise NVIDIA targets ahead of earnings",
                    "url": "https://nasdaq.com/nvidia-targets",
                    "snippet": "Street revisions remain constructive ahead of the Feb 26 report.",
                    "published": "2025-02-18",
                },
            ]
        },
    }
)

# %% [markdown]
# The second is a narrower query on the business line that will decide the outcome. It carries
# the same `search_results` type: the type says what kind of evidence this is, not which query
# produced it.

# %%
search_evidence.append(
    {
        "type": "search_results",
        "source": "web_search",
        "timestamp": as_of.isoformat(),
        "query": "NVIDIA data center revenue growth",
        "content": {
            "results": [
                {
                    "title": "Data center revenue expected to exceed $30B",
                    "url": "https://reuters.com/nvidia-data-center",
                    "snippet": "Consensus estimates point to another record quarter for data center.",
                    "published": "2025-02-15",
                },
            ]
        },
    }
)

# %% [markdown]
# The third is a **base rate**: how often the event has happened before, independent of
# anything specific to this quarter. It is typed separately because the coverage gate below
# requires it, for a reason worth stating plainly. Asked to forecast from current reporting
# alone, a model reasons from the narrative in front of it and ignores the frequency; a
# historical anchor is what a probability has to be an adjustment away from.

# %%
search_evidence.append(
    {
        "type": "base_rate",
        "source": "web_search",
        "timestamp": as_of.isoformat(),
        "query": "NVIDIA historical earnings beat rate",
        "content": {
            "results": [
                {
                    "title": "NVIDIA has beaten earnings estimates 8 of last 8 quarters",
                    "url": "https://nasdaq.com/nvidia-earnings-history",
                    "snippet": "Strong track record of exceeding consensus, with average surprise of 12%.",
                    "published": "2025-02-10",
                },
            ]
        },
    }
)

# %% [markdown]
# Appending them to the state pairs each evidence item with its trace entry.

# %%
for item in search_evidence:
    state.evidence.append(item)
    state.tool_trace.append({"tool": "search", "query": item["query"], "status": "success"})

print(f"Evidence collected: {len(state.evidence)} items")
print(f"Search calls made: {len(state.tool_trace)}")
for item in state.evidence:
    n_results = len(item["content"].get("results", []))
    print(f'  [{item["type"]}] "{item["query"][:45]}" -> {n_results} results')
# %% [markdown]
# ## Quality Gates
#
# A **quality gate** is a check that runs on the state before the agent is allowed to produce
# an answer. It is a contract about evidence: a statement, in code, of what a run must have
# gathered for its output to be worth reading. When one fails the agent gathers more or
# abstains, and the failure is recorded, so a thin run is visibly thin rather than quietly
# confident.
#
# Three gates cover three different ways the evidence can be inadequate:
#
# 1. **Coverage** asks whether there is enough evidence, of the required kinds.
# 2. **Freshness** asks how long ago the agent went and looked.
# 3. **Consistency** asks whether anything it read was published after the cutoff.

# %% [markdown]
# ### Coverage gate
#
# Two conditions. The required types make the gate an argument about *what kind* of evidence a
# forecast needs: `search_results` for what is happening now and `base_rate` for how often the
# thing has happened before. A model given only current reporting will follow the narrative and
# ignore the frequency, which is the classic base-rate neglect, and requiring both types is a
# cheap structural defence against it. `min_items` is the separate question of how much: three
# records is the chapter's working minimum, low enough that a normal run clears it and high
# enough that a single search cannot.


# %% [markdown]
# ### Freshness gate
#
# This gate reads the `timestamp` on each evidence item, which is when the agent retrieved the
# document, not when the document was published. The two come apart in both directions: a run
# that fetched a 2019 paper an hour ago is fresh, and a run that fetched this morning's
# reporting three days ago is not. Twenty-four hours is the default window because the
# questions in this chapter resolve on news cycles; a question about a quarterly filing would
# take a wider one.
#
# A timestamp in the future is treated as a failure rather than as maximal freshness. It means
# the clock, the fixture or the checkpoint is wrong, and a gate that reads it as fresh would
# pass a run precisely when its record cannot be trusted.


# %% [markdown]
# ### Consistency gate
#
# Freshness is about the run; consistency is about the documents. Every search result has to
# carry a publication date that parses and that falls before the question's cutoff. The three
# ways a result can fail are reported separately, because they mean different things: a missing
# date is evidence the tool could not date, an unparseable one is a provider bug, and a
# post-cutoff one is hindsight that has already entered the run.
#
# [`02_tool_contracts`](02_tool_contracts.ipynb) filters at retrieval, so in a normal run
# nothing reaches here to fail. This gate is the second check on the same property, at the
# other end of the pipeline, and it exists because the first one can be bypassed: evidence
# arrives from fixtures, from checkpoints, and from tools written after the filter was.

# %% [markdown]
# The three functions live in `agent_schemas.py` beside `AgentState` itself, so the agent in
# [`04_research_agent`](04_research_agent.ipynb) checks its evidence against exactly the
# definitions demonstrated here. One parser handles the cutoff and every result date, so a date
# the cutoff comparison would accept cannot be one a result comparison rejects, and each gate
# collects every failure rather than stopping at the first, so one run reports the full extent
# of the problem.

# %% [markdown]
# ### Running the gates
#
# `run_quality_gates` runs the three, stores the outcomes on the state, and returns them, so a
# checkpoint carries not only the evidence but the judgement made about it. What to do on a
# failure is the caller's decision, not the gate's:
# [`04_research_agent`](04_research_agent.ipynb) runs these same three over a finished agent
# run and reports what they say about it.

# %%
gates = run_quality_gates(state, as_of=as_of)

print("Quality gate results:")
for g in gates:
    status = "PASS" if g.passed else "FAIL"
    print(f"  [{status}] {g.gate_name}: {g.reason}")

all_passed = all(g.passed for g in gates)
print(f"\nAll gates passed: {all_passed}")

# %% [markdown]
# All three pass. The run gathered both required evidence types across three records, it
# retrieved them at the declared as-of time, and every document it read was published before
# the cutoff. Passing gates say the evidence is admissible, and nothing more: none of them has
# read a word of what the documents actually claim.

# %% [markdown]
# ## What each gate catches
#
# A gate that has never been seen to fail is a gate nobody should trust. Each state below is
# built to breach exactly one contract, so the failure can be attributed to the check that
# raised it rather than guessed at.

# %% [markdown]
# ### Too few evidence types
#
# One search, one result, no historical anchor. The coverage gate checks required types before
# it checks the count, so this is reported as a missing type.

# %%
sparse_state = AgentState(question="one search only", cutoff_date="2025-02-20")
sparse_state.evidence.append(
    {
        "type": "search_results",
        "source": "web_search",
        "timestamp": as_of.isoformat(),
        "query": "NVIDIA Q4 FY2025 earnings expectations",
        "content": {"results": [{"title": "Single result", "published": "2025-02-15"}]},
    }
)

sparse_gates = run_quality_gates(sparse_state, as_of=as_of)
print("Gates on the one-search state:")
for g in sparse_gates:
    print(f"  [{'PASS' if g.passed else 'FAIL'}] {g.gate_name}: {g.reason}")

# %% [markdown]
# ### Both types, too little of either
#
# Coverage is two conditions, not one. This state satisfies the type requirement and still
# fails, because two records is below the minimum the gate is configured to demand.

# %%
thin_state = AgentState(question="both types, two records", cutoff_date="2025-02-20")
thin_state.evidence.extend(
    [
        {
            "type": "search_results",
            "source": "web_search",
            "timestamp": as_of.isoformat(),
            "query": "NVIDIA Q4 FY2025 earnings expectations",
            "content": {"results": [{"title": "Supply chain check", "published": "2025-02-17"}]},
        },
        {
            "type": "base_rate",
            "source": "web_search",
            "timestamp": as_of.isoformat(),
            "query": "NVIDIA historical earnings beat rate",
            "content": {"results": [{"title": "Beat history", "published": "2025-02-10"}]},
        },
    ]
)

thin_gates = run_quality_gates(thin_state, as_of=as_of)
print("Gates on the two-record state:")
for g in thin_gates:
    print(f"  [{'PASS' if g.passed else 'FAIL'}] {g.gate_name}: {g.reason}")

# %% [markdown]
# ### A document published after the cutoff
#
# This is the failure that matters most and shows up least. The state has enough evidence of
# both types, retrieved on time, and one of its search results was published on 2025-02-27:
# the day after NVIDIA reported. An agent reading it is not forecasting, and its forecast will
# look excellent.

# %%
bad_state = AgentState(question="reads past the cutoff", cutoff_date="2025-02-20")
bad_state.evidence.extend(
    [
        {
            "type": "search_results",
            "source": "web_search",
            "timestamp": as_of.isoformat(),
            "query": "NVIDIA Q4 FY2025 earnings expectations",
            "content": {
                "results": [
                    {"title": "Pre-cutoff article", "published": "2025-02-18"},
                    {"title": "Earnings beat confirmed", "published": "2025-02-27"},
                ]
            },
        },
        {
            "type": "search_results",
            "source": "web_search",
            "timestamp": as_of.isoformat(),
            "query": "NVIDIA data center revenue growth",
            "content": {"results": [{"title": "Data center outlook", "published": "2025-02-15"}]},
        },
        {
            "type": "base_rate",
            "source": "web_search",
            "timestamp": as_of.isoformat(),
            "query": "NVIDIA historical earnings beat rate",
            "content": {"results": [{"title": "Base rate data", "published": "2025-02-10"}]},
        },
    ]
)

bad_gates = run_quality_gates(bad_state, as_of=as_of)
print("Gates on the state that read past its cutoff:")
for g in bad_gates:
    print(f"  [{'PASS' if g.passed else 'FAIL'}] {g.gate_name}: {g.reason}")
    for issue in g.details.get("issues", []) if not g.passed else []:
        print(f"    -> {issue}")

# %% [markdown]
# ### Stale evidence
#
# Freshness asks a different question from consistency. Consistency asks whether a document
# was published before the cutoff; freshness asks how long ago the agent went and looked. A
# run that gathered its evidence two days ago and is producing a forecast now has been reading
# a stale snapshot of the world, whatever the publication dates say.
#
# The state below carries both required types and dates every document well before the cutoff,
# so coverage and consistency pass. Its retrieval timestamps sit 48 hours before the declared
# as-of time, which is past the gate's 24-hour default.

# %%
stale_ts = (as_of - timedelta(hours=48)).isoformat()
stale_state = AgentState(question="gathered two days ago", cutoff_date="2025-02-20")
stale_state.evidence.extend(
    [
        {
            "type": "search_results",
            "source": "web_search",
            "timestamp": stale_ts,
            "query": "NVIDIA earnings (stale)",
            "content": {"results": [{"title": "Old article", "published": "2025-02-12"}]},
        },
        {
            "type": "base_rate",
            "source": "web_search",
            "timestamp": stale_ts,
            "query": "NVIDIA beat rate (stale)",
            "content": {"results": [{"title": "Historical beats", "published": "2025-02-10"}]},
        },
        {
            "type": "search_results",
            "source": "web_search",
            "timestamp": stale_ts,
            "query": "NVIDIA guidance (stale)",
            "content": {"results": [{"title": "Guidance recap", "published": "2025-02-11"}]},
        },
    ]
)
stale_gates = run_quality_gates(stale_state, as_of=as_of)
print("Stale state gates:")
for g in stale_gates:
    status = "PASS" if g.passed else "FAIL"
    print(f"  [{status}] {g.gate_name}: {g.reason}")

# %% [markdown]
# ## Checkpointing
#
# `AgentState` is a dataclass of plain lists and strings, so serialising it is
# `json.dumps(asdict(self))` and restoring it is the constructor. That simplicity is the point:
# a checkpoint format that needs a custom encoder is a format that will silently drop a field
# the day someone adds one.
#
# The record is JSON rather than a pickle for the same reason it lives outside the context
# window. A pickle needs this codebase at this version to be read at all; a JSON checkpoint can
# be opened in a text editor, diffed against another run, queried without loading Python, and
# read in five years by something that has never heard of `AgentState`.

# %%
checkpoint = state.to_json()
print(f"Checkpoint size: {len(checkpoint):,} bytes")

data = json.loads(checkpoint)
print(f"Top-level fields: {list(data.keys())}")
print(f"Evidence items:   {len(data['evidence'])}")
print(f"Quality gates:    {len(data['quality_gates'])}")

# %% [markdown]
# Restoring it has to give back the same record, not merely a similar one. Comparing the two
# serialised forms checks every field at once, including the ones a hand-written comparison
# would forget to look at.

# %%
restored = AgentState.from_json(checkpoint)
print(f"Restored run ID:   {restored.run_id}")
print(f"Restored evidence: {len(restored.evidence)} items")
print(f"Restored gates:    {len(restored.quality_gates)} results")

assert restored.to_json() == checkpoint, "checkpoint did not survive the round trip"
print("Round trip: byte-identical")

# %% [markdown]
# ## Replay: what happens without the historical anchor
#
# The reason to checkpoint is not only to resume. A saved state is also the input to an
# experiment: restore it, change one thing, re-run the gates, and read what moved. The
# question below is whether the base-rate evidence is load-bearing or decorative.

# %%
print("=== Original run ===")
for g in state.quality_gates:
    print(f"  [{('PASS' if g.passed else 'FAIL')}] {g.gate_name}")

ablation = AgentState.from_json(checkpoint)
ablation.evidence = [e for e in ablation.evidence if e["type"] != "base_rate"]
ablation_gates = run_quality_gates(ablation, as_of=as_of)

print("\n=== Without base-rate evidence ===")
print(f"Evidence items: {len(ablation.evidence)} (was {len(state.evidence)})")
for g in ablation_gates:
    print(f"  [{('PASS' if g.passed else 'FAIL')}] {g.gate_name}")

# %%
if not all(g.passed for g in ablation_gates):
    ablation.synthesis_status = "abstained"
    print(f"Synthesis status: {ablation.synthesis_status}")
    print("No forecast is produced: the coverage contract is not satisfied.")
else:
    print("All gates still pass; the base rate was not required for coverage.")

# %% [markdown]
# Dropping the base rate takes the coverage gate below its required types, and the run
# abstains. That is the behaviour worth arguing about rather than the code: whether a
# forecast should be refused because no historical frequency was found is a research
# decision, and writing it as a gate is what makes it a decision someone can point at
# rather than an accident of what the agent happened to search for.

# %% [markdown]
# ## Where this sits in the memory hierarchy
#
# Three kinds of memory are usually distinguished, and they are easy to conflate because a
# language model presents all three as text in one prompt. **Working memory** is the context
# window: what the model can see this turn, built in
# [`01_react_reasoning`](01_react_reasoning.ipynb). **Short-term memory** is the run's own
# durable record, which is the `AgentState` above: it outlives the conversation and can be
# read by something other than the model. **Long-term memory** is knowledge carried across
# runs, retrieved on demand rather than held in the prompt, which is what Chapter 22 builds
# with retrieval-augmented generation.

# %% [markdown]
# ## Key Takeaways
#
# 1. **State the agent can be audited on has to live outside the context window.** A message
#    history is a transcript, not a record: it cannot be queried, diffed, or replayed, and it is
#    gone when the conversation ends.
# 2. **A gate is a contract about evidence, checked before synthesis.** It says what a run must
#    have gathered to be allowed to produce an answer, in code, so a thin run abstains instead
#    of guessing confidently.
# 3. **Separate when evidence was retrieved from when it was published.** The two answer
#    different questions - is this run stale, and did this run read the future - and conflating
#    them lets one hide the other.
# 4. **Abstention is an outcome, not a failure.** A forecast the evidence does not support costs
#    more than no forecast, because it is scored as if it were a judgement.
# 5. **A checkpoint turns an ablation into a one-line experiment.** Restore, drop a class of
#    evidence, re-run the gates, and read what changed.
#
# **Known limitations of what is built here.** The gates check that evidence exists, is recent,
# and predates the cutoff; none of them looks at whether it is any good, whether five results
# are five sources or one wire story copied five times, or whether the sources contradict each
# other. Freshness is measured against a declared as-of rather than wall-clock time, which is
# what makes replay possible and also means a stale run replays as fresh. And the gates run
# once before synthesis, so evidence that arrives during synthesis is ungated.
#
# **Next**: [`04_research_agent`](04_research_agent.ipynb) combines the provider, the search
# tool, and this state into an agent that gathers evidence, runs these gates, and either
# forecasts or abstains.
#
# **Book**: Section 24.3 covers the memory hierarchy in depth, including vector stores
# and RAG-backed long-term memory.
