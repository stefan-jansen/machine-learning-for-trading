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
# This notebook introduces **explicit agent state**, **quality gates**, and
# **checkpoint/replay**, which support auditability and reproducible inspection.
# Without explicit records, agent runs are difficult to debug, audit, or compare.
#
# **Learning Objectives**:
# - Design an explicit state schema for the information a workflow elects to persist
# - Implement quality gates that catch stale, insufficient, or inconsistent evidence
# - Checkpoint state to JSON for persistence and replay
# - Compare runs by restoring from checkpoints
#
# **Book Reference**: Chapter 24, Section 24.3 (Agent Memory: State, Persistence,
# and Replay)
#
# **Prerequisites**: [`01_react_reasoning`](01_react_reasoning.ipynb) (providers),
# [`02_tool_contracts`](02_tool_contracts.ipynb) (tools).

# %%
"""Agent State, Memory, and Quality Gates: explicit, inspectable run state."""

import json
import warnings
from datetime import date, datetime, timedelta

warnings.filterwarnings("ignore")

from agent_fixtures import get_demo_question
from agent_schemas import AgentState, QualityGateResult

# %% tags=["parameters"]
AS_OF_ISO = "2025-03-13T12:00:00"
RUN_ID = "ch24-state-demo"

# %%
as_of = datetime.fromisoformat(AS_OF_ISO)

# %% [markdown]
# ## The State Problem
#
# Most agent tutorials treat state implicitly: it lives in the LLM's message history
# and disappears when the conversation ends. For financial agents, this is dangerous:
#
# - You cannot audit what evidence the agent considered
# - You cannot replay a run with different parameters
# - You cannot detect when evidence is stale or insufficient
# - You cannot compare two runs systematically
#
# An **explicit state schema** records the workflow state selected for persistence,
# separate from the LLM's context window.

# %% [markdown]
# ## The AgentState Schema
#
# Our `AgentState` dataclass captures:
# - **Identity**: `run_id`, `ticker` (question identifier), `cutoff_date`
# - **Evidence**: list of structured evidence items from search calls
# - **Open questions**: what the agent still needs to investigate
# - **Tool trace**: full record of search calls and results
# - **Quality gates**: pass/fail results for each gate
# - **Synthesis status**: pending -> in_progress -> complete (or abstained)

# %%
question = get_demo_question()
state = AgentState(
    ticker=question.question[:50],
    cutoff_date=question.cutoff_date,
    run_id=RUN_ID,
)
print(f"Run ID: {state.run_id}")
print(f"Question: {state.ticker}")
print(f"Cutoff: {state.cutoff_date}")
print(f"Status: {state.synthesis_status}")
print(f"Evidence items: {len(state.evidence)}")
print(f"Quality gates: {len(state.quality_gates)}")

# %% [markdown]
# ## Collecting Evidence
#
# As the agent calls search tools, we record structured evidence items in the state.
# Each item has a type, source, timestamp, and content. In the AIA Forecaster,
# all evidence comes from web search results.
#
# Simulate a research workflow: three search queries returning canned
# results. We add each evidence item to `state.evidence` (along with a
# matching `tool_trace` entry) one at a time so the structure of each item
# is visible on its own.

# %%
search_evidence: list[dict] = []

# Evidence item 1: forward-looking search on the question itself.
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

# %%
# Evidence item 2: a second forward-looking query for context.
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

# %%
# Evidence item 3: historical base-rate evidence, required by the
# coverage gate alongside forward-looking search.
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
# Quality gates are **automated checks** that run before the agent produces its
# final output. They catch problems that would otherwise lead to unreliable
# forecasts. We implement three gates:
#
# 1. **Coverage gate**: Does the agent have enough evidence records and required types?
# 2. **Freshness gate**: Was any evidence retrieved before the allowed window?
# 3. **Consistency gate**: Do any search results violate the cutoff date?

# %% [markdown]
# ### Coverage gate
#
# Verifies that the agent has gathered enough evidence and that it covers
# both recent developments and historical base rates.


# %%
def check_coverage_gate(
    state: AgentState,
    min_items: int = 2,
    required_types: list[str] | None = None,
) -> QualityGateResult:
    """Check that evidence covers required types and has minimum items.

    The default `required_types` includes `search_results` and `base_rate`,
    so the gate fails when either is missing. That matches the chapter's
    coverage rule: a credible forecast needs both forward-looking evidence
    and a historical anchor.
    """
    required_types = required_types or ["search_results", "base_rate"]
    present_types = {item["type"] for item in state.evidence}
    missing = set(required_types) - present_types

    if missing:
        return QualityGateResult(
            gate_name="coverage",
            passed=False,
            reason=f"Missing evidence types: {', '.join(sorted(missing))}",
            details={"required": required_types, "present": sorted(present_types)},
        )
    if len(state.evidence) < min_items:
        return QualityGateResult(
            gate_name="coverage",
            passed=False,
            reason=f"Only {len(state.evidence)} evidence items (need {min_items})",
            details={"count": len(state.evidence), "min_required": min_items},
        )
    return QualityGateResult(
        gate_name="coverage",
        passed=True,
        reason=f"{len(state.evidence)} items covering {len(present_types)} types",
        details={"count": len(state.evidence), "types": sorted(present_types)},
    )


# %% [markdown]
# ### Freshness gate
#
# Checks when evidence was retrieved. This differs from the publication date
# of the underlying document, which the consistency gate checks separately.


# %%
def check_freshness_gate(
    state: AgentState,
    *,
    as_of: datetime,
    max_age_hours: int = 24,
) -> QualityGateResult:
    """Check that no evidence is older than the allowed window."""
    stale_items = []

    for item in state.evidence:
        ts_str = item.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str)
            age = as_of - ts
            if age < timedelta(0):
                stale_items.append({"type": item["type"], "age_hours": "future"})
            elif age > timedelta(hours=max_age_hours):
                stale_items.append(
                    {"type": item["type"], "age_hours": round(age.total_seconds() / 3600, 1)}
                )
        except (ValueError, TypeError):
            stale_items.append({"type": item["type"], "age_hours": "unknown"})

    if stale_items:
        return QualityGateResult(
            gate_name="freshness",
            passed=False,
            reason=f"{len(stale_items)} evidence items exceed {max_age_hours}h age limit",
            details={"stale_items": stale_items, "max_age_hours": max_age_hours},
        )
    return QualityGateResult(
        gate_name="freshness",
        passed=True,
        reason=f"All evidence within {max_age_hours}h window",
        details={"max_age_hours": max_age_hours, "item_count": len(state.evidence)},
    )


# %% [markdown]
# ### Consistency gate
#
# Checks that every search result has a parseable publication date before the
# cutoff. Missing or post-cutoff dates cannot satisfy the point-in-time contract.

# %% [markdown]
# ### ISO date parser
#
# The consistency gate uses one parser for the cutoff and every result date.


# %%
def parse_iso_date(value: str) -> date | None:
    """Parse an ISO date, returning None for missing or malformed values."""
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


# %% [markdown]
# ### Consistency check


# %%
def check_consistency_gate(state: AgentState) -> QualityGateResult:
    issues = []
    cutoff = parse_iso_date(state.cutoff_date)
    if cutoff is None:
        return QualityGateResult(
            gate_name="consistency",
            passed=False,
            reason="Cutoff date is missing or invalid",
            details={"cutoff_date": state.cutoff_date},
        )

    for item in state.evidence:
        content = item.get("content", {})
        if isinstance(content, dict):
            for r in content.get("results", []):
                pub = r.get("published", "")
                if not pub:
                    issues.append(f"Missing publication date: '{r.get('title', '')[:50]}'")
                    continue
                published = parse_iso_date(pub)
                if published is None:
                    issues.append(f"Invalid publication date: '{r.get('title', '')[:50]}'")
                    continue
                if published >= cutoff:
                    issues.append(
                        f"Post-cutoff result: '{r.get('title', '')[:50]}' "
                        f"(published {published}, cutoff {cutoff})"
                    )

    if issues:
        return QualityGateResult(
            gate_name="consistency",
            passed=False,
            reason=f"{len(issues)} cutoff violations found",
            details={"issues": issues},
        )
    return QualityGateResult(
        gate_name="consistency",
        passed=True,
        reason="All result dates precede the cutoff",
        details={"checks_run": ["date_presence", "date_parse", "cutoff_enforcement"]},
    )


# %% [markdown]
# ### Running quality gates
#
# Gates run independently and their results are stored in the agent state. If any
# gate fails, the agent should either gather more evidence or abstain.


# %%
def run_quality_gates(state: AgentState, *, as_of: datetime) -> list[QualityGateResult]:
    """Run all quality gates and store results in state."""
    gates = [
        check_coverage_gate(state),
        check_freshness_gate(state, as_of=as_of),
        check_consistency_gate(state),
    ]
    state.quality_gates = gates
    return gates


# %%
gates = run_quality_gates(state, as_of=as_of)

print("Quality Gate Results:")
for g in gates:
    status = "PASS" if g.passed else "FAIL"
    print(f"  [{status}] {g.gate_name}: {g.reason}")

all_passed = all(g.passed for g in gates)
print(f"\nAll gates passed: {all_passed}")

# %% [markdown]
# **Interpretation**: The positive fixture passes because it includes both required
# evidence types, its retrieval timestamps equal the declared as-of time, and every
# result has a parseable pre-cutoff publication date.

# %% [markdown]
# ## Demonstrating Gate Failures
#
# Let's deliberately trigger failures to show how gates catch problems.

# %%
# SPARSE state: only one evidence item
sparse_state = AgentState(ticker="sparse_test", cutoff_date="2025-02-20")
sparse_state.evidence.append(
    {
        "type": "search_results",
        "source": "web_search",
        "timestamp": as_of.isoformat(),
        "query": "test query",
        "content": {"results": [{"title": "Single result", "published": "2025-02-15"}]},
    }
)

sparse_gates = run_quality_gates(sparse_state, as_of=as_of)
print("Sparse state gates:")
for g in sparse_gates:
    status = "PASS" if g.passed else "FAIL"
    print(f"  [{status}] {g.gate_name}: {g.reason}")

# %%
# INCONSISTENT state: search result published AFTER cutoff
bad_state = AgentState(ticker="cutoff_test", cutoff_date="2025-02-20")
bad_state.evidence.append(
    {
        "type": "search_results",
        "source": "web_search",
        "timestamp": as_of.isoformat(),
        "query": "test query",
        "content": {
            "results": [
                {"title": "Pre-cutoff article", "published": "2025-02-18"},
                {"title": "POST-CUTOFF: earnings beat", "published": "2025-02-27"},
            ]
        },
    }
)
bad_state.evidence.append(
    {
        "type": "base_rate",
        "source": "web_search",
        "timestamp": as_of.isoformat(),
        "query": "historical base rate",
        "content": {"results": [{"title": "Base rate data", "published": "2025-02-10"}]},
    }
)

bad_gates = run_quality_gates(bad_state, as_of=as_of)
print("\nInconsistent state gates:")
for g in bad_gates:
    status = "PASS" if g.passed else "FAIL"
    print(f"  [{status}] {g.gate_name}: {g.reason}")
    if not g.passed and g.details.get("issues"):
        for issue in g.details["issues"]:
            print(f"    -> {issue}")

# %% [markdown]
# **Finding**: The coverage gate catches the deliberately sparse fixture, and the
# consistency gate catches the post-cutoff result. A production policy can use these
# failures to gather more evidence or abstain.

# %% [markdown]
# ### Stale evidence
#
# The third gate, freshness, fires when an evidence item's timestamp is
# older than `max_age_hours`. To make the failure observable, the fixture's
# retrieval timestamp predates the declared as-of time by more than the default
# window.

# %%
stale_ts = (as_of - timedelta(hours=48)).isoformat()
stale_state = AgentState(ticker="stale_test", cutoff_date="2025-02-20")
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
    ]
)
stale_gates = run_quality_gates(stale_state, as_of=as_of)
print("Stale state gates:")
for g in stale_gates:
    status = "PASS" if g.passed else "FAIL"
    print(f"  [{status}] {g.gate_name}: {g.reason}")

# %% [markdown]
# ## Checkpointing and Replay
#
# State serialization enables two critical capabilities:
# 1. **Persistence**: Save mid-run state and resume later
# 2. **Replay**: Re-run analysis from a known state with different parameters


# %%
def checkpoint_state(state: AgentState) -> str:
    """Serialize state to JSON for persistence."""
    return state.to_json()


# %%
def restore_checkpoint(json_str: str) -> AgentState:
    """Restore state from JSON checkpoint."""
    return AgentState.from_json(json_str)


# %%
checkpoint = checkpoint_state(state)
print(f"Checkpoint size: {len(checkpoint):,} bytes")
print(f"Run ID: {state.run_id}")

data = json.loads(checkpoint)
print(f"\nCheckpoint keys: {list(data.keys())}")
print(f"Evidence items: {len(data['evidence'])}")
print(f"Quality gates: {len(data['quality_gates'])}")

# %%
restored = restore_checkpoint(checkpoint)
print(f"Restored run ID: {restored.run_id}")
print(f"Restored evidence: {len(restored.evidence)} items")
print(f"Restored gates: {len(restored.quality_gates)} results")

assert restored.run_id == state.run_id
assert len(restored.evidence) == len(state.evidence)
print("\nRound-trip verification: PASSED")

# %% [markdown]
# ## Replay: Comparing Runs
#
# With checkpoints, we can systematically compare runs with different configurations.
# For example: "What happens if we drop the base rate evidence?"

# %%
print("=== Original Run ===")
for g in state.quality_gates:
    print(f"  [{('PASS' if g.passed else 'FAIL')}] {g.gate_name}")

# Ablation: remove base_rate evidence
ablation = restore_checkpoint(checkpoint)
ablation.evidence = [e for e in ablation.evidence if e["type"] != "base_rate"]
ablation_gates = run_quality_gates(ablation, as_of=as_of)

print("\n=== Ablation (no base rate) ===")
print(f"Evidence items: {len(ablation.evidence)} (was {len(state.evidence)})")
for g in ablation_gates:
    print(f"  [{('PASS' if g.passed else 'FAIL')}] {g.gate_name}")

# %%
if not all(g.passed for g in ablation_gates):
    ablation.synthesis_status = "abstained"
    print(f"Synthesis status: {ablation.synthesis_status}")
    print("The policy abstains when the coverage contract is not satisfied.")
else:
    print("Ablation still passes all gates; the base rate was not required for coverage.")

# %% [markdown]
# **Interpretation**: Removing the `base_rate` evidence trips the coverage
# gate: the default `required_types=["search_results", "base_rate"]`
# treats the historical anchor as part of the minimum coverage contract.
# That is the point of an explicit gate: ablations that previously
# "looked fine" now fail loudly, and checkpoint/replay turns the ablation
# into a one-line experiment.

# %% [markdown]
# ## Memory hierarchy in one line
#
# This notebook covers **short-term memory** (the explicit `AgentState`).
# **Working memory** is the LLM context window from NB01; **long-term
# memory** via RAG is the subject of Chapter 22.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Explicit state** records selected workflow state outside the LLM context,
#    enabling audit, replay, and comparison
# 2. **Quality gates** catch insufficient, stale, and contaminated evidence before
#    synthesis, so the policy can gather more evidence or abstain
# 3. **Checkpointing** enables persistence and replay: run the same analysis with
#    different parameters or evidence subsets
# 4. **Ablation via replay** reveals which evidence types satisfy the configured
#    quality gates
#
# **Next**: [`04_research_agent`](04_research_agent.ipynb), which combines providers,
# tools, and state into a ResearchAgent that produces probability forecasts.
#
# **Book**: Section 24.3 covers the memory hierarchy in depth, including vector stores
# and RAG-backed long-term memory.
