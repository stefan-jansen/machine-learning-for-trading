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
import sys

from utils.paths import get_chapter_dir

sys.path.insert(0, str(get_chapter_dir(24)))

"""Agent State, Memory, and Quality Gates - explicit state for reliable agents."""

import json
from datetime import datetime, timedelta

from agent_fixtures import get_demo_question
from agent_schemas import AgentState, QualityGateResult

# %% tags=["parameters"]
# (Parameters cell kept for Papermill - no notebook-level overrides are
# currently exposed; future tunables like MIN_EVIDENCE_ITEMS / MAX_AGE_HOURS
# would be added here.)

# %% [markdown]
# ## The State Problem
#
# Most agent tutorials treat state implicitly - it lives in the LLM's message history
# and disappears when the conversation ends. For financial agents, this is dangerous:
#
# - You can't audit what evidence the agent considered
# - You can't replay a run with different parameters
# - You can't detect when evidence is stale or insufficient
# - You can't compare two runs systematically
#
# The solution is an **explicit state schema** that captures everything the agent knows,
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
# - **Synthesis status**: pending → in_progress → complete (or abstained)

# %%
question = get_demo_question()
state = AgentState(
    ticker=question.question[:50],
    cutoff_date=question.cutoff_date,
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
        "timestamp": datetime.now().isoformat(),
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
        "timestamp": datetime.now().isoformat(),
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
# Evidence item 3: historical base-rate evidence - required by the
# coverage gate alongside forward-looking search.
search_evidence.append(
    {
        "type": "base_rate",
        "source": "web_search",
        "timestamp": datetime.now().isoformat(),
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
    print(f'  [{item["type"]}] "{item["query"][:45]}" → {n_results} results')

# %% [markdown]
# ## Quality Gates
#
# Quality gates are **automated checks** that run before the agent produces its
# final output. They catch problems that would otherwise lead to unreliable
# forecasts. We implement three gates:
#
# 1. **Coverage gate**: Does the agent have enough search results?
# 2. **Freshness gate**: Is any evidence older than the allowed window?
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

    The default `required_types` is two - `search_results` and `base_rate`
    - so the gate fails when either is missing. That matches the chapter's
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
# Checks that evidence was gathered recently - stale search results may not
# reflect the current state of the world.


# %%
def check_freshness_gate(
    state: AgentState,
    max_age_hours: int = 24,
) -> QualityGateResult:
    """Check that no evidence is older than the allowed window."""
    now = datetime.now()
    stale_items = []

    for item in state.evidence:
        ts_str = item.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str)
            age = now - ts
            if age > timedelta(hours=max_age_hours):
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
# Checks whether retained publication dates violate the cutoff. This detects
# explicit date violations. It cannot establish historical page content or
# remove information already present in the model's training data.


# %%
def check_consistency_gate(state: AgentState) -> QualityGateResult:
    """Check that no search results violate the cutoff date."""
    issues = []

    for item in state.evidence:
        content = item.get("content", {})
        if isinstance(content, dict):
            for r in content.get("results", []):
                pub = r.get("published", "")
                if pub and state.cutoff_date and pub >= state.cutoff_date:
                    issues.append(
                        f"Post-cutoff result: '{r.get('title', '')[:50]}' "
                        f"(published {pub}, cutoff {state.cutoff_date})"
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
        reason="No cutoff violations detected",
        details={"checks_run": ["cutoff_enforcement"]},
    )


# %% [markdown]
# ### Running quality gates
#
# Gates run independently and their results are stored in the agent state. If any
# gate fails, the agent should either gather more evidence or abstain.


# %%
def run_quality_gates(state: AgentState) -> list[QualityGateResult]:
    """Run all quality gates and store results in state."""
    gates = [
        check_coverage_gate(state),
        check_freshness_gate(state),
        check_consistency_gate(state),
    ]
    state.quality_gates = gates
    return gates


# %%
gates = run_quality_gates(state)

print("Quality Gate Results:")
for g in gates:
    status = "PASS" if g.passed else "FAIL"
    print(f"  [{status}] {g.gate_name}: {g.reason}")

all_passed = all(g.passed for g in gates)
print(f"\nAll gates passed: {all_passed}")

# %% [markdown]
# **Interpretation**: All three gates pass because we collected multiple search results
# (coverage), all evidence was just gathered (freshness), and all published dates are
# before the cutoff (consistency).

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
        "timestamp": datetime.now().isoformat(),
        "query": "test query",
        "content": {"results": [{"title": "Single result", "published": "2025-02-15"}]},
    }
)

sparse_gates = run_quality_gates(sparse_state)
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
        "timestamp": datetime.now().isoformat(),
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
        "timestamp": datetime.now().isoformat(),
        "query": "historical base rate",
        "content": {"results": [{"title": "Base rate data", "published": "2025-02-10"}]},
    }
)

bad_gates = run_quality_gates(bad_state)
print("\nInconsistent state gates:")
for g in bad_gates:
    status = "PASS" if g.passed else "FAIL"
    print(f"  [{status}] {g.gate_name}: {g.reason}")
    if not g.passed and g.details.get("issues"):
        for issue in g.details["issues"]:
            print(f"    → {issue}")

# %% [markdown]
# **Finding**: The coverage gate catches insufficient evidence (only 1 item), and the
# consistency gate catches the cutoff violation (a result published after the cutoff
# date). These failures would trigger the agent to either gather additional evidence
# or abstain rather than produce an unreliable forecast.

# %% [markdown]
# ### Stale evidence
#
# The third gate, freshness, fires when an evidence item's timestamp is
# older than `max_age_hours`. To make the failure observable we construct
# a state whose evidence was written 48 hours ago - twice the default
# 24-hour window - and check that the freshness gate fails while coverage
# and consistency still pass.

# %%
stale_ts = (datetime.now() - timedelta(hours=48)).isoformat()
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
stale_gates = run_quality_gates(stale_state)
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
ablation_gates = run_quality_gates(ablation)

print("\n=== Ablation (no base rate) ===")
print(f"Evidence items: {len(ablation.evidence)} (was {len(state.evidence)})")
for g in ablation_gates:
    print(f"  [{('PASS' if g.passed else 'FAIL')}] {g.gate_name}")

# %%
if not all(g.passed for g in ablation_gates):
    ablation.synthesis_status = "abstained"
    print(f"Synthesis status: {ablation.synthesis_status}")
    print("Agent correctly abstains when evidence is insufficient.")
else:
    print("Ablation still passes all gates - base rate wasn't critical for coverage.")

# %% [markdown]
# **Interpretation**: Removing the `base_rate` evidence trips the coverage
# gate - the default `required_types=["search_results", "base_rate"]`
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
# 1. **Explicit state** captures everything the agent knows, separate from the LLM's
#    context window - enabling audit, replay, and comparison
# 2. **Quality gates** catch insufficient, stale, and contaminated evidence before
#    the agent produces output - abstention is better than unreliable forecasts
# 3. **Checkpointing** enables persistence and replay - run the same analysis with
#    different parameters or evidence subsets
# 4. **Ablation via replay** reveals which evidence sources actually matter for
#    forecast quality
#
# **Next**: [`research_agent`](04_research_agent.ipynb) - combines providers, tools, and state into
# a complete ResearchAgent that produces calibrated probability forecasts.
#
# **Book**: Section 24.3 covers the memory hierarchy in depth, including vector stores
# and RAG-backed long-term memory.
