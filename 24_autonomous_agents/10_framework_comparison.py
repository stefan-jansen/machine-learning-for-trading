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
# # Framework Comparison: Native SDK vs CrewAI vs LangGraph
#
# **Docker image**: `ml4t`
#
# The pipeline of notebooks 04 to 08 was built end to end with plain Python classes,
# no orchestration framework. That choice keeps every transition visible and
# every artifact inspectable, but it pushes the reader to write the
# orchestration logic by hand. The agentic-framework ecosystem offers
# alternatives that automate parts of that work in exchange for adopting a
# framework's conventions.
#
# This notebook expresses the same four-phase forecasting flow in three styles, and only two
# of them are comparable:
#
# 1. **Native Python SDK**: direct composition of the chapter's
#    `ResearchAgent` / `DebateAgent` / `SupervisorAgent` classes, same shape
#    as `AIAForecaster` in [`08_forecasting_pipeline`](08_forecasting_pipeline.ipynb).
# 2. **LangGraph**: the pipeline as a `StateGraph` with typed state and four
#    nodes that delegate to the same specialist classes the native variant
#    uses. Native and LangGraph call the same research, aggregation, debate and
#    supervisor functions, so what differs between them is the orchestration and
#    nothing about how evidence is accepted.
# 3. **CrewAI**: a **configuration sketch**, not a run. CrewAI's primitives
#    (`Agent`, `Task`, `Crew`) are shown as they would be written, and the code is
#    not executed here. An earlier pass reported fixed probabilities for this
#    variant with no capture behind them; a number with nothing to inspect is worse
#    than no number, so it is gone and the syntax is what remains.
#
# So the comparison is of the **orchestration layer** between two implementations that share
# their internals: how each expresses a four-phase flow, where state lives, and what happens
# when something fails midway. It is a teaching implementation and not a performance
# benchmark - two runs of one question decide nothing about which framework forecasts better.
#
# **Learning objectives**:
# - Express the same specialist, aggregate, debate and supervise flow twice, once by composing
#   classes directly and once as a typed state graph, and read off what the graph adds
# - Say where a run's state lives in each implementation, and find the entry point you would
#   attach a debugger to when a phase fails midway
# - Decide whether a project should adopt an orchestration framework, from what the framework
#   takes over and what it costs to inspect a run through it
#
# **Book reference**: §24.5 (The Engineering Stack: Frameworks and Migration).
#
# **Prerequisites**: [`04_research_agent`](04_research_agent.ipynb) through
# [`08_forecasting_pipeline`](08_forecasting_pipeline.ipynb), which build the specialist
# classes and the pipeline that the native and LangGraph variants compose.
#
# **Replay vs live mode**: by default (`RUN_LIVE = False`) the notebook replays the pinned
# 2026-06-09 native and LangGraph captures, which hold the real model exchanges and the
# durations measured that day. Those two ran under the chapter's earlier aggregation, so they
# are labelled legacy where their numbers appear. No API call is made and the output is
# identical on every machine. Set `RUN_LIVE = True` to run both variants against a current
# question through the chapter's `LLMClient`, which costs money and reproduces nothing; use
# `LLM_PROVIDER=mock` for the deterministic client.

# %%
"""Framework comparison: one four-phase pipeline, native and as a LangGraph."""

import sys

from utils.paths import get_chapter_dir

sys.path.insert(0, str(get_chapter_dir(24)))

import math
import time
from datetime import date
from typing import TypedDict

import polars as pl
from agent_fixtures import get_chapter_contested_question
from agent_observability import TRACES_DIR, RunTrace, replay_llm_calls, trace_llm
from agent_pipeline import logodds_extremize, validate_probabilities
from agent_providers import TokenUsage, create_llm_client
from agent_research import ResearchAgent, format_agent_summary
from agent_schemas import AggregationResult, ForecastQuestion, ForecastResult
from agent_specialists import DebateAgent, SupervisorAgent
from agent_tools import create_search_client

# %% [markdown]
# ## Settings
#
# `RUN_LIVE` left at `False` replays the pinned 2026-06-09 run: the saved native and LangGraph
# traces are reloaded, no API calls are made, and the reported figures are the ones measured
# that day. Set it to `True` to run both executable variants live against a current question,
# which costs money and reproduces nothing.
#
# `N_AGENTS`, `MAX_STEPS`, `DEBATE_ROUNDS` and `SUPERVISOR_QUERIES` configure the chapter's
# specialist classes, which the native and LangGraph variants both compose, so a change moves
# both of them identically. That is what makes the statement count and the state-location
# column the comparison, rather than the two probabilities.
#
# The turn budget matters more here than elsewhere: an agent that runs out of turns returns no
# probability at all, and a variant whose agents all did that would look like it had failed
# when the budget was what failed.
#
# `LLM_PROVIDER` is empty so the factory picks the first provider whose key is set. The CrewAI
# section calls no model: it is printed configuration, and CrewAI's own LiteLLM layer is one
# reason there was nothing for the chapter's tracing to capture.

# %% tags=["parameters"]
RUN_LIVE = False
LLM_PROVIDER = ""
N_AGENTS = 3
MAX_STEPS = 5
DEBATE_ROUNDS = 2
SUPERVISOR_QUERIES = 2
PINNED_TRACES = {
    "Native Python": "10_framework_comparison_20260609T150909Z_ba1855847f4c.json",
    "LangGraph": "10_framework_comparison_20260609T151202Z_c8cba3007562.json",
}


# %% [markdown]
# ## Shared stages
#
# The state retains all research attempts. Only supported probabilities enter the
# mean. Abstention or a failed stage stops further model calls, preserving the
# partial evidence. Confidence is self-reported, not a calibrated reliability score.


# %%
class ForecastState(TypedDict, total=False):
    question: ForecastQuestion
    artifacts: list
    aggregation: object
    debate: object
    supervisor: object
    final_probability: float | None
    status: str
    failure_reason: str


def research_node(state, llm, search):
    artifacts = [
        ResearchAgent(llm, search, agent_id=f"agent_{i}", max_steps=MAX_STEPS).run(
            state["question"], market_price=None
        )
        for i in range(N_AGENTS)
    ]
    return {"artifacts": artifacts}


def aggregate_node(state):
    probs = [a.p_yes for a in state["artifacts"] if a.status == "accepted" and a.p_yes is not None]
    if not probs:
        return {"status": "abstained", "failure_reason": "No supported research forecasts"}
    validate_probabilities(probs)
    return {
        "aggregation": AggregationResult("mean", sum(probs) / len(probs), input_probabilities=probs)
    }


def summaries(state):
    return "\n\n".join(
        format_agent_summary(a) for a in state["artifacts"] if a.status == "accepted"
    )


def debate_node(state, llm):
    agent = DebateAgent(llm, max_rounds=DEBATE_ROUNDS)
    try:
        debate = agent.run(
            state["question"].question, summaries(state), state["aggregation"].raw_probability
        )
    finally:
        state["debate"] = agent.artifact
    return {"debate": debate}


def supervise_node(state, llm, search):
    q = state["question"]
    agent = SupervisorAgent(llm, search, max_queries=SUPERVISOR_QUERIES)
    try:
        sup = agent.run(
            q.question,
            summaries(state),
            date.fromisoformat(q.cutoff_date) if q.cutoff_date else None,
        )
    finally:
        state["supervisor"] = agent.artifact
    probability = state["aggregation"].raw_probability
    debate = state["debate"]
    if debate.consensus_reached:
        probability = (debate.bull_final_probability + debate.bear_final_probability) / 2
    if sup.confidence == "high" and sup.p_yes is not None:
        probability = sup.p_yes
    return {"supervisor": sup, "final_probability": logodds_extremize(probability, math.sqrt(3))}


def apply_stage(state, function, *args):
    if state.get("status") != "accepted":
        return state
    try:
        return {**state, **function(state, *args)}
    except Exception as exc:
        return {**state, "status": "failed", "failure_reason": str(exc), "final_probability": None}


def forecast_result(state):
    total = TokenUsage()
    for artifact in [*state.get("artifacts", []), state.get("debate"), state.get("supervisor")]:
        if artifact is not None:
            total += artifact.token_usage
    return ForecastResult(
        question=state["question"],
        agents=state.get("artifacts", []),
        aggregation=state.get("aggregation"),
        debate=state.get("debate"),
        supervisor=state.get("supervisor"),
        final_probability=state.get("final_probability"),
        status=state["status"],
        failure_reason=state.get("failure_reason"),
        total_token_usage=total,
    )


# %% [markdown]
# ## Native Python
#
# Ordinary function calls own the state and execute the stages in order. Debugging
# and failure handling remain visible in the Python call stack.


# %%
def native_sdk_pipeline(question, llm, search):
    state = {"question": question, "status": "accepted"}
    for function, args in [
        (research_node, (llm, search)),
        (aggregate_node, ()),
        (debate_node, (llm,)),
        (supervise_node, (llm, search)),
    ]:
        state = apply_stage(state, function, *args)
    return forecast_result(state)


# %% [markdown]
# ## LangGraph
#
# The graph declares the same four stages and edges. Each node delegates to the
# same function and retains the same artifacts. This example demonstrates the
# execution graph; persistence/checkpointing requires separate configuration.


# %%
def langgraph_pipeline(question, llm, search):
    from langgraph.graph import END, StateGraph

    graph = StateGraph(ForecastState)
    graph.add_node("research", lambda s: apply_stage(s, research_node, llm, search))
    graph.add_node("aggregate", lambda s: apply_stage(s, aggregate_node))
    graph.add_node("debate", lambda s: apply_stage(s, debate_node, llm))
    graph.add_node("supervise", lambda s: apply_stage(s, supervise_node, llm, search))
    graph.set_entry_point("research")
    graph.add_edge("research", "aggregate")
    graph.add_edge("aggregate", "debate")
    graph.add_edge("debate", "supervise")
    graph.add_edge("supervise", END)
    state = graph.compile().invoke({"question": question, "status": "accepted"})
    return forecast_result(state)


# %% [markdown]
# ## Exercise both orchestrators with a failure fixture
#
# This explicitly synthetic client attempts an unsupported forecast. Neither
# orchestrator should produce an aggregate or final probability. No timing or
# accuracy claims are made from this test.


# %%
class UnsupportedForecastFixture:
    model_name = "synthetic-unsupported-forecast"

    def complete_with_usage(self, messages, json_mode=True):
        return '{"action":"forecast","p_yes":0.9,"rationale":"No evidence"}', TokenUsage()


for pipeline in (native_sdk_pipeline, langgraph_pipeline):
    result = pipeline(
        ForecastQuestion("Synthetic unsupported question?"), UnsupportedForecastFixture(), None
    )
    assert result.status == "abstained" and result.final_probability is None
    assert result.aggregation is None
    print(f"{pipeline.__name__}: {result.status}; {len(result.agents)} attempts retained")

# %% [markdown]
# ## CrewAI syntax example
#
# CrewAI uses agents, tasks and a process. The following configuration is shown as
# source, not executed. It has no research tools or trace integration and therefore
# cannot be compared as an evidence-backed forecasting pipeline. Implement those
# contracts before using its output as a researched forecast.

# %%
CREWAI_EXAMPLE = """from crewai import Agent, Task, Crew, Process
analyst = Agent(role="Research analyst", goal="Review supplied evidence", backstory="Forecasting analyst", llm=model)
task = Task(description=question_with_evidence, expected_output="Probability and source-linked rationale", agent=analyst)
crew = Crew(agents=[analyst], tasks=[task], process=Process.sequential)
# A live application would attach tools, validate outputs and retain actual messages.
"""
print(CREWAI_EXAMPLE)
print("CrewAI: syntax only; no probability, timing or performance measurement")

# %% [markdown]
# ## Inspect measured captures or run live
#
# The default table reports the original measured durations and probabilities
# without pretending to rerun today's implementation. A fresh live run retains full
# research, stage outputs, actual messages and timing. Different stochastic model
# calls are not a controlled test of framework effects.

# %%
rows, records = [], []
if RUN_LIVE:
    llm = create_llm_client(LLM_PROVIDER)
    search = create_search_client()
    question = get_chapter_contested_question()
    for label, pipeline in [
        ("Native Python", native_sdk_pipeline),
        ("LangGraph", langgraph_pipeline),
    ]:
        tracer = trace_llm(llm, label=label)
        started = time.perf_counter()
        result = pipeline(question, tracer, search)
        result.duration_seconds = time.perf_counter() - started
        run = RunTrace.from_result(
            result,
            notebook="10_framework_comparison",
            provider=llm.model_name,
            params={
                "variant": label,
                "n_agents": N_AGENTS,
                "max_steps": MAX_STEPS,
                "debate_rounds": DEBATE_ROUNDS,
                "supervisor_queries": SUPERVISOR_QUERIES,
                "statistical_correction_a": math.sqrt(3),
            },
            llm_calls=tracer.calls,
        )
        print(f"Saved {run.save().name}")
        records.append(run)
        rows.append(
            {
                "variant": label,
                "mode": run.execution_mode,
                "status": result.status,
                "duration_s": result.duration_seconds,
                "final_p": result.final_probability,
            }
        )
else:
    for label, filename in PINNED_TRACES.items():
        run = RunTrace.load(TRACES_DIR / filename)
        records.append(run)
        rows.append(
            {
                "variant": label,
                "mode": "replay",
                "status": "legacy_capture",
                "duration_s": run.params["elapsed_s"],
                "final_p": run.params["result"]["final_p"],
            }
        )
        print(f"{label}: {run.created_at}; {len(run.llm_calls)} actual captured model calls")
print(pl.DataFrame(rows))
print("CrewAI omitted: no inspectable measured capture")

# %%
print(replay_llm_calls(records[0].call_log()[:2], content_chars=600))
print("Full messages remain in the saved JSON; display excerpts are abbreviated.")

# %% [markdown]
# The statement count covers the orchestration layer, not the agent classes.
# CrewAI is the most verbose at the orchestration layer because each agent
# carries a role / goal / backstory triple and each task carries a
# description / expected_output / context list. LangGraph's overhead is
# the typed state schema and four short node functions; once the schema is
# defined, each node is a handful of statements. Native SDK is a single
# function.
#
# The two `final_p` values are not expected to match, and their gap measures nothing about
# orchestration: native and LangGraph run the same specialist classes over the same evidence
# rule, so what separates their numbers is that they are two samples from a stochastic model.
# That is the argument for comparing structure - statements written, where state lives, where a
# debugger attaches - rather than forecast agreement.
#
# ## Comparison matrix
#
# | Dimension | Native SDK | CrewAI | LangGraph |
# |---|---|---|---|
# | **Orchestration shape** | function call stack | role-based agents in a `Crew` | `StateGraph` with typed nodes |
# | **State location** | local variables | task outputs threaded by framework | explicit `TypedDict` |
# | **Parallelism** | manual (threads / asyncio) | opt-in asynchronous tasks | conditional / parallel edges |
# | **Checkpointing** | write it yourself, as in notebook 03 | not first-class | first-class (`BaseCheckpointSaver`) |
# | **LLM layer** | chapter `LLMClient` | LiteLLM (provider auto-detect) | passes through whatever the nodes use |
# | **Debug entry point** | the line that raised | inside CrewAI's loop | `app.get_state(thread)` snapshot |
# | **Full-trace capture** | one client wrapper | framework callbacks / verbose logs | one client wrapper |
# | **Visual debugging** | print / pdb | verbose logs | graph rendering + state inspection |
# | **Dep weight** | LLM SDK only | crewai + langchain transitives | langgraph + langchain transitives |
# | **Learning curve** | low (Python only) | medium (roles / tasks / processes) | medium-high (graph + state) |
#
# The differences that matter for a Chapter 24-shaped pipeline cluster in
# two columns: LangGraph exposes an explicit checkpoint abstraction, while
# native Python adds no orchestration dependency. Migration paths, observability stories, framework-momentum
# trade-offs, and a longer discussion of when persistence /
# parallelism / conditional-flow pressures justify a framework live in
# §24.5 of the book rather than in the notebook.
#
# ## Key Takeaways
#
# 1. **Start with plain Python for a pipeline this shape.** It asks the caller to write the
#    least orchestration and puts every failure on the line that caused it. That is structural
#    evidence about this pipeline, not a ranking of the frameworks.
# 2. **A framework earns its place against a named pressure, not a preference.** Crash recovery
#    mid-run is what LangGraph's checkpoint abstraction is for; named personas with distinct
#    charters are what CrewAI's role and goal slots are for. Without one of those pressures, the
#    dependency and the indirection are the whole trade.
# 3. **Ask where the run's state lives before adopting anything.** It decides what you can
#    inspect when a run goes wrong: local variables you can print, a framework's task outputs,
#    or a typed state object you can snapshot.
# 4. **A framework that owns the model client owns the observability.** CrewAI drives LiteLLM
#    directly, so nothing in this chapter's tracing sees its calls. That is why it appears here
#    as syntax rather than as a run: there was no capture to inspect, and the concrete cost of
#    the boundary moving is that there could not be one.
# 5. **Two probabilities from two runs are not a benchmark.** The native and LangGraph variants
#    share their internals, so what differs between their numbers is sampling, not
#    orchestration. Comparing frameworks needs a structural measure, which is what the
#    statement count and the state-location column are for.
#
# **Known limitations of what is built here.** One question, one run per variant, one model, and
# only two of the three variants execute at all. The statement count is a proxy for how much a
# caller writes and says nothing about how much there is to understand: a framework can trade
# statements for concepts. The elapsed times are single measurements of two different call paths
# on one machine, taken under the chapter's earlier aggregation, and the comparison matrix
# describes framework versions that move.
#
# **Next**: [`11_research_operator`](11_research_operator.ipynb) replays a
# production-shaped operator loop against a real case-study registry.
#
# **Book**: Section 24.5 covers migration paths, the persistence, parallelism and
# conditional-flow pressures that justify a framework, and the ecosystem-version note.
