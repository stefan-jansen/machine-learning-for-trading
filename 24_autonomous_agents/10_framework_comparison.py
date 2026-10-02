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
# # Framework comparison with inspectable runs
#
# Native Python and LangGraph below call the same research, mean aggregation,
# debate and supervisor functions. New live runs use the high-only supervisor rule
# and fixed $\sqrt{3}$ correction. The framework changes orchestration, not the
# evidence acceptance rule. This is a teaching implementation, not a forecasting
# performance benchmark.
#
# Default replay retains the genuine June Native and LangGraph captures, including
# actual model I/O and measured duration. Those captures used the earlier Neyman
# pipeline and are labeled legacy. CrewAI is a syntax example: the former fixed
# numbers had no inspectable capture and are removed from the comparison.
#
# **Book:** §24.5. **Prerequisites:** research, debate and forecasting pipeline.

# %%
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
# ## What the example establishes
#
# Both executable orchestrators use the same specialist contracts, and the synthetic
# failure check exercises the actual LangGraph graph offline. Genuine captured
# conversations support inspection of older runs. One run per framework does not
# establish a speed or accuracy ranking. Adopt a framework for concrete needs such
# as conditional execution or checkpointing, and verify those capabilities separately.
