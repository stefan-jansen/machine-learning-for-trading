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
# # The Research Agent
#
# **Docker image**: `ml4t`
#
# The first three notebooks built the pieces separately: a model client, a search tool with a
# cutoff and a source policy, and a state record with gates over it. This notebook assembles
# them into one object that takes a question and returns a probability with everything behind
# it attached, which is the unit every later notebook in the chapter operates on.
#
# **Learning Objectives**:
# - Assemble a model client, a search tool and a turn budget into one agent that answers a
#   question end to end
# - Parse and validate a model reply that arrives fenced, prefixed, or in the wrong shape, and
#   return it for correction instead of failing
# - Derive confidence, sentiment, evidence class and key findings from a forecast in code, so
#   that every agent's metadata is produced the same way
# - Tell an agent that committed to a probability apart from one that ran out of turns, and
#   keep the second out of any average
# - Read a saved run record and reproduce a forecast made months earlier with no API calls
#
# **Book Reference**: Chapter 24, Section 24.6 (Designing the research agent at the heart of
# the pipeline)
#
# **Prerequisites**: [`01_react_reasoning`](01_react_reasoning.ipynb) (providers),
# [`02_tool_contracts`](02_tool_contracts.ipynb) (tools),
# [`03_state_and_memory`](03_state_and_memory.ipynb) (state and gates).

# %%
import sys

from utils.paths import get_chapter_dir

sys.path.insert(0, str(get_chapter_dir(24)))

from agent_pipeline import validate_probabilities

"""The Research Agent — ReAct loop with rich output extraction."""

import json
import re
import warnings
from datetime import date

warnings.filterwarnings("ignore")

import polars as pl
from agent_fixtures import get_chapter_contested_question
from agent_observability import (
    TRACES_DIR,
    RunTrace,
    merge_calls,
    show_agent_timeline,
    trace_llm,
)
from agent_providers import ChatMessage, LLMClient, TokenUsage, create_llm_client
from agent_schemas import (
    AgentForecastArtifact,
    AgentTrace,
    EvidenceQuality,
    ForecastQuestion,
    SearchResult,
    Sentiment,
)
from agent_tools import (
    MockSearchClient,
    SearchClient,
    ToolExecutor,
    create_search_client,
    eligible_results,
    format_search_results,
)

# %% tags=["parameters"]
# RUN_LIVE=False (the default) replays the pinned 2026-06-09 trace named below:
# the notebook reloads that saved run and makes no API calls, so the agent
# outputs are stable and match the chapter. Set RUN_LIVE=True (with API keys) to
# forecast a current question live; that path produces different numbers.
RUN_LIVE = False
PINNED_TRACE = "04_research_agent_20260609T141730Z_b694ab4d0453.json"

LLM_PROVIDER = ""  # blank = read .env (LLM_PROVIDER); "mock" forces mock
MAX_STEPS = 5
MAX_SEARCH_RESULTS = 5

# %% [markdown]
# ## Prompt Templates
#
# The system prompt and step prompt define the agent's behavior. They are shown
# inline so readers can see exactly how the LLM is instructed. These are the same
# prompts used by the AIA Forecaster production system.

# %%
AGENT_SYSTEM_PROMPT = """\
You are a forecasting agent in a multi-agent forecasting system.

Your job:
1) Gather evidence by issuing web/news search queries when needed.
2) Then produce a binary probability forecast for the question.

You must follow the action schema exactly and output valid JSON only.
You must not browse prediction market prices unless they are explicitly provided.
Cite a retained source URL or title in your rationale. If evidence is insufficient,
return {"action":"abstain","rationale":"what is missing"}."""

# %% [markdown]
# ### Step prompt
#
# Provides the question, optional market context, and the action schema.
# The agent must output exactly one JSON action per step.


# %%
def build_step_prompt(
    question: ForecastQuestion,
    market_price: float | None = None,
    max_steps: int | None = None,
) -> str:
    """Format the step prompt with question context.

    When ``max_steps`` is given, the agent is told its search budget and is
    nudged to commit once it has enough evidence — searching is not free, and an
    agent that knows its budget stops reformulating the same query and forecasts.
    """
    prompt = f"QUESTION:\n{question.question}\n\n"
    if question.description:
        prompt += f"MARKET CONTEXT:\n{question.description}\n\n"
    if market_price is not None:
        prompt += f"MARKET IMPLIED PROBABILITY (p_yes):\n{market_price}\n\n"
    if max_steps is not None:
        prompt += (
            f"SEARCH BUDGET: you have at most {max_steps} steps. Search only when a "
            "specific, named fact is missing — do not reformulate a query you already "
            "ran. As soon as you have a base rate and the current signal, forecast.\n\n"
        )
    prompt += (
        "NEXT ACTION SCHEMA (output JSON only):\n"
        'If you need more info:\n{"action":"search","query":"..."}\n'
        "If you are ready to forecast:\n"
        '{"action":"forecast","p_yes":0.XX,"rationale":"short explanation '
        'grounded in evidence and base rates"}\n\n'
        "Pick exactly one action."
    )
    return prompt


# %% [markdown]
# ## JSON Parsing
#
# LLMs sometimes wrap JSON in markdown code blocks or add trailing text.
# Robust parsing tries `json.loads` first, then falls back to extracting
# the first `{...}` block.


# %%
def parse_json(raw: str) -> dict:
    """Parse JSON from LLM output with fallback bracket extraction."""
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        return (
            parsed
            if isinstance(parsed, dict)
            else {"action": "invalid", "error": "Expected JSON object"}
        )
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
        return (
            parsed
            if isinstance(parsed, dict)
            else {"action": "invalid", "error": "Expected JSON object"}
        )
    except json.JSONDecodeError:
        pass

    # Extract first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return (
                parsed
                if isinstance(parsed, dict)
                else {"action": "invalid", "error": "Expected JSON object"}
            )
        except json.JSONDecodeError:
            pass

    return {"action": "invalid", "error": "JSON parse failure"}


# %% [markdown]
# ## Rich Output Extraction
#
# After the agent produces a forecast, we extract additional metadata from the
# raw LLM output. These fields enrich the `AgentForecastArtifact` with
# confidence, sentiment, key findings, uncertainties, and evidence quality.

# %% [markdown]
# ### Confidence extraction
#
# If the forecast JSON includes a `confidence` field, we use it directly (clamped
# to $[0, 1]$). Otherwise, we infer confidence from probability extremity:
# $\text{confidence} = 2 \cdot |p_{\text{yes}} - 0.5|$.


# %%
def extract_confidence(action: dict) -> float:
    """Extract confidence from action dict or infer from probability."""
    if "confidence" in action:
        try:
            return max(0.0, min(1.0, float(action["confidence"])))
        except (TypeError, ValueError):
            pass
    p = float(action.get("p_yes", 0.5))
    return round(abs(p - 0.5) * 2, 3)


# %% [markdown]
# ### Sentiment extraction
#
# Maps $p_{\text{yes}}$ to a five-level sentiment scale.


# %%
def extract_sentiment(p_yes: float) -> Sentiment:
    """Infer sentiment from probability."""
    if p_yes > 0.8:
        return Sentiment.STRONGLY_BULLISH
    if p_yes > 0.6:
        return Sentiment.BULLISH
    if p_yes > 0.4:
        return Sentiment.NEUTRAL
    if p_yes > 0.2:
        return Sentiment.BEARISH
    return Sentiment.STRONGLY_BEARISH


# %% [markdown]
# ### Key findings and uncertainties
#
# Parses bullet points and numbered items from the rationale text, and identifies
# sentences containing uncertainty language.


# %%
def extract_key_findings(rationale: str) -> list[str]:
    """Extract bullet points and numbered items from rationale."""
    findings = []
    for line in rationale.split("\n"):
        line = line.strip()
        if re.match(r"^[-•*]\s+", line):
            findings.append(re.sub(r"^[-•*]\s+", "", line).strip())
        elif re.match(r"^\d+[.)]\s+", line):
            findings.append(re.sub(r"^\d+[.)]\s+", "", line).strip())
    return findings[:10]


# %% [markdown]
# ### Uncertainty extraction
#
# Identifies sentences containing uncertainty language in the rationale.


# %%
def extract_uncertainties(rationale: str) -> list[str]:
    """Identify sentences mentioning uncertainty."""
    uncertainty_words = {
        "uncertain",
        "unclear",
        "unknown",
        "risk",
        "caveat",
        "however",
        "but",
        "although",
    }
    sentences = re.split(r"[.!?]+", rationale)
    return [
        s.strip() for s in sentences if s.strip() and any(w in s.lower() for w in uncertainty_words)
    ][:5]


# %% [markdown]
# ### Evidence quality assessment
#
# Based on how many search queries the agent made and how many sources it consulted.


# %%
def assess_evidence_quality(sources_consulted: int, queries_made: int) -> EvidenceQuality:
    """Assess evidence quality from search metrics."""
    if sources_consulted >= 10 and queries_made >= 3:
        return EvidenceQuality.HIGH
    if sources_consulted >= 5 or queries_made >= 2:
        return EvidenceQuality.MEDIUM
    return EvidenceQuality.LOW


# %% [markdown]
# ## ReAct step handlers
#
# The ReAct loop processes one of three action types per step: a search,
# a forecast, or an unrecognised action. Extracting each branch as a small
# free function keeps the agent class focused on iteration and assembly.


# %%
# Check the returned sources, including supplied context.


# %% [markdown]
# ## The ResearchAgent Class
#
# The agent owns the LLM, the search executor, and the iteration budget.
# Its `run()` method walks the ReAct loop, dispatches each step to the
# handlers above, and returns an `AgentForecastArtifact` with the
# accumulated metadata. The class is reused unchanged in NB06 (multi-agent)
# and NB08 (full pipeline).


# %%
class ResearchAgent:
    """ReAct research agent producing evidence-linked probability forecasts.

    Implements the same loop as the AIA Forecaster's research agent:
    search for evidence, then forecast with rich metadata.
    """

    def __init__(
        self,
        llm: LLMClient,
        search: SearchClient | None = None,
        agent_id: str = "agent_0",
        max_steps: int = 6,
        max_search_results: int = 5,
    ) -> None:
        self.llm = llm
        self.executor = ToolExecutor(search=search)
        self.agent_id = agent_id
        self.max_steps = max_steps
        self.max_search_results = max_search_results

    def run(
        self,
        question: ForecastQuestion,
        market_price: float | None = None,
        evidence: list[SearchResult] | None = None,
    ) -> AgentForecastArtifact:
        """Research within the budget; retain unsupported attempts as abstentions."""
        if self.max_steps < 1 or self.max_search_results < 1:
            raise ValueError("Research budgets must be positive")
        cutoff = date.fromisoformat(question.cutoff_date) if question.cutoff_date else None
        messages = [
            ChatMessage(role="system", content=AGENT_SYSTEM_PROMPT),
            ChatMessage(
                role="user", content=build_step_prompt(question, market_price, self.max_steps)
            ),
        ]
        traces: list[AgentTrace] = []
        sources = eligible_results(evidence or [], cutoff)
        if sources:
            traces.append(AgentTrace(step=0, action="supplied_evidence", results=sources))
            messages.append(ChatMessage(role="user", content=format_search_results(sources)))
        total_tokens = TokenUsage()
        queries_made = 0
        p_yes = None
        rationale = ""
        raw_action = {}
        reason = "Budget exhausted without a supported forecast"
        status = "abstained"

        for step in range(1, self.max_steps + 2):
            synthesis_only = step > self.max_steps
            if synthesis_only:
                if not sources:
                    break
                messages.append(
                    ChatMessage(
                        role="user",
                        content=(
                            "Search budget exhausted. Use only the retained evidence to forecast, "
                            "citing its URL or title. If it is insufficient, abstain. No further search."
                        ),
                    )
                )
            try:
                response, usage = self.llm.complete_with_usage(messages, json_mode=True)
            except Exception as exc:
                reason = f"Model call failed ({type(exc).__name__})"
                status = "failed"
                traces.append(AgentTrace(step=step, action="provider_error", llm_raw=reason))
                break
            total_tokens = total_tokens + usage
            messages.append(ChatMessage(role="assistant", content=response))
            action = parse_json(response)
            action_type = action.get("action", "invalid")
            if action_type == "search" and not synthesis_only:
                query = action.get("query", "")
                results = self.executor.execute_search(query, self.max_search_results, cutoff)
                log = self.executor.execution_log[-1]
                queries_made += int(log.status in {"success", "error"})
                sources.extend(results)
                traces.append(
                    AgentTrace(
                        step=step, action="search", query=query, results=results, llm_raw=response
                    )
                )
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=(
                            format_search_results(results)
                            if results
                            else f"No eligible evidence. Search status: {log.status}. {log.result_preview}"
                        ),
                    )
                )
                continue
            traces.append(AgentTrace(step=step, action=str(action_type), llm_raw=response))
            if action_type == "abstain":
                reason = str(action.get("rationale") or "Model reported insufficient evidence")
                break
            if action_type == "forecast":
                try:
                    candidate = float(action["p_yes"])
                    validate_probabilities([candidate])
                except (KeyError, TypeError, ValueError):
                    reason = "Invalid probability; forecast rejected"
                else:
                    explanation = action.get("rationale")
                    if not sources:
                        reason = "No eligible evidence; forecast rejected"
                    elif not isinstance(explanation, str) or not explanation.strip():
                        reason = "Missing rationale; forecast rejected"
                    elif not any(
                        (r.url and r.url in explanation) or (r.title and r.title in explanation)
                        for r in sources
                    ):
                        reason = "Rationale must identify a retained source URL or title"
                    else:
                        p_yes, rationale, raw_action = candidate, explanation, action
                        status = "accepted"
                        break
            else:
                reason = "Invalid action or search requested after the budget was exhausted"
            messages.append(ChatMessage(role="tool", content=reason))

        return AgentForecastArtifact(
            agent_id=self.agent_id,
            p_yes=p_yes,
            rationale=rationale,
            traces=traces,
            status=status,
            failure_reason=None if status == "accepted" else reason,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            execution_mode=(
                "synthetic"
                if isinstance(self.executor.search, MockSearchClient)
                or any(
                    label in getattr(self.llm, "model_name", "").lower()
                    for label in ("mock", "synthetic")
                )
                else "live"
            ),
            confidence=extract_confidence(raw_action) if p_yes is not None else 0.0,
            sentiment=extract_sentiment(p_yes) if p_yes is not None else Sentiment.NEUTRAL,
            key_findings=extract_key_findings(rationale),
            evidence_quality=assess_evidence_quality(len(sources), queries_made),
            uncertainties=extract_uncertainties(rationale),
            token_usage=total_tokens,
            search_queries_made=queries_made,
            sources_consulted=len(sources),
        )


# %% [markdown]
# ## Running the Research Agent
#
# We run the agent on the pinned `CHAPTER_CONTESTED_QUESTION` from
# `agent_fixtures.py` (*"Will the Federal Reserve hike rates in 2026?"*, where
# credible evidence cuts both ways) and inspect the full `AgentForecastArtifact`.
# The whole forecasting arc forecasts this same pinned question (NB07 debates it,
# NB08 runs it through the full pipeline, NB10 ports it across frameworks); NB06
# uses the companion `CHAPTER_CLEAR_QUESTION`, a one-directional question on which
# the agents instead agree. The numbers are a dated point-in-time capture
# (provider `claude-sonnet-4`, Tavily search, 2026-06-09). By default the notebook
# *replays* that pinned run (`RUN_LIVE = False`): it reloads the saved artifacts
# and raw conversation and makes no API calls, so the outputs are stable. Set
# `RUN_LIVE = True` (with `LLM_PROVIDER` + `LLM_API_KEY` + `TAVILY_API_KEY` in `.env`) to forecast a
# current question live, which is not reproducible.

# %%
if RUN_LIVE:
    llm = create_llm_client(LLM_PROVIDER)
    search = create_search_client()

    # Wrap the client in a TracingLLMClient so every prompt and raw response is
    # captured for the run trace we persist at the end of the notebook.
    tracer_0 = trace_llm(llm, label="agent_0")
    agent = ResearchAgent(
        llm=tracer_0,
        search=search,
        agent_id="agent_0",
        max_steps=MAX_STEPS,
        max_search_results=MAX_SEARCH_RESULTS,
    )

    question = get_chapter_contested_question()
    provider_name = llm.model_name
    search_name = type(search).__name__
    artifact = agent.run(question)
else:
    # Replay: reload the pinned trace and rehydrate its two saved agents. The
    # first is the single-agent forecast inspected here; the second is the
    # preview pair shown later. Display cells consume these unchanged.
    pinned_run = RunTrace.load(TRACES_DIR / PINNED_TRACE)
    question = pinned_run.question_obj()
    provider_name = pinned_run.provider
    search_name = "replay (pinned trace)"
    replayed_artifacts = pinned_run.agent_artifacts()
    artifact = replayed_artifacts[0]

print(f"Mode: {'LIVE' if RUN_LIVE else 'REPLAY (pinned 2026-06-09 trace)'}")
print(f"Provider: {provider_name}")
print(f"Search: {search_name}")
print(f"Question: {question.question}\n")

# %% [markdown]
# ## Inspecting the Forecast Artifact
#
# The `AgentForecastArtifact` captures everything about the agent's run —
# the probability, reasoning, evidence trail, and metadata.

# %%
print("=== Forecast ===")
print(f"Agent:       {artifact.agent_id}")
print(f"Status:      {artifact.status}")
print(f"p(YES):      {artifact.p_yes}")
if artifact.failure_reason:
    print(f"Reason:      {artifact.failure_reason}")
print(f"Confidence:  {artifact.confidence:.2f}")
print(f"Sentiment:   {artifact.sentiment.value}")
print(f"Evidence:    {artifact.evidence_quality.value}")
print(f"Queries:     {artifact.search_queries_made}")
print(f"Sources:     {artifact.sources_consulted}")
print(f"Tokens:      {artifact.token_usage.total_tokens:,}")

# %%
print(f"\nRationale:\n{artifact.rationale[:400]}")

# %%
if artifact.key_findings:
    print("\nKey Findings:")
    for f in artifact.key_findings:
        print(f"  • {f}")

if artifact.uncertainties:
    print("\nUncertainties:")
    for u in artifact.uncertainties:
        print(f"  • {u}")

# %% [markdown]
# ## Execution Trace
#
# Every search query and its results are captured in the trace. `show_agent_timeline`
# from `agent_observability` renders the whole run in order — each query, the
# documents it returned (title, date, URL, and a snippet), and the forecast with
# the untruncated rationale — so the agent's path from question to probability is
# fully visible. This is the per-agent observability view reused across NB06–NB08.

# %%
print(show_agent_timeline(artifact))

# %% [markdown]
# ## Tool Execution Audit
#
# The executor's independent log captures timing and provenance for every
# search call. Rendering it as a Polars DataFrame puts the query / status /
# duration in three sortable columns rather than a hand-aligned string
# table — the same audit data, in a form that downstream analysis code can
# read without parsing. (A live run also records per-call `duration_ms`; the
# replay path reconstructs the same query/status/result-count audit from the
# saved traces, since wall-clock timing is not part of the persisted record.)

# %%
if RUN_LIVE:
    audit_df = (
        pl.DataFrame(
            [
                {
                    "query": entry.args.get("query", "?"),
                    "status": entry.status,
                    "duration_ms": round(entry.duration_ms, 1),
                }
                for entry in agent.executor.execution_log
            ]
        )
        if agent.executor.execution_log
        else pl.DataFrame({"query": [], "status": [], "duration_ms": []})
    )
else:
    # Replay: reconstruct the search audit from the saved agent traces.
    search_steps = [t for t in artifact.traces if t.action == "search"]
    audit_df = (
        pl.DataFrame(
            [
                {
                    "query": t.query or "?",
                    "status": "ok",
                    "results": len(t.results),
                }
                for t in search_steps
            ]
        )
        if search_steps
        else pl.DataFrame({"query": [], "status": [], "results": []})
    )
audit_df

# %% [markdown]
# ## Agent Summary Format
#
# When multiple agents run in parallel (NB06), their outputs are summarized
# for the supervisor and debate stages. This is the format used downstream.


# %%
def format_agent_summary(a: AgentForecastArtifact) -> str:
    """Format an agent artifact as a summary for supervisor/debate prompts."""
    lines = [
        f"Agent: {a.agent_id}",
        f"Probability (p_yes): {a.p_yes}",
        f"Status: {a.status}",
        f"Confidence: {a.confidence:.2f}",
        f"Rationale: {a.rationale}",
    ]
    if a.key_findings:
        lines.append("Key findings:")
        for f in a.key_findings[:3]:
            lines.append(f"  - {f}")
    for trace in a.traces:
        for source in trace.results:
            lines.append(f"Source: {source.title} | {source.url}")
            lines.append(f"Excerpt: {(source.snippet or '')[:400]}")
    return "\n".join(lines)


# %%
print(format_agent_summary(artifact))

# %% [markdown]
# ## Running Multiple Agents
#
# A preview of the multi-agent notebooks: the same agent class with different
# IDs, run on this *contested* question, can already produce different forecasts,
# driven by which evidence each agent emphasises. That spread is real but
# question-dependent. NB06 runs the same class on the one-directional
# `CHAPTER_CLEAR_QUESTION` and the agents instead agree. The contrast is why the
# later notebooks add debate (NB07) and a role-specialized pipeline (NB08) for
# contested questions rather than relying on temperature alone.

# %%
if RUN_LIVE:
    tracer_1 = trace_llm(llm, label="agent_1")
    agent_b = ResearchAgent(llm=tracer_1, search=search, agent_id="agent_1", max_steps=MAX_STEPS)
    artifact_b = agent_b.run(question)
else:
    # Replay: the pinned trace's second saved agent is this preview run.
    artifact_b = replayed_artifacts[1]

for a in [artifact, artifact_b]:
    print(f"{a.agent_id}: status={a.status}, p_yes={a.p_yes}")
if artifact.p_yes is not None and artifact_b.p_yes is not None:
    print(f"Difference: {abs(artifact.p_yes - artifact_b.p_yes):.2f}")

# %% [markdown]
# **Interpretation**: The captured agents searched and weighted evidence differently.
# Their probability spread can reflect retrieval, sampling, prompts and model biases.
# These two runs do not isolate those effects or establish independent information.
# Notebooks 06 and 07 inspect an ensemble and debate on other questions;
# resolved comparable runs are needed to assess whether either improves forecasting.

# %% [markdown]
# ## Persisting the Run Trace
#
# The artifact holds the structured forecast; the `TracingLLMClient`s wrapped
# around each agent hold the raw conversation. `RunTrace.capture` bundles both —
# the question, both agents' artifacts, and every prompt/response — into one JSON
# record under `forecast_traces/`, the same auditable format the multi-agent
# notebooks (NB06–NB08) write. Reload it with `RunTrace.load` to replay exactly
# what the agent saw and said — which is what the default `RUN_LIVE = False` path
# does above. A live run writes a fresh trace here; the default replay run reports
# the pinned trace it loaded rather than overwriting it.

# %%
if RUN_LIVE:
    run = RunTrace.capture(
        notebook="04_research_agent",
        provider=provider_name,
        question=question,
        params={"max_steps": MAX_STEPS, "max_search_results": MAX_SEARCH_RESULTS},
        agents=[artifact, artifact_b],
        notes="Single research agent plus a two-agent preview on the contested question.",
        llm_calls=merge_calls(tracer_0, tracer_1),
    )
    trace_path = run.save()
    print(
        f"Saved {len(run.llm_calls)} model calls "
        f"({run.total_tokens():,} tokens) → {trace_path.relative_to(trace_path.parents[1])}"
    )
else:
    # Replay: report the pinned trace we loaded rather than writing a new file.
    run = pinned_run
    trace_path = TRACES_DIR / PINNED_TRACE
    print(
        f"Replayed {len(run.llm_calls)} model calls "
        f"({run.total_tokens():,} tokens) from {trace_path.name}"
    )


# %% [markdown]
# ## Key Takeaways
#
# 1. **ResearchAgent** combines the ReAct loop with rich output extraction — every
#    forecast comes with confidence, sentiment, evidence quality, and uncertainties
# 2. **Structured artifacts**: `AgentForecastArtifact` captures the complete run —
#    probability, reasoning, traces, and token usage
# 3. **Robust parsing**: JSON extraction handles markdown code blocks and malformed
#    LLM output gracefully
# 4. **Reusable**: This class is the building block for multi-agent (NB06),
#    debate (NB07), and full pipeline (NB08) notebooks
# 5. **Token tracking**: Every LLM call is metered for cost analysis
#
# **Next**: [`aggregation`](05_aggregation_math.ipynb) — the mathematical foundation for combining
# multiple probability estimates (Neyman extremization, weighted aggregation).
#
# **Book**: Section 24.6 discusses agent design patterns, including the trade-off
# between agent complexity and forecast calibration.

# %% [markdown]
# ### Failure test: a probability without evidence
#
# This scripted model deliberately tries to forecast before acquiring evidence. The actual research loop must retain the attempt and abstain. No live model or invented performance result is involved.


# %%
class UnsupportedForecastFixture:
    model_name = "synthetic-failure-fixture"

    def complete_with_usage(self, messages, **kwargs):
        return '{"action":"forecast","p_yes":0.9,"rationale":"No sources consulted"}', TokenUsage()


unsupported = ResearchAgent(UnsupportedForecastFixture(), max_steps=1).run(
    ForecastQuestion("Deliberately unsupported teaching forecast")
)
assert unsupported.status == "abstained" and unsupported.p_yes is None
print(unsupported.status, unsupported.failure_reason)

# %% [markdown]
# ## Fresh verification of the current loop
#
# This October 2, 2026 capture uses genuine Sonnet 4.6 and Tavily calls on an
# unresolved October FOMC question. Inspect its retained evidence and model I/O
# alongside the original June capture above. The forecast has no outcome score;
# acceptance means a valid probability, rationale and eligible cited evidence.

# %%
verified_run = RunTrace.load(TRACES_DIR / "verified_research_20261002T193656Z_f169a52a0918.json")
print(verified_run.question_obj().question)
print(f"Status: {verified_run.status}; p={verified_run.final_probability}")
print(f"Model calls: {len(verified_run.llm_calls)}; tokens: {verified_run.total_tokens():,}")
print(verified_run.agent_artifacts()[0].rationale)
print(show_agent_timeline(verified_run.agent_artifacts()[0]))
