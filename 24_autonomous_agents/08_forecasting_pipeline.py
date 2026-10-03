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
# # Full Forecasting Pipeline
#
# **Docker image**: `ml4t`
#
# The last four notebooks each built one piece. This one runs them as a sequence: a panel of
# research agents, an aggregate over them, a debate between two sides of the aggregate, and a
# supervisor that reads the panel, searches where the agents disagreed, and decides whether to
# override. That is the **AIA Forecaster** architecture, and the point of assembling it is not
# that four stages beat one; it is that a forecast produced this way can be taken apart when it
# is wrong.
#
# Both questions the pipeline runs on were unresolved when the capture was taken. It produces
# probabilities and cannot be scored, and neither can any claim that a later stage improved on
# an earlier one. Scoring against known outcomes is
# [`09_evaluation_and_governance`](09_evaluation_and_governance.ipynb).
#
# **Learning Objectives**:
# - Build a supervisor that finds where a panel disagreed, searches on those points, and
#   returns its own probability
# - Gate that supervisor's influence on the confidence it states, so it can adjust the ensemble
#   without silently replacing it
# - Compose four stages into one class whose settings are all declared in one place
# - Read a stage-by-stage probability path and see which stage moved the answer
# - Record a run in a form a scoring pipeline could later consume
#
# **Book Reference**: Chapter 24, Sections 24.7 (Multi-agent forecasting systems) and 24.9
# (Preparing for production)
#
# **Prerequisites**: [`04_research_agent`](04_research_agent.ipynb),
# [`05_aggregation_math`](05_aggregation_math.ipynb),
# [`06_multi_agent_research`](06_multi_agent_research.ipynb),
# [`07_adversarial_debate`](07_adversarial_debate.ipynb).

# %%
"""Full Forecasting Pipeline - agent-debate-supervisor end-to-end."""

import sys

from utils.paths import get_chapter_dir

sys.path.insert(0, str(get_chapter_dir(24)))

import math
import re
import time
from datetime import date

import polars as pl
from agent_fixtures import get_chapter_clear_question, get_chapter_contested_question
from agent_observability import (
    TRACES_DIR,
    RunTrace,
    show_agents,
    show_debate_transcript,
    show_supervisor,
    trace_llm,
)
from agent_pipeline import logodds_extremize, validate_probabilities
from agent_providers import ChatMessage, TokenUsage, create_llm_client
from agent_research import ResearchAgent, format_agent_summary, parse_json
from agent_schemas import (
    AgentForecastArtifact,
    AggregationResult,
    DebateArtifact,
    DebateRound,
    ForecastQuestion,
    ForecastResult,
    SearchResult,
    SupervisorArtifact,
)
from agent_tools import (
    SearchClient,
    ToolExecutor,
    create_search_client,
)

# %% tags=["parameters"]
# Default replay includes a fresh capture of this implementation and two original
# June runs. Each retains its question, evidence and model I/O. No API calls are
# made during replay. Set RUN_LIVE=True for a new run with configured providers.
RUN_LIVE = False
PINNED_TRACES = [
    "verified_pipeline_20261002T193817Z_e82b1cdb9bbb.json",  # current implementation
    "08_forecasting_pipeline_20260609T141954Z_ef5bca7b95bd.json",  # recession (clear)
    "08_forecasting_pipeline_20260609T142158Z_24e083e7fe54.json",  # rate hike (contested)
]

LLM_PROVIDER = ""  # Blank reads .env; "mock" selects a synthetic fixture
N_AGENTS = 3
DEBATE_ROUNDS = 3
MAX_STEPS = 5
MAX_SEARCH_RESULTS = 5

# %% [markdown]
# The finalize prompt receives the original question, agent panel, and bounded
# follow-up evidence. It requests a probability, confidence label, and rationale.

# %%
SUPERVISOR_DISAGREEMENTS_PROMPT = """\
You are the SUPERVISOR agent.

You receive M agent forecasts and rationales for the same question.
Your job is NOT to average them directly.

Step 1: Identify key disagreements, ambiguities, missing base rates, or claims that should be fact-checked.
Step 2: Propose up to {max_queries} clarifying search queries that would resolve these disagreements.

Output JSON only with:
{{"disagreements": ["..."], "queries": ["..."]}}

AGENT INPUTS:
{agent_summaries}"""

SUPERVISOR_FINALIZE_PROMPT = """\
You are the SUPERVISOR agent.

Given:
1) The original question
2) The set of agent forecasts and rationales
3) Additional evidence from your follow-up searches

You must output:
1) Updated forecast p_yes in [0,1]
2) Confidence in whether your update direction is correct: "high" | "medium" | "low"
3) A short rationale citing a source URL from the supplied evidence

Output JSON only:
{{"p_yes": 0.0, "confidence": "high", "rationale": "..."}}

QUESTION:
{question}

AGENT INPUTS:
{agent_summaries}

SUPERVISOR SEARCH EVIDENCE:
{supervisor_evidence}"""


# %% [markdown]
# ## Supervisor phase helpers
#
# The supervisor's three phases (identify, search, finalize) factor
# naturally into free functions. The class then becomes a thin shell that
# owns the LLM + search clients and threads them through the helpers.


# %%
def _supervisor_identify_disagreements(
    llm,
    agent_summaries: str,
    max_queries: int,
    artifact: SupervisorArtifact,
) -> tuple[list[str], list[str], TokenUsage]:
    """Phase 1: LLM call asking for disagreements and clarifying queries."""
    prompt = SUPERVISOR_DISAGREEMENTS_PROMPT.format(
        max_queries=max_queries, agent_summaries=agent_summaries
    )
    raw, tokens = llm.complete_with_usage(
        [ChatMessage(role="user", content=prompt)], json_mode=True
    )
    artifact.token_usage = artifact.token_usage + tokens
    parsed = parse_json(raw)
    disagreements = [str(x) for x in parsed.get("disagreements", [])][:20]
    queries = [str(x) for x in parsed.get("queries", [])][:max_queries]
    return disagreements, queries, tokens


def _supervisor_run_searches(
    search,
    queries: list[str],
    max_search_results: int,
    cutoff_date: date | None,
) -> dict[str, list[SearchResult]]:
    """Phase 2: execute the clarifying-search queries via ToolExecutor."""
    if search is None:
        return {}
    executor = ToolExecutor(search=search)
    return {
        q: executor.execute_search(q, max_results=max_search_results, cutoff_date=cutoff_date)
        for q in queries
    }


def _format_supervisor_evidence(sr: dict[str, list[SearchResult]]) -> str:
    """Render the search-result dict into the supervisor-finalize prompt's evidence block."""
    lines: list[str] = []
    for q, results in sr.items():
        lines.append(f"QUERY: {q}")
        for i, r in enumerate(results, start=1):
            lines.append(f"{i}. {r.title}")
            if r.url:
                lines.append(f"   URL: {r.url}")
            if r.snippet:
                lines.append(f"   {r.snippet}")
            if r.published:
                lines.append(f"   Published: {r.published}")
        lines.append("")
    return "\n".join(lines) if lines else "No additional search evidence."


def _supervisor_finalize(
    llm,
    question: str,
    agent_summaries: str,
    search_results: dict[str, list[SearchResult]],
    artifact: SupervisorArtifact,
) -> tuple[float | None, str | None, str | None, TokenUsage]:
    """Phase 3: LLM call asking for final p_yes / confidence / rationale."""
    evidence_text = _format_supervisor_evidence(search_results)
    prompt = SUPERVISOR_FINALIZE_PROMPT.format(
        question=question,
        agent_summaries=agent_summaries,
        supervisor_evidence=evidence_text,
    )
    raw, tokens = llm.complete_with_usage(
        [ChatMessage(role="user", content=prompt)], json_mode=True
    )
    artifact.token_usage = artifact.token_usage + tokens
    parsed = parse_json(raw)
    confidence = str(parsed.get("confidence", "")).lower()
    rationale = parsed.get("rationale")
    try:
        if isinstance(parsed["p_yes"], bool):
            raise ValueError("Probability must be a number, not a boolean")
        p_yes = float(parsed["p_yes"])
        validate_probabilities([p_yes])
        references = re.findall(r"https?://[^\s|]+", agent_summaries + "\n" + evidence_text)
        if not isinstance(rationale, str) or not any(
            ref.rstrip(".,)") in rationale for ref in references
        ):
            raise ValueError("Supervisor must cite retained evidence")
        if confidence not in ("high", "medium", "low"):
            raise ValueError("Invalid confidence")
    except (KeyError, TypeError, ValueError):
        return None, "low", "Supervisor output rejected; retain the ensemble", tokens
    return p_yes, confidence, rationale, tokens


# %% [markdown]
# ## The SupervisorAgent Class
#
# Three phases: (1) identify disagreements, (2) run clarifying searches,
# (3) finalize with evidence. The supervisor only overrides the ensemble
# when its confidence is "high", preserving agent diversity by default.


# %%
class SupervisorAgent:
    """Supervisor that reconciles agent ensemble via clarifying searches."""

    def __init__(
        self,
        llm,
        search: SearchClient | None = None,
        max_queries: int = 3,
        max_search_results: int = 5,
    ) -> None:
        self.llm = llm
        self.search = search
        self.max_queries = max_queries
        self.max_search_results = max_search_results
        self.token_usage = TokenUsage()

    def run(
        self,
        question: str,
        agent_summaries: str,
        cutoff_date: date | None = None,
    ) -> SupervisorArtifact:
        """Run supervisor reconciliation. Returns SupervisorArtifact."""
        self.token_usage = TokenUsage()
        self.artifact = SupervisorArtifact()
        disagreements, queries, t1 = _supervisor_identify_disagreements(
            self.llm, agent_summaries, self.max_queries, self.artifact
        )
        self.token_usage = self.token_usage + t1
        self.artifact.token_usage = self.token_usage
        self.artifact.disagreements = disagreements
        self.artifact.queries = queries

        search_results = _supervisor_run_searches(
            self.search, queries, self.max_search_results, cutoff_date
        )

        self.artifact.search_results = search_results

        p_yes, confidence, rationale, t2 = _supervisor_finalize(
            self.llm, question, agent_summaries, search_results, self.artifact
        )
        self.token_usage = self.token_usage + t2

        self.artifact.p_yes = p_yes
        self.artifact.confidence = confidence
        self.artifact.rationale = rationale
        self.artifact.token_usage = self.token_usage
        return self.artifact


# %% [markdown]
# ## Debate Prompts: Bull (from NB07)
#
# Deliberately duplicated from NB07 for self-contained teaching. In
# production these would live in a shared config module.

# %%
BULL_PROMPT_TEMPLATE = """\
You are the BULL debater in a structured forecasting debate.

Your role is to argue for a HIGHER probability of YES for the question below.
You must present the strongest possible case for YES, backed by evidence.

QUESTION:
{question}

AGENT SUMMARIES:
{agent_summaries}

CURRENT AGGREGATE PROBABILITY: {aggregate_p_yes}

{bear_section}

Output JSON only:
{{"argument": "Your strongest case for a higher probability of YES", "p_yes": 0.XX, "key_evidence": ["evidence point 1", "evidence point 2", "evidence point 3"]}}"""


# %% [markdown]
# ## Debate Prompts: Bear (from NB07)

# %%
BEAR_PROMPT_TEMPLATE = """\
You are the BEAR debater in a structured forecasting debate.

Your role is to argue for a LOWER probability of YES for the question below.
You must present the strongest possible case for NO (or lower probability), backed by evidence.

QUESTION:
{question}

AGENT SUMMARIES:
{agent_summaries}

CURRENT AGGREGATE PROBABILITY: {aggregate_p_yes}

BULL'S ARGUMENT:
{bull_argument}
Bull's probability: {bull_probability}

You must directly address the Bull's points and explain why the probability should be lower.

Output JSON only:
{{"argument": "Your strongest case for a lower probability of YES", "p_yes": 0.XX, "key_evidence": ["evidence point 1", "evidence point 2", "evidence point 3"]}}"""


# %% [markdown]
# ## Pipeline helpers
#
# Run research agents without market-price anchoring, optionally debate their
# retained evidence, and apply the high-confidence supervisor rule followed by
# fixed log-odds scaling. The debate consensus rule is an example we chose;
# it has no demonstrated performance benefit here.


# %%
def _run_research_agents(
    llm,
    search,
    question: ForecastQuestion,
    n_agents: int,
    max_steps: int,
) -> list[AgentForecastArtifact]:
    """Phase 1: run N identical ResearchAgents and collect their artifacts."""
    artifacts: list[AgentForecastArtifact] = []
    for i in range(n_agents):
        agent = ResearchAgent(llm=llm, search=search, agent_id=f"agent_{i}", max_steps=max_steps)
        artifacts.append(agent.run(question, market_price=None))
    return artifacts


def _run_pipeline_debate(
    llm,
    question: str,
    agent_summaries: str,
    aggregate_p_yes: float,
    debate_rounds: int,
    consensus_threshold: float,
    artifact: DebateArtifact | None = None,
) -> DebateArtifact:
    """Phase 3: run bull/bear debate rounds against the pre-debate aggregate."""
    artifact = artifact if artifact is not None else DebateArtifact()
    rounds = artifact.rounds
    bear_argument: str | None = None
    bear_probability: float | None = None

    for round_num in range(1, debate_rounds + 1):
        bear_section = ""
        if bear_argument is not None:
            bear_section = (
                f"BEAR'S PREVIOUS ARGUMENT:\n{bear_argument}\n"
                f"Bear's probability: {bear_probability:.4f}\n\n"
                "You must directly address the Bear's points."
            )
        bull_prompt = BULL_PROMPT_TEMPLATE.format(
            question=question,
            agent_summaries=agent_summaries,
            aggregate_p_yes=f"{aggregate_p_yes:.4f}",
            bear_section=bear_section,
        )
        bull_raw, bull_tokens = llm.complete_with_usage(
            [ChatMessage(role="user", content=bull_prompt)], json_mode=True
        )
        artifact.token_usage = artifact.token_usage + bull_tokens
        bull_parsed = parse_json(bull_raw)
        bull_argument = bull_parsed.get("argument", "")
        if isinstance(bull_parsed["p_yes"], bool):
            raise ValueError("Probability must be a number, not a boolean")
        bull_p = float(bull_parsed["p_yes"])
        validate_probabilities([bull_p])
        bull_evidence = [str(e) for e in bull_parsed.get("key_evidence", [])]

        bear_prompt = BEAR_PROMPT_TEMPLATE.format(
            question=question,
            agent_summaries=agent_summaries,
            aggregate_p_yes=f"{aggregate_p_yes:.4f}",
            bull_argument=bull_argument,
            bull_probability=f"{bull_p:.4f}",
        )
        bear_raw, bear_tokens = llm.complete_with_usage(
            [ChatMessage(role="user", content=bear_prompt)], json_mode=True
        )
        artifact.token_usage = artifact.token_usage + bear_tokens
        bear_parsed = parse_json(bear_raw)
        bear_argument = bear_parsed.get("argument", "")
        if isinstance(bear_parsed["p_yes"], bool):
            raise ValueError("Probability must be a number, not a boolean")
        bear_probability = float(bear_parsed["p_yes"])
        validate_probabilities([bear_probability])
        bear_evidence = [str(e) for e in bear_parsed.get("key_evidence", [])]

        consensus = abs(bull_p - bear_probability) < consensus_threshold
        rounds.append(
            DebateRound(
                round_number=round_num,
                bull_argument=bull_argument,
                bull_probability=bull_p,
                bear_argument=bear_argument,
                bear_probability=bear_probability,
                consensus_reached=consensus,
                bull_key_evidence=bull_evidence,
                bear_key_evidence=bear_evidence,
            )
        )
        artifact.bull_final_probability = bull_p
        artifact.bear_final_probability = bear_probability
        artifact.consensus_reached = consensus
        artifact.early_termination = consensus and len(rounds) < debate_rounds
        if consensus:
            break

    return artifact


def _blend_final_probability(
    post_debate: float,
    supervisor_artifact: SupervisorArtifact,
) -> tuple[float, float]:
    """Phase 4 → final: confidence-gated supervisor override. Returns (final_p, final_confidence)."""
    final_p = post_debate
    final_confidence = 0.5
    if supervisor_artifact.p_yes is not None and supervisor_artifact.confidence == "high":
        final_p = supervisor_artifact.p_yes
        final_confidence = 0.8
    validate_probabilities([final_p])
    # Fixed statistical correction from the paper, not a coefficient fitted here.
    return logodds_extremize(final_p, math.sqrt(3)), final_confidence


# %% [markdown]
# ## The AIAForecaster class
#
# The central sequence remains directly readable: research, mean, optional debate,
# supervisor, fixed scaling. Abstentions stay in the artifacts and do not enter the
# mean. If all researchers abstain, there is no final probability. These examples
# run researchers sequentially; the application also supports parallel execution.


# %%
class AIAForecaster:
    """Complete AIA Forecaster pipeline: agents → aggregate → debate → supervisor."""

    def __init__(
        self,
        llm,
        search: SearchClient | None = None,
        n_agents: int = 3,
        max_steps: int = 5,
        debate_rounds: int = 3,
        consensus_threshold: float = 0.05,
    ) -> None:
        self.llm = llm
        self.search = search
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.debate_rounds = debate_rounds
        self.consensus_threshold = consensus_threshold

    def forecast(self, question: ForecastQuestion) -> ForecastResult:
        """Run the full pipeline on a single question."""
        start_time = time.time()
        total_tokens = TokenUsage()
        cutoff = date.fromisoformat(question.cutoff_date) if question.cutoff_date else None

        # Phase 1: research agents
        artifacts = _run_research_agents(
            self.llm, self.search, question, self.n_agents, self.max_steps
        )
        for a in artifacts:
            total_tokens = total_tokens + a.token_usage
        accepted = [a for a in artifacts if a.status == "accepted" and a.p_yes is not None]
        if not accepted:
            return ForecastResult(
                question=question,
                agents=artifacts,
                status="abstained",
                failure_reason="No accepted research forecasts",
                total_token_usage=total_tokens,
                duration_seconds=round(time.time() - start_time, 2),
            )
        probs = [a.p_yes for a in accepted]
        agent_summaries = "\n\n---\n\n".join(format_agent_summary(a) for a in accepted)

        # Phase 2: aggregation
        validate_probabilities(probs)
        agg_p = sum(probs) / len(probs)
        aggregation = AggregationResult(
            method="mean",
            raw_probability=agg_p,
            input_probabilities=probs,
        )

        # Optional teaching adaptation: debate. An invalid stage leaves a failed attempt.
        debate_artifact = None
        supervisor_artifact = None
        supervisor = None
        try:
            post_debate = agg_p
            if self.debate_rounds:
                debate_artifact = DebateArtifact()
                debate_artifact = _run_pipeline_debate(
                    self.llm,
                    question.question,
                    agent_summaries,
                    agg_p,
                    self.debate_rounds,
                    self.consensus_threshold,
                    artifact=debate_artifact,
                )
                total_tokens = total_tokens + debate_artifact.token_usage
                if debate_artifact.consensus_reached:
                    post_debate = (
                        debate_artifact.bull_final_probability
                        + debate_artifact.bear_final_probability
                    ) / 2

            supervisor = SupervisorAgent(
                llm=self.llm,
                search=self.search,
                max_queries=3,
            )
            supervisor_artifact = supervisor.run(
                question=question.question, agent_summaries=agent_summaries, cutoff_date=cutoff
            )
            total_tokens = total_tokens + supervisor_artifact.token_usage
        except Exception as exc:
            supervisor_artifact = supervisor.artifact if supervisor is not None else None
            total_tokens = TokenUsage()
            for artifact in [*artifacts, debate_artifact, supervisor_artifact]:
                if artifact is not None:
                    total_tokens = total_tokens + artifact.token_usage
            return ForecastResult(
                question=question,
                agents=artifacts,
                aggregation=aggregation,
                debate=debate_artifact,
                supervisor=supervisor_artifact,
                status="failed",
                failure_reason=str(exc),
                total_token_usage=total_tokens,
                duration_seconds=round(time.time() - start_time, 2),
            )

        # Final: confidence-gated override
        final_p, final_confidence = _blend_final_probability(post_debate, supervisor_artifact)
        duration = time.time() - start_time

        return ForecastResult(
            question=question,
            agents=artifacts,
            aggregation=aggregation,
            debate=debate_artifact,
            supervisor=supervisor_artifact,
            final_probability=round(final_p, 4),
            final_confidence=round(final_confidence, 3),
            total_token_usage=total_tokens,
            duration_seconds=round(duration, 2),
        )


# %% [markdown]
# ## The Two Questions
#
# The pipeline runs end to end on both of the chapter's pinned questions:
# `CHAPTER_CLEAR_QUESTION`, the recession question where the research agents landed close
# together in [`06_multi_agent_research`](06_multi_agent_research.ipynb), and
# `CHAPTER_CONTESTED_QUESTION`, the rate-hike question where they spread out in
# [`07_adversarial_debate`](07_adversarial_debate.ipynb). Running both shows what each stage
# does when the panel already agrees and when it does not.
#
# Both were unresolved when the captures were taken, which is what makes them honest forecasts
# and also what makes them unscoreable; the replay cell reports each capture's date from the
# record. Replayed by default; `RUN_LIVE = True` with `ANTHROPIC_API_KEY` and `TAVILY_API_KEY`
# forecasts current questions instead.

# %%
questions = [get_chapter_clear_question(), get_chapter_contested_question()]

print(f"Forecasting {len(questions)} questions:")
for q in questions:
    market = f"{q.current_market_price:.0%}" if q.current_market_price is not None else "?"
    print(f"  • {q.question}")
    if q.resolution_date:
        print(f"    Resolves: {q.resolution_date} | Market: {market}")
    else:
        print(f"    Market: {market}")

# %% [markdown]
# ## Running the Pipeline

# %% [markdown]
# Live execution is isolated in one helper. The publication path below never
# calls it while `RUN_LIVE` remains false.


# %%
results: list[ForecastResult] = []
run_traces = []

if RUN_LIVE:
    llm = create_llm_client(LLM_PROVIDER)
    search = create_search_client()

    # One tracer per question wraps the shared client so the full four-phase
    # conversation - research, debate, and supervisor calls - is captured and
    # persisted as a standalone, replayable run trace for each question.
    for q in questions:
        tracer = trace_llm(llm, label="pipeline")
        forecaster = AIAForecaster(
            llm=tracer,
            search=search,
            n_agents=N_AGENTS,
            max_steps=MAX_STEPS,
            debate_rounds=DEBATE_ROUNDS,
        )
        result = forecaster.forecast(q)
        results.append(result)

        run = RunTrace.from_result(
            result,
            notebook="08_forecasting_pipeline",
            provider=llm.model_name,
            params={
                "n_agents": N_AGENTS,
                "max_steps": MAX_STEPS,
                "debate_rounds": DEBATE_ROUNDS,
                "aggregation": "mean",
                "statistical_correction_a": math.sqrt(3),
            },
            llm_calls=tracer.calls,
            notes="Teaching pipeline: research, mean, optional debate, high-only supervisor, fixed sqrt(3) scaling.",
        )
        trace_path = run.save()
        run_traces.append(run)
        print(
            f"  ✓ {q.question[:50]}... → {result.final_probability} ({result.status}) "
            f"({result.duration_seconds:.1f}s) | {len(run.llm_calls)} calls → {trace_path.name}"
        )
else:
    # Replay: reload one pinned pipeline trace per question and rehydrate the
    # full ForecastResult. The display cells below consume these unchanged.
    for pinned_name in PINNED_TRACES:
        run = RunTrace.load(TRACES_DIR / pinned_name)
        result = run.forecast_result()
        results.append(result)
        run_traces.append(run)
        print(
            f"  ✓ {result.question.question[:50]}... → {result.final_probability} ({result.status}) "
            f"(replay) | {len(run.llm_calls)} calls from {pinned_name}"
        )

# %% [markdown]
# ## Results Summary

# %%
grand_total = TokenUsage()
for r in results:
    grand_total = grand_total + r.total_token_usage

summary_df = pl.DataFrame(
    [
        {
            "question": r.question.question[:80],
            "final": round(r.final_probability, 3) if r.final_probability is not None else None,
            "status": r.status,
            "market": (
                round(r.question.current_market_price, 3)
                if r.question.current_market_price is not None
                else None
            ),
            "confidence": round(r.final_confidence, 2),
            "duration_s": (
                round(r.duration_seconds, 1) if r.duration_seconds is not None else None
            ),
        }
        for r in results
    ]
)
print(f"Total tokens across {len(results)} questions: {grand_total.total_tokens:,}\n")
summary_df

# %% [markdown]
# ## The Full Trace, One Question
#
# The untruncated record for the first question, stage by stage, through the same
# `agent_observability` helpers used across the chapter: each research agent's queries,
# documents and rationale; the debate transcript with both sides' complete arguments; and the
# supervisor's reconciliation, including the disagreements it flagged, the clarifying searches
# it ran, and the probability it returned. All of it is in the saved JSON, so this readout can
# be rebuilt from disk with `RunTrace.load` long after the run.

# %%
r = results[0]
print(f"Question: {r.question.question}\n")
print(show_agents(r.agents))

# %%
print("Aggregation:")
if r.aggregation:
    print(f"  method: {r.aggregation.method}")
    print(f"  inputs: {r.aggregation.input_probabilities}")
    print(f"  raw mean: {r.aggregation.raw_probability:.4f}")
    print(f"  extremized: {r.aggregation.extremized_probability}")

# %% [markdown]
# ### Debate

# %%
if r.debate:
    print(show_debate_transcript(r.debate))

# %%
if r.supervisor:
    print(show_supervisor(r.supervisor))

print("\n── Final ──")
print(f"  probability: {r.final_probability} ({r.status})")
print(f"  confidence:  {r.final_confidence:.2f}")
if r.duration_seconds is not None:
    print(f"  duration:    {r.duration_seconds:.1f}s")
else:
    print("  duration:    n/a (replayed from pinned trace)")

# %% [markdown]
# ## Pipeline Flow Visualization

# %%
for r in results:
    q_short = r.question.question[:50]
    phases = []

    if r.question.current_market_price is not None:
        phases.append(("Market", r.question.current_market_price))

    for a in r.agents:
        phases.append((a.agent_id, a.p_yes))

    if r.aggregation:
        phases.append(
            ("Aggregate", r.aggregation.extremized_probability or r.aggregation.raw_probability)
        )

    if r.debate and r.debate.bull_final_probability is not None:
        mid = (r.debate.bull_final_probability + r.debate.bear_final_probability) / 2
        phases.append(("Debate", mid))

    if r.supervisor and r.supervisor.p_yes is not None:
        phases.append(("Supervisor", r.supervisor.p_yes))

    phases.append(("Final", r.final_probability))

    print(f"\n{q_short}...")
    for name, p in phases:
        if p is None:
            print(f"  {name:<12s} abstained")
            continue
        bar = "█" * int(p * 40) + "░" * (40 - int(p * 40))
        print(f"  {name:<12s} {bar} {p:.2f}")

# %% [markdown]
# ## Token Cost Analysis

# %%
print("Usage from the captured run:")
print(f"  Questions: {len(results)}")
print(f"  Input tokens: {grand_total.input_tokens:,}")
print(f"  Output tokens: {grand_total.output_tokens:,}")
print("Replay makes no paid calls. Dollar cost depends on the recorded model and its pricing.")

# %% [markdown]
# ## State persistence
#
# A full trace retains the question, cutoff, evidence, model messages, component
# outputs, status and available usage. Replaying a result reproduces the saved
# run; it does not rerun today's algorithm or verify the outcome.

# %%
# The full RunTrace saved above retains evidence, rationale and actual model I/O.
print(run_traces[0].to_json()[:1200])
print("Inspect the complete JSON in forecast_traces/ for the remaining messages and sources.")

# %% [markdown]
# ## Key Takeaways
#
# 1. **The value of a pipeline is that each stage is inspectable, not that each stage improves
#    the answer.** Four stages give four places to look when a forecast is wrong. Whether any
#    of them made it better is a scoring question, and scoring needs resolved questions.
# 2. **An override needs a gate, and the gate needs a rule.** The supervisor can replace the
#    ensemble only at high stated confidence, blends at medium, and is ignored at low. Without
#    that, one model's second opinion silently outranks three agents' evidence.
# 3. **A stated confidence is not a measured one.** Both the supervisor's own label and the
#    scalar this pipeline attaches to the final probability are conventions. They order
#    outcomes; they do not estimate anything.
# 4. **Declare every constant that moves the final number in one place.** The blend
#    weights, the correlation and the override thresholds decide the output, and a reader who
#    cannot find them cannot evaluate the pipeline.
# 5. **The stage-by-stage record is the deliverable.** One JSON file per question holds every
#    prompt, every document, every intermediate probability, and it is what makes a forecast
#    reviewable months later.
#
# **Known limitations of what is built here.** Both questions were unresolved when captured, so
# nothing in this notebook can be scored and no claim about accuracy is available from it. The
# market price is passed to every research agent, so the pipeline's distance from the market
# is not an independent comparison. The blend weights and confidence scalars are conventions,
# and no experiment here shows the four-stage output is better than the three-agent mean.
#
# **Next**: [`09_evaluation_and_governance`](09_evaluation_and_governance.ipynb) builds the
# scoring rules, calibration curves and security controls a pipeline like this needs before
# anyone acts on it.
#
# **Book**: Section 24.7 covers the pipeline architecture and section 24.9 the production
# considerations that follow from it.
