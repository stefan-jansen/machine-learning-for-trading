"""Reusable specialist definitions from the debate and composed teaching notebooks."""

from __future__ import annotations

import re
from datetime import date

from agent_pipeline import validate_probabilities
from agent_providers import ChatMessage
from agent_research import parse_json
from agent_schemas import DebateArtifact, DebateRound, SearchResult, SupervisorArtifact, TokenUsage
from agent_tools import SearchClient, ToolExecutor

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


def _run_debate_round(
    llm,
    round_num: int,
    question: str,
    agent_summaries: str,
    aggregate_p_yes: float,
    prev_bear_argument: str | None,
    prev_bear_probability: float | None,
    consensus_threshold: float,
    artifact: DebateArtifact,
) -> tuple[DebateRound, TokenUsage]:
    """Run one bull→bear round. Returns (DebateRound, round_token_usage)."""
    round_tokens = TokenUsage()

    bear_section = ""
    if prev_bear_argument is not None:
        bear_section = (
            f"BEAR'S PREVIOUS ARGUMENT:\n{prev_bear_argument}\n"
            f"Bear's probability: {prev_bear_probability:.4f}\n\n"
            "You must directly address the Bear's points and explain "
            "why the probability should be higher."
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
    round_tokens = round_tokens + bull_tokens
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
    round_tokens = round_tokens + bear_tokens
    artifact.token_usage = artifact.token_usage + bear_tokens
    bear_parsed = parse_json(bear_raw)
    bear_argument = bear_parsed.get("argument", "")
    if isinstance(bear_parsed["p_yes"], bool):
        raise ValueError("Probability must be a number, not a boolean")
    bear_probability = float(bear_parsed["p_yes"])
    validate_probabilities([bear_probability])
    bear_evidence = [str(e) for e in bear_parsed.get("key_evidence", [])]

    consensus = abs(bull_p - bear_probability) < consensus_threshold
    debate_round = DebateRound(
        round_number=round_num,
        bull_argument=bull_argument,
        bull_probability=bull_p,
        bear_argument=bear_argument,
        bear_probability=bear_probability,
        consensus_reached=consensus,
        bull_key_evidence=bull_evidence,
        bear_key_evidence=bear_evidence,
    )
    return debate_round, round_tokens


class DebateAgent:
    """Structured adversarial debate between bull and bear positions."""

    def __init__(
        self,
        llm,
        max_rounds: int = 3,
        consensus_threshold: float = 0.05,
    ) -> None:
        self.llm = llm
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.token_usage = TokenUsage()

    def run(
        self,
        question: str,
        agent_summaries: str,
        aggregate_p_yes: float,
    ) -> DebateArtifact:
        """Run the debate. Returns a DebateArtifact with full transcript."""
        self.token_usage = TokenUsage()
        self.artifact = DebateArtifact()
        rounds = self.artifact.rounds
        prev_bear_argument: str | None = None
        prev_bear_probability: float | None = None

        for round_num in range(1, self.max_rounds + 1):
            debate_round, round_tokens = _run_debate_round(
                self.llm,
                round_num,
                question,
                agent_summaries,
                aggregate_p_yes,
                prev_bear_argument,
                prev_bear_probability,
                self.consensus_threshold,
                self.artifact,
            )
            self.token_usage = self.token_usage + round_tokens
            rounds.append(debate_round)
            if debate_round.consensus_reached:
                break
            prev_bear_argument = debate_round.bear_argument
            prev_bear_probability = debate_round.bear_probability

        final_bull = rounds[-1].bull_probability if rounds else None
        final_bear = rounds[-1].bear_probability if rounds else None
        consensus_reached = rounds[-1].consensus_reached if rounds else False

        return DebateArtifact(
            rounds=rounds,
            bull_final_probability=final_bull,
            bear_final_probability=final_bear,
            consensus_reached=consensus_reached,
            early_termination=consensus_reached and len(rounds) < self.max_rounds,
            token_usage=self.token_usage,
        )


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
