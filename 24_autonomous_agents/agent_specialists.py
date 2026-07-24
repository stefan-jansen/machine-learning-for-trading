"""Specialist agent classes for the multi-agent forecasting pipeline.

`DebateAgent` and `SupervisorAgent` are built step-by-step in NB07 and NB08
respectively; this module mirrors those classes for downstream notebooks
(NB10 framework comparison) that need to reuse the full pipeline.

Each class is functionally identical to the inline version in its teaching
notebook. Prompts live here as module-level constants so all three frameworks
in NB10 can route the same text through their orchestrators.
"""

from __future__ import annotations

from datetime import date

from agent_providers import ChatMessage
from agent_research import parse_json
from agent_schemas import (
    DebateArtifact,
    DebateRound,
    SearchResult,
    SupervisorArtifact,
    TokenUsage,
)
from agent_tools import SearchClient, ToolExecutor

# ---------------------------------------------------------------------------
# Debate prompts
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Supervisor prompts
# ---------------------------------------------------------------------------

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
3) A short rationale

Output JSON only:
{{"p_yes": 0.0, "confidence": "high", "rationale": "..."}}

QUESTION:
{question}

AGENT INPUTS:
{agent_summaries}

SUPERVISOR SEARCH EVIDENCE:
{supervisor_evidence}"""


# ---------------------------------------------------------------------------
# DebateAgent
# ---------------------------------------------------------------------------


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
        rounds: list[DebateRound] = []

        bear_argument: str | None = None
        bear_probability: float | None = None

        for round_num in range(1, self.max_rounds + 1):
            bear_section = ""
            if bear_argument is not None:
                bear_section = (
                    f"BEAR'S PREVIOUS ARGUMENT:\n{bear_argument}\n"
                    f"Bear's probability: {bear_probability:.4f}\n\n"
                    "You must directly address the Bear's points and explain "
                    "why the probability should be higher."
                )

            bull_prompt = BULL_PROMPT_TEMPLATE.format(
                question=question,
                agent_summaries=agent_summaries,
                aggregate_p_yes=f"{aggregate_p_yes:.4f}",
                bear_section=bear_section,
            )
            bull_raw, bull_tokens = self.llm.complete_with_usage(
                [ChatMessage(role="user", content=bull_prompt)], json_mode=True
            )
            self.token_usage = self.token_usage + bull_tokens
            bull_parsed = parse_json(bull_raw)
            bull_argument = bull_parsed.get("argument", "")
            bull_p = float(bull_parsed.get("p_yes", aggregate_p_yes))
            bull_evidence = [str(e) for e in bull_parsed.get("key_evidence", [])]

            bear_prompt = BEAR_PROMPT_TEMPLATE.format(
                question=question,
                agent_summaries=agent_summaries,
                aggregate_p_yes=f"{aggregate_p_yes:.4f}",
                bull_argument=bull_argument,
                bull_probability=f"{bull_p:.4f}",
            )
            bear_raw, bear_tokens = self.llm.complete_with_usage(
                [ChatMessage(role="user", content=bear_prompt)], json_mode=True
            )
            self.token_usage = self.token_usage + bear_tokens
            bear_parsed = parse_json(bear_raw)
            bear_argument = bear_parsed.get("argument", "")
            bear_probability = float(bear_parsed.get("p_yes", aggregate_p_yes))
            bear_evidence = [str(e) for e in bear_parsed.get("key_evidence", [])]

            consensus = abs(bull_p - bear_probability) < self.consensus_threshold

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

            if consensus:
                break

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


# ---------------------------------------------------------------------------
# SupervisorAgent
# ---------------------------------------------------------------------------


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

        msg1 = SUPERVISOR_DISAGREEMENTS_PROMPT.format(
            max_queries=self.max_queries,
            agent_summaries=agent_summaries,
        )
        raw1, tokens1 = self.llm.complete_with_usage(
            [ChatMessage(role="user", content=msg1)], json_mode=True
        )
        self.token_usage = self.token_usage + tokens1
        parsed1 = parse_json(raw1)

        disagreements = [str(x) for x in parsed1.get("disagreements", [])][:20]
        queries = [str(x) for x in parsed1.get("queries", [])][: self.max_queries]

        search_results: dict[str, list[SearchResult]] = {}
        if self.search is not None:
            executor = ToolExecutor(search=self.search)
            for q in queries:
                results = executor.execute_search(
                    q, max_results=self.max_search_results, cutoff_date=cutoff_date
                )
                search_results[q] = results

        evidence_text = self._format_evidence(search_results)
        msg2 = SUPERVISOR_FINALIZE_PROMPT.format(
            question=question,
            agent_summaries=agent_summaries,
            supervisor_evidence=evidence_text,
        )
        raw2, tokens2 = self.llm.complete_with_usage(
            [ChatMessage(role="user", content=msg2)], json_mode=True
        )
        self.token_usage = self.token_usage + tokens2
        parsed2 = parse_json(raw2)

        p_yes = parsed2.get("p_yes")
        confidence = parsed2.get("confidence")
        rationale = parsed2.get("rationale")

        if confidence is not None:
            conf_str = str(confidence).lower()
            if conf_str not in ("high", "medium", "low"):
                conf_str = "medium"
            confidence = conf_str

        return SupervisorArtifact(
            disagreements=disagreements,
            queries=queries,
            search_results=search_results,
            p_yes=float(p_yes) if p_yes is not None else None,
            confidence=str(confidence) if confidence is not None else None,
            rationale=str(rationale) if rationale is not None else None,
            token_usage=self.token_usage,
        )

    @staticmethod
    def _format_evidence(sr: dict[str, list[SearchResult]]) -> str:
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
