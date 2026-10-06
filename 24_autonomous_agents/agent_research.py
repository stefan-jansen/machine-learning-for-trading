"""Research agent class and supporting utilities.

Built step-by-step in NB04; downstream notebooks (NB06–NB08) import from here
rather than rebuilding the agent inline each time.
"""

from __future__ import annotations

import json
import re
from datetime import date

from agent_pipeline import validate_probabilities
from agent_providers import ChatMessage, LLMClient, TokenUsage
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
    eligible_results,
    format_search_results,
)

# ---------------------------------------------------------------------------
# Prompt templates (inline for teaching visibility)
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """\
You are a forecasting agent in a multi-agent forecasting system.

Your job:
1) Gather evidence by issuing web/news search queries when needed.
2) Then produce a binary probability forecast for the question.

You must follow the action schema exactly and output valid JSON only.
You must not browse prediction market prices unless they are explicitly provided.
Cite a retained source URL or title in your rationale. If evidence is insufficient,
return {"action":"abstain","rationale":"what is missing"}."""


def build_step_prompt(
    question: ForecastQuestion,
    market_price: float | None = None,
    max_steps: int | None = None,
) -> str:
    """Format the step prompt with question context.

    When ``max_steps`` is given, the agent is told its search budget and is
    nudged to commit once it has enough evidence - searching is not free, and an
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
            "specific, named fact is missing - do not reformulate a query you already "
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


# ---------------------------------------------------------------------------
# JSON parsing with fallback
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Rich output extraction
# ---------------------------------------------------------------------------


def extract_confidence(action: dict) -> float:
    """Extract confidence from action dict or infer from probability."""
    if "confidence" in action:
        try:
            return max(0.0, min(1.0, float(action["confidence"])))
        except (TypeError, ValueError):
            pass
    p = float(action.get("p_yes", 0.5))
    return round(abs(p - 0.5) * 2, 3)


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


INLINE_ENUMERATION = re.compile(r"\(\d+\)\s*")
SENTENCE_END = re.compile(r"(?<![A-Z]\.)(?<=[.!?])\s+(?=[A-Z])")


def extract_key_findings(rationale: str) -> list[str]:
    """Extract the items a rationale enumerates, on their own lines or inline.

    Line-leading bullets and numbers win where they exist. Otherwise the inline
    ``(1) ... (2) ...`` form models routinely use inside a sentence is split on its
    markers; at least two are required so a lone parenthesised digit in ordinary
    prose is not read as a list.
    """
    findings = []
    for line in rationale.split("\n"):
        line = line.strip()
        if re.match(r"^[-•*]\s+", line):
            findings.append(re.sub(r"^[-•*]\s+", "", line).strip())
        elif re.match(r"^\d+[.)]\s+", line):
            findings.append(re.sub(r"^\d+[.)]\s+", "", line).strip())
    if findings:
        return findings[:10]
    items = [part.strip() for part in INLINE_ENUMERATION.split(rationale)[1:]]
    if len(items) < 2:
        return []
    # The text after the last marker runs on into whatever the model wrote next, so the
    # last item ends at its own sentence boundary rather than at the end of the rationale.
    # The boundary has to survive an abbreviation: "U.S. inflation remains elevated" is one
    # sentence, so a full stop only ends the item when the letter before it is not a capital
    # and the next word begins with one.
    items[-1] = SENTENCE_END.split(items[-1])[0]
    return [item.rstrip(";.").strip() for item in items][:10]


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


def assess_evidence_quality(sources_consulted: int, queries_made: int) -> EvidenceQuality:
    """Assess evidence quality from search metrics."""
    if sources_consulted >= 10 and queries_made >= 3:
        return EvidenceQuality.HIGH
    if sources_consulted >= 5 or queries_made >= 2:
        return EvidenceQuality.MEDIUM
    return EvidenceQuality.LOW


# ---------------------------------------------------------------------------
# ResearchAgent
# ---------------------------------------------------------------------------


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
                    if isinstance(action["p_yes"], bool):
                        raise ValueError("Probability must be a number, not a boolean")
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


# ---------------------------------------------------------------------------
# Agent summary formatting (for supervisor/debate prompts)
# ---------------------------------------------------------------------------


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
