"""Research agent class and supporting utilities.

Built step-by-step in NB04; downstream notebooks (NB06–NB08) import from here
rather than rebuilding the agent inline each time.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date

from agent_providers import ChatMessage, LLMClient, TokenUsage
from agent_schemas import (
    AgentForecastArtifact,
    AgentTrace,
    EvidenceQuality,
    ForecastQuestion,
    SearchResult,
    Sentiment,
)
from agent_tools import SearchClient, ToolExecutor, format_search_results

# ---------------------------------------------------------------------------
# Prompt templates (inline for teaching visibility)
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """\
You are a forecasting agent in a multi-agent forecasting system.

Your job:
1) Gather evidence by issuing web/news search queries when needed.
2) Then produce a binary probability forecast for the question.

You must follow the action schema exactly and output valid JSON only.
You must not browse prediction market prices unless they are explicitly provided."""


def build_step_prompt(
    question: ForecastQuestion,
    market_price: float | None = None,
) -> str:
    """Format the step prompt with question context."""
    prompt = f"QUESTION:\n{question.question}\n\n"
    if question.description:
        prompt += f"MARKET CONTEXT:\n{question.description}\n\n"
    if market_price is not None:
        prompt += f"MARKET IMPLIED PROBABILITY (p_yes):\n{market_price}\n\n"
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
    """Parse one JSON object from LLM output, including fenced output."""
    stripped = raw.strip()
    unfenced = re.sub(r"^```(?:json)?\s*", "", stripped)
    unfenced = re.sub(r"\s*```$", "", unfenced)
    match = re.search(r"\{.*\}", unfenced, re.DOTALL)
    candidates = [stripped, unfenced]
    if match is not None:
        candidates.append(match.group())
    for candidate in dict.fromkeys(candidates):
        try:
            decoded = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(decoded, dict):
            return decoded
    return {"action": "parse_failure", "error": "JSON parse failure"}


def validate_action(action: dict) -> tuple[str, dict | None]:
    """Validate and normalize one search or forecast action."""
    action_type = action.get("action")
    if not isinstance(action_type, str):
        return "schema_failure", None

    normalized = dict(action)
    if action_type == "search":
        query = action.get("query")
        if not isinstance(query, str) or not query.strip():
            return "schema_failure", None
        normalized["query"] = query.strip()
        return action_type, normalized

    if action_type == "forecast":
        raw_p_yes = action.get("p_yes")
        rationale = action.get("rationale")
        if isinstance(raw_p_yes, bool) or not isinstance(raw_p_yes, (int, float)):
            return "schema_failure", None
        try:
            p_yes = float(raw_p_yes)
        except (TypeError, ValueError, OverflowError):
            return "schema_failure", None
        if not math.isfinite(p_yes) or not isinstance(rationale, str):
            return "schema_failure", None
        normalized["p_yes"] = max(0.0, min(1.0, p_yes))

        if "confidence" in action:
            raw_confidence = action["confidence"]
            if isinstance(raw_confidence, bool) or not isinstance(raw_confidence, (int, float)):
                return "schema_failure", None
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError, OverflowError):
                return "schema_failure", None
            if not math.isfinite(confidence):
                return "schema_failure", None
            normalized["confidence"] = max(0.0, min(1.0, confidence))
        return action_type, normalized

    return action_type, None


# ---------------------------------------------------------------------------
# Rich output extraction
# ---------------------------------------------------------------------------


def extract_confidence(action: dict) -> float:
    """Extract confidence from action dict or infer from probability."""
    if "confidence" in action:
        try:
            confidence = float(action["confidence"])
            if math.isfinite(confidence):
                return max(0.0, min(1.0, confidence))
        except (TypeError, ValueError, OverflowError):
            pass
    try:
        p = float(action.get("p_yes", 0.5))
    except (TypeError, ValueError, OverflowError):
        p = 0.5
    if not math.isfinite(p):
        p = 0.5
    p = max(0.0, min(1.0, p))
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
    """Classify evidence volume from query and result counts."""
    if sources_consulted >= 10 and queries_made >= 3:
        return EvidenceQuality.HIGH
    if sources_consulted >= 5 or queries_made >= 2:
        return EvidenceQuality.MEDIUM
    return EvidenceQuality.LOW


# ---------------------------------------------------------------------------
# ResearchAgent
# ---------------------------------------------------------------------------


class ResearchAgent:
    """ReAct-based research agent producing probability forecasts.

    Implements the same loop as the AIA Forecaster's research agent:
    search for evidence, then forecast with rich metadata.
    """

    def __init__(
        self,
        llm: LLMClient,
        search: SearchClient | None = None,
        agent_id: str = "agent_0",
        max_steps: int = 5,
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
    ) -> AgentForecastArtifact:
        """Run the agent on a question. Returns a rich forecast artifact."""
        cutoff = date.fromisoformat(question.cutoff_date) if question.cutoff_date else None

        messages = [
            ChatMessage(role="system", content=AGENT_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=build_step_prompt(question, market_price),
            ),
        ]

        traces: list[AgentTrace] = []
        total_tokens = TokenUsage()
        queries_made = 0
        sources_consulted = 0
        p_yes = 0.5
        rationale = ""
        raw_action: dict = {}

        for step in range(1, self.max_steps + 1):
            response, usage = self.llm.complete_with_usage(messages, json_mode=True)
            total_tokens = total_tokens + usage

            parsed = parse_json(response)
            action_type, action = validate_action(parsed)

            if action_type == "search" and action is not None:
                query = action["query"]
                results = self.executor.execute_search(
                    query,
                    max_results=self.max_search_results,
                    cutoff_date=cutoff,
                )
                queries_made += 1
                sources_consulted += len(results)

                traces.append(
                    AgentTrace(
                        step=step,
                        action="search",
                        query=query,
                        results=results,
                        llm_raw=response,
                    )
                )
                messages.append(ChatMessage(role="assistant", content=response))
                messages.append(ChatMessage(role="tool", content=format_search_results(results)))

            elif action_type == "forecast" and action is not None:
                p_yes = action["p_yes"]
                rationale = action["rationale"]
                raw_action = action
                traces.append(AgentTrace(step=step, action="forecast", llm_raw=response))
                break

            else:
                # Unrecognized action: record in trace, feed error back to LLM
                traces.append(AgentTrace(step=step, action=action_type, llm_raw=response))
                messages.append(ChatMessage(role="assistant", content=response))
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=f"Error: unrecognized action '{action_type}'. "
                        "Use 'search' or 'forecast'.",
                    )
                )
        else:
            # Loop exhausted without a forecast: record forced default
            rationale = "Max steps reached without forecast"
            traces.append(AgentTrace(step=self.max_steps, action="forced_default"))

        return AgentForecastArtifact(
            agent_id=self.agent_id,
            p_yes=p_yes,
            rationale=rationale,
            traces=traces,
            confidence=extract_confidence(raw_action),
            sentiment=extract_sentiment(p_yes),
            key_findings=extract_key_findings(rationale),
            evidence_quality=assess_evidence_quality(sources_consulted, queries_made),
            uncertainties=extract_uncertainties(rationale),
            token_usage=total_tokens,
            search_queries_made=queries_made,
            sources_consulted=sources_consulted,
        )


# ---------------------------------------------------------------------------
# Agent summary formatting (for supervisor/debate prompts)
# ---------------------------------------------------------------------------


def format_agent_summary(a: AgentForecastArtifact) -> str:
    """Format an agent artifact as a summary for supervisor/debate prompts."""
    lines = [
        f"Agent: {a.agent_id}",
        f"Probability (p_yes): {a.p_yes:.2f}",
        f"Confidence: {a.confidence:.2f}",
        f"Rationale: {a.rationale[:200]}",
    ]
    if a.key_findings:
        lines.append("Key findings:")
        for f in a.key_findings[:3]:
            lines.append(f"  - {f}")
    return "\n".join(lines)
