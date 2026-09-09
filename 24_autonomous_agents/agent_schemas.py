"""Data schemas for the multi-agent forecasting pipeline.

Plain dataclasses — no Pydantic dependency — so the teaching code stays
transparent and inspectable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum, StrEnum
from uuid import uuid4

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Sentiment(StrEnum):
    """Agent sentiment towards YES outcome."""

    STRONGLY_BULLISH = "strongly_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONGLY_BEARISH = "strongly_bearish"


class EvidenceQuality(StrEnum):
    """Quality of evidence found during research."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TokenUsage:
    """Accumulates token counts across LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


# ---------------------------------------------------------------------------
# Search results
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SearchResult:
    """A single search result from Tavily or mock provider."""

    title: str
    url: str | None = None
    snippet: str | None = None
    published: str | None = None
    score: float | None = None


# ---------------------------------------------------------------------------
# Agent trace and artifact
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AgentTrace:
    """One step in the agent's ReAct loop."""

    step: int
    action: str  # "search" or "forecast"
    query: str | None = None
    results: list[SearchResult] = field(default_factory=list)
    llm_raw: str | None = None


@dataclass(slots=True)
class AgentForecastArtifact:
    """Rich output from a single research agent run.

    Holds the probability, the reasoning trace behind it, heuristic
    confidence and sentiment metadata, and token usage for cost tracking.

    `forecast_produced` separates an agent that committed to a probability
    from one that ran out of turns. Both carry a `p_yes`; only the first is a
    forecast, and treating the second as one puts an unearned 0.5 into every
    average downstream.
    """

    agent_id: str
    p_yes: float
    rationale: str
    traces: list[AgentTrace] = field(default_factory=list)

    # False when the loop ended on its step budget rather than on a forecast
    # action. `p_yes` then holds the loop's fallback value, which is not a
    # judgement and must not be aggregated as one.
    forecast_produced: bool = True

    # Rich output fields
    confidence: float = 0.5
    sentiment: Sentiment = Sentiment.NEUTRAL
    key_findings: list[str] = field(default_factory=list)
    evidence_quality: EvidenceQuality = EvidenceQuality.MEDIUM
    uncertainties: list[str] = field(default_factory=list)

    # Token usage
    token_usage: TokenUsage = field(default_factory=TokenUsage)

    # Search metadata
    search_queries_made: int = 0
    sources_consulted: int = 0


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SupervisorArtifact:
    """Output from the supervisor reconciliation phase.

    Two-phase process: (1) identify disagreements and propose clarifying
    queries, (2) search and finalize with confidence-gated override.
    """

    disagreements: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    search_results: dict[str, list[SearchResult]] = field(default_factory=dict)
    p_yes: float | None = None
    confidence: str | None = None  # "high" / "medium" / "low"
    rationale: str | None = None
    resolved_disagreements: list[str] = field(default_factory=list)
    unresolved_uncertainties: list[str] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)


# ---------------------------------------------------------------------------
# Debate
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DebateRound:
    """A single bull-vs-bear debate round."""

    round_number: int
    bull_argument: str
    bull_probability: float
    bear_argument: str
    bear_probability: float
    consensus_reached: bool = False
    bull_key_evidence: list[str] = field(default_factory=list)
    bear_key_evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DebateArtifact:
    """Complete debate transcript with convergence tracking."""

    rounds: list[DebateRound] = field(default_factory=list)
    bull_final_probability: float | None = None
    bear_final_probability: float | None = None
    consensus_reached: bool = False
    early_termination: bool = False
    token_usage: TokenUsage = field(default_factory=TokenUsage)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AggregationResult:
    """Detailed result from probability aggregation."""

    method: str  # mean, median, neyman, neyman_weighted
    raw_probability: float
    extremized_probability: float | None = None
    extremization_factor: float | None = None
    input_probabilities: list[float] = field(default_factory=list)
    input_weights: list[float] = field(default_factory=list)
    effective_n: float | None = None


# ---------------------------------------------------------------------------
# Forecast question and result
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ForecastQuestion:
    """A binary prediction market question with resolution data."""

    question: str
    description: str = ""
    resolution_date: str = ""
    cutoff_date: str = ""
    current_market_price: float | None = None
    resolved_outcome: float | None = None  # 1.0 = YES, 0.0 = NO


@dataclass(slots=True)
class ForecastResult:
    """Complete output from a forecasting pipeline run."""

    question: ForecastQuestion

    # Agent phase
    agents: list[AgentForecastArtifact] = field(default_factory=list)

    # Aggregation phase
    aggregation: AggregationResult | None = None

    # Debate phase
    debate: DebateArtifact | None = None

    # Supervisor phase
    supervisor: SupervisorArtifact | None = None

    # Final output
    final_probability: float = 0.5
    final_confidence: float = 0.5

    # Operational
    total_token_usage: TokenUsage = field(default_factory=TokenUsage)
    duration_seconds: float | None = None


# ---------------------------------------------------------------------------
# Agent state (for NB03 state management teaching)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class QualityGateResult:
    """Result from a single quality gate check."""

    gate_name: str
    passed: bool
    reason: str
    details: dict = field(default_factory=dict)


@dataclass(slots=True)
class AgentState:
    """Explicit agent state for checkpoint/replay.

    Holds what the agent has established at a point in the run, separate from
    the LLM's context window: the question and its cutoff, the evidence
    gathered, the tool calls that gathered it, the quality-gate verdicts, and
    whether synthesis has run. `to_json` / `from_json` round-trip the whole
    record, which is what makes a run resumable and an ablation cheap.
    """

    question: str = ""
    cutoff_date: str = ""
    run_id: str = field(default_factory=lambda: uuid4().hex[:12])
    evidence: list[dict] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    tool_trace: list[dict] = field(default_factory=list)
    quality_gates: list[QualityGateResult] = field(default_factory=list)
    synthesis_status: str = "pending"

    def to_json(self) -> str:
        """Serialize state to JSON for checkpointing."""
        return json.dumps(asdict(self), indent=2, default=_json_default)

    @classmethod
    def from_json(cls, json_str: str) -> AgentState:
        """Restore state from JSON checkpoint."""
        data = json.loads(json_str)
        quality_gates = [QualityGateResult(**item) for item in data.get("quality_gates", [])]
        return cls(
            question=data.get("question", ""),
            cutoff_date=data.get("cutoff_date", ""),
            run_id=data.get("run_id", uuid4().hex[:12]),
            evidence=data.get("evidence", []),
            open_questions=data.get("open_questions", []),
            tool_trace=data.get("tool_trace", []),
            quality_gates=quality_gates,
            synthesis_status=data.get("synthesis_status", "pending"),
        )


# ---------------------------------------------------------------------------
# Quality gates over AgentState
# ---------------------------------------------------------------------------


def parse_iso_date(value: str) -> date | None:
    """Parse an ISO date, returning None for missing or malformed values."""
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def check_coverage_gate(
    state: AgentState,
    min_items: int = 3,
    required_types: list[str] | None = None,
) -> QualityGateResult:
    """Check that evidence covers the required types and meets the minimum count."""
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


def check_freshness_gate(
    state: AgentState,
    *,
    as_of: datetime,
    max_age_hours: int = 24,
) -> QualityGateResult:
    """Check that no evidence was retrieved outside the allowed age window."""
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


def check_consistency_gate(state: AgentState) -> QualityGateResult:
    """Check that every search result is dated and published before the cutoff."""
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


def run_quality_gates(state: AgentState, *, as_of: datetime) -> list[QualityGateResult]:
    """Run every gate, store the outcomes on the state, and return them."""
    gates = [
        check_coverage_gate(state),
        check_freshness_gate(state, as_of=as_of),
        check_consistency_gate(state),
    ]
    state.quality_gates = gates
    return gates


def _json_default(value: object) -> object:
    """JSON serialization hook for enums and dataclasses."""
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
