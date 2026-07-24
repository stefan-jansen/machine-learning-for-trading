"""Data schemas for the multi-agent forecasting pipeline.

Plain dataclasses — no Pydantic dependency — so the teaching code stays
transparent and inspectable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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

    Includes the forecast, reasoning trace, confidence/sentiment
    metadata, and token usage for cost tracking.
    """

    agent_id: str
    p_yes: float
    rationale: str
    traces: list[AgentTrace] = field(default_factory=list)

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

    Captures everything the agent knows at a point in time,
    separate from the LLM's context window.
    """

    ticker: str = ""
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
            ticker=data.get("ticker", ""),
            cutoff_date=data.get("cutoff_date", ""),
            run_id=data.get("run_id", uuid4().hex[:12]),
            evidence=data.get("evidence", []),
            open_questions=data.get("open_questions", []),
            tool_trace=data.get("tool_trace", []),
            quality_gates=quality_gates,
            synthesis_status=data.get("synthesis_status", "pending"),
        )


def _json_default(value: object) -> object:
    """JSON serialization hook for enums and dataclasses."""
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
