from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "24_autonomous_agents"))

import agent_pipeline as metrics  # noqa: E402
from agent_observability import RunTrace, show_agent_timeline  # noqa: E402
from agent_providers import TokenUsage, create_llm_client  # noqa: E402
from agent_research import ResearchAgent, parse_json  # noqa: E402
from agent_schemas import ForecastQuestion, SearchResult  # noqa: E402
from agent_tools import ToolExecutor, create_search_client  # noqa: E402


@pytest.mark.parametrize(
    ("predictions", "outcomes"),
    [
        ([], []),
        ([0.7], []),
        ([0.2, 0.3], [1]),
        ([math.nan], [1]),
        ([math.inf], [0]),
        ([1.2], [1]),
        ([-0.1], [0]),
        ([0.3], [None]),
        ([0.3], [0.5]),
        ([0.3], [2]),
    ],
)
@pytest.mark.parametrize(
    "score",
    [
        metrics.brier_score,
        metrics.log_score,
        metrics.expected_calibration_error,
        metrics.reliability_bins,
        metrics.brier_ci,
        metrics.find_optimal_d,
    ],
)
def test_invalid_scored_panels_are_rejected(score, predictions, outcomes):
    with pytest.raises(ValueError):
        score(predictions, outcomes)


def test_scores_and_paired_intervals_use_corresponding_outcomes():
    predictions, outcomes = [0.2, 0.8], [0, 1]
    assert metrics.brier_score(predictions, outcomes) == pytest.approx(0.04)
    assert metrics.log_score(predictions, outcomes) == pytest.approx(-math.log(0.8))
    assert metrics.brier_ci(predictions, outcomes, n_boot=50) == pytest.approx((0.04, 0.04))
    assert metrics.brier_diff_ci(predictions, [0.5, 0.5], outcomes, n_boot=50) == pytest.approx(
        (-0.21, -0.21)
    )
    assert math.isfinite(metrics.log_score([0, 1], [1, 0]))


@pytest.mark.parametrize("probabilities", [[], [True], [False], [math.nan], [math.inf], [-1], [2]])
def test_aggregation_rejects_invalid_probability_sets(probabilities):
    with pytest.raises(ValueError):
        metrics.neyman_extremize(probabilities)


@pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [math.nan, 1], [1]])
def test_weighted_aggregation_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        metrics.neyman_extremize_weighted([0.2, 0.8], weights)


def test_bootstrap_and_bin_parameters_are_checked():
    with pytest.raises(ValueError):
        metrics.brier_ci([0.5], [1], n_boot=0)
    with pytest.raises(ValueError):
        metrics.brier_diff_ci([0.5], [], [1])
    with pytest.raises(ValueError):
        metrics.reliability_bins([0.5], [1], n_bins=0)
    with pytest.raises(ValueError):
        metrics.neyman_extremize([0.5], correlation=-1)


class ScriptedModel:
    model_name = "synthetic-failure-fixture"

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = 0

    def complete_with_usage(self, messages, **kwargs):
        self.calls += 1
        return next(self.responses), TokenUsage(input_tokens=5, output_tokens=5, total_tokens=10)


class RecordedSearch:
    def __init__(self, results):
        self.results = results
        self.calls = 0

    def search(self, query, max_results=5, cutoff_date=None):
        self.calls += 1
        return self.results


def test_unsupported_forecast_is_an_abstention_with_a_trace():
    llm = ScriptedModel('{"action":"forecast","p_yes":0.9,"rationale":"No sources"}')
    artifact = ResearchAgent(llm, max_steps=1).run(ForecastQuestion("Teaching failure test"))
    assert artifact.status == "abstained"
    assert artifact.p_yes is None
    assert "evidence" in artifact.failure_reason.lower()
    assert artifact.messages[-2]["content"].startswith('{"action":"forecast"')
    assert llm.calls == 1
    assert "n/a" in show_agent_timeline(artifact)


def test_budget_limited_synthesis_uses_retained_evidence():
    source = SearchResult(
        title="Recorded source",
        url="https://example.org/report",
        snippet="The measured indicator rose.",
    )
    llm = ScriptedModel(
        '{"action":"search","query":"indicator"}',
        '{"action":"forecast","p_yes":0.7,"rationale":"Based on https://example.org/report"}',
    )
    artifact = ResearchAgent(llm, RecordedSearch([source]), max_steps=1).run(
        ForecastQuestion("Indicator rises?")
    )
    assert artifact.status == "accepted"
    assert artifact.p_yes == 0.7
    assert artifact.sources_consulted == 1
    assert artifact.token_usage.total_tokens == 20
    assert any(source.url in m["content"] for m in artifact.messages)


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        '{"action":"forecast","p_yes":true,"rationale":"Recorded source"}',
        '{"action":"forecast","p_yes":false,"rationale":"Recorded source"}',
        "[]",
        '{"action":"forecast","p_yes":2,"rationale":"Recorded source"}',
        '{"action":"forecast","p_yes":0.7,"rationale":"Unlinked assertion"}',
        '{"action":"forecast","p_yes":0.7,"rationale":""}',
    ],
)
def test_malformed_or_unlinked_forecasts_are_not_accepted(response):
    source = SearchResult(
        title="Recorded source", url="https://example.org/report", snippet="Evidence"
    )
    llm = ScriptedModel(response, '{"action":"abstain","rationale":"Cannot support this forecast"}')
    artifact = ResearchAgent(llm, max_steps=1).run(ForecastQuestion("Question"), evidence=[source])
    assert artifact.p_yes is None
    assert artifact.status == "abstained"
    assert parse_json("not JSON")["action"] == "invalid"


def test_supplied_evidence_does_not_require_a_redundant_search(tmp_path):
    source = SearchResult(
        title="Recorded source", url="https://example.org/report", snippet="Evidence"
    )
    llm = ScriptedModel(
        '{"action":"forecast","p_yes":0.2,"rationale":"Recorded source indicates low odds"}'
    )
    artifact = ResearchAgent(llm, max_steps=1).run(ForecastQuestion("Question"), evidence=[source])
    assert artifact.status == "accepted" and artifact.search_queries_made == 0
    trace = RunTrace.capture(notebook="fixture", provider=llm.model_name, agents=[artifact])
    restored = RunTrace.load(trace.save(tmp_path)).agent_artifacts()[0]
    assert restored.messages == artifact.messages
    assert restored.status == artifact.status


def test_date_and_argument_contracts_apply_to_the_actual_search_call():
    from datetime import date

    search = RecordedSearch(
        [
            SearchResult(
                title="Undated", url="https://example.org/undated", snippet="Unknown date"
            ),
            SearchResult(
                title="Future",
                url="https://example.org/future",
                snippet="Later",
                published="2025-03-02",
            ),
            SearchResult(
                title="Eligible",
                url="https://example.org/eligible",
                snippet="Earlier",
                published="2025-02-28",
            ),
        ]
    )
    executor = ToolExecutor(search)
    assert executor.execute_search(" ") == []
    assert search.calls == 0 and executor.execution_log[-1].status == "blocked"
    results = executor.execute_search("report", cutoff_date=date(2025, 3, 1))
    assert [r.title for r in results] == ["Eligible"]


def test_requested_live_providers_fail_instead_of_returning_mocks(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        create_llm_client("openai")
    with pytest.raises(RuntimeError):
        create_search_client("tavily")
    with pytest.raises(ValueError):
        create_llm_client("misspelled-provider")


def test_provider_and_tool_failures_preserve_diagnostics():
    class UnavailableModel:
        def complete_with_usage(self, messages, **kwargs):
            raise OSError("synthetic provider failure")

    failed = ResearchAgent(UnavailableModel(), max_steps=1).run(ForecastQuestion("Question"))
    assert failed.status == "failed" and failed.p_yes is None
    assert failed.traces[-1].action == "provider_error"

    class UnavailableSearch:
        def search(self, *args):
            raise OSError("synthetic search failure")

    agent = ResearchAgent(
        ScriptedModel('{"action":"search","query":"report"}'), UnavailableSearch(), max_steps=1
    )
    abstained = agent.run(ForecastQuestion("Question"))
    assert abstained.p_yes is None and agent.executor.execution_log[-1].status == "error"
    assert any("Search status: error" in m["content"] for m in abstained.messages)


def test_trace_save_failure_does_not_leave_a_partial_record(tmp_path, monkeypatch):
    def fail_replace(self, target):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    trace = RunTrace.capture(notebook="failure", provider="synthetic-fixture")
    with pytest.raises(OSError):
        trace.save(tmp_path)
    assert list(tmp_path.iterdir()) == []


def _pipeline_classes(notebook_name="08_forecasting_pipeline.ipynb"):
    import ast
    import json

    notebook_root = REPO_ROOT / "24_autonomous_agents"
    notebook = json.loads((notebook_root / notebook_name).read_text())
    definitions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if (
                isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
                or isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and "PROMPT" in t.id for t in node.targets)
            ):
                definitions.append(node)
    namespace = {}
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            str(notebook_root),
            "exec",
            dont_inherit=True,
        ),
        namespace,
    )
    return namespace


class SyntheticPipelineClient:
    model_name = "synthetic-pipeline-fixture"

    def __init__(self, confidence="high", supervisor_p=0.7):
        self.confidence = confidence
        self.supervisor_p = supervisor_p
        self.calls = []

    def complete_with_usage(self, messages, json_mode=True):
        import json

        self.calls.append(messages)
        prompt = messages[-1].content.lower()
        if "supervisor" in prompt and "step 1:" in prompt:
            reply = {"disagreements": ["Synthetic disagreement"], "queries": ["clarify"]}
        elif "supervisor" in prompt:
            reply = {
                "p_yes": self.supervisor_p,
                "confidence": self.confidence,
                "rationale": "Synthetic support: https://fixture.example/evidence",
            }
        elif "bull debater" in prompt or "bear debater" in prompt:
            reply = {"p_yes": 0.6, "argument": "Synthetic argument", "key_evidence": []}
        elif any(m.role == "tool" for m in messages):
            reply = {
                "action": "forecast",
                "p_yes": 0.6,
                "rationale": "Synthetic support: https://fixture.example/evidence",
            }
        else:
            reply = {"action": "search", "query": "synthetic evidence"}
        return json.dumps(reply), TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)


class SyntheticPipelineSearch:
    def search(self, query, max_results, cutoff_date=None):
        return [
            SearchResult(
                "Synthetic pipeline evidence",
                "https://fixture.example/evidence",
                "Synthetic mathematical fixture",
                "2025-01-01",
            )
        ]


@pytest.mark.parametrize(
    "confidence,supervisor_p,pre_scaling",
    [
        ("high", 0.7, 0.7),
        ("medium", 0.7, 0.6),
        ("high", 1.2, 0.6),
        ("high", True, 0.6),
        ("high", False, 0.6),
    ],
)
def test_composed_notebook_executes_high_only_supervisor_and_fixed_correction(
    confidence,
    supervisor_p,
    pre_scaling,
):
    namespace = _pipeline_classes()
    client = SyntheticPipelineClient(confidence, supervisor_p)
    result = namespace["AIAForecaster"](
        client,
        SyntheticPipelineSearch(),
        n_agents=2,
        max_steps=1,
        debate_rounds=1,
    ).forecast(ForecastQuestion("Synthetic question?", current_market_price=0.99))
    expected = 1 / (1 + math.exp(-math.sqrt(3) * math.log(pre_scaling / (1 - pre_scaling))))
    assert result.status == "accepted"
    assert result.final_probability == pytest.approx(expected, abs=0.00005)
    assert result.aggregation.raw_probability == pytest.approx(0.6)
    assert len(result.debate.rounds) == 1
    assert len(result.supervisor.search_results["clarify"]) == 1
    assert result.agents[0].traces[0].results[0].url in result.agents[0].rationale


def test_composed_abstention_survives_trace_roundtrip(tmp_path):
    namespace = _pipeline_classes()
    result = namespace["AIAForecaster"](
        SyntheticPipelineClient(),
        search=None,
        n_agents=1,
        max_steps=1,
    ).forecast(ForecastQuestion("Synthetic unsupported question?"))
    assert result.status == "abstained" and result.final_probability is None
    run = RunTrace.from_result(result, notebook="fixture", provider="synthetic")
    replay = RunTrace.load(run.save(tmp_path)).forecast_result()
    assert replay.status == "abstained" and replay.final_probability is None
    assert replay.agents[0].messages


def test_recovered_panel_preserves_probabilities_and_corrects_scores():
    import json

    from agent_fixtures import get_evaluation_panel
    from agent_observability import TRACES_DIR

    panel = get_evaluation_panel()
    capture = json.loads(
        (TRACES_DIR / "09_legacy_panel_20260615T191431Z_d5030378899c.json").read_text()
    )
    assert [q.resolved_outcome for q in panel] == [0, 1, 1, 0, 1, 1, 1, 1, 0, 0]
    assert all(q.outcome_sources for q in panel)
    assert [i for i, q in enumerate(panel) if q.known_before_cutoff] == [6, 7]
    scored = [
        (q, r)
        for q, r in zip(panel, capture["params"]["panel"], strict=True)
        if not q.known_before_cutoff
    ]
    assert len(scored) == 8
    assert metrics.brier_score(
        [r["final_p"] for _, r in scored], [q.resolved_outcome for q, _ in scored]
    ) == pytest.approx(0.0001)


@pytest.mark.parametrize("stage", ["debate", "supervisor"])
def test_failed_reconciliation_keeps_completed_call_usage(stage):
    class LaterFailure(SyntheticPipelineClient):
        bears = 0

        def complete_with_usage(self, messages, json_mode=True):
            import json

            prompt = messages[-1].content.lower()
            if stage == "supervisor" and "supervisor" in prompt and "step 1:" not in prompt:
                raise OSError("Synthetic final supervisor failure")
            if "bear debater" in prompt:
                self.bears += 1
                if stage == "debate" and self.bears == 2:
                    raise OSError("Synthetic second-round failure")
                raw, usage = super().complete_with_usage(messages, json_mode)
                data = json.loads(raw)
                data["p_yes"] = 0.4
                return json.dumps(data), usage
            return super().complete_with_usage(messages, json_mode)

    result = _pipeline_classes()["AIAForecaster"](
        LaterFailure(),
        SyntheticPipelineSearch(),
        n_agents=1,
        max_steps=1,
        debate_rounds=2,
        consensus_threshold=0.01,
    ).forecast(ForecastQuestion("Synthetic failure?"))
    assert result.status == "failed" and result.final_probability is None
    assert result.debate and result.debate.rounds
    assert result.total_token_usage.total_tokens == (75 if stage == "debate" else 105)
    if stage == "supervisor":
        assert result.supervisor and result.supervisor.token_usage.total_tokens == 15


EVALUATION_NOTEBOOK = REPO_ROOT / "24_autonomous_agents/09_evaluation_and_governance.ipynb"
DEBATE_NOTEBOOK = REPO_ROOT / "24_autonomous_agents/07_adversarial_debate.ipynb"
PROVIDER_MODULE = "agent_providers"
SCHEMA_MODULE = "agent_schemas"
NOTEBOOK_CWD = REPO_ROOT


def _execute_notebook_fixture(path, setup, final_check=""):
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    parameters = next(
        i for i, c in enumerate(notebook.cells) if "parameters" in c.metadata.get("tags", [])
    )
    notebook.cells.insert(parameters + 1, nbformat.v4.new_code_cell(setup))
    if final_check:
        notebook.cells.append(nbformat.v4.new_code_cell(final_check))
    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_CWD)}},
    )
    client.create_kernel_manager()
    client.km.kernel_spec.argv[0] = sys.executable
    client.execute()
    return notebook


@pytest.mark.parametrize("accepted_count", [0, 1])
def test_sparse_live_panel_finishes_and_preserves_every_attempt(tmp_path, accepted_count):
    import json

    setup = (
        """
import socket
from SCHEMA_MODULE import SearchResult
def no_network(self, address):
    raise AssertionError("Failure fixture must not access the network")
socket.socket.connect = no_network
class SparsePanelModel:
    model_name = "synthetic-sparse-panel-test"
    questions = 0
    def complete_with_usage(self, messages, json_mode=True):
        if any(m.role == "tool" for m in messages):
            response = {"action": "forecast", "p_yes": 0.6,
                        "rationale": "https://fixture.example/evidence"}
        else:
            self.questions += 1
            response = ({"action": "search", "query": "fixture"}
                        if self.questions <= ACCEPTED_COUNT
                        else {"action": "abstain", "rationale": "Synthetic missing evidence"})
        from PROVIDER_MODULE import TokenUsage
        return json.dumps(response), TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
class FixtureSearch:
    def search(self, query, max_results=5, cutoff_date=None):
        return [SearchResult("Synthetic evidence", "https://fixture.example/evidence",
                             "Synthetic failure fixture", "2024-01-01")]
RUN_LIVE = True
N_AGENTS = 1
MAX_STEPS = 1
create_llm_client = lambda *args, **kwargs: SparsePanelModel()
create_search_client = lambda *args, **kwargs: FixtureSearch()
original_save = RunTrace.save
RunTrace.save = lambda self, directory=None: original_save(self, TRACE_DIRECTORY)
""".replace("ACCEPTED_COUNT", str(accepted_count))
        .replace("PROVIDER_MODULE", PROVIDER_MODULE)
        .replace("SCHEMA_MODULE", SCHEMA_MODULE)
        .replace("TRACE_DIRECTORY", repr(str(tmp_path)))
    )
    _execute_notebook_fixture(
        EVALUATION_NOTEBOOK,
        setup,
        f"assert len(scored_results) == {accepted_count}\nassert not configs\nassert warden.calls == 2",
    )
    captures = list(tmp_path.glob("*.json"))
    assert len(captures) == 1
    panel = json.loads(captures[0].read_text())["params"]["panel"]
    assert len(panel) == 10
    assert sum(r["status"] == "accepted" for r in panel) == accepted_count
    assert sum(r["status"] == "abstained" for r in panel) == 8 - accepted_count
    assert all(r["agents"][0]["messages"] for r in panel if r["status"] != "excluded_known_outcome")


def test_invalid_live_debate_is_saved_before_notebook_stops(tmp_path):
    import json

    from nbclient.exceptions import CellExecutionError

    setup = (
        """
import json
import socket
from SCHEMA_MODULE import SearchResult
def no_network(self, address):
    raise AssertionError("Failure fixture must not access the network")
socket.socket.connect = no_network
class InvalidDebateModel:
    model_name = "synthetic-invalid-debate-test"
    def complete_with_usage(self, messages, json_mode=True):
        if "bull debater" in messages[-1].content.lower():
            response = {"p_yes": 2, "argument": "Invalid synthetic probability"}
        elif any(m.role == "tool" for m in messages):
            response = {"action": "forecast", "p_yes": 0.6,
                        "rationale": "https://fixture.example/evidence"}
        else:
            response = {"action": "search", "query": "fixture"}
        from PROVIDER_MODULE import TokenUsage
        return json.dumps(response), TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
class FixtureSearch:
    def search(self, query, max_results=5, cutoff_date=None):
        return [SearchResult("Synthetic evidence", "https://fixture.example/evidence",
                             "Synthetic failure fixture", "2024-01-01")]
RUN_LIVE = True
N_AGENTS = 1
MAX_STEPS = 1
create_llm_client = lambda *args, **kwargs: InvalidDebateModel()
create_search_client = lambda *args, **kwargs: FixtureSearch()
original_save = RunTrace.save
RunTrace.save = lambda self, directory=None: original_save(self, TRACE_DIRECTORY)
""".replace("PROVIDER_MODULE", PROVIDER_MODULE)
        .replace("SCHEMA_MODULE", SCHEMA_MODULE)
        .replace("TRACE_DIRECTORY", repr(str(tmp_path)))
    )
    with pytest.raises(CellExecutionError, match="probabilit"):
        _execute_notebook_fixture(DEBATE_NOTEBOOK, setup)
    captures = list(tmp_path.glob("*.json"))
    assert len(captures) == 1
    capture = json.loads(captures[0].read_text())
    assert capture["status"] == "failed" and capture["final_probability"] is None
    assert len(capture["llm_calls"]) == 3
    assert json.loads(capture["llm_calls"][-1]["response"])["p_yes"] == 2


def test_malformed_supervisor_response_retains_completed_usage(tmp_path):
    import json

    from agent_observability import trace_llm

    class MalformedSupervisor(SyntheticPipelineClient):
        def complete_with_usage(self, messages, json_mode=True):
            raw, usage = super().complete_with_usage(messages, json_mode)
            if (
                "supervisor" in messages[-1].content.lower()
                and "step 1:" in messages[-1].content.lower()
            ):
                raw = json.dumps({"disagreements": [], "queries": None})
            return raw, usage

    tracer = trace_llm(MalformedSupervisor(), label="synthetic-malformed-supervisor")
    result = _pipeline_classes()["AIAForecaster"](
        tracer, SyntheticPipelineSearch(), n_agents=1, max_steps=1, debate_rounds=0
    ).forecast(ForecastQuestion("Synthetic malformed response?"))
    assert result.status == "failed" and result.final_probability is None
    assert result.supervisor.token_usage.total_tokens == 15
    assert result.total_token_usage.total_tokens == 45
    run = RunTrace.from_result(
        result, notebook="fixture", provider=tracer.model_name, llm_calls=tracer.calls
    )
    replay = RunTrace.load(run.save(tmp_path)).forecast_result()
    assert replay.total_token_usage.total_tokens == 45


@pytest.mark.parametrize("pipeline_name", ["native_sdk_pipeline", "langgraph_pipeline"])
def test_framework_failure_preserves_completed_artifacts(tmp_path, pipeline_name):
    import json

    from agent_observability import trace_llm

    if pipeline_name == "langgraph_pipeline":
        pytest.importorskip("langgraph.graph", reason="Requires the LangGraph runtime dependency")

    class MalformedSupervisor(SyntheticPipelineClient):
        def complete_with_usage(self, messages, json_mode=True):
            raw, usage = super().complete_with_usage(messages, json_mode)
            if (
                "supervisor" in messages[-1].content.lower()
                and "step 1:" in messages[-1].content.lower()
            ):
                raw = json.dumps({"disagreements": [], "queries": None})
            return raw, usage

    namespace = _pipeline_classes("10_framework_comparison.ipynb")
    namespace.update(N_AGENTS=1, MAX_STEPS=1, DEBATE_ROUNDS=1, SUPERVISOR_QUERIES=1)
    tracer = trace_llm(MalformedSupervisor(), label="synthetic-malformed-supervisor")
    result = namespace[pipeline_name](
        ForecastQuestion("Synthetic malformed response?"), tracer, SyntheticPipelineSearch()
    )
    assert result.status == "failed" and result.final_probability is None
    assert result.debate.token_usage.total_tokens == 30
    assert result.supervisor.token_usage.total_tokens == 15
    assert result.total_token_usage.total_tokens == 75
    run = RunTrace.from_result(
        result, notebook="fixture", provider=tracer.model_name, llm_calls=tracer.calls
    )
    assert RunTrace.load(run.save(tmp_path)).forecast_result().total_token_usage.total_tokens == 75
