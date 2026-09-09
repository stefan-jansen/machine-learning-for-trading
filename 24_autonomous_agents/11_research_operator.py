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
# # The ML4T Research Operator: one autonomous iteration on a real case study
#
# **Docker image**: `ml4t`
#
# The forecasting workflows of notebooks 04 to 09 hand the LLM a tightly typed action
# surface, a `search` query and a `forecast` probability, and stop when the
# agent returns a calibrated number. That surface is the right one when the
# experiment space is bounded by the library author. Research-line iteration
# against a real case study is shaped differently: each follow-up is a small
# amount of code that wires together the chapter's libraries against a real
# run-log registry, and pre-enumerating those moves across nine case studies
# would either grow without limit or block the moves that matter.
#
# This notebook adopts the **operator** shape that production coding agents
# have converged on: a thin orchestrator that hands an LLM a
# small set of **general-purpose** tools to read, write, and edit files, run
# bash, query a SQLite registry, and inspect parquet. It then decides what to do.
# The task-specific layer lives outside the operator: the **ml4t-data /
# ml4t-engineer / ml4t-diagnostic / ml4t-backtest** libraries provide the
# runtime, and the companion **[ml4t/skills](https://github.com/ml4t/skills)**
# repository (versioned `SKILL.md` files with `WRONG/CORRECT` patterns and
# library callouts) provides the discipline.
# Skills are a key feature of the book: they distill the methodology that
# earlier chapters teach in long-form prose into a corpus the agent can
# consult on demand. The operator stitches the LLM, the libraries, and the
# skills together.
#
# We point the operator at the ETFs case study and ask it to execute the
# **§20.9 next-step suggestion** verbatim:
#
# > *Ensemble GBM, tabular deep learning, and the CAE configuration, and
# > evaluate whether the combined signal stabilizes holdout Sharpe.*
#
# **Learning Objectives**:
# - Inspect the ten-tool operator surface and how the agent discovers skills
#   on demand via `list_skills`/`read_skill`.
# - Replay a captured operator run end-to-end against the ETFs case-study
#   registry, including the agent's IC-vs-Sharpe diagnosis on the
#   §20.9 ensemble follow-up.
# - Read the side-by-side summary of two runs (ETFs negative result,
#   `us_firm_characteristics` quantified capacity hit) without leaving the
#   notebook.
# - Recognise where the operator + skills + libraries split responsibility:
#   the operator is a thin loop, the skills carry methodology, and the
#   libraries do the math.
#
# **Prerequisites**: [`04_research_agent`](04_research_agent.ipynb) (the agent loop),
# [`09_evaluation_and_governance`](09_evaluation_and_governance.ipynb) (evaluation),
# Ch20 §20.9 (case-study next-step suggestions). Familiarity with the
# `ml4t-diagnostic` API (`cross_sectional_ic_series`, `compute_ic_hac_stats`)
# is helpful but not required. The agent's trace shows them in context.
#
# This notebook re-displays saved traces by default. `RUN_LIVE = False` is the
# publication path and makes no API calls or model-supplied shell calls.

# %%
"""Replay (or run) one iteration of the ML4T Research Operator on ETFs §20.9."""

from __future__ import annotations

import json
import re

import matplotlib.pyplot as plt
import polars as pl
import research_operator as ro
from IPython.display import Markdown, display

from utils.paths import get_chapter_dir
from utils.style import COLORS, FIGSIZE, add_message_title, show_with_alt

# %% [markdown]
# ## Settings
#
# `RUN_LIVE` left at `False` replays the two captured operator runs and makes no API calls and
# no model-supplied shell calls. `True` runs the operator live, which needs `OPENROUTER_API_KEY`
# and executes whatever commands the model decides to issue; the security note below is about
# that path and only that path.
#
# `PRICE_IN_PER_MTOK` and `PRICE_OUT_PER_MTOK` are the DeepSeek v4 Pro rates on OpenRouter as of
# May 2026, in dollars per million tokens. They convert the recorded token counts into an
# approximate cost. Model pricing moves faster than anything else in this chapter, so treat the
# figure as an order of magnitude and change these two values rather than the arithmetic.

# %% tags=["parameters"]
RUN_LIVE = False
PRICE_IN_PER_MTOK = 0.50
PRICE_OUT_PER_MTOK = 1.50

# %%
NOTEBOOK_DIR = get_chapter_dir(24)
ARTIFACTS_DIR = NOTEBOOK_DIR / "operator_artifacts"
ETFS_TRACE = ARTIFACTS_DIR / "run_etfs_20260504T223150.json"
US_FIRMS_TRACE = ARTIFACTS_DIR / "run_us_firm_characteristics_20260504T225521.json"
DEFAULT_TRACE = ETFS_TRACE

# %% [markdown]
# ## Tool surface
#
# Ten tools, one compact operator. Seven are generic file, bash, SQL, and
# parquet primitives that any coding agent would expose. Two (`list_skills`,
# `read_skill`) make the standalone skills repo discoverable at runtime.
# One (`done`) terminates the loop with a structured summary.

# %%
for schema in ro.TOOL_SCHEMAS:
    fn = schema["function"]
    desc = " ".join(fn["description"].split())
    print(f"  {fn['name']:18s}: {desc[:96]}")

# %% [markdown]
# ## Skills as task-specific knowledge
#
# The skills repo is **the task-specific layer**. Each `SKILL.md` is a short
# concept-first teaching document (problem statement → WRONG/CORRECT example →
# `## Production Implementation` block pointing at the right `ml4t-*` library
# function). The operator does not embed them in its prompt; the agent
# *discovers* them when it needs them.
#
# The skill library is a separate companion repo. Clone it next to the code
# repo (the operator's default location), or anywhere, and point the
# `RESEARCH_OPERATOR_SKILLS_ROOT` env var at it:
#
# ```bash
# git clone https://github.com/ml4t/skills    # alongside the code repo
# # or: export RESEARCH_OPERATOR_SKILLS_ROOT=/path/to/skills
# ```
#
# If the library is missing, `list_skills`/`read_skill` return a clear hint
# instead of failing. The rest of the notebook still runs.

# %%
res = ro.tool_list_skills(category="validation")
if "error" in res:
    print(res["error"])
    print(res.get("hint", ""))
else:
    print(f"validation skills ({res['n_skills']}):")
    for s in res["skills"]:
        print(
            f"  {s['name']:32s} | library: {s['library'] or '(none)':18s} | {s['description'][:60]}"
        )

# %%
out = ro.tool_read_skill("walk-forward-cv")
if "error" in out:
    print(out["error"])
    print(out.get("hint", ""))
else:
    display(Markdown(out["content"][:1200] + "\n\n*…(truncated)*"))

# %% [markdown]
# ## The task
#
# The system prompt hands the agent §20.9's suggestion verbatim, plus the baseline it has to
# beat: the LSTM's validation and holdout Sharpe with their intervals, printed below as the
# agent received them. It is also told the one fact that decides the shape of the experiment,
# which is that only the LSTM has holdout predictions in the registry, and asked to choose
# between comparing on validation alone and retraining three model families to produce holdout
# predictions for the rest. Handing it the constraint and the choice rather than the answer is
# what makes the decision it reaches worth reading.

# %%
print(ro.CASE_STUDY_TASKS["etfs"])

# %% [markdown]
# ## Run vs replay
#
# By default, this notebook replays a saved trace from May 4, 2026. Setting
# `RUN_LIVE = True` launches a fresh run (requires `OPENROUTER_API_KEY`
# in the environment; budget ~\$1 on DeepSeek v4 Pro).
#
# **Security warning for live runs only.** A live run lets the model issue
# arbitrary `bash` commands with `shell=True` on the host. The
# `ML4T_OUTPUT_DIR` redirect and the directory allowlist are convenience
# guardrails for cooperative models. They are not a sandbox. A jailbroken
# or confused model can escape via shell redirection (`> ~/anything`,
# `rm -rf …`, network egress) at the host user's privileges. For live runs,
# isolate the host: run inside a container or firejail with restricted
# filesystem and network. The trace-replay path (`RUN_LIVE = False`)
# never executes model-supplied commands and is the only fully-safe option.

# %%
if not RUN_LIVE:
    trace_path = DEFAULT_TRACE
    print(f"Replaying saved trace: {trace_path.name}")
    result = json.loads(trace_path.read_text())
else:
    print("Running the operator live. This will spend money.")
    result = ro.run_operator()
    # Persist the same commit-ready form the pinned traces ship in: the host's
    # home directory is normalized to ``~`` so a captured trace can be committed
    # as a replay artifact without leaking the capture host's username.
    out_path = ARTIFACTS_DIR / "run_etfs_live.json"
    out_path.write_text(
        json.dumps(ro.sanitize_operator_trace_for_commit(result), indent=2, default=str)
    )
    print(f"Trace saved to: {out_path}")

# %% [markdown]
# The numbers below are read out of the trace rather than transcribed into the notebook. The
# operator's own comparison script printed a fixed block per model, and parsing that block is
# what keeps this table and the captured run from drifting apart: a re-captured trace changes
# the table, and a trace that no longer contains the block fails here instead of silently
# showing yesterday's figures.


# %%
_METRIC_BLOCK = re.compile(
    r"^\s*(?P<model>\S.*?):\s*\n"
    r"\s*IC=(?P<ic>-?[\d.]+), IC_IR=(?P<ic_ir>-?[\d.]+), t\(HAC\)=(?P<t>-?[\d.]+), p=[\d.]+\s*\n"
    r"\s*IC CI95=\[(?P<ic_lo>-?[\d.]+), (?P<ic_hi>-?[\d.]+)\], pct_pos=[\d.]+\s*\n"
    r"\s*Sharpe=(?P<sharpe>-?[\d.]+), CI95=\[(?P<sr_lo>-?[\d.]+), (?P<sr_hi>-?[\d.]+)\], "
    r"PSR p=(?P<psr>[\d.]+)\s*\n"
    r"\s*MaxDD=(?P<mdd>-?[\d.]+),",
    re.MULTILINE,
)


def _matched_comparison(trace: dict) -> tuple[str, dict[str, dict[str, float]]]:
    """Return the header and blocks of the last run that scored both models together.

    The operator scored several model sets during the session under different
    allocation configurations. Only a set containing both the baseline and the
    ensemble was scored under one allocation, so only that one is a comparison;
    taking the last block per model across the whole trace would put Sharpe
    figures from different allocators in the same table.
    """
    matched: tuple[str, dict[str, dict[str, float]]] | None = None
    for entry in trace["trace"]:
        result = entry.get("result")
        if not isinstance(result, dict):
            continue
        stdout = result.get("stdout_tail") or ""
        blocks: dict[str, dict[str, float]] = {}
        for match in _METRIC_BLOCK.finditer(stdout):
            row = match.groupdict()
            model = row.pop("model")
            blocks[model] = {k: float(v) for k, v in row.items()}
        has_pair = any(k.startswith("LSTM") for k in blocks) and any(
            k.startswith("ENSEMBLE") for k in blocks
        )
        if has_pair:
            header = next(
                (line.strip() for line in stdout.splitlines() if line.startswith("===")),
                "(allocation not recorded in the captured output)",
            )
            matched = (header, blocks)
    if matched is None:
        raise RuntimeError("the trace holds no run that scored both models under one allocation")
    return matched


# %%
allocation_header, etf_blocks = _matched_comparison(result)
baseline_key = next(k for k in etf_blocks if k.startswith("LSTM"))
ensemble_key = next(k for k in etf_blocks if k.startswith("ENSEMBLE"))
baseline, ensemble = etf_blocks[baseline_key], etf_blocks[ensemble_key]
print(f"Comparison scored under: {allocation_header}")


# %% [markdown]
# ## Run summary


# %%
def _run_cost(run: dict) -> float:
    """Approximate what a run cost, at the rates declared in the parameters cell."""
    cost = (
        run["total_in_tokens"] * PRICE_IN_PER_MTOK + run["total_out_tokens"] * PRICE_OUT_PER_MTOK
    ) / 1e6
    return round(cost, 2)


def _human_money(run: dict) -> str:
    """Format `_run_cost` for a printed summary."""
    return f"~${_run_cost(run):.2f}"


print(f"model:          {result['model']}")
print(f"case study:     {result.get('case_study', '(unset)')}")
print(f"turns:          {result['iterations']}")
print(f"tokens (in):    {result['total_in_tokens']:>12,}")
print(f"tokens (out):   {result['total_out_tokens']:>12,}")
print(f"elapsed:        {result['elapsed_s']:.0f}s")
print(f"approx cost:    {_human_money(result)}")
# %% [markdown]
# The agent's own closing summary describes this validation-window experiment as a holdout
# conclusion, which it is not. The raw artifact keeps that text unchanged, because an audit
# record that has been edited is not one. What follows states the same result within the
# evidence the run actually produced, and the gap between the two is the reason a human still
# reads the summary before anyone acts on it.

# %%
display(
    Markdown(
        "### What the captured run established\n\n"
        "The z-score ensemble raises mean cross-sectional IC from "
        f"{baseline['ic']:.4f} to {ensemble['ic']:.4f} and lowers validation Sharpe from "
        f"{baseline['sharpe']:.3f} to {ensemble['sharpe']:.3f}. Several ensemble folds carry "
        "negative IC where the baseline's stay positive. That is an association the run "
        "observed, not a mechanism it isolated, and the experiment produced no ensemble "
        "holdout result at all."
    )
)

# %% [markdown]
# ## Where the Turns Went
#
# The trace records every tool call, so the histogram below is what the run actually spent
# itself on rather than an impression of it. The shape is the thing to read: inspection of the
# registry and the files, discovery and reading of skills, and then a loop of writing, editing
# and running one experiment script. A run that is mostly reading has not got started; a run
# that is mostly running has stopped checking what it produced.
#
# How much of that shape is the agent's is worth asking of any count like this. The counts are
# the agent's choices made through a surface the operator built: ten tools and no others, one
# skill per `read_skill` call so consulting five means five calls, and a bash tool general
# enough that a whole experiment is one invocation of it. A different surface with the same
# agent behind it draws a different histogram. What the counts support is a comparison between
# runs on this surface, not a statement about how agents allocate effort in general.

# %%
calls = [
    {"turn": e["turn"], "tool": e["name"]} for e in result["trace"] if e.get("type") == "tool_call"
]
hist = (
    pl.DataFrame(calls)
    .group_by("tool")
    .agg(pl.len().alias("n_calls"))
    .sort("n_calls", descending=True)
)

fig, ax = plt.subplots()
ax.barh(
    hist["tool"].to_list(),
    hist["n_calls"].to_list(),
    color=COLORS["blue"],
)
ax.invert_yaxis()
ax.bar_label(ax.containers[0], padding=3)
ax.set_xlabel("Tool calls")
ax.set_ylabel("Operator tool")
add_message_title(
    ax,
    "Running experiments and reading the registry take most of the turns",
    subtitle=f"{result['iterations']} turns in the pinned ETFs replay",
)
show_with_alt(
    fig,
    f"Horizontal bar chart of tool-call counts across {int(hist['n_calls'].sum())} calls in "
    f"{result['iterations']} turns, led by "
    + ", ".join(f"{row['tool']} at {row['n_calls']}" for row in hist.head(3).iter_rows(named=True))
    + ".",
)

# %%
skill_reads = [
    e["args"].get("name_or_path")
    for e in result["trace"]
    if e.get("type") == "tool_call" and e.get("name") == "read_skill"
]
print("Skills consulted:")
for s in skill_reads:
    print(f"  {s}")

# %% [markdown]
# ## Result vs the §20.9 baseline
#
# Chapter 20 ranks the case study's models on holdout Sharpe and puts an LSTM at the top. The
# agent ran its ensemble on the validation window only, and said why: the registry holds
# holdout predictions for the LSTM alone, and producing them for the other three families
# means retraining all three, which costs an order of magnitude more than the experiment it
# ran. It then matched the LSTM baseline's exact backtest
# long-only) and computed IC + Sharpe with `ml4t.diagnostic.api`.

# %%
comparison = pl.DataFrame(
    [
        {
            "model": name,
            "eval_basis": "validation",
            "ic_mean": row["ic"],
            "ic_ir": row["ic_ir"],
            "val_sharpe": row["sharpe"],
            "val_sharpe_ci_lo": row["sr_lo"],
            "val_sharpe_ci_hi": row["sr_hi"],
            "psr_pvalue": row["psr"],
            "max_drawdown": row["mdd"],
        }
        for name, row in etf_blocks.items()
    ]
)
comparison

# %% [markdown]
# The intervals in that table are Sharpe intervals from the backtest, and they are the run's
# own. The IC column carries a separate uncertainty question the table does not show. The
# operator's script computed its IC t statistics with a five-lag HAC adjustment while the labels
# are 21-day forward returns, so consecutive observations overlap for twenty sessions and five
# lags does not span that overlap. A HAC correction that stops short of the dependence it is
# correcting for is not enough of one; how much it is out by is not something the run measures.
#
# The registry holds a 20-lag figure for the baseline, printed below beside the operator's, and
# nothing in the trace holds a 20-lag figure for the ensemble. So the two models' IC
# uncertainties cannot be compared with each other, and the point estimates are what the
# comparison rests on.

# %%
registry_lstm = next(
    row
    for entry in result["trace"]
    if entry.get("name") == "query_registry"
    for row in (entry.get("result") or {}).get("rows", [])
    if isinstance(row, dict)
    and row.get("config_name") == "lstm_h64"
    and row.get("split") == "validation"
    and "ic_t_hac" in row
)
print(f"Registry LSTM, 20-lag HAC:  t = {registry_lstm['ic_t_hac']:.4f}")
print(
    f"                            95% CI [{registry_lstm['ic_ci_lo']:.5f}, "
    f"{registry_lstm['ic_ci_hi']:.5f}]"
)
print(f"Operator script, 5-lag HAC: t = {baseline['t']:.4f}")

# %% [markdown]
# The two panels separate rank correlation from portfolio performance, which is the whole point
# of the run. Sharpe error bars are the intervals the operator's script recorded.

# %%
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["dual_h_tall"])
model_labels = [name.split(" (")[0].split("_")[0] for name in comparison["model"].to_list()]
axes[0].bar(
    model_labels,
    comparison["ic_mean"].to_list(),
    color=COLORS["blue"],
)
axes[0].set_ylabel("Mean cross-sectional IC")
axes[0].set_ylim(bottom=0)
axes[1].bar(
    model_labels,
    comparison["val_sharpe"].to_list(),
    color=COLORS["amber"],
)
axes[1].errorbar(
    model_labels,
    comparison["val_sharpe"].to_list(),
    yerr=[
        (comparison["val_sharpe"] - comparison["val_sharpe_ci_lo"]).to_list(),
        (comparison["val_sharpe_ci_hi"] - comparison["val_sharpe"]).to_list(),
    ],
    fmt="none",
    color=COLORS["neutral"],
    capsize=3,
)
axes[1].set_ylabel("Validation Sharpe ratio")
for ax in axes:
    ax.tick_params(axis="x", labelrotation=30)
add_message_title(
    axes[0],
    "The highest rank correlation is not the highest Sharpe",
    subtitle="Validation window only; Sharpe bars carry the intervals the run recorded",
)
show_with_alt(
    fig,
    f"Two bar charts over the same {comparison.height} models. On the left, mean "
    f"cross-sectional IC, where the ensemble is higher at {ensemble['ic']:.4f} against the "
    f"baseline's {baseline['ic']:.4f}. On the right, validation Sharpe ratio with confidence "
    f"intervals, where the ensemble is {ensemble['sharpe']:.2f} against the baseline's "
    f"{baseline['sharpe']:.2f} and its interval spans zero.",
)

# %%
display(
    Markdown(
        f"**What the run found.** The ensemble reaches the higher rank correlation, "
        f"{ensemble['ic']:.4f} against the baseline's {baseline['ic']:.4f}, and the "
        f"lower Sharpe, {ensemble['sharpe']:.2f} "
        f"against {baseline['sharpe']:.2f}, a difference of "
        f"{ensemble['sharpe'] - baseline['sharpe']:+.2f}. Its Sharpe interval "
        f"[{ensemble['sr_lo']:.2f}, {ensemble['sr_hi']:.2f}] spans zero; the baseline's does "
        f"not."
    )
)

# %% [markdown]
# The agent's own diagnosis, recorded in the trace, is that the ensemble's per-fold IC swings
# from negative to strongly positive while the baseline's stays positive and small, and that
# the `score_weighted_top_k` allocator turns that instability into portfolio losses by sizing
# positions on the score. That is an association the run observed rather than a mechanism it
# isolated, and it arrives at the point Chapter 20 already makes: the family with the highest
# isolated, and it arrives at the point Chapter 20 already makes: the family with the highest
# rank correlation is not the family with the highest portfolio Sharpe, and the allocator is
# where the two come apart. The operator was not told any of that.
#
# **What the run does not establish.** This is a validation-window result and the chapter's
# ranking is a holdout ranking, so it cannot displace it. Producing an ensemble holdout number
# would need holdout predictions for all three constituents, which the registry does not have,
# or a retrain of three model families, which costs an order of magnitude more than the
# experiment the operator ran. The agent chose the cheap path, said so, and reported a negative
# result rather than an improvement, which is the behaviour worth having.

# %% [markdown]
# ## Second case study: US firm characteristics, §20.9 mcap-quartile filter
#
# A second operator run on a different case study. Same operator, same skill
# repo, and the same library surface. Only `RESEARCH_OPERATOR_CASE_STUDY` and the
# `CASE_STUDY_TASKS` entry change. The §20.9 next-step suggestion for
# US firms is:
#
# > *Filter the universe to the top three quartiles by market capitalization
# > and re-run to see how the Sharpe behaves under realistic capacity.*
#
# §20.1 flags that this strategy's highest-Sharpe long and short legs both cluster in small-cap
# names, on validation and on holdout alike, which makes its headline figure a claim about
# stocks it may not be able to trade at size. The hypothesis: removing the bottom market-cap
# quartile erodes the Sharpe materially.

# %%
us_firms = json.loads(US_FIRMS_TRACE.read_text())

print(f"model:          {us_firms['model']}")
print(f"case study:     {us_firms.get('case_study', '(unset)')}")
print(f"turns:          {us_firms['iterations']}")
print(f"tokens (in):    {us_firms['total_in_tokens']:>12,}")
print(f"tokens (out):   {us_firms['total_out_tokens']:>12,}")
print(f"elapsed:        {us_firms['elapsed_s']:.0f}s")
print(f"approx cost:    {_human_money(us_firms)}")

# %% [markdown]
# The raw operator artifact remains unchanged for audit. Its interpretation
# overstates what a signal-level universe filter identifies, so the
# reader-facing replay labels the experiment by its actual evaluation scope.

# %%
display(
    Markdown(
        "### Captured validation sensitivity\n\n"
        "The pinned operator filtered existing validation predictions to the "
        "top three market-cap quartiles and reran the same backtest specification. "
        "It did not retrain the model or estimate market impact. The following "
        "results therefore measure sensitivity to a capacity-oriented signal "
        "screen, not the return of a scalable implementation."
    )
)

# %% [markdown]
# The agent's summary reports its comparison as a markdown table. Parsing that table is what
# keeps the figures below tied to the capture: the operator's script printed its full results
# past the end of the captured stdout, so the summary is the only complete record of them, and
# retyping its numbers into the notebook would put a second copy beside the artifact with
# nothing to keep the two in step.


# %%
def _summary_table(summary: str) -> dict[str, tuple[float, float]]:
    """Read the agent's markdown results table as {metric: (baseline, filtered)}."""
    parsed: dict[str, tuple[float, float]] = {}
    for line in summary.splitlines():
        cells = [c.replace("*", "").replace("\u2212", "-").strip() for c in line.split("|")]
        if len(cells) < 5:
            continue
        numbers = [re.match(r"-?[\d,.]+", c) for c in cells[2:4]]
        if all(numbers) and cells[1]:
            parsed[cells[1]] = tuple(float(m.group().replace(",", "")) for m in numbers)
    return parsed


# %%
us_metrics = _summary_table(us_firms["final_summary"])
us_firms_comparison = pl.DataFrame(
    [
        {
            "metric": metric,
            "baseline_full_universe": before,
            "top3_quartile_mcap": after,
            "change_pct": round((after - before) / abs(before) * 100, 1) if before else None,
        }
        for metric, (before, after) in us_metrics.items()
    ]
)
us_firms_comparison

# %%
us_panels = ["Sharpe", "IC mean (HAC)", "Assets/period"]
fig, axes = plt.subplots(len(us_panels), 1, figsize=FIGSIZE["grid_3x2"])
for ax, metric in zip(axes, us_panels, strict=True):
    before, after = us_metrics[metric]
    ax.barh(
        ["Full universe", "Top 3 quartiles"],
        [before, after],
        color=[COLORS["neutral"], COLORS["blue"]],
    )
    ax.bar_label(ax.containers[0], padding=3)
    ax.set_xlabel(metric)
    ax.set_xlim(left=0)
    ax.invert_yaxis()
add_message_title(
    axes[0],
    "Dropping the smallest quartile takes the Sharpe with it",
    subtitle="Validation-window signal filter; no retraining and no impact-cost estimate",
)
show_with_alt(
    fig,
    "Three horizontal bar charts comparing the full universe against the top three market-cap "
    "quartiles: "
    + "; ".join(f"{m} falls from {us_metrics[m][0]:g} to {us_metrics[m][1]:g}" for m in us_panels)
    + ".",
)

# %%
display(
    Markdown(
        "**What the filter did.** Removing the smallest quartile takes "
        f"{abs((us_metrics['Assets/period'][1] - us_metrics['Assets/period'][0]) / us_metrics['Assets/period'][0]):.0%}"
        f" of the universe with it. Sharpe falls from {us_metrics['Sharpe'][0]:.2f} to "
        f"{us_metrics['Sharpe'][1]:.2f}, mean IC from {us_metrics['IC mean (HAC)'][0]:.3f} to "
        f"{us_metrics['IC mean (HAC)'][1]:.3f}, and maximum drawdown deepens from "
        f"{us_metrics['Max Drawdown'][0]:.0%} to {us_metrics['Max Drawdown'][1]:.0%}. Turnover "
        "barely moves."
    )
)

# %% [markdown]
# The strategy's validation result depends materially on the names it is no longer allowed to
# hold. That is the finding the §20.9 suggestion was fishing for, and it arrives with two
# caveats the agent states and a reader should hold on to.
#
# The filter is applied to signals, not to a retrained model, so nothing here says what a model
# fitted on the larger-cap universe would find. And no market impact is estimated anywhere, so
# this measures sensitivity to a capacity screen rather than the return a scalable
# implementation would realize. Turnover barely moving is the tell: a signal filter does not
# change how the portfolio is constructed, only which names are eligible.
#
# It is tempting to reach for the fundamental law here, and worth being careful about what it
# says. It relates a portfolio's information ratio to the signal's IC and the number of
# independent bets, so a smaller universe lowers the information ratio at a *fixed* IC. It does
# not predict that IC itself falls, and it says nothing about drawdown. Both of those moved
# here, and both are separate empirical outcomes: the filter did not only shrink the universe,
# it changed which firms are in it, and this experiment does not separate the two.
#
# A natural follow-up the agent flagged: **retrain on the filtered universe**
# A natural follow-up the agent flagged: **retrain on the filtered universe**
# (rather than just signal-filter the existing predictions) to see whether
# the model can find alpha in the larger-cap names that the original training
# universe diluted with small-cap signal.

# %% [markdown]
# ## Two case studies, side by side
#
# The same operator loop, skills, and libraries handled both case studies.
# Only `RESEARCH_OPERATOR_CASE_STUDY` and the task configuration changed.
# The two outcomes differ, and the operator records both.

# %%
runs = [
    {
        "case_study": "etfs",
        "next_step": "Ensemble GBM+TabDL+CAE",
        "turns": result["iterations"],
        "tokens_in": result["total_in_tokens"],
        "tokens_out": result["total_out_tokens"],
        "elapsed_s": result["elapsed_s"],
        "cost_usd": _run_cost(result),
        "headline": "no improvement (SR 0.92 → 0.56); diagnosed IC-vs-SR gap",
    },
    {
        "case_study": "us_firm_characteristics",
        "next_step": "Top-3-quartile mcap filter",
        "turns": us_firms["iterations"],
        "tokens_in": us_firms["total_in_tokens"],
        "tokens_out": us_firms["total_out_tokens"],
        "elapsed_s": us_firms["elapsed_s"],
        "cost_usd": _run_cost(us_firms),
        "headline": "SR 4.27 → 2.24 (−48%); shows small-cap sensitivity",
    },
]
pl.DataFrame(runs)

# %% [markdown]
# ## What this demonstrates for Chapter 24
#
# 1. **The libraries are the tools.** The operator never imports anything
#    from `ml4t.*` directly. The LLM does, via `run_bash`, when it decides
#    that's the right move. `cross_sectional_ic_series` and
#    `compute_ic_hac_stats` got pulled because the agent read
#    `validation/evaluate-factor` and `concepts/information-coefficient`
#    and followed the `## Production Implementation` block.
# 2. **The skills are task-specific knowledge, not the agent's harness.**
#    `list_skills` returned a one-line summary per file; the agent picked
#    five and called `read_skill` on each. No skill content sits in the
#    system prompt; everything is pulled on demand.
# 3. **The operator is a thin loop.** `ro.run_operator()` mainly dispatches
#    schemas and records results; the domain logic stays in skills and
#    libraries.
#    Within the explicit task and tool constraints, the LLM chooses what
#    experiment to run, how to ensemble, and which backtest specification to
#    match, using the registry, skills, and libraries as evidence.
# 4. **Negative results are first-class results.** The agent did not
#    confabulate an improvement. It diagnosed the IC-vs-Sharpe gap and
#    reported `done()` with a defensible "no", matching the practitioner-
#    workflow discipline §20.9 prescribes.

# %% [markdown]
# ## Key Takeaways
#
# 1. **The operator is generic; the knowledge is not.** One loop handled both case studies, and
#    only the task description changed. The skills carry the methodology, the libraries do the
#    arithmetic, and neither is in the operator's prompt.
# 2. **General tools beat an enumerated action space once the experiment space is open.** The
#    forecasting agent earlier in this chapter had two actions because two were enough. A
#    research follow-up is a small program, and pre-enumerating the programs worth writing
#    across nine case studies is not a thing anyone can do.
# 3. **Knowledge fetched on demand scales; knowledge in the prompt does not.** The agent listed
#    the available skills, chose a handful, and read only those. The corpus can grow without
#    the system prompt growing with it.
# 4. **A negative result reported as a negative result is the behaviour to check for.** Both
#    runs found less than they were looking for and said so. An agent that confabulates an
#    improvement is worse than no agent, because its output looks like the thing you wanted.
# 5. **Read the agent's own conclusion against its own evidence.** The ETFs summary describes a
#    validation experiment as a holdout conclusion. The numbers in it are right and the claim
#    on top of them is not, which is the shape of overclaiming that gets past every check but
#    someone reading it.
# 6. **The trace is the deliverable.** Every command, output and decision is on disk, which is
#    what makes an autonomous run reviewable rather than merely repeatable.
#
# **Known limitations of what is built here.** Two runs, one model, two case studies: nothing
# here says how often the operator produces something worth having. Neither experiment
# retrained anything, so both measure sensitivity of an existing signal rather than what a
# model fitted for the new setting would do. A live run executes model-supplied shell commands
# at the host user's privileges, and the guardrails described above are conveniences rather
# than a sandbox.
#
# **Reader follow-ups**:
#
# - **Try a different case study.** Set
#   `RESEARCH_OPERATOR_CASE_STUDY=us_firm_characteristics` (or any of the
#   nine case studies) and add a one-paragraph §20.9 task to
#   `CASE_STUDY_TASKS`.
# - **Try a different model.** Set `RESEARCH_OPERATOR_MODEL` to any
#   OpenAI-compatible endpoint; model cost figures here are illustrative
#   and become stale quickly.
# - **Add a skill.** Drop a new `SKILL.md` under `~/ml4t/skills/{category}/`
#   with the standard frontmatter; `list_skills` picks it up at runtime
#   without a code change.
# - **Promote a successful pattern.** When an experiment produces a validated
#   improvement (this one did not), the agent's trace is the design record
#   for upgrading the headline configuration in the case-study registry.
#
# **Book**: Chapter 24 §24.8 frames the operator as the production-side
# counterpart to the forecasting workflow of §24.6–§24.7.
