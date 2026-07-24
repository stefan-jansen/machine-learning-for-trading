"""ML4T Research Operator — one autonomous iteration on a real case study.

A thin orchestrator (~600 lines) that loads case-study context, hands an LLM a
small set of general-purpose tools (registry SQL, file I/O, bash, parquet I/O,
skill discovery), and asks it to execute the §20.9 "what's next?" suggestion
for one of the chapter-20 case studies. The runtime is the deterministic ml4t
library stack; the discipline lives in the standalone skills repo at
`~/ml4t/skills`. The operator imports neither — both are reached through the
generic tool surface and discovered at run time.

Sandbox: writes are confined to `ML4T_OUTPUT_DIR=$SANDBOX` so the production
case-study registry is never mutated. Reads span the sandbox, the code repo,
and the skills repo. Bash and file edits stay inside the sandbox.

Companion notebook: `11_research_operator`.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

import polars as pl
from dotenv import load_dotenv
from openai import OpenAI

if _env_file := os.environ.get("RESEARCH_OPERATOR_ENV_FILE"):
    load_dotenv(_env_file)
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CODE_REPO = Path(os.environ.get("RESEARCH_OPERATOR_CODE_REPO", Path(__file__).resolve().parents[1]))
# Companion skill library (56 SKILL.md files). Clone it next to the code repo,
# or point RESEARCH_OPERATOR_SKILLS_ROOT at it.
SKILLS_REPO_URL = "https://github.com/ml4t/skills"
SKILLS_ROOT = Path(os.environ.get("RESEARCH_OPERATOR_SKILLS_ROOT", CODE_REPO.parent / "skills"))

CASE_STUDY_NAME = os.environ.get("RESEARCH_OPERATOR_CASE_STUDY", "etfs")
SANDBOX = Path(
    os.environ.get(
        "RESEARCH_OPERATOR_SANDBOX",
        f"/tmp/{CASE_STUDY_NAME}_operator_sandbox",
    )
)
CASE_STUDY = SANDBOX / CASE_STUDY_NAME
REGISTRY_DB = CASE_STUDY / "run_log" / "registry.db"

MAX_TOOL_CALLS = int(os.environ.get("RESEARCH_OPERATOR_MAX_CALLS", "60"))
BASH_TIMEOUT_S = int(os.environ.get("RESEARCH_OPERATOR_BASH_TIMEOUT", "600"))


def _select_openai_endpoint() -> tuple[str, str | None, str]:
    """Pick an OpenAI-compatible (api_key, base_url, model) for tool-calling.

    Priority matches ``agent_providers.create_llm_client`` for the providers
    that ship a tool-calling endpoint reachable via the OpenAI SDK:
    OPENAI_API_KEY (direct) > OPENROUTER_API_KEY (any model). Override the
    default model with ``RESEARCH_OPERATOR_MODEL``.

    Anthropic and Google can be reached via OpenRouter (e.g.
    ``RESEARCH_OPERATOR_MODEL=anthropic/claude-sonnet-4``).
    """
    explicit_model = os.environ.get("RESEARCH_OPERATOR_MODEL", "")

    if key := os.environ.get("OPENAI_API_KEY"):
        return key, None, explicit_model or "gpt-4.1-mini"
    if key := os.environ.get("OPENROUTER_API_KEY"):
        return (
            key,
            "https://openrouter.ai/api/v1",
            explicit_model or "anthropic/claude-sonnet-4",
        )
    raise RuntimeError(
        "Set OPENAI_API_KEY or OPENROUTER_API_KEY to run the operator. "
        "Tool-calling requires an OpenAI-compatible chat-completions endpoint."
    )


ARTIFACTS_DIR = Path(__file__).resolve().parent / "operator_artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


READ_ALLOWED_ROOTS = [SANDBOX, CODE_REPO, SKILLS_ROOT]


def _resolve_in_allowed(path: str) -> Path:
    """Resolve a path and verify it is inside one of the allowed read roots."""

    p = Path(path).expanduser().resolve()
    allowed = [r.resolve() for r in READ_ALLOWED_ROOTS]
    if not any(p == root or root in p.parents for root in allowed):
        raise ValueError(
            f"path {p} is outside allowed roots ({', '.join(str(r) for r in allowed)})"
        )
    return p


def _writable_in_sandbox(path: str) -> Path:
    """Resolve a path and verify it is inside the sandbox (writes only)."""
    p = Path(path).expanduser().resolve()
    if SANDBOX.resolve() != p and SANDBOX.resolve() not in p.parents:
        raise ValueError(f"writes restricted to sandbox; got {p}")
    return p


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def tool_query_registry(sql: str) -> dict[str, Any]:
    """Read-only SQL over the sandboxed registry. Returns rows + columns."""

    if any(
        kw in sql.lower() for kw in ("insert ", "update ", "delete ", "drop ", "alter ", "attach ")
    ):
        return {"error": "read-only: INSERT/UPDATE/DELETE/DROP/ALTER/ATTACH disallowed"}
    try:
        con = sqlite3.connect(f"file:{REGISTRY_DB}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        rows = list(con.execute(sql).fetchmany(200))
        cols = [d[0] for d in con.execute(sql).description] if rows else []
        con.close()
        return {
            "columns": cols,
            "rows": [dict(r) for r in rows],
            "n_rows_returned": len(rows),
            "truncated_at_200": len(rows) == 200,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def tool_read_file(path: str, max_bytes: int = 80_000) -> dict[str, Any]:
    """Read a file under the sandbox, code repo, or skills repo (read-only)."""

    try:
        p = _resolve_in_allowed(path)
        if not p.is_file():
            return {"error": f"not a file: {p}"}
        data = p.read_bytes()[:max_bytes]
        truncated = p.stat().st_size > max_bytes
        return {
            "path": str(p),
            "size_bytes": p.stat().st_size,
            "truncated": truncated,
            "content": data.decode("utf-8", errors="replace"),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def tool_edit_file(path: str, old_text: str, new_text: str) -> dict[str, Any]:
    """Exact-string replace in a sandboxed file."""

    try:
        p = _writable_in_sandbox(path)
        if not p.is_file():
            return {"error": f"not a file: {p}"}
        body = p.read_text(encoding="utf-8")
        if old_text not in body:
            return {"error": "old_text not found in file"}
        if body.count(old_text) > 1:
            return {"error": f"old_text matches {body.count(old_text)} sites; make it unique"}
        p.write_text(body.replace(old_text, new_text, 1), encoding="utf-8")
        return {"path": str(p), "replaced": True}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def tool_write_file(path: str, content: str) -> dict[str, Any]:
    """Create or overwrite a sandboxed file."""

    try:
        p = _writable_in_sandbox(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"path": str(p), "bytes_written": len(content)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def tool_run_bash(command: str, cwd: str | None = None) -> dict[str, Any]:
    """Run an LLM-supplied bash command via ``shell=True``.

    SECURITY THREAT MODEL — read before flipping ``REPLAY_TRACE = False``:

    This function executes arbitrary commands on the host with the calling
    user's permissions. The ``ML4T_OUTPUT_DIR`` redirect and the ``cwd``
    allowlist are convenience guardrails for cooperative agents — they are
    NOT a sandbox. A confused or jailbroken model can trivially escape via
    shell redirection (``> ~/anything``, ``rm -rf …``, ``tar c / | nc …``),
    pipes, or by spawning subprocesses outside ``cwd``.

    For live runs against an external LLM, isolate the host: run inside a
    container/firejail/seatbelt, drop privileges, mount only required paths
    read-only, and disable network egress unless you mean to allow it. The
    in-process trace replay (``REPLAY_TRACE = True``) does NOT execute any
    LLM-supplied commands and is the only fully-safe path in this module.
    """

    cwd_path = _resolve_in_allowed(cwd) if cwd else CODE_REPO
    env = os.environ.copy()
    env["ML4T_OUTPUT_DIR"] = str(SANDBOX)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PLOTLY_RENDERER", "json")
    env.setdefault("PYTHONUNBUFFERED", "1")
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd_path),
            env=env,
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT_S,
        )
        return {
            "exit_code": proc.returncode,
            "stdout_tail": proc.stdout[-4_000:],
            "stderr_tail": proc.stderr[-4_000:],
            "elapsed_s": round(time.time() - started, 1),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "error": f"timeout after {BASH_TIMEOUT_S}s",
            "stdout_tail": (exc.stdout or b"").decode(errors="replace")[-4_000:],
            "stderr_tail": (exc.stderr or b"").decode(errors="replace")[-4_000:],
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def tool_read_parquet(
    path: str, n_rows: int = 5, columns: list[str] | None = None
) -> dict[str, Any]:
    """Inspect a parquet file: schema, height, head, optional column subset."""

    try:
        p = _resolve_in_allowed(path)
        df = pl.read_parquet(p)
        if columns:
            df = df.select(columns)
        head_rows = df.head(n_rows).to_dicts()
        return {
            "path": str(p),
            "shape": list(df.shape),
            "columns": df.columns,
            "schema": {k: str(v) for k, v in df.schema.items()},
            "head": head_rows,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def tool_write_parquet(path: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Write a list of dict rows to a sandboxed parquet path."""

    try:
        p = _writable_in_sandbox(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame(rows)
        df.write_parquet(p)
        return {"path": str(p), "n_rows": df.height, "columns": df.columns}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _skills_missing_error() -> dict[str, str]:
    """Single source for the 'skills root missing' error + clone hint."""
    return {
        "error": f"skills root missing: {SKILLS_ROOT}",
        "hint": (
            f"Clone the companion skill library: `git clone {SKILLS_REPO_URL}` "
            "next to the code repo, or set RESEARCH_OPERATOR_SKILLS_ROOT to its path."
        ),
    }


def _parse_skill_frontmatter(text: str) -> dict[str, str]:
    """Cheap YAML-frontmatter parser for `name`, `description`, `library`."""

    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    block = text[3:end].strip()
    out: dict[str, str] = {}
    in_metadata = False
    for line in block.splitlines():
        stripped = line.rstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("metadata:"):
            in_metadata = True
            continue
        if in_metadata and (line.startswith("  ") or line.startswith("\t")):
            kv = stripped.lstrip()
            if ":" in kv:
                k, _, v = kv.partition(":")
                out[f"metadata.{k.strip()}"] = v.strip().strip('"')
            continue
        in_metadata = False
        if ":" in stripped:
            k, _, v = stripped.partition(":")
            out[k.strip()] = v.strip().strip('"')
    return out


def tool_list_skills(category: str | None = None) -> dict[str, Any]:
    """List ML4T skills (category-filtered if given) with one-line descriptions.

    Categories: backtest, concepts, data, features, infrastructure, portfolio,
    production, validation, workflows.
    """

    try:
        if not SKILLS_ROOT.is_dir():
            return _skills_missing_error()
        if category:
            search_root = SKILLS_ROOT / category
            if not search_root.is_dir():
                return {
                    "error": f"unknown category {category!r}",
                    "available_categories": sorted(
                        p.name
                        for p in SKILLS_ROOT.iterdir()
                        if p.is_dir() and not p.name.startswith(".")
                    ),
                }
            paths = sorted(search_root.glob("*/SKILL.md"))
        else:
            paths = sorted(SKILLS_ROOT.glob("*/*/SKILL.md"))

        skills = []
        for p in paths:
            try:
                fm = _parse_skill_frontmatter(p.read_text(encoding="utf-8"))
            except Exception:
                fm = {}
            skills.append(
                {
                    "name": fm.get("name", p.parent.name),
                    "category": p.parent.parent.name,
                    "path": str(p),
                    "description": fm.get("description", ""),
                    "library": fm.get("metadata.library", ""),
                }
            )
        return {"category": category, "n_skills": len(skills), "skills": skills}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def tool_read_skill(name_or_path: str) -> dict[str, Any]:
    """Read a SKILL.md by name (e.g. `walk-forward-cv`) or absolute path."""

    try:
        # Both the name-based and path-based branches need the skills root;
        # check once up front so either path returns the same clone hint.
        if not SKILLS_ROOT.is_dir():
            return _skills_missing_error()
        if "/" in name_or_path:
            p = _resolve_in_allowed(name_or_path)
        else:
            matches = list(SKILLS_ROOT.glob(f"*/{name_or_path}/SKILL.md"))
            if not matches:
                return {
                    "error": f"no skill matches {name_or_path!r}",
                    "hint": "use list_skills to discover names",
                }
            if len(matches) > 1:
                return {
                    "error": f"ambiguous skill name {name_or_path!r}",
                    "matches": [str(m) for m in matches],
                }
            p = matches[0]
        if not p.is_file():
            return {"error": f"not a file: {p}"}
        body = p.read_text(encoding="utf-8")
        return {"path": str(p), "size_bytes": len(body.encode("utf-8")), "content": body}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def tool_done(summary: str) -> dict[str, Any]:
    """The agent calls this to signal the iteration is complete."""
    return {"summary": summary}


# ---------------------------------------------------------------------------
# Tool dispatch & schema
# ---------------------------------------------------------------------------


TOOL_FUNCTIONS = {
    "query_registry": tool_query_registry,
    "read_file": tool_read_file,
    "edit_file": tool_edit_file,
    "write_file": tool_write_file,
    "run_bash": tool_run_bash,
    "read_parquet": tool_read_parquet,
    "write_parquet": tool_write_parquet,
    "list_skills": tool_list_skills,
    "read_skill": tool_read_skill,
    "done": tool_done,
}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_registry",
            "description": (
                "Run a read-only SQL query against the case-study registry "
                "(SQLite). Tables: training_runs, prediction_sets, prediction_metrics, "
                "fold_metrics, backtest_runs, backtest_metrics, backtest_fold_metrics, "
                "backtest_paired_metrics, causal_runs. Truncated to 200 rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file under the sandbox, ml4t code repo, or ml4t skills repo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_bytes": {"type": "integer", "default": 80000},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": ("Exact-string replace in a sandboxed file. old_text must be unique."),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a sandboxed file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "Run a bash command from the ml4t code-repo root with "
                "ML4T_OUTPUT_DIR pointed at the sandbox. Use this to invoke "
                "pipeline scripts or arbitrary `uv run python -c '...'`. "
                "Output is truncated to the last 4000 chars of stdout/stderr. "
                f"Timeout: {BASH_TIMEOUT_S}s."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string", "description": "optional working directory"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_parquet",
            "description": "Inspect a parquet file: schema, shape, head.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "n_rows": {"type": "integer", "default": 5},
                    "columns": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_parquet",
            "description": "Write a list of dict rows to a sandboxed parquet path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "rows": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["path", "rows"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": (
                "List ML4T skills with one-line descriptions and library tags. "
                "Filter by category (backtest, concepts, data, features, "
                "infrastructure, portfolio, production, validation, workflows). "
                "Each skill is a SKILL.md file with a WRONG/CORRECT pattern and a "
                "Production Implementation block pointing at the right ml4t-* library. "
                "Consult these before designing or implementing any experiment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "optional category filter",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill",
            "description": (
                "Read a SKILL.md by short name (e.g. `walk-forward-cv`) or absolute path. "
                "Returns the full skill body including WRONG/CORRECT examples and library "
                "recommendation."
            ),
            "parameters": {
                "type": "object",
                "properties": {"name_or_path": {"type": "string"}},
                "required": ["name_or_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": (
                "Signal the iteration is complete. Provide a clear summary: "
                "what experiment you ran, the metric delta vs baseline, which "
                "skills/libraries you used, and your honest recommendation."
            ),
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """\
You are the ML4T Research Operator. The book's Chapter 20 has produced first-pass
results for the {case_study_name} case study and recorded a specific next-step
suggestion. Your job is to execute that next step end-to-end, using the existing
ML4T pipeline, libraries, and skills.

THE TASK

{task_description}

THE SETUP

- Sandbox path: {sandbox}
- Case-study path: {case_study} (writable copy of the production case study)
- Registry: {registry} (read-only via query_registry)
- Code-repo root for running scripts: {code_repo} (read-only)
- Pipeline scripts live at {case_study}/NN_*.py (jupytext format). Run via:
    `uv run python case_studies/{case_study_name}/14_backtest.py` etc.
  When run from the code-repo root with ML4T_OUTPUT_DIR set (run_bash sets
  this for you), all writes redirect to the sandbox.
- The ml4t libraries (ml4t-data, ml4t-engineer, ml4t-diagnostic, ml4t-backtest)
  are installed in the active venv. Import them in any script you write.

SKILLS — USE THEM

The ML4T skills repo lives at {skills_root} and contains 56 SKILL.md files,
each with a WRONG/CORRECT pattern and a `## Production Implementation` block
that points at the right `ml4t-*` library function. Categories: backtest,
concepts, data, features, infrastructure, portfolio, production, validation,
workflows.

BEFORE you implement anything, call `list_skills` (optionally filtered by
category) to see what guidance is available. When a skill matches your task
(e.g. ensembling predictions touches `validation/walk-forward-cv`,
`infrastructure/registry-system`, `concepts/information-coefficient`,
`validation/deflated-sharpe`), call `read_skill` to read it in full and follow
the CORRECT pattern + library recommendation.

The skills are not optional reading. They encode the reviewer-grade discipline
this case study is built on. Pulling library functions through them is the
whole point of the operator-skills-libraries triad.

WORKFLOW

1. Inspect the registry, README, and key configs to orient yourself.
2. Discover relevant skills (`list_skills`, then `read_skill` on the most
   relevant 3-6).
3. State a clear hypothesis and the experiment you will run, citing the skills
   and libraries you will use.
4. Execute it. Edit configs or write code as needed. Run pipeline scripts.
5. Read the new registry rows and/or output artifacts; compare to the baseline
   the chapter cites.
6. Call `done(summary=...)` with your conclusion. Be honest: report a
   no-improvement result as no-improvement.

BUDGET

You have at most {max_calls} tool calls. Be efficient. Prefer `query_registry`
over re-reading scripts you already understand. State your plan before running
expensive bash commands.

Begin by reading the case-study README and listing skills.
"""


CASE_STUDY_TASKS: dict[str, str] = {
    "etfs": (
        "ETFs case study, §20.9 next step:\n"
        "  *Ensemble GBM, tabular deep learning, and the CAE configuration, and\n"
        "  evaluate whether the combined signal stabilizes holdout Sharpe.*\n\n"
        "Baseline (rank-1 single model): deep_learning/lstm_h64 on fwd_ret_21d:\n"
        "  validation Sharpe +0.92 [+0.40, +1.49] (PSR p=0.005)\n"
        "  holdout Sharpe   +0.77 [-0.45, +2.17]\n"
        "  validation IC    +0.052 [+0.009, +0.095]\n"
        "  holdout IC       +0.093 [+0.037, +0.149]\n\n"
        "Concretely, your job is to:\n"
        "  1. Identify the rank-1 prediction set for each of GBM, tabular_dl, and\n"
        "     latent_factors|cae on label fwd_ret_21d.\n"
        "  2. Combine their signals (rank-average or z-score average — choose and\n"
        "     justify, citing the relevant skill).\n"
        "  3. Evaluate the ensemble's daily IC and per-fold IC stability vs each\n"
        "     of the three component models AND vs the LSTM rank-1 baseline.\n"
        "     Use ml4t-diagnostic for IC + HAC standard errors.\n"
        "  4. If the ensemble produces a stable signal, run the same backtest\n"
        "     stage the LSTM rank-1 went through (cost-aware, monthly cadence,\n"
        "     long-only top-N) and report the Sharpe delta vs the LSTM baseline.\n"
        "  5. Note: only the LSTM has a holdout prediction set in the registry.\n"
        "     Decide whether to (a) limit comparison to the validation window or\n"
        "     (b) retrain GBM/tabular_dl/CAE on the full pre-holdout window to\n"
        "     produce holdout predictions. Choose based on tractability."
    ),
    "us_firm_characteristics": (
        "US firm characteristics case study, §20.9 next step:\n"
        "  *Filter the universe to the top three quartiles by market capitalization\n"
        "  and re-run to see how the Sharpe behaves under realistic capacity.*\n\n"
        "Background (§20.1 binding constraint): the long leg's notional concentrates\n"
        "in bottom-quartile market-cap firms whose historical spreads likely run\n"
        "materially wider than the chapter's static cost assumption. The chapter's\n"
        "rank-1 strategy is reported at validation Sharpe 4.27 [3.51, 5.15], DSR 0.71\n"
        "(p≈0), then holdout Sharpe +2.48 [+0.67, +5.36] on a 12-month draw with\n"
        "wide CIs. This case study runs at MONTHLY cadence (not daily) on ~2,500 US\n"
        "firms, label `fwd_ret_1m` (regression) and `fwd_class_1m` (classification).\n\n"
        "Concretely, your job is to:\n"
        "  1. Identify the rank-1 prediction set in the registry — query\n"
        "     prediction_metrics joined to backtest_metrics. The chapter cites\n"
        "     gbm with binary classification (`fwd_class_1m`) on the highest-Sharpe\n"
        "     side; on `fwd_ret_1m` the leader is gbm/leaves_15_mae or sae. Pick\n"
        "     ONE rank-1 to test and justify the choice.\n"
        "  2. Locate market-cap data. The standard proxy in this case study is the\n"
        "     `LME` column (lagged market equity) in\n"
        "     `case_studies/us_firm_characteristics/features/financial.parquet`.\n"
        "     Confirm the schema before assuming.\n"
        "  3. For each rebalance date, compute the monthly mcap-quartile cutoff\n"
        "     across the universe and produce a filtered prediction set that\n"
        "     EXCLUDES the bottom-quartile names (i.e. keep top 3 quartiles).\n"
        "  4. Run the SAME backtest stage / spec the rank-1 used (long-short decile,\n"
        "     monthly rebalance, era-dependent costs). Match the original spec by\n"
        "     querying backtest_runs.spec_json for the rank-1 backtest_hash.\n"
        "  5. Compare Sharpe, IC, max drawdown, AND turnover/notional-per-leg\n"
        "     between the unfiltered baseline and the top-3-quartile filtered run.\n"
        "     The honest question is: does Sharpe survive the capacity filter, and\n"
        "     does max drawdown / cost-sensitivity behave as the chapter expects?\n"
        "  6. Use ml4t-diagnostic for IC + HAC; use the case-study `backtest_runner`\n"
        "     for the backtest. Match the rank-1 baseline's spec exactly before\n"
        "     reporting any delta — a spec mismatch will silently produce a wrong\n"
        "     Sharpe (this happened on the ETFs run; learn from it).\n"
        "  7. Stick to the SIGNAL-LEVEL filter (apply mcap mask to predictions\n"
        "     before backtest). Do NOT retrain the model on a filtered universe —\n"
        "     that's a follow-up experiment and would multiply runtime."
    ),
}


# ---------------------------------------------------------------------------
# Trace persistence
# ---------------------------------------------------------------------------


def sanitize_operator_trace_for_commit(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of an operator run dict with the capture host's home path normalized.

    The operator's tools echo absolute paths from the capture host into the
    trace: ``run_bash`` commands (``cd /home/<user>/ml4t/code && …``),
    ``read_skill``/``list_skills`` paths, and the tracebacks in ``run_bash``
    stderr. A committed replay artifact ships in version control, so it should
    not carry the capture host's username. This rewrites the home directory to
    ``~`` everywhere it appears and leaves everything else verbatim — the
    notebook's rendered cells (``final_summary``, the tool-call histogram, the
    consulted-skill list) are byte-identical except that the one full skill path
    the agent passed to ``read_skill`` now reads ``~/ml4t/skills/…``.

    Operator traces need no other redaction: the operator never reads ``.env``,
    has no network-egress tool, and queries only the book's own registry, so no
    secrets or third-party payloads reach the trace. The ``/tmp/…_sandbox_…``
    paths are ephemeral and carry no identifying information, so they are kept
    as the agent saw them.
    """
    home = str(Path.home())
    return json.loads(json.dumps(data, default=str).replace(home, "~"))


# ---------------------------------------------------------------------------
# Operator loop
# ---------------------------------------------------------------------------


def run_operator() -> dict[str, Any]:
    api_key, base_url, model = _select_openai_endpoint()
    client = OpenAI(api_key=api_key, base_url=base_url)

    task_description = CASE_STUDY_TASKS.get(
        CASE_STUDY_NAME,
        f"(no canonical task registered for {CASE_STUDY_NAME!r}; ask the user)",
    )

    system = SYSTEM_PROMPT.format(
        sandbox=SANDBOX,
        case_study=CASE_STUDY,
        case_study_name=CASE_STUDY_NAME,
        registry=REGISTRY_DB,
        code_repo=CODE_REPO,
        skills_root=SKILLS_ROOT,
        max_calls=MAX_TOOL_CALLS,
        task_description=task_description,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Execute the §20.9 next step for the {CASE_STUDY_NAME} case study. "
                "Follow the workflow in the system prompt. Cite the skills you used."
            ),
        },
    ]

    trace: list[dict[str, Any]] = []
    total_in_tokens = total_out_tokens = 0
    iteration = 0
    final_summary: str | None = None

    while iteration < MAX_TOOL_CALLS:
        iteration += 1
        print(f"\n=== turn {iteration} — calling {model} ===", flush=True)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                max_completion_tokens=4096,
                temperature=0.2,
            )
        except Exception as exc:
            print(f"!!! LLM call failed: {type(exc).__name__}: {exc}", flush=True)
            trace.append({"turn": iteration, "type": "llm_error", "error": str(exc)})
            break

        usage = getattr(resp, "usage", None)
        if usage is not None:
            total_in_tokens += usage.prompt_tokens
            total_out_tokens += usage.completion_tokens
            print(
                f"  tokens: prompt={usage.prompt_tokens} "
                f"completion={usage.completion_tokens} "
                f"(cumulative in={total_in_tokens} out={total_out_tokens})",
                flush=True,
            )

        choice = resp.choices[0]
        msg = choice.message
        if msg.content:
            print(f"  reasoning: {msg.content[:500]}", flush=True)
        messages.append(msg.model_dump(exclude_none=True))

        tool_calls = msg.tool_calls or []
        if not tool_calls:
            trace.append({"turn": iteration, "type": "final_text", "content": msg.content})
            final_summary = msg.content or "(model returned no tool call and no text)"
            break

        for call in tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                args = {"_parse_error": str(exc)}
            print(f"  -> tool: {name}({json.dumps(args)[:300]})", flush=True)
            fn = TOOL_FUNCTIONS.get(name)
            if fn is None:
                result: Any = {"error": f"unknown tool {name!r}"}
            else:
                try:
                    result = fn(**args)
                except TypeError as exc:
                    result = {"error": f"bad arguments to {name}: {exc}"}
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
            preview = json.dumps(result, default=str)[:300]
            print(f"     result: {preview}", flush=True)
            trace.append(
                {
                    "turn": iteration,
                    "type": "tool_call",
                    "name": name,
                    "args": args,
                    "result": result,
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str),
                }
            )
            if name == "done":
                final_summary = result.get("summary", "")

        if final_summary is not None:
            break

    return {
        "model": model,
        "case_study": CASE_STUDY_NAME,
        "sandbox": str(SANDBOX),
        "iterations": iteration,
        "total_in_tokens": total_in_tokens,
        "total_out_tokens": total_out_tokens,
        "final_summary": final_summary,
        "trace": trace,
    }


def main() -> None:
    run_id = time.strftime("%Y%m%dT%H%M%S")
    log_path = ARTIFACTS_DIR / f"run_{CASE_STUDY_NAME}_{run_id}.json"
    print(f"writing trace to {log_path}", flush=True)

    started = time.time()
    result = run_operator()
    result["elapsed_s"] = round(time.time() - started, 1)
    committable = sanitize_operator_trace_for_commit(result)
    log_path.write_text(json.dumps(committable, indent=2, default=str), encoding="utf-8")

    print()
    print("=" * 70)
    print(
        f"FINAL SUMMARY (model={result['model']}, "
        f"case_study={result['case_study']}, "
        f"turns={result['iterations']}, "
        f"in={result['total_in_tokens']}, out={result['total_out_tokens']}, "
        f"elapsed={result['elapsed_s']}s)"
    )
    print("=" * 70)
    print(result.get("final_summary") or "(no summary returned)")
    print()
    print(f"trace saved to: {log_path}")


if __name__ == "__main__":
    main()
