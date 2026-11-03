#!/usr/bin/env python3
"""A result number belongs in a figure, a table or a cell tagged `results` - not in prose.

Clause N1/N2 of `rules/notebook-standards.md`: numbers appear in figures and in cells tagged
`results`, nowhere else in prose, and no result value is typed into an untagged markdown cell.
The reason is mechanical rather than stylistic. Every value a notebook computes moves when the
notebook is re-executed, and a value typed into prose does not move with it, so the sentence
and the output beside it disagree and nothing in the repository can see that they do. A value
shown through `display(Markdown(f"..."))` or printed into a table cannot go stale, because it
is the number just computed.

What this gate buys is a read that no longer has to happen. Re-rendering a notebook used to
mean a human reading every markdown cell to find the numbers that moved; where this rule holds
there are none to find, and the read is the figures alone.

**The rule is gated against NEW sites, not swept.** Stefan's ruling of 2026-09-11 that
introduced the clause says it in as many words: apply it to whatever notebook is open and to
anything written from now on, and leave a violation alone in a notebook nobody is otherwise
touching. `.github/ci/prose-numerals.txt` lists the sites that exist today, one per line as
`<path>\t<numeral>\t<count>`. An unlisted site fails. A listed site that is gone also fails, so
the list shrinks as notebooks are re-executed for other reasons rather than rotting into a
description of a corpus that has moved on. The same shape as
`.github/ci/figure-prose-recomputed.txt`, and for the same reason: the fix is a code-cell
change that costs a re-execution.

What counts as a numeral here is deliberately narrow, because the cost of a false positive is
a red gate on an honest sentence. Only a decimal or a percentage counts - a bare integer is a
count, a year, a parameter or a page far more often than it is a result, and reading them all
would report every "two layers of 64 units" in the repository. Cross-references are skipped,
including ranges and lists of them ("Sections 19.4 (drawdowns), 19.7 and 19.8"), as are
numbered headings, inline code, math, link targets and URLs, each of which is a literal whose
staleness a different rule owns.

What it does NOT decide: whether a sentence narrates a result without naming one. "The ensemble
does better" goes stale exactly as a typed number does, and no regex reaches it;
`rules/notebook-standards.md` line 1012 already says so of clause N8. This file is the half a
machine can settle, which is also the half that was costing a human read.

Stdlib only, nothing imported or executed. `--selftest` proves each rule fires and does not fire.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ALLOWLIST = Path(__file__).resolve().parents[2] / ".github" / "ci" / "prose-numerals.txt"

# A decimal or a percentage. A bare integer is excluded: see the module docstring.
NUMERAL = re.compile(r"(?<![\w.$])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?(?![\w])")

# The words that introduce a cross-reference, optionally through "Reference" and any markdown
# emphasis or punctuation between: "Section 13.9", "**Section Reference**: 8.2", "Chapter 8,".
XREF_WORDS = (
    r"(?:chapters?|sections?|figures?|tables?|appendix|appendices|equations?|eq\.|§|parts?"
    r"|steps?|listings?|volumes?|editions?|notebooks?|versions?|python|cuda|isbn|rfc|pages?"
    r"|pp\.)"
)
XREF = re.compile(XREF_WORDS + r"(?:\s*\**\s*references?)?\s*\**\s*[:.\-]?[\s,]*$", re.I)
# The tail of one element of a cross-reference run, so "19.4 (drawdowns and recovery), 19.7"
# walks back to the "Sections" that introduced it.
XREF_RUN = re.compile(r"[\d.]+\s*(?:\([^()]*\))?\s*(?:[-–—,&]|\bto\b|\band\b|\s)+$", re.I)
XREF_HOPS = 8
XREF_WINDOW = 48

# A numeral whose staleness a different rule owns, blanked before the scan so offsets survive.
MASKED = re.compile(r"`[^`]*`|\$\$.*?\$\$|\$[^$\n]*\$|\[[^\]]*\]\([^)]*\)|https?://\S+", re.S)
# "### 12.3 D3.js Web Export" numbers the heading, it does not report a value.
HEADING_NUMBER = re.compile(r"^\s{0,3}#{1,6}\s+[\d.]*$")


def mask(text: str) -> str:
    """Blank code spans, math, link targets and URLs, keeping every offset."""
    return MASKED.sub(lambda m: " " * len(m.group(0)), text)


def introduces_cross_reference(text: str, start: int) -> bool:
    """True when the numeral at `start` continues a cross-reference, however long the run."""
    cursor = start
    for _ in range(XREF_HOPS):
        before = text[max(0, cursor - XREF_WINDOW) : cursor]
        if XREF.search(before):
            return True
        run = XREF_RUN.search(before)
        if run is None:
            return False
        cursor = max(0, cursor - XREF_WINDOW) + run.start()
    return False


def numerals(text: str):
    """Yield every result numeral typed into one markdown cell's text."""
    masked = mask(text)
    for match in NUMERAL.finditer(masked):
        token = match.group(0)
        bare = token.rstrip("%").replace(",", "")
        if "." not in bare and not token.endswith("%"):
            continue  # a bare integer is a count, a year or a parameter far more often
        line_start = masked.rfind("\n", 0, match.start()) + 1
        if HEADING_NUMBER.match(masked[line_start : match.end()]):
            continue
        if introduces_cross_reference(masked, match.start()):
            continue
        yield token


def markdown_cells(src: str):
    """Yield (text, tagged_results) for each jupytext percent markdown cell."""
    lines = src.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("# %% [markdown]"):
            tagged = "results" in lines[i]
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith("# %%"):
                line = lines[i]
                body.append(line[2:] if line.startswith("# ") else "")
                i += 1
            yield "\n".join(body), tagged
        else:
            i += 1


def check_source(src: str) -> dict[str, int]:
    """Count each result numeral typed into an untagged markdown cell of one notebook."""
    found: dict[str, int] = {}
    for text, tagged in markdown_cells(src):
        if tagged:
            continue
        for token in numerals(text):
            found[token] = found.get(token, 0) + 1
    return found


def notebook_sources(roots: list[Path]) -> list[Path]:
    """Every paired notebook source, and nothing else.

    A notebook source is a `.py` with a committed `.ipynb` beside it, which is what separates
    a notebook from the string fixtures under `tests/`. A named file is taken as given, so a
    scratch copy can still be checked directly. Dot-directories are skipped: `.venv` holds tens
    of thousands of files and none of them is a notebook.
    """
    out: list[Path] = []
    for root in roots:
        if root.is_file():
            out.append(root)
            continue
        for path in sorted(root.rglob("*.py")):
            if any(part.startswith(".") for part in path.parts):
                continue
            if path.with_suffix(".ipynb").exists():
                out.append(path)
    return out


def read_allowlist() -> dict[tuple[str, str], int]:
    """The sites that exist today, as {(path, numeral): count}."""
    allowed: dict[tuple[str, str], int] = {}
    if not ALLOWLIST.exists():
        return allowed
    for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        path, token, count = line.split("\t")
        allowed[(path, token)] = int(count)
    return allowed


SELFTEST_MUST_FIRE = [
    ("the mean validation IC is **+0.072** on the primary label", "+0.072"),
    ("Coverage is 76.0% of the keys the label file needs", "76.0%"),
    ("its probabilistic Sharpe p-value is 0.0107, so it survives", "0.0107"),
    ("the eight fold ICs run +0.018, -0.023 and +0.073 across folds", "+0.018"),
]

SELFTEST_MUST_NOT_FIRE = [
    "**Book Reference**: Chapter 13, Section 13.5 (Modern transformer variants).",
    "the authoritative comparison is the walk-forward evaluation in Section 13.9.",
    "## Book reference Sections 19.4 (drawdowns and recovery), 19.7 (adaptive controls), "
    "and 19.8 (kill switches).",
    "**Chapter 8: Feature Engineering** **Section Reference**: 8.2 - Price-Derived Features",
    "### 12.3 D3.js Web Export",
    "the window is `LOOKBACK` long and the stride is `patch_len / 2`",
    "under exchangeability the intervals have at least $1-\\alpha$ marginal coverage",
    "see [the Vovk reference](https://example.org/vovk/2005.1) for the proof",
    "two LSTM layers with a hidden state of 64 and 50 Monte Carlo samples",
    "Table 13.5 reports the results this notebook produces.",
]


def selftest() -> int:
    failures = []
    for text, expected in SELFTEST_MUST_FIRE:
        got = list(numerals(text))
        if expected not in got:
            failures.append(f"must fire but did not: {text!r} (found {got})")
    for text in SELFTEST_MUST_NOT_FIRE:
        got = list(numerals(text))
        if got:
            failures.append(f"must not fire but did: {text!r} (found {got})")
    tagged = '# %% [markdown] tags=["results"]\n# The validation IC is +0.072.\n'
    if check_source(tagged):
        failures.append("a cell tagged results must be exempt, and was not")
    untagged = "# %% [markdown]\n# The validation IC is +0.072.\n"
    if not check_source(untagged):
        failures.append("an untagged cell carrying the same sentence must fire, and did not")
    for failure in failures:
        print(failure)
    if failures:
        print(f"\n{len(failures)} selftest case(s) failed")
        return 1
    print(
        f"selftest: {len(SELFTEST_MUST_FIRE)} cases fire, "
        f"{len(SELFTEST_MUST_NOT_FIRE)} do not, tagging exempts"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()

    repo = Path(__file__).resolve().parents[2]
    roots = [p.resolve() for p in args.paths] or [repo]
    allowed = read_allowlist()
    problems: list[str] = []
    seen: dict[tuple[str, str], int] = {}

    for path in notebook_sources(roots):
        try:
            found = check_source(path.read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:  # an unreadable file is not a clean one
            print(f"{path}: cannot read: {exc}", file=sys.stderr)
            return 2
        try:  # a path under the repo reads better relative; one outside it stays absolute
            rel = str(path.resolve().relative_to(repo))
        except ValueError:
            rel = str(path)
        for token, count in sorted(found.items()):
            seen[(rel, token)] = count
            budget = allowed.get((rel, token), 0)
            if count > budget:
                problems.append(
                    f"{rel}: {token} is typed into prose "
                    f"({count} time(s), {budget} listed) - clause N1/N2 puts a result number "
                    "in a figure, a table, or a cell tagged results"
                )

    # A listed site that is gone means the notebook was re-executed and the line was not
    # deleted. Reporting it is what makes the list shrink rather than rot. Only on a
    # whole-repository run: pointed at one file, every other line is absent by construction.
    if roots == [repo]:
        for (rel, token), count in sorted(allowed.items()):
            if seen.get((rel, token), 0) < count:
                problems.append(
                    f"{ALLOWLIST.relative_to(repo)}: {rel} no longer types {token} into prose "
                    f"{count} time(s) - lower the count or delete this line"
                )

    for problem in problems:
        print(problem)
    if problems:
        print(f"\n{len(problems)} prose numeral(s) outside a figure, a table or a results cell")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
