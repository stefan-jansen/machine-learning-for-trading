#!/usr/bin/env python3
"""Classify a JUnit report's testcases so the weekly workflow can file and close issues.

The weekly jobs used to parse their JUnit reports with an inline regex in the
workflow YAML, twice, and `name="([^"]*)"` matched the ``name="`` inside
``classname="``. Every failure was therefore filed under the test class rather
than the notebook, which is why one `weekly-flake` issue absorbed every weekly
failure and never closed.

Parsing here instead of in the workflow buys three things: one copy of the rule,
an XML parser rather than a regex, and a `passed` list, which is what lets a
previously filed issue close itself once its notebook runs clean.

Outputs JSON on stdout::

    {"failed": ["22_rag_financial_research::02_x.py"], "passed": [...], "skipped": [...]}

A skipped testcase appears in neither `failed` nor `passed`: a notebook that
skipped did not pass, and closing its issue on a skip would hide the failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from xml.etree import ElementTree

# test_chapter_notebook[22_rag_financial_research::02_domain_embeddings_comparison.py]
#   -> 22_rag_financial_research::02_domain_embeddings_comparison.py
# test_etf_yahoo_reachable -> etf_yahoo
ID_STYLES = ("notebook", "drift")


def testcase_id(name: str, style: str) -> str:
    """Reduce a pytest testcase name to the id the issue title is keyed on."""
    if style == "notebook":
        start = name.find("[")
        if start == -1 or not name.endswith("]"):
            return name
        return name[start + 1 : -1]
    if style == "drift":
        return name.removeprefix("test_").removesuffix("_reachable")
    raise ValueError(f"unknown id style {style!r}; expected one of {ID_STYLES}")


def classify(xml: str, style: str) -> dict[str, list[str]]:
    """Split every testcase in a JUnit report into failed, passed and skipped ids."""
    root = ElementTree.fromstring(xml)
    outcomes: dict[str, list[str]] = {"failed": [], "passed": [], "skipped": []}
    for case in root.iter("testcase"):
        name = case.get("name")
        if not name:
            continue
        case_id = testcase_id(name, style)
        if case.find("failure") is not None or case.find("error") is not None:
            bucket = "failed"
        elif case.find("skipped") is not None:
            bucket = "skipped"
        else:
            bucket = "passed"
        if case_id not in outcomes[bucket]:
            outcomes[bucket].append(case_id)
    # A parametrization that failed once and passed on a rerun is still a failure
    # to report: pytest-rerunfailures writes both the failed attempt and the
    # passing one, and the id must not land in two buckets at once.
    outcomes["passed"] = [i for i in outcomes["passed"] if i not in outcomes["failed"]]
    outcomes["skipped"] = [
        i
        for i in outcomes["skipped"]
        if i not in outcomes["failed"] and i not in outcomes["passed"]
    ]
    return outcomes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("junit", type=Path, help="path to the JUnit XML report")
    parser.add_argument("--id-style", choices=ID_STYLES, required=True)
    args = parser.parse_args(argv)

    if not args.junit.exists():
        # The workflow calls this unconditionally; an absent report means the
        # pytest step died before writing one, and there is nothing to file.
        print(json.dumps({"failed": [], "passed": [], "skipped": []}))
        return 0

    print(json.dumps(classify(args.junit.read_text(), args.id_style)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
