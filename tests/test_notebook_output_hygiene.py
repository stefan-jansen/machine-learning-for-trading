"""Guards on what committed notebooks expose to readers.

Two hygiene defects have reached readers from committed ``.ipynb`` files:

* machine-specific absolute paths (``/home/<user>/...``) baked into cell
  outputs and papermill metadata, and
* an empty ``tags: []`` stamped on every cell by papermill, which desynced the
  notebook from its jupytext-paired ``.py`` and made JupyterLab refuse to open
  it (public issue #372).

Each test scans every tracked ``.ipynb`` and names the script that fixes it.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sanitize_notebook_paths import (  # noqa: E402
    _iter_notebooks,
    sanitize_notebook_text,
    source_home_path_leaks,
)
from strip_empty_cell_tags import (  # noqa: E402
    notebook_targets,
    paired_py_has_fossil,
    strip_text,
)


def test_no_machine_specific_paths_in_committed_notebooks() -> None:
    offenders: list[str] = []
    for nb in _iter_notebooks():
        raw = nb.read_text(encoding="utf-8")
        source_indexes = source_home_path_leaks(raw)
        if source_indexes:
            offenders.append(
                f"{nb.relative_to(REPO_ROOT)} (machine path in source cells {source_indexes})"
            )
        _, n = sanitize_notebook_text(raw)
        if n:
            offenders.append(f"{nb.relative_to(REPO_ROOT)} ({n})")
    assert not offenders, (
        "Notebooks leak machine-specific absolute paths in their committed "
        "outputs/metadata. Run `uv run python scripts/sanitize_notebook_paths.py` "
        "to fix:\n  " + "\n  ".join(offenders)
    )


def test_path_sanitizer_does_not_rewrite_cell_source() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": ['path = "/home/reader/ml4t/code/data/file.parquet"\n'],
                "metadata": {},
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": ["/home/reader/ml4t/code/data/file.parquet\n"],
                    }
                ],
            }
        ],
        "metadata": {},
    }

    clean, count = sanitize_notebook_text(json.dumps(notebook))
    clean_notebook = json.loads(clean)

    assert count == 1
    assert clean_notebook["cells"][0]["source"] == notebook["cells"][0]["source"]
    assert clean_notebook["cells"][0]["outputs"][0]["text"] == ["data/file.parquet\n"]
    assert source_home_path_leaks(json.dumps(notebook)) == [0]


KNOWN_MISSING_MATPLOTLIB = {
    "03_market_microstructure/03_itch_lob_analysis.ipynb": 1,
    "03_market_microstructure/17_databento_bar_sampling.ipynb": 1,
    "09_model_based_features/06_path_signatures.ipynb": 1,
    "09_model_based_features/11_hmm_regimes.ipynb": 5,
    "09_model_based_features/13_regime_as_feature.ipynb": 2,
    "case_studies/etfs/04_model_based_features.ipynb": 1,
}


class _TopLevelCallCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[ast.Call] = []

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _attribute_root(node: ast.expr) -> str | None:
    while isinstance(node, ast.Attribute | ast.Subscript):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _parse_cell(source: str) -> ast.Module | None:
    sanitized = "\n".join(
        "pass" if line.lstrip().startswith(("%", "!")) else line for line in source.splitlines()
    )
    try:
        return ast.parse(sanitized)
    except SyntaxError:
        return None


_PLT_FIGURE_METHODS = {
    "acorr",
    "angle_spectrum",
    "arrow",
    "axhline",
    "axhspan",
    "axline",
    "axvline",
    "axvspan",
    "axes",
    "bar",
    "barbs",
    "barh",
    "boxplot",
    "broken_barh",
    "bxp",
    "cohere",
    "contour",
    "contourf",
    "csd",
    "ecdf",
    "errorbar",
    "eventplot",
    "figure",
    "figimage",
    "fill",
    "fill_between",
    "fill_betweenx",
    "hexbin",
    "hist",
    "hist2d",
    "hlines",
    "imshow",
    "loglog",
    "magnitude_spectrum",
    "matshow",
    "pcolor",
    "pcolorfast",
    "pcolormesh",
    "phase_spectrum",
    "pie",
    "plot",
    "plot_date",
    "polar",
    "psd",
    "quiver",
    "quiverkey",
    "scatter",
    "semilogx",
    "semilogy",
    "specgram",
    "spy",
    "stackplot",
    "stairs",
    "stem",
    "step",
    "streamplot",
    "subplots",
    "subplot",
    "subplot2grid",
    "subplot_mosaic",
    "table",
    "tricontour",
    "tricontourf",
    "tripcolor",
    "triplot",
    "violin",
    "violinplot",
    "vlines",
    "xcorr",
}
_AXES_FIGURE_METHODS = _PLT_FIGURE_METHODS - {
    "axes",
    "figure",
    "subplot",
    "subplot2grid",
    "subplots",
    "subplot_mosaic",
} | {
    "bar3d",
    "contour3D",
    "plot3D",
    "plot_surface",
    "plot_trisurf",
    "plot_wireframe",
    "scatter3D",
    "stem3D",
}
_SNS_FIGURE_METHODS = {
    "barplot",
    "boxenplot",
    "boxplot",
    "catplot",
    "clustermap",
    "countplot",
    "displot",
    "distplot",
    "ecdfplot",
    "FacetGrid",
    "heatmap",
    "histplot",
    "jointplot",
    "JointGrid",
    "kdeplot",
    "lmplot",
    "lineplot",
    "pairplot",
    "PairGrid",
    "palplot",
    "pointplot",
    "regplot",
    "relplot",
    "residplot",
    "rugplot",
    "scatterplot",
    "stripplot",
    "swarmplot",
    "violinplot",
}


def _call_produces_matplotlib_figure(
    call: ast.Call, axes_receivers: set[str] | None = None
) -> bool:
    if not isinstance(call.func, ast.Attribute):
        return False
    root = _attribute_root(call.func)
    method = call.func.attr
    return (
        (root == "plt" and method in _PLT_FIGURE_METHODS)
        or (root in (axes_receivers or set()) and method in _AXES_FIGURE_METHODS)
        or (root == "sns" and method in _SNS_FIGURE_METHODS)
    )


def _top_level_call_starts_figure(call: ast.Call) -> bool:
    if not isinstance(call.func, ast.Attribute):
        return False
    return _attribute_root(call.func) in {"plt", "sns"} and _call_produces_matplotlib_figure(call)


def _matplotlib_helper_names(notebook: dict) -> set[str]:
    helpers = set()
    for cell in notebook.get("cells", []):
        tree = _parse_cell("".join(cell.get("source", [])))
        if tree is None:
            continue
        for definition in (node for node in tree.body if isinstance(node, ast.FunctionDef)):
            calls = [node for node in ast.walk(definition) if isinstance(node, ast.Call)]
            args = definition.args
            parameters = (*args.posonlyargs, *args.args, *args.kwonlyargs)
            axes_receivers = {
                arg.arg
                for arg in parameters
                if arg.arg.lower() in {"ax", "axs", "axes", "axis"}
                or (
                    arg.annotation is not None
                    and isinstance(arg.annotation, ast.Name | ast.Attribute)
                    and getattr(arg.annotation, "id", getattr(arg.annotation, "attr", ""))
                    in {"Axes", "PolarAxes"}
                )
            }
            if args.vararg is not None:
                axes_receivers.add(args.vararg.arg)
            if args.kwarg is not None:
                axes_receivers.add(args.kwarg.arg)
            if any(_call_produces_matplotlib_figure(call, axes_receivers) for call in calls):
                helpers.add(definition.name)
    return helpers


def _expects_matplotlib_png(source: str, helpers: set[str] | None = None) -> bool:
    tree = _parse_cell(source)
    if tree is None:
        return "plt.show(" in source

    collector = _TopLevelCallCollector()
    collector.visit(tree)
    calls = collector.calls
    helpers = helpers or set()
    uses_matplotlib = any(
        isinstance(call.func, ast.Attribute)
        and _attribute_root(call.func) in {"plt", "ax", "axes", "matplotlib"}
        for call in calls
    )
    calls_show = any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "show"
        and (_attribute_root(call.func) == "plt" or uses_matplotlib)
        for call in calls
    )
    # An ``ax`` call can intentionally continue a figure created in a prior cell.
    # Require that cell's eventual ``show`` instead of demanding a PNG from every
    # intermediate mutation. Axes calls inside plotting helpers remain detected.
    creates_plot = any(_top_level_call_starts_figure(call) for call in calls)
    calls_matplotlib_helper = any(
        isinstance(call.func, ast.Name) and call.func.id in helpers for call in calls
    )
    return calls_show or creates_plot or calls_matplotlib_helper


def _missing_matplotlib_outputs() -> dict[str, int]:
    """Return notebooks with rendered Matplotlib calls but no embedded PNG."""
    offenders: dict[str, int] = {}
    for nb_path in _iter_notebooks():
        notebook = json.loads(nb_path.read_text(encoding="utf-8"))
        helpers = _matplotlib_helper_names(notebook)
        count = 0
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            if not _expects_matplotlib_png("".join(cell.get("source", [])), helpers):
                continue
            has_png = any(
                "image/png" in output.get("data", {}) for output in cell.get("outputs", [])
            )
            if not has_png:
                count += 1
        if count:
            offenders[str(nb_path.relative_to(REPO_ROOT))] = count
    return offenders


def test_no_new_missing_matplotlib_outputs() -> None:
    offenders = _missing_matplotlib_outputs()
    regressions = sorted(
        path for path, count in offenders.items() if count > KNOWN_MISSING_MATPLOTLIB.get(path, 0)
    )
    assert not regressions, (
        "Notebooks call Matplotlib show without an embedded image/png output. "
        "Execute committed notebooks with the default renderers:\n  " + "\n  ".join(regressions)
    )


def test_known_missing_matplotlib_list_has_no_stale_entries() -> None:
    offenders = _missing_matplotlib_outputs()
    stale = sorted(
        path for path, count in KNOWN_MISSING_MATPLOTLIB.items() if offenders.get(path, 0) < count
    )
    assert not stale, (
        "These notebooks now embed every Matplotlib figure. Remove them from "
        "KNOWN_MISSING_MATPLOTLIB:\n  " + "\n  ".join(stale)
    )


def test_matplotlib_detector_covers_common_display_patterns() -> None:
    assert _expects_matplotlib_png("ax.plot(x, y); plt.show()")
    assert _expects_matplotlib_png("canvas, axes = plt.subplots(); canvas.show()")
    assert _expects_matplotlib_png("plt.figure()\nplt.plot(x, y)")
    assert _expects_matplotlib_png("plt.hexbin(x, y)")
    assert _expects_matplotlib_png("plt.ecdf(values)")
    assert _expects_matplotlib_png("plt.pcolor(values)")
    assert _expects_matplotlib_png("plt.stairs(values)")
    assert _expects_matplotlib_png("plt.polar(theta, radius)")
    assert _expects_matplotlib_png("ax.stackplot(x, y); plt.show()")
    assert _expects_matplotlib_png("sns.scatterplot(data=frame, x='x', y='y')")
    assert _expects_matplotlib_png("sns.palplot(palette)")
    assert _expects_matplotlib_png("sns.PairGrid(frame)")
    assert _expects_matplotlib_png("sns.JointGrid(data=frame, x='x', y='y')")
    assert not _expects_matplotlib_png("def build():\n    return plt.subplots()")
    assert not _expects_matplotlib_png("sns.despine()\nsns.set_palette('deep')")
    assert not _expects_matplotlib_png("penguins = sns.load_dataset('penguins')")
    assert not _expects_matplotlib_png("fig.update_layout(title='Plotly')\nfig.show()")
    assert _expects_matplotlib_png("chart = plot_splits(data)\nchart.show()", {"plot_splits"})

    non_rendering = {
        "cells": [
            {
                "source": [
                    "def configure():\n    sns.set_theme()\n\n"
                    "def cleanup():\n    plt.close('all')\n\n"
                    "def load():\n    return sns.load_dataset('penguins')\n"
                ]
            }
        ]
    }
    assert _matplotlib_helper_names(non_rendering) == set()

    axes_helpers = {
        "cells": [
            {"source": ["def chart(axis, x, y):\n    axis.plot(x, y)\n"]},
            {"source": ["def panels(axes, x, y):\n    axes[0].plot(x, y)\n"]},
        ]
    }
    assert _matplotlib_helper_names(axes_helpers) == {"chart", "panels"}

    method_collision = {"cells": [{"source": ["def initialize(array):\n    array.fill(0)\n"]}]}
    assert _matplotlib_helper_names(method_collision) == set()


KNOWN_BARE_PLOTLY_JSON = {"case_studies/etfs/03_financial_features.ipynb": 3}


def _is_bare_plotly_bundle(bundle: dict) -> bool:
    payload = bundle.get("application/json")
    is_plotly = isinstance(payload, dict) and "data" in payload and "layout" in payload
    has_rendered_plot = "application/vnd.plotly.v1+json" in bundle or "image/png" in bundle
    return is_plotly and not has_rendered_plot


def _bare_plotly_json_outputs() -> dict[str, int]:
    offenders: dict[str, int] = {}
    for nb_path in _iter_notebooks():
        notebook = json.loads(nb_path.read_text(encoding="utf-8"))
        count = 0
        for cell in notebook.get("cells", []):
            for output in cell.get("outputs", []):
                bundle = output.get("data", {})
                if _is_bare_plotly_bundle(bundle):
                    count += 1
        if count:
            offenders[str(nb_path.relative_to(REPO_ROOT))] = count
    return offenders


def test_no_new_bare_plotly_json_outputs() -> None:
    offenders = _bare_plotly_json_outputs()
    regressions = sorted(
        path for path, count in offenders.items() if count > KNOWN_BARE_PLOTLY_JSON.get(path, 0)
    )
    assert not regressions, "Plotly output is bare JSON and will not render:\n  " + "\n  ".join(
        regressions
    )


def test_known_bare_plotly_json_list_has_no_stale_entries() -> None:
    offenders = _bare_plotly_json_outputs()
    stale = sorted(
        path for path, count in KNOWN_BARE_PLOTLY_JSON.items() if offenders.get(path, 0) < count
    )
    assert not stale, "Remove repaired notebooks from KNOWN_BARE_PLOTLY_JSON:\n  " + "\n  ".join(
        stale
    )


def test_plotly_detector_validates_each_bundle_and_ignores_generic_json() -> None:
    plotly = {"application/json": {"data": [], "layout": {}}}
    generic = {"application/json": {"records": []}}
    rendered = {**plotly, "image/png": "encoded"}
    assert _is_bare_plotly_bundle(plotly)
    assert not _is_bare_plotly_bundle(generic)
    assert not _is_bare_plotly_bundle(rendered)
    assert not _is_bare_plotly_bundle({"image/png": "unrelated"})


# Notebooks still carrying the fossil, all in chapters not yet shipped to readers
# (case studies -> Beat 5+). They are desynced for additional reasons too, so the
# empty tags cannot be stripped in isolation: doing so is churn that leaves the
# notebook just as unopenable. Clear these before the beats that ship them; the
# list must only ever shrink, which the second test below enforces.
KNOWN_DESYNCED = frozenset(
    {
        "case_studies/cme_futures/10a_pca.ipynb",
        "case_studies/cme_futures/10b_stochastic_discount_factor.ipynb",
        "case_studies/crypto_perps_funding/05_evaluation.ipynb",
        "case_studies/crypto_perps_funding/_archive/11_autoencoder.ipynb",
        "case_studies/etfs/11a_pca.ipynb",
        "case_studies/etfs/11b_ipca.ipynb",
        "case_studies/etfs/11c_conditional_autoencoder.ipynb",
        "case_studies/etfs/11d_stochastic_discount_factor.ipynb",
        "case_studies/etfs/11e_supervised_autoencoder.ipynb",
        "case_studies/fx_pairs/06_linear.ipynb",
        "case_studies/nasdaq100_microstructure/05_evaluation.ipynb",
        "case_studies/sp500_equity_option_analytics/05_evaluation.ipynb",
        "case_studies/sp500_equity_option_analytics/06_linear.ipynb",
        "case_studies/sp500_equity_option_analytics/08_tabular_dl.ipynb",
        "case_studies/sp500_equity_option_analytics/11a_pca.ipynb",
        "case_studies/sp500_equity_option_analytics/11b_ipca.ipynb",
        "case_studies/sp500_equity_option_analytics/11c_conditional_autoencoder.ipynb",
        "case_studies/sp500_equity_option_analytics/11d_stochastic_discount_factor.ipynb",
        "case_studies/sp500_equity_option_analytics/11e_supervised_autoencoder.ipynb",
        "case_studies/sp500_options/01_feasibility_analysis.ipynb",
        "case_studies/sp500_options/05_evaluation.ipynb",
        "case_studies/us_equities_panel/04_model_based_features.ipynb",
        "case_studies/us_equities_panel/05_evaluation.ipynb",
        "case_studies/us_firm_characteristics/04_evaluation.ipynb",
        "case_studies/us_firm_characteristics/08a_ipca.ipynb",
        "case_studies/us_firm_characteristics/08b_conditional_autoencoder.ipynb",
        "case_studies/us_firm_characteristics/08c_stochastic_discount_factor.ipynb",
        "case_studies/us_firm_characteristics/08d_supervised_autoencoder.ipynb",
    }
)


def _empty_tag_offenders() -> dict[str, int]:
    """{relative path: count} for notebooks whose paired .py lacks the empty tags."""
    out: dict[str, int] = {}
    for nb in _iter_notebooks():
        if paired_py_has_fossil(nb):
            continue  # pair agrees; stripping one side is what would break it
        _, n = strip_text(nb.read_text(encoding="utf-8"))
        if n:
            out[str(nb.relative_to(REPO_ROOT))] = n
    return out


def test_no_empty_cell_tags_in_committed_notebooks() -> None:
    """Empty `tags: []` desyncs a notebook from its .py, so JupyterLab won't open it."""
    offenders = [f"{p} ({n})" for p, n in _empty_tag_offenders().items() if p not in KNOWN_DESYNCED]
    assert not offenders, (
        "Notebooks carry empty `tags: []` cell metadata their paired .py lacks, so "
        "JupyterLab shows a 'File Load Error' instead of the notebook (cf. public "
        "#372). Pass only the listed paths to "
        "`uv run python scripts/strip_empty_cell_tags.py`:\n  " + "\n  ".join(offenders)
    )


def test_empty_tag_stripper_accepts_explicit_notebook_targets() -> None:
    relative = "01_process_is_edge/factor_regimes.ipynb"
    assert notebook_targets([relative]) == [REPO_ROOT / relative]


def test_known_desynced_list_has_no_stale_entries() -> None:
    """The debt list must only shrink: a fixed notebook has to leave it.

    Entries whose notebook is absent are ignored, not stale: this file is mirrored
    to the public repo, which ships only a subset of the case studies.
    """
    offenders = _empty_tag_offenders()
    stale = sorted(e for e in KNOWN_DESYNCED - set(offenders) if (REPO_ROOT / e).exists())
    assert not stale, (
        "These notebooks are listed in KNOWN_DESYNCED but are now clean. Remove them "
        "from the list in this file so it cannot silently mask a regression:\n  "
        + "\n  ".join(stale)
    )
