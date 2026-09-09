"""The default Plotly renderer must not require Chrome.

Regression gate for #432. `utils` used to hard-code `plotly_mimetype+png`; the
`png` half drives kaleido, kaleido v1 drives a Chrome it does not bundle, and the
Docker images ship no Chrome. Every `fig.show()` in a figure-producing notebook
therefore died on ChromeNotFoundError.

The probe is deliberately written so that *any* failure means "no Chrome", since
this code runs at `import utils` and must never be what breaks the import. These
tests inject a fake `choreographer` rather than depending on what is installed on
the runner.
"""

from __future__ import annotations

import sys
import types
from importlib.metadata import PackageNotFoundError

import pytest

import utils
from utils import _default_plotly_renderer

WITH_PNG = "plotly_mimetype+png"
WITHOUT_PNG = "plotly_mimetype"


@pytest.fixture
def fake_choreographer(monkeypatch):
    """Install a stand-in `choreographer.browsers.chromium` with a scripted result."""

    def install(find_browser):
        chromium = types.ModuleType("choreographer.browsers.chromium")
        chromium.Chromium = type("Chromium", (), {"find_browser": staticmethod(find_browser)})

        browsers = types.ModuleType("choreographer.browsers")
        browsers.chromium = chromium
        root = types.ModuleType("choreographer")
        root.browsers = browsers

        for name, module in {
            "choreographer": root,
            "choreographer.browsers": browsers,
            "choreographer.browsers.chromium": chromium,
        }.items():
            monkeypatch.setitem(sys.modules, name, module)

    return install


@pytest.fixture
def fake_kaleido(monkeypatch):
    """Report a kaleido version regardless of what the runner has installed.

    The version decides whether the Chrome probe runs at all, so a test about the
    probe has to pin it: test-unit installs no kaleido, and without this the lookup
    would short-circuit before reaching the fake browser.
    """

    def install(version):
        def _version(name):
            if name == "kaleido":
                if version is None:
                    raise PackageNotFoundError(name)
                return version
            raise AssertionError(f"unexpected metadata lookup: {name}")

        monkeypatch.setattr("importlib.metadata.version", _version)

    return install


@pytest.fixture(autouse=True)
def _kaleido_v1(fake_kaleido):
    """Default every test to kaleido 1.x, the version whose probe these tests cover."""
    fake_kaleido("1.3.0")


def test_png_requested_without_a_browser_on_kaleido_0x(fake_kaleido, fake_choreographer):
    """kaleido 0.x bundles its own Chromium, so a missing system Chrome is irrelevant.

    The py312 container: the probe found no Chrome and PNG was dropped, while
    `fig.to_image()` worked there the whole time.
    """
    fake_kaleido("0.2.1")
    fake_choreographer(lambda **_: None)
    assert _default_plotly_renderer() == WITH_PNG


def test_png_dropped_when_kaleido_is_absent(fake_kaleido):
    """No kaleido at all means no PNG to ask for, whatever browsers exist."""
    fake_kaleido(None)
    assert _default_plotly_renderer() == WITHOUT_PNG


def test_png_requested_when_chrome_is_found(fake_choreographer):
    fake_choreographer(lambda **_: "/usr/bin/google-chrome")
    assert _default_plotly_renderer() == WITH_PNG


def test_png_dropped_when_chrome_is_absent(fake_choreographer):
    """The Docker case: kaleido is installed, Chrome is not."""
    fake_choreographer(lambda **_: None)
    assert _default_plotly_renderer() == WITHOUT_PNG


def test_png_dropped_when_the_probe_raises(fake_choreographer):
    """A choreographer API change must degrade, not break `import utils`."""

    def explode(**_):
        raise RuntimeError("choreographer changed its API")

    fake_choreographer(explode)
    assert _default_plotly_renderer() == WITHOUT_PNG


def test_png_dropped_when_choreographer_is_missing(monkeypatch):
    """Plotly without kaleido: the import inside the probe fails outright."""
    for name in list(sys.modules):
        if name == "choreographer" or name.startswith("choreographer."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "choreographer", None)

    assert _default_plotly_renderer() == WITHOUT_PNG


def test_env_var_still_wins(monkeypatch):
    """`PLOTLY_RENDERER=json` is how CI and papermill force a headless renderer."""
    import importlib

    import plotly.io as pio

    monkeypatch.setenv("PLOTLY_RENDERER", "json")
    pio.renderers.default = "browser"

    import utils

    importlib.reload(utils)

    assert pio.renderers.default == "browser", (
        "utils must leave the renderer alone when PLOTLY_RENDERER is set, "
        "so the env var chosen by CI/papermill survives"
    )
