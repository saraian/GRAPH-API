"""The dashboard's extension seam: extra pages, extra launcher settings, and the product name.

WHY. The dashboard ships in this repository, which may not name the package that extends it
(`lost3dsg/test/check_no_extension_refs.py` fails the build on such a reference). But the dashboard
genuinely has extension-specific surfaces: pages that render an extension's own artefacts, launcher
settings that are an extension's policy vocabulary, and the product name in the title bar. Deleting
them would make the upstream copy poorer for no reason; hard-coding them is what the boundary check
forbids.

So the dashboard declares WHAT KINDS of thing an extension may add, and an extension declares the
things. This is the same arrangement `run.sh` has with `EXT_ENV_FILE` and `preflight_gate.py`
has with `load_external_probes`: the repository ships the seam, the deployment ships the wiring.

WHAT AN EXTENSION MAY ADD
  brand      the product name in the page title and header. Default: the generic one.
  pages      extra routes, each with a render function and an optional tools-menu entry.
  env_groups extra groups in the launcher form, each a (name, blurb, [(var, default, kind, help)]).

HOW ONE REGISTERS. Set `DASHBOARD_EXT` to `module:attribute`, or call `register()` directly before
the app is built. Absent is normal and silent: a dashboard with no extension is a complete dashboard
of this stack alone, which is exactly the copy this repository is meant to ship.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Page:
    """One extension page. `route` is the path AND the menu href, so the two cannot drift.

    `render(bundle_name)` returns the page body as HTML; the dashboard wraps it in the tools menu.
    `menu_label` absent means the page exists but is not advertised — used for pages reached from
    another page rather than from the menu.
    """
    route: str
    render: Callable[[str], str]
    menu_label: str | None = None
    raw_response: bool = False
    """Register `render` VERBATIM instead of wrapping it.

    A page that only produces a body takes `bundle_name` and gets the tools menu around it. Some
    pages cannot: a concept resolver reads a query parameter and answers with a redirect, a 404 or
    HTML depending on the argument. Forcing those through the body-only shape would mean either
    losing the status codes or teaching this seam every response type. So the handler is registered
    as it stands and the framework reads its own signature; it renders its own chrome."""


@dataclass
class Extension:
    brand: str | None = None
    pages: list[Page] = field(default_factory=list)
    env_groups: list[tuple] = field(default_factory=list)


_EXT = Extension()
_LOADED = False


def register(ext: Extension) -> None:
    """Install an extension. Later calls replace the previous one; the dashboard reads it lazily."""
    global _EXT, _LOADED
    _EXT, _LOADED = ext, True


def _load_from_env() -> None:
    """`DASHBOARD_EXT=module:attr`. A NAMED extension that does not import is an ERROR, not a
    silent fallback: somebody meant to load something and it is not there. Absent is silent."""
    global _LOADED
    if _LOADED:
        return
    spec = os.environ.get("DASHBOARD_EXT", "").strip()
    _LOADED = True
    if not spec:
        return
    mod_name, _, attr = spec.partition(":")
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr or "EXTENSION")
    register(obj() if callable(obj) else obj)


def extension() -> Extension:
    _load_from_env()
    return _EXT


def brand(default: str = "Scene graph") -> str:
    return extension().brand or default


def pages() -> list[Page]:
    return list(extension().pages)


def env_groups() -> list[tuple]:
    return list(extension().env_groups)


def _selfcheck():
    import types
    global _EXT, _LOADED
    saved_env, saved_ext, saved_loaded = os.environ.get("DASHBOARD_EXT"), _EXT, _LOADED
    try:
        # 1. with nothing registered the dashboard is whole and generic: no pages, no extra
        #    settings, and a brand that names no product.
        _EXT, _LOADED = Extension(), False
        os.environ.pop("DASHBOARD_EXT", None)
        assert pages() == [] and env_groups() == []
        assert brand() == "Scene graph" and "OUND" not in brand()

        # 2. a registered extension is visible through every accessor
        ext = Extension(brand="Widget", env_groups=[("W", "widget policy", [("W_A", "1", "bool", "a")])],
                        pages=[Page("widget", lambda b: f"<p>{b}</p>", menu_label="WIDGET")])
        register(ext)
        assert brand() == "Widget" and len(pages()) == 1 and len(env_groups()) == 1
        assert pages()[0].route == "widget" and pages()[0].render("x") == "<p>x</p>"

        # 3. the accessors hand back COPIES: a caller that mutates its list must not edit the
        #    extension's own, or one page render could change the menu for every later request.
        pages().append(Page("sneak", lambda b: ""))
        env_groups().append(("S", "", []))
        assert len(pages()) == 1 and len(env_groups()) == 1, "an accessor leaked its internals"

        # 4. DASHBOARD_EXT loads by module:attr
        mod = types.ModuleType("_dash_ext_probe")
        mod.EXTENSION = Extension(brand="FromEnv")
        import sys
        sys.modules["_dash_ext_probe"] = mod
        _EXT, _LOADED = Extension(), False
        os.environ["DASHBOARD_EXT"] = "_dash_ext_probe:EXTENSION"
        assert brand() == "FromEnv", brand()

        # 5. a NAMED extension that cannot be imported RAISES. A silent fallback here would run the
        #    generic dashboard while somebody believed their pages were live -- the failure mode the
        #    gate's own external-probe loader is written to avoid.
        _EXT, _LOADED = Extension(), False
        os.environ["DASHBOARD_EXT"] = "_no_such_module_at_all:EXTENSION"
        try:
            brand()
            raise AssertionError("a named extension that does not import must raise")
        except ModuleNotFoundError:
            pass
        sys.modules.pop("_dash_ext_probe", None)

        # 6. this file must name no deployment; the token is built, not typed
        from pathlib import Path
        token = "F" + "OUND"
        hits = [ln.strip()[:60] for ln in Path(__file__).read_text().splitlines()
                if token in ln and "token" not in ln]
        assert not hits, f"the seam names a deployment: {hits}"
    finally:
        _EXT, _LOADED = saved_ext, saved_loaded
        if saved_env is None:
            os.environ.pop("DASHBOARD_EXT", None)
        else:
            os.environ["DASHBOARD_EXT"] = saved_env
    print("dash_ext self-check OK - generic by default, extension visible, named-but-missing raises")


if __name__ == "__main__":
    _selfcheck()
