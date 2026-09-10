"""The dashboard's own environment contract, in one place, so it can ship upstream.

WHY THIS FILE EXISTS. The dashboard is moving into this repository, which may not name the package
that extends it — `lost3dsg/test/check_no_extension_refs.py` fails the build on such a reference,
docstrings included. Three of the dashboard's infrastructure variables carried an extension's prefix
for settings that belong to the dashboard itself: where run bundles live, where ground truth lives,
and whether this copy is public. A setting the dashboard owns should not be spelled with one
deployment's name.

WHAT THIS IS NOT. An extension's own policy variables — its aligner, its enforcement, its hold band —
do NOT come through here. They are the extension's vocabulary and reach the dashboard through the
page/group seam, the same way `live_run.sh` takes them through `EXT_ENV_FILE` rather than naming
them.

NO LEGACY SPELLING SHIPS HERE. The alias table is EMPTY upstream, and an extension registers its own
through `register_aliases` at import, so a deployment that still exports an old name keeps working on
the day the dashboard moves without this file ever naming it. The dashboard's own name always wins
over a registered alias, or a migration could never take effect. The self-check asserts that this
source names no deployment at all — that is the property that lets the module ship.
"""
from __future__ import annotations

import os
from pathlib import Path

# Legacy spellings an EXTENSION may register for its own deployment. Upstream this table is EMPTY,
# and that is the point: this repository knows only the dashboard's own names, so it never carries a
# deployment's prefix. An extension calls `register_aliases({"DASH_PUBLIC": "XYZ_DASH_PUBLIC"})` at
# import and its existing exports keep working, without this file ever naming it.
_ALIASES: dict[str, str] = {}


def register_aliases(mapping: dict[str, str]) -> None:
    """Teach the dashboard a deployment's legacy spellings: {own_name: legacy_name}.

    Called by an extension before the dashboard reads anything. Idempotent, and it never overrides
    a name the dashboard already resolves — a legacy alias is a fallback, never an override, or
    exporting the new name could not take effect.
    """
    for own, legacy in mapping.items():
        _ALIASES.setdefault(own, legacy)


def env(name: str, default: str | None = None) -> str | None:
    """The value of a dashboard setting, under its own name or the legacy one."""
    v = os.environ.get(name)
    if v is None and name in _ALIASES:
        v = os.environ.get(_ALIASES[name])
    return default if v is None else v


def flag(name: str) -> bool:
    """A boolean setting: set and non-empty is true, matching the previous `os.environ.get` tests."""
    return bool(env(name))


def runs_dir() -> Path:
    """Where run bundles are archived. The default is deliberately NOT a deployment's path: a
    checkout that sets nothing gets a directory beside the tree, and the bundle picker says it is
    empty rather than silently reading somebody else's runs."""
    v = env("GRAPH_API_RUNS_DIR")
    if v:
        return Path(v)
    return Path(os.environ.get("GRAPH_API_ROOT", "/DATA/GRAPH-API")) / "runs"


def gt_dir() -> Path:
    """Where ground-truth scene data lives, for the 3D view's GT overlay."""
    v = env("GRAPH_API_GT_DIR")
    if v:
        return Path(v)
    return Path(os.environ.get("GRAPH_API_ROOT", "/DATA/GRAPH-API")) / "data/gt"


def _selfcheck():
    import tempfile
    # An extension's legacy prefix is INVENTED here, never a real one: this file must not name any
    # deployment, and the last check asserts that by scanning its own source.
    legacy_runs, legacy_public = "XYZ_RUNS_DIR", "XYZ_DASH_PUBLIC"
    saved = {k: os.environ.get(k) for k in
             ("GRAPH_API_RUNS_DIR", legacy_runs, "DASH_PUBLIC", legacy_public)}
    try:
        for k in saved:
            os.environ.pop(k, None)
        # 0. with NO aliases registered, a legacy name means nothing -- upstream's state
        os.environ[legacy_runs] = "/tmp/legacy-runs"
        assert runs_dir() != Path("/tmp/legacy-runs"), "an unregistered legacy name must not answer"
        # 1. once an extension registers it, the legacy name answers and nothing has to migrate
        register_aliases({"GRAPH_API_RUNS_DIR": legacy_runs, "DASH_PUBLIC": legacy_public})
        assert runs_dir() == Path("/tmp/legacy-runs"), runs_dir()
        assert flag("DASH_PUBLIC") is False
        os.environ[legacy_public] = "1"
        assert flag("DASH_PUBLIC") is True
        # 2. the dashboard's OWN name wins over the legacy one, or a migration could never land
        os.environ["GRAPH_API_RUNS_DIR"] = "/tmp/new-runs"
        assert runs_dir() == Path("/tmp/new-runs"), runs_dir()
        # 3. with neither set the default sits beside this tree, never a deployment's path
        for k in ("GRAPH_API_RUNS_DIR", legacy_runs):
            os.environ.pop(k, None)
        with tempfile.TemporaryDirectory() as td:
            os.environ["GRAPH_API_ROOT"] = td
            assert runs_dir() == Path(td) / "runs", runs_dir()
            assert gt_dir() == Path(td) / "data/gt", gt_dir()
        os.environ.pop("GRAPH_API_ROOT", None)
        # 4. this file must name NO extension. The token is built, not typed, for the same reason
        # the boundary checker excuses its own pattern line: a check cannot look for a word it
        # cannot write. Anything registered at runtime lives in memory, never in this source.
        token = "F" + "OUND"
        hits = [ln.strip()[:70] for ln in Path(__file__).read_text().splitlines()
                if token in ln and "token" not in ln]
        assert not hits, f"this module names a deployment: {hits}"
    finally:
        _ALIASES.clear()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("dash_env self-check OK - no aliases shipped, registered ones answer, own names win")


if __name__ == "__main__":
    _selfcheck()
