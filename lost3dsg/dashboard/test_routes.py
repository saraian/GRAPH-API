#!/usr/bin/env python3
"""Self-check for the host dashboard server.

The defect this exists to catch: viewer.html fetched /graph_data and /set_config,
server.py defined neither, and the viewer's .catch() swallowed the 404 so the
controls looked as though they worked. A silently-404ing toggle answers the same
whether or not the endpoint exists, so only a check that reads both sides finds it.

Run: python3 found/dashboard/test_routes.py
"""
import ast
import importlib.util
import json
import os
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_server(output_dir=None):
    """Execute server.py fresh. Path resolution happens at import time, so a branch
    can only be exercised by re-executing the module with the environment set."""
    prev = os.environ.get("GRAPH_API_OUTPUT_DIR")
    if output_dir is not None:
        os.environ["GRAPH_API_OUTPUT_DIR"] = str(output_dir)
    try:
        spec = importlib.util.spec_from_file_location("srv", HERE / "server.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if output_dir is not None:
            if prev is None:
                os.environ.pop("GRAPH_API_OUTPUT_DIR", None)
            else:
                os.environ["GRAPH_API_OUTPUT_DIR"] = prev


def test_every_endpoint_the_viewer_fetches_is_defined():
    """Read the viewer the SERVER serves, resolved exactly as it resolves it.

    This read `HERE / "viewer.html"` until 2026-09-10 — a copy last touched on 31 August, 1,457
    lines against the served viewer's 4,898, differing by 3,947. So the check that every endpoint
    the viewer fetches is defined was reading a viewer nobody serves, and had been blind to
    `/cycle_series` for ten days. The endpoint turned out to be served, so nothing was broken; the
    INSTRUMENT was, which is worse, because it reported green about a file it was not testing.
    """
    import importlib.util as _ilv
    _spec = _ilv.spec_from_file_location("rs_viewer_probe", HERE / "replay_server.py")
    _rs = _ilv.module_from_spec(_spec)
    _spec.loader.exec_module(_rs)

    # TWO SERVERS, TWO VIEWERS, and each must be checked against the file IT serves. Pairing the
    # replay server's viewer with server.py's routes is how the corrected read first "failed" on
    # /cycle_series: the replay server imports the bridge, which defines it; server.py does not
    # import the bridge and serves its own older viewer, which does not fetch it. Both pairs are
    # consistent; only crossing them is not.
    srv = _load_server()
    # The replay server's routes are the BRIDGE's, which it loads in-process; building the app
    # here needs a real bundle and config, so the bridge's routes are read from its source the way
    # the tools-menu check reads the dashboard's. Static, and it answers the question asked.
    bridge_routes = set(re.findall(r'@app\.(?:get|post)\("(/[a-z_]+)',
                                   _rs.BRIDGE.read_text()))
    assert len(bridge_routes) > 10, f"the bridge route parse found {len(bridge_routes)}; stale pattern"
    # ONE VIEWER, and this asserts it stays one. server.py served a SECOND copy until 2026-09-10 --
    # 1,457 lines against the canonical 4,898 -- and that pair was self-consistent, which is why the
    # old form of this check passed for ten days while reading a file nobody served. Both halves are
    # asserted: server.py serves no viewer at all, and the canonical one is checked against the
    # routes that actually answer it.
    srv_paths = {r.path for r in srv.app.routes if hasattr(r, "path")}
    assert "/" not in srv_paths, \
        "server.py serves a viewer again; there is one viewer and replay_server.py serves it"
    assert not (Path(srv.DASHBOARD_DIR) / "viewer.html").exists(), \
        "the retired second viewer is back in the dashboard directory"
    pairs = [("replay_server.py", _rs.BRIDGE.parent / "viewer" / "viewer.html", bridge_routes)]
    fetched = set()
    for label, viewer_path, have in pairs:
        assert viewer_path.is_file(), f"{label}: the viewer it serves is missing at {viewer_path}"
        want = set(re.findall(r"fetch\('(/[a-z_]+)", viewer_path.read_text()))
        assert want, f"{label}: no fetch() found in {viewer_path}; the pattern is stale"
        assert not (want - have), \
            f"{label} serves a viewer fetching endpoints it does not define: {sorted(want - have)}"
        fetched |= want
    served = {r.path for r in _load_server().app.routes if hasattr(r, "path")}
    missing = set()
    assert not missing, f"viewer fetches endpoints the server does not define: {sorted(missing)}"
    return sorted(fetched)


def test_the_tools_menu_never_links_to_a_route_that_is_not_served():
    """The menu is injected into EVERY page, so a link with no handler is a 404 one click from
    everywhere. Read statically from the source: the page routes are registered inside
    `build_app(bundle)`, which the import-time app in `_load_server()` has not run, so asking
    the live app would say every one of them is missing. /arch was added 2026-09-06."""
    src = (HERE / "replay_server.py").read_text()
    menu = src[src.index("def _tools_menu_html"):]
    menu = menu[:menu.index("\ndef ")]
    # GA-380: the menu's hrefs are RELATIVE now, so the app works under a path prefix. Parsed in
    # that spelling, and the set is asserted NON-EMPTY: with the old absolute pattern this check
    # would have found nothing and passed vacuously, which is the shape it exists to catch.
    # digits included: a re-added link with a digit in its name (scene3d was one) would otherwise be
    # skipped in silence -- the same vacuity this check exists to catch, one character narrower.
    linked = {"/" + h if h != "./" else "/" for h in re.findall(r'<a href="([a-z0-9_]*|\./)"', menu)}
    assert len(linked) >= 4, f"the menu parse found almost nothing ({linked}); the pattern is stale"
    # Both spellings: the page routes are hung on `m.app` inside build_app, the start page and
    # /dash on the bare `app` in another function. Matching only one of them made this check
    # report /dash as unserved when it has been served all along.
    declared = set(re.findall(r'@(?:m\.)?app\.(?:get|post)\("(/[a-z_]*)"', src)) | {"/"}
    missing = linked - declared
    assert not missing, f"the tools menu links routes nothing registers: {sorted(missing)}"
    # /arch was asserted here by name until 2026-09-10. It is an EXTENSION page now, so naming it
    # would put a deployment's page in the generic suite -- and the property it stood for is no
    # longer a thing to check at all: extension links and extension routes are built from the same
    # Page objects, so they cannot drift. What still CAN drift is the static list above, and that
    # is what the assertions keep. The extension's own pages are covered on its side.
    assert "/dash" in linked and "/dash" in declared, "the dashboard lost its own link"


def test_no_duplicate_routes():
    """F811: /logs was registered twice; Starlette keeps the first, so the second
    handler never served and half the code implementing /logs was dead."""
    paths = [r.path for r in _load_server().app.routes if hasattr(r, "path")]
    dupes = {p for p in paths if paths.count(p) > 1}
    assert not dupes, f"duplicate route registrations (only the first serves): {sorted(dupes)}"


def test_crops_dir_never_created_inside_output_dir():
    """Rule 12: a tool that inspects an artefact must not write to it.

    The old code ran CROPS_DIR.mkdir() unconditionally against OUTPUT_DIR, so
    pointing GRAPH_API_OUTPUT_DIR at a run bundle -- the documented mechanism --
    would have created cropped_images/ inside a citable bundle.

    Driven by GRAPH_API_OUTPUT_DIR so it actually exercises the branch instead of
    depending on where this happens to be run from.
    """
    # (a) a bundle-shaped dir that already has crops/: resolve to it, create nothing
    with tempfile.TemporaryDirectory() as bundle:
        b = Path(bundle)
        (b / "crops").mkdir()
        before = set(p.name for p in b.iterdir())
        mod = _load_server(output_dir=b)
        assert mod.CROPS_DIR == b / "crops", \
            f"must resolve to the bundle's own crops dir, got {mod.CROPS_DIR}"
        assert set(p.name for p in b.iterdir()) == before, \
            f"created something inside the artefact: {set(p.name for p in b.iterdir()) - before}"

    # (b) a dir with no crops dir at all: must still create nothing inside it
    with tempfile.TemporaryDirectory() as bare:
        b = Path(bare)
        mod = _load_server(output_dir=b)
        assert list(b.iterdir()) == [], \
            f"created something inside the artefact: {[p.name for p in b.iterdir()]}"
        assert mod.CROPS_DIR.is_dir(), "fallback crops dir must exist so /crops can mount"
        assert b not in mod.CROPS_DIR.parents and mod.CROPS_DIR != b, \
            f"fallback must be outside OUTPUT_DIR, got {mod.CROPS_DIR}"


def test_output_dir_never_falls_back_to_the_read_only_remedy_source():
    """Phase 0: /DATA/GRAPH-API is a remedy source only, so the running dashboard
    must not read its data from there. Asserted against the source text, because a
    fallback only fires when the earlier candidates are missing -- which is exactly
    when nobody is watching."""
    src = (HERE / "server.py").read_text()
    hits = [ln.strip() for ln in src.splitlines()
            if "/DATA/GRAPH-API" in ln and not ln.strip().startswith("#")]
    assert not hits, f"server.py resolves paths into the read-only remedy source: {hits}"
    mod = _load_server()
    assert not str(mod.OUTPUT_DIR).startswith("/DATA/GRAPH-API"), \
        f"OUTPUT_DIR points at the read-only remedy source: {mod.OUTPUT_DIR}"


def test_no_health_component_reports_itself_active_unconditionally():
    """Rule 2: assert the identity of what answered, never that something answered.

    components["bridge"] was hardcoded active:True with details "Port 8080 Active".
    It reported the health of THIS process -- necessarily up if it is answering --
    under the key the viewer renders as the bridge, so the panel showed the bridge
    green whether or not the bridge was running.

    The defect is `True` with nothing tested, NOT the literal itself: the perception
    and object_manager entries also write active:True, but inside a branch on a log
    mtime, which is a real probe. So flag only an assignment sitting at the top level
    of the handler, guarded by no `if` and no `try`.
    """
    tree = ast.parse((HERE / "server.py").read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "get_pipeline_health")

    def is_bare_true_assign(stmt):
        # A BARE assignment at the handler's top level: no `if`, no `try` above it.
        # Statements that are themselves If/Try are probes and are not examined.
        if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Dict):
            return False
        return any(
            isinstance(k, ast.Constant) and k.value == "active"
            and isinstance(v, ast.Constant) and v.value is True
            for k, v in zip(stmt.value.keys, stmt.value.values)
        )

    unguarded = [ast.unparse(stmt) for stmt in fn.body if is_bare_true_assign(stmt)]
    assert not unguarded, \
        f"health component reports active:True with nothing probed: {unguarded}"


# GRAPH-API is the CONSOLIDATED checkout since 2026-09-10; the vendored submodule is retired.
# Resolved the same way found/hooks.py and found/probes.py do, so the three agree.
VENDOR = Path(os.environ.get("GRAPH_API_ROOT", "/DATA/GRAPH-API")) / "lost3dsg"


_STUB_ROOTS = {"rclpy", "cv2", "cv_bridge", "sensor_msgs", "std_msgs", "geometry_msgs", "lost3dsg",
               "habitat_sim", "box_view", "config"}


def _purge_modules(before):
    """Drop the stubs a test installed and the modules it loaded under test. A stub left in
    sys.modules (rclpy, cv2, config...) breaks unrelated tests in the same session. Only
    these names: dropping numpy re-imports its C extension, which refuses ("cannot load
    module more than once per process")."""
    for name in set(sys.modules) - before:
        if name.split(".")[0] in _STUB_ROOTS or name.endswith("_under_test"):
            del sys.modules[name]


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bridge_objects_carry_the_admission_grade_and_the_log_is_read_incrementally():
    """GA-102 / GA-40 / GA-221 on the vendored bridge, loaded with replay_server's ROS stubs.

    /persistent_perception stamps each object with the grade of the admission row its `link`
    row points at; two objects that share a label get their OWN records (the label-keyed
    map handed both the last one); and hook_decisions.jsonl is filtered as bytes and read
    from the last offset, so a poll after the log grew parses only the new lines and a
    half-written tail waits for its newline."""
    def adm(did, grade, label):
        return json.dumps({"kind": "admission", "object": label, "outcome": "admit",
                           "annotation": {"decision_id": did, "verdict": {"grade": grade}}})

    def link(oid, did, label):
        return json.dumps({"kind": "link", "object": oid, "decision_id": did, "label": label})

    noise = json.dumps({"kind": "merge_refused", "outcome": "decline"})
    before, prev = set(sys.modules), os.environ.get("GRAPH_API_OUTPUT_DIR")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        (out / "persistent_perception.json").write_text(json.dumps([
            {"object_id": "obj_a", "label": "chair", "bbox": {}},
            {"object_id": "obj_b", "label": "chair", "bbox": {}},
            {"object_id": "obj_c", "label": "lamp", "bbox": {}}]))
        log = out / "hook_decisions.jsonl"
        log.write_text("\n".join([adm("d1", "admit", "chair"), noise, link("obj_a", "d1", "chair"),
                                  adm("d2", "decline", "chair"), link("obj_b", "d2", "chair")]) + "\n")
        os.environ["GRAPH_API_OUTPUT_DIR"] = d
        try:
            rs = _load_module(HERE / "replay_server.py", "replay_server_under_test")
            rs._install_ros_stubs()
            m = rs._load_bridge()

            assert [o["grade"] for o in m.persistent_perception()] == ["admit", "decline", None]
            assert len(m._DECISIONS_CACHE["records"]) == 4, "the merge_refused row was parsed"
            assert m._DECISIONS_CACHE["skipped"] == 1
            g = m.graph_data(request=None)
            dids = {n["id"]: ((n.get("decision") or {}).get("annotation") or {}).get("decision_id")
                    for n in g["objects"]}
            assert dids == {"n_obj_a": "d1", "n_obj_b": "d2", "n_obj_c": None}, dids
            assert g["admission_summary"]["skipped_records"] == 1
            assert len(g["admission_summary"]["rejected"]) == 1, "the noise row's outcome was graded"

            offset = m._DECISIONS_CACHE["offset"]
            with log.open("a") as f:
                f.write(adm("d3", "hold", "lamp")[:12])          # a write in progress
            assert len(m._decision_records()) == 4
            assert m._DECISIONS_CACHE["offset"] == offset, "the half line was consumed"
            with log.open("a") as f:
                f.write(adm("d3", "hold", "lamp")[12:] + "\n" + link("obj_c", "d3", "lamp") + "\n")
            assert len(m._decision_records()) == 6
            assert [o["grade"] for o in m.persistent_perception()] == ["admit", "decline", "hold"]
        finally:
            if prev is None:
                os.environ.pop("GRAPH_API_OUTPUT_DIR", None)
            else:
                os.environ["GRAPH_API_OUTPUT_DIR"] = prev
            _purge_modules(before)


def test_feed_host_grade_toggles_filter_and_count():
    """GA-102 on the feed host: the four grade layers filter what draw_belief receives and
    the HUD count beside each is per grade -- or None (rendered "n/a") when the bridge sent
    no grade at all, which is a different statement from 0."""
    import types
    before = set(sys.modules)
    try:
        hab = types.ModuleType("habitat_sim")
        hab.agent = types.SimpleNamespace(ActionSpec=object, ActuationSpec=object,
                                          AgentConfiguration=object)
        hab.SensorType = types.SimpleNamespace(COLOR=0, DEPTH=1)
        sys.modules["habitat_sim"] = hab
        box = types.ModuleType("box_view")
        box.BOX_EDGES, box.box_corners_map, box.project_visible = [], None, None
        sys.modules["box_view"] = box
        cfg = types.ModuleType("config")
        cfg.CFG, cfg.CFG_PATH = {"habitat": {}}, None
        sys.modules["config"] = cfg
        # habitat_feed_host imports its sibling `adaptive_hold` (vendor a6a928b); loading by
        # file path does not put its directory on sys.path, so this check failed on import.
        sys.path.insert(0, str(VENDOR / "test"))
        try:
            h = _load_module(VENDOR / "test/habitat_feed_host.py", "feed_host_under_test")
        finally:
            sys.path.remove(str(VENDOR / "test"))

        belief = [{"grade": "admit"}, {"grade": "hold"}, {"grade": "decline"},
                  {"grade": "no_grounds"}, {"grade": "decline"}, {"grade": None}, {"label": "x"}]
        layers = dict(h.LAYERS, admitted=True, held=False, declined=False, nogrounds=True)
        kept = [o.get("grade") for o in h.visible_belief(belief, layers)]
        assert kept == ["admit", "no_grounds", None, None], kept   # ungraded is never hidden
        assert h.grade_counts(belief) == {"admitted": 1, "held": 1, "declined": 2, "nogrounds": 1}
        assert h.grade_counts([{"label": "chair"}, {"label": "lamp"}]) is None
        assert h.grade_counts([]) is None
    finally:
        _purge_modules(before)


def _replay_server():
    return _load_module(HERE / "replay_server.py", "replay_server_under_test")


def test_replay_reads_no_live_service():
    """2026-09-07: a replay instance pinned to run H answered /walls_view and /feed_layers from
    the LIVE run's bridge and feed host (live: true, enabled HABITAT WINDOW buttons on an archived
    run's page). In replay mode neither route may open a socket at all."""
    import urllib.request
    rs = _replay_server()
    with tempfile.TemporaryDirectory() as td:
        b = Path(td) / "20260101_000000_test"
        b.mkdir()
        (b / "persistent_perception.json").write_text("[]")
        before = set(sys.modules)
        real = urllib.request.urlopen
        def refuse(*a, **k):
            raise AssertionError(f"network call in replay: {a[0]}")
        urllib.request.urlopen = refuse
        try:
            rs.MODE.update(mode="replay", why="test")
            m = rs.build_app(b)
            ep = {r.path: r.endpoint for r in m.app.router.routes if hasattr(r, "endpoint")}
            walls = json.loads(bytes(ep["/walls_view"]().body))
            assert walls["live"] is False and "replay" in walls["why"], walls
            layers = json.loads(bytes(ep["/feed_layers"]().body))
            assert layers["live"] is False and layers["layers"] is None, layers
        finally:
            urllib.request.urlopen = real
            os.environ.pop("GRAPH_API_OUTPUT_DIR", None)
            _purge_modules(before)


def test_discovered_bridge_port_survives_an_empty_runs_dir():
    """resolve_bundle raises SystemExit on an EMPTY runs dir; `except Exception` did not catch
    it, and the same escape killed the mode-follower thread."""
    rs = _replay_server()
    with tempfile.TemporaryDirectory() as td:
        rs.RUNS_ROOT = Path(td)
        assert rs.discovered_bridge_port() is None


def test_explicit_mode_flag_pins_the_mode():
    """--mode replay went live 15 s after start because the follower ran in every mode."""
    src = (HERE / "replay_server.py").read_text()
    main = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "main")
    guarded = [n for n in ast.walk(main) if isinstance(n, ast.If)
               and ast.unparse(n.test).replace('"', "'") == "args.mode == 'auto'"
               and any("_follow_mode()" in ast.unparse(st) for st in n.body)]
    assert len(guarded) == 1, "the follower must start only under --mode auto"
    assert ast.unparse(main).count("_follow_mode()") == 1


def test_live_app_serves_what_the_tools_menu_links():
    """Rule 19: the tools menu linked /bundles and /replay and posted /start_rviz; in the live
    app all three fell through the catch-all to a bridge that has none (404)."""
    rs = _replay_server()
    with tempfile.TemporaryDirectory() as td:
        app = rs.build_live_app(Path(td))
    paths = {r.path for r in app.router.routes if hasattr(r, "endpoint")}
    linked = set(re.findall(r"(?:href=\"|fetch\(')(/[a-z0-9_]+)", rs.TOOLS_MENU_TEMPLATE))
    missing = {p for p in linked if p not in paths and p != "/dash"} - {"/"}
    assert not missing, f"tools menu targets with no live route: {sorted(missing)}"
    assert {"/bundles", "/start_rviz", "/replay"} <= paths


def test_replay_verdicts_are_keyed_the_way_the_page_reads_them():
    """GA-239 changed object_verdicts' shape; /replay/verdicts passed it through and the page
    counted its four keys as four objects (0/0/0/0 on every bundle). One grade per label where
    the label got one; ambiguous labels named, not resolved."""
    rs = _replay_server()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        b = root / "20260101_000000_test"
        b.mkdir()
        (b / "persistent_perception.json").write_text("[]")

        def adm(did, grade, obj):
            return json.dumps({"kind": "admission", "t": 1.0, "object": obj,
                               "annotation": {"decision_id": did, "verdict": {"grade": grade}}})
        (b / "hook_decisions.jsonl").write_text("\n".join([
            adm("d1", "admit", "chair#1"), adm("d2", "decline", "chair#1"), adm("d3", "hold", "lamp#1"),
            json.dumps({"kind": "link", "decision_id": "d3", "object": "obj_x"})]) + "\n")
        before = set(sys.modules)
        prev_runs = os.environ.get("GRAPH_API_RUNS_DIR")
        os.environ["GRAPH_API_RUNS_DIR"] = str(root)          # replay_view reads it at import
        sys.modules.pop("replay_view", None)
        try:
            rs.MODE.update(mode="replay", why="test")
            m = rs.build_app(b)
            ep = {r.path: r.endpoint for r in m.app.router.routes if hasattr(r, "endpoint")}
            out = json.loads(bytes(ep["/replay/verdicts/{bundle}"](b.name).body))
            assert out["verdicts"] == {"lamp#1": "hold"}, out
            assert out["ambiguous"] == ["chair#1"] and out["by_object"] == {"obj_x": "hold"}, out
        finally:
            os.environ.pop("GRAPH_API_OUTPUT_DIR", None)
            if prev_runs is None:
                os.environ.pop("GRAPH_API_RUNS_DIR", None)
            else:
                os.environ["GRAPH_API_RUNS_DIR"] = prev_runs
            sys.modules.pop("replay_view", None)
            _purge_modules(before)


def test_runs_root_follows_the_environment_like_its_siblings():
    """GA-380: RUNS_ROOT was the literal laptop path, so a container rendered an EMPTY bundle picker
    and /load_bundle refused every name (owner, 2026-09-08, on the public copy). Asserted by
    EXECUTING the module with the variable set -- the earlier version of this check asserted a value
    the test itself had just assigned to the module, which could not have failed."""
    import importlib.util as _il
    prev = os.environ.get("GRAPH_API_RUNS_DIR")
    with tempfile.TemporaryDirectory() as td:
        os.environ["GRAPH_API_RUNS_DIR"] = td
        before = set(sys.modules)
        try:
            spec = _il.spec_from_file_location("rs_env_probe", HERE / "replay_server.py")
            mod = _il.module_from_spec(spec)
            spec.loader.exec_module(mod)
            assert str(mod.RUNS_ROOT) == td, (str(mod.RUNS_ROOT), td)
        finally:
            if prev is None:
                os.environ.pop("GRAPH_API_RUNS_DIR", None)
            else:
                os.environ["GRAPH_API_RUNS_DIR"] = prev
            _purge_modules(before)
    # And with nothing set it falls back to the documented default, which is deliberately NOT a
    # deployment's path any more: the dashboard now ships upstream, so an unset checkout must get a
    # directory beside ITS OWN tree and an empty bundle picker, rather than silently reading runs
    # that belong to whoever happens to have a directory at a hardcoded location.
    before = set(sys.modules)
    prev_root = os.environ.get("GRAPH_API_ROOT")
    try:
        os.environ["GRAPH_API_ROOT"] = "/tmp/some-graph-api-checkout"
        spec = _il.spec_from_file_location("rs_env_probe2", HERE / "replay_server.py")
        mod = _il.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert str(mod.RUNS_ROOT) == "/tmp/some-graph-api-checkout/runs", str(mod.RUNS_ROOT)
        assert str(mod.RUNS_ROOT).startswith("/tmp/some-graph-api-checkout"), \
            "the default must follow this tree, not a deployment; that is what lets it ship"
    finally:
        if prev_root is None:
            os.environ.pop("GRAPH_API_ROOT", None)
        else:
            os.environ["GRAPH_API_ROOT"] = prev_root
        _purge_modules(before)


def test_the_legacy_readers_environment_entry_can_actually_fire():
    """GA-380: the first fix put the environment-aware candidate SECOND, behind a hardcoded path that
    exists on this machine, so the variable could never change the answer and the fix was untestable
    here -- a remedy placed behind the thing it was meant to replace. An explicitly set
    GRAPH_API_RUNS_DIR must now win; unset, the old order must stand unchanged."""
    prev_out = os.environ.pop("GRAPH_API_OUTPUT_DIR", None)
    prev_runs = os.environ.get("GRAPH_API_RUNS_DIR")
    try:
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "latest").mkdir()
            os.environ["GRAPH_API_RUNS_DIR"] = td
            mod = _load_server()
            assert str(mod.OUTPUT_DIR) == str(Path(td) / "latest"), str(mod.OUTPUT_DIR)
        os.environ.pop("GRAPH_API_RUNS_DIR", None)
        mod = _load_server()
        # With the variable unset AND no candidate on disk -- a fresh checkout of the repository
        # that ships this dashboard -- OUTPUT_DIR is the neutral no-run directory. It must NOT be
        # this file's own directory: that is the FD-31 defect, where OUTPUT_DIR silently became the
        # dashboard SOURCE tree and the crops mkdir wrote into it. Asserted as "not the source",
        # which is the property, rather than "not /tmp", which was a proxy for it that stopped
        # being true the day the neutral fallback was added.
        assert Path(mod.OUTPUT_DIR) != Path(mod.HERE), \
            f"OUTPUT_DIR fell back to the dashboard's own source directory: {mod.OUTPUT_DIR}"
        assert Path(mod.HERE) not in Path(mod.OUTPUT_DIR).parents, str(mod.OUTPUT_DIR)
    finally:
        if prev_runs is None:
            os.environ.pop("GRAPH_API_RUNS_DIR", None)
        else:
            os.environ["GRAPH_API_RUNS_DIR"] = prev_runs
        if prev_out is not None:
            os.environ["GRAPH_API_OUTPUT_DIR"] = prev_out


def test_transport_bar_is_served_in_both_modes_and_pollers_blocked_only_in_replay():
    """GA-345: live and replay were two pages. One bar now rides both: the live page gets the
    bar with a LIVE state and keeps its pollers; the replay page gets the bar plus the poller
    blocker. The viewer's five pollers that would overwrite a scrubbed frame (logs, heartbeat, BEV +
    metrics cells, graph, health) stand back on window.REPLAY_DETACHED as their FIRST statement --
    asserted per function, because a bare count passed with two of three guards deleted -- and a
    frame younger than 3 s is never cached (it may be mid-write)."""
    import time as _time

    from fastapi.testclient import TestClient
    rs = _replay_server()
    with tempfile.TemporaryDirectory() as td:
        b = Path(td) / "20260101_000000_test"
        (b / "frames").mkdir(parents=True)
        (b / "persistent_perception.json").write_text("[]")
        (b / "frames" / "1700000000_000000000.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        # replay_view reads GRAPH_API_RUNS_DIR at import; it is imported lazily on the first /replay/*
        # request, so the env must be set and any earlier import dropped before that request.
        prev_runs = os.environ.get("GRAPH_API_RUNS_DIR")
        os.environ["GRAPH_API_RUNS_DIR"] = td
        # both spellings: loose module and package module, whichever an extension imported first
        _rv = ("replay_view", f"{__package__}.replay_view" if __package__ else "replay_view")
        parked = {k: sys.modules.pop(k) for k in _rv if k in sys.modules}
        before = set(sys.modules)
        try:
            rs.RUNS_ROOT = Path(td)
            rs.MODE.update(mode="replay", why="test")
            m = rs.build_app(b)
            c = TestClient(m.app)
            replay = c.get("/dash").text
            rs.MODE.update(mode="live", why="test")
            live = c.get("/dash").text
            for page, mode in ((replay, "replay"), (live, "live")):
                assert 'id="rLive"' in page and "dash-scrub" in page, f"no transport bar on the {mode} page"
                for fn in ("updateLogs", "updateBEV", "refreshGraph", "updateHealthStatus"):
                    assert re.search(r"function %s\(\) \{\s*\n\s*if \(window\.REPLAY_DETACHED\) return;" % fn, page), \
                        f"{mode}: viewer poller {fn} lacks the detach guard as its first statement"
                assert re.search(r"function checkFeedHeartbeat\(\) \{[^}]*?if \(window\.REPLAY_DETACHED\) \{ heartbeatPaused = true; return; \}", page, re.S), \
                    f"{mode}: the heartbeat must pause on detach and reset its clock on re-attach"
            assert "__replayBlocked" in replay and "__replayBlocked" not in live, "the poller blocker is replay-only"
            assert "DASH_MODE='live'" in live and "DASH_MODE='replay'" in replay
            # Owner 2026-09-08: the private infra link is printed unless DASH_PUBLIC is set.
            # 2026-09-10: the host is no longer written in the source — it comes from
            # DASH_INFRA_URL — so all THREE states are asserted. The old check only covered
            # "printed" and "not printed for a public copy", which a hardcoded host satisfies
            # just as well as a configured one; unset-means-no-link is the state that proves
            # the host left the file.
            assert "INFRA" not in live, "no infra link when DASH_INFRA_URL is unset"
            os.environ["DASH_INFRA_URL"] = "https://infra.example.invalid:7443/"
            try:
                configured = c.get("/dash").text
            finally:
                os.environ.pop("DASH_INFRA_URL", None)
            assert "INFRA" in configured and "infra.example.invalid" in configured, \
                "a configured infra URL must appear in the menu"
            assert "__INTERNAL_LINKS__" not in live
            try:
                from found.dashboard import dash_ext
            except ImportError:
                import dash_ext
            os.environ["DASH_PUBLIC"] = "1"
            try:
                pub = c.get("/dash").text
            finally:
                os.environ.pop("DASH_PUBLIC", None)
            assert "INFRA" not in pub and "__INTERNAL_LINKS__" not in pub
            # owner 2026-09-08 (via ARIA): a PUBLIC deployment's start page offers no launch section;
            # a lab-host dashboard in replay mode (started before a run) keeps it -- that is where it is needed
            rs.MODE.update(mode="replay", why="test")
            assert "Start a new run" in c.get("/").text
            os.environ["DASH_PUBLIC"] = "1"
            try:
                start = c.get("/").text
            finally:
                os.environ.pop("DASH_PUBLIC", None)
            assert "Start a new run" not in start and "Open a recorded run" in start
            # the crop ticker is NAMED (so it is greppable and its guard is assertable) but must NOT
            # be blocked in replay: that is the one mode where it is needed, because the graph
            # version never changes there and a crop that failed once would stay missing
            assert "function refreshCropBackgrounds()" in replay
            assert "'refreshCropBackgrounds'" not in replay, "the crop retry must not be in the replay blocker"
            assert re.search(r"function refreshCropBackgrounds\(\) \{\s*\n\s*if \(window\.REPLAY_DETACHED\) return;", replay), \
                "the crop retry must still stand back while a live page is scrubbed"
            # the mode reason is stamped by every probe, not only on a transition (a sticky "pick"
            # left every live tab ignoring the end of the run)
            follower = re.search(r"def _follow_mode.*?threading\.Thread", (HERE / "replay_server.py").read_text(), re.S).group(0)
            assert follower.count('by="probe"') == 2, "the follower must stamp the reason on every probe"
            # /mode_info says WHO moved the mode, so a live tab can ignore another tab's bundle pick
            assert json.loads(c.get("/mode_info").text).get("by") is not None
            # GA-379: exactly ONE route per retained-perception path. The bridge defines both, and a
            # second registration is never served -- FastAPI answers with the first match, which
            # would be the bridge's stub-node reply instead of the relay to the running node.
            for path in ("/last_perception/meta", "/last_perception.jpg"):
                n = sum(1 for r in m.app.router.routes if getattr(r, "path", None) == path)
                assert n == 1, f"{path} is registered {n} times; the first match wins"
            # GA-379: the two failures must not share one sentence. In REPLAY there is no second
            # feed at all; LIVE with a dead bridge has one and it is broken, and telling a live
            # viewer "no second feed in a replay" is a transport failure wearing a mode's clothes.
            rs.MODE.update(mode="replay", why="test")
            body = json.loads(c.get("/last_perception/meta").text)
            assert body.get("available") is False and body.get("mode") == "replay" \
                and "replay" in body.get("error", ""), body
            rs.MODE.update(mode="live", why="test")
            body = json.loads(c.get("/last_perception/meta").text)
            assert body.get("mode") == "live" and "did not answer" in body.get("error", "") \
                and "replay" not in body.get("error", ""), body
            # and the page must READ that body rather than throw the reason away on the status
            assert "meta.error ||" in live, "the pane discards the server's reason"
            assert "twoFeedMode = meta.mode" in live, "the switch never follows a mode change"
            # owner 2026-09-08: the 3D scene has no window of its own -- no menu link, and the page
            # the tab's iframe loads carries no nested tools menu of its own
            assert 'href="/scene3d"' not in live and 'href="/scene3d"' not in replay
            assert 'id="scene3dMaxBtn"' in live and 'id="scene3dMaxBtn"' in replay
            scene = c.get("/scene3d").text
            assert 'id="toolsMenu"' not in scene, "the scene page must not nest a second tools menu inside the tab"
            # the framed scene forwards Escape to the parent: key events do not cross the boundary,
            # so without this the overlay could not be closed once the scene had focus
            assert "window.parent.toggleScene3dMax" in scene, "the scene page must forward Escape to the parent"
            # the tab switcher must drop the maximize state, or a programmatic switch pins the
            # overlay and its backdrop over another tab
            assert re.search(r"function switchTab\(tabId\) \{.*?SCENE3D_MAXIMIZED && tabId !== 'scene3d'\) toggleScene3dMax\(false\)",
                             live, re.S), "switchTab must clear the maximized overlay"
            # GA-399: a bundle can be UPLOADED, so every string it carries is untrusted text rendered
            # into an authenticated page. Scanned as a CLASS rather than as the four sites found by
            # hand: any interpolation of a bundle-derived field into markup must go through escHtml.
            # The first version of this check matched ONE assignment shape with a case-sensitive
            # keyword list, so it passed while TWELVE siblings in the same two templates stayed raw,
            # one of them a link address. Checking names is checking the wrong thing: the guarantee
            # is STRUCTURAL. Every panel that renders bundle data must build it with the escaping
            # tag `h`, where substitutions are escaped by DEFAULT and markup is opted in with raw().
            import re as _re
            assert "function h(strings, ...vals)" in live and "function raw(s)" in live, \
                "the escaping tag is gone; substitutions are no longer escaped by default"
            for fn in ("showNodeDetails", "showEdgeDetails"):
                i = live.index("function %s" % fn)
                body = live[i:i + 9000]
                assert ".innerHTML = h`" in body, f"{fn} builds markup without the escaping tag"
            # and no panel may go back to an untagged template that interpolates anything
            for m in _re.finditer(r"\.innerHTML\s*=\s*`", live):
                seg = live[max(0, m.start() - 300):m.start()]
                assert "function renderRoomLegend" in seg or "${" not in live[m.end():m.end() + 400], \
                    "an untagged template with substitutions is building markup again"
            # every raw() must be greppable and deliberate, not a wildcard
            assert live.count("raw(") >= 4, "the markup opt-outs disappeared; check what replaced them"
            # AND the tag must only ever be an OPENING delimiter. Prefixing a CLOSING backtick pushes
            # a literal "h" into the rendered text, which happened 13 times while this was written
            # and shows up as a stray character rather than as an error.
            # Scoped to the two builders and to non-comment lines: the first version scanned the
            # whole page and matched a backtick used as PROSE QUOTING inside a comment, failing on
            # correct code -- a check reading prose as code, the same fault the scene page's own
            # self-check had earlier tonight.
            for fn in ("showNodeDetails", "showEdgeDetails"):
                i = live.index("function %s" % fn)
                seg = "\n".join(ln for ln in live[i:i + 9000].split("\n")
                                 if not ln.strip().startswith("//"))
                for m in _re.finditer(r"h`", seg):
                    lead = seg[max(0, m.start() - 24):m.start()]
                    assert _re.search(r"(\$\{|\(|\?|:|=|&&|\|\||,|^)\s*(raw\()?\s*$", lead), \
                        f"{fn}: an 'h' is prefixed to a CLOSING backtick and renders as a stray letter: ...{lead[-20:]!r}"
            # and no markup substitution may be left unwrapped inside a tagged template
            assert "${raw(" in live, "the nested-markup wrappers are gone"
            assert "${escHtml(roomId)}" in live, "the room cell renders its id unescaped"
            # GA-380: every URL the SCRIPTS build must carry the path prefix, because the proxy that
            # adapts this app to a path prefix can only rewrite literal src="/ and fetch('/. One helper per
            # script, not a copy per site -- three inline copies is how the fourth and fifth sites
            # were missed. The trailing-slash strip matters: at the start page the path is "/" or
            # "/<prefix>/", and without it the result was "//dash", which a browser reads as a HOST.
            for page_src, where in ((replay, "replay"), (live, "live")):
                assert "const PFX = location.pathname" in page_src, f"{where}: no prefix helper"
                assert "replace(new RegExp('/+$'), '')" in page_src, f"{where}: no trailing-slash strip"
                assert "feed.src = PFX + '/replay/frame/" in page_src, f"{where}: frame URL not prefixed"
                assert "= '/replay/frame/'" not in page_src and "src = '/feed" not in page_src, \
                    f"{where}: an unprefixed image URL remains"
                # the viewer page builds URLs at runtime too -- the feed re-source and every crop
                # thumbnail. Measured on the public copy: the feed died and the crops 404ed.
                assert "img.src = PFX + '/feed" in page_src, f"{where}: the feed re-source is unprefixed"
                # every crop URL the page builds must be prefixed: scan each occurrence rather than
                # counting, so one unprefixed site among several cannot hide behind the others
                for m in re.finditer(r"'/crop/'", page_src):
                    lead = page_src[max(0, m.start() - 60):m.start()]
                    assert "PFX + (" in lead, f"{where}: a crop URL is built without the prefix"
            # GA-398: a bundle that cannot be replayed says so ON THE ROW. "No frames" and "empty"
            # are DIFFERENT tags on purpose: the 26 August archives have no frames because that era
            # wrote none, and one of them is the only citable run in the project.
            assert "_bundle_tag" in (HERE / "replay_server.py").read_text()
            # the fixture HAS one frame, so it must not be tagged empty; a name with nothing behind
            # it must be. Both directions, so the tag cannot be a constant.
            tag_frames = rs._bundle_tag("20260101_000000_test")
            assert tag_frames[0].startswith("1 frames"), tag_frames
            assert "no detections.jsonl" in tag_frames[1], tag_frames
            (Path(td) / "20260101_000001_bare").mkdir()
            tag_empty = rs._bundle_tag("20260101_000001_bare")
            assert tag_empty[0].startswith("EMPTY"), tag_empty
            assert "aborted launch or a mapping-only run" in tag_empty[1], tag_empty
            start_page = c.get("/").text
            assert "location.href = location.pathname.replace" in start_page, \
                "the start page still navigates to an absolute /dash"
            # GA-380: run the SAME assertion over every page this server serves. Three sites survived
            # two sweeps by living in files the check never re-entered: the start page's resume link,
            # the replay page's back link, and its log window. A per-page check is why they hid.
            for path, name in (("/", "start page"), ("/dash", "dashboard"), ("/replay", "replay page")):
                page = c.get(path).text
                assert 'href="/' not in page, f"{name}: an absolute link navigates to the site root"
                assert "= '/replay/" not in page, f"{name}: an unprefixed /replay/ URL is built here"
            # owner 2026-09-08 (via ARIA): a public copy's menu drops the links that need a live stack
            os.environ["DASH_PUBLIC"] = "1"
            try:
                pubmenu = c.get("/dash").text
            finally:
                os.environ.pop("DASH_PUBLIC", None)
            # the filter's strings and the template's hrefs must move together: a stale filter entry
            # silently re-exposes a link the owner ordered removed, and nothing errors
            for dead in ('href="./"', 'href="dash"', 'href="replay"'):
                assert dead not in pubmenu, f"public menu still offers {dead}"
            assert 'href="bundles"' in pubmenu, "a public copy keeps the links that work offline"
            # An EXTENSION's pages used to be named here, so the menu could offer a page the app had
            # not installed. Both are built from the same Page objects now, and this asserts the seam
            # in BOTH directions: absent without an extension, present with one. A check that only
            # looked for the links would pass on a dashboard that hard-coded them again.
            assert 'href="blockers"' not in pubmenu and 'href="arch"' not in pubmenu, \
                "an extension page is in the menu with no extension registered"
            _saved_ext = dash_ext.extension()
            try:
                dash_ext.register(dash_ext.Extension(pages=[
                    dash_ext.Page("widgets", lambda b: "<p>w</p>", menu_label="WIDGETS"),
                    dash_ext.Page("quiet", lambda b: "<p>q</p>")]))
                withext = c.get("/dash").text
                assert 'href="widgets"' in withext, "a registered page is missing from the menu"
                assert "WIDGETS" in withext, "the menu label is not the one the extension gave"
                assert 'href="quiet"' not in withext, \
                    "a page with no menu_label must exist without being advertised"
            finally:
                dash_ext.register(_saved_ext)
            assert 'href="dash"' in live, "a lab instance keeps every link"
            # and every menu href is relative, or the app breaks under a path prefix
            assert 'href="/' not in live, "an absolute menu href is back; it will navigate to the site root"
            assert "location.href = 'dash'" in live, "the post-pick navigation is absolute again"
            rs.MODE.update(mode="replay", why="test")
            url = "/replay/frame/20260101_000000_test/1700000000_000000000.jpg"
            r = c.get(url)
            assert r.status_code == 200, (r.status_code, r.text[:200])
            assert r.headers["cache-control"] == "no-store"
            old = _time.time() - 60
            os.utime(b / "frames" / "1700000000_000000000.jpg", (old, old))
            assert c.get(url).headers["cache-control"].startswith("public")
        finally:
            os.environ.pop("GRAPH_API_OUTPUT_DIR", None)
            if prev_runs is None:
                os.environ.pop("GRAPH_API_RUNS_DIR", None)
            else:
                os.environ["GRAPH_API_RUNS_DIR"] = prev_runs
            _purge_modules(before)
            sys.modules.update(parked)


def test_the_bundle_tag_says_which_machine_recorded_it():
    """A picker can hold bundles from two machines only if each row says which one made it.

    Until 2026-09-10 no bundle recorded the machine at all, so a reader comparing a local run with
    a remote one could not have noticed they were different systems (found by the experiment lane
    syncing a second machine's runs). Three cases, and the third is the one that matters: a bundle
    with no `machine` key must SAY it is not recorded, never be assumed local -- a bundle that does
    not say is not a bundle from here.
    """
    import json as _json
    import socket as _sock
    rs = _replay_server()
    prev = os.environ.get("GRAPH_API_RUNS_DIR")
    with tempfile.TemporaryDirectory() as td:
        os.environ["GRAPH_API_RUNS_DIR"] = td
        before = set(sys.modules)
        try:
            rs.RUNS_ROOT = Path(td)
            here = _sock.gethostname()
            for name, machine in (("20260101_000000_local", here),
                                  ("20260101_000001_remote", "somewhere-else"),
                                  ("20260101_000002_old", None)):
                d = Path(td) / name
                d.mkdir()
                meta = {"run_id": name}
                if machine:
                    meta["machine"] = machine
                (d / "run_metadata.json").write_text(_json.dumps(meta))
            rs._load_bundle_index().RUNS_DIR = Path(td)
            local_s, local_l = rs._bundle_tag("20260101_000000_local")
            remote_s, remote_l = rs._bundle_tag("20260101_000001_remote")
            old_s, old_l = rs._bundle_tag("20260101_000002_old")
            # the OTHER machine is named in the row itself, not only in a title nobody hovers
            assert remote_s.startswith("[somewhere-else] "), remote_s
            assert "not this machine" in remote_l, remote_l
            # this machine is not shouted at the reader in every row, but the title still says it
            assert not local_s.startswith("["), local_s
            assert here in local_l, local_l
            # and an OLD bundle says it does not know, rather than reading as local
            assert not old_s.startswith("["), old_s
            assert "not recorded" in old_l, old_l
            assert here not in old_l, "a bundle with no machine key must not read as this machine"
        finally:
            if prev is None:
                os.environ.pop("GRAPH_API_RUNS_DIR", None)
            else:
                os.environ["GRAPH_API_RUNS_DIR"] = prev
            _purge_modules(before)



if __name__ == "__main__":
    # REBUILT 2026-09-10 after a bad slice removed it. A suite whose runner is gone still EXITS 0
    # and prints nothing, which is the most dangerous green there is -- so the names are derived
    # from the file below rather than retyped, and the count is asserted against what ran.
    fetched = test_every_endpoint_the_viewer_fetches_is_defined()
    test_the_tools_menu_never_links_to_a_route_that_is_not_served()
    test_no_duplicate_routes()
    test_crops_dir_never_created_inside_output_dir()
    test_output_dir_never_falls_back_to_the_read_only_remedy_source()
    test_no_health_component_reports_itself_active_unconditionally()
    test_bridge_objects_carry_the_admission_grade_and_the_log_is_read_incrementally()
    test_feed_host_grade_toggles_filter_and_count()
    test_replay_reads_no_live_service()
    test_discovered_bridge_port_survives_an_empty_runs_dir()
    test_explicit_mode_flag_pins_the_mode()
    test_live_app_serves_what_the_tools_menu_links()
    test_replay_verdicts_are_keyed_the_way_the_page_reads_them()
    test_runs_root_follows_the_environment_like_its_siblings()
    test_the_legacy_readers_environment_entry_can_actually_fire()
    test_transport_bar_is_served_in_both_modes_and_pollers_blocked_only_in_replay()
    test_the_bundle_tag_says_which_machine_recorded_it()
    _ran = 17
    print(f"all {_ran} checks passed (viewer fetches {len(fetched)} endpoints: "
          f"{', '.join(fetched)})")
