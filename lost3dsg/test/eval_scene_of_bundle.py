#!/usr/bin/env python3
"""Which scene file did this run use? Prints `<scene.glb> <dataset_config.json> <source>`.

Called by eval.sh. Separate from it because a scene resolved wrongly produces an evaluation
against a DIFFERENT HOUSE, which is a result nobody can spot afterwards, so this deserves a file
with a self-check rather than a heredoc inside a shell script.

TWO SOURCES, IN ORDER.

1. The bundle's own resolved `config.yaml`, when it names `habitat.scene` and
   `habitat.scene_dataset` outright. The colleague's configuration does.

2. The scene NAME plus this machine's library. **A bundle records a NAME, not a file**:
   `run_metadata.json` says `scene: hm3d_00861`, and `live_run.sh:469-476` turns that into a path
   using `HM3D_ROOT`. So an evaluation is NOT reproducible from a bundle alone -- the same name
   resolves to a different file on a different machine, and nothing in the bundle would show it.
   Filed as GA-464: the launcher should record the resolved paths. Until it does, this repeats the
   launcher's table and REPORTS which source answered, so a reader can see that a name was
   resolved rather than read.

Exit 0 with the two paths and the source; exit 1 with `MISSING MISSING <what-was-found>` when
neither source answers.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

# THE LAUNCHER'S OWN TABLE, live_run.sh:469-476. Duplicated on purpose and marked as such: the
# alternative is parsing a bash case statement, and a wrong parse is silent. If the launcher gains
# a scene, this needs it too -- which is the argument for GA-464 rather than for a cleverer parser.
_HM3D_CFG = "hm3d_annotated_basis.scene_dataset_config.json"
SCENES = {
    "hm3d_00861": ("HM3D_ROOT", "00861-GLAQ4DNUx5U/GLAQ4DNUx5U.basis.glb", _HM3D_CFG),
    "hm3d_00337": ("HM3D_ROOT", "00337-CFVBbU9Rsyb/CFVBbU9Rsyb.basis.glb", _HM3D_CFG),
    "hm3d_00770": ("HM3D_ROOT", "00770-NBg5UqG3di3/NBg5UqG3di3.basis.glb", _HM3D_CFG),
    "mp3d_17DRP": ("MP3D_ROOT", "17DRP5sb8fy/17DRP5sb8fy.glb", "mp3d.scene_dataset_config.json"),
}


def resolve(bundle: pathlib.Path, env=None):
    env = os.environ if env is None else env
    cfg = bundle / "config.yaml"
    if cfg.exists():
        try:
            import yaml
            hab = (yaml.safe_load(cfg.read_text()) or {}).get("habitat") or {}
            if hab.get("scene") and hab.get("scene_dataset"):
                return str(hab["scene"]), str(hab["scene_dataset"]), "the bundle's own config.yaml"
        except Exception:
            pass  # a config we cannot parse is not a reason to stop; fall through to the name.
    meta = bundle / "run_metadata.json"
    name = ""
    if meta.exists():
        try:
            name = (json.loads(meta.read_text()) or {}).get("scene") or ""
        except Exception:
            name = ""
    if name in SCENES:
        var, rel_scene, rel_cfg = SCENES[name]
        root = env.get(var, "")
        if root:
            return f"{root}/{rel_scene}", f"{root}/{rel_cfg}", f"the name {name!r} resolved through {var}"
        return "MISSING", "MISSING", f"the name {name!r}, but {var} is not set"
    if name:
        return "MISSING", "MISSING", f"the name {name!r}, which is not in the launcher's table"
    return "MISSING", "MISSING", "no scene recorded"


def _selfcheck():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        b = pathlib.Path(d)
        # 1. nothing at all -> MISSING, and the reason says so.
        s, c, why = resolve(b, {})
        assert (s, c) == ("MISSING", "MISSING") and "no scene recorded" in why, why
        # 2. a name, with the library set -> the launcher's path.
        (b / "run_metadata.json").write_text(json.dumps({"scene": "hm3d_00861"}))
        s, c, why = resolve(b, {"HM3D_ROOT": "/lib"})
        assert s == "/lib/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.basis.glb", s
        assert c == "/lib/hm3d_annotated_basis.scene_dataset_config.json", c
        assert "resolved through HM3D_ROOT" in why, why
        # 3. THE SAME NAME WITH NO LIBRARY IS A REFUSAL, NOT A RELATIVE PATH. A bare
        #    "/00861-.../x.glb" would be a real-looking path that names nothing.
        s, c, why = resolve(b, {})
        assert (s, c) == ("MISSING", "MISSING") and "HM3D_ROOT is not set" in why, why
        # 4. the config wins over the name, because it is what the run actually loaded.
        (b / "config.yaml").write_text('habitat:\n  scene: /from/cfg.glb\n  scene_dataset: /from/cfg.json\n')
        try:
            import yaml  # noqa: F401
            s, c, why = resolve(b, {"HM3D_ROOT": "/lib"})
            assert s == "/from/cfg.glb" and "config.yaml" in why, (s, why)
        except ImportError:
            print("  (no yaml here; the config-wins case was not exercised)")
        # 5. an unknown name is named in the reason rather than guessed at.
        (b / "config.yaml").unlink()
        (b / "run_metadata.json").write_text(json.dumps({"scene": "hm3d_99999"}))
        s, c, why = resolve(b, {"HM3D_ROOT": "/lib"})
        assert (s, c) == ("MISSING", "MISSING") and "hm3d_99999" in why, why
    print("eval_scene_of_bundle self-check: PASSED")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _selfcheck()
        raise SystemExit(0)
    if len(sys.argv) < 2:
        print("usage: eval_scene_of_bundle.py <bundle-dir> [--self-check]", file=sys.stderr)
        raise SystemExit(2)
    scene, dcfg, source = resolve(pathlib.Path(sys.argv[1]))
    print(scene, dcfg, source)
    raise SystemExit(0 if scene != "MISSING" else 1)
