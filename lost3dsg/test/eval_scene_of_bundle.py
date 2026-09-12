#!/usr/bin/env python3
"""Which scene file did this run use? Prints `<scene.glb> <dataset_config.json> <source>`.

Called by eval.sh. Separate from it because a scene resolved wrongly produces an evaluation
against a DIFFERENT HOUSE, which is a result nobody can spot afterwards, so this deserves a file
with a self-check rather than a heredoc inside a shell script.

TWO SOURCES, IN ORDER.

1. The bundle's own resolved `config.yaml`, when it names `habitat.scene` and
   `habitat.scene_dataset` outright. The colleague's configuration does.

2. The scene NAME plus this machine's library. **A bundle records a NAME, not a file**:
   `run_metadata.json` says `scene: hm3d_00861`, and `run_sim.sh-476` turns that into a path
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

# THE LAUNCHER'S OWN TABLE, run_sim.sh-476. Duplicated on purpose and marked as such: the
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
    meta = bundle / "run_metadata.json"
    name = ""
    if meta.exists():
        try:
            name = (json.loads(meta.read_text()) or {}).get("scene") or ""
        except Exception:
            name = ""

    # THE NAME IS READ FIRST, AND IT POLICES THE PATH. The launcher owns the scene: `run_sim.sh`
    # picks DEF_SCENE from its own argument (:538) and never reads `habitat.scene`, so a config can
    # carry a scene the run did not drive. MEASURED 2026-09-11: schedules/configs/06 and 07 named
    # 00824-Dd4bFSTQ8gi while every run of them drives hm3d_00861. Trusting the config there would
    # score the belief against a DIFFERENT HOUSE and report the result as ground truth.
    cfg = bundle / "config.yaml"
    if cfg.exists():
        try:
            import yaml
            hab = (yaml.safe_load(cfg.read_text()) or {}).get("habitat") or {}
            if hab.get("scene") and hab.get("scene_dataset"):
                path = str(hab["scene"])
                token = SCENES.get(name, (None, "", None))[1].split("/")[0] if name in SCENES else None
                if token and token not in path:
                    return ("MISSING", "MISSING",
                            f"the bundle's config.yaml names {path!r}, which is not the recorded "
                            f"scene {name!r} ({token}). Refusing rather than scoring against the "
                            f"wrong house; remove habitat.scene from the run config.")
                return path, str(hab["scene_dataset"]), "the bundle's own config.yaml"
        except Exception:
            pass  # a config we cannot parse is not a reason to stop; fall through to the name.
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
        # A CONFIG NAMING ANOTHER HOUSE IS REFUSED, not preferred. schedules/configs/06 and 07
        # carried 00824-Dd4bFSTQ8gi while every run of them drives hm3d_00861, and the old order
        # would have scored the belief against that other house and called it ground truth.
        (b / "run_metadata.json").write_text(json.dumps({"scene": "hm3d_00861"}))
        (b / "config.yaml").write_text(
            "habitat:\n  scene: /x/00824-Dd4bFSTQ8gi/Dd4bFSTQ8gi.basis.glb\n"
            "  scene_dataset: /x/cfg.json\n")
        s, c, why = resolve(b, {"HM3D_ROOT": "/lib"})
        assert (s, c) == ("MISSING", "MISSING"), (s, c)
        assert "not the recorded scene" in why, why
        # The SAME house in the config is used as written.
        (b / "config.yaml").write_text(
            "habitat:\n  scene: /x/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.basis.glb\n"
            "  scene_dataset: /x/cfg.json\n")
        s, c, why = resolve(b, {"HM3D_ROOT": "/lib"})
        assert s == "/x/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.basis.glb", s
        (b / "config.yaml").unlink()
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
        # 4. the config wins over the name ONLY WHEN IT NAMES THE SAME HOUSE. It is what the run
        #    loaded, so its exact path is preferred -- but the recorded name polices it (case 6).
        (b / "config.yaml").write_text(
            'habitat:\n  scene: /from/00861-GLAQ4DNUx5U/cfg.glb\n  scene_dataset: /from/cfg.json\n')
        try:
            import yaml  # noqa: F401
            s, c, why = resolve(b, {"HM3D_ROOT": "/lib"})
            assert s == "/from/00861-GLAQ4DNUx5U/cfg.glb" and "config.yaml" in why, (s, why)
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
