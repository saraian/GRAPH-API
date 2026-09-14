#!/usr/bin/env python3
"""Rewrite the run configurations in schedules/configs/ from the tracked config.

    python3 lost3dsg/test/regen_arm_configs.py            # rewrite them
    python3 lost3dsg/test/regen_arm_configs.py --check     # say which are stale, change nothing
    python3 lost3dsg/test/regen_arm_configs.py --self-check

WHY THIS EXISTS, and it is a measured fault rather than tidiness.

Each file in schedules/configs/ is a COMPLETE snapshot of the tracked config plus the two or three
keys that make it an arm. It has to be complete: `GRAPH_API_CONFIG` REPLACES config.yaml rather
than layering on it (config.py), so a file holding only the differences would fall back to the
module defaults for everything else, not to the tracked config.

A complete snapshot goes stale the moment a key is ADDED to the tracked config, and it goes stale
SILENTLY. The snapshot simply does not mention the new key, so the code takes its module fallback
and the run looks like a fair test of whatever the key controls.

MEASURED 2026-09-11. `perception.frame_queue_max` was added to the tracked config at 8, to let the
detection cycle work on queued frames while the robot moves. The snapshots were generated at
09:32:49 and predate it. Its module default is 0, which means OFF. So a run of arm 06 or arm 07
would have read no frame_queue_max, run with the queue disabled, and produced a bundle that reads
exactly like a test of the queue and contains none. The same shape had already bitten the
post-scan merge: `association.scan_merge_settle_s` was absent from every snapshot too, and there
it was harmless only because the module fallback happened to equal the intended value.

SO THE SNAPSHOT RECORDS WHEN IT WAS GENERATED AND WHAT FROM. `generated_from` carries the tracked
config's sha256, and `--check` compares it against the tracked config as it stands. A reader who
finds a key missing from a bundle's config can then tell "this arm predates the key" from "this
arm sets it differently", which is the distinction that was unavailable this morning.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
TRACKED = REPO / "lost3dsg/src/perception_module/config.yaml"
OUT_DIR = REPO / "schedules/configs"

# THE ARMS. `diff` is what makes each one an arm; everything else comes from the tracked config.
# `note` is the paragraph a reader of a bundle needs in order to know what the arm was FOR -- it
# is not decoration, and an arm whose purpose is not written down cannot be reported.
ARMS = [
    ("09_legacy_merge", {"association.merge_engine": "legacy"}, [
        "THE MERGE ABLATION BASELINE (merge-algorithm lane, 2026-09-14). Identical to 01_reference",
        "except that the LEGACY merge engine decides: the pre-ruling gates (room on any two rooms,",
        "0.8 m centre distance, similarity >= 0.925 on the weighted attribute score) instead of the",
        "evidence engine with the ontology channel off and the bounded attribute channel. Run it",
        "ONLY through this file: the MERGE_ENGINE environment override also switches the engine",
        "but run_metadata's effective_config stamps association.merge_engine from the config, so",
        "an env-switched run would carry a stamp that names the other engine (rules 2 and 5).",
        "Compare against the reference on the SAME tour: merges, adds, updates, final objects,",
        "cross-kind merges, and the evaluator's precision at IoU 0.25 / 0.50.",
    ]),
    ("01_reference", {}, [
        "THE REFERENCE RUN. Everything else is compared against this one, and it is the tracked",
        "configuration with nothing changed: local detector, rtabmap pose (owner ruling",
        "2026-09-10), the scheduled exploration, the camera tilted 30 degrees down, no cap.",
    ]),
    ("02_no_mapping_phase", {}, [
        "RETIRED 2026-09-11, AND IT MUST NOT BE RUN AS AN ARM. It was the control for the",
        "reference: identical except `mapping_seconds: 0`. The mapping phase was removed with the",
        "sampling policy on the owner's instruction, so the one setting that made this arm",
        "different no longer exists and this file is now identical to 01_reference. Running both",
        "would report a difference of zero as a finding. Kept rather than deleted because bundles",
        "already name it and a reader needs to find out what it meant.",
    ]),
    ("03_size_gate", {"hooks.filter": "envelope_size:SizeFilter"}, [
        "THE SIZE ENVELOPES, ANNOTATING ONLY. The reference plus the envelope filter. It abstains",
        "rather than admits where the corpus has never measured a kind, so it annotates without",
        "refusing anything. Compare its hook_decisions against 01 to see what the envelopes would",
        "have said.",
    ]),
    ("04_ground_truth_pose", {"habitat.localization_mode": "ground_truth"}, [
        "GROUND-TRUTH POSE. Separates localisation error from perception error: every box position",
        "in the archive rests on a pose the real robot will never have, and this arm is the upper",
        "bound the rtabmap arm is measured against. NOT a base run -- the owner's regime is",
        "rtabmap.",
    ]),
    ("05_noise_floor", {}, [
        "THE NOISE FLOOR, and it needs `repeat: 3` in the schedule. Identical to the reference on",
        "purpose: no two archived bundles are tier-one identical AND carry the covariate that",
        "normalises for run length, so the spread between runs of ONE configuration cannot be",
        "recovered by analysis. Until it is measured, no figure from any run supports a regression",
        "claim -- and one map per storey makes the band strictly wider than before.",
    ]),
    ("06_vlm_online", {"vlm.base_url": "https://api.regolo.ai/v1", "vlm.model": "gemma4-31b"}, [
        "THE LABELLING VLM IS THE ONLINE MODEL. Owner 2026-09-10: gemma4-31b on regolo.",
        "Everything else is the tracked default.",
        "THE KEY IS NOT HERE. It lives in lost3dsg/test/env.local.sh as REGOLO_API_KEY, gitignored,",
        "because a key in a tracked config is GA-319 and was committed once already.",
    ]),
    ("08_no_filter", {"vlm.base_url": "https://api.regolo.ai/v1", "vlm.model": "gemma4-31b",
                      "hooks.filter": "",
                      "habitat.revisit_scan_deg": 180.0, "habitat.revisit_offset_m": 1.0}, [
        "NO ADMISSION FILTER. Owner 2026-09-11. Identical to 06_vlm_online except that nothing",
        "judges a proposal: hooks.Filter's pass-through admits everything, so every new-object",
        "proposal becomes an object.",
        "WHAT IT ISOLATES. On 20260911_160215 the envelope filter admitted 27 of 43 proposals and",
        "ABSTAINED on 16 -- 'no envelope for this kind: it has never been measured'. An abstain is",
        "admissible by this seam's definition, so those 16 entered anyway; what the filter changed",
        "was the annotation, not the map. This arm is the control that shows whether that holds.",
        "The admission rows are still written, so the two arms are compared on the same log.",
        "THE MULTI-STOP TOUR IS ARMED HERE. habitat.revisit_scan_deg 180 and revisit_offset_m 1.0",
        "(simulator lane, 71620ec and 0416f55): the agent turns 180 degrees again every time the",
        "route re-enters a waypoint, standing a metre to the side. Their module defaults are 0,",
        "which is the single-stop tour, so an arm that omits them does NOT test the tour.",
    ]),
    ("07_vlm_offline", {"vlm.base_url": "http://localhost:11434/v1", "vlm.model": "gemma3:27b"}, [
        "THE SAME RUN WITH NOTHING LEAVING THE MACHINE. The labelling VLM is served by ollama on",
        "the host; the detector and segmenter were already local, so this arm makes the run fully",
        "offline. gemma3:27b rather than gemma4-31b because the 31b is not in ollama's registry --",
        "the pair is therefore NOT a clean online/offline comparison of one model, and any",
        "difference between the two arms includes the difference between two different models.",
        "Said here so no report reads it as a hosting effect.",
    ]),
]


def _set_dotted(tree: dict, dotted: str, value):
    parts = dotted.split(".")
    node = tree
    for p in parts[:-1]:
        nxt = node.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            node[p] = nxt
        node = nxt
    node[parts[-1]] = value


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _mask_stamp(text: str) -> str:
    """Drop the generated_at line. Every other byte of a snapshot is reproducible from the
    tracked config and the ARMS table, so a remaining difference is a hand edit."""
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith("# generated_at:"))


def header(name: str, diff: dict, note: list[str], base_sha: str, stamp: str) -> str:
    lines = [f"# {name}.yaml — a complete run configuration. Use it directly:",
             f"#     ./run_sim_headless.sh --config schedules/configs/{name}.yaml",
             "# or name it from a schedule arm with `config_file:`.",
             "#"]
    lines += [f"# {ln}" for ln in note]
    lines += ["#",
              "# WHAT DIFFERS FROM THE TRACKED config.yaml:"]
    if diff:
        lines += [f"#   {k}: {v!r}" for k, v in sorted(diff.items())]
    else:
        lines += ["#   nothing. This arm IS the tracked configuration."]
    lines += ["#",
              "# GENERATED, DO NOT HAND-EDIT. Rewrite it with:",
              "#     python3 lost3dsg/test/regen_arm_configs.py",
              f"# generated_at:   {stamp}",
              f"# generated_from: lost3dsg/src/perception_module/config.yaml sha256 {base_sha}",
              "#",
              "# A SNAPSHOT CANNOT MENTION A KEY THAT DID NOT EXIST WHEN IT WAS WRITTEN, and the",
              "# code then takes its module fallback instead. If a bundle's config is missing a key",
              "# you expected, compare the sha above against the tracked config before concluding",
              "# the arm set it deliberately: `regen_arm_configs.py --check` answers that.",
              ""]
    return "\n".join(lines) + "\n"


def build(base_text: str, stamp: str):
    """-> {name: file text}. Pure, so the self-check can exercise it without touching disk."""
    import yaml
    base = yaml.safe_load(base_text) or {}
    base_sha = _sha16(base_text)
    out = {}
    for name, diff, note in ARMS:
        tree = copy.deepcopy(base)
        for dotted, value in diff.items():
            _set_dotted(tree, dotted, value)
        out[name] = header(name, diff, note, base_sha, stamp) + yaml.safe_dump(tree, sort_keys=False)
    return out, base_sha


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report which snapshots are stale against the tracked config; write nothing")
    args = ap.parse_args()
    if not TRACKED.is_file():
        print(f"!! the tracked config is missing: {TRACKED}", file=sys.stderr)
        return 2
    base_text = TRACKED.read_text()
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    files, base_sha = build(base_text, stamp)

    if args.check:
        stale = []
        for name in files:
            p = OUT_DIR / f"{name}.yaml"
            if not p.is_file():
                stale.append((name, "absent"))
                continue
            on_disk = p.read_text()
            recorded = ""
            for line in on_disk.splitlines():
                if line.startswith("# generated_from:"):
                    recorded = line.rsplit(" ", 1)[-1]
                    break
            if recorded != base_sha:
                stale.append((name, f"generated from {recorded or 'an unrecorded config'}"))
            elif _mask_stamp(on_disk) != _mask_stamp(files[name]):
                # The sha matches, so the snapshot is not stale -- the BODY has diverged, which is
                # a hand edit. A regen would silently delete it, so name it here instead.
                stale.append((name, "hand-edited: a regen would overwrite it. Move the change "
                                    "into the ARMS table, or into the tracked config"))
        if stale:
            print(f"{len(stale)} of {len(files)} snapshots are stale "
                  f"(tracked config sha256 {base_sha}):")
            for name, why in stale:
                print(f"  {name}.yaml — {why}")
            return 1
        print(f"all {len(files)} snapshots match the tracked config (sha256 {base_sha})")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (OUT_DIR / f"{name}.yaml").write_text(text)
    print(f"rewrote {len(files)} run configurations in {OUT_DIR}")
    print(f"  from {TRACKED} (sha256 {base_sha})")
    print(f"  at   {stamp}")
    return 0


def _selfcheck():
    base = ("perception:\n  backend: local\n  frame_queue_max: 8\n"
            "vlm:\n  base_url: http://old\n  model: old-model\n"
            "habitat:\n  localization_mode: rtabmap\n")
    files, sha = build(base, "2026-01-01T00:00:00")
    import yaml
    # THE ARM'S OWN KEYS WIN, and nothing else moves.
    six = yaml.safe_load(files["06_vlm_online"])
    assert six["vlm"]["base_url"] == "https://api.regolo.ai/v1", six["vlm"]
    assert six["vlm"]["model"] == "gemma4-31b", six["vlm"]
    assert six["habitat"]["localization_mode"] == "rtabmap", six["habitat"]
    # THE WHOLE REASON THIS SCRIPT EXISTS: a key added to the tracked config reaches every arm.
    # Exercised on frame_queue_max itself, because its module default is 0 and 0 means OFF, so an
    # arm that omits it runs with the queue disabled and looks like a test of the queue.
    for name in files:
        assert yaml.safe_load(files[name])["perception"]["frame_queue_max"] == 8, name
    # An arm with no difference is the tracked config, and says so rather than leaving it blank.
    assert "#   nothing. This arm IS the tracked configuration." in files["01_reference"]
    # The recorded sha is the BASE's, so --check can tell staleness from a new stamp.
    assert f"sha256 {sha}" in files["01_reference"]
    assert sha == _sha16(base)
    # A changed base changes the sha, which is what makes --check work at all.
    _, sha2 = build(base + "extra: 1\n", "2026-01-01T00:00:00")
    assert sha2 != sha
    # ARM 08 CARRIES THE MULTI-STOP TOUR. Exercised because the two keys reached the file by hand
    # on 2026-09-12 and a regen that did not know them would have deleted both, turning the tour
    # off in the one arm whose purpose is to run it.
    eight = yaml.safe_load(files["08_no_filter"])
    assert eight["habitat"]["revisit_scan_deg"] == 180.0, eight["habitat"]
    assert eight["habitat"]["revisit_offset_m"] == 1.0, eight["habitat"]
    # THE HAND-EDIT DETECTOR. A new stamp is not a difference; an added key is.
    a = files["01_reference"]
    assert _mask_stamp(a) == _mask_stamp(a.replace("2026-01-01T00:00:00", "2027-02-02T00:00:00"))
    assert _mask_stamp(a) != _mask_stamp(a + "hand_added: 1\n")
    # dotted assignment does not destroy siblings
    t = {"a": {"b": 1}}
    _set_dotted(t, "a.c", 2)
    assert t == {"a": {"b": 1, "c": 2}}, t
    print("regen_arm_configs self-check: PASSED")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _selfcheck()
        raise SystemExit(0)
    raise SystemExit(main())
