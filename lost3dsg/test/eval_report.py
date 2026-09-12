#!/usr/bin/env python3
"""A PDF report of a run's statistics, or a comparison of several runs.

    python3 eval_report.py <bundle> [<bundle> ...] --output Comparison_<stamp>.pdf

Reads each bundle's `eval/metrics.json` (written by metrics_eval.py) and its
`run_metadata.json`, and lays out every table side by side, one column per run.

WHY A REPORT AND NOT A DUMP. metrics_eval.py already writes the numbers as JSON. What a reader
cannot get from that file is which run each column belongs to, what differed between them, and
WHICH NUMBERS ARE NOT MEASUREMENTS. This project has spent days on figures that turned out to
describe something else, so the report's job is attribution:

  - **A missing input is printed as "not measured", never as 0.** metrics_eval declares its own
    `missing_inputs`, and Tables III, IV's classification and V depend on inputs the pipeline does
    not yet produce. A 0.0% next to a real 3.1% invites a comparison that has no meaning.
  - **Each column names its config file and its sha**, because two runs of "the same" thing have
    differed on a local override all week.
  - **Each column says how the run ended.** A bundle from a run that died on a viewer is not
    comparable with one that completed its tour, and `terminating_node.ended` is the only field
    that says so.
  - **The construction time is printed in hours** when it exceeds one, because 8605.2 seconds is a
    number people read as milliseconds.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

# WHAT EACH TABLE SHOWS, in the reader's terms. Owner 2026-09-11: "add a small explanation to each
# section (what do the metrics show)". Written for someone opening the PDF without the code beside
# them, and each one says what a BAD number would mean -- a description that only defines the
# number leaves the reader no better off than the label already did.
TABLE_NOTES = {
    "table_ii_floor_regions":
        "Did the run find the right storeys, and did it carve them into the right rooms? Floor "
        "accuracy compares the storeys the run mapped against the storeys the scene has. Region "
        "precision and recall compare the ROOMS the run segmented against the ground truth's "
        "regions: low recall means the run merged several real rooms into one, low precision "
        "means it split one room into several.",
    "table_iii_rooms":
        "Given a room the run did find, did it call it the right KIND -- kitchen, corridor, "
        "bedroom? Exact accuracy demands the same label; approximate accuracy accepts a near "
        "neighbour. This says nothing about whether the room's shape was right, which is Table II.",
    "table_iv_objects":
        "Did the run put objects in the right places? Precision is the share of objects it "
        "reported that are really there; recall is the share of real objects it found. Mean IoU "
        "is how well a matched box overlaps the true one -- above 0.5 is the usual bar for "
        "'the same object'. The top-k rows ask whether the right label was in the k most likely, "
        "and they can only be computed for matches that were CLASSIFIED.",
    "table_v_retrieval":
        "Could the finished graph answer a question and could the robot act on the answer? "
        "Retrieval success @10 is how often the right object was in the ten returned; navigation "
        "success is how often driving to it arrived. These need a trial set; with none, the run "
        "produced a map nobody asked anything of.",
    "table_vi_room_objects":
        "The same objects as Table IV, but scored INSIDE each room -- it asks whether the right "
        "things ended up in the right room. A run can score well in Table IV and nothing here: "
        "that happens when the objects are correctly placed in space but the room segmentation "
        "does not line up with the ground truth's regions, so every object falls outside the "
        "room it belongs to. Read this table together with Table II's region precision.",
    "table_vii_representation":
        "How much the map costs to keep. The point of comparison is between arms, not against a "
        "target: a representation that is much larger for the same recall is paying for nothing.",
}

# The tables, in the order the paper reads them. `(key, title, [(field, label, unit)])`.
TABLES = [
    ("table_ii_floor_regions", "Table II — floors and regions", [
        ("acc_f_pct", "floor accuracy", "%"),
        ("floor_matches", "floors matched", ""),
        ("floor_gt", "floors in ground truth", ""),
        ("region_precision_pct", "region precision", "%"),
        ("region_recall_pct", "region recall", "%"),
        ("region_matches", "regions matched", ""),
        ("predicted_regions", "regions predicted", ""),
        ("ground_truth_regions", "regions in ground truth", ""),
    ]),
    ("table_iii_rooms", "Table III — room typing", [
        ("acc_exact_pct", "exact accuracy", "%"),
        ("acc_approx_pct", "approximate accuracy", "%"),
        ("rooms_evaluated", "rooms evaluated", ""),
    ]),
    ("table_iv_objects", "Table IV — objects", [
        ("object_precision_pct", "precision", "%"),
        ("object_recall_pct", "recall", "%"),
        ("matched_object_iou_mean", "mean IoU of matches", ""),
        ("matched_objects_iou_gt_0.5", "matches with IoU > 0.5", ""),
        ("classified_matched_objects", "matches that were classified", ""),
        ("top5_pct", "top-5", "%"),
        ("top25_pct", "top-25", "%"),
        ("top100_pct", "top-100", "%"),
    ]),
    ("table_v_retrieval", "Table V — retrieval and navigation", [
        ("trials", "trials", ""),
        ("retrieval_sr_at_10_pct", "retrieval success @10", "%"),
        ("navigation_sr_pct", "navigation success", "%"),
    ]),
    ("table_vi_room_objects", "Table VI — objects per room", [
        ("rooms_evaluated", "rooms evaluated", ""),
        ("expected_objects", "expected", ""),
        ("predicted_objects", "predicted", ""),
        ("matched_objects", "matched", ""),
        ("missing_objects", "missing", ""),
        ("extra_objects", "extra", ""),
        ("precision_pct", "precision", "%"),
        ("recall_pct", "recall", "%"),
        ("f1_pct", "F1", "%"),
    ]),
    ("table_vii_representation", "Table VII — representation size", [
        ("size_mb_total", "total", "MB"),
    ]),
]


def fmt(v, unit=""):
    """A number, or the words that stop a missing input reading as a zero."""
    if v is None:
        return "not measured"
    if isinstance(v, float):
        s = f"{v:.4g}"
    else:
        s = str(v)
    return f"{s}{unit}" if unit and v is not None else s


def read_bundle(p: pathlib.Path):
    """Everything the report says about one run. Absent files are reported, not guessed."""
    out = {"bundle": p, "name": p.name, "metrics": None, "meta": {}, "problems": []}
    m = p / "eval" / "metrics.json"
    if m.exists():
        try:
            out["metrics"] = json.loads(m.read_text())
        except Exception as e:
            out["problems"].append(f"metrics.json is unreadable: {e}")
    else:
        out["problems"].append("no eval/metrics.json — run ./eval.sh on this bundle first")
    tm = p / "eval" / "time_metrics.json"
    if tm.exists():
        try:
            out["time"] = json.loads(tm.read_text())
        except Exception as e:
            out["problems"].append(f"time_metrics.json is unreadable: {e}")
    else:
        out["time"] = None
        out["problems"].append("no eval/time_metrics.json — time_metrics.py did not run")
    r = p / "run_metadata.json"
    if r.exists():
        try:
            out["meta"] = json.loads(r.read_text())
        except Exception as e:
            out["problems"].append(f"run_metadata.json is unreadable: {e}")
    else:
        out["problems"].append("no run_metadata.json")
    return out


def provenance_rows(runs):
    """The rows that say WHICH run each column is. Without these the numbers are anonymous."""
    def cfg(meta):
        c = (meta.get("config_resolved") or {})
        return c.get("feed_host_path") or meta.get("config_name") or "?"

    def ended(meta):
        t = meta.get("terminating_node") or {}
        return t.get("ended") or t.get("node") or "unrecorded"

    def secs(m):
        v = (m or {}).get("construction_time_s")
        if v is None:
            return "not measured"
        # 8605.2 reads as milliseconds to anyone skimming. Give the hours too.
        return f"{v:.0f} s" + (f" ({v / 3600:.1f} h)" if v >= 3600 else "")

    return [
        ("bundle", [r["name"] for r in runs]),
        ("scene", [str((r["meta"].get("scene") or "?")) for r in runs]),
        ("config", [pathlib.Path(cfg(r["meta"])).name for r in runs]),
        ("config sha", [str(((r["meta"].get("config_resolved") or {}).get("file_sha256_16")) or "?")
                        for r in runs]),
        ("how it ended", [ended(r["meta"]) for r in runs]),
        ("construction time", [secs(r["metrics"]) for r in runs]),
    ]


def build_pdf(runs, out_path: pathlib.Path, title: str):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

    styles = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8, leading=10)
    note = ParagraphStyle("note", parent=small, textColor=colors.HexColor("#8a4b00"))
    # A separate style from `note`: note is amber and means "something is wrong here".
    # An explanation of what a table shows is not a warning, and colouring it the same
    # would make every section look like it had a problem.
    explain = ParagraphStyle("explain", parent=small, textColor=colors.HexColor("#444444"))
    page = landscape(A4) if len(runs) > 2 else A4
    doc = SimpleDocTemplate(str(out_path), pagesize=page,
                            leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm,
                            title=title)
    flow = [Paragraph(title, styles["Title"]),
            Paragraph(f"Generated {datetime.datetime.now().isoformat(timespec='seconds')} from "
                      f"each bundle's <font face='Courier'>eval/metrics.json</font>.", small),
            Spacer(1, 4 * mm)]

    def grid(rows, header):
        t = Table([header] + rows, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bbbbbb")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        return t

    head = [""] + [f"run {i + 1}" for i in range(len(runs))]
    flow += [Paragraph("Which runs these are", styles["Heading2"]),
             grid([[k] + [Paragraph(str(v), small) for v in vals]
                   for k, vals in provenance_rows(runs)], head),
             Spacer(1, 4 * mm)]

    missing_any = False
    for key, ttitle, fields in TABLES:
        rows = []
        for field, label, unit in fields:
            cells = []
            for r in runs:
                tbl = ((r["metrics"] or {}).get(key) or {})
                cells.append(fmt(tbl.get(field), unit))
            # A row nobody measured is still printed: its absence is the finding.
            if all(c == "not measured" for c in cells):
                missing_any = True
            rows.append([label] + cells)
        flow += [Paragraph(ttitle, styles["Heading2"])]
        # The explanation goes BEFORE the numbers: a reader who meets the table first has
        # already formed a reading of it by the time an explanation underneath arrives.
        if TABLE_NOTES.get(key):
            flow += [Paragraph(TABLE_NOTES[key], explain), Spacer(1, 1.5 * mm)]
        flow += [grid(rows, head), Spacer(1, 3 * mm)]

    # WHAT THE PIPELINE DID NOT PRODUCE, in metrics_eval's own words. This is the section that
    # stops a reader treating "not measured" as a poor score.
    flow += [Paragraph("Inputs the pipeline did not produce", styles["Heading2"])]
    any_missing_inputs = False
    for i, r in enumerate(runs, 1):
        mi = (r["metrics"] or {}).get("missing_inputs") or []
        if mi:
            any_missing_inputs = True
            flow.append(Paragraph(f"run {i} — {r['name']}:", small))
            for line in mi:
                flow.append(Paragraph(f"&nbsp;&nbsp;• {line}", note))
    if not any_missing_inputs:
        flow.append(Paragraph("None declared.", small))
    if missing_any:
        flow += [Spacer(1, 2 * mm),
                 Paragraph("A row reading <b>not measured</b> is an input the pipeline never "
                           "produced. It is not a score of zero, and it must not be compared "
                           "with one.", note)]

    # WHERE THE TIME WENT. Printed as its own section because a duration is not a score: the
    # same run can be sound and unusably slow, and the six rows below are the only place the
    # difference between "moving", "online" and "waiting" is visible.
    flow += [Paragraph("Time", styles["Heading2"])]
    trows = []
    for field, label, unit in [
        ("run_time_s", "run time (first to last timestamp)", " s"),
        ("exploration_time_s", "exploration (first to last frame)", " s"),
        ("movement_time_s", "of which moving — no perception fires", " s"),
        ("online_time_s", "inside remote calls", " s"),
        ("first_frame_to_kg_s", "first frame to first admission", " s"),
        ("cycles", "perception cycles", ""),
    ]:
        trows.append([label] + [fmt((r.get("time") or {}).get(field), unit) for r in runs])
    flow += [grid(trows, head), Spacer(1, 2 * mm)]
    # EVERY STAGE, not one cycle number. median / p95 / max, because the first cycle of every
    # archived run used to include a model download and a mean hides that.
    stage_names = []
    for r in runs:
        for st in ((r.get("time") or {}).get("latency_ms") or {}):
            if st not in stage_names:
                stage_names.append(st)
    if stage_names:
        srows = []
        for st in stage_names:
            cells = []
            for r in runs:
                d = ((r.get("time") or {}).get("latency_ms") or {}).get(st)
                cells.append("not measured" if not d
                             else f"{d['median']} / {d['p95']} / {d['max']}  (n={d['n']})")
            srows.append([st.replace("_ms", "")] + cells)
        flow += [Paragraph("Stage latency, median / p95 / max in ms", styles["Heading3"]),
                 grid(srows, head), Spacer(1, 2 * mm)]
    else:
        flow += [Paragraph("No per-stage latency was recorded: perception_latencies.jsonl is "
                           "absent from every bundle here.", note), Spacer(1, 2 * mm)]
    for i, r in enumerate(runs, 1):
        for n in ((r.get("time") or {}).get("notes") or []):
            flow.append(Paragraph(f"&nbsp;&nbsp;• run {i}: {n}", note))

    problems = [(r["name"], p) for r in runs for p in r["problems"]]
    if problems:
        flow += [Spacer(1, 3 * mm), Paragraph("Problems reading these bundles", styles["Heading2"])]
        for n, p in problems:
            flow.append(Paragraph(f"&nbsp;&nbsp;• {n}: {p}", note))

    doc.build(flow)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles", nargs="+", type=pathlib.Path)
    ap.add_argument("--output", type=pathlib.Path, default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    runs = [read_bundle(b.resolve()) for b in args.bundles]
    if not any(r["metrics"] for r in runs):
        print("!! none of these bundles has eval/metrics.json — run ./eval.sh first", file=sys.stderr)
        for r in runs:
            for p in r["problems"]:
                print(f"   {r['name']}: {p}", file=sys.stderr)
        return 2

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or (runs[0]["bundle"] / "eval" / f"Comparison_{stamp}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    title = args.title or (f"Comparison of {len(runs)} runs" if len(runs) > 1
                           else f"Run statistics — {runs[0]['name']}")
    build_pdf(runs, out, title)
    print(f"wrote {out}")
    for r in runs:
        for p in r["problems"]:
            print(f"   note: {r['name']}: {p}")
    return 0


def _selfcheck():
    import tempfile
    # A missing input must print as words, never as a zero -- the whole point of the report.
    assert fmt(None) == "not measured"
    assert fmt(None, "%") == "not measured"
    assert fmt(0.0, "%") == "0%", fmt(0.0, "%")
    assert fmt(3.125, "%") == "3.125%"
    assert fmt(8605.2) == "8605"
    with tempfile.TemporaryDirectory() as d:
        b = pathlib.Path(d) / "20260101_000000_scene"
        (b / "eval").mkdir(parents=True)
        (b / "eval" / "metrics.json").write_text(json.dumps({
            "table_ii_floor_regions": {"acc_f_pct": 100.0, "region_recall_pct": 0.0},
            "table_iii_rooms": {"acc_exact_pct": None, "rooms_evaluated": 0},
            "missing_inputs": ["Table III: ground-truth room labels"],
            "construction_time_s": 8605.2,
        }))
        (b / "eval" / "time_metrics.json").write_text(json.dumps({
            "run_time_s": 8605.2, "exploration_time_s": 8605.2, "movement_time_s": 529.5,
            "online_time_s": None, "first_frame_to_kg_s": 20.9, "cycles": 12,
            "latency_ms": {"total_ms": {"median": 2000.0, "p95": 3000.0, "max": 3100.0, "n": 12}},
            "notes": ["online_time_s is absent because no cycle recorded a remote stage"]}))
        (b / "run_metadata.json").write_text(json.dumps({
            "scene": "hm3d_00861",
            "config_resolved": {"feed_host_path": "/x/config.yaml", "file_sha256_16": "abc123"},
            "terminating_node": {"ended": "tour_complete"}}))
        r = read_bundle(b)
        assert r["problems"] == [], r["problems"]
        rows = dict((k, v) for k, v in provenance_rows([r]))
        assert rows["config"] == ["config.yaml"], rows
        assert rows["how it ended"] == ["tour_complete"], rows
        # the hours are added because 8605 seconds reads as milliseconds
        assert "2.4 h" in rows["construction time"][0], rows["construction time"]
        out = pathlib.Path(d) / "Comparison_test.pdf"
        build_pdf([r], out, "self-check")
        assert out.exists() and out.stat().st_size > 1200, out.stat().st_size
        # a bundle with no metrics is REPORTED, not skipped silently
        b2 = pathlib.Path(d) / "20260101_000001_scene"
        b2.mkdir()
        r2 = read_bundle(b2)
        assert any("no eval/metrics.json" in p for p in r2["problems"]), r2["problems"]
    print("eval_report self-check: PASSED")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _selfcheck()
        raise SystemExit(0)
    raise SystemExit(main())
