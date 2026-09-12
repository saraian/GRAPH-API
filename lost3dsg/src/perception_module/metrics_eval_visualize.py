#!/usr/bin/env python3
"""Visualize predicted and ground-truth 3D bounding boxes from a manifest.

Produces a self-contained HTML page with top (X-Z), front (X-Y), and side
(Z-Y) projections. Blue = ground truth; orange = unmatched prediction;
green = prediction whose 3D IoU exceeds the chosen threshold.

Esempio:
    python3 metrics_eval_visualize.py manifest.json --output boxes.html
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np

from metrics_eval import assignment, filtered_scene, geometry_iou, load, object_assignment, _centre


BLUE = "#2563eb"
ORANGE = "#f97316"
GREEN = "#16a34a"
INK = "#172033"
GRID = "#d7dee9"


def _box(row):
    """Return (min, max) for a valid AABB, otherwise ``None``."""
    try:
        low = np.asarray(row["aabb_min_m"], dtype=float)
        high = np.asarray(row["aabb_max_m"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if low.shape != (3,) or high.shape != (3,) or not np.all(np.isfinite(low + high)):
        return None
    return np.minimum(low, high), np.maximum(low, high)


def _identifier(row, index, gt=False):
    key = next((name for name in ("object_id", "room_id", "region_id", "id")
                if row.get(name) is not None), None)
    label = row.get("category_name") if gt else row.get("label", row.get("predicted_label"))
    identity = row.get(key) if key else index
    return f"{identity}: {label}" if label else str(identity)


def _scene_xz_bounds(scene):
    """Return a shared padded X-Z extent for object and region visualizations."""
    extents = []
    for row in scene.get("predicted_objects", []) + scene.get("ground_truth_objects", []):
        box = _box(row)
        if box is not None:
            extents.append((box[0][[0, 1]], box[1][[0, 1]]))
    for row in scene.get("predicted_regions", []) + scene.get("ground_truth_regions", []):
        polygon = _polygon(row)
        if polygon is not None:
            extents.append((polygon.min(axis=0), polygon.max(axis=0)))
    if not extents:
        return np.zeros(2), np.ones(2)
    low = np.min([pair[0] for pair in extents], axis=0)
    high = np.max([pair[1] for pair in extents], axis=0)
    padding = max(float((high - low).max()) * .06, .15)
    return low - padding, high + padding


def _scene_svg(scene, threshold, xz_bounds, width=1260, panel_w=390, panel_h=420):
    pred_rows = scene.get("predicted_objects", [])
    gt_rows = scene.get("ground_truth_objects", [])
    pred = [(index, row, *_box(row)) for index, row in enumerate(pred_rows) if _box(row) is not None]
    gt = [(index, row, *_box(row)) for index, row in enumerate(gt_rows) if _box(row) is not None]
    # assignment needs the original rows, not the filtered display lists.
    matches = object_assignment(pred_rows, gt_rows, threshold)
    matched_pred = {pi for pi, _, _ in matches}
    matched_gt = {gi for _, gi, _ in matches}
    all_boxes = [(low, high) for _, _, low, high in pred + gt]
    if all_boxes:
        low = np.min([b[0] for b in all_boxes], axis=0)
        high = np.max([b[1] for b in all_boxes], axis=0)
    else:
        low, high = np.zeros(3), np.ones(3)
    extent = np.maximum(high - low, 0.1)
    padding = max(float(extent.max()) * 0.06, 0.15)
    low -= padding
    high += padding
    # Keep the top object view in exactly the same horizontal window as regions.
    low[[0, 1]], high[[0, 1]] = xz_bounds[0], xz_bounds[1]

    def point(value, axes, ox, oy):
        # SVG y grows down: invert the second displayed coordinate.
        x = ox + 45 + (value[axes[0]] - low[axes[0]]) / (high[axes[0]] - low[axes[0]]) * (panel_w - 70)
        y = oy + 25 + (high[axes[1]] - value[axes[1]]) / (high[axes[1]] - low[axes[1]]) * (panel_h - 70)
        return x, y

    panels = (("Top view  X-Y", (0, 1)), ("Front view  X-Z", (0, 2)),
              ("Side view  Y-Z", (1, 2)))
    centre_matches = object_assignment(pred_rows, gt_rows, threshold)
    centre_by_pred = {pi: (gi, score) for pi, gi, score in centre_matches}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="560" viewBox="0 0 {width} 560">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="24" y="32" fill="{INK}" font-family="sans-serif" font-size="20" font-weight="bold">Scene {html.escape(str(scene.get("scene", "unknown")))} — object bounding boxes</text>',
             f'<text x="24" y="55" fill="{INK}" font-family="sans-serif" font-size="13">Valid GT boxes: {len(gt)} · predictions: {len(pred)} · 3D centre matches (&le; {threshold:g} m): {len(centre_matches)} · ROS frame (Z-up)</text>']
    for panel_index, (title, axes) in enumerate(panels):
        ox, oy = 20 + panel_index * (panel_w + 20), 85
        parts += [f'<rect x="{ox}" y="{oy}" width="{panel_w}" height="{panel_h}" rx="6" fill="#fbfcfe" stroke="{GRID}"/>',
                  f'<text x="{ox + 14}" y="{oy + 22}" fill="{INK}" font-family="sans-serif" font-size="14" font-weight="bold">{title}</text>']
        # Light border for the coordinate area, then GT below predictions.
        parts.append(f'<rect x="{ox + 45}" y="{oy + 25}" width="{panel_w - 70}" height="{panel_h - 70}" fill="none" stroke="{GRID}"/>')
        for index, row, bmin, bmax in gt:
            x1, y1 = point(bmin, axes, ox, oy); x2, y2 = point(bmax, axes, ox, oy)
            x, y, w, h = min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)
            label = html.escape("GT " + _identifier(row, index, True) + "; GT covered: " +
                                ("yes" if index in {gi for _, gi, _ in centre_matches} else "no"))
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 1):.1f}" height="{max(h, 1):.1f}" fill="{BLUE}" fill-opacity=".10" stroke="{BLUE}" stroke-width="1.5"><title>{label}</title></rect>')
        for index, row, bmin, bmax in pred:
            x1, y1 = point(bmin, axes, ox, oy); x2, y2 = point(bmax, axes, ox, oy)
            x, y, w, h = min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)
            color = GREEN if index in centre_by_pred else ORANGE
            best_iou = max((geometry_iou(row, candidate) for candidate in gt_rows), default=0.0)
            if index in centre_by_pred:
                gi, _ = centre_by_pred[index]
                distance = float(np.linalg.norm(_centre(row) - _centre(gt_rows[gi])))
                relation = f"; centre distance: {distance:.3f} m; IoU: {geometry_iou(row, gt_rows[gi]):.1f}"
            else:
                relation = "; GT covered: no"
            label = html.escape("Prediction " + _identifier(row, index) +
                                (" (matched)" if index in centre_by_pred else " (unmatched)") +
                                relation + f"; best 3D IoU: {best_iou:.3f}")
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 1):.1f}" height="{max(h, 1):.1f}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="5 3"><title>{label}</title></rect>')
        parts.append(f'<text x="{ox + panel_w / 2:.1f}" y="{oy + panel_h - 12}" text-anchor="middle" fill="#566176" font-family="sans-serif" font-size="12">metres</text>')
    legend = ((BLUE, "Ground truth"), (GREEN, "Prediction: matched"), (ORANGE, "Prediction: unmatched"))
    for index, (color, text) in enumerate(legend):
        x = 28 + index * 250
        parts += [f'<rect x="{x}" y="530" width="18" height="10" fill="{color}" fill-opacity=".2" stroke="{color}"/>',
                  f'<text x="{x + 25}" y="540" fill="{INK}" font-family="sans-serif" font-size="13">{text}</text>']
    return "".join(parts) + "</svg>"


def _polygon(row):
    """Return a valid X-Z polygon, otherwise ``None``."""
    try:
        polygon = np.asarray(row["polygon_xz_m"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    return polygon if polygon.ndim == 2 and polygon.shape[0] >= 3 and polygon.shape[1] == 2 and np.all(np.isfinite(polygon)) else None


def _region_svg(scene, threshold, xz_bounds, width=1260, height=560):
    pred_rows = scene.get("predicted_regions", [])
    gt_rows = scene.get("ground_truth_regions", [])
    pred = [(index, row, _polygon(row)) for index, row in enumerate(pred_rows) if _polygon(row) is not None]
    gt = [(index, row, _polygon(row)) for index, row in enumerate(gt_rows) if _polygon(row) is not None]
    matches = assignment(pred_rows, gt_rows, threshold)
    matched_pred = {pi for pi, _, _ in matches}
    low, high = xz_bounds
    plot_x, plot_y, plot_w, plot_h = 95, 80, width - 150, height - 140
    # RViz renders the map with an equal metric scale on X and Y.  Scaling
    # the two coordinates independently makes the same room polygon look
    # translated/deformed in the browser, especially for this tall scene.
    data_extent = np.maximum(high - low, 1e-9)
    scale = min(plot_w / data_extent[0], plot_h / data_extent[1])
    used_w, used_h = float(data_extent[0] * scale), float(data_extent[1] * scale)
    origin_x = plot_x + (plot_w - used_w) * 0.5
    origin_y = plot_y + (plot_h - used_h) * 0.5

    def points(polygon):
        x = origin_x + (polygon[:, 0] - low[0]) * scale
        y = origin_y + (high[1] - polygon[:, 1]) * scale
        return " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))

    scene_name = html.escape(str(scene.get("scene", "unknown")))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="24" y="32" fill="{INK}" font-family="sans-serif" font-size="20" font-weight="bold">Scene {scene_name} — region polygons (top view X-Y)</text>',
             f'<text x="24" y="55" fill="{INK}" font-family="sans-serif" font-size="13">Valid GT regions: {len(gt)} · predictions: {len(pred)} · IoU matches (&gt; {threshold:g}): {len(matches)}</text>',
             f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" fill="#fbfcfe" stroke="{GRID}"/>']
    for index, row, polygon in gt:
        label = html.escape("GT region " + _identifier(row, index, True))
        parts.append(f'<polygon points="{points(polygon)}" fill="{BLUE}" fill-opacity=".10" stroke="{BLUE}" stroke-width="1.5"><title>{label}</title></polygon>')
    for index, row, polygon in pred:
        color = GREEN if index in matched_pred else ORANGE
        best_iou = max((geometry_iou(row, candidate) for candidate in gt_rows), default=0.0)
        label = html.escape("Predicted region " + _identifier(row, index) +
                            (" (matched)" if index in matched_pred else " (unmatched)") +
                            f"; best IoU: {best_iou:.3f}")
        parts.append(f'<polygon points="{points(polygon)}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="6 3"><title>{label}</title></polygon>')
    legend = ((BLUE, "Ground truth"), (GREEN, "Prediction: matched"), (ORANGE, "Prediction: unmatched"))
    for index, (color, text) in enumerate(legend):
        x = 105 + index * 260
        parts += [f'<rect x="{x}" y="{height - 35}" width="18" height="10" fill="{color}" fill-opacity=".2" stroke="{color}"/>',
                  f'<text x="{x + 25}" y="{height - 25}" fill="{INK}" font-family="sans-serif" font-size="13">{text}</text>']
    return "".join(parts) + "</svg>"


def render(scenes, output, threshold=.5, region_threshold=.5):
    """Write an HTML page with object and region visualizations per scene."""
    scenes = [filtered_scene(scene, include_regions=True) for scene in scenes]
    svg = "\n".join(
        _scene_svg(scene, threshold, _scene_xz_bounds(scene)) +
        _region_svg(scene, region_threshold, _scene_xz_bounds(scene))
        for scene in scenes)
    document = "<!doctype html><html><head><meta charset=\"utf-8\"><title>Metrics: bounding boxes</title></head><body>" + svg + "</body></html>\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="JSON manifest or manifest directory")
    parser.add_argument("--output", type=Path,
                        help="HTML output path (default: next to the manifest)")
    parser.add_argument("--object-iou", type=float, default=.5,
                        help="3D IoU threshold used to mark matches")
    parser.add_argument("--region-iou", type=float, default=.5,
                        help="2D polygon IoU threshold used to mark region matches")
    args = parser.parse_args()
    if not 0 <= args.object_iou <= 1 or not 0 <= args.region_iou <= 1:
        parser.error("--object-iou and --region-iou must be between 0 and 1")
    scenes = load(args.path)
    output = args.output or ((args.path.parent if args.path.is_file() else args.path) /
                             "metrics_eval_boxes.html")
    render(scenes, output, args.object_iou, args.region_iou)
    print(f"Created {output} ({len(scenes)} scene(s)). Open it in a browser; hover over boxes for IDs and labels.")


if __name__ == "__main__":
    main()
