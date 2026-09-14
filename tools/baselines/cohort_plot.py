"""Tour PNGs with floor-separated native navmeshes and explicit event connectors."""
import argparse
import html
import json
from pathlib import Path

import numpy as np


def _resolve(path, root):
    path = Path(path)
    return path if path.is_absolute() else root / path


def render(row, output, root=Path('.')):
    import habitat_sim
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    root = Path(root)
    levels = json.loads(_resolve(row['assets']['schedule']['path'], root).read_text())['schedule']
    lap_policy = row.get('lap_policy', {})
    configured_laps = int(lap_policy.get('configured_execution_laps', 1))
    pf = habitat_sim.PathFinder()
    if not pf.load_nav_mesh(str(_resolve(row['assets']['navmesh']['path'], root))):
        raise ValueError('Native navmesh unavailable')
    vertices = np.asarray(pf.build_navmesh_vertices()).reshape(-1, 3, 3)
    n = len(levels)
    fig, axes = plt.subplots(1, n, figsize=(11*n, 11), dpi=150, squeeze=False)
    fig.patch.set_facecolor('#f5f8fc')
    heights = np.array([level['height'] for level in levels])
    assignments = np.abs(vertices.mean(axis=1)[:, 1, None]-heights[None, :]).argmin(axis=1)
    colors = {'spawn': '#238c46', 'move': '#d77700', 'remove': '#c72c46'}
    lines = []
    templates = set()
    event_laps = set()
    for floor, (ax, level) in enumerate(zip(axes[0], levels)):
        ax.set_facecolor('#f5f8fc')
        cov = level.get('coverage')
        if cov:
            with np.load(_resolve(cov['grid']['path'], root)) as grid:
                mask = grid['mask']
                origin = grid['origin_xz']
                res = float(grid['resolution_m'])
                ax.imshow(mask, origin='lower', extent=(origin[0], origin[0]+mask.shape[1]*res,
                          origin[1], origin[1]+mask.shape[0]*res), interpolation='nearest',
                          cmap=ListedColormap(['#f5f8fc', '#596776', '#f4cf80', '#a0d6b7']), vmin=0, vmax=3)
        ax.add_collection(PolyCollection(vertices[assignments==floor][:, :, [0, 2]],
                                         facecolors='none' if cov else '#d6e0e9', edgecolors='#b1c0cc', linewidths=.12))
        points = np.asarray([p['xyz'] for p in level['trajectory']])
        for a, b in zip(points, np.roll(points, -1, axis=0)):
            path = habitat_sim.ShortestPath()
            path.requested_start, path.requested_end = pf.snap_point(a), pf.snap_point(b)
            if not pf.find_path(path):
                raise ValueError('Tour path disappeared')
            p = np.asarray(path.points)
            ax.plot(p[:, 0], p[:, 2], color='#168fa7', lw=1.3, alpha=.85, zorder=2)
        scans = [p for p in level['trajectory'] if p.get('scan_deg', 0) > 0]
        for i, p in enumerate(scans, 1):
            x, _, z = p['xyz']
            ax.scatter(x, z, s=17, c='#60bfd0', edgecolors='#146579', zorder=3)
            ax.annotate(str(i), (x, z), xytext=(3, 3), textcoords='offset points', fontsize=7,
                        bbox={'fc': 'white', 'ec': 'none', 'alpha': .75, 'pad': .3}, zorder=4)
        ax.scatter(points[0, 0], points[0, 2], marker='*', s=160, c='#603bb1', edgecolors='white', zorder=7)
        plan = next(p for p in row['dynamic_plans'] if p['floor_index']==floor)
        template = plan.get('action_coverage', {}).get('template', 'dynamic object').replace('_', ' ')
        extent = plan.get('action_coverage', {}).get('template_scaled_max_extent_m')
        if plan['events']:
            templates.add(f'{template} ({extent:.3f} m max extent)' if extent is not None else template)
        for event in plan['events']:
            event_laps.add(int(event.get('trigger', {}).get('lap', 0)) + 1)
            robot, target = np.array(event['robot_position'])[[0, 2]], np.array(event['object_position'])[[0, 2]]
            color = colors[event['action']]
            ax.scatter(*robot, marker='s', s=85, c='white', edgecolors=color, linewidths=2, zorder=6)
            ax.annotate('', xy=target, xytext=robot, arrowprops={'arrowstyle': '-|>', 'color': color,
                        'linestyle': ':', 'linewidth': 2, 'connectionstyle': 'arc3,rad=.12'}, zorder=6)
            marker = {'spawn': 'P', 'move': 'D', 'remove': 'X'}[event['action']]
            ax.scatter(*target, marker=marker, s=110, c=color, edgecolors='white', linewidths=.8, zorder=7)
            offset = (10, -22) if event['action']=='remove' else (10, 15)
            human_lap = int(event.get('trigger', {}).get('lap', 0)) + 1
            ax.annotate(event['id']+' '+event['action']+f' L{human_lap}', target,
                        xytext=offset, textcoords='offset points',
                        color=color, fontsize=9, weight='bold', bbox={'fc': 'white', 'ec': color, 'alpha': .9, 'pad': 2}, zorder=8)
            if event['previous_object_position'] is not None:
                old = np.array(event['previous_object_position'])[[0, 2]]
                ax.annotate('', xy=target, xytext=old, arrowprops={'arrowstyle': '->', 'color': color,
                            'linestyle': '--', 'linewidth': 1.5, 'connectionstyle': 'arc3,rad=-.12'}, zorder=5)
            object_number = event['object'].removeprefix('dynamic_object_').lstrip('0') or '0'
            lines.append(
                f"{event['id']} {event['action']} object {object_number} "
                f"@ lap {human_lap}, scan {event['scan_visit_number']}"
            )
        if not plan['events']:
            ax.text(.02, .02, 'No compiled object changes on this floor', transform=ax.transAxes, fontsize=9,
                    bbox={'fc': 'white', 'ec': '#8795a6'})
        for transition in row['floor_transitions']:
            if floor not in (transition['from_floor'], transition['to_floor']):
                continue
            path = np.array(transition['points'])
            assigned = np.abs(path[:, 1, None]-heights).argmin(axis=1)
            segments = [path[i:i+2] for i in range(len(path)-1) if assigned[i]==floor or assigned[i+1]==floor]
            for seg in segments:
                ax.plot(seg[:, 0], seg[:, 2], c='#8552bf', ls='--', lw=2, alpha=.8)
        ax.autoscale()
        ax.margins(.12)
        ax.set_aspect('equal')
        ax.grid(alpha=.13)
        ax.set_xlabel('Habitat X (m)')
        ax.set_ylabel('Habitat Z (m)')
        duration = level.get('duration')
        timing = ''
        if duration:
            configured_seconds = duration.get(
                'configured_laps_seconds',
                duration['first_lap_seconds'] * configured_laps,
            )
            timing = (f'\nNominal: {duration["first_lap_seconds"]/60:.1f} min / first lap; '
                      f'{configured_seconds/60:.1f} min / {configured_laps} laps at {duration["fps"]:g} fps')
            if not duration['valid_complete_tour']:
                timing = '\nDuration invalid: native navigation skipped waypoints'
        ax.set_title(f'Floor {floor+1} · height {level["height"]:+.2f} m\n{len(scans)} scan stops · {len(points)} trajectory points' + (f'\nFree-floor coverage {cov["coverage_pct"]:.1f}% ({cov["covered_area_m2"]:.1f} / {cov["free_area_m2"]:.1f} m²), range {cov["range_m"]:g} m' if cov else '') + timing, fontsize=12)
    fig.suptitle(f'{row["scene_id"]} · {row["cohort_group"]} · {configured_laps} complete laps and expected object changes\n'
                 f'{row["annotated_room_regions"]} annotated room regions · {row["native_gt_objects"]} native GT objects', fontsize=16, y=.99)
    legend = [Line2D([0], [0], color='#168fa7', label='Full tour'),
              Line2D([0], [0], marker='*', color='none', markerfacecolor='#603bb1', markersize=12, label='Start'),
              Line2D([0], [0], color='#8552bf', linestyle='--', label='Cross-floor navmesh route (not certified)'),
              Line2D([0], [0], marker='s', markerfacecolor='white', color='#333333', linestyle=':', label='Robot trigger → change location'),
              Line2D([0], [0], color='#d77700', linestyle='--', label='Object relocation')]
    if any('coverage' in level for level in levels):
        legend += [Line2D([0], [0], color='#a0d6b7', lw=8, label='Covered free floor'),
                   Line2D([0], [0], color='#f4cf80', lw=8, label='Uncovered free floor'),
                   Line2D([0], [0], color='#596776', lw=8, label='Obstacle on floor')]
    event_lines = ['   |   '.join(lines[index:index + 3]) for index in range(0, len(lines), 3)]
    lap_text = ', '.join(map(str, sorted(event_laps))) or 'none'
    confirmations = lap_policy.get('confirmation_laps_human', [])
    confirmation_text = (
        f' Confirmation-only lap(s): {", ".join(map(str, confirmations))}.'
        if confirmations else ''
    )
    event_summary = (
        f'Full tour repeats for {configured_laps} complete lap(s); object actions trigger on lap(s) {lap_text}.'
        f'{confirmation_text} '
        f'Template: {", ".join(sorted(templates))}\n' + '\n'.join(event_lines)
    )
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .20), ncol=3, fontsize=10)
    fig.text(.5, .105, event_summary, ha='center', va='center', fontsize=8)
    fig.text(.5, .025, 'Expected waypoint-triggered changes, not recorded outcomes. Mesh outlines = navmesh; numbers = scan visit order.\n'
             'Floor panels use separate height slices. Multi-floor execution remains gated on certification.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .32, 1, .92))
    fig.savefig(output)
    plt.close(fig)


def export_cohort(cohort_path):
    from PIL import Image, ImageDraw
    cohort_path = Path(cohort_path)
    root = cohort_path.parent
    rows = json.loads(cohort_path.read_text())['scenes']
    cards = []
    for row in rows:
        png = root/(row['scene_id']+'-tour-changes.png')
        render(row, png, root=root)
        with Image.open(png) as original:
            tile = Image.new('RGB', (1000, 620), '#f5f8fc')
            preview = original.copy()
            preview.thumbnail((990, 570))
            tile.paste(preview, ((1000-preview.width)//2, 35+(570-preview.height)//2))
            label = (row['scene_id']+' | '+row['cohort_group']+
                     f' | {row.get("lap_policy", {}).get("configured_execution_laps", 1)} laps')
            ImageDraw.Draw(tile).text((16, 12), label, fill='#14283d')
            cards.append(tile)
        print(png, flush=True)
    overview = Image.new('RGB', (2000, 620*((len(cards)+1)//2)), 'white')
    for i, tile in enumerate(cards):
        overview.paste(tile, (1000*(i%2), 620*(i//2)))
    overview.save(root/'overview.png')
    body = ''.join(f'<section><h2>{html.escape(r["scene_id"])}</h2><a href="{r["scene_id"]}-tour-changes.png"><img src="{r["scene_id"]}-tour-changes.png"></a></section>' for r in rows)
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Baseline cohort tours</title><style>body{font-family:system-ui;background:#f5f8fc;margin:24px}img{max-width:100%}section{margin-bottom:36px}</style><h1>Baseline tour pack</h1><p>Tours, native navmeshes, free-floor coverage and expected waypoint-triggered object changes.</p>'+body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', type=Path)
    args = parser.parse_args()
    export_cohort(args.cohort)


if __name__ == '__main__':
    main()
