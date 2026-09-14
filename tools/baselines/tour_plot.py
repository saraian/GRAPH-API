"""Export planned tours over actual native navmesh triangles (display only)."""
import argparse
import json
from pathlib import Path

import numpy as np


def plot(navmesh, trajectory, output, title, trail=None):
    import habitat_sim
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    pf = habitat_sim.PathFinder()
    if not pf.load_nav_mesh(str(navmesh)):
        raise ValueError('Cannot load navmesh')
    vertices = np.asarray(pf.build_navmesh_vertices())
    if len(vertices) % 3:
        raise ValueError('Navmesh triangles have incomplete vertex triples')
    triangles = vertices.reshape(-1, 3, 3)[:, :, [0, 2]]
    fig, ax = plt.subplots(figsize=(11, 11), dpi=180)
    fig.patch.set_facecolor('#f6f8fc')
    ax.set_facecolor('#f6f8fc')
    ax.add_collection(PolyCollection(triangles, facecolors='#d7e2eb', edgecolors='#a9bac8', linewidths=.15))
    goals = np.asarray([p['xyz'] for p in trajectory])
    snapped = [pf.snap_point(p) for p in goals]
    for i, (a, b) in enumerate(zip(snapped, np.roll(snapped, -1, axis=0))):
        path = habitat_sim.ShortestPath()
        path.requested_start, path.requested_end = a, b
        if not pf.find_path(path):
            raise ValueError(f'No path for leg {i}')
        points = np.asarray(path.points)
        closure = i == len(snapped)-1
        ax.plot(points[:, 0], points[:, 2], color='#008ba5', lw=1.6,
                alpha=.7, linestyle='--' if closure else '-',
                label='Planned tour on navmesh' if i == 0 else ('Lap closure' if closure else None))
    scans = [p for p in trajectory if p.get('scan_deg', 0) > 0]
    for i, stop in enumerate(scans, 1):
        x, _, z = stop['xyz']
        ax.scatter([x], [z], s=30, c='#ffd174', edgecolors='#734e00', zorder=4,
                   label='Scan stops (visit order)' if i == 1 else None)
        ax.annotate(str(i), (x, z), xytext=(5, 5), textcoords='offset points', fontsize=8,
                    bbox={'boxstyle': 'round,pad=.1', 'fc': '#ffffff', 'ec': 'none', 'alpha': .85}, zorder=5)
    if trail is not None:
        positions = np.asarray(trail)
        ax.plot(positions[:, 0], positions[:, 2], c='#29904a', lw=1.4, label='Recorded robot trail', zorder=3)
    ax.scatter(goals[0, 0], goals[0, 2], marker='*', s=180, c='#b92241', edgecolors='white', label='Start', zorder=6)
    ax.autoscale()
    ax.margins(.08)
    ax.set_aspect('equal')
    ax.set_xlabel('Habitat world X (m)')
    ax.set_ylabel('Habitat world Z (m)')
    ax.set_title(f'{title}\n{len(scans)} scan stops · {len(trajectory)} trajectory points · native navmesh {pf.navigable_area:.2f} m²', fontsize=13, pad=16)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=.15)
    fig.text(.5, .015, 'Grey: actual navmesh triangles. Cyan: planned geodesic route; not proof of executed room coverage.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference-model', type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    doc = json.loads(args.selection.read_text())
    for row in doc['scenes']:
        schedule = json.loads(Path(row['assets']['schedule']['path']).read_text())['schedule']
        if len(schedule) != 1:
            raise ValueError('This plot expects a single scheduled floor')
        target = args.output_dir / (row['scene_id'] + '-tour.png')
        plot(row['assets']['navmesh']['path'], schedule[0]['trajectory'], target, row['scene_id'])
        print(target, flush=True)
    if args.reference_model:
        model = json.loads(args.reference_model.read_text())
        root = Path(__file__).resolve().parents[2]
        scene = model['scene']
        navmesh = root / 'lost3dsg/FOUND-Dataset/habitat/hm3d-val-habitat-v0.2' / scene / (scene.split('-')[1]+'.basis.navmesh')
        target = args.output_dir / (scene + '-short-replay-tour.png')
        plot(navmesh, model['trajectory'], target, scene + ' — shortened replay fixture',
             [np.asarray(f['pose'])[:3, 3].tolist() for f in model['frames']])
        print(target, flush=True)


if __name__ == '__main__':
    main()
