"""One four-panel renderer for native baseline replay, HTTP preview and MP4 export."""
from __future__ import annotations

import argparse
import bisect
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .replay_model import SCHEMA, read_json

BG = '#09111e'
PANEL = '#101d2d'
TEXT = '#edf4fc'
MUTED = '#9baec4'
CYAN = '#4dd9ef'
GREEN = '#62e0a1'
AMBER = '#ffcc70'
RED = '#ff788a'
COLORS = {'object': CYAN, 'place': '#739bcb', 'room': AMBER, 'floor': '#cf9eff'}
BOX_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if (a ^ b) in (1, 2, 4)]
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def text(draw, xy, value, size=18, fill=TEXT, bold=False):
    draw.text(xy, str(value), fill=fill, font=font(size, bold))


def project_segments(corners, pose, width, height, hfov):
    """OpenGL camera (-Z forward) projection, clipped at the near plane."""
    pose = np.asarray(pose)
    camera = (np.asarray(corners) - pose[:3, 3]) @ pose[:3, :3]
    optical = camera * np.array([1, -1, -1])
    focal = width / (2 * math.tan(math.radians(hfov) / 2))
    lines = []
    for i, j in BOX_EDGES:
        a, b = optical[i].copy(), optical[j].copy()
        if a[2] < 0.05 and b[2] < 0.05:
            continue
        if a[2] < 0.05:
            a += (b - a) * ((0.05 - a[2]) / (b[2] - a[2]))
        if b[2] < 0.05:
            b += (a - b) * ((0.05 - b[2]) / (a[2] - b[2]))
        pa = np.array([focal * a[0] / a[2] + width / 2, focal * a[1] / a[2] + height / 2])
        pb = np.array([focal * b[0] / b[2] + width / 2, focal * b[1] / b[2] + height / 2])
        if not (max(pa[0], pb[0]) < 0 or min(pa[0], pb[0]) > width or
                max(pa[1], pb[1]) < 0 or min(pa[1], pb[1]) > height):
            lines.append((tuple(pa), tuple(pb)))
    return lines


class Replay:
    def __init__(self, model):
        self.model = read_json(model) if isinstance(model, (str, Path)) else model
        if self.model['schema'] != SCHEMA:
            raise ValueError('Unknown replay schema')
        self.final_metrics = dict(self.model.get('final_metrics', {}))
        self.metric_report = None
        from .replay_metrics import validate_metrics
        validate_metrics(self.model['metrics_history'])
        validate_metrics([{'time_s': 0, 'metrics': self.final_metrics}])
        self.frames = self.model['frames']
        self.times = [f['time_s'] for f in self.frames]
        self.start, self.end = self.times[0], self.times[-1]
        self.positions = np.array([f['pose'] for f in self.frames])[:, :3, 3]
        self.graph_cache = {}
        self.map_base, self.map_xy = self.make_map()

    def index(self, time_s):
        return max(0, min(len(self.frames) - 1, bisect.bisect_right(self.times, time_s) - 1))

    def state(self, time_s):
        i = self.index(time_s)
        frame = self.frames[i]
        actions = [a for a in self.model['actions'] if a['frame_index'] <= i]
        active, removed = {}, {}
        for a in actions:
            result = a['result']
            oid = str(result['object_id'])
            if a['action'] == 'remove':
                removed[oid] = dict(active.pop(oid, {}), position=result.get('position'), time_s=a['time_s'])
            else:
                active[oid] = {'position': result['position'], 'action': a['action'],
                               'time_s': a['time_s'], 'id': oid}
                removed.pop(oid, None)
        metrics = {}
        for row in self.model['metrics_history']:
            if row['time_s'] <= frame['time_s']:
                metrics.update(row['metrics'])
        history = [g for g in self.model['graph_history'] if g['time_s'] <= frame['time_s']]
        if self.model['graph_history']:
            graph = history[-1] if history else {'nodes': [], 'edges': [], 'scope': 'awaiting_recorded_snapshot'}
        else:
            graph = self.model['graph']
        scans = [event for f in self.frames[:i + 1] if f['reason'] == 'tour' for event in (f.get('event') or [])]
        return {'index': i, 'frame': frame, 'actions': actions, 'active': active, 'removed': removed,
                'metrics': metrics, 'graph': graph, 'scans': scans,
                'action_counts': dict(Counter(a['action'] for a in actions))}

    def make_map(self):
        image = Image.new('RGB', (764, 354), PANEL)
        draw = ImageDraw.Draw(image)
        points = [self.positions[:, [0, 2]], np.array([p['xyz'] for p in self.model['trajectory']])[:, [0, 2]]]
        cloud = np.array(self.model['map_points'])
        if len(cloud):
            points.append(cloud[:, [0, 2]])
        for a in self.model['actions']:
            if a['result'].get('position'):
                points.append(np.array([a['result']['position']])[:, [0, 2]])
        allpoints = np.vstack(points)
        lo, hi = allpoints.min(0), allpoints.max(0)
        center = (lo + hi) / 2
        scale = min(700 / max(hi[0] - lo[0], 1), 300 / max(hi[1] - lo[1], 1))
        def xy(position):
            a = np.array([position[0], position[2]])
            return tuple(((a - center) * scale + [382, 177]).tolist())
        for x in range(math.floor(lo[0]), math.ceil(hi[0]) + 1):
            draw.line([xy([x, 0, lo[1]]), xy([x, 0, hi[1]])], fill='#1b2a3c')
        for z in range(math.floor(lo[1]), math.ceil(hi[1]) + 1):
            draw.line([xy([lo[0], 0, z]), xy([hi[0], 0, z])], fill='#1b2a3c')
        for p in cloud:
            draw.point(xy(p), fill='#405264')
        path = [xy(p['xyz']) for p in self.model['trajectory']]
        if len(path) > 1:
            draw.line(path, fill='#428295', width=2)
        for p in self.model['trajectory']:
            if p.get('scan_deg', 0) > 0:
                x, y = xy(p['xyz'])
                draw.ellipse((x-5, y-5, x+5, y+5), outline=CYAN, width=2)
                text(draw, (x+7, y-10), p['stop'], 12, CYAN)
        return image, xy

    def feed(self, state, object_id=None):
        f = state['frame']
        with Image.open(f['image']) as source:
            image = source.convert('RGB')
        draw = ImageDraw.Draw(image)
        candidates = []
        for n in state['graph']['nodes']:
            if n['type'] != 'object' or 'corners' not in n:
                continue
            if object_id is not None and n['id'] != object_id:
                continue
            lines = project_segments(n['corners'], f['pose'], f['width'], f['height'], f['hfov_deg'])
            if lines:
                distance = np.linalg.norm(np.asarray(n['position']) - self.positions[state['index']])
                candidates.append((distance, n, lines))
        for _, n, lines in sorted(candidates, key=lambda x: x[0])[:12]:
            for a, b in lines:
                draw.line([a, b], fill=CYAN, width=2)
            points = np.array([p for line in lines for p in line])
            x, y = np.clip(points.min(0), [0, 0], [f['width'] - 155, f['height'] - 22])
            label = n['label'][:22]
            draw.rectangle((x, y, x+max(55, len(label)*7), y+19), fill='#092331')
            text(draw, (x+3, y+1), label, 11, CYAN)
        return image, len(candidates)

    def graph_image(self, graph):
        key = graph.get('time_s', 'final')
        if key in self.graph_cache:
            return self.graph_cache[key]
        image = Image.new('RGB', (764, 354), PANEL)
        draw = ImageDraw.Draw(image)
        nodes, edges = graph['nodes'], graph['edges']
        points = [n['position'] for n in nodes if n.get('position') is not None]
        if not points:
            text(draw, (30, 130), 'Waiting for a recorded graph snapshot', 21, MUTED)
            return image
        p = np.array(points)[:, [0, 2]]
        lo, hi = p.min(0), p.max(0)
        scale = min(690 / max(hi[0]-lo[0], 1), 286 / max(hi[1]-lo[1], 1))
        center = (lo + hi) / 2
        coords = {n['id']: tuple((np.array(n['position'])[[0, 2]] - center)*scale + [382, 170])
                  for n in nodes if n.get('position') is not None}
        for edge in edges:
            if edge['source'] in coords and edge['target'] in coords:
                draw.line([coords[edge['source']], coords[edge['target']]], fill='#334760', width=1)
        for n in sorted(nodes, key=lambda n: n['type'] != 'object'):
            if n['id'] not in coords:
                continue
            x, y = coords[n['id']]
            radius = 4 if n['type'] in ('object', 'place') else 8
            color = COLORS.get(n['type'], CYAN)
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline=PANEL)
        # A bounded label budget keeps the graph legible; every node and edge is drawn.
        labeled = [n for n in nodes if n['type'] in ('floor', 'room')]
        labeled += [n for n in nodes if n['type'] == 'object'][::max(1, len(nodes)//8)][:8]
        for n in labeled:
            if n['id'] in coords:
                x, y = coords[n['id']]
                text(draw, (max(2, min(x+5, 620)), max(2, min(y-20, 320))), n['label'][:22], 12, COLORS.get(n['type'], TEXT))
        self.graph_cache[key] = image
        return image

    def render(self, time_s, object_id=None):
        state = self.state(time_s)
        image = Image.new('RGB', (1600, 1000), BG)
        draw = ImageDraw.Draw(image)
        baseline = {'clio': 'Clio', 'hovsg': 'HOV-SG', 'dynamicgsg': 'DynamicGSG',
                    'dashboard': 'GRAPH-API / dashboard'}[self.model['baseline']]
        text(draw, (26, 17), baseline.upper(), 28, TEXT, True)
        text(draw, (350, 22), self.model['scene'] + '  |  shared scheduled input', 19, MUTED)
        elapsed = state['frame']['time_s'] - self.start
        text(draw, (1250, 20), f'{elapsed:05.1f} / {self.end-self.start:05.1f} s', 22, CYAN, True)
        for x, y in ((18, 68), (810, 68), (18, 522), (810, 522)):
            draw.rounded_rectangle((x, y, x+772, y+440), radius=14, fill=PANEL, outline='#25374d')
        text(draw, (36, 81), '01  CAMERA + OVERLAYS', 19, TEXT, True)
        graph_scope = state['graph']['scope']
        scope = 'Final graph boxes reprojected' if graph_scope == 'final_snapshot' else 'Recorded snapshot boxes'
        text(draw, (36, 110), scope + ' · nearest 12 · no occlusion test', 14, MUTED)
        feed, visible = self.feed(state, object_id)
        feed.thumbnail((744, 354), Image.Resampling.LANCZOS)
        image.paste(feed, (32+(744-feed.width)//2, 143+(354-feed.height)//2))
        text(draw, (829, 81), '02  KNOWLEDGE / SCENE GRAPH', 19, TEXT, True)
        scope_label = 'Final reconstruction · graph history was not recorded' if graph_scope == 'final_snapshot' else 'Recorded graph snapshot'
        text(draw, (829, 110), scope_label, 14, AMBER)
        image.paste(self.graph_image(state['graph']), (814, 145))
        text(draw, (36, 535), '03  DYNAMIC SCHEDULE', 19, TEXT, True)
        text(draw, (36, 564), 'Cyan: tour  ·  green: robot trail  ·  amber: object  ·  red: removed', 13, MUTED)
        minimap = self.map_base.copy()
        md = ImageDraw.Draw(minimap)
        path = [self.map_xy(p) for p in self.positions[:state['index']+1]]
        if len(path) > 1:
            md.line(path, fill=GREEN, width=3)
        for a in state['active'].values():
            x, y = self.map_xy(a['position'])
            md.ellipse((x-7, y-7, x+7, y+7), fill=AMBER, outline=TEXT, width=1)
            text(md, (x+10, y-10), 'object ' + a['id'], 13, AMBER)
        for a in state['removed'].values():
            if a.get('position'):
                x, y = self.map_xy(a['position'])
                md.line((x-6, y-6, x+6, y+6), fill=RED, width=3)
                md.line((x-6, y+6, x+6, y-6), fill=RED, width=3)
        pose = np.array(state['frame']['pose'])
        x, y = self.map_xy(pose[:3, 3])
        forward = -(pose[:3, 2])[[0, 2]]
        angle = math.atan2(forward[1], forward[0])
        triangle = [(x+14*math.cos(angle), y+14*math.sin(angle)),
                    (x+9*math.cos(angle+2.5), y+9*math.sin(angle+2.5)),
                    (x+9*math.cos(angle-2.5), y+9*math.sin(angle-2.5))]
        md.polygon(triangle, fill=GREEN, outline=TEXT)
        image.paste(minimap, (22, 596))
        text(draw, (829, 535), '04  RECORDED METRICS', 19, TEXT, True)
        text(draw, (829, 564), 'Cursor counters · final box scores use whole-scene GT', 14, MUTED)
        counts = Counter(n['type'] for n in state['graph']['nodes'])
        values = [
            ('Input frame', f'{state["index"]+1} / {len(self.frames)}'),
            ('Scans completed', str(len(state['scans']))),
            ('Schedule added / moved / removed', ' / '.join(str(state['action_counts'].get(k, 0)) for k in ('spawn', 'move', 'remove'))),
            ('Scheduled objects active', str(len(state['active']))),
            ('Graph objects (final)' if graph_scope == 'final_snapshot' else 'Graph objects (snapshot)', str(counts['object'])),
            ('Graph nodes / edges', f'{len(state["graph"]["nodes"])} / {len(state["graph"]["edges"])}'),
        ]
        report = self.metric_report
        values.append(('Final GT / TP / FP / FN', ' / '.join(str(report[k]) for k in ('gt_objects', 'tp', 'fp', 'fn')) if report else 'N/A: no GT evaluation'))
        fields = [('stage_latency_ms', 'Backend stage mean', 'stage not recorded'),
                  ('run_wall_time_s', 'Run wall time (final)', 'timing not recorded'),
                  ('objects_added', 'Model objects added', 'no object history'),
                  ('avg_latency_ms', 'Input-to-result latency', 'no paired timestamps'),
                  ('precision', 'Final box precision', 'no GT evaluation'),
                  ('recall', 'Final box recall', 'no GT evaluation'),
                  ('f1', 'Final box F1', 'no GT evaluation')]
        for key, label, reason in fields:
            metric = self.final_metrics.get(key) or state['metrics'].get(key)
            if metric and metric.get('label'):
                label = metric['label']
            value = f'{metric["value"]} {metric["unit"]}' if metric and metric['value'] is not None else 'N/A: ' + reason
            values.append((label, value))
        for i, (label, value) in enumerate(values):
            y = 599 + i * 24
            draw.line((829, y+23, 1560, y+23), fill='#233248')
            text(draw, (832, y), label, 14, MUTED)
            text(draw, (1230, y), value, 14, TEXT if not value.startswith('N/A') else AMBER)
        latest = state['actions'][-1] if state['actions'] else None
        event = f'{latest["action"].upper()} object {latest["result"]["object_id"]} at {latest["time_s"]-self.start:.1f}s' if latest else 'Waiting for the first object action'
        text(draw, (36, 971), event, 14, AMBER)
        text(draw, (810, 971), 'External replay adapter · native baseline algorithms unchanged', 14, MUTED)
        return image


def export_video(replay, output, fps=10, speed=1):
    if fps <= 0 or speed <= 0:
        raise ValueError('FPS and speed must be positive')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = math.ceil((replay.end-replay.start) / speed * fps) + 1
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
               '-pixel_format', 'rgb24', '-video_size', '1600x1000', '-framerate', str(fps),
               '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '21',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for i in range(count):
            proc.stdin.write(replay.render(min(replay.end, replay.start + i*speed/fps)).tobytes())
            if i % 100 == 0:
                print(f'{output.name}: {i}/{count}', flush=True)
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode:
        raise RuntimeError(f'ffmpeg failed ({proc.returncode})')
    result = {'complete': True, 'frames': count, 'fps': fps, 'speed': speed,
              'source_start_s': replay.start, 'source_end_s': replay.end,
              'baseline': replay.model['baseline'], 'limitations': replay.model['limitations']}
    output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model')
    p.add_argument('--video')
    p.add_argument('--still')
    p.add_argument('--time', type=float, default=0, help='Seconds from recording start')
    p.add_argument('--fps', type=float, default=10)
    p.add_argument('--speed', type=float, default=1)
    args = p.parse_args()
    replay = Replay(args.model)
    if args.still:
        replay.render(replay.start+args.time).save(args.still)
    if args.video:
        export_video(replay, args.video, args.fps, args.speed)
    if not args.still and not args.video:
        p.error('Choose --still or --video')


if __name__ == '__main__':
    main()
