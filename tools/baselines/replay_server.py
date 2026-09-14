"""Reduced read-only baseline dashboard: one timeline, four panels, shared readers."""
from __future__ import annotations

import argparse
import io
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from .replay_model import file_stamp, read_json
from .replay_render import Replay


def create_app(models, video_root=None):
    models = list(models)
    replays = {Path(path).stem.removesuffix('.replay'): Replay(path) for path in models}
    scene_paths = {Path(p).stem.removesuffix('.replay'): Path(p).with_suffix('.scene.json') for p in models}
    model_hashes = {Path(p).stem.removesuffix('.replay'): file_stamp(p)['sha256'] for p in models}
    from .replay_metrics import validate_metrics
    for path in models:
        key = Path(path).stem.removesuffix('.replay')
        report_path = Path(path).with_suffix('.metrics.json')
        if report_path.is_file():
            report = read_json(report_path)
            if report.get('schema') != 'graphapi.baseline_metrics.v1' or report.get('model_sha256') != model_hashes[key]:
                raise ValueError('Metrics report belongs to a different replay model')
            validate_metrics([{'time_s': 0, 'metrics': report['metrics']}])
            replays[key].final_metrics.update(report['metrics'])
            replays[key].metric_report = report
    verified_scenes = {}
    app = FastAPI(title='Baseline replay', docs_url=None, redoc_url=None)
    vendor = Path(__file__).resolve().parents[2] / 'lost3dsg/dashboard/vendor_three'
    if vendor.is_dir():
        app.mount('/vendor', StaticFiles(directory=vendor), name='three_vendor')
    video_root = Path(video_root).resolve() if video_root else None

    def selected(name):
        if name not in replays:
            raise HTTPException(404, 'Unknown replay')
        return replays[name]

    def video_ready(name):
        return bool(video_root and (video_root/(name+'.mp4')).is_file() and
                    (video_root/(name+'.json')).is_file() and read_json(video_root/(name+'.json')).get('complete'))

    @app.get('/', response_class=HTMLResponse)
    def page():
        return Path(__file__).with_name('replay.html').read_text()

    @app.get('/scene', response_class=HTMLResponse)
    def scene_page():
        return Path(__file__).with_name('replay_scene.html').read_text()

    @app.get('/api/scene/{name}')
    def scene(name: str):
        selected(name)
        path = scene_paths[name]
        if not path.is_file():
            raise HTTPException(404, 'Native 3D geometry has not been exported for this bundle')
        identity = (path.stat().st_mtime_ns, path.stat().st_size)
        if verified_scenes.get(name) != identity:
            data = read_json(path)
            if data.get('model_sha256') != model_hashes[name]:
                raise HTTPException(409, '3D export belongs to a different replay model; export it again')
            verified_scenes[name] = identity
        return FileResponse(path, media_type='application/json')

    @app.get('/api/replays')
    def catalog():
        return [{'id': key, 'baseline': r.model['baseline'], 'scene': r.model['scene'],
                 'duration_s': r.end-r.start, 'frames': len(r.frames),
                 'graph_scope': r.model['graph']['scope'], 'limitations': r.model['limitations'],
                 'scene_available': scene_paths[key].is_file(),
                 'video_available': video_ready(key)}
                for key, r in replays.items()]

    @app.get('/api/state/{name}')
    def state(name: str, t: float = 0):
        r = selected(name)
        s = r.state(r.start+t)
        return {'index': s['index'], 'time_s': s['frame']['time_s']-r.start,
                'camera_pose': s['frame']['pose'],
                'camera_hfov_deg': s['frame']['hfov_deg'],
                'camera_aspect': s['frame']['width'] / s['frame']['height'],
                'camera_trail': r.positions[:s['index']+1].tolist(),
                'frame_count': len(r.frames), 'action_counts': s['action_counts'],
                'active_objects': s['active'], 'completed_scans': len(s['scans']),
                'metrics': s['metrics'], 'final_metrics': r.final_metrics,
                'final_evaluation': r.metric_report, 'graph_scope': s['graph']['scope'],
                'nodes': len(s['graph']['nodes']), 'edges': len(s['graph']['edges']),
                'provenance': r.model['provenance']}

    @app.get('/api/panel/{name}.jpg')
    def panel(name: str, t: float = 0, object_id: str | None = None):
        r = selected(name)
        out = io.BytesIO()
        r.render(r.start+t, object_id=object_id).save(out, format='JPEG', quality=88)
        return Response(out.getvalue(), media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    @app.get('/api/feed/{name}.jpg')
    def feed(name: str, t: float = 0, object_id: str | None = None):
        r = selected(name)
        state = r.state(r.start+t)
        if object_id is not None and not any(n['id'] == object_id and n['type'] == 'object'
                                             for n in state['graph']['nodes']):
            raise HTTPException(404, 'Object is absent from this graph snapshot')
        out = io.BytesIO()
        r.feed(state, object_id)[0].save(out, format='JPEG', quality=90)
        return Response(out.getvalue(), media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    @app.get('/api/graph/{name}')
    def graph(name: str, t: float = 0):
        """Cytoscape elements matching the existing dashboard's graph interface."""
        r = selected(name)
        native = r.state(r.start+t)['graph']
        nodes = []
        for n in native['nodes']:
            a = {k: v for k, v in n.items() if k != 'corners'}
            if n.get('position') is not None:
                x, y, z = n['position']
                a['position'] = [x, -z, y]  # dashboard map frame (Z up)
            nodes.append({'data': a})
        return {'elements': {'nodes': nodes, 'edges': [
            {'data': dict(e, id=f'edge-{i}')} for i, e in enumerate(native['edges'])]},
            'version': name + ':' + str(native.get('time_s', 'final')),
            'scope': native['scope'], 'coordinate_frame': 'dashboard Z-up'}

    @app.get('/api/video/{name}')
    def video(name: str):
        selected(name)
        if not video_ready(name):
            raise HTTPException(404, 'Video has not been exported')
        return FileResponse(video_root/(name+'.mp4'), media_type='video/mp4', filename=name+'.mp4')

    return app


def main():
    import uvicorn
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('models', nargs='+')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8097)
    p.add_argument('--video-root')
    args = p.parse_args()
    uvicorn.run(create_app(args.models, args.video_root), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
