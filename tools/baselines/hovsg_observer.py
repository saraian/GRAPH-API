"""Read-only Python profiler recording native HOV outputs at function returns.

No native function is replaced, and arguments, return values and model state are
never modified. Frame-local SAM mask IDs are not persistent graph object IDs.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np


class NativeObserver:
    def __init__(self, graph, recording, output):
        self.output = Path(output) / 'native_observations'
        self.output.mkdir()
        self.rows = [json.loads(line) for line in
            (Path(recording) / 'frames.jsonl').read_text().splitlines()]
        extract = graph.create_feature_map.__func__.__globals__['extract_feats_per_pixel']
        self.codes = {extract.__code__: 'sam_clip',
            graph.dataset.create_3d_masks.__func__.__code__: 'masks_3d'}
        self.starts = {}
        self.counts = {'sam_clip': 0, 'masks_3d': 0}
        self.write_time = 0.0

    def __enter__(self):
        if sys.getprofile() is not None:
            raise RuntimeError('A profiler is already installed; refusing to replace it')
        sys.setprofile(self._profile)
        return self

    def _profile(self, frame, event, value):
        kind = self.codes.get(frame.f_code)
        if kind is None:
            return
        if event == 'call':
            self.starts[id(frame)] = time.perf_counter()
        elif event == 'return' and value is not None:
            finished = time.perf_counter()
            elapsed = finished - self.starts.pop(id(frame))
            native_index = int(frame.f_back.f_locals['i'])
            source_index = int(self.rows[native_index].get('source_frame_index', native_index))
            row = {'frame_index': source_index, 'native_frame_index': native_index,
                'source_time_s': self.rows[native_index]['time_s'],
                'stage': kind, 'stage_wall_time_s': elapsed}
            if kind == 'sam_clip':
                masks = value[2]
                features = np.asarray(value[1])
                global_feature = np.asarray(value[3])
                if features.ndim != 2 or len(features) != len(masks):
                    raise RuntimeError('Native HOV mask features do not align with SAM masks')
                row['masks'] = [{k: np.asarray(v).tolist() for k, v in mask.items()
                    if k != 'segmentation'} for mask in masks]
                pixels = np.asarray([mask['segmentation'] for mask in masks], dtype=bool)
                np.savez_compressed(self.output / f'{source_index:06d}.masks.npz',
                    packed=np.packbits(pixels, axis=-1), shape=np.asarray(pixels.shape),
                    mask_features=features.astype(np.float16),
                    global_feature=global_feature.astype(np.float16))
            else:
                row['masks'] = []
                for mask_index, cloud in enumerate(value):
                    points = np.asarray(cloud.points)
                    row['masks'].append({'mask_index': mask_index, 'points': len(points),
                        'aabb_min': points.min(axis=0).tolist() if len(points) else None,
                        'aabb_max': points.max(axis=0).tolist() if len(points) else None})
            with (self.output / 'observations.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            self.counts[kind] += 1
            self.write_time += time.perf_counter() - finished

    def __exit__(self, kind, value, traceback):
        sys.setprofile(None)
        (self.output / 'observer.json').write_text(json.dumps({
            'schema': 'graphapi.hovsg_native_observer.v2',
            'counts': self.counts, 'observer_serialization_wall_time_s': self.write_time,
            'scope': 'Native per-frame SAM masks, aligned CLIP mask features and projected 3D masks before batch merging; frame-local IDs; no online graph or persistent tracks',
            'mask_archive': 'Each <recording-index>.masks.npz stores packed masks, exact shape, float16 mask_features and float16 global_feature',
            'timing': 'Measured native function call-to-return wall time with read-only profiling enabled; excludes this observer serialization, includes profiling dispatch overhead',
            'native_functions_replaced': False, 'native_state_modified': False,
            'native_execution_failed': kind is not None}, indent=2))
