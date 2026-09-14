"""Run native HOV-SG on a shared scheduled acquisition, without regenerating poses."""
import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .timing import phase, save


def select_hov_frames(frames, stride):
    """Uniform sampling only; GT and action outcomes never select baseline input."""
    if stride < 1:
        raise ValueError('HOV sampling stride must be positive')
    selected = set(range(0, len(frames), stride))
    reasons = {index: {'uniform_stride'} for index in selected}
    indices = sorted(selected)
    return indices, [{'frame_index': index, 'reasons': sorted(reasons[index])}
                     for index in indices]


def prepare_sampled_recording(recording, output, stride):
    recording, output = Path(recording), Path(output)
    frames = [json.loads(line) for line in (recording / 'frames.jsonl').read_text().splitlines()
              if line]
    indices, selections = select_hov_frames(frames, stride)
    view = output / 'sampled_input'
    view.mkdir()
    for folder in ('rgb', 'depth', 'pose'):
        (view / folder).mkdir()
    sampled_rows = []
    for native_index, source_index in enumerate(indices):
        source = frames[source_index]
        original_index = int(source.get('source_frame_index', source_index))
        stem = f'{native_index:06d}'
        row = dict(source, index=native_index, stem=stem,
                   source_frame_index=original_index,
                   source_stem=source.get('source_stem', source['stem']))
        sampled_rows.append(row)
        for folder, suffix in (('rgb', '.png'), ('depth', '.png'), ('pose', '.txt')):
            target = view / folder / (stem + suffix)
            target.symlink_to((recording / folder / (source['stem'] + suffix)).resolve())
    (view / 'frames.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in sampled_rows))
    acquisition = json.loads((recording / 'acquisition.json').read_text())
    acquisition.update(frames=len(indices), source_recording=str(recording),
                       source_frames=len(frames), sampling_stride=stride,
                       sampling_strategy='uniform GT-independent frame stride')
    (view / 'acquisition.json').write_text(json.dumps(acquisition, indent=2) + '\n')
    source_hash = hashlib.sha256((recording / 'frames.jsonl').read_bytes()).hexdigest()
    receipt = {'schema': 'graphapi.hovsg_sampled_input.v1', 'complete': True,
        'source_recording': str(recording), 'source_frames': len(frames),
        'source_frames_sha256': source_hash, 'uniform_stride': stride,
        'sampled_frames': len(indices),
        'sampled_source_frame_indices': indices, 'selections': selections,
        'method': ('Uniform temporal coverage only. GT visibility, object actions, and '
                   'evaluation outcomes do not select HOV-SG input frames.')}
    (output / 'sampling_manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return view, indices, receipt


def save_evaluation_assets(graph, output, source_indices=None):
    """Persist the native embedding spaces needed by the external evaluator.

    This calls HOV-SG's existing label encoders after graph construction.  It
    does not change feature fusion, clustering, hierarchy construction, or the
    native graph files.
    """
    import numpy as np
    from hovsg.utils.constants import HM3DSEM_ROOM_CATEGORIES
    from hovsg.utils.label_feats import get_label_feats
    from hovsg.utils.clip_utils import get_text_feats_multiple_templates

    target = Path(output) / 'evaluation_assets'
    target.mkdir()
    object_features, object_names = get_label_feats(
        graph.clip_model, graph.clip_feat_dim, 'HM3DSEM_LABELS', str(target))
    room_names = list(HM3DSEM_ROOM_CATEGORIES)
    room_features = get_text_feats_multiple_templates(
        room_names, graph.clip_model, graph.clip_feat_dim)
    np.save(target / 'object_category_features.npy', np.asarray(object_features))
    np.save(target / 'room_category_features.npy', np.asarray(room_features))
    predictions = []
    for room in graph.rooms:
        embeddings = np.asarray(room.embeddings, dtype=float)
        if embeddings.ndim != 2 or not len(embeddings):
            predicted = None
            votes = []
        else:
            choices = np.argmax(embeddings @ np.asarray(room_features).T, axis=1)
            values, counts = np.unique(choices, return_counts=True)
            selected = int(values[int(np.argmax(counts))])
            predicted = room_names[selected]
            votes = [room_names[int(index)] for index in choices]
        representatives = [int(value) for value in room.represent_images]
        if source_indices is not None:
            if any(index < 0 or index >= len(source_indices) for index in representatives):
                raise RuntimeError('HOV room representative image index is out of range')
            representatives = [source_indices[index] for index in representatives]
        predictions.append({'room_id': str(room.room_id),
            'predicted_label': predicted, 'view_votes': votes,
            'represent_images': representatives})
    metadata = {
        'schema': 'graphapi.hovsg_evaluation_assets.v1',
        'embedding_model': str(graph.cfg.models.clip.type),
        'embedding_dimension': int(graph.clip_feat_dim),
        'object_categories': [str(value) for value in object_names],
        'room_categories': room_names,
        'room_predictions': predictions,
        'scope': ('Post-construction evaluation assets from HOV-SG native CLIP and '
                  'the native view-embedding room vote; graph construction is unchanged.'),
    }
    (target / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def verify_evaluation_outputs(recording, output, skip_frames, observations, assets):
    """Fail a measurement run when an adapter output is incomplete or misaligned."""
    recording, output = Path(recording), Path(output)
    frames = [json.loads(line) for line in
              (recording / 'frames.jsonl').read_text().splitlines() if line]
    expected_native = list(range(0, len(frames), skip_frames))
    expected = [int(frames[index].get('source_frame_index', index))
                for index in expected_native]
    readiness = {'expected_sampled_frame_indices': expected,
                 'native_observations': False, 'semantic_assets': False}
    if observations:
        metadata = json.loads((output / 'native_observations/observer.json').read_text())
        rows = [json.loads(line) for line in
                (output / 'native_observations/observations.jsonl').read_text().splitlines()
                if line]
        sam = [row for row in rows if row['stage'] == 'sam_clip']
        masks3d = [row for row in rows if row['stage'] == 'masks_3d']
        if ([row['frame_index'] for row in sam] != expected or
                [row['frame_index'] for row in masks3d] != expected or
                [row.get('native_frame_index') for row in sam] != expected_native or
                [row.get('native_frame_index') for row in masks3d] != expected_native):
            raise RuntimeError('HOV native observation indices do not match sampled inputs')
        if metadata.get('schema') != 'graphapi.hovsg_native_observer.v2':
            raise RuntimeError('HOV observer did not save the temporal-evaluation schema')
        for row in sam:
            path = output / f'native_observations/{row["frame_index"]:06d}.masks.npz'
            with np.load(path) as archive:
                required = {'packed', 'shape', 'mask_features', 'global_feature'}
                if not required <= set(archive.files):
                    raise RuntimeError(f'HOV mask archive is incomplete: {path}')
                features = archive['mask_features']
                if features.ndim != 2 or len(features) != len(row['masks']):
                    raise RuntimeError(f'HOV mask features are misaligned: {path}')
        readiness['native_observations'] = True
    if assets is not None:
        metadata = json.loads((output / 'evaluation_assets/manifest.json').read_text())
        object_features = np.load(output / 'evaluation_assets/object_category_features.npy')
        room_features = np.load(output / 'evaluation_assets/room_category_features.npy')
        if (object_features.ndim != 2 or room_features.ndim != 2
                or len(object_features) != len(metadata['object_categories'])
                or len(room_features) != len(metadata['room_categories'])
                or object_features.shape[1] != metadata['embedding_dimension']
                or room_features.shape[1] != metadata['embedding_dimension']):
            raise RuntimeError('HOV semantic evaluation assets are inconsistent')
        readiness['semantic_assets'] = True
    readiness['ready_for_end_to_end_evaluation'] = (
        readiness['native_observations'] and readiness['semantic_assets'])
    return readiness


def run(baseline_root, recording, output, skip_frames=10, build_graph=True, record_observations=False):
    root, recording, output = map(lambda x: Path(x).resolve(), (baseline_root, recording, output))
    metadata = json.loads((recording / 'acquisition.json').read_text())
    if not metadata.get('complete') or metadata.get('frames', 0) < 1:
        raise ValueError('Acquisition is incomplete or empty')
    output.mkdir(parents=True, exist_ok=False)
    native_recording, sampled_indices, sampling = prepare_sampled_recording(
        recording, output, skip_frames)
    sys.path.insert(0, str(root))
    from hovsg.graph.graph import Graph
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(root / 'config/create_graph.yaml')
    cfg.main.dataset_path = str(native_recording)
    cfg.main.save_path = str(output)
    cfg.main.package_path = str(root / 'hovsg')
    cfg.main.scene_id = Path(metadata['scene']).parent.name
    cfg.models.clip.checkpoint = str(root / cfg.models.clip.checkpoint)
    cfg.models.sam.checkpoint = str(root / cfg.models.sam.checkpoint)
    for path in (cfg.models.clip.checkpoint, cfg.models.sam.checkpoint):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    cfg.pipeline.skip_frames = 1
    cfg.pipeline.create_graph = build_graph
    OmegaConf.save(cfg, output / 'resolved_config.yaml')
    phases = {}
    started = time.perf_counter()
    with phase(phases, 'model_initialization'):
        graph = Graph(cfg)
    with phase(phases, 'feature_map'):
        from .hovsg_observer import NativeObserver
        observer = NativeObserver(graph, native_recording, output) if record_observations else nullcontext()
        with observer:
            graph.create_feature_map()
    if len(graph.full_pcd.points) == 0 or len(graph.mask_pcds) == 0:
        raise RuntimeError('HOV-SG produced an empty feature map or no object masks')
    graph.save_masked_pcds(path=str(output), state='both')
    graph.save_full_pcd(path=str(output))
    graph.save_full_pcd_feats(path=str(output))
    if build_graph:
        with phase(phases, 'hierarchy'):
            graph.build_graph(save_path=str(output))
    phases['run_wall_time'] = time.perf_counter() - started
    evaluation_assets = None
    if build_graph:
        with phase(phases, 'evaluation_asset_export'):
            evaluation_assets = save_evaluation_assets(graph, output, sampled_indices)
    readiness = verify_evaluation_outputs(
        native_recording, output, 1, record_observations, evaluation_assets)
    readiness['sampling_manifest'] = 'sampling_manifest.json'
    readiness['sampled_source_frame_indices'] = sampled_indices
    save(output, phases, 'Native model initialization through feature mapping, saved point clouds/features and optional hierarchy; excludes adapter setup and validation')
    result = {'complete': True, 'container_image_id': os.environ.get('BASELINE_IMAGE_ID'), 'baseline': 'HOV-SG', 'recording': str(recording),
              'input_frames': metadata['frames'], 'skip_frames': skip_frames,
              'sampled_frames': sampling['sampled_frames'],
              'sampling_strategy': sampling['method'],
              'points': len(graph.full_pcd.points), 'object_masks': len(graph.mask_pcds),
              'hierarchy_requested': build_graph, 'source': str(root),
              'evaluation_assets': ('evaluation_assets/manifest.json'
                                    if evaluation_assets is not None else None),
              'native_observations_recorded': bool(record_observations),
              'evaluation_readiness': readiness}
    (output / 'baseline_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root', required=True)
    p.add_argument('--recording', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--skip-frames', type=int, default=10)
    p.add_argument('--feature-map-only', action='store_true')
    p.add_argument('--record-native-observations', action='store_true')
    a = p.parse_args(argv)
    if a.skip_frames < 1:
        p.error('--skip-frames must be positive')
    run(a.baseline_root, a.recording, a.output, a.skip_frames, not a.feature_map_only,
        a.record_native_observations)


if __name__ == '__main__':
    main()
