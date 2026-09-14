"""Run Clio's native ROS pipeline using a bag made from shared tour observations."""
import argparse
import gzip
import json
import math
import os
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from .clio_bag import export, validate_existing_bag
from .timing import save


ROSLAUNCH_SIGINT_TIMEOUT_S = 600
ROSLAUNCH_SIGTERM_TIMEOUT_S = 120
MIN_SEMANTIC_OUTPUT_FRACTION = .85  # owner ruling 2026-09-14; see .handoff/lanes/FOUND-DGX-Clio/CLIO_SEMANTIC_OUTPUT_GATE.md


def roslaunch_command(launch):
    """Give native Clio enough time to serialize a large graph on shutdown."""
    return ['roslaunch', '--sigint-timeout', str(ROSLAUNCH_SIGINT_TIMEOUT_S),
            '--sigterm-timeout', str(ROSLAUNCH_SIGTERM_TIMEOUT_S), str(launch)]


def prepare(root, recording, output, models, clip, tasks, segmentation_confidence=None,
            semantic_mapping_only=False):
    root, recording, output = map(Path, (root, recording, output))
    row = json.loads((recording / 'frames.jsonl').read_text().splitlines()[0])
    pipeline = yaml.safe_load((root / 'clio_ros/config/realsense/pipeline.yaml').read_text())
    sensor = pipeline['input']['inputs']['camera']['sensor']
    focal = row['width'] / (2 * math.tan(math.radians(row['hfov_deg']) / 2))
    sensor.update(width=row['width'], height=row['height'], fx=focal, fy=focal,
                  cx=row['width']/2, cy=row['height']/2)
    pipeline_path = output / 'pipeline.yaml'
    pipeline_path.write_text(yaml.safe_dump(pipeline))
    config = yaml.safe_load((root / 'clio_ros/config/segmentation/small_clip.yaml').read_text())
    config['segmentation']['model_name'] = str(Path(models) / 'FastSAM-x.pt')
    config['clip_model']['model_name'] = str(Path(clip) / 'ViT-B-32.pt')
    if segmentation_confidence is not None:
        if not 0 < segmentation_confidence < 1:
            raise ValueError('Segmentation confidence must be between zero and one')
        config['segmentation']['confidence'] = segmentation_confidence
    for path in (config['segmentation']['model_name'], config['clip_model']['model_name']):
        if not Path(path).is_file():
            raise FileNotFoundError(f'Cache the model before the run: {path}')
    config_path = output / 'segmentation.yaml'
    config_path.write_text(yaml.safe_dump(config))
    # Native task_server's list path reads a literal '~{prefix}' key. Its
    # documented file path works; use it without forking the native task server.
    object_file, place_file = output / 'object_tasks.yaml', output / 'place_tasks.yaml'
    object_tasks = {} if semantic_mapping_only else {task: {} for task in tasks}
    place_tasks = {} if semantic_mapping_only else {'room': {}}
    object_file.write_text(yaml.safe_dump(object_tasks, sort_keys=False))
    place_file.write_text(yaml.safe_dump(place_tasks, sort_keys=False))
    launch = ET.Element('launch')
    include = ET.SubElement(launch, 'include', pass_all_args='true', file=str(root / 'clio_ros/launch/realsense.launch'))
    args = {'start_rviz': 'false', 'start_visualizer': 'false',
            'map_frame': 'world', 'odom_frame': 'world', 'robot_frame': 'camera_optical',
            'sensor_frame': 'camera_optical', 'config_path': str(pipeline_path),
            'segmenter_config': str(config_path), 'log_path': str(output / 'graph'),
            'object_tasks_file': str(object_file), 'place_tasks_file': str(place_file), 'exit_after_clock': 'false'}
    for name, value in args.items():
        ET.SubElement(include, 'arg', name=name, value=value)
    path = output / 'scheduled.launch'
    ET.ElementTree(launch).write(path, encoding='unicode')
    return path


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    launch = prepare(args.baseline_root, args.recording, output, args.models,
                     args.clip_cache, args.tasks, args.segmentation_confidence,
                     args.semantic_mapping_only)
    if args.input_bag:
        bag = Path(args.input_bag).resolve(strict=True)
        validation = validate_existing_bag(args.recording, bag)
        (output / 'input_bag_reference.json').write_text(json.dumps(validation, indent=2))
        counts = validation['topic_counts']
    else:
        bag = output / 'input.bag'
        counts = export(args.recording, bag)
    env = dict(os.environ, PYTHONUNBUFFERED='1', ROS_LOG_DIR=str(output / 'ros-logs'))
    started = time.perf_counter()
    with (output / 'roslaunch.log').open('w') as log:
        proc = subprocess.Popen(roslaunch_command(launch), stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        observer = None
        try:
            import rosgraph
            master = rosgraph.Master('/baseline_verification')
            deadline = time.monotonic() + 120
            ready = False
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f'Clio exited during startup ({proc.returncode}); see {log.name}')
                # rosnode/rosgraph transport failures before roscore starts are expected.
                import socket
                try:
                    pubs, subs, _ = master.getSystemState()
                except (socket.error, ConnectionRefusedError):
                    time.sleep(0.25)
                    continue
                topics = {topic for topic, nodes in pubs if nodes}
                rgb_subs = dict(subs).get('/dominic/forward/color/image_raw', [])
                if ({'/task_server/objects', '/task_server/places'} <= topics and
                        '/semantic_inference' in rgb_subs):
                    ready = True
                    break
                time.sleep(0.25)
            if not ready:
                raise RuntimeError('Clio did not advertise its feature inputs and subscribe to RGB within 120 seconds')
            import rospy
            from semantic_inference_msgs.msg import FeatureVectors
            rospy.init_node('baseline_verification', disable_signals=True)
            embedded = rospy.wait_for_message('/task_server/objects', FeatureVectors, timeout=120)
            places = rospy.wait_for_message('/task_server/places', FeatureVectors, timeout=120)
            expected_places = [] if args.semantic_mapping_only else ['room']
            if (list(embedded.names) != args.tasks or
                    len(embedded.features) != len(args.tasks) or
                    list(places.names) != expected_places or
                    len(places.features) != len(expected_places)):
                raise RuntimeError('Native task server returned incorrect object/place feature inputs')
            (output / 'task_embeddings.json').write_text(json.dumps({
                'mode': ('task_free_semantic_mapping' if args.semantic_mapping_only
                         else 'task_conditioned'),
                'object_names': list(embedded.names),
                'object_feature_vectors': len(embedded.features),
                'place_names': list(places.names),
                'place_feature_vectors': len(places.features)}, indent=2))
            from .clio_observer import NativeObserver
            observer = NativeObserver(output, master, relay_rgb=True)
            from .clio_rgb_relay import REMAP
            with (output / 'playback.log').open('w') as playback:
                player = subprocess.Popen(['rosbag', 'play', '--clock', '--delay=3', '-r', str(args.rate), str(bag), REMAP],
                                          stdout=playback, stderr=subprocess.STDOUT, env=env)
                try:
                    while player.poll() is None:
                        observer.discover(master)
                        if proc.poll() is not None:
                            raise RuntimeError('Clio exited during bag playback')
                        time.sleep(0.5)
                    if player.returncode:
                        raise subprocess.CalledProcessError(player.returncode, player.args)
                finally:
                    if player.poll() is None:
                        player.terminate()
                        player.wait(timeout=15)
            # Let the native asynchronous reconstruction/backend drain before graceful shutdown.
            expected = max(counts.values())
            drain_deadline = time.monotonic() + args.drain_seconds
            while (time.monotonic() < drain_deadline and
                   observer.counts.get('/dominic/forward/semantic/image_raw', 0) <
                   MIN_SEMANTIC_OUTPUT_FRACTION * expected):
                observer.discover(master)
                time.sleep(.5)
            if proc.poll() is not None:
                raise RuntimeError(f'Clio exited before playback/drain completed; see {log.name}')
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=ROSLAUNCH_SIGINT_TIMEOUT_S +
                              ROSLAUNCH_SIGTERM_TIMEOUT_S + 60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=30)
                    raise RuntimeError('Clio did not finish native graph serialization during shutdown')
            if observer is not None:
                observer.close()
    elapsed = time.perf_counter() - started
    result = verify_output(args, counts)
    save(output, {'run_wall_time': elapsed}, 'Native ROS launch through startup, bag playback, configured drain and shutdown/save; excludes bag conversion and validation; playback rate and drain affect this duration')
    return result


def verify_output(args, counts):
    """Recheck saved native results without rerunning a completed pipeline."""
    output = Path(args.output).resolve()
    graph_path = output / 'graph/backend/dsg.json'
    backend = json.loads(graph_path.read_text())
    nodes = backend.get('nodes', [])
    objects = [n for n in nodes if isinstance(n.get('id'), int) and
               chr(n['id'] >> 56) == 'O']
    segments = [n for n in nodes if isinstance(n.get('id'), int) and
                chr(n['id'] >> 56) == 's']
    if args.semantic_mapping_only:
        if objects:
            raise RuntimeError('Task-free Clio unexpectedly produced task-clustered object nodes')
        if not segments:
            raise RuntimeError('Task-free Clio produced no semantic segment primitives')
        semantic = [n for n in segments
                    if n.get('attributes', {}).get('semantic_feature', {}).get('data')]
        boxed = [n for n in segments if
                 len(n.get('attributes', {}).get('bounding_box', {}).get('dimensions', [])) == 3 and
                 all(float(x) > 0 for x in
                     n['attributes']['bounding_box']['dimensions'])]
        meshed = [n for n in segments
                  if n.get('attributes', {}).get('mesh', {}).get('points')]
        if min(len(semantic), len(boxed), len(meshed)) != len(segments):
            raise RuntimeError('Clio semantic primitives are missing a feature, 3-D box, or mesh')
    elif not objects:
        raise RuntimeError('Clio produced no task-clustered object nodes')
    graph_root = output / 'graph'
    mesh_path = graph_root / 'backend/mesh.ply'
    if not mesh_path.is_file() or mesh_path.stat().st_size <= 100:
        raise RuntimeError('Clio saved no nonempty reconstructed mesh')
    graphs = list(graph_root.glob('*/dsg.json'))
    nonempty = []
    for path in graphs:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and data.get('nodes'):
            nonempty.append({'path': str(path), 'nodes': len(data['nodes'])})
    if not nonempty:
        raise RuntimeError('Clio saved no nonempty scene graph; inspect roslaunch.log')
    observer_path = output / 'native_observer.json'
    observer = json.loads(observer_path.read_text())
    semantic_messages = sum(value for topic, value in observer.get('messages', {}).items()
                            if 'semantic' in topic)
    semantic_images = observer.get('messages', {}).get(
        '/dominic/forward/semantic/image_raw', 0)
    expected_frames = max(counts.values())
    semantic_fraction = semantic_images / expected_frames if expected_frames else 0
    if (observer.get('error') or
            observer.get('input_frames_observed') != expected_frames or
            observer.get('rgb_frames_relayed') != expected_frames or
            semantic_fraction < MIN_SEMANTIC_OUTPUT_FRACTION or semantic_messages < 1):
        raise RuntimeError('Clio observer did not capture complete semantic inference output')
    graph_snapshots = 0
    semantic_history_ids = set()
    semantic_history_boxes = 0
    with gzip.open(output / 'native_graph_history.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            graph_snapshots += 1
            for node in row.get('nodes', []):
                if node.get('type') != 'semantic_primitive':
                    continue
                semantic_history_ids.add(node['id'])
                semantic_history_boxes += 'corners' in node
                if ('native_last_observed_ns' not in node or
                        'native_is_active' not in node):
                    raise RuntimeError('Clio semantic history lacks activity provenance')
    if graph_snapshots != observer.get('graph_snapshots'):
        raise RuntimeError('Clio native graph history count is incomplete')
    if args.semantic_mapping_only and (not semantic_history_ids or not semantic_history_boxes):
        raise RuntimeError('Clio semantic history lacks primitive boxes')
    mode = ('task_free_semantic_mapping' if args.semantic_mapping_only
            else 'task_conditioned')
    result = {'complete': True, 'container_image_id': os.environ.get('BASELINE_IMAGE_ID'), 'baseline': 'Clio', 'recording': str(Path(args.recording).resolve()),
              'mode': mode, 'bag_messages': counts, 'graphs': nonempty,
              'tasks': args.tasks, 'semantic_primitives': len(segments),
              'semantic_primitives_with_features': len(semantic) if args.semantic_mapping_only else None,
              'semantic_primitives_with_boxes': len(boxed) if args.semantic_mapping_only else None,
              'semantic_primitives_with_meshes': len(meshed) if args.semantic_mapping_only else None,
              'semantic_output_messages': semantic_messages,
              'semantic_image_output_fraction': semantic_fraction,
              'reconstructed_mesh_bytes': mesh_path.stat().st_size,
              'playback_rate': args.rate, 'source': args.baseline_root, 'task_clustered_objects': len(objects),
              'segmentation_confidence': yaml.safe_load((output / 'segmentation.yaml').read_text())['segmentation']['confidence'],
              'evaluation_readiness': {
                  'native_graph_snapshots': graph_snapshots,
                  'semantic_history_unique_primitives': len(semantic_history_ids),
                  'semantic_history_box_records': semantic_history_boxes,
                  'ready_for_temporal_object_action_evaluation': bool(
                      semantic_history_ids and semantic_history_boxes)}}
    (output / 'baseline_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-root', 'recording', 'output', 'models', 'clip-cache'):
        p.add_argument('--' + name, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--semantic-mapping-only', action='store_true', default=True,
                      help='Run open-set semantic inference and persist segment primitives without task prompts (default)')
    mode.add_argument('--task-conditioned', action='store_false', dest='semantic_mapping_only',
                      help='Enable Clio task clustering; requires explicit --tasks')
    p.add_argument('--tasks', nargs='+', default=[])
    p.add_argument('--rate', type=float, default=0.5)
    p.add_argument('--segmentation-confidence', type=float, help='Native FastSAM confidence configuration; leaves native default unless specified')
    p.add_argument('--input-bag', help='Reuse an existing ROS bag only after verifying RGB/depth against every original frame')
    p.add_argument('--drain-seconds', type=float, default=20)
    args = p.parse_args(argv)
    if args.segmentation_confidence is not None and not 0 < args.segmentation_confidence < 1:
        p.error('segmentation-confidence must be between zero and one')
    if len(set(args.tasks)) != len(args.tasks):
        p.error('tasks must be distinct')
    if args.semantic_mapping_only and args.tasks:
        p.error('--semantic-mapping-only does not accept --tasks')
    if not args.semantic_mapping_only and not args.tasks:
        p.error('--task-conditioned requires explicit --tasks')
    if args.rate <= 0 or args.drain_seconds < 0:
        p.error('rate must be positive and drain-seconds nonnegative')
    run(args)


if __name__ == '__main__':
    main()
