"""Shared scheduled Habitat acquisition for baseline adapters.

Import the selected GRAPH-API checkout at runtime. Never copy the tour, compiler,
script executor or rigid-object implementation into a baseline repository.
One thread owns Habitat; ScriptRunner callbacks dispatch operations to that thread.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue

import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_paths(root):
    root = Path(root).resolve(strict=True)
    return root / 'lost3dsg/test', root / 'lost3dsg/src/perception_module'


def source_module(root, name):
    """Import current source, rejecting accidental resolution to another checkout."""
    test, src = source_paths(root)
    expected = (test if name in ('habitat_feed_host', 'schedule_batch') else src) / (name + '.py')
    if not expected.is_file():
        raise FileNotFoundError(expected)
    for directory in (test, src):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != expected.resolve():
        raise RuntimeError(f'{name} resolved to {module.__file__}, expected {expected}')
    return module


def provenance(root, names):
    test, src = source_paths(root)
    result = {}
    for name in names:
        path = (test if name in ('habitat_feed_host', 'schedule_batch', 'voronoi_roadmap') else src) / (name + '.py')
        result[name] = {'path': str(path), 'sha256': digest(path)}
    return result


def matching_event(event, trigger):
    return (event.get('stop') == trigger['stop'] and
            ('lap' not in trigger or event.get('lap') == trigger['lap']))


def validate_triggers(runner_class, script, schedule, laps):
    """Use the real executor's parser; reject impossible or backwards triggers."""
    stops = [p.get('stop') for p in schedule['trajectory'] if p.get('scan_deg', 0) > 0]
    events = [{'stop': stop, 'lap': lap} for lap in range(laps) for stop in stops]
    cursor = 0
    for index, step in enumerate(script['steps']):
        trigger = runner_class._waypoint_trigger(step, index)
        if trigger is None:
            continue
        candidates = [i for i in range(cursor, len(events)) if matching_event(events[i], trigger)]
        if not candidates:
            raise ValueError(f'Step {index}: unreachable or out-of-order waypoint {trigger}')
        cursor = candidates[0]  # Several consecutive actions may share one event.


class Recording:
    """HOV-native files plus lossless metre depths for Clio, one shared frame order."""
    def __init__(self, output, fps, hfov, compress_depth=False, record_gt=False,
                 shuffle_depth=False, dynamic_semantic_id_offset=0):
        self.output = Path(output)
        self.fps, self.hfov = fps, hfov
        self.index = 0
        self.started_at = time.monotonic()
        self.rows = []
        self.compress_depth, self.record_gt = compress_depth, record_gt
        self.shuffle_depth = shuffle_depth
        self.dynamic_semantic_id_offset = dynamic_semantic_id_offset
        self.controller = None
        self.evaluation_ids = {}
        for name in ('rgb', 'depth', 'depth_m', 'pose'):
            (self.output / name).mkdir()
        if record_gt:
            (self.output / 'semantic').mkdir()

    def capture(self, sim, agent, reason, event=None):
        obs = sim.get_sensor_observations()
        rgb = np.asarray(obs['color_sensor'])[..., :3]
        depth = np.asarray(obs['depth_sensor'], dtype=np.float32)
        if not np.isfinite(depth).all() or (depth < 0).any() or (depth > 65.535).any():
            raise ValueError('Depth is outside the finite 0..65.535 m HOV PNG range')
        state = agent.get_state().sensor_states['color_sensor']
        import quaternion
        pose = np.eye(4)
        pose[:3, :3] = quaternion.as_rotation_matrix(state.rotation)
        pose[:3, 3] = state.position
        stem = f'{self.index:06d}'
        Image.fromarray(rgb).save(self.output / 'rgb' / (stem + '.png'))
        Image.fromarray(np.rint(depth * 1000).astype(np.uint16)).save(self.output / 'depth' / (stem + '.png'))
        if self.shuffle_depth:
            from .depth_codec import write_shuffled
            write_shuffled(self.output / 'depth_m' / (stem + '.npz'), depth)
        elif self.compress_depth:
            np.savez_compressed(self.output / 'depth_m' / (stem + '.npz'), depth=depth)
        else:
            np.save(self.output / 'depth_m' / (stem + '.npy'), depth)
        # HOV's reader reads exactly one line, then applies the OpenGL->optical flip.
        np.savetxt(self.output / 'pose' / (stem + '.txt'), pose.reshape(1, 16), fmt='%.12g')
        h, w = depth.shape
        row = {'index': self.index, 'time_s': time.monotonic() - self.started_at + 1.0 / self.fps,
               'stem': stem, 'reason': reason, 'event': event,
               'width': w, 'height': h, 'hfov_deg': self.hfov}
        base = agent.get_state()
        row['base_position'] = np.asarray(base.position).tolist()
        row['base_rotation_wxyz'] = quaternion.as_float_array(base.rotation).tolist()
        if self.record_gt:
            semantic = np.asarray(obs['semantic_sensor'], dtype=np.uint32)
            np.savez_compressed(self.output / 'semantic' / (stem + '.npz'), semantic=semantic)
            row['visible_semantic_ids'] = np.unique(semantic).tolist()
            dynamic = []
            import itertools
            import magnum as mn
            for object_id, obj in self.controller.spawned_objects.items():
                bounds = obj.root_scene_node.cumulative_bb
                transform = obj.root_scene_node.absolute_transformation()
                corners = np.asarray([list(transform.transform_point(mn.Vector3(c)))
                    for c in itertools.product(*zip(bounds.min, bounds.max))])
                dynamic.append({'object_id': object_id, 'handle': obj.handle,
                    'evaluation_object_id': self.evaluation_ids.get(
                        int(object_id), str(object_id)),
                    'semantic_id': int(obj.semantic_id),
                    'position': list(obj.translation), 'bbox_corners_world': corners.tolist(),
                    'aabb_min': corners.min(axis=0).tolist(), 'aabb_max': corners.max(axis=0).tolist()})
            row['dynamic_ground_truth'] = dynamic
        with (self.output / 'frames.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        self.rows.append(row)
        self.index += 1
        return str(self.output / 'rgb' / (stem + '.png'))


class Session:
    def __init__(self, feed, sim, agent, tour, recording):
        self.feed, self.sim, self.agent = feed, sim, agent
        self.tour, self.recording = tour, recording
        self.controller = feed.DynamicObjectController(sim, agent)
        recording.controller = self.controller
        self.last_event = None
        self.responses = {}
        self.actions = []
        self.dynamic_instance_count = 0
        self.evaluation_ids = {}
        recording.evaluation_ids = self.evaluation_ids

    def tick(self):
        if self.tour.house_done:
            raise RuntimeError('Tour ended before the requested waypoint')
        self.tour.step(self.agent)
        self.sim.step_physics(1.0 / self.recording.fps)
        events = []
        while self.feed.CTRL.scan_events:
            self.last_event = self.feed.CTRL.scan_events.popleft()
            events.append(self.last_event)
        self.recording.capture(self.sim, self.agent, 'tour', events or None)

    def setup_dispatch(self):
        self.condition = threading.Condition()
        self.requests = Queue()
        self.closed = False
        self.active_request = None

    def wait_waypoint(self, trigger):
        with self.condition:
            while self.last_event is None or not matching_event(self.last_event, trigger):
                if self.closed or self.tour.house_done:
                    raise RuntimeError(f'Tour ended before waypoint {trigger}')
                self.condition.wait()
            return {'success': True, 'requested': dict(trigger), 'event': dict(self.last_event)}

    def _request(self, kind, payload):
        from concurrent.futures import Future
        future = Future()
        with self.condition:
            if self.closed:
                raise RuntimeError('Acquisition has stopped')
            self.requests.put((kind, payload, future))
            self.condition.notify_all()
        return future.result()

    def publish(self, action, payload):
        result = self._request('action', dict(payload, action=action))
        self.responses[payload['request_id']] = result

    def wait_result(self, action, timeout, request_id=None):
        result = self.responses.pop(request_id)
        if result['action'] != action:
            raise RuntimeError('Mismatched object response')
        return result

    def capture(self, action, index, result):
        return self._request('capture', action)

    def dispatch(self):
        """Execute queued operations on the simulation's owning thread."""
        while True:
            try:
                kind, payload, future = self.requests.get_nowait()
            except Empty:
                break
            self.active_request = future
            if kind == 'action':
                previous_evaluation_id = self.evaluation_ids.get(
                    int(payload['object_id'])) if payload.get('object_id') is not None else None
                result = self.controller.execute(payload)
                if result.get('success') and payload['action'] == 'spawn':
                    self.dynamic_instance_count += 1
                    evaluation_id = f'dynamic-{self.dynamic_instance_count:06d}'
                    self.evaluation_ids[int(result['object_id'])] = evaluation_id
                    result['evaluation_object_id'] = evaluation_id
                    if self.recording.dynamic_semantic_id_offset:
                        # Annotation-only label: every lifecycle gets a unique GT identity even
                        # when Habitat reuses its physical rigid-object ID after deletion.
                        obj = self.controller.spawned_objects[result['object_id']]
                        obj.semantic_id = (self.recording.dynamic_semantic_id_offset +
                                           self.dynamic_instance_count)
                        result['gt_semantic_id'] = int(obj.semantic_id)
                elif result.get('success') and previous_evaluation_id is not None:
                    result['evaluation_object_id'] = previous_evaluation_id
                    if payload['action'] == 'remove':
                        self.evaluation_ids.pop(int(payload['object_id']), None)
                entry = {'action': payload['action'], 'request': payload, 'result': result,
                         'after_frame': self.recording.index - 1, 'event': self.last_event}
                self.actions.append(entry)
                with (self.recording.output / 'object_actions.jsonl').open('a') as f:
                    f.write(json.dumps(entry) + '\n')
                future.set_result(result)
            else:
                # No event: this frame follows an object action, it does not complete a
                # scan. Passing last_event here wrote the trigger scan's scan_complete a
                # second time, and every reader that keys scans by (lap, stop) saw one
                # scan twice. object_actions.jsonl already records last_event per action.
                future.set_result(self.recording.capture(
                    self.sim, self.agent, f'object:{payload}'))

    def drive(self, runner, script):
        """Run canonical waits in a worker while the shared tour keeps advancing.

        This mirrors the ROS runner/feed split: script waits use wall time; only
        the simulator thread touches Habitat. Scan events and command results
        are delivered without polling timeouts or stale event replay.
        """
        self.setup_dispatch()
        executor = ThreadPoolExecutor(max_workers=1)
        result = executor.submit(runner.run, script)
        next_frame = time.monotonic()
        try:
            while not self.tour.house_done or not result.done():
                self.dispatch()
                if result.done():
                    outcome = result.result()  # Propagate executor failures immediately.
                    if not outcome.get('success') or not outcome.get('dataset_complete'):
                        return outcome
                now = time.monotonic()
                if now >= next_frame:
                    if not self.tour.house_done:
                        self.tick()
                    else:
                        self.sim.step_physics(1.0 / self.recording.fps)
                        self.recording.capture(self.sim, self.agent, 'script-after-tour')
                    with self.condition:
                        self.condition.notify_all()
                    next_frame = time.monotonic() + 1.0 / self.recording.fps
                with self.condition:
                    if self.requests.empty():
                        self.condition.wait(timeout=max(0, next_frame - time.monotonic()))
            self.dispatch()
            return result.result()
        finally:
            with self.condition:
                self.closed = True
                self.condition.notify_all()
            if self.active_request is not None and not self.active_request.done():
                self.active_request.set_exception(RuntimeError('Acquisition stopped'))
            # Resolve pending transports on failure so shutdown cannot deadlock.
            while not self.requests.empty():
                _, _, pending = self.requests.get_nowait()
                pending.set_exception(RuntimeError('Acquisition stopped'))
            executor.shutdown(wait=True)


def acquire(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Set configuration BEFORE importing modules that resolve it at import time.
    os.environ.update(HABITAT_SCENE=args.scene, HABITAT_DATASET=args.dataset,
                      HABITAT_EXAMPLE_OBJECTS_DIR=args.objects,
                      FEED_WIDTH=str(args.width), FEED_HEIGHT=str(args.height),
                      FEED_HFOV='90', FEED_FPS=str(args.fps), FEED_SHOW='0',
                      FEED_TOUR_ALL_FLOORS='0', FEED_POST_SCAN_HOOK='',
                      FEED_GT_SEMANTIC='1' if args.record_gt else '0',
                      GRAPH_API_OUTPUT_DIR=str(output), RUN_DIR=str(output), OUT_DIR=str(output))
    if args.config:
        os.environ['GRAPH_API_CONFIG'] = args.config
    names = ('habitat_feed_host', 'schedule_batch', 'voronoi_roadmap',
             'script_runner', 'scene_script', 'generate_scene_batch',
             'run_habitat_script', 'hm3d_ground_truth_manifest')
    if args.record_gt:
        names += ('hm3d_ground_truth_manifest',)
    sources = provenance(args.graph_api_root, names)
    metadata = {'schema': 'graphapi.baseline_acquisition.v1', 'complete': False,
                'sources': sources, 'adapter': {'path': __file__, 'sha256': digest(__file__)}, 'schedule': str(Path(args.schedule).resolve()),
                'schedule_sha256': digest(args.schedule), 'script_sha256': digest(args.script),
                'scene': args.scene, 'dataset': args.dataset, 'frames': 0,
                'pose_convention': 'Habitat/OpenGL camera-to-world',
                'depth_png_units': 'millimetres', 'depth_npy_units': 'metres',
                'purpose': args.purpose, 'fps': args.fps, 'created_at': time.time(),
                'depth_storage': ('npz-shuffle-lossless' if args.shuffle_depth else
                                  'npz-lossless' if args.compress_depth else 'npy'),
                'dynamic_semantic_id_offset': args.dynamic_semantic_id_offset,
                'ground_truth_recorded': args.record_gt, 'requested_laps': args.laps,
                'selected_floor_height_m': args.floor,
                'selected_floor_height_m': args.floor,
                'script_timing': 'wall-clock waits concurrent with tour',
                'config_sha256': digest(args.config) if args.config else None}
    def save_metadata():
        (output / 'acquisition.json').write_text(json.dumps(metadata, indent=2) + '\n')
    save_metadata()
    feed = source_module(args.graph_api_root, 'habitat_feed_host')
    scripts = source_module(args.graph_api_root, 'script_runner')
    if 'wait_waypoint' not in inspect.signature(scripts.HabitatScriptRunner).parameters:
        raise RuntimeError('Shared script_runner lacks wait_waypoint; update the selected GRAPH-API checkout')
    schedule = feed.load_schedule(args.schedule, args.floor)
    if not schedule['trajectory']:
        raise ValueError('Empty schedule')
    script = json.loads(Path(args.script).read_text())
    if script.get('scene') and Path(script['scene']).name != Path(args.scene).name:
        raise ValueError('Object script was compiled for a different scene')
    planned_turn = schedule.get('scan_plan', {}).get('turn_step_deg')
    if planned_turn is not None and not np.isclose(planned_turn, feed.TURN_STEP_DEG):
        raise ValueError('Schedule turn step disagrees with the shared simulator configuration')
    validate_triggers(scripts.HabitatScriptRunner, script, schedule, args.laps)
    from .observation_protocol import validate_script_contract
    observation_plan = validate_script_contract(script, schedule, args.laps)
    sim = feed.make_sim()
    try:
        if not sim.pathfinder.is_loaded:
            raise RuntimeError('The selected scene has no loaded navmesh')
        if args.record_gt:
            # Generate the complete static benchmark manifest from the same
            # loaded simulator before any scripted object enters the world.
            # Dynamic objects are recorded separately in every frame below.
            gt_module = source_module(args.graph_api_root, 'hm3d_ground_truth_manifest')
            static_gt = gt_module.extract(sim, Path(args.scene), selected_floor=None)
            required = ('ground_truth_floors_m', 'ground_truth_regions',
                        'ground_truth_objects', 'rooms', 'categories')
            if any(key not in static_gt for key in required):
                raise RuntimeError('Static GT generator returned an incomplete evaluation manifest')
            (output / 'static-gt.json').write_text(json.dumps(
                static_gt, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
            metadata['static_ground_truth'] = {
                'path': str(output / 'static-gt.json'),
                'sha256': digest(output / 'static-gt.json'),
                'floors': len(static_gt['ground_truth_floors_m']),
                'regions': len(static_gt['ground_truth_regions']),
                'objects': len(static_gt['ground_truth_objects']),
                'scope': 'whole scene before scheduled dynamic actions'}
            save_metadata()
        agent = sim.initialize_agent(0)
        state = agent.get_state()
        state.position = np.asarray(schedule['trajectory'][0]['xyz'], dtype=np.float32)
        if not sim.pathfinder.is_navigable(state.position):
            raise ValueError('First scheduled point is not navigable in this scene')
        agent.set_state(state)
        tour = feed.ScheduledTour(sim, schedule, args.laps, args.move)
        recording = Recording(output, args.fps, 90, args.compress_depth, args.record_gt,
                              args.shuffle_depth, args.dynamic_semantic_id_offset)
        session = Session(feed, sim, agent, tour, recording)
        runner = scripts.HabitatScriptRunner(Path(args.script).parent,
            output / 'current_script.json', session.publish, session.wait_result,
            'spawn', 'move', 'remove', capture_frame=session.capture,
            wait_waypoint=session.wait_waypoint)
        recording.capture(sim, agent, 'initial')
        result = session.drive(runner, str(Path(args.script).resolve()))
        (output / 'script_result.json').write_text(json.dumps(result, indent=2) + '\n')
        if not result.get('success') or not result.get('dataset_complete'):
            raise RuntimeError(f'Object script failed: {result}')
        if tour.skipped:
            raise RuntimeError(f'Tour skipped {len(tour.skipped)} points')
        current = provenance(args.graph_api_root, names)
        if sources != current or digest(__file__) != metadata['adapter']['sha256']:
            raise RuntimeError('Shared source changed during acquisition; repeat with a stable checkout')
        metadata.update(complete=True, frames=recording.index, tour=tour.report(),
                        actions=len(session.actions), script_success=True,
                        action_observation_plan=observation_plan)
        save_metadata()
        print(json.dumps(metadata, indent=2))
        return metadata
    finally:
        sim.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph-api-root', required=True)
    p.add_argument('--dataset-tools-root', help='Optional canonical dataset compiler/ROS runner directory; keeps its own config imports')
    sub = p.add_subparsers(dest='command', required=True)
    run = sub.add_parser('acquire')
    for name in ('scene', 'dataset', 'objects', 'schedule', 'script', 'output'):
        run.add_argument('--' + name, required=True)
    run.add_argument('--floor', type=float, required=True)
    run.add_argument('--laps', type=int, default=1)
    run.add_argument('--fps', type=float, default=3)
    run.add_argument('--width', type=int, default=640)
    run.add_argument('--height', type=int, default=480)
    run.add_argument('--move', choices=('navigate', 'teleport'), default='navigate')
    run.add_argument('--config')
    run.add_argument('--compress-depth', action='store_true', help='Lossless float32 depth compression')
    run.add_argument('--shuffle-depth', action='store_true', help='Lossless byte-shuffled float32 NPZ storage; reduces archive size')
    run.add_argument('--record-gt', action='store_true', help='Record native semantic pixels and dynamic world boxes without mutating the scene')
    run.add_argument('--dynamic-semantic-id-offset', type=int, default=0,
                     help='Optional reserved semantic label namespace for dynamic GT only; does not change RGB, depth or native algorithms')
    run.add_argument('--purpose', choices=('verification', 'measurement'), default='verification')
    tool = sub.add_parser('source-tool', help='Import and invoke the current shared compiler/generator/ROS2 runner')
    tool.add_argument('name', choices=('scene_script', 'generate_scene_batch', 'schedule_batch', 'run_habitat_script'))
    tool.add_argument('arguments', nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    if args.command == 'source-tool':
        if args.dataset_tools_root and args.name != 'schedule_batch':
            root = Path(args.dataset_tools_root).resolve(strict=True)
            sys.path.insert(0, str(root))
            module = importlib.import_module(args.name)
            if Path(module.__file__).resolve() != root / (args.name + '.py'):
                raise RuntimeError(f'{args.name} resolved outside the selected dataset tools root')
        else:
            module = source_module(args.graph_api_root, args.name)
        sys.argv = [module.__file__] + args.arguments
        return module.main()
    if args.laps < 1 or args.fps <= 0 or args.width <= 0 or args.height <= 0:
        p.error('laps, fps and image dimensions must be positive')
    if args.dynamic_semantic_id_offset and (not args.record_gt or not 1 <= args.dynamic_semantic_id_offset < 2**30):
        p.error('Dynamic semantic labels require --record-gt and an offset in [1, 2**30)')
    acquire(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
