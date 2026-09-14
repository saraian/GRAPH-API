"""Exercise staged scene assets, shared trigger parsing, GT labels and depth storage."""
import argparse
import json
import os
from concurrent.futures import Future
from pathlib import Path

import numpy as np

from .depth_codec import read_depth
from .runtime import Recording, Session, digest, provenance, source_module, validate_triggers


def run(root, scene, output):
    inputs = root / 'inputs' / scene
    manifest = json.loads((root / 'prepared-scenes.json').read_text())
    entry = next(row for row in manifest['scenes'] if row['scene'] == scene)
    mesh = next((root / 'assets' / scene).glob('*.basis.glb'))
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(HABITAT_SCENE=str(mesh),
        HABITAT_DATASET=str(inputs / 'dataset.scene_dataset_config.json'),
        HABITAT_EXAMPLE_OBJECTS_DIR='/home/phd_student/Musumeci/baseline-integration/objects/configs',
        GRAPH_API_CONFIG=str(inputs / 'config.yaml'), FEED_GT_SEMANTIC='1',
        FEED_WIDTH='640', FEED_HEIGHT='480', FEED_SHOW='0', FEED_TOUR_ALL_FLOORS='0',
        GRAPH_API_OUTPUT_DIR=str(output), RUN_DIR=str(output), OUT_DIR=str(output))
    feed = source_module(root / 'source', 'habitat_feed_host')
    scripts = source_module(root / 'source', 'script_runner')
    schedule = feed.load_schedule(str(inputs / 'schedule.json'), entry['floor'])
    script = json.loads((inputs / 'script.json').read_text())
    validate_triggers(scripts.HabitatScriptRunner, script, schedule, 2)
    truth = json.loads((inputs / 'static-gt.json').read_text())
    offset = 1000000
    if max(row['semantic_id'] for row in truth['ground_truth_objects']) >= offset:
        raise ValueError('Static and dynamic semantic ID namespaces would overlap')
    sim = feed.make_sim()
    try:
        import magnum as mn
        import quaternion  # registers np.quaternion
        del quaternion
        if not sim.pathfinder.is_loaded:
            raise ValueError('Missing native navmesh')
        goals = [sim.pathfinder.is_navigable(np.asarray(point['xyz'], dtype=np.float32))
            for point in schedule['trajectory']]
        if not all(goals):
            raise ValueError('A staged scheduled point is not native-navigable')
        agent = sim.initialize_agent(0)
        state = agent.get_state()
        state.position = np.asarray(schedule['trajectory'][0]['xyz'], dtype=np.float32)
        agent.set_state(state)
        recording = Recording(output, 3, 90, True, True, True, offset)
        session = Session(feed, sim, agent, feed.ScheduledTour(sim, schedule, 2, 'navigate'), recording)
        session.setup_dispatch()
        recording.capture(sim, agent, 'preflight-first-tour-pose')
        if len(recording.rows[-1]['visible_semantic_ids']) < 2:
            raise ValueError('Semantic sensor is empty')
        command = next(step for step in script['steps'] if step['action'] == 'spawn')
        future = Future()
        session.requests.put(('action', command, future))
        session.dispatch()
        spawned = future.result()
        if not spawned['success'] or spawned['gt_semantic_id'] < offset:
            raise ValueError('Native spawn or external GT annotation failed')
        eye, target = command['capture_eye'], command['position']
        matrix = mn.Matrix4.look_at(mn.Vector3(eye), mn.Vector3(target), mn.Vector3.y_axis())
        q = mn.Quaternion.from_matrix(matrix.rotation())
        rotation = np.quaternion(q.scalar, *q.vector)
        state = agent.get_state()
        for sensor in state.sensor_states.values():
            sensor.position, sensor.rotation = np.asarray(eye), rotation
        agent.set_state(state, infer_sensor_states=False)
        recording.capture(sim, agent, 'preflight-object-view')
        obs = sim.get_sensor_observations()
        dynamic_pixels = int(np.count_nonzero(obs['semantic_sensor'] == spawned['gt_semantic_id']))
        if not dynamic_pixels:
            raise ValueError('Dynamic object has no GT pixels from its compiled capture view')
        decoded = read_depth(output / 'depth_m', '000001', 'npz-shuffle-lossless')
        if not np.array_equal(decoded.view(np.uint32), obs['depth_sensor'].view(np.uint32)):
            raise ValueError('Lossless depth round trip changed native values')
        future = Future()
        session.requests.put(('action', {'action': 'remove', 'object_id': spawned['object_id']}, future))
        session.dispatch()
        if not future.result()['success']:
            raise ValueError('Native removal failed')
        recording.capture(sim, agent, 'preflight-removed')
        if recording.rows[-1]['dynamic_ground_truth']:
            raise ValueError('Removed object remains in GT state')
        fixture = {'schema': 'graphapi.baseline_storage_fixture.v1', 'complete': True,
            'purpose': 'Three-pose codec/GT smoke only; not a completed tour or baseline run',
            'requested_laps': 0, 'frames': recording.index, 'scene': str(mesh),
            'depth_storage': 'npz-shuffle-lossless'}
        (output / 'acquisition.json').write_text(json.dumps(fixture, indent=2))
        report = {'passed': True, 'scene': scene, 'native_navigable_points': len(goals),
            'frames': recording.index, 'dynamic_gt_pixels': dynamic_pixels,
            'float_depth_bit_exact': True, 'native_baseline_algorithms_executed': False,
            'scope': 'Input assets, native navigation point membership, shared trigger validity, GT annotation and storage; not full tour execution or baseline certification',
            'assets_sha256': {p.name: digest(p) for p in (root / 'assets' / scene).iterdir()},
            'sources': provenance(root / 'source', ['habitat_feed_host', 'script_runner'])}
        (output / 'preflight.json').write_text(json.dumps(report, indent=2))
        return report
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.scene, args.output), indent=2))


if __name__ == '__main__':
    main()
