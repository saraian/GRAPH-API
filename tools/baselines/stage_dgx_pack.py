"""Stage one single-scene Pack A root for another machine, repathed, from a frozen Gin pack.

The Gin pack pins two absolute prefixes: the HM3D scene root and the pack root itself
(`graph_api_root`, `objects`). Everything else is relative. This copies what a scene needs
into a fresh root, substitutes the two prefixes for the target machine's, and refuses to
finish while any load-bearing file still names the source machine. The FREEZE is not
done here: `freeze_pack` pins scene assets and native revisions from the machine it runs
on, so it runs on the target after the copy.

ponytail: string substitution of two prefixes, checked by a grep that must come back empty.
"""
import argparse
import json
import shutil
from pathlib import Path

GIN_SCENE_ROOT = '/home/phd_student/Argenziano/habitat-MP3D/data/scene_datasets/hm3d/val'
LOAD_BEARING = ('pack-manifest.json', '{scene}.scene_dataset_config.json', 'floor-0.script.json')


def substitute(path, mapping):
    text = path.read_text()
    for old, new in mapping.items():
        text = text.replace(old, new)
    path.write_text(text)


def stage(source_pack, scene, replan_dir, out_parent, dgx_scene_root, dgx_pack_parent):
    root = out_parent / f'pack-a-dgx-{scene}'
    if root.exists():
        raise FileExistsError(root)
    dgx_root = f'{dgx_pack_parent.rstrip("/")}/pack-a-dgx-{scene}'
    plan = json.loads((source_pack / 'pack-manifest.json').read_text())
    entries = [e for e in plan['scenes'] if e['scene'] == scene]
    if not entries:
        # The Gin pack lists only the validated scene; the rest sit in remaining_pack_a_scenes.
        template = plan['scenes'][0]
        entry = dict(template)
        tag = scene.split('-', 1)[1]
        entry.update(scene=scene,
                     mesh=f'{GIN_SCENE_ROOT}/{scene}/{tag}.basis.glb',
                     input_root=f'inputs/{scene}',
                     dataset=f'inputs/{scene}/{scene}.scene_dataset_config.json',
                     schedule=f'inputs/{scene}/{scene}.schedule.json',
                     script=f'inputs/{scene}/floor-0.script.json',
                     config=f'inputs/{scene}/config.yaml',
                     action_visibility_preflight=f'inputs/{scene}/action-visibility-preflight.json')
        for key in ('planned_coverage_pct', 'planned_duration_s', 'schedule_sha256',
                    'script_sha256', 'scheduled_actions', 'floor'):
            entry.pop(key, None)
    else:
        entry = dict(entries[0])

    root.mkdir(parents=True)
    # source/ is a symlink on Gin; the target needs the real files, same bytes.
    shutil.copytree(source_pack / 'source', root / 'source', symlinks=False)
    shutil.copytree(source_pack / 'assets', root / 'assets')
    shutil.copytree(source_pack / 'integration', root / 'integration')
    shutil.copytree(source_pack / 'inputs' / scene, root / 'inputs' / scene)
    # The replan supersedes the pack's staged script and its preflight receipt.
    for name in ('floor-0.script.json', 'action-visibility-preflight.json', 'placement_points.json',
                 'dynamic-plans.json'):
        if (replan_dir / name).is_file():
            shutil.copy2(replan_dir / name, root / 'inputs' / scene / name)

    entry['graph_api_root'] = f'{dgx_root}/source'
    entry['objects'] = f'{dgx_root}/assets/habitat_objects/configs'
    plan = dict(plan, scenes=[entry], pack=f'{plan.get("pack", "A")}-dgx-{scene}')
    plan.pop('remaining_pack_a_scenes', None)
    (root / 'pack-manifest.json').write_text(json.dumps(plan, indent=2) + '\n')

    # Three prefixes, longest first: the real directory behind the pack's `source` symlink
    # (the replan scripts record their compiler by that path), the pack itself, the scene root.
    mapping = {str((source_pack / 'source').resolve()): f'{dgx_root}/source',
               str(source_pack): dgx_root,
               GIN_SCENE_ROOT: dgx_scene_root.rstrip('/')}
    substitute(root / 'pack-manifest.json', mapping)
    substitute(root / 'inputs' / scene / f'{scene}.scene_dataset_config.json', mapping)
    substitute(root / 'inputs' / scene / 'floor-0.script.json', mapping)

    leftovers = []
    for name in LOAD_BEARING:
        path = root / ('inputs/' + scene + '/' + name.format(scene=scene) if name != 'pack-manifest.json'
                       else name)
        if '/home/phd_student' in path.read_text():
            leftovers.append(str(path))
    if leftovers:
        raise RuntimeError('source-machine paths remain in: ' + ', '.join(leftovers))
    (root / 'STAGED_FROM.json').write_text(json.dumps({
        'source_pack': str(source_pack), 'scene': scene, 'replan_dir': str(replan_dir),
        'repath': mapping, 'freeze': 'run freeze_pack on the target machine; not frozen here'},
        indent=2) + '\n')
    return root


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-pack', type=Path, required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--replan-dir', type=Path, required=True)
    parser.add_argument('--out-parent', type=Path, required=True)
    parser.add_argument('--dgx-scene-root', required=True)
    parser.add_argument('--dgx-pack-parent', required=True)
    args = parser.parse_args(argv)
    root = stage(args.source_pack, args.scene, args.replan_dir, args.out_parent,
                 args.dgx_scene_root, args.dgx_pack_parent)
    print(root)


if __name__ == '__main__':
    main()
