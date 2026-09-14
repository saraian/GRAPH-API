"""Create the immutable input/adapter inventory consumed by the Pack A driver."""
import argparse
import json
import os
from pathlib import Path

from .pack_a_driver import digest
from .run_pair import native_state, write


SCOPES = ('integration', 'inputs', 'source/lost3dsg/test',
          'source/lost3dsg/src/perception_module')


def files_under(path):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(item for item in path.rglob('*')
                  if item.is_file() and not item.is_symlink()
                  and item.suffix != '.pyc' and '__pycache__' not in item.parts)


def json_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from json_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from json_strings(item)


def declared_assets(root, plan):
    """Resolve every external runtime asset declared by the Pack A manifest."""
    files = []
    for entry in plan.get('scenes', []):
        mesh = Path(entry['mesh']).resolve()
        navmesh = mesh.with_suffix('.navmesh')
        if not mesh.is_file() or not navmesh.is_file():
            raise FileNotFoundError(f'Scene mesh/navmesh pair is incomplete: {mesh}')
        files += [mesh, navmesh]

        dataset_path = (root / entry['dataset']).resolve()
        dataset = json.loads(dataset_path.read_text())
        for value in json_strings(dataset):
            candidate = Path(value)
            if candidate.is_absolute() and candidate.suffix in {'.glb', '.txt', '.navmesh'}:
                if not candidate.is_file():
                    raise FileNotFoundError(candidate)
                files.append(candidate.resolve())

        script = json.loads((root / entry['script']).read_text())
        templates = {step['template'] for step in script.get('steps', [])
                     if step.get('action') == 'spawn'}
        configs = Path(entry['objects']).resolve()
        for template in templates:
            config = configs / f'{template}.object_config.json'
            if not config.is_file():
                raise FileNotFoundError(
                    f'Script template {template!r} is absent from {configs}')
            files.append(config)
            attributes = json.loads(config.read_text())
            for key in ('collision_asset', 'render_asset'):
                asset = (config.parent / attributes[key]).resolve()
                if not asset.is_file():
                    raise FileNotFoundError(asset)
                files.append(asset)
    if not plan.get('scenes'):
        raise ValueError('Pack manifest has no scenes')
    return files


def write_native_revisions(root, baseline_roots):
    state = native_state(Path(baseline_roots).resolve())
    if any(row['tracked_changes'] for row in state.values()):
        raise RuntimeError('Native baseline has tracked changes; refusing to freeze')
    value = {'schema': 'graphapi.native_baseline_revisions.v1',
             'baseline_roots': str(Path(baseline_roots).resolve()),
             'repositories': state}
    write(Path(root).resolve() / 'native-revisions.json', value)
    return value


def inventory(root):
    root = Path(root).resolve()
    plan_path = root / 'pack-manifest.json'
    native_revisions = root / 'native-revisions.json'
    if not native_revisions.is_file():
        raise FileNotFoundError(native_revisions)
    plan = json.loads(plan_path.read_text())
    files = [plan_path, native_revisions] + declared_assets(root, plan)
    for relative in SCOPES:
        scope = root / relative
        if not scope.is_dir():
            raise FileNotFoundError(scope)
        for directory, names, filenames in os.walk(scope, followlinks=False):
            names[:] = sorted(name for name in names if name != '__pycache__')
            for filename in sorted(filenames):
                path = Path(directory) / filename
                if path.suffix == '.pyc' or path.is_symlink() or not path.is_file():
                    continue
                files.append(path)
    result = {}
    for path in sorted(set(Path(item) for item in files), key=str):
        try:
            key = path.relative_to(root).as_posix()
        except ValueError:
            key = str(path)
        result[key] = digest(path)
    if not result or len(result) != len(set(result)):
        raise RuntimeError('Frozen inventory is empty or ambiguous')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack-root', type=Path, required=True)
    parser.add_argument('--baseline-roots', type=Path,
                        default=Path('/home/phd_student/Musumeci'))
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args(argv)
    output = args.pack_root.resolve() / 'frozen-files.json'
    if output.exists() and not args.replace:
        raise FileExistsError(output)
    write_native_revisions(args.pack_root, args.baseline_roots)
    values = inventory(args.pack_root)
    write(output, values)
    print(json.dumps({'path': str(output), 'files': len(values)}))


if __name__ == '__main__':
    main()
