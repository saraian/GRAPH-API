import json
import tempfile
import unittest
from pathlib import Path

from tools.baselines.stage_dgx_pack import GIN_SCENE_ROOT, stage

GIN_PACK = '/home/phd_student/Musumeci/graphapi-pack-a-20260914'


class StageDgxPackTests(unittest.TestCase):
    def test_every_load_bearing_path_is_repathed_and_the_replan_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            pack = tmp / 'gin-pack'
            scene, tag = '00824-Dd4bFSTQ8gi', 'Dd4bFSTQ8gi'
            real_source = tmp / 'realrun' / 'source'
            (real_source / 'lost3dsg').mkdir(parents=True)
            (real_source / 'lost3dsg' / 'x.py').write_text('x')
            pack.mkdir()
            (pack / 'source').symlink_to(real_source, target_is_directory=True)  # as on Gin
            (pack / 'assets' / 'habitat_objects' / 'configs').mkdir(parents=True)
            (pack / 'integration' / 'tools').mkdir(parents=True)
            inputs = pack / 'inputs' / scene
            inputs.mkdir(parents=True)
            (inputs / f'{scene}.scene_dataset_config.json').write_text(json.dumps({
                'stages': [f'{GIN_SCENE_ROOT}/{scene}/{tag}.basis.glb',
                           f'{GIN_SCENE_ROOT}/{scene}/{tag}.semantic.glb']}))
            (inputs / 'floor-0.script.json').write_text(json.dumps({'steps': [], 'old': True}))
            (inputs / 'config.yaml').write_text('a: 1\n')
            (pack / 'pack-manifest.json').write_text(json.dumps({'pack': 'A', 'scenes': [{
                'scene': '00813-svBbv1Pavdk', 'floor': 0.08,
                'mesh': f'{GIN_SCENE_ROOT}/00813-svBbv1Pavdk/svBbv1Pavdk.basis.glb',
                'graph_api_root': f'{GIN_PACK}/source',
                'objects': f'{GIN_PACK}/assets/habitat_objects/configs',
                'dataset': 'inputs/00813-svBbv1Pavdk/x.json', 'script_sha256': 'stale'}],
                'remaining_pack_a_scenes': [scene]}))
            replan = tmp / 'replan'
            replan.mkdir()
            (replan / 'floor-0.script.json').write_text(json.dumps({
                'steps': [], 'scene': f'{GIN_SCENE_ROOT}/{scene}/{tag}.basis.glb',
                'source_compiler': {'path': f'{real_source}/lost3dsg/FOUND-Dataset/scene_script.py'}}))
            (replan / 'action-visibility-preflight.json').write_text('{}')

            root = stage(pack, scene, replan, tmp / 'out', '/raid/hm3d', '/raid/packs')

            plan = json.loads((root / 'pack-manifest.json').read_text())
            entry = plan['scenes'][0]
            self.assertEqual(entry['scene'], scene)
            self.assertEqual(entry['mesh'], f'/raid/hm3d/{scene}/{tag}.basis.glb')
            self.assertEqual(entry['graph_api_root'], f'/raid/packs/pack-a-dgx-{scene}/source')
            self.assertNotIn('script_sha256', entry)
            self.assertNotIn('remaining_pack_a_scenes', plan)
            script = json.loads((root / 'inputs' / scene / 'floor-0.script.json').read_text())
            self.assertNotIn('old', script)                       # the replan won
            self.assertEqual(script['scene'], f'/raid/hm3d/{scene}/{tag}.basis.glb')
            self.assertEqual(script['source_compiler']['path'],
                             f'/raid/packs/pack-a-dgx-{scene}/source/lost3dsg/FOUND-Dataset/scene_script.py')
            dataset = (root / 'inputs' / scene / f'{scene}.scene_dataset_config.json').read_text()
            self.assertNotIn('/home/phd_student', dataset)
            self.assertFalse((root / 'source').is_symlink())
            self.assertTrue((root / 'source' / 'lost3dsg' / 'x.py').is_file())

    def test_refuses_when_a_source_path_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            pack = tmp / 'gin-pack'
            scene = '00824-Dd4bFSTQ8gi'
            for sub in ('source', 'assets', 'integration', f'inputs/{scene}'):
                (pack / sub).mkdir(parents=True)
            (pack / 'inputs' / scene / f'{scene}.scene_dataset_config.json').write_text(
                json.dumps({'stages': ['/home/phd_student/elsewhere/unmapped.glb']}))
            (pack / 'inputs' / scene / 'floor-0.script.json').write_text('{}')
            (pack / 'pack-manifest.json').write_text(json.dumps({'scenes': [{
                'scene': scene, 'mesh': 'm', 'graph_api_root': 'g', 'objects': 'o'}]}))
            with self.assertRaisesRegex(RuntimeError, 'source-machine paths remain'):
                stage(pack, scene, tmp / 'none', tmp / 'out', '/raid/hm3d', '/raid/packs')


if __name__ == '__main__':
    unittest.main()
