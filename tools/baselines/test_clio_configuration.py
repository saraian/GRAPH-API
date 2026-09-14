"""External native configuration selection must remain explicit and scoped."""
import tempfile
import unittest
from pathlib import Path
import gzip
import json
import yaml
from types import SimpleNamespace

from tools.baselines.clio_run import prepare, roslaunch_command, verify_output


class Configuration(unittest.TestCase):
    def test_confidence_uses_native_config_without_changing_other_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            r=Path(directory);base=r/'native';recording=r/'recording';output=r/'output';models=r/'models';clip=r/'clip'
            for p in (base/'clio_ros/config/realsense',base/'clio_ros/config/segmentation',recording,output,models,clip):p.mkdir(parents=True)
            (models/'FastSAM-x.pt').write_bytes(b'fixture-model');(clip/'ViT-B-32.pt').write_bytes(b'fixture-model')
            (recording/'frames.jsonl').write_text(json.dumps({'width':640,'height':480,'hfov_deg':90})+'\n')
            (base/'clio_ros/config/realsense/pipeline.yaml').write_text(yaml.safe_dump({'input':{'inputs':{'camera':{'sensor':{}}}}}))
            original={'segmentation':{'confidence':.55,'iou':.85,'output_size':640},'clip_model':{'type':'clip'}}
            config=base/'clio_ros/config/segmentation/small_clip.yaml';config.write_text(yaml.safe_dump(original));before=config.read_bytes()
            prepare(base,recording,output,models,clip,['find a chair'],.25)
            actual=yaml.safe_load((output/'segmentation.yaml').read_text())
            self.assertEqual(actual['segmentation']['confidence'],.25)
            self.assertEqual(actual['segmentation']['iou'],.85)
            self.assertEqual(actual['segmentation']['output_size'],640)
            self.assertEqual(config.read_bytes(),before)
            prepare(base,recording,output,models,clip,['find a chair'])
            self.assertEqual(yaml.safe_load((output/'segmentation.yaml').read_text())['segmentation']['confidence'],.55)
            for value in (0,1,-.1,float('nan')):
                with self.assertRaises(ValueError):prepare(base,recording,output,models,clip,['find a chair'],value)

    def test_semantic_mapping_uses_empty_native_feature_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            r=Path(directory);base=r/'native';recording=r/'recording';output=r/'output';models=r/'models';clip=r/'clip'
            for p in (base/'clio_ros/config/realsense',base/'clio_ros/config/segmentation',recording,output,models,clip):p.mkdir(parents=True)
            (models/'FastSAM-x.pt').write_bytes(b'fixture-model');(clip/'ViT-B-32.pt').write_bytes(b'fixture-model')
            (recording/'frames.jsonl').write_text(json.dumps({'width':640,'height':480,'hfov_deg':90})+'\n')
            (base/'clio_ros/config/realsense/pipeline.yaml').write_text(yaml.safe_dump({'input':{'inputs':{'camera':{'sensor':{}}}}}))
            (base/'clio_ros/config/segmentation/small_clip.yaml').write_text(yaml.safe_dump({'segmentation':{'confidence':.55},'clip_model':{}}))
            prepare(base,recording,output,models,clip,[],.25,semantic_mapping_only=True)
            self.assertEqual(yaml.safe_load((output/'object_tasks.yaml').read_text()),{})
            self.assertEqual(yaml.safe_load((output/'place_tasks.yaml').read_text()),{})

    def test_semantic_mapping_verifies_native_primitives(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); output=root/'output'; recording=root/'recording'
            (output/'graph/backend').mkdir(parents=True); recording.mkdir()
            segment={'id':ord('s') << 56,'layer':1,'attributes':{
                'semantic_feature':{'rows':2,'cols':1,'data':[.1,.2]},
                'bounding_box':{'dimensions':[1,2,3]},
                'mesh':{'points':[[0,0,0],[1,0,0],[0,1,0]]}}}
            (output/'graph/backend/dsg.json').write_text(json.dumps({'nodes':[segment]}))
            (output/'graph/backend/mesh.ply').write_bytes(b'x'*101)
            (output/'segmentation.yaml').write_text(yaml.safe_dump({'segmentation':{'confidence':.25}}))
            (output/'native_observer.json').write_text(json.dumps({
                'messages':{'/dominic/forward/semantic/image_raw':1},
                'input_frames_observed':1,'rgb_frames_relayed':1,
                'graph_snapshots':1,'error':None}))
            with gzip.open(output/'native_graph_history.jsonl.gz','wt') as stream:
                stream.write(json.dumps({'nodes':[{'id':'s1','type':'semantic_primitive',
                    'corners':[[0,0,0]]*8,'native_last_observed_ns':[1],
                    'native_is_active':True}]})+'\n')
            args=SimpleNamespace(output=output,recording=recording,semantic_mapping_only=True,
                tasks=[],rate=1.0,baseline_root='/baseline')
            result=verify_output(args,{'rgb':1})
            self.assertEqual(result['mode'],'task_free_semantic_mapping')
            self.assertEqual(result['semantic_primitives'],1)
            self.assertEqual(result['task_clustered_objects'],0)
            self.assertTrue(result['evaluation_readiness'][
                'ready_for_temporal_object_action_evaluation'])

    def test_roslaunch_allows_large_native_graph_to_serialize(self):
        command = roslaunch_command('/tmp/scheduled.launch')
        self.assertEqual(command[0], 'roslaunch')
        self.assertGreaterEqual(int(command[command.index('--sigint-timeout') + 1]), 600)
        self.assertIn('/tmp/scheduled.launch', command)


if __name__=='__main__':unittest.main()
