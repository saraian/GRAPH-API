import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

SRC = Path(__file__).resolve().parents[1] / 'src/perception_module'
spec = importlib.util.spec_from_file_location('object_evidence', SRC / 'object_evidence.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def obj():
    return NS(label='bed#1', color='red', material='wood', description='red bed',
              object_id='stable-id', room_id='room', onto_aligned=True)


def add(o, capture, label='sofa', observation=None, **kw):
    e.record(o, dict(capture_id=capture, observation_id=observation or capture,
                     label=label, description='a sofa', **kw))


class EvidenceTests(unittest.TestCase):
    def test_manager_consumes_new_semantics_after_accepted_association(self):
        tree = ast.parse((SRC / 'object_manager_6.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == '_stash_cycle_observation')
        o = obj()
        env = dict(object_evidence=e, wm=NS(snapshot=lambda: [o]),
                   world2vec=None, get_embedding=lambda model, text: ['vector', text])
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<stash>', 'exec'), env)
        manager = NS(room_manager=NS(scene_graph={'room': {'objects': ['bed#1']}}))
        for capture in ('a', 'b'):
            evidence = dict(capture_id=capture, observation_id=capture, label='sofa',
                            description='a sofa', color='blue', material='fabric')
            env['_stash_cycle_observation'](manager, o,
                dict(observation=evidence, semantic_evidence=evidence))
        self.assertEqual(o.label, 'sofa#1')
        self.assertEqual(o.embedding, ['vector', 'a sofa'])
        self.assertEqual(o._cycle_capture_id, 'b')
        self.assertEqual(e.summary(o)['capture_count'], 2)

    def test_duplicates_do_not_confirm_or_vote_twice(self):
        o = obj()
        for i in range(10):
            add(o, 'one', observation=str(i))
        e.stabilize(o, {})
        self.assertEqual(e.summary(o)['state'], 'tentative')
        self.assertEqual(o.label, 'bed#1')
        add(o, 'two')
        e.stabilize(o, {})
        self.assertEqual(o.label, 'sofa#1')
        self.assertEqual(o.object_id, 'stable-id')
        self.assertFalse(o.onto_aligned)

    def test_unknown_and_ties_preserve_incumbent(self):
        o = obj()
        for capture, label in [('a', 'bed'), ('b', 'bed'), ('c', 'sofa'), ('d', 'sofa'), ('e', 'unknown')]:
            add(o, capture, label, color='unknown')
        e.stabilize(o, {})
        self.assertEqual(o.label, 'bed#1')
        self.assertEqual(o.color, 'red')

    def test_conflicting_same_capture_abstains(self):
        o = obj()
        add(o, 'a', 'bed', 'a1')
        add(o, 'a', 'sofa', 'a2')
        add(o, 'b', 'sofa')
        e.stabilize(o, {})
        self.assertEqual(o.label, 'bed#1')

    def test_merge_deduplicates_and_preserves_confidence_provenance(self):
        a, b = obj(), obj()
        add(a, 'one', boxer_confidence=.7)
        add(b, 'one', boxer_confidence=.7)
        e.merge(a, b)
        self.assertEqual(e.summary(a)['capture_count'], 1)
        add(b, 'two', boxer_confidence=.9)
        e.merge(a, b)
        self.assertEqual(e.summary(a)['state'], 'confirmed')
        self.assertAlmostEqual(e.summary(a)['confidence_mean'], .8)
        self.assertEqual(e.summary(a)['confidence_source'], 'boxer_3d')

    def test_missing_capture_or_score_is_unmeasured(self):
        o = obj()
        add(o, None)
        self.assertEqual(e.summary(o)['state'], 'unmeasured')
        add(o, 'a')
        self.assertIsNone(e.summary(o)['confidence_mean'])

    def test_rename_keeps_shared_room_labels_and_other_rooms(self):
        o, other = obj(), obj()
        graph = {'room': {'objects': ['bed#1']}, 'elsewhere': {'objects': ['bed#1']}}
        add(o, 'a'); add(o, 'b')
        self.assertTrue(e.stabilize(o, graph, [o, other]))
        self.assertEqual(graph['room']['objects'], ['bed#1', 'sofa#1'])
        self.assertEqual(graph['elsewhere']['objects'], ['bed#1'])


if __name__ == '__main__':
    unittest.main()
