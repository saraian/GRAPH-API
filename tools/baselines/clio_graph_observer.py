"""Compact native DSG messages into their actual nodes, boxes and graph edges."""
import sys
from pathlib import Path

import numpy as np

from .replay_model import ZUP_TO_HABITAT


def load_bindings():
    # This image builds the matching native bindings but does not install them.
    abi = f'cpython-{sys.version_info.major}{sys.version_info.minor}'
    candidates = list(Path('/root/catkin_ws/src/spark_dsg/build').glob(f'lib.*-{abi}'))
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one native Spark DSG binding build, found {candidates}')
    sys.path.insert(0, str(candidates[0]))
    import spark_dsg
    return spark_dsg


def snapshot(graph, stamp, topic):
    """Serialize the native layers needed for replay and temporal evaluation.

    Task-free Clio deliberately has no ``O`` layer.  Its semantic map lives in
    the ``s`` layer, so excluding that layer made it impossible to tell whether
    a scheduled object caused map evidence to appear, move, or remain stale.
    Features and meshes stay in the final native DSG; repeating them in every
    snapshot would make the history needlessly large.
    """
    nodes = []
    for native in graph.nodes:
        value = native.id.value
        prefix = chr(value >> 56)
        if prefix not in ('O', 's', 'p', 'l'):
            continue
        attrs = native.attributes
        node = {'id': str(value), 'label': attrs.name or str(native.id),
            'type': {'O': 'object', 's': 'semantic_primitive',
                     'p': 'place', 'l': 'room'}[prefix],
            'position': (ZUP_TO_HABITAT @ attrs.position).tolist(),
            'native_first_observed_ns': np.asarray(
                getattr(attrs, 'first_observed_ns', None)).tolist()}
        if prefix in ('O', 's'):
            node['native_last_observed_ns'] = np.asarray(
                getattr(attrs, 'last_observed_ns', None)).tolist()
            node['native_is_active'] = bool(getattr(attrs, 'is_active', False))
        if attrs.bounding_box.is_valid():
            node['corners'] = (np.asarray(attrs.bounding_box.corners()) @ ZUP_TO_HABITAT.T).tolist()
        nodes.append(node)
    ids = {node['id'] for node in nodes}
    pairs = {(str(edge.source), str(edge.target)) for edge in graph.edges}
    pairs.update((str(edge.source), str(edge.target)) for edge in graph.interlayer_edges)
    edges = [{'source': a, 'target': b, 'label': 'native edge'}
        for a, b in sorted(pairs) if a in ids and b in ids]
    return {'time_s': stamp / 1e9, 'source_stamp_ns': stamp, 'scope': 'recorded_snapshot',
        'source': {'topic': topic, 'native_message': 'hydra_msgs/DsgUpdate'},
        'nodes': nodes, 'edges': edges,
        'excluded_layers': ['robot poses', 'dense mesh geometry',
                            'semantic feature vectors (retained in final native DSG)']}
