"""Observe native ROS outputs and input receipt times without changing Clio."""
import json
import gzip
import queue
import threading
import time
from collections import Counter
from pathlib import Path


class NativeObserver:
    def __init__(self, output, master, relay_rgb=False):
        import rosbag
        import rospy
        from sensor_msgs.msg import Image

        self.root = Path(output)
        self.started = time.perf_counter()
        self.inputs = {}
        self.receipts = []
        self.counts = Counter()
        self.written = 0
        self.graph_written = 0
        self.error = None
        self.pending = queue.Queue(maxsize=256)
        self.bag = rosbag.Bag(str(self.root / 'native_outputs.bag'), 'w', compression='lz4')
        self.graph_stream = gzip.open(self.root / 'native_graph_history.jsonl.gz', 'wt', compresslevel=1)
        from .clio_graph_observer import load_bindings
        self.dsg = load_bindings()
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.writer.start()
        self.relay = None
        self.subscribers = []
        if relay_rgb:
            from .clio_rgb_relay import RGBRelay
            self.relay = RGBRelay(self._input)
            self.relay.wait_for_subscriber()
        else:
            self.subscribers.append(rospy.Subscriber('/dominic/forward/color/image_raw', Image,
                self._input, queue_size=100, buff_size=64 * 1024 * 1024))
        self.topics = {}
        self.discover(master)
        if '/dominic/forward/semantic/image_raw' not in self.topics:
            raise RuntimeError('Native semantic image publisher is missing')

    def discover(self, master):
        """Hydra advertises graph topics after its task-dependent initialization."""
        import roslib.message
        import rospy
        for topic, kind in master.getTopicTypes():
            if topic in self.topics:
                continue
            if (kind == 'hydra_msgs/DsgUpdate' and '/backend/' in topic) or ('semantic' in topic and
                    (kind == 'sensor_msgs/Image' or kind.startswith('semantic_inference_msgs/'))):
                cls = roslib.message.get_message_class(kind)
                if cls is None:
                    raise RuntimeError(f'Cannot deserialize native output {topic}: {kind}')
                self.topics[topic] = kind
                self.subscribers.append(rospy.Subscriber(topic, cls,
                    lambda msg, name=topic: self._output(name, msg),
                    queue_size=100, buff_size=64 * 1024 * 1024))

    def _input(self, msg):
        self.inputs[msg.header.stamp.to_nsec()] = time.perf_counter() - self.started

    def _output(self, topic, msg):
        received = time.perf_counter() - self.started
        stamp = msg.header.stamp.to_nsec() if hasattr(msg, 'header') else None
        self.receipts.append({'topic': topic, 'stamp_ns': stamp, 'receipt_elapsed_s': received})
        self.counts[topic] += 1
        try:
            self.pending.put_nowait((topic, msg, received))
        except queue.Full:
            self.error = 'Native observer write queue overflowed; outputs are incomplete'

    def _write(self):
        import rospy
        while True:
            item = self.pending.get()
            if item is None:
                break
            topic, msg, received = item
            if self.topics[topic] == 'hydra_msgs/DsgUpdate':
                from .clio_graph_observer import snapshot
                if not msg.full_update:
                    raise NotImplementedError('Native graph recorder requires full DSG updates')
                graph = self.dsg.DynamicSceneGraph.from_binary(msg.layer_contents)
                row = snapshot(graph, msg.header.stamp.to_nsec(), topic)
                self.graph_stream.write(json.dumps(row, separators=(',', ':')) + '\n')
                self.graph_written += 1
            else:
                self.bag.write(topic, msg, rospy.Time.from_sec(received))
            self.written += 1

    def close(self):
        if self.relay is not None:
            self.relay.close()
        for sub in self.subscribers:
            sub.unregister()
        self.pending.put(None, timeout=30)
        self.writer.join(timeout=60)
        if self.writer.is_alive():
            raise RuntimeError('Native observer writer did not drain')
        self.bag.close()
        self.graph_stream.close()
        if self.written != sum(self.counts.values()):
            self.error = f'Observer saved {self.written} of {sum(self.counts.values())} received messages'
        with (self.root / 'native_receipts.jsonl').open('w') as stream:
            for row in self.receipts:
                source = self.inputs.get(row['stamp_ns'])
                row['input_receipt_elapsed_s'] = source
                row['paired_transport_latency_s'] = (
                    row['receipt_elapsed_s'] - source if source is not None else None)
                stream.write(json.dumps(row) + '\n')
        summary = {'topics': self.topics, 'messages': dict(self.counts),
            'graph_snapshots': self.graph_written,
            'graph_storage': 'All received backend updates as native semantic-primitive/object/place/room nodes, boxes, activity timestamps and edges in native_graph_history.jsonl.gz; dense meshes, repeated semantic vectors and robot layers excluded',
            'input_frames_observed': len(self.inputs), 'error': self.error,
            'input_timing_origin': 'before_rgb_publish' if self.relay else 'independent_rgb_subscriber',
            'rgb_frames_relayed': self.relay.count if self.relay else None,
            'latency_scope': ('External RGB publish start to native output receipt with identical source timestamp; includes serialization, transport and scheduling, not isolated algorithm time' if self.relay else
                'Signed difference between independent RGB/output subscriber receipts; callback order does not establish causal processing latency'),
            'bag_time': 'Observer monotonic elapsed seconds; original source stamps retained in message headers'}
        (self.root / 'native_observer.json').write_text(json.dumps(summary, indent=2))
        if self.error:
            raise RuntimeError(self.error)
