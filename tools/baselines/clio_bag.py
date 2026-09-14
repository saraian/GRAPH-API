"""Convert shared scheduled observations to Clio's native ROS 1 input topics.

Run in Clio's ROS Noetic environment. No ROS master or simulator is needed.
The Habitat world is rotated to Z-up; camera poses use ROS optical axes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .depth_codec import read_depth


def export(recording, output):
    import rosbag
    import rospy
    from geometry_msgs.msg import TransformStamped
    from sensor_msgs.msg import CameraInfo
    from sensor_msgs.msg import Image as RosImage
    from tf.transformations import quaternion_from_matrix
    from tf2_msgs.msg import TFMessage

    recording, output = Path(recording), Path(output)
    metadata = json.loads((recording / 'acquisition.json').read_text())
    if not metadata.get('complete'):
        raise ValueError('Recording is incomplete')
    if output.exists():
        raise FileExistsError(output)
    rows = [json.loads(line) for line in (recording / 'frames.jsonl').read_text().splitlines()]
    if len(rows) != metadata['frames'] or not rows:
        raise ValueError('Frame count disagrees with the acquisition manifest')
    world = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    optical = np.diag([1, -1, -1, 1])
    with rosbag.Bag(str(output), 'w', compression='lz4') as bag:
        for row in rows:
            stem = row['stem']
            rgb = np.asarray(Image.open(recording / 'rgb' / (stem + '.png')))
            depth = read_depth(recording / 'depth_m', stem, metadata.get('depth_storage', 'npy'))
            pose = world @ np.loadtxt(recording / 'pose' / (stem + '.txt')).reshape(4, 4) @ optical
            stamp = rospy.Time.from_sec(row['time_s'])
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = 'world'
            transform.child_frame_id = 'camera_optical'
            t, q = pose[:3, 3], quaternion_from_matrix(pose)
            transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = t
            transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = q
            bag.write('/tf', TFMessage([transform]), stamp)
            for topic, data, encoding in (('/dominic/forward/color/image_raw', rgb, 'rgb8'),
                ('/dominic/forward/depth/image_rect_raw', depth.astype('<f4'), '32FC1')):
                msg = RosImage()
                msg.header.stamp, msg.header.frame_id = stamp, 'camera_optical'
                msg.height, msg.width = data.shape[:2]
                msg.encoding, msg.is_bigendian = encoding, False
                msg.step = data.strides[0]
                msg.data = data.tobytes()
                bag.write(topic, msg, stamp)
            info = CameraInfo()
            info.header.stamp, info.header.frame_id = stamp, 'camera_optical'
            info.width, info.height = row['width'], row['height']
            focal = info.width / (2 * np.tan(np.deg2rad(row['hfov_deg']) / 2))
            info.K = [focal, 0, info.width / 2, 0, focal, info.height / 2, 0, 0, 1]
            info.P = [focal, 0, info.width / 2, 0, 0, focal, info.height / 2, 0, 0, 0, 1, 0]
            info.R = np.eye(3).reshape(-1).tolist()
            info.distortion_model, info.D = 'plumb_bob', [0.0] * 5
            bag.write('/dominic/forward/color/camera_info', info, stamp)
    with rosbag.Bag(str(output)) as bag:
        counts = {k: v.message_count for k, v in bag.get_type_and_topic_info().topics.items()}
        if set(counts.values()) != {len(rows)} or len(counts) != 4:
            raise RuntimeError(f'Bag readback mismatch: {counts}')
    return counts


def validate_existing_bag(recording, bag_path):
    """Verify all original RGB/depth pixels and source stamps before bag reuse."""
    import rosbag
    import rospy
    recording, bag_path = Path(recording), Path(bag_path)
    metadata = json.loads((recording / 'acquisition.json').read_text())
    rows = [json.loads(line) for line in (recording / 'frames.jsonl').read_text().splitlines()]
    if not metadata['complete'] or metadata['frames'] != len(rows) or not rows:
        raise ValueError('Cannot reuse a bag for an incomplete recording')
    rgb_topic = '/dominic/forward/color/image_raw'
    depth_topic = '/dominic/forward/depth/image_rect_raw'
    indices = {rgb_topic: 0, depth_topic: 0}
    with rosbag.Bag(str(bag_path)) as bag:
        counts = {k: v.message_count for k, v in bag.get_type_and_topic_info().topics.items()}
        if set(counts) != {rgb_topic, depth_topic, '/tf', '/dominic/forward/color/camera_info'} or set(counts.values()) != {len(rows)}:
            raise ValueError('Existing bag topic counts differ from this recording')
        for topic, msg, _ in bag.read_messages(topics=list(indices)):
            row = rows[indices[topic]]
            if msg.header.stamp.to_nsec() != rospy.Time.from_sec(row['time_s']).to_nsec():
                raise ValueError('Existing bag source timestamps differ')
            if topic == rgb_topic:
                expected = np.asarray(Image.open(recording / 'rgb' / (row['stem'] + '.png')))
                encoding = 'rgb8'
            else:
                expected = read_depth(recording / 'depth_m', row['stem'], metadata.get('depth_storage', 'npy')).astype('<f4')
                encoding = '32FC1'
            if msg.encoding != encoding or (msg.height, msg.width) != expected.shape[:2] or bytes(msg.data) != expected.tobytes():
                raise ValueError('Existing bag pixels differ at ' + topic + ' frame ' + row['stem'])
            indices[topic] += 1
    h = hashlib.sha256()
    with bag_path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return {'bag': str(bag_path), 'sha256': h.hexdigest(), 'topic_counts': counts,
        'rgb_depth_frames_verified': indices, 'frames_sha256': hashlib.sha256((recording / 'frames.jsonl').read_bytes()).hexdigest(),
        'scope': 'Every RGB/depth pixel and source timestamp verified; original bag TF/camera topics retained'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    print(json.dumps(export(args.recording, args.output), indent=2))


if __name__ == '__main__':
    main()
