"""Timestamp RGB immediately before publishing it to unchanged native Clio.

The private playback topic prevents the observer's independent subscriber
scheduling from producing non-causal input/output receipt differences. Pixel
bytes, source timestamp and camera frame remain unchanged. ROS assigns the
publisher's own header sequence number, which is not the source frame identity.
"""
import time

INPUT = '/baseline_replay/input/color'
OUTPUT = '/dominic/forward/color/image_raw'
REMAP = OUTPUT + ':=' + INPUT


class RGBRelay:
    def __init__(self, before_publish):
        import rospy
        from sensor_msgs.msg import Image
        self.before_publish = before_publish
        self.count = 0
        self.publisher = rospy.Publisher(OUTPUT, Image, queue_size=100)
        self.subscriber = rospy.Subscriber(INPUT, Image, self.receive,
            queue_size=100, buff_size=64*1024*1024)

    def receive(self, message):
        self.before_publish(message)
        self.publisher.publish(message)
        self.count += 1

    def wait_for_subscriber(self, timeout=15):
        deadline = time.monotonic()+timeout
        while self.publisher.get_num_connections() == 0:
            if time.monotonic() >= deadline:
                raise RuntimeError('Native RGB subscriber did not connect to the timed publisher')
            time.sleep(0.05)

    def close(self):
        self.subscriber.unregister()
        self.publisher.unregister()
