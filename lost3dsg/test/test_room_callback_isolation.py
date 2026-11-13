"""Real ROS executor regression for lock contention in the room callback wiring.

Run in a ROS environment, without rosstub:
  python3 -m unittest discover -s lost3dsg/test -p test_room_callback_isolation.py
The slow segmentation is controlled with an Event, not a performance benchmark.
"""
import ast
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest

try:
    import rclpy
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from std_msgs.msg import String
    REAL_ROS = hasattr(rclpy, "__file__") and "rosstub" not in str(rclpy.__file__)
except ImportError:
    REAL_ROS = False

MODULES = Path(__file__).resolve().parents[1] / "src/perception_module"


def room_wiring():
    """Use the repository's actual group selection and camera-info callback."""
    tree = ast.parse((MODULES / "room_manager.py").read_text())
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Attribute) and t.attr == "_cb_group" for t in n.targets))
    sensor_assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                             and any(isinstance(t, ast.Attribute) and t.attr == "_sensor_cb_group" for t in n.targets))
    callback = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == "_camera_info_callback")
    cache = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                 and n.name == "_cache_sensor_state")
    room = SimpleNamespace(_lock=threading.RLock(), _sensor_lock=threading.Lock(),
                           _pending_sensor_state={})
    env = dict(self=room, MutuallyExclusiveCallbackGroup=MutuallyExclusiveCallbackGroup,
               ReentrantCallbackGroup=ReentrantCallbackGroup)
    exec(compile(ast.Module(body=[assignment, sensor_assignment, callback, cache], type_ignores=[]), "room_wiring", "exec"), env)
    room._cache_sensor_state = lambda **state: env["_cache_sensor_state"](room, **state)
    return room, env["_camera_info_callback"]


def wall_group(room):
    tree = ast.parse((MODULES / "object_manager_6.py").read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "create_subscription"
                and any(isinstance(a, ast.Constant) and a.value == "/detected_wall_segments" for a in n.args))
    group = next((kw.value for kw in call.keywords if kw.arg == "callback_group"), None)
    if group is None:
        return None  # ROS default mutually-exclusive group
    return eval(compile(ast.Expression(group), "wall_wiring", "eval"),
                {"self": SimpleNamespace(room_manager=room)})


@unittest.skipUnless(REAL_ROS, "requires a real ROS installation")
class RoomCallbackIsolation(unittest.TestCase):
    def probe(self, with_wall):
        context = rclpy.context.Context()
        rclpy.init(context=context)
        node = Node("room_callback_isolation", context=context)
        executor = MultiThreadedExecutor(num_threads=4, context=context)
        executor.add_node(node)
        room, camera = room_wiring()
        entered, release, received = threading.Event(), threading.Event(), threading.Event()
        timers = []

        def once(callback, group):
            timer = None
            def run():
                timer.cancel()
                callback()
            timer = node.create_timer(.01, run, callback_group=group)
            timers.append(timer)

        def segment():
            with room._lock:
                entered.set()
                release.wait(5)

        node.create_subscription(String, "/isolation_probe_bbox", lambda msg: received.set(), 10)
        publisher = node.create_publisher(String, "/isolation_probe_bbox", 10)
        once(segment, room._cb_group)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        try:
            self.assertTrue(entered.wait(2), "controlled segmentation never started")
            if with_wall:
                # ingest_detected_walls waits on this same room lock.
                def ingest_wall():
                    with room._lock:
                        pass
                once(ingest_wall, wall_group(room))
            else:
                for _ in range(3):
                    once(lambda: camera(room, "fresh"), room._sensor_cb_group)
            time.sleep(.2)  # let ready lock contenders reach the executor
            deadline = time.monotonic() + 2
            while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertGreater(publisher.get_subscription_count(), 0)
            publisher.publish(String(data="bbox"))
            self.assertTrue(received.wait(1),
                            "bbox reception blocked while room segmentation holds its lock")
            self.assertFalse(release.is_set())
            if not with_wall:
                self.assertEqual(room._pending_sensor_state.get("_camera_info"), "fresh",
                                 "sensor acquisition must also progress during segmentation")
        finally:
            release.set()
            executor.shutdown(timeout_sec=3)
            thread.join(timeout=3)
            node.destroy_node()
            rclpy.shutdown(context=context)

    def test_room_waiters_do_not_exhaust_executor(self):
        self.probe(with_wall=False)

    def test_wall_waiter_does_not_block_bbox_group(self):
        self.probe(with_wall=True)
