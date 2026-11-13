"""Host-side RPC client for the upstream Boxer inference worker.

Boxer's repository uses a top-level ``utils`` package, which collides with the
legacy GRAPH-API ``utils.py`` module. Run the unmodified upstream imports in a
dedicated interpreter and exchange only numpy arrays and plain dictionaries.
"""
from __future__ import annotations

import multiprocessing.connection
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time


class BoxerRuntime:
    def __init__(self, config):
        self._tmp = tempfile.TemporaryDirectory(prefix="graphapi-boxer-")
        self._address = str(Path(self._tmp.name) / "worker.sock")
        self._authkey = secrets.token_bytes(32)
        self._listener = multiprocessing.connection.Listener(
            self._address, family="AF_UNIX", authkey=self._authkey)
        env = dict(os.environ)
        env.setdefault("BOXER_ROOT", "/opt/boxer")
        self._process = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("boxer_worker.py")),
             "--worker", self._address, self._authkey.hex()],
            env=env, stdin=subprocess.DEVNULL)
        self._connection = None
        try:
            deadline = time.monotonic() + float(config.get("startup_timeout_s", 240))
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise RuntimeError(
                        f"Boxer worker exited during startup (code {self._process.returncode})")
                if self._listener._listener._socket is not None:
                    # accept() is blocking; use a short local timeout so startup failure
                    # remains diagnosable instead of hanging ROS initialization forever.
                    self._listener._listener._socket.settimeout(1.0)
                try:
                    self._connection = self._listener.accept()
                    break
                except TimeoutError:
                    continue
            if self._connection is None:
                raise TimeoutError("timed out waiting for the Boxer process")
            self._connection.send({"command": "init", "config": config})
            if not self._connection.poll(float(config.get("startup_timeout_s", 240))):
                raise TimeoutError("Boxer model initialization timed out")
            response = self._connection.recv()
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "Boxer initialization failed"))
        except Exception:
            self.close()
            raise
        finally:
            self._listener.close()

    def infer(self, rgb, depth, camera_info, transform, on_boxes_2d=None):
        r = transform.transform.rotation
        t = transform.transform.translation
        request = {
            "command": "infer",
            "rgb_bgr": rgb,
            "depth_m": depth,
            "intrinsics": {
                "width": int(camera_info.width), "height": int(camera_info.height),
                "fx": float(camera_info.k[0]), "fy": float(camera_info.k[4]),
                "cx": float(camera_info.k[2]), "cy": float(camera_info.k[5]),
            },
            "T_world_camera": {
                "translation": [float(t.x), float(t.y), float(t.z)],
                "rotation_xyzw": [float(r.x), float(r.y), float(r.z), float(r.w)],
            },
        }
        self._connection.send(request)
        response = self._connection.recv()
        if response.get("phase") == "boxes_2d":
            if on_boxes_2d is not None:
                on_boxes_2d(response.get("detections") or [])
            response = self._connection.recv()
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Boxer inference failed"))
        return response

    def close(self):
        connection = getattr(self, "_connection", None)
        if connection is not None:
            try:
                connection.send({"command": "stop"})
                connection.close()
            except (BrokenPipeError, EOFError, OSError):
                pass
            self._connection = None
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
        listener = getattr(self, "_listener", None)
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        temp = getattr(self, "_tmp", None)
        if temp is not None:
            temp.cleanup()
