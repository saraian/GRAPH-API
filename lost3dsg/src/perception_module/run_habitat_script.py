#!/usr/bin/env python3

"""Esegue uno script Habitat JSON senza usare l'assistente OpenAI.

Esempi:
    python3 run_habitat_script.py --list
    python3 run_habitat_script.py organize_objects_v1
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from queue import Empty, Queue
import threading
import time
import uuid

import rclpy
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from script_runner import HabitatScriptRunner


class HabitatScriptNode(Node):
    def __init__(self, scripts_dir: Path, state_path: Path, frames_dir: Path):
        super().__init__("habitat_script_runner")
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.results = Queue()
        self.frames_dir = Path(frames_dir)
        self._frame_condition = threading.Condition()
        self._latest_rgb = None
        self._frame_sequence = 0
        self.spawn_pub = self.create_publisher(String, "/habitat/spawn_object", qos)
        self.move_pub = self.create_publisher(String, "/habitat/set_object_position", qos)
        self.remove_pub = self.create_publisher(String, "/habitat/remove_object", qos)
        self.capture_pub = self.create_publisher(String, "/habitat/capture_object_view", qos)
        self.create_subscription(
            String,
            "/habitat/object_command_result",
            self._on_result,
            qos,
        )
        self.create_subscription(
            Image, "/habitat/object_capture/rgb", self._on_object_capture, qos
        )
        self.runner = HabitatScriptRunner(
            scripts_dir=scripts_dir,
            state_path=state_path,
            publish=self.publish_command,
            wait_result=self.wait_result,
            spawn_publisher=self.spawn_pub,
            move_publisher=self.move_pub,
            remove_publisher=self.remove_pub,
            capture_frame=self.capture_frame,
        )

    def _on_result(self, msg: String):
        try:
            self.results.put(json.loads(msg.data))
        except json.JSONDecodeError:
            self.get_logger().warning(f"Risultato non JSON: {msg.data}")

    def _on_object_capture(self, msg: Image):
        if msg.encoding not in {"rgb8", "rgba8"}:
            self.get_logger().warning(f"Cattura oggetto ignorata: encoding {msg.encoding}")
            return
        channels = 3 if msg.encoding == "rgb8" else 4
        expected_row_bytes = int(msg.width) * channels
        if msg.height <= 0 or msg.width <= 0 or int(msg.step) < expected_row_bytes:
            self.get_logger().warning("Cattura oggetto ignorata: dimensioni non valide")
            return
        raw = bytes(msg.data)
        if len(raw) < int(msg.step) * int(msg.height):
            self.get_logger().warning("Cattura oggetto ignorata: dati incompleti")
            return
        # Manteniamo solo l'ultimo frame. La copia elimina l'eventuale padding
        # a fine riga e rende l'immagine adatta a Pillow.
        packed = b"".join(
            raw[row * int(msg.step):row * int(msg.step) + expected_row_bytes]
            for row in range(int(msg.height))
        )
        with self._frame_condition:
            self._latest_rgb = (int(msg.width), int(msg.height), channels, packed)
            self._frame_sequence += 1
            self._frame_condition.notify_all()

    def capture_frame(self, action: str, step_index: int, result):
        """Request and save an object view, retrying transient capture failures."""
        if not isinstance(result, dict):
            return {"success": False, "attempts": 0, "error": "risultato non valido"}
        # Dopo spawn o teletrasporto l'oggetto dinamico deve compiere qualche
        # frame di fisica prima che la foto descriva davvero la posa finale.
        # L'attesa e' deliberatamente qui, mai nel risultato dello spawn:
        # bloccare quel risultato rendeva l'intero script soggetto a timeout.
        if action in {"spawn", "move", "remove_after"}:
            time.sleep(1.50)
        if action == "remove_after":
            position = result.get("position")
            if not isinstance(position, (list, tuple)) or len(position) != 3:
                return {
                    "success": False, "attempts": 0,
                    "error": "posizione rimossa non disponibile",
                }
            base_payload = {"position": [float(value) for value in position]}
        elif result.get("object_id") is not None:
            base_payload = {"object_id": int(result["object_id"])}
        else:
            return {"success": False, "attempts": 0, "error": "object_id mancante"}
        capture_eye = result.get("capture_eye")
        if isinstance(capture_eye, (list, tuple)) and len(capture_eye) == 3:
            base_payload["capture_eye"] = [float(value) for value in capture_eye]

        try:
            max_attempts = int(os.environ.get("HABITAT_CAPTURE_ATTEMPTS", "3"))
        except ValueError:
            max_attempts = 3
        max_attempts = min(max(max_attempts, 1), 5)
        last_error = "errore di cattura sconosciuto"
        for attempt in range(1, max_attempts + 1):
            with self._frame_condition:
                start_sequence = self._frame_sequence
            request_id = uuid.uuid4().hex
            payload = dict(base_payload)
            payload["request_id"] = request_id
            payload["capture_attempt"] = attempt
            self.publish_command(self.capture_pub, payload)
            capture_result = self.wait_result(
                "capture", timeout=3.0, request_id=request_id
            )
            if not capture_result.get("success"):
                last_error = str(
                    capture_result.get("message", "cattura rifiutata dal viewer")
                )
            else:
                with self._frame_condition:
                    deadline = time.monotonic() + 2.0
                    while self._frame_sequence <= start_sequence:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._frame_condition.wait(timeout=remaining)
                    frame = (
                        self._latest_rgb
                        if self._frame_sequence > start_sequence else None
                    )
                if frame is None:
                    last_error = "nessuna vista ricevuta dal viewer"
                else:
                    width, height, channels, packed = frame
                    image = PILImage.frombytes(
                        "RGB" if channels == 3 else "RGBA",
                        (width, height), packed,
                    )
                    if channels == 4:
                        image = image.convert("RGB")
                    if max(channel_max for _, channel_max in image.getextrema()) <= 2:
                        last_error = "vista oggetto interamente nera"
                    else:
                        self.frames_dir.mkdir(parents=True, exist_ok=True)
                        object_id = result.get("object_id", "unknown")
                        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
                        output = self.frames_dir / (
                            f"step_{step_index:02d}_{action}_object_"
                            f"{object_id}_{timestamp}.jpg"
                        )
                        image.save(output, format="JPEG", quality=95, subsampling=0)
                        return {
                            "success": True,
                            "attempts": attempt,
                            "image": str(output),
                            "error": None,
                        }
            self.get_logger().warning(
                f"Cattura step {step_index}, tentativo {attempt}/{max_attempts} "
                f"fallito: {last_error}"
            )
            if attempt < max_attempts:
                time.sleep(0.25)
        return {
            "success": False,
            "attempts": max_attempts,
            "image": None,
            "error": last_error,
        }

    def publish_command(self, publisher, payload):
        msg = String()
        msg.data = json.dumps(payload)
        publisher.publish(msg)

    def wait_result(self, action: str, timeout: float, request_id=None):
        deadline = time.time() + timeout
        deferred = []
        while time.time() < deadline:
            try:
                result = self.results.get(timeout=0.1)
            except Empty:
                continue
            if result.get("action") == action and (
                request_id is None or result.get("request_id") == request_id
            ):
                for other in deferred:
                    self.results.put(other)
                return result
            deferred.append(result)
        for other in deferred:
            self.results.put(other)
        return {
            "success": False,
            "action": action,
            "message": "timeout in attesa del risultato Habitat",
        }


def main():
    parser = argparse.ArgumentParser(description="Esegue uno script Habitat JSON")
    parser.add_argument("script_id", nargs="?", help="nome o percorso del file JSON da eseguire")
    parser.add_argument("--list", action="store_true", help="mostra gli script disponibili")
    parser.add_argument(
        "--frames-dir",
        default=os.environ.get("HABITAT_SCRIPT_FRAMES_DIR", "state/script_frames"),
        help="directory per i JPEG post-azione (default: state/script_frames)",
    )
    args = parser.parse_args()

    scripts_dir = Path(os.environ.get("HABITAT_SCRIPTS_DIR", Path(__file__).with_name("scripts")))
    state_path = Path(os.environ.get(
        "HABITAT_CURRENT_SCRIPT",
        Path(__file__).parent / "state/current_script.json",
    ))

    if args.list:
        dummy = HabitatScriptRunner(
            scripts_dir, state_path, lambda *_: None, lambda *_: {}, None, None, None
        )
        for script in dummy.available_scripts():
            print(f"{script['script_id']}: {script['description']}")
        return 0

    if not args.script_id:
        parser.error("indica uno script_id oppure usa --list")

    rclpy.init()
    node = HabitatScriptNode(scripts_dir, state_path, Path(args.frames_dir))
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        result = node.runner.run(args.script_id)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("success") else 1
    except Exception as exc:
        print(json.dumps({"success": False, "message": str(exc)}, indent=2))
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    raise SystemExit(main())
