#!/usr/bin/env python3

"""Assistente conversazionale OpenAI per controllare Habitat via ROS 2.

Avvio:
    python3 openai_habitat_assistant.py

Prerequisiti:
    pip install openai pillow

Il programma legge la chiave da api.txt, riceve l'ultima immagine da
/camera/rgb e traduce le function call del modello nei topic JSON del nodo
Habitat.
"""

import base64
import io
import json
import os
from pathlib import Path
from queue import Empty, Queue
import threading
import time
from typing import Any, Dict, Optional

import rclpy
from openai import OpenAI
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from script_runner import HabitatScriptRunner


SCRIPT_DIR = Path(os.environ.get(
    "HABITAT_SCRIPTS_DIR", Path(__file__).with_name("scripts")
))
SCRIPT_STATE = Path(os.environ.get(
    "HABITAT_CURRENT_SCRIPT", Path(__file__).parent / "state/current_script.json"
))


MODEL = os.environ.get("OPENAI_MODEL", "gpt-5")

SYSTEM_PROMPT = """
Sei l'assistente robotico per una scena Habitat-Sim.
Rispondi in italiano e usa gli strumenti quando l'utente chiede di creare,
spostare o rimuovere oggetti.

Quando l'utente chiede di eseguire una procedura composta, usa run_script con
uno degli script disponibili. Se esiste uno script persistente corrente,
riusalo per le richieste successive compatibili; cambialo solo se l'utente
chiede esplicitamente una procedura diversa.

L'immagine allegata e' la scena corrente, con risoluzione 640x480.
Quando devi scegliere una posizione, restituisci al tool un pixel [u, v]
all'interno della superficie desiderata. Il sistema converte il pixel in
coordinate 3D. Usa il centro della superficie e stai lontano dai bordi.
I pixel validi sono 0 <= u < 640 e 0 <= v < 480. Non correggere o troncare
pixel fuori immagine: chiedi all'utente un pixel valido.
Non inventare object_id: usa solo quelli restituiti dai tool.
Per uno spawn, usa il nome naturale richiesto dall'utente come template,
ad esempio "banana" o "mug"; il nodo risolve automaticamente il nome nel
template caricato. Usa template="random" solo se l'utente non specifica
alcun tipo di oggetto.
Dopo ogni operazione comunica l'object_id e l'esito all'utente.
"""


TOOLS = [
    {
        "type": "function",
        "name": "run_script",
        "description": "Seleziona, salva ed esegue uno script Habitat disponibile su file.",
        "parameters": {
            "type": "object",
            "properties": {
                "script_id": {
                    "type": "string",
                    "description": "ID esatto dello script disponibile.",
                }
            },
            "required": ["script_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "spawn_object",
            "description": (
            "Crea un oggetto Habitat. Template puo' essere un nome naturale "
            "come banana, mug o bottle, oppure un percorso completo. Usa "
            "random solo se l'utente non specifica il tipo. Se l'utente "
            "indica una superficie, scegli il pixel target nell'immagine."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "template": {
                    "type": "string",
                    "description": "Nome dell'oggetto, percorso completo oppure random.",
                },
                "target_pixel": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Pixel [u, v] della superficie target.",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "move_object",
        "description": "Sposta un oggetto esistente sul pixel target indicato.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {"type": "integer"},
                "target_pixel": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "required": ["object_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "remove_object",
        "description": "Rimuove un oggetto esistente usando il suo object_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {"type": "integer"},
            },
            "required": ["object_id"],
            "additionalProperties": False,
        },
    },
]


class HabitatAssistant(Node):
    def __init__(self):
        super().__init__("openai_habitat_assistant")

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._latest_image: Optional[bytes] = None
        self._latest_image_lock = threading.Lock()
        self._results: Queue[Dict[str, Any]] = Queue()

        self.create_subscription(Image, "/camera/rgb", self._on_image, qos)
        self.create_subscription(
            String,
            "/habitat/object_command_result",
            self._on_result,
            qos,
        )
        self._command_pub = self.create_publisher(
            String,
            "/habitat/spawn_object",
            qos,
        )
        self._move_pub = self.create_publisher(
            String,
            "/habitat/set_object_position",
            qos,
        )
        self._remove_pub = self.create_publisher(
            String,
            "/habitat/remove_object",
            qos,
        )

        key_path = Path(__file__).with_name("api.txt")
        if not key_path.is_file():
            raise FileNotFoundError(f"API key non trovata: {key_path}")
        api_key = key_path.read_text(encoding="utf-8").strip()
        if not api_key:
            raise RuntimeError(f"API key vuota: {key_path}")

        self._client = OpenAI(api_key=api_key)
        self._history = []
        self._ask_lock = threading.Lock()
        self._auto_stop = threading.Event()
        self._auto_thread: Optional[threading.Thread] = None
        self._auto_interval = 15.0
        self._auto_cycles = 0
        self._auto_max_cycles = 100
        self._known_object_ids = set()
        self._script_runner = HabitatScriptRunner(
            scripts_dir=SCRIPT_DIR,
            state_path=SCRIPT_STATE,
            publish=self._publish,
            wait_result=self._wait_result,
            spawn_publisher=self._command_pub,
            move_publisher=self._move_pub,
            remove_publisher=self._remove_pub,
        )

    def _on_image(self, msg: Image) -> None:
        if msg.encoding not in ("rgb8", "rgba8"):
            self.get_logger().warn(f"Encoding immagine non supportato: {msg.encoding}")
            return

        channels = 3 if msg.encoding == "rgb8" else 4
        raw = bytes(msg.data)
        image = PILImage.frombytes("RGB" if channels == 3 else "RGBA", (msg.width, msg.height), raw)
        if channels == 4:
            image = image.convert("RGB")

        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=95,
            subsampling=0,
        )
        with self._latest_image_lock:
            self._latest_image = base64.b64encode(buffer.getvalue()).decode("ascii")

    def _on_result(self, msg: String) -> None:
        try:
            self._results.put(json.loads(msg.data))
        except json.JSONDecodeError:
            self.get_logger().warn(f"Risultato ROS non JSON: {msg.data}")

    def _current_image_data_url(self) -> str:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self._latest_image_lock:
                image = self._latest_image
            if image:
                return f"data:image/jpeg;base64,{image}"
            time.sleep(0.05)
        raise RuntimeError("Nessuna immagine ricevuta da /camera/rgb")

    def _publish(self, publisher, payload: Dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(payload)
        publisher.publish(msg)

    def _wait_result(self, action: str, timeout: float = 5.0) -> Dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                result = self._results.get(timeout=0.1)
            except Empty:
                continue
            if result.get("action") == action:
                return result
        return {
            "success": False,
            "action": action,
            "message": "timeout in attesa del risultato Habitat",
        }

    def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name == "run_script":
            try:
                result = self._script_runner.run(arguments["script_id"])
                if result.get("success"):
                    self._known_object_ids.update(result.get("object_ids", {}).values())
                return result
            except Exception as exc:
                return {"success": False, "action": "run_script", "message": str(exc)}

        if name == "spawn_object":
            payload: Dict[str, Any] = {}
            template = arguments.get("template")
            if template and template.lower() != "random":
                payload["template"] = template

            self._publish(self._command_pub, payload)
            result = self._wait_result("spawn")
            if not result.get("success"):
                return result

            self._known_object_ids.add(int(result["object_id"]))

            pixel = arguments.get("target_pixel")
            if pixel is not None:
                move_result = self._move_to_pixel(result["object_id"], pixel)
                result["placement"] = move_result
            return result

        if name == "move_object":
            return self._move_to_pixel(
                int(arguments["object_id"]), arguments["target_pixel"]
            )

        if name == "remove_object":
            object_id = int(arguments["object_id"])
            self._publish(self._remove_pub, {"object_id": object_id})
            result = self._wait_result("remove")
            if result.get("success"):
                self._known_object_ids.discard(object_id)
            return result

        return {"success": False, "message": f"Tool sconosciuto: {name}"}

    def _move_to_pixel(self, object_id: int, pixel) -> Dict[str, Any]:
        self._publish(
            self._move_pub,
            {"object_id": int(object_id), "pixel": [int(pixel[0]), int(pixel[1])]},
        )
        return self._wait_result("move")

    def ask(self, user_text: str) -> str:
        """Esegue una richiesta serializzata verso il modello."""
        with self._ask_lock:
            return self._ask_impl(user_text)

    def _ask_impl(self, user_text: str) -> str:
        image_url = self._current_image_data_url()
        available = self._script_runner.available_scripts()
        script_context = (
            "\nScript disponibili (scegline uno solo quando appropriato):\n"
            + json.dumps(available, ensure_ascii=False)
            + f"\nScript persistente corrente: {self._script_runner.current_script_id()}\n"
            + "Se la richiesta è compatibile con lo script corrente, usa run_script "
              "con quell'ID invece di ricreare i comandi manualmente.\n"
        )
        self._history.append({
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_text},
                {"type": "input_image", "image_url": image_url},
            ],
        })

        response = self._client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT + script_context,
            input=self._history,
            tools=TOOLS,
        )

        while True:
            self._history.extend(response.output)
            calls = [
                item for item in response.output
                if item.type == "function_call"
            ]
            if not calls:
                return response.output_text

            for call in calls:
                try:
                    arguments = json.loads(call.arguments)
                    result = self._call_tool(call.name, arguments)
                except Exception as exc:
                    result = {"success": False, "message": str(exc)}

                self._history.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result),
                })

            response = self._client.responses.create(
                model=MODEL,
                instructions=SYSTEM_PROMPT,
                input=self._history,
                tools=TOOLS,
            )

    def start_auto(self, interval: float = 15.0) -> bool:
        """Avvia il ciclo autonomo; ritorna False se e' gia' attivo."""
        if self._auto_thread is not None and self._auto_thread.is_alive():
            return False

        self._auto_interval = max(5.0, float(interval))
        self._auto_cycles = 0
        self._auto_stop.clear()
        self._auto_thread = threading.Thread(
            target=self._auto_loop,
            name="habitat-auto-mode",
            daemon=True,
        )
        self._auto_thread.start()
        return True

    def stop_auto(self) -> bool:
        """Ferma il ciclo autonomo."""
        active = self._auto_thread is not None and self._auto_thread.is_alive()
        self._auto_stop.set()
        return active

    def auto_status(self) -> str:
        active = self._auto_thread is not None and self._auto_thread.is_alive()
        if not active:
            return "chat"
        return f"auto, intervallo={self._auto_interval:.1f}s, cicli={self._auto_cycles}"

    def _auto_loop(self) -> None:
        # Esegue subito il primo ciclo, poi attende tra un ciclo e l'altro.
        while not self._auto_stop.is_set():
            if self._auto_cycles >= self._auto_max_cycles:
                print("Auto> limite cicli raggiunto; modalità automatica fermata.")
                self._auto_stop.set()
                return

            self._auto_cycles += 1
            try:
                known_ids = sorted(self._known_object_ids)
                result = self.ask(
                    f"""
Modalità AUTONOMA. Devi eseguire esattamente UNA azione in questo ciclo:
non rispondere NO_ACTION e non limitarti a descrivere cosa faresti.

Gli object_id creati e gestiti dall'assistente sono: {known_ids}.

Regole:
- se non ci sono object_id gestiti, chiama spawn_object con template=random
  e scegli un pixel valido al centro di un tavolo o ripiano visibile,
  lontano dai bordi;
- se esistono object_id, scegli uno di essi e chiama move_object verso un
  nuovo pixel valido al centro di un tavolo o ripiano, oppure remove_object;
- non inventare ID e non chiamare più di un tool;
- evita pavimento, pareti, bordi dei mobili e parti occluse;
- per spawn e move usa sempre un pixel [u,v] valido nell'immagine 640x480.

Dopo la chiamata spiega brevemente l'azione eseguita.
"""
                )
                print(f"Auto> {result}")
            except Exception as exc:
                print(f"Auto> errore: {exc}")

            if self._auto_stop.wait(self._auto_interval):
                return


def main():
    rclpy.init()
    node = HabitatAssistant()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print(
        "Assistente Habitat pronto. Comandi: :auto [secondi], :pause, "
        ":chat, :status, esci."
    )
    try:
        while rclpy.ok():
            user_text = input("Tu> ").strip()
            if not user_text:
                continue
            if user_text.lower() in {"esci", "exit", "quit"}:
                break

            if user_text.lower().startswith(":auto"):
                parts = user_text.split()
                interval = float(parts[1]) if len(parts) > 1 else 15.0
                if node.start_auto(interval):
                    print(f"Sistema> modalità auto attiva, intervallo minimo {max(5.0, interval):.1f}s")
                else:
                    print("Sistema> modalità auto già attiva.")
                continue

            if user_text.lower() == ":pause":
                node.stop_auto()
                print("Sistema> modalità auto in pausa.")
                continue

            if user_text.lower() == ":chat":
                node.stop_auto()
                print("Sistema> modalità chat attiva.")
                continue

            if user_text.lower() == ":status":
                print(f"Sistema> modalità {node.auto_status()}.")
                continue

            try:
                print(f"Claude/OpenAI> {node.ask(user_text)}")
            except Exception as exc:
                print(f"Errore: {exc}")
    finally:
        node.stop_auto()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
