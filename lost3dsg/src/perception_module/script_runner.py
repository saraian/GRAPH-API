#!/usr/bin/env python3

"""Loader ed esecutore di script dichiarativi per Habitat.

Lo script contiene solo azioni JSON; nessun codice Python viene eseguito dal
file. Gli object_id restituiti da Habitat vengono associati ai nomi logici
definiti nello script. Un'azione spawn/move/remove con ``at_waypoint`` viene
pubblicata solo dopo la barriera di schedule fornita dal nodo chiamante.
"""

import json
import math
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# scene_script accepts at most 64 logical actions and may insert one settling
# wait between each pair. Keep the executor limit aligned with that expansion.
MAX_COMPILED_STEPS = 127


class HabitatScriptRunner:
    def __init__(
        self,
        scripts_dir: Path,
        state_path: Path,
        publish: Callable[[Any, Dict[str, Any]], None],
        wait_result: Callable[[str, float], Dict[str, Any]],
        spawn_publisher: Any,
        move_publisher: Any,
        remove_publisher: Any,
        capture_frame: Optional[Callable[[str, int, Dict[str, Any]], Optional[str]]] = None,
        wait_waypoint: Optional[Callable[[Dict[str, int]], Dict[str, Any]]] = None,
    ) -> None:
        self.scripts_dir = Path(scripts_dir)
        self.state_path = Path(state_path)
        self.publish = publish
        self.wait_result = wait_result
        self.spawn_publisher = spawn_publisher
        self.move_publisher = move_publisher
        self.remove_publisher = remove_publisher
        self.capture_frame = capture_frame
        self.wait_waypoint = wait_waypoint
        self.object_ids: Dict[str, int] = {}
        self.created_object_ids: Dict[str, int] = {}

    def _capture(self, action: str, step_index: int, result: Dict[str, Any]) -> Dict[str, Any]:
        """Return an explicit, serializable capture outcome for every action."""
        if self.capture_frame is None:
            return {
                "success": False, "attempts": 0,
                "error": "capture disabilitata",
            }
        try:
            outcome = self.capture_frame(action, step_index, result)
            if isinstance(outcome, dict):
                return {
                    "success": bool(outcome.get("success")),
                    "attempts": int(outcome.get("attempts", 1)),
                    "image": outcome.get("image"),
                    "error": outcome.get("error"),
                }
            if outcome:
                # Backward compatibility for custom callbacks returning a path.
                return {
                    "success": True, "attempts": 1,
                    "image": str(outcome), "error": None,
                }
            return {"success": False, "attempts": 1, "error": "nessuna immagine"}
        except Exception as exc:
            # La mancata acquisizione e' diagnostica, non deve annullare
            # un'azione Habitat gia' completata con successo.
            return {"success": False, "attempts": 1, "error": str(exc)}

    @staticmethod
    def _attach_capture(entry: Dict[str, Any], capture: Dict[str, Any]) -> None:
        entry["capture_success"] = bool(capture.get("success"))
        entry["capture_attempts"] = int(capture.get("attempts", 0))
        if capture.get("image"):
            entry["image"] = str(capture["image"])
        if capture.get("error"):
            entry["capture_error"] = str(capture["error"])

    def available_scripts(self) -> List[Dict[str, str]]:
        result = []
        if not self.scripts_dir.is_dir():
            return result
        for path in sorted(self.scripts_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                script_id = str(data.get("script_id") or path.stem)
                result.append({
                    "script_id": script_id,
                    "description": str(data.get("description", "")),
                })
            except (OSError, json.JSONDecodeError, TypeError):
                continue
        return result

    def load(self, script_id: str) -> Dict[str, Any]:
        requested = str(script_id).strip()
        direct_path = Path(requested)
        if direct_path.is_file():
            data = json.loads(direct_path.read_text(encoding="utf-8"))
            if not isinstance(data.get("steps"), list):
                raise ValueError(f"Lo script '{requested}' non contiene steps validi")
            return data
        for path in sorted(self.scripts_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if str(data.get("script_id", path.stem)) == requested:
                if not isinstance(data.get("steps"), list):
                    raise ValueError(f"Lo script '{requested}' non contiene steps validi")
                return data
        raise FileNotFoundError(f"Script non trovato: {requested}")

    def select(self, script_id: str) -> Dict[str, Any]:
        data = self.load(script_id)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({"script_id": str(script_id)}, indent=2),
            encoding="utf-8",
        )
        return data

    def current_script_id(self) -> Optional[str]:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            value = data.get("script_id")
            return str(value) if value else None
        except (OSError, json.JSONDecodeError, TypeError):
            return None

    def _resolve_object_id(self, reference: Any) -> int:
        if isinstance(reference, str) and reference in self.object_ids:
            return int(self.object_ids[reference])
        return int(reference)

    @staticmethod
    def _waypoint_trigger(step: Dict[str, Any], index: int) -> Optional[Dict[str, int]]:
        """Validate the optional schedule barrier attached to a physical action.

        ``stop`` is the stable visit-order number written by schedule_batch.py.
        ``lap`` is optional and uses the feed event's zero-based numbering.
        """
        raw = step.get("at_waypoint")
        if raw is None:
            return None
        if isinstance(raw, bool):
            raise ValueError(f"Step {index}: at_waypoint non valido")
        if isinstance(raw, int):
            raw = {"stop": raw}
        if not isinstance(raw, dict) or set(raw) - {"stop", "lap"}:
            raise ValueError(
                f"Step {index}: at_waypoint deve essere un intero oppure "
                "un oggetto con stop e lap opzionale"
            )
        if "stop" not in raw or isinstance(raw["stop"], bool):
            raise ValueError(f"Step {index}: at_waypoint.stop mancante o non valido")
        try:
            trigger = {"stop": int(raw["stop"])}
            if "lap" in raw:
                if isinstance(raw["lap"], bool):
                    raise ValueError
                trigger["lap"] = int(raw["lap"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Step {index}: stop/lap del waypoint non validi") from exc
        if trigger["stop"] < 0 or trigger.get("lap", 0) < 0:
            raise ValueError(f"Step {index}: stop/lap del waypoint devono essere >= 0")
        return trigger

    def _build_ledger(self, steps, results, step_times):
        """-> {"steps": [...]} -- what the script did, and when, for the evaluator.

        One entry per EXECUTED step, carrying the object identity, the placement and the stamp.
        A step that never ran is ABSENT rather than recorded with a null time: a step with no time
        cannot be placed on the timeline, and writing one would let a reader believe the world
        changed at an instant nobody measured.
        """
        ran = {entry.get("step") for entry in results if isinstance(entry, dict)}
        # The simulator reports the object's REAL AABB with the command result. Without it the
        # evaluator falls back to a fabricated cube and every volume metric on that pose is
        # meaningless, so it is carried here rather than dropped (script_ledger marks a pose
        # extents_source="measured" or "default" accordingly).
        extents_by_step = {}
        for entry in results:
            if isinstance(entry, dict) and isinstance(entry.get("result"), dict):
                size = entry["result"].get("extents")
                if isinstance(size, (list, tuple)) and len(size) == 3:
                    extents_by_step[entry.get("step")] = [float(v) for v in size]
        out = []
        for index, step in enumerate(steps):
            if index not in ran or not isinstance(step, dict):
                continue
            action = str(step.get("action", "")).strip().lower()
            name = step.get("name") or step.get("object")
            entry = {
                "step": index,
                "action": action,
                "at": step_times.get(index),
                "stamp": "step_start; the effect lands before the next step's stamp",
            }
            if name is not None:
                entry["name"] = str(name)
                entry["object_id"] = str(name)
            if step.get("template"):
                entry["template"] = str(step["template"])
            if step.get("target_category"):
                entry["target_category"] = str(step["target_category"])
            position = step.get("position")
            if isinstance(position, (list, tuple)) and len(position) == 3:
                entry["position"] = [float(v) for v in position]
            if index in extents_by_step:
                entry["extents"] = extents_by_step[index]
            out.append(entry)
        return {"steps": out,
                "note": ("scene-script ground truth. Each entry is one executed step; a move is "
                         "read as closing the previous pose and opening a new one.")}

    def run(self, script_id: str, timeout: float = 8.0) -> Dict[str, Any]:
        data = self.select(script_id)
        if not isinstance(data.get("steps"), list) or not data["steps"]:
            raise ValueError("Lo script deve contenere almeno uno step")
        if len(data["steps"]) > MAX_COMPILED_STEPS:
            raise ValueError(
                f"Lo script supera il limite di {MAX_COMPILED_STEPS} step"
            )
        object_scale = data.get("object_scale")
        if object_scale is not None:
            try:
                object_scale = float(object_scale)
            except (TypeError, ValueError) as exc:
                raise ValueError("object_scale non valida") from exc
            if not math.isfinite(object_scale) or object_scale <= 0:
                raise ValueError("object_scale deve essere un numero positivo")
        self.object_ids = {}
        self.created_object_ids = {}
        results = []

        step_times = {}
        for index, step in enumerate(data["steps"]):
            if not isinstance(step, dict):
                raise ValueError(f"Step {index} non valido")
            action = str(step.get("action", "")).lower()
            # WHEN this step took effect. A scripted object exists, moves or disappears at a
            # moment, and the evaluator needs that moment to score a prediction against the world
            # as it stood when the robot looked (see script_ledger.py). The stamp is taken as the
            # step BEGINS: the command is published and acknowledged within it, so the true instant
            # lies between this stamp and the next step's. That bound is recorded, not implied.
            step_times[index] = time.time()

            if action not in {"spawn", "move", "remove", "wait"}:
                raise ValueError(f"Azione non supportata nello step {index}: {action}")

            trigger = self._waypoint_trigger(step, index)
            trigger_result = None
            if trigger is not None:
                if action == "wait":
                    raise ValueError(f"Step {index}: un wait non può avere at_waypoint")
                if self.wait_waypoint is None:
                    raise RuntimeError(
                        f"Step {index}: lo script richiede il waypoint {trigger}, "
                        "ma questo runner non riceve /habitat/scan_complete"
                    )
                trigger_result = self.wait_waypoint(trigger)
                if not trigger_result.get("success"):
                    return {
                        "success": False,
                        "failed_step": index,
                        "results": results,
                        "error": trigger_result,
                    }

            if action == "wait":
                seconds = float(step.get("seconds", 0))
                if not math.isfinite(seconds) or not 0 <= seconds <= 60:
                    raise ValueError(f"Step {index}: seconds deve essere tra 0 e 60")
                time.sleep(seconds)
                results.append({"step": index, "action": action, "success": True})
                continue

            if action == "spawn":
                if not step.get("name") or not step.get("template"):
                    raise ValueError(f"Step {index}: spawn richiede name e template")
                if step.get("visual_surface_validated") is True and (
                    not step.get("target_category")
                    or not isinstance(step.get("target_surface_point"), (list, tuple))
                    or len(step["target_surface_point"]) != 3
                ):
                    raise ValueError(
                        f"Step {index}: script compilato obsoleto; rigenerarlo per "
                        "includere categoria e punto fisico del supporto"
                    )
                if step["name"] in self.object_ids:
                    raise ValueError(f"Step {index}: nome oggetto duplicato")
                request_id = uuid.uuid4().hex
                payload: Dict[str, Any] = {}
                if step.get("template"):
                    payload["template"] = step["template"]
                if "position" in step:
                    if not isinstance(step["position"], (list, tuple)) or len(step["position"]) != 3:
                        raise ValueError(f"Step {index}: position non valida")
                    if not all(math.isfinite(float(value)) for value in step["position"]):
                        raise ValueError(f"Step {index}: position non valida")
                    payload["position"] = step["position"]
                if step.get("visual_surface_validated") is True:
                    payload["visual_surface_validated"] = True
                if step.get("target_category"):
                    payload["target_category"] = step["target_category"]
                if step.get("target_surface_point"):
                    payload["target_surface_point"] = step["target_surface_point"]
                if object_scale is not None:
                    payload["object_scale"] = object_scale
                payload["request_id"] = request_id
                self.publish(self.spawn_publisher, payload)
                result = self.wait_result("spawn", timeout, request_id)
                if not result.get("success"):
                    return {"success": False, "failed_step": index, "results": results, "error": result}

                name = step.get("name")
                if name:
                    self.object_ids[str(name)] = int(result["object_id"])
                    self.created_object_ids[str(name)] = int(result["object_id"])

                # Un nuovo oggetto può essere spawnato in una posizione visibile
                # usando un pixel, come già fa l'assistente conversazionale.
                if "target_pixel" in step:
                    move_result = self._move(result["object_id"], step["target_pixel"], timeout)
                    result["placement"] = move_result
                    if not move_result.get("success"):
                        return {"success": False, "failed_step": index, "results": results, "error": move_result}
                if step.get("capture_eye") is not None:
                    result["capture_eye"] = step["capture_eye"]
                entry = {"step": index, "action": action, "result": result}
                if trigger_result is not None:
                    entry["waypoint_trigger"] = trigger_result
                self._attach_capture(entry, self._capture(action, index, result))
                results.append(entry)
                continue

            if action == "move":
                reference = step.get("object") or step.get("name")
                if reference is None and len(self.object_ids) == 1:
                    reference = next(iter(self.object_ids))
                if reference is None:
                    raise ValueError(f"Step {index}: move richiede object")
                if step.get("visual_surface_validated") is True and (
                    not step.get("target_category")
                    or not isinstance(step.get("target_surface_point"), (list, tuple))
                    or len(step["target_surface_point"]) != 3
                ):
                    raise ValueError(
                        f"Step {index}: script compilato obsoleto; rigenerarlo per "
                        "includere categoria e punto fisico del supporto"
                    )
                object_id = self._resolve_object_id(reference)
                request_id = uuid.uuid4().hex
                target_pixel = step.get("target_pixel")
                payload = {"object_id": object_id, "request_id": request_id}
                if target_pixel is not None:
                    payload["pixel"] = target_pixel
                elif "position" in step:
                    payload["position"] = step["position"]
                    if step.get("visual_surface_validated") is True:
                        payload["visual_surface_validated"] = True
                    if step.get("target_category"):
                        payload["target_category"] = step["target_category"]
                    if step.get("target_surface_point"):
                        payload["target_surface_point"] = step["target_surface_point"]
                else:
                    raise ValueError(f"Step {index}: move richiede target_pixel o position")
                self.publish(self.move_publisher, payload)
                result = self.wait_result("move", timeout, request_id)
                if not result.get("success"):
                    results.append({"step": index, "action": action, "result": result})
                    return {"success": False, "failed_step": index, "results": results, "error": result}
                if step.get("capture_eye") is not None:
                    result["capture_eye"] = step["capture_eye"]
                entry = {"step": index, "action": action, "result": result}
                if trigger_result is not None:
                    entry["waypoint_trigger"] = trigger_result
                self._attach_capture(entry, self._capture(action, index, result))
                results.append(entry)
                continue

            if action == "remove":
                reference = step.get("object") or step.get("name")
                if reference is None and len(self.object_ids) == 1:
                    reference = next(iter(self.object_ids))
                if reference is None:
                    raise ValueError(f"Step {index}: remove richiede object")
                object_id = self._resolve_object_id(reference)
                request_id = uuid.uuid4().hex
                self.publish(self.remove_publisher, {"object_id": object_id, "request_id": request_id})
                result = self.wait_result("remove", timeout, request_id)
                entry = {"step": index, "action": action, "result": result}
                if trigger_result is not None:
                    entry["waypoint_trigger"] = trigger_result
                if not result.get("success"):
                    results.append(entry)
                    return {"success": False, "failed_step": index, "results": results, "error": result}
                self.object_ids = {
                    key: value for key, value in self.object_ids.items()
                    if value != object_id
                }
                # Inquadra la posa appena svuotata, quindi dopo la rimozione.
                self._attach_capture(
                    entry, self._capture("remove_after", index, result)
                )
                results.append(entry)
                continue

            raise ValueError(f"Azione non supportata nello step {index}: {action}")

        capture_entries = [
            entry for entry in results if "capture_success" in entry
        ]
        capture_failures = [
            {
                "step": entry["step"],
                "action": entry["action"],
                "error": entry.get("capture_error", "cattura fallita"),
            }
            for entry in capture_entries if not entry["capture_success"]
        ]
        self.last_ledger = self._build_ledger(data["steps"], results, step_times)
        return {
            "ledger": self.last_ledger,
            "success": True,
            "dataset_complete": not capture_failures if self.capture_frame is not None else None,
            "script_id": str(script_id),
            "object_ids": dict(self.object_ids),
            "active_object_count": len(self.object_ids),
            "created_object_ids": dict(self.created_object_ids),
            "created_object_count": len(self.created_object_ids),
            "capture_failures": capture_failures,
            "results": results,
        }

    def _move(self, object_id: int, pixel: Any, timeout: float) -> Dict[str, Any]:
        request_id = uuid.uuid4().hex
        self.publish(self.move_publisher, {
            "object_id": int(object_id),
            "pixel": [int(pixel[0]), int(pixel[1])],
            "request_id": request_id,
        })
        return self.wait_result("move", timeout, request_id)
