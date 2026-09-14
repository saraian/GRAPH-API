#!/usr/bin/env python3
"""State and identity contract for persistent-Habitat floor sessions.

This module has no ROS or Habitat dependency.  The launcher and the persistent
feed use the same journal so a floor switch cannot be inferred from process
liveness or from a height alone.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


IDLE = "IDLE"
PREPARING = "PREPARING"
LOCALIZING = "LOCALIZING"
READY = "READY"
ACTIVE = "ACTIVE"
DRAINING = "DRAINING"
TRANSITION = "TRANSITION"
COMPLETE = "COMPLETE"
FAILED = "FAILED"


class TransitionRefused(RuntimeError):
    """The requested transition would weaken the floor-session contract."""


def floor_id(value: str | float) -> str:
    """Return the stable directory-safe identity used by existing floor maps."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"floor height must be finite, got {value!r}")
    return f"floor_{number + 0.0:+.2f}"


def parse_floor_sequence(raw: str) -> list[float]:
    """Parse an explicit ordered visit sequence, retaining repeated floors."""
    values = raw.replace(",", " ").split()
    if not values:
        raise ValueError("multi-floor mode requires an explicit floor sequence")
    result = [float(value) for value in values]
    for value in result:
        if not math.isfinite(value):
            raise ValueError(f"floor height must be finite, got {value!r}")
    return result


def _identity4() -> tuple[tuple[float, ...], ...]:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def validate_rigid_transform(value: Sequence[Sequence[float]]) -> list[list[float]]:
    """Validate and normalize one homogeneous rigid transform."""
    if len(value) != 4 or any(len(row) != 4 for row in value):
        raise ValueError("building_to_map must be a 4x4 matrix")
    matrix = [[float(cell) for cell in row] for row in value]
    if not all(math.isfinite(cell) for row in matrix for cell in row):
        raise ValueError("building_to_map contains a non-finite value")
    if any(abs(matrix[3][i] - expected) > 1e-7
           for i, expected in enumerate((0.0, 0.0, 0.0, 1.0))):
        raise ValueError("building_to_map has an invalid homogeneous last row")
    rotation = [row[:3] for row in matrix[:3]]
    for i in range(3):
        for j in range(3):
            dot = sum(rotation[k][i] * rotation[k][j] for k in range(3))
            expected = 1.0 if i == j else 0.0
            if abs(dot - expected) > 1e-5:
                raise ValueError("building_to_map rotation is not orthonormal")
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > 1e-5:
        raise ValueError("building_to_map rotation determinant is not +1")
    return matrix


@dataclasses.dataclass(frozen=True)
class SessionSpec:
    floor: float
    floor_id: str
    visit_index: int
    session_id: str
    transform_epoch: int
    mode: str
    map_id: str
    database_path: str
    source_database_path: str | None
    output_dir: str
    building_to_map: tuple[tuple[float, ...], ...] = dataclasses.field(default_factory=_identity4)

    def as_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["building_to_map"] = validate_rigid_transform(self.building_to_map)
        return result


def build_session_specs(
    floors: Iterable[float], house_dir: str | os.PathLike[str]
) -> list[SessionSpec]:
    """Build first-visit mapping and revisit localization sessions."""
    root = Path(house_dir).resolve()
    seen: dict[str, Path] = {}
    specs: list[SessionSpec] = []
    for visit_index, value in enumerate(floors):
        fid = floor_id(value)
        session_id = f"visit-{visit_index:03d}-{fid}"
        output = root / "sessions" / session_id
        database = output / "ros" / "rtabmap.db"
        source = seen.get(fid)
        mode = "mapping" if source is None else "localization"
        specs.append(SessionSpec(
            floor=float(value),
            floor_id=fid,
            visit_index=visit_index,
            session_id=session_id,
            transform_epoch=visit_index,
            mode=mode,
            map_id=f"{fid}-map",
            database_path=str(database),
            source_database_path=str(source) if source is not None else None,
            output_dir=str(output),
        ))
        if source is None:
            seen[fid] = database
    return specs


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SessionJournal:
    """Atomic state plus append-only events for one ordered floor sequence."""

    def __init__(self, directory: str | os.PathLike[str], clock=time.time) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / "state.json"
        self.events_path = self.directory / "events.jsonl"
        self.clock = clock
        if self.state_path.exists():
            with open(self.state_path, encoding="utf-8") as stream:
                self.state = json.load(stream)
            if self.state.get("state") not in {
                IDLE, PREPARING, LOCALIZING, READY, ACTIVE, DRAINING,
                TRANSITION, COMPLETE, FAILED,
            }:
                raise ValueError(f"unknown persisted coordinator state: {self.state.get('state')!r}")
        else:
            self.state: dict[str, Any] = {
                "state": IDLE,
                "session": None,
                "updated_at": float(self.clock()),
                "accepted_observations": 0,
                "rejected_observations": 0,
            }
            self._write_state()

    def _write_state(self) -> None:
        temporary = self.state_path.with_suffix(".json.tmp")
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(self.state, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.state_path)

    def _event(self, event: str, **detail: Any) -> None:
        record = {"event": event, "t": float(self.clock()), **detail}
        with open(self.events_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _set(self, state: str, event: str, **detail: Any) -> None:
        self.state["state"] = state
        self.state["updated_at"] = float(self.clock())
        self.state.update(detail)
        self._write_state()
        self._event(event, state=state, session=self.state.get("session"), **detail)

    def prepare(self, spec: SessionSpec) -> None:
        if self.state["state"] not in (IDLE, TRANSITION):
            raise TransitionRefused(f"cannot prepare from {self.state['state']}")
        if self.state.get("session") is not None:
            previous = self.state["session"]
            if spec.visit_index != int(previous["visit_index"]) + 1:
                raise TransitionRefused("visit indices must increase by one")
            if spec.transform_epoch <= int(previous["transform_epoch"]):
                raise TransitionRefused("transform epoch must increase at every floor switch")
        session = spec.as_dict()
        self.state = {
            "state": PREPARING,
            "session": session,
            "updated_at": float(self.clock()),
            "accepted_observations": 0,
            "rejected_observations": 0,
        }
        self._write_state()
        self._event("session_preparing", state=PREPARING, session=session)
        self._set(LOCALIZING, "localization_started", readiness=None)

    def mark_ready(self, evidence: Mapping[str, Any]) -> None:
        if self.state["state"] != LOCALIZING:
            raise TransitionRefused(f"cannot mark ready from {self.state['state']}")
        session = self.state["session"]
        for key in ("session_id", "floor_id", "map_id", "transform_epoch"):
            if evidence.get(key) != session[key]:
                raise TransitionRefused(f"readiness {key} does not match the prepared session")
        pose_stamp = float(evidence.get("pose_stamp", -1))
        if pose_stamp < float(self.state["updated_at"]):
            raise TransitionRefused("destination pose predates localization startup")
        authorities = evidence.get("tf_authorities")
        if not isinstance(authorities, list) or len(authorities) != 1:
            raise TransitionRefused("exactly one map-to-odom TF authority is required")
        if evidence.get("pose_source") not in ("simulator", "rtabmap"):
            raise TransitionRefused("pose_source must identify simulator or rtabmap")
        if not isinstance(evidence.get("quality"), Mapping):
            raise TransitionRefused("localization readiness requires a quality record")
        if evidence["quality"].get("passed") is not True:
            raise TransitionRefused("localization quality did not pass")
        normalized = dict(evidence)
        normalized["building_to_map"] = validate_rigid_transform(
            evidence.get("building_to_map", session["building_to_map"])
        )
        self._set(READY, "localization_ready", readiness=normalized)

    def activate(self) -> None:
        if self.state["state"] != READY:
            raise TransitionRefused(f"cannot activate from {self.state['state']}")
        self._set(ACTIVE, "session_active", active_since=float(self.clock()))

    def accept_observation(self, metadata: Mapping[str, Any]) -> bool:
        session = self.state.get("session") or {}
        reasons = []
        if self.state["state"] != ACTIVE:
            reasons.append(f"state={self.state['state']}")
        for key in ("session_id", "floor_id", "transform_epoch"):
            if metadata.get(key) != session.get(key):
                reasons.append(f"{key} mismatch")
        stamp = float(metadata.get("stamp", -1))
        if stamp < float(self.state.get("active_since", math.inf)):
            reasons.append("observation predates active barrier")
        if reasons:
            self.state["rejected_observations"] += 1
            self._write_state()
            self._event("observation_rejected", reasons=reasons, metadata=dict(metadata))
            return False
        self.state["accepted_observations"] += 1
        self._write_state()
        return True

    def begin_drain(self) -> None:
        if self.state["state"] != ACTIVE:
            raise TransitionRefused(f"cannot drain from {self.state['state']}")
        self._set(DRAINING, "drain_started", drain_started_at=float(self.clock()))

    def finish_visit(self, database_sha256: str, more_visits: bool) -> None:
        if self.state["state"] != DRAINING:
            raise TransitionRefused(f"cannot finish from {self.state['state']}")
        if len(database_sha256) != 64 or any(c not in "0123456789abcdef" for c in database_sha256):
            raise TransitionRefused("database_sha256 must be a lowercase SHA-256 digest")
        target = TRANSITION if more_visits else COMPLETE
        self._set(target, "visit_finished", database_sha256=database_sha256)

    def fail(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("failure reason must not be empty")
        if self.state["state"] in (COMPLETE, FAILED):
            raise TransitionRefused(f"cannot fail from terminal state {self.state['state']}")
        self._set(FAILED, "session_failed", failure_reason=reason)


def write_plan(path: str | os.PathLike[str], specs: Sequence[SessionSpec], transport: str) -> None:
    if transport not in ("teleport", "stairs"):
        raise ValueError("transport must be teleport or stairs")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "transport": transport,
        "transport_claim": (
            "floor-session switching only; autonomous stairs are not certified"
            if transport == "teleport" else
            "continuous stair traversal requested; success requires observed connected motion"
        ),
        "sessions": [spec.as_dict() for spec in specs],
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, destination)


def apply_transforms(specs: Sequence[SessionSpec], path: str | os.PathLike[str]) -> list[SessionSpec]:
    """Apply explicit building-to-map transforms keyed by floor_id."""
    with open(path, encoding="utf-8") as stream:
        transforms = json.load(stream)
    if not isinstance(transforms, dict):
        raise ValueError("transform file must be an object keyed by floor_id")
    required = {spec.floor_id for spec in specs}
    missing = sorted(required - set(transforms))
    if missing:
        raise ValueError(f"transform file is missing: {', '.join(missing)}")
    return [dataclasses.replace(
        spec,
        building_to_map=tuple(tuple(row) for row in validate_rigid_transform(transforms[spec.floor_id])),
    ) for spec in specs]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a persistent-world floor-session plan")
    parser.add_argument("--sequence", required=True, help="ordered floor heights; repeats are visits")
    parser.add_argument("--house-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--transport", choices=("teleport", "stairs"), default="teleport")
    parser.add_argument("--transforms", required=True,
                        help="JSON object of explicit 4x4 building-to-map transforms by floor_id")
    args = parser.parse_args()
    specs = build_session_specs(parse_floor_sequence(args.sequence), args.house_dir)
    specs = apply_transforms(specs, args.transforms)
    write_plan(args.output, specs, args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
