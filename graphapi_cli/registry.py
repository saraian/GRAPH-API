"""Atomic resource ownership and exact operation IDs shared by all frontends."""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import uuid


def alive(pid, identity=None):
    try:
        os.kill(int(pid), 0)
        return not identity or process_identity(pid) == identity
    except (OSError, ValueError, TypeError):
        return False


def process_identity(pid):
    try:
        # /proc starttime prevents PID reuse; other supported hosts use PID liveness.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def container_alive(name):
    if not name:
        return False
    try:
        check = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name],
                               capture_output=True, text=True, timeout=5)
        return check.returncode == 0 and check.stdout.strip() == "true"
    except (OSError, subprocess.TimeoutExpired):
        return False


def actor_alive(actor, pid, identity=None):
    if actor.startswith("pal:"):
        return container_alive(actor[4:])
    if not container_alive(actor):
        return False
    if pid is None:
        # A detached Docker supervisor exists before its worker publishes its PID.
        return True
    script = "import os,sys; from pathlib import Path; p=int(sys.argv[1]); os.kill(p,0); expected=sys.argv[2]; actual=Path('/proc/%s/stat'%p).read_text().rsplit(')',1)[1].split()[19]; sys.exit(0 if not expected or actual==expected else 1)"
    try:
        result = subprocess.run(["docker", "exec", actor, "python3", "-c", script, str(pid), identity or ""],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def owned_containers(operation):
    try:
        result = subprocess.run(["docker", "ps", "--filter", "label=graphapi.operation=" + operation,
                                 "--format", "{{.Names}}"], capture_output=True, text=True, timeout=5)
        return result.stdout.splitlines() if result.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired):
        return []


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


class Registry:
    def __init__(self, workspace, resource_root=None):
        self.workspace = Path(workspace)
        self.operations = self.workspace / "results/operations"
        state = Path(resource_root or os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "graphapi"
        state.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(state / "resources.sqlite", timeout=30)
        self.db.execute("CREATE TABLE IF NOT EXISTS leases (resource TEXT PRIMARY KEY, operation TEXT, pid INTEGER, identity TEXT)")
        if "actor" not in {r[1] for r in self.db.execute("PRAGMA table_info(leases)")}:
            self.db.execute("ALTER TABLE leases ADD COLUMN actor TEXT")
            self.db.commit()

    def __del__(self):
        if hasattr(self, "db"):
            self.db.close()

    def reserve(self, mode, gpu=None, robot=None, ports=None, domain=None, extra=(), need_domain=True):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            retained = {}
            for resource, operation, pid, identity, actor in self.db.execute("SELECT resource,operation,pid,identity,actor FROM leases").fetchall():
                live = actor_alive(actor, pid, identity) if actor else alive(pid, identity)
                if not live:
                    retained.setdefault(operation, owned_containers(operation))
                if not live and not retained.get(operation):
                    self.db.execute("DELETE FROM leases WHERE resource=?", (resource,))
            op = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:10]
            resources = list(extra)
            if gpu is not None:
                resources.append(f"gpu:{gpu}")
            if robot:
                resources.append(f"robot:{robot}")
            requested = ports or {}
            allocated = {}
            held = []
            try:
                for label, fixed in requested.items():
                    candidates = [fixed] if fixed else range(18000, 19000)
                    for port in candidates:
                        if port in allocated.values():
                            if fixed:
                                raise ValueError(f"{label} port {port} is already allocated to another endpoint in this operation")
                            continue
                        owner = self.db.execute("SELECT operation FROM leases WHERE resource=?", (f"port:{port}",)).fetchone()
                        if owner:
                            if fixed:
                                raise ValueError(f"{label} port {port} is reserved by operation {owner[0]}")
                            continue
                        sock = socket.socket()
                        # HTTP servers permit reuse after an earlier connection
                        # enters TIME_WAIT. Probe the same way, while still
                        # rejecting a live listener and retaining exclusive leases.
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        try:
                            sock.bind(("127.0.0.1", port))
                        except OSError as exc:
                            sock.close()
                            if fixed:
                                raise ValueError(f"{label} port {port} is unavailable: {exc.strerror}") from exc
                            continue
                        held.append(sock)
                        allocated[label] = port
                        resources.append(f"port:{port}")
                        break
                    else:
                        raise ValueError(f"no available port for {label}")
                selected = None
                if need_domain:
                    domains = [domain] if domain is not None else range(20, 101)
                    selected = next((d for d in domains if not self.db.execute(
                        "SELECT 1 FROM leases WHERE resource=?", (f"dds:{d}",)).fetchone()), None)
                    if selected is None:
                        raise ValueError("DDS domain is occupied")
                    resources.append(f"dds:{selected}")
                for resource in resources:
                    self.db.execute("INSERT INTO leases (resource,operation,pid,identity,actor) VALUES (?,?,?,?,?)",
                                    (resource, op, os.getpid(), process_identity(os.getpid()), os.environ.get("GRAPHAPI_RESOURCE_ACTOR")))
                self.db.commit()
            finally:
                for sock in held:
                    sock.close()
            return op, allocated, selected
        except sqlite3.IntegrityError as exc:
            self.db.rollback()
            raise ValueError("requested GPU/robot/resource is owned by another GRAPH-API operation") from exc
        except BaseException:
            self.db.rollback()
            raise

    def release(self, operation):
        self.db.execute("DELETE FROM leases WHERE operation=?", (operation,))
        self.db.commit()

    def rows(self):
        result = []
        for path in sorted(self.operations.glob("*/status.json")):
            try:
                row = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            live = actor_alive(row["actor"], row.get("pid"), row.get("process_identity")) if row.get("actor") else alive(row.get("pid"), row.get("process_identity"))
            if row.get("state") in ("PREPARING", "RUNNING", "DRAINING") and not live:
                row["state"] = "FAILED"
                row["error"] = "supervisor exited without recording a terminal state"
            result.append(row)
        return sorted(result, key=lambda row: row.get("created", 0))

    def select(self, selector):
        rows = self.rows()
        if selector == "active":
            matches = [r for r in rows if r["state"] in ("PREPARING", "RUNNING", "DRAINING")]
            if len(matches) != 1:
                raise ValueError(f"active requires exactly one operation; found {len(matches)}")
            return matches[0]
        if selector in ("latest", "latest-started"):
            matches = [r for r in rows if selector == "latest-started" or (r["state"] == "COMPLETED" and r.get("bundles"))]
            if not matches:
                raise ValueError(f"no operation matches {selector}")
            return max(matches, key=lambda row: row.get("finished", 0)) if selector == "latest" else matches[-1]
        return next((r for r in rows if r["operation_id"] == selector), None) or self._missing(selector)

    @staticmethod
    def _missing(selector):
        raise ValueError(f"unknown operation: {selector}")
