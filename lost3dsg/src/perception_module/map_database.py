"""
map_database.py
---------------
Persistenza temporale per ObjectManagerNode (TIAGo).
Records every change with a timestamp; nothing is overwritten.
Include il supporto per la Mappatura Semantica Topologica (room_id).
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

from hooks import Store


class MapDatabase(Store):
    """Default `hooks.Store`: the SQLite temporal map. Another backend (config
    `hooks.store`) subclasses Store and receives the same four events."""
    name = "sqlite"

    def __init__(self, db_path: str):
        """
        db_path: percorso ASSOLUTO del file .db — sempre passato dall'esterno.
        Esempio: os.path.join(log_dir, "tiago_temporal_map.db")
        """
        self.db_path = str(Path(db_path).expanduser().resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        print(f"[MapDB] ✅ Database: {self.db_path}")

    # ------------------------------------------------------------------ #
    #  SETUP                                                               #
    # ------------------------------------------------------------------ #

    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS objects (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_uuid  TEXT,
                    label        TEXT NOT NULL,
                    color        TEXT DEFAULT '',
                    material     TEXT DEFAULT '',
                    description  TEXT DEFAULT '',
                    x            REAL,
                    y            REAL,
                    z            REAL,
                    bbox_json    TEXT,
                    is_active    INTEGER DEFAULT 1,
                    is_uncertain INTEGER DEFAULT 0,
                    first_seen   TEXT NOT NULL,
                    last_seen    TEXT NOT NULL,
                    last_event   TEXT DEFAULT 'detected',
                    room_id      TEXT DEFAULT 'unknown'  -- <--- NUOVA COLONNA STANZA
                );

                CREATE TABLE IF NOT EXISTS object_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    -- object_id is the objects.id ROW NUMBER, not the object's
                    -- identity. It is kept because history() and query_map.storia()
                    -- are called with it (tests/test_store.py, hooks.py). GA-44 adds
                    -- the durable identity beside it rather than repointing it,
                    -- because changing a value readers already use breaks them
                    -- without an error.
                    object_id   INTEGER NOT NULL,
                    object_uuid TEXT,
                    label       TEXT NOT NULL DEFAULT '',
                    color       TEXT NOT NULL DEFAULT '',
                    timestamp   TEXT NOT NULL,
                    event_type  TEXT NOT NULL,
                    phase       TEXT DEFAULT 'exploration',
                    step        INTEGER DEFAULT 0,
                    x_old REAL, y_old REAL, z_old REAL,
                    x_new REAL, y_new REAL, z_new REAL,
                    distance    REAL,
                    iou         REAL,
                    bbox_old    TEXT,
                    bbox_new    TEXT,
                    notes       TEXT DEFAULT '',
                    room_id     TEXT DEFAULT 'unknown',  -- <--- NUOVA COLONNA STANZA
                    FOREIGN KEY(object_id) REFERENCES objects(id)
                );

                CREATE INDEX IF NOT EXISTS idx_obj_label  ON objects(label);
                CREATE INDEX IF NOT EXISTS idx_hist_obj   ON object_history(object_id);
                CREATE INDEX IF NOT EXISTS idx_hist_time  ON object_history(timestamp);
                CREATE INDEX IF NOT EXISTS idx_hist_event ON object_history(event_type);
            """)

            # GA-43 removed the DROP TABLE pair that used to precede the CREATEs, so
            # this file now SURVIVES across constructions -- which is what the module
            # docstring promised all along. The drops were also the only thing keeping
            # the schema current: CREATE TABLE IF NOT EXISTS does nothing to a table
            # that already exists, so a database written before GA-44 would keep the
            # old object_history and the next insert would fail with
            # "no such column: object_uuid". Migrate it forward instead.
            cols = {r[1] for r in conn.execute("PRAGMA table_info(object_history)")}
            if cols and "object_uuid" not in cols:
                conn.execute("ALTER TABLE object_history ADD COLUMN object_uuid TEXT")
                # Backfill from the row number while it still resolves. After this the
                # history keeps a durable identity even if the rowids are ever reused.
                conn.execute("""UPDATE object_history
                                   SET object_uuid = (SELECT o.object_uuid FROM objects o
                                                      WHERE o.id = object_history.object_id)
                                 WHERE object_uuid IS NULL""")
                print("[MapDB] migrated object_history: added object_uuid and backfilled")

            # AFTER the migration, never inside the CREATE script above: on a
            # pre-GA-44 database the table exists without the column, and indexing a
            # column that is not there yet raises before the migration can run.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hist_uuid ON object_history(object_uuid)"
            )

    # ------------------------------------------------------------------ #
    #  HELPERS                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _centroid(bbox: dict):
        if not bbox:
            return None, None, None
        return (
            (bbox["x_min"] + bbox["x_max"]) / 2.0,
            (bbox["y_min"] + bbox["y_max"]) / 2.0,
            (bbox["z_min"] + bbox["z_max"]) / 2.0,
        )

    def _find_active(self, conn, obj):
        """Find an active row by immutable object ID, with legacy fallback."""
        conn.row_factory = sqlite3.Row
        object_uuid = getattr(obj, 'object_id', None)
        if object_uuid:
            row = conn.execute(
                """SELECT * FROM objects
                   WHERE object_uuid=? AND is_active=1
                   ORDER BY last_seen DESC LIMIT 1""",
                (object_uuid,)
            ).fetchone()
            if row:
                return row
        return conn.execute(
            """SELECT * FROM objects
               WHERE label=? AND color=? AND material=? AND is_active=1
               ORDER BY last_seen DESC LIMIT 1""",
            (obj.label, obj.color or "", obj.material or "")
        ).fetchone()

    # ------------------------------------------------------------------ #
    #  4 METODI DA CHIAMARE NEL TUO NODO                                  #
    # ------------------------------------------------------------------ #

    def on_new_object(self, obj, phase: str = "exploration", step: int = 0):
        """Called from add_new_object() — records a new object."""
        now = datetime.now().isoformat()
        x, y, z = self._centroid(obj.bbox)
        bbox_json = json.dumps(obj.bbox) if obj.bbox else None
        
        # Take the room from the object
        room = getattr(obj, 'room_id', 'unknown')

        with sqlite3.connect(self.db_path) as conn:
            obj_id = conn.execute(
                """INSERT INTO objects
                   (object_uuid,label,color,material,description,x,y,z,bbox_json,first_seen,last_seen,last_event,room_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,'detected',?)""",
                (getattr(obj, 'object_id', None), obj.label, obj.color or "", obj.material or "",
                 obj.description or "", x, y, z, bbox_json, now, now, room)
            ).lastrowid

            conn.execute(
                """INSERT INTO object_history
                   (object_id,object_uuid,label,color,timestamp,event_type,phase,step,
                    x_new,y_new,z_new,bbox_new,room_id)
                   VALUES (?,?,?,?,?,'detected',?,?,?,?,?,?,?)""",
                (obj_id, getattr(obj, 'object_id', None), obj.label, obj.color or "", now, phase, step,
                 x, y, z, bbox_json, room)
            )

        print(f"[MapDB] ✅ NEW [{phase}] '{obj.label}' in {room} @ ({x:.2f},{y:.2f},{z:.2f})")

    def on_object_moved(self, obj, old_bbox: dict, new_bbox: dict,
                        distance: float, iou: float,
                        phase: str = "tracking", step: int = 0):
        """Called from modify_existing_object() — the low-IoU case (object moved)."""
        now = datetime.now().isoformat()
        x_old, y_old, z_old = self._centroid(old_bbox)
        x_new, y_new, z_new = self._centroid(new_bbox)
        room = getattr(obj, 'room_id', 'unknown')

        with sqlite3.connect(self.db_path) as conn:
            row = self._find_active(conn, obj)
            if not row:
                print(f"[MapDB] ⚠️  on_object_moved: '{obj.label}' not found in the DB, skipping.")
                return
            obj_id = row["id"]
            conn.execute(
                """UPDATE objects
                   SET x=?,y=?,z=?,bbox_json=?,last_seen=?,last_event='moved'
                   WHERE id=?""",
                (x_new, y_new, z_new, json.dumps(new_bbox), now, obj_id)
            )
            conn.execute(
                """INSERT INTO object_history
                   (object_id,object_uuid,label,color,timestamp,event_type,phase,step,
                    x_old,y_old,z_old,x_new,y_new,z_new,
                    distance,iou,bbox_old,bbox_new,room_id)
                   VALUES (?,?,?,?,?,'moved',?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (obj_id, row["object_uuid"], obj.label, obj.color or "", now, phase, step,
                 x_old, y_old, z_old, x_new, y_new, z_new,
                 distance, iou, json.dumps(old_bbox), json.dumps(new_bbox), room)
            )

        print(f"[MapDB] 🔄 MOVED '{obj.label}'  Δ={distance:.2f}m  IoU={iou:.2f}")

    def on_object_room_changed(self, obj, old_room, new_room, step: int = 0):
        """Persist a room reassignment caused by updated room geometry."""
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            row = self._find_active(conn, obj)
            if not row:
                return
            conn.execute(
                "UPDATE objects SET room_id=?, last_seen=?, last_event='room_reassigned' WHERE id=?",
                (new_room, now, row['id'])
            )
            conn.execute(
                """INSERT INTO object_history
                   (object_id,object_uuid,label,color,timestamp,event_type,phase,step,notes,room_id)
                   VALUES (?,?,?,?,?,'room_reassigned','tracking',?,?,?)""",
                (row['id'], row['object_uuid'], obj.label, obj.color or "", now,
                 step, f"{old_room} -> {new_room}", new_room)
            )

    def on_object_deleted(self, obj, reason: str = "",
                          phase: str = "tracking", step: int = 0):
        """Called from delete_undetected_objects() and delete_uncertain_objects()."""
        now = datetime.now().isoformat()
        x, y, z = self._centroid(obj.bbox)
        room = getattr(obj, 'room_id', 'unknown')

        with sqlite3.connect(self.db_path) as conn:
            row = self._find_active(conn, obj)
            if not row:
                print(f"[MapDB] ⚠️  on_object_deleted: '{obj.label}' not found in the DB, skipping.")
                return
            obj_id = row["id"]
            conn.execute(
                """UPDATE objects
                   SET is_active=0,last_seen=?,last_event='disappeared'
                   WHERE id=?""",
                (now, obj_id)
            )
            conn.execute(
                """INSERT INTO object_history
                   (object_id,object_uuid,label,color,timestamp,event_type,phase,step,
                    x_old,y_old,z_old,bbox_old,notes,room_id)
                   VALUES (?,?,?,?,?,'disappeared',?,?,?,?,?,?,?,?)""",
                (obj_id, row["object_uuid"], obj.label, obj.color or "", now, phase, step,
                 x, y, z, json.dumps(obj.bbox), reason, room)
            )

        print(f"[MapDB] ❌ DELETED '{obj.label}' ({reason})")

    # ------------------------------------------------------------------ #
    #  READS (the Store contract; same rows query_map.py reads offline)    #
    # ------------------------------------------------------------------ #

    def objects(self, only_active: bool = True):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM objects" + (" WHERE is_active=1" if only_active else "")).fetchall()
        return [{"id": r["id"], "label": r["label"], "color": r["color"], "material": r["material"],
                 "description": r["description"], "bbox": json.loads(r["bbox_json"]) if r["bbox_json"] else None,
                 "room_id": r["room_id"], "is_active": bool(r["is_active"]), "is_uncertain": bool(r["is_uncertain"]),
                 "first_seen": r["first_seen"], "last_seen": r["last_seen"], "last_event": r["last_event"]}
                for r in rows]

    def history(self, object_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM object_history WHERE object_id=? ORDER BY timestamp", (object_id,)).fetchall()
        return [{"timestamp": r["timestamp"], "event_type": r["event_type"], "phase": r["phase"], "step": r["step"],
                 "bbox_old": json.loads(r["bbox_old"]) if r["bbox_old"] else None,
                 "bbox_new": json.loads(r["bbox_new"]) if r["bbox_new"] else None,
                 "distance": r["distance"], "iou": r["iou"], "notes": r["notes"]} for r in rows]

    def on_uncertain_added(self, obj, step: int = 0):
        """Called when adding to uncertain_objects in modify_existing_object()."""
        now = datetime.now().isoformat()
        room = getattr(obj, 'room_id', 'unknown')
        
        with sqlite3.connect(self.db_path) as conn:
            row = self._find_active(conn, obj)
            if not row:
                print(f"[MapDB] ⚠️  on_uncertain_added: '{obj.label}' not found in the DB, skipping.")
                return
            obj_id = row["id"]
            conn.execute(
                "UPDATE objects SET is_uncertain=1,last_seen=? WHERE id=?",
                (now, obj_id)
            )
            conn.execute(
                """INSERT INTO object_history
                   (object_id,object_uuid,label,color,timestamp,event_type,phase,step,notes,room_id)
                   VALUES (?,?,?,?,?,'uncertain_added','tracking',?,'large displacement',?)""",
                (obj_id, row["object_uuid"], obj.label, obj.color or "", now, step, room)
            )
        print(f"[MapDB] ⚠️  UNCERTAIN '{obj.label}'")
