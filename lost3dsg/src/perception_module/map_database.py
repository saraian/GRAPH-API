"""
map_database.py
---------------
Persistenza temporale per ObjectManagerNode (TIAGo).
Salva ogni cambiamento con timestamp — nessuna sovrascrittura.
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
                DROP TABLE IF EXISTS object_history;
                DROP TABLE IF EXISTS objects;

                CREATE TABLE IF NOT EXISTS objects (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
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
                    object_id   INTEGER NOT NULL,
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

    def _find_active(self, conn, label, color, material):
        """Cerca l'oggetto attivo con match ESATTO su label+color+material."""
        conn.row_factory = sqlite3.Row
        return conn.execute(
            """SELECT * FROM objects
               WHERE label=? AND color=? AND material=? AND is_active=1
               ORDER BY last_seen DESC LIMIT 1""",
            (label, color or "", material or "")
        ).fetchone()

    # ------------------------------------------------------------------ #
    #  4 METODI DA CHIAMARE NEL TUO NODO                                  #
    # ------------------------------------------------------------------ #

    def on_new_object(self, obj, phase: str = "exploration", step: int = 0):
        """Chiama in add_new_object() — registra un nuovo oggetto."""
        now = datetime.now().isoformat()
        x, y, z = self._centroid(obj.bbox)
        bbox_json = json.dumps(obj.bbox) if obj.bbox else None
        
        # Estrai la stanza dall'oggetto
        room = getattr(obj, 'room_id', 'unknown')

        with sqlite3.connect(self.db_path) as conn:
            obj_id = conn.execute(
                """INSERT INTO objects
                   (label,color,material,description,x,y,z,bbox_json,first_seen,last_seen,last_event,room_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,'detected',?)""",
                (obj.label, obj.color or "", obj.material or "",
                 obj.description or "", x, y, z, bbox_json, now, now, room)
            ).lastrowid

            conn.execute(
                """INSERT INTO object_history
                   (object_id,label,color,timestamp,event_type,phase,step,
                    x_new,y_new,z_new,bbox_new,room_id)
                   VALUES (?,?,?,?,'detected',?,?,?,?,?,?,?)""",
                (obj_id, obj.label, obj.color or "", now, phase, step,
                 x, y, z, bbox_json, room)
            )

        print(f"[MapDB] ✅ NEW [{phase}] '{obj.label}' in {room} @ ({x:.2f},{y:.2f},{z:.2f})")

    def on_object_moved(self, obj, old_bbox: dict, new_bbox: dict,
                        distance: float, iou: float,
                        phase: str = "tracking", step: int = 0):
        """Chiama in modify_existing_object() — caso IoU bassa (oggetto spostato)."""
        now = datetime.now().isoformat()
        x_old, y_old, z_old = self._centroid(old_bbox)
        x_new, y_new, z_new = self._centroid(new_bbox)
        room = getattr(obj, 'room_id', 'unknown')

        with sqlite3.connect(self.db_path) as conn:
            row = self._find_active(conn, obj.label, obj.color, obj.material)
            if not row:
                print(f"[MapDB] ⚠️  on_object_moved: '{obj.label}' non trovato nel DB, salto.")
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
                   (object_id,label,color,timestamp,event_type,phase,step,
                    x_old,y_old,z_old,x_new,y_new,z_new,
                    distance,iou,bbox_old,bbox_new,room_id)
                   VALUES (?,?,?,?,'moved',?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (obj_id, obj.label, obj.color or "", now, phase, step,
                 x_old, y_old, z_old, x_new, y_new, z_new,
                 distance, iou, json.dumps(old_bbox), json.dumps(new_bbox), room)
            )

        print(f"[MapDB] 🔄 MOVED '{obj.label}'  Δ={distance:.2f}m  IoU={iou:.2f}")

    def on_object_deleted(self, obj, reason: str = "",
                          phase: str = "tracking", step: int = 0):
        """Chiama in delete_undetected_objects() e delete_uncertain_objects()."""
        now = datetime.now().isoformat()
        x, y, z = self._centroid(obj.bbox)
        room = getattr(obj, 'room_id', 'unknown')

        with sqlite3.connect(self.db_path) as conn:
            row = self._find_active(conn, obj.label, obj.color, obj.material)
            if not row:
                print(f"[MapDB] ⚠️  on_object_deleted: '{obj.label}' non trovato nel DB, salto.")
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
                   (object_id,label,color,timestamp,event_type,phase,step,
                    x_old,y_old,z_old,bbox_old,notes,room_id)
                   VALUES (?,?,?,?,'disappeared',?,?,?,?,?,?,?,?)""",
                (obj_id, obj.label, obj.color or "", now, phase, step,
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
        """Chiama quando aggiungi a uncertain_objects in modify_existing_object()."""
        now = datetime.now().isoformat()
        room = getattr(obj, 'room_id', 'unknown')
        
        with sqlite3.connect(self.db_path) as conn:
            row = self._find_active(conn, obj.label, obj.color, obj.material)
            if not row:
                print(f"[MapDB] ⚠️  on_uncertain_added: '{obj.label}' non trovato nel DB, salto.")
                return
            obj_id = row["id"]
            conn.execute(
                "UPDATE objects SET is_uncertain=1,last_seen=? WHERE id=?",
                (now, obj_id)
            )
            conn.execute(
                """INSERT INTO object_history
                   (object_id,label,color,timestamp,event_type,phase,step,notes,room_id)
                   VALUES (?,?,?,?,'uncertain_added','tracking',?,'large displacement',?)""",
                (obj_id, obj.label, obj.color or "", now, step, room)
            )
        print(f"[MapDB] ⚠️  UNCERTAIN '{obj.label}'")