"""On-disk cache. The whole point: face detection, landmarks, and
embeddings are the expensive operations; compute them once, persist, then
iterate the cheap scoring/selection/render stages dozens of times against
the cache. Without this the project is unusable to tune.

SQLite over a pile of JSON files because bucketing and selection are
queries, and one inspectable file beats thousands of tiny ones. Embeddings
and keypoints are stored as raw float32 BLOBs (np.tobytes / np.frombuffer).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from .records import Face, ImageRecord, Timestamp, TsSource


def _blob(a: np.ndarray | None) -> bytes | None:
    return None if a is None else np.ascontiguousarray(a, dtype=np.float32).tobytes()


def _arr(b: bytes | None, shape) -> np.ndarray | None:
    if b is None:
        return None
    return np.frombuffer(b, dtype=np.float32).reshape(shape)


SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    content_hash TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    width INTEGER, height INTEGER,
    ts_iso TEXT, ts_source INTEGER,
    decoded_ok INTEGER, error TEXT
);
CREATE TABLE IF NOT EXISTS faces (
    content_hash TEXT NOT NULL,
    face_id INTEGER NOT NULL,
    bbox BLOB, kps BLOB, embedding BLOB, dense BLOB,
    yaw REAL, pitch REAL, roll REAL, det_score REAL,
    interocular_px REAL, sharpness REAL, ear REAL, mar REAL,
    neutral_penalty REAL, id_sim REAL, id_margin REAL,
    is_self INTEGER DEFAULT 0,
    passes_pose INTEGER, passes_neutral INTEGER,
    composite REAL, selected INTEGER DEFAULT 0, bucket TEXT,
    PRIMARY KEY (content_hash, face_id),
    FOREIGN KEY (content_hash) REFERENCES images(content_hash)
);
CREATE TABLE IF NOT EXISTS seeds (
    path TEXT PRIMARY KEY,
    ts_iso TEXT,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_faces_self ON faces(is_self);
CREATE INDEX IF NOT EXISTS idx_faces_bucket ON faces(bucket);
"""

# Applied in order to bring PRAGMA user_version from i to i + 1.
MIGRATIONS = [
    """
    ALTER TABLE images ADD COLUMN id_outcome TEXT;
    CREATE TABLE promotions (
        content_hash TEXT NOT NULL,
        face_id INTEGER NOT NULL,
        PRIMARY KEY (content_hash, face_id)
    );
    """,
    "ALTER TABLE images ADD COLUMN phash TEXT;",
    """
    CREATE TABLE funnel (
        stage TEXT PRIMARY KEY,
        kept INTEGER, total INTEGER,
        reasons TEXT
    );
    """,
    """
    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
    CREATE TABLE seeds2 (
        person TEXT NOT NULL DEFAULT '', path TEXT, ts_iso TEXT, embedding BLOB,
        PRIMARY KEY (person, path)
    );
    INSERT INTO seeds2 SELECT '', path, ts_iso, embedding FROM seeds;
    DROP TABLE seeds;
    ALTER TABLE seeds2 RENAME TO seeds;
    CREATE TABLE promotions2 (
        person TEXT NOT NULL DEFAULT '', content_hash TEXT, face_id INTEGER,
        PRIMARY KEY (person, content_hash, face_id)
    );
    INSERT INTO promotions2 SELECT '', content_hash, face_id FROM promotions;
    DROP TABLE promotions;
    ALTER TABLE promotions2 RENAME TO promotions;
    CREATE TABLE funnel2 (
        person TEXT NOT NULL DEFAULT '', stage TEXT, kept INTEGER, total INTEGER, reasons TEXT,
        PRIMARY KEY (person, stage)
    );
    INSERT INTO funnel2 SELECT '', stage, kept, total, reasons FROM funnel;
    DROP TABLE funnel;
    ALTER TABLE funnel2 RENAME TO funnel;
    CREATE TABLE person_faces (
        person TEXT, content_hash TEXT, face_id INTEGER,
        is_self INTEGER, id_sim REAL, id_margin REAL, passes_pose INTEGER,
        passes_neutral INTEGER, neutral_penalty REAL, ear REAL, mar REAL,
        composite REAL, selected INTEGER, bucket TEXT,
        PRIMARY KEY (person, content_hash, face_id)
    );
    CREATE TABLE person_images (
        person TEXT, content_hash TEXT, id_outcome TEXT,
        PRIMARY KEY (person, content_hash)
    );
    """,
]

# Per-person results. The faces/images columns hold the active person's
# results; switching person saves them to person_faces/person_images and
# loads the other person's, so every query elsewhere stays person-agnostic.
PERSON_COLS = [
    "is_self",
    "id_sim",
    "id_margin",
    "passes_pose",
    "passes_neutral",
    "neutral_penalty",
    "ear",
    "mar",
    "composite",
    "selected",
    "bucket",
]


class Cache:
    def __init__(self, db_path: str, person: str = ""):
        """person namespaces seeds, identity results and selections, so one
        analyzed archive can yield one timelapse per person."""
        self.person = person
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        # Enable foreign key constraints and WAL mode for better concurrency
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(SCHEMA)
        self._migrate()
        self.db.commit()
        self._activate(person)

    def _activate(self, person: str) -> None:
        row = self.db.execute("SELECT value FROM meta WHERE key='active_person'").fetchone()
        active = row[0] if row else ""
        if active == person:
            return
        cols = ", ".join(PERSON_COLS)
        with self.db:
            self.db.execute("DELETE FROM person_faces WHERE person=?", (active,))
            self.db.execute(
                f"INSERT INTO person_faces SELECT ?, content_hash, face_id, {cols} FROM faces",
                (active,),
            )
            self.db.execute("DELETE FROM person_images WHERE person=?", (active,))
            self.db.execute(
                "INSERT INTO person_images SELECT ?, content_hash, id_outcome FROM images",
                (active,),
            )
            self.db.execute(
                "UPDATE faces SET is_self=0, id_sim=NULL, id_margin=NULL, passes_pose=NULL, "
                "passes_neutral=NULL, neutral_penalty=NULL, ear=NULL, mar=NULL, composite=NULL, "
                "selected=0, bucket=NULL"
            )
            self.db.execute("UPDATE images SET id_outcome=NULL")
            assign = ", ".join(f"{c}=p.{c}" for c in PERSON_COLS)
            self.db.execute(
                f"UPDATE faces SET {assign} FROM person_faces p WHERE p.person=? "
                "AND p.content_hash=faces.content_hash AND p.face_id=faces.face_id",
                (person,),
            )
            self.db.execute(
                "UPDATE images SET id_outcome=p.id_outcome FROM person_images p "
                "WHERE p.person=? AND p.content_hash=images.content_hash",
                (person,),
            )
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('active_person', ?)", (person,))

    def _migrate(self) -> None:
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        for i in range(version, len(MIGRATIONS)):
            self.db.executescript(MIGRATIONS[i])
            self.db.execute(f"PRAGMA user_version = {i + 1}")

    def close(self) -> None:
        self.db.close()

    # ---- images + faces ----
    def has_image(self, content_hash: str) -> bool:
        cur = self.db.execute("SELECT 1 FROM images WHERE content_hash=?", (content_hash,))
        return cur.fetchone() is not None

    def upsert_image(self, rec: ImageRecord) -> None:
        ts_iso = rec.ts.value.isoformat() if rec.ts.value else None
        self.db.execute(
            "INSERT OR REPLACE INTO images (content_hash, path, width, height, ts_iso, "
            "ts_source, decoded_ok, error) VALUES (?,?,?,?,?,?,?,?)",
            (
                rec.content_hash,
                rec.path,
                rec.width,
                rec.height,
                ts_iso,
                int(rec.ts.source),
                int(rec.decoded_ok),
                rec.error,
            ),
        )
        self.db.execute("DELETE FROM faces WHERE content_hash=?", (rec.content_hash,))
        for f in rec.faces:
            self._insert_face(f)
        self.db.commit()

    def _insert_face(self, f: Face) -> None:
        self.db.execute(
            """INSERT OR REPLACE INTO faces
               (content_hash, face_id, bbox, kps, embedding, dense,
                yaw, pitch, roll, det_score, interocular_px, sharpness,
                ear, mar, neutral_penalty, id_sim, id_margin, is_self,
                passes_pose, passes_neutral, composite, selected, bucket)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                f.content_hash,
                f.face_id,
                _blob(f.bbox),
                _blob(f.kps),
                _blob(f.embedding),
                _blob(f.dense),
                f.yaw,
                f.pitch,
                f.roll,
                f.det_score,
                f.interocular_px,
                f.sharpness,
                f.ear,
                f.mar,
                f.neutral_penalty,
                f.id_sim,
                f.id_margin,
                int(f.is_self),
                _int(f.passes_pose),
                _int(f.passes_neutral),
                f.composite,
                int(f.selected),
                f.bucket,
            ),
        )

    def update_face(self, f: Face) -> None:
        self._insert_face(f)
        self.db.commit()

    def update_faces(self, faces: list[Face]) -> None:
        with self.db:  # explicit transaction
            for f in faces:
                self._insert_face(f)

    def reset_scores(self) -> None:
        """Clear identity and gate results. Not committed: the caller's next
        update_faces commits it together with the new results."""
        self.db.execute(
            "UPDATE faces SET is_self=0, id_sim=NULL, id_margin=NULL, passes_pose=NULL, "
            "passes_neutral=NULL, neutral_penalty=NULL"
        )
        self.db.execute("UPDATE images SET id_outcome=NULL")

    def set_outcome(self, content_hash: str, outcome: str) -> None:
        """Not committed; written with the score results."""
        self.db.execute(
            "UPDATE images SET id_outcome=? WHERE content_hash=?", (outcome, content_hash)
        )

    def hashes_with_outcome(self, outcome: str) -> list[str]:
        rows = self.db.execute(
            "SELECT content_hash FROM images WHERE id_outcome=? ORDER BY ts_iso", (outcome,)
        )
        return [r[0] for r in rows]

    def phash(self, content_hash: str) -> str | None:
        r = self.db.execute("SELECT phash FROM images WHERE content_hash=?", (content_hash,))
        row = r.fetchone()
        return row[0] if row else None

    def set_phash(self, content_hash: str, value: str) -> None:
        with self.db:
            self.db.execute("UPDATE images SET phash=? WHERE content_hash=?", (value, content_hash))

    def record_stage(self, stage: str, kept: int, total: int, reasons: dict[str, int]) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO funnel VALUES (?,?,?,?,?)",
                (self.person, stage, kept, total, json.dumps(reasons)),
            )

    def stages(self) -> dict[str, tuple[int, int, dict[str, int]]]:
        """Analyze stages are shared ('' person); later stages are per person."""
        rows = self.db.execute(
            "SELECT stage, kept, total, reasons FROM funnel WHERE person IN ('', ?) "
            "ORDER BY person",
            (self.person,),
        )
        return {s: (k, t, json.loads(r)) for s, k, t, r in rows}

    def faces_of(self, content_hash: str) -> list[Face]:
        cur = self.db.execute("SELECT * FROM faces WHERE content_hash=?", (content_hash,))
        cols = [c[0] for c in cur.description]
        return [self._row_to_face(dict(zip(cols, r))) for r in cur.fetchall()]

    def add_promotions(self, keys: list[tuple[str, int]]) -> None:
        with self.db:
            self.db.executemany(
                "INSERT OR IGNORE INTO promotions VALUES (?,?,?)",
                [(self.person, h, i) for h, i in keys],
            )

    def promotions(self) -> set[tuple[str, int]]:
        rows = self.db.execute(
            "SELECT content_hash, face_id FROM promotions WHERE person=?", (self.person,)
        )
        return {(h, int(i)) for h, i in rows}

    def reset_selection(self) -> None:
        """Clear bucket winners. Not committed, same as reset_scores."""
        self.db.execute("UPDATE faces SET selected=0, bucket=NULL, composite=NULL")

    def all_faces(self, only_self: bool = False, only_selected: bool = False) -> list[Face]:
        # Build query with proper WHERE clause (safe since conds are hardcoded)
        q = "SELECT * FROM faces"
        conds = []
        if only_self:
            conds.append("is_self=1")
        if only_selected:
            conds.append("selected=1")
        if conds:
            q += " WHERE " + " AND ".join(conds)
        rows = self.db.execute(q).fetchall()
        cols = [c[0] for c in self.db.execute("SELECT * FROM faces LIMIT 0").description]
        return [self._row_to_face(dict(zip(cols, r))) for r in rows]

    def image_ts(self, content_hash: str) -> Timestamp:
        r = self.db.execute(
            "SELECT ts_iso, ts_source FROM images WHERE content_hash=?", (content_hash,)
        ).fetchone()
        if r is None:
            return Timestamp(None, TsSource.NONE)
        ts_iso, src = r
        val = datetime.fromisoformat(ts_iso) if ts_iso else None
        return Timestamp(val, TsSource(src))

    def image_path(self, content_hash: str) -> str:
        r = self.db.execute(
            "SELECT path FROM images WHERE content_hash=?", (content_hash,)
        ).fetchone()
        return r[0] if r else ""

    @staticmethod
    def _row_to_face(d: dict) -> Face:
        return Face(
            content_hash=d["content_hash"],
            face_id=d["face_id"],
            bbox=_arr(d["bbox"], (4,)),
            kps=_arr(d["kps"], (5, 2)),
            embedding=_arr(d["embedding"], (-1,)),
            dense=_arr(d["dense"], (-1, 2)) if d["dense"] else None,
            yaw=d["yaw"],
            pitch=d["pitch"],
            roll=d["roll"],
            det_score=d["det_score"],
            interocular_px=d["interocular_px"],
            sharpness=d["sharpness"],
            ear=d["ear"],
            mar=d["mar"],
            neutral_penalty=d["neutral_penalty"],
            id_sim=d["id_sim"],
            id_margin=d["id_margin"],
            is_self=bool(d["is_self"]),
            passes_pose=_bool(d["passes_pose"]),
            passes_neutral=_bool(d["passes_neutral"]),
            composite=d["composite"],
            selected=bool(d["selected"]),
            bucket=d["bucket"],
        )

    # ---- seeds (reference chain) ----
    def add_seed(self, path: str, ts: datetime | None, embedding: np.ndarray) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO seeds VALUES (?,?,?,?)",
            (self.person, path, ts.isoformat() if ts else None, _blob(embedding)),
        )
        self.db.commit()

    def seeds(self) -> list[tuple[str, datetime | None, np.ndarray]]:
        out = []
        rows = self.db.execute(
            "SELECT path, ts_iso, embedding FROM seeds WHERE person=?", (self.person,)
        )
        for path, ts_iso, emb in rows:
            out.append((path, datetime.fromisoformat(ts_iso) if ts_iso else None, _arr(emb, (-1,))))
        return out


def _int(b: bool | None) -> int | None:
    return None if b is None else int(b)


def _bool(v: int | None) -> bool | None:
    return None if v is None else bool(v)
