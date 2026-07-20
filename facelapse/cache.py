"""On-disk cache. The whole point: face detection, landmarks, and
embeddings are the expensive operations; compute them once, persist, then
iterate the cheap scoring/selection/render stages dozens of times against
the cache. Without this the project is unusable to tune.

SQLite over a pile of JSON files because bucketing and selection are
queries, and one inspectable file beats thousands of tiny ones. Embeddings
and keypoints are stored as raw float32 BLOBs (np.tobytes / np.frombuffer).
"""
from __future__ import annotations

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


class Cache:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # ---- images + faces ----
    def has_image(self, content_hash: str) -> bool:
        cur = self.db.execute("SELECT 1 FROM images WHERE content_hash=?", (content_hash,))
        return cur.fetchone() is not None

    def upsert_image(self, rec: ImageRecord) -> None:
        ts_iso = rec.ts.value.isoformat() if rec.ts.value else None
        self.db.execute(
            "INSERT OR REPLACE INTO images VALUES (?,?,?,?,?,?,?,?)",
            (rec.content_hash, rec.path, rec.width, rec.height,
             ts_iso, int(rec.ts.source), int(rec.decoded_ok), rec.error),
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
            (f.content_hash, f.face_id, _blob(f.bbox), _blob(f.kps),
             _blob(f.embedding), _blob(f.dense),
             f.yaw, f.pitch, f.roll, f.det_score,
             f.interocular_px, f.sharpness, f.ear, f.mar, f.neutral_penalty,
             f.id_sim, f.id_margin, int(f.is_self),
             _int(f.passes_pose), _int(f.passes_neutral),
             f.composite, int(f.selected), f.bucket),
        )

    def update_face(self, f: Face) -> None:
        self._insert_face(f)
        self.db.commit()

    def update_faces(self, faces: list[Face]) -> None:
        for f in faces:
            self._insert_face(f)
        self.db.commit()

    def all_faces(self, only_self: bool = False, only_selected: bool = False) -> list[Face]:
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
        r = self.db.execute("SELECT path FROM images WHERE content_hash=?", (content_hash,)).fetchone()
        return r[0] if r else ""

    @staticmethod
    def _row_to_face(d: dict) -> Face:
        return Face(
            content_hash=d["content_hash"], face_id=d["face_id"],
            bbox=_arr(d["bbox"], (4,)), kps=_arr(d["kps"], (5, 2)),
            embedding=_arr(d["embedding"], (-1,)),
            dense=_arr(d["dense"], (-1, 2)) if d["dense"] else None,
            yaw=d["yaw"], pitch=d["pitch"], roll=d["roll"], det_score=d["det_score"],
            interocular_px=d["interocular_px"], sharpness=d["sharpness"],
            ear=d["ear"], mar=d["mar"], neutral_penalty=d["neutral_penalty"],
            id_sim=d["id_sim"], id_margin=d["id_margin"], is_self=bool(d["is_self"]),
            passes_pose=_bool(d["passes_pose"]), passes_neutral=_bool(d["passes_neutral"]),
            composite=d["composite"], selected=bool(d["selected"]), bucket=d["bucket"],
        )

    # ---- seeds (reference chain) ----
    def add_seed(self, path: str, ts: datetime | None, embedding: np.ndarray) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO seeds VALUES (?,?,?)",
            (path, ts.isoformat() if ts else None, _blob(embedding)),
        )
        self.db.commit()

    def seeds(self) -> list[tuple[str, datetime | None, np.ndarray]]:
        out = []
        for path, ts_iso, emb in self.db.execute("SELECT path, ts_iso, embedding FROM seeds"):
            out.append((path, datetime.fromisoformat(ts_iso) if ts_iso else None,
                        _arr(emb, (-1,))))
        return out


def _int(b: bool | None) -> int | None:
    return None if b is None else int(b)


def _bool(v) -> bool | None:
    return None if v is None else bool(v)
