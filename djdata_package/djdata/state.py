"""Run state in one sqlite file: mixes, tracks, seams and their statuses. This is the resume log.

Statuses
  mixes : pending → downloading → windows_ready | failed
  tracks: pending → downloading → ready | failed
  seams : pending → ready (window + both tracks on disk) → analysing → done | failed

Archival: a track file is safe to move once every seam that references it is done or failed; a
window file once its seam is done or failed. `archivable()` lists them, `mark_archived()` records
the move so nothing looks for the file again.
"""

import json
from pathlib import Path
import re
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS mixes (
  mix_id TEXT PRIMARY KEY, title TEXT, url TEXT, source TEXT, year INTEGER, genres TEXT,
  status TEXT NOT NULL DEFAULT 'pending', path TEXT, error TEXT, worker TEXT, updated REAL);
CREATE TABLE IF NOT EXISTS tracks (
  track_id TEXT PRIMARY KEY, title TEXT, url TEXT, duration REAL,
  status TEXT NOT NULL DEFAULT 'pending', path TEXT, error TEXT, worker TEXT, updated REAL, archived INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS seams (
  seam_id TEXT PRIMARY KEY, mix_id TEXT NOT NULL, a TEXT NOT NULL, b TEXT NOT NULL, tier TEXT, source TEXT,
  coarse TEXT NOT NULL,           -- json: anchors, rates, overlap, window
  status TEXT NOT NULL DEFAULT 'pending', window_path TEXT, error TEXT, worker TEXT, updated REAL, archived INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS seams_mix ON seams(mix_id);
CREATE INDEX IF NOT EXISTS seams_status ON seams(status);
"""


class State:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._local = threading.local()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=60, isolation_level=None, check_same_thread=False)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA busy_timeout=60000")
            c.row_factory = sqlite3.Row
            self._local.conn = c
        return c

    # ── manifest loading ────────────────────────────────────────────────────────
    def add_mix(self, mix_id, title, url, source, year, genres):
        self._conn().execute(
            "INSERT OR IGNORE INTO mixes(mix_id,title,url,source,year,genres,updated) VALUES(?,?,?,?,?,?,?)",
            (mix_id, title, url, source, year, json.dumps(genres), time.time()))

    def add_track(self, track_id, title, url, duration):
        self._conn().execute(
            "INSERT OR IGNORE INTO tracks(track_id,title,url,duration,updated) VALUES(?,?,?,?,?)",
            (track_id, title, url, duration, time.time()))

    def add_seam(self, seam_id, mix_id, a, b, tier, source, coarse: dict):
        self._conn().execute(
            "INSERT OR IGNORE INTO seams(seam_id,mix_id,a,b,tier,source,coarse,updated) VALUES(?,?,?,?,?,?,?,?)",
            (seam_id, mix_id, a, b, tier, source, json.dumps(coarse), time.time()))

    # ── claims: one worker takes one item, atomically ───────────────────────────
    def claim_mix(self, tiers: list, worker: str):
        """Next pending mix that has at least one seam in the run tiers, oldest first."""
        c = self._conn()
        c.execute("BEGIN IMMEDIATE")
        row = c.execute(
            "SELECT m.* FROM mixes m WHERE m.status='pending' AND EXISTS ("
            " SELECT 1 FROM seams s WHERE s.mix_id=m.mix_id AND s.tier IN (%s)) ORDER BY m.updated LIMIT 1"
            % ",".join("?" * len(tiers)), tiers).fetchone()
        if row:
            c.execute("UPDATE mixes SET status='downloading', worker=?, updated=? WHERE mix_id=?",
                      (worker, time.time(), row["mix_id"]))
        c.execute("COMMIT")
        return dict(row) if row else None

    def claim_track(self, tiers: list, worker: str):
        """Next pending track needed by a seam of a mix whose windows are ready, in mix order."""
        c = self._conn()
        c.execute("BEGIN IMMEDIATE")
        row = c.execute(
            "SELECT t.* FROM tracks t JOIN seams s ON (s.a=t.track_id OR s.b=t.track_id)"
            " JOIN mixes m ON m.mix_id=s.mix_id"
            " WHERE t.status='pending' AND m.status='windows_ready' AND s.tier IN (%s)"
            " ORDER BY m.updated LIMIT 1" % ",".join("?" * len(tiers)), tiers).fetchone()
        if row:
            c.execute("UPDATE tracks SET status='downloading', worker=?, updated=? WHERE track_id=?",
                      (worker, time.time(), row["track_id"]))
        c.execute("COMMIT")
        return dict(row) if row else None

    def claim_seam(self, tiers: list, worker: str):
        """Next seam whose window is cut and both tracks are on disk."""
        c = self._conn()
        c.execute("BEGIN IMMEDIATE")
        row = c.execute(
            "SELECT s.*, ta.path AS a_path, tb.path AS b_path FROM seams s"
            " JOIN tracks ta ON ta.track_id=s.a JOIN tracks tb ON tb.track_id=s.b"
            " WHERE s.status='ready' AND ta.status='ready' AND tb.status='ready' AND s.tier IN (%s)"
            " ORDER BY s.updated LIMIT 1" % ",".join("?" * len(tiers)), tiers).fetchone()
        if row:
            c.execute("UPDATE seams SET status='analysing', worker=?, updated=? WHERE seam_id=?",
                      (worker, time.time(), row["seam_id"]))
        c.execute("COMMIT")
        return dict(row) if row else None

    # ── status updates ──────────────────────────────────────────────────────────
    def set_mix(self, mix_id, status, path=None, error=None):
        self._conn().execute("UPDATE mixes SET status=?, path=COALESCE(?,path), error=?, updated=? WHERE mix_id=?",
                             (status, path, error, time.time(), mix_id))

    def set_track(self, track_id, status, path=None, error=None, duration=None):
        self._conn().execute(
            "UPDATE tracks SET status=?, path=COALESCE(?,path), error=?, duration=COALESCE(?,duration), updated=? WHERE track_id=?",
            (status, path, error, duration, time.time(), track_id))

    def set_seam(self, seam_id, status, window_path=None, error=None):
        self._conn().execute(
            "UPDATE seams SET status=?, window_path=COALESCE(?,window_path), error=?, updated=? WHERE seam_id=?",
            (status, window_path, error, time.time(), seam_id))

    def seams_of_mix(self, mix_id) -> list[dict]:
        return [dict(r) for r in self._conn().execute("SELECT * FROM seams WHERE mix_id=?", (mix_id,))]

    def release_stale(self):
        """Items left 'downloading'/'analysing' by a killed run go back to the queue. Called at start."""
        c = self._conn()
        c.execute("UPDATE mixes SET status='pending' WHERE status='downloading'")
        c.execute("UPDATE tracks SET status='pending' WHERE status='downloading'")
        c.execute("UPDATE seams SET status='ready' WHERE status='analysing'")

    def retry_tracks(self, match: str) -> dict:
        """Failed tracks whose error contains `match` go back to pending, and the seams they failed
        go back to ready (window cut) or pending, unless the seam's other track is failed too."""
        c = self._conn()
        like = f"%{match}%"
        c.execute("BEGIN IMMEDIATE")
        n_tracks = c.execute("UPDATE tracks SET status='pending', error=NULL, updated=? WHERE status='failed' AND error LIKE ?",
                             (time.time(), like)).rowcount
        n_seams = c.execute(
            "UPDATE seams SET status=CASE WHEN window_path IS NULL THEN 'pending' ELSE 'ready' END, error=NULL, updated=?"
            " WHERE status='failed' AND error LIKE ? AND NOT EXISTS ("
            "  SELECT 1 FROM tracks t WHERE t.track_id IN (seams.a, seams.b) AND t.status='failed')",
            (time.time(), "track: " + like)).rowcount
        c.execute("COMMIT")
        return {"tracks": n_tracks, "seams": n_seams}

    # ── queries ─────────────────────────────────────────────────────────────────
    def counts(self, tiers: list | None = None) -> dict:
        c = self._conn()
        out = {}
        for table in ("mixes", "tracks", "seams"):
            if table == "seams" and tiers:
                rows = c.execute("SELECT status, COUNT(*) n FROM seams WHERE tier IN (%s) GROUP BY status"
                                 % ",".join("?" * len(tiers)), tiers)
            else:
                rows = c.execute(f"SELECT status, COUNT(*) n FROM {table} GROUP BY status")
            out[table] = {r["status"]: r["n"] for r in rows}
        return out

    def failure_summary(self, tiers: list, top: int = 8) -> dict:
        """Failed and waiting items grouped by error text (ids stripped), most common first.
        `waiting` = pending tracks/mixes that carry an error: refused by YouTube, retried when the gate reopens."""
        c = self._conn()
        q = ",".join("?" * len(tiers))

        def bucket(err):
            err = re.sub(r"\[youtube\] [\w-]+: ", "[youtube] ", err or "")
            err = re.sub(r"\bmix\d+\b", "mix", err)
            return err[:90]

        def group(rows):
            counts = {}
            for (e,) in rows:
                counts[bucket(e)] = counts.get(bucket(e), 0) + 1
            return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:top])

        return {
            "tracks_failed": group(c.execute("SELECT error FROM tracks WHERE status='failed'")),
            "tracks_waiting": group(c.execute("SELECT error FROM tracks WHERE status='pending' AND error IS NOT NULL")),
            "mixes_failed": group(c.execute("SELECT error FROM mixes WHERE status='failed'")),
            "seams_failed": group(c.execute("SELECT error FROM seams WHERE status='failed' AND tier IN (%s)" % q, tiers)),
        }

    def pending_unanalysed_mixes(self, tiers: list) -> int:
        """Mixes downloaded whose seams (in the run tiers) are not all finished. Gates the mix worker."""
        return self._conn().execute(
            "SELECT COUNT(*) FROM mixes m WHERE m.status='windows_ready' AND EXISTS ("
            " SELECT 1 FROM seams s WHERE s.mix_id=m.mix_id AND s.tier IN (%s) AND s.status NOT IN ('done','failed'))"
            % ",".join("?" * len(tiers)), tiers).fetchone()[0]

    def work_left(self, tiers: list) -> bool:
        c = self._conn()
        q = ",".join("?" * len(tiers))
        seams = c.execute("SELECT COUNT(*) FROM seams WHERE tier IN (%s) AND status NOT IN ('done','failed')" % q, tiers).fetchone()[0]
        return seams > 0

    def archivable(self, tiers: list) -> dict:
        c = self._conn()
        q = ",".join("?" * len(tiers))
        tracks = c.execute(
            "SELECT t.track_id, t.path FROM tracks t WHERE t.status='ready' AND t.archived=0 AND t.path IS NOT NULL"
            " AND NOT EXISTS (SELECT 1 FROM seams s WHERE (s.a=t.track_id OR s.b=t.track_id) AND s.tier IN (%s)"
            "                 AND s.status NOT IN ('done','failed'))" % q, tiers).fetchall()
        windows = c.execute(
            "SELECT seam_id, window_path FROM seams WHERE status IN ('done','failed') AND archived=0 AND window_path IS NOT NULL"
            " AND tier IN (%s)" % q, tiers).fetchall()
        return {"tracks": [dict(r) for r in tracks], "windows": [dict(r) for r in windows]}

    def mark_archived(self, track_ids: list, seam_ids: list):
        c = self._conn()
        c.executemany("UPDATE tracks SET archived=1 WHERE track_id=?", [(t,) for t in track_ids])
        c.executemany("UPDATE seams SET archived=1 WHERE seam_id=?", [(s,) for s in seam_ids])

    def seam_rows(self, tiers: list, status: str = "done") -> list[dict]:
        q = ",".join("?" * len(tiers))
        return [dict(r) for r in self._conn().execute(
            "SELECT * FROM seams WHERE status=? AND tier IN (%s)" % q, [status, *tiers])]
