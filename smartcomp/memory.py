"""Persistent memory: sessions, learned skills, observations and tool stats.

Three things live here, and each exists to make the next run cheaper:

* **Sessions** own their screenshots. When a session ends its image directory is
  deleted, so a daemon that takes a screenshot per step cannot fill the disk.
* **Skills** are named action sequences distilled from what actually worked.
  A run that clicked through a flow can save it and replay it later in one call.
* **Observations** are small facts learned the hard way: which window class a
  launch produces, which label a button carries, what the retry delay is.

Statistics per tool (calls, failures, latency) are the effectiveness measure: the
`stats` tool reports them so tuning is driven by data, not vibes.

The database is SQLite in WAL mode under the state directory; text is capped so a
runaway OCR loop cannot balloon it.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started REAL NOT NULL,
    ended REAL,
    actions INTEGER DEFAULT 0,
    screenshots INTEGER DEFAULT 0,
    note TEXT
);
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL,
    ts REAL NOT NULL,
    tool TEXT NOT NULL,
    args TEXT,
    ok INTEGER NOT NULL,
    ms REAL NOT NULL,
    err TEXT,
    FOREIGN KEY (session) REFERENCES sessions(id)
);
CREATE INDEX IF NOT EXISTS actions_session ON actions(session);
CREATE TABLE IF NOT EXISTS skills (
    name TEXT PRIMARY KEY,
    description TEXT,
    steps TEXT NOT NULL,
    uses INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    hits INTEGER DEFAULT 1,
    created REAL NOT NULL,
    updated REAL NOT NULL,
    ttl REAL
);
"""

MAX_TEXT = 4000


class Memory:
    def __init__(self, path: Path | None = None) -> None:
        self.root = config.state_dir()
        self.sessions_dir = self.root / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.path = path or self.root / "memory.db"
        # A daemon lives for days and serves several agents; without a task
        # boundary its single rolling session blends unrelated work (observed
        # live: clipboard probes from a fix-verification run appeared inside
        # a later browser-research session's trace). A gap longer than this
        # between two actions rotates the session; 0 disables the rotation.
        try:
            minutes = float(os.environ.get("SMARTCOMP_SESSION_GAP_MINUTES", "30"))
        except ValueError:
            minutes = 30.0
        self.idle_gap_s = minutes * 60.0 if minutes > 0 else None
        self.last_action_ts: float | None = None
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        # Databases created before these columns existed keep working: SQLite
        # has no ADD COLUMN IF NOT EXISTS, so check the live schema once.
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(actions)").fetchall()}
        if "err" not in cols:
            self.db.execute("ALTER TABLE actions ADD COLUMN err TEXT")
        ocols = {r["name"] for r in self.db.execute("PRAGMA table_info(observations)").fetchall()}
        for col in ("created", "ttl"):
            if col not in ocols:
                self.db.execute(f"ALTER TABLE observations ADD COLUMN {col} REAL")
        self.db.commit()
        self.session = self.start_session()

    # -- sessions ---------------------------------------------------------
    def start_session(self) -> str:
        session_id = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        self.db.execute(
            "INSERT INTO sessions (id, started) VALUES (?, ?)", (session_id, time.time())
        )
        self.db.commit()
        (self.sessions_dir / session_id).mkdir(parents=True, exist_ok=True)
        self.session = session_id
        self.last_action_ts = None  # a fresh session is never idle-aged
        return session_id

    def session_dir(self) -> Path:
        path = self.sessions_dir / self.session
        path.mkdir(parents=True, exist_ok=True)
        return path

    def log(self, tool: str, args: dict, ok: bool, ms: float,
            error: str | None = None) -> None:
        now = time.time()
        if (self.idle_gap_s and self.last_action_ts is not None
                and now - self.last_action_ts > self.idle_gap_s):
            # Task boundary: the previous session's trace stays queryable,
            # but this action belongs to a fresh session, not a task that
            # ended hours ago.
            self.end_session()
        self.last_action_ts = now
        payload = json.dumps(args, default=str)[:MAX_TEXT]
        # The error text (class + message) is what makes failures explainable:
        # stats() histograms it, so "screenshot fails sometimes" becomes
        # "screenshot fails with BrokenPipeError when the locker grabs X".
        self.db.execute(
            "INSERT INTO actions (session, ts, tool, args, ok, ms, err) "
            "VALUES (?,?,?,?,?,?,?)",
            (self.session, now, tool, payload, 1 if ok else 0, ms,
             (error or "")[:200] or None),
        )
        self.db.execute(
            "UPDATE sessions SET actions = actions + 1 WHERE id = ?", (self.session,)
        )
        self.db.commit()

    def count_screenshot(self) -> None:
        self.db.execute(
            "UPDATE sessions SET screenshots = screenshots + 1 WHERE id = ?", (self.session,)
        )
        self.db.commit()

    def end_session(self, note: str = "") -> dict:
        """Close the current session, delete its screenshots, return what it did.

        Also called automatically: a gap longer than
        SMARTCOMP_SESSION_GAP_MINUTES (default 30) between two actions
        rotates the session, so unrelated tasks do not share one trace.
        """
        session_id = self.session
        rows = self.db.execute(
            "SELECT tool, args, ok, ms, err FROM actions WHERE session = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        dir_path = self.sessions_dir / session_id
        deleted = 0
        if dir_path.exists():
            deleted = len([p for p in dir_path.rglob("*") if p.is_file()])
            shutil.rmtree(dir_path, ignore_errors=True)
        self.db.execute(
            "UPDATE sessions SET ended = ?, note = ? WHERE id = ?",
            (time.time(), note[:500], session_id),
        )
        self.db.commit()
        trace = [
            {"tool": r["tool"], "args": json.loads(r["args"] or "{}"), "ok": bool(r["ok"]),
             "ms": round(r["ms"], 1), "err": r["err"]}
            for r in rows
        ]
        summary = {
            "session": session_id,
            "actions": len(trace),
            "failures": sum(1 for t in trace if not t["ok"]),
            "screenshots_deleted": deleted,
            "duration_s": round(
                (time.time() - (self.db.execute(
                    "SELECT started FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()["started"])),
                1,
            ),
        }
        self.start_session()
        summary["next_session"] = self.session
        summary["trace"] = trace[-40:]
        return summary

    def prune_sessions(self, older_than_hours: float = 24.0) -> int:
        """Delete screenshot directories left behind by crashed runs.

        Already-empty directories go no matter their age: end_session
        deletes the screenshots but leaves the folder behind, and that
        deletion refreshes the folder mtime -- age alone would keep empty
        husks around forever (8 measured live after a day of restarts).
        """
        cutoff = time.time() - older_than_hours * 3600
        removed = 0
        for path in self.sessions_dir.iterdir():
            if not path.is_dir() or path.name == self.session:
                continue
            empty = next(path.iterdir(), None) is None
            if empty or path.stat().st_mtime < cutoff:
                removed += 1
                shutil.rmtree(path, ignore_errors=True)
        return removed

    # -- skills -----------------------------------------------------------
    def learn_skill(self, name: str, steps: list[dict], description: str = "") -> dict:
        now = time.time()
        clean = [s for s in steps if isinstance(s, dict) and s.get("tool")]
        if not clean:
            raise ValueError("a skill needs at least one step with a tool name")
        payload = json.dumps(clean)[:MAX_TEXT]
        self.db.execute(
            """INSERT INTO skills (name, description, steps, created, updated)
               VALUES (?,?,?,?,?)
               ON CONFLICT(name) DO UPDATE SET
                 description=excluded.description, steps=excluded.steps,
                 updated=excluded.updated""",
            (name, description[:500], payload, now, now),
        )
        self.db.commit()
        return {"name": name, "steps": len(clean)}

    def skill(self, name: str) -> dict | None:
        row = self.db.execute("SELECT * FROM skills WHERE name = ?", (name,)).fetchone()
        if row is None:
            return None
        return {
            "name": row["name"],
            "description": row["description"],
            "steps": json.loads(row["steps"]),
            "uses": row["uses"],
            "wins": row["wins"],
        }

    def record_skill_use(self, name: str, ok: bool) -> None:
        self.db.execute(
            "UPDATE skills SET uses = uses + 1, wins = wins + ?, updated = ? WHERE name = ?",
            (1 if ok else 0, time.time(), name),
        )
        self.db.commit()

    def skills(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT name, description, uses, wins, updated FROM skills ORDER BY wins DESC, uses DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def forget_skill(self, name: str) -> bool:
        cur = self.db.execute("DELETE FROM skills WHERE name = ?", (name,))
        self.db.commit()
        return cur.rowcount > 0

    def distill(self, name: str, description: str = "") -> dict:
        """Turn the successful actions of this session into a replayable skill."""
        rows = self.db.execute(
            "SELECT tool, args FROM actions WHERE session = ? AND ok = 1 "
            "AND tool NOT IN ('get_changes','screen_state','health','stats','read_window',"
            "'read_region','find','wait_for_change','screenshot','session_end','list_skills',"
            "'run_skill','learn_skill','remember','recall') ORDER BY id",
            (self.session,),
        ).fetchall()
        steps = [{"tool": r["tool"], "args": json.loads(r["args"] or "{}")} for r in rows]
        if not steps:
            raise ValueError("no successful actions to distil yet")
        self.learn_skill(name, steps, description or f"learned {len(steps)} steps")
        return {"name": name, "steps": len(steps)}

    # -- observations -----------------------------------------------------
    def remember(self, key: str, value, ttl_s: float | None = None) -> None:
        now = time.time()
        self.db.execute(
            """INSERT INTO observations (key, value, created, updated, ttl)
               VALUES (?,?,?,?,?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value,
                 created=excluded.created, updated=excluded.updated,
                 ttl=excluded.ttl, hits=hits+1""",
            (key[:200], json.dumps(value, default=str)[:MAX_TEXT], now, now,
             float(ttl_s) if ttl_s is not None else None),
        )
        self.db.commit()

    def recall(self, key: str) -> dict:
        """Return the value plus its provenance: age and staleness.

        A fact stored with a ttl that has expired comes back stale with the
        value nulled -- a stale claim ("window class is X" after an app
        update) must not be handed back as if it were current. Age is always
        reported, so a caller can judge freshness even without a ttl.
        """
        row = self.db.execute(
            "SELECT value, updated, ttl FROM observations WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return {"value": None, "age_s": None, "stale": None}
        age_s = round(time.time() - row["updated"], 1)
        stale = row["ttl"] is not None and age_s > row["ttl"]
        return {
            "value": None if stale else json.loads(row["value"]),
            "age_s": age_s,
            "stale": stale,
        }

    def observations(self, limit: int = 100) -> list[dict]:
        now = time.time()
        rows = self.db.execute(
            "SELECT key, value, hits, updated, ttl FROM observations "
            "ORDER BY hits DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"key": r["key"], "value": json.loads(r["value"]), "hits": r["hits"],
             "age_s": round(now - r["updated"], 1), "ttl_s": r["ttl"]}
            for r in rows
        ]

    # -- effectiveness ----------------------------------------------------
    def stats(self, hours: float = 24.0) -> dict:
        rows = self.db.execute(
            """SELECT tool, COUNT(*) AS calls, SUM(ok) AS wins, AVG(ms) AS avg_ms,
                      MAX(ms) AS max_ms FROM actions GROUP BY tool ORDER BY calls DESC"""
        ).fetchall()
        out = {}
        for r in rows:
            calls = r["calls"] or 0
            wins = r["wins"] or 0
            out[r["tool"]] = {
                "calls": calls,
                "wins": wins,
                "success_rate": round(wins / calls, 3) if calls else 0.0,
                "avg_ms": round(r["avg_ms"] or 0, 1),
                "max_ms": round(r["max_ms"] or 0, 1),
            }
        total = self.db.execute("SELECT COUNT(*) AS c FROM actions").fetchone()["c"]
        # Failure histogram: which tool failed, with what, how often. This is
        # what let a live review prove the scroll and clipboard bugs were the
        # code's fault instead of guessing.
        err_rows = self.db.execute(
            """SELECT tool, substr(err, 1, 160) AS err, COUNT(*) AS n FROM actions
               WHERE ok = 0 AND err IS NOT NULL GROUP BY tool, substr(err, 1, 160)
               ORDER BY n DESC LIMIT 24"""
        ).fetchall()
        # A recent window next to the lifetime one: the lifetime rate of a
        # tool whose bug was just fixed reads as still-broken forever
        # (screenshot: 11% over 561 calls while the fixed path scores 100%).
        recent: dict = {"window_h": hours, "tools": {}, "errors": []}
        if hours and hours > 0:
            cutoff = time.time() - hours * 3600
            rrows = self.db.execute(
                """SELECT tool, COUNT(*) AS calls, SUM(ok) AS wins, AVG(ms) AS avg_ms
                   FROM actions WHERE ts > ? GROUP BY tool ORDER BY calls DESC""",
                (cutoff,),
            ).fetchall()
            for r in rrows:
                calls = r["calls"] or 0
                wins = r["wins"] or 0
                recent["tools"][r["tool"]] = {
                    "calls": calls,
                    "wins": wins,
                    "success_rate": round(wins / calls, 3) if calls else 0.0,
                    "avg_ms": round(r["avg_ms"] or 0, 1),
                }
            rerrs = self.db.execute(
                """SELECT tool, substr(err, 1, 160) AS err, COUNT(*) AS n FROM actions
                   WHERE ok = 0 AND err IS NOT NULL AND ts > ?
                   GROUP BY tool, substr(err, 1, 160) ORDER BY n DESC LIMIT 12""",
                (cutoff,),
            ).fetchall()
            recent["errors"] = [dict(r) for r in rerrs]
        return {
            "tools": out,
            "errors": [dict(r) for r in err_rows],
            "recent": recent,
            "actions_total": total,
            "skills": len(self.skills()),
            "observations": len(self.observations()),
            "db_path": str(self.path),
            "db_kb": round(self.path.stat().st_size / 1024, 1) if self.path.exists() else 0,
        }
