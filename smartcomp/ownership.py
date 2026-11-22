"""Shared app-ownership registry: which smartcomp owns which app.

Problem: with several smartcomps on one machine (main daemon + session
daemons, or different operators), a naive `launch_app("firefox")` resumes
whatever Firefox is open -- possibly another agent's, mid-task. This file is
the shared knowledge point: every daemon reads and writes claims in one
0600 JSON file under the state dir, so each agent knows which apps are in
use and by whom, and launches its OWN instance when an app is taken.

Claims are soft leases: a claim has an owner (daemon/session id), the app,
an optional window id and a timestamp. An owner releases on session_end /
session_close; stale claims (owner process gone or older than TTL) are
ignored. No locks needed for the small critical section: atomic replace.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from . import config

_CLAIM_TTL = 6 * 3600.0


def _root() -> Path:
    return Path(os.environ.get("SMARTCOMP_OWNERS_DIR",
                               str(Path.home() / ".local" / "state" / "smartcomp")))


def _path() -> Path:
    p = _root() / "app-owners.json"
    if not p.exists():
        _atomic_write({"claims": {}})
    return p


def _atomic_write(data: dict) -> None:
    path = _root() / "app-owners.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".owners.")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read() -> dict:
    try:
        return json.loads(_path().read_text())
    except Exception:
        return {"claims": {}}


def owner_id() -> str:
    """Stable id for this daemon: the session name, or 'main'."""
    return os.environ.get("SMARTCOMP_SESSION_NAME", "main")


def claim(app: str, window_id: str | None = None, pid: int | None = None) -> dict:
    """Record that this daemon is using `app`."""
    data = _read()
    key = f"{app.lower()}@{owner_id()}"
    data["claims"][key] = {
        "app": app.lower(), "owner": owner_id(), "window": window_id, "pid": pid,
        "at": time.time(),
    }
    _atomic_write(data)
    return {"ok": True, "app": app, "owner": owner_id()}


def release(app: str) -> dict:
    data = _read()
    key = f"{app.lower()}@{owner_id()}"
    if key in data["claims"]:
        data["claims"].pop(key, None)
        _atomic_write(data)
    return {"ok": True, "released": app}


def release_all() -> dict:
    data = _read()
    mine = [k for k, v in data["claims"].items() if v.get("owner") == owner_id()]
    for k in mine:
        data["claims"].pop(k, None)
    if mine:
        _atomic_write(data)
    return {"ok": True, "released": mine}


def _alive(pid: int | None) -> bool:
    if not pid:
        return True  # no pid recorded: trust the lease until TTL
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def owner_of(app: str) -> dict | None:
    """Who owns `app` right now (None if free / stale / self)."""
    entries = [e for e in _read()["claims"].values()
               if e.get("app") == app.lower()]
    for entry in entries:
        if time.time() - entry.get("at", 0) > _CLAIM_TTL or not _alive(entry.get("pid")):
            continue
        if entry.get("owner") == owner_id():
            continue  # our own app: normal resume is fine
        return entry
    return None


def status() -> dict:
    """All live claims (shared knowledge of app usage)."""
    data = _read()
    live = {}
    now = time.time()
    for app, e in data["claims"].items():
        if now - e.get("at", 0) <= _CLAIM_TTL and _alive(e.get("pid")):
            live[app] = {"owner": e.get("owner"), "window": e.get("window"),
                         "age_s": round(now - e.get("at", 0), 1)}
    return {"ok": True, "owner": owner_id(), "path": str(_path()),
            "claims": live}
