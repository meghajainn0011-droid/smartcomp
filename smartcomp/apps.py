"""App census and resume semantics: what is running, what is open, what to reuse.

launch_app used to spawn a duplicate of an app that was already open (two
Firefoxes from two calls), and nothing could answer "how many tabs do I have
open right now". This module gives the agent both:

* an identity-based census of the allowlisted apps -- live processes from
  /proc, mapped windows from the window manager, the focused one marked --
  so "is it already open?" is one cheap call instead of a guess;
* resume semantics for launch_app: an app that is already on screen is
  focused, not duplicated (a second instance is opt-in via force=true);
* for Firefox, the exact per-window tab counts read out of its own session
  store (recovery.jsonlz4, mozLZ4 container) -- no extension or CDP needed.
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path

try:  # the venv has it via system site packages; degrade cleanly without
    import lz4.block as _lz4
except Exception:  # pragma: no cover - import guard
    _lz4 = None

# name -> binary, the launch_app allowlist. Single source here; tools imports it.
APP_ALLOWLIST = {
    "brave": "brave-browser",
    "chromium": "chromium",
    "firefox": "firefox",
    "terminal": "qterminal",
    "text-editor": "mousepad",
    "file-manager": "thunar",
    "image-viewer": "ristretto",
}

_MOZLZ4_MAGIC = b"mozLz40\0"


def _proc_comm(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return ""


def _proc_state(pid: int) -> str:
    """Single state letter from /proc/<pid>/stat (Z = zombie, not an app).

    comm is wrapped in parens and may contain spaces or parens itself, so
    the state letter is whatever follows the LAST ')'. Splitting on tokens
    misses it: "(firefox-esr)" is one token and never equals ")". Found
    live: a defunct firefox-esr zombie passed as a real instance and the
    resume guard refused a launch that should have happened.
    """
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        after = text.rsplit(")", 1)[1].split()
        return after[0] if after else ""
    except (OSError, IndexError):
        return ""


def _proc_cmd(pid: int) -> str:
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        first = next((a for a in argv if a), b"")
        return Path(first.decode("utf-8", "replace")).name
    except OSError:
        return ""


def processes(binary: str) -> list[dict]:
    """Live, non-zombie processes whose executable is `binary` (or an ESR-style
    suffixed variant of it, e.g. firefox -> firefox-esr)."""
    want = binary.lower()
    out: list[dict] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if _proc_state(pid) == "Z":
            continue
        comm = _proc_comm(pid).lower()
        cmd = _proc_cmd(pid).lower()
        if comm == want or comm.startswith(want) or cmd == want or cmd.startswith(want):
            out.append({"pid": pid, "comm": comm or cmd})
    return out


def window_matches(win: dict, binary: str) -> bool:
    """A census window belongs to `binary` if any identity field names it."""
    want = binary.lower()
    fields = [win.get("comm", ""), win.get("cmd", ""), win.get("class", ""),
              win.get("instance", "")]
    return any(f and (f.lower() == want or f.lower().startswith(want)) for f in fields)


def census(windows: list[dict], names: dict[str, str] | None = None) -> dict:
    """Per allowlisted app: running processes, windows, focused window id.

    `processes` is the count of matching executables, which for multi-process
    apps (Firefox runs ~7 of them on purpose) is NOT the answer to "did I
    open two?": that question is answered by `windows` and `focused`.
    """
    out: dict = {}
    for name, binary in (names or APP_ALLOWLIST).items():
        pids = [p["pid"] for p in processes(binary)]
        app_windows = [w for w in windows if window_matches(w, binary)]
        focused = next((w["id"] for w in app_windows if w.get("active")), None)
        entry: dict = {
            "binary": binary,
            "running": bool(pids),
            "processes": len(pids),
            "pids": pids,
            "windows": [
                {"id": w["id"], "title": w.get("title", ""), "active": w.get("active", False)}
                for w in app_windows
            ],
            "focused": focused,
        }
        if binary == "firefox":
            # Tab counts only mean something while Firefox is running: a
            # recent recovery.jsonlz4 outlives the process (crash or close)
            # and would describe tabs nothing can see.
            entry["tabs"] = firefox_tabs() if pids else None
        out[name] = entry
    return out


def resume_target(windows: list[dict], binary: str) -> dict | None:
    """The window to focus instead of launching a duplicate: an active one
    first, else the first match."""
    matches = [w for w in windows if window_matches(w, binary)]
    if not matches:
        return None
    return next((w for w in matches if w.get("active")), matches[0])


# -- firefox session store --------------------------------------------------

def _mozlz4_paths() -> list[Path]:
    profiles = Path.home() / ".mozilla" / "firefox"
    out: list[Path] = []
    if profiles.is_dir():
        for profile in profiles.iterdir():
            backups = profile / "sessionstore-backups"
            for name in ("recovery.jsonlz4", "previous.jsonlz4"):
                p = backups / name
                if p.exists():
                    out.append(p)
    # newest first: recovery beats previous even across profiles
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)


def _mozlz4_decode(path: Path):
    """Decode a Firefox mozlz4 file: 8-byte magic, u32 LE size, raw LZ4 block."""
    raw = path.read_bytes()
    if len(raw) < 12 or raw[:8] != _MOZLZ4_MAGIC:
        raise ValueError(f"not a mozLZ4 file: {path.name}")
    (size,) = struct.unpack("<I", raw[8:12])
    if _lz4 is None:
        raise RuntimeError("lz4 module not available")
    return _lz4.decompress(raw[12:], uncompressed_size=size)


def _parse_sessionstore(obj: dict) -> dict:
    """Tab counts and active titles out of a sessionstore JSON object."""
    out_windows: list[dict] = []
    total = 0
    for win in obj.get("windows", []):
        tabs = win.get("tabs", [])
        selected = win.get("selected", 1) - 1
        title = ""
        if 0 <= selected < len(tabs):
            entries = tabs[selected].get("entries", [])
            idx = tabs[selected].get("index", len(entries)) - 1
            if 0 <= idx < len(entries):
                title = entries[idx].get("title", "")
        out_windows.append({"tabs": len(tabs), "selected_title": title[:80]})
        total += len(tabs)
    return {"windows": out_windows, "total_tabs": total,
            "source_age_s": None}


def _fresh_enough(path: Path, now: float) -> bool:
    """recovery.jsonlz4 is the live store; previous.jsonlz4 only when recent.

    Firefox rewrites recovery.jsonlz4 while it runs and leaves
    previous.jsonlz4 behind after a clean quit. A day-old previous snapshot
    describes a session that is over -- serving it as the answer to "how
    many tabs are open NOW" would hand the agent a wrong fact (measured
    live: a 25-minute-old previous.jsonlz4 said 1 tab while 3 were open).
    """
    if path.name == "recovery.jsonlz4":
        return True
    return now - path.stat().st_mtime < 600.0


def firefox_tabs() -> dict | None:
    """How many tabs Firefox has open, per window, from its session store.

    Firefox rewrites recovery.jsonlz4 roughly every 15 s while it runs, so
    this reflects the live state within one save interval. None when there
    is no usable source (Firefox never ran, or only an old post-quit
    snapshot exists) -- never an exception: it is a census field.
    """
    now = time.time()
    for path in _mozlz4_paths():
        if not _fresh_enough(path, now):
            continue
        try:
            obj = json.loads(_mozlz4_decode(path))
            parsed = _parse_sessionstore(obj)
            parsed["source"] = path.parent.parent.name + "/" + path.name
            parsed["source_age_s"] = round(now - path.stat().st_mtime, 1)
            return parsed
        except Exception:
            continue  # try the next backup file
    return None
