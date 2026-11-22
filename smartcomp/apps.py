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
import os
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


def profile_args(binary: str, name: str) -> list[str]:
    """Args that give a second instance its OWN profile (no shared session)."""
    ident = "".join(c for c in name if c.isalnum() or c in "-_")[:24]
    base = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "smartcomp-profiles" / ident
    base.mkdir(parents=True, exist_ok=True)
    if binary in ("firefox", "firefox-esr"):
        return ["-profile", str(base / "firefox")]
    if binary in ("brave-browser", "chromium", "google-chrome"):
        return [f"--user-data-dir={base / binary}"]
    return []


#: Profile files that are pure cache. Copying them makes a clone slow and buys
#: nothing: the browser rebuilds them on first use.
_PROFILE_SKIP = (
    "cache", "cache2", "startupcache", "startupCache", "shader-cache", "shadercache",
    "cache2*", "thumbnails", "crashes", "minidumps", "datareporting", "saved-telemetry-pings",
    "security_state", "storage", "lock", ".parentlock", "lockfile",
    "SingletonLock", "SingletonCookie", "SingletonSocket",
)


def _firefox_default_profile() -> Path | None:
    """The profile Firefox actually opens (from profiles.ini, not a guess).

    [Install*] is the launcher's own choice and wins: on this box it points at
    the populated `*.default-esr` profile while [Profile1] still carries
    Default=1 on an empty one, so preferring the Profile flag clones a browser
    with no logins in it.
    """
    import configparser

    base = Path.home() / ".mozilla" / "firefox"
    ini = base / "profiles.ini"
    if not ini.exists():
        return None
    parser = configparser.ConfigParser()
    try:
        parser.read(ini)
    except Exception:
        return None

    def resolve(path: str) -> Path:
        return Path(path) if Path(path).is_absolute() else base / path

    for section in parser.sections():
        if section.startswith("Install"):
            chosen = parser.get(section, "Default", fallback="")
            if chosen and resolve(chosen).is_dir():
                return resolve(chosen)
    candidates: list[tuple[bool, Path]] = []
    for section in parser.sections():
        if not section.startswith("Profile"):
            continue
        path = parser.get(section, "Path", fallback="")
        if not path:
            continue
        is_default = parser.get(section, "Default", fallback="0") == "1"
        full = resolve(path)
        candidates.append((is_default, full))
    for is_default, full in candidates:
        if is_default and full.is_dir():
            return full
    for _, full in candidates:
        if full.is_dir():
            return full
    return None


def _chromium_profile(binary: str) -> tuple[Path, Path] | None:
    """(user-data dir, Default profile dir) for a Chromium-family browser."""
    roots = {
        "brave-browser": Path.home() / ".config" / "BraveSoftware" / "Brave-Browser",
        "chromium": Path.home() / ".config" / "chromium",
        "google-chrome": Path.home() / ".config" / "google-chrome",
    }
    root = roots.get(binary)
    if root is None or not (root / "Default").is_dir():
        return None
    return root, root / "Default"


def clone_profile(binary: str, dest: Path) -> list[str]:
    """Copy the user's real browser profile into ``dest``; return launch args.

    A nested session has to look like the user's browser -- signed in, same
    extensions, same cookies -- or every site asks to log in again and the
    agent is testing a browser nobody uses. The running profile cannot be
    shared (it is locked, and its windows belong to the host display), so the
    directory is copied once per session and the nested instance runs on the
    copy. Cache directories are skipped; logins, cookies, history and
    preferences are kept.

    Chromium keeps the cookie encryption key in "Local State" next to the
    profile, so that file is copied too -- without it the cookies arrive but
    cannot be decrypted.
    """
    import shutil

    dest.mkdir(parents=True, exist_ok=True)
    if binary in ("firefox", "firefox-esr"):
        src = _firefox_default_profile()
        if src is None:
            return profile_args(binary, dest.name)
        target = dest / "firefox"
        if not (target / "prefs.js").exists():
            shutil.copytree(src, target, ignore=shutil.ignore_patterns(*_PROFILE_SKIP),
                            dirs_exist_ok=True)
        return ["-profile", str(target)]
    found = _chromium_profile(binary)
    if found is None:
        return profile_args(binary, dest.name)
    root, default = found
    target = dest / binary
    if not (target / "Default" / "Preferences").exists():
        (target / "Default").mkdir(parents=True, exist_ok=True)
        shutil.copytree(default, target / "Default",
                        ignore=shutil.ignore_patterns(*_PROFILE_SKIP), dirs_exist_ok=True)
        local_state = root / "Local State"
        if local_state.exists():
            shutil.copy2(local_state, target / "Local State")
    return [f"--user-data-dir={target}"]


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


def processes(binary: str, display: str | None = None) -> list[dict]:
    """Live, non-zombie processes for `binary`, scoped to a display.

    A session daemon must not be blocked by a host app on another X display:
    with `display` set, only processes whose DISPLAY matches are returned.
    Reads /proc/<pid>/environ (best-effort: the kernel hides some entries).
    """
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
        if not (comm == want or comm.startswith(want) or cmd == want or cmd.startswith(want)):
            continue
        if display:
            try:
                env = (entry / "environ").read_bytes().decode("utf-8", "replace")
                pdisp = next((p.split("=", 1)[1] for p in env.split("\0")
                              if p.startswith("DISPLAY=")), None)
                if pdisp is not None and pdisp.rstrip(".0") != display.rstrip(".0"):
                    continue
            except OSError:
                pass  # unreadable env: keep the process (conservative)
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
