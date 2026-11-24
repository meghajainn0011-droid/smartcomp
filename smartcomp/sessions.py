"""Isolated per-session X desktops: own cursor, own memory, shared training.

One X11 display has exactly one pointer. Agents that share a display fight
over the cursor and can see/interrupt each other's windows. The standard fix
(x11-mcp, desktop-sandbox, deskpal, codemux) is a private Xvfb per session:
XTEST is display-local, so each session's clicks stay on its own screen, and
the virtual display has no screen lock -- it keeps working while the host is
locked, blanked or asleep, without waking or unlocking the host.

Each session is a full SmartComp daemon (its own port, token, state dir and
Memory). `align()` merges every session's skills/observations into a shared
pool and pushes the pool back; `train()` distils successful traces and feeds
the pool -- so the fleet gets better the more it is used.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import config
from .memory import Memory

log = logging.getLogger("smartcomp.sessions")

_DISPLAY_BASE = 1000          # avoid host :0-:2 and the CI convention :99
_GEOMETRY = "1920x1080x24"
_PORT_BASE_OFFSET = 1         # main daemon is config.MCP_PORT; sessions follow
_HEALTH_TIMEOUT = 40.0        # detector load on this 2-core box is ~20 s

_mgr: "SessionManager | None" = None


def manager() -> "SessionManager":
    global _mgr
    if _mgr is None:
        _mgr = SessionManager()
    return _mgr


class SessionManager:
    def __init__(self) -> None:
        self.root = config.state_dir() / "sessions"
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "registry.json"
        self.shared = Memory(path=config.state_dir() / "shared-memory.db")

    # -- registry --------------------------------------------------------
    def _load(self) -> dict:
        try:
            return json.loads(self.registry_path.read_text()) if self.registry_path.exists() else {"sessions": {}}
        except Exception:
            return {"sessions": {}}

    def _save(self, reg: dict) -> None:
        tmp = self.registry_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(reg, indent=2))
        tmp.replace(self.registry_path)

    @staticmethod
    def ensure_token(sdir: Path) -> str:
        """Token exists before the daemon starts (endpoint needs it at once)."""
        import secrets
        token_file = sdir / "token"
        if not token_file.exists():
            token_file.write_text(secrets.token_urlsafe(32))
            token_file.chmod(0o600)
        return token_file.read_text().strip()

    # -- allocation ------------------------------------------------------
    def _free_display(self, reg: dict) -> str:
        used = set()
        for s in reg["sessions"].values():
            used.add(s["display"])
        for path in Path("/tmp/.X11-unix").glob("X*"):
            used.add(":" + path.name[1:])
        n = _DISPLAY_BASE
        while f":{n}" in used:
            n += 1
        return f":{n}"

    def _free_port(self, reg: dict) -> int:
        # Bind to port 0: the kernel assigns a guaranteed-free port. The
        # old "MCP_PORT + offset" scheme collided when two sessions were
        # created close together (both picked 8795), leaving one dead.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    @staticmethod
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _healthy(self, port: int, token: str) -> bool:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/mcp",
            headers={"Authorization": f"Bearer {token}"}, method="GET")
        try:
            urllib.request.urlopen(req, timeout=3)
            return True
        except urllib.error.HTTPError as exc:
            # Any JSON-RPC-level refusal means the server answered: 405 is the
            # spec'd "no stream for a sessionless GET", 406 the older
            # "Accept must include event-stream". A dead port raises instead.
            return exc.code in (200, 405, 406)
        except Exception:
            return False

    # -- lifecycle -------------------------------------------------------
    @staticmethod
    def _visible_slot(reg: dict) -> tuple[int, int, int, int]:
        """Where the next visible session window goes on the host screen.

        Two 940x740 windows side by side leave the bottom of a 1920x1080
        screen free: the point is to watch two agents work, not to cover the
        human's desktop with one maximised window.
        """
        slots = [(20, 60), (960, 60), (20, 810), (960, 810)]
        taken = sum(1 for s in reg["sessions"].values() if s.get("visible"))
        x, y = slots[taken % len(slots)]
        return x, y, 940, 740

    def create(self, name: str = "agent", visible: bool = False,
               position: tuple[int, int] | None = None,
               size: tuple[int, int] | None = None) -> dict:
        reg = self._load()
        if name in reg["sessions"]:
            return {"ok": False, "error": f"session {name!r} already exists"}
        display = self._free_display(reg)
        port = self._free_port(reg)
        sdir = self.root / name
        sdir.mkdir(parents=True, exist_ok=True)
        # Create the token NOW so session_endpoint()/connect works immediately,
        # without waiting for the daemon's lazy first read (HTTP 406 check).
        self.ensure_token(sdir)
        global_root = Path.home() / ".local" / "state" / "smartcomp"
        env = dict(os.environ, DISPLAY=display,
                   SMARTCOMP_OWNERS_DIR=str(global_root))

        logf = sdir / "daemon.log"
        def start(argv: list[str], use_env: dict | None = None, logfile: Path | None = None) -> int:
            out = open(logfile, "ab") if logfile else subprocess.DEVNULL
            proc = subprocess.Popen(
                argv, env=use_env or env, cwd=str(config.PROJECT_ROOT),
                stdout=out, stderr=subprocess.STDOUT if out is not subprocess.DEVNULL else subprocess.DEVNULL,
                start_new_session=True)
            return proc.pid

        slot = self._visible_slot(reg)
        x, y, width, height = (position[0], position[1],
                               (size or slot)[0], (size or slot)[1]) if position else slot
        if visible:
            # Xephyr: a nested X server whose root window is a window on the
            # host screen. The session keeps its own pointer (XTEST is
            # display-local) but the human can SEE it work, and nothing an
            # agent does can reach the host's apps.
            host = os.environ.get("DISPLAY") or ":0"
            xenv = dict(os.environ, DISPLAY=host)
            screen_pid = start(["Xephyr", display, "-screen", f"{width}x{height}x24",
                                "-ac", "-noreset", "-name", f"smartcomp-{name}"],
                               use_env=xenv)
        else:
            screen_pid = start(["Xvfb", display, "-screen", "0", _GEOMETRY,
                                "-nolisten", "tcp", "-ac", "-noreset",
                                "+extension", "RANDR", "-extension", "GLX"])
        wm_pid = None
        if subprocess.run(["sh", "-c", "command -v xfwm4"],
                          capture_output=True).returncode == 0:
            time.sleep(0.5)
            # xfwm4 needs --replace to claim the screen and must be detached
            # (it runs forever; subprocess.run would kill it on timeout).
            # The session display has no other WM, so this claims cleanly.
            wm_pid = start(["xfwm4", "--replace"])
            time.sleep(0.6)
        if visible:
            self._place_host_window(f"smartcomp-{name}", x, y, width, height)
        # A translucent (20%) but visible cursor for the virtual display.
        from .cursor import make_visible_cursor
        cursor = make_visible_cursor(str(sdir), alpha=51)
        cursor_theme = cursor.get("theme") if cursor.get("ok") else None
        # A full daemon per session: own port/token/state dir; the virtual
        # display has no lock of its own, so the host-lock gate is off (it is
        # already skipped for a daemon carrying SMARTCOMP_SESSION_NAME). The
        # daemon and its apps inherit the visible cursor theme.
        cenv = dict(env,
                    SMARTCOMP_PORT=str(port),
                    SMARTCOMP_STATE_DIR=str(sdir),
                    SMARTCOMP_SESSION_NAME=name,
                    SMARTCOMP_HOST_LOCK_GATE="0")
        if cursor_theme:
            cenv["XCURSOR_THEME"] = cursor_theme
            cenv["XCURSOR_SIZE"] = "32"
            env["XCURSOR_THEME"] = cursor_theme
            env["XCURSOR_SIZE"] = "32"
        bin_path = Path(sys.executable).parent / "smartcomp"
        daemon_pid = (start([str(bin_path), "serve"], cenv, logf) if bin_path.exists()
                      else start([sys.executable, "-m", "smartcomp", "serve"], cenv, logf))
        rec = {"display": display, "port": port, "pid": daemon_pid,
               "xvfb_pid": screen_pid, "wm_pid": wm_pid, "dir": str(sdir),
               "created": time.time(), "visible": visible,
               "geometry": [x, y, width, height]}
        reg["sessions"][name] = rec
        self._save(reg)
        return {"ok": True, "session": name, **rec,
                "cursor": {"visible": bool(cursor_theme), "alpha": 51},
                "endpoint": f"http://127.0.0.1:{port}/mcp",
                "token_path": str(sdir / "token")}

    @staticmethod
    def _place_host_window(title: str, x: int, y: int, width: int, height: int,
                           timeout: float = 8.0) -> bool:
        """Move the nested display's window to its slot on the host screen.

        Xephyr maps its window wherever the WM decides; the session is only
        watchable if the two windows sit side by side, so they are placed
        explicitly (a ConfigureWindow request -- no synthetic input). The
        window is matched by WM_CLASS instance name, which is what Xephyr sets
        from -name; its WM_NAME is "Xephyr on :1000.0 (ctrl+shift grabs
        mouse and keyboard)", so matching the title never found it.
        """
        try:
            from Xlib import X, display as xdisplay
        except Exception:
            return False
        try:
            d = xdisplay.Display(os.environ.get("DISPLAY") or None)
        except Exception:
            return False
        try:
            root = d.screen().root
            atom = d.intern_atom("_NET_CLIENT_LIST")
            deadline = time.time() + timeout
            while time.time() < deadline:
                prop = root.get_full_property(atom, X.AnyPropertyType)
                for wid in (prop.value if prop else []):
                    try:
                        win = d.create_resource_object("window", wid)
                        cls = win.get_wm_class()
                    except Exception:
                        continue
                    if cls and cls[0] == title:
                        win.configure(x=x, y=y, width=width, height=height)
                        d.sync()
                        return True
                time.sleep(0.2)
            return False
        finally:
            d.close()

    def list(self) -> list[dict]:
        reg = self._load()
        out = []
        for name, s in reg["sessions"].items():
            token = ""
            try:
                token = (Path(s["dir"]) / "token").read_text().strip()
            except OSError:
                pass
            out.append({
                "name": name, "display": s["display"], "port": s["port"],
                "alive": self._alive(s["pid"]) and self._alive(s["xvfb_pid"]),
                "healthy": self._healthy(s["port"], token) if token else False,
                "created": s["created"],
            })
        return out

    def close(self, name: str) -> dict:
        reg = self._load()
        s = reg["sessions"].pop(name, None)
        if not s:
            return {"ok": False, "error": f"no session {name!r}"}
        # Kill every app running on this session's display FIRST, so no app
        # is left on screen or holding RAM after the session ends. Match by
        # the process's DISPLAY (the same way apps.processes scopes it).
        killed = self._kill_display_apps(s["display"])
        # Drop this owner's app claims from the shared registry.
        from . import ownership as own
        saved = os.environ.get("SMARTCOMP_SESSION_NAME")
        os.environ["SMARTCOMP_SESSION_NAME"] = name
        try:
            own.release_all()
        finally:
            if saved is not None:
                os.environ["SMARTCOMP_SESSION_NAME"] = saved
            else:
                os.environ.pop("SMARTCOMP_SESSION_NAME", None)
        for key in ("pid", "wm_pid", "xvfb_pid"):
            pid = s.get(key)
            if not pid:
                continue
            try:
                os.killpg(os.getpgid(pid), 15)
            except (OSError, ProcessLookupError):
                try:
                    os.kill(pid, 15)
                except OSError:
                    pass
        self._save(reg)
        return {"ok": True, "closed": name, "apps_killed": killed}

    @staticmethod
    def _kill_display_apps(display: str) -> list[dict]:
        """Terminate GUI processes bound to `display` (app cleanup on close)."""
        killed = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                env = (entry / "environ").read_bytes().decode("utf-8", "replace")
                pdisp = next((p.split("=", 1)[1] for p in env.split("\0")
                              if p.startswith("DISPLAY=")), None)
                if not pdisp or pdisp.rstrip(".0") != display.rstrip(".0"):
                    continue
                comm = (entry / "comm").read_text().strip()
                if comm in ("Xvfb", "xfwm4", "smartcomp", "python3"):
                    continue  # session infrastructure handled separately
                os.kill(pid, 15)
                killed.append({"pid": pid, "comm": comm})
            except (OSError, ProcessLookupError):
                continue
        return killed

    def endpoint(self, name: str) -> dict:
        reg = self._load()
        s = reg["sessions"].get(name)
        if not s:
            return {"ok": False, "error": f"no session {name!r}"}
        return {
            "ok": True, "session": name,
            "mcp": {"name": f"smartcomp-{name}", "transport": "streamable-http",
                    "url": f"http://127.0.0.1:{s['port']}/mcp",
                    "headers": {
                        "Authorization": f"Bearer {(Path(s['dir']) / 'token').read_text().strip()}"},
                    "docs": str(config.PROJECT_ROOT / "README.md")},
        }

    # -- aligned memory + training loop ----------------------------------
    def _session_memory(self, s: dict) -> Memory:
        return Memory(path=Path(s["dir"]) / "memory.db")

    def align(self) -> dict:
        """Merge every session's skills/observations into the shared pool and
        push the pool back to all sessions (alignment across memories)."""
        reg = self._load()
        pushed = 0
        for name, s in reg["sessions"].items():
            mem = self._session_memory(s)
            for skill in mem.skills():
                self.shared.learn_skill(
                    f"{name}/{skill['name']}", skill.get("steps", []),
                    skill.get("description", ""))
                pushed += 1
            for obs in mem.observations(limit=50):
                self.shared.remember(f"{name}/{obs['key']}", obs["value"])
                pushed += 1
        pool = [(k["name"], k.get("steps", []), k.get("description", ""))
                for k in self.shared.skills()]
        pulled = 0
        for name, s in reg["sessions"].items():
            mem = self._session_memory(s)
            for skill_name, steps, desc in pool:
                if not mem.skill(skill_name):
                    mem.learn_skill(skill_name, steps, desc)
                    pulled += 1
        self.shared.remember("last-align", str(time.time()))
        return {"ok": True, "sessions": len(reg["sessions"]),
                "pushed": pushed, "pulled": pulled}

    def train(self) -> dict:
        """Training loop: distil each session's successful trace into a skill,
        feed the shared pool, then align so every session inherits it."""
        reg = self._load()
        distilled = []
        for name, s in reg["sessions"].items():
            mem = self._session_memory(s)
            try:
                out = mem.distill(f"train-{name}-{int(time.time())}",
                                  "training loop: distilled from successful actions")
                if out.get("steps"):
                    distilled.append({"session": name, "steps": out["steps"]})
            except ValueError:
                continue
        aligned = self.align()
        self.shared.remember("last-train", str(time.time()))
        return {"ok": True, "distilled": distilled, "alignment": aligned}
