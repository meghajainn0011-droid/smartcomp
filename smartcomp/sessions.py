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
        used = {s["port"] for s in reg["sessions"].values()}
        used.add(config.MCP_PORT)
        port = config.MCP_PORT + _PORT_BASE_OFFSET
        while port in used:
            port += 1
        # also refuse a port another process already owns
        for _ in range(500):
            with socket.socket() as sock:
                try:
                    sock.bind(("127.0.0.1", port))
                except OSError:
                    port += 1
                    continue
            return port
        raise RuntimeError("no free session port")

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
        except urllib.error.HTTPError as exc:  # 406 = live, auth passed
            return exc.code in (200, 406)
        except Exception:
            return False

    # -- lifecycle -------------------------------------------------------
    def create(self, name: str = "agent") -> dict:
        reg = self._load()
        if name in reg["sessions"]:
            return {"ok": False, "error": f"session {name!r} already exists"}
        display = self._free_display(reg)
        port = self._free_port(reg)
        sdir = self.root / name
        sdir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, DISPLAY=display)

        def start(argv: list[str], use_env: dict | None = None) -> int:
            proc = subprocess.Popen(
                argv, env=use_env or env, cwd=str(config.PROJECT_ROOT),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            return proc.pid

        xvfb_pid = start(["Xvfb", display, "-screen", "0", _GEOMETRY,
                          "-nolisten", "tcp", "-ac", "+extension", "RANDR"])
        wm_pid = None
        if subprocess.run(["sh", "-c", "command -v xfwm4"],
                          capture_output=True).returncode == 0:
            time.sleep(0.4)
            wm_pid = start(["xfwm4"])
        time.sleep(0.3)
        # A full daemon per session: own port/token/state dir; the virtual
        # display has no lock, so the wake/unlock guard is off.
        cenv = dict(env,
                    SMARTCOMP_PORT=str(port),
                    SMARTCOMP_STATE_DIR=str(sdir),
                    SMARTCOMP_SCREEN_GUARD="0")
        bin_path = Path(sys.executable).parent / "smartcomp"
        daemon_pid = start([str(bin_path), "serve"], cenv) if bin_path.exists() else start(
            [sys.executable, "-m", "smartcomp", "serve"], cenv)
        rec = {"display": display, "port": port, "pid": daemon_pid,
               "xvfb_pid": xvfb_pid, "wm_pid": wm_pid, "dir": str(sdir),
               "created": time.time()}
        reg["sessions"][name] = rec
        self._save(reg)
        return {"ok": True, "session": name, **rec,
                "endpoint": f"http://127.0.0.1:{port}/mcp",
                "token_path": str(sdir / "token")}

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
        return {"ok": True, "closed": name}

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
