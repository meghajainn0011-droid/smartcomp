"""Start/stop/restart/status/doctor for the local daemon.

`smartcomp` (no arguments) starts it, and the other commands do what they say.
systemd --user owns the service when its unit is installed -- the normal case,
because it survives logout and restarts on failure. The process path exists
for a machine without a user bus, and both paths are probed the same way: the
daemon is "up" when its MCP endpoint answers `tools/list`.

Nothing here touches the host's lock or blank state. The doctor only reads
(`xset q`, ScreenSaver.GetActive), and the "sleep" answer is honest: a blank
monitor does not stop the daemon, a suspended machine stops everything.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import config

UNIT = "smartcomp.service"
STATE_DIR = config.state_dir
PID_FILE = STATE_DIR() / "daemon.pid"
LOG_FILE = Path("/tmp/smartcomp_server.log")


# -- process plumbing ----------------------------------------------------

def _run(cmd: list[str], timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def unit_installed() -> bool:
    try:
        return _run(["systemctl", "--user", "cat", UNIT], timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _systemd_active() -> bool:
    try:
        out = _run(["systemctl", "--user", "is-active", UNIT], timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.stdout.strip() == "active"


def _systemd_pid() -> int | None:
    try:
        out = _run(["systemctl", "--user", "show", "-p", "MainPID", "--value", UNIT], timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        pid = int(out.stdout.strip())
    except ValueError:
        return None
    return pid or None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _pidfile_pid() -> int | None:
    try:
        pid = int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if _pid_alive(pid) else None


def running_pid() -> int | None:
    if unit_installed() and _systemd_active():
        return _systemd_pid()
    return _pidfile_pid()


def backend() -> str:
    if unit_installed() and _systemd_active():
        return "systemd"
    return "pidfile" if _pidfile_pid() else "none"


def _probe(timeout: float = 3.0) -> tuple[bool, str, int | None]:
    """(answering, detail, tool count) against the local endpoint."""
    from . import rpc

    try:
        token = config.read_token()
    except Exception as exc:  # unreadable/absent token is a real diagnosis
        return False, f"token unavailable: {exc}", None
    url = f"http://{config.MCP_HOST}:{config.MCP_PORT}/mcp"
    try:
        tools = rpc.list_tools(url, token, timeout=timeout)
    except Exception as exc:
        return False, str(exc)[:200], None
    return True, "ok", len(tools)


def wait_ready(seconds: float = 90.0) -> tuple[bool, int | None]:
    """Poll the endpoint until it answers tools/list (or the deadline)."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        ok, _detail, count = _probe(timeout=2.0)
        if ok and count:
            return True, count
        time.sleep(1.0)
    return False, None


# -- commands ------------------------------------------------------------

def start(wait: bool = True) -> int:
    if running_pid():
        ok, count = wait_ready(10)
        print(f"smartcomp is already running (pid {running_pid()}, {count or '?'} tools)")
        return 0 if ok else 1
    if unit_installed():
        print(f"starting {UNIT} ...")
        result = _run(["systemctl", "--user", "start", UNIT], timeout=60)
        if result.returncode != 0:
            print(f"systemctl start failed: {result.stderr.strip() or result.stdout.strip()}")
            return 1
    else:
        print("no systemd user unit; starting a detached process")
        log = open(LOG_FILE, "ab")
        proc = subprocess.Popen(
            [sys.executable, "-m", "smartcomp", "serve"],
            cwd=str(config.PROJECT_ROOT), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        log.close()
        STATE_DIR().mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(f"{proc.pid}\n")
    if not wait:
        return 0
    t0 = time.time()
    ok, count = wait_ready(config.START_TIMEOUT)
    if not ok:
        print(f"smartcomp did not answer within {config.START_TIMEOUT:.0f}s; "
              f"run `smartcomp doctor`")
        return 1
    print(f"smartcomp is up: {count} tools in {time.time() - t0:.0f}s "
          f"(pid {running_pid()})")
    return 0


def stop() -> int:
    pid = running_pid()
    if not pid:
        print("smartcomp is not running")
        return 0
    if unit_installed():
        result = _run(["systemctl", "--user", "stop", UNIT], timeout=60)
        if result.returncode != 0:
            print(f"systemctl stop failed: {result.stderr.strip()}")
            return 1
    else:
        try:
            os.kill(pid, 15)
        except OSError:
            pass
        deadline = time.time() + 15
        while time.time() < deadline and _pid_alive(pid):
            time.sleep(0.2)
        if _pid_alive(pid):
            os.kill(pid, 9)
    try:
        PID_FILE.unlink()
    except OSError:
        pass
    print(f"smartcomp stopped (pid {pid})")
    return 0


def restart() -> int:
    if unit_installed():
        print(f"restarting {UNIT} ...")
        result = _run(["systemctl", "--user", "restart", UNIT], timeout=90)
        if result.returncode != 0:
            print(f"systemctl restart failed: {result.stderr.strip()}")
            return 1
    else:
        stop()
        return start(wait=False)
    t0 = time.time()
    ok, count = wait_ready(config.START_TIMEOUT)
    if not ok:
        print(f"smartcomp did not answer within {config.START_TIMEOUT:.0f}s; "
              f"run `smartcomp doctor`")
        return 1
    print(f"smartcomp restarted: {count} tools in {time.time() - t0:.0f}s "
          f"(pid {running_pid()})")
    return 0


def status() -> int:
    from . import lock

    pid = running_pid()
    ok, detail, count = _probe()
    try:
        from . import sessions as sessions_mod

        session_count = len(sessions_mod.manager().list())
    except Exception:
        session_count = -1
    print(f"smartcomp: {'running' if pid else 'stopped'}"
          f"{f' ({backend()})' if pid else ''}")
    if pid:
        print(f"  pid       {pid}")
    print(f"  endpoint  http://{config.MCP_HOST}:{config.MCP_PORT}/mcp")
    print(f"  token     {config.token_path()}")
    print(f"  mcp       {f'{count} tools' if ok else 'NOT ANSWERING: ' + detail}")
    locked = lock.is_locked()
    print(f"  screen    {'locked' if locked else 'unlocked' if locked is False else 'unknown'}"
          f"  (display {os.environ.get('DISPLAY', ':0')})")
    fallback = "agent display while locked" if config.auto_session_enabled() else "off"
    print(f"  fallback  {fallback}")
    if session_count >= 0:
        print(f"  sessions  {session_count}")
    return 0 if pid and ok else 1


# -- doctor --------------------------------------------------------------

def _check(name: str, ok: bool | None, detail: str = "") -> dict:
    """ok=True pass, False fail, None warn."""
    return {"name": name, "ok": ok, "detail": detail}


def _orphans() -> list[str]:
    found = []
    mine = running_pid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            raw = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
            cmd = raw.decode(errors="replace").strip()
            _, _, rest = (entry / "stat").read_text().partition(") ")
            state, ppid = rest.split()[0], int(rest.split()[1])
        except (OSError, IndexError, ValueError):
            continue
        if not cmd:
            continue
        if state == "Z" and ppid == mine:
            found.append(f"zombie {pid} (child of the daemon): {cmd[:80]}")
        elif ppid == 1 and (cmd.startswith("Xvfb ") or cmd.startswith("Xephyr ")
                            or "smartcomp serve" in cmd or cmd.startswith("xfwm4")):
            found.append(f"orphan {pid}: {cmd[:80]}")
    return found


def doctor() -> dict:
    from . import lock, rpc
    from .display import Display
    from . import sessions as sessions_mod

    checks: list[dict] = []
    checks.append(_check("systemd unit", True if unit_installed() else None,
                         str(Path.home() / ".config/systemd/user" / UNIT)))
    pid = running_pid()
    checks.append(_check("service", bool(pid),
                         f"pid {pid}" if pid else "not running -- `smartcomp start`"))

    host, port = config.MCP_HOST, config.MCP_PORT
    try:
        with socket.create_connection((host, port), timeout=3):
            listening = True
    except OSError:
        listening = False
    checks.append(_check("endpoint listening", listening, f"{host}:{port}"))

    token_path = config.token_path()
    try:
        token = config.read_token()
        mode = token_path.stat().st_mode & 0o777
        loose = mode & 0o077
        checks.append(_check("token", True if not loose else None,
                             f"{token_path} mode {oct(mode)[2:]}"
                             + (" (group/world readable!)" if loose else "")))
    except Exception as exc:
        token = ""
        checks.append(_check("token", False, str(exc)[:160]))

    if token:
        try:
            tools = rpc.list_tools(f"http://{host}:{port}/mcp", token, timeout=10)
            checks.append(_check("mcp tools/list", bool(tools), f"{len(tools)} tools"))
        except Exception as exc:
            checks.append(_check("mcp tools/list", False, str(exc)[:200]))
    else:
        checks.append(_check("mcp tools/list", False, "no token to authenticate with"))

    disp = Display().status()
    checks.append(_check("X display", disp.get("dpms_enabled") is not None,
                         f"{disp.get('display')} blank={disp.get('display_blank')}"))
    locked = lock.is_locked()
    checks.append(_check("lock state readable", None if locked is None else True,
                         "unknown (no screensaver on the bus)" if locked is None
                         else "locked" if locked else "unlocked"))

    has_ocr = bool(shutil.which("tesseract"))
    checks.append(_check("ocr engine", True if has_ocr else None,
                         "tesseract" if has_ocr else "tesseract not found on PATH"))
    has_xvfb, has_xephyr = bool(shutil.which("Xvfb")), bool(shutil.which("Xephyr"))
    checks.append(_check("virtual displays", True if (has_xvfb and has_xephyr) else None,
                         f"Xvfb={'yes' if has_xvfb else 'NO'} "
                         f"Xephyr={'yes' if has_xephyr else 'NO'} (needed for locked work)"))

    try:
        listed = sessions_mod.manager().list()
        bad = [s["name"] for s in listed if not (s.get("alive") and s.get("healthy"))]
        checks.append(_check("sessions", True if not bad else None,
                             f"{len(listed)} registered"
                             + (f", unhealthy: {', '.join(bad)}" if bad else "")))
    except Exception as exc:
        checks.append(_check("sessions", None, f"registry unreadable: {exc}"[:160]))

    orphans = _orphans()
    checks.append(_check("stray processes", True if not orphans else None,
                         f"{len(orphans)}: " + "; ".join(orphans[:3]) if orphans
                         else "none"))

    failed = [c for c in checks if c["ok"] is False]
    return {"ok": not failed, "checks": checks}


def print_doctor(report: dict, log_tail: int = 8) -> int:
    for check in report["checks"]:
        mark = "PASS" if check["ok"] else "WARN" if check["ok"] is None else "FAIL"
        print(f"{mark}  {check['name']:<20} {check['detail']}")
    failed = [c for c in report["checks"] if c["ok"] is False]
    if failed:
        print(f"\n{len(failed)} problem(s). Last daemon log lines:")
        try:
            lines = LOG_FILE.read_text(errors="replace").splitlines()[-log_tail:]
            for line in lines:
                print(f"  {line}")
        except OSError as exc:
            print(f"  (log unreadable: {exc})")
    else:
        print("\nno errors found")
    return 0 if report["ok"] else 1
