"""Input: pointer, keyboard and clipboard over XTEST.

Everything goes through the X server's TEST extension, so the same code drives
the real screen and a per-session Xephyr display.

Two details that break naive implementations:
  * One long-lived Display connection is held for the life of the daemon. A
    pointer warp made by a short-lived client is discarded when that client
    disconnects, and a drag that reconnects between press and release silently
    becomes a click.
  * Motions are sent as a stream, not a jump. Pages and canvases coalesce or
    drop a single teleport, so drags interpolate with a duration.
"""

from __future__ import annotations

import subprocess
import time

from Xlib import X, XK, display as xdisplay
from Xlib.ext import xtest


class InputError(RuntimeError):
    pass


_MODIFIERS = {
    "ctrl": "Control_L",
    "control": "Control_L",
    "alt": "Alt_L",
    "shift": "Shift_L",
    "super": "Super_L",
    "meta": "Super_L",
}

_SPECIAL = {
    " ": "space",
    "\n": "Return",
    "\t": "Tab",
    "\r": "Return",
}

BUTTONS = {"left": 1, "middle": 2, "right": 3}

# Wheel ticks per scroll() call. One tick is one press/release pair; 20 is
# more than a page on any viewport. Without the cap, a model asking to
# "scroll down" with amount=400 fires 400 wheel clicks and the page is gone.
MAX_SCROLL_TICKS = 20


class Input:
    def __init__(self, display: str | None = None, screen_size: tuple[int, int] = (1920, 1080)):
        self._d = xdisplay.Display(display)
        geometry = self._d.screen().root.get_geometry()
        self.screen_size = (geometry.width, geometry.height) if screen_size is None else screen_size
        self._held: list[int] = []

    # -- geometry ---------------------------------------------------------
    def to_pixels(self, x_pm: int, y_pm: int) -> tuple[int, int]:
        """Permille of the logical frame -> real screen pixels."""
        w, h = self.screen_size
        return int(x_pm / 1000 * w), int(y_pm / 1000 * h)

    def position(self) -> tuple[int, int]:
        q = self._d.screen().root.query_pointer()
        return q.root_x, q.root_y

    # -- pointer ----------------------------------------------------------
    def move(self, x: int, y: int) -> None:
        xtest.fake_input(self._d, X.MotionNotify, x=int(x), y=int(y))
        self._d.sync()

    def button(self, name_or_number: int | str, press: bool) -> None:
        button = (
            BUTTONS.get(name_or_number.lower(), 1)
            if isinstance(name_or_number, str)
            else int(name_or_number)
        )
        xtest.fake_input(self._d, X.ButtonPress if press else X.ButtonRelease, button)
        self._d.sync()

    def click(self, x: int | None = None, y: int | None = None, button: int | str = "left",
              count: int = 1, delay: float = 0.05) -> None:
        if x is not None and y is not None:
            self.move(x, y)
            time.sleep(0.02)
        number = BUTTONS.get(button.lower(), 1) if isinstance(button, str) else int(button)
        for _ in range(count):
            xtest.fake_input(self._d, X.ButtonPress, number)
            self._d.sync()
            time.sleep(0.012)
            xtest.fake_input(self._d, X.ButtonRelease, number)
            self._d.sync()
            time.sleep(delay)

    def drag(self, points: list[tuple[int, int]], button: int | str = "left",
             duration: float = 0.0, step_delay: float = 0.004) -> None:
        """Press at the first point, travel through the rest, release at the last.

        duration > 0 spreads the travel over that many seconds (a "human" drag
        for canvases and sliders); otherwise each point costs step_delay.
        """
        if len(points) < 2:
            raise InputError("drag needs at least two points")
        number = BUTTONS.get(button.lower(), 1) if isinstance(button, str) else int(button)
        per_point = duration / max(1, len(points) - 1) if duration else step_delay
        self.move(*points[0])
        time.sleep(0.02)
        xtest.fake_input(self._d, X.ButtonPress, number)
        self._d.sync()
        for x, y in points[1:]:
            xtest.fake_input(self._d, X.MotionNotify, x=int(x), y=int(y))
            self._d.sync()
            if per_point:
                time.sleep(per_point)
        time.sleep(0.01)
        xtest.fake_input(self._d, X.ButtonRelease, number)
        self._d.sync()

    def scroll(self, amount: int, axis: str = "vertical", x: int | None = None,
               y: int | None = None) -> None:
        """Positive scrolls down/right, negative up/left.

        amount is wheel ticks and is capped at MAX_SCROLL_TICKS per call;
        the scroll tool reports when the cap bit so the model can call again
        instead of assuming the whole request went through.
        """
        if x is not None and y is not None:
            self.move(x, y)
        ticks = max(-MAX_SCROLL_TICKS, min(MAX_SCROLL_TICKS, int(amount)))
        if axis.lower().startswith("h"):
            button = 7 if ticks > 0 else 6
        else:
            button = 5 if ticks > 0 else 4
        for _ in range(abs(ticks)):
            xtest.fake_input(self._d, X.ButtonPress, button)
            xtest.fake_input(self._d, X.ButtonRelease, button)
            self._d.sync()
            time.sleep(0.01)

    # -- keyboard ---------------------------------------------------------
    # LLMs write key names the way people say them ("backspace", "esc",
    # "enter", "pgdn"); X11 keysym lookup is case-sensitive and canonical
    # ("BackSpace", "Escape", "Return", "Page_Down"), so the plain name
    # failed with "unknown key" -- the agent's natural vocabulary rejected
    # by the tool that exists to translate for it. Aliases for the common
    # ones, plus a case-insensitive sweep of the keysym table, both bound
    # once at import.
    _KEY_ALIASES = {
        "backspace": "BackSpace",
        "del": "Delete",
        "esc": "Escape",
        "enter": "Return",
        "ins": "Insert",
        "pagedown": "Page_Down",
        "pageup": "Page_Up",
        "pgdn": "Page_Down",
        "pgup": "Page_Up",
        "spacebar": "space",
        "tab": "Tab",
    }
    _KEYSYM_BY_LOWER = {
        name[len("XK_"):].lower(): value
        for name, value in vars(XK).items()
        if name.startswith("XK_") and isinstance(value, int)
    }

    @classmethod
    def _keysym(cls, name: str) -> int:
        keysym = XK.string_to_keysym(name)
        if not keysym:
            keysym = XK.string_to_keysym(cls._KEY_ALIASES.get(name.lower(), ""))
        if not keysym:
            keysym = cls._KEYSYM_BY_LOWER.get(name.lower(), 0)
        if not keysym and len(name) == 1:
            keysym = ord(name)
        return keysym

    def _keycode(self, name: str) -> tuple[int, bool]:
        keysym = self._keysym(name)
        if keysym == 0:
            raise InputError(f"unknown key: {name!r}")
        keycode = self._d.keysym_to_keycode(keysym)
        if not keycode:
            raise InputError(f"no keycode for {name!r}")
        needs_shift = self._d.keycode_to_keysym(keycode, 0) != keysym
        return keycode, needs_shift

    def key(self, combo: str) -> None:
        """Press a chord such as 'ctrl+l' or a single key such as 'Return'."""
        parts = [p.strip() for p in combo.split("+") if p.strip()]
        if not parts:
            return
        modifiers = [_MODIFIERS[p.lower()] for p in parts[:-1] if p.lower() in _MODIFIERS]
        target = parts[-1]
        held = []
        for mod in modifiers:
            code, _ = self._keycode(mod)
            xtest.fake_input(self._d, X.KeyPress, code)
            held.append(code)
        code, shift = self._keycode(target)
        if shift:
            shift_code, _ = self._keycode("Shift_L")
            xtest.fake_input(self._d, X.KeyPress, shift_code)
            held.append(shift_code)
        self._d.sync()
        xtest.fake_input(self._d, X.KeyPress, code)
        xtest.fake_input(self._d, X.KeyRelease, code)
        self._d.sync()
        for code in reversed(held):
            xtest.fake_input(self._d, X.KeyRelease, code)
        self._d.sync()

    def key_batch(self, combo: str) -> None:
        """Hold a modifier for a series of keys: 'ctrl+a ctrl+c'.

        Not implemented as a tool yet; a plain sequence of chords is cheaper.
        """
        for part in combo.split():
            self.key(part)

    def key_down(self, combo: str) -> None:
        for name in [p.strip() for p in combo.split("+") if p.strip()]:
            code, _ = self._keycode(_MODIFIERS.get(name.lower(), name))
            xtest.fake_input(self._d, X.KeyPress, code)
            self._held.append(code)
        self._d.sync()

    def key_up(self, combo: str) -> None:
        for name in reversed([p.strip() for p in combo.split("+") if p.strip()]):
            code, _ = self._keycode(_MODIFIERS.get(name.lower(), name))
            xtest.fake_input(self._d, X.KeyRelease, code)
            if code in self._held:
                self._held.remove(code)
        self._d.sync()

    def type_text(self, text: str, delay: float = 0.012) -> None:
        for ch in text:
            self.key(_SPECIAL.get(ch, ch))
            time.sleep(delay)

    # -- clipboard --------------------------------------------------------
    def clipboard_get(self) -> str:
        try:
            out = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"], capture_output=True, timeout=5
            )
        except FileNotFoundError as exc:
            raise InputError("xclip is not installed") from exc
        return out.stdout.decode("utf-8", "replace")

    def clipboard_set(self, text: str) -> None:
        """Hand text to xclip, which forks and keeps owning the selection.

        The owner process must not inherit our stdout/stderr: run() with
        capture_output waits for those pipes to close, they stay open for
        the owner's lifetime, and every call eats the full timeout -- which
        is why this tool was 0/9 in the action log while clipboard_get was
        10/10. stdin is the only pipe the transfer needs and it is closed
        the moment the data is written; everything else is DEVNULL, and the
        owner runs in its own session like any other detached process.
        """
        try:
            proc = subprocess.Popen(
                ["xclip", "-selection", "clipboard", "-i"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise InputError("xclip is not installed") from exc
        try:
            # The parent exits right after forking the selection owner, so
            # this returns quickly; the timeout only bites on a broken xclip.
            proc.communicate(text.encode(), timeout=5)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            raise InputError("xclip did not take the clipboard in time") from exc
        if proc.returncode:
            raise InputError(f"xclip exited with code {proc.returncode}")


def launch(binary: str, args: list[str] | None = None, display: str | None = None) -> int:
    """Start a GUI app so it outlives this daemon, using the name the desktop knows.

    The daemon is a systemd user service, and systemd kills the whole
    service cgroup when the service stops: a child app died with every
    daemon restart (measured live -- a Firefox launched through this path
    was swept away mid-session by the next restart's stop sweep, with no
    OOM and no crash of its own). So under a user manager the app goes
    into its own transient service (smartcomp-app-*), a cgroup the daemon
    does not own; without one it falls back to the old detached child.
    """
    cmd = [binary, *(args or [])]
    pid = _launch_via_user_manager(cmd, display)
    if pid is not None:
        return pid
    return _launch_detached(cmd, display)


def _launch_env(display: str | None) -> dict[str, str]:
    """The GUI-relevant env a transient service would not inherit.

    A systemd user service does not get the daemon's environment, so the
    display connection has to be forwarded explicitly (XDG_RUNTIME_DIR and
    the rest of the user-session env the manager already provides).
    """
    import os

    env: dict[str, str] = {}
    wanted = display or os.environ.get("DISPLAY")
    if wanted:
        env["DISPLAY"] = wanted
    for key in ("XAUTHORITY", "WAYLAND_DISPLAY", "DBUS_SESSION_BUS_ADDRESS"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def _launch_via_user_manager(cmd: list[str], display: str | None) -> int | None:
    """Run `cmd` as its own transient user service; return its pid.

    None means "no user manager here or the start was refused" and the
    caller falls back to a plain detached child. Returns 0 when the unit
    started but the pid could not be read in time -- the census finds the
    process by name, so nothing depends on the exact pid.
    """
    import os
    import re
    import shutil

    if shutil.which("systemd-run") is None:
        return None
    base = re.sub(r"[^A-Za-z0-9]", "", os.path.basename(cmd[0])) or "app"
    unit = f"smartcomp-app-{base}-{int(time.time() * 1000) % 100000:05d}"
    argv = ["systemd-run", "--user", "--collect", f"--unit={unit}"]
    argv += [f"--setenv={k}={v}" for k, v in _launch_env(display).items()]
    argv += ["--", *cmd]
    try:
        started = subprocess.run(argv, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if started.returncode != 0:
        return None
    deadline = time.time() + 1.5
    while time.time() < deadline:
        pid = _unit_main_pid(unit)
        if pid:
            return pid
        time.sleep(0.1)
    return 0


def _unit_main_pid(unit: str) -> int:
    try:
        out = subprocess.run(
            ["systemctl", "--user", "show", "-p", "MainPID", "--value", unit],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    if out.returncode:
        return 0
    try:
        return int(out.stdout.strip())
    except ValueError:
        return 0


def _launch_detached(cmd: list[str], display: str | None) -> int:
    """The fallback: a child in its own session, no user manager involved."""
    import os

    env = None
    if display:
        env = dict(os.environ, DISPLAY=display)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
    except FileNotFoundError as exc:
        raise InputError(f"{cmd[0]} is not installed") from exc
    return proc.pid
