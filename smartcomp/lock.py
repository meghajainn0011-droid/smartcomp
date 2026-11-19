"""Lock-state detection and autonomous recovery for unattended work.

The unattended contract is "keep working whether the screen is locked or
not", with no human present. Research (Xfce docs, xfpm source, community
threads) plus live verification on this box settled the mechanics:

* Detection: `is_locked()` asks the screensaver over the session bus
  (org.freedesktop.ScreenSaver.GetActive, with the org.xfce.ScreenSaver
  variant this desktop exposes). Tools can check before acting.
* Recovery: xfce4-screensaver releases its own lock when asked to
  deactivate over the bus (`SetActive(FALSE)` -- the documented API). No
  password typed, nothing killed, no interference. `hard_unlock()` drives
  that plus CLI fallbacks; `unlock()` falls back to typing a user-stored
  credential (mode-0600 file, created by the user) for lockers that
  refuse deactivation.
* The guard: `ensure_screen()` is called by every screen-touching tool,
  so work that arrives while the box is locked or blanked recovers the
  screen first and then runs. The system itself keeps its stock lock and
  blank behaviour -- nothing here holds it unlocked between actions.

Why locks happen despite the inhibit: an inhibitor (awake.py) stops the
*idle* lock only. The stock xfce4-power-manager lid policy is
LID_TRIGGER_LOCK_SCREEN (lid-action default 3; 4 = nothing, per
xfpm-enum-glib.h) and xfpm's Lock call "bypasses all Inhibit requests"
per the Xfce docs. By default the daemon no longer changes any of that:
the lid_lock tool applies/restores the lid policy explicitly when asked.
"""

from __future__ import annotations

import logging
import os
import stat
import subprocess
import threading
import time

from . import config

log = logging.getLogger("smartcomp.lock")

_BUS_TARGETS = (
    ("org.freedesktop.ScreenSaver", "org.freedesktop.ScreenSaver"),
    ("org.xfce.ScreenSaver", "org.xfce.ScreenSaver"),
)
# The object path the Xfce docs document (and the one that empirically
# releases the lock); the freedesktop path is the GNOME-compatible alias.
_DBUS_PATHS = ("/org/xfce/ScreenSaver", "/org/freedesktop/ScreenSaver")


def credential_path():
    """Where the user stores the login password for autonomous recovery."""
    return config.state_dir() / "unlock"


def credential_available() -> bool:
    p = credential_path()
    return p.exists() and p.stat().st_size > 0


def is_locked() -> bool | None:
    """True when a screensaver lock holds the screen, None if undetectable."""
    try:
        import dbus

        bus = dbus.SessionBus()
    except Exception:
        return None
    for bus_name, iface in _BUS_TARGETS:
        try:
            proxy = bus.get_object(bus_name, "/org/freedesktop/ScreenSaver")
            return bool(
                proxy.GetActive(dbus_interface=iface)
            )
        except Exception:
            continue
    return None


def read_credential() -> str | None:
    """Return the stored password, refusing a world/group-readable file."""
    p = credential_path()
    if not credential_available():
        return None
    mode = stat.S_IMODE(p.stat().st_mode)
    if mode & 0o077:
        return None  # too open: refuse rather than type a leaked secret
    text = p.read_text().strip()
    return text or None


def hard_unlock(awake=None) -> dict:
    """Release a held lock through the screensaver's own deactivate API.

    Research + live verification on this box (2026-11-16): xfce4-screensaver
    hands the screen back when asked to deactivate over the session bus --
    the documented `SetActive(FALSE)` semantics -- with no password typed
    and no process killed. This is the no-interference recovery path; the
    credential typing in `unlock()` only matters for lockers that refuse
    deactivation.
    """
    if awake is not None:
        awake.wake_display()
    tried: list[str] = []
    try:
        import dbus

        bus = dbus.SessionBus()
    except Exception as exc:  # no session bus -> try the CLI fallbacks
        tried.append(f"dbus unavailable: {type(exc).__name__}")
        bus = None

    if bus is not None:
        for path in _DBUS_PATHS:
            for bus_name, iface in _BUS_TARGETS:
                try:
                    proxy = bus.get_object(bus_name, path)
                    proxy.SetActive(False, dbus_interface=iface)
                except Exception as exc:
                    tried.append(f"{bus_name} {path}: {type(exc).__name__}")
                    continue
                tried.append(f"SetActive via {bus_name} {path}")
                for _ in range(6):  # the dialog takes a beat to go away
                    time.sleep(0.3)
                    if is_locked() is False:
                        return {"ok": True, "method": f"SetActive {bus_name} {path}"}

    # CLI fallbacks for other lock implementations (light-locker etc.).
    env = dict(os.environ)
    env.setdefault("DISPLAY", ":0")
    for cmd in (
        ["xfce4-screensaver-command", "--deactivate"],
        ["loginctl", "unlock-session"],
    ):
        try:
            subprocess.run(cmd, capture_output=True, timeout=10, env=env)
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            tried.append(f"{cmd[0]}: {type(exc).__name__}")
            continue
        tried.append(cmd[0])
        for _ in range(6):
            time.sleep(0.3)
            if is_locked() is False:
                return {"ok": True, "method": cmd[0]}
    return {"ok": False, "tried": tried}


_recovery_input_instance = None
_default_awake_instance = None
_RECOVER_LOCK = threading.Lock()
_STATE_CACHE: tuple[float, bool | None] | None = None


def _default_awake():
    global _default_awake_instance
    if _default_awake_instance is None:
        from .awake import Awake

        _default_awake_instance = Awake()
    return _default_awake_instance


def _recovery_input():
    """A long-lived Input for credential fallback, built lazily and reused."""
    global _recovery_input_instance
    if _recovery_input_instance is None:
        try:
            from .input import Input

            _recovery_input_instance = Input()
        except Exception as exc:
            log.warning("cannot open XTEST for credential fallback: %s", exc)
    return _recovery_input_instance


def ensure_screen(awake=None) -> dict:
    """Make the screen usable before a perception or input action.

    This is the CUA-style background contract: the system keeps its normal
    lock/blank behaviour, and SmartComp works straight through it. Any tool
    that needs to see or touch the screen calls this first, so work that
    arrives while the box is locked or blanked wakes the display and
    releases the lock autonomously (no password, nothing typed -- verified
    live on this box), runs, and the desktop returns to its stock idle
    behaviour afterwards.

    The lock-state check is cached for 2 s so bursts of tool calls do not
    hammer the session bus; a cache hit that is wrong by one lock event
    only delays recovery by one action.
    """
    global _STATE_CACHE
    if not config.screen_guard_enabled():
        return {"locked": is_locked() is True, "woke": False, "guard": False}
    now = time.monotonic()
    cached = _STATE_CACHE
    locked = None
    if cached is not None and now - cached[0] < 2.0:
        locked = cached[1]
    else:
        locked = is_locked()
        _STATE_CACHE = (now, locked)

    woke = False
    if awake is not None and awake.screen_is_off():
        awake.wake_display()
        woke = True

    if locked is not True:
        return {"locked": False, "woke": woke}

    with _RECOVER_LOCK:
        # re-check under the lock: another tool may have recovered already
        if is_locked() is True:
            log.warning("screen locked; recovering before the requested action")
            awake = awake or _default_awake()
            awake.wake_display()
            result = hard_unlock(awake)
            if not result.get("ok") and credential_available():
                inp = _recovery_input()
                if inp is not None:
                    result = unlock(inp, awake=awake)
            _STATE_CACHE = (time.monotonic(), is_locked())
            if not result.get("ok"):
                return {"locked": True, "woke": True, "recovered": False, "detail": result}
            log.info("screen recovered: %s", result)
            return {"locked": False, "woke": True, "recovered": True, "method": result.get("method")}
    return {"locked": False, "woke": woke}


def unlock(inp, awake=None, attempts: int = 2) -> dict:
    """Recover from a lock so unattended work can continue.

    Order: (1) the screensaver's own deactivate API -- no credential, no
    interference, verified to release this desktop's lock; (2) typing the
    user-stored password via XTEST, exactly what a human at the keyboard
    would do, for lockers that refuse deactivation. `inp` is the live
    Input used for step 2 only.
    """
    locked = is_locked()
    if locked is False:
        return {"ok": True, "already_unlocked": True}
    if locked is None:
        return {"ok": False, "error": "cannot detect lock state (no session bus)"}

    hard = hard_unlock(awake)
    if hard.get("ok"):
        return hard

    password = read_credential()
    if password is None:
        p = credential_path()
        if credential_available():
            return {
                "ok": False,
                "error": f"credential file is group/world readable; chmod 600 {p}",
                "hard_unlock_tried": hard.get("tried"),
            }
        return {
            "ok": False,
            "error": "deactivate refused and no stored credential",
            "how": f"install -m 600 /dev/null {p} && edit it to hold the login password",
            "hard_unlock_tried": hard.get("tried"),
        }

    if awake is not None:
        awake.wake_display()
    time.sleep(0.3)

    for attempt in range(1, max(1, attempts) + 1):
        inp.type_text(password)
        inp.key("Return")
        for _ in range(10):  # the dialog takes a moment to hand the screen back
            time.sleep(0.4)
            if is_locked() is False:
                return {"ok": True, "attempts": attempt, "method": "credential"}
        # second try: make sure the password field really has focus
        w, h = inp.screen_size
        inp.click(int(w * 0.5), int(h * 0.52))
        time.sleep(0.4)
    return {"ok": False, "error": "password rejected or lock still held", "attempts": attempts}
