"""Host lock-state detection. Read-only, by design.

SmartComp never unlocks the screen, never wakes a blanked monitor, never
holds a screensaver inhibit and never changes the lid policy. Those actions
are the user's, and an agent that performs them is not "working through a
lock" -- it is taking the lock away.

The contract is the opposite one:

* On the **host display**, a locked screen makes perception and XTEST input
  impossible (the lock screen owns the pixels and the input grab), and typing
  into a lock prompt is actively harmful -- it can put a password into the
  wrong field. So the tool wrapper REFUSES screen-touching calls with
  `locked: true` and a pointer to a session. Nothing is woken or released.
* On a **session display** (`session_create`), the daemon owns its own X
  server. Its captures and input never reach the host's lock screen, so work
  continues while the host is locked, blanked or asleep -- the same
  background behaviour other computer-use stacks have. Sessions are
  detected by `SMARTCOMP_SESSION_NAME`, not by a flag an agent can flip.
"""

from __future__ import annotations

import os
import time

from . import config

_BUS_TARGETS = (
    ("org.freedesktop.ScreenSaver", "org.freedesktop.ScreenSaver"),
    ("org.xfce.ScreenSaver", "org.xfce.ScreenSaver"),
)

# Lock-state reads are cached for a second: a burst of tool calls must not
# hammer the session bus, and a one-second stale answer can only delay a
# refusal, never cause a mutation.
_LOCK_CACHE: tuple[float, bool | None] | None = None


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
            return bool(proxy.GetActive(dbus_interface=iface))
        except Exception:
            continue
    return None


def lock_state(refresh: bool = False) -> bool | None:
    """Cached `is_locked()`: cheap enough to call before every screen action."""
    global _LOCK_CACHE
    now = time.monotonic()
    if not refresh and _LOCK_CACHE is not None and now - _LOCK_CACHE[0] < 1.0:
        return _LOCK_CACHE[1]
    state = is_locked()
    _LOCK_CACHE = (now, state)
    return state


def host_display_isolated() -> bool:
    """True when this daemon drives its own display instead of the host's.

    A session daemon is started with SMARTCOMP_SESSION_NAME, so the gate
    applies only to the host-attached daemon. SMARTCOMP_HOST_LOCK_GATE=0 is
    the explicit escape hatch for an operator who knows the attached display
    has no lock of its own.
    """
    if os.environ.get("SMARTCOMP_SESSION_NAME"):
        return True
    return not config.host_lock_gate_enabled()


def blocked_reason() -> dict | None:
    """Why a screen-touching tool cannot run here, or None when it can.

    Pure read: no wake, no unlock, no inhibit. Returning a reason is the
    whole recovery strategy -- the caller switches to a session display.
    """
    if host_display_isolated():
        return None
    if lock_state() is not True:
        return None
    return {
        "error": "host display is locked; smartcomp never unlocks or wakes it",
        "locked": True,
        "display": os.environ.get("DISPLAY", ":0"),
        "hint": "the host desktop stays untouched until the user unlocks it. "
                "session_create(name) gives an agent its own X display that "
                "keeps working while the host is locked; send the tool call "
                "to that session's endpoint (session_endpoint).",
    }
