"""Read-only display status: blanked or not, and the current timeouts.

This module used to hold a screensaver inhibit and mutate the X timeouts with
`xset`, and the tool surface could wake the monitor, disable blanking and
change the lid policy. All of that is gone. The host's lock/blank behaviour
belongs to the user; an agent must not change it, not on purpose and not as a
side effect of "recovering" before an action.

Nothing here writes: the only external call is `xset q`, which reports state.
There is no `ScreenSaver.Inhibit`, no `xset s off`, no `xset dpms force on`
and no `xfconf` write anywhere in smartcomp. A locked host is reported by
`lock.py` and left alone; work continues on session displays, which have
their own X server.
"""

from __future__ import annotations

import os
import re
import subprocess


def _env() -> dict:
    env = dict(os.environ)
    env.setdefault("DISPLAY", ":0")
    return env


def _xset_query() -> str:
    """`xset q` output for this daemon's display; empty when unavailable."""
    try:
        out = subprocess.run(
            ["xset", "q"], capture_output=True, text=True, timeout=5, env=_env()
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout


def _parse_xset(text: str) -> dict:
    """Pull the screensaver and DPMS timeouts out of `xset q` output."""
    state: dict = {"dpms_enabled": "DPMS is Enabled" in text}
    m = re.search(r"timeout:\s*(\d+)\s+cycle:\s*(\d+)", text)
    if m:
        state["ss_timeout"] = int(m.group(1))
        state["ss_cycle"] = int(m.group(2))
    m = re.search(r"Standby:\s*(\d+)\s+Suspend:\s*(\d+)\s+Off:\s*(\d+)", text)
    if m:
        state["dpms"] = [int(m.group(1)), int(m.group(2)), int(m.group(3))]
    return state


class Display:
    """Read-only description of the display this daemon is attached to."""

    def status(self) -> dict:
        text = _xset_query()
        q = _parse_xset(text)
        return {
            "display": os.environ.get("DISPLAY", ":0"),
            "dpms_enabled": q.get("dpms_enabled"),
            "dpms_timeouts": q.get("dpms"),
            "ss_timeout": q.get("ss_timeout"),
            "display_blank": "Monitor is Off" in text,
        }

    def screen_is_off(self) -> bool:
        return "Monitor is Off" in _xset_query()
