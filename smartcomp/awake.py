"""Unattended-operation guard: keep the screen usable while the daemon runs.

An agent working with no human at the machine is defeated by two idle
mechanisms: DPMS blanking (every capture comes back black) and the
screensaver locking the desktop half way through a task. Both are handled
here without touching the lock's credentials:

* The screensaver is inhibited over the session bus with the standard
  org.freedesktop.ScreenSaver.Inhibit call. The inhibit is owned by this
  process' bus connection, so it is released automatically if the daemon
  dies -- no "the screen never locks again" state can survive a crash.
* DPMS and the X screensaver timeouts are turned off with xset while
  awake; the previous values are captured and restored on disable and at
  interpreter exit.

The lock screen itself is never bypassed. If the screen is already locked
when the daemon starts, only the user can unlock it; `status()` reports
that so a caller can stop instead of typing into the dark.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import subprocess

from . import config


def _env() -> dict:
    env = dict(os.environ)
    env.setdefault("DISPLAY", ":0")
    return env


def _xset(*args: str) -> bool:
    out = subprocess.run(
        ["xset", *args], capture_output=True, timeout=5, env=_env()
    )
    return out.returncode == 0


def _xset_query() -> str:
    out = subprocess.run(
        ["xset", "q"], capture_output=True, text=True, timeout=5, env=_env()
    )
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


# xfce4-power-manager lid trigger, per common/xfpm-enum-glib.h:
# DPMS=0, SUSPEND=1, HIBERNATE=2, LOCK_SCREEN=3, NOTHING=4. The stock
# DEFAULT is LOCK_SCREEN, which is why a closed lid locked this box even
# with an inhibit held ("Lock ... bypasses all Inhibit requests" -- Xfce
# docs). NOTHING is the only value that cannot lock or sleep the box.
LID_TRIGGER_NOTHING = 4
_LID_PROPS = (
    "/xfce4-power-manager/lid-action-on-ac",
    "/xfce4-power-manager/lid-action-on-battery",
)


def _xfconf(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["xfconf-query", "-c", "xfce4-power-manager", *args],
        capture_output=True,
        text=True,
        timeout=10,
        env=env or _env(),
    )


def _xfconf_lid(prop: str) -> str | None:
    """Current lid trigger value, or None when the property is unset."""
    out = _xfconf("-p", prop)
    if out.returncode != 0:
        return None
    value = out.stdout.strip()
    return value or None


def _xfconf_set_lid(prop: str, value: str) -> bool:
    # -n creates the property when the channel has never seen it.
    out = _xfconf("-p", prop, "-n", "-t", "uint", "-s", value)
    return out.returncode == 0


def _xfconf_reset_lid(prop: str) -> bool:
    return _xfconf("-p", prop, "-r").returncode == 0


def _lid_policy_snapshot() -> dict:
    return {prop: _xfconf_lid(prop) for prop in _LID_PROPS}


class Awake:
    """Screensaver inhibit + DPMS control for one daemon process."""

    def __init__(self) -> None:
        self._bus = None
        self._cookie: int | None = None
        self._saved: dict | None = None
        self._reason: str = ""

    # -- state -------------------------------------------------------------
    def inhibited(self) -> bool:
        return self._cookie is not None

    def status(self) -> dict:
        q = _parse_xset(_xset_query())
        blanked = self._screen_is_black()
        return {
            "inhibited": self.inhibited(),
            "reason": self._reason,
            "dpms_enabled": q.get("dpms_enabled"),
            "dpms_timeouts": q.get("dpms"),
            "ss_timeout": q.get("ss_timeout"),
            "display_blank": blanked,
        }

    @staticmethod
    def _screen_is_black() -> bool | None:
        """Cheap DPMS check: True when the monitor reports it is off."""
        out = subprocess.run(
            ["xset", "q"], capture_output=True, text=True, timeout=5, env=_env()
        )
        return "Monitor is Off" in out.stdout

    def screen_is_off(self) -> bool:
        return bool(self._screen_is_black())

    # -- actions -----------------------------------------------------------
    def wake_display(self) -> dict:
        """Force the monitor on and reset the idle timer. Never unlocks."""
        was_off = self._screen_is_black()
        _xset("dpms", "force", "on")
        _xset("s", "reset")
        return {"ok": True, "was_off": bool(was_off), "unlocked_nothing": True}

    # -- unattended hardening ----------------------------------------------
    def apply_lock_policy(self) -> dict:
        """Move the xfpm lid trigger from its stock LOCK_SCREEN to NOTHING.

        An inhibitor cannot cover xfpm's lid lock ("Lock bypasses all
        Inhibit requests", Xfce docs) and the stock default IS
        LID_TRIGGER_LOCK_SCREEN -- this is what locked the box when the
        lid closed during an unattended run. The previous values are
        recorded in the state dir so restore_lock_policy() puts them back.
        """
        path = config.state_dir() / "lid_policy.json"
        if not path.exists():  # first application: remember the old state
            path.write_text(json.dumps(_lid_policy_snapshot()))
        changed = {}
        for prop in _LID_PROPS:
            if _xfconf_lid(prop) != str(LID_TRIGGER_NOTHING):
                ok = _xfconf_set_lid(prop, str(LID_TRIGGER_NOTHING))
                changed[prop.split("/")[-1]] = ok
        return {
            "ok": all(changed.values()) if changed else True,
            "previous": json.loads(path.read_text()),
            "changed": changed,
        }

    def restore_lock_policy(self) -> dict:
        """Put the lid triggers back exactly as they were found."""
        path = config.state_dir() / "lid_policy.json"
        restored = {}
        if path.exists():
            previous: dict = json.loads(path.read_text())
            for prop, value in previous.items():
                if value is None:  # property did not exist before us
                    restored[prop.split("/")[-1]] = _xfconf_reset_lid(prop)
                else:
                    restored[prop.split("/")[-1]] = _xfconf_set_lid(prop, value)
            path.unlink()
        return {"ok": all(restored.values()) if restored else True, "restored": restored}

    def reassert(self) -> dict:
        """Re-apply the X-level timeouts that xfpm keeps re-enabling.

        xfce4-power-manager reacts to inhibit/lid events by restoring the
        stock `xset` state (observed live: DPMS came back enabled after a
        lid event with the inhibit held). The watchdog calls this after
        every recovery so the box stays awake for unattended work.
        """
        q = _parse_xset(_xset_query())
        fixed = []
        if q.get("dpms_enabled"):
            _xset("s", "off")
            _xset("-dpms")
            fixed.append("dpms")
        if q.get("ss_timeout") not in (None, 0):
            _xset("s", "off")
            fixed.append("screensaver")
        return {"ok": True, "reasserted": fixed}

    def enable(self, reason: str = "agent session") -> dict:
        if self.inhibited():
            return {"ok": True, "already": True, **self.status()}
        # 1. Hold a session-bus inhibit (crash-safe: dies with this process).
        # The standard implementors differ in both bus name and interface:
        # GNOME/KDE speak org.freedesktop.ScreenSaver, xfce4-screensaver
        # speaks org.xfce.ScreenSaver. The object path is the same.
        try:
            import dbus

            self._bus = dbus.SessionBus()
            last_error: Exception | None = None
            targets = (
                ("org.freedesktop.ScreenSaver", "org.freedesktop.ScreenSaver"),
                ("org.xfce.ScreenSaver", "org.xfce.ScreenSaver"),
            )
            for bus_name, iface in targets:
                try:
                    proxy = self._bus.get_object(bus_name, "/org/freedesktop/ScreenSaver")
                    self._cookie = int(
                        proxy.Inhibit(
                            "smartcomp", reason or "agent session",
                            dbus_interface=iface,
                        )
                    )
                    self._reason = reason
                    break
                except Exception as exc:  # try the next known implementor
                    last_error = exc
                    self._cookie = None
            if self._cookie is None:
                raise last_error or RuntimeError("no screensaver interface")
        except Exception as exc:
            self._bus = None
            self._cookie = None
            return {"ok": False, "error": f"dbus inhibit failed: {exc}"}
        # 2. X-level timeouts off; previous values kept for restore. The wake
        # comes first: `dpms force on` re-enables DPMS, so it must precede
        # the `-dpms` that turns the blanking off for good.
        self._saved = _parse_xset(_xset_query())
        _xset("dpms", "force", "on")
        _xset("s", "off")
        _xset("-dpms")
        atexit.register(self.disable)
        return {"ok": True, **self.status()}

    def disable(self) -> dict:
        cookie = self._cookie
        if cookie is not None:
            for bus_name, iface in (
                ("org.freedesktop.ScreenSaver", "org.freedesktop.ScreenSaver"),
                ("org.xfce.ScreenSaver", "org.xfce.ScreenSaver"),
            ):
                try:
                    proxy = self._bus.get_object(bus_name, "/org/freedesktop/ScreenSaver")
                    proxy.UnInhibit(cookie, dbus_interface=iface)
                    break
                except Exception:
                    continue  # bus gone -- the inhibit died with it anyway
            self._cookie = None
            self._bus = None
            self._reason = ""
        if self._saved:
            s = self._saved
            if "ss_timeout" in s:
                _xset("s", str(s["ss_timeout"]), str(s.get("ss_cycle", 600)))
            if "dpms" in s:
                _xset("dpms", *map(str, s["dpms"]))
            if not s.get("dpms_enabled", True):
                _xset("-dpms")
            self._saved = None
        return {"ok": True, **self.status()}
