"""Window management over EWMH.

The plan wants an agent to be able to focus, move and close what a human could.
The window manager already exposes that (via _NET_* client messages), so this
talks to it directly instead of shelling out to wmctrl, which is not installed
on this box anyway.
"""

from __future__ import annotations

import time
from pathlib import Path

from Xlib import X, display as xdisplay, protocol

_STATE_ATOMS = {
    "maximize": ("_NET_WM_STATE_MAXIMIZED_VERT", "_NET_WM_STATE_MAXIMIZED_HORZ"),
    "maximized": ("_NET_WM_STATE_MAXIMIZED_VERT", "_NET_WM_STATE_MAXIMIZED_HORZ"),
    "fullscreen": ("_NET_WM_STATE_FULLSCREEN",),
}


class Windows:
    def __init__(self, display: str | None = None) -> None:
        self._d = xdisplay.Display(display)
        self._root = self._d.screen().root
        self._cache: dict[str, int] = {}

    # -- atoms ------------------------------------------------------------
    def _atom(self, name: str) -> int:
        if name not in self._cache:
            self._cache[name] = self._d.intern_atom(name)
        return self._cache[name]

    def _win(self, raw):
        """Client-list entries come back as bare window ids, not objects."""
        if hasattr(raw, "get_geometry"):
            return raw
        return self._d.create_resource_object("window", int(raw))

    def _prop(self, wid: int, name: str):
        try:
            prop = wid.get_full_property(self._atom(name), X.AnyPropertyType)
        except Exception:
            return None
        return prop.value if prop else None

    def _title(self, wid) -> str:
        value = self._prop(wid, "_NET_WM_NAME")
        if value is None:
            value = self._prop(wid, "WM_NAME")
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", "replace")
        return str(value)

    def _wm_class(self, wid) -> str:
        value = self._prop(wid, "WM_CLASS")
        if not value:
            return ""
        if isinstance(value, bytes):
            parts = value.split(b"\0")
            return parts[-1].decode("utf-8", "replace") if parts else ""
        return str(value)

    def _wm_instance(self, wid) -> str:
        """The resource *name* (first field of WM_CLASS), not the class."""
        value = self._prop(wid, "WM_CLASS")
        if not value:
            return ""
        if isinstance(value, bytes):
            parts = value.split(b"\0")
            return parts[0].decode("utf-8", "replace") if parts else ""
        return str(value).split("\0")[0]

    def _pid(self, wid) -> int | None:
        """_NET_WM_PID, set by most toolkits; None when the client hides it."""
        value = self._prop(wid, "_NET_WM_PID")
        try:
            return int(value[0]) if value else None
        except (TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _comm(pid: int | None) -> str:
        """Executable name from /proc: the identity when WM_CLASS is empty."""
        if not pid:
            return ""
        try:
            return Path(f"/proc/{pid}/comm").read_text().strip()[:64]
        except OSError:
            return ""

    @staticmethod
    def _cmd(pid: int | None) -> str:
        """Basename of the first argv entry; a title can lie, a command cannot."""
        if not pid:
            return ""
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            first = next((a for a in argv if a), b"")
            return Path(first.decode("utf-8", "replace")).name[:64]
        except OSError:
            return ""

    def _send(self, wid: int, message: str, data: tuple) -> None:
        event = protocol.event.ClientMessage(
            window=wid, client_type=self._atom(message), data=(32, data)
        )
        self._root.send_event(
            event, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask
        )
        self._d.flush()
        self._d.sync()

    # -- queries ----------------------------------------------------------
    def list(self) -> list[dict]:
        clients = self._prop(self._root, "_NET_CLIENT_LIST") or []
        active = self._prop(self._root, "_NET_ACTIVE_WINDOW")
        active_id = active[0] if active else None
        out: list[dict] = []
        for raw in clients:
            wid = self._win(raw)
            try:
                geom = wid.get_geometry()
                # Translating the window's origin into root space gives position.
                coords = self._root.translate_coords(wid, 0, 0)
                cls = self._wm_class(wid)
            except Exception:
                continue
            title = self._title(wid)
            pid = self._pid(wid)
            if not title and not cls and not pid:
                continue
            out.append(
                {
                    "id": hex(wid.id) if hasattr(wid, "id") else hex(wid),
                    "title": title,
                    "class": cls,
                    "instance": self._wm_instance(wid),
                    "pid": pid,
                    "comm": self._comm(pid),
                    "cmd": self._cmd(pid),
                    "x": coords.x,
                    "y": coords.y,
                    "w": geom.width,
                    "h": geom.height,
                    "active": (wid.id if hasattr(wid, "id") else wid) == active_id,
                }
            )
        return out

    def find(self, needle: str) -> list[dict]:
        needle = needle.lower()
        return [
            w
            for w in self.list()
            if needle in w["title"].lower() or needle in w["class"].lower()
        ]

    def _by_id(self, window_id: str | int):
        target = window_id
        if isinstance(target, str):
            target = int(target, 16) if target.lower().startswith("0x") else int(target)
        for raw in self._prop(self._root, "_NET_CLIENT_LIST") or []:
            wid = self._win(raw)
            if (wid.id if hasattr(wid, "id") else wid) == target:
                return wid
        raise KeyError(f"no window {window_id}")

    # -- actions ----------------------------------------------------------
    def activate(self, window_id: str | int) -> dict:
        wid = self._by_id(window_id)
        # 2 = source pager; timestamp 0 lets the WM use its own clock.
        self._send(wid, "_NET_ACTIVE_WINDOW", (2, X.CurrentTime, 0, 0, 0))
        # xfwm4 sometimes only raises without marking focus; stacking is direct
        # and works regardless of what the WM decides about activation.
        try:
            wid.configure(stack_mode=X.Above)
            self._d.sync()
        except Exception:
            pass
        time.sleep(0.15)
        self._d.sync()
        try:
            wid.set_input_focus(X.RevertToParent, X.CurrentTime)
            self._d.sync()
        except Exception:
            pass
        return {"ok": True, "id": hex(wid.id)}

    def close(self, window_id: str | int) -> dict:
        wid = self._by_id(window_id)
        self._send(wid, "_NET_CLOSE_WINDOW", (0, 2, 0, 0, 0))
        return {"ok": True}

    def move_resize(self, window_id: str | int, x: int, y: int, w: int, h: int) -> dict:
        wid = self._by_id(window_id)
        # EWMH data.l[0]: gravity in the low byte, flags in bits 8-11 saying
        # which of x|y|w|h follow. The old (0xF, ...) set gravity 15 with
        # flags 0 -- "no fields present" -- and xfwm4 quietly ignored the
        # whole message (observed live: geometry unchanged).
        self._send(wid, "_NET_MOVERESIZE_WINDOW", (0x1 | 0xF00, x, y, w, h))
        time.sleep(0.25)
        if self._geometry_applied(wid, x, y, w, h):
            return {"ok": True, "geometry": [x, y, w, h], "via": "ewmh"}
        # xfwm4 ignores synthetic messages in general; ask for the geometry
        # directly, the way activate() and the maximize fallback do. The WM
        # still validates it through ConfigureRequest (hints, work area).
        wid.configure(x=x, y=y, width=w, height=h)
        self._d.sync()
        time.sleep(0.15)
        return {"ok": True, "geometry": [x, y, w, h], "via": "configure"}

    def _geometry_applied(self, wid, x: int, y: int, w: int, h: int) -> bool:
        try:
            geom = wid.get_geometry()
            coords = self._root.translate_coords(wid, 0, 0)
        except Exception:
            return False
        # 40 px slack: frame extents shift the client inside the frame, and
        # the WM may snap to size-increment steps.
        return (abs(coords.x - x) <= 40 and abs(coords.y - y) <= 40
                and abs(geom.width - w) <= 40 and abs(geom.height - h) <= 40)

    def set_state(self, window_id: str | int, state: str) -> dict:
        wid = self._by_id(window_id)
        key = state.lower()
        if key not in ("normal", "restore", "maximize", "maximized", "fullscreen"):
            raise KeyError(f"unknown state {state!r}")
        action = 0 if key in ("normal", "restore") else 1
        atom_names = _STATE_ATOMS.get(key, _STATE_ATOMS["maximize"])
        atoms = [self._atom(name) for name in atom_names]
        # One message with both atoms, the way wmctrl sends it.
        self._send(wid, "_NET_WM_STATE",
                   (action, atoms[0], atoms[1] if len(atoms) > 1 else 0, 1, 0))
        time.sleep(0.25)
        if self._state_applied(wid, atoms, add=action == 1):
            return {"ok": True, "state": key, "via": "ewmh"}
        if action == 1:
            # xfwm4 ignores synthetic _NET_WM_STATE messages surprisingly
            # often (observed live: state property unchanged, geometry
            # untouched). Achieve the observable result directly: fill the
            # WM's work area, which is what a maximized window means to an
            # agent. activate() already uses this direct-call fallback.
            wa = (self._prop(self._root, "_NET_WORKAREA") or [0, 0, 0, 0])[:4]
            if wa[2] and wa[3]:
                wid.configure(x=wa[0], y=wa[1], width=wa[2], height=wa[3])
                self._d.sync()
                time.sleep(0.15)
                return {"ok": True, "state": key, "via": "configure"}
        return {"ok": True, "state": key, "via": "ewmh"}

    def _state_applied(self, wid, atoms: list[int], add: bool) -> bool:
        current = self._prop(wid, "_NET_WM_STATE") or []
        if add:
            return all(a in current for a in atoms)
        return all(a not in current for a in atoms)
