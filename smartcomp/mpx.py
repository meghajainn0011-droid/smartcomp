"""One real X cursor per agent: Multi-Pointer X (XInput2) support.

A display has one *core* pointer. Every XTEST client drives it, so two agents
sharing a screen fight over one cursor and the human's mouse jumps under their
clicks. XInput2 gives the server *master* pointers instead: each master is an
independent on-screen cursor with its own keyboard focus and its own XTEST
slave device. Moving one never moves the core pointer, so an agent can work on
the visible desktop while the human keeps their mouse.

The hierarchy calls (add/remove master) go through the ``xinput`` tool: the
libXi entry point for them corrupts the request stream on this build, while the
CLI builds the same request correctly. Everything else -- warp, position,
buttons, keys -- is called straight into libXi/libXtst through ctypes, which is
safe now that every struct is declared with its real layout.

Only available on X11 (XInput 2.0+). ``available()`` says whether this display
can do it; callers fall back to core-pointer input when it cannot.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess

# XInput2 device uses.
MASTER_POINTER = 1
MASTER_KEYBOARD = 2
SLAVE_POINTER = 3
SLAVE_KEYBOARD = 4
XI_ALL_DEVICES = 0

POINTER_ROOT = 1


class MpxError(RuntimeError):
    pass


class XIButtonState(ctypes.Structure):
    _fields_ = [("mask_len", ctypes.c_int), ("mask", ctypes.POINTER(ctypes.c_ubyte))]


class XIModifierState(ctypes.Structure):
    _fields_ = [("base", ctypes.c_int), ("latched", ctypes.c_int),
                ("locked", ctypes.c_int), ("effective", ctypes.c_int)]


class XIGroupState(ctypes.Structure):
    _fields_ = [("base", ctypes.c_int), ("latched", ctypes.c_int),
                ("locked", ctypes.c_int), ("effective", ctypes.c_int)]


class XIDeviceInfo(ctypes.Structure):
    _fields_ = [("deviceid", ctypes.c_int), ("name", ctypes.c_char_p), ("use", ctypes.c_int),
                ("attachment", ctypes.c_int), ("enabled", ctypes.c_int),
                ("num_classes", ctypes.c_int), ("classes", ctypes.c_void_p)]


class XDevice(ctypes.Structure):
    """The client-side handle libXtst reads. Only ``device_id`` is used, so a
    fabricated instance is enough -- XOpenDevice is not needed (and is broken
    in this libXi build: it sends the wrong id and the server rejects it)."""

    _fields_ = [("device_id", ctypes.c_ulong), ("num_classes", ctypes.c_int),
                ("classes", ctypes.c_void_p)]


class Device:
    __slots__ = ("id", "name", "use", "attachment", "enabled")

    def __init__(self, deviceid: int, name: str, use: int, attachment: int, enabled: int):
        self.id = deviceid
        self.name = name
        self.use = use
        self.attachment = attachment
        self.enabled = enabled

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "use": self.use,
                "attachment": self.attachment, "enabled": self.enabled}


_libs: dict | None = None


def _load() -> dict:
    """Bind libX11/libXi/libXtst once, with the real signatures.

    A wrong prototype here does not raise: ctypes writes the reply through a
    pointer of the wrong type and the process dies in the library. XIQueryPointer
    takes XIButtonState*/XIModifierState*/XIGroupState*, not ints.
    """
    global _libs
    if _libs is not None:
        return _libs

    x11 = ctypes.CDLL("libX11.so.6")
    xi = ctypes.CDLL("libXi.so.6")
    xtst = ctypes.CDLL("libXtst.so.6")

    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    x11.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
    x11.XDefaultRootWindow.restype = ctypes.c_ulong
    x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]

    xi.XIQueryVersion.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                  ctypes.POINTER(ctypes.c_int)]
    xi.XIQueryVersion.restype = ctypes.c_int
    xi.XIQueryDevice.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    xi.XIQueryDevice.restype = ctypes.c_void_p
    xi.XIFreeDeviceInfo.argtypes = [ctypes.c_void_p]
    xi.XIWarpPointer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                                 ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_double, ctypes.c_double]
    xi.XIWarpPointer.restype = ctypes.c_int
    xi.XIQueryPointer.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(XIButtonState), ctypes.POINTER(XIModifierState),
        ctypes.POINTER(XIGroupState),
    ]
    xi.XIQueryPointer.restype = ctypes.c_int
    xi.XISetFocus.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong]
    xi.XISetFocus.restype = ctypes.c_int

    xtst.XTestFakeDeviceMotionEvent.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_ulong,
    ]
    xtst.XTestFakeDeviceMotionEvent.restype = ctypes.c_int
    xtst.XTestFakeDeviceButtonEvent.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_ulong,
    ]
    xtst.XTestFakeDeviceButtonEvent.restype = ctypes.c_int
    xtst.XTestFakeDeviceKeyEvent.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_ulong,
    ]
    xtst.XTestFakeDeviceKeyEvent.restype = ctypes.c_int

    _libs = {"x11": x11, "xi": xi, "xtst": xtst}
    return _libs


class _Conn:
    """A private display connection for MPX calls."""

    def __init__(self, display: str | None):
        libs = _load()
        self._libs = libs
        self._dpy = libs["x11"].XOpenDisplay((display or "").encode() or None)
        if not self._dpy:
            raise MpxError(f"cannot open display {display or '(default)'}")
        self._root = libs["x11"].XDefaultRootWindow(self._dpy)

    def sync(self) -> None:
        self._libs["x11"].XSync(self._dpy, 0)

    def close(self) -> None:
        if self._dpy:
            self._libs["x11"].XCloseDisplay(self._dpy)
            self._dpy = None

    def version(self) -> tuple[int, int]:
        maj, mi = ctypes.c_int(2), ctypes.c_int(0)
        status = self._libs["xi"].XIQueryVersion(self._dpy, ctypes.byref(maj), ctypes.byref(mi))
        if status != 0:
            return (0, 0)
        return (maj.value, mi.value)

    def devices(self) -> list[Device]:
        n = ctypes.c_int(0)
        base = self._libs["xi"].XIQueryDevice(self._dpy, XI_ALL_DEVICES, ctypes.byref(n))
        if not base:
            return []
        try:
            arr = ctypes.cast(base, ctypes.POINTER(XIDeviceInfo))
            return [
                Device(arr[i].deviceid, (arr[i].name or b"").decode("utf-8", "replace"),
                       arr[i].use, arr[i].attachment, arr[i].enabled)
                for i in range(n.value)
            ]
        finally:
            self._libs["xi"].XIFreeDeviceInfo(base)

    def position(self, pointer_id: int) -> tuple[int, int]:
        root = ctypes.c_ulong(); child = ctypes.c_ulong()
        rx = ctypes.c_double(); ry = ctypes.c_double()
        wx = ctypes.c_double(); wy = ctypes.c_double()
        buttons = XIButtonState(); mods = XIModifierState(); group = XIGroupState()
        ok = self._libs["xi"].XIQueryPointer(
            self._dpy, pointer_id, self._root,
            ctypes.byref(root), ctypes.byref(child),
            ctypes.byref(rx), ctypes.byref(ry), ctypes.byref(wx), ctypes.byref(wy),
            ctypes.byref(buttons), ctypes.byref(mods), ctypes.byref(group),
        )
        if buttons.mask:
            ctypes.CDLL(None).free(buttons.mask)
        if not ok:
            raise MpxError(f"no such pointer device {pointer_id}")
        return int(rx.value), int(ry.value)

    def warp(self, pointer_id: int, x: float, y: float) -> None:
        status = self._libs["xi"].XIWarpPointer(
            self._dpy, pointer_id, 0, self._root, 0, 0, 0, 0, float(x), float(y)
        )
        self.sync()
        if status != 0:
            raise MpxError(f"XIWarpPointer failed ({status}) for device {pointer_id}")

    def _fake(self, device_id: int) -> XDevice:
        return XDevice(device_id=ctypes.c_ulong(device_id), num_classes=0, classes=None)

    def button(self, device_id: int, button: int, press: bool) -> None:
        dev = self._fake(device_id)
        ok = self._libs["xtst"].XTestFakeDeviceButtonEvent(
            self._dpy, ctypes.byref(dev), int(button), 1 if press else 0, None, 0, 0
        )
        self.sync()
        if not ok:
            raise MpxError(f"XTestFakeDeviceButtonEvent failed for device {device_id}")

    def key(self, device_id: int, keycode: int, press: bool) -> None:
        dev = self._fake(device_id)
        ok = self._libs["xtst"].XTestFakeDeviceKeyEvent(
            self._dpy, ctypes.byref(dev), int(keycode), 1 if press else 0, None, 0, 0
        )
        self.sync()
        if not ok:
            raise MpxError(f"XTestFakeDeviceKeyEvent failed for device {device_id}")

    def pointer_root_focus(self, keyboard_id: int) -> None:
        """Keys follow the window under this master's own cursor."""
        self._libs["xi"].XISetFocus(self._dpy, keyboard_id, POINTER_ROOT, 0)
        self.sync()


class Master:
    """A master pointer + keyboard pair and the XTEST slaves that drive them."""

    __slots__ = ("name", "pointer_id", "keyboard_id", "slave_pointer", "slave_keyboard")

    def __init__(self, name, pointer_id, keyboard_id, slave_pointer, slave_keyboard):
        self.name = name
        self.pointer_id = pointer_id
        self.keyboard_id = keyboard_id
        self.slave_pointer = slave_pointer
        self.slave_keyboard = slave_keyboard

    def as_dict(self) -> dict:
        return {"name": self.name, "pointer": self.pointer_id, "keyboard": self.keyboard_id,
                "slave_pointer": self.slave_pointer, "slave_keyboard": self.slave_keyboard}


def _xinput(*args: str, display: str | None = None) -> subprocess.CompletedProcess:
    if shutil.which("xinput") is None:
        raise MpxError("xinput is not installed (needed to add/remove master pointers)")
    env = None
    if display:
        import os

        env = dict(os.environ, DISPLAY=display)
    return subprocess.run(["xinput", *args], capture_output=True, text=True, timeout=15, env=env)


def _match(conn: _Conn, name: str) -> Master | None:
    """Find the master pair for ``name`` ("alpha" or "alpha pointer").

    The server names a master pair "<name> pointer" / "<name> keyboard", so the
    caller's short name has to be expanded before comparing.
    """
    pointer_name = name if name.endswith(" pointer") else f"{name} pointer"
    keyboard_name = pointer_name[: -len("pointer")] + "keyboard"
    devices = conn.devices()
    pointer = next((d for d in devices if d.use == MASTER_POINTER and d.name == pointer_name), None)
    if pointer is None:
        return None
    keyboard = next((d for d in devices if d.use == MASTER_KEYBOARD
                     and d.name == keyboard_name), None)
    if keyboard is None:
        keyboard = next((d for d in devices if d.use == MASTER_KEYBOARD
                         and d.attachment == pointer.id), None)
    slave_p = next((d for d in devices if d.use == SLAVE_POINTER
                    and d.attachment == pointer.id), None)
    slave_k = next((d for d in devices if d.use == SLAVE_KEYBOARD
                    and keyboard and d.attachment == keyboard.id), None)
    return Master(pointer.name, pointer.id, keyboard.id if keyboard else -1,
                  slave_p.id if slave_p else -1, slave_k.id if slave_k else -1)


def available(display: str | None = None) -> bool:
    try:
        conn = _Conn(display)
    except (OSError, MpxError):
        return False
    try:
        maj, _ = conn.version()
        return maj >= 2 and shutil.which("xinput") is not None
    finally:
        conn.close()


def list_masters(display: str | None = None) -> list[dict]:
    conn = _Conn(display)
    try:
        return [d.as_dict() for d in conn.devices() if d.use == MASTER_POINTER]
    finally:
        conn.close()


def ensure(display: str | None, name: str) -> Master:
    """Return the master called ``name``, creating it when missing.

    The name must be unique per agent: the server keeps it until it is removed,
    so two agents never share a cursor by accident.
    """
    conn = _Conn(display)
    try:
        if conn.version()[0] < 2:
            raise MpxError("this X server has no XInput 2.0 (no multi-pointer support)")
        existing = _match(conn, name)
        if existing is not None:
            conn.pointer_root_focus(existing.keyboard_id)
            return existing
        done = _xinput("create-master", name, display=display)
        if done.returncode != 0:
            raise MpxError(f"xinput create-master failed: {done.stderr.strip() or done.stdout.strip()}")
        master = _match(conn, name)
        if master is None:
            raise MpxError(f"master {name!r} was not created")
        conn.pointer_root_focus(master.keyboard_id)
        return master
    finally:
        conn.close()


def destroy(display: str | None, name: str) -> bool:
    conn = _Conn(display)
    try:
        master = _match(conn, name)
        if master is None:
            return False
        done = _xinput("remove-master", master.name, display=display)
        if done.returncode != 0:
            # The 1.x device lookup xinput uses cannot see a master created
            # after it started; removing by id works through the same request.
            done = _xinput("remove-master", str(master.pointer_id), display=display)
        return _match(conn, name) is None
    finally:
        conn.close()
