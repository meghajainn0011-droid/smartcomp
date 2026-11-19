"""Panel tray app: status indicator for the perception daemon.

Two backends exist on Linux:
  * XEmbed via Gtk.StatusIcon -- needs a system-tray host (panel plugin) that
    owns the `_NET_SYSTEM_TRAY_S0` selection.
  * StatusNotifierItem via pystray/AppIndicator -- needs an SNI host plugin.
This box ships both libraries but Xfce 4.20 dropped the SNI panel plugin, so the
XEmbed path is the one that can actually show pixels. `--diagnose` reports which
host, if any, is present before you spend time wondering why nothing appeared.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys

from . import config

ICON_SIZE = 22


def _icon_pixbuf(scale: float = 1.0):
    import cairo
    import gi

    gi.require_version("GdkPixbuf", "2.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import GdkPixbuf, GLib

    size = int(ICON_SIZE * scale)
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, size, size)
    ctx = cairo.Context(surface)
    s = size / 22.0
    ctx.set_source_rgba(0.13, 0.55, 0.85, 1.0)
    ctx.rectangle(1 * s, 4 * s, 20 * s, 14 * s)
    ctx.fill_preserve()
    ctx.set_source_rgba(1, 1, 1, 0.95)
    ctx.set_line_width(1.6 * s)
    ctx.stroke()
    ctx.set_source_rgba(1, 1, 1, 1.0)
    ctx.arc(11 * s, 11 * s, 4.2 * s, 0, 6.2832)
    ctx.fill()
    ctx.set_source_rgba(0.13, 0.55, 0.85, 1.0)
    ctx.arc(11 * s, 11 * s, 1.9 * s, 0, 6.2832)
    ctx.fill()
    surface.flush()
    data = bytes(surface.get_data())
    stride = surface.get_stride()
    # cairo is premultiplied BGRA on little endian; the pixbuf wants RGBA.
    rgba = bytearray(size * size * 4)
    for i in range(size * size):
        b, g, r, a = data[i * 4 : i * 4 + 4]
        if a not in (0, 255):
            r = min(255, r * 255 // a)
            g = min(255, g * 255 // a)
            b = min(255, b * 255 // a)
        rgba[i * 4 : i * 4 + 4] = bytes((r, g, b, a))
    pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
        GLib.Bytes.new(bytes(rgba)),
        GdkPixbuf.Colorspace.RGB,
        True,
        8,
        size,
        size,
        size * 4,
    )
    if scale != 1.0:
        pixbuf = pixbuf.scale_simple(int(ICON_SIZE), int(ICON_SIZE), GdkPixbuf.InterpType.BILINEAR)
    return pixbuf


def diagnose() -> int:
    """Print whether any tray host is listening, without starting an icon."""
    from Xlib import display as xdisplay

    xd = xdisplay.Display()
    atom = xd.intern_atom("_NET_SYSTEM_TRAY_S0")
    owner = xd.get_selection_owner(atom)
    xembed = owner if getattr(owner, "id", 0) else None
    print(f"XEmbed tray host (_NET_SYSTEM_TRAY_S0): {xembed or 'NONE'}")
    sni = _sni_present()
    print(f"StatusNotifier watcher (DBus): {'present' if sni else 'NONE'}")
    panel = subprocess.run(
        ["xfconf-query", "-c", "xfce4-panel", "-p", "/panels/panel-1/plugin-ids"],
        capture_output=True,
        text=True,
    )
    if panel.returncode == 0:
        print("panel plugin-ids:", " ".join(panel.stdout.split()))
    if xembed:
        print("verdict: XEmbed path is available -> Gtk.StatusIcon will show")
        return 0
    if sni:
        print("verdict: SNI path only -> use pystray/AppIndicator")
        return 0
    print("verdict: NO tray host -> an icon cannot appear until a systray panel plugin is added")
    return 1


def _sni_present() -> bool:
    try:
        out = subprocess.run(
            [
                "dbus-send",
                "--session",
                "--dest=org.freedesktop.DBus",
                "--type=method_call",
                "--print-reply",
                "/org/freedesktop/DBus",
                "org.freedesktop.DBus.NameHasOwner",
                "string:org.kde.StatusNotifierWatcher",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return "boolean true" in out.stdout
    except Exception:
        return False


class Tray:
    def __init__(self) -> None:
        import gi

        gi.require_version("Gtk", "3.0")
        from gi.repository import Gtk

        self.Gtk = Gtk
        self.status_text = "perception: idle"
        self.icon = Gtk.StatusIcon.new_from_pixbuf(_icon_pixbuf())
        self.icon.set_tooltip_text("SmartComp perception")
        self.icon.set_visible(True)
        self.icon.connect("popup-menu", self._on_popup)
        self.icon.connect("activate", self._on_activate)

        menu = Gtk.Menu()
        self.status_item = Gtk.MenuItem(label=self.status_text)
        self.status_item.set_sensitive(False)
        menu.append(self.status_item)
        menu.append(Gtk.SeparatorMenuItem())

        full = Gtk.MenuItem(label="Force full refresh")
        full.connect("activate", self._force_full)
        menu.append(full)

        token = Gtk.MenuItem(label="Copy MCP token")
        token.connect("activate", self._copy_token)
        menu.append(token)

        quit_item = Gtk.MenuItem(label="Quit tray")
        quit_item.connect("activate", lambda *_: Gtk.main_quit())
        menu.append(quit_item)
        menu.show_all()
        self.menu = menu

        from gi.repository import GLib

        GLib.timeout_add_seconds(5, self._poll)
        self._poll()

    # -- menu actions -----------------------------------------------------
    def _force_full(self, *_):
        # The next get_changes call picks this up and ignores the tile diff, so
        # a client that thinks it lost sync can ask for a fresh screen.
        (config.state_dir() / "force_full").touch()
        self._notify("Full refresh requested")

    def _copy_token(self, *_):
        token = config.read_token()
        self.Gtk.Clipboard.get(self.Gtk.Gdk.SELECTION_CLIPBOARD).set_text(token, -1)

    def _on_popup(self, icon, button, time):
        self.menu.popup(None, None, self.Gtk.StatusIcon.position_menu, icon, button, time)

    def _on_activate(self, *_):
        self._notify(self.status_text)

    def _notify(self, message: str):
        try:
            subprocess.run(["notify-send", "-a", "SmartComp", message], timeout=5)
        except Exception:
            pass
        print(message, flush=True)

    def _poll(self) -> bool:
        try:
            with socket.create_connection((config.MCP_HOST, config.MCP_PORT), timeout=0.4):
                self.status_text = "perception: daemon up"
        except OSError:
            self.status_text = "perception: daemon down"
        self.status_item.set_label(self.status_text)
        self.icon.set_tooltip_text(f"SmartComp — {self.status_text}")
        return True

    def run(self) -> None:
        print("tray: started (Gtk.StatusIcon / XEmbed)")
        self._notify("SmartComp tray running")
        self.Gtk.main()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smartcomp-tray")
    parser.add_argument("--diagnose", action="store_true", help="report tray hosts and exit")
    args = parser.parse_args(argv)
    if args.diagnose:
        return diagnose()
    if not os.environ.get("DISPLAY"):
        print("DISPLAY is not set; a tray app needs an X session", file=sys.stderr)
        return 2
    Tray().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
