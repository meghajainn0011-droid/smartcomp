"""Popup detection: find dialogs/menus the screen is showing right now.

The agent must never miss a popup: a close-confirmation, a save prompt, a
permission dialog. Detection is cheap (window geometry + element store); the
expensive dual read (OCR + Mirage) only runs when a popup is actually found,
so watching stays fast and the LLM gets a small, first-class `popups` block.
"""

from __future__ import annotations

DIALOG_TITLE_WORDS = (
    "confirm", "save", "close", "warning", "question", "alert", "dialog",
    "error", "remove", "delete", "permission", "allow", "restart", "update",
    "sign in", "preferences", "settings", "properties",
)
BUTTON_WORDS = (
    "ok", "cancel", "close", "save", "don't save", "yes", "no", "allow",
    "deny", "accept", "reject", "continue", "restart", "remove", "delete",
    "close all tabs", "force quit", "discard",
)


def _center(box: list[int]) -> list[int]:
    return [box[0] + box[2] // 2, box[1] + box[3] // 2]


def _extra_x_windows() -> list[dict]:
    """Mapped windows X knows about but the WM's client list omits.

    Notification bubbles (xfce4-notifyd), menus, tooltips and override-
    redirect dialogs are NOT in _NET_CLIENT_LIST (what windows.list reads) --
    the gap that made a visible popup report as zero. xwininfo is cheap.
    """
    try:
        import os
        import subprocess

        out = subprocess.run(
            ["xwininfo", "-root", "-children"], env=dict(os.environ),
            capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return []
    found = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("0x"):
            continue
        wid = line.split(None, 1)[0]
        rest = line[len(wid):].strip()
        title = rest.split('"')[1] if '"' in rest else ""
        low = (title + " " + rest).lower()
        # Skip the 10x10 / 1x1 offscreen utility windows.
        if "+-100" in line or " 1x1+" in line or " 10x10+" in line:
            continue
        if any(k in low for k in ("notif", "menu", "dialog", "warning",
                                  "confirm", "popup", "alert", "tooltip")):
            found.append({"id": wid, "title": title, "app": "x-popup",
                          "source": "x-tree"})
    return found


def detect(wins, perception, fresh_lines: list[str] | None = None) -> list[dict]:
    """Return popup candidates: dialogs, X override-redirect popups, banners."""
    out: list[dict] = []
    # Only true dialog/popup windows: a confirmation title, or a small
    # transient window. A maximized app with "Settings" in its title is not
    # a popup; require small geometry for the generic case.
    for w in wins.list():
        title = (w.get("title") or "").lower()
        small = w.get("w", 9999) < 700 and 0 < w.get("h", 9999) < 500
        explicit = any(k in title for k in ("confirm", "save changes", "are you sure",
                                             "close all tabs", "question", "alert",
                                             "permission", "dialog"))
        if explicit or (small and w.get("w", 0) > 60):
            out.append({
                "source": "window", "id": w["id"], "app": w.get("comm", ""),
                "title": w.get("title", ""),
                "box": [w["x"], w["y"], w["w"], w["h"]],
            })
    # In-page warning banners (Firefox/Chrome show these INSIDE the page,
    # no separate X window): detect from the element store text, but ONLY
    # inside the active window -- the agent's own terminal may quote the
    # same words in its conversation and must not be mistaken for a popup.
    BANNER_WORDS = ("will not be functional", "in use by another", "cannot be",
                    "failed to", "warning:", "error:", "not responding",
                    "unable to", "something went wrong", "learn more")
    active = next((w for w in wins.list() if w.get("active")), None)
    nw, nh = perception.capturer.native_size
    bx = int(active["x"] / max(1, nw) * 1000) if active else 0
    by = int(active["y"] / max(1, nh) * 1000) if active else 0
    bw = int(active["w"] / max(1, nw) * 1000) if active else 1000
    bh = int(active["h"] / max(1, nh) * 1000) if active else 1000
    banners: list[dict] = []
    # The element store is a delta and holds stale text from OTHER windows
    # (e.g. the agent's own terminal quoting an error) -- filter it to the
    # active window. Fresh OCR lines already come from that window only.
    banner_texts = []
    for el in perception.elements.values():
        cx, cy = _center(el["b"])
        if bx <= cx <= bx + bw and by <= cy <= by + bh:
            banner_texts.append(el.get("x") or "")
    banner_texts += list(fresh_lines or [])
    for i, text in enumerate(banner_texts):
        text = (text or "").strip()
        if not text or not any(k in text.lower() for k in BANNER_WORDS):
            continue
        banners.append({"id": f"banner-{i}", "text": text[:120],
                        "center": [bx + bw // 2, by + 8]})
    if banners:
        out.append({"source": "banner", "count": len(banners),
                    "banners": banners[:6]})

    # Override-redirect popups invisible to the WM's client list.
    for xw in _extra_x_windows():
        out.append({"source": "x-tree", "id": xw["id"], "app": xw["app"],
                    "title": xw["title"], "box": [0, 0, 1920, 1080]})
    # Only buttons physically inside a detected popup's rect. This excludes
    # stale elements from old dialogs (a "Close all tabs?" from a closed
    # browser) and the main window's ordinary buttons.
    regions = [(f["box"][0], f["box"][1], f["box"][2], f["box"][3])
               for f in out if f["source"] in ("window", "x-tree")]
    native_w, native_h = perception.capturer.native_size
    buttons = []
    for el in perception.elements.values():
        text = (el.get("x") or "").strip()
        if not text:
            continue
        key = text.lower().strip(" .!?:…")
        cx, cy = _center(el["b"])
        px, py = cx / 1000 * native_w, cy / 1000 * native_h
        in_popup = any(rx <= px <= rx + rw and ry <= py <= ry + rh
                       for rx, ry, rw, rh in regions) if regions else False
        if in_popup and any(key == b or key.startswith(b + " ") for b in BUTTON_WORDS):
            buttons.append({"id": el["id"], "text": text[:60],
                            "center": [cx, cy]})
    if buttons:
        out.append({"source": "elements", "count": len(buttons),
                    "buttons": buttons[:8]})
    return out


def summarize(found: list[dict]) -> dict:
    """Compact first-class report of what popped up."""
    windows = [f for f in found if f["source"] in ("window", "x-tree")]
    buttons = []
    banners = []
    for f in found:
        if f["source"] == "elements":
            buttons.extend(f["buttons"])
        if f["source"] == "banner":
            banners.extend(f["banners"])
    return {
        "count": len(windows) or len(banners) or (1 if buttons else 0),
        "windows": windows[:6],
        "buttons": buttons[:8],
        "banners": banners[:6],
    }
