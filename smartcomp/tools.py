"""The MCP tool surface: perception, full computer control, and learned memory.

Every coordinate is permille of the 960x540 logical frame, identical to what
`get_changes` reports, so a model never converts units.

Every tool call is wrapped by `tool()` below, which times it, records it in the
session log (the raw material for `stats` and `distill`) and turns exceptions
into `{"ok": false, "error": ...}` instead of killing the transport.
"""

from __future__ import annotations

import functools
import os
import json
import time
from pathlib import Path

import numpy as np

from . import apps as apps_mod
from . import config
from . import lock as lock_mod
from .apps import APP_ALLOWLIST
from .display import Display
from .input import Input, InputError, MAX_SCROLL_TICKS, launch
from .memory import Memory
from .perception import Perception
def _lazy(module: str):
    """Import a heavy module only when its tool is first used (fast startup)."""
    import importlib

    return importlib.import_module("." + module, __package__)


def _vision():
    return _lazy("vision")


def _sessions():
    return _lazy("sessions")


def _media():
    return _lazy("media")


def _skillbook():
    return _lazy("skillbook")


# Heavy optional subsystems imported on demand, not at daemon startup:
# media (yt-dlp/ffmpeg), sessions (Xvfb), skillbook, vision (Mirage).
from . import popups as popups_mod  # noqa: E402  (cheap, used by every action report)
from .windows import Windows


def _px(perception: Perception, x: int, y: int) -> tuple[int, int]:
    w, h = perception.capturer.native_size
    return int(x / 1000 * w), int(y / 1000 * h)


def _center(box: list[int]) -> tuple[int, int]:
    x, y, w, h = box
    return x + w // 2, y + h // 2


# An action's report caps the elements it attaches for the same reason the
# delta caps deleted ids: a popup is 3-6 elements, a page repainting is 200,
# and only the first handful is actionable context.
MAX_REPORT_ADDS = 8


def _describe(el: dict) -> dict:
    """The action-addressable form of an element: id, kind, label, center.

    Coordinates are deliberately left out except the center -- the agent
    addresses elements by id or label, never by geometry, so the full box
    is tokens without a use here.
    """
    cx, cy = _center(el["b"])
    out = {"id": el["id"], "k": el["k"], "center": [cx, cy]}
    if el.get("x"):
        out["text"] = el["x"][:60]
    return out


def _compact_adds(items: list[dict], cap: int = MAX_REPORT_ADDS) -> tuple[list[dict], int]:
    """Trim delta adds for an action report; returns (entries, dropped)."""
    out = [_describe(el) for el in items[:cap]]
    return out, max(0, len(items) - cap)


def _dominant_colors(img: np.ndarray, limit: int = 6) -> list[dict]:
    """Quantized dominant colors of a BGR image, as hex plus pixel share.

    Colors are part of the screen's plain-language vocabulary: the agent
    should be able to say "the red button" without ever seeing pixels.
    Quantizing to 8 levels per channel (//32) merges antialiasing and
    gradient noise into the flat theme colors that actually dominate a UI.
    """
    h, w = img.shape[:2]
    sy, sx = max(1, h // 96), max(1, w // 96)
    px = img[::sy, ::sx, :3].reshape(-1, 3)
    q = (px // 32).astype(np.int32) * 32
    b, g, r = q[:, 0], q[:, 1], q[:, 2]
    packed = (r << 16) | (g << 8) | b
    uniq, counts = np.unique(packed, return_counts=True)
    total = int(counts.sum())
    top = np.argsort(counts)[::-1][:limit]
    out = []
    for i in top:
        p = int(uniq[i])
        out.append({
            "hex": f"#{(p >> 16) & 0xff:02x}{(p >> 8) & 0xff:02x}{p & 0xff:02x}",
            "share": round(int(counts[i]) / total, 2),
        })
    return out


def _blank_spot(wins: Windows, perception: Perception,
                grid: int = 9) -> list[int]:
    """A permille point inside the active window, far from all elements.

    Lets context_menu work with no coordinates from the agent: sample a grid
    in the active window, score each point by distance to the nearest element
    center, return the emptiest. Falls back to the window center if the store
    is empty (nothing to avoid).
    """
    win = next((w for w in wins.list() if w.get("active")), None)
    if not win:
        return [500, 500]
    nat_w, nat_h = perception.capturer.native_size
    bx = int(win["x"] / max(1, nat_w) * 1000)
    by = int(win["y"] / max(1, nat_h) * 1000)
    bw = int(win["w"] / max(1, nat_w) * 1000)
    bh = int(win["h"] / max(1, nat_h) * 1000)
    centers = [_center(el["b"]) for el in perception.elements.values()]
    if not centers:
        return [bx + bw // 2, by + bh // 2]
    best, best_score = None, -1
    for i in range(1, grid):
        for j in range(1, grid):
            p = [bx + bw * i // grid, by + bh * j // grid]
            score = min((p[0] - cx) ** 2 + (p[1] - cy) ** 2 for cx, cy in centers)
            if score > best_score:
                best, best_score = p, score
    return best or [bx + bw // 2, by + bh // 2]


MAX_DESCRIBE_CONTROLS = 12


def _click_box(perception: Perception, el: dict) -> list[int]:
    """The box to actually click -- snap tiny glyphs to a detector icon.

    OCR reads a close button's "x" as a 5x9 permille text box, but the real
    hit target is the larger icon box the detector drew around it (measured
    live: clicking the glyph's center did nothing, clicking the overlapping
    icon dismissed the bubble). Use the icon when one covers the glyph.
    """
    x, y, w, h = el["b"]
    if w >= 25 or h >= 25 or el.get("k") == "i":
        return el["b"]  # already a real-sized target
    cx, cy = x + w // 2, y + h // 2
    best = None
    for other in perception.elements.values():
        if other.get("k") != "i":
            continue
        ox, oy, ow, oh = other["b"]
        # icon near the glyph and at least as big
        if (abs(ox + ow // 2 - cx) <= max(20, ow)
                and abs(oy + oh // 2 - cy) <= max(20, oh)
                and ow >= w and oh >= h):
            if best is None or ow * oh < best[2] * best[3]:
                best = other["b"]
    return best or el["b"]


def _compose_description(active: dict | None, windows: list[dict],
                         elements: list[dict], colors: list[dict],
                         screen: list[int]) -> dict:
    """The screen's story as plain JSON: frontmost app, its controls, palette.

    This is what the agent reads instead of a screenshot. Controls are the
    elements inside the active window's box, ordered top-to-bottom (reading
    order) and capped -- a Firefox page carries a hundred elements, and a
    description that lists them all is a screenshot in JSON with extra
    steps; read_window digs deeper into one window.
    """
    out: dict = {
        "screen": screen,
        "colors": colors,
        "active": ({"app": active["comm"],
                    "title": (active.get("title") or "")[:60],
                    "id": active["id"]} if active else None),
    }
    others = [{"app": w["comm"], "title": (w.get("title") or "")[:40], "id": w["id"]}
              for w in windows if not w.get("active")]
    out["other_windows"] = others[:6]
    if len(others) > 6:
        out["other_windows_dropped"] = len(others) - 6
    if active:
        w, h = screen
        bx = int(active["x"] / max(1, w) * 1000)
        by = int(active["y"] / max(1, h) * 1000)
        bw = int(active["w"] / max(1, w) * 1000)
        bh = int(active["h"] / max(1, h) * 1000)

        def inside(el: dict) -> bool:
            cx, cy = _center(el["b"])
            return bx <= cx <= bx + bw and by <= cy <= by + bh

        controls = [el for el in elements if inside(el)]
        controls.sort(key=lambda e: (_center(e["b"])[1], _center(e["b"])[0]))
        out["controls"] = [_describe(el) for el in controls[:MAX_DESCRIBE_CONTROLS]]
        if len(controls) > MAX_DESCRIBE_CONTROLS:
            out["controls_dropped"] = len(controls) - MAX_DESCRIBE_CONTROLS
            out["hint"] = "read_window(active.id) for the full list of this window"
    return out


def _resolve_element(perception: Perception, element_id: str | None,
                     label: str | None) -> tuple[dict | None, dict | None]:
    """Address an element by id or by its visible text; (element, error).

    The agent speaks in labels ("click Paste") or in ids from a previous
    report, and this resolves the geometry -- that is the division of
    labor that keeps the agent out of pixels. A label that matches several
    elements is refused with the candidates: clicking the wrong "OK" is
    worse than asking. A label that matches nothing lists what IS visible
    so the label can be corrected in one retry instead of a full re-read.
    """
    if element_id:
        el = perception.elements.get(element_id)
        if el is None:
            return None, {
                "ok": False,
                "error": f"unknown element id {element_id}",
                "hint": ("ids come from get_changes or an action's changed "
                         "block; the screen may have moved on -- get_changes"),
            }
        return el, None
    needle = (label or "").strip().lower()
    if not needle:
        return None, {"ok": False, "error": "give element_id or label"}
    exact: list[dict] = []
    partial: list[dict] = []
    for el in perception.elements.values():
        text = (el.get("x") or "").strip()
        if not text:
            continue
        low = text.lower()
        if low == needle:
            exact.append(el)
        elif needle in low:
            partial.append(el)
    pool = exact or partial
    if len(pool) == 1:
        return pool[0], None
    if not pool:
        visible: list[str] = []
        for el in perception.elements.values():
            text = (el.get("x") or "").strip()
            if text and text[:48] not in visible:
                visible.append(text[:48])
            if len(visible) >= 8:
                break
        return None, {
            "ok": False,
            "error": f"no element labelled {label!r}",
            "visible_text": visible,
            "hint": ("labels are OCR text of kind t; icon-only buttons (kind i) "
                     "have no label -- address them by id from get_changes"),
        }
    return None, {
        "ok": False,
        "error": f"{len(pool)} elements match {label!r}",
        "matches": [_describe(el) for el in pool[:8]],
    }


# A report answers "what did MY action bring up": an element within this
# permille radius of the action point counts as its answer (a dialog
# opens centred on its parent window, a menu centred on the click).
REPORT_NEAR_RADIUS = 200
# And the answer can still be en route when the pass runs: a WM close is
# an IPC round trip before the prompt maps (measured: the report pass ran
# 55 ms after close on a fully idle screen, prompt painted after it).
REPORT_LATE_PAINT_S = 0.5


def _report_delta(perception: Perception, result: dict,
                  at: list[int] | None = None) -> dict:
    """Attach the screen's answer to an action's result.

    An input action used to come back bare: the agent had to make a second
    get_changes call to learn what its action did, and a popup that appeared
    meanwhile was invisible until it polled. One capture pass right after
    the action attaches what appeared as element ids -- a popup report is
    ids the agent can act on, not pixels. Cheap when nothing changed (the
    tile diff short-circuits OCR and detection); it costs a normal delta
    pass when the screen answered.

    `at` (the action's permille point) orders the reported elements by
    distance from the action: a click that opens a menu AND scrolls a
    terminal behind it reports the menu first -- the scrollback is noise
    around the thing the agent just did (measured live: the first 8 adds
    in OCR order were scrollback lines while the opened menu fell into
    the dropped count).
    """
    payload = perception.get_changes()
    if (at is not None and not payload.get("add") and not payload.get("del")
            and not payload.get("stats", {}).get("tiles")
            and not payload.get("full")):
        # The action point expects an answer yet the screen is stone idle:
        # the consequence may not have painted yet (a close's prompt maps
        # after an IPC round trip -- measured: the pass ran 55 ms after the
        # close, idle, and the prompt painted after it). Give the paint one
        # beat, then take one more pass; the retry below still covers the
        # mid-paint cases on top of this.
        time.sleep(REPORT_LATE_PAINT_S)
        payload = perception.get_changes()
    stats = payload.get("stats", {})
    moving = stats.get("tiles") and not payload.get("full")
    silent = not payload.get("add") and not payload.get("del")
    answered = True
    if at is not None and payload.get("add"):
        answered = any(abs(_center(el["b"])[0] - at[0]) <= REPORT_NEAR_RADIUS
                       and abs(_center(el["b"])[1] - at[1]) <= REPORT_NEAR_RADIUS
                       for el in payload["add"])
    if moving and (silent or not answered):
        # Two ways a pass can miss the action's answer, both measured live:
        # a popup paints its background before its text (tiles moved, zero
        # elements), or the answer painted after this pass while unrelated
        # windows repainted around it (Firefox noise beside a late save
        # dialog). One more pass sees the settled paint; capped at one so
        # an animating screen cannot spin.
        payload = perception.get_changes()
    delta: dict = {"seq": payload.get("seq"), "full": payload.get("full", False)}
    if payload.get("full"):
        # First pass since startup or reset: every element is "new". Listing
        # a page's whole inventory inside a click's result would be the
        # token flood the report exists to avoid.
        delta["note"] = ("full capture (first since startup/reset); call "
                         "get_changes once for the baseline instead")
    else:
        adds = payload.get("add", [])
        if at is not None and adds:
            def _dist(el: dict) -> float:
                cx, cy = _center(el["b"])
                return (cx - at[0]) ** 2 + (cy - at[1]) ** 2

            adds = sorted(adds, key=_dist)
        compact, dropped = _compact_adds(adds)
        if compact:
            delta["add"] = compact
        if dropped:
            delta["add_dropped"] = dropped
        if payload.get("del"):
            delta["del"] = payload["del"]
        if payload.get("del_dropped"):
            delta["del_dropped"] = payload["del_dropped"]
    delta["ms"] = payload.get("stats", {}).get("total")
    # A newly appeared window (OAuth / sign-in popup) has no reliable pixel
    # signature until it paints and may be missed by the element delta, so
    # report window identity from the WM client list explicitly. Never leave
    # an agent blind to a window its own click just opened.
    if _WINS is not None:
        try:
            appeared = popups_mod.new_windows(_WINS)
            if appeared:
                delta["windows"] = popups_mod.describe_new_windows(appeared)
        except Exception:
            pass  # window census must never break an action report
    result["changed"] = delta
    return result


def _clip_box(box: dict, bounds: dict) -> dict | None:
    """Intersect a grab rectangle with the screen; None when they do not meet.

    XGetImage (major opcode 73) answers BadMatch for any rectangle that
    leaves the root window, and a window's translated origin can be negative
    or run past the edge (frame offsets, a window dragged partly offscreen).
    Clamping is what keeps the window-scoped screenshot alive; the reviewed
    session lost it on Firefox to exactly this error.
    """
    left = max(box["left"], bounds["left"])
    top = max(box["top"], bounds["top"])
    right = min(box["left"] + box["width"], bounds["left"] + bounds["width"])
    bottom = min(box["top"] + box["height"], bounds["top"] + bounds["height"])
    if right - left <= 0 or bottom - top <= 0:
        return None
    return {"left": left, "top": top, "width": right - left, "height": bottom - top}


def _looks_degenerate(lines: list[dict], w: int, h: int) -> bool:
    """True when a window read came back with far less text than its size holds.

    Dark or heavily graphic UIs (modern browsers) can defeat tesseract almost
    completely -- a real read of Firefox returned "a x / - a x / fal / x] +"
    over a 1.2 M px window. Flagging it saves the caller from treating a
    near-empty read as the window's content, and points at the paths that do
    work there.
    """
    area = w * h
    if area <= 200_000:
        return not lines and area > 60_000
    chars = sum(len(line.get("x", "")) for line in lines)
    return len(lines) < 10 and chars < 80


_ACTIONS_NEEDING_SCREEN = frozenset({
    # perception: pixels must show the desktop, not a lock screen
    "get_changes", "find", "read_window", "read_region", "wait_for_change",
    "screen_state", "screenshot", "describe_screen", "explain_screen",
    "popups", "watch_popups",
    # input: XTEST events go to the lock screen's grab while locked
    "context_menu", "click_element", "drag_element", "draw_stroke", "scroll",
    "type_text", "key", "key_down", "key_up",
})
# Window management (list/focus/move/close/launch) and the clipboard are
# deliberately NOT gated: they work through a lock over EWMH/selections, and
# they cannot type into the lock screen.
#
# When the host is locked these tools return ok:false + locked:true instead of
# acting. Nothing is woken or unlocked -- a session display is the answer.


# Set by register() so _report_delta can include newly-appeared windows in
# every action's `changed` block (e.g. a Google sign-in popup after a click).
_WINS: Windows | None = None


def register(mcp, perception: Perception, inp: Input, wins: Windows, memory: Memory,
             display: Display | None = None) -> None:
    global _WINS
    _WINS = wins
    registry: dict[str, callable] = {}
    display = display or Display()

    def tool(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                blocked = (lock_mod.blocked_reason()
                           if fn.__name__ in _ACTIONS_NEEDING_SCREEN else None)
                if blocked:
                    # Read-only refusal: the host lock is the user's. A
                    # session display (own X server) is how work continues.
                    memory.log(fn.__name__, kwargs, False,
                               (time.perf_counter() - start) * 1000,
                               error="host display locked")
                    return {"ok": False, **blocked}
                result = fn(*args, **kwargs)
                ok = not (isinstance(result, dict) and (result.get("ok") is False
                                                        or result.get("error")))
            except Exception as exc:  # a tool must never take the server down
                memory.log(fn.__name__, kwargs, False,
                           (time.perf_counter() - start) * 1000,
                           error=f"{type(exc).__name__}: {exc}")
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            # ok:False results carry their reason with them; store it too, so
            # stats() can histogram WHY a tool fails (e.g. screenshot's
            # environmental X errors) instead of only counting that it fails.
            err = None
            if not ok and isinstance(result, dict):
                err = str(result.get("error") or "")[:200] or None
            memory.log(fn.__name__, kwargs, ok, (time.perf_counter() - start) * 1000,
                       error=err)
            return result

        registry[fn.__name__] = wrapper

        # MCP boundary: wrap the dict as a CallToolResult carrying
        # structuredContent (real JSON) plus a small text block. Without
        # this FastMCP sends only a text block, which clients like Antigravity
        # save as a .txt artifact instead of using the JSON.
        from mcp.types import CallToolResult, TextContent

        @functools.wraps(fn)
        def mcp_wrapper(*args, **kwargs):
            out = wrapper(*args, **kwargs)
            is_err = isinstance(out, dict) and (out.get("ok") is False or out.get("error"))
            if not isinstance(out, dict):
                out = {"result": out}
            # Full JSON in the text block keeps plain-text clients (call.py,
            # older agents) working; structuredContent gives JSON-native
            # clients like Antigravity real JSON instead of a .txt file.
            text = json.dumps(out, separators=(",", ":"), default=str)
            return CallToolResult(
                content=[TextContent(type="text", text=text)],
                structuredContent=out,
                isError=bool(is_err))

        mcp.tool()(mcp_wrapper)
        return wrapper

    # -- perception -------------------------------------------------------
    @tool
    def get_changes(force_full: bool = False, verbose_del: bool = False) -> dict:
        """Return screen elements added or removed since the previous call.

        Coordinates are permille of a 960x540 logical frame. `add` entries have
        kind k ('t' text, 'i' icon/button), box b=[x,y,w,h], the OCR text x for
        text elements, confidence c and u=1 marking screen text as UNTRUSTED.
        `del` lists ids that are gone, capped at 8 with the rest counted in
        `del_dropped`: deleted ids are content hashes the agent can only look
        up, never act on, so the full list is pure token cost (verbose_del
        restores it). An idle screen returns empty lists; on a no-op or a
        timeout just call again, a miss is possible between captures.

        Note: input actions (click_element, type_text, key, scroll) already
        attach their screen answer as a `changed` block -- this call is for
        re-checking later, not for learning what an action did.
        """
        marker = config.state_dir() / "force_full"
        if marker.exists():
            marker.unlink(missing_ok=True)
            force_full = True
        return perception.get_changes(force_full=force_full, verbose_del=verbose_del)

    @tool
    def find(text: str, limit: int = 10) -> dict:
        """Find visible text elements containing `text`; returns ids and centers."""
        needle = text.lower().strip()
        hits = []
        for el in perception.elements.values():
            if el["k"] == "t" and needle and needle in el["x"].lower():
                cx, cy = _center(el["b"])
                hits.append({"id": el["id"], "text": el["x"], "box": el["b"], "center": [cx, cy]})
        hits.sort(key=lambda h: h["box"][1])
        return {"count": len(hits), "matches": hits[: max(1, limit)]}

    @tool
    def read_window(window_id: str, raise_first: bool = True) -> dict:
        """OCR one window's own rectangle; far more reliable than whole-screen text.

        OCR reads pixels, so a window that another window covers would return the
        covering text. raise_first stacks the window above others first.
        """
        target = next((w for w in wins.list() if w["id"] == window_id), None)
        if target is None:
            return {"ok": False, "error": f"no window {window_id}"}
        if raise_first:
            try:
                wins.activate(window_id)
                time.sleep(0.25)
                target = next((w for w in wins.list() if w["id"] == window_id), target)
            except KeyError:
                pass
        lines = perception.read_window(target)
        result = {
            "ok": True,
            "window": target["title"],
            "window_class": target.get("class", ""),
            "pid": target.get("pid"),
            "lines": lines,
            "text": "\n".join(line["x"] for line in lines),
            "ms": perception.last_stats.get("window_ocr"),
        }
        if _looks_degenerate(lines, target["w"], target["h"]):
            result["low_yield"] = True
            result["hint"] = (
                "far too little OCR text for a window this size: dark/graphic "
                "UIs (browsers) defeat tesseract here. Do not treat this as "
                "the window's content -- use screenshot for the visual, "
                "keyboard-driven reading (key + clipboard_get), or get_changes."
            )
        return result

    @tool
    def read_region(x: int, y: int, width: int, height: int) -> dict:
        """OCR a screen rectangle given in permille coordinates."""
        native_w, native_h = perception.capturer.native_size
        px = _px(perception, x, y)
        lines = perception.read_region(
            px[0], px[1], int(width / 1000 * native_w), int(height / 1000 * native_h)
        )
        return {"ok": True, "lines": lines, "text": "\n".join(l["x"] for l in lines)}

    @tool
    def wait_for_change(timeout_s: float = 5.0) -> dict:
        """Block until the screen changes (or the timeout), returning the delta.

        Polls with an adaptive interval (0.1 s growing to 0.5 s) so waiting
        for a slow page load does not turn into dozens of full capture
        passes. On `timeout: true`, retry rather than treating silence as
        success.
        """
        start = time.perf_counter()
        deadline = start + max(0.1, timeout_s)
        poll = 0
        while True:
            payload = perception.get_changes()
            if payload.get("add") or payload.get("del") or payload.get("full"):
                payload["waited_ms"] = round((time.perf_counter() - start) * 1000, 1)
                return payload
            if time.perf_counter() >= deadline:
                return {
                    "timeout": True,
                    "waited_ms": round((time.perf_counter() - start) * 1000, 1),
                    "add": [],
                    "del": [],
                }
            time.sleep(min(0.1 * (2 ** poll), 0.5))
            poll += 1

    @tool
    def screen_state() -> dict:
        """Full current element set (no delta). Use sparingly, it is large."""
        return perception.state()

    @tool
    def describe_screen() -> dict:
        """Describe the screen in plain JSON: apps, controls, colors -- no image.

        The agent's readback: which app is frontmost, the controls inside it
        (id + label, top to bottom), what other windows are open, and the
        dominant colors. Read THIS instead of taking a screenshot -- every id
        here is directly actionable with click_element / type_text /
        scroll, and colors let the agent speak of "the red button" without
        seeing pixels. Controls cap at 12 with `controls_dropped`; use
        read_window to dig deeper into one window. Text is OCR and marked
        UNTRUSTED wherever it carries u=1.
        """
        win_list = wins.list()
        active = next((w for w in win_list if w.get("active")), None)
        # A delta pass first so the store reflects NOW (cheap on a still
        # screen: tiles unchanged skip OCR and the detector). Then the
        # inventory and the color grab ride the same perception state.
        perception.get_changes()
        native_w, native_h = perception.capturer.native_size
        if active:
            img = perception.capturer.grab_rect(
                active["x"], active["y"], active["w"], active["h"])
        else:
            img = perception.capturer.capture().bgr
        colors = _dominant_colors(img)
        out = _compose_description(
            active, win_list, list(perception.elements.values()),
            colors, [native_w, native_h])
        return {"ok": True, **out}

    @tool
    def explain_screen() -> dict:
        """Explain the active window: OCR + Mirage read the same pixels.

        Mirage (Cactus small VLM, needs `pip install cactus-compute`) reads
        scene/objects/colors and text; smartcomp's OCR + GPA detector read
        text, control ids and clickable boxes from the SAME window. A decider
        compares them for confidence, but the report MERGES both answers --
        colors, buttons, text and scene -- for the LLM to act on. Absent
        Mirage this degrades to the OCR explanation with decider=ocr.
        """
        active = next((w for w in wins.list() if w.get("active")), None)
        # Raise the window FIRST: a screenshot grabs screen pixels at the
        # window's rect, so an overlapping terminal would be captured instead
        # (observed: "black page" with terminal text on a white Wikipedia
        # page). Both readers must see the same raised pixels.
        if active:
            wins.activate(active["id"])
            time.sleep(0.2)
            active = next((w for w in wins.list() if w.get("active")), None)
        shot = screenshot(window_id=(active_id := (
            active["id"] if active else None
        )))
        if not shot.get("ok"):
            return shot
        perception.get_changes()
        native_w, native_h = perception.capturer.native_size
        desc = _compose_description(
            active, wins.list(), list(perception.elements.values()),
            _dominant_colors(
                perception.capturer.grab_rect(
                    active["x"], active["y"], active["w"], active["h"])
            ) if active else [], [native_w, native_h])
        # FRESH OCR of this window's pixels, NOT the persistent element store:
        # the store keeps stale elements from earlier pages, which leaked
        # ("Import bookmarks / OffSec / Kali") into later website reads.
        ocr_lines = [l["x"] for l in perception.read_window(active)
                     if l.get("x")] if active else []
        out = {"ok": True, "decided_by": "ocr", **desc}
        # Mirage reads from the saved window PNG (its documented input form).
        mir =  _vision().describe(shot["path"])
        out["mirage"] = mir
        # MERGE both readers of the same screen into ONE understanding for
        # the LLM: OCR text + Mirage text (deduplicated), Mirage's scene /
        # relations, smartcomp's colors, controls and clickable ids. Nothing
        # is thrown away. The decider only annotates confidence.
        out["decider"] =  _vision().decide(" ".join(ocr_lines),
                                        (mir.get("lines") or [])) if mir.get("ok") else {
            "decided": "ocr", "why": mir.get("error", "unknown")}
        mir_lines = mir.get("lines") or [] if mir.get("ok") else []
        merged, seen = [], set()
        for line in ocr_lines + mir_lines:
            key = " ".join((line or "").lower().split())
            if key and key not in seen:
                seen.add(key)
                merged.append(line)
        out["mode"] = "merged"
        out["lines"] = merged[:24]
        out["ocr_lines"] = ocr_lines[:12]
        out["mirage_lines"] = mir_lines[:12]
        if mir.get("ok"):
            out["scene"] = mir.get("scene")
            out["relations"] = mir.get("relations")
            out["reasoning"] = mir.get("reasoning")
        return out

    @tool
    def screenshot(path: str | None = None, window_id: str | None = None) -> dict:
        """Save a PNG of the screen, or of one window, into this session's folder.

        Passing `window_id` grabs only that window's rectangle: no unrelated
        windows in the image, a fraction of the pixels, and nothing private
        that happens to be behind it. The rectangle is clamped to the screen
        (an unclamped rect makes XGetImage fail with BadMatch, which is what
        killed window screenshots on Firefox) and a grab that still fails
        falls back to a full-screen grab cropped in software. Session
        screenshots are deleted at session end.
        """
        import mss.tools

        box = None
        if window_id:
            target = next((w for w in wins.list() if w["id"] == window_id), None)
            if target is None:
                return {"ok": False, "error": f"no window {window_id}"}
            box = {"left": target["x"], "top": target["y"],
                   "width": target["w"], "height": target["h"]}
        if path:
            out = Path(path)
        else:
            name = f"win-{int(time.time() * 1000)}.png" if window_id else \
                f"shot-{int(time.time() * 1000)}.png"
            out = memory.session_dir() / name
        with mss.mss() as sct:
            bounds = sct.monitors[0]  # the virtual screen: root-window bounds
            if box is not None:
                box = _clip_box(box, bounds)
                if box is None:
                    return {"ok": False, "window": target["title"],
                            "window_class": target.get("class", ""),
                            "error": "window rectangle is entirely offscreen"}
            try:
                shot = sct.grab(box or sct.monitors[1])
                mss.tools.to_png(shot.rgb, shot.size, output=str(out))
                size = [shot.width, shot.height]
            except Exception:
                # BadMatch and friends: grab the whole root and crop after the
                # fact, which cannot fail on coordinates. Only the window path
                # can fall back; a failing full-screen grab is a real failure.
                if box is None:
                    raise
                full = sct.grab(bounds)
                img = np.frombuffer(full.rgb, np.uint8).reshape(
                    full.height, full.width, 3)
                crop = np.ascontiguousarray(
                    img[box["top"]:box["top"] + box["height"],
                        box["left"]:box["left"] + box["width"]])
                mss.tools.to_png(crop.tobytes(),
                                 (crop.shape[1], crop.shape[0]), output=str(out))
                size = [crop.shape[1], crop.shape[0]]
        memory.count_screenshot()
        result = {"ok": True, "path": str(out), "size": size}
        if window_id:
            result["window"] = target["title"]
            result["window_class"] = target.get("class", "")
        return result

    @tool
    def health() -> dict:
        """Report OCR engine, detector status, screen size, session and timings."""
        return {
            "ok": True,
            "ocr": perception.ocr.engine,
            "detector": "ready" if perception.detector.ready else perception.detector.error,
            "screen": list(perception.capturer.native_size),
            "session": memory.session,
            "display": display.status(),
            "locked": lock_mod.lock_state(),
            "stats": perception.last_stats,
        }

    # -- lock state (read-only) -------------------------------------------
    @tool
    def lock_state() -> dict:
        """Report whether the host screen is locked or blanked. Read-only.

        SmartComp never unlocks the screen, wakes a blanked monitor or holds
        a screensaver inhibit, so this is the whole "lock handling" surface.
        When the host display is locked, screen-touching tools return
        ok:false + locked:true and leave the lock screen alone; work that
        must continue runs on a session display (session_create), whose own
        X server is unaffected by the host lock.
        """
        disp = display.status()
        return {
            "locked": lock_mod.lock_state(refresh=True),
            "blank": disp.get("display_blank"),
            "display": disp.get("display"),
            "isolated": lock_mod.host_display_isolated(),
            "would_block": lock_mod.blocked_reason() is not None,
        }

    # -- pointer ----------------------------------------------------------
    @tool
    def popups(report: bool = True) -> dict:
        """List every popup/dialog currently on screen, as ids to act on.

        Watches the window list (small dialog windows), the X tree
        (override-redirect notifications), AND a FRESH OCR of the active
        window for in-page warning banners. When a popup is found, both
        readers describe it -- smartcomp's OCR and Mirage -- and the decider
        merges the result, so the agent gets one first-class `popups` block.
        """
        perception.get_changes()
        # Snapshot windows BEFORE raising: activate() can re-stack/replace the
        # client-list entries, so detecting after the raise misses a window
        # the action report already saw (the sign-in popup case).
        found = popups_mod.detect(wins, perception, fresh_lines=[])
        active = next((w for w in wins.list() if w.get("active")), None)
        if active:
            wins.activate(active["id"])
            time.sleep(0.2)
            active = next((w for w in wins.list() if w.get("active")), None)
        if active:
            shot = screenshot(window_id=active["id"] if active else None)
            ocr = [l["x"] for l in perception.read_window(active)
                   if l.get("x")] if active else []
            mir =  _vision().describe(shot["path"]) if shot.get("ok") else {"ok": False}
            mir_lines = (mir.get("lines") or []) if mir.get("ok") else []
            # Feed the fresh text into banner detection without re-consuming
            # the window snapshot (would drop the new-window event).
            banners = popups_mod.detect_banners(ocr + mir_lines, wins, perception)
            found.extend(banners)
            out = popups_mod.summarize(found)
            if found and report:
                out["ocr_lines"] = ocr[:12]
                if mir.get("ok"):
                    out["mirage_lines"] = mir_lines[:12]
                    out["decider"] =  _vision().decide(" ".join(ocr), mir_lines)
                    out["scene"] = mir.get("scene")
        else:
            out = popups_mod.summarize(found)
        return {"ok": True, "popups": out}

    @tool
    def watch_popups(timeout_s: float = 5.0) -> dict:
        """Block until a popup appears (or timeout), then report it.

        Polls the element/window state for up to timeout_s; the instant a
        dialog or menu shows, runs the dual read (OCR + Mirage + decider)
        and returns the same first-class `popups` block. Use after an action
        that might open a confirmation you must not miss.
        """
        deadline = time.time() + max(0.0, timeout_s)
        while True:
            perception.get_changes()
            found = popups_mod.detect(wins, perception)
            if found:
                return popups()
            if time.time() >= deadline:
                return {"ok": True, "popups": popups_mod.summarize([]),
                        "waited_s": timeout_s}
            time.sleep(min(0.3, max(0.0, deadline - time.time())))

    @tool
    def context_menu(element_id: str | None = None, label: str | None = None,
                     report: bool = True) -> dict:
        """Right-click an element, or a blank spot smartcomp finds itself.

        The agent supplies NO coordinates: give an element id or label to
        right-click it, or give nothing and smartcomp picks an empty spot in
        the active window (furthest from every known element center) -- so
        the desktop's / a canvas's context menu opens without the agent ever
        seeing pixels. The menu arrives in `changed` as element ids.
        """
        at = None
        if element_id or label:
            el, err = _resolve_element(perception, element_id, label)
            if err:
                return err
            at = list(_center(el["b"]))
        else:
            # Blank spot: the point in the active window maximally far from
            # every element center. Computed here, never by the LLM.
            at = _blank_spot(wins, perception)
        inp.click(*_px(perception, at[0], at[1]), button="right")
        result = {"ok": True, "right_clicked": at, "button": "right"}
        return _report_delta(perception, result, at=at) if report else result

    @tool
    def click_element(element_id: str | None = None, label: str | None = None,
                      button: str = "left", count: int = 1,
                      report: bool = True) -> dict:
        """Click an element by id, or by its visible label text.

        This is the normal way to act: the agent says WHAT ("click Accept"),
        smartcomp computes where. `label` matches OCR text case-insensitively;
        one match clicks, several return the candidates instead of guessing,
        none returns what is visible so the label can be corrected. Element
        ids come from get_changes or from any action's `changed` block. The
        result carries `changed` -- popups and menus the click opened arrive
        as element ids in the same call, no polling.
        """
        el, err = _resolve_element(perception, element_id, label)
        if err:
            return err
        click_b = _click_box(perception, el)
        cx, cy = _center(click_b)
        inp.click(*_px(perception, cx, cy), button=button, count=count)
        result = {"ok": True, "clicked": _describe(el),
                  "hit_box": click_b, "button": button, "count": count}
        return _report_delta(perception, result, at=[cx, cy]) if report else result

    @tool
    def draw_stroke(element_id: str, shape: str = "circle", scale: float = 0.6,
                    duration: float = 1.2) -> dict:
        """Draw a stroke inside an element: canvas, signature pad, slider.

        The agent names the element and the shape; smartcomp computes the path
        inside that element's box and drags the pointer along it -- nothing
        here is in pixels. Shapes: circle, zigzag, spiral, line.
        """
        el, err = _resolve_element(perception, element_id, None)
        if err:
            return err
        box = el["b"]
        x0, y0, w, h = box[0], box[1], box[2], box[3]
        cx, cy = x0 + w / 2, y0 + h / 2
        shape = (shape or "circle").lower()
        pts: list[tuple[float, float]] = []
        if shape == "circle":
            import math

            r = max(2.0, min(w, h) / 2 * max(0.05, min(1.0, scale)))
            steps = 48
            for i in range(steps + 1):
                a = 2 * math.pi * i / steps - math.pi / 2
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        elif shape == "spiral":
            import math

            r_max = max(2.0, min(w, h) / 2 * max(0.05, min(1.0, scale)))
            steps = 96
            for i in range(steps + 1):
                a = 4 * math.pi * i / steps
                r = r_max * i / steps
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        elif shape == "zigzag":
            steps = 12
            span = w * max(0.05, min(1.0, scale))
            left = cx - span / 2
            amp = h * max(0.05, min(1.0, scale)) / 2
            for i in range(steps + 1):
                pts.append((left + span * i / steps, cy + (amp if i % 2 else -amp)))
        else:  # line
            span = w * max(0.05, min(1.0, scale))
            pts = [(cx - span / 2, cy), (cx + span / 2, cy)]
        pixels = [_px(perception, px, py) for px, py in pts]
        inp.drag(pixels, duration=duration)
        result = {"ok": True, "shape": shape, "element": _describe(el),
                  "points": len(pixels)}
        return _report_delta(perception, result, at=list(_center(box)))

    @tool
    def drag_element(start_id: str, end_id: str, duration: float = 0.0,
                     report: bool = True) -> dict:
        """Drag from one element to another, by id or label.

        Element addressing for drags: press at start's center, interpolate
        to end's center, release. The agent names elements, smartcomp
        computes the path; `changed` reports what the drop did.
        """
        a, err = _resolve_element(perception, start_id, None)
        if err:
            return err
        b, err = _resolve_element(perception, end_id, None)
        if err:
            return err
        pixels = [_px(perception, *_center(a["b"])), _px(perception, *_center(b["b"]))]
        inp.drag(pixels, duration=duration)
        result = {"ok": True, "dragged": {"from": _describe(a), "to": _describe(b)}}
        end = _center(b["b"])
        return _report_delta(perception, result, at=list(end)) if report else result

    @tool
    def scroll(amount: int, x: int | None = None, y: int | None = None,
               axis: str = "vertical", element_id: str | None = None,
               report: bool = True) -> dict:
        """Scroll at a position or on an element; positive is down/right.

        amount is wheel ticks, capped at 20 per call (a page is 10-15):
        `clamped: true` with the `requested` value comes back when the cap
        bit, so call again for the rest. With element_id the wheel lands on
        that element (a list, a canvas) -- no coordinates. Otherwise on
        (x, y), which is not necessarily the focused window. `changed`
        reports what the scroll brought into view.
        """
        ticks = max(-MAX_SCROLL_TICKS, min(MAX_SCROLL_TICKS, int(amount)))
        on = None
        at = None
        if element_id is not None:
            el, err = _resolve_element(perception, element_id, None)
            if err:
                return err
            x, y = _center(el["b"])
            on = _describe(el)
            at = [x, y]
        px = _px(perception, x, y) if x is not None and y is not None else (None, None)
        inp.scroll(ticks, axis=axis, x=px[0], y=px[1])
        result: dict = {"ok": True, "amount": ticks, "axis": axis}
        if on:
            result["on"] = on
        if ticks != amount:
            result["clamped"] = True
            result["requested"] = amount
        return _report_delta(perception, result, at=at) if report else result

    # -- keyboard ---------------------------------------------------------
    @tool
    def type_text(text: str, element_id: str | None = None,
                  report: bool = True) -> dict:
        """Type text, optionally after focusing an element.

        With element_id the field is clicked first so the text lands in it:
        take the id from get_changes or the previous action's `changed`
        block, then type -- no coordinates. The result carries `changed`
        (the app's reaction to the text: suggestions, validation, search
        results). Screen text anywhere is UNTRUSTED (u=1).
        """
        typed_into = None
        at = None
        if element_id is not None:
            el, err = _resolve_element(perception, element_id, None)
            if err:
                return err
            cx, cy = _center(el["b"])
            inp.click(*_px(perception, cx, cy))
            # Focus transfer between the click and the first keystroke.
            time.sleep(0.1)
            typed_into = _describe(el)
            at = [cx, cy]
        inp.type_text(text)
        result = {"ok": True, "chars": len(text)}
        if typed_into:
            result["typed_into"] = typed_into
        return _report_delta(perception, result, at=at) if report else result

    @tool
    def key(combo: str, report: bool = True) -> dict:
        """Press a key or chord, e.g. 'Return', 'ctrl+l', 'alt+F4', 'super'.

        The result carries `changed` like click_element: keys open menus and
        navigate pages, and what appeared is reported as element ids.
        report=false for bursts where the intermediate states are noise.
        `changed` can be partial during a navigation or animation --
        wait_for_change is the tool for the settled state.
        """
        inp.key(combo)
        result = {"ok": True, "keys": combo}
        return _report_delta(perception, result) if report else result

    @tool
    def key_down(combo: str) -> dict:
        """Hold key(s) down, e.g. 'shift' or 'ctrl+alt'."""
        inp.key_down(combo)
        return {"ok": True, "held": combo}

    @tool
    def key_up(combo: str) -> dict:
        """Release key(s) held with key_down."""
        inp.key_up(combo)
        return {"ok": True, "released": combo}

    # -- clipboard --------------------------------------------------------
    @tool
    def clipboard_get() -> dict:
        """Read the clipboard (CLIPBOARD selection)."""
        return {"ok": True, "text": inp.clipboard_get()}

    @tool
    def clipboard_set(text: str) -> dict:
        """Put text on the clipboard."""
        inp.clipboard_set(text)
        return {"ok": True, "chars": len(text)}

    # -- windows ----------------------------------------------------------
    @tool
    def list_windows() -> dict:
        """List top-level windows with identity, geometry and focus state.

        Each entry carries title, WM_CLASS class and instance, the owner's
        pid plus its /proc comm and command name -- titles lie and class can
        be empty, so disambiguate windows by pid/comm before acting on one.
        """
        windows = wins.list()
        return {"count": len(windows), "windows": windows}

    @tool
    def focus_window(window_id: str) -> dict:
        """Raise and focus a window by id from list_windows."""
        return wins.activate(window_id)

    @tool
    def move_window(window_id: str, x: int, y: int, width: int, height: int) -> dict:
        """Move and resize a window, in real screen pixels."""
        return wins.move_resize(window_id, x, y, width, height)

    @tool
    def window_state(window_id: str, state: str) -> dict:
        """Set a window's state: maximize|fullscreen|normal."""
        return wins.set_state(window_id, state)

    @tool
    def close_window(window_id: str, report: bool = True) -> dict:
        """Ask a window to close (the app may prompt to save).

        The prompt IS the answer to this action, so it arrives in the same
        call: the result carries `changed` with the save dialog's elements
        ("Don't Save" as an id to click). Without the report the agent had
        to poll wait_for_change for a dialog it could not have predicted,
        paying a separate round trip and swallowing the whole desktop's
        delta (measured: a 7 kB poll vs a ~1 kB in-call report).
        """
        # The dialog arrives in this same call; ordering it by the closing
        # window's center puts it ahead of unrelated repaints. Geometry is
        # read BEFORE the close: afterwards the window is usually already
        # gone from the client list. wins.list() is native pixels; the
        # report's `at` is permille like every element center.
        win = next((w for w in wins.list() if w["id"] == window_id), None)
        result = wins.close(window_id)
        if isinstance(result, dict) and result.get("ok") and report:
            at = None
            if win:
                nat_w, nat_h = perception.capturer.native_size
                at = [int((win["x"] + win["w"] / 2) / max(1, nat_w) * 1000),
                      int((win["y"] + win["h"] / 2) / max(1, nat_h) * 1000)]
            result = _report_delta(perception, result, at=at)
        return result

    # -- apps -------------------------------------------------------------
    @tool
    def launch_app(name: str, args: list[str] | None = None,
                   force: bool = False, profile: str = "") -> dict:
        """Open an allowlisted app, or focus the one that is already open.

        brave, chromium, firefox, terminal, text-editor, file-manager,
        image-viewer. An app with a live window is RESUMED (focused, no new
        process) -- two calls used to mean two Firefoxes; pass force=true to
        deliberately start a second instance. A fresh launch runs as its own
        transient user service, so daemon restarts do not kill it (a child
        app used to die with every restart's cgroup sweep). app_status
        shows what is open.

        profile="real" copies the user's own browser profile (cookies,
        logins, extensions) into this session and launches on the copy, so
        a nested session browses as the signed-in user without touching the
        profile the host browser is using.
        """
        binary = APP_ALLOWLIST.get(name.lower())
        if binary is None:
            return {
                "ok": False,
                "error": f"not allowlisted: {name}",
                "allowed": sorted(APP_ALLOWLIST),
            }
        from . import ownership as own

        open_windows = wins.list()
        display = os.environ.get("DISPLAY")
        procs = apps_mod.processes(binary, display=display)
        target = apps_mod.resume_target(open_windows, binary)
        holder = own.owner_of(name.lower())
        # If another smartcomp/session owns this app, NEVER resume its window:
        # start our own instance with a private profile instead.
        taken = holder is not None
        if target is not None and not force and not taken:
            own.claim(name.lower(), target["id"], target.get("pid"))
            wins.activate(target["id"])
            return {
                "ok": True,
                "resumed": True,
                "already_running": True,
                "window": target["id"],
                "title": target.get("title", ""),
                "processes": len(procs),
                "windows": sum(1 for w in open_windows
                               if apps_mod.window_matches(w, binary)),
                "note": "focused the open instance instead of launching a duplicate",
            }
        if procs and target is None and not force:
            # Running but windowless (a window was closed, the process stayed):
            # launching again would stack a second instance on the first.
            return {
                "ok": False,
                "running_windowless": True,
                "processes": len(procs),
                "pids": [p["pid"] for p in procs],
                "error": "app is running but has no window; force=true starts "
                         "another instance",
            }
        launch_args = list(args or [])
        if taken or force:
            launch_args += apps_mod.profile_args(binary, name.lower())
        if profile == "real":
            base = Path(os.environ.get("SMARTCOMP_STATE_DIR")
                        or Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "smartcomp-profiles"
                        / name.lower())
            launch_args += apps_mod.clone_profile(binary, base / "profiles")
        pid = launch(binary, launch_args)
        own.claim(name.lower(), None, pid if pid else None)
        memory.remember(f"app:{name.lower()}", {"binary": binary, "pid": pid})
        result: dict = {"ok": True, "pid": pid, "binary": binary,
                        "resumed": False, "private_profile": bool(taken or force)}
        if taken:
            result["note"] = (f"app was in use by smartcomp '{holder['owner']}' "
                              "-- launched a separate instance with its own profile")
        if pid == 0:
            result["pid_note"] = (
                "started as a transient user service; its pid was not readable "
                "in time -- app_status reports the real process count"
            )
        # Report the new window in THIS call (the Google-sign-in case: the
        # agent must not be blind to a window its own click launched). Wait
        # briefly for the window to map, then attach a `changed` block with
        # the new window's id/title so the agent can act on it immediately.
        for _ in range(12):
            time.sleep(0.25)
            win = apps_mod.resume_target(wins.list(), binary)
            if win:
                result["window"] = win["id"]
                result["title"] = win.get("title", "")
                break
        return _report_delta(perception, result)

    # -- isolated agent sessions -------------------------------------------
    @tool
    def session_create(name: str = "agent", visible: bool = False) -> dict:
        """Create a private desktop with its own cursor and daemon.

        Each session gets its own display (XTEST cursor), MCP port, token,
        memory and skills, so two agents never fight over one pointer. The
        display is headless by default and keeps working while the host is
        locked/asleep. visible=true makes it an Xephyr window on the host
        screen instead: the session's cursor and apps are then watchable by
        a human, and nothing it does can reach the host's windows. Connect
        with session_endpoint(name).
        """
        return _sessions().manager().create(name, visible=visible)

    @tool
    def session_list() -> dict:
        """List isolated sessions with display, port and health."""
        return {"ok": True, "sessions": _sessions().manager().list()}

    @tool
    def session_close(name: str) -> dict:
        """Stop a session's daemon, WM and virtual display."""
        return _sessions().manager().close(name)

    @tool
    def session_endpoint(name: str) -> dict:
        """MCP connect config (url + bearer token) for an isolated session."""
        return _sessions().manager().endpoint(name)

    @tool
    def session_align() -> dict:
        """Align memory across sessions: merge every session's skills and
        observations into the shared pool, then push the pool back to all."""
        return _sessions().manager().align()

    @tool
    def session_train() -> dict:
        """Training loop: distil each session's successful actions into a
        skill, feed the shared pool, then align so every session inherits it."""
        return _sessions().manager().train()

    # -- video/voice understanding ----------------------------------------
    @tool
    def media_understand(source: str, model: str = "tiny", max_frames: int = 8,
                         duration: int = 180, captions: bool = True,
                         voice: bool = True) -> dict:
        """Understand a video URL or file -> structured JSON (no images).

        Captions first (zero RAM); else Whisper int8 on 16 kHz audio in 30 s
        chunks (tiny ~40 MB / ~300 MB RAM, base ~75 MB); frames sampled every
        8 s with OCR for burned-in text (reels/songs). Returns transcript,
        frame text, hashtags and URLs. Low RAM: one model at a time, frames
        released per loop, temp files cleaned. Good for research on YouTube
        videos and social-media reels/songs/music.
        """
        return _media().understand(source, model=model, max_frames=max_frames,
                                    duration=duration, captions=captions,
                                    voice=voice)

    @tool
    def media_listen(seconds: float = 8.0, model: str = "tiny") -> dict:
        """Listen to the system audio (what the browser is playing) and
        transcribe with Whisper. Records the PulseAudio monitor of the default
        output, so a playing YouTube video / reel is heard as the user hears
        it -- no captions, no download.
        """
        return _media().listen(seconds=seconds, model=model)

    @tool
    def media_watch(seconds: float = 6.0, frames: int = 4) -> dict:
        """Watch the current screen while a video plays: sample real frames
        and read them with BOTH OCR and Mirage as JSON. Combined with
        media_listen(), this understands a playing reel's graphics AND voice.
        """
        return _media().watch_screen(seconds=seconds, frames=frames)

    @tool
    def media_audio_check() -> dict:
        """Diagnose why a playing video has no sound: system mute, browser
        audio stream present/muted (autoplay tab mute), or no monitor source.
        """
        return _media().audio_check()

    @tool
    def media_capabilities() -> dict:
        """Which media engines are available (yt-dlp/ffmpeg/whisper/audio)."""
        return _media().capabilities()

    # -- pre-made + self-building skills ----------------------------------
    @tool
    def skills_install() -> dict:
        """Install the built-in skills (video, audio, tabs, apps, popups,
        isolated agents). Idempotent -- safe to call every startup.
        """
        return _skillbook().install_premade(memory)

    @tool
    def skill_create(name: str, steps: list[dict], description: str = "") -> dict:
        """Create a new skill from steps that worked (self-building). The
        skill can be replayed with run_skill and improves with each use.
        """
        return _skillbook().auto_skill(memory, name, steps, description)

    @tool
    def skill_learn_recent(limit: int = 12) -> dict:
        """Build a skill automatically from the most recent successful
        actions -- the daemon teaches itself from its own work.
        """
        return _skillbook().learn_from_recent(memory, limit=limit)

    @tool
    def app_owners() -> dict:
        """Shared knowledge: which apps are in use by which smartcomp/session.
        Use before resuming an app so you never hijack another agent's window.
        """
        from . import ownership as own

        return own.status()

    @tool
    def app_release(name: str) -> dict:
        """Release this smartcomp's claim on an app (e.g. after closing it)."""
        from . import ownership as own

        return own.release(name)

    @tool
    def app_status() -> dict:
        """Census of the allowlisted apps: live instances, open windows, focus.

        Answers "did I already open this?" and "how many windows does it
        have" without launching anything. Firefox also reports per-window
        tab counts read from its own session store (tabs: null when
        unavailable, e.g. Firefox never ran).
        """
        return {"ok": True, "apps": apps_mod.census(wins.list())}

    # -- memory and learning ----------------------------------------------
    @tool
    def learn_skill(name: str, description: str = "", steps: list[dict] | None = None) -> dict:
        """Save a reusable skill. With no steps, distil this session's successful actions."""
        if steps:
            return {"ok": True, **memory.learn_skill(name, steps, description)}
        return {"ok": True, **memory.distill(name, description)}

    @tool
    def run_skill(name: str, args: dict | None = None) -> dict:
        """Replay a skill step by step. `args` fills placeholders like {url}."""
        skill = memory.skill(name)
        if skill is None:
            return {"ok": False, "error": f"unknown skill {name}", "available": memory.skills()}
        values = args or {}
        results = []
        for step in skill["steps"]:
            fn = registry.get(step["tool"])
            if fn is None:
                results.append({"tool": step["tool"], "ok": False, "error": "not a tool"})
                continue
            step_args = dict(step.get("args") or {})
            for k, v in list(step_args.items()):
                if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                    key = v[1:-1]
                    if key in values:
                        step_args[k] = values[key]
                    else:
                        return {"ok": False, "error": f"missing skill arg {{{key}}}",
                                "skill": name, "provided": list(values)}
            outcome = fn(**step_args)
            ok = not (isinstance(outcome, dict) and (outcome.get("ok") is False
                                                    or outcome.get("error")))
            results.append({"tool": step["tool"], "ok": ok})
            if not ok:
                memory.record_skill_use(name, False)
                return {"ok": False, "skill": name, "stopped_at": step["tool"],
                        "results": results}
        memory.record_skill_use(name, True)
        return {"ok": True, "skill": name, "steps": len(results)}

    @tool
    def list_skills() -> dict:
        """List learned skills with use and win counts."""
        return {"skills": memory.skills()}

    @tool
    def forget_skill(name: str) -> dict:
        """Delete a stale or wrong skill."""
        return {"ok": memory.forget_skill(name), "name": name}

    @tool
    def remember(key: str, value: str, ttl_s: float | None = None) -> dict:
        """Store a small fact for later runs (window class, button label, timing).

        Every fact is stamped with the time it was written; a ttl_s makes
        recall report it stale (and stop returning the value) once expired.
        Facts without a ttl persist, and recall always reports their age.
        """
        memory.remember(key, value, ttl_s=ttl_s)
        return {"ok": True, "key": key, "ttl_s": ttl_s}

    @tool
    def recall(key: str) -> dict:
        """Read a fact stored with remember: value, age in seconds, staleness.

        A fact whose ttl has expired comes back stale with value null --
        stale claims ("window class is X" after an app update) must not be
        acted on silently. Unknown keys return value null.
        """
        return {"ok": True, "key": key, **memory.recall(key)}

    @tool
    def stats(hours: float = 24.0) -> dict:
        """Per-tool calls, successes and latency plus skill counts.

        Lifetime aggregates sit next to a `recent` window (default 24 h):
        the lifetime rate of a tool whose bug was just fixed otherwise reads
        as still-broken forever (screenshot showed 11% over 561 calls while
        the fixed path scored 100%). hours=0 gives lifetime only.
        """
        return {"ok": True, **memory.stats(hours=hours)}

    @tool
    def session_end(note: str = "", auto_improve: bool = True) -> dict:
        """Close the session: delete its screenshots, report what worked.

        auto_improve (default on) distils the session's successful actions
        into a skill BEFORE closing: every session leaves smartcomp with a
        reusable flow (skills gain use/win counts on replay), so more use
        means a smarter, faster daemon. A no-op when there was nothing
        worth distilling.
        """
        # Distil BEFORE end_session rotates the session id -- otherwise the
        # skill is built from the NEW (empty) session and always fails.
        distilled = None
        if auto_improve:
            from datetime import datetime
            try:
                distilled = memory.distill(
                    f"auto-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    "self-improvement: distilled from this session's successful actions")
            except ValueError as exc:
                distilled = {"skipped": str(exc)}
        from . import ownership as own

        released = own.release_all()
        result = {"ok": True, **memory.end_session(note),
                  "apps_released": released.get("released", [])}
        if distilled is not None:
            result["skill"] = distilled
        return result

    return registry
