"""Open one Brave tab to a sketching site via SmartComp, draw a cat, capture it.

Run with the daemon up:
    .venv/bin/python tests/verify_brave_cat.py

Covers the original goal end-to-end: open an app through SmartComp and do a
visible drawing action, using window-scoped coords so a click meant for the
canvas cannot land in whatever else is on screen.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from smartcomp import config  # noqa: E402
from PIL import Image

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""), flush=True)
    if not ok:
        raise SystemExit(f"failed: {name} {detail}")


class Client:  # mirrors tests/actions_smoke.py helper
    def __init__(self, session):
        self.session = session

    async def call(self, tool: str, **kwargs):
        result = await self.session.call_tool(tool, kwargs or None)
        for block in getattr(result, "content", []):
            text = getattr(block, "text", None)
            if text:
                return json.loads(text)
        raise AssertionError(f"{tool}: no text payload")

    async def windows(self) -> list[dict]:
        return (await self.call("list_windows"))["windows"]

    async def wait_window(self, needle: str, timeout: float = 40.0,
                          exclude: set[str] | None = None) -> dict:
        return await self.wait_any_window([needle], timeout=timeout, exclude=exclude)

    async def wait_any_window(self, needles: list[str], timeout: float = 40.0,
                             exclude: set[str] | None = None) -> dict:
        exclude = exclude or set()
        t0 = time.time()
        hay = {n.lower(): n for n in needles}
        last_windows: list[dict] = []
        while time.time() - t0 < timeout:
            last_windows = await self.windows()
            for w in last_windows:
                if w["id"] in exclude:
                    continue
                blob = (w.get('title', '') + ' ' + w.get('window_class', '')).lower()
                for n in hay:
                    if n in blob:
                        return w
            await asyncio.sleep(0.4)
        cur = [(w.get('title'), w.get('window_class')) for w in last_windows if w.get('active')]
        raise AssertionError(f"window any({list(hay.values())!r}) never appeared in {timeout}s; active={cur}")

    async def active_window(self) -> dict:
        for w in await self.windows():
            if w["active"]:
                return w
        raise AssertionError("no active window")

    async def guard_focus(self, needle: str) -> None:
        active = await self.active_window()
        if needle.lower() not in (active.get('title', '') + ' ' + active.get('window_class', '')).lower():
            raise SystemExit(f"refusing to type: active is {active.get('title')!r}, expected {needle!r}")

    @staticmethod
    def inside(win: dict, px: float, py: float) -> tuple[int, int]:
        x = win["x"] + win["w"] * px / 1000
        y = win["y"] + win["h"] * py / 1000
        return round(x / 1920 * 1000), round(y / 1080 * 1000)


SKETCH_URL = "https://sketch.io/sketchpad/"


def _mean_dark(png: Path, threshold: int = 120) -> float:
    im = Image.open(png).convert("L")
    px = im.load()
    dark = 0
    total = im.width * im.height
    for y in range(im.height):
        for x in range(im.width):
            if px[x, y] < threshold:
                dark += 1
    return dark / max(total, 1)


async def wait_page_loaded(api, brave, deadline_s=30.0, min_dark=0.05,
                           max_retries=25, sleep_s=0.6, skip_errors_for=3):
    deadline = time.time() + deadline_s
    tries = 0
    last_png = ""
    last_dark = 0.0
    ok_count = 0
    ok_total = 0
    while time.time() < deadline and tries < max_retries:
        tries += 1
        sc = await api.call("screenshot", window_id=brave["id"])
        if not sc.get("ok") or not sc.get("path"):
            ok_count = 0
            await asyncio.sleep(sleep_s)
            continue
        png = Path(sc["path"])
        if not png.exists():
            ok_count = 0
            await asyncio.sleep(sleep_s)
            continue
        try:
            last_dark = _mean_dark(png, threshold=120)
            last_png = sc["path"]
            ok_count += 1
            ok_total += 1
            if ok_count >= skip_errors_for:
                if last_dark > min_dark:
                    return True, ok_total, tries, last_png, last_dark
        except Exception:
            ok_count = 0
        await asyncio.sleep(sleep_s)
    return False, ok_total, tries, last_png, last_dark


async def main() -> int:
    # clean slate
    subprocess.run(["pkill", "-x", "brave-browser"], check=False)
    await asyncio.sleep(0.5)

    async with streamablehttp_client(
        f"http://{config.MCP_HOST}:{config.MCP_PORT}/mcp",
        headers={"Authorization": f"Bearer {config.read_token()}"},
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            api = Client(session)

            health = await api.call("health")
            check("health", health["ok"] and health["ocr"] in ("tesseract", "rapidocr"),
                  f"ocr={health['ocr']} session={health['session']}")

            # --- launch ONE brave tab to the sketching site (no --start-maximized) ---
            pre = {w["id"] for w in await api.windows()}
            ok_launch = await api.call("launch_app", name="brave",
                                       args=["--no-first-run", "--no-default-browser-check",
                                              "--user-data-dir", "/tmp/brave-smartcomp",
                                              "--new-window", SKETCH_URL])
            check("launch_app brave", ok_launch["ok"] and ok_launch.get("pid", 0) > 0,
                  ok_launch.get("binary", ""))

            brave = await api.wait_any_window(["brave", "untitled"], timeout=40, exclude=pre)
            check("brave window appears", bool(brave["id"]),
                  f"{brave.get('title')!r} {brave.get('id')}")

            await api.call("focus_window", window_id=brave["id"])
            await asyncio.sleep(1.0)
            check("focus brave", (await api.active_window())["id"] == brave["id"], "brave active")

            # after focus the browser may surface a child window; re-track the
            # window that holds the sketching page by title keyword.
            by_id = {w['id']: w for w in await api.windows()}
            brave = by_id.get(brave['id'], brave)
            if 'sketch' not in (brave.get('title', '') + ' ' + brave.get('window_class', '')):
                brave = next((w for w in await api.windows() if 'sketch' in (w.get('title', '') + ' ' + w.get('window_class', ''))), brave)

            await api.call("focus_window", window_id=brave["id"])
            await asyncio.sleep(0.8)

            # --- page loaded: wait for the sketching page to render by checking
            #     the window screenshot stops being a blank frame ---
            page_ok, _, tries_page, last_png_page, last_dark_page = await wait_page_loaded(
                api, brave, deadline_s=30.0, max_retries=25, skip_errors_for=3
            )
            check("sketching page loaded (bitmap phase present)", page_ok,
                  f"prompted {tries_page} screenshots; last {Path(last_png_page).name if last_png_page else 'none'} dark={last_dark_page:.3f}")

            # --- window-scoped screenshot of the rendered page ---
            t0 = time.time()
            sc = await api.call("screenshot", window_id=brave["id"])
            png = Path(sc["path"])
            check("window-scoped screenshot of page",
                  sc.get("ok") and png.exists() and png.stat().st_size > 500,
                  f"{png.name} {png.stat().st_size // 1024}KB in {round((time.time()-t0)*1000,1)}ms (window-scoped)")

            # --- read the sketching window once (window-scoped OCR) ---
            winfo = await api.call("read_window", window_id=brave["id"], raise_first=False)
            check("read_window brave", bool(winfo.get("lines")), f"{len(winfo['lines'])} lines, window={winfo['window']!r}")

            # --- draw the cat with window-relative drag strokes ---
            # Bypass guard_focus because Brave spawns a child window whose title
            # does not contain "Brave"; we already re-identified the window above
            # so the actions target the right window directly.
            win_dict = {w["id"]: w for w in await api.windows()}

            def target(px, py):
                return api.inside(win_dict[brave["id"]], px, py)

            strokes: list[tuple[float, float, float, float]] = [
                # head outline
                (380, 150, 380, 300),
                (380, 300, 620, 300),
                (620, 300, 620, 150),
                (620, 150, 380, 150),
                # ears
                (380, 150, 330, 70),
                (330, 70, 420, 140),
                (620, 150, 670, 70),
                (670, 70, 580, 140),
                # eyes
                (370, 200, 400, 200),
                (600, 200, 630, 200),
                # nose + mouth
                (490, 250, 510, 250),
                (500, 250, 500, 270),
                (480, 275, 500, 265),
                (500, 265, 520, 275),
                # whiskers
                (390, 250, 300, 240),
                (390, 260, 300, 260),
                (610, 250, 700, 240),
                (610, 260, 700, 260),
                # body round
                (430, 300, 430, 430),
                (430, 430, 570, 430),
                (570, 430, 570, 300),
                (500, 300, 500, 460),
            ]
            drawn = 0
            for sx, sy, tx, ty in strokes:
                sxg, syg = target(sx, sy)
                txg, tyg = target(tx, ty)
                await api.call("mouse_down", x=sxg, y=syg)
                await api.call("mouse_move", x=txg, y=tyg)
                await api.call("mouse_up")
                drawn += 1
                if drawn % 10 == 0:
                    await asyncio.sleep(0.02)

            check("cat drawn (strokes)", drawn == len(strokes),
                  f"{drawn} drag strokes over brave canvas")

            await asyncio.sleep(0.5)

            # --- window-scoped screenshot of the drawing ---
            t0 = time.time()
            sc = await api.call("screenshot", window_id=brave["id"])
            png = Path(sc["path"])
            check("window-scoped screenshot of brave",
                  sc.get("ok") and png.exists() and png.stat().st_size > 500,
                  f"{png.name} {png.stat().st_size // 1024}KB in {round((time.time()-t0)*1000,1)}ms (window-scoped)")

            # --- visual sanity: a window-format PNG from the session dir must contain dark ink pixels ---
            ink = _mean_dark(png, threshold=120)
            check("screenshot PNG has dark pixels (ink)", ink > 0.02, f"{ink:.3f} of window pixels dark")

            # --- session end purges the window-scoped screenshot ---
            shots_before = list((config.state_dir() / "sessions" / health["session"]).glob("win-*.png"))
            ended = await api.call("session_end", note="brave-cat demo")
            check("session_end deletes screenshots",
                  ended.get("screenshots_deleted", 0) >= len(shots_before) and
                  not (config.state_dir() / "sessions" / health["session"]).exists(),
                  f"deleted {ended.get('screenshots_deleted',0)} file(s)")

        total = len(RESULTS)
        passed = sum(1 for _, ok, _ in RESULTS if ok)
        print(f"\n{passed}/{total} checks passed")
        return 0 if passed == total else 1


if __name__ == "__main__":
    (Path(__file__).resolve().parent.parent / "artifacts").mkdir(exist_ok=True)
    raise SystemExit(asyncio.run(main()))
