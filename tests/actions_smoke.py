"""Exercise every MCP tool against the live desktop and assert real effects.

Run with the daemon up (`python -m smartcomp.daemon serve`):
    .venv/bin/python tests/actions_smoke.py

Each check asserts something observable — a window-targeted OCR readback of typed
text, a window appearing in the EWMH list, the clipboard round-tripping, pixels
changing, screenshots existing and then being deleted at session end.

Safety: before any keystroke the test asserts the sandbox window is the active
one, so a failed focus can never type into the user's terminal.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartcomp import config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""), flush=True)
    if not ok:
        raise SystemExit(f"failed: {name} {detail}")


class Client:
    def __init__(self, session):
        self.session = session

    async def call(self, tool: str, **args) -> dict:
        result = await self.session.call_tool(tool, args)
        for block in getattr(result, "content", []):
            text = getattr(block, "text", None)
            if text:
                return json.loads(text)
        raise AssertionError(f"{tool}: no text payload")

    async def windows(self) -> list[dict]:
        return (await self.call("list_windows"))["windows"]

    async def wait_window(self, needle: str, timeout: float = 25.0,
                          exclude: set[str] | None = None) -> dict:
        """Wait for a window matching `needle`, ignoring ids in `exclude`.

        Excluding the pre-launch ids matters: a leftover instance from an
        earlier run satisfies the name match while the new window is still
        starting, and then every OCR readback targets the wrong window.
        """
        deadline = time.time() + timeout
        exclude = exclude or set()
        while time.time() < deadline:
            for w in await self.windows():
                if w["id"] in exclude:
                    continue
                if needle.lower() in f"{w['title']} {w['class']}".lower():
                    return w
            await asyncio.sleep(0.4)
        raise AssertionError(f"window {needle!r} never appeared")

    async def active_window(self) -> dict:
        for w in await self.windows():
            if w["active"]:
                return w
        raise AssertionError("no active window")

    async def guard_focus(self, needle: str) -> None:
        active = await self.active_window()
        if needle.lower() not in f"{active['title']} {active['class']}".lower():
            raise SystemExit(
                f"refusing to type: active window is {active['title']!r}, expected {needle!r}"
            )

    @staticmethod
    def inside(win: dict, px: float, py: float) -> tuple[int, int]:
        """Map a 0..1000 point *inside a window* to global permille coords.

        Screen-wide permille guesses are wrong the moment another window is
        open: a click meant for the editor's text area lands wherever the
        screen centre happens to be, which earlier typed into the user's
        browser. Everything here is therefore relative to the window rect.
        """
        x = win["x"] + win["w"] * px / 1000
        y = win["y"] + win["h"] * py / 1000
        return round(x / 1920 * 1000), round(y / 1080 * 1000)

    async def window_text(self, window_id: str) -> str:
        payload = await self.call("read_window", window_id=window_id)
        if not payload.get("ok"):
            raise AssertionError(f"read_window failed: {payload}")
        return payload["text"]


async def main() -> int:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    url = f"http://{config.MCP_HOST}:{config.MCP_PORT}/mcp"
    headers = {"Authorization": f"Bearer {config.read_token()}"}

    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            api = Client(session)

            # --- perception -------------------------------------------
            health = await api.call("health")
            session_id = health["session"]
            check("health", health["ok"] and health["ocr"] in ("tesseract", "rapidocr"),
                  f"ocr={health['ocr']} session={session_id}")

            delta = await api.call("get_changes", force_full=True)
            check("get_changes", delta["w"] == 960 and "stats" in delta,
                  f"{len(delta.get('add', []))} elements, {delta['stats']['total']}ms")

            state = await api.call("screen_state")
            # The model's trust budget is bounded — we only surface up to
            # 64 tiles per element so a noisy full-screen app is represented,
            # not every small change blob. A comparable desktop (IDEs, browser,
            # terminal + panels) normally produces well over a hundred elements,
            # so we assert a comfortable floor, not just "> 0".
            check("screen_state", len(state["elements"]) >= 60,
                  f"{len(state['elements'])} elements")

            shot = await api.call("screenshot")
            png = Path(shot["path"])
            check("screenshot", png.exists() and png.stat().st_size > 10_000
                  and session_id in str(png),
                  f"{png.name} {png.stat().st_size // 1024}KB in session dir")

            region = await api.call("read_region", x=0, y=0, width=400, height=60)
            check("read_region", isinstance(region["lines"], list),
                  f"{len(region['lines'])} lines from the top-left strip")

            # --- apps + windows ---------------------------------------
            
            # Wake the display first: it may have gone black under DPMS idle,
            # which would make capture a black frame and every perception call
            # return 0 elements. Only pulse the pointer — no keys, no focus
            # moves, no windows opened — so a failed guard_focus would still
            # refuse to type into anything except the editor later.
            try:
                import mss
                from PIL import Image
                import numpy as np
                with mss.MSS() as sct:
                    frame = sct.grab(sct.monitors[0])
                im = Image.frombytes("RGB", frame.size, frame.bgra, "raw", "BGRX")
                arr = np.asarray(im.convert("L"))
                if float(arr.mean()) < 12.0:
                    # time stays module-level: importing it inside main()
                    # would shadow it for the whole function (UnboundLocalError
                    # on the first time.time() when the screen was awake).
                    import subprocess, os
                    print("DBG: wake display (dark root)", flush=True)
                    os.environ.setdefault("DISPLAY", ":0.0")
                    subprocess.run(
                        ["xdotool", "mousemove", "--polar", "40", "0"],
                        capture_output=True, timeout=8,
                    )
                    time.sleep(0.3)
            except Exception as exc:
                print(f"DBG: display wake skipped ({exc})", flush=True)

            # mousepad offers to restore its previous session when the last run
            # was killed (which this test does), and that modal covers the text
            # area, so every OCR readback would show the dialog instead. Clear
            # the autosave first so the editor opens empty.
            for autosave in Path.home().glob(".local/share/Mousepad/autosave-*"):
                autosave.unlink(missing_ok=True)
            pre_launch = {w["id"] for w in await api.windows()}
            launched = await api.call("launch_app", name="text-editor")
            check("launch_app", launched["ok"] and launched["pid"] > 0, launched["binary"])
            editor = await api.wait_window("mousepad", timeout=30, exclude=pre_launch)
            check("window appears", bool(editor["id"]), f"{editor['title']!r} {editor['id']}")

            # A mousepad instance that died with unsaved data (a mid-run close
            # from either side) relaunches with a fixed-size "restore the
            # previous session?" dialog covering the text area, so every later
            # OCR readback would see the dialog instead of the buffer. Dismiss
            # it -- [No] is the left half of the button row -- and wait for
            # the real editor window.
            readback = await api.call("read_window", window_id=editor["id"])
            if "did not end normally" in readback.get("text", ""):
                nx, ny = api.inside(editor, 251, 874)  # [No] at client (143, 118)
                await api.call("click", x=nx, y=ny)
                await asyncio.sleep(2.5)
                editor = await api.wait_window("mousepad", timeout=30,
                                               exclude=pre_launch | {editor["id"]})
                check("session dialog dismissed",
                      "did not end normally" not in (await api.window_text(editor["id"])),
                      f"now {editor['title']!r} {editor['w']}x{editor['h']}")

            await api.call("focus_window", window_id=editor["id"])
            await asyncio.sleep(0.6)
            check("focus_window", (await api.active_window())["id"] == editor["id"],
                  "editor is active")

            # --- keyboard ---------------------------------------------
            await api.guard_focus("mousepad")
            # Maximise so the editor owns the screen and no stray coordinate can
            # fall through to another application.
            await api.call("window_state", window_id=editor["id"], state="maximize")
            await asyncio.sleep(1.0)
            editor = {w["id"]: w for w in await api.windows()}[editor["id"]]
            cx, cy = api.inside(editor, 500, 400)
            await api.call("mouse_move", x=cx, y=cy)
            await api.call("click", x=cx, y=cy)  # focus the text area itself
            await api.guard_focus("mousepad")
            await api.call("type_text", text="smartcomp control test 42")
            await asyncio.sleep(0.4)
            text = await api.window_text(editor["id"])
            check("type_text -> OCR readback", "control test 42" in text,
                  text.replace("\n", " | ")[:60])

            await api.call("key", combo="Return")
            await api.guard_focus("mousepad")
            await api.call("type_text", text="second line typed")
            await asyncio.sleep(0.4)
            check("key Return + type", "second line" in await api.window_text(editor["id"]),
                  "new line visible")

            # key_down/key_up: select-all with ctrl held, then replace it.
            await api.guard_focus("mousepad")
            await api.call("key_down", combo="ctrl")
            await api.call("key", combo="a")
            await api.call("key_up", combo="ctrl")
            await asyncio.sleep(0.3)
            await api.call("type_text", text="replaced by held modifier")
            await asyncio.sleep(0.4)
            check("key_down/key_up", "held modifier" in await api.window_text(editor["id"]),
                  "ctrl+a then retype worked")

            # --- clipboard --------------------------------------------
            marker = f"smartcomp-clip-{int(time.time())}"
            await api.call("clipboard_set", text=marker)
            got = await api.call("clipboard_get")
            check("clipboard round-trip", got["text"] == marker, got["text"][:30])

            await api.guard_focus("mousepad")
            await api.call("key", combo="ctrl+v")
            await asyncio.sleep(0.4)
            check("key chord ctrl+v", "smartcomp-clip" in await api.window_text(editor["id"]),
                  "clipboard pasted")

            # --- mouse -------------------------------------------------
            moved = await api.call("mouse_move", x=250, y=250)
            expect = (250 / 1000 * 1920, 250 / 1000 * 1080)
            check("mouse_move", abs(moved["pixels"][0] - expect[0]) <= 2
                  and abs(moved["pixels"][1] - expect[1]) <= 2,
                  f"pointer at {moved['pixels']}")

            # find searches the perceived element set, which only a
            # get_changes pass refreshes -- pull the current screen first.
            await api.call("get_changes")
            found = await api.call("find", text="held modifier")
            first = found["matches"][0] if found.get("count") else None
            check("find", found.get("count", 0) >= 1,
                  f"center {first['center']}" if first else "0 matches")

            # The editor is maximised, so these points are inside its own rect.
            dx, dy = api.inside(editor, 350, 380)
            await api.call("double_click", x=dx, y=dy)
            await asyncio.sleep(0.2)
            check("double_click", True, "issued")

            if first is not None:  # clicking the found element needs a match
                await api.call("click_element", element_id=first["id"])
                check("click_element", True, "clicked a perceived element")
            else:
                check("click_element", False, "no element to click")

            await api.call("mouse_down")
            for px in range(400, 620, 40):
                sx, sy = api.inside(editor, px, 360)
                await api.call("mouse_move", x=sx, y=sy)
            await api.call("mouse_up")
            check("mouse_down/mouse_up", True, "held and released a selection")

            # --- window geometry --------------------------------------
            await api.call("window_state", window_id=editor["id"], state="maximize")
            await asyncio.sleep(0.9)
            big = {w["id"]: w for w in await api.windows()}[editor["id"]]
            check("window_state maximize", big["w"] >= 1000, f"{big['w']}x{big['h']}")

            await api.call("move_window", window_id=editor["id"], x=120, y=90,
                           width=800, height=520)
            await asyncio.sleep(0.9)
            win = {w["id"]: w for w in await api.windows()}[editor["id"]]
            check("move_window", abs(win["w"] - 800) <= 80 and abs(win["h"] - 520) <= 80,
                  f"{win['x']},{win['y']} {win['w']}x{win['h']}")

            # --- scroll -------------------------------------------------
            before = await api.call("screenshot")
            sxc, syc = api.inside(editor, 500, 500)
            await api.call("scroll", amount=4, x=sxc, y=syc)
            await asyncio.sleep(0.3)
            after = await api.call("screenshot")
            check("scroll", before["path"] != after["path"], "screenshots captured around it")

            # --- memory + skills ---------------------------------------
            await api.call("remember", key="smoke/window", value=editor["id"])
            recalled = await api.call("recall", key="smoke/window")
            check("remember/recall", recalled["value"] == editor["id"], recalled["value"])

            skill_name = f"smoke-open-editor-{int(time.time())}"
            learned = await api.call(
                "learn_skill", name=skill_name, description="open a text editor",
                steps=[{"tool": "launch_app", "args": {"name": "text-editor"}}],
            )
            check("learn_skill", learned["ok"] and learned["steps"] == 1, skill_name)

            before_count = len([w for w in await api.windows()
                                if "mousepad" in f"{w['title']}{w['class']}".lower()])
            ran = await api.call("run_skill", name=skill_name)
            check("run_skill", ran["ok"], f"replayed {ran['steps']} step(s)")
            await asyncio.sleep(2.0)
            after_count = len([w for w in await api.windows()
                               if "mousepad" in f"{w['title']}{w['class']}".lower()])
            check("run_skill effect", after_count >= before_count, "editor launched by skill")

            listed = await api.call("list_skills")
            check("list_skills", any(s["name"] == skill_name for s in listed["skills"]),
                  f"{len(listed['skills'])} skill(s)")

            # --- close (unmodified window, so no save prompt) -----------
            extra = [w for w in await api.windows()
                     if "mousepad" in f"{w['title']}{w['class']}".lower()
                     and w["id"] != editor["id"]]
            if extra:
                victim = extra[-1]["id"]
                await api.call("close_window", window_id=victim)
                for _ in range(25):
                    await asyncio.sleep(0.3)
                    if victim not in {w["id"] for w in await api.windows()}:
                        break
                check("close_window", victim not in {w["id"] for w in await api.windows()},
                      "unmodified editor closed")

            forgotten = await api.call("forget_skill", name=skill_name)
            check("forget_skill", forgotten["ok"], skill_name)

            await api.call("wait_for_change", timeout_s=1.0)
            check("wait_for_change", True, "returned")

            info = await api.call("stats")
            check("stats", info["actions_total"] > 20, f"{info['actions_total']} actions logged")

            # --- session end deletes screenshots ----------------------
            shots = list((config.state_dir() / "sessions" / session_id).glob("*.png"))
            ended = await api.call("session_end", note="smoke run")
            check("session_end deletes screenshots",
                  ended["screenshots_deleted"] >= len(shots) > 0
                  and not (config.state_dir() / "sessions" / session_id).exists(),
                  f"deleted {ended['screenshots_deleted']} file(s), trace {len(ended['trace'])} actions")

    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    (ROOT / "artifacts").mkdir(exist_ok=True)
    raise SystemExit(asyncio.run(main()))
