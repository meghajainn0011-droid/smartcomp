"""Reproduce the key_down/key_up failure step by step with readbacks."""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartcomp import config


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

    async def guard(self, needle: str) -> None:
        a = await self.active_window()
        blob = f"{a['title']} {a['class']}".lower()
        if needle.lower() not in blob:
            raise AssertionError(f"active={a['title']!r}, expected {needle!r}")

    @staticmethod
    def inside(win: dict, px: float, py: float) -> tuple[int, int]:
        x = win["x"] + win["w"] * px / 1000
        y = win["y"] + win["h"] * py / 1000
        return round(x / 1920 * 1000), round(y / 1080 * 1000)


async def main() -> int:
    subprocess.run(["pkill", "-x", "mousepad"], check=False)
    await asyncio.sleep(0.8)
    for autosave in Path.home().glob(".local/share/Mousepad/autosave-*"):
        autosave.unlink(missing_ok=True)

    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    url = f"http://{config.MCP_HOST}:{config.MCP_PORT}/mcp"
    headers = {"Authorization": f"Bearer {config.read_token()}"}
    async with streamablehttp_client(url, headers=headers) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            api = Client(s)

            pre = {x["id"] for x in await api.windows()}
            await api.call("launch_app", name="text-editor")
            ed = await api.wait_window("mousepad", 30, exclude=pre)
            await api.call("focus_window", window_id=ed["id"])
            await asyncio.sleep(0.6)
            await api.call("window_state", window_id=ed["id"], state="maximize")
            await asyncio.sleep(1.0)
            ed = {x["id"]: x for x in await api.windows()}[ed["id"]]
            cx, cy = api.inside(ed, 500, 400)
            await api.call("click", x=cx, y=cy)
            await api.guard("mousepad")

            await api.call("type_text", text="hello world")
            await asyncio.sleep(0.4)
            t = await api.call("read_window", window_id=ed["id"])
            print("STEP1 typed :", repr(t["text"][:60]))

            await api.guard("mousepad")
            await api.call("key_down", combo="ctrl")
            await asyncio.sleep(0.15)
            await api.call("key", combo="a")
            await asyncio.sleep(0.15)
            await api.call("key_up", combo="ctrl")
            await asyncio.sleep(0.4)
            shot = await api.call("screenshot")
            print("STEP2 select-all screenshot:", shot["path"])
            t = await api.call("read_window", window_id=ed["id"])
            print("STEP2 after ctrl+a :", repr(t["text"][:60]))

            await api.guard("mousepad")
            await api.call("type_text", text="X")
            await asyncio.sleep(0.5)
            t = await api.call("read_window", window_id=ed["id"])
            print("STEP3 retype  :", repr(t["text"][:60]))

            await api.call("close_window", window_id=ed["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
