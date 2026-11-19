"""End-to-end smoke test: a real MCP client calling get_changes over HTTP.

Run the daemon first (`python -m smartcomp.daemon serve`), then:
    .venv/bin/python tests/mcp_smoke.py
It asserts the auth boundary (no token => 401) and that consecutive calls
return valid delta payloads.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartcomp import config  # noqa: E402


def _payload(result) -> dict:
    for block in getattr(result, "content", []):
        text = getattr(block, "text", None)
        if text:
            return json.loads(text)
    raise AssertionError(f"no text content in tool result: {result!r}")


async def main() -> int:
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    token = config.read_token()
    url = f"http://{config.MCP_HOST}:{config.MCP_PORT}/mcp"

    async with streamablehttp_client(url, headers={"Authorization": f"Bearer {token}"}) as (
        read,
        write,
        _get_session_id,
    ):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print("server:", init.serverInfo.name, init.serverInfo.version)
            tools = await session.list_tools()
            print("tools:", [t.name for t in tools.tools])

            first = _payload(await session.call_tool("get_changes", {"force_full": True}))
            print(
                f"call 1: seq={first['seq']} full={first['full']} "
                f"add={len(first.get('add', []))} del={len(first.get('del', []))} "
                f"stats={first['stats']}"
            )
            second = _payload(await session.call_tool("get_changes", {}))
            print(
                f"call 2: seq={second['seq']} full={second['full']} "
                f"add={len(second.get('add', []))} del={len(second.get('del', []))} "
                f"stats={second['stats']}"
            )
            health = _payload(await session.call_tool("health", {}))
            print("health:", health)

            assert first["w"] == config.LOGICAL_W and first["h"] == config.LOGICAL_H
            assert first["seq"] == 1, "sequence should start at 1 for a fresh server"
            assert second["seq"] == 2
            texts = [e for e in first.get("add", []) if e["k"] == "t"]
            assert texts, "expected OCR text on a screen with a terminal open"
            assert all(e.get("u") == 1 for e in texts), "screen text must be marked untrusted"
            for el in first.get("add", []):
                x, y, w, h = el["b"]
                assert 0 <= x <= 1000 and 0 <= y <= 1000 and w > 0 and h > 0, el
            print(f"sample text ({len(texts)}):", [e["x"] for e in texts[:5]])

    # Auth boundary: the same endpoint without a token must refuse.
    async with httpx.AsyncClient() as client:
        r = await client.post(
            url,
            headers={"Accept": "application/json, text/event-stream"},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )
        print("no token:", r.status_code, r.text[:60])
        assert r.status_code == 401, "unauthenticated requests must be rejected"

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
