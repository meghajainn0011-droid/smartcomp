"""Quick live check: connect to the daemon, call health + get_changes."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartcomp import config  # noqa: E402


async def main() -> int:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    url = f"http://{config.MCP_HOST}:{config.MCP_PORT}/mcp"
    headers = {"Authorization": f"Bearer {config.read_token()}"}
    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("tools:", sorted(t.name for t in tools.tools))
            h = await session.call_tool("health", None)
            print("health:", h.content[0].text)
            d = await session.call_tool("get_changes", {"force_full": True})
            payload = json.loads(d.content[0].text)
            print("get_changes: elements=%d tiles=%d total=%.0fms" % (
                len(payload.get("add", [])), payload.get("stats", {}).get("tiles", 0),
                payload.get("stats", {}).get("total", 0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
