#!/usr/bin/env python3
"""Convenience CLI client for the SmartComp MCP daemon."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/damon/smartcomp")
from smartcomp import config
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    if len(sys.argv) < 2:
        print("Usage: call.py <tool_name> [json_args]")
        sys.exit(1)

    tool_name = sys.argv[1]
    args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}

    token = config.read_token()
    url = f"http://{config.MCP_HOST}:{config.MCP_PORT}/mcp"

    async with streamablehttp_client(url, headers={"Authorization": f"Bearer {token}"}) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool(tool_name, args)
            for block in getattr(res, "content", []):
                text = getattr(block, "text", None)
                if text:
                    print(text)
                    return
            print(res)

if __name__ == "__main__":
    asyncio.run(main())
