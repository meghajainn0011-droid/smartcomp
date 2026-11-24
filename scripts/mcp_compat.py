#!/usr/bin/env python3
"""MCP interoperability matrix for smartcomp.

Different agents speak slightly different dialects of streamable HTTP: some
open a session and keep it, some replay a stale session id after the daemon
restarted, some never send a session id at all, some send only
`Accept: application/json` instead of the spec's `application/json,
text/event-stream`. `Session not found`, `Invalid request parameters` and 406s
are the ways those differences show up, and they look like "the MCP server is
broken" to whoever hits them.

This script starts a throwaway daemon on a spare port and exercises every one
of those paths, plus the ordinary SDK flow, so a regression is one command
away instead of a support thread.

    .venv/bin/python scripts/mcp_compat.py

Exit code is 0 only when every check passes.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
TOKEN = (Path.home() / ".local/state/smartcomp/token").read_text().strip()
BOTH = "application/json, text/event-stream"

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    results.append((ok, name, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""), flush=True)
    return ok


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_daemon(port: int) -> subprocess.Popen:
    env = dict(os.environ, SMARTCOMP_PORT=str(port))
    proc = subprocess.Popen(
        [str(ROOT / ".venv/bin/smartcomp"), "serve"],
        cwd=str(ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            r = httpx.post(f"http://127.0.0.1:{port}/mcp",
                           headers={"Authorization": f"Bearer {TOKEN}", "Accept": BOTH},
                           json={"jsonrpc": "2.0", "id": 1, "method": "ping"}, timeout=2)
            if r.status_code < 500:
                return proc
        except Exception:
            time.sleep(1)
    proc.kill()
    raise SystemExit(f"daemon on {port} did not come up")


def parse(resp: httpx.Response) -> list[dict]:
    """Accept either a JSON body or an SSE frame containing JSON."""
    ctype = resp.headers.get("content-type", "")
    text = resp.text
    if "text/event-stream" in ctype:
        out = []
        for block in text.split("\n\n"):
            for line in block.splitlines():
                if line.startswith("data:"):
                    try:
                        out.append(json.loads(line[5:].strip()))
                    except json.JSONDecodeError:
                        pass
        return out
    try:
        return [resp.json()]
    except Exception:
        return []


def post(base: str, payload: dict, session: str | None = None,
         accept: str = BOTH, auth: str | None = None) -> httpx.Response:
    headers = {"Accept": accept, "Content-Type": "application/json"}
    if auth is not False:
        headers["Authorization"] = f"Bearer {auth or TOKEN}"
    if session:
        headers["Mcp-Session-Id"] = session
    return httpx.post(f"{base}/mcp", headers=headers, json=payload, timeout=120)


def initialize(base: str, version: str = "2025-06-18") -> tuple[str | None, dict]:
    r = post(base, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": version, "capabilities": {},
                               "clientInfo": {"name": "compat", "version": "1"}}})
    sess = r.headers.get("mcp-session-id")
    msgs = parse(r)
    return sess, (msgs[0] if msgs else {})


def main() -> int:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    proc = start_daemon(port)
    ok_overall = True
    try:
        # -- SDK-style: initialize, keep the session, call tools ------------
        sess, init = initialize(base)
        ok_overall &= check(bool(sess), "initialize returns a session id", str(sess)[:12])

        r = post(base, {"jsonrpc": "2.0", "method": "notifications/initialized"}, session=sess)
        ok_overall &= check(r.status_code in (200, 202), "notifications/initialized accepted",
                            str(r.status_code))

        r = post(base, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, session=sess)
        tools = (parse(r) or [{}])[0].get("result", {}).get("tools", [])
        ok_overall &= check(r.status_code == 200 and len(tools) > 40,
                            "tools/list on a live session", f"{len(tools)} tools")

        r = post(base, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                        "params": {"name": "list_windows", "arguments": {}}}, session=sess)
        res = (parse(r) or [{}])[0].get("result", {})
        ok_overall &= check(r.status_code == 200 and "content" in res,
                            "tools/call returns content", json.dumps(res)[:60])
        ok_overall &= check("structuredContent" in res,
                            "tools/call also returns structuredContent")

        # -- stateless client: no session id at all -------------------------
        r = post(base, {"jsonrpc": "2.0", "id": 4, "method": "tools/list"})
        count = len(((parse(r) or [{}])[0].get("result") or {}).get("tools", []))
        ok_overall &= check(r.status_code == 200 and count > 40,
                            "tools/list without a session id (stateless client)",
                            f"{r.status_code}, {count} tools")

        r = post(base, {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                        "params": {"name": "list_windows", "arguments": {}}})
        ok_overall &= check(r.status_code == 200, "tools/call without a session id",
                            str(r.status_code))

        # -- stale session id (client replayed one from before a restart) ---
        r = post(base, {"jsonrpc": "2.0", "id": 6, "method": "tools/list"},
                 session="deadbeefdeadbeefdeadbeefdeadbeef")
        count = len(((parse(r) or [{}])[0].get("result") or {}).get("tools", []))
        ok_overall &= check(r.status_code == 200 and count > 40,
                            "tools/list with a stale session id", f"{r.status_code}, {count}")

        # -- Accept header variants -----------------------------------------
        r = post(base, {"jsonrpc": "2.0", "id": 7, "method": "tools/list"},
                 session=sess, accept="application/json")
        ok_overall &= check(r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"),
                            "existing session, Accept: application/json only",
                            f"HTTP {r.status_code} {r.headers.get('content-type')}")

        r = post(base, {"jsonrpc": "2.0", "id": 7, "method": "tools/list"},
                 accept="application/json")
        ok_overall &= check(r.status_code == 200,
                            "stateless + json-only tools/list", f"HTTP {r.status_code}")

        # A client that echoes the version the server negotiated must be
        # accepted; a mismatch here breaks every call after initialize.
        s4, init4 = initialize(base)
        negotiated = (init4.get("result") or {}).get("protocolVersion")
        r = httpx.post(f"{base}/mcp", headers={
            "Authorization": f"Bearer {TOKEN}", "Accept": BOTH,
            "Content-Type": "application/json", "Mcp-Session-Id": s4,
            "MCP-Protocol-Version": str(negotiated),
        }, json={"jsonrpc": "2.0", "id": 14, "method": "tools/list"}, timeout=60)
        ok_overall &= check(r.status_code == 200,
                            f"tools/list echoing MCP-Protocol-Version {negotiated}",
                            str(r.status_code))

        # -- protocol version negotiation -----------------------------------
        for version in ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"):
            s2, msg = initialize(base, version)
            got = (msg.get("result") or {}).get("protocolVersion")
            ok_overall &= check(bool(s2) and bool(got),
                                f"initialize with protocolVersion {version}", f"-> {got}")

        # -- errors, not crashes --------------------------------------------
        r = post(base, {"jsonrpc": "2.0", "id": 8, "method": "tools/call",
                        "params": {"name": "no_such_tool", "arguments": {}}}, session=sess)
        body = (parse(r) or [{}])[0]
        ok_overall &= check(r.status_code in (200, 400) and ("error" in body or
                            body.get("result", {}).get("isError")),
                            "unknown tool returns a JSON-RPC error", json.dumps(body)[:70])

        r = post(base, {"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                        "params": {"name": "click_element", "arguments": {}}}, session=sess)
        body = (parse(r) or [{}])[0]
        ok_overall &= check(r.status_code in (200, 400) and ("error" in body or
                            body.get("result", {}).get("isError") is not None),
                            "missing arguments return an error, not a crash")

        r = post(base, {"jsonrpc": "2.0", "id": 10, "method": "tools/list"}, auth=False)
        ok_overall &= check(r.status_code == 401, "no token -> 401", str(r.status_code))
        r = post(base, {"jsonrpc": "2.0", "id": 11, "method": "tools/list"}, auth="wrong")
        ok_overall &= check(r.status_code == 401, "wrong token -> 401", str(r.status_code))

        # -- concurrency in one session -------------------------------------
        def one(i: int) -> int:
            rr = post(base, {"jsonrpc": "2.0", "id": 100 + i, "method": "tools/call",
                             "params": {"name": "list_windows", "arguments": {}}}, session=sess)
            return rr.status_code
        with ThreadPoolExecutor(max_workers=6) as ex:
            codes = list(ex.map(one, range(6)))
        ok_overall &= check(codes == [200] * 6, "6 concurrent calls on one session", str(codes))

        # -- two agents at once, separate sessions --------------------------
        def other(i: int) -> int:
            s3, _ = initialize(base)
            return post(base, {"jsonrpc": "2.0", "id": 200 + i, "method": "tools/list"},
                        session=s3).status_code
        with ThreadPoolExecutor(max_workers=3) as ex:
            codes = list(ex.map(other, range(3)))
        ok_overall &= check(codes == [200] * 3, "3 clients initializing at once", str(codes))

        # -- transport edge cases clients try ------------------------------
        r = httpx.get(f"{base}/mcp", headers={"Authorization": f"Bearer {TOKEN}",
                                              "Accept": "text/event-stream"}, timeout=3)
        ok_overall &= check(r.status_code == 405,
                            "sessionless GET probe -> 405 (not 400)", str(r.status_code))

        # Regression: a GET for a session id nobody owns used to create a
        # transport and hold the session-creation lock for the life of the SSE
        # stream, so every later POST waited forever (the daemon looked "half
        # alive": GET answered, POSTs never arrived). The id must be one the
        # daemon has never adopted -- an earlier check replays "deadbeef...",
        # which makes that id live, and a live-session GET is meant to stream.
        r = httpx.get(f"{base}/mcp", headers={"Authorization": f"Bearer {TOKEN}",
                                              "Accept": "text/event-stream",
                                              "Mcp-Session-Id": "0" * 32}, timeout=3)
        ok_overall &= check(r.status_code == 405,
                            "GET with an unknown session id -> 405", str(r.status_code))
        r = post(base, {"jsonrpc": "2.0", "id": 11, "method": "tools/list",
                        "params": {}})
        ok_overall &= check(r.status_code == 200,
                            "POST after that GET still answers (no wedged lock)",
                            str(r.status_code))

        r = post(base, {"jsonrpc": "2.0", "id": 12, "method": "tools/call",
                        "params": {"name": "screen_state", "arguments": {}}}, session=sess)
        size = len(r.text)
        ok_overall &= check(r.status_code == 200 and 0 < size < 400_000,
                            "large payload (screen_state) survives the transport",
                            f"{size} bytes")

        r = httpx.request("DELETE", f"{base}/mcp", headers={
            "Authorization": f"Bearer {TOKEN}", "Mcp-Session-Id": sess}, timeout=10)
        ok_overall &= check(r.status_code in (200, 202, 204, 400),
                            "DELETE session is answered", str(r.status_code))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    failed = [r for r in results if not r[0]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
