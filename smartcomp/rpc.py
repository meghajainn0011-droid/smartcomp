"""Minimal MCP JSON-RPC over HTTP for internal callers.

Two things inside smartcomp need to talk to the daemon's own endpoint: the
`doctor` command (is it answering?) and the locked-host fallback (run this tool
on the agent's display). Both use the public surface, over localhost, with the
same bearer token agents use -- so an internal caller cannot drift from what a
real client sees, and no second tool implementation exists to fall out of date.

The MCP SDK client would work but pulls an async stack into the daemon's tool
path for one request. This is the wire format only: POST JSON, read a JSON body
or SSE frames, return the matching id.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


def parse_messages(text: str, content_type: str = "") -> list[dict]:
    """JSON-RPC messages from an SSE or JSON body (tolerant, never raises)."""
    if "text/event-stream" in (content_type or "") or text.lstrip().startswith("event:"):
        out: list[dict] = []
        for block in text.split("\n\n"):
            data = [line[5:].strip() for line in block.splitlines() if line.startswith("data:")]
            if not data:
                continue
            try:
                out.append(json.loads("\n".join(data)))
            except json.JSONDecodeError:
                continue
        return out
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else [parsed]


def call(url: str, token: str, method: str, params: dict | None = None, *,
         timeout: float = 60.0, request_id: int = 1) -> dict:
    """One JSON-RPC request; returns its `result`, raises RuntimeError on error."""
    payload = {"jsonrpc": "2.0", "id": request_id, "method": method,
               "params": params or {}}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            # Both: SSE-aware servers answer normally, JSON-only ones answer
            # with a body instead of 406.
            "Accept": "application/json, text/event-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            ctype = resp.headers.get("content-type", "")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        messages = parse_messages(body, exc.headers.get("content-type", ""))
        detail = ""
        if messages:
            err = messages[-1].get("error") or {}
            detail = str(err.get("message") or messages[-1].get("result") or "")[:200]
        raise RuntimeError(f"HTTP {exc.code}: {detail or body[:200]}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"cannot reach {url}: {exc}") from exc

    for message in reversed(parse_messages(body, ctype)):
        if message.get("id") != request_id:
            continue
        if "error" in message:
            err = message["error"] or {}
            raise RuntimeError(str(err.get("message") or err)[:400])
        return message.get("result") or {}
    raise RuntimeError(f"no JSON-RPC reply (HTTP body: {body[:200]!r})")


def call_tool(url: str, token: str, name: str, arguments: dict | None = None, *,
              timeout: float = 120.0) -> dict:
    """Call one MCP tool and return its JSON payload (structured or text)."""
    result = call(url, token, "tools/call",
                  {"name": name, "arguments": arguments or {}}, timeout=timeout)
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        out = dict(structured)
    else:
        text = ""
        for block in result.get("content") or []:
            if block.get("type") == "text":
                text = block.get("text", "")
                break
        try:
            out = json.loads(text) if text else {}
        except json.JSONDecodeError:
            out = {"text": text}
        if not isinstance(out, dict):
            out = {"result": out}
    if result.get("isError") and out.get("ok") is not False:
        out["ok"] = False
    return out


def list_tools(url: str, token: str, *, timeout: float = 15.0) -> list[dict]:
    """The tool catalogue, via the stateless path (no session is created)."""
    result = call(url, token, "tools/list", {}, timeout=timeout)
    return list(result.get("tools") or [])
