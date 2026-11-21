"""Pre-made skills + self-building skill creation.

The daemon ships with a set of ready-to-run workflows (video, audio, tabs,
apps) and a mechanism that turns what actually worked in a session into a
new skill -- so it improves the more it is used. Skills are stored in the
shared Memory DB and replayed by tools.run_skill.
"""

from __future__ import annotations

import time

# Each skill is a list of MCP tool steps. Placeholders (e.g. {url}) are
# filled by run_skill from the arguments it is given.
PREMADE: dict[str, dict] = {
    "watch-video": {
        "description": "Research a YouTube/video URL: read the page, then understand the video (captions + frames).",
        "steps": [
            {"tool": "launch_app", "args": {"name": "firefox"}},
            {"tool": "explain_screen", "args": {}},
            {"tool": "media_understand", "args": {"source": "{url}", "captions": True, "max_frames": 4}},
        ],
    },
    "listen-reel": {
        "description": "Hear and see the reel/video currently playing in the browser live (system audio + screen frames).",
        "steps": [
            {"tool": "media_listen", "args": {"seconds": 8}},
            {"tool": "media_watch", "args": {"seconds": 6, "frames": 4}},
        ],
    },
    "fix-no-audio": {
        "description": "When a video has no voice: check permissions/routes/mute, report the cause, then retry listening. Browsers autoplay muted, so a silent media_listen is usually a tab-mute, not a permission.",
        "steps": [
            {"tool": "media_audio_check", "args": {}},
            {"tool": "media_capabilities", "args": {}},
            {"tool": "media_listen", "args": {"seconds": 6}},
        ],
    },
    "research-tabs": {
        "description": "Inspect the browser tabs/apps and the current page before acting.",
        "steps": [
            {"tool": "app_status", "args": {}},
            {"tool": "list_windows", "args": {}},
            {"tool": "describe_screen", "args": {}},
        ],
    },
    "open-app": {
        "description": "Open an allowlisted app and report the window it created.",
        "steps": [
            {"tool": "launch_app", "args": {"name": "{name}"}},
            {"tool": "app_status", "args": {}},
        ],
    },
    "read-popups": {
        "description": "Check for and report any popup/dialog on screen, with buttons to act on.",
        "steps": [
            {"tool": "popups", "args": {"report": True}},
        ],
    },
    "isolated-agent": {
        "description": "Give a second agent its own cursor/desktop with aligned memory.",
        "steps": [
            {"tool": "session_create", "args": {"name": "{name}"}},
            {"tool": "session_endpoint", "args": {"name": "{name}"}},
            {"tool": "session_align", "args": {}},
        ],
    },
}


def install_premade(memory) -> dict:
    """Install/refresh the built-in skills into a Memory (idempotent)."""
    added = 0
    for name, spec in PREMADE.items():
        full = {"name": f"builtin/{name}", "description": spec["description"],
                "steps": spec["steps"]}
        memory.learn_skill(full["name"], full["steps"], full["description"])
        added += 1
    return {"ok": True, "installed": added, "names": [f"builtin/{n}" for n in PREMADE]}


def auto_skill(memory, name: str, steps: list[dict], description: str = "") -> dict:
    """Create a skill from steps that just worked (self-building loop)."""
    clean = [s for s in steps if isinstance(s, dict) and s.get("tool")]
    if not clean:
        return {"ok": False, "error": "no steps to learn"}
    full_name = name if name.startswith("self/") else f"self/{name}"
    memory.learn_skill(full_name, clean, description or
                       f"self-built from {len(clean)} working steps")
    return {"ok": True, "name": full_name, "steps": len(clean)}


def learn_from_recent(memory, limit: int = 12) -> dict:
    """Turn the most recent successful action sequence into a skill (no
    explicit name needed -- the daemon builds it from what it just did)."""
    rows = memory.db.execute(
        "SELECT tool, args FROM actions WHERE ok = 1 AND tool NOT IN "
        "('get_changes','screen_state','health','stats','list_skills','learn_skill',"
        "'remember','recall','run_skill','session_end','wait_for_change','find') "
        "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    steps = [{"tool": r["tool"], "args": __import__("json").loads(r["args"] or "{}")}
             for r in reversed(rows)]
    if not steps:
        return {"ok": False, "error": "no recent successful actions"}
    return auto_skill(memory, f"auto-{int(time.time())}", steps,
                      "self-built from recent successful actions")
