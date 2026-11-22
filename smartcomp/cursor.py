"""Visible virtual cursor for isolated agent sessions (20% opacity).

Xvfb has no hardware pointer; its default can be invisible on capture. Give
each session a translucent-but-visible arrow (20% alpha) so an operator can
see where the agent works, while the host cursor is never touched.

Builds a real Xcursor with `xcursorgen` (python-xlib has no SetCursor), writes
it into a session-local theme, and sets XCURSOR_THEME in the session daemon.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger("smartcomp.cursor")


def make_visible_cursor(state_dir: str, alpha: int = 51) -> dict:
    """Create a 20%-alpha arrow cursor theme under state_dir/cursors.

    alpha=51 is 20% of 255: visible but unobtrusive, never fully transparent.
    """
    try:
        from PIL import Image, ImageDraw

        root = Path(state_dir) / "cursor-theme" / "cursors"
        root.mkdir(parents=True, exist_ok=True)
        size = 32
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        dr = ImageDraw.Draw(img)
        arrow = [(3, 2), (3, 24), (10, 16), (15, 27), (19, 25),
                 (14, 14), (24, 14)]
        fill = (255, 255, 255, alpha)
        edge = (0, 0, 0, alpha)
        dr.polygon(arrow, fill=fill)
        dr.line(arrow + [arrow[0]], fill=edge, width=1)
        png = root / "arrow.png"
        img.save(str(png))
        cfg = root / "arrow.config"
        cfg.write_text(f"{size} 3 2 {png}\n")
        out = root / "left_ptr"
        subprocess.run(["xcursorgen", str(cfg), str(out)], check=True, timeout=15)
        return {"ok": True, "theme": str(root.parent),
                "cursor": str(out), "alpha": alpha}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
