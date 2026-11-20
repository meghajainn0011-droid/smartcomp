"""Vision-engine explainer: Mirage (Cactus) description of a screen + decider.

Mirage (see web research: 22MB 2-bit VLM) describes a scene and reads text.
smartcomp's native path (OCR + GPA detector + colors) is the faster, robust
default for desktop UI. The decider here compares both descriptions and
reports which it deems more accurate — deliberately at describe time, so the
agent can ask "explain_screen" when it wants deeper understanding.
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger("smartcomp.vision")

_MIRRALE = None
_MIRRAGE_ERR = None


def load_mirage():
    """Return a Mirage instance, or None when cactus-compute isn't installed."""
    global _MIRRALE, _MIRRAGE_ERR
    if _MIRRALE is not None:
        return _MIRRALE
    if _MIRRAGE_ERR is not None:
        return None
    try:
        import cactus

        _MIRRALE = cactus.Mirage()
        log.info("Mirage loaded")
        return _MIRRALE
    except Exception as exc:  # not installed or load failed: OCR path only
        _MIRRAGE_ERR = exc
        log.info("Mirage unavailable: %s", exc)
        return None


def describe(img_path: str) -> dict:
    """Mirage description of an image path. {"ok": bool, "lines": [...], ...}"""
    vision = load_mirage()
    if vision is None:
        return {"ok": False, "error": "mirage unavailable"}
    try:
        t0 = time.perf_counter()
        res = vision.describe(img_path)
        lines = []
        for e in (res or {}).get("text", []) if isinstance(res, dict) else []:
            if isinstance(e, dict):
                if e.get("string"):
                    lines.append(e["string"])
            elif str(e).strip():
                lines.append(str(e))
        out = {"ok": True, "lines": lines}
        if isinstance(res, dict):
            for key in ("scene", "relations", "reasoning"):
                if res.get(key):
                    out[key] = res[key]
        out["ms"] = (time.perf_counter() - t0) * 1000.0
        return out
    except Exception as exc:  # per-image failures must not break the OCR path
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _words(text: str) -> set:
    import re

    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 1}


def decide(ocr_text: str, mirage_lines: list[str]) -> dict:
    """Pick the more accurate explanation of a screen/window.

    Word-coverage comparison: the description that repeats more of the other's
    distinguishing words is the better reader (both see the same screen).
    """
    if not mirage_lines:
        return {"decided": "ocr", "why": "mirage produced no text"}
    mirage_all = " ".join(mirage_lines)
    ocr_words = _words(ocr_text)
    mir_words = _words(mirage_all)
    if not mir_words:
        return {"decided": "ocr", "why": "mirage text is empty"}
    overlap = len(ocr_words & mir_words)
    coverage = overlap / max(1, len(mir_words))
    status = "ok" if coverage >= 0.5 else "low"
    return {
        "decided": "mirage" if coverage >= 0.5 else "ocr",
        "why": (f"mirage covers {overlap}/{len(mir_words)} words "
                f"(duplicated={status}, {len(mirage_lines)} lines)"),
    }
