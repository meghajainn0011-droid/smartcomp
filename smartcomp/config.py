"""Process-wide configuration for the SmartComp perception daemon."""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "smartcomp"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

# The logical frame the model reasons about. Everything the MCP client sees is
# normalized against this, so a 4K screen and a 1024x768 screen produce the same
# coordinate space.
LOGICAL_W = 960
LOGICAL_H = 540

# Tile grid used for the change detector. 16x9 = 144 tiles of 60x60 px at the
# logical resolution; the plan measured this grid at 12.4 ms including capture.
GRID_COLS = 16
GRID_ROWS = 9

# A tile counts as changed when more than this fraction of its pixels moved.
TILE_CHANGED_PIXEL_FRACTION = 0.012
TILE_PIXEL_DELTA = 18  # per-channel 0-255 difference that counts as a move
FULL_FRAME_FRACTION = 0.5  # enough of the screen changed => treat as a new screen
# Changed regions smaller than this (logical px^2) are not worth OCR: a clock
# tick or a cursor blink would otherwise cost a whole tesseract process.
MIN_REGION_PX = 4000

DETECTOR_MODEL = MODEL_DIR / "gpa_gui_detector.onnx"
DETECTOR_INPUT = 640
DETECTOR_CONF = 0.25
DETECTOR_IOU = 0.45
# 3 threads measured fastest for this fixed 640x640 export; 4 threads is slower
# than 3 on a 2-core CPU because the runtime oversubscribes.
DETECTOR_THREADS = int(os.environ.get("SMARTCOMP_DET_THREADS", "3"))

OCR_MIN_CONFIDENCE = 55.0

# Failed rows are rescued by re-OCR at 2x on the inverted image. Those strips
# are stacked into one image and read by a single tesseract process: a busy
# screen fails ~25 rows, and one process per row cost 3.8 s for a handful of
# recovered words, versus ~0.6 s for one composite pass. The cap bounds the
# composite (and tesseract's memory) on a pathological screen.
OCR_STRIP_BATCH = int(os.environ.get("SMARTCOMP_OCR_STRIP_BATCH", "16"))
# White separation between stacked strips, in scaled pixels: enough that word
# boxes never span two strips.
OCR_STRIP_GAP = 12

# OCR jobs and the detector run in a small thread pool. Both are C extensions
# that release the GIL (a tesseract subprocess, an ONNX session run), so a
# delta with several changed regions now overlaps its own OCR work instead of
# paying for it one region at a time. Tesseract is pinned to one thread each
# (OMP_THREAD_LIMIT), so workers are the parallelism knob.
WORKERS = max(1, int(os.environ.get("SMARTCOMP_WORKERS", "3")))

# A crop over this many (scaled) pixels skips the whole-image 2x rescue pass
# on perception scans -- it would cost seconds. read_window overrides the
# budget with the strip-based rescue (see ocr.Ocr._strip_rescue).
OCR_ALT_PASS_MAX_PIXELS = 1_500_000
# Crops are taken at native resolution (14px UI glyphs survive there) so the
# tesseract fallback does not need to upscale; the PP-OCR path resizes itself.
OCR_SCALE = 1.0

MCP_HOST = "127.0.0.1"
# 8791 rather than the usual 8765: that port belongs to the user's browserd here.
MCP_PORT = int(os.environ.get("SMARTCOMP_PORT", "8791"))

# Unattended behaviour is opt-in now: the daemon does NOT inhibit the
# screensaver or change the lid policy by default, so the box locks and
# blanks exactly as the user configured it. Instead, every screen-touching
# tool carries a recovery guard (lock.ensure_screen): when work is
# requested, a locked/blanked screen is woken and released autonomously
# (verified live: xfce4-screensaver hands the screen back to SetActive
# deactivation with no password), the action runs, and the system returns
# to its normal idle behaviour afterwards. SMARTCOMP_AWAKE=1 opts into the
# old always-awake inhibit for long critical tasks.
def awake_default_enabled() -> bool:
    return os.environ.get("SMARTCOMP_AWAKE", "0") not in ("", "0", "false", "no")


# The recovery guard itself: SMARTCOMP_SCREEN_GUARD=0 turns off the
# wake+unlock-before-action behaviour (the box then behaves purely stock
# and tools refuse while locked instead of recovering).
def screen_guard_enabled() -> bool:
    return os.environ.get("SMARTCOMP_SCREEN_GUARD", "1") not in ("0", "false", "no")


def state_dir() -> Path:
    """Directory holding the local token, the SQLite store and logs."""
    base = Path(os.environ.get("SMARTCOMP_STATE_DIR", Path.home() / ".local" / "state" / APP_NAME))
    base.mkdir(parents=True, exist_ok=True)
    return base


def token_path() -> Path:
    return state_dir() / "token"


def read_token() -> str:
    """Local MCP token; created on first use, readable only by the owner."""
    path = token_path()
    if not path.exists():
        import secrets

        path.write_text(secrets.token_urlsafe(32))
        path.chmod(0o600)
    return path.read_text().strip()
