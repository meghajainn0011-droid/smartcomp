"""Screen capture and the tile-level change detector.

Capture and downscale happen in one pass through mss, which on this box costs
~12 ms for a 1920x1080 screen including the 144 tile hashes (see README).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import mss
import numpy as np

from . import config


@dataclass
class Frame:
    gray: np.ndarray  # LOGICAL_H x LOGICAL_W uint8 luma, for the diff
    bgr: np.ndarray  # LOGICAL_H x LOGICAL_W x 3 uint8, for the detector
    native: np.ndarray  # full resolution BGR, for OCR
    source_size: tuple[int, int]  # native screen size, for coordinate mapping
    capture_ms: float


@dataclass
class Diff:
    tiles: list[int] = field(default_factory=list)  # changed tile indices
    full: bool = False  # so much changed that the whole screen is new
    ms: float = 0.0


class Capturer:
    """Grabs the display, downscales it, and diffs consecutive frames."""

    def __init__(self, display: str | None = None) -> None:
        # mss keeps one X connection per instance; the daemon reuses it so each
        # capture is a single XGetImage instead of a connect + grab.
        self._sct = mss.mss(display=display)
        self._prev_gray: np.ndarray | None = None
        mon = self._sct.monitors[1]
        self.native_size = (mon["width"], mon["height"])
        self._cols, self._rows = config.GRID_COLS, config.GRID_ROWS
        self._tw = config.LOGICAL_W // self._cols
        self._th = config.LOGICAL_H // self._rows

    def capture(self) -> Frame:
        t0 = time.perf_counter()
        raw = self._sct.grab(self._sct.monitors[1])
        native = np.asarray(raw, dtype=np.uint8)[:, :, :3].copy()
        if native.shape[1] != config.LOGICAL_W or native.shape[0] != config.LOGICAL_H:
            small = cv2.resize(
                native, (config.LOGICAL_W, config.LOGICAL_H), interpolation=cv2.INTER_AREA
            )
        else:
            small = native
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return Frame(
            gray=gray,
            bgr=small,
            native=native,
            source_size=(raw.width, raw.height),
            capture_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def grab_rect(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Grab one screen rectangle at native resolution, and nothing else.

        Whole-screen grabs cost the same whether the caller wants one window or
        everything, and they drag unrelated windows' pixels through the OCR and
        detection path. mss can ask X for a sub-rectangle, so a window read only
        pays for the window. Clamps to native screen bounds to prevent X11 BadMatch.
        """
        sw, sh = self.native_size
        left = max(0, min(int(x), sw - 1))
        top = max(0, min(int(y), sh - 1))
        width = max(1, min(int(w), sw - left))
        height = max(1, min(int(h), sh - top))
        box = {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }
        try:
            raw = self._sct.grab(box)
            return np.asarray(raw, dtype=np.uint8)[:, :, :3].copy()
        except Exception:
            # Fallback to full capture cropped in software if sub-rect grab fails
            full = self.capture().native
            return full[top:top + height, left:left + width].copy()

    def tile_matrix(self, gray: np.ndarray) -> np.ndarray:
        """Mean absolute difference per tile against the previous frame."""
        if self._prev_gray is None:
            self._prev_gray = gray.copy()
            return np.ones((self._rows, self._cols), dtype=bool)
        delta = cv2.absdiff(gray, self._prev_gray)
        # Per-tile count of pixels that actually moved, as a fraction.
        moved = (delta > config.TILE_PIXEL_DELTA).astype(np.uint8)
        frac = np.empty((self._rows, self._cols), dtype=np.float32)
        for r in range(self._rows):
            y = r * self._th
            band = moved[y : y + self._th]
            for c in range(self._cols):
                x = c * self._tw
                frac[r, c] = band[:, x : x + self._tw].mean()
        self._prev_gray = gray.copy()
        return frac > config.TILE_CHANGED_PIXEL_FRACTION

    def diff(self, frame: Frame) -> Diff:
        t0 = time.perf_counter()
        changed = self.tile_matrix(frame.gray)
        idx = np.flatnonzero(changed.reshape(-1))
        full = changed.mean() >= config.FULL_FRAME_FRACTION
        return Diff(
            tiles=[int(i) for i in idx],
            full=bool(full),
            ms=(time.perf_counter() - t0) * 1000.0,
        )

    def reset(self) -> None:
        """Forget the previous frame so the next diff reports everything."""
        self._prev_gray = None

    def tile_rect(self, index: int) -> tuple[int, int, int, int]:
        """Pixel rect (x, y, w, h) of a tile index in logical coordinates."""
        r, c = divmod(index, self._cols)
        return c * self._tw, r * self._th, self._tw, self._th
