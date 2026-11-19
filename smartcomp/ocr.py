"""Text recognition for screen regions.

Tesseract 5 is the engine this box can actually run today: its wheel-free binary
is installed and works offline in ~100 ms for a region. RapidOCR (the PP-OCR
ONNX pipeline the design document asks for) is used automatically when it is
importable -- it was blocked here by a 27 MB download that never finished, so the
tesseract path is what ships. Both return the same structure, so swapping the
engine later is a one-line change.
"""

from __future__ import annotations

import csv
import io
import os
import statistics
import subprocess
import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass
class TextLine:
    x: int  # pixel coords inside the crop that was recognized
    y: int
    w: int
    h: int
    text: str
    conf: float  # 0-100


def _rapidocr():
    try:
        from rapidocr import RapidOCR  # type: ignore

        return RapidOCR(
            params={
                "Global.use_cls": False,  # upright UI text needs no angle model
                "Global.log_level": "error",
                "EngineConfig.onnxruntime.intra_op_num_threads": 2,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            }
        )
    except Exception:
        return None


class Ocr:
    """Engine selection: tesseract by default, RapidOCR when asked for.

    Both were measured on this box (see README): tesseract does a 700x300
    region of dense text in ~0.9 s, RapidOCR needs ~3.9 s for a 400x120 crop.
    The design document's 82 ms RapidOCR figure does not reproduce on this
    hardware, so the faster engine is the default and SMARTCOMP_OCR chooses.
    """

    def __init__(self, engine: str | None = None) -> None:
        chosen = (engine or os.environ.get("SMARTCOMP_OCR") or "tesseract").lower()
        self._rapid = None
        if chosen == "rapidocr":
            self._rapid = _rapidocr()
            chosen = "rapidocr" if self._rapid is not None else "tesseract"
        self.engine = chosen if chosen != "auto" else "tesseract"
        self.last_ms = 0.0

    def available(self) -> bool:
        return self.engine == "rapidocr" or _tesseract_ok()

    def read(self, crop_bgr: np.ndarray, deep: bool = False) -> list[TextLine]:
        t0 = time.perf_counter()
        if self.engine == "rapidocr":
            lines = self._read_rapid(crop_bgr)
        else:
            lines = self._read_tesseract(crop_bgr, deep=deep)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return lines

    def _read_rapid(self, crop_bgr: np.ndarray) -> list[TextLine]:
        result = self._rapid(crop_bgr)  # type: ignore[misc]
        out: list[TextLine] = []
        for box, text, conf in _rapid_items(result):
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            out.append(
                TextLine(
                    x=int(min(xs)),
                    y=int(min(ys)),
                    w=int(max(xs) - min(xs)),
                    h=int(max(ys) - min(ys)),
                    text=str(text).strip(),
                    conf=float(conf) * 100.0,
                )
            )
        return [line for line in out if line.text and line.conf >= config.OCR_MIN_CONFIDENCE]

    def _read_tesseract(self, crop_bgr: np.ndarray, deep: bool = False) -> list[TextLine]:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if config.OCR_SCALE != 1.0:
            gray = cv2.resize(
                gray, None, fx=config.OCR_SCALE, fy=config.OCR_SCALE,
                interpolation=cv2.INTER_CUBIC,
            )
        lines = self._tesseract_words(gray, config.OCR_SCALE)
        gate = config.OCR_MIN_CONFIDENCE
        passed = [line for line in lines if line.conf >= gate]

        # Second opinion for readback-style crops. Tesseract's per-word
        # confidence on this box wobbles with the preprocessing path: the same
        # editor line scored mean 48.9 (dropped) in raw grey and 60-96 (kept)
        # after 2x + Otsu on the inverted image. A line that fails the gate in
        # one path but clears it in the other is real text; a row that fails
        # in both is the toolbar noise the gate exists for.
        #
        # The budget: a whole-image 2x rescue is only cheap on small crops.
        # Rescue when the main pass has failed lines, at any window size: a
        # failed line means the main pass read badly there, and reading it
        # differently is the whole point of the rescue. Small crops get the
        # whole-image 2x+Otsu alt pass; over-budget crops (maximized
        # Firefox pages) get the strip-based rescue, which costs a fraction
        # of a full 2x pass. Measured on synthetic dense text (8 px):
        # alt/strip rescue never changed CER vs the main pass (0.514) -- the
        # gain is for busy pages the main pass reads poorly, not for the
        # clean cases the plain pass already handles.
        failed = [line for line in lines if line.conf < gate]
        if not failed:
            return passed
        if gray.shape[0] * gray.shape[1] > config.OCR_ALT_PASS_MAX_PIXELS:
            alt_ok = self._strip_rescue(gray, failed, gate)
        else:
            big = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            _, bw = cv2.threshold(255 - big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            alt = self._tesseract_words(bw, 2.0)
            alt_ok = [a for a in alt if a.conf >= gate]

        def overlaps(a: TextLine, b: TextLine) -> float:
            x1, y1 = max(a.x, b.x), max(a.y, b.y)
            x2 = min(a.x + a.w, b.x + b.w)
            y2 = min(a.y + a.h, b.y + b.h)
            if x2 <= x1 or y2 <= y1:
                return 0.0
            return ((x2 - x1) * (y2 - y1)) / float(max(1, a.w * a.h))

        for a in alt_ok:  # lines only the alternate path found
            if not any(overlaps(a, other) > 0.3 for other in lines):
                passed.append(a)
        for f in failed:
            alts = [a for a in alt_ok if overlaps(a, f) > 0.3]
            if alts:  # rescued by the alternate path
                passed.append(max(alts, key=lambda a: a.conf))
        return passed

    def _strip_rescue(self, gray: np.ndarray, failed: list[TextLine],
                      gate: float) -> list[TextLine]:
        """Re-OCR narrow strips around the failed rows at 2x on the inverted image.

        A maximized window is over the whole-image alt-pass budget, but the
        words that failed sit in a few rows. Cropping each failed row with a
        small pad and running the same 2x + Otsu path on that strip gives the
        identical rescue for a fraction of the cost of upscaling the entire
        window (a 1920x1039 readback re-OCRs a few 2000x40 strips, not a
        3840x2078 image). Lines come back in `gray`'s coordinate space.
        """
        scale = config.OCR_SCALE
        seen: list[TextLine] = []
        for line in failed:
            pad_x, pad_y = 3 * max(line.h, 10), 2 * max(line.h, 10)
            x0 = max(0, int(line.x * scale) - pad_x)
            y0 = max(0, int(line.y * scale) - pad_y)
            x1 = min(gray.shape[1], int((line.x + line.w) * scale) + pad_x)
            y1 = min(gray.shape[0], int((line.y + line.h) * scale) + pad_y)
            strip = gray[y0:y1, x0:x1]
            if strip.size == 0:
                continue
            big = cv2.resize(strip, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            _, bw = cv2.threshold(255 - big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            for a in self._tesseract_words(bw, 2.0):
                if a.conf < gate:
                    continue
                # _cluster_lines already divided by the strip's 2x; undo the
                # strip origin and the pre-scale so coords match `gray`'s
                # crop-space, where the overlap comparison happens.
                a.x = int((a.x + x0) / scale)
                a.y = int((a.y + y0) / scale)
                if any(a.text == s.text and abs(a.x - s.x) < 8 and abs(a.y - s.y) < 8
                       for s in seen):
                    continue  # two failed fragments can share one strip row
                seen.append(a)
        return seen

    @staticmethod
    def _tesseract_words(gray: np.ndarray, scale: float) -> list[TextLine]:
        ok, buf = cv2.imencode(".png", gray)
        if not ok:
            return []
        try:
            # oem 1 = LSTM only; the legacy engine roughly doubles the runtime
            # for no accuracy gain on UI text. OMP_THREAD_LIMIT=1 matters far
            # more: letting tesseract use all 4 threads on this i3 takes the
            # same 700x300 crop from 0.9 s to 13.6 s (measured, see README).
            env = dict(os.environ, OMP_THREAD_LIMIT="1")
            proc = subprocess.run(
                ["tesseract", "-", "-", "--oem", "1", "--psm", "11", "--dpi", "96",
                 "-l", "eng", "tsv"],
                input=buf.tobytes(),
                capture_output=True,
                timeout=20,
                env=env,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []
        if proc.returncode != 0:
            return []
        lines = _parse_tsv(proc.stdout.decode("utf-8", "replace"), scale)
        return lines


def _rapid_items(result):
    """Normalize RapidOCR 1.x/2.x tuples and 3.x output objects."""
    if result is None:
        return []
    boxes = getattr(result, "boxes", None)
    txts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if boxes is not None and txts is not None:
        return list(zip(boxes, txts, scores if scores is not None else [1.0] * len(txts)))
    if isinstance(result, tuple) and len(result) == 2:
        return result[0] or []
    return result or []


def _tesseract_ok() -> bool:
    from shutil import which

    return which("tesseract") is not None


def _parse_tsv(raw: str, scale: float) -> list[TextLine]:
    """Turn tesseract words into lines.

    psm 11 reports sparse text word by word and its own line numbers are
    unreliable on UI screens, so words are clustered geometrically: same row
    (60% vertical overlap) and no more than 1.5 glyph-heights apart. Without
    this the element list is a pile of 7px fragments the model cannot use.
    """
    rows = list(csv.DictReader(io.StringIO(raw), delimiter="\t", quoting=csv.QUOTE_NONE))
    words = []
    for row in rows:
        if row.get("level") != "5":
            continue
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            left, top = int(row["left"]), int(row["top"])
            width, height = int(row["width"]), int(row["height"])
            conf = float(row["conf"])
        except (TypeError, ValueError):
            continue
        if conf < 0:
            continue
        words.append((left, top, width, height, conf, text))
    return _cluster_lines(words, scale)


def _cluster_lines(words, scale: float) -> list[TextLine]:
    """Group words into visual rows, then split each row at wide gaps.

    Row membership is decided by vertical position alone. The earlier version
    matched a word to the first line whose bbox it overlapped *and* was close
    to horizontally, so on a real window "test 42" was pulled onto the stray
    garbage row above it instead of the text row it belongs to, and the readback
    came back as fragments. Vertical grouping cannot make that mistake; wide
    horizontal gaps then keep separate columns (label / value, menu / status)
    from being glued into one line.
    """
    if not words:
        return []
    heights = sorted(w[3] for w in words if w[3] > 0)
    med_h = heights[len(heights) // 2] if heights else 10

    rows: list[list[tuple]] = []
    for word in sorted(words, key=lambda w: (w[1] + w[3] / 2.0, w[0])):
        centre = word[1] + word[3] / 2.0
        best, best_gap = None, None
        for row in rows:
            r_centre = (min(w[1] for w in row) + max(w[1] + w[3] for w in row)) / 2.0
            gap = abs(centre - r_centre)
            if gap <= 0.5 * max(med_h, word[3]) and (best_gap is None or gap < best_gap):
                best, best_gap = row, gap
        if best is None:
            rows.append([word])
        else:
            best.append(word)

    split_gap = max(3.0 * med_h, 30.0)
    lines: list[list[tuple]] = []
    for row in rows:
        row.sort(key=lambda w: w[0])
        current = [row[0]]
        for word in row[1:]:
            prev_right = max(w[0] + w[2] for w in current)
            if word[0] - prev_right > split_gap:
                lines.append(current)
                current = [word]
            else:
                current.append(word)
        lines.append(current)

    out: list[TextLine] = []
    for line in lines:
        line.sort(key=lambda w: w[0])
        left = min(w[0] for w in line)
        top = min(w[1] for w in line)
        right = max(w[0] + w[2] for w in line)
        bottom = max(w[1] + w[3] for w in line)
        confs = [w[4] for w in line]
        text = " ".join(w[5] for w in line)
        # One stray glyph is an icon, not a word; it pollutes the element list.
        if len(text) <= 1 and max(confs) < 80:
            continue
        out.append(
            TextLine(
                x=int(left / scale),
                y=int(top / scale),
                w=int((right - left) / scale),
                h=int((bottom - top) / scale),
                text=text,
                # Median, not mean: tesseract psm 11 sometimes scores the
                # first words of a perfectly readable row conf 0.0 (observed
                # on mousepad's dark theme: "1 replaced" 0.0, "by held
                # modifier" ~96.7). A mean let two broken words drop a real
                # line below the confidence gate; the median of the row's
                # word confidences is what a human would score it.
                conf=float(statistics.median(confs)),
            )
        )
    return out
