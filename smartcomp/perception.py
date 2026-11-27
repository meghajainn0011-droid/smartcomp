"""The perception tier: capture -> tile diff -> OCR + detector -> compact delta.

The contract with the model is "tell me what changed since you last asked", not
"here is the screen". Every element gets a stable id derived from its kind, box
and text, so an unchanged screen costs one call with empty `add`/`del` lists.

Coordinates are permille of the logical 960x540 frame (0-1000), which keeps the
JSON small and resolution independent. Anything the model would read as an
instruction -- `x` on a text element -- is untrusted content that came off the
screen; it is flagged with `u: 1` so the caller's prompt can fence it.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import config
from .capture import Capturer
from .detector import Detector
from .ocr import Ocr


def _permille(v: float, span: int) -> int:
    return int(round(v / span * 1000))


def _eid(kind: str, box: list[int], text: str) -> str:
    raw = f"{kind}|{box[0]}|{box[1]}|{box[2]}|{box[3]}|{text}".encode()
    return hashlib.blake2b(raw, digest_size=3).hexdigest()


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return ((x2 - x1) * (y2 - y1)) / float(max(1, aw * ah))


# Deleted ids are content hashes the agent can only ever look up, never act
# on, so a window switch sending 200 of them is pure token cost. The delta
# lists the first MAX_DEL_IDS and counts the rest in `del_dropped`.
MAX_DEL_IDS = 8


class Perception:
    def __init__(self, display: str | None = None) -> None:
        self.capturer = Capturer(display=display)
        self.ocr = Ocr()
        self.detector = Detector(
            config.detector_model(),
            input_size=config.DETECTOR_INPUT,
            conf=config.DETECTOR_CONF,
            iou=config.DETECTOR_IOU,
        )
        self.elements: dict[str, dict] = {}
        self.seq = 0
        self.last_stats: dict[str, float] = {}
        # OCR submits a process per changed region and the detector runs an
        # ONNX session; both release the GIL, so running them together turns
        # serial waits into overlap (see config.WORKERS). One worker keeps the
        # old serial behaviour for a single-core box or a constrained run.
        self._pool: ThreadPoolExecutor | None = None

    # -- public -----------------------------------------------------------

    def get_changes(self, force_full: bool = False, verbose_del: bool = False,
                    detect: bool = True) -> dict:
        t_start = time.perf_counter()
        self.seq += 1
        if force_full:
            self.capturer.reset()

        frame = self.capturer.capture()
        diff = self.capturer.diff(frame)

        removed: list[str] = []
        added: list[dict] = []
        ocr_ms = det_ms = 0.0
        rects: list[tuple[int, int, int, int]] = []

        if diff.full:
            removed = list(self.elements)
            rects = [(0, 0, config.LOGICAL_W, config.LOGICAL_H)]
        elif diff.tiles:
            rects = self._cluster(diff.tiles)
            for eid, el in list(self.elements.items()):
                box = self._px_box(el["b"])
                if any(_overlap(box, r) > 0.15 for r in rects):
                    removed.append(eid)

        # A ticking clock changes a couple of tiles and nothing else. Paying
        # ~200 ms of tesseract process start (or ~1 s of detector) for it makes
        # idle polling unusable, so regions below the noise floor are ignored.
        work = [r for r in rects if r[2] * r[3] >= config.MIN_REGION_PX]
        kept: list[tuple[int, int, int, int]] = []
        for rect in work:
            kept.append(rect)

        # Icon detections are filtered down to the changed rects (_icons),
        # so a pass with nothing changed cannot gain icons: running the
        # detector there is a fixed tax on every idle poll, which is what
        # made get_changes cost ~5 s even when the screen stood still.
        # Threads are only worth starting when there is more than one region to
        # spread: a single full-screen region already holds the CPU, and letting
        # a 3-thread ONNX run compete with tesseract measured slower (6.1 s vs
        # 5.4 s for the whole call) on this box.
        parallel = len(kept) > 1
        det_ready = detect and self.detector.available and (diff.full or bool(kept))
        det_future = (
            self._workers().submit(self.detector.detect, frame.bgr)
            if det_ready and parallel
            else None
        )

        if kept:
            # The wall time of the whole phase is what the stats report;
            # summing per-region times would count overlapped work twice.
            t_ocr = time.perf_counter()
            if parallel:
                pool = self._workers()
                futures = [pool.submit(self._recognize, frame.native, rect) for rect in kept]
                for future in futures:
                    added.extend(future.result())
            else:
                added.extend(self._recognize(frame.native, kept[0]))
            ocr_ms = (time.perf_counter() - t_ocr) * 1000.0

        if det_ready:
            # Multi-region calls already had the detector running alongside the
            # OCR jobs; a single-region call runs it now, serially.
            dets = (
                det_future.result()
                if det_future is not None
                else self.detector.detect(frame.bgr)
            )
            det_ms = self.detector.last_ms
            added.extend(self._icons(dets, kept))

        for eid in removed:
            self.elements.pop(eid, None)

        new_items: list[dict] = []
        for el in added:
            eid = el["id"]
            prev = self.elements.get(eid)
            if prev is None:
                self.elements[eid] = el
                new_items.append(el)
            elif prev.get("x") != el.get("x") or prev["b"] != el["b"]:
                self.elements[eid] = el
                new_items.append(el)

        removed = [eid for eid in removed if eid not in self.elements]
        total_ms = (time.perf_counter() - t_start) * 1000.0
        self.last_stats = {
            "cap": round(frame.capture_ms, 1),
            "diff": round(diff.ms, 1),
            "ocr": round(ocr_ms, 1),
            "det": round(det_ms, 1),
            "total": round(total_ms, 1),
            "els": len(self.elements),
            "tiles": len(diff.tiles),
        }

        payload: dict = {
            "seq": self.seq,
            "ts": int(time.time() * 1000),
            "w": config.LOGICAL_W,
            "h": config.LOGICAL_H,
            "full": diff.full,
        }
        if new_items:
            payload["add"] = new_items
        if removed:
            if verbose_del:
                payload["del"] = removed
            else:
                payload["del"] = removed[:MAX_DEL_IDS]
                if len(removed) > MAX_DEL_IDS:
                    payload["del_dropped"] = len(removed) - MAX_DEL_IDS
        payload["stats"] = self.last_stats
        return payload

    def read_region(self, x: int, y: int, w: int, h: int) -> list[dict]:
        """OCR an arbitrary rect of the native screen.

        Whole-screen OCR is fine for what is on top, but a 640x480 window with
        14 px text is where tesseract's full-frame pass starts dropping lines.
        Reading one window's rect directly is both faster and more accurate.
        """
        return self._region(x, y, w, h, stats_key="region_ocr")

    def read_window(self, win: dict) -> list[dict]:
        """OCR exactly one window, grabbing only that rectangle.

        Same text as `read_region` over the window's rect, but the grab asks X
        for the window's pixels rather than the whole screen, so the cost and the
        pixels touched scale with the window instead of the display. This is the
        agent's explicit readback call, so it always gets the OCR rescue pass
        (deep=True) even when the window is too big for the whole-image budget.
        """
        return self._region(win["x"], win["y"], win["w"], win["h"],
                            stats_key="window_ocr", deep=True)

    def _region(self, x: int, y: int, w: int, h: int, stats_key: str,
                deep: bool = False) -> list[dict]:
        native_w, native_h = self.capturer.native_size
        x = max(0, min(int(x), native_w - 1))
        y = max(0, min(int(y), native_h - 1))
        w = max(1, min(int(w), native_w - x))
        h = max(1, min(int(h), native_h - y))
        t0 = time.perf_counter()
        crop = self.capturer.grab_rect(x, y, w, h)
        grab_ms = (time.perf_counter() - t0) * 1000.0
        out: list[dict] = []
        for line in self.ocr.read(crop, deep=deep):
            box = [
                _permille(x + line.x, native_w),
                _permille(y + line.y, native_h),
                _permille(line.w, native_w),
                _permille(line.h, native_h),
            ]
            text = line.text[:160]
            eid = _eid("t", box, text)
            el = {
                "id": eid,
                "k": "t",
                "b": box,
                "x": text,
                "c": round(line.conf / 100.0, 2),
                "u": 1,
            }
            self.elements[eid] = el
            out.append(el)
        self.last_stats = {
            **(self.last_stats or {}),
            stats_key: round(self.ocr.last_ms, 1),
            "grab": round(grab_ms, 1),
        }
        return out

    def state(self) -> dict:
        return {
            "elements": list(self.elements.values()),
            "seq": self.seq,
            "screen": [config.LOGICAL_W, config.LOGICAL_H],
            "stats": self.last_stats,
            "ocr_engine": self.ocr.engine,
            "detector": ("ready" if self.detector.ready else "idle (loads on first use)"
                         if self.detector.available else f"unavailable ({self.detector.error})"),
        }

    # -- internals --------------------------------------------------------

    def _workers(self) -> ThreadPoolExecutor:
        """The OCR/detector pool, created on first use.

        Lazily: tests build a Perception without __init__, and a daemon that
        never makes a perception call should not own three idle threads.
        """
        pool = getattr(self, "_pool", None)
        if pool is None:
            pool = self._pool = ThreadPoolExecutor(
                max_workers=config.WORKERS, thread_name_prefix="smartcomp-perc"
            )
        return pool

    def _px_box(self, permille: list[int]) -> tuple[int, int, int, int]:
        return (
            int(permille[0] / 1000 * config.LOGICAL_W),
            int(permille[1] / 1000 * config.LOGICAL_H),
            max(1, int(permille[2] / 1000 * config.LOGICAL_W)),
            max(1, int(permille[3] / 1000 * config.LOGICAL_H)),
        )

    def _cluster(self, tiles: list[int]) -> list[tuple[int, int, int, int]]:
        """Merge changed tiles into the fewest rectangles that contain them."""
        grid = np.zeros((config.GRID_ROWS, config.GRID_COLS), dtype=np.uint8)
        for i in tiles:
            r, c = divmod(i, config.GRID_COLS)
            grid[r, c] = 1
        n, labels = cv2.connectedComponents(grid, connectivity=4)
        rects: list[tuple[int, int, int, int]] = []
        for label in range(1, n):
            ys, xs = np.where(labels == label)
            x0, y0, tw, th = self.capturer.tile_rect(int(ys.min() * config.GRID_COLS + xs.min()))
            x1, y1, _, _ = self.capturer.tile_rect(int(ys.max() * config.GRID_COLS + xs.max()))
            left, top = max(0, x0 - 4), max(0, y0 - 4)
            right = min(config.LOGICAL_W, x1 + tw + 4)
            bottom = min(config.LOGICAL_H, y1 + th + 4)
            rects.append((left, top, right - left, bottom - top))
        # One huge region is cheaper to OCR in a single pass than in pieces.
        if len(rects) > 1:
            area = sum(w * h for _, _, w, h in rects)
            # One huge region is cheaper to OCR in a single pass than in pieces.
            if area > 0.5 * config.LOGICAL_W * config.LOGICAL_H:
                return [(0, 0, config.LOGICAL_W, config.LOGICAL_H)]
        return rects

    def _recognize(self, native: np.ndarray, rect: tuple[int, int, int, int]) -> list[dict]:
        """OCR a logical-space rect out of the *native* frame.

        Text is recognized at native resolution: halving a 1920x1080 screen
        before OCR destroys 14px UI glyphs, which is what made the first
        downscaled attempt return garbage.
        """
        x, y, w, h = rect
        sx = native.shape[1] / config.LOGICAL_W
        sy = native.shape[0] / config.LOGICAL_H
        nx, ny = int(x * sx), int(y * sy)
        crop = native[ny : ny + int(h * sy), nx : nx + int(w * sx)]
        if crop.size == 0:
            return []
        out: list[dict] = []
        for line in self.ocr.read(crop):
            box = [
                _permille(x + line.x / sx, config.LOGICAL_W),
                _permille(y + line.y / sy, config.LOGICAL_H),
                _permille(line.w / sx, config.LOGICAL_W),
                _permille(line.h / sy, config.LOGICAL_H),
            ]
            text = line.text[:120]
            out.append(
                {
                    "id": _eid("t", box, text),
                    "k": "t",
                    "b": box,
                    "x": text,
                    "c": round(line.conf / 100.0, 2),
                    "u": 1,
                }
            )
        return out

    def _icons(self, dets, rects: list[tuple[int, int, int, int]]) -> list[dict]:
        text_boxes = [self._px_box(el["b"]) for el in self.elements.values() if el["k"] == "t"]
        out: list[dict] = []
        for det in dets:
            box_px = (det.x, det.y, det.w, det.h)
            if not any(_overlap(box_px, r) > 0.3 for r in rects):
                continue
            # A detection sitting on top of recognized text is the text's
            # container; reporting both doubles the element count for no gain.
            if any(_overlap(box_px, tb) > 0.5 for tb in text_boxes):
                continue
            box = [
                _permille(det.x, config.LOGICAL_W),
                _permille(det.y, config.LOGICAL_H),
                _permille(det.w, config.LOGICAL_W),
                _permille(det.h, config.LOGICAL_H),
            ]
            out.append(
                {"id": _eid("i", box, ""), "k": "i", "b": box, "c": round(det.conf, 2)}
            )
        return out
