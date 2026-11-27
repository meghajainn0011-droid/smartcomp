"""Single-class icon/button detector.

Weights: salesforce/GPA-GUI-Detector (MIT), exported to ONNX. The MIT export was
chosen over OmniParser V2 because those weights are AGPL-2.0, which the design
document flags as a licensing blocker before shipping. Only the ONNX file is
needed at runtime, so no torch/ultralytics dependency is pulled in.
"""

from __future__ import annotations

import ctypes
import gc
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config


@dataclass
class Detection:
    x: int  # pixel coords in the frame that was passed in
    y: int
    w: int
    h: int
    conf: float


class Detector:
    """Icon/button detector, loaded on first use and released when idle.

    The fp32 weights are 80 MB of Conv filters; ONNX Runtime turns that into
    ~200 MB of RSS, which is most of a daemon that must stay small on this
    8 GB box. The shipped int8 build is 21 MB and 2.3x faster, but a loaded
    session still costs ~90 MB plus a transient allocation on the first run.
    So the session is lazy and is dropped after `DETECTOR_IDLE_S` without a
    detection: idle daemons stay small, and an agent that only needs text
    never pays for the icon model at all.
    """

    def __init__(self, model_path: Path, input_size: int = 640, conf: float = 0.25,
                 iou: float = 0.45, idle_s: float | None = None) -> None:
        self.path = Path(model_path)
        self.input_size = input_size
        self.conf = conf
        self.iou = iou
        self.idle_s = config.DETECTOR_IDLE_S if idle_s is None else idle_s
        self.last_ms = 0.0
        self._session = None
        self._input_name = ""
        self._lock = threading.RLock()
        self._last_use = 0.0
        self._watch: threading.Thread | None = None
        self.error: str | None = None
        if not config.detector_enabled():
            self.error = "disabled (SMARTCOMP_DETECTOR=0)"
        elif not self.path.exists():
            self.error = f"weights missing: {self.path}"

    @property
    def available(self) -> bool:
        """Whether a detection could run here (file present, not disabled)."""
        return self.error is None

    @property
    def ready(self) -> bool:
        """Whether the session is currently loaded (False while idle)."""
        return self._session is not None

    def ensure(self) -> bool:
        if self._session is not None:
            return True
        if not self.available:
            return False
        with self._lock:
            if self._session is not None:
                return True
            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = config.DETECTOR_THREADS
                opts.inter_op_num_threads = 1
                opts.log_severity_level = 3
                # The arena reserves activation memory for the life of the
                # session and measured 444 MB RSS on this model; without it the
                # same session settles at 155 MB and reuses buffers per run.
                opts.enable_cpu_mem_arena = False
                opts.enable_mem_pattern = False
                self._session = ort.InferenceSession(
                    str(self.path), sess_options=opts, providers=["CPUExecutionProvider"]
                )
                self._input_name = self._session.get_inputs()[0].name
            except Exception as exc:  # corrupt download, missing runtime, ...
                self.error = f"{type(exc).__name__}: {exc}"
                self._session = None
                return False
            self._last_use = time.time()
            self._start_watch()
            return True

    def _start_watch(self) -> None:
        if self.idle_s <= 0:
            return
        if self._watch is not None and self._watch.is_alive():
            return
        self._watch = threading.Thread(target=self._idle_loop, daemon=True,
                                       name="smartcomp-detector-idle")
        self._watch.start()

    def _idle_loop(self) -> None:
        while True:
            time.sleep(min(15.0, max(1.0, self.idle_s / 4)))
            with self._lock:
                if self._session is None:
                    return
                if time.time() - self._last_use < self.idle_s:
                    continue
                self._release()
                return

    def _release(self) -> None:
        self._session = None
        self._input_name = ""
        gc.collect()
        try:  # hand the freed pages back to the OS, not just to glibc
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass

    def unload(self) -> None:
        with self._lock:
            self._release()

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if not self.ensure():
            return []
        t0 = time.perf_counter()
        blob, scale, pad_x, pad_y = _letterbox(frame_bgr, self.input_size)
        with self._lock:  # never unload a session mid-inference
            if self._session is None:
                return []
            out = self._session.run(None, {self._input_name: blob})[0]
            self._last_use = time.time()
        boxes = self._decode(np.asarray(out), scale, pad_x, pad_y, frame_bgr.shape[:2])
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return boxes

    def _decode(self, out: np.ndarray, scale: float, pad_x: float, pad_y: float,
                frame_hw: tuple[int, int]) -> list[Detection]:
        pred = out[0] if out.ndim == 3 else out
        if pred.ndim != 2:
            return []
        # Exports differ: (N, 5+classes) rows, or transposed (5+classes, N).
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T
        cols = pred.shape[1]
        if cols < 5:
            return []
        boxes = pred[:, :4]
        # Single class -> one score column; multi-class -> take the max.
        scores = pred[:, 4] if cols == 5 else pred[:, 4:].max(axis=1)
        keep = scores >= self.conf
        if not np.any(keep):
            return []
        boxes, scores = boxes[keep], scores[keep]

        cx, cy, w, h = boxes.T
        x = (cx - w / 2 - pad_x) / scale
        y = (cy - h / 2 - pad_y) / scale
        w = w / scale
        h = h / scale

        h_img, w_img = frame_hw
        x = np.clip(x, 0, w_img)
        y = np.clip(y, 0, h_img)
        w = np.clip(w, 0, w_img - x)
        h = np.clip(h, 0, h_img - y)

        rects = [[float(a), float(b), float(c), float(d)] for a, b, c, d in zip(x, y, w, h)]
        idx = cv2.dnn.NMSBoxes(rects, scores.astype(np.float32).tolist(), self.conf, self.iou)
        if isinstance(idx, tuple):  # some OpenCV builds return (indices, scores)
            idx = idx[0]
        out_boxes: list[Detection] = []
        for i in np.array(idx, dtype=np.int64).reshape(-1):
            a, b, c, d = rects[int(i)]
            if c < 6 or d < 6:  # sub-6px blobs are noise at this resolution
                continue
            out_boxes.append(
                Detection(int(a), int(b), int(c), int(d), float(scores[int(i)]))
            )
        return out_boxes


def _letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - nw) / 2, (size - nh) / 2
    top, left = int(round(pad_y)), int(round(pad_x))
    canvas[top : top + nh, left : left + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(rgb, (2, 0, 1))[None]
    return np.ascontiguousarray(blob), scale, pad_x, pad_y
