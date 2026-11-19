"""Single-class icon/button detector.

Weights: salesforce/GPA-GUI-Detector (MIT), exported to ONNX. The MIT export was
chosen over OmniParser V2 because those weights are AGPL-2.0, which the design
document flags as a licensing blocker before shipping. Only the ONNX file is
needed at runtime, so no torch/ultralytics dependency is pulled in.
"""

from __future__ import annotations

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
    def __init__(self, model_path: Path, input_size: int = 640, conf: float = 0.25,
                 iou: float = 0.45) -> None:
        self.path = Path(model_path)
        self.input_size = input_size
        self.conf = conf
        self.iou = iou
        self.last_ms = 0.0
        self._session = None
        self._input_name = ""
        self.error: str | None = None
        if not self.path.exists():
            self.error = f"weights missing: {self.path}"
            return
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = config.DETECTOR_THREADS
            opts.inter_op_num_threads = 1
            opts.log_severity_level = 3
            self._session = ort.InferenceSession(
                str(self.path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
        except Exception as exc:  # corrupt download, missing runtime, ...
            self.error = f"{type(exc).__name__}: {exc}"
            self._session = None

    @property
    def ready(self) -> bool:
        return self._session is not None

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if self._session is None:
            return []
        t0 = time.perf_counter()
        blob, scale, pad_x, pad_y = _letterbox(frame_bgr, self.input_size)
        out = self._session.run(None, {self._input_name: blob})[0]
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
