"""Video/voice understanding for agents, at low RAM cost.

Designed from research (watch-video, yt-mcp, video-vision-mcp, whisper.cpp
memory tables): captions first (zero RAM), then Whisper tiny/base int8 in
30 s chunks (the 220 MB/speech-minute spike from unchunked CTranslate2 is the
known OOM trap -- chunking bounds it), and ffmpeg frame sampling with the
existing Tesseract OCR for burned-in text (reels, songs, captions). One model
loaded at a time; frames processed in a loop and released; temp dir cleaned.

Returns small structured JSON for the LLM -- never images.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

log = logging.getLogger("smartcomp.media")

_WHISPER = None          # loaded lazily, one model for the daemon's life
_HAS_FASTER = None
_HAS_YTDLP = shutil.which("yt-dlp") is not None
_HAS_FFMPEG = shutil.which("ffmpeg") is not None


def has_faster_whisper() -> bool:
    global _HAS_FASTER
    if _HAS_FASTER is None:
        try:
            import faster_whisper  # noqa: F401
            _HAS_FASTER = True
        except Exception:
            _HAS_FASTER = False
    return _HAS_FASTER


def _whisper(model: str):
    global _WHISPER
    if _WHISPER is None:
        from faster_whisper import WhisperModel
        # int8 keeps weights tiny (tiny ~40 MB, base ~75 MB) and RAM ~300-460 MB.
        # cpu_threads capped at 2 for this 2-core box.
        _WHISPER = WhisperModel(model, device="cpu", compute_type="int8",
                                cpu_threads=2, num_workers=1)
    return _WHISPER


def _run(argv: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                          env=dict(os.environ, OMP_THREAD_LIMIT="1"))


def _captions(source: str, work: Path) -> list[dict]:
    """Platform captions via yt-dlp -- zero RAM, preferred path."""
    if not _HAS_YTDLP:
        return []
    out = work / "subs"
    out.mkdir(exist_ok=True)
    _run(["yt-dlp", "--skip-download", "--write-auto-subs", "--write-subs",
          "--sub-langs", "en.*,en", "--sub-format", "vtt",
          "-o", str(out / "%(id)s.%(ext)s"), source], timeout=180)
    vtts = list(out.glob("*.vtt"))
    if not vtts:
        return []
    lines = []
    start = ""
    for line in vtts[0].read_text(errors="replace").splitlines():
        line = line.strip()
        # VTT cue: "00:00:01.500 --> 00:00:03.000"
        if "-->" in line:
            start = line.split("-->")[0].strip().split(".")[0]
            continue
        if not line or line in ("WEBVTT", "NOTE") or line.startswith(("Kind:", "Language:")):
            continue
        text = re.sub(r"<[^>]+>", "", line).strip()
        if not text:
            continue
        if lines and lines[-1]["text"] == text:
            continue  # auto-captions repeat each word as a rolling window
        lines.append({"t": start, "text": text[:200]})
    return lines[:400]


def _audio(source: str, work: Path, duration: int) -> Path | None:
    if not _HAS_FFMPEG:
        return None
    audio = work / "audio.wav"
    cmd = ["ffmpeg", "-y", "-i", source, "-vn", "-ac", "1", "-ar", "16000",
           "-f", "wav", str(audio)]
    if duration > 0:
        cmd[1:1] = ["-t", str(duration)]
    res = _run(cmd, timeout=300)
    return audio if res.returncode == 0 and audio.exists() else None


def _transcribe(audio: Path, model: str) -> list[dict]:
    if not has_faster_whisper() or not audio.exists():
        return []
    engine = _whisper(model)
    out = []
    # faster-whisper 1.2.1's decode_audio() calls av.open(..., metadata_errors=)
    # which newer PyAV removed. Passing a numpy float32 array skips that path
    # entirely: read the 16 kHz mono wav with the stdlib `wave` module.
    import wave
    import numpy as np
    try:
        with wave.open(str(audio), "rb") as wf:
            pcm = wf.readframes(wf.getnframes())
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        return []
    step = 30 * 16000
    for start in range(0, max(len(samples), 1), step):
        chunk = samples[start:start + step]
        if chunk.size < 1600:
            continue
        try:
            segments, _info = engine.transcribe(chunk, beam_size=3,
                                                vad_filter=True)
            for seg in segments:
                out.append({"t": round((start + seg.start * 16000) / 16000, 1),
                            "text": seg.text.strip()[:200]})
        except Exception as exc:
            log.warning("transcribe window %s failed: %s", start, exc)
    return out[:600]


def _frames(source: str, work: Path, count: int, ocr) -> list[dict]:
    """Sample frames, downscale, OCR burned-in text, release each frame."""
    if not _HAS_FFMPEG:
        return []
    frames = work / "frames"
    frames.mkdir(exist_ok=True)
    # even sampling across the whole source; 640px wide keeps OCR RAM low
    cmd = ["ffmpeg", "-y", "-i", source, "-vf",
           f"fps=1/8,scale=640:-1", "-frames:v", str(count),
           str(frames / "f-%03d.jpg")]
    _run(cmd, timeout=300)
    out = []
    for img in sorted(frames.glob("f-*.jpg")):
        import cv2
        import numpy as np
        bgr = cv2.imread(str(img))
        if bgr is None:
            continue
        lines = [ln.text for ln in ocr.read(bgr) if ln.text]
        out.append({"frame": img.stem, "text": lines[:8]})
        del bgr
    return out[:count]


def understand(source: str, model: str = "tiny", max_frames: int = 8,
               duration: int = 180, captions: bool = True,
               voice: bool = True) -> dict:
    """Understand a video URL or local file -> small JSON for the LLM."""
    t0 = time.time()
    if not (_HAS_YTDLP or Path(source).exists()):
        return {"ok": False, "error": "source not reachable and yt-dlp missing"}
    work = Path(tempfile.mkdtemp(prefix="smartcomp-media-"))
    try:
        meta: dict = {}
        local = source
        if source.startswith("http") and _HAS_YTDLP:
            info = _run(["yt-dlp", "--dump-single-json", "--skip-download", source],
                        timeout=120)
            if info.returncode == 0:
                try:
                    d = json.loads(info.stdout)
                    meta = {"title": d.get("title", "")[:120],
                            "duration": d.get("duration"),
                            "uploader": (d.get("uploader") or "")[:80]}
                except Exception:
                    pass
            # ffmpeg cannot read a YouTube page URL: fetch a low-res local
            # copy (worst mp4) so frame sampling and audio extraction work
            # without holding a high-res video in RAM.
            video = work / "video.mp4"
            dl = _run(["yt-dlp", "-f", "worst[ext=mp4]/worst", "-o", str(video),
                       "--no-playlist", source], timeout=300)
            if dl.returncode == 0 and video.exists():
                local = str(video)
                meta["downloaded_bytes"] = video.stat().st_size
        transcript = _captions(source, work) if captions else []
        mode = "captions" if transcript else "offline"
        if not transcript and voice:
            audio = _audio(local, work, duration)
            transcript = _transcribe(audio, model) if audio else []
            mode = "whisper-" + model if transcript else mode
        from .ocr import Ocr
        frames = _frames(local, work, max_frames, Ocr())
        text = " ".join(c["text"] for c in transcript)
        tags = sorted(set(re.findall(r"#\w+", text)))[:12]
        urls = sorted(set(re.findall(r"https?://\S+", text)))[:8]
        return {
            "ok": True, "source": source[:160], "meta": meta,
            "mode": mode, "seconds": round(time.time() - t0, 1),
            "transcript": transcript[:60],
            "frames": frames,
            "hashtags": tags, "urls": urls,
            "note": ("reels/songs often have burned-in text and music with no "
                     "speech: frame text is the main signal when transcript is empty"),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)
