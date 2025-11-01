import io
import os
import tempfile
from typing import Dict, Iterable, List, Tuple

import numpy as np
import cv2


def _write_tempfile(data: bytes, suffix: str = ".mp4") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _read_video_meta(path: str) -> Tuple[int, float, int]:
    """Return (num_frames, fps, duration_seconds_int)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return 0, 0.0, 0

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if fps > 0 else 0
    duration = int(num_frames / fps) if fps > 0 and num_frames > 0 else 0
    cap.release()
    return num_frames, fps, duration


def explode_into_clips(
    row: Dict,
    clip_seconds: int = 30,
    frame_every_sec: float = 5.0,
) -> Iterable[Dict]:
    """
    Ray Data UDF: from (path, bytes) -> rows of clips (no frames).

    Input row:
      - 'path': str
      - 'bytes': bytes

    Output row:
      - 'path': str
      - 'bytes': bytes
      - 'start_sec': float
      - 'end_sec': float
      - 'frame_every_sec': float
    """
    path_hint: str = row.get("path", "unknown.mp4")
    data: bytes = row["bytes"]

    tmp = _write_tempfile(data, suffix=os.path.splitext(path_hint)[-1] or ".mp4")

    try:
        num_frames, fps, duration = _read_video_meta(tmp)
        if fps <= 0 or num_frames <= 0 or duration <= 0:
            # fallback: treat as single pseudo-clip
            yield {
                "path": path_hint,
                "bytes": data,
                "start_sec": 0.0,
                "end_sec": float(clip_seconds),
                "frame_every_sec": frame_every_sec,
            }
            return

        start = 0.0
        while start < duration:
            end = min(start + clip_seconds, duration)
            yield {
                "path": path_hint,
                "bytes": data,
                "start_sec": float(start),
                "end_sec": float(end),
                "frame_every_sec": frame_every_sec,
            }
            start += clip_seconds

    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass