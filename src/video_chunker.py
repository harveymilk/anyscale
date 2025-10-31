# src/video_chunker.py
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
    # Some containers don't report frame count reliably; best-effort.
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if fps > 0 else 0
    duration = int(num_frames / fps) if fps > 0 and num_frames > 0 else 0
    cap.release()
    return num_frames, fps, duration


def _sample_frame_indices(start_sec: float, end_sec: float, fps: float, every_sec: float) -> List[int]:
    if fps <= 0:
        return []
    times = np.arange(start_sec, end_sec, step=max(every_sec, 1e-6))
    idxs = (times * fps).astype(int)
    return sorted(list(set(map(int, idxs))))


def _read_frame_bgr_at(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    """Seek to frame index and return BGR frame or None."""
    # OpenCV seek: set frame position then read.
    ok = cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
    if not ok:
        return None
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def explode_into_clips(row: Dict, clip_seconds: int = 30, frame_every_sec: float = 5.0) -> Iterable[Dict]:
    """Ray Data UDF: from (path, bytes) -> rows of clips with sampled frames.

    Input row:
      - 'path': str      (from read_binary_files(..., include_paths=True))
      - 'bytes': bytes

    Output row:
      - 'video_path': str
      - 'clip_index': int
      - 'start_sec': float
      - 'end_sec': float
      - 'frames': List[np.ndarray] (RGB uint8)
    """
    path_hint: str = row.get("path", "unknown.mp4")
    data: bytes = row["bytes"]

    # Write to a temp file because OpenCV VideoCapture expects a file path.
    tmp = _write_tempfile(data, suffix=os.path.splitext(path_hint)[-1] or ".mp4")

    try:
        num_frames, fps, _ = _read_video_meta(tmp)
        cap = cv2.VideoCapture(tmp)

        if not cap.isOpened() or fps <= 0 or num_frames <= 0:
            # Fallback: try to interpret bytes as a single image
            try:
                img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    yield {
                        "video_path": path_hint,
                        "clip_index": 0,
                        "start_sec": 0.0,
                        "end_sec": float(clip_seconds),
                        "frames": [rgb],
                    }
                    return
            except Exception:
                pass
            # Could not decode; return an empty-frames pseudo-clip
            yield {
                "video_path": path_hint,
                "clip_index": 0,
                "start_sec": 0.0,
                "end_sec": float(clip_seconds),
                "frames": [],
            }
            return

        duration = num_frames / fps

        clip_idx = 0
        start = 0.0
        while start < duration:
            end = min(start + clip_seconds, duration)
            frame_indices = _sample_frame_indices(start, end, fps, frame_every_sec)

            frames: List[np.ndarray] = []
            for idx in frame_indices:
                idx = int(min(max(idx, 0), max(num_frames - 1, 0)))
                bgr = _read_frame_bgr_at(cap, idx)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(rgb)

            yield {
                "video_path": path_hint,
                "clip_index": clip_idx,
                "start_sec": float(start),
                "end_sec": float(end),
                "frames": frames,
            }

            clip_idx += 1
            start += clip_seconds

    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            os.remove(tmp)
        except Exception:
            pass
