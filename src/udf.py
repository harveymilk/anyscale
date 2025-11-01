from typing import List, Dict, Any
import pandas as pd
import numpy as np
from io import BytesIO
import imageio.v3 as iio

import torch
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import tempfile, os, cv2, numpy as np

class FrameSummarizer:
    """
    Ray Data class UDF, constructed once per actor.

    Expected input columns per batch:
      - 'path': str
      - 'bytes': bytes (entire video)
      - 'start_sec': float
      - 'end_sec': float
      - 'frame_every_sec': float

    Output:
      - 'summary': str
    """
    def __init__(self, topk: int = 3, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights).to(self.device).eval()
        self.preprocess = weights.transforms()
        self.categories = weights.meta.get("categories", None)
        self.topk = topk

    # ------------------------------------------------------------------
    # Sampling + summarization
    # ------------------------------------------------------------------

    def _sample_frames(self, video_bytes: bytes,
                    start_sec: float, end_sec: float,
                    every_sec: float, max_frames: int = 16) -> List[np.ndarray]:
        if not video_bytes:
            return []

        # Write bytes to a temp file (reliable for OpenCV)
        fd, tmp = tempfile.mkstemp(suffix=".mp4")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(video_bytes)

            cap = cv2.VideoCapture(tmp)
            if not cap.isOpened():
                cap.release()
                return []

            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            if fps <= 0:
                fps = 30.0

            start_idx = max(0, int(start_sec * fps))
            end_idx = int(end_sec * fps) if end_sec > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            stride = max(int(every_sec * fps), 1)

            frames: List[np.ndarray] = []
            # Jump directly to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

            idx = start_idx
            while idx < end_idx:
                # Grab current frame
                ok, frame = cap.read()
                if not ok:
                    break
                # OpenCV gives BGR; convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                if len(frames) >= max_frames:
                    break

                # Jump ahead by stride
                idx += stride
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

            cap.release()
            return frames

        except Exception as e:
            print(f"[WARN] OpenCV sampling failed: {e}")
            return []
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass
            
    def _summarize_frames(self, frames: List[np.ndarray]) -> str:
        if not frames:
            return ""
        imgs = [self.preprocess(Image.fromarray(f)) for f in frames]
        x = torch.stack(imgs).to(self.device)
        with torch.inference_mode():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1).mean(0)
        top_probs, top_ids = torch.topk(probs, k=min(self.topk, probs.shape[-1]))
        labels = [
            self.categories[idx] if self.categories else str(int(idx))
            for idx in top_ids.tolist()
        ]
        return ", ".join(labels)

    # ------------------------------------------------------------------

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        summaries: List[str] = []
        for _, r in batch.iterrows():
            frames = self._sample_frames(
                r["bytes"], r["start_sec"], r["end_sec"], r["frame_every_sec"]
            )
            summaries.append(self._summarize_frames(frames))
        return pd.DataFrame({
            "path": batch["path"],
            "start_sec": batch["start_sec"],
            "end_sec": batch["end_sec"],
            "summary": summaries,
        })