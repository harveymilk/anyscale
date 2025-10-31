# src/udf.py
from typing import List, Dict, Any
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image


class FrameSummarizer:
    """Ray Data class UDF, constructed once per actor.

    Expected input (pandas.DataFrame) columns per batch:
      - 'video_path': str
      - 'clip_index': int
      - 'start_sec': float
      - 'end_sec': float
      - 'frames': list[np.ndarray]  # RGB uint8 per frame

    Output: pandas.DataFrame with same identifiers + 'summary': str
    """
    def __init__(self, topk: int = 3, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights).to(self.device).eval()
        # This transform expects a PIL Image; it includes resize/center-crop/ToTensor/normalize.
        self.preprocess = weights.transforms()
        self.categories = weights.meta.get("categories", None)
        self.topk = topk

    def _summarize_frames(self, frames: List[np.ndarray]) -> str:
        if not frames:
            return ""

        # Convert each numpy RGB frame -> PIL Image -> preprocess -> tensor
        imgs = [self.preprocess(Image.fromarray(f)) for f in frames]
        x = torch.stack(imgs, dim=0).to(self.device, non_blocking=True)

        with torch.inference_mode():
            logits = self.model(x)                           # [N, 1000]
            probs = F.softmax(logits, dim=-1).mean(dim=0)   # average over frames

        top_probs, top_ids = torch.topk(probs, k=min(self.topk, probs.shape[-1]))
        labels = [
            self.categories[idx] if self.categories else str(int(idx))
            for idx in top_ids.tolist()
        ]
        return ", ".join(labels)

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        # Defensive: if Ray hands us a dict of columns (shouldn't with pandas format), convert.
        if not isinstance(batch, pd.DataFrame):
            batch = pd.DataFrame(batch)

        summaries: List[str] = []
        frames_col = batch["frames"].tolist()
        for frames in frames_col:
            # Some decoders may yield None; normalize to list
            frames = frames or []
            summaries.append(self._summarize_frames(frames))

        out = batch.copy()
        out["summary"] = summaries
        return out
