"""Image / mask preprocessing and batched loading."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from PIL import Image


def load_image(path_or_img, size: int) -> np.ndarray:
    """Grayscale SEM image -> float32 array in [0, 1] of shape (size, size)."""
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    img = img.convert("L")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_mask(path, size: int, threshold: int = 128) -> np.ndarray:
    """Binary defect mask (bool) at model resolution. Pixels >= threshold are defect."""
    with Image.open(path) as m:
        arr = (np.asarray(m.convert("L")) >= threshold).astype(np.uint8) * 255
    if arr.shape != (size, size):
        arr = np.asarray(Image.fromarray(arr).resize((size, size), Image.BILINEAR))
    return arr >= 128


def to_tensor(batch: np.ndarray) -> torch.Tensor:
    """(B, H, W) -> (B, 1, H, W) float tensor."""
    return torch.from_numpy(np.ascontiguousarray(batch)).float().unsqueeze(1)


def iterate_batches(records: pd.DataFrame, size: int, batch_size: int,
                    with_masks: bool = False, mask_threshold: int = 128
                    ) -> Iterator[tuple[pd.DataFrame, np.ndarray, np.ndarray | None]]:
    """Yield (records_chunk, images[B,H,W], masks[B,H,W] or None)."""
    for start in range(0, len(records), batch_size):
        chunk = records.iloc[start:start + batch_size]
        imgs = np.stack([load_image(p, size) for p in chunk["image_path"]])
        masks = None
        if with_masks:
            masks = np.stack([
                load_mask(p, size, mask_threshold) if a == 1 else np.zeros((size, size), bool)
                for p, a in zip(chunk["mask_path"], chunk["is_anomaly"])
            ])
        yield chunk, imgs, masks


def load_all_images(records: pd.DataFrame, size: int) -> np.ndarray:
    return np.stack([load_image(p, size) for p in records["image_path"]]).astype(np.float32)


def image_to_uint8(arr: np.ndarray) -> np.ndarray:
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def exists(p) -> bool:
    return Path(p).exists()
