"""Tiny SEM-like SYNTHETIC fixture used ONLY by the automated tests / smoke runs.

It mimics the Carinthia-S file layout (images/, masks/, carinthia-s.csv) so the pipeline can be
exercised without the real dataset. Results computed on this fixture are meaningless and are
never reported as research results.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image


def _background(rng: np.random.Generator, size: int) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size] / size
    f = rng.uniform(8, 14)
    img = 0.45 + 0.12 * np.sin(2 * np.pi * f * x + rng.uniform(0, 6)) * np.sin(2 * np.pi * f * y)
    img += rng.normal(0, 0.03, (size, size))
    return img


def make_fixture(root: Path, n_ref: int = 60, n_anom: int = 120, size: int = 96, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    d = root / "data" / "raw" / "carinthia_s" / "data"
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_ref + n_anom):
        img = _background(rng, size)
        mask = np.zeros((size, size), np.uint8)
        anomalous = i >= n_ref
        if anomalous:
            cy, cx = rng.integers(size // 5, 4 * size // 5, 2)
            ry, rx = rng.integers(size // 14, size // 6, 2)
            yy, xx = np.ogrid[:size, :size]
            blob = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
            img[blob] = rng.choice([0.95, 0.05])
            mask[blob] = 255
        name = f"syn_{i:04d}"
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "L").save(d / "images" / f"{name}.jpg", quality=95)
        Image.fromarray(mask, "L").save(d / "masks" / f"{name}.png")
        label = 6 if not anomalous else int(rng.choice([1, 2, 3, 4, 5], p=[0.1, 0.05, 0.7, 0.1, 0.05]))
        rows.append({"image_path": f"data/images/{name}.jpg", "mask_path": f"data/masks/{name}.png",
                     "filename": name, "label": label})
    pd.DataFrame(rows).to_csv(d / "carinthia-s.csv", index=False)
    return d


def make_config(root: Path, base_config: Path, image_size: int = 64, epochs: int = 3) -> Path:
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cfg["data"]["image_size"] = image_size
    cfg["training"].update(epochs=epochs, patience=2, batch_size=8, device="cpu")
    cfg["phase2"]["smoothing_sigma"] = 1.0
    cfg["localization"]["pixel_auc_bins"] = 1000
    cfg["drift"]["conditions"] = cfg["drift"]["conditions"][:3]
    path = root / "experiment_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path
