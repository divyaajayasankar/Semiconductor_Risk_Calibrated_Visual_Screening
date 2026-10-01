"""Deterministic imaging-drift perturbations applied to images in [0, 1]."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def apply_perturbation(images: np.ndarray, cond: dict, indices: np.ndarray | None = None,
                       seed: int = 42) -> np.ndarray:
    t = cond.get("type", "none")
    v = cond.get("value")
    x = images.astype(np.float32, copy=True)
    if t == "none":
        return x
    if t == "brightness":
        x = x + v
    elif t == "contrast":
        mean = x.mean(axis=(1, 2), keepdims=True)
        x = (x - mean) * v + mean
    elif t == "gamma":
        x = np.power(np.clip(x, 0, 1), v)
    elif t == "noise":
        idx = np.arange(len(x)) if indices is None else indices
        for j, i in enumerate(idx):  # per-image deterministic noise
            rng = np.random.default_rng(seed * 100003 + int(i))
            x[j] = x[j] + rng.normal(0, v, x[j].shape).astype(np.float32)
    elif t == "blur":
        for j in range(len(x)):
            x[j] = ndimage.gaussian_filter(x[j], sigma=v, mode="reflect")
    else:
        raise ValueError(f"unknown perturbation type {t}")
    return np.clip(x, 0.0, 1.0)


def make_perturb(cond: dict, seed: int = 42, offset: int = 0):
    if cond.get("type", "none") == "none":
        return None
    return lambda imgs, idx: apply_perturbation(imgs, cond, idx + offset, seed)
