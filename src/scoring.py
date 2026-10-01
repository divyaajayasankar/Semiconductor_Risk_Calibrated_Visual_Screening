"""Reconstruction-error maps, global/local scores, robust normalisation, fusion.

E_i        = (x_i - x_hat_i)^2                       (per-pixel error map)
S_global   = mean(E)                                 (Phase 1 score)
S_local(k) = mean of the top-k fraction of G_sigma * E (Phase 2 local evidence)
Z          = (S - median) / (1.4826 * MAD)           (robust normalisation, VALIDATION refs)
S_fused    = Z_global + lambda * Z_local
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .preprocessing import iterate_batches, to_tensor


def local_col(k: float) -> str:
    return f"local_{k:g}"


# ----------------------------------------------------------- smoothing
def gaussian_kernel(sigma: float) -> torch.Tensor:
    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k1 = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    k1 = k1 / k1.sum()
    return (k1[:, None] * k1[None, :])[None, None]


def smooth_maps(e: torch.Tensor, sigma: float) -> torch.Tensor:
    """Fixed Gaussian smoothing of (B,1,H,W) error maps (reflect padding)."""
    if sigma is None or sigma <= 0:
        return e
    k = gaussian_kernel(sigma).to(e.device)
    r = k.shape[-1] // 2
    return F.conv2d(F.pad(e, (r, r, r, r), mode="reflect"), k)


# ----------------------------------------------------------- scoring
@torch.no_grad()
def error_maps(model, images: np.ndarray, device, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (reconstructions, raw error maps, smoothed error maps) as float32 arrays [B,H,W]."""
    x = to_tensor(images).to(device)
    xh = model(x)
    e = (x - xh) ** 2
    es = smooth_maps(e, sigma)
    return xh[:, 0].cpu().numpy(), e[:, 0].cpu().numpy(), es[:, 0].cpu().numpy()


def topk_mean(flat: np.ndarray, frac: float) -> np.ndarray:
    """Mean of the largest `frac` fraction of values in each row of a [B, N] array."""
    n = flat.shape[1]
    k = max(1, int(round(frac * n)))
    part = np.partition(flat, n - k, axis=1)[:, n - k:]
    return part.mean(axis=1)


def image_scores(raw: np.ndarray, smooth: np.ndarray, topk_grid) -> dict[str, np.ndarray]:
    flat = smooth.reshape(len(smooth), -1)
    out = {"global": raw.reshape(len(raw), -1).mean(axis=1)}
    for k in topk_grid:
        out[local_col(k)] = topk_mean(flat, k)
    return out


def compute_scores(model, records: pd.DataFrame, cfg: dict, device,
                   perturb: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
                   map_callback: Callable | None = None, with_masks: bool = False) -> pd.DataFrame:
    """Score every record. `perturb(images, idx)` applies a drift transform.
    `map_callback(chunk, images, recon, raw, smooth, masks)` receives maps for localisation."""
    size = cfg["data"]["image_size"]
    sigma = cfg["phase2"]["smoothing_sigma"]
    grid = cfg["phase2"]["topk_grid"]
    bs = cfg["training"].get("eval_batch_size", 16)
    rows = []
    offset = 0
    for chunk, imgs, masks in iterate_batches(records, size, bs, with_masks=with_masks,
                                              mask_threshold=cfg["data"]["mask_threshold"]):
        if perturb is not None:
            imgs = perturb(imgs, np.arange(offset, offset + len(imgs)))
        offset += len(imgs)
        recon, raw, sm = error_maps(model, imgs, device, sigma)
        sc = image_scores(raw, sm, grid)
        part = chunk[["filename", "label", "is_anomaly", "role"]].copy()
        for k, v in sc.items():
            part[k] = v
        rows.append(part)
        if map_callback is not None:
            map_callback(chunk, imgs, recon, raw, sm, masks)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ----------------------------------------------------- normalisation
def fit_robust(values: np.ndarray, method: str = "mad") -> dict:
    v = np.asarray(values, dtype=float)
    med = float(np.median(v))
    if method == "iqr":
        q1, q3 = np.percentile(v, [25, 75])
        scale = float((q3 - q1) / 1.349)
    else:
        scale = float(1.4826 * np.median(np.abs(v - med)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(v) or 1e-12)
    return {"method": method, "median": med, "scale": scale, "n": int(len(v))}


def apply_robust(values, params: dict) -> np.ndarray:
    return (np.asarray(values, dtype=float) - params["median"]) / params["scale"]


def fused_score(df: pd.DataFrame, norm: dict, topk: float, lam: float) -> np.ndarray:
    zg = apply_robust(df["global"], norm["global"])
    zl = apply_robust(df[local_col(topk)], norm[local_col(topk)])
    return zg + lam * zl


def fit_normalization(ref_scores: pd.DataFrame, topk_grid, method: str) -> dict:
    """Fit robust statistics on reference scores only (VALIDATION references)."""
    assert (ref_scores["is_anomaly"] == 0).all(), "normalisation must be fitted on reference images only"
    assert not (ref_scores["role"] == "TEST").any(), "normalisation must never see TEST data"
    norm = {"global": fit_robust(ref_scores["global"], method)}
    for k in topk_grid:
        norm[local_col(k)] = fit_robust(ref_scores[local_col(k)], method)
    return norm
