"""Pixel-level defect localisation from reconstruction-error maps.

Phase 1: raw error map, fixed threshold = q-quantile of VALIDATION-reference pixel errors.
Phase 2: Gaussian-smoothed error map; pixel threshold quantile and morphological opening radius
         selected by mean Dice on VALIDATION anomalies (never TEST).
Pixel AUROC / AP: streaming histogram estimator (memory-safe for thousands of images).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y) <= radius * radius


def predict_mask(err_map: np.ndarray, threshold: float, morph_radius: int = 0) -> np.ndarray:
    m = err_map >= threshold
    if morph_radius and morph_radius > 0:
        m = ndimage.binary_opening(m, structure=disk(morph_radius))
    return m


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    ps, gs = pred.sum(), gt.sum()
    union = np.logical_or(pred, gt).sum()
    if ps + gs == 0:
        return 1.0, 1.0
    return float(2 * inter / (ps + gs)), float(inter / union) if union else 1.0


def pixel_threshold_from_reference(ref_maps: np.ndarray, quantile: float) -> float:
    return float(np.quantile(np.asarray(ref_maps, dtype=np.float32).ravel(), quantile))


def select_localization(val_ref_maps: np.ndarray, val_anom_maps: np.ndarray, val_anom_masks: np.ndarray,
                        quantile_grid, morph_grid) -> tuple[dict, list[dict]]:
    """Grid-search (quantile, morphology) maximising mean Dice on VALIDATION anomalies."""
    flat_ref = np.asarray(val_ref_maps, dtype=np.float32).ravel()
    rows = []
    for q in quantile_grid:
        thr = float(np.quantile(flat_ref, q))
        for r in morph_grid:
            d = [dice_iou(predict_mask(m, thr, r), g) for m, g in zip(val_anom_maps, val_anom_masks)]
            dice = float(np.mean([a for a, _ in d])) if d else 0.0
            iou = float(np.mean([b for _, b in d])) if d else 0.0
            rows.append({"pixel_quantile": float(q), "threshold": thr, "morph_radius": int(r),
                         "val_dice": dice, "val_iou": iou})
    best = sorted(rows, key=lambda z: (-z["val_dice"], -z["pixel_quantile"], z["morph_radius"]))[0]
    return best, rows


class StreamingPixelAUC:
    """Histogram-based pixel AUROC and average precision."""

    def __init__(self, max_value: float, bins: int = 10000):
        self.max_value = float(max_value) if max_value > 0 else 1.0
        self.bins = int(bins)
        self.pos = np.zeros(self.bins, dtype=np.int64)
        self.neg = np.zeros(self.bins, dtype=np.int64)

    def update(self, scores: np.ndarray, gt: np.ndarray) -> None:
        idx = np.clip((np.asarray(scores).ravel() / self.max_value * (self.bins - 1)).astype(np.int64),
                      0, self.bins - 1)
        g = np.asarray(gt).ravel().astype(bool)
        self.pos += np.bincount(idx[g], minlength=self.bins)
        self.neg += np.bincount(idx[~g], minlength=self.bins)

    def auroc(self) -> float | None:
        P, N = self.pos.sum(), self.neg.sum()
        if P == 0 or N == 0:
            return None
        # descending thresholds
        tp = np.cumsum(self.pos[::-1]) / P
        fp = np.cumsum(self.neg[::-1]) / N
        tpr = np.concatenate([[0], tp])
        fpr = np.concatenate([[0], fp])
        return float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))

    def average_precision(self) -> float | None:
        P = self.pos.sum()
        if P == 0:
            return None
        tp = np.cumsum(self.pos[::-1]).astype(float)
        fp = np.cumsum(self.neg[::-1]).astype(float)
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / P
        dr = np.diff(np.concatenate([[0], recall]))
        return float(np.sum(dr * precision))


class LocalizationAccumulator:
    """Accumulates Dice/IoU (anomalous images) and pixel AUROC/AP (all images)."""

    def __init__(self, threshold: float, morph_radius: int, auc_max: float, bins: int, keep_examples: int = 0):
        self.threshold, self.morph = threshold, morph_radius
        self.auc = StreamingPixelAUC(auc_max, bins)
        self.dice, self.iou, self.names = [], [], []
        self.examples: list[dict] = []
        self.keep = keep_examples

    def update(self, names, images, recons, maps, masks, is_anomaly) -> None:
        for n, img, rec, m, g, a in zip(names, images, recons, maps, masks, is_anomaly):
            self.auc.update(m, g)
            if a == 1:
                pred = predict_mask(m, self.threshold, self.morph)
                d, i = dice_iou(pred, g)
                self.dice.append(d)
                self.iou.append(i)
                self.names.append(n)
                if len(self.examples) < self.keep:
                    self.examples.append({"filename": n, "image": img, "recon": rec, "map": m,
                                          "gt": g, "pred": pred, "dice": d})

    def summary(self) -> dict:
        return {
            "dice": float(np.mean(self.dice)) if self.dice else None,
            "iou": float(np.mean(self.iou)) if self.iou else None,
            "dice_median": float(np.median(self.dice)) if self.dice else None,
            "pixel_auroc": self.auc.auroc(),
            "pixel_ap": self.auc.average_precision(),
            "n_anomalous_images": len(self.dice),
            "threshold": self.threshold,
            "morph_radius": self.morph,
        }
