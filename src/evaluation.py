"""Shared experiment logic: scoring caches, policy fitting (leak-free), application, localisation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .conformal import conformal_pvalues, empirical_threshold, pvalue_diagnostics
from .cost_policy import select_alpha
from .localization import LocalizationAccumulator, pixel_threshold_from_reference, select_localization
from .metrics import image_metrics
from .models_convae import load_checkpoint
from .scoring import compute_scores, fit_normalization, fused_score, local_col, apply_robust
from .utils import file_sha256, get_device, load_json, log, save_json


# ------------------------------------------------------------ loading
def load_context(cfg: dict, paths) -> dict:
    split = D.load_split(paths.split_manifest, paths.data_dir, paths.csv)
    device = get_device(cfg["training"].get("device", "auto"))
    model, ckpt = load_checkpoint(paths.checkpoint, device)
    return {"split": split, "device": device, "model": model, "ckpt": ckpt}


def clean_scores(cfg: dict, paths, ctx: dict, force: bool = False) -> pd.DataFrame:
    """Scores for every non-TRAIN image under clean conditions (cached, keyed by checkpoint hash)."""
    cache = paths.cache / "clean_scores.csv"
    ck = file_sha256(paths.checkpoint)
    meta_path = paths.cache / "clean_scores.meta.json"
    if cache.exists() and meta_path.exists() and not force:
        m = load_json(meta_path)
        if m.get("checkpoint_sha") == ck and m.get("manifest_hash") == D.manifest_hash(ctx["split"]) \
                and m.get("topk_grid") == list(cfg["phase2"]["topk_grid"]) \
                and m.get("sigma") == cfg["phase2"]["smoothing_sigma"]:
            log("Using cached clean scores.")
            df = pd.read_csv(cache)
            df["filename"] = df["filename"].astype(str)
            return df
    recs = ctx["split"][ctx["split"].role != "TRAIN"].reset_index(drop=True)
    log(f"Scoring {len(recs)} images with frozen ConvAE ...")
    df = compute_scores(ctx["model"], recs, cfg, ctx["device"])
    df.to_csv(cache, index=False)
    save_json({"checkpoint_sha": ck, "manifest_hash": D.manifest_hash(ctx["split"]),
               "topk_grid": list(cfg["phase2"]["topk_grid"]),
               "sigma": cfg["phase2"]["smoothing_sigma"]}, meta_path)
    return df


def by_role(df: pd.DataFrame, role: str, anomaly: int | None = None) -> pd.DataFrame:
    d = df[df.role == role]
    if anomaly is not None:
        d = d[d.is_anomaly == anomaly]
    return d.reset_index(drop=True)


# ---------------------------------------------------------- Phase 1
def fit_phase1_policy(scores: pd.DataFrame, cfg: dict) -> dict:
    cal = by_role(scores, "CALIBRATION", 0)
    assert len(cal) and (cal.is_anomaly == 0).all()
    q = cfg["phase1"]["threshold_quantile"]
    thr = empirical_threshold(cal["global"], q)
    return {
        "phase": 1, "score": "global_mse", "threshold": thr,
        "threshold_source": f"{q:.3f}-quantile of CALIBRATION reference global scores (finite-sample corrected)",
        "calibration_size": int(len(cal)),
        "calibration_global_scores": cal["global"].tolist(),
    }


def apply_phase1(df: pd.DataFrame, pol: dict) -> pd.DataFrame:
    out = df[["filename", "label", "is_anomaly", "role"]].copy()
    out["score"] = df["global"].values
    out["p_value"] = conformal_pvalues(pol["calibration_global_scores"], df["global"].values)
    out["threshold"] = pol["threshold"]
    out["pred"] = (out["score"] >= pol["threshold"]).astype(int)
    out["decision"] = np.where(out.pred == 1, "ANOMALY", "REFERENCE")
    return out


# ---------------------------------------------------------- Phase 2
def select_topk_lambda(scores: pd.DataFrame, norm: dict, cfg: dict) -> tuple[float, float, pd.DataFrame]:
    """Validation-only selection: maximise ROC-AUC, then PR-AUC.
    Ties (common when validation AUC saturates) -> equal-weight prior |lambda - 1| smallest, then smaller k."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    val = by_role(scores, "VALIDATION")
    assert not (val.role == "TEST").any()
    rows = []
    for k in cfg["phase2"]["topk_grid"]:
        for lam in cfg["phase2"]["lambda_grid"]:
            s = fused_score(val, norm, k, lam)
            rows.append({"topk": k, "lambda": lam,
                         "val_roc_auc": roc_auc_score(val.is_anomaly, s),
                         "val_pr_auc": average_precision_score(val.is_anomaly, s)})
    t = pd.DataFrame(rows)
    t["_r"] = t.val_roc_auc.round(6)
    t["_p"] = t.val_pr_auc.round(6)
    t["_l"] = (t["lambda"] - 1.0).abs()
    best = t.sort_values(["_r", "_p", "_l", "topk"], ascending=[False, False, True, True]).iloc[0]
    return float(best.topk), float(best["lambda"]), t.drop(columns=["_r", "_p", "_l"])


def fit_phase2_policy(scores: pd.DataFrame, cfg: dict, variant: str = "full") -> tuple[dict, dict]:
    """variant: 'full' (global+local), 'local_only'. Returns (policy, diagnostics)."""
    p2 = cfg["phase2"]
    grid = p2["topk_grid"]
    val_ref = by_role(scores, "VALIDATION", 0)
    norm = fit_normalization(val_ref, grid, p2["normalization"])
    sel_table = None
    if variant == "local_only":
        from sklearn.metrics import roc_auc_score
        val = by_role(scores, "VALIDATION")
        aucs = {k: roc_auc_score(val.is_anomaly, val[local_col(k)]) for k in grid}
        topk = sorted(grid, key=lambda k: (-round(aucs[k], 6), k))[0]
        lam = None
    else:
        topk, lam, sel_table = select_topk_lambda(scores, norm, cfg)

    def score_fn(df):
        if variant == "local_only":
            return apply_robust(df[local_col(topk)], norm[local_col(topk)])
        return fused_score(df, norm, topk, lam)

    cal = by_role(scores, "CALIBRATION", 0)
    cal_scores = np.asarray(score_fn(cal))
    pol_val = by_role(scores, "POLICY_VALIDATION")
    p_pol = conformal_pvalues(cal_scores, score_fn(pol_val))
    alpha, alpha_table = select_alpha(p_pol, pol_val.is_anomaly, p2["alpha_grid"], p2["cost_fn"], p2["cost_fp"],
                                      p2.get("max_flag_rate"))
    if min(p2["alpha_grid"]) < 1.0 / (len(cal) + 1):
        log(f"NOTE: alpha values below 1/(n+1) = {1.0 / (len(cal) + 1):.4f} can never flag an image "
            f"(calibration n = {len(cal)}).")
    policy = {
        "phase": 2, "variant": variant, "topk": topk, "lambda": lam, "alpha": alpha,
        "normalization": norm, "calibration_size": int(len(cal)),
        "calibration_scores": cal_scores.tolist(),
        "cost_fn": p2["cost_fn"], "cost_fp": p2["cost_fp"], "smoothing_sigma": p2["smoothing_sigma"],
        "selection": {"topk_lambda": "VALIDATION (max ROC-AUC, then PR-AUC; ties -> lambda nearest 1, smaller k)",
                      "normalization": "VALIDATION references (robust median/MAD)",
                      "calibration": "CALIBRATION references only",
                      "alpha": "POLICY_VALIDATION (min 10*FN + 1*FP; ties -> smaller alpha)"},
        "min_attainable_p": 1.0 / (len(cal) + 1),
    }
    diag = {"topk_lambda_table": sel_table, "alpha_table": alpha_table}
    return policy, diag


def phase2_score(df: pd.DataFrame, pol: dict) -> np.ndarray:
    if pol.get("variant") == "local_only":
        k = pol["topk"]
        return apply_robust(df[local_col(k)], pol["normalization"][local_col(k)])
    return fused_score(df, pol["normalization"], pol["topk"], pol["lambda"])


def apply_phase2(df: pd.DataFrame, pol: dict, cal_scores=None, alpha: float | None = None) -> pd.DataFrame:
    cal_scores = pol["calibration_scores"] if cal_scores is None else cal_scores
    alpha = pol["alpha"] if alpha is None else alpha
    out = df[["filename", "label", "is_anomaly", "role"]].copy()
    k = pol["topk"]
    out["global_score"] = df["global"].values
    out["local_score"] = df[local_col(k)].values
    out["z_global"] = apply_robust(df["global"], pol["normalization"]["global"])
    out["z_local"] = apply_robust(df[local_col(k)], pol["normalization"][local_col(k)])
    out["score"] = phase2_score(df, pol)
    out["p_value"] = conformal_pvalues(cal_scores, out["score"].values)
    out["alpha"] = alpha
    out["pred"] = (out["p_value"] <= alpha).astype(int)
    out["decision"] = np.where(out.pred == 1, "ANOMALY", "REFERENCE")
    return out


def evaluate_predictions(pred: pd.DataFrame, cfg: dict) -> dict:
    p2 = cfg["phase2"]
    return image_metrics(pred.is_anomaly, pred.score, pred.pred, p2["cost_fn"], p2["cost_fp"])


def pvalue_diag(pred_test: pd.DataFrame, cfg: dict) -> dict:
    return pvalue_diagnostics(pred_test.loc[pred_test.is_anomaly == 0, "p_value"].values,
                              pred_test.loc[pred_test.is_anomaly == 1, "p_value"].values,
                              cfg["phase2"]["alpha_grid"])


# ------------------------------------------------------ localisation
def fit_localization(cfg: dict, ctx: dict) -> dict:
    """Uses VALIDATION images only: reference pixels define thresholds, anomalies select (q, morphology)."""
    val = D.role(ctx["split"], "VALIDATION")
    raw_ref, sm_ref, sm_anom, masks = [], [], [], []

    def cb(chunk, imgs, rec, raw, sm, m):
        for a, r, s, g in zip(chunk.is_anomaly.values, raw, sm, m):
            if a == 0:
                raw_ref.append(r.astype(np.float32))
                sm_ref.append(s.astype(np.float32))
            else:
                sm_anom.append(s.astype(np.float32))
                masks.append(g)

    log(f"Localisation selection on {len(val)} VALIDATION images ...")
    compute_scores(ctx["model"], val, cfg, ctx["device"], map_callback=cb, with_masks=True)
    lc = cfg["localization"]
    p1_thr = pixel_threshold_from_reference(np.stack(raw_ref), lc["phase1_pixel_quantile"])
    best, grid = select_localization(np.stack(sm_ref), sm_anom, masks, lc["pixel_quantile_grid"],
                                     lc["morphology_grid"])
    # AUC histogram ranges from reference statistics
    raw_max = float(np.quantile(np.stack(raw_ref), 0.999) * 50)
    sm_max = float(np.quantile(np.stack(sm_ref), 0.999) * 50)
    return {
        "phase1": {"threshold": p1_thr, "morph_radius": 0, "map": "raw",
                   "rule": f"{lc['phase1_pixel_quantile']}-quantile of VALIDATION reference raw pixel errors",
                   "auc_max": raw_max},
        "phase2": {"threshold": best["threshold"], "pixel_quantile": best["pixel_quantile"],
                   "morph_radius": best["morph_radius"], "map": "gaussian_smoothed",
                   "smoothing_sigma": cfg["phase2"]["smoothing_sigma"], "val_dice": best["val_dice"],
                   "rule": "(quantile, opening radius) maximising mean Dice on VALIDATION anomalies",
                   "auc_max": sm_max},
        "grid": grid,
    }


def evaluate_localization(cfg: dict, ctx: dict, loc: dict, records: pd.DataFrame, keep_examples: int = 8,
                          perturb=None) -> tuple[dict, dict, list, list]:
    bins = cfg["localization"]["pixel_auc_bins"]
    a1 = LocalizationAccumulator(loc["phase1"]["threshold"], 0, loc["phase1"]["auc_max"], bins, keep_examples)
    a2 = LocalizationAccumulator(loc["phase2"]["threshold"], loc["phase2"]["morph_radius"],
                                 loc["phase2"]["auc_max"], bins, keep_examples)

    def cb(chunk, imgs, rec, raw, sm, m):
        a1.update(chunk.filename.values, imgs, rec, raw, m, chunk.is_anomaly.values)
        a2.update(chunk.filename.values, imgs, rec, sm, m, chunk.is_anomaly.values)

    compute_scores(ctx["model"], records, cfg, ctx["device"], map_callback=cb, with_masks=True, perturb=perturb)
    return a1.summary(), a2.summary(), a1.examples, a2.examples


def per_image_dice(acc_names, dice) -> pd.DataFrame:
    return pd.DataFrame({"filename": acc_names, "dice": dice})


def ensure_exists(p: Path, hint: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. {hint}")
