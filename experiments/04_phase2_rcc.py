"""Step 05/06 - Phase-2 RCC-ConvAE on the FROZEN Phase-1 ConvAE.

  --stage fit      : top-k + lambda (VALIDATION), robust normalisation (VALIDATION refs),
                     conformal calibration (CALIBRATION refs), alpha* (POLICY_VALIDATION),
                     localisation threshold + morphology (VALIDATION). TEST is never read.
  --stage evaluate : single final TEST evaluation of the frozen policies (image + pixel level).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src import data as D  # noqa: E402
from src import evaluation as E  # noqa: E402
from src.config import get_paths, load_config  # noqa: E402
from src.utils import file_sha256, load_json, log, save_json  # noqa: E402


def _check_frozen(paths) -> dict:
    p1 = load_json(paths.phase1_policy)
    if not p1.get("frozen"):
        raise RuntimeError("Phase-1 policy is not frozen. Run step 04 (03_evaluate_phase1.py --stage freeze).")
    if p1.get("checkpoint_sha256") != file_sha256(paths.checkpoint):
        raise RuntimeError("Checkpoint changed after Phase-1 freeze - Phase 2 must use the frozen ConvAE.")
    return p1


def fit(cfg=None) -> dict:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    p1 = _check_frozen(paths)
    ctx = E.load_context(cfg, paths)
    scores = E.clean_scores(cfg, paths, ctx)
    no_test = scores[scores.role != "TEST"].reset_index(drop=True)   # physically remove TEST rows

    policy, diag = E.fit_phase2_policy(no_test, cfg, "full")
    d = paths.phase2_dir
    save_json(policy["normalization"], d / "normalization.json")
    np.save(d / "calibration_scores.npy", np.asarray(policy["calibration_scores"]))
    slim = {k: v for k, v in policy.items() if k not in ("normalization", "calibration_scores")}
    slim["backbone_checkpoint_sha256"] = p1["checkpoint_sha256"]
    slim["frozen"] = True
    save_json(slim, d / "rcc_policy.json")
    out = paths.results / "phase2"
    diag["topk_lambda_table"].to_csv(out / "selection_topk_lambda_validation.csv", index=False)
    diag["alpha_table"].to_csv(out / "selection_alpha_policy_validation.csv", index=False)

    loc = E.fit_localization(cfg, ctx)                  # VALIDATION only
    import pandas as pd
    pd.DataFrame(loc.pop("grid")).to_csv(out / "selection_localization_validation.csv", index=False)
    save_json(loc, d / "localization_threshold.json")
    log(f"RCC policy frozen: top-k={policy['topk']} lambda={policy['lambda']} alpha*={policy['alpha']} "
        f"| calib n={policy['calibration_size']} | loc q={loc['phase2']['pixel_quantile']} "
        f"morph={loc['phase2']['morph_radius']} (val Dice={loc['phase2']['val_dice']:.4f})")
    return slim


def load_policy(paths) -> dict:
    d = paths.phase2_dir
    pol = load_json(d / "rcc_policy.json")
    pol["normalization"] = load_json(d / "normalization.json")
    pol["calibration_scores"] = np.load(d / "calibration_scores.npy").tolist()
    return pol


def evaluate(cfg=None) -> dict:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    ctx = E.load_context(cfg, paths)
    scores = E.clean_scores(cfg, paths, ctx)
    pol = load_policy(paths)
    test = E.by_role(scores, "TEST")
    pred = E.apply_phase2(test, pol)
    m = E.evaluate_predictions(pred, cfg)
    m.update(selected_topk=pol["topk"], selected_lambda=pol["lambda"], selected_alpha=pol["alpha"],
             calibration_reference_size=pol["calibration_size"], min_attainable_p=pol["min_attainable_p"],
             score="fused robust z-score -> conformal p-value", pvalue_diagnostics=E.pvalue_diag(pred, cfg))
    out = paths.results / "phase2"
    pred.to_csv(out / "predictions.csv", index=False)

    # ---- pixel-level localisation (both phases, single TEST pass) ----
    loc = load_json(paths.phase2_dir / "localization_threshold.json")
    test_recs = D.role(ctx["split"], "TEST")
    log(f"Localisation on {len(test_recs)} TEST images ...")
    l1, l2, ex1, ex2 = E.evaluate_localization(cfg, ctx, loc, test_recs, keep_examples=8)
    m["localization"] = l2
    save_json(m, out / "metrics.json")
    m1 = load_json(paths.results / "phase1" / "metrics.json")
    m1["localization"] = l1
    save_json(m1, paths.results / "phase1" / "metrics.json")
    np.savez_compressed(out / "localization_examples.npz", **{
        f"{k}_{i}": (e[k].astype(np.float16) if k not in ("filename", "dice") else np.array(e[k]))
        for i, e in enumerate(ex2) for k in ("image", "recon", "map", "gt", "pred", "filename", "dice")
    }, **{f"p1pred_{i}": e["pred"] for i, e in enumerate(ex1)}, **{f"p1map_{i}": e["map"].astype(np.float16)
                                                                   for i, e in enumerate(ex1)})
    log(f"PHASE 2 (TEST): ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} F1={m['f1']:.4f} "
        f"BalAcc={m['balanced_accuracy']:.4f} FPR={m['fpr']:.4f} FNR={m['fnr']:.4f} Cost/img={m['cost_per_image']:.4f}")
    log(f"LOCALISATION  P1: Dice={l1['dice']:.4f} IoU={l1['iou']:.4f} PixAUROC={l1['pixel_auroc']:.4f} | "
        f"P2: Dice={l2['dice']:.4f} IoU={l2['iou']:.4f} PixAUROC={l2['pixel_auroc']:.4f}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fit", "evaluate", "all"], default="all")
    s = ap.parse_args().stage
    if s in ("fit", "all"):
        fit()
    if s in ("evaluate", "all"):
        evaluate()
