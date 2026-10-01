"""Step 03/04 - Phase-1 baseline: calibrate + freeze the global-MSE threshold, then evaluate on TEST.

  --stage evaluate : fit threshold on CALIBRATION references, save policy, evaluate TEST once
  --stage freeze   : lock the Phase-1 checkpoint + policy (SHA-256) so Phase 2 reuses them unchanged
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluation as E  # noqa: E402
from src.config import get_paths, load_config  # noqa: E402
from src.utils import file_sha256, load_json, log, save_json  # noqa: E402


def evaluate(cfg=None) -> dict:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    ctx = E.load_context(cfg, paths)
    scores = E.clean_scores(cfg, paths, ctx)

    policy = E.fit_phase1_policy(scores, cfg)          # CALIBRATION references only
    save_json(policy, paths.phase1_policy)             # policy fixed BEFORE touching TEST
    test = E.by_role(scores, "TEST")
    pred = E.apply_phase1(test, policy)
    m = E.evaluate_predictions(pred, cfg)
    m.update(decision_threshold=policy["threshold"], threshold_source=policy["threshold_source"],
             calibration_reference_size=policy["calibration_size"], score="global reconstruction MSE")
    out = paths.results / "phase1"
    pred.to_csv(out / "predictions.csv", index=False)
    save_json(m, out / "metrics.json")
    log(f"PHASE 1 (TEST): ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} F1={m['f1']:.4f} "
        f"BalAcc={m['balanced_accuracy']:.4f} FPR={m['fpr']:.4f} FNR={m['fnr']:.4f} Cost/img={m['cost_per_image']:.4f}")
    return m


def freeze(cfg=None) -> dict:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    pol = load_json(paths.phase1_policy)
    pol["frozen"] = True
    pol["frozen_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    pol["checkpoint_sha256"] = file_sha256(paths.checkpoint)
    save_json(pol, paths.phase1_policy)
    log(f"Phase-1 frozen: checkpoint sha={pol['checkpoint_sha256']} threshold={pol['threshold']:.6g}")
    return pol


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["evaluate", "freeze", "all"], default="all")
    s = ap.parse_args().stage
    if s in ("evaluate", "all"):
        evaluate()
    if s in ("freeze", "all"):
        freeze()
