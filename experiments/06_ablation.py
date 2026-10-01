"""Step 07 - Ablation study (all components fitted without TEST; TEST evaluated once per variant).

A  Phase 1: global reconstruction error + calibration-quantile threshold
B  Global + local evidence (fused) + calibration-quantile threshold (no conformal)
C  Global + local + conformal calibration, fixed alpha = 0.05
D  Global + local + conformal calibration + cost-aware alpha*   (= RCC-ConvAE)
E  Local evidence only + conformal calibration + cost-aware alpha*
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import evaluation as E  # noqa: E402
from src.conformal import empirical_threshold  # noqa: E402
from src.config import get_paths, load_config  # noqa: E402
from src.utils import load_json, log  # noqa: E402


def _phase2_module():
    spec = importlib.util.spec_from_file_location("p2", Path(__file__).with_name("04_phase2_rcc.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    ctx = E.load_context(cfg, paths)
    scores = E.clean_scores(cfg, paths, ctx)
    no_test = scores[scores.role != "TEST"].reset_index(drop=True)
    test = E.by_role(scores, "TEST")
    p1 = load_json(paths.phase1_policy)
    p2 = _phase2_module().load_policy(paths)
    loc1 = load_json(paths.results / "phase1" / "metrics.json").get("localization", {})
    loc2 = load_json(paths.results / "phase2" / "metrics.json").get("localization", {})

    rows = []

    def add(key, name, pred, loc, extra=""):
        m = E.evaluate_predictions(pred, cfg)
        rows.append({"variant": key, "description": name, "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
                     "f1": m["f1"], "balanced_accuracy": m["balanced_accuracy"], "recall": m["recall"],
                     "fpr": m["fpr"], "fnr": m["fnr"], "fp": m["fp"], "fn": m["fn"],
                     "cost_per_image": m["cost_per_image"], "dice": loc.get("dice"), "iou": loc.get("iou"),
                     "pixel_auroc": loc.get("pixel_auroc"), "decision_rule": extra})

    add("A", "Phase 1: global reconstruction error", E.apply_phase1(test, p1), loc1,
        f"global >= q{cfg['phase1']['threshold_quantile']} (calibration)")

    # B: fused score, calibration-quantile threshold (same quantile as Phase 1), no conformal
    cal = E.by_role(no_test, "CALIBRATION", 0)
    thr_b = empirical_threshold(E.phase2_score(cal, p2), cfg["phase1"]["threshold_quantile"])
    pb = E.apply_phase2(test, p2)
    pb["pred"] = (pb["score"] >= thr_b).astype(int)
    add("B", "Global + local evidence (fused)", pb, loc2, f"fused >= q{cfg['phase1']['threshold_quantile']}")

    a_fix = cfg["phase2"]["ablation_fixed_alpha"]
    add("C", "Global + local + conformal calibration", E.apply_phase2(test, p2, alpha=a_fix), loc2,
        f"p <= {a_fix} (fixed)")
    add("D", "Global + local + conformal + cost-aware alpha* (RCC-ConvAE)", E.apply_phase2(test, p2), loc2,
        f"p <= alpha*={p2['alpha']}")

    pe, _ = E.fit_phase2_policy(no_test, cfg, "local_only")
    add("E", "Local evidence only (+ conformal + cost-aware alpha)", E.apply_phase2(test, pe), loc2,
        f"top-k={pe['topk']}, p <= alpha*={pe['alpha']}")

    df = pd.DataFrame(rows)
    out = paths.results / "ablation"
    df.to_csv(out / "ablation.csv", index=False)
    df.to_csv(paths.reports / "ablation_table.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        log("Ablation (TEST):\n" + df[["variant", "roc_auc", "pr_auc", "f1", "balanced_accuracy", "fpr", "fnr",
                                        "cost_per_image", "dice"]].round(4).to_string(index=False))
    return df


if __name__ == "__main__":
    main()
